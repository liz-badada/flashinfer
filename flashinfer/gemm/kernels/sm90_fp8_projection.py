# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait


@tr.jit
def _raw_projection(
    X,
    WD,
    RAW,
    T: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    BN: tl.constexpr,
    PDL: tl.constexpr = False,
):
    if PDL:
        gdc_launch_dependents()
        gdc_wait()
    panel = tl.program_id(0)
    tile = tl.program_id(1)
    group = tl.program_id(2)
    rows = panel * 8 + tl.arange(0, 8)
    n = tile * BN + tl.arange(0, BN)
    k = tl.arange(0, 128)
    for offset in range(GROUP):
        kb = group * GROUP + offset
        if kb < K // 128:
            x = tl.load(
                X + rows[None, :] * K + kb * 128 + k[:, None], rows[None, :] < T, 0.0
            )
            w = WD.load([tile * BN, kb * 128])
            raw = tl.dot(w, x, out_dtype=tl.float32, max_num_imprecise_acc=0)
            tl.store(
                RAW + (kb * T + rows[None, :]) * N + n[:, None], raw, rows[None, :] < T
            )


@tr.jit
def _dense_bf16(
    X,
    XS,
    WD,
    WS,
    OUT,
    T: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)
    valid = rows < T
    tile = tl.program_id(1)
    n = tile * BN + tl.arange(0, BN)
    k = tl.arange(0, 128)
    acc = tl.full((BN, 8), 0.0, tl.float32)
    for kb in range(K // 128):
        x = tl.load(X + rows[None, :] * K + kb * 128 + k[:, None], valid[None, :], 0.0)
        w = WD.load([tile * BN, kb * 128])
        xs = tl.load(XS + rows * (K // 128) + kb, valid, 0.0)
        ws = tl.load(WS + tile * BN // 128 * (K // 128) + kb)
        partial = tl.dot(w, x, out_dtype=tl.float32, max_num_imprecise_acc=0)
        acc += partial * xs[None, :] * ws
    tl.store(
        OUT + rows[None, :] * N + n[:, None], acc, valid[None, :] & (n[:, None] < N)
    )


@tr.jit
def _project_mix(
    X,
    XS,
    WD,
    WS,
    SHARED,
    ALPHA,
    SHARED_GATE,
    OUT,
    H: tl.constexpr,
    L: tl.constexpr,
    BN: tl.constexpr,
    T: tl.constexpr,
):
    slots = tl.program_id(0) * 8 + tl.arange(0, 8)
    valid = slots < T
    rows = tl.maximum(slots, 0)
    tile = tl.program_id(1)
    n = tile * BN + tl.arange(0, BN)
    k = tl.arange(0, 128)
    acc = tl.full((BN, 8), 0.0, tl.float32)
    for kb in range(L // 128):
        x = tl.load(X + rows[None, :] * L + kb * 128 + k[:, None], valid[None, :], 0.0)
        w = WD.load([tile * BN, kb * 128])
        xs = tl.load(XS + rows * (L // 128) + kb, valid, 0.0)
        ws = tl.load(WS + tile * BN // 128 * (L // 128) + kb)
        partial = tl.dot(w, x, out_dtype=tl.float32, max_num_imprecise_acc=0)
        acc += partial * xs[None, :] * ws
    routed = acc.to(tl.bfloat16).to(tl.float32)
    alpha = tl.load(ALPHA + rows, valid, 0.0).to(tl.bfloat16).to(tl.float32)
    routed = tl.inline_asm_elementwise(
        "mul.rn.bf16 $0, $1, $2;",
        "=h,h,h",
        [routed.to(tl.bfloat16), alpha[None, :].to(tl.bfloat16)],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    ).to(tl.float32)
    shared = tl.load(
        SHARED + rows[None, :] * H + n[:, None], valid[None, :] & (n[:, None] < H), 0.0
    ).to(tl.float32)
    shared_gate = tl.load(SHARED_GATE + rows, valid, 0.0).to(tl.bfloat16).to(tl.float32)
    shared = tl.inline_asm_elementwise(
        "mul.rn.bf16 $0, $1, $2;",
        "=h,h,h",
        [shared.to(tl.bfloat16), shared_gate[None, :].to(tl.bfloat16)],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    ).to(tl.float32)
    tl.store(
        OUT + rows[None, :] * H + n[:, None],
        routed + shared,
        valid[None, :] & (n[:, None] < H),
    )


@tr.jit
def _finish_quant(
    RAW, XS, WS, Q, QS, T: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    row, group = (tl.program_id(0), tl.program_id(1))
    n = group * 128 + tl.arange(0, 128)
    acc = tl.full((128,), 0.0, tl.float32)
    for kb in range(K // 128):
        raw = tl.load(RAW + (kb * T + row) * N + n)
        sx = tl.load(XS + row * (K // 128) + kb)
        sw = tl.load(WS + group * (K // 128) + kb)
        acc += raw * sx * sw
    value = acc.to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value)), 1e-10) / 448.0
    quant = tl.minimum(tl.maximum(value / scale, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + row * N + n, quant)
    tl.store(QS + row * (N // 128) + group, scale)
