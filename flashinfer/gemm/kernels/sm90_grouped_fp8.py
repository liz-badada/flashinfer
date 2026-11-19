# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for license information.

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait


@tr.jit
def grouped(
    X,
    XS,
    W_DESC,
    WS,
    SLOTS,
    EXPERTS,
    RW,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    INPUT_REUSE: tl.constexpr,
    WEIGHTED: tl.constexpr,
    BN: tl.constexpr,
    UNROLL: tl.constexpr,
    PDL: tl.constexpr,
    CACHE_POLICY: tl.constexpr,
):
    if PDL:
        gdc_launch_dependents()
        gdc_wait()
    panel = tl.program_id(0)
    expert = tl.load(EXPERTS + panel).to(tl.int64)
    if expert >= 0:
        slots = tl.load(SLOTS + panel * 8 + tl.arange(0, 8))
        valid = slots >= 0
        rows = tl.maximum(slots, 0) // INPUT_REUSE
        tile = tl.program_id(1)
        n = tile * BN + tl.arange(0, BN)
        k = tl.arange(0, 128)
        acc = tl.full((8, BN), 0.0, tl.float32)
        for kb in tl.range(K // 128, loop_unroll_factor=UNROLL if UNROLL > 1 else None):
            x = tl.load(
                X + rows[:, None] * K + kb * 128 + k[None, :], valid[:, None], 0.0
            )
            w = tl.trans(W_DESC.load([(expert * N + tile * BN).to(tl.int32), kb * 128]))
            xs = tl.load(XS + rows * (K // 128) + kb, valid, 0.0).to(tl.float32)
            ws = tl.load(WS + (expert * (N // 128) + n // 128) * (K // 128) + kb).to(
                tl.float32
            )
            partial = tl.trans(
                tl.dot(
                    tl.trans(w),
                    tl.trans(x),
                    out_dtype=tl.float32,
                    max_num_imprecise_acc=0,
                )
            )
            acc += partial * xs[:, None] * ws[None, :]
        if WEIGHTED:
            acc *= tl.load(RW + slots, valid, 0.0)[:, None]
        tl.store(OUT + slots[:, None] * N + n[None, :], acc, valid[:, None])
