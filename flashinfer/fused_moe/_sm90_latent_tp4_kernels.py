"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice, gdc_launch_dependents, gdc_wait


@tr.jit
def tanh_ten(value):
    x = value * 0.1
    z = tl.abs(x)
    square = x * x
    p = tl.fma(0.01573968306183815, square, -0.05230396240949631)
    p = tl.fma(p, square, 0.13315297663211823)
    p = tl.fma(p, square, -0.33332768082618713)
    p = tl.fma(p, square, 0.0)
    small = tl.fma(p, x, x)
    reciprocal = tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[1.0 + tl.exp2(z * 2.885390043258667)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    large = tl.fma(-2.0, reciprocal, 1.0)
    large = libdevice.copysign(tl.where(z >= 9.010913848876953, 1.0, large), x)
    result = tl.where(z < 0.6000000238418579, small, large)
    return tl.where(z < 1.1754943508222875e-38, libdevice.copysign(0.0, x), result)


@tr.jit
def situ_quant(
    INPUT,
    Q,
    SCALE,
    F: tl.constexpr,
    BETA: tl.constexpr,
    UP_BETA: tl.constexpr,
    PDL: tl.constexpr = False,
):
    if PDL:
        gdc_launch_dependents()
        gdc_wait()
    row = tl.program_id(0)
    n = tl.program_id(1) * 128 + tl.arange(0, 128)
    g = tl.load(INPUT + row * 2 * F + n).to(tl.float32)
    u = tl.load(INPUT + row * 2 * F + F + n).to(tl.float32)
    tg = tanh_ten(g) if BETA == 10.0 else libdevice.tanh(g * (1.0 / BETA))
    gate = BETA * tg * (1.0 / (1.0 + tl.exp(-g)))
    if UP_BETA > 0:
        tu = tanh_ten(u) if UP_BETA == 10.0 else libdevice.tanh(u * (1.0 / UP_BETA))
        u = UP_BETA * tu
    h = (gate * u).to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(h)), 1e-10) / 448.0
    value = tl.minimum(tl.maximum(h / scale, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + row * F + n, value)
    tl.store(SCALE + row * (F // 128) + tl.program_id(1), scale)


@tr.jit
def quant_plan(
    X,
    Q,
    QS,
    IDS,
    SLOTS,
    EXPERTS,
    T: tl.constexpr,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
):
    block = tl.program_id(0)
    n = block * 128 + tl.arange(0, 128)
    row = block // (H // 128)
    col = block % (H // 128) * 128 + tl.arange(0, 128)
    value = tl.load(X + row * ROW_STRIDE + col).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value)), 1e-10) / 448.0
    q = tl.minimum(tl.maximum(value / scale, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + n, q)
    tl.store(QS + block, scale)
    if block < T * TOPK:
        expert = tl.load(IDS + block)
        candidates = tl.arange(0, tr.next_power_of_2(T * TOPK))
        matched = (tl.load(IDS + candidates, candidates < T * TOPK, -1) == expert) & (
            candidates < T * TOPK
        )
        before = tl.sum((matched & (candidates < block)).to(tl.int32), 0)
        active = before % 8 == 0
        tl.store(EXPERTS + block, tl.where(active, expert, -1))
        if active:
            position = tl.cumsum(matched.to(tl.int32)) - 1
            wanted = before + tl.arange(0, 8)
            found = tl.min(
                tl.where(
                    matched[:, None] & (position[:, None] == wanted[None, :]),
                    candidates[:, None],
                    T * TOPK,
                ),
                0,
            )
            tl.store(
                SLOTS + block * 8 + tl.arange(0, 8),
                tl.where(found < T * TOPK, found, -1),
            )


@tr.jit
def finish_sum(
    RAW,
    XS,
    WS,
    SHARED,
    ALPHA,
    GAMMA,
    WORKSPACE,
    EPOCHS,
    OUT,
    H: tl.constexpr,
    L: tl.constexpr,
    T: tl.constexpr,
    RANK: tl.constexpr,
    BN: tl.constexpr,
    COMM_BYTES: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cta = row * (H // BN) + tile
    grids = tl.num_programs(0) * tl.num_programs(1)
    epoch = tl.load(EPOCHS + cta * 32, volatile=True)
    comm_bytes = COMM_BYTES
    clear_size = 4 * T * H
    own = tl.load(WORKSPACE + 8 + RANK).to(tl.pointer_type(tl.uint8))
    data = (own + epoch.to(tl.int64) * comm_bytes).to(tl.pointer_type(tl.bfloat16))
    clear = (own + ((epoch + 2) % 3).to(tl.int64) * comm_bytes).to(
        tl.pointer_type(tl.uint16)
    )
    n = tile * BN + tl.arange(0, BN)
    acc = tl.full((BN,), 0.0, tl.float32)
    for kb in range(L // 128):
        raw = tl.load(RAW + (kb * T + row) * H + n)
        sx = tl.load(XS + row * (L // 128) + kb)
        sw = tl.load(WS + tile * (L // 128) + kb)
        acc += raw * sx * sw
    a = tl.load(ALPHA + row).to(tl.bfloat16)
    g = tl.load(GAMMA + row).to(tl.bfloat16)
    shared = tl.load(SHARED + row * H + n)
    routed = tl.inline_asm_elementwise(
        "mul.rn.bf16 $0, $1, $2;",
        "=h,h,h",
        [acc.to(tl.bfloat16), a],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )
    shared = tl.inline_asm_elementwise(
        "mul.rn.bf16 $0, $1, $2;",
        "=h,h,h",
        [shared, g],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )
    mixed = (routed.to(tl.float32) + shared.to(tl.float32)).to(tl.bfloat16)
    bits = mixed.to(tl.uint16, bitcast=True)
    mixed = tl.where(bits == 32768, 0, bits).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    offsets = row * H + n
    for peer in tl.static_range(4):
        remote = tl.load(WORKSPACE + 8 + peer).to(tl.pointer_type(tl.uint8))
        remote = (remote + epoch.to(tl.int64) * comm_bytes).to(
            tl.pointer_type(tl.bfloat16)
        )
        tl.store(remote + RANK * (T * H) + offsets, mixed)
    i = cta * 128 + tl.arange(0, 128)
    for start in range(0, clear_size, grids * 128):
        idx = start + i
        tl.store(clear + idx, 32768, idx < clear_size)
    v0 = tl.full((BN,), 0.0, tl.bfloat16)
    v1 = tl.full((BN,), 0.0, tl.bfloat16)
    v2 = tl.full((BN,), 0.0, tl.bfloat16)
    v3 = tl.full((BN,), 0.0, tl.bfloat16)
    done = False
    while not done:
        v0 = tl.load(data + offsets, volatile=True)
        v1 = tl.load(data + T * H + offsets, volatile=True)
        v2 = tl.load(data + 2 * T * H + offsets, volatile=True)
        v3 = tl.load(data + 3 * T * H + offsets, volatile=True)
        ready = (v0.to(tl.uint16, bitcast=True) != 32768) & (
            v1.to(tl.uint16, bitcast=True) != 32768
        )
        ready &= (v2.to(tl.uint16, bitcast=True) != 32768) & (
            v3.to(tl.uint16, bitcast=True) != 32768
        )
        done = tl.sum((~ready).to(tl.int32), 0) == 0
    reduced = (
        v0.to(tl.float32) + v1.to(tl.float32) + v2.to(tl.float32) + v3.to(tl.float32)
    )
    tl.store(OUT + offsets, reduced)
    tl.store(EPOCHS + cta * 32, (epoch + 1) % 3)


@tr.jit
def peer_initialize(WORKSPACE, RANK: tl.constexpr, WORDS: tl.constexpr):
    own = tl.load(WORKSPACE + 8 + RANK).to(tl.pointer_type(tl.uint32))
    i = tl.program_id(0) * 128 + tl.arange(0, 128)
    tl.store(own + i, 4294967295, i < WORDS)


@tr.jit
def peer_publish(
    RAW,
    XS,
    WS,
    WORKSPACE,
    RANK: tl.constexpr,
    T: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
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
    q = tl.minimum(tl.maximum(value / scale, -448.0), 448.0).to(tl.float8e4nv)
    bytes = tl.reshape(q.to(tl.uint8, bitcast=True), (32, 4)).to(tl.uint32)
    words = tl.sum(bytes << tl.arange(0, 4)[None, :] * 8, 1).to(tl.uint32)
    offsets = row * N + RANK * (N // 4) + group * 32 + tl.arange(0, 32)
    scale_offset = T * N + row * (N // 32) + RANK * (N // 128) + group
    for peer in tl.static_range(4):
        dest = tl.load(WORKSPACE + 8 + peer).to(tl.pointer_type(tl.uint32))
        tl.store(dest + offsets, words)
        tl.store(dest + scale_offset, scale.to(tl.uint32, bitcast=True))


@tr.jit
def peer_consume(
    WORKSPACE, Q, QS, RANK: tl.constexpr, T: tl.constexpr, N: tl.constexpr
):
    row, group = (tl.program_id(0), tl.program_id(1))
    own = tl.load(WORKSPACE + 8 + RANK).to(tl.pointer_type(tl.uint32))
    words = row * (N // 4) + group * 32 + tl.arange(0, 32)
    scale = T * (N // 4) + row * (N // 128) + group
    data = tl.full((32,), 4294967295, tl.uint32)
    sbits = tl.full((), 4294967295, tl.uint32)
    done = False
    while not done:
        data = tl.load(own + words, volatile=True)
        sbits = tl.load(own + scale, volatile=True)
        done = (tl.sum((data == 4294967295).to(tl.int32), 0) == 0) & (
            sbits != 4294967295
        )
    tl.store(Q.to(tl.pointer_type(tl.uint32)) + words, data)
    tl.store(QS + row * (N // 128) + group, sbits.to(tl.float32, bitcast=True))
    tl.store(own + words, 4294967295)
    tl.store(own + scale, 4294967295)


@tr.jit
def combine_quant(PART, Q, QS, L: tl.constexpr, TOPK: tl.constexpr, PDL: tl.constexpr):
    if PDL:
        gdc_launch_dependents()
        gdc_wait()
    token, block = (tl.program_id(0), tl.program_id(1))
    n = block * 128 + tl.arange(0, 128)
    value = tl.full((128,), 0.0, tl.float32)
    for slot in range(TOPK):
        value += tl.load(PART + (token * TOPK + slot) * L + n).to(tl.float32)
    value = value.to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value)), 1e-10) / 448.0
    quant = tl.minimum(tl.maximum(value / scale, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + token * L + n, quant)
    tl.store(QS + token * (L // 128) + block, scale)
