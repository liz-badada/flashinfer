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
    reciprocal = tl.inline_asm_elementwise('rcp.approx.ftz.f32 $0, $1;', constraints='=f,f', args=[1.0 + tl.exp2(z * 2.885390043258667)], dtype=tl.float32, is_pure=True, pack=1)
    large = tl.fma(-2.0, reciprocal, 1.0)
    large = libdevice.copysign(tl.where(z >= 9.010913848876953, 1.0, large), x)
    result = tl.where(z < 0.6000000238418579, small, large)
    return tl.where(z < 1.1754943508222875e-38, libdevice.copysign(0.0, x), result)

@tr.jit
def situ_quant(INPUT, Q, SCALE, F: tl.constexpr, BETA: tl.constexpr, UP_BETA: tl.constexpr, PDL: tl.constexpr=False):
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
def quant_plan(X, Q, QS, IDS, SLOTS, EXPERTS, T: tl.constexpr, H: tl.constexpr, TOPK: tl.constexpr, ROW_STRIDE: tl.constexpr):
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
        matched = (tl.load(IDS + candidates, candidates < T * TOPK, -1) == expert) & (candidates < T * TOPK)
        before = tl.sum((matched & (candidates < block)).to(tl.int32), 0)
        active = before % 8 == 0
        tl.store(EXPERTS + block, tl.where(active, expert, -1))
        if active:
            position = tl.cumsum(matched.to(tl.int32)) - 1
            wanted = before + tl.arange(0, 8)
            found = tl.min(tl.where(matched[:, None] & (position[:, None] == wanted[None, :]), candidates[:, None], T * TOPK), 0)
            tl.store(SLOTS + block * 8 + tl.arange(0, 8), tl.where(found < T * TOPK, found, -1))

@tr.jit
def raw_projection(X, WD, RAW, T: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr, BN: tl.constexpr, PDL: tl.constexpr=False):
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
            x = tl.load(X + rows[None, :] * K + kb * 128 + k[:, None], rows[None, :] < T, 0.0)
            w = WD.load([tile * BN, kb * 128])
            raw = tl.dot(w, x, out_dtype=tl.float32, max_num_imprecise_acc=0)
            tl.store(RAW + (kb * T + rows[None, :]) * N + n[:, None], raw, rows[None, :] < T)

@tr.jit
def dense_bf16(X, XS, WD, WS, OUT, T: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr):
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
    tl.store(OUT + rows[None, :] * N + n[:, None], acc, valid[None, :] & (n[:, None] < N))

@tr.jit
def finish_sum(RAW, XS, WS, SHARED, ALPHA, GAMMA, WORKSPACE, EPOCHS, OUT, H: tl.constexpr, L: tl.constexpr, T: tl.constexpr, RANK: tl.constexpr, BN: tl.constexpr, COMM_BYTES: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cta = row * (H // BN) + tile
    grids = tl.num_programs(0) * tl.num_programs(1)
    epoch = tl.load(EPOCHS + cta * 32, volatile=True)
    comm_bytes = COMM_BYTES
    clear_size = 4 * T * H
    own = tl.load(WORKSPACE + 8 + RANK).to(tl.pointer_type(tl.uint8))
    data = (own + epoch.to(tl.int64) * comm_bytes).to(tl.pointer_type(tl.bfloat16))
    clear = (own + ((epoch + 2) % 3).to(tl.int64) * comm_bytes).to(tl.pointer_type(tl.uint16))
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
    routed = tl.inline_asm_elementwise('mul.rn.bf16 $0, $1, $2;', '=h,h,h', [acc.to(tl.bfloat16), a], dtype=tl.bfloat16, is_pure=True, pack=1)
    shared = tl.inline_asm_elementwise('mul.rn.bf16 $0, $1, $2;', '=h,h,h', [shared, g], dtype=tl.bfloat16, is_pure=True, pack=1)
    mixed = (routed.to(tl.float32) + shared.to(tl.float32)).to(tl.bfloat16)
    bits = mixed.to(tl.uint16, bitcast=True)
    mixed = tl.where(bits == 32768, 0, bits).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    offsets = row * H + n
    for peer in tl.static_range(4):
        remote = tl.load(WORKSPACE + 8 + peer).to(tl.pointer_type(tl.uint8))
        remote = (remote + epoch.to(tl.int64) * comm_bytes).to(tl.pointer_type(tl.bfloat16))
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
        ready = (v0.to(tl.uint16, bitcast=True) != 32768) & (v1.to(tl.uint16, bitcast=True) != 32768)
        ready &= (v2.to(tl.uint16, bitcast=True) != 32768) & (v3.to(tl.uint16, bitcast=True) != 32768)
        done = tl.sum((~ready).to(tl.int32), 0) == 0
    reduced = v0.to(tl.float32) + v1.to(tl.float32) + v2.to(tl.float32) + v3.to(tl.float32)
    tl.store(OUT + offsets, reduced)
    tl.store(EPOCHS + cta * 32, (epoch + 1) % 3)

@tr.jit
def peer_initialize(WORKSPACE, RANK: tl.constexpr, WORDS: tl.constexpr):
    own = tl.load(WORKSPACE + 8 + RANK).to(tl.pointer_type(tl.uint32))
    i = tl.program_id(0) * 128 + tl.arange(0, 128)
    tl.store(own + i, 4294967295, i < WORDS)

@tr.jit
def peer_publish(RAW, XS, WS, WORKSPACE, RANK: tl.constexpr, T: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
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
def peer_consume(WORKSPACE, Q, QS, RANK: tl.constexpr, T: tl.constexpr, N: tl.constexpr):
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
        done = (tl.sum((data == 4294967295).to(tl.int32), 0) == 0) & (sbits != 4294967295)
    tl.store(Q.to(tl.pointer_type(tl.uint32)) + words, data)
    tl.store(QS + row * (N // 128) + group, sbits.to(tl.float32, bitcast=True))
    tl.store(own + words, 4294967295)
    tl.store(own + scale, 4294967295)

@tr.jit
def project_mix(X, XS, WD, WS, SHARED, ALPHA, SHARED_GATE, OUT, H: tl.constexpr, L: tl.constexpr, BN: tl.constexpr, T: tl.constexpr):
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
    routed = tl.inline_asm_elementwise('mul.rn.bf16 $0, $1, $2;', '=h,h,h', [routed.to(tl.bfloat16), alpha[None, :].to(tl.bfloat16)], dtype=tl.bfloat16, is_pure=True, pack=1).to(tl.float32)
    shared = tl.load(SHARED + rows[None, :] * H + n[:, None], valid[None, :] & (n[:, None] < H), 0.0).to(tl.float32)
    shared_gate = tl.load(SHARED_GATE + rows, valid, 0.0).to(tl.bfloat16).to(tl.float32)
    shared = tl.inline_asm_elementwise('mul.rn.bf16 $0, $1, $2;', '=h,h,h', [shared.to(tl.bfloat16), shared_gate[None, :].to(tl.bfloat16)], dtype=tl.bfloat16, is_pure=True, pack=1).to(tl.float32)
    tl.store(OUT + rows[None, :] * H + n[:, None], routed + shared, valid[None, :] & (n[:, None] < H))

@tr.jit
def finish_quant(RAW, XS, WS, Q, QS, T: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
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

@tr.jit
def grouped(X, XS, W_DESC, WS, SLOTS, EXPERTS, RW, OUT, N: tl.constexpr, K: tl.constexpr, INPUT_REUSE: tl.constexpr, WEIGHTED: tl.constexpr, BN: tl.constexpr, UNROLL: tl.constexpr, PDL: tl.constexpr, CACHE_POLICY: tl.constexpr):
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
            x = tl.load(X + rows[:, None] * K + kb * 128 + k[None, :], valid[:, None], 0.0)
            w = tl.trans(W_DESC.load([(expert * N + tile * BN).to(tl.int32), kb * 128]))
            xs = tl.load(XS + rows * (K // 128) + kb, valid, 0.0).to(tl.float32)
            ws = tl.load(WS + (expert * (N // 128) + n // 128) * (K // 128) + kb).to(tl.float32)
            partial = tl.trans(tl.dot(tl.trans(w), tl.trans(x), out_dtype=tl.float32, max_num_imprecise_acc=0))
            acc += partial * xs[:, None] * ws[None, :]
        if WEIGHTED:
            acc *= tl.load(RW + slots, valid, 0.0)[:, None]
        tl.store(OUT + slots[:, None] * N + n[None, :], acc, valid[:, None])

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
