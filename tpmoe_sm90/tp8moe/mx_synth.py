"""Synthetic Kimi K3 expert weights for tp8moe: the kernel's Humming fused-E8M0 MXFP4 layout (DIM = 3584 latent, INTER = 384 per-rank
intermediate), the K3 checkpoint format (compressed-tensors mxfp4-pack: weight_packed uint8 [N, K/2], weight_scale E8M0 uint8 [N, K/32])
and the conversion checkpoint -> kernel layout.

Block internals (the kernel's register dequant: lane_of / dequant_word / make_pools in tp8moe.cu and the TMA maps in set_weights):
  * a 1 KB "Humming block" (HLB) holds 64 rows x 32 k of fp4 codes.  Inside a block: 16-B chunk c (0..63): h32 = c >> 5, lane = c & 31,
    g = lane >> 2, tig = lane & 3; the two 8-B halves sel = 0/1 hold rows R0 = 32*h32 + 16*sel + g (word 0) and R0 + 8 (word 1);
    word nibble n: bits 0..2 = magnitude code of element n, where element n < 4 is k = 4*tig + n and element n >= 4 is
    k = 16 + 4*tig + (n - 4); nibble bit 3 = sign of element (n even: n/2, n odd: 4 + (n-1)/2).  Codes 0..7 = {0, .5, 1, 1.5, 2, 3, 4, 6}.
  * one scale byte o per (row, k32 slice): scale bytes of a slice follow the same block order, 64 B per block, byte index
    8*g + 4*h32 + 2*sel + b for row R = 32*h32 + 16*sel + g + 8*b.  The dequantized e4m3 value is code * 2^(o-6); the kernel then
    multiplies by factor[e] * 64, so the real weight is code * 2^o * factor[e], o in [0, 12].
Slice order (the k32 slice of an expert is a sequence of blocks):
  * FC1 (w13 = [w1 rows 0..INTER-1 (gate); w3 rows INTER.. (up)], K = DIM): slice stride 2 * ITEMS * 2 KB =
    [w13 (2)][128-row item q (ITEMS)][64-row block (2)][1 KB] -> rows INTER * w13 + 128 q + 64 blk + R
  * FC2 (w2, K = INTER, N = DIM): slice stride (DIM / 64) KB = [DIM / 64 blocks][1 KB] -> rows 64 blk + R
"""
import torch

DIM, INTER = 3584, 384
ITEMS = INTER // 128
S1, S2, NB2 = DIM // 32, INTER // 32, DIM // 64
O_MAX = 12
FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_EDGES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])   # round-to-nearest boundaries between the E2M1 magnitudes


# ---- Humming block primitives
def _block_perm():
    """(word index 0..255, nibble 0..7) of the magnitude and of the sign of every (row, k) of a 1 KB block."""
    mag_word = torch.empty(64, 32, dtype=torch.long)
    mag_nib = torch.empty(64, 32, dtype=torch.long)
    sgn_word = torch.empty(64, 32, dtype=torch.long)
    sgn_nib = torch.empty(64, 32, dtype=torch.long)
    for c in range(64):   # chunk c = words 4c..4c+3: two 8-B halves (sel) of two words (b)
        h32, lane = c >> 5, c & 31
        g, tig = lane >> 2, lane & 3
        for sel in range(2):
            for b in range(2):
                R = 32 * h32 + 16 * sel + g + 8 * b
                w = c * 4 + sel * 2 + b
                for n in range(8):
                    k = 4 * tig + n if n < 4 else 16 + 4 * tig + (n - 4)
                    mag_word[R, k] = w
                    mag_nib[R, k] = n
                    sgn_word[R, k] = w
                    sgn_nib[R, k] = 2 * n if n < 4 else 2 * (n - 4) + 1   # sign of element n lives in nibble 2n (n < 4) or 2(n-4)+1
    return mag_word, mag_nib, sgn_word, sgn_nib


def _scale_perm():
    idx = torch.empty(64, dtype=torch.long)
    for h32 in range(2):
        for sel in range(2):
            for g in range(8):
                for b in range(2):
                    R = 32 * h32 + 16 * sel + g + 8 * b
                    idx[R] = 8 * g + 4 * h32 + 2 * sel + b
    return idx


_MW, _MN, _SW, _SN = _block_perm()
_SP = _scale_perm()


def encode_blocks(codes, signs):
    """codes [B, 64, 32] int (0..7), signs [B, 64, 32] (0/1) -> uint32 words [B, 256] in Humming block order (int32 bit pattern)."""
    B = codes.shape[0]
    words = torch.zeros(B, 256, dtype=torch.int64, device=codes.device)
    mw = _MW.to(codes.device).reshape(-1)
    mn = _MN.to(codes.device).reshape(-1)
    sn = _SN.to(codes.device).reshape(-1)
    cf = codes.reshape(B, -1).to(torch.int64)
    sf = signs.reshape(B, -1).to(torch.int64)
    words.scatter_add_(1, mw.expand(B, -1), cf << (4 * mn))
    words.scatter_add_(1, mw.expand(B, -1), (sf << 3) << (4 * sn))
    return words.to(torch.int32)


def decode_blocks(words):
    """uint32 words [B, 256] -> (codes [B, 64, 32], signs [B, 64, 32])."""
    B = words.shape[0]
    w = words.to(torch.int64) & 0xFFFFFFFF
    mw = _MW.to(words.device).reshape(-1)
    mn = _MN.to(words.device).reshape(-1)
    sn = _SN.to(words.device).reshape(-1)
    codes = (torch.gather(w, 1, mw.expand(B, -1)) >> (4 * mn)) & 7
    signs = (torch.gather(w, 1, mw.expand(B, -1)) >> (4 * sn + 3)) & 1
    return codes.reshape(B, 64, 32), signs.reshape(B, 64, 32)


def quantize_mxfp4_fused(W, o_max=O_MAX):
    """W [rows, K] float -> (codes, signs, o, f) per 32-group with the fused exponent o in [0, o_max] and the per-tensor factor f:
    scale(group) = 2^o * f, f = max_scale / 2^o_max; groups whose scale would need o < 0 are clamped (extra error on tiny groups)."""
    rows, K = W.shape
    Wg = W.float().reshape(rows, K // 32, 32)
    amax = Wg.abs().amax(-1).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 6.0))      # scale = 2^e so that amax / scale <= 6
    e_max = e.max()
    f = torch.exp2(e_max - o_max)
    o = (e - e_max + o_max).clamp_min(0)
    q = Wg / (torch.exp2(o) * f).unsqueeze(-1)
    codes = torch.bucketize(q.abs().clamp_max(6.0), _E2M1_EDGES.to(W.device)).to(torch.int64)
    signs = (q < 0).to(torch.int64)
    return codes, signs, o.to(torch.int64), f


def dequant_rows(codes, signs, o, f):
    mag = FP4_MAG.to(codes.device)[codes]
    val = torch.where(signs.bool(), -mag, mag) * torch.exp2(o.float()).unsqueeze(-1) * f
    return val.reshape(codes.shape[0], -1)


# ---- kernel layout (tp8moe shapes)
def _scales_fc1(o, device):
    sp = _SP.to(device)
    scales = torch.zeros(S1, 2, ITEMS, 2, 64, dtype=torch.uint8, device=device)
    scales[..., sp] = o.permute(1, 0).reshape(S1, 2, ITEMS, 2, 64).to(torch.uint8)
    return scales.reshape(-1)


def _scales_fc2(o, device):
    sp = _SP.to(device)
    scales = torch.zeros(S2, NB2, 64, dtype=torch.uint8, device=device)
    scales[..., sp] = o.permute(1, 0).reshape(S2, NB2, 64).to(torch.uint8)
    return scales.reshape(-1)


def _words_fc1(codes, signs):
    c = codes.permute(1, 0, 2).reshape(S1, 2, ITEMS, 2, 64, 32).reshape(-1, 64, 32)
    s = signs.permute(1, 0, 2).reshape(S1, 2, ITEMS, 2, 64, 32).reshape(-1, 64, 32)
    return encode_blocks(c, s).reshape(-1)


def _words_fc2(codes, signs):
    c = codes.permute(1, 0, 2).reshape(S2 * NB2, 64, 32)
    s = signs.permute(1, 0, 2).reshape(S2 * NB2, 64, 32)
    return encode_blocks(c, s).reshape(-1)


def pack_fc1(W13, o_max=O_MAX):
    """W13 [2 * INTER, DIM] float -> (words int32 [S1 * 2 * ITEMS * 2 * 256], scales uint8 [S1 * 2 * INTER], f, Wdq [2 * INTER, DIM])."""
    assert tuple(W13.shape) == (2 * INTER, DIM)
    codes, signs, o, f = quantize_mxfp4_fused(W13, o_max)   # [2 * INTER, S1, 32]
    return _words_fc1(codes, signs), _scales_fc1(o, W13.device), f, dequant_rows(codes, signs, o, f)


def pack_fc2(W2, o_max=O_MAX):
    """W2 [DIM, INTER] -> (words int32 [S2 * NB2 * 256], scales uint8 [S2 * DIM], f, Wdq)."""
    assert tuple(W2.shape) == (DIM, INTER)
    codes, signs, o, f = quantize_mxfp4_fused(W2, o_max)      # [DIM, S2, 32]
    return _words_fc2(codes, signs), _scales_fc2(o, W2.device), f, dequant_rows(codes, signs, o, f)


def unpack_fc1(words, scales, f):
    c, s = decode_blocks(words.reshape(S1 * 2 * ITEMS * 2, 256))
    codes = c.reshape(S1, 2 * INTER, 32).permute(1, 0, 2)
    signs = s.reshape(S1, 2 * INTER, 32).permute(1, 0, 2)
    sp = _SP.to(words.device)
    o = scales.reshape(S1, 2, ITEMS, 2, 64)[..., sp].reshape(S1, 2 * INTER).permute(1, 0).to(torch.int64)
    return dequant_rows(codes, signs, o, f)


def unpack_fc2(words, scales, f):
    c, s = decode_blocks(words.reshape(S2 * NB2, 256))
    codes = c.reshape(S2, DIM, 32).permute(1, 0, 2)
    signs = s.reshape(S2, DIM, 32).permute(1, 0, 2)
    sp = _SP.to(words.device)
    o = scales.reshape(S2, NB2, 64)[..., sp].reshape(S2, DIM).permute(1, 0).to(torch.int64)
    return dequant_rows(codes, signs, o, f)


# ---- K3 checkpoint format (compressed-tensors mxfp4-pack-quantized, group 32, symmetric): E2M1 nibbles (low nibble = even k), E8M0 scales
def mx_quantize(W):
    """W [N, K] float -> (weight_packed uint8 [N, K/2], weight_scale uint8 [N, K/32]) with the standard MX recipe:
    shared exponent e = floor(log2(amax)) - 2 (so the largest magnitude maps to the E2M1 range <= 6), scale byte = e + 127."""
    N, K = W.shape
    g = W.float().reshape(N, K // 32, 32)
    amax = g.abs().amax(-1).clamp_min(2.0 ** -120)
    e = torch.floor(torch.log2(amax)) - 2
    q = g / torch.exp2(e).unsqueeze(-1)
    code = torch.bucketize(q.abs().clamp_max(6.0), _E2M1_EDGES.to(W.device)).to(torch.uint8)
    nib = code | ((q < 0).to(torch.uint8) << 3)
    nib = nib.reshape(N, K)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return packed.contiguous(), (e + 127).to(torch.uint8).contiguous()


def mx_dequant(packed, scale):
    N = packed.shape[0]
    nib = torch.stack([packed & 15, packed >> 4], -1).reshape(N, -1)
    mag = FP4_MAG.to(packed.device)[(nib & 7).long()]
    val = torch.where((nib & 8).bool(), -mag, mag)
    e = scale.float() - 127.0
    return (val.reshape(N, -1, 32) * torch.exp2(e).unsqueeze(-1)).reshape(N, -1)


def _ckpt_to_codes(packed, scale, o_max=O_MAX):
    """(packed, E8M0 scale) -> (codes, signs, o, f, n_clamped): per-tensor factor f = 2^(e_max - o_max); groups whose exponent is below
    e_max - o_max are re-rounded onto the coarser grid (the Humming fused format spans 2^o_max of exponent per tensor)."""
    N = packed.shape[0]
    nib = torch.stack([packed & 15, packed >> 4], -1).reshape(N, -1, 32)
    codes = (nib & 7).long()
    signs = ((nib >> 3) & 1).long()
    e = scale.long() - 127
    e_max = int(e.max())
    o = e - e_max + o_max
    clamped = o < 0
    n_cl = int(clamped.sum())
    if n_cl:
        mag = FP4_MAG.to(packed.device)[codes] * torch.exp2(o.float()).unsqueeze(-1)   # magnitude in units of 2^(e_max - o_max)
        re = torch.bucketize(mag.clamp_max(6.0), _E2M1_EDGES.to(packed.device))
        codes = torch.where(clamped.unsqueeze(-1), re, codes)
        o = o.clamp_min(0)
    f = torch.tensor(2.0 ** (e_max - o_max), device=packed.device)
    return codes, signs, o, f, n_cl


def ckpt_to_kernel_fc1(w1_packed, w1_scale, w3_packed, w3_scale):
    """K3 experts.N.w1 / w3 (each [INTER_shard, DIM/2] + [INTER_shard, DIM/32]) -> kernel FC1 (words, scales, f, Wdq, n_clamped)."""
    packed = torch.cat([w1_packed, w3_packed], 0); scale = torch.cat([w1_scale, w3_scale], 0)
    codes, signs, o, f, ncl = _ckpt_to_codes(packed, scale)
    return _words_fc1(codes, signs), _scales_fc1(o, packed.device), f, dequant_rows(codes, signs, o, f), ncl


def ckpt_to_kernel_fc2(w2_packed, w2_scale):
    """K3 experts.N.w2 ([DIM, INTER_shard/2] + [DIM, INTER_shard/32]) -> kernel FC2 (words, scales, f, Wdq, n_clamped)."""
    codes, signs, o, f, ncl = _ckpt_to_codes(w2_packed, w2_scale)
    return _words_fc2(codes, signs), _scales_fc2(o, w2_packed.device), f, dequant_rows(codes, signs, o, f), ncl


def make_expert_weights(n_exp, device, seed=0, std=0.02, keep_dq=False, via_ckpt=False):
    """Random experts in the kernel layout: w1 int32, s1 uint8, f1 fp32 [E], w2, s2, f2 (+ dq1 [E, 2*INTER, DIM], dq2 [E, DIM, INTER] bf16).
    via_ckpt: generate in the K3 checkpoint format first (standard MX quantization) and convert (exercises the loader path)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    w1 = torch.empty(n_exp, S1 * 2 * ITEMS * 2 * 256, dtype=torch.int32, device=device)
    s1 = torch.empty(n_exp, S1 * 2 * INTER, dtype=torch.uint8, device=device)
    w2 = torch.empty(n_exp, S2 * NB2 * 256, dtype=torch.int32, device=device)
    s2 = torch.empty(n_exp, S2 * DIM, dtype=torch.uint8, device=device)
    f1 = torch.empty(n_exp, dtype=torch.float32, device=device)
    f2 = torch.empty(n_exp, dtype=torch.float32, device=device)
    dq1, dq2 = [], []
    for e in range(n_exp):
        W13 = (torch.randn(2 * INTER, DIM, generator=g) * std).to(device)
        W2 = (torch.randn(DIM, INTER, generator=g) * std).to(device)
        if via_ckpt:
            p1, q1 = mx_quantize(W13[:INTER]); p3, q3 = mx_quantize(W13[INTER:]); p2, q2 = mx_quantize(W2)
            a, b, c, d, _ = ckpt_to_kernel_fc1(p1, q1, p3, q3); w1[e], s1[e], f1[e] = a, b, c
            if keep_dq: dq1.append(d.to(torch.bfloat16))
            a, b, c, d, _ = ckpt_to_kernel_fc2(p2, q2); w2[e], s2[e], f2[e] = a, b, c
            if keep_dq: dq2.append(d.to(torch.bfloat16))
        else:
            a, b, c, d = pack_fc1(W13); w1[e], s1[e], f1[e] = a, b, c
            if keep_dq: dq1.append(d.to(torch.bfloat16))
            a, b, c, d = pack_fc2(W2); w2[e], s2[e], f2[e] = a, b, c
            if keep_dq: dq2.append(d.to(torch.bfloat16))
    out = dict(w1=w1.reshape(-1), s1=s1.reshape(-1), f1=f1, w2=w2.reshape(-1), s2=s2.reshape(-1), f2=f2)
    if keep_dq:
        out['dq1'] = torch.stack(dq1); out['dq2'] = torch.stack(dq2)
    return out


if __name__ == '__main__':   # round trips: kernel layout, and checkpoint format -> kernel layout
    dev = torch.device('cuda')
    torch.manual_seed(0)
    W13 = (torch.randn(2 * INTER, DIM) * 0.02).to(dev)
    w, s, f, dq = pack_fc1(W13)
    print('fc1 roundtrip max|diff|', (unpack_fc1(w, s, f) - dq).abs().max().item(), 'quant rel err', ((dq - W13).norm() / W13.norm()).item())
    W2 = (torch.randn(DIM, INTER) * 0.02).to(dev)
    w, s, f, dq = pack_fc2(W2)
    print('fc2 roundtrip max|diff|', (unpack_fc2(w, s, f) - dq).abs().max().item(), 'quant rel err', ((dq - W2).norm() / W2.norm()).item())
    p1, q1 = mx_quantize(W13[:INTER]); p3, q3 = mx_quantize(W13[INTER:])
    ck = torch.cat([mx_dequant(p1, q1), mx_dequant(p3, q3)], 0)
    w, s, f, dq, ncl = ckpt_to_kernel_fc1(p1, q1, p3, q3)
    print('ckpt fc1: mx quant rel err', ((ck - W13).norm() / W13.norm()).item(), '| ckpt->kernel exact:', bool((unpack_fc1(w, s, f) == ck).all()), 'clamped groups', ncl)
    p2, q2 = mx_quantize(W2)
    w, s, f, dq, ncl = ckpt_to_kernel_fc2(p2, q2)
    print('ckpt fc2: ckpt->kernel exact:', bool((unpack_fc2(w, s, f) == mx_dequant(p2, q2)).all()), 'clamped groups', ncl)
