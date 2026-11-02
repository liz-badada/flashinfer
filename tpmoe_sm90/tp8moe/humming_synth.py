"""Synthetic weights in the Humming fused-E8M0 MXFP4 layout that tp4moe2 consumes in place, plus the exact inverse (decode).

The layout is derived from the kernel's register dequant (tp4moe2_sh.cu: lane_of / dequant_word / make_pools and the TMA maps
in set_weights).  Conventions:
  * a 1 KB "Humming block" (HLB) holds 64 rows x 32 k of fp4 codes; the k32 slice of an expert is a sequence of blocks:
      FC1 (w13): slice stride 16 KB = [w1|w3 (2)][128-row group q (4)][64-row block (2)][1 KB]  -> rows 512*w13 + 128*q + 64*blk + R
      FC2 (w2):  slice stride 64 KB = [64 blocks][1 KB]                                          -> rows 64*blk + R
  * inside a block: 16-B chunk c (0..63): h32 = c >> 5, lane = c & 31, g = lane >> 2, tig = lane & 3; the two 8-B halves sel = 0/1
    hold rows R0 = 32*h32 + 16*sel + g (word 0) and R0 + 8 (word 1); word nibble n: bits 0..2 = magnitude code of element n, where
    element n < 4 is k = 4*tig + n and element n >= 4 is k = 16 + 4*tig + (n - 4); nibble bit 3 = sign of element (n even: n/2,
    n odd: 4 + (n-1)/2).  Magnitude codes 0..7 = {0, .5, 1, 1.5, 2, 3, 4, 6}.
  * one scale byte o per (row, k32 slice): scale bytes of a slice follow the same block order, 64 B per block, byte index
    8*g + 4*h32 + 2*sel + b for row R = 32*h32 + 16*sel + g + 8*b.  The dequantized e4m3 value is code * 2^(o-6); the kernel then
    multiplies by factor[e] * 64, so the real weight is code * 2^o * factor[e].
"""
import torch

DIM, INTER, NEXP = 4096, 512, 256
FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _block_perm():
    """row/k of every nibble of a 1 KB block: returns (row[64,32] -> (word index 0..255, nibble 0..7) for magnitude), sign nibble."""
    # word w (0..255) of the block: chunk c = w // 4, sub = w % 4 (two 8-B halves = words {0,1} sel 0, {2,3} sel 1)
    mag_word = torch.empty(64, 32, dtype=torch.long)
    mag_nib = torch.empty(64, 32, dtype=torch.long)
    sgn_word = torch.empty(64, 32, dtype=torch.long)
    sgn_nib = torch.empty(64, 32, dtype=torch.long)
    for c in range(64):
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
                    # sign of element n lives in nibble 2n (n < 4) or 2(n-4)+1 (n >= 4)
                    sgn_word[R, k] = w
                    sgn_nib[R, k] = 2 * n if n < 4 else 2 * (n - 4) + 1
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
    """codes [B, 64, 32] int (0..7), signs [B, 64, 32] (0/1) -> uint32 words [B, 256] in Humming block order."""
    B = codes.shape[0]
    words = torch.zeros(B, 256, dtype=torch.int64, device=codes.device)
    mw = _MW.to(codes.device).reshape(-1)
    mn = _MN.to(codes.device).reshape(-1)
    sn = _SN.to(codes.device).reshape(-1)
    cf = codes.reshape(B, -1).to(torch.int64)
    sf = signs.reshape(B, -1).to(torch.int64)
    # magnitude bits: 3 bits at nibble mn; sign at nibble sn bit 3 (same word)
    words.scatter_add_(1, mw.expand(B, -1), cf << (4 * mn))
    words.scatter_add_(1, mw.expand(B, -1), (sf << 3) << (4 * sn))
    return words.to(torch.int32)   # bit pattern (int32 view of uint32)


def decode_blocks(words):
    """uint32 words [B, 256] -> (codes [B,64,32], signs [B,64,32])."""
    B = words.shape[0]
    w = words.to(torch.int64) & 0xFFFFFFFF
    mw = _MW.to(words.device).reshape(-1)
    mn = _MN.to(words.device).reshape(-1)
    sn = _SN.to(words.device).reshape(-1)
    codes = (torch.gather(w, 1, mw.expand(B, -1)) >> (4 * mn)) & 7
    signs = (torch.gather(w, 1, mw.expand(B, -1)) >> (4 * sn + 3)) & 1
    return codes.reshape(B, 64, 32), signs.reshape(B, 64, 32)


def quantize_mxfp4_fused(W, o_max=12):
    """W [rows, K] float -> (codes, signs, o) per 32-group, with e8m0 'fused' exponent o in [0, o_max] and the per-tensor factor f.
    scale(group) = 2^o * f, f = max_scale / 2^o_max; groups whose scale would need o < 0 are clamped (extra error on tiny groups)."""
    rows, K = W.shape
    Wg = W.float().reshape(rows, K // 32, 32)
    amax = Wg.abs().amax(-1).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 6.0))      # scale = 2^e so that amax/scale <= 6
    e_max = e.max()
    f = torch.exp2(e_max - o_max)                # per-tensor (expert) factor
    o = (e - e_max + o_max).clamp_min(0)         # fused exponent
    scale = torch.exp2(o) * f
    q = Wg / scale.unsqueeze(-1)
    mag = FP4_MAG.to(W.device)
    # nearest fp4 magnitude (ties to even-ish: use midpoints)
    aq = q.abs().clamp_max(6.0)
    idx = torch.bucketize(aq, torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=W.device))
    codes = idx.to(torch.int64)
    signs = (q < 0).to(torch.int64)
    return codes, signs, o.to(torch.int64), f


def dequant_rows(codes, signs, o, f):
    mag = FP4_MAG.to(codes.device)[codes]
    val = torch.where(signs.bool(), -mag, mag) * torch.exp2(o.float()).unsqueeze(-1) * f
    return val.reshape(codes.shape[0], -1)


def pack_fc1(W13, o_max=12):
    """W13 [1024, 4096] (rows 0..511 gate, 512..1023 up) -> (words int32 [128 slices * 4096 B / 4], scales uint8 [128 * 1024], f, Wdq)."""
    codes, signs, o, f = quantize_mxfp4_fused(W13, o_max)   # codes [1024, 128, 32]
    rows, S = codes.shape[0], codes.shape[1]
    # block order per slice: [w13 (2)][q (4)][blk (2)] -> rows 512*w13 + 128*q + 64*blk
    c = codes.permute(1, 0, 2).reshape(S, 2, 4, 2, 64, 32).reshape(S * 16, 64, 32)
    s = signs.permute(1, 0, 2).reshape(S, 2, 4, 2, 64, 32).reshape(S * 16, 64, 32)
    words = encode_blocks(c, s)                               # [S*16, 256]
    sp = _SP.to(W13.device)
    ob = o.permute(1, 0).reshape(S, 2, 4, 2, 64)              # [S][w13][q][blk][R]
    scales = torch.zeros(S, 2, 4, 2, 64, dtype=torch.uint8, device=W13.device)
    scales[..., sp] = ob.to(torch.uint8)
    Wdq = dequant_rows(codes, signs, o, f)
    return words.reshape(-1), scales.reshape(-1), f, Wdq


def pack_fc2(W2, o_max=12):
    """W2 [4096, 512] -> (words int32, scales uint8 [16 * 4096], f, Wdq)."""
    codes, signs, o, f = quantize_mxfp4_fused(W2, o_max)      # [4096, 16, 32]
    rows, S = codes.shape[0], codes.shape[1]
    c = codes.permute(1, 0, 2).reshape(S * 64, 64, 32)
    s = signs.permute(1, 0, 2).reshape(S * 64, 64, 32)
    words = encode_blocks(c, s)
    sp = _SP.to(W2.device)
    ob = o.permute(1, 0).reshape(S, 64, 64)
    scales = torch.zeros(S, 64, 64, dtype=torch.uint8, device=W2.device)
    scales[..., sp] = ob.to(torch.uint8)
    Wdq = dequant_rows(codes, signs, o, f)
    return words.reshape(-1), scales.reshape(-1), f, Wdq


def unpack_fc1(words, scales, f):
    """inverse of pack_fc1 -> dequantized W13 [1024, 4096] (float32)."""
    S = 128
    c, s = decode_blocks(words.reshape(S * 16, 256))
    codes = c.reshape(S, 2, 4, 2, 64, 32).reshape(S, 1024, 32).permute(1, 0, 2)
    signs = s.reshape(S, 2, 4, 2, 64, 32).reshape(S, 1024, 32).permute(1, 0, 2)
    sp = _SP.to(words.device)
    o = scales.reshape(S, 2, 4, 2, 64)[..., sp].reshape(S, 1024).permute(1, 0).to(torch.int64)
    return dequant_rows(codes, signs, o, f)


def unpack_fc2(words, scales, f):
    S = 16
    c, s = decode_blocks(words.reshape(S * 64, 256))
    codes = c.reshape(S, 4096, 32).permute(1, 0, 2)
    signs = s.reshape(S, 4096, 32).permute(1, 0, 2)
    sp = _SP.to(words.device)
    o = scales.reshape(S, 64, 64)[..., sp].reshape(S, 4096).permute(1, 0).to(torch.int64)
    return dequant_rows(codes, signs, o, f)


def make_expert_weights(n_exp, device, seed=0, std=0.02, keep_dq=False):
    """Random experts in the kernel's layout: w1 int32 [E*128*4096 B/4], s1 uint8 [E*128*1024], f1 fp32 [E], w2, s2, f2 (+ dequantized)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    w1 = torch.empty(n_exp, 128 * 4096, dtype=torch.int32, device=device)
    s1 = torch.empty(n_exp, 128 * 1024, dtype=torch.uint8, device=device)
    w2 = torch.empty(n_exp, 16 * 16384, dtype=torch.int32, device=device)
    s2 = torch.empty(n_exp, 16 * 4096, dtype=torch.uint8, device=device)
    f1 = torch.empty(n_exp, dtype=torch.float32, device=device)
    f2 = torch.empty(n_exp, dtype=torch.float32, device=device)
    dq1, dq2 = [], []
    for e in range(n_exp):
        W13 = (torch.randn(1024, 4096, generator=g) * std).to(device)
        W2 = (torch.randn(4096, 512, generator=g) * std).to(device)
        a, b, c, d = pack_fc1(W13); w1[e], s1[e], f1[e] = a, b, c
        if keep_dq: dq1.append(d.to(torch.bfloat16))
        a, b, c, d = pack_fc2(W2); w2[e], s2[e], f2[e] = a, b, c
        if keep_dq: dq2.append(d.to(torch.bfloat16))
    out = dict(w1=w1.reshape(-1), s1=s1.reshape(-1), f1=f1, w2=w2.reshape(-1), s2=s2.reshape(-1), f2=f2)
    if keep_dq:
        out['dq1'] = torch.stack(dq1); out['dq2'] = torch.stack(dq2)
    return out


if __name__ == '__main__':   # round-trip self test
    torch.manual_seed(0)
    W13 = torch.randn(1024, 4096) * 0.02
    w, s, f, dq = pack_fc1(W13)
    back = unpack_fc1(w, s, f)
    print('fc1 roundtrip max abs diff', (back - dq).abs().max().item(), 'quant rel err', ((dq - W13).norm() / W13.norm()).item())
    W2 = torch.randn(4096, 512) * 0.02
    w, s, f, dq = pack_fc2(W2)
    back = unpack_fc2(w, s, f)
    print('fc2 roundtrip max abs diff', (back - dq).abs().max().item(), 'quant rel err', ((dq - W2).norm() / W2.norm()).item())
