"""Torch reference of tp8moe's numerics (one rank's routed experts in the latent space, TP combine) and of the whole K3 MoE block."""
import torch

DIM, INTER, NEXP, TOPK = 3584, 384, 896, 16
HIDDEN = 7168
INV448 = 0.0022321429569274187
SITU_B1, SITU_B2 = 4.0, 25.0


def bf16r(x):
    return x.to(torch.bfloat16).to(torch.float32)


def e4m3(x):
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)


def quant_x(xb):
    """per-token e4m3 (as the kernel's quant_token): scale = max(amax, 1e-30) * fp32(1/448), q = e4m3(x / scale)."""
    x = xb.float()
    amax = x.abs().amax(1).clamp_min(1e-30)
    xs = amax * INV448
    return e4m3(x * (1.0 / xs).unsqueeze(1)), xs


def situ_glu(gate, up):
    """Kimi K3 SiTU-GLU (SituAndMul): b1 tanh(g / b1) sigmoid(g) * b2 tanh(u / b2), fp32."""
    sa = SITU_B1 * torch.tanh(gate / SITU_B1) * torch.sigmoid(gate)
    return sa * (SITU_B2 * torch.tanh(up / SITU_B2))


def routed_local(zb, ids, rw, dq1, dq2, factor=1.0):
    """One rank's routed experts on the latent input zb [M, DIM] (BF16): BF16(factor * sum_k rw[t,k] y_k) [M, DIM].
    dq1 [E, 2*INTER, DIM], dq2 [E, DIM, INTER]: the dequantized weights (code * 2^o * f, exact in bf16).  Slot order as the kernel."""
    M, topk = ids.shape
    xq, xs = quant_x(zb)
    acc = torch.zeros(M, DIM, dtype=torch.float32, device=zb.device)
    for k in range(topk):
        e = ids[:, k]
        valid = (e >= 0) & (e < dq1.shape[0])
        for t in torch.nonzero(valid).flatten().tolist():
            ee = int(e[t])
            gu = xq[t] @ dq1[ee].float().t()           # fp32 [2 * INTER]
            v = bf16r(gu * xs[t])                      # BF16 gate / up (GEMM output dtype)
            act = bf16r(situ_glu(v[:INTER], v[INTER:]))
            hs = act.abs().amax().clamp_min(1e-30) * INV448
            hq = e4m3(act / hs)
            y = bf16r((hq @ dq2[ee].float().t()) * hs)
            acc[t] += y * rw[t, k]
    return bf16r(acc * factor)


def combine_tp(locals_):
    """rank-ordered fp32 sum of the ranks' BF16 outputs, one BF16 rounding (as the one-shot tail)."""
    acc = locals_[0].float().clone()
    for v in locals_[1:]:
        acc = acc + v.float()
    return bf16r(acc)


def gate_ref(logits, bias, topk=TOPK, renormalize=True, scaling=1.0):
    """K3 KimiMoEGate: sigmoid scores (fp32), + e_score_correction_bias for the selection only, top-k, weights = scores / sum, x scaling."""
    scores = torch.sigmoid(logits.float())
    top = torch.topk(scores + bias.float().unsqueeze(0), topk, dim=1, sorted=False).indices
    w = torch.gather(scores, 1, top)
    if renormalize:
        w = w / (w.sum(1, keepdim=True) + 1e-20)
    return top.int(), (w * scaling).float()


def rmsnorm(x, w, eps=1e-5):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)) * w.float()


def moe_block_ref(x, gate_w, gate_bias, down_w, up_w, norm_w, sh_gate, sh_up, sh_down, rank_fn):
    """Whole KimiSparseMoeBlock (eval): x [M, 7168] BF16 -> y [M, 7168] BF16.  rank_fn(z_bf16, ids, rw) returns the TP-reduced routed
    latent output [M, DIM] (BF16) -- the kernel path or routed_local + combine_tp.  BF16 module boundaries as the HF modeling code."""
    logits = x.float() @ gate_w.float().t()
    ids, rw = gate_ref(logits, gate_bias)
    z = (x.float() @ down_w.float().t()).to(torch.bfloat16)          # routed_expert_down_proj (BF16 Linear)
    u = rank_fn(z, ids, rw)                                            # [M, DIM] BF16
    un = rmsnorm(u, norm_w).to(torch.bfloat16)                         # routed_expert_norm (latent_moe_use_norm)
    y = (un.float() @ up_w.float().t()).to(torch.bfloat16)             # routed_expert_up_proj
    g = (x.float() @ sh_gate.float().t()).to(torch.bfloat16); p = (x.float() @ sh_up.float().t()).to(torch.bfloat16)
    a = situ_glu(g.float(), p.float()).to(torch.bfloat16)
    shared = (a.float() @ sh_down.float().t()).to(torch.bfloat16)      # shared_experts (KimiMLP, intermediate 6144)
    return (y.float() + shared.float()).to(torch.bfloat16)
