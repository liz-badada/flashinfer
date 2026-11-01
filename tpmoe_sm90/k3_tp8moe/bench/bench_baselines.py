"""Industry MoE kernels on the K3 TP8 per-rank shard (E = 896 experts, K = 3584 latent, per-rank intermediate N = 384, top-16, SiTU),
one H200, CUDA graph of --copies back-to-back calls, us per call (min / median over --reps replays).  Routing: uniform random top-16 of
896 (seeded per M) like test_k3, so the unique-expert count (the bytes) matches our kernel's runs.  Each baseline runs in its own try
block; a failing one is reported and skipped.  All are expert-only on one GPU: add nccl_ar.py's 8-GPU all-reduce to compare with our
TP8 figure, or compare with our ndev-1 figure.
  marlin_mxfp4_w4a16 : SGLang marlin runner for MXFP4 checkpoints on Hopper (prepare_moe_mxfp4_layer_for_marlin + fused_marlin_moe,
                       E8M0 scales, BF16 activations, fused SiTU) -- SGLang's `--moe-runner-backend marlin` path for K3 on H200
  fi_cutlass_w4a16   : FlashInfer cutlass_fused_moe(use_w4_group_scaling=True) with the SM90 interleave helpers -- SGLang's
                       `flashinfer_mxfp4` SM90 default (W4A16)
  fi_humming_w4a8    : FlashInfer cutlass_fused_moe(use_wfp4afp8_humming=True): Humming W4A8 (FP8 activations) -- SGLang's
                       `flashinfer_mxfp4` SM90 with precision=fp8 (the kernel tp4moe2 was measured against in DeepSeek serving)
  triton_bf16        : SGLang Triton fused_experts with BF16 weights (generic path, 4x the bytes; for reference only)
"""
import argparse, inspect, time, traceback
import torch

E, K, N, TOPK = 896, 3584, 384, 16
dev = torch.device('cuda')


def routing(M, seed):
    """Exactly test_k3's routing (K3 sigmoid gate on seeded random logits, seed 100 + M): same ids, same unique-expert count as our runs."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from test_k3 import random_routing
    ids, w = random_routing(M, E, dev, seed)
    return ids.to(torch.int32).contiguous(), w.float().contiguous(), int(ids.unique().numel())


def mx_weights(rows, k, seed):
    """MXFP4 checkpoint-style tensors: packed E2M1 nibbles uint8 [E, rows, k/2], E8M0 scales uint8 [E, rows, k/32] (2^-9 .. 2^-2)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    wq = torch.randint(0, 256, (E, rows, k // 2), generator=g, dtype=torch.uint8).to(dev)
    sc = torch.randint(118, 126, (E, rows, k // 32), generator=g, dtype=torch.uint8).to(dev)
    return wq, sc


def timed(fn, copies, reps):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2): fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(copies): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e6 / copies)
    ts.sort(); return ts[0], ts[len(ts) // 2]


def report(name, M, U, mn, md, note=''):
    print(f'RESULT M={M:3d} U={U:3d} {name:<22s} per-call us min {mn:7.1f} median {md:7.1f} {note}', flush=True)


class _Layer(torch.nn.Module):
    pass


# ---------------------------------------------------------------- SGLang marlin (MXFP4 W4A16, SiTU)
def bench_marlin(M, x, ids, w, copies, reps, U):
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    L = _Layer()
    w13, s13 = mx_weights(2 * N, K, 11); w2, s2 = mx_weights(K, N, 12)   # [E, 2N, K/2] (gate rows then up rows), [E, K, N/2]
    L.w13_weight = torch.nn.Parameter(w13, requires_grad=False); L.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
    L.w13_weight_scale = torch.nn.Parameter(s13, requires_grad=False); L.w2_weight_scale = torch.nn.Parameter(s2, requires_grad=False)
    L.orig_dtype = torch.bfloat16
    prepare_moe_mxfp4_layer_for_marlin(L)
    print('  marlin w13', tuple(L.w13_weight.shape), L.w13_weight.dtype, 'scale', tuple(L.w13_weight_scale.shape), L.w13_weight_scale.dtype, flush=True)
    gating = torch.zeros(M, E, device=dev, dtype=torch.bfloat16)
    call = lambda: fused_marlin_moe(x, L.w13_weight, L.w2_weight, L.w13_weight_scale, L.w2_weight_scale, gating, w, ids,
                                    workspace=L.workspace, num_bits=4, activation='situ', gemm1_alpha=4.0, clamp_limit=25.0)
    out = call(); assert out.shape == (M, K) and torch.isfinite(out.float()).all(), out.shape
    mn, md = timed(call, copies, reps); report('marlin_mxfp4_w4a16', M, U, mn, md)


# ---------------------------------------------------------------- FlashInfer cutlass SM90 W4A16 / Humming W4A8
def _fi_call(x, ids, w, w13_il, w2_il, qs, humming):
    from flashinfer import fused_moe
    from flashinfer.tllm_enums import ActivationType
    out = torch.empty(x.shape[0], K, device=dev, dtype=torch.bfloat16)
    kw = dict(quant_scales=qs, use_w4_group_scaling=True, output=out)
    if humming: kw['use_wfp4afp8_humming'] = True
    def call(act):
        fused_moe.cutlass_fused_moe(x, ids, w, w13_il, w2_il, torch.bfloat16, activation_type=act, **kw); return out
    try:
        call(ActivationType.Situ); torch.cuda.synchronize(); act = ActivationType.Situ; note = 'SiTU'
    except Exception as e:
        print('  Situ activation failed, falling back to Swiglu:', type(e).__name__, str(e)[:200], flush=True)
        act = ActivationType.Swiglu; note = 'SwiGLU(fallback)'
    return (lambda: call(act)), note


def bench_fi_w4a16(M, x, ids, w, copies, reps, U):
    from flashinfer import fused_moe
    w13, s13 = mx_weights(2 * N, K, 21); w2, s2 = mx_weights(K, N, 22)   # [up; gate] halves, K % 128 == 0, N % 128 == 0: no padding
    w13_il = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(w13, 'fp4'); w2_il = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(w2, 'fp4')
    s13_il = fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(s13, group_size=32); s2_il = fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(s2, group_size=32)
    qs = [s13_il.view(torch.int32), s2_il.view(torch.int32)]
    call, note = _fi_call(x, ids, w, w13_il, w2_il, qs, humming=False)
    out = call(); assert torch.isfinite(out.float()).all()
    mn, md = timed(call, copies, reps); report('fi_cutlass_w4a16', M, U, mn, md, note)


def bench_fi_humming(M, x, ids, w, copies, reps, U):
    from flashinfer import fused_moe
    w13, s13 = mx_weights(2 * N, K, 31); w2, s2 = mx_weights(K, N, 32)
    w13_il, s13_il, r13 = fused_moe.preprocess_moe_weights_for_sm90_mixed_gemm_humming(w13, s13)
    w2_il, s2_il, r2 = fused_moe.preprocess_moe_weights_for_sm90_mixed_gemm_humming(w2, s2)
    qs = [s13_il.view(torch.int32), (r13 * 64.0).contiguous(), torch.ones((), device=dev, dtype=torch.float32), s2_il.view(torch.int32), (r2 * 64.0).contiguous()]
    call, note = _fi_call(x, ids, w, w13_il, w2_il, qs, humming=True)
    out = call(); assert torch.isfinite(out.float()).all()
    mn, md = timed(call, copies, reps); report('fi_humming_w4a8', M, U, mn, md, note)


# ---------------------------------------------------------------- SGLang Triton fused_experts, BF16 weights
def bench_triton(M, x, ids, w, copies, reps, U):
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_experts
    from types import SimpleNamespace
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config as _C
    _C.get_exec = lambda: SimpleNamespace(deterministic=SimpleNamespace(enable_deterministic_inference=False))   # no server context here
    sig = inspect.signature(fused_experts); print('  fused_experts', sig, flush=True)
    w13 = (torch.randn(E, 2 * N, K, device=dev) * 0.02).to(torch.bfloat16); w2 = (torch.randn(E, K, N, device=dev) * 0.02).to(torch.bfloat16)
    P = sig.parameters
    if 'topk_weights' in P and 'topk_ids' in P:
        kw = {k: v for k, v in dict(inplace=False, activation='silu').items() if k in P}
        call = lambda: fused_experts(x, w13, w2, w, ids, **kw)
    elif 'topk_output' in P:
        from sglang.srt.layers.moe import topk as T
        cls = getattr(T, 'StandardTopKOutput', None) or getattr(T, 'TopKOutput')
        to = cls(w, ids, torch.zeros(M, E, device=dev, dtype=torch.bfloat16))
        kw = {}
        if 'moe_runner_config' in P:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            kw['moe_runner_config'] = MoeRunnerConfig(inplace=False)
        call = lambda: fused_experts(x, w13, w2, to, **kw)
    else:
        raise RuntimeError(f'unknown fused_experts signature {sig}')
    out = call(); out = out[0] if isinstance(out, (list, tuple)) else out; assert torch.isfinite(out.float()).all()
    mn, md = timed(call, copies, reps); report('triton_bf16_w16a16', M, U, mn, md)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ms', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--copies', type=int, default=32); ap.add_argument('--reps', type=int, default=30); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only', nargs='*', default=None)
    a = ap.parse_args()
    import sglang, sgl_kernel, flashinfer
    print('device', torch.cuda.get_device_name(), 'torch', torch.__version__, 'sglang', sglang.__version__, 'sgl_kernel', sgl_kernel.__version__, 'flashinfer', flashinfer.__version__, flush=True)
    benches = [('marlin', bench_marlin), ('fi_w4a16', bench_fi_w4a16), ('fi_humming', bench_fi_humming), ('triton', bench_triton)]
    for name, fn in benches:
        if a.only and name not in a.only: continue
        print(f'==== {name}', flush=True)
        for M in a.ms:
            x = torch.randn(M, K, device=dev).to(torch.bfloat16)
            ids, w, U = routing(M, 100 + M)
            try:
                fn(M, x, ids, w, a.copies, a.reps, U)
            except Exception as e:
                print(f'RESULT M={M:3d} {name}: FAILED {type(e).__name__}: {str(e)[:300]}', flush=True); traceback.print_exc(); break
            torch.cuda.empty_cache()
    print('baselines done', flush=True)


if __name__ == '__main__':
    main()
