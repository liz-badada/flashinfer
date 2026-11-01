"""k3tp8moe checks on synthetic K3 weights.

  python test_k3.py --local --ms 1 2 4 8 16 32 [--nexp 128]      # one GPU: kernel (ndev 1) vs the torch reference
  python test_k3.py --tp8 --ms 1 8 32 [--nexp 128] [--b2b]       # 8 GPUs: one-shot LL all-reduce, every rank vs the 8-rank reference
"""
import argparse
import ctypes
import glob
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k3_host as H   # noqa: E402
import k3_ref as R   # noqa: E402
import k3_synth as S   # noqa: E402


def random_routing(M, n_exp, device, seed, topk=16):
    g = torch.Generator(device='cpu').manual_seed(seed)
    logits = torch.randn(M, n_exp, generator=g) * 2.0
    bias = torch.randn(n_exp, generator=g) * 0.1
    ids, rw = R.gate_ref(logits.to(device), bias.to(device), topk=min(topk, n_exp))
    return ids.contiguous(), rw.contiguous()


def stats(a, b):
    d = a.float() - b.float()
    tok = d.square().mean(1).sqrt() / b.float().square().mean(1).sqrt().clamp_min(1e-30)
    return float(d.square().mean().sqrt() / b.float().square().mean().sqrt().clamp_min(1e-30)), float(tok.max())


def local(args):
    dev = torch.device('cuda')
    E = H.ext()
    st = H.RankState(dev, alloc_comm=False)
    w = S.make_expert_weights(args.nexp, dev, seed=1, keep_dq=True, via_ckpt=args.ckpt)
    for M in args.ms:
        g = torch.Generator(device='cpu').manual_seed(M)
        z = torch.randn(M, S.DIM, generator=g).to(torch.bfloat16).to(dev)
        ids, rw = random_routing(M, args.nexp, dev, 100 + M)
        out = torch.empty(M, S.DIM, dtype=torch.bfloat16, device=dev)
        H.run(st, w, z, ids, rw, out, 0, 1)
        torch.cuda.synchronize()
        out2 = torch.empty_like(out)
        H.run(st, w, z, ids, rw, out2, 0, 1)
        torch.cuda.synchronize()
        ref = R.routed_local(z, ids, rw, w['dq1'], w['dq2'])
        rr, tr = stats(out, ref)
        print(f'M={M:3d} nexp={args.nexp} U={len(torch.unique(ids)):3d} kernel vs ref: rel_rms={rr:.4f} tok_max={tr:.4f} '
              f'finite={bool(torch.isfinite(out.float()).all())} call1==call2={bool((out == out2).all())}', flush=True)
        if args.time:   # single-GPU kernel time (ndev 1: no LL reduce), graph of --copies calls
            import statistics
            n = args.copies
            zs = [torch.randn(M, S.DIM).to(torch.bfloat16).to(dev) for _ in range(n)]
            rts = [random_routing(M, args.nexp, dev, 900 + M + i) for i in range(n)]
            outs = [torch.empty(M, S.DIM, dtype=torch.bfloat16, device=dev) for _ in range(n)]
            for i in range(n): H.run(st, w, zs[i], rts[i][0], rts[i][1], outs[i], 0, 1)
            torch.cuda.synchronize()
            gph = torch.cuda.CUDAGraph(); sstream = torch.cuda.Stream()
            with torch.cuda.stream(sstream):
                with torch.cuda.graph(gph, stream=sstream):
                    for i in range(n): H.run(st, w, zs[i], rts[i][0], rts[i][1], outs[i], 0, 1)
            torch.cuda.synchronize()
            ts = []
            for rep in range(args.reps):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); gph.replay(); e1.record(); torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) * 1e3 / n)
            print(f'M={M:3d} kernel local (ndev 1) b2b graph of {n}: per-call us min {min(ts):7.1f} median {statistics.median(ts):7.1f}', flush=True)


class IpcHandle(ctypes.Structure):
    _fields_ = [('reserved', ctypes.c_byte * 64)]


class CudaRT:
    def __init__(self):
        cands = glob.glob(os.path.join(sys.prefix, 'lib', 'python*', 'site-packages', 'nvidia', '**', 'libcudart.so*'), recursive=True)
        cands += glob.glob(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libcudart*.so*')) + ['/usr/local/cuda/lib64/libcudart.so']
        self.lib = ctypes.CDLL(cands[0])
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.lib.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(IpcHandle), ctypes.c_void_p]
        self.lib.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), IpcHandle, ctypes.c_uint]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def ck(self, r):
        assert r == 0, self.lib.cudaGetErrorString(r)

    def malloc(self, n):
        p = ctypes.c_void_p(); self.ck(self.lib.cudaMalloc(ctypes.byref(p), n)); self.ck(self.lib.cudaMemset(p, 0, n)); return p.value

    def handle(self, ptr):
        h = IpcHandle(); self.ck(self.lib.cudaIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(ptr))); return bytes(h)

    def open(self, raw):
        p = ctypes.c_void_p(); self.ck(self.lib.cudaIpcOpenMemHandle(ctypes.byref(p), IpcHandle.from_buffer_copy(raw), 1)); return p.value


def worker(rank, args):
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dev = torch.device('cuda', rank)
    dist.init_process_group('gloo', init_method='file://' + args.rdv, world_size=8, rank=rank)
    E = H.ext()
    st = H.RankState(dev, alloc_comm=False)
    rt = CudaRT(); rt.ck(rt.lib.cudaSetDevice(rank))
    ptr = rt.malloc(H.RankState.RS_BYTES())
    raw = [None] * 8
    dist.all_gather_object(raw, rt.handle(ptr))
    st.link_ptrs([ptr if peer == rank else rt.open(h) for peer, h in enumerate(raw)])
    torch.cuda.synchronize(); dist.barrier()
    w = S.make_expert_weights(args.nexp, dev, seed=1 + rank, keep_dq=True, via_ckpt=args.ckpt)   # rank-specific expert shards
    for M in args.ms:
        g = torch.Generator(device='cpu').manual_seed(M)
        z = torch.randn(M, S.DIM, generator=g).to(torch.bfloat16).to(dev)   # identical latent input and routing on every rank
        ids, rw = random_routing(M, args.nexp, dev, 100 + M)
        dist.barrier()
        out = torch.empty(M, S.DIM, dtype=torch.bfloat16, device=dev)
        H.run(st, w, z, ids, rw, out, rank, 8)
        torch.cuda.synchronize(); dist.barrier()
        out2 = torch.empty_like(out)
        H.run(st, w, z, ids, rw, out2, rank, 8)   # second call: epoch / parity handling
        torch.cuda.synchronize()
        mine = R.routed_local(z, ids, rw, w['dq1'], w['dq2']).cpu()
        allv = [None] * 8
        dist.all_gather_object(allv, mine)
        full = R.combine_tp([v.to(dev) for v in allv])
        rr, tr = stats(out, full)
        outs = [None] * 8
        dist.all_gather_object(outs, out.cpu())
        if rank == 0:
            eq = [bool((outs[0] == o).all()) for o in outs[1:]]
            print(f'M={M:3d} nexp={args.nexp} U={len(torch.unique(ids)):3d} TP8 vs ref: rel_rms={rr:.4f} tok_max={tr:.4f} '
                  f'finite={bool(torch.isfinite(out.float()).all())} call1==call2={bool((out == out2).all())} ranks_identical={all(eq)}', flush=True)
    if args.b2b:   # back-to-back calls captured in one graph and replayed: the parity-double-buffered one-shot buffers under replay
        for M in args.ms:
            n = args.copies
            zs = [torch.randn(M, S.DIM, generator=torch.Generator(device='cpu').manual_seed(4321 + M + 7919 * i)).to(torch.bfloat16).to(dev) for i in range(n)]
            rts = [random_routing(M, args.nexp, dev, 500 + M + i) for i in range(n)]
            outs = [torch.empty(M, S.DIM, dtype=torch.bfloat16, device=dev) for _ in range(n)]
            for i in range(n):   # warm (eager) so every launch's state is initialized
                H.run(st, w, zs[i], rts[i][0], rts[i][1], outs[i], rank, 8)
            torch.cuda.synchronize(); dist.barrier()
            gph = torch.cuda.CUDAGraph()
            sstream = torch.cuda.Stream()
            with torch.cuda.stream(sstream):
                with torch.cuda.graph(gph, stream=sstream):
                    for i in range(n):
                        H.run(st, w, zs[i], rts[i][0], rts[i][1], outs[i], rank, 8)
            torch.cuda.synchronize(); dist.barrier()
            refs = []
            for i in range(n):
                mine = R.routed_local(zs[i], rts[i][0], rts[i][1], w['dq1'], w['dq2']).cpu()
                allv = [None] * 8
                dist.all_gather_object(allv, mine)
                refs.append(R.combine_tp([v.to(dev) for v in allv]))
            worst = 0.0
            for rep in range(args.reps):
                for o in outs: o.zero_()
                gph.replay(); torch.cuda.synchronize()
                for i in range(n): worst = max(worst, stats(outs[i], refs[i])[0])
                dist.barrier()
            if rank == 0:
                print(f'M={M:3d} b2b graph of {n} calls x {args.reps} replays: worst rel_rms={worst:.4f}', flush=True)
    if args.time:   # kernel-only: one graph of --copies back-to-back calls, CUDA-event time per call, worst rank
        import statistics
        for M in args.ms:
            n = args.copies
            zs = [torch.randn(M, S.DIM).to(torch.bfloat16).to(dev) for _ in range(n)]
            rts = [random_routing(M, args.nexp, dev, 900 + M + i) for i in range(n)]
            outs = [torch.empty(M, S.DIM, dtype=torch.bfloat16, device=dev) for _ in range(n)]
            for i in range(n): H.run(st, w, zs[i], rts[i][0], rts[i][1], outs[i], rank, 8)
            torch.cuda.synchronize(); dist.barrier()
            gph = torch.cuda.CUDAGraph(); sstream = torch.cuda.Stream()
            with torch.cuda.stream(sstream):
                with torch.cuda.graph(gph, stream=sstream):
                    for i in range(n): H.run(st, w, zs[i], rts[i][0], rts[i][1], outs[i], rank, 8)
            torch.cuda.synchronize(); dist.barrier()
            ts = []
            for rep in range(args.reps):
                dist.barrier()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); gph.replay(); e1.record(); torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) * 1e3 / n)
            mine = (min(ts), statistics.median(ts))
            allt = [None] * 8
            dist.all_gather_object(allt, mine)
            if rank == 0:
                print(f'M={M:3d} kernel TP8 b2b graph: per-call us worst-rank min {max(t[0] for t in allt):7.1f} median {max(t[1] for t in allt):7.1f} '
                      f'(rank mins {[round(t[0], 1) for t in allt]})', flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--local', action='store_true'); ap.add_argument('--tp8', action='store_true')
    ap.add_argument('--ms', type=int, nargs='+', default=[1, 8, 32]); ap.add_argument('--nexp', type=int, default=128)
    ap.add_argument('--ckpt', action='store_true', help='generate weights in the K3 checkpoint format and convert (loader path)')
    ap.add_argument('--b2b', action='store_true'); ap.add_argument('--time', action='store_true'); ap.add_argument('--copies', type=int, default=4); ap.add_argument('--reps', type=int, default=10)
    args = ap.parse_args()
    if args.local:
        local(args)
    if args.tp8:
        import torch.multiprocessing as mp
        args.rdv = tempfile.mktemp(prefix='k3_rdv_')
        mp.spawn(worker, args=(args,), nprocs=8, join=True)


if __name__ == '__main__':
    main()
