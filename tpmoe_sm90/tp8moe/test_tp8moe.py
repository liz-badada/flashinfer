"""tp8moe checks on synthetic K3 weights.

  python test_tp8moe.py --local --ms 1 2 4 8 16 32 [--nexp 128]      # one GPU: kernel (ndev 1) vs the torch reference
  python test_tp8moe.py --tp8 --ms 1 8 32 [--nexp 128] [--b2b]       # 8 GPUs: one-shot LL all-reduce, every rank vs the 8-rank reference
"""
import argparse
import ctypes
import glob
import os
import statistics
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp8moe_host as H   # noqa: E402
import moe_ref as R   # noqa: E402
import mx_synth as S   # noqa: E402


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


def capture_graph(run_i, n, barrier=lambda: None):
    """One eager warm pass of run_i(0..n-1) (initializes every launch's state), then the n calls captured in one CUDA graph."""
    for i in range(n): run_i(i)
    torch.cuda.synchronize(); barrier()
    gph = torch.cuda.CUDAGraph(); sstream = torch.cuda.Stream()
    sstream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(sstream):
        with torch.cuda.graph(gph, stream=sstream):
            for i in range(n): run_i(i)
    torch.cuda.synchronize(); barrier()
    return gph


def replay_times(gph, n, reps, barrier=lambda: None):
    """CUDA-event time per call (us) of reps replays of a graph of n calls."""
    ts = []
    for _ in range(reps):
        barrier()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); gph.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / n)
    return ts


def local(args):
    dev = torch.device('cuda')
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
            n = args.copies
            zs = [torch.randn(M, S.DIM).to(torch.bfloat16).to(dev) for _ in range(n)]
            rts = [random_routing(M, args.nexp, dev, 900 + M + i) for i in range(n)]
            outs = [torch.empty(M, S.DIM, dtype=torch.bfloat16, device=dev) for _ in range(n)]
            gph = capture_graph(lambda i: H.run(st, w, zs[i], rts[i][0], rts[i][1], outs[i], 0, 1), n)
            ts = replay_times(gph, n, args.reps)
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


def ipc_link(st, dev_id, tp_rank, group=None):
    """Allocate this rank's one-shot LL buffer with the runtime API (IPC-shareable), exchange handles inside the TP group
    (torch.distributed, 8 ranks) and point st at all 8 buffers."""
    import torch.distributed as dist
    rt = CudaRT(); rt.ck(rt.lib.cudaSetDevice(dev_id))
    ptr = rt.malloc(H.RankState.RS_BYTES())
    raw = [None] * 8
    dist.all_gather_object(raw, rt.handle(ptr), group=group)
    st.link_ptrs([ptr if peer == tp_rank else rt.open(h) for peer, h in enumerate(raw)])
    torch.cuda.synchronize(); dist.barrier()


def worker(rank, args):
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dev = torch.device('cuda', rank)
    dist.init_process_group('gloo', init_method='file://' + args.rdv, world_size=8, rank=rank)
    st = H.RankState(dev, alloc_comm=False)
    ipc_link(st, rank, rank)
    w = S.make_expert_weights(args.nexp, dev, seed=1 + rank, keep_dq=True, via_ckpt=args.ckpt)   # rank-specific expert shards

    def full_ref(z, ids, rw):   # every rank's torch reference, combined as the one-shot tail does
        mine = R.routed_local(z, ids, rw, w['dq1'], w['dq2']).cpu()
        allv = [None] * 8
        dist.all_gather_object(allv, mine)
        return R.combine_tp([v.to(dev) for v in allv])

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
        rr, tr = stats(out, full_ref(z, ids, rw))
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
            gph = capture_graph(lambda i: H.run(st, w, zs[i], rts[i][0], rts[i][1], outs[i], rank, 8), n, dist.barrier)
            refs = [full_ref(zs[i], rts[i][0], rts[i][1]) for i in range(n)]
            worst = 0.0
            for rep in range(args.reps):
                for o in outs: o.zero_()
                gph.replay(); torch.cuda.synchronize()
                for i in range(n): worst = max(worst, stats(outs[i], refs[i])[0])
                dist.barrier()
            if rank == 0:
                print(f'M={M:3d} b2b graph of {n} calls x {args.reps} replays: worst rel_rms={worst:.4f}', flush=True)
    if args.time:   # kernel-only: one graph of --copies back-to-back calls, CUDA-event time per call, worst rank
        for M in args.ms:
            n = args.copies
            zs = [torch.randn(M, S.DIM).to(torch.bfloat16).to(dev) for _ in range(n)]
            rts = [random_routing(M, args.nexp, dev, 900 + M + i) for i in range(n)]
            outs = [torch.empty(M, S.DIM, dtype=torch.bfloat16, device=dev) for _ in range(n)]
            gph = capture_graph(lambda i: H.run(st, w, zs[i], rts[i][0], rts[i][1], outs[i], rank, 8), n, dist.barrier)
            ts = replay_times(gph, n, args.reps, dist.barrier)
            allt = [None] * 8
            dist.all_gather_object(allt, (min(ts), statistics.median(ts)))
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
        args.rdv = tempfile.mktemp(prefix='tp8moe_rdv_')
        mp.spawn(worker, args=(args,), nprocs=8, join=True)


if __name__ == '__main__':
    main()
