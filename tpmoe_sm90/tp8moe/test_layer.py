"""End-to-end TP8 check of one K3 MoE layer (moe_layer.MoELayerTP8) against moe_ref.moe_block_ref with torch-reference experts.

  python test_layer.py --ms 1 8 32 [--nexp 128] [--time]
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp8moe_host as H   # noqa: E402
import moe_ref as R   # noqa: E402
import mx_synth as S   # noqa: E402
from moe_layer import MoELayerTP8, HID, LAT, SH_INTER, NDEV   # noqa: E402
from test_tp8moe import CudaRT, stats   # noqa: E402


def dense_weights(device, seed=7):
    g = torch.Generator(device='cpu').manual_seed(seed)
    f = lambda *shape, std: (torch.randn(*shape, generator=g) * std).to(torch.bfloat16).to(device)
    return dict(gate_w=f(896, HID, std=0.02), gate_bias=(torch.randn(896, generator=g) * 0.05).to(device),
                down_w=f(LAT, HID, std=0.02), up_w=f(HID, LAT, std=0.02), norm_w=(1.0 + 0.1 * torch.randn(LAT, generator=g)).to(torch.bfloat16).to(device),
                sh_gate=f(SH_INTER, HID, std=0.02), sh_up=f(SH_INTER, HID, std=0.02), sh_down=f(HID, SH_INTER, std=0.02))


def worker(rank, args):
    torch.cuda.set_device(rank)
    dev = torch.device('cuda', rank)
    dist.init_process_group('nccl', init_method='file://' + args.rdv, world_size=NDEV, rank=rank)
    H.ext()
    st = H.RankState(dev, alloc_comm=False)
    rt = CudaRT(); rt.ck(rt.lib.cudaSetDevice(rank))
    ptr = rt.malloc(H.RankState.RS_BYTES())
    raw = [None] * NDEV
    dist.all_gather_object(raw, rt.handle(ptr))
    st.link_ptrs([ptr if peer == rank else rt.open(h) for peer, h in enumerate(raw)])
    torch.cuda.synchronize(); dist.barrier()
    experts = S.make_expert_weights(args.nexp, dev, seed=1 + rank, keep_dq=True)   # this rank's intermediate shard of every expert
    D = dense_weights(dev)
    layer = MoELayerTP8(rank, st, experts, **D)

    def rank_fn_ref(z, ids, rw):   # torch-reference experts on every rank + exact TP combine
        mine = R.routed_local(z, ids, rw, experts['dq1'], experts['dq2'])
        allv = [torch.empty_like(mine) for _ in range(NDEV)]
        dist.all_gather(allv, mine)
        return R.combine_tp(allv)

    for M in args.ms:
        g = torch.Generator(device='cpu').manual_seed(M)
        x = torch.randn(M, HID, generator=g).to(torch.bfloat16).to(dev)
        y = layer.forward(x)
        torch.cuda.synchronize()
        y2 = layer.forward(x)
        torch.cuda.synchronize()
        yref = R.moe_block_ref(x, D['gate_w'], D['gate_bias'], D['down_w'], D['up_w'], D['norm_w'], D['sh_gate'], D['sh_up'], D['sh_down'], rank_fn_ref)
        rr, tr = stats(y, yref)
        ys = [torch.empty_like(y) for _ in range(NDEV)]
        dist.all_gather(ys, y)
        if rank == 0:
            print(f'M={M:3d} nexp={args.nexp} layer TP8 vs ref: rel_rms={rr:.4f} tok_max={tr:.4f} finite={bool(torch.isfinite(y.float()).all())} '
                  f'call1==call2={bool((y == y2).all())} ranks_identical={all(bool((ys[0] == o).all()) for o in ys[1:])}', flush=True)
    if args.time:
        for M in args.ms:
            x = torch.randn(M, HID).to(torch.bfloat16).to(dev)
            for _ in range(5): layer.forward(x)
            torch.cuda.synchronize(); dist.barrier()
            n = 50
            t0 = time.perf_counter()
            for _ in range(n): layer.forward(x)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / n * 1e6
            ts = [None] * NDEV
            dist.all_gather_object(ts, dt)
            if rank == 0:
                print(f'M={M:3d} layer eager time per call (worst rank): {max(ts):8.1f} us  (ranks: {[round(t) for t in ts]})', flush=True)
            # the same layer captured in one CUDA graph (NCCL collectives included): per-call time of a graph of --copies calls
            n = args.copies
            xs = [torch.randn(M, HID).to(torch.bfloat16).to(dev) for _ in range(n)]
            for xx in xs: layer.forward(xx)
            torch.cuda.synchronize(); dist.barrier()
            gph = torch.cuda.CUDAGraph(); sstream = torch.cuda.Stream()
            sstream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(sstream):
                with torch.cuda.graph(gph, stream=sstream):
                    ys = [layer.forward(xx) for xx in xs]
            torch.cuda.synchronize(); dist.barrier()
            tsg = []
            for rep in range(args.reps):
                dist.barrier()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); gph.replay(); e1.record(); torch.cuda.synchronize()
                tsg.append(e0.elapsed_time(e1) * 1e3 / n)
            allt = [None] * NDEV
            dist.all_gather_object(allt, (min(tsg), sorted(tsg)[len(tsg) // 2]))
            if rank == 0:
                print(f'M={M:3d} layer CUDA-graph time per call (worst rank): min {max(t[0] for t in allt):8.1f} us  median {max(t[1] for t in allt):8.1f} us', flush=True)
    dist.barrier(); torch.cuda.synchronize()
    # exit without tearing down: destroying the NCCL communicator while captured graphs still hold its work hung for 40 min on
    # viking-prod-322 (all 8 workers spinning after the last print); the OS reclaims everything
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


def main():
    # the torch reference of 896 experts runs for minutes without a collective: keep the NCCL heartbeat monitor from aborting the ranks
    os.environ.setdefault('TORCH_NCCL_ENABLE_MONITORING', '0')
    os.environ.setdefault('TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC', '3600')
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', type=int, nargs='+', default=[1, 8, 32]); ap.add_argument('--nexp', type=int, default=128)
    ap.add_argument('--time', action='store_true'); ap.add_argument('--copies', type=int, default=8); ap.add_argument('--reps', type=int, default=20)
    args = ap.parse_args()
    args.rdv = tempfile.mktemp(prefix='tp8moe_rdv_')
    mp.spawn(worker, args=(args,), nprocs=NDEV, join=True)


if __name__ == '__main__':
    main()
