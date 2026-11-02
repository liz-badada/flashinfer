"""Two-node layout driver for K3 MoE layers: TP8 inside each node (tp8moe + moe_layer glue), PP2 across the nodes.

Stage s (node s) holds layers [s * L, (s + 1) * L); the hidden states (BF16 [M, 7168]) move stage 0 -> stage 1 with
torch.distributed send / recv (NCCL; across nodes that is the IB / EFA link, inside one node NVLink).  Each stage's 8 ranks form a TP
group with its own IPC-linked one-shot LL buffers.  World = 16 ranks: rank r is stage r // 8, TP rank r % 8, device r % 8 (so the whole
two-node layout can be rehearsed on ONE 8-GPU node with two ranks per GPU: --colocated).

  python pp2.py --layers 2 --ms 1 8 --nexp 32 --colocated          # 16 ranks on 8 GPUs: pipeline output == all layers on one TP8 group
  torchrun --nnodes 2 --nproc_per_node 8 ... pp2.py --layers 46 ... # the real two-node run (env:// rendezvous, device = LOCAL_RANK)
"""
import argparse
import os
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp8moe_host as H   # noqa: E402
import mx_synth as S   # noqa: E402
from moe_layer import MoELayerTP8, HID, NDEV   # noqa: E402
from test_tp8moe import ipc_link, stats   # noqa: E402
from test_layer import dense_weights   # noqa: E402

NSTAGE = 2


def make_layers(stage, tp_rank, st, dev, L, nexp, group):
    layers = []
    for li in range(stage * L, (stage + 1) * L):   # deterministic per (layer, tp_rank): both stages can rebuild each other's layers
        experts = S.make_expert_weights(nexp, dev, seed=1000 * li + tp_rank)
        D = dense_weights(dev, seed=77 + li)
        layers.append(MoELayerTP8(tp_rank, st, experts, group=group, **D))
    return layers


def worker(rank, args):
    world = NSTAGE * NDEV
    stage, tp_rank = rank // NDEV, rank % NDEV
    dev_id = int(os.environ['LOCAL_RANK']) if args.torchrun else (tp_rank if args.colocated else rank)
    torch.cuda.set_device(dev_id)
    dev = torch.device('cuda', dev_id)
    os.environ.setdefault('TORCH_NCCL_ENABLE_MONITORING', '0')
    # colocated rehearsal: two ranks share a GPU, which NCCL forbids inside one communicator -> the world group is gloo and the stage hop
    # goes through host memory; the TP groups (distinct GPUs) are NCCL.  Real two-node run: everything NCCL.
    to = timedelta(hours=2)   # the 896-expert torch reference runs for minutes between collectives
    if args.torchrun:   # real two-node run: torchrun --nnodes 2 --nproc_per_node 8, everything NCCL (env:// rendezvous)
        dist.init_process_group('nccl', init_method='env://', world_size=world, rank=rank, timeout=to)
    else:
        dist.init_process_group('gloo' if args.colocated else 'nccl', init_method='file://' + args.rdv, world_size=world, rank=rank, timeout=to)
    groups = [dist.new_group(list(range(s * NDEV, (s + 1) * NDEV)), backend='nccl', timeout=to) for s in range(NSTAGE)]
    if rank == 0:
        print(f'world {world} ranks, torchrun={args.torchrun} colocated={args.colocated} host {os.uname().nodename}', flush=True)
    group = groups[stage]
    H.ext()
    st = H.RankState(dev, alloc_comm=False)
    ipc_link(st, dev_id, tp_rank, group=group)   # IPC link inside the TP group only
    layers = make_layers(stage, tp_rank, st, dev, args.layers, args.nexp, group)
    peer = rank + NDEV if stage == 0 else rank - NDEV   # same TP rank on the other stage

    def hop_send(t):
        dist.send(t.cpu() if args.colocated else t, peer)

    def hop_recv(t):
        if args.colocated:
            buf = torch.empty_like(t, device='cpu'); dist.recv(buf, peer); t.copy_(buf)
        else:
            dist.recv(t, peer)

    def stage_forward(x):
        for layer in layers:
            x = layer.forward(x)
        return x

    for M in args.ms:
        x0 = torch.randn(M, HID, generator=torch.Generator(device='cpu').manual_seed(M)).to(torch.bfloat16).to(dev)
        # ---- pipeline: stage 0 computes and sends, stage 1 receives and computes
        dist.barrier()
        if stage == 0:
            y0 = stage_forward(x0)
            hop_send(y0)
            out = None
        else:
            xin = torch.empty(M, HID, dtype=torch.bfloat16, device=dev)
            hop_recv(xin)
            out = stage_forward(xin)
        torch.cuda.synchronize(); dist.barrier()
        # ---- check: stage 1's output == every layer applied in order on stage 0's TP group (it rebuilds stage 1's layers)
        if stage == 0:
            ref_layers = layers + make_layers(1, tp_rank, st, dev, args.layers, args.nexp, group)
            yref = x0
            for layer in ref_layers:
                yref = layer.forward(yref)
            torch.cuda.synchronize()
            hop_send(yref)
        else:
            yref = torch.empty(M, HID, dtype=torch.bfloat16, device=dev)
            hop_recv(yref)
            rr, _ = stats(out, yref)
            if rank == NDEV:
                print(f'M={M:3d} PP2 x TP8 ({2 * args.layers} MoE layers, {args.layers} per stage): pipeline == single-group bitwise: {bool((out == yref).all())} '
                      f'(rel_rms {rr:.2e}) finite={bool(torch.isfinite(out.float()).all())}', flush=True)
        dist.barrier()
        if args.time:   # single-stream latency: a token's worth of layers through both stages, incl. the stage hop
            xin = torch.empty(M, HID, dtype=torch.bfloat16, device=dev)
            for it in range(3 + args.reps):
                dist.barrier()
                t0 = time.perf_counter()
                if stage == 0:
                    y0 = stage_forward(x0); hop_send(y0)
                else:
                    hop_recv(xin); out = stage_forward(xin)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1e6
                dist.barrier()
                if it >= 3 and rank == NDEV:
                    best = dt if it == 3 else min(best, dt)
            if rank == NDEV:
                print(f'M={M:3d} PP2 single-stream wall time, {2 * args.layers} layers through both stages (eager, min): {best:8.1f} us '
                      f'({best / (2 * args.layers):6.1f} us per layer incl. the hop)', flush=True)
    dist.barrier(); torch.cuda.synchronize()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # no NCCL teardown with captured work alive (hung on viking-prod-322)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', type=int, default=2, help='MoE layers per stage')
    ap.add_argument('--ms', type=int, nargs='+', default=[1, 8]); ap.add_argument('--nexp', type=int, default=32)
    ap.add_argument('--colocated', action='store_true', help='16 ranks on one 8-GPU node (two per GPU)')
    ap.add_argument('--time', action='store_true'); ap.add_argument('--reps', type=int, default=10)
    args = ap.parse_args()
    os.environ.setdefault('TORCH_NCCL_ENABLE_MONITORING', '0')
    args.torchrun = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if args.torchrun:
        assert int(os.environ['WORLD_SIZE']) == NSTAGE * NDEV, os.environ['WORLD_SIZE']
        args.rdv = None
        worker(int(os.environ['RANK']), args)
    else:
        args.rdv = tempfile.mktemp(prefix='tp8moe_rdv_')
        mp.spawn(worker, args=(args,), nprocs=NSTAGE * NDEV, join=True)


if __name__ == '__main__':
    main()
