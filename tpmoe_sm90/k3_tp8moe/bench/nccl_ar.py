"""8-GPU all-reduce cost of the MoE output [M, 3584] BF16 (what a single-GPU baseline kernel must add under TP8): NCCL via torch.distributed,
graph of N back-to-back calls, us per call (min / median over reps), worst rank."""
import argparse, os, time, sys
import torch, torch.distributed as dist, torch.multiprocessing as mp
DIM = 3584
def worker(rank, args):
    torch.cuda.set_device(rank); dev = torch.device('cuda', rank)
    dist.init_process_group('nccl', init_method='file://' + args.rdv, world_size=8, rank=rank)
    res = {}
    for M in args.ms:
        x = torch.randn(M, DIM, device=dev).to(torch.bfloat16)
        for _ in range(5): dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3): dist.all_reduce(x)
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s):
                for _ in range(args.copies): dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        ts = []
        for _ in range(args.reps):
            dist.barrier(); torch.cuda.synchronize()
            t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e6 / args.copies)
        ts.sort(); res[M] = (ts[0], ts[len(ts) // 2])
    out = [None] * 8
    dist.all_gather_object(out, res)
    if rank == 0:
        for M in args.ms:
            mn = max(r[M][0] for r in out); md = max(r[M][1] for r in out)
            print(f'M={M:3d} NCCL all-reduce [M,{DIM}] bf16 x8 GPUs, graph of {args.copies}: per-call us min {mn:7.1f} median {md:7.1f} (worst rank)', flush=True)
    dist.barrier(); torch.cuda.synchronize(); sys.stdout.flush(); os._exit(0)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ms', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--copies', type=int, default=32); ap.add_argument('--reps', type=int, default=30)
    a = ap.parse_args(); import tempfile; a.rdv = tempfile.mktemp(prefix='nccl_ar_')
    os.environ.setdefault('TORCH_NCCL_ENABLE_MONITORING', '0')
    mp.spawn(worker, args=(a,), nprocs=8, join=True)
