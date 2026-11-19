"""Synthetic TP4 liveness, rank agreement, graph and storage-lifetime smoke."""
import argparse
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parents[1]))
from generic_tp4 import BlockWeight, ExecutionConfig, LocalWeights, ProjectionConfig, Workspace
from generic_tp4.operator import shard_experts


def random_weight(shape):
    values = torch.randn(shape, device='cuda') * .025
    blocks = values.reshape(*shape[:-2], shape[-2] // 128, 128, shape[-1] // 128, 128)
    scales = blocks.abs().amax((-3, -1)).clamp_min(1e-10) / 448
    q = (blocks / scales[..., :, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return BlockWeight(q.reshape(shape), scales.contiguous())


def check_output(value):
    assert torch.isfinite(value).all()
    ranks = [torch.empty_like(value) for _ in range(4)]
    dist.all_gather(ranks, value)
    assert all(torch.equal(ranks[0], v) for v in ranks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', nargs='+', type=int, default=[1, 3, 9])
    p.add_argument('--topk', type=int, default=2)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--latent', type=int, default=512)
    p.add_argument('--experts', type=int, default=8)
    p.add_argument('--local-width', type=int, default=128)
    p.add_argument('--shared-width', type=int, default=128)
    p.add_argument('--down', choices=['split', 'peer'], default='split')
    p.add_argument('--up', choices=['split', 'separate'], default='split')
    p.add_argument('--first', choices=['triton', 'rs'], default='triton')
    p.add_argument('--pdl', action='store_true')
    p.add_argument('--bridge', action='store_true')
    p.add_argument('--down-bn', type=int, default=128)
    p.add_argument('--down-warps', type=int, default=4)
    args = p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    torch.manual_seed(7)
    e, h, latent, f, sf = args.experts, args.hidden, args.latent, args.local_width, args.shared_width
    first, second = shard_experts(random_weight((e, 8 * f, latent)), random_weight((e, latent, 4 * f)), rank)
    s1, s2 = shard_experts(random_weight((1, 8 * sf, h)), random_weight((1, h, 4 * sf)), rank)
    weights = LocalWeights(first, second, random_weight((latent, h)), random_weight((h, latent)),
                           BlockWeight(s1.values[0], s1.scales[0]), BlockWeight(s2.values[0], s2.scales[0]))
    cfg = ExecutionConfig(first=ProjectionConfig(), second=ProjectionConfig(),
        first_backend=args.first, down_mode=args.down, up_mode=args.up,
        pdl=args.pdl, bridge=args.bridge, down_bn=args.down_bn, down_warps=args.down_warps)
    for tokens in args.tokens:
        torch.manual_seed(13 + tokens)
        x = torch.randn(tokens, h, dtype=torch.bfloat16, device='cuda')
        ids = (torch.arange(tokens * args.topk, device='cuda').reshape(tokens, args.topk) % e).int()
        routing = torch.rand(tokens, args.topk, device='cuda')
        alpha, gamma = torch.rand(tokens, 1, device='cuda'), torch.rand(tokens, 1, device='cuda')
        inputs = (x, ids, routing, alpha, gamma)
        workspace = Workspace(weights, tokens=tokens, topk=args.topk, config=cfg).prepare(*inputs)
        baseline = workspace(*inputs).clone()
        check_output(baseline)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(2):
                output = workspace(*inputs)
        graph.replay()
        torch.cuda.synchronize()
        check_output(output)
        assert torch.equal(output, baseline)
        tensors = (*inputs, *(v for w in vars(weights).values() for v in (w.values, w.scales)))
        saved = [v.clone() for v in tensors]
        x.mul_(1.125)
        ids.fill_(e - 1)
        routing.mul_(-.625)
        alpha.mul_(.5)
        gamma.mul_(.625)
        for w in vars(weights).values():
            w.values.view(torch.uint8).bitwise_xor_(128)
            w.scales.mul_(1.125)
        if workspace._up_raw is not None:
            workspace._up_raw.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        check_output(output)
        assert not torch.equal(output, baseline)
        alpha.zero_()
        gamma.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert not torch.count_nonzero(output)
        for value, old in zip(tensors, saved):
            value.copy_(old)
        graph.replay()
        torch.cuda.synchronize()
        check_output(output)
        assert torch.equal(output, baseline)
        graph.reset()
        workspace.synchronize()
        owned = workspace.capture(*inputs, copies=2)
        assert torch.equal(owned.replay(), baseline)
        owned.close()
        workspace.close()
    if rank == 0:
        print('PASS synthetic TP4 rank agreement, graphs, live mutation and restoration')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
