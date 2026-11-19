"""Run with torchrun --standalone --nproc-per-node=4 on one SM90 host."""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from flashinfer.fused_moe import sm90_latent_tp4 as api


def random_weight(shape):
    value = torch.randn(shape, device="cuda") / math.sqrt(shape[-1])
    blocks = value.reshape(*shape[:-2], shape[-2] // 128, 128, shape[-1] // 128, 128)
    scales = blocks.abs().amax((-3, -1)).clamp_min(1e-10) / 448
    q = (
        (blocks / scales[..., :, None, :, None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    return api.FP8BlockWeight(q.reshape(shape), scales.contiguous())


def check(value):
    assert torch.isfinite(value).all()
    peers = [torch.empty_like(value) for _ in range(4)]
    dist.all_gather(peers, value)
    assert all(torch.equal(peers[0], v) for v in peers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 3, 9])
    parser.add_argument("--first-backend", choices=("triton", "rs"), default="triton")
    parser.add_argument("--down-mode", choices=("split", "peer"), default="split")
    parser.add_argument("--up-mode", choices=("split", "separate"), default="split")
    parser.add_argument("--pdl", action="store_true")
    parser.add_argument("--bridge", action="store_true")
    parser.add_argument(
        "--cache", choices=("normal", "first", "last"), default="normal"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    assert (
        Path(api.__file__).resolve() == root / "flashinfer/fused_moe/sm90_latent_tp4.py"
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.manual_seed(7)
    e, h, latent, f, sf, topk = 8, 512, 512, 128, 128, 2
    first, second = api.shard_fp8_experts_tp4(
        random_weight((e, 8 * f, latent)), random_weight((e, latent, 4 * f)), rank
    )
    shared_first, shared_second = api.shard_fp8_experts_tp4(
        random_weight((1, 8 * sf, h)), random_weight((1, h, 4 * sf)), rank
    )
    weights = api.LatentMoeTP4Weights(
        first,
        second,
        random_weight((latent, h)),
        random_weight((h, latent)),
        api.FP8BlockWeight(shared_first.values[0], shared_first.scales[0]),
        api.FP8BlockWeight(shared_second.values[0], shared_second.scales[0]),
    )
    config = api.LatentMoeTP4Config(
        first=api.LatentMoeProjectionConfig(cache=args.cache),
        second=api.LatentMoeProjectionConfig(cache=args.cache),
        first_backend=args.first_backend,
        down_mode=args.down_mode,
        up_mode=args.up_mode,
        pdl=args.pdl,
        bridge=args.bridge,
    )
    for tokens in args.tokens:
        torch.manual_seed(13 + tokens)
        x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16)
        ids = (
            torch.arange(tokens * topk, device="cuda").reshape(tokens, topk) % e
        ).int()
        rw = torch.rand(tokens, topk, device="cuda")
        alpha, gamma = (
            torch.rand(tokens, 1, device="cuda"),
            torch.rand(tokens, 1, device="cuda"),
        )
        inputs = (x, ids, rw, alpha, gamma)
        owner = api.LatentMoeTP4(
            weights, tokens=tokens, topk=topk, config=config
        ).prepare(*inputs)
        initial_output = owner(*inputs).clone()
        torch.cuda.synchronize()
        check(initial_output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(2):
                output = owner(*inputs)
        graph.replay()
        torch.cuda.synchronize()
        check(output)
        assert torch.equal(output, initial_output)
        tensors = (
            *inputs,
            *(v for w in vars(weights).values() for v in (w.values, w.scales)),
        )
        saved = [v.clone() for v in tensors]
        x.mul_(1.125)
        ids.fill_(e - 1)
        rw.mul_(-0.625)
        alpha.mul_(0.5)
        gamma.mul_(0.625)
        for weight in vars(weights).values():
            weight.values.view(torch.uint8).bitwise_xor_(128)
            weight.scales.mul_(1.125)
        if owner._up_raw is not None:
            owner._up_raw.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check(output)
        assert not torch.equal(output, initial_output)
        alpha.zero_()
        gamma.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert not torch.count_nonzero(output)
        for value, previous in zip(tensors, saved, strict=False):
            value.copy_(previous)
        graph.replay()
        torch.cuda.synchronize()
        check(output)
        assert torch.equal(output, initial_output)
        graph.reset()
        owner.synchronize()
        owned = owner.capture(*inputs, copies=2)
        check(owned.replay())
        owned.close()
        owner.close()
    if rank == 0:
        print(
            "PASS synthetic latent TP4 rank agreement, graphs, live values and storage lifetime"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
