# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compare prepared Triton and RS projections with resident synthetic weights."""

import argparse
import json
import statistics

import torch

from flashinfer.gemm.sm90_grouped_fp8 import SM90GroupedFP8Projection
from flashinfer.testing import bench_gpu_time_with_cudagraph


def run(args):
    torch.manual_seed(31)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    weight = (torch.randn((args.experts, args.n, args.k), device=device) * 32).to(
        torch.float8_e4m3fn
    )
    scales = (
        torch.rand((args.experts, args.n // 128, args.k // 128), device=device)
        .add_(0.5)
        .mul_(1 / (32 * args.k**0.5))
    )
    print(
        json.dumps(
            dict(
                gpu=props.name,
                uuid=str(props.uuid),
                sms=props.multi_processor_count,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                shape=dict(experts=args.experts, n=args.n, k=args.k, topk=args.topk),
                scope="single unweighted prepared projection; routing and quantization excluded",
                method="CUDA events around 16-call graphs; resident weights; no cache flush",
                triton=dict(block_n=32, num_warps=1, num_stages=3),
                rs=dict(tile_groups=2, pipe_stages=2),
                cache="first",
                repeats=args.repeats,
            )
        ),
        flush=True,
    )
    for tokens in args.tokens:
        q = (torch.randn((tokens, args.k), device=device) * 32).to(torch.float8_e4m3fn)
        qs = torch.rand((tokens, args.k // 128), device=device).add_(0.5).mul_(1 / 32)
        by_expert = [[] for _ in range(args.experts)]
        for slot in range(tokens * args.topk):
            by_expert[slot % args.experts].append(slot)
        panels, expert_ids = [], []
        for expert, indices in enumerate(by_expert):
            for start in range(0, len(indices), 8):
                chunk = indices[start : start + 8]
                panels.append(chunk + [-1] * (8 - len(chunk)))
                expert_ids.append(expert)
        slots = torch.tensor(panels, device=device, dtype=torch.int32)
        experts = torch.tensor(expert_ids, device=device, dtype=torch.int32)
        owners, outputs = {}, {}
        for backend in ("triton", "rs"):
            outputs[backend] = torch.empty(
                (tokens * args.topk, args.n), device=device, dtype=torch.bfloat16
            )
            owners[backend] = SM90GroupedFP8Projection(
                weight, scales, input_reuse=args.topk, backend=backend, cache="first"
            )
            owners[backend].prepare(
                q, qs, slots, experts, outputs[backend], validate_indices=True
            )
            owners[backend](q, qs, slots, experts, outputs[backend])
        torch.testing.assert_close(
            outputs["rs"], outputs["triton"], rtol=0.01, atol=0.01
        )
        samples = {backend: [] for backend in owners}
        windows = []
        for order in (("triton", "rs"), ("rs", "triton")) * 2:
            block = {}
            for backend in order:
                op, output = owners[backend], outputs[backend]
                values = bench_gpu_time_with_cudagraph(
                    lambda: op(q, qs, slots, experts, output),
                    dry_run_iters=5,
                    repeat_iters=args.repeats,
                    num_iters_within_graph=16,
                    cold_l2_cache=False,
                )
                block[backend] = [value * 1000 for value in values]
                samples[backend].extend(block[backend])
            windows.append(dict(order=order, samples_us=block))
        medians = {key: statistics.median(values) for key, values in samples.items()}
        print(
            json.dumps(
                dict(
                    tokens=tokens,
                    panels=len(panels),
                    median_us=medians,
                    rs_speedup=medians["triton"] / medians["rs"],
                    windows=windows,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if (
        min(args.experts, args.n, args.k, args.topk, args.repeats, *args.tokens) < 1
        or args.topk > args.experts
    ):
        parser.error("positive dimensions and topk <= experts are required")
    run(args)
