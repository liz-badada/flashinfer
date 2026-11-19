# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CUDA graph microbenchmark for ordered FP8 projection and requantization."""

import argparse
import json
import statistics

import torch

from flashinfer.gemm.sm90_fp8_projection import SM90FP8Projection
from flashinfer.testing import bench_gpu_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--block-n", type=int, choices=[16, 128], default=128)
    parser.add_argument("--num-warps", type=int, choices=[1, 4], default=4)
    parser.add_argument("--k-group", type=int, choices=[1, 4], default=1)
    args = parser.parse_args()
    torch.manual_seed(42)
    weight = torch.randn((args.n, args.k), device="cuda").to(torch.float8_e4m3fn)
    ws = torch.rand((args.n // 128, args.k // 128), device="cuda")
    for tokens in args.tokens:
        q = torch.randn((tokens, args.k), device="cuda").to(torch.float8_e4m3fn)
        xs = torch.rand((tokens, args.k // 128), device="cuda")
        projection = SM90FP8Projection(
            weight,
            ws,
            tokens,
            block_n=args.block_n,
            num_warps=args.num_warps,
            k_group=args.k_group,
        )
        raw = torch.empty(projection.raw_shape, device="cuda")
        out = torch.empty((tokens, args.n), device="cuda", dtype=q.dtype)
        scales = torch.empty((tokens, args.n // 128), device="cuda")
        projection.warmup("raw", q, out=raw)
        projection.warmup("quantize", raw, xs, output_q=out, output_scales=scales)

        def run():
            projection(q, xs, raw_output=raw, output_q=out, output_scales=scales)

        samples = bench_gpu_time(
            run,
            use_cuda_graph=True,
            cold_l2_cache=False,
            dry_run_iters=10,
            repeat_iters=30,
            num_iters_within_graph=16,
        )
        print(
            json.dumps(
                dict(
                    gpu=torch.cuda.get_device_name(),
                    tokens=tokens,
                    n=args.n,
                    k=args.k,
                    block_n=args.block_n,
                    num_warps=args.num_warps,
                    k_group=args.k_group,
                    median_us=1000 * statistics.median(samples),
                )
            )
        )


if __name__ == "__main__":
    main()
