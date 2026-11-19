# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

import flashinfer.gemm.sm90_fp8_projection as projection_module


assert Path(projection_module.__file__).resolve() == (
    Path(__file__).resolve().parents[2] / "flashinfer/gemm/sm90_fp8_projection.py"
)
SM90FP8Projection = projection_module.SM90FP8Projection
HAS_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
requires_sm90 = pytest.mark.skipif(not HAS_SM90, reason="requires SM90")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tokens": 0},
        {"tokens": True},
        {"block_n": 64},
        {"num_warps": 2},
        {"k_group": 2},
    ],
)
def test_invalid_configuration(kwargs):
    w = torch.empty((512, 512), dtype=torch.float8_e4m3fn)
    ws = torch.empty((4, 4))
    with pytest.raises(ValueError):
        SM90FP8Projection(w, ws, **({"tokens": 1} | kwargs))


def test_metadata_validation_without_cuda():
    w = torch.empty((512, 512), dtype=torch.float8_e4m3fn)
    ws = torch.empty((4, 4))
    with pytest.raises(TypeError, match="E4M3"):
        SM90FP8Projection(w.float(), ws, 1)
    with pytest.raises(ValueError, match="weight_scales"):
        SM90FP8Projection(w, ws[:, :2], 1)
    with pytest.raises(ValueError, match="contiguous"):
        SM90FP8Projection(w.T, ws, 1)
    with pytest.raises(ValueError, match="CUDA"):
        SM90FP8Projection(w, ws, 1)
    large = torch.empty((128, 2**24), dtype=torch.float8_e4m3fn, device="meta")
    with pytest.raises(ValueError, match="indexing"):
        SM90FP8Projection(large, torch.empty((1, 2**17), device="meta"), 1)


def inputs(tokens, n, k):
    # Dyadic values keep the unscaled dot products exactly representable in FP32.
    q = (torch.randint(-8, 9, (tokens, k), device="cuda").float() / 4).to(
        torch.float8_e4m3fn
    )
    w = (torch.randint(-8, 9, (n, k), device="cuda").float() / 4).to(
        torch.float8_e4m3fn
    )
    xs = torch.randint(1, 5, (tokens, k // 128), device="cuda").float() / 16
    ws = torch.randint(1, 5, (n // 128, k // 128), device="cuda").float() / 32
    return q, w, xs, ws


def reference(q, w, xs, ws):
    raw = torch.stack(
        [
            q[:, i : i + 128].float() @ w[:, i : i + 128].float().T
            for i in range(0, w.shape[1], 128)
        ]
    )
    acc = torch.zeros((q.shape[0], w.shape[0]), device=q.device)
    for block, partial in enumerate(raw):
        acc += (
            partial * xs[:, block, None] * ws[:, block].repeat_interleave(128)[None, :]
        )
    return raw, acc.to(torch.bfloat16)


@requires_sm90
@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 32])
@pytest.mark.parametrize(
    "n,k,bn,warps,group",
    [
        (512, 512, 16, 1, 1),
        (1024, 2048, 128, 4, 1),
        (2048, 1024, 128, 4, 4),
        (512, 1024, 16, 4, 4),
        (1024, 512, 128, 1, 1),
    ],
)
def test_projection_methods(tokens, n, k, bn, warps, group):
    torch.manual_seed(17)
    q, w, xs, ws = inputs(tokens, n, k)
    op = SM90FP8Projection(w, ws, tokens, block_n=bn, num_warps=warps, k_group=group)
    raw = torch.full(op.raw_shape, float("nan"), device="cuda")
    oq = torch.empty((tokens, n), dtype=torch.float8_e4m3fn, device="cuda")
    os = torch.empty((tokens, n // 128), device="cuda")
    bf16 = torch.empty((tokens, n), dtype=torch.bfloat16, device="cuda")
    shared = (torch.randint(-16, 17, (tokens, n), device="cuda").float() / 4).bfloat16()
    alpha = torch.randint(-8, 9, (tokens, 1), device="cuda").float() / 8 + 1 / 1024
    gamma = torch.randint(-8, 9, (tokens,), device="cuda").float() / 8 + 1 / 1024
    mixed = torch.empty_like(bf16)
    op.warmup("raw", q, out=raw)
    op.warmup("quantize", raw, xs, output_q=oq, output_scales=os)
    op.warmup("bf16", q, xs, out=bf16)
    op.warmup("mix", q, xs, shared, alpha, gamma, out=mixed)
    assert torch.isnan(raw).all(), "warmup launched the raw kernel"
    assert op.raw(q, out=raw) is raw
    assert op.bf16(q, xs, out=bf16) is bf16
    op.quantize(raw, xs, output_q=oq, output_scales=os)
    op.mix(q, xs, shared, alpha, gamma, out=mixed)
    expected_raw, expected = reference(q, w, xs, ws)
    torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)
    torch.testing.assert_close(bf16, expected, rtol=0, atol=0)
    expected_scales = (
        expected.float().reshape(tokens, n // 128, 128).abs().amax(-1).clamp_min(1e-10)
        / 448
    )
    torch.testing.assert_close(os, expected_scales, rtol=1e-6, atol=0)
    dequantized = oq.float().reshape(tokens, n // 128, 128) * os[:, :, None]
    # Half of the largest finite E4M3 quantization step is 16 scale units.
    assert (
        (dequantized - expected.float().reshape_as(dequantized)).abs()
        <= 16.01 * os[:, :, None]
    ).all()
    expected_mix = (
        (expected.float() * alpha.bfloat16().float()).bfloat16().float()
        + (shared.float() * gamma[:, None].bfloat16().float()).bfloat16().float()
    ).bfloat16()
    torch.testing.assert_close(mixed, expected_mix, rtol=0, atol=0)
    assert set(op.compiled_kernels) == {"raw", "quantize", "bf16", "mix"}


@requires_sm90
def test_graph_replay_and_live_weights():
    q, w, xs, ws = inputs(4, 512, 1024)
    op = SM90FP8Projection(w, ws, 4, pdl=True)
    raw = torch.full(op.raw_shape, float("nan"), device="cuda")
    out = torch.empty((4, 512), dtype=torch.bfloat16, device="cuda")
    op.warmup("raw", q, out=raw).warmup("bf16", q, xs, out=out)
    assert torch.isnan(raw).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.raw(q, out=raw)
        op.bf16(q, xs, out=out)
    for sign in (False, True):
        if sign:
            w.view(torch.uint8).bitwise_xor_(128)
            q.copy_((q.float() / 2).to(q.dtype))
        raw.fill_(float("nan"))
        graph.replay()
        expected_raw, expected = reference(q, w, xs, ws)
        torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    graph.reset()
    new_q = (q.float() / 2).to(q.dtype)
    torch.testing.assert_close(
        op.bf16(new_q, xs), reference(new_q, w, xs, ws)[1], rtol=0, atol=0
    )


@requires_sm90
def test_partial_k_group_and_weight_metadata():
    q, w, xs, ws = inputs(4, 512, 896)
    op = SM90FP8Projection(w, ws, 4, k_group=4)
    torch.testing.assert_close(op.raw(q), reference(q, w, xs, ws)[0], rtol=0, atol=0)
    w.resize_(512, 1024)
    with pytest.raises(RuntimeError, match="metadata changed"):
        op.raw(q)


@requires_sm90
def test_capture_requires_warmup_and_no_alias():
    q, w, xs, ws = inputs(32, 512, 512)
    op = SM90FP8Projection(w, ws, 32)
    alias = w.view(torch.float32).reshape(op.raw_shape)
    with pytest.raises(ValueError, match="overlap"):
        op.raw(q, out=alias)
    raw = torch.empty(op.raw_shape, device="cuda")
    # Reject before recording an uninitialized specialization in the graph.
    with (
        pytest.raises(RuntimeError, match="warm up raw"),
        torch.cuda.graph(torch.cuda.CUDAGraph()),
    ):
        op.raw(q, out=raw)
