# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for license information.

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import torch

from flashinfer.gemm.sm90_grouped_fp8 import (
    SM90GroupedFP8Projection,
    _Config,
    _validate_indices,
)
from flashinfer.gemm._sm90_grouped_fp8_native import _build_identity
from triton.tools.tensor_descriptor import TensorDescriptor
import flashinfer.gemm.kernels.sm90_grouped_fp8 as kernel_implementation


def test_config_cpu():
    assert _Config(32, 1, 3, 1, "normal", None).policy_key == 0
    assert _Config(32, 1, 3, 1, "last", None).policy_key == 2
    assert _Config(32, 1, 3, 1, "first", 8).policy_key == 10
    for args in (
        (17, 1, 3, 1, "normal", None),
        (32, True, 3, 1, "normal", None),
        (32, 1, 0, 1, "normal", None),
        (32, 1, 3, 1, "last", 8),
    ):
        with pytest.raises(ValueError):
            _Config(*args)


def test_indices_cpu():
    slots = torch.tensor(
        [[0, 1, -1, -1, -1, -1, -1, -1], [2**31 - 1] * 8], dtype=torch.int32
    )
    experts = torch.tensor([1, -1], dtype=torch.int32)
    kwargs = dict(expert_count=3, input_rows=1, output_rows=2, reuse=2)
    _validate_indices(slots, experts, **kwargs)
    for value in (0, 2):
        invalid = slots.clone()
        invalid[0, 1] = value
        with pytest.raises(ValueError):
            _validate_indices(invalid, experts, **kwargs)
    with pytest.raises(ValueError):
        _validate_indices(slots, torch.tensor([3, -1]), **kwargs)


def test_build_identity_cpu():
    source = dict(path="/opt/kernels/project.cu", name="project.cu", sha256="a" * 64)
    _, first = _build_identity(
        source, {"project.cu": "a" * 64}, {"nvcc": "compiler"}, ["-O3"]
    )
    for entry, sources, toolchain, flags in (
        (
            {**source, "sha256": "b" * 64},
            {"project.cu": "a" * 64},
            {"nvcc": "compiler"},
            ["-O3"],
        ),
        (source, {"project.cu": "b" * 64}, {"nvcc": "compiler"}, ["-O3"]),
        (source, {"project.cu": "a" * 64}, {"nvcc": "different"}, ["-O3"]),
        (source, {"project.cu": "a" * 64}, {"nvcc": "compiler"}, ["-O2"]),
    ):
        assert _build_identity(entry, sources, toolchain, flags)[1] != first


def test_reject_cpu_weights():
    weight = torch.empty((2, 256, 512), dtype=torch.float8_e4m3fn)
    scales = torch.ones((2, 2, 4))
    with pytest.raises(ValueError, match="CUDA"):
        SM90GroupedFP8Projection(weight, scales)


@pytest.fixture
def sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires an SM90 GPU")
    torch.manual_seed(23)
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device("cuda", torch.cuda.current_device())


def _quantize(x):
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 128, 128)
    scales = blocks.abs().amax(-1).clamp_min(1e-8) / 448.0
    q = (blocks / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(x.shape).contiguous(), scales.contiguous()


def _fixture(device, weighted, n=256, k=512):
    values = torch.randn((5, n // 128, 128, k // 128, 128), device=device) / k**0.5
    scales = values.abs().amax((2, 4)).clamp_min(1e-8) / 448
    weight = (
        (values / scales[:, :, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    )
    weight = weight.reshape(5, n, k).contiguous()
    reuse = 1 if weighted else 2
    q, qs = _quantize(torch.randn((24 // reuse, k), device=device))
    slots = torch.tensor(
        [
            [0, 2, 4, 6, 8, 10, -1, -1],
            [2**31 - 1] * 8,
            [1, 3, 5, 7, 9, 11, 12, 13],
            [14, 15, -1, 16, 17, 18, 19, -1],
        ],
        device=device,
        dtype=torch.int32,
    )
    experts = torch.tensor([0, -1, 3, 1], device=device, dtype=torch.int32)
    out = torch.full((24, n), 3.25, device=device, dtype=torch.bfloat16)
    rw = torch.linspace(-0.7, 0.9, 24, device=device) if weighted else None
    return weight, scales, q, qs, slots, experts, out, rw, reuse


def _reference(weight, ws, q, qs, slots, experts, out, rw, reuse):
    result = out.clone()
    for panel, expert in enumerate(experts.cpu().tolist()):
        if expert < 0:
            continue
        valid = slots[panel][slots[panel] >= 0].to(torch.int64)
        rows = valid // reuse
        acc = torch.zeros((valid.numel(), weight.shape[1]), device=q.device)
        for kb in range(q.shape[1] // 128):
            raw = (
                q[rows, kb * 128 : (kb + 1) * 128].float()
                @ weight[expert, :, kb * 128 : (kb + 1) * 128].float().T
            )
            sw = ws[expert, :, kb].repeat_interleave(128)
            acc = torch.addcmul(acc, raw * qs[rows, kb, None], sw[None, :])
        if rw is not None:
            acc = acc * rw[valid, None]
        result[valid] = acc.to(torch.bfloat16)
    return result


@pytest.mark.parametrize("n,k", [(256, 512), (512, 1024)])
@pytest.mark.parametrize(
    "backend,weighted,pdl,cache,hot",
    [
        ("triton", False, False, "normal", None),
        ("triton", True, False, "last", None),
        ("triton", True, True, "first", 2),
        ("rs", False, False, "first", None),
    ],
)
def test_projection(sm90, n, k, backend, weighted, pdl, cache, hot):
    w, ws, q, qs, slots, experts, out, rw, reuse = _fixture(sm90, weighted, n, k)
    op = SM90GroupedFP8Projection(
        w,
        ws,
        backend=backend,
        weighted=weighted,
        pdl=pdl,
        cache=cache,
        hot_panels=hot,
        input_reuse=reuse,
    )
    args = q, qs, slots, experts, out, rw
    before = out.clone()
    op.prepare(*args, validate_indices=True)
    assert torch.equal(out, before)
    expected = _reference(w, ws, *args, reuse)
    op(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=0.01, atol=0.01)
    artifacts = op.artifacts
    assert artifacts["cubin"]
    if backend == "triton":
        assert bool(artifacts["resources"]["launch_pdl"]) == pdl
    else:
        assert artifacts["resources"]["local_bytes"] == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*args)
    saved = [t.clone() for t in (q, qs, w, ws, experts)]
    for phase in range(4):
        if phase == 0:
            q.view(torch.uint8).bitwise_xor_(128)
            ws.mul_(1.125)
        elif phase == 1:
            w.view(torch.uint8).bitwise_xor_(128)
            qs.mul_(0.875)
            experts.copy_(torch.tensor([2, -1, 0, 4], device=sm90, dtype=torch.int32))
        elif phase == 2:
            experts.fill_(-1)
        else:
            for value, initial in zip((q, qs, w, ws, experts), saved, strict=True):
                value.copy_(initial)
        out.fill_(3.25)
        expected = _reference(w, ws, *args, reuse)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=0.01, atol=0.01)
    graph.reset()
    with pytest.raises(RuntimeError, match="storage"):
        op(q.clone(), qs, slots, experts, out, rw)


def test_shared_compiled_module(sm90, monkeypatch):
    torch.set_num_threads(1)
    w, ws, q, qs, slots, experts, _, rw, reuse = _fixture(sm90, True)
    barrier = threading.Barrier(4)
    descriptor = TensorDescriptor.from_tensor(w.reshape(-1, q.shape[1]), [32, 128])
    initial_out = torch.empty((24, 256), device=sm90, dtype=torch.bfloat16)
    compiled = kernel_implementation.grouped.warmup(
        q,
        qs,
        descriptor,
        ws,
        slots,
        experts,
        rw,
        initial_out,
        256,
        q.shape[1],
        1,
        True,
        32,
        1,
        False,
        3,
        grid=(experts.numel(), 8, 1),
        num_warps=1,
        num_stages=3,
        launch_pdl=False,
    )
    assert compiled.module is None
    monkeypatch.setattr(
        kernel_implementation.grouped, "warmup", lambda *args, **kwargs: compiled
    )
    torch.cuda.synchronize()

    def initialize(_):
        with torch.cuda.device(sm90):
            out = torch.empty((24, 256), device=sm90, dtype=torch.bfloat16)
            op = SM90GroupedFP8Projection(
                w, ws, weighted=True, cache="first", hot_panels=1
            )
            barrier.wait()
            op.prepare(q, qs, slots, experts, out, rw)
            op(q, qs, slots, experts, out, rw)
            return op, out

    with ThreadPoolExecutor(max_workers=4) as executor:
        owners = list(executor.map(initialize, range(4)))
    torch.cuda.synchronize()
    assert all(op.compiled_kernel is owners[0][0].compiled_kernel for op, _ in owners)
    for _, out in owners:
        expected = _reference(w, ws, q, qs, slots, experts, out, rw, reuse)
        active = torch.cat([slots[i][slots[i] >= 0] for i in (0, 2, 3)]).long()
        torch.testing.assert_close(out[active], expected[active], rtol=0.01, atol=0.01)
