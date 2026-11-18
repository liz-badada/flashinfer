# SPDX-License-Identifier: Apache-2.0
"""Small synthetic correctness tests."""

import pytest
import torch

from .projection import GroupedFP8Projection


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 GPU required",
)


def fixture(n, k, dtype, weighted):
    torch.manual_seed(73)
    x = torch.randn(3, k, device="cuda").to(torch.float8_e4m3fn)
    xs = torch.full((3, k // 128), 0.125, device="cuda")
    signs = torch.where(torch.arange(n, device="cuda") % 2 == 0, 1.0, -1.0)
    factors = torch.arange(1, 4, device="cuda").float() * 0.5
    weights = (factors[:, None, None] * signs[None, :, None]).expand(3, n, k)
    weights = weights.contiguous().to(torch.float8_e4m3fn)
    scales = torch.arange(3 * (n // 128) * (k // 128), device="cuda") % 4 + 1
    scales = scales.float().reshape(3, n // 128, k // 128) / 16
    output = torch.empty(6, n, device="cuda", dtype=dtype)
    slots = torch.full((3, 8), -1, device="cuda", dtype=torch.int32)
    slots[:, :2] = torch.tensor([[0, 4], [1, 5], [2, 3]], device="cuda")
    groups = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int64)
    count = torch.tensor([3], device="cuda", dtype=torch.int32)
    coefficients = (
        torch.tensor([0.5, -0.25, 0.75, -0.5, 0.25, -0.75], device="cuda")
        if weighted
        else None
    )
    op = GroupedFP8Projection(weights, scales, output, max_panels=3, input_reuse=2)
    return op, x, xs, slots, groups, count, coefficients, signs, factors


@pytest.mark.parametrize(
    "n,k,dtype,weighted",
    [
        (128, 128, torch.float32, False),
        (128, 256, torch.bfloat16, True),
        (256, 128, torch.bfloat16, False),
        (256, 256, torch.float32, True),
    ],
)
def test_known_dense_rows(n, k, dtype, weighted):
    op, x, xs, slots, groups, count, coefficients, signs, factors = fixture(
        n, k, dtype, weighted
    )
    actual = op(x, xs, slots, groups, count, coefficients)
    block_sums = x.float().reshape(3, k // 128, 128).sum(-1) * xs
    expected = torch.empty_like(actual)
    for group, indices in [(2, (0, 4)), (0, (1, 5)), (1, (2, 3))]:
        for slot in indices:
            values = (block_sums[slot // 2][None, :] * op.scales[group]).sum(-1)
            values = values.repeat_interleave(128) * factors[group] * signs
            if coefficients is not None:
                values *= coefficients[slot]
            expected[slot] = values
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_graph_live_weights_scales_and_count():
    op, x, xs, slots, groups, count, coefficients, _, _ = fixture(
        128, 256, torch.float32, True
    )
    op(x, xs, slots, groups, count, coefficients)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, xs, slots, groups, count, coefficients)
    graph.replay()
    initial = op.output.clone()
    op.weights.view(torch.uint8)[2].bitwise_xor_(128)
    op.scales[2].mul_(2)
    graph.replay()
    expected = initial.clone()
    expected[[0, 4]] *= -2
    torch.testing.assert_close(op.output, expected, rtol=0, atol=0)
    count.fill_(2)
    op.output.fill_(17)
    graph.replay()
    torch.testing.assert_close(
        op.output[[2, 3]], torch.full_like(op.output[[2, 3]], 17)
    )


def test_invalid_scale_metadata():
    op, _, _, _, _, _, _, _, _ = fixture(128, 128, torch.float32, False)
    with pytest.raises(ValueError, match="FP32 block multipliers"):
        GroupedFP8Projection(
            op.weights, op.scales.to(torch.float16), op.output, max_panels=3
        )
