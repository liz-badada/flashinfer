# SPDX-License-Identifier: Apache-2.0
"""Strict block-scaled FP8 grouped projection for SM90.

Callers provide a panel plan. Each eight-slot panel selects one weight matrix;
negative slots are padding. Scales multiply each 128-element dot partial in
FP32 before accumulation. The optional coefficient applies before output cast.
"""

import torch
import triton as tr
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@tr.jit
def _project(
    X,
    XS,
    WS,
    SLOTS,
    EXPERTS,
    COUNT,
    COEFFICIENTS,
    OUT,
    W_DESC,
    N: tl.constexpr,
    K: tl.constexpr,
    REUSE: tl.constexpr,
    WEIGHTED: tl.constexpr,
    BN: tl.constexpr,
    UNROLL: tl.constexpr,
):
    panel = tl.program_id(0)
    if panel < tl.load(COUNT):
        slots = tl.load(SLOTS + panel * 8 + tl.arange(0, 8))
        expert = tl.load(EXPERTS + panel).to(tl.int64)
        valid = slots >= 0
        rows = tl.maximum(slots, 0) // REUSE
        n_tile = tl.program_id(1)
        n = n_tile * BN + tl.arange(0, BN)
        k = tl.arange(0, 128)
        acc = tl.full((8, BN), 0.0, tl.float32)
        for kb in tl.range(K // 128, loop_unroll_factor=UNROLL):
            x = tl.load(
                X + rows[:, None] * K + kb * 128 + k[None, :],
                valid[:, None],
                0.0,
            )
            w = W_DESC.load([(expert * N + n_tile * BN).to(tl.int32), kb * 128])
            xs = tl.load(XS + rows * (K // 128) + kb, valid, 0.0)
            ws = tl.load(
                WS + (expert * (N // 128) + n // 128) * (K // 128) + kb,
                n < N,
                0.0,
            )
            partial = tl.trans(
                tl.dot(
                    w,
                    tl.trans(x),
                    out_dtype=tl.float32,
                    max_num_imprecise_acc=0,
                )
            )
            acc += partial * xs[:, None] * ws[None, :]
        if WEIGHTED:
            acc *= tl.load(COEFFICIENTS + slots, valid, 0.0)[:, None]
        tl.store(
            OUT + slots[:, None] * N + n[None, :],
            acc,
            valid[:, None] & (n[None, :] < N),
        )


class GroupedFP8Projection:
    """Preallocated grouped projection with live weight and routing tensors.

    Weights are contiguous E4M3FN [groups, N, K], scales are FP32
    [groups, N/128, K/128], and N/K must be multiples of 128. Output is
    contiguous BF16 or FP32 [slots, N]. A plan must cover each output slot
    exactly once, with valid group indices and count <= max_panels. Validate
    plan values before capture. A workspace belongs to one execution stream.
    """

    def __init__(
        self,
        weights,
        scales,
        output,
        *,
        max_panels,
        input_reuse=1,
        config=(64, 4, 3, 1),
    ):
        if weights.ndim != 3 or weights.dtype != torch.float8_e4m3fn:
            raise ValueError("weights must be E4M3FN [groups,N,K]")
        groups, n, k = weights.shape
        if min(groups, n, k) <= 0 or n % 128 or k % 128:
            raise ValueError("positive N/K dimensions must be multiples of 128")
        if (
            scales.shape != (groups, n // 128, k // 128)
            or scales.dtype != torch.float32
        ):
            raise ValueError("scales must be matching FP32 block multipliers")
        if (
            output.ndim != 2
            or output.shape[1] != n
            or output.dtype
            not in (
                torch.float32,
                torch.bfloat16,
            )
        ):
            raise ValueError("output must be BF16 or FP32 [slots,N]")
        if (
            type(input_reuse) is not int
            or input_reuse <= 0
            or output.shape[0] % input_reuse
        ):
            raise ValueError("input_reuse must divide the output slot count")
        if type(max_panels) is not int or max_panels <= 0 or output.shape[0] <= 0:
            raise ValueError("panel and output capacities must be positive")
        if (
            len(config) != 4
            or any(type(v) is not int for v in config)
            or config[0] not in (16, 32, 64, 128, 256)
            or config[1] not in (1, 2, 4, 8)
            or config[2] not in (2, 3, 4)
            or config[3] not in (1, 2, 4)
        ):
            raise ValueError("invalid tile/warps/stages/unroll configuration")
        self.device = weights.device
        if self.device.type != "cuda":
            raise ValueError("SM90 CUDA tensors are required")
        for tensor in (weights, scales, output):
            if tensor.device != self.device or not tensor.is_contiguous():
                raise ValueError("tensors must be contiguous on one CUDA device")
        if torch.cuda.get_device_capability(self.device) != (9, 0):
            raise ValueError("this kernel requires SM90")
        self.weights, self.scales, self.output = weights, scales, output
        self.max_panels, self.input_reuse, self.config = max_panels, input_reuse, config
        self.n, self.k = n, k
        with torch.cuda.device(self.device):
            self.descriptor = TensorDescriptor.from_tensor(
                weights.view(groups * n, k),
                [config[0], 128],
            )

    def __call__(self, x, x_scales, slots, groups, count, coefficients=None):
        rows = self.output.shape[0] // self.input_reuse
        if x.shape != (rows, self.k) or x.dtype != torch.float8_e4m3fn:
            raise ValueError("activation shape or FP8 dtype mismatch")
        if x_scales.shape != (rows, self.k // 128) or x_scales.dtype != torch.float32:
            raise ValueError("activation scales must be FP32 block multipliers")
        if slots.shape != (self.max_panels, 8) or slots.dtype != torch.int32:
            raise ValueError("slots must be int32 [max_panels,8]")
        if groups.shape != (self.max_panels,) or groups.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("group indices must be int32/int64 [max_panels]")
        if count.shape != (1,) or count.dtype != torch.int32:
            raise ValueError("count must be int32 [1]")
        tensors = [x, x_scales, slots, groups, count]
        if coefficients is not None:
            if (
                coefficients.shape != (self.output.shape[0],)
                or coefficients.dtype != torch.float32
            ):
                raise ValueError("coefficients must be FP32 [slots]")
            tensors.append(coefficients)
        if any(t.device != self.device or not t.is_contiguous() for t in tensors):
            raise ValueError("inputs must be contiguous on the weight device")
        bn, warps, stages, unroll = self.config
        with torch.cuda.device(self.device):
            _project[(self.max_panels, tr.cdiv(self.n, bn))](
                x,
                x_scales,
                self.scales,
                slots,
                groups,
                count,
                coefficients if coefficients is not None else count,
                self.output,
                self.descriptor,
                self.n,
                self.k,
                self.input_reuse,
                coefficients is not None,
                bn,
                unroll,
                num_warps=warps,
                num_stages=stages,
            )
        return self.output
