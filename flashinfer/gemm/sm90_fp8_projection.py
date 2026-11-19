# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""SM90 FP8 projections with ordered block-scale accumulation."""

import threading
from types import MappingProxyType
from typing import Any, Optional

import torch

from ..api_logging import flashinfer_api


def _signature(tensor):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _no_overlap(output, *inputs):
    start, stop = (
        output.data_ptr(),
        output.data_ptr() + output.numel() * output.element_size(),
    )
    for tensor in inputs:
        other = tensor.data_ptr()
        if start < other + tensor.numel() * tensor.element_size() and other < stop:
            raise ValueError("output must not overlap an input or another output")


class SM90FP8Projection:
    r"""Project FP8 activations using 128-by-128 block-scaled FP8 weights.

    ``weight`` has shape ``[N, K]`` and dtype ``float8_e4m3fn``;
    ``weight_scales`` contains FP32 multiplicative scales shaped
    ``[N // 128, K // 128]``. Inputs use per-token, per-128 FP32 scales.
    N and K must be positive multiples of 128.

    :meth:`raw` returns unscaled FP32 dot products shaped ``[K // 128, T, N]``.
    :meth:`quantize` accumulates these in increasing K-block order, rounds to
    BF16, then quantizes each 128 output elements to E4M3. :meth:`bf16` performs
    the scaled projection directly; :meth:`mix` adds two independently rounded
    BF16 products without communicating between devices.

    Outputs omitted by the caller are allocated lazily and reused. Calls using
    the same output storage must be ordered across streams. Keep this object,
    inputs, and outputs alive while a CUDA graph uses them. Weight values and
    scales may change in place, but their storage and metadata must stay fixed.

    Parameters
    ----------
    weight : torch.Tensor
        Contiguous E4M3 weight matrix on an SM90 CUDA device.
    weight_scales : torch.Tensor
        Contiguous FP32 block scales on the weight device.
    tokens : int
        Fixed number of activation rows T.
    block_n : int
        Output tile width, 16 or 128.
    num_warps : int
        Number of warps for raw projection, 1 or 4.
    k_group : int
        K128 blocks assigned to each raw-projection CTA, 1 or 4.
    pdl : bool
        Enable programmatic dependent launch for :meth:`raw`. The raw kernel
        triggers dependents and then waits before reading its activation input.
        Consumers must wait for its output before reading partials.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        tokens: int,
        *,
        block_n: int = 128,
        num_warps: int = 4,
        k_group: int = 1,
        pdl: bool = False,
    ):
        for name, value in (
            ("tokens", tokens),
            ("block_n", block_n),
            ("num_warps", num_warps),
            ("k_group", k_group),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if block_n not in (16, 128) or num_warps not in (1, 4) or k_group not in (1, 4):
            raise ValueError(
                "supported block_n/num_warps/k_group are (16,128)/(1,4)/(1,4)"
            )
        if not isinstance(pdl, bool):
            raise TypeError("pdl must be bool")
        if weight.dtype != torch.float8_e4m3fn or weight_scales.dtype != torch.float32:
            raise TypeError("weight must be E4M3 and weight_scales must be FP32")
        if weight.ndim != 2 or any(size < 1 or size % 128 for size in weight.shape):
            raise ValueError(
                "weight must be [N,K] with positive N and K divisible by 128"
            )
        n, k = weight.shape
        if weight_scales.shape != (n // 128, k // 128):
            raise ValueError("weight_scales must have shape [N//128,K//128]")
        if not weight.is_contiguous() or not weight_scales.is_contiguous():
            raise ValueError("weights and scales must be contiguous")
        if max(n * k, tokens * k, (k // 128) * tokens * n) >= 2**31:
            raise ValueError("projection exceeds the supported 32-bit indexing range")
        if weight.device.type != "cuda" or weight_scales.device != weight.device:
            raise ValueError("weights and scales must use the same CUDA device")
        self.device = weight.device
        with torch.cuda.device(self.device):
            if torch.cuda.get_device_capability(self.device) != (9, 0):
                raise ValueError("SM90 CUDA device required")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("construct the projection before CUDA graph capture")
            if weight.data_ptr() % 16 or weight_scales.data_ptr() % 16:
                raise ValueError("weights and scales must be 16-byte aligned")
            from triton.tools.tensor_descriptor import TensorDescriptor
            from .kernels import sm90_fp8_projection as kernels

            self._descriptor = TensorDescriptor.from_tensor(weight, [block_n, 128])
        self._weight, self._weight_scales = weight, weight_scales
        self._signatures = (_signature(weight), _signature(weight_scales))
        self._t, self._n, self._k = tokens, n, k
        self._block_n, self._warps, self._k_group, self._pdl = (
            block_n,
            num_warps,
            k_group,
            pdl,
        )
        self._kernels = kernels
        self._buffers: dict[str, torch.Tensor] = {}
        self._compiled: dict[str, Any] = {}
        self._lock = threading.RLock()

    @property
    def raw_shape(self):
        return (self._k // 128, self._t, self._n)

    @property
    def compiled_kernels(self):
        """Read-only mapping from method name to its compiled specialization."""
        return MappingProxyType(self._compiled)

    def _weights_unchanged(self):
        if self._signatures != (
            _signature(self._weight),
            _signature(self._weight_scales),
        ):
            raise RuntimeError(
                "weight storage or metadata changed; create a new projection"
            )

    def _tensor(self, tensor, shape, dtype, name):
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
        if tuple(tensor.shape) != tuple(shape) or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous with shape {shape}")
        if tensor.device != self.device or tensor.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned on {self.device}")

    def _output(self, value, name, shape, dtype):
        if value is None:
            if name not in self._buffers:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "provide output buffers or warm up before capture"
                    )
                self._buffers[name] = torch.empty(
                    shape, dtype=dtype, device=self.device
                )
            value = self._buffers[name]
        self._tensor(value, shape, dtype, name)
        return value

    def _launch(self, method, kernel, arguments, grid, options, compile_only):
        capturing = torch.cuda.is_current_stream_capturing()
        if compile_only:
            if capturing:
                raise RuntimeError("warmup must run before CUDA graph capture")
            compiled = kernel.warmup(*arguments, grid=grid, **options)
            compiled._init_handles()
        else:
            if capturing and method not in self._compiled:
                raise RuntimeError(f"warm up {method} before CUDA graph capture")
            compiled = kernel[grid](*arguments, **options)
        self._compiled[method] = compiled

    @flashinfer_api
    def warmup(self, method, *args, **kwargs):
        """Compile and initialize one method without launching a GPU kernel.

        For example, ``warmup("raw", q, out=raw)`` accepts the same arguments
        as :meth:`raw`. Supported methods are raw, quantize, bf16, and mix.
        Caller-provided buffers are left untouched, including for PDL kernels.
        """
        if method not in ("raw", "quantize", "bf16", "mix"):
            raise ValueError("method must be raw, quantize, bf16, or mix")
        if "_warmup" in kwargs:
            raise TypeError("_warmup is reserved")
        getattr(self, "_" + method)(*args, _warmup=True, **kwargs)
        return self

    @flashinfer_api
    def raw(
        self, q: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return FP32 [K//128,T,N] unscaled K128 dot products."""
        return self._raw(q, out=out)

    def _raw(self, q, *, out=None, _warmup=False):
        with self._lock, torch.cuda.device(self.device):
            self._weights_unchanged()
            self._tensor(q, (self._t, self._k), torch.float8_e4m3fn, "q")
            out = self._output(out, "raw", self.raw_shape, torch.float32)
            _no_overlap(out, q, self._weight, self._weight_scales)
            arguments = (
                q,
                self._descriptor,
                out,
                self._t,
                self._n,
                self._k,
                self._k_group,
                self._block_n,
                self._pdl,
            )
            grid = (
                (self._t + 7) // 8,
                self._n // self._block_n,
                (self._k // 128 + self._k_group - 1) // self._k_group,
            )
            self._launch(
                "raw",
                self._kernels._raw_projection,
                arguments,
                grid,
                dict(num_warps=self._warps, num_stages=2, launch_pdl=self._pdl),
                _warmup,
            )
            return out

    @flashinfer_api
    def quantize(
        self,
        raw: torch.Tensor,
        activation_scales: torch.Tensor,
        *,
        output_q: Optional[torch.Tensor] = None,
        output_scales: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply scales in K order, round to BF16, and requantize per 128."""
        return self._quantize(
            raw, activation_scales, output_q=output_q, output_scales=output_scales
        )

    def _quantize(
        self,
        raw,
        activation_scales,
        *,
        output_q=None,
        output_scales=None,
        _warmup=False,
    ):
        with self._lock, torch.cuda.device(self.device):
            self._weights_unchanged()
            self._tensor(raw, self.raw_shape, torch.float32, "raw")
            self._tensor(
                activation_scales,
                (self._t, self._k // 128),
                torch.float32,
                "activation_scales",
            )
            q = self._output(
                output_q, "output_q", (self._t, self._n), torch.float8_e4m3fn
            )
            scales = self._output(
                output_scales, "output_scales", (self._t, self._n // 128), torch.float32
            )
            _no_overlap(
                q, raw, activation_scales, self._weight, self._weight_scales, scales
            )
            _no_overlap(
                scales, raw, activation_scales, self._weight, self._weight_scales
            )
            self._launch(
                "quantize",
                self._kernels._finish_quant,
                (
                    raw,
                    activation_scales,
                    self._weight_scales,
                    q,
                    scales,
                    self._t,
                    self._n,
                    self._k,
                ),
                (self._t, self._n // 128),
                dict(num_warps=1, num_stages=3),
                _warmup,
            )
            return q, scales

    @flashinfer_api
    def bf16(
        self,
        q: torch.Tensor,
        activation_scales: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a BF16 scaled projection in one kernel."""
        return self._bf16(q, activation_scales, out=out)

    def _bf16(self, q, activation_scales, *, out=None, _warmup=False):
        with self._lock, torch.cuda.device(self.device):
            self._weights_unchanged()
            self._tensor(q, (self._t, self._k), torch.float8_e4m3fn, "q")
            self._tensor(
                activation_scales,
                (self._t, self._k // 128),
                torch.float32,
                "activation_scales",
            )
            out = self._output(out, "bf16", (self._t, self._n), torch.bfloat16)
            _no_overlap(out, q, activation_scales, self._weight, self._weight_scales)
            self._launch(
                "bf16",
                self._kernels._dense_bf16,
                (
                    q,
                    activation_scales,
                    self._descriptor,
                    self._weight_scales,
                    out,
                    self._t,
                    self._n,
                    self._k,
                    self._block_n,
                ),
                ((self._t + 7) // 8, self._n // self._block_n),
                dict(num_warps=1, num_stages=3),
                _warmup,
            )
            return out

    @flashinfer_api
    def mix(
        self,
        q: torch.Tensor,
        activation_scales: torch.Tensor,
        shared: torch.Tensor,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return BF16(RN_BF16(projection*alpha) + RN_BF16(shared*gamma)).

        Projection, alpha, shared, and gamma are rounded to BF16 before their
        products. Gates are contiguous FP32 [T] or [T,1] tensors.
        """
        return self._mix(q, activation_scales, shared, alpha, gamma, out=out)

    def _mix(
        self, q, activation_scales, shared, alpha, gamma, *, out=None, _warmup=False
    ):
        with self._lock, torch.cuda.device(self.device):
            self._weights_unchanged()
            self._tensor(q, (self._t, self._k), torch.float8_e4m3fn, "q")
            self._tensor(
                activation_scales,
                (self._t, self._k // 128),
                torch.float32,
                "activation_scales",
            )
            self._tensor(shared, (self._t, self._n), torch.bfloat16, "shared")
            for name, gate in (("alpha", alpha), ("gamma", gamma)):
                if tuple(gate.shape) not in ((self._t,), (self._t, 1)):
                    raise ValueError(f"{name} must be [T] or [T,1]")
                self._tensor(gate, gate.shape, torch.float32, name)
            out = self._output(out, "mix", (self._t, self._n), torch.bfloat16)
            _no_overlap(
                out,
                q,
                activation_scales,
                shared,
                alpha,
                gamma,
                self._weight,
                self._weight_scales,
            )
            self._launch(
                "mix",
                self._kernels._project_mix,
                (
                    q,
                    activation_scales,
                    self._descriptor,
                    self._weight_scales,
                    shared,
                    alpha,
                    gamma,
                    out,
                    self._n,
                    self._k,
                    self._block_n,
                    self._t,
                ),
                ((self._t + 7) // 8, self._n // self._block_n),
                dict(num_warps=1, num_stages=3),
                _warmup,
            )
            return out

    @flashinfer_api
    def __call__(
        self,
        q,
        activation_scales,
        *,
        raw_output=None,
        output_q=None,
        output_scales=None,
    ):
        """Run raw projection and ordered requantization using two kernels."""
        with self._lock:
            raw = self._raw(q, out=raw_output)
            return self._quantize(
                raw, activation_scales, output_q=output_q, output_scales=output_scales
            )
