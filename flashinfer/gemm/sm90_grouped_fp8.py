# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for license information.

"""Prepared eight-slot FP8 grouped projections with ordered block128 scales."""

from __future__ import annotations

from dataclasses import dataclass
import re
import threading

import torch


@dataclass(frozen=True)
class _Config:
    bn: int
    warps: int
    stages: int
    unroll: int
    cache: str
    hot_panels: int | None

    def __post_init__(self):
        if any(
            type(v) is not int for v in (self.bn, self.warps, self.stages, self.unroll)
        ):
            raise ValueError("integer tile and pipeline parameters required")
        if self.bn not in (16, 32, 64, 128) or self.warps not in (1, 2, 4, 8):
            raise ValueError("unsupported tile size or warp count")
        if self.stages < 1 or self.unroll < 1:
            raise ValueError("positive pipeline stages and loop unroll required")
        if self.cache not in ("normal", "first", "last"):
            raise ValueError("cache must be normal, first or last")
        if self.hot_panels is not None and (
            type(self.hot_panels) is not int
            or self.hot_panels < 1
            or self.cache != "first"
        ):
            raise ValueError("hot panels require a positive count and first policy")

    @property
    def policy_key(self):
        if self.hot_panels is not None:
            return self.hot_panels + 2
        return {"normal": 0, "first": 1, "last": 2}[self.cache]


_PATCH_LOCK = threading.RLock()
_MODULES: dict[int, tuple[object, int]] = {}


def _signature(tensor):
    return (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.data_ptr(),
    )


def _validate_indices(slots, experts, *, expert_count, input_rows, output_rows, reuse):
    active = experts >= 0
    if bool((experts[active] >= expert_count).any().item()):
        raise ValueError("an active expert index exceeds the weight tensor")
    assigned = slots[active]
    assigned = assigned[assigned >= 0].to(torch.int64)
    if assigned.numel() and (
        bool((assigned >= output_rows).any().item())
        or bool((assigned // reuse >= input_rows).any().item())
    ):
        raise ValueError("an active slot exceeds the input or output tensor")
    if assigned.unique().numel() != assigned.numel():
        raise ValueError("active slots must have unique output owners")


def _patch_cache(compiled, config, pdl):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend, CUDAOptions

    if compiled.module is not None:
        raise RuntimeError("cache policy must be applied before loading the module")
    ptx = compiled.asm["ptx"]
    if "cache_hint" in ptx:
        raise RuntimeError("unexpected cache hint in an unpatched specialization")
    position = ptx.index("{", ptx.index(".reqntid"))
    declarations = "\n\t.reg .b64 %weight_policy;"
    if config.hot_panels is not None:
        declarations += "\n\t.reg .b64 %weight_hot;\n\t.reg .b32 %weight_panel;\n\t.reg .pred %weight_keep;"
    ptx = ptx[: position + 1] + declarations + ptx[position + 1 :]
    pattern = r"(?m)^(\s*)(@%[A-Za-z0-9_]+\s+)?(cp\.async\.bulk\.tensor\.2d\.shared::(?:cta|cluster)\.global\.mbarrier::complete_tx::bytes)([^\n]*);$"

    def replace(match):
        indent, predicate, operation, operands = match.groups()
        name = "evict_last" if config.cache == "last" else "evict_first"
        code = (
            indent
            + f"createpolicy.fractional.L2::{name}.b64 %weight_policy, 0f3F800000;\n"
        )
        if config.hot_panels is not None:
            code += (
                indent
                + "createpolicy.fractional.L2::evict_last.b64 %weight_hot, 0f3F800000;\n"
                + indent
                + "mov.u32 %weight_panel, %ctaid.x;\n"
                + indent
                + f"setp.lt.u32 %weight_keep, %weight_panel, {config.hot_panels};\n"
                + indent
                + "selp.b64 %weight_policy, %weight_hot, %weight_policy, %weight_keep;\n"
            )
        return (
            code
            + indent
            + (predicate or "")
            + operation
            + ".L2::cache_hint"
            + operands
            + ", %weight_policy;"
        )

    ptx, count = re.subn(pattern, replace, ptx)
    if not count or count != compiled.asm["ptx"].count("cp.async.bulk.tensor.2d."):
        raise RuntimeError("incomplete weight cache-policy specialization")
    options = CUDAOptions(
        launch_pdl=pdl,
        num_warps=compiled.metadata.num_warps,
        num_stages=compiled.metadata.num_stages,
        enable_fp_fusion=compiled.metadata.enable_fp_fusion,
    )
    cubin = CUDABackend(GPUTarget("cuda", 90, 32)).make_cubin(ptx, {}, options, 90)
    compiled.asm["ptx"], compiled.asm["cubin"], compiled.kernel = ptx, cubin, cubin


class SM90GroupedFP8Projection:
    """Apply prepared grouped FP8 projections with BF16 output.

    ``weight`` is contiguous E4M3FN ``[E,N,K]`` and ``weight_scales`` is
    contiguous FP32 ``[E,N/128,K/128]`` containing multiplier scales. N and K
    must be positive multiples of 128. Each K128 dot uses FP16 operands and
    FP32 accumulation, followed by an FP32 input-scale multiplication and an
    ordered weight-scale FMA. Weighted output multiplies in FP32 before the
    final BF16 store.

    Prepared ``slots[P,8]`` and ``experts[P]`` use int32. A negative expert
    disables a panel; negative slots disable individual rows. Active slots
    must be unique, index the output rows, and satisfy
    ``slot//input_reuse < x.shape[0]``. Unwritten output rows are unchanged.
    Routing values may change after preparation if these conditions remain
    true. ``prepare(validate_indices=True)`` checks the initial values and
    synchronizes; normal calls perform only metadata checks.

    Call ``prepare`` before CUDA graph capture. Input/output/weight storage
    and metadata are fixed thereafter; tensor values remain live. Keep this
    object and its bound tensors alive through external graph replay. Calls
    sharing output storage require stream ordering. Triton specializations
    use one CUDA primary context per process.

    ``backend='rs'`` selects the unweighted, two-stage/two-warpgroup F16
    WGMMA RS implementation; it requires PDL disabled and no hot-panel
    subset. Backend selection is explicit; Triton is the default.
    ``pdl=True`` preserves the dependent-launch trigger/wait prefix and
    launch attribute; callers must establish producer dependencies.
    """

    def __init__(
        self,
        weight,
        weight_scales,
        *,
        input_reuse=1,
        weighted=False,
        backend="triton",
        block_n=32,
        num_warps=1,
        num_stages=3,
        unroll=1,
        cache="normal",
        hot_panels=None,
        pdl=False,
    ):
        self._config = _Config(
            block_n, num_warps, num_stages, unroll, cache, hot_panels
        )
        if type(input_reuse) is not int or not 0 < input_reuse <= 2**31 - 1:
            raise ValueError("input_reuse must be a positive int32 value")
        if type(weighted) is not bool or type(pdl) is not bool:
            raise ValueError("weighted and pdl must be boolean")
        if backend not in ("triton", "rs"):
            raise ValueError("backend must be triton or rs")
        if backend == "rs" and (weighted or pdl or hot_panels is not None):
            raise ValueError(
                "RS requires unweighted output, no PDL and no hot-panel subset"
            )
        if weight.ndim != 3 or weight.dtype != torch.float8_e4m3fn:
            raise ValueError("weight must be E4M3FN [E,N,K]")
        e, n, k = weight.shape
        if min(e, n, k) < 1 or n % 128 or k % 128 or e * n > 2**31 - 1:
            raise ValueError(
                "positive block128 dimensions within int32 TMA coordinates required"
            )
        if (
            weight_scales.shape != (e, n // 128, k // 128)
            or weight_scales.dtype != torch.float32
        ):
            raise ValueError("weight_scales must be FP32 [E,N/128,K/128]")
        if weight.device.type != "cuda" or weight_scales.device != weight.device:
            raise ValueError("weights and scales must share one CUDA device")
        if (
            not weight.is_contiguous()
            or not weight_scales.is_contiguous()
            or weight.data_ptr() % 16
        ):
            raise ValueError("contiguous, aligned weights and scales required")
        self.weight, self.weight_scales = weight, weight_scales
        self._input_reuse, self._weighted, self._pdl = input_reuse, weighted, pdl
        self._backend = backend
        self.device, self.e, self.n, self.k = weight.device, e, n, k
        self._weight_signature = _signature(weight), _signature(weight_scales)
        self._lock = threading.RLock()
        self._bindings = self._binding_signature = None
        self._compiled = self._native = self._descriptor = None
        self._context = None

    @property
    def config(self):
        return self._config

    @property
    def input_reuse(self):
        return self._input_reuse

    @property
    def weighted(self):
        return self._weighted

    @property
    def pdl(self):
        return self._pdl

    @property
    def backend(self):
        return self._backend

    def _validate(self, x, xs, slots, experts, out, routing_weights):
        if (
            _signature(self.weight),
            _signature(self.weight_scales),
        ) != self._weight_signature:
            raise RuntimeError("weight storage or metadata changed")
        if x.ndim != 2 or x.shape[1] != self.k or x.dtype != torch.float8_e4m3fn:
            raise ValueError("x must be E4M3FN [M,K]")
        if xs.shape != (x.shape[0], self.k // 128) or xs.dtype != torch.float32:
            raise ValueError("x_scales must be FP32 [M,K/128]")
        if slots.ndim != 2 or slots.shape[1] != 8 or slots.dtype != torch.int32:
            raise ValueError("slots must be int32 [P,8]")
        if experts.shape != (slots.shape[0],) or experts.dtype != torch.int32:
            raise ValueError("experts must be int32 [P]")
        if out.ndim != 2 or out.shape[1] != self.n or out.dtype != torch.bfloat16:
            raise ValueError("out must be BF16 [S,N]")
        tensors = (x, xs, slots, experts, out)
        if self.weighted:
            if (
                routing_weights is None
                or routing_weights.dtype != torch.float32
                or routing_weights.numel() != out.shape[0]
            ):
                raise ValueError(
                    "weighted output requires one FP32 routing weight per output row"
                )
            tensors += (routing_weights,)
        elif routing_weights is not None:
            raise ValueError("routing_weights require weighted=True")
        if min(x.shape[0], slots.shape[0], out.shape[0]) < 1:
            raise ValueError("positive input, output and panel counts required")
        if any(t.device != self.device or not t.is_contiguous() for t in tensors):
            raise ValueError(
                "all inputs and output must be contiguous on the weight device"
            )
        if max(x.numel(), out.numel(), slots.numel()) > 2**31 - 1:
            raise ValueError("input and output indexing must fit int32")
        if x.data_ptr() % 16 or out.data_ptr() % 16:
            raise ValueError("input and output bases must be 16-byte aligned")
        lo, hi = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
        for tensor in (*tensors[:4], self.weight, self.weight_scales, *tensors[5:]):
            start, stop = (
                tensor.data_ptr(),
                tensor.data_ptr() + tensor.numel() * tensor.element_size(),
            )
            if max(start, lo) < min(stop, hi):
                raise ValueError("output storage must not overlap any input")
        return tensors

    def _arguments(self, tensors):
        x, xs, slots, experts, out, *weights = tensors
        return (
            x,
            xs,
            self._descriptor,
            self.weight_scales,
            slots,
            experts,
            weights[0] if weights else None,
            out,
            self.n,
            self.k,
            self.input_reuse,
            self.weighted,
            self.config.bn,
            self.config.unroll,
            self.pdl,
            self.config.policy_key,
        )

    def prepare(
        self,
        x,
        x_scales,
        slots,
        experts,
        out,
        routing_weights=None,
        *,
        validate_indices=False,
    ):
        """Compile and bind storage before capture, without executing a projection."""
        with self._lock, torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare must precede CUDA graph capture")
            tensors = self._validate(x, x_scales, slots, experts, out, routing_weights)
            signature = tuple(_signature(t) for t in tensors)
            if type(validate_indices) is not bool:
                raise ValueError("validate_indices must be boolean")
            if validate_indices:
                _validate_indices(
                    slots,
                    experts,
                    expert_count=self.e,
                    input_rows=x.shape[0],
                    output_rows=out.shape[0],
                    reuse=self.input_reuse,
                )
            if self._bindings is not None:
                if signature != self._binding_signature:
                    raise RuntimeError(
                        "construct a new projection for different bound storage"
                    )
                return self
            if torch.cuda.get_device_capability(self.device) != (9, 0):
                raise ValueError("SM90 hardware required")
            from . import _sm90_grouped_fp8_native as native

            self._native_tools = native
            self._context = native._retain_context(self.device)
            if self.backend == "rs":
                self._native = native.Projection(
                    self.weight,
                    self.weight_scales,
                    self.input_reuse,
                    self.config.policy_key,
                )
            else:
                from triton.tools.tensor_descriptor import TensorDescriptor

                from .kernels.sm90_grouped_fp8 import grouped

                self._descriptor = TensorDescriptor.from_tensor(
                    self.weight.reshape(-1, self.k), [self.config.bn, 128]
                )
                args = self._arguments(tensors)
                compiled = grouped.warmup(
                    *args,
                    grid=(experts.numel(), self.n // self.config.bn, 1),
                    num_warps=self.config.warps,
                    num_stages=self.config.stages,
                    launch_pdl=self.pdl,
                )
                with _PATCH_LOCK:
                    retained = _MODULES.get(id(compiled))
                    if retained is None:
                        if self.config.policy_key:
                            _patch_cache(compiled, self.config, self.pdl)
                        compiled._init_handles()
                        _MODULES[id(compiled)] = compiled, self._context
                    elif retained[1] != self._context:
                        raise RuntimeError(
                            "one CUDA primary context per process is required"
                        )
                self._compiled = compiled
            self._bindings, self._binding_signature = tensors, signature
            return self

    def __call__(self, x, x_scales, slots, experts, out, routing_weights=None):
        """Run against the prepared storage; tensor values are read on each launch."""
        with self._lock, torch.cuda.device(self.device):
            if self._bindings is None:
                raise RuntimeError("call prepare before running the projection")
            tensors = self._validate(x, x_scales, slots, experts, out, routing_weights)
            if tuple(_signature(t) for t in tensors) != self._binding_signature:
                raise RuntimeError("prepared tensor storage or metadata changed")
            if self._native_tools.current_context() != self._context:
                raise RuntimeError("the prepared CUDA context changed")
            if self._native is not None:
                self._native.run(x, x_scales, slots, experts, out)
            else:
                self._compiled[(experts.numel(), self.n // self.config.bn, 1)](
                    *self._arguments(tensors)
                )
            return out

    @property
    def compiled_kernel(self):
        """The prepared Triton kernel, or None for the native backend."""
        return self._compiled

    @property
    def native_projection(self):
        """The prepared native owner, or None for the Triton backend."""
        return self._native

    @property
    def artifacts(self):
        """Actual executable bytes and launch resources after preparation."""
        if self._bindings is None:
            raise RuntimeError("prepare the projection before requesting artifacts")
        if self._native is not None:
            return dict(
                backend="rs",
                cubin=self._native.cubin.read_bytes(),
                ptx=None,
                resources=dict(self._native.resources),
                build_record=self._native.build_record,
            )
        c = self._compiled
        return dict(
            backend="triton",
            cubin=c.asm["cubin"],
            ptx=c.asm["ptx"],
            resources=dict(
                registers=c.n_regs,
                spills=c.n_spills,
                shared=c.metadata.shared,
                launch_pdl=c.metadata.launch_pdl,
                warps=c.metadata.num_warps,
                stages=c.metadata.num_stages,
            ),
        )
