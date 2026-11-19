"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import socket
import threading
from typing import Any

import torch
import torch.distributed as dist

from ._sm90_latent_tp4_config import (
    LatentMoeProjectionConfig,
    LatentMoeTP4Config,
    _check_dimensions,
)

__all__ = [
    "FP8BlockWeight",
    "LatentMoeTP4Weights",
    "LatentMoeProjectionConfig",
    "LatentMoeTP4Config",
    "LatentMoeTP4",
    "LatentMoeTP4Graph",
    "shard_fp8_experts_tp4",
]


_GRAPH_LEASE = object()


@dataclass(frozen=True)
class FP8BlockWeight:
    """Contiguous E4M3FN matrix with FP32 multiplier scales per 128x128 block."""

    values: torch.Tensor
    scales: torch.Tensor

    def __post_init__(self):
        if self.values.ndim not in (2, 3) or self.values.dtype != torch.float8_e4m3fn:
            raise ValueError("canonical rank2/rank3 E4M3FN weights required")
        n, width = self.values.shape[-2:]
        if any(dim < 1 for dim in self.values.shape) or n % 128 or width % 128:
            raise ValueError("nonempty block128 matrices required")
        if (
            self.scales.shape != (*self.values.shape[:-2], n // 128, width // 128)
            or self.scales.dtype != torch.float32
        ):
            raise ValueError("FP32 multiplier scales must match block128 layout")
        if (
            self.values.device != self.scales.device
            or not self.values.is_contiguous()
            or not self.scales.is_contiguous()
        ):
            raise ValueError("contiguous values/scales on one device required")

    @property
    def shape(self):
        return self.values.shape


@dataclass(frozen=True)
class LatentMoeTP4Weights:
    """Local TP4 expert/shared shards and replicated down/up matrices."""

    first: FP8BlockWeight
    second: FP8BlockWeight
    down: FP8BlockWeight
    up: FP8BlockWeight
    shared_first: FP8BlockWeight
    shared_second: FP8BlockWeight


def shard_fp8_experts_tp4(first, second, rank):
    if type(rank) is not int or rank not in range(4) or second.values.ndim != 3:
        raise ValueError("TP4 expert matrices and rank required")
    e, latent, width = second.shape
    if first.shape != (e, 2 * width, latent) or width % 512:
        raise ValueError("paired gate/up and block128 TP shards required")
    start, stop = rank * (width // 4), (rank + 1) * (width // 4)
    a, b, mid = start // 128, stop // 128, width // 128
    return (
        FP8BlockWeight(
            torch.cat(
                (
                    first.values[:, start:stop],
                    first.values[:, width + start : width + stop],
                ),
                1,
            ),
            torch.cat((first.scales[:, a:b], first.scales[:, mid + a : mid + b]), 1),
        ),
        FP8BlockWeight(
            second.values[:, :, start:stop].contiguous(),
            second.scales[:, :, a:b].contiguous(),
        ),
    )


def _signature(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


def _byte_span(tensor):
    low = high = 0
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        extent = (size - 1) * stride
        low += min(0, extent)
        high += max(0, extent)
    start = tensor.data_ptr()
    item = tensor.element_size()
    return start + low * item, start + (high + 1) * item


def _check_input_workspace_overlap(inputs, buffers):
    for value in inputs:
        if not value.numel():
            continue
        low, high = _byte_span(value)
        for buffer in buffers:
            if buffer.device != value.device or not buffer.numel():
                continue
            other_low, other_high = _byte_span(buffer)
            if low < other_high and other_low < high:
                raise ValueError("inputs must not overlap workspace-owned storage")


class _IPC:
    def __init__(self, rank, tokens, hidden, group):
        from flashinfer.comm.trtllm_ar import (
            trtllm_create_ipc_workspace_for_all_reduce_fusion,
        )

        self.rank, self.group = rank, group
        self.handles, self.workspace, self.metadata = (
            trtllm_create_ipc_workspace_for_all_reduce_fusion(
                tp_rank=rank,
                tp_size=4,
                max_token_num=tokens,
                hidden_dim=hidden,
                group=group,
                create_metadata=True,
                use_symm_dev_mem=False,
            )
        )
        self.comm_bytes = self.metadata["lamport_comm_size"]

    def reduce(self, value, output):
        from flashinfer.comm.trtllm_ar import (
            AllReduceFusionPattern,
            trtllm_allreduce_fusion,
        )

        trtllm_allreduce_fusion(
            value,
            world_size=4,
            world_rank=self.rank,
            token_num=value.shape[0],
            hidden_dim=value.shape[1],
            workspace_ptrs=self.workspace,
            launch_with_pdl=False,
            trigger_completion_at_end=False,
            fp32_acc=True,
            pattern_code=AllReduceFusionPattern.kAllReduce,
            use_oneshot=True,
            allreduce_out=output,
            residual_in=None,
            residual_out=None,
            norm_out=None,
            quant_out=None,
            scale_out=None,
            rms_gamma=None,
            rms_eps=None,
            scale_factor=None,
            layout_code=None,
            metadata=self.metadata,
        )
        return output

    def close(self):
        from flashinfer.comm.trtllm_ar import (
            trtllm_destroy_ipc_workspace_for_all_reduce_fusion,
        )

        trtllm_destroy_ipc_workspace_for_all_reduce_fusion(self.handles, self.group)


class LatentMoeTP4:
    """SM90 latent MoE execution on one host with four distinct CUDA ranks.

    ``first`` stores gate then up rows, and ``second`` consumes each rank's
    local expert width. Shared FFN weights are also TP shards. Down/up weights
    are replicated; peer down computes four output-row shards and exchanges
    their quantized results. Input and output are BF16 ``[tokens, hidden]``.

    Call :meth:`prepare` collectively before execution or CUDA graph capture.
    Inputs and weights remain live at their prepared storage addresses. Calls
    on one workspace must be serialized. Keep an externally captured graph
    and its workspace alive together, or use :meth:`capture` for an owned lease.
    Close all graphs before collectively closing the workspace. This class
    owns scratch and IPC resources; it does not own the supplied weights.
    """

    def __init__(
        self,
        weights: LatentMoeTP4Weights,
        *,
        tokens: int,
        topk: int,
        config: LatentMoeTP4Config | None = None,
        group=None,
    ):
        if not dist.is_initialized() or dist.get_world_size(group) != 4:
            raise ValueError("an initialized four-rank NCCL process group is required")
        if dist.get_backend(group) != "nccl":
            raise ValueError("NCCL process group required")
        self.group = group
        self.rank = dist.get_rank(group)
        self.weights, self.tokens, self.topk = weights, tokens, topk
        self.config = config or LatentMoeTP4Config()
        self._lock = threading.RLock()
        self._graph = None
        self._stream = None
        self._closed = self._prepared = False
        failure = None
        try:
            self.device = weights.down.values.device
            if self.device.type != "cuda" or torch.cuda.get_device_capability(
                self.device
            ) != (9, 0):
                raise ValueError("SM90 CUDA weights required")
            if torch.cuda.current_device() != self.device.index:
                raise ValueError(
                    "select the weight device before constructing a workspace"
                )
            device_uuid = str(torch.cuda.get_device_properties(self.device).uuid)
            self.e, self.l, self.f = weights.second.shape
            self.h = weights.down.shape[1]
            self.sf = weights.shared_second.shape[1]
            expected = (
                (self.e, 2 * self.f, self.l),
                (self.l, self.h),
                (self.h, self.l),
                (2 * self.sf, self.h),
                (self.h, self.sf),
            )
            actual = (
                weights.first.shape,
                weights.down.shape,
                weights.up.shape,
                weights.shared_first.shape,
                weights.shared_second.shape,
            )
            if actual != expected:
                raise ValueError("inconsistent local projection dimensions")
            for weight in vars(weights).values():
                if weight.values.device != self.device:
                    raise ValueError("all weights must use the same device")
            _check_dimensions(
                tokens, self.h, self.l, self.e, self.f, self.sf, topk, self.config
            )
            spec = (
                tokens,
                topk,
                self.h,
                self.l,
                self.e,
                self.f,
                self.sf,
                asdict(self.config),
                socket.gethostname(),
            )
        except Exception as exc:
            failure, spec, device_uuid = str(exc), None, None
        statuses: list[Any] = [None] * 4
        dist.all_gather_object(statuses, (failure, spec, device_uuid), group=group)
        if (
            any(x[0] for x in statuses)
            or any(x[1] != statuses[0][1] for x in statuses)
            or len({x[2] for x in statuses}) != 4
        ):
            raise ValueError(statuses)
        self._weight_refs = tuple(
            t for w in vars(weights).values() for t in (w.values, w.scales)
        )
        self._weight_signatures = tuple(_signature(t) for t in self._weight_refs)
        cap_t = self.config.comm_tokens or tokens
        cap_h = self.config.comm_hidden or max(self.h, self.l)
        self._sum_link = _IPC(self.rank, cap_t, cap_h, group)
        self._peer_link = (
            _IPC(self.rank, cap_t, cap_h, group)
            if self.config.down_mode == "peer"
            else None
        )
        failure = None
        try:
            with torch.cuda.device(self.device):
                self._allocate()
                if self.config.up_mode == "split":
                    self._preflight()
        except Exception as exc:
            failure = str(exc)
        failures = [None] * 4
        dist.all_gather_object(failures, failure, group=group)
        if any(failures):
            torch.cuda.synchronize(self.device)
            if self._peer_link is not None:
                self._peer_link.close()
            self._sum_link.close()
            self._closed = True
            raise RuntimeError(failures)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=group)

    def _empty(self, shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=self.device)

    def _allocate(self):
        from . import _sm90_latent_tp4_kernels

        self._kernels = _sm90_latent_tp4_kernels
        from flashinfer.gemm.sm90_fp8_projection import SM90FP8Projection
        from flashinfer.gemm.sm90_grouped_fp8 import SM90GroupedFP8Projection

        w, c, t, s = self.weights, self.config, self.tokens, self.tokens * self.topk
        self.panel_slots = self._empty((s, 8), torch.int32)
        self.panel_experts = self._empty((s,), torch.int32)
        self.xq, self.xs = (
            self._empty((t, self.h), torch.float8_e4m3fn),
            self._empty((t, self.h // 128), torch.float32),
        )
        self.q, self.qs = (
            self._empty((t, self.l), torch.float8_e4m3fn),
            self._empty((t, self.l // 128), torch.float32),
        )
        self.gate_up = self._empty((s, 2 * self.f))
        self.hq, self.hs = (
            self._empty((s, self.f), torch.float8_e4m3fn),
            self._empty((s, self.f // 128), torch.float32),
        )
        self.partials = self._empty((s, self.l))
        self.shared_gate_up, self.shared_output = (
            self._empty((t, 2 * self.sf)),
            self._empty((t, self.h)),
        )
        self.sq, self.ss = (
            self._empty((t, self.sf), torch.float8_e4m3fn),
            self._empty((t, self.sf // 128), torch.float32),
        )
        self.uq, self.us = (
            self._empty((t, self.l), torch.float8_e4m3fn),
            self._empty((t, self.l // 128), torch.float32),
        )
        self.output, self.local_output = (
            self._empty((t, self.h)),
            self._empty((t, self.h)),
        )
        self._shared_stream = (
            torch.cuda.Stream(device=self.device) if c.overlap_shared else None
        )
        self._shared_first = SM90FP8Projection(
            w.shared_first.values,
            w.shared_first.scales,
            t,
            block_n=c.shared_bn,
            num_warps=1,
        )
        self._shared_second = SM90FP8Projection(
            w.shared_second.values,
            w.shared_second.scales,
            t,
            block_n=c.shared_bn,
            num_warps=1,
        )
        self._up = SM90FP8Projection(
            w.up.values,
            w.up.scales,
            t,
            block_n=128 if c.up_mode == "split" else c.mix_bn,
            num_warps=4 if c.up_mode == "split" else 1,
            k_group=c.up_k_group,
            pdl=c.bridge,
        )
        self._up_raw = (
            self._empty((self.l // 128, t, self.h), torch.float32)
            if c.up_mode == "split"
            else None
        )
        first = c.first
        self._first = SM90GroupedFP8Projection(
            w.first.values,
            w.first.scales,
            backend=c.first_backend,
            block_n=first.bn,
            num_warps=first.warps,
            num_stages=first.stages,
            unroll=first.unroll,
            cache=first.cache,
            hot_panels=first.hot_panels,
            input_reuse=self.topk,
            weighted=False,
            pdl=c.pdl if c.first_backend == "triton" else False,
        )
        second = c.second
        self._second = SM90GroupedFP8Projection(
            w.second.values,
            w.second.scales,
            backend="triton",
            block_n=second.bn,
            num_warps=second.warps,
            num_stages=second.stages,
            unroll=second.unroll,
            cache=second.cache,
            hot_panels=second.hot_panels,
            input_reuse=1,
            weighted=True,
            pdl=c.pdl,
        )
        down = w.down
        if self._peer_link is not None:
            n = self.l // 4
            down = FP8BlockWeight(
                down.values[self.rank * n : (self.rank + 1) * n],
                down.scales[self.rank * (n // 128) : (self.rank + 1) * (n // 128)],
            )
        self._down_weight = down
        dn, dk = down.shape
        self._down = SM90FP8Projection(
            down.values,
            down.scales,
            t,
            block_n=c.down_bn,
            num_warps=c.down_warps,
            k_group=c.down_k_group,
        )
        self._down_raw = self._empty((dk // 128, t, dn), torch.float32)
        self._epochs = (
            torch.zeros(
                (t * (self.h // 128), 32), dtype=torch.int32, device=self.device
            )
            if c.up_mode == "split"
            else None
        )
        if self._sum_link.comm_bytes < 4 * t * self.h * 2:
            raise ValueError("SUM workspace is too small")
        if self._peer_link is not None:
            words = t * (self.l // 4 + self.l // 128)
            if words * 4 > self._peer_link.comm_bytes:
                raise ValueError("peer workspace is too small")
            self._kernels.peer_initialize[((words + 127) // 128,)](
                self._peer_link.workspace, self.rank, words, num_warps=1
            )

    def _preflight(self):
        from flashinfer.gemm._sm90_grouped_fp8_native import cooperative_grid_capacity

        t, h, l = self.tokens, self.h, self.l
        gates = self._empty((t, 1), torch.float32)
        self._finish_kernel = self._kernels.finish_sum.warmup(
            self._up_raw,
            self.us,
            self.weights.up.scales,
            self.shared_output,
            gates,
            gates,
            self._sum_link.workspace,
            self._epochs,
            self.output,
            h,
            l,
            t,
            self.rank,
            128,
            self._sum_link.comm_bytes,
            grid=(t, h // 128),
            num_warps=1,
            num_stages=3,
            launch_cooperative_grid=True,
        )
        capacity = cooperative_grid_capacity(
            self._finish_kernel, self.device, threads=32
        )
        if (
            self._finish_kernel.n_spills
            or not self._finish_kernel.metadata.launch_cooperative_grid
        ):
            raise ValueError("nonspilling cooperative finish required")
        if capacity < t * (h // 128):
            raise ValueError("finish grid exceeds cooperative resident capacity")

    def _check_storage(self):
        if tuple(_signature(t) for t in self._weight_refs) != self._weight_signatures:
            raise RuntimeError(
                "weight storage or metadata changed; construct a new workspace"
            )
        current = tuple(
            t for w in vars(self.weights).values() for t in (w.values, w.scales)
        )
        if any(a is not b for a, b in zip(current, self._weight_refs, strict=False)):
            raise RuntimeError("weight tensors were replaced")

    def _check_inputs_bound(self, inputs):
        if tuple(_signature(v) for v in inputs) != self._input_signatures:
            raise RuntimeError(
                "input storage or metadata changed; construct a new workspace"
            )

    def _inputs(self, x, ids, rw, alpha, gamma):
        self._check_storage()
        expected = (
            (x, (self.tokens, self.h), torch.bfloat16),
            (ids, (self.tokens, self.topk), torch.int32),
            (rw, (self.tokens, self.topk), torch.float32),
            (alpha, (self.tokens, 1), torch.float32),
            (gamma, (self.tokens, 1), torch.float32),
        )
        for v, shape, dtype in expected:
            if (
                v.shape != shape
                or v.dtype != dtype
                or v.device != self.device
                or v.stride(-1) != 1
            ):
                raise ValueError("input tensor metadata does not match the workspace")
            if v is not x and not v.is_contiguous():
                raise ValueError("routing and gate inputs must be contiguous")
        if x.stride(0) < self.h:
            raise ValueError("overlapping input rows are unsupported")

    def _shared(self):
        c, t = self.config, self.tokens
        self._shared_first.bf16(self.xq, self.xs, out=self.shared_gate_up)
        self._kernels.situ_quant[(t, self.sf // 128)](
            self.shared_gate_up,
            self.sq,
            self.ss,
            self.sf,
            c.beta,
            c.up_beta,
            False,
            num_warps=1,
        )
        self._shared_second.bf16(self.sq, self.ss, out=self.shared_output)

    def _down_project(self):
        t = self.tokens
        n, width = self._down_weight.shape
        self._down.raw(self.xq, out=self._down_raw)
        if self._peer_link is None:
            self._down.quantize(
                self._down_raw, self.xs, output_q=self.q, output_scales=self.qs
            )
        else:
            self._kernels.peer_publish[(t, n // 128)](
                self._down_raw,
                self.xs,
                self._down_weight.scales,
                self._peer_link.workspace,
                self.rank,
                t,
                n,
                width,
                num_warps=1,
            )
            self._kernels.peer_consume[(t, self.l // 128)](
                self._peer_link.workspace,
                self.q,
                self.qs,
                self.rank,
                t,
                self.l,
                num_warps=1,
            )

    def _raw_up(self):
        self._up.raw(self.uq, out=self._up_raw)

    def _call(self, x, ids, rw, alpha, gamma):
        c, t = self.config, self.tokens
        self._kernels.quant_plan[(t * (self.h // 128),)](
            x,
            self.xq,
            self.xs,
            ids,
            self.panel_slots,
            self.panel_experts,
            t,
            self.h,
            self.topk,
            x.stride(0),
            num_warps=1,
            num_stages=1,
        )
        main = torch.cuda.current_stream(self.device)
        if self._shared_stream is not None:
            self._shared_stream.wait_stream(main)
            with torch.cuda.stream(self._shared_stream):
                self._shared()
        else:
            self._shared()
        self._down_project()
        self._first(self.q, self.qs, self.panel_slots, self.panel_experts, self.gate_up)
        self._kernels.situ_quant[(t * self.topk, self.f // 128)](
            self.gate_up,
            self.hq,
            self.hs,
            self.f,
            c.beta,
            c.up_beta,
            c.pdl,
            num_warps=1,
            launch_pdl=c.pdl,
        )
        self._second(
            self.hq,
            self.hs,
            self.panel_slots,
            self.panel_experts,
            self.partials,
            routing_weights=self._routing_weights,
        )
        self._kernels.combine_quant[(t, self.l // 128)](
            self.partials,
            self.uq,
            self.us,
            self.l,
            self.topk,
            c.pdl,
            num_warps=1,
            launch_pdl=c.pdl,
        )
        if c.up_mode == "split" and c.bridge:
            self._raw_up()
        if self._shared_stream is not None:
            main.wait_stream(self._shared_stream)
        if c.up_mode == "split":
            if not c.bridge:
                self._raw_up()
            self._kernels.finish_sum[(t, self.h // 128)](
                self._up_raw,
                self.us,
                self.weights.up.scales,
                self.shared_output,
                alpha,
                gamma,
                self._sum_link.workspace,
                self._epochs,
                self.output,
                self.h,
                self.l,
                t,
                self.rank,
                128,
                self._sum_link.comm_bytes,
                num_warps=1,
                num_stages=3,
                launch_cooperative_grid=True,
            )
            return self.output
        self._up.mix(
            self.uq, self.us, self.shared_output, alpha, gamma, out=self.local_output
        )
        return self._sum_link.reduce(self.local_output, self.output)

    def __call__(self, x, ids, rw, alpha, gamma):
        """Return the workspace-owned BF16 output for the prepared inputs."""
        with self._lock, torch.cuda.device(self.device):
            if self._closed or self._graph is not None or not self._prepared:
                raise RuntimeError(
                    "prepare an open workspace without an owned graph before calling"
                )
            self._inputs(x, ids, rw, alpha, gamma)
            self._check_inputs_bound((x, ids, rw, alpha, gamma))
            capturing = torch.cuda.is_current_stream_capturing()
            stream = torch.cuda.current_stream(self.device).cuda_stream
            if not capturing:
                if self._stream is not None and self._stream != stream:
                    raise RuntimeError(
                        "eager calls require one stream until synchronize()"
                    )
                self._stream = stream
            return self._call(x, ids, rw, alpha, gamma)

    def prepare(self, x, ids, rw, alpha, gamma):
        """Collectively compile and warm up once, binding these input storages.

        IDs must remain in ``[0, E)`` on every replay. Duplicate experts and
        signed routing weights are supported. Values may change in place;
        replacing tensor storage requires a new workspace.
        """
        with self._lock, torch.cuda.device(self.device):
            if (
                self._closed
                or self._graph is not None
                or torch.cuda.is_current_stream_capturing()
            ):
                raise RuntimeError("prepare requires an open, uncaptured workspace")
            failure = None
            try:
                self._inputs(x, ids, rw, alpha, gamma)
                if self._prepared:
                    self._check_inputs_bound((x, ids, rw, alpha, gamma))
                else:
                    buffers = [
                        value
                        for value in vars(self).values()
                        if isinstance(value, torch.Tensor)
                    ]
                    buffers.append(self._sum_link.workspace)
                    if self._peer_link is not None:
                        buffers.append(self._peer_link.workspace)
                    _check_input_workspace_overlap((x, ids, rw, alpha, gamma), buffers)
                if not bool(((ids >= 0) & (ids < self.e)).all()):
                    raise ValueError("expert IDs must be in [0, E)")
            except Exception as exc:
                failure = str(exc)
            failures = [None] * 4
            dist.all_gather_object(failures, failure, group=self.group)
            if any(failures):
                raise ValueError(failures)
            if self._prepared:
                return self
            self._input_refs = (x, ids, rw, alpha, gamma)
            self._input_signatures = tuple(_signature(v) for v in self._input_refs)
            self._routing_weights = rw.view(-1)
            failure = None
            try:
                self._first.prepare(
                    self.q, self.qs, self.panel_slots, self.panel_experts, self.gate_up
                )
                self._second.prepare(
                    self.hq,
                    self.hs,
                    self.panel_slots,
                    self.panel_experts,
                    self.partials,
                    routing_weights=self._routing_weights,
                )
                self._down.warmup("raw", self.xq, out=self._down_raw)
                if self._peer_link is None:
                    self._down.warmup(
                        "quantize",
                        self._down_raw,
                        self.xs,
                        output_q=self.q,
                        output_scales=self.qs,
                    )
                self._shared_first.warmup(
                    "bf16", self.xq, self.xs, out=self.shared_gate_up
                )
                self._shared_second.warmup(
                    "bf16", self.sq, self.ss, out=self.shared_output
                )
                if self.config.up_mode == "split":
                    self._up.warmup("raw", self.uq, out=self._up_raw)
                else:
                    self._up.warmup(
                        "mix",
                        self.uq,
                        self.us,
                        self.shared_output,
                        alpha,
                        gamma,
                        out=self.local_output,
                    )
            except Exception as exc:
                failure = str(exc)
            failures = [None] * 4
            dist.all_gather_object(failures, failure, group=self.group)
            if any(failures):
                raise RuntimeError(failures)
            for _ in range(3):
                self._call(x, ids, rw, alpha, gamma)
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
            self._prepared = True
            self._stream = None
            return self

    def capture(self, x, ids, rw, alpha, gamma, *, copies=1):
        if type(copies) is not int or copies < 1:
            raise ValueError("copies must be positive")
        with self._lock, torch.cuda.device(self.device):
            if not self._prepared or self._closed or self._graph is not None:
                raise RuntimeError("prepare an open workspace before capture")
            self._inputs(x, ids, rw, alpha, gamma)
            self._check_inputs_bound((x, ids, rw, alpha, gamma))
            torch.cuda.synchronize(self.device)
            owned = LatentMoeTP4Graph(
                self, (x, ids, rw, alpha, gamma), copies, _lease=_GRAPH_LEASE
            )
            self._graph = owned
            return owned

    def synchronize(self):
        with self._lock, torch.cuda.device(self.device):
            if self._graph is not None:
                raise RuntimeError("close the graph before rebinding its workspace")
            torch.cuda.synchronize(self.device)
            self._stream = None

    def close(self):
        with self._lock, torch.cuda.device(self.device):
            if self._graph is not None:
                raise RuntimeError("close the owned graph before closing its workspace")
            if not self._closed:
                torch.cuda.synchronize(self.device)
                dist.barrier(group=self.group)
                if self._peer_link is not None:
                    self._peer_link.close()
                self._sum_link.close()
                self._closed = True


class LatentMoeTP4Graph:
    """A captured execution lease retaining its workspace, weights and inputs."""

    def __init__(self, owner, inputs, copies, *, _lease=None):
        if _lease is not _GRAPH_LEASE:
            raise RuntimeError("create owned graphs with workspace.capture()")
        self.owner, self.inputs = owner, inputs
        self._signatures = tuple(_signature(t) for t in inputs)
        self._lock = threading.RLock()
        self._closed = False
        self._stream = None
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            for _ in range(copies):
                self.output = owner._call(*inputs)

    def replay(self):
        with self._lock, torch.cuda.device(self.owner.device):
            if self._closed or self.owner._closed:
                raise RuntimeError("graph or workspace is closed")
            self.owner._check_storage()
            if tuple(_signature(t) for t in self.inputs) != self._signatures:
                raise RuntimeError("captured input storage or metadata changed")
            stream = torch.cuda.current_stream(self.owner.device).cuda_stream
            if self._stream is not None and self._stream != stream:
                raise RuntimeError("graph replays must use one stream")
            self._stream = stream
            self._graph.replay()
            return self.output

    def close(self):
        with self.owner._lock, self._lock:
            if not self._closed:
                torch.cuda.synchronize(self.owner.device)
                self._graph.reset()
                self.owner._graph = None
                self.owner._stream = None
                self._closed = True
