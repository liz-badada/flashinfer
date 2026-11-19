from __future__ import annotations

import ctypes as C
from dataclasses import asdict, dataclass
import re
import socket
import threading

import torch
import torch.distributed as dist
import triton as tr
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend, CUDAOptions
from triton.tools.tensor_descriptor import TensorDescriptor

from .config import ExecutionConfig, check_dimensions
from . import kernels as k


@dataclass(frozen=True)
class BlockWeight:
    values: torch.Tensor
    scales: torch.Tensor

    def __post_init__(self):
        if self.values.ndim not in (2, 3) or self.values.dtype != torch.float8_e4m3fn:
            raise ValueError('canonical rank2/rank3 E4M3FN weights required')
        n, width = self.values.shape[-2:]
        if not n or not width or n % 128 or width % 128:
            raise ValueError('nonempty block128 matrices required')
        if self.scales.shape != (*self.values.shape[:-2], n // 128, width // 128) or self.scales.dtype != torch.float32:
            raise ValueError('FP32 multiplier scales must match block128 layout')
        if self.values.device != self.scales.device or not self.values.is_contiguous() or not self.scales.is_contiguous():
            raise ValueError('contiguous values/scales on one device required')

    @property
    def shape(self):
        return self.values.shape


@dataclass(frozen=True)
class LocalWeights:
    first: BlockWeight
    second: BlockWeight
    down: BlockWeight
    up: BlockWeight
    shared_first: BlockWeight
    shared_second: BlockWeight


def shard_experts(first, second, rank):
    if rank not in range(4) or second.values.ndim != 3:
        raise ValueError('TP4 expert matrices and rank required')
    e, latent, width = second.shape
    if first.shape != (e, 2 * width, latent) or width % 512:
        raise ValueError('paired gate/up and block128 TP shards required')
    start, stop = rank * (width // 4), (rank + 1) * (width // 4)
    a, b, mid = start // 128, stop // 128, width // 128
    return (
        BlockWeight(torch.cat((first.values[:, start:stop], first.values[:, width + start:width + stop]), 1),
                    torch.cat((first.scales[:, a:b], first.scales[:, mid + a:mid + b]), 1)),
        BlockWeight(second.values[:, :, start:stop].contiguous(), second.scales[:, :, a:b].contiguous()),
    )


def _signature(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


_PATCH_LOCK = threading.RLock()
_PATCHED = {}


class _Grouped:
    def __init__(self, weight, config, pdl):
        self.weight, self.config, self.pdl = weight, config, pdl
        self.descriptor = TensorDescriptor.from_tensor(weight.values.reshape(-1, weight.shape[-1]), [config.bn, 128])

    def __call__(self, q, qs, slots, experts, rw, output, reuse, weighted):
        from .native import _retain_context
        c = self.config
        with torch.cuda.device(q.device):
            context = _retain_context(q.device)
            args = (q, qs, self.descriptor, self.weight.scales, slots, experts, rw, output,
                    self.weight.shape[-2], self.weight.shape[-1], reuse, weighted,
                    c.bn, c.unroll, self.pdl, c.policy_key)
            grid = (experts.numel(), self.weight.shape[-2] // c.bn)
            compiled = k.grouped.warmup(*args, grid=grid, num_warps=c.warps, num_stages=c.stages, launch_pdl=self.pdl)
            with _PATCH_LOCK:
                record = _PATCHED.get(id(compiled))
                if record is None:
                    if c.policy_key:
                        self._patch(compiled)
                    compiled._init_handles()
                    _PATCHED[id(compiled)] = (compiled, context)
                elif record[1] != context:
                    raise RuntimeError('one CUDA primary context per process is required')
            compiled[(*grid, 1)](*args)

    def _patch(self, compiled):
        if compiled.module is not None:
            raise RuntimeError('cache specialization was loaded before patching')
        ptx = compiled.asm['ptx']
        if 'cache_hint' in ptx:
            raise RuntimeError('unexpected cache hint in unpatched specialization')
        pos = ptx.index('{', ptx.index('.reqntid'))
        declarations = '\n\t.reg .b64 %weight_policy;'
        if self.config.hot_panels is not None:
            declarations += '\n\t.reg .b64 %weight_hot;\n\t.reg .b32 %weight_panel;\n\t.reg .pred %weight_keep;'
        ptx = ptx[:pos + 1] + declarations + ptx[pos + 1:]
        pattern = r'(?m)^(\s*)(@%[A-Za-z0-9_]+\s+)?(cp\.async\.bulk\.tensor\.2d\.shared::(?:cta|cluster)\.global\.mbarrier::complete_tx::bytes)([^\n]*);$'

        def replace(match):
            indent, predicate, op, operands = match.groups()
            name = 'evict_last' if self.config.cache == 'last' else 'evict_first'
            code = indent + f'createpolicy.fractional.L2::{name}.b64 %weight_policy, 0f3F800000;\n'
            if self.config.hot_panels is not None:
                code += (indent + 'createpolicy.fractional.L2::evict_last.b64 %weight_hot, 0f3F800000;\n'
                         + indent + 'mov.u32 %weight_panel, %ctaid.x;\n'
                         + indent + f'setp.lt.u32 %weight_keep, %weight_panel, {self.config.hot_panels};\n'
                         + indent + 'selp.b64 %weight_policy, %weight_hot, %weight_policy, %weight_keep;\n')
            return code + indent + (predicate or '') + op + '.L2::cache_hint' + operands + ', %weight_policy;'

        ptx, count = re.subn(pattern, replace, ptx)
        if not count or count != compiled.asm['ptx'].count('cp.async.bulk.tensor.2d.'):
            raise RuntimeError('incomplete TMA cache-policy patch')
        options = CUDAOptions(launch_pdl=self.pdl, num_warps=compiled.metadata.num_warps,
                              num_stages=compiled.metadata.num_stages,
                              enable_fp_fusion=compiled.metadata.enable_fp_fusion)
        cubin = CUDABackend(GPUTarget('cuda', 90, 32)).make_cubin(ptx, {}, options, 90)
        compiled.asm['ptx'], compiled.asm['cubin'], compiled.kernel = ptx, cubin, cubin


class _IPC:
    def __init__(self, rank, tokens, hidden, group):
        from flashinfer.comm.trtllm_ar import trtllm_create_ipc_workspace_for_all_reduce_fusion
        self.rank, self.group = rank, group
        self.handles, self.workspace, self.metadata = trtllm_create_ipc_workspace_for_all_reduce_fusion(
            tp_rank=rank, tp_size=4, max_token_num=tokens, hidden_dim=hidden,
            group=group, create_metadata=True, use_symm_dev_mem=False)
        self.comm_bytes = self.metadata['lamport_comm_size']

    def reduce(self, value, output):
        from flashinfer.comm.trtllm_ar import AllReduceFusionPattern, trtllm_allreduce_fusion
        trtllm_allreduce_fusion(
            value, world_size=4, world_rank=self.rank, token_num=value.shape[0],
            hidden_dim=value.shape[1], workspace_ptrs=self.workspace,
            launch_with_pdl=False, trigger_completion_at_end=False, fp32_acc=True,
            pattern_code=AllReduceFusionPattern.kAllReduce, use_oneshot=True,
            allreduce_out=output, residual_in=None, residual_out=None, norm_out=None,
            quant_out=None, scale_out=None, rms_gamma=None, rms_eps=None,
            scale_factor=None, layout_code=None, metadata=self.metadata)
        return output

    def close(self):
        from flashinfer.comm.trtllm_ar import trtllm_destroy_ipc_workspace_for_all_reduce_fusion
        trtllm_destroy_ipc_workspace_for_all_reduce_fusion(self.handles, self.group)


class Workspace:
    def __init__(self, weights: LocalWeights, *, tokens: int, topk: int,
                 config: ExecutionConfig | None = None, group=None):
        if not dist.is_initialized() or dist.get_world_size(group) != 4:
            raise ValueError('an initialized four-rank NCCL process group is required')
        if dist.get_backend(group) != 'nccl':
            raise ValueError('NCCL process group required')
        self.group = group
        self.rank = dist.get_rank(group)
        self.weights, self.tokens, self.topk = weights, tokens, topk
        self.config = config or ExecutionConfig()
        self._lock = threading.RLock()
        self._graph = None
        self._stream = None
        self._closed = self._prepared = False
        failure = None
        try:
            self.device = weights.down.values.device
            if self.device.type != 'cuda' or torch.cuda.get_device_capability(self.device) != (9, 0):
                raise ValueError('SM90 CUDA weights required')
            if torch.cuda.current_device() != self.device.index:
                raise ValueError('select the weight device before constructing a workspace')
            device_uuid = str(torch.cuda.get_device_properties(self.device).uuid)
            self.e, self.l, self.f = weights.second.shape
            self.h = weights.down.shape[1]
            self.sf = weights.shared_second.shape[1]
            expected = ((self.e, 2 * self.f, self.l), (self.l, self.h),
                        (self.h, self.l), (2 * self.sf, self.h), (self.h, self.sf))
            actual = (weights.first.shape, weights.down.shape, weights.up.shape,
                      weights.shared_first.shape, weights.shared_second.shape)
            if actual != expected:
                raise ValueError('inconsistent local projection dimensions')
            for weight in vars(weights).values():
                if weight.values.device != self.device:
                    raise ValueError('all weights must use the same device')
            check_dimensions(tokens, self.h, self.l, self.e, self.f, self.sf, topk, self.config)
            spec = (tokens, topk, self.h, self.l, self.e, self.f, self.sf, asdict(self.config), socket.gethostname())
        except Exception as exc:
            failure, spec, device_uuid = str(exc), None, None
        statuses = [None] * 4
        dist.all_gather_object(statuses, (failure, spec, device_uuid), group=group)
        if any(x[0] for x in statuses) or any(x[1] != statuses[0][1] for x in statuses) or len({x[2] for x in statuses}) != 4:
            raise ValueError(statuses)
        self._weight_refs = tuple(t for w in vars(weights).values() for t in (w.values, w.scales))
        self._weight_signatures = tuple(_signature(t) for t in self._weight_refs)
        cap_t = self.config.comm_tokens or tokens
        cap_h = self.config.comm_hidden or max(self.h, self.l)
        self._sum_link = _IPC(self.rank, cap_t, cap_h, group)
        self._peer_link = _IPC(self.rank, cap_t, cap_h, group) if self.config.down_mode == 'peer' else None
        failure = None
        try:
            with torch.cuda.device(self.device):
                self._allocate()
                if self.config.up_mode == 'split':
                    self._preflight()
        except Exception as exc:
            failure = str(exc)
        failures = [None] * 4
        dist.all_gather_object(failures, failure, group=group)
        if any(failures):
            raise RuntimeError(failures)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=group)

    def _empty(self, shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=self.device)

    def _allocate(self):
        w, c, t, s = self.weights, self.config, self.tokens, self.tokens * self.topk
        self.panel_rows, self.inline_routing, self.max_panels = 8, True, s
        self.fc1, self.fc2 = w.first, w.second
        self.panel_slots = self._empty((s, 8), torch.int32)
        self.panel_experts = self._empty((s,), torch.int32)
        self.xq, self.xs = self._empty((t, self.h), torch.float8_e4m3fn), self._empty((t, self.h // 128), torch.float32)
        self.q, self.qs = self._empty((t, self.l), torch.float8_e4m3fn), self._empty((t, self.l // 128), torch.float32)
        self.gate_up = self._empty((s, 2 * self.f))
        self.hq, self.hs = self._empty((s, self.f), torch.float8_e4m3fn), self._empty((s, self.f // 128), torch.float32)
        self.partials = self._empty((s, self.l))
        self.shared_gate_up, self.shared_output = self._empty((t, 2 * self.sf)), self._empty((t, self.h))
        self.sq, self.ss = self._empty((t, self.sf), torch.float8_e4m3fn), self._empty((t, self.sf // 128), torch.float32)
        self.uq, self.us = self._empty((t, self.l), torch.float8_e4m3fn), self._empty((t, self.l // 128), torch.float32)
        self.output, self.local_output = self._empty((t, self.h)), self._empty((t, self.h))
        self._shared_stream = torch.cuda.Stream(device=self.device) if c.overlap_shared else None
        self._shared_d1 = TensorDescriptor.from_tensor(w.shared_first.values, [c.shared_bn, 128])
        self._shared_d2 = TensorDescriptor.from_tensor(w.shared_second.values, [c.shared_bn, 128])
        self._up_desc = TensorDescriptor.from_tensor(w.up.values, [128 if c.up_mode == 'split' else c.mix_bn, 128])
        self._up_raw = self._empty((self.l // 128, t, self.h), torch.float32) if c.up_mode == 'split' else None
        if c.first_backend == 'rs':
            from .native import Projection
            self._native_first = Projection(self, False, c.native_stages, c.native_groups, c.first.policy_key)
            if self._native_first.resources['local_bytes']:
                raise ValueError('native projection spills are unsupported')
        else:
            self._native_first = None
        self._first = _Grouped(w.first, c.first, c.pdl) if self._native_first is None else None
        self._second = _Grouped(w.second, c.second, c.pdl)
        self._down = w.down
        if self._peer_link is not None:
            n = self.l // 4
            self._down = BlockWeight(w.down.values[self.rank * n:(self.rank + 1) * n],
                                     w.down.scales[self.rank * (n // 128):(self.rank + 1) * (n // 128)])
        dn, dk = self._down.shape
        self._down_desc = TensorDescriptor.from_tensor(self._down.values, [c.down_bn, 128])
        self._down_raw = self._empty((dk // 128, t, dn), torch.float32)
        self._epochs = torch.zeros((t * (self.h // 128), 32), dtype=torch.int32, device=self.device) if c.up_mode == 'split' else None
        if self._peer_link is not None:
            words = t * (self.l // 4 + self.l // 128)
            if words * 4 > self._peer_link.comm_bytes:
                raise ValueError('peer workspace is too small')
            k.peer_initialize[(tr.cdiv(words, 128),)](self._peer_link.workspace, self.rank, words, num_warps=1)

    def _preflight(self):
        from .native import _DRIVER, _retain_context, _check
        _retain_context(self.device)
        t, h, l = self.tokens, self.h, self.l
        if self._sum_link.comm_bytes < 8 * t * h:
            raise ValueError('SUM workspace is too small')
        dummy = self._empty((t, 1), torch.float32)
        self._finish_kernel = k.finish_sum.warmup(
            self._up_raw, self.us, self.weights.up.scales, self.shared_output, dummy, dummy,
            self._sum_link.workspace, self._epochs, self.output, h, l, t,
            self.rank, 128, self._sum_link.comm_bytes, grid=(t, h // 128),
            num_warps=1, num_stages=3, launch_cooperative_grid=True)
        with _PATCH_LOCK:
            self._finish_kernel._init_handles()
        active = C.c_int()
        fn = _DRIVER.cuOccupancyMaxActiveBlocksPerMultiprocessor
        fn.argtypes = [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t]
        _check(fn(C.byref(active), C.c_void_p(self._finish_kernel.function), 32, self._finish_kernel.metadata.shared))
        props = torch.cuda.get_device_properties(self.device)
        if self._finish_kernel.n_spills or not self._finish_kernel.metadata.launch_cooperative_grid:
            raise ValueError('nonspilling cooperative finish required')
        if active.value * props.multi_processor_count < t * (h // 128):
            raise ValueError('finish grid exceeds cooperative resident capacity')

    def _check_storage(self):
        if tuple(_signature(t) for t in self._weight_refs) != self._weight_signatures:
            raise RuntimeError('weight storage or metadata changed; construct a new workspace')
        current = tuple(t for w in vars(self.weights).values() for t in (w.values, w.scales))
        if any(a is not b for a, b in zip(current, self._weight_refs)):
            raise RuntimeError('weight tensors were replaced')

    def _inputs(self, x, ids, rw, alpha, gamma):
        self._check_storage()
        expected = ((x, (self.tokens, self.h), torch.bfloat16),
                    (ids, (self.tokens, self.topk), torch.int32),
                    (rw, (self.tokens, self.topk), torch.float32),
                    (alpha, (self.tokens, 1), torch.float32),
                    (gamma, (self.tokens, 1), torch.float32))
        for v, shape, dtype in expected:
            if v.shape != shape or v.dtype != dtype or v.device != self.device or v.stride(-1) != 1:
                raise ValueError('input tensor metadata does not match the workspace')
            if v is not x and not v.is_contiguous():
                raise ValueError('routing and gate inputs must be contiguous')
        if x.stride(0) < self.h:
            raise ValueError('overlapping input rows are unsupported')

    def _shared(self):
        c, w, t = self.config, self.weights, self.tokens
        k.dense_bf16[(tr.cdiv(t, 8), 2 * self.sf // c.shared_bn)](
            self.xq, self.xs, self._shared_d1, w.shared_first.scales, self.shared_gate_up,
            t, 2 * self.sf, self.h, c.shared_bn, num_warps=1, num_stages=3)
        k.situ_quant[(t, self.sf // 128)](self.shared_gate_up, self.sq, self.ss, self.sf, c.beta, c.up_beta, False, num_warps=1)
        k.dense_bf16[(tr.cdiv(t, 8), self.h // c.shared_bn)](
            self.sq, self.ss, self._shared_d2, w.shared_second.scales, self.shared_output,
            t, self.h, self.sf, c.shared_bn, num_warps=1, num_stages=3)

    def _down_project(self):
        c, t = self.config, self.tokens
        n, width = self._down.shape
        k.raw_projection[(tr.cdiv(t, 8), n // c.down_bn, tr.cdiv(width // 128, c.down_k_group))](
            self.xq, self._down_desc, self._down_raw, t, n, width, c.down_k_group,
            c.down_bn, False, num_warps=c.down_warps, num_stages=2)
        if self._peer_link is None:
            k.finish_quant[(t, n // 128)](self._down_raw, self.xs, self._down.scales, self.q, self.qs, t, n, width, num_warps=1)
        else:
            k.peer_publish[(t, n // 128)](self._down_raw, self.xs, self._down.scales,
                                         self._peer_link.workspace, self.rank, t, n, width, num_warps=1)
            k.peer_consume[(t, self.l // 128)](self._peer_link.workspace, self.q, self.qs,
                                             self.rank, t, self.l, num_warps=1)

    def _raw_up(self):
        c = self.config
        k.raw_projection[(tr.cdiv(self.tokens, 8), self.h // 128, tr.cdiv(self.l // 128, c.up_k_group))](
            self.uq, self._up_desc, self._up_raw, self.tokens, self.h, self.l,
            c.up_k_group, 128, c.bridge, num_warps=4, num_stages=2, launch_pdl=c.bridge)

    def _call(self, x, ids, rw, alpha, gamma):
        c, t = self.config, self.tokens
        k.quant_plan[(t * (self.h // 128),)](x, self.xq, self.xs, ids, self.panel_slots,
            self.panel_experts, t, self.h, self.topk, x.stride(0), num_warps=1, num_stages=1)
        main = torch.cuda.current_stream(self.device)
        if self._shared_stream is not None:
            self._shared_stream.wait_stream(main)
            with torch.cuda.stream(self._shared_stream):
                self._shared()
        else:
            self._shared()
        self._down_project()
        if self._native_first is not None:
            self._native_first.run(self.q, self.qs, None, self.gate_up)
        else:
            self._first(self.q, self.qs, self.panel_slots, self.panel_experts, None, self.gate_up, self.topk, False)
        k.situ_quant[(t * self.topk, self.f // 128)](self.gate_up, self.hq, self.hs,
            self.f, c.beta, c.up_beta, c.pdl, num_warps=1, launch_pdl=c.pdl)
        self._second(self.hq, self.hs, self.panel_slots, self.panel_experts, rw, self.partials, 1, True)
        k.combine_quant[(t, self.l // 128)](self.partials, self.uq, self.us, self.l, self.topk, c.pdl, num_warps=1, launch_pdl=c.pdl)
        if c.up_mode == 'split' and c.bridge:
            self._raw_up()
        if self._shared_stream is not None:
            main.wait_stream(self._shared_stream)
        if c.up_mode == 'split':
            if not c.bridge:
                self._raw_up()
            k.finish_sum[(t, self.h // 128)](self._up_raw, self.us, self.weights.up.scales,
                self.shared_output, alpha, gamma, self._sum_link.workspace, self._epochs,
                self.output, self.h, self.l, t, self.rank, 128, self._sum_link.comm_bytes,
                num_warps=1, num_stages=3, launch_cooperative_grid=True)
            return self.output
        k.project_mix[(tr.cdiv(t, 8), self.h // c.mix_bn)](
            self.uq, self.us, self._up_desc, self.weights.up.scales,
            self.shared_output, alpha, gamma, self.local_output,
            self.h, self.l, c.mix_bn, t, num_warps=1, num_stages=3)
        return self._sum_link.reduce(self.local_output, self.output)

    def __call__(self, x, ids, rw, alpha, gamma):
        with self._lock, torch.cuda.device(self.device):
            if self._closed or self._graph is not None:
                raise RuntimeError('workspace is closed or leased by its owned graph')
            self._inputs(x, ids, rw, alpha, gamma)
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing and not self._prepared:
                raise RuntimeError('warm up this workspace before external graph capture')
            stream = torch.cuda.current_stream(self.device).cuda_stream
            if not capturing:
                if self._stream is not None and self._stream != stream:
                    raise RuntimeError('eager calls require one stream until synchronize()')
                self._stream = stream
            output = self._call(x, ids, rw, alpha, gamma)
            self._prepared = True
            return output

    def prepare(self, x, ids, rw, alpha, gamma):
        for _ in range(3):
            self(x, ids, rw, alpha, gamma)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self._prepared = True
        self._stream = None
        return self

    def capture(self, x, ids, rw, alpha, gamma, *, copies=1):
        if copies < 1:
            raise ValueError('copies must be positive')
        with self._lock, torch.cuda.device(self.device):
            if not self._prepared or self._closed or self._graph is not None:
                raise RuntimeError('prepare an open workspace before capture')
            self._inputs(x, ids, rw, alpha, gamma)
            torch.cuda.synchronize(self.device)
            owned = _Graph(self, (x, ids, rw, alpha, gamma), copies)
            self._graph = owned
            return owned

    def synchronize(self):
        with self._lock, torch.cuda.device(self.device):
            if self._graph is not None:
                raise RuntimeError('close the graph before rebinding its workspace')
            torch.cuda.synchronize(self.device)
            self._stream = None

    def close(self):
        with self._lock, torch.cuda.device(self.device):
            if self._graph is not None:
                raise RuntimeError('close the owned graph before closing its workspace')
            if not self._closed:
                torch.cuda.synchronize(self.device)
                dist.barrier(group=self.group)
                if self._peer_link is not None:
                    self._peer_link.close()
                self._sum_link.close()
                self._closed = True


class _Graph:
    def __init__(self, owner, inputs, copies):
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
            if self._closed:
                raise RuntimeError('graph is closed')
            self.owner._check_storage()
            if tuple(_signature(t) for t in self.inputs) != self._signatures:
                raise RuntimeError('captured input storage or metadata changed')
            stream = torch.cuda.current_stream(self.owner.device).cuda_stream
            if self._stream is not None and self._stream != stream:
                raise RuntimeError('graph replays must use one stream')
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
