"""Host side of tp8moe.cu: extension loader, per-rank workspace (+ one-shot LL all-reduce buffers), launch wrapper.

run(): latent input z (BF16 [M, 3584], quantized in-kernel per token), routing ids / weights [M, 16] (K3 sigmoid gate, computed
outside), expert weights in the kernel layout (mx_synth) -> BF16 [M, 3584] = this rank's routed sum (ndev 1) or the 8-rank sum (ndev 8).
"""
import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
TP4_DIR = os.path.join(os.path.dirname(HERE), 'serving_tp4moe2_v26')   # kpipe.cuh / chain4.cuh: the k-loop pipeline headers shared with tp4moe2
_ext = None


def ext(flags=(), name=None):
    global _ext
    if _ext is None:
        flags = tuple(flags) + tuple(os.environ.get('TP8MOE_FLAGS', '').split())
        tag = ''.join(c if c.isalnum() else '_' for c in ''.join(flags))
        if os.environ.get('TP8MOE_SKIP_CUDA_CHECK') == '1':  # container with nvcc major != torch's cuda major
            import torch.utils.cpp_extension as _ce
            _ce._check_cuda_version = lambda *a, **k: None
        saved = os.environ.get('TORCH_CUDA_ARCH_LIST')
        os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
        build = os.path.join(HERE, 'build', name or ('tp8moe' + tag))
        os.makedirs(build, exist_ok=True)
        try:
            _ext = load(name=name or ('tp8moe' + tag), sources=[os.path.join(HERE, 'tp8moe.cu')], build_directory=build,
                        extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', f'-I{HERE}', f'-I{TP4_DIR}', '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK'] + list(flags),
                        verbose=os.environ.get('TP8MOE_BUILD_VERBOSE') == '1')
        finally:
            if saved is None:
                os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
            else:
                os.environ['TORCH_CUDA_ARCH_LIST'] = saved
    return _ext


class RankState:
    """Per-rank workspace.  Counters / flags are epoch-tagged or parity-reset by the kernel itself (no host resets)."""

    def __init__(self, device, alloc_comm=True):
        E = ext()
        self.device = device
        grid = torch.cuda.get_device_properties(device).multi_processor_count
        z = dict(device=device)
        self.act = torch.zeros(E.MAXE * E.INTER, dtype=torch.bfloat16, **z)
        self.amax = torch.zeros(E.MAXE * 4, dtype=torch.float32, **z)
        self.part = torch.zeros(2 * grid * E.NCONS * 32, dtype=torch.float32, **z)
        self.part_flags = torch.zeros(max(2 * grid, 256), dtype=torch.int32, **z)
        self.rt_cnt = torch.zeros(2 * E.NRT, dtype=torch.int32, **z)
        self.work = torch.zeros(4, dtype=torch.int32, **z)
        self.outp = torch.zeros(E.MAXM * E.TOPK_MAX * E.DIM, dtype=torch.bfloat16, **z)
        self.hq = torch.zeros(E.MAXP * E.NKT2 * E.NMAX * 128, dtype=torch.uint8, **z)
        self.hs = torch.zeros(E.MAXP * E.NMAX, dtype=torch.float32, **z)
        self.h_flags = torch.zeros(E.MAXP, dtype=torch.int32, **z)
        self.pan_cnt = torch.zeros(2 * E.MAXP, dtype=torch.int32, **z)
        self.xg = torch.zeros((E.MAXE + 8 * E.MAXP) * E.DIM, dtype=torch.uint8, **z)
        self.xg_cnt = torch.zeros(2 * E.MAXP, dtype=torch.int32, **z)
        self.fc2_next = torch.zeros(4, dtype=torch.int32, **z)
        self.xqw = torch.zeros(E.MAXM * E.DIM, dtype=torch.uint8, **z)   # fused input quantization workspace (e4m3 x, scales, flags)
        self.xsw = torch.zeros(E.MAXM, dtype=torch.float32, **z)
        self.xflag = torch.zeros(E.MAXM, dtype=torch.int32, **z)
        self.seq = 0
        self.rs_ptrs = []
        if alloc_comm:
            self.rs_buf = torch.zeros(self.RS_BYTES() // 4, dtype=torch.int32, **z)

    @staticmethod
    def RS_BYTES():   # two epoch-parity halves of the one-shot layout [NDEV src][NRT][OS_MAXM][NMSG] x 16 B
        E = ext()
        return 2 * E.NDEV * E.NRT * E.OS_MAXM * E.NMSG * 16

    def link_ptrs(self, rs_ptrs):
        """rs_ptrs[d]: device pointer of rank d's one-shot buffer (own or IPC-mapped), RS_BYTES each."""
        self.rs_ptrs = [int(p) for p in rs_ptrs]


_EMPTY = {}


def _empty(device, dtype):
    key = (str(device), dtype)
    if key not in _EMPTY:
        _EMPTY[key] = torch.empty(0, dtype=dtype, device=device)
    return _EMPTY[key]


def run(state, w, zb, ids, rw, out, my_rank, ndev, dbg=None, tl=None, factor=1.0):
    """zb: BF16 latent input [M, 3584] (quantized in-kernel, per token e4m3); ids int32 / rw fp32 [M, topk <= 16]; out BF16 [M, 3584].
    dbg / tl: TP2_STAMPS builds only (per-CTA globaltimer stamps / FC2 unit timeline, int64 tensors)."""
    state.seq += 1
    E = ext()
    e_i64 = _empty(out.device, torch.int64)
    E.run(state.xqw, state.xsw, w['w1'], w['s1'], w['f1'], w['w2'], w['s2'], w['f2'], ids, rw, float(factor),
          state.act, state.amax, state.part, state.part_flags, state.rt_cnt, state.work, state.outp,
          state.hq, state.hs, state.h_flags, state.pan_cnt, state.xg, state.xg_cnt, state.fc2_next,
          state.rs_ptrs if ndev > 1 else [], out, my_rank, ndev,
          dbg if dbg is not None else e_i64, tl if tl is not None else e_i64,
          zb, state.xflag, state.seq)
