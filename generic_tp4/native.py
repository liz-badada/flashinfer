"""Prepared-panel F16 WGMMA RS projections, with process-lifetime modules."""
from __future__ import annotations
import ctypes as C
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import threading
import torch
_ROOT = Path(__file__).resolve().parent

class _Driver:
    _library = None

    def __getattr__(self, name):
        with _LOCK:
            if self._library is None:
                lib = C.CDLL('libcuda.so.1')
                lib.cuLaunchKernel.argtypes = [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
                lib.cuLaunchKernel.restype = C.c_int
                lib.cuStreamIsCapturing.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
                lib.cuCtxGetCurrent.argtypes = [C.POINTER(C.c_void_p)]
                self._library = lib
            return getattr(self._library, name)
_DRIVER = _Driver()
_LOCK = threading.RLock()
_MODULES = {}
_CONTEXTS = {}

class Unsupported(ValueError):
    """Caller should keep the existing projection fallback for this input."""

def _check(code):
    if code:
        message = C.c_char_p()
        _DRIVER.cuGetErrorString(C.c_int(code), C.byref(message))
        raise RuntimeError(f'CUDA driver {code}: {message.value!r}')

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _source_manifest():
    suffixes = {'.cu', '.cuh', '.h', '.hpp', '.py'}
    return {str(p.relative_to(_ROOT)): _sha(p) for p in sorted(_ROOT.rglob('*')) if p.is_file() and p.suffix in suffixes and ('__pycache__' not in p.parts)}

def _dependency_paths(nvcc):
    version = subprocess.run([nvcc, '--version'], capture_output=True, text=True, check=True).stdout
    result = subprocess.run([nvcc, '-M', '-std=c++20', '-arch=sm_90a', '-DN_WIDTH=128', '-DK_WIDTH=128', '-DTOP_K=1', str(_ROOT / 'kernel.cu')], capture_output=True, text=True, check=True)
    dependencies = shlex.split(result.stdout.replace('\\\n', ' ').split(':', 1)[1])
    paths = tuple(sorted({str(Path(p).resolve()) for p in dependencies if Path(p).resolve() != (_ROOT / 'kernel.cu').resolve()}))
    return (version, paths)

def _toolchain(nvcc):
    compiler_hash = _sha(nvcc)
    version, paths = _dependency_paths(nvcc)
    return {'nvcc': nvcc, 'nvcc_sha256': compiler_hash, 'nvcc_version': version, 'headers': {p: _sha(p) for p in paths}, 'nvcc_environment': {name: os.environ.get(name, '') for name in ('NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS', 'NVCC_CCBIN', 'CPATH', 'CPLUS_INCLUDE_PATH')}}

def _build_identity(entry_source, sources, toolchain, flags):
    identity = dict(entry_source=entry_source, sources=sources, toolchain=toolchain, flags=flags)
    return identity, hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _compile(n, k, stages, weighted, groups=1, cache_policy=0, topk=1):
    nvcc = os.environ.get('TP4_NATIVE_NVCC') or shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    nvcc = str(Path(nvcc).resolve())
    sources = _source_manifest()
    toolchain = _toolchain(nvcc)
    flags = ['-cubin', '-std=c++20', '-arch=sm_90a', '-O3', '--ptxas-options=-v', f'-DN_WIDTH={n}', f'-DK_WIDTH={k}', f'-DPIPE_STAGES={stages}', f'-DTILE_GROUPS={groups}', f'-DWEIGHTED={int(weighted)}', f'-DCACHE_POLICY={cache_policy}', f'-DTOP_K={topk}']
    entry_source = {'path': str((_ROOT / 'kernel.cu').resolve()), 'name': 'kernel.cu', 'sha256': sources['kernel.cu']}
    flags.append('-DPROGRAMMATIC_DEPENDENT_LAUNCH=0')
    identity, key = _build_identity(entry_source, sources, toolchain, flags)
    cache = Path(os.environ.get('TP4_NATIVE_CACHE_DIR', Path('/tmp') / 'generic-tp4-native'))
    cache.mkdir(parents=True, exist_ok=True)
    cubin, manifest = (cache / f'{key}.cubin', cache / f'{key}.json')
    with (cache / f'{key}.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cubin.is_file() and manifest.is_file():
            try:
                record = json.loads(manifest.read_text())
                if record['key'] == key and record['cubin_sha256'] == _sha(cubin) and (record.get('entry_source') == entry_source) and (record['command'][1] == entry_source['path']):
                    return (cubin, record, True)
            except (OSError, KeyError, ValueError, TypeError, IndexError):
                pass
        with tempfile.TemporaryDirectory(prefix=f'.{key[:12]}-', dir=cache) as temporary:
            temporary = Path(temporary)
            built = temporary / 'kernel.cubin'
            command = [nvcc, str(_ROOT / 'kernel.cu'), '-o', str(built), *flags]
            result = subprocess.run(command, capture_output=True, text=True)
            log = result.stdout + result.stderr
            if result.returncode:
                failed = temporary / 'failed.log'
                failed.write_text(log)
                os.replace(failed, cache / f'{key}.failed.log')
                raise RuntimeError(f'native NVCC failed:\n{log}')
            record = {'key': key, **identity, 'command': command, 'cubin_sha256': _sha(built), 'compiler_log': log}
            pending = temporary / 'manifest.json'
            pending.write_text(json.dumps(record, indent=2) + '\n')
            os.replace(built, cubin)
            os.replace(pending, manifest)
        return (cubin, record, False)

class _TensorMap:

    def __init__(self, address, rows, width, block_rows):
        if address % 16 or width % 16:
            raise Unsupported('TMA requires a 16-byte aligned base and row stride')
        self._storage = C.create_string_buffer(255)
        self.address = C.addressof(self._storage) + 127 & ~127
        dimensions = (C.c_uint64 * 2)(width, rows)
        stride = (C.c_uint64 * 1)(width)
        box = (C.c_uint32 * 2)(128, block_rows)
        element = (C.c_uint32 * 2)(1, 1)
        _check(_DRIVER.cuTensorMapEncodeTiled(C.c_void_p(self.address), C.c_int(0), C.c_uint32(2), C.c_void_p(address), dimensions, stride, box, element, C.c_int(0), C.c_int(3), C.c_int(2), C.c_int(0)))

def _retain_context(device):
    current = C.c_void_p()
    _check(_DRIVER.cuCtxGetCurrent(C.byref(current)))
    if not current.value:
        raise Unsupported('native projection needs an initialized CUDA context')
    with _LOCK:
        if current.value not in _CONTEXTS:
            ordinal = C.c_int()
            _check(_DRIVER.cuDeviceGet(C.byref(ordinal), C.c_int(device.index)))
            primary = C.c_void_p()
            _check(_DRIVER.cuDevicePrimaryCtxRetain(C.byref(primary), ordinal))
            if primary.value != current.value:
                _DRIVER.cuDevicePrimaryCtxRelease_v2(ordinal)
                raise Unsupported('native module cache supports Torch primary contexts')
            _CONTEXTS[current.value] = (primary, ordinal)
    return current.value

def _load_module(cubin, record, shared_bytes, device):
    context = _retain_context(device)
    key = (context, record['cubin_sha256'])
    with _LOCK:
        if key not in _MODULES:
            module = C.c_void_p()
            _check(_DRIVER.cuModuleLoad(C.byref(module), C.c_char_p(os.fsencode(cubin))))
            functions = {}
            try:
                for dtype, name in ((torch.int32, b'panel_project'),):
                    function = C.c_void_p()
                    _check(_DRIVER.cuModuleGetFunction(C.byref(function), module, C.c_char_p(name)))
                    if shared_bytes > 48 * 1024:
                        _check(_DRIVER.cuFuncSetAttribute(function, C.c_int(8), C.c_int(shared_bytes)))
                    functions[dtype] = function
            except Exception:
                _DRIVER.cuModuleUnload(module)
                raise
            _MODULES[key] = (module, functions)
        return (context, _MODULES[key])

def _weight_reason(weight):
    if not hasattr(weight, 'values') or not hasattr(weight, 'scales'):
        return 'canonical values/scales are required'
    values, scales = (weight.values, weight.scales)
    if values.ndim != 3 or values.dtype != torch.float8_e4m3fn:
        return 'canonical contiguous rank3 E4M3FN weights are required'
    e, n, k = values.shape
    if not e or n % 128 or k % 128 or (not n) or (not k) or (e * n > 2 ** 31 - 1):
        return 'unsupported block128 weight dimensions or TMA row coordinate range'
    if values.device.type != 'cuda' or scales.device != values.device:
        return 'weights/scales must share one CUDA device'
    if scales.shape != (e, n // 128, k // 128) or scales.dtype != torch.float32:
        return 'raw multiplier scales must be FP32 [E,N/128,K/128]'
    if not values.is_contiguous() or not scales.is_contiguous():
        return 'weights/scales must be contiguous'
    if values.data_ptr() % 16:
        return 'TMA weight base must be 16-byte aligned'
    if torch.cuda.get_device_capability(values.device) != (9, 0):
        return 'native projection targets SM90a'
    return None

class Projection:
    supported_id_dtypes = (torch.int32, torch.int64)

    def __init__(self, runner, weighted=False, stages=2, groups=2, cache_policy=0):
        self.runner, self.weighted = (runner, bool(weighted))
        self.weight = runner.fc2 if weighted else runner.fc1
        reason = _weight_reason(self.weight)
        if reason:
            raise Unsupported(reason)
        if not runner.inline_routing or runner.panel_rows != 8:
            raise Unsupported('prepared eight-slot panels required')
        if runner.tokens < 1 or runner.topk < 1:
            raise Unsupported('positive tokens and topk required')
        self.stages, self.groups = (stages, groups)
        self.threads = 128 * groups + 32
        self.shared_bytes = stages * ((8 * groups + 2) * 1024 + 24)
        self.e, self.n, self.k = self.weight.shape
        assert self.n % (64 * groups) == 0 and groups in (1, 2, 4)
        assert cache_policy in (0, 1, 2)
        self.cache_policy = cache_policy
        cubin, record, _ = _compile(self.n, self.k, stages, self.weighted, groups=groups, cache_policy=cache_policy, topk=runner.topk)
        self.context, (self.module, self.functions) = _load_module(cubin, record, self.shared_bytes, self.weight.values.device)
        self.wmap = _TensorMap(self.weight.values.data_ptr(), self.e * self.n, self.k, 64 * groups)
        self.cubin, self.build_record = (cubin, record)
        self.address = self.weight.values.data_ptr()
        self.resources = {}
        for label, attribute in [('static_shared_bytes', 1), ('local_bytes', 3), ('registers', 4)]:
            value = C.c_int()
            _check(_DRIVER.cuFuncGetAttribute(C.byref(value), C.c_int(attribute), self.functions[torch.int32]))
            self.resources[label] = value.value

    def run(self, q, qs, rw, out):
        r, w = (self.runner, self.weight)
        assert w.values.data_ptr() == self.address
        assert q.dtype == torch.float8_e4m3fn and qs.dtype == torch.float32
        assert q.is_contiguous() and qs.is_contiguous() and out.is_contiguous()
        rows = r.tokens * r.topk if self.weighted else r.tokens
        if q.shape != (rows, self.k) or qs.shape != (rows, self.k // 128) or out.shape != (r.tokens * r.topk, self.n):
            raise ValueError('projection input/output shape mismatch')
        if out.dtype != torch.bfloat16 or any((t.device != w.values.device for t in (q, qs, out))):
            raise ValueError('BF16 output on the weight device required')
        pointers = [C.c_uint64(t.data_ptr()) for t in (q, qs, w.scales, r.panel_slots, r.panel_experts)]
        pointers += [C.c_uint64(rw.data_ptr() if self.weighted else 0), C.c_uint64(out.data_ptr())]
        arguments = (C.c_void_p * 8)(self.wmap.address, *[C.addressof(p) for p in pointers])
        stream = torch.cuda.current_stream(w.values.device).cuda_stream
        capture_status = C.c_int()
        _check(_DRIVER.cuStreamIsCapturing(C.c_void_p(stream), C.byref(capture_status)))
        if capture_status.value == 2:
            raise RuntimeError('native projection capture invalidated')
        context = C.c_void_p()
        _check(_DRIVER.cuCtxGetCurrent(C.byref(context)))
        if context.value != self.context:
            raise Unsupported('native projection current CUDA context changed')
        _check(_DRIVER.cuLaunchKernel(self.functions[torch.int32], C.c_uint(self.n // (64 * self.groups)), C.c_uint(r.max_panels), C.c_uint(1), C.c_uint(self.threads), C.c_uint(1), C.c_uint(1), C.c_uint(self.shared_bytes), C.c_void_p(stream), arguments, None))
