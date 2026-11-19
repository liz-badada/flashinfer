# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for license information.

"""CUDA context ownership and the prepared-panel SM90 native projection."""

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

_LOCK = threading.RLock()
_MODULES = {}
_CONTEXTS = {}


class Unsupported(ValueError):
    """The native backend cannot serve this configuration."""


class _Driver:
    _library = None

    def __getattr__(self, name):
        with _LOCK:
            if self._library is None:
                library = C.CDLL("libcuda.so.1")
                prototypes = {
                    "cuGetErrorString": [C.c_int, C.POINTER(C.c_char_p)],
                    "cuCtxGetCurrent": [C.POINTER(C.c_void_p)],
                    "cuDeviceGet": [C.POINTER(C.c_int), C.c_int],
                    "cuDevicePrimaryCtxRetain": [C.POINTER(C.c_void_p), C.c_int],
                    "cuDevicePrimaryCtxRelease_v2": [C.c_int],
                    "cuModuleLoad": [C.POINTER(C.c_void_p), C.c_char_p],
                    "cuModuleGetFunction": [
                        C.POINTER(C.c_void_p),
                        C.c_void_p,
                        C.c_char_p,
                    ],
                    "cuModuleUnload": [C.c_void_p],
                    "cuFuncGetAttribute": [C.POINTER(C.c_int), C.c_int, C.c_void_p],
                    "cuLaunchKernel": [C.c_void_p]
                    + [C.c_uint] * 7
                    + [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)],
                    "cuOccupancyMaxActiveBlocksPerMultiprocessor": [
                        C.POINTER(C.c_int),
                        C.c_void_p,
                        C.c_int,
                        C.c_size_t,
                    ],
                    "cuTensorMapEncodeTiled": [
                        C.c_void_p,
                        C.c_int,
                        C.c_uint,
                        C.c_void_p,
                        C.POINTER(C.c_uint64),
                        C.POINTER(C.c_uint64),
                        C.POINTER(C.c_uint32),
                        C.POINTER(C.c_uint32),
                        C.c_int,
                        C.c_int,
                        C.c_int,
                        C.c_int,
                    ],
                }
                for symbol, arguments in prototypes.items():
                    fn = getattr(library, symbol)
                    fn.argtypes, fn.restype = arguments, C.c_int
                self._library = library
            return getattr(self._library, name)


_DRIVER = _Driver()


def _check(code):
    if code:
        message = C.c_char_p()
        _DRIVER.cuGetErrorString(code, C.byref(message))
        raise RuntimeError(f"CUDA driver {code}: {message.value!r}")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_paths():
    from ..jit import env as jit_env

    root = Path(__file__).resolve().parent
    return {
        "sm90_grouped_fp8.cuh": jit_env.FLASHINFER_INCLUDE_DIR
        / "flashinfer/gemm/sm90_grouped_fp8.cuh",
        "sm90_grouped_fp8.py": root / "sm90_grouped_fp8.py",
        "_sm90_grouped_fp8_native.py": Path(__file__).resolve(),
        "kernels/sm90_grouped_fp8.py": root / "kernels/sm90_grouped_fp8.py",
    }


def _build_identity(entry_source, sources, toolchain, flags):
    identity = dict(
        entry_source=entry_source, sources=sources, toolchain=toolchain, flags=flags
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity, key


def _toolchain(nvcc, source):
    version = subprocess.run(
        [nvcc, "--version"], capture_output=True, text=True, check=True
    ).stdout
    command = [
        nvcc,
        "-M",
        "-x",
        "cu",
        "-std=c++20",
        "-arch=sm_90a",
        "-DN_WIDTH=128",
        "-DK_WIDTH=128",
        "-DTOP_K=1",
        str(source),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    dependencies = shlex.split(result.stdout.replace("\\\n", " ").split(":", 1)[1])
    paths = sorted(
        {
            str(Path(p).resolve())
            for p in dependencies
            if Path(p).resolve() != source.resolve()
        }
    )
    return dict(
        nvcc=nvcc,
        nvcc_sha256=_sha(nvcc),
        nvcc_version=version,
        headers={p: _sha(p) for p in paths},
        nvcc_environment={
            name: os.environ.get(name, "")
            for name in (
                "NVCC_PREPEND_FLAGS",
                "NVCC_APPEND_FLAGS",
                "NVCC_CCBIN",
                "CPATH",
                "CPLUS_INCLUDE_PATH",
            )
        },
    )


def _compile(n, k, cache_policy, input_reuse):
    from ..jit import env as jit_env

    executable = os.environ.get("CUDACXX") or shutil.which("nvcc")
    if executable is None:
        raise Unsupported("nvcc is required for the native backend")
    nvcc = str(Path(executable).resolve())
    paths = _source_paths()
    source = paths["sm90_grouped_fp8.cuh"]
    sources = {name: _sha(path) for name, path in paths.items()}
    toolchain = _toolchain(nvcc, source)
    flags = [
        "-x",
        "cu",
        "-cubin",
        "-std=c++20",
        "-arch=sm_90a",
        "-O3",
        "--ptxas-options=-v",
        f"-DN_WIDTH={n}",
        f"-DK_WIDTH={k}",
        "-DPIPE_STAGES=2",
        "-DTILE_GROUPS=2",
        f"-DCACHE_POLICY={cache_policy}",
        f"-DTOP_K={input_reuse}",
    ]
    entry = dict(
        path=str(source.resolve()),
        name=source.name,
        sha256=sources["sm90_grouped_fp8.cuh"],
    )
    identity, key = _build_identity(entry, sources, toolchain, flags)
    cache = jit_env.FLASHINFER_JIT_DIR / "sm90_grouped_fp8"
    cache.mkdir(parents=True, exist_ok=True)
    cubin, manifest = cache / f"{key}.cubin", cache / f"{key}.json"
    with (cache / f"{key}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cubin.is_file() and manifest.is_file():
            try:
                record = json.loads(manifest.read_text())
                if (
                    record["key"] == key
                    and record["cubin_sha256"] == _sha(cubin)
                    and record["entry_source"] == entry
                    and record["command"][1] == entry["path"]
                ):
                    return cubin, record
            except (OSError, KeyError, ValueError, TypeError, IndexError):
                pass
        with tempfile.TemporaryDirectory(
            prefix=f".{key[:12]}-", dir=cache
        ) as temporary:
            temporary = Path(temporary)
            built = temporary / "kernel.cubin"
            command = [nvcc, str(source), "-o", str(built), *flags]
            result = subprocess.run(command, capture_output=True, text=True)
            log = result.stdout + result.stderr
            if result.returncode:
                failed = temporary / "failed.log"
                failed.write_text(log)
                os.replace(failed, cache / f"{key}.failed.log")
                raise RuntimeError(f"native NVCC failed:\n{log}")
            record = dict(
                key=key,
                **identity,
                command=command,
                cubin_sha256=_sha(built),
                compiler_log=log,
            )
            pending = temporary / "manifest.json"
            pending.write_text(json.dumps(record, indent=2) + "\n")
            os.replace(built, cubin)
            os.replace(pending, manifest)
        return cubin, record


def current_context():
    current = C.c_void_p()
    _check(_DRIVER.cuCtxGetCurrent(C.byref(current)))
    return current.value


def _retain_context(device):
    current = C.c_void_p(current_context())
    if not current.value:
        raise Unsupported("an initialized CUDA primary context is required")
    with _LOCK:
        if current.value not in _CONTEXTS:
            ordinal = C.c_int()
            _check(_DRIVER.cuDeviceGet(C.byref(ordinal), device.index))
            primary = C.c_void_p()
            _check(_DRIVER.cuDevicePrimaryCtxRetain(C.byref(primary), ordinal))
            if primary.value != current.value:
                _DRIVER.cuDevicePrimaryCtxRelease_v2(ordinal)
                raise Unsupported(
                    "the native module cache requires CUDA primary contexts"
                )
            _CONTEXTS[current.value] = primary, ordinal
    return current.value


def cooperative_grid_capacity(compiled_kernel, device, threads=32):
    """Return cooperative CTA capacity for a prepared kernel on one device."""
    device = torch.device(device)
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if device.type != "cuda":
        raise ValueError("a CUDA device is required")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with torch.cuda.device(device), _LOCK:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cooperative preflight must precede graph capture")
        _retain_context(device)
        compiled_kernel._init_handles()
        active = C.c_int()
        _check(
            _DRIVER.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                C.byref(active),
                C.c_void_p(compiled_kernel.function),
                threads,
                compiled_kernel.metadata.shared,
            )
        )
        return (
            active.value
            * torch.cuda.get_device_properties(device).multi_processor_count
        )


class _TensorMap:
    def __init__(self, address, rows, width):
        self._storage = C.create_string_buffer(255)
        self.address = (C.addressof(self._storage) + 127) & ~127
        dimensions = (C.c_uint64 * 2)(width, rows)
        stride = (C.c_uint64 * 1)(width)
        box = (C.c_uint32 * 2)(128, 128)
        element = (C.c_uint32 * 2)(1, 1)
        _check(
            _DRIVER.cuTensorMapEncodeTiled(
                self.address,
                0,
                2,
                address,
                dimensions,
                stride,
                box,
                element,
                0,
                3,
                2,
                0,
            )
        )


def _load_module(cubin, record, device):
    context = _retain_context(device)
    key = context, record["cubin_sha256"]
    with _LOCK:
        if key not in _MODULES:
            module, function = C.c_void_p(), C.c_void_p()
            _check(_DRIVER.cuModuleLoad(C.byref(module), os.fsencode(cubin)))
            try:
                _check(
                    _DRIVER.cuModuleGetFunction(
                        C.byref(function), module, b"panel_project"
                    )
                )
            except Exception:
                _DRIVER.cuModuleUnload(module)
                raise
            _MODULES[key] = module, function
        return context, _MODULES[key]


class Projection:
    """Unweighted two-warpgroup, two-stage prepared-panel projection."""

    def __init__(self, weight, scales, input_reuse, cache_policy):
        self.weight, self.scales = weight, scales
        self.device = weight.device
        self.e, self.n, self.k = weight.shape
        self.cubin, self.build_record = _compile(
            self.n, self.k, cache_policy, input_reuse
        )
        self.context, (self.module, self.function) = _load_module(
            self.cubin, self.build_record, self.device
        )
        self.tensor_map = _TensorMap(weight.data_ptr(), self.e * self.n, self.k)
        self.shared_bytes, self.threads = 2 * ((8 * 2 + 2) * 1024 + 24), 128 * 2 + 32
        self.resources = {}
        for label, attribute in (
            ("static_shared_bytes", 1),
            ("local_bytes", 3),
            ("registers", 4),
        ):
            value = C.c_int()
            _check(_DRIVER.cuFuncGetAttribute(C.byref(value), attribute, self.function))
            self.resources[label] = value.value
        if self.resources["local_bytes"]:
            raise Unsupported(
                "native projection with local-memory spills is unsupported"
            )

    def run(self, q, qs, slots, experts, out):
        pointers = [
            C.c_uint64(t.data_ptr()) for t in (q, qs, self.scales, slots, experts)
        ]
        pointers += [C.c_uint64(out.data_ptr())]
        arguments = (C.c_void_p * 7)(
            self.tensor_map.address, *[C.addressof(p) for p in pointers]
        )
        stream = torch.cuda.current_stream(self.device).cuda_stream
        context = C.c_void_p()
        _check(_DRIVER.cuCtxGetCurrent(C.byref(context)))
        if context.value != self.context:
            raise RuntimeError("the prepared CUDA context changed")
        _check(
            _DRIVER.cuLaunchKernel(
                self.function,
                self.n // 128,
                experts.numel(),
                1,
                self.threads,
                1,
                1,
                self.shared_bytes,
                stream,
                arguments,
                None,
            )
        )
