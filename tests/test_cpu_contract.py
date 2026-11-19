import ast
import ctypes as C
from dataclasses import replace
from pathlib import Path
import sys

import unittest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from generic_tp4 import BlockWeight, ExecutionConfig, LocalWeights, ProjectionConfig, Workspace
from generic_tp4.config import check_dimensions
from generic_tp4.operator import _signature, shard_experts
from generic_tp4 import native


def test_import_does_not_initialize_cuda():
    assert not torch.cuda.is_initialized()
    assert native._DRIVER._library is None


def test_invalid_launch_and_communication_configs():
    for kwargs in [dict(bridge=True), dict(bridge=True, pdl=True, up_mode='separate'),
                   dict(native_groups=3), dict(down_mode='last_cta'), dict(comm_tokens=0)]:
        with unittest.TestCase().assertRaises(ValueError):
            ExecutionConfig(**kwargs)
    with unittest.TestCase().assertRaises(ValueError):
        ProjectionConfig(cache='last', hot_panels=3)
    with unittest.TestCase().assertRaises(ValueError):
        ProjectionConfig(hot_panels=3.5)
    cfg = ExecutionConfig(pdl=True, bridge=True)
    check_dimensions(3, 512, 512, 16, 256, 128, 3, cfg)
    for kwargs in [dict(comm_tokens=2), dict(comm_hidden=256)]:
        with unittest.TestCase().assertRaises(ValueError):
            check_dimensions(3, 512, 512, 16, 256, 128, 3, replace(cfg, **kwargs))
    with unittest.TestCase().assertRaises(ValueError):
        check_dimensions(3, 128, 512, 16, 256, 128, 3, cfg)


def test_pair_preserving_shard_and_scale_blocks():
    e, f, l = 2, 512, 128
    v = (torch.arange(2 * f).view(1, -1, 1).expand(e, -1, l) % 11).float().to(torch.float8_e4m3fn).contiguous()
    s = torch.arange(e * (2 * f // 128)).reshape(e, 2 * f // 128, 1).float() + 1
    a = BlockWeight(v, s)
    b = BlockWeight(torch.zeros(e, l, f).to(torch.float8_e4m3fn), torch.ones(e, l // 128, f // 128))
    for rank in range(4):
        x, y = shard_experts(a, b, rank)
        start, stop = rank * 128, (rank + 1) * 128
        assert torch.equal(x.values.view(torch.uint8), torch.cat((v[:, start:stop], v[:, f + start:f + stop]), 1).view(torch.uint8))
        assert torch.equal(x.scales, torch.cat((s[:, rank:rank + 1], s[:, 4 + rank:5 + rank]), 1))
        assert y.shape == (e, l, 128)
    with unittest.TestCase().assertRaises(ValueError):
        BlockWeight(v, torch.ones(e, 2 * f // 128, 2))


def test_storage_lease_allows_values_but_rejects_rebinding():
    w = BlockWeight(torch.zeros(128, 128).to(torch.float8_e4m3fn), torch.ones(1, 1))
    owner = Workspace.__new__(Workspace)
    owner.weights = LocalWeights(w, w, w, w, w, w)
    owner._weight_refs = tuple(t for z in vars(owner.weights).values() for t in (z.values, z.scales))
    owner._weight_signatures = tuple(_signature(t) for t in owner._weight_refs)
    w.scales.mul_(2)
    owner._check_storage()
    w.values.data = torch.zeros_like(w.values)
    with unittest.TestCase().assertRaises(RuntimeError):
        owner._check_storage()


def test_native_entry_and_topk_are_build_identity_inputs():
    entry = dict(name='kernel.cu', path='/tmp/a/kernel.cu', sha256='a' * 64)
    args = dict(entry_source=entry, sources={'kernel.cu': 'a' * 64}, toolchain={'version': 'x'}, flags=['-DTOP_K=3'])
    _, key = native._build_identity(**args)
    for variant in [dict(entry_source={**entry, 'path': '/tmp/b/kernel.cu'}),
                    dict(entry_source={**entry, 'sha256': 'b' * 64}), dict(flags=['-DTOP_K=5'])]:
        assert native._build_identity(**{**args, **variant})[1] != key


def test_native_launch_abi_and_pdl_metadata():
    root = Path(__file__).parents[1] / 'generic_tp4'
    tree = ast.parse((root / 'native.py').read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(ast.unparse(x) == 'lib.cuLaunchKernel.argtypes' for x in n.targets))
    types = eval(compile(ast.Expression(node.value), '<ABI>', 'eval'), {'C': C})
    assert len(types) == 11
    assert types == [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    source = (root / 'kernel.cu').read_text()
    assert '/TOP_K' in source and '#error TOP_K must be supplied' in source
    assert len(source.split('void panel_project(', 1)[1].split(') {', 1)[0].split(',')) == 8
    tree = ast.parse((root / 'operator.py').read_text())
    warmup = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'k.grouped.warmup')
    assert any(x.arg == 'launch_pdl' and ast.unparse(x.value) == 'self.pdl' for x in warmup.keywords)


def test_warmed_external_capture_accepts_capture_stream():
    import contextlib
    import threading
    from types import SimpleNamespace
    from unittest.mock import patch
    import generic_tp4.operator as module

    owner = Workspace.__new__(Workspace)
    owner._lock = threading.RLock()
    owner._closed, owner._graph, owner._prepared, owner._stream = False, None, True, 11
    owner.device = torch.device('cuda:0')
    owner._inputs = lambda *args: None
    owner._call = lambda *args: 'queued'
    with patch.object(module.torch.cuda, 'device', lambda *a: contextlib.nullcontext()), \
         patch.object(module.torch.cuda, 'is_current_stream_capturing', return_value=True), \
         patch.object(module.torch.cuda, 'current_stream', return_value=SimpleNamespace(cuda_stream=22)):
        assert owner(None, None, None, None, None) == 'queued'
        owner._prepared = False
        with unittest.TestCase().assertRaises(RuntimeError):
            owner(None, None, None, None, None)
    assert not torch.cuda.is_initialized()


def test_cold_compiled_launch_has_pdl_flag_and_three_grid_axes():
    import contextlib
    from types import SimpleNamespace
    from unittest.mock import patch
    import generic_tp4.operator as module

    seen = {}
    class Compiled:
        def _init_handles(self):
            seen['initialized'] = True
        def __getitem__(self, grid):
            assert grid == (6, 4, 1)
            return lambda *args: seen.update(launched=True)
    compiled = Compiled()
    class Jit:
        def warmup(self, *args, **kwargs):
            assert kwargs['launch_pdl'] is True
            return compiled
    grouped = module._Grouped.__new__(module._Grouped)
    grouped.weight = SimpleNamespace(shape=(2, 128, 128), scales=object())
    grouped.config, grouped.pdl, grouped.descriptor = ProjectionConfig(), True, object()
    q, experts = SimpleNamespace(device=torch.device('cuda:0')), SimpleNamespace(numel=lambda: 6)
    try:
        with patch.object(module.torch.cuda, 'device', lambda *a: contextlib.nullcontext()), \
             patch.object(module.k, 'grouped', Jit()), \
             patch.object(native, '_retain_context', return_value=9):
            grouped(q, object(), object(), experts, None, object(), 2, False)
        assert seen == dict(initialized=True, launched=True)
    finally:
        module._PATCHED.pop(id(compiled), None)
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in sorted(globals().copy().items()) if name.startswith("test_"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
