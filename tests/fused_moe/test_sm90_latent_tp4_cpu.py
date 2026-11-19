"""CPU validation of configuration, sharding and prepared-storage ownership."""

import contextlib
from dataclasses import replace
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

_cuda_initialized_before_import = torch.cuda.is_initialized()
with patch.object(
    torch.cuda, "_lazy_init", side_effect=RuntimeError("CPU import guard")
):
    from flashinfer.fused_moe import sm90_latent_tp4 as module
    from flashinfer.fused_moe._sm90_latent_tp4_config import _check_dimensions


class TestLatentMoeTP4CPU(unittest.TestCase):
    def test_import_does_not_initialize_cuda(self):
        self.assertEqual(torch.cuda.is_initialized(), _cuda_initialized_before_import)
        self.assertEqual(
            Path(module.__file__).resolve(),
            Path(__file__).resolve().parents[2]
            / "flashinfer/fused_moe/sm90_latent_tp4.py",
        )

    def test_invalid_configuration(self):
        for kwargs in (
            dict(bridge=True),
            dict(bridge=True, pdl=True, up_mode="separate"),
            dict(first_backend="unsupported"),
            dict(comm_tokens=0),
        ):
            with self.assertRaises(ValueError):
                module.LatentMoeTP4Config(**kwargs)
        with self.assertRaises(ValueError):
            module.LatentMoeProjectionConfig(cache="last", hot_panels=1)
        cfg = module.LatentMoeTP4Config(pdl=True, bridge=True)
        _check_dimensions(3, 512, 512, 8, 128, 128, 2, cfg)
        for kwargs in (dict(comm_tokens=2), dict(comm_hidden=256)):
            with self.assertRaises(ValueError):
                _check_dimensions(3, 512, 512, 8, 128, 128, 2, replace(cfg, **kwargs))

    def test_sharding_preserves_gate_up_and_scale_blocks(self):
        e, f, width = 2, 512, 128
        values = torch.arange(2 * f).view(1, -1, 1).expand(e, -1, width) % 11 - 5
        values = values.float().to(torch.float8_e4m3fn).contiguous()
        scales = torch.arange(e * (2 * f // 128)).reshape(e, -1, 1).float() + 1
        first = module.FP8BlockWeight(values, scales)
        second = module.FP8BlockWeight(
            torch.ones(e, width, f).to(torch.float8_e4m3fn),
            torch.ones(e, width // 128, f // 128),
        )
        for rank in range(4):
            a, b = module.shard_fp8_experts_tp4(first, second, rank)
            lo, hi = rank * 128, (rank + 1) * 128
            expected = torch.cat((values[:, lo:hi], values[:, f + lo : f + hi]), 1)
            self.assertTrue(
                torch.equal(a.values.view(torch.uint8), expected.view(torch.uint8))
            )
            self.assertTrue(
                torch.equal(
                    a.scales,
                    torch.cat(
                        (scales[:, rank : rank + 1], scales[:, 4 + rank : 5 + rank]), 1
                    ),
                )
            )
            self.assertEqual(b.shape, (e, width, 128))

    def test_live_weights_and_storage_lease(self):
        weight = module.FP8BlockWeight(
            torch.ones(128, 128).to(torch.float8_e4m3fn), torch.ones(1, 1)
        )
        owner = module.LatentMoeTP4.__new__(module.LatentMoeTP4)
        owner.weights = module.LatentMoeTP4Weights(*([weight] * 6))
        owner._weight_refs = tuple(
            v for w in vars(owner.weights).values() for v in (w.values, w.scales)
        )
        owner._weight_signatures = tuple(
            module._signature(v) for v in owner._weight_refs
        )
        weight.scales.mul_(2)
        owner._check_storage()
        weight.values.data = torch.ones_like(weight.values)
        with self.assertRaises(RuntimeError):
            owner._check_storage()

    def test_input_workspace_aliases_use_byte_spans(self):
        storage = torch.empty(32, dtype=torch.bfloat16)
        x = storage.as_strided((2, 4), (8, 1))
        with self.assertRaisesRegex(ValueError, "overlap"):
            module._check_input_workspace_overlap((x,), (storage[9:10],))
        module._check_input_workspace_overlap((x,), (storage[16:20],))
        output = storage.view(2, 16)
        gate = output.view(torch.float32).flatten()[:2].view(2, 1)
        with self.assertRaisesRegex(ValueError, "overlap"):
            module._check_input_workspace_overlap((gate,), (output,))

    def test_owned_graph_requires_factory_lease(self):
        with patch.object(
            module.torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA touched")
        ):
            for kwargs in ({}, {"_lease": object()}):
                with self.assertRaisesRegex(RuntimeError, "workspace.capture"):
                    module.LatentMoeTP4Graph(None, None, 1, **kwargs)
        self.assertEqual(torch.cuda.is_initialized(), _cuda_initialized_before_import)

    def test_prepared_capture_uses_capture_stream(self):
        owner = module.LatentMoeTP4.__new__(module.LatentMoeTP4)
        owner._lock = threading.RLock()
        owner._closed, owner._graph, owner._prepared, owner._stream = (
            False,
            None,
            True,
            11,
        )
        owner.device = torch.device("cuda:0")
        owner._inputs = owner._check_inputs_bound = lambda *args: None
        owner._call = lambda *args: "queued"
        with (
            patch.object(
                module.torch.cuda, "device", lambda *a: contextlib.nullcontext()
            ),
            patch.object(
                module.torch.cuda, "is_current_stream_capturing", return_value=True
            ),
            patch.object(
                module.torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=22),
            ),
        ):
            self.assertEqual(owner(None, None, None, None, None), "queued")
            owner._prepared = False
            with self.assertRaises(RuntimeError):
                owner(None, None, None, None, None)
        self.assertEqual(torch.cuda.is_initialized(), _cuda_initialized_before_import)


if __name__ == "__main__":
    unittest.main()
