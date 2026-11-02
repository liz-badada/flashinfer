# tpmoe_sm90: TP fused MoE decode kernels for Hopper (sm_90a)

Two packages share one kernel design (persistent MXFP4 W4A8 expert kernel with the TP all-reduce fused as low-latency NVLink messages):

| directory | model / layout | status |
|---|---|---|
| `serving_tp4moe2_v26/` | DeepSeek-V4-Flash, TP4 on 4x H20 | frozen serving package (SHA-pinned), measured in SGLang 1P1D |
| `tp8moe/` | Kimi K3 Stable LatentMoE (896 experts, top-16, latent 3584), TP8 in-node, PP2 across two nodes on H200 | v0: correct on H20 and H200 (1, 8, 2 x 8 GPUs), SOL analysis, not in an engine yet; see its README |

`tp8moe/tp8moe.cu` is the tp4moe2 kernel generalized to the K3 shapes and includes the pipeline headers from `serving_tp4moe2_v26/`.

## serving_tp4moe2_v26: TP4 fused MoE decode kernel for H20

`serving_tp4moe2_v26/` is the frozen serving package of `tp4moe2`. It is a
persistent TP4 MoE kernel for DeepSeek-V4-Flash Decode on four H20 GPUs.
One launch per MoE layer replaces SGLang's routed-expert MoE (Humming MXFP4
W4A8) plus the TP all-reduce:

- router gate (fused top-k), or the given top-k for hash-routed layers
- per-token FP8 input quantization
- FC1 + SwiGLU + FC2 over the routed experts, using the Humming fused-E8M0
  W4A8 weights in place
- shared expert, fused into the kernel for M <= 32 (the `sh` build); above
  that it runs on SGLang's alternate stream (the `ns` build)
- the cross-rank reduce, as low-latency messages over NVLink peer memory
  (IPC-linked per-rank workspace)

Shapes are hidden 4096, per-rank intermediate 512, at most 256 experts, top-k
at most 8, and M (tokens per call) at most 512. Larger calls fall back to the
original SGLang path.

The directory is self-contained and is not wired into the FlashInfer build or
Python API. Its contents are frozen: `sources.json` pins every source file by
SHA-256, and `install_tp4moe2.py` refuses to load if any file differs.

## Files

| File | Role |
|---|---|
| `tp4moe2_ns.cu` | kernel + PyTorch binding, built with `-DTP2_SHARED=0` (no shared-expert code; used for M > 32) |
| `tp4moe2_sh.cu` | kernel + binding with the fused FP8 shared expert (used for M <= 32) |
| `kpipe.cuh`, `chain4.cuh`, `chain_ss.cuh` | k-loop pipeline: MXFP4 dequant + RS-wgmma chains; SS-wgmma for the shared expert |
| `shared_num.cuh` | shared-expert numerics matching the production FP8 kernels instruction for instruction |
| `tp4moe2.py` | host side: per-rank workspace (`RankState`), LL all-reduce buffers, launch wrapper |
| `tp4_ext.py` | `torch.utils.cpp_extension.load` of both builds (`sm_90a`, build dirs `build_ns/`, `build_sh/`) |
| `build.py` | prebuild both extensions with the exact `load()` calls of the hook (no GPU needed) |
| `install_tp4moe2.py` | SGLang hook: patches `DeepseekV2MoE.forward_normal_dual_stream` / `forward_normal` and `Mxfp4HummingMoEMethod.process_weights_after_loading` |
| `tp4_lazy.py` | `sitecustomize` entry: installs the hook only after `sglang.srt.models.deepseek_v2` is imported (frontend / detokenizer processes never touch CUDA) |
| `test_v26_mix.py` | TP4 test of the hook (bitwise isolated vs mixed vs CUDA graph, gate vs production, timing); needs captured activations and the model, not included |
| `sources.json` | variant name, extension names, SHA-256 of every source |

## Use in SGLang (TP4 Decode)

```bash
# prebuild once, inside the serving image (as the user the server runs as)
python3 serving_tp4moe2_v26/build.py

# in every Decode worker process, e.g. from sitecustomize.py
export FAC_FMOE_MODE=tp4moe2 TP2_PDL=1
```

```python
import sys
sys.path.insert(0, "/path/to/tpmoe_sm90/serving_tp4moe2_v26")
import tp4_lazy   # installs the hook when deepseek_v2 is imported
```

Knobs:
- `TP2_PDL=1` uses programmatic dependent launch; it is the measured setting.
- `FAC_TP4_SHADOW=N` (default 1): the first N eager calls per (layer, M) also
  run the original path and log the difference. CUDA-graph warmup runs every
  captured M eagerly, so a server start validates every graph size.
- `FAC_TP4_SH_FUSED_MAXM` (default 32) is the largest M that uses the fused
  shared expert.

The hook logs `FAC_TP4_INSTALLED`, `FAC_TP4_LINKED`, `FAC_TP4_LAYER_READY`,
`FAC_TP4_SHAPE` and `FAC_TP4_SHADOW` JSON records. Gate a deployment on them:
- 4 linked ranks
- every MoE layer ready (46 per rank with DSpark: 43 target + 3 draft)
- no fallback M
- every shadow case within rel RMS 0.05 and per-token rel 0.25

## Measured (DeepSeek-V4-Flash, 4x H20, 1P1D with DSpark Decode)

The Decode step, independent of DSpark acceptance, is compared against the
stock Humming MoE. The first figure is real P/D; the second is fake-KV pure
Decode:

| ISL/OSL, concurrency | Step speed-up |
|---|---|
| 1K/1K, C1 | +3.0% / +2.6% |
| 1K/1K, C4 | +3.5% / +1.6% |
| 1K/1K, C16 | +8.6% / +10.1% |
| 1K/1K, C64 | +8.1% / +6.6% |
| 8K/1K, C16 | +7.7% (ITL p50) / +9.3% |

All cells are beyond the A/B/A noise floor. Start-up shadow gate against
stock Humming over 4,920 (layer, M) cases: worst rel RMS 0.0235, worst
per-token rel 0.051.
