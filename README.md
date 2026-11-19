# SM90 Latent MoE TP4 kernels

This is a preview for integration and synthetic validation. The current checks cover four H20 GPUs (SM90); H200 configuration and performance qualification remains pending.

Tested dependencies: Python 3.12, PyTorch 2.14.1+cu130, Triton 3.8.0 and FlashInfer 0.7.1.

The `generic_tp4` package implements BF16 input quantization, a replicated projection, routed local expert shards, a shared local FFN, a replicated output projection, BF16 gate mixing, and TP4 SUM. Inputs include external expert IDs, routing weights, and both gate tensors. Each matrix uses canonical E4M3FN values with FP32 multiplier scales per 128×128 block.

Install into an environment providing CUDA PyTorch, Triton 3.8 and the FlashInfer IPC workspace/all-reduce API. The package performs no CUDA initialization on import. It targets one process per GPU and four ranks on one host, with an initialized NCCL process group.

```python
from generic_tp4 import BlockWeight, ExecutionConfig, LocalWeights, Workspace

# Each pair contains a contiguous value tensor and its scale tensor.
weights = LocalWeights(*(BlockWeight(values, scales) for values, scales in pairs))
config = ExecutionConfig(pdl=True, bridge=True)
op = Workspace(weights, tokens=tokens, topk=topk, config=config)
op.prepare(x, ids, routing_weights, alpha, shared_gate)
y = op(x, ids, routing_weights, alpha, shared_gate)
```

`pairs` are local expert FC1/FC2, replicated down/up, and local shared FC1/FC2, in that order. Expert FC1 is `[E,2F,L]` in gate/up row order; FC2 is `[E,L,F]`. Dense pairs are `[L,H]`, `[H,L]`, `[2S,H]`, `[H,S]`. `x` is BF16 `[T,H]`, IDs are int32 `[T,K]`, routing weights are FP32 `[T,K]`, and both gates are FP32 `[T,1]`. Matrices have block128 dimensions; the fused plan requires `H/128 >= K`. All values must be finite; scales positive and IDs in range. Down/up replicas and correct local TP shards are the caller's responsibility.

`ProjectionConfig` exposes BN, warps, stages, unroll and cache hints. A positive `hot_panels` count selects prepared panel IDs for last eviction priority, with first priority on the remaining panels. `ExecutionConfig` controls activation parameters, native RS or Triton FC1, raw down BN/warps/K grouping, optional peer down, shared dense BN, split or separate up, PDL and the up bridge. There are no shape-based profiles. Communication capacities default to the requested tensor dimensions and may be supplied explicitly. Split up checks the actual cooperative resident capacity collectively before use.

Calls on one workspace are ordered and nonconcurrent. Returned output storage is reused. An eager stream change requires `synchronize()`. After warmup, external `torch.cuda.graph` capture is supported; keep the workspace, weights, input storage and graph alive together, and synchronize before capturing or switching use. Never replace weight storage or concurrently mutate weights. In-place input and weight value updates before replay are supported. Reset external graphs before calling `close()` on the workspace. Close workspaces collectively across the four ranks.

The optional `op.capture(..., copies=1)` returns an owned graph wrapper with `replay()` and `close()`. It retains its workspace and inputs and rejects changed storage. Each captured instance is replayed on one stream. It is optional; normal external capture remains available.

The peer-down workspace is private to the complete operation. Its reuse requires the final SUM completion. Split-up IPC uses dedicated triple buffers, per-CTA epochs, signed-zero normalization and cooperative launch. Do not invoke another all-reduce implementation on those buffers or bypass the final SUM. Preparation and construction contain collectives and must execute in matching rank order outside capture.

Numerics retain K128 products, FP32 ordered scale accumulation, BF16 FC1/activation/weighted-slot/combine boundaries, per128 FP8 quantization, separate BF16 RN gate products and a final BF16 add. The activation includes a beta-ten numerical specialization. Validate correctness and performance on the intended hardware for every selected configuration.

CPU tests do not initialize CUDA:

```sh
python tests/test_cpu_contract.py
```

Synthetic four-rank validation, to run explicitly on an allocated machine:

```sh
torchrun --standalone --nproc-per-node=4 tests/smoke_tp4.py --pdl --bridge
torchrun --standalone --nproc-per-node=4 tests/smoke_tp4.py --pdl --bridge --first rs --down peer
torchrun --standalone --nproc-per-node=4 tests/smoke_tp4.py --up separate --tokens 1 16 32 --down-bn 16 --down-warps 1
```

The smoke script checks nonzero synthetic TP shards, finite output, rank agreement, external and owned graph replay, value mutations and restoration. Run each required configuration explicitly on the intended hardware. Native helpers retain their Apache, MIT and BSD notices under `generic_tp4/licenses`.
