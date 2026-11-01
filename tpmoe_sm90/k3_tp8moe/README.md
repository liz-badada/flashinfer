# k3_tp8moe: Kimi K3 Stable-LatentMoE TP megamoe for Hopper (sm_90a), TP8 intra-node

Port of the DeepSeek-V4-Flash `tp4moe2` kernel (`../serving_tp4moe2_v26`) to the Kimi K3 MoE structure, for H200 two-node
serving. Status: **v0, correctness-verified on 8x H20-3e and on H200 (1 GPU, 8 GPUs, and 2 nodes x 8 GPUs TP8 x PP2 on computelab)**; routed experts + 8-way one-shot LL reduce run in one persistent kernel,
the rest of the block is torch + NCCL (`k3_layer.py`). Not wired into FlashInfer; not yet hooked into an engine.

## Why TP8 (not TP16) for the two-node H200 layout

Official recipes for K3 on Hopper: SGLang's H200 profile is 2x8 with **TP16 + EP16** (cross-node TP over the NIC, `symm-mem`,
marlin W4A16 experts; a community deployment report measured 16.8 tok/s at concurrency 1), vLLM/Dynamo's H200 recipe is **TP8 x PP4**
on 32 GPUs, and SGLang's B200 2x8 profile is **PP2 x TP8**. This kernel's low-latency all-reduce writes directly into NVLink peer memory,
so its TP group must be one node: the two-node H200 layout for it is **TP8 inside the node, PP2 across the nodes** (the same shape as the
Dynamo Hopper and SGLang B200 recipes). Under a TP16 deployment the kernel can still do the intra-node 8-way stage of a hierarchical
reduce, with the 2-way cross-node step outside it. Memory: the 1.56 TB MXFP4 checkpoint is 98 GB per GPU at 16 GPUs with the weights
consumed in place (marlin's W4A16 repack is +33%, which is what makes vLLM run out of memory on H200). H20-3e also has 141 GB.

## K3 MoE structure implemented (verified from the model code, not from descriptions)

`modeling_kimi_linear.py` `KimiSparseMoeBlock` (HF module name `block_sparse_moe`), config.json, safetensors headers:

| step | shape (BF16 unless noted) | where it runs in v0 |
|---|---|---|
| router: sigmoid(x W_r^T), + bias for selection only, top-16, renormalize | `gate.weight` [896, 7168], bias F32 [896] | torch (`k3_ref.gate_ref`), replicated |
| latent down projection z = W_down x (shared by all experts, before the experts) | [3584, 7168] | torch, rows sharded 448 / rank + NCCL all-gather |
| routed experts in latent space: w1/w3 3584 -> 3072, w2 3072 -> 3584, **SiTU-GLU** (4 tanh(g/4) sigmoid(g) * 25 tanh(u/25)) | MXFP4 group 32, `weight_packed` U8 [3072, 1792] + E8M0 `weight_scale` U8 [3072, 112]; w2 [3584, 1536] + [3584, 96] | **kernel**: this rank's 384 intermediate rows of every expert, per-token FP8 activations |
| weighted sum over the 16 slots and over the 8 ranks | | **kernel** (T1 combine + one-shot LL reduce) |
| RMSNorm(3584) (`routed_expert_norm`, present in all 92 MoE layers) | [3584] | torch, replicated |
| latent up projection W_up | [7168, 3584] | torch, K-sharded (448 latent rows / rank) -> fp32 partial |
| shared expert: ONE `KimiMLP` with intermediate 6144 (= 3072 x 2), input = the pre-routing x, SiTU | [6144, 7168] x2, [7168, 6144] | torch, intermediate sharded 768 / rank -> fp32 partial |
| y = W_up RMSNorm(u) + shared(x) | | NCCL fp32 all-reduce of the partials |

Layer 0 is dense (intermediate 33792); layers 1..92 are MoE. Per-layer `mlp_res_proj` / `self_attention_res_proj` [1, 7168] residual
scalings exist in the checkpoint (attn_res_block_size 12) and are outside this block.

## Kernel changes vs tp4moe2 (`k3tp8moe.cu`)

- Shapes: `DIM` 4096 -> 3584 (latent), `INTER` 512 -> 384 (3072 / 8), `NEXP` 256 -> 896, `TOPK_MAX` 8 -> 16, `NDEV` 4 -> 8, `MAXM` 128,
  `ITEMS` = INTER/128 = 3 FC1 items per panel, 28 FC1 k-tiles, 3 FC2 k-tiles, 14 row tiles. Stream-K cost / index math uses
  `ITEMS` / `NKT1` instead of 4 / 32; panel info packs e in 10 bits; FC2 unit words pack pid at bit 20; `km` slot masks are 16-bit.
- Tables: the block-wide builder scans 896 experts with 512 threads in two rounds; `MAXP` 512 panels -> host requires M * topk <= 512
  (M <= 32). Small tables (M <= 8) use the bitmap builder. Dense mode and the fused sqrt-softplus gate are disabled (DeepSeek-specific).
- Epilogue: SiTU-GLU instead of the clamped SwiGLU (`situ_glu`, fp32, result BF16).
- Tail: 8-way one-shot only (every rank pushes to all 8, each reduces in rank order, fp32, one BF16 rounding), parity double-buffered;
  RS + AG mode is not adapted (14 row tiles do not split evenly over 8 ranks).
- Fused per-token quantization bounded to DIM / 8 chunks (the 4096 assumption overran rows: a race at M >= 4).
- Shared expert is not fused (`TP2_SHARED 0`): K3's is full-width (7168), it belongs to the second reduce stage.
- ptxas: 96 registers (640-thread launch bound), 168 B spill stores / 80 B loads (the 16-slot T1 combine; tp4moe2 had none).

**Shared with `serving_tp4moe2_v26/`, and why the kernel is a separate file.** The k-loop pipeline headers (`kpipe.cuh`, `chain4.cuh`,
`chain_ss.cuh`, `shared_num.cuh`) are the tp4moe2 files, included from `../serving_tp4moe2_v26` (`k3_host.py` adds the `-I`; the cubin
built that way is byte-identical to the one built from copies). `k3tp8moe.cu` is `tp4moe2_ns.cu` generalized: 423 changed lines in
125 hunks, almost all of them the hard-coded 4 / 32 / 16 / 256 / 8 / 4 becoming `ITEMS` / `NKT1` / `NRT` / `NEXP` / `TOPK_MAX` /
`NDEV`, wider bit fields (expert 8 -> 10 bits, pid at bit 20, 16-bit slot masks) and the K3-only pieces listed above (multi-round
896-expert table scan, SiTU-GLU, one-shot-only tail, no dense / fused-gate / fused-shared paths). It is not folded into the tp4moe2
sources because that package is frozen (`sources.json` pins every file by SHA-256 and the serving hook refuses to load otherwise) and
the wider bit fields change the DeepSeek kernel's table layout and register allocation, i.e. the validated binary. The intended end
state is one kernel body parameterized by a model config (DIM / INTER / NEXP / TOPK / NDEV / activation) with tp4moe2 re-frozen as v27
after an H20 re-validation; until then the two `.cu` files sit side by side on this branch and a change to the shared logic has to be
applied to both.

Weights: `k3_synth.py` packs the kernel layout (generalized Humming fused-E8M0 blocks) and converts the **K3 checkpoint format**
(`mxfp4-pack-quantized`: E2M1 nibbles, low nibble = even k, E8M0 scales) exactly (per-expert factor f = 2^(e_max - 12), groups more than
2^12 below e_max are re-rounded; none on synthetic weights -- check on the real checkpoint).

## Results (8x H20-3e, 78 SMs, 4.8 TB/s; synthetic weights)

Correctness (`test_k3.py`, `test_k3_layer.py`; rel RMS vs the torch reference of the same numerics):

| test | result |
|---|---|
| single GPU M = 1, 2, 4, 8, 16, 32 (64 experts; 896 experts via the checkpoint-format converter at M = 8, 32) | 0.0019-0.0047, two calls bitwise identical |
| TP8 M = 1..32, 896 experts | 0.0043-0.0049, all 8 ranks identical, back-to-back graph of 4 calls x 10 replays OK |
| whole layer (`K3MoELayerTP8`) vs `k3_block_ref` M = 1, 8, 32 | 0.0007-0.0036 |

Kernel time per layer (TP8, 896 experts, graph of 8 back-to-back calls, worst-rank min, us):

| M | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| kernel | 41.7 | 52.1 | 70.4 | 100.5 | 127.2 | 157.3 |
| unique experts | 16 | 28 | 51 | 88 | 121 | 153 |
| SOL at 4.3 TB/s (2.2 MB per expert per rank) | 8 | 14 | 26 | 45 | 62 | 78 |

M = 1 is latency-bound (16 narrow panels of 3 x 28 k-tiles, 8-way one-shot, 16-slot combine); 20 % of SOL at M = 1, ~50 % at M >= 8
(the DeepSeek kernel is at 23-54 % on the same machine). PDL makes no difference in a back-to-back graph.

Whole layer (`K3MoELayerTP8`: kernel + torch router / W_down / RMSNorm / W_up / shared expert + NCCL all-gather and all-reduce),
captured in one CUDA graph of 8 calls, worst-rank min, us:

| M | 1 | 8 | 32 |
|---|---|---|---|
| layer (graph) | 198 | 314 | 453 |
| of which the kernel | 42 | 100 | 157 |

The torch + NCCL part (about 150-250 us: ~20 small kernels, two NCCL collectives, bf16 GEMMs over 7168-wide weights) is the next thing
to fuse; it is already the majority of the layer time.


## Results on H200 (computelab, 1 and 8 GPUs; synthetic weights)

Jobs 5367215 / 5369265 (1 GPU: ipp2-0042, smc-522ga-0030), 5365960 / 5369818 (8 GPUs: viking-prod-322, viking-cr-194),
`nvcr.io/nvidia/pytorch:25.09-py3` (torch 2.9.0+cu130, nvcc 13.0), H200 141 GB, HBM 3201 MHz.

Correctness is identical to H20 (same sm_90a binary): single GPU M = 1..32 x {64, 896} experts rel RMS 0.0019-0.0047; TP8 M = 1..32,
896 experts, 0.0043-0.0049 with all 8 ranks bitwise identical and graph replay of 4 back-to-back calls x 10 OK; whole layer vs the
block reference 0.0040-0.0046; checkpoint-format converter exact.

Kernel time, 896 experts, graph of 32 back-to-back calls, us per call (TP8: worst-rank min; the 8 rank minima agree within 0.3 us):

| M | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| H200 single GPU, ndev 1 (no reduce) | 41.0 | 49.9 | 62.1 | 82.2 | 99.3 | 120.2 |
| **H200 TP8, incl. 8-way one-shot LL reduce** | **39.5** | **47.4** | **60.4** | **79.2** | **98.7** | **126.0** |
| H20 TP8 (78 SMs) | 41.7 | 52.1 | 70.4 | 100.5 | 127.2 | 157.3 |
| unique experts | 16 | 28 | 51 | 88 | 121 | 153 |
| SOL at 4.0 TB/s (2.2 MB per unique expert per rank) | 9 | 15 | 28 | 48 | 67 | 84 |

The TP8 run is not slower than the single-GPU run: the LL reduce is hidden behind the panel tail, and the single-GPU numbers came from
a node with other users' jobs. M = 1 is the same ~40 us as on H20: the small-M path is latency-bound (16 narrow panels of 3 x 28
k-tiles, 16-slot combine, one-shot barrier), not bandwidth-bound, so the 132-SM / 4.8 TB/s H200 only helps from M >= 4 (-14 % at
M = 4, -20 % at M = 32 vs H20). Graph medians sit 8-11 us above the minima with 32 calls per graph (30-70 us above with 4 calls):
that is the launch skew of 8 uncoupled processes on a shared node, not kernel variance (the layer graphs below, which contain NCCL
collectives, have min == median).

Whole layer (`K3MoELayerTP8`), one CUDA graph of 32 layer calls, worst rank, us per layer:

| M | 1 | 8 | 32 |
|---|---|---|---|
| H200 layer graph (min / median) | 213.8 / 214.3 | 279.8 / 280.2 | 371.3 / 372.0 |
| H200 layer graph, sglang image, NCCL 2.30.7 NVLS off (job 5509039, viking-prod-214) | 227.6 | 319.6 | 405.6 |
| of which the kernel | 40 | 79 | 126 |
| H200 layer eager | 802 | 471 | 479 |
| H20 layer graph | 198 | 314 | 453 |

The torch + NCCL part is 175-245 us per layer on H200 (about 20 small kernels, bf16 GEMMs over the 7168-wide W_down / W_up / shared
expert, one NCCL all-gather and one all-reduce): 4-5x the MoE kernel at M = 1. Fusing it (tail fusion + sharded router / W_down with an
NVLink all-gather) is the next step; the eager numbers show why CUDA graphs are mandatory on this path.

### SOL analysis (H200, TP8, per rank)

The block is memory-bound: FP8 tensor-core work is 0.13 GFLOP per token per rank (16 experts x 3 GEMMs of 384 x 3584), 2.1 us at
M = 32 against a 1979 TFLOPS peak, so SOL = bytes read per rank / HBM bandwidth. H200 HBM3e peak is 4.8 TB/s; 4.0 TB/s (83 %) is the
usual achievable copy bandwidth and is used as the SOL reference (not measured on these nodes; both columns are given).

Bytes per rank: one unique expert costs 2.19 MB (w1/w3 2 x 384 x 3584 nibbles + E8M0 scales = 1.46 MB, w2 3584 x 384 nibbles + scales
0.73 MB). The unique-expert counts below are from the synthetic uniform routing of the benchmark; real routing is more skewed, so real
bytes and SOL at M > 1 are lower and the kernel's fraction correspondingly lower.

Kernel (routed experts + 8-way reduce), 896 experts:

| M | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| unique experts / bytes per rank, MB | 16 / 35 | 28 / 61 | 51 / 112 | 88 / 193 | 121 / 265 | 153 / 336 |
| SOL at 4.8 TB/s peak, us | 7.3 | 12.8 | 23.3 | 40.2 | 55.3 | 69.9 |
| SOL at 4.0 TB/s achievable, us | 8.8 | 15.4 | 28.0 | 48.3 | 66.4 | 83.9 |
| measured H200 TP8, us | 39.5 | 47.4 | 60.4 | 79.2 | 98.7 | 126.0 |
| achieved HBM bandwidth, TB/s | 0.89 | 1.30 | 1.85 | 2.44 | 2.69 | 2.66 |
| **% of SOL (4.0 TB/s)** | **22** | **32** | **46** | **61** | **67** | **67** |
| % of peak 4.8 TB/s | 19 | 27 | 39 | 51 | 56 | 55 |
| H20 TP8, % of its 4.0 TB/s SOL | 21 | 29 | 40 | 48 | 52 | 53 |

M = 1 is 22 % of SOL and the gap is latency, not bandwidth: 16 panels x 3 items x 28 k-tiles is one wave on 132 SMs with a dependent
FC1 -> FC2 chain, a 16-slot combine and a one-shot barrier, which is why M = 1 costs the same ~40 us on H20 and H200. From M >= 8 the
kernel sits at 61-67 % of achievable bandwidth (2.4-2.7 TB/s), a +8-14 point gain over H20 and in line with the DeepSeek tp4moe2 kernel
(23-54 % on H20). The remaining third at large M is the FC2 k-tile serialization and the tail (combine + reduce) that no longer hides
behind loads.

Whole layer, counting every weight the block must read per rank per layer with everything sharded 8 ways (router 1.6 MB, W_down 6.4,
W_up 6.4, shared expert 33.0, total 47.5 MB dense + the routed experts above; the current glue reads 71.6 MB of dense weights because
the fp32 router is replicated):

| M | 1 | 8 | 32 |
|---|---|---|---|
| bytes per rank, MB | 83 | 241 | 383 |
| SOL at 4.0 TB/s, us | 20.6 | 60.1 | 95.8 |
| measured layer graph, us | 214 | 280 | 371 |
| **% of SOL** | **10** | **21** | **26** |
| per token through 92 MoE layers: SOL / measured, ms | 1.9 / 19.7 | 5.5 / 25.8 | 8.8 / 34.1 |

The layer is at 10-26 % of SOL and the kernel is not the bottleneck: 47.5 MB of dense weights is 12 us of HBM time but the torch + NCCL
glue that reads them takes 175-245 us (about 20 kernels and two NCCL collectives, launch- and sync-bound). Closing the layer gap is
therefore (1) fusing the tail (RMSNorm, W_up, shared expert, final reduce) into the persistent kernel, (2) sharding the router and
W_down with the all-gather on NVLink inside the kernel, and only then (3) the kernel's own M = 1 latency. At M = 1 that is a
reachable 5-8x on the MoE block; the decode step as a whole also carries attention and the 46 x 2 pipeline stages.

### Comparison with the industry kernels on H200 (`bench/bench_baselines.py`, `bench/nccl_ar.py`)

Same idle node (viking-prod-309, computelab job 5508653, `sglang_v0.5.22` image: torch 2.14.1+cu130, sgl_kernel 0.4.9, flashinfer 0.7.1,
NCCL 2.30.7), same TP8 shard (896 experts, 3584 x 384 per rank, top-16, SiTU), **identical routing** (`test_k3.random_routing`, so the
unique-expert counts and the bytes match), same measurement (CUDA graph of 32 back-to-back calls, us per call, min over 30 replays).
The baselines are the paths SGLang runs for Kimi K3 MXFP4 on Hopper, built through SGLang's own weight preparation code:

- **Marlin MXFP4 W4A16** (`--moe-runner-backend marlin`: `prepare_moe_mxfp4_layer_for_marlin` + `fused_marlin_moe`, E8M0 scales, BF16
  activations, fused SiTU) -- the same bytes as our kernel.
- **FlashInfer cutlass W4A16** (`flashinfer_mxfp4` on SM90: `cutlass_fused_moe(use_w4_group_scaling=True)` + SM90 interleave, SiTU).
- **FlashInfer Humming W4A8** (`flashinfer_mxfp4` with `precision=fp8`: `use_wfp4afp8_humming=True`, FP8 activations, SiTU) -- the kernel
  family tp4moe2 was measured against in DeepSeek serving.
- Not run: vLLM (not in the image; its K3 Hopper path is the same Marlin kernel), SGLang Triton BF16 fused_experts (needs the server's
  parallel context; W16, 4x the bytes, not a contender), and this repo's `cake_fused_moe` / `kimi_k3_latent_moe` K3 kernels (sm_100a /
  sm_103a only).

Routed experts only, one GPU (the baselines have no cross-rank reduce; our TP8 figure includes the 8-way reduce):

| M | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| unique experts / bytes per rank, MB | 16 / 35 | 28 / 61 | 51 / 112 | 88 / 193 | 121 / 265 | 153 / 336 |
| **ours, single GPU (ndev 1)** | **36.7** | **44.3** | **56.7** | **74.5** | **92.5** | **113.6** |
| **ours, TP8 incl. 8-way reduce** | **39.6** | **47.3** | **60.4** | **78.9** | **98.4** | **125.8** |
| Marlin MXFP4 W4A16 | 33.4 | 44.5 | 70.3 | 101.1 | 133.9 | 168.2 |
| FlashInfer cutlass W4A16 | 81.4 | 102.8 | 137.0 | 191.4 | 240.1 | 292.2 |
| FlashInfer Humming W4A8 | 78.2 | 95.6 | 125.4 | 173.2 | 215.8 | 259.2 |
| NCCL all-reduce [M, 3584] bf16 x 8 (what a baseline adds under TP8) | 21.9 | 22.3 | 23.4 | 27.2 | 26.7 | 21.8 |
| ours ndev 1 vs Marlin | 0.91x | 1.00x | 1.24x | 1.36x | 1.45x | 1.48x |
| ours ndev 1 vs Humming W4A8 | 2.13x | 2.16x | 2.21x | 2.32x | 2.33x | 2.28x |
| ours TP8 vs Marlin + NCCL | 1.40x | 1.41x | 1.55x | 1.63x | 1.63x | 1.51x |
| achieved HBM TB/s: ours ndev 1 / Marlin / Humming | 0.96 / 1.05 / 0.45 | 1.39 / 1.38 / 0.64 | 1.97 / 1.59 / 0.89 | 2.59 / 1.91 / 1.11 | 2.87 / 1.98 / 1.23 | 2.95 / 2.00 / 1.29 |
| % of SOL at 4.0 TB/s: ours ndev 1 / Marlin / Humming | 24 / 26 / 11 | 35 / 35 / 16 | 49 / 40 / 22 | 65 / 48 / 28 | 72 / 50 / 31 | 74 / 50 / 32 |

Reading: from M = 4 the kernel is the fastest of the set by a growing margin (1.24-1.48x over Marlin, 2.1-2.3x over Humming, 2.2-2.6x
over the cutlass W4A16 path) and reaches 65-74 % of achievable HBM bandwidth where Marlin plateaus at 50 % and Humming at ~30 %. As a
TP8 layer (kernel + reduce) it is 1.4-1.6x ahead of Marlin + NCCL at every M. **At M = 1 it is not the fastest kernel: Marlin is 3.3 us
(9 %) faster on the expert part** (33.4 vs 36.7 us; both ~25 % of SOL), and the two tie at M = 2. That is the latency floor described
in the SOL section (one wave of 16 narrow panels, FC1 -> FC2 dependency, 16-slot combine, one-shot barrier); Marlin's single
moe_wna16 GEMM per stage has less fixed cost at one token. The NCCL figure is torch.distributed NCCL 2.30.7 with NVLS disabled
(`NCCL_NVLS_ENABLE=0`: the multicast bind fails in this container); SGLang's custom one-shot all-reduce would be a few us lower, which
does not change the ranking.

**Two-node run, TP8 x PP2 on 2 x 8 H200** (job 5377802, viking-prod-214 + viking-prod-297, torchrun 2 x 8, NCCL 2.27.7 world over
the inter-node link, 2 MoE layers per stage, 896 experts; `k3_pp2.sbatch`):

| M | 1 | 8 | 32 |
|---|---|---|---|
| stage-1 output == all 4 layers on one TP8 group | bitwise | bitwise | bitwise |
| single-stream wall time per layer incl. the stage hop (eager, min, us) | 434 | 600 | 1190 |

The pipeline result is bitwise identical to running every layer on one TP8 group, as it must be (the hop moves BF16 hidden states, the
math per stage is unchanged). The wall times are eager (no CUDA graph, host launch overhead, one NCCL send/recv per 2 layers across the
nodes) and are an upper bound; the graph numbers above are what the per-layer cost looks like in an engine.

The one-node colocated rehearsal (16 ranks, two per GPU, gloo world) also matched bitwise at M = 1 on H200 but timed out at M = 8
(two persistent kernels from two processes on one GPU have no co-scheduling guarantee); use the real two-node mode.

### Running on computelab

Scratch copy: `/home/scratch.jinyanc_wwfo/k3_tp8moe/` (`k3_payload.sh` single|tp8 runs inside the container; `k3_launch.sh` reads the
image from `IMAGE`; `k3_single.sbatch` / `k3_tp8.sbatch`; logs in `logs/`). Of the images on scratch only the sglang `.sqsh` are valid
(`pytorch_2_9_0_cuda13_devel.sqsh` and `vllm_0.19.0_cu130.sqsh` are zero-filled), and the sglang image has nvcc 12.9 against torch
cu130, so the jobs use the nvcr pytorch 25.09 image (pulled by pyxis, ~5 min on a fresh node). `K3_SKIP_CUDA_CHECK=1` bypasses the
cpp_extension CUDA major-version check when nvcc and torch disagree.
Two-node job (`k3_pp2.sbatch` -> `k3_pp2_node.sh`, one torchrun per node, `--rdzv_backend c10d` on the master resolved by the sbatch
script since `scontrol` is not in the container): the viking nodes expose one routable NIC (`ens6f0np0`, 10.176/16) next to a dozen
docker `br-*` bridges that NCCL does not exclude by default, so without `NCCL_SOCKET_IFNAME=ens6f0np0` the cross-node bootstrap hangs
(stage-0 ranks spin at 100 % GPU in the first world collective, stage-1 ranks poll sockets; job 5373544, 40 min, no output).

## Two-node layout: `k3_pp2.py` (TP8 x PP2)

Stage s holds layers [s L, (s + 1) L) on its own TP8 group (own IPC-linked LL buffers, NCCL group); the BF16 hidden states hop
stage 0 -> 1 with `send` / `recv` (NCCL over the inter-node link in the real run). Rehearsed on one 8-GPU node with 16 ranks (two
per GPU, `--colocated`: the world group is gloo and the hop goes through host memory, since NCCL refuses two ranks on one GPU):
the pipeline output is **bitwise equal** to all layers run on a single TP8 group (M = 1, 8; 2 x 2 layers). The real two-node run (`torchrun --nnodes 2 --nproc_per_node 8 ... k3_pp2.py`, env:// rendezvous, device = LOCAL_RANK) was executed on
2 x 8 H200 on computelab: bitwise equal at M = 1, 8, 32 (see the H200 section).

## Run

```bash
PY=python   # torch cu130 (nvcc 13.1), TORCH_CUDA_ARCH_LIST=9.0a is set by k3_host
# needs ../serving_tp4moe2_v26/*.cuh (shared pipeline headers)
$PY k3_synth.py                                              # layout round trips (kernel layout, checkpoint format -> kernel layout)
$PY test_k3.py --local --ms 1 2 4 8 16 32 --nexp 64          # one GPU vs torch reference
$PY test_k3.py --local --ms 8 32 --nexp 896 --ckpt            # 896 experts, weights generated in the checkpoint format and converted
$PY test_k3.py --tp8 --ms 1 8 32 --nexp 128 --b2b             # 8 GPUs: one-shot LL reduce, every rank vs the 8-rank reference, graph replay
$PY test_k3.py --tp8 --ms 1 2 4 8 16 32 --nexp 896 --time     # kernel-only timing (graph of --copies calls)
$PY test_k3_layer.py --ms 1 8 32 --nexp 896 --time            # whole layer vs block reference, eager + CUDA-graph timing
$PY k3_pp2.py --layers 2 --ms 1 8 --nexp 32 --colocated --time  # TP8 x PP2 rehearsal: 16 ranks on 8 GPUs, pipeline vs single group
```

## Next

1. Tail fusion: RMSNorm + W_up K-slice + shared-expert shard + 7168-wide reduce inside the kernel (second LL stage) instead of torch + NCCL.
2. Sharded router + W_down GEMM with an NVLink all-gather of z (today: torch GEMM + NCCL all-gather).
3. Dense mode (M <= 8) and small-M latency work: the M = 1 gap to SOL and the 3 us Marlin leads by at one token.
4. Activation quantization: K3 was trained with MXFP8 (per-32-group) activations; the kernel uses per-token FP8. Evaluate on real weights.
5. RS + AG tail for M > 32 / larger prefill-ish batches; M * topk > 512 tables.
6. Engine hook (SGLang / vLLM K3 model) with 46 layers per stage; CUDA-graph the whole stage incl. the hop.
