// tp4moe2: TP4 fused MoE decode kernel for H20 (sm_90a), DeepSeek-V4-Flash routed experts (hidden 4096, per-rank
// intermediate 512, <= 256 experts, top-k <= 8), Humming fused-e8m0 W4A8 weights used in place.
// Design + rationale: pr13_h20/tp4_round1/DESIGN_tp4moe2.md.  One persistent kernel, one CTA per SM (640 threads:
// 4 consumer warpgroups + 1 producer warpgroup):
//   prologue  every CTA builds the same routing tables from (ids, weights): union (ascending expert id), per-expert token
//             entries (ascending token), token panels of <= 32 entries.  Deterministic, no inter-CTA traffic.
//   FC1       work = (panel, 128-col group q) items of 32 k-tiles; stream-K over all CTAs (equal k-tile shares).  The
//             CTA owning an item's k-tile 0 adds the other CTAs' fp32 partials in CTA order, applies x scale * s2*64 ->
//             BF16 -> clamp -> SiLU -> BF16 -> * up -> BF16 activation, and publishes act (bf16) + per-token partial amax.
//   FC2       work = (panel, 256-row tile) units, static round robin.  A unit waits for its panel's 4 activation flags,
//             quantizes act per token over all 512 columns (Humming's per-token quant) into the smem B tile, runs K=512,
//             and writes BF16(y * hs * s2*64) per (token, top-k slot).
//   tail      T1 per (row tile, token): sum_k w_k * y_k (fp32, k order) -> x factor -> BF16 -> + shared -> BF16, LL push to
//             the owner rank; T2 owner: rank-ordered fp32 sum -> BF16, LL push to all ranks; T3: pull -> output.
// Inner k-tile step (both GEMMs): dequant 4 k32 slices (Humming LUT), wait the previous K128 chain, D += C, issue the next
// chain (single asm block of 4 wgmma m64nNk32 into one accumulator set).
// Fused shared expert (P.shw, v20): the production FP8 shared expert (128x128 block-scaled gate_up [1024][4096], down [4096][512])
// runs as extra panels after the routed ones (tokens 32 sp .. as entries); its numerics follow the production kernels instruction
// for instruction (shared_num.cuh): group-128 input quantization, per-K128 promotion D = fma(sa * sb, C, D), BF16 gate_up, fast-math
// SiLU, group-128 activation quantization, BF16 output -- added in T1 exactly where the external shared output was.  Its k-steps
// are SS-wgmma (both operands TMA-loaded, 128B swizzle): one ring stage = 128 weight rows x K128 for one warpgroup pair.
#include <cfloat>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <string>
#include <unordered_map>
#include "chain4.cuh"
#include "chain_ss.cuh"
#include "shared_num.cuh"
#ifndef TP2_KPIPE_PF
#define TP2_KPIPE_PF 0   // 1: kpipe_pf.cuh (slice s + 1's smem loads issued under slice s's wgmma, no bar.warp.sync between slices; same arithmetic)
#endif
#if TP2_KPIPE_PF
#include "kpipe_pf.cuh"
#else
#include "kpipe.cuh"
#endif

#ifndef TP2_NS
#define TP2_NS tp4moe2_v20
#endif
#ifndef TP2_DBG_EXIT
#define TP2_DBG_EXIT 0   // timing diagnostic (outputs invalid): 1 return at kernel start, 2 return after the prologue (routing + tables)
#endif
#ifndef TP2_QUANT_PRE
#define TP2_QUANT_PRE 0   // (measured neutral) PDL launch + fused quantization: the quantizing threads run before griddepcontrol.wait (x precedes the routing)
#endif
#ifndef TP2_LOCAL_QUANT
#define TP2_LOCAL_QUANT 0   // (measured slower: all CTAs read all x rows; kept for A/B) dense + fused quantization: every CTA quantizes all M tokens itself into its private 16-row slot of the
                            // workspace (warps 8..15, one token per warp), no cross-CTA token flags on the first x load
#endif
#ifndef TP2_DBG_KLOOP
#define TP2_DBG_KLOOP 0   // timing diagnostics (garbage outputs): 1 = consumers skip dequant + MMA (wait / release only), 2 = producer
                          // skips every weight / x / h load (arrive without tx: consumers compute on stale smem)
#endif
#ifndef TP2_GATHER_FENCE
#define TP2_GATHER_FENCE 0   // 1: consumers fence.proxy.async after an FC1 stage lands (cp.async-written x tile read by wgmma)
#endif

namespace TP2_NS {

// Kimi K3 Stable LatentMoE on TP8: experts work in the latent space (DIM = 3584 in / out), per-rank intermediate 3072 / 8 = 384,
// 896 routed experts, top-16.  One launch = routed experts of one layer on this rank + 8-way one-shot LL reduce of the latent sum.
constexpr int DIM = 3584, INTER = 384, NEXP = 896, TOPK_MAX = 16, MAXM = 128, NDEV = 8;
constexpr int ITEMS = INTER / 128;           // FC1 items per panel (128 gate + 128 up rows each)
constexpr int MAXE = MAXM * TOPK_MAX;        // entries (token, expert) upper bound
constexpr int MAXP = 512;                    // panels upper bound: <= entries <= M * topk (host: M * topk <= MAXP); one block-wide table thread per panel
constexpr int NKT1 = DIM / 128;              // 28 FC1 k-tiles per item
constexpr int NKT2 = INTER / 128;            // 3 FC2 k-tiles per unit
constexpr int NRT = DIM / 256;               // 14 FC2 row tiles
constexpr int NMSG = 43;                     // LL messages per (row tile, token): 6 bf16 + tag each, last carries 4
constexpr int NCONS = 512, NTHREADS = 640, PROD_WARP = 16, GATHER_WARP = 17;
constexpr int NMAX = 32;                     // token panel width (wgmma N <= 32)
constexpr int HLB = 1024;                    // one 64-row x k32 Humming block
constexpr int STAGE_W = 16384, STAGE_S = 1024, STAGE_WS = STAGE_W + STAGE_S;   // [4 k32 slices][4 blocks][1 KB] + [4][4 x 64 B]
constexpr int STAGE_X = NMAX * 128;          // FC1 x tile: [32 tokens][128 B] (128B swizzle)
constexpr int STAGE = STAGE_WS + STAGE_X;    // 21504 (multiple of 1024)
#ifndef TP2_KPS
#define TP2_KPS 1   // (2 measured 0.5-4 us SLOWER at every M, oracle-correct) k-tiles per ring stage: one full / empty mbarrier round trip per KPS k-tiles (the per-stage handshake ~300 cycles
                    // is exposed in every k-step: 1 -> 0.46 us / k-tile at N8, 2 -> ~0.39 in tma_bw.cu); stream-K ranges are KPS-aligned
#endif
constexpr int KPS = TP2_KPS;
constexpr int NST = KPS == 2 ? 6 : 7;        // ring slots (one k-tile each); barrier stages = NST / KPS (slot pairs)
constexpr int OFF_STG = NST * STAGE;         // epilogue staging: FC1 up exchange [2][128][NMAX/2] fp32 + act tile [NMAX][128] bf16 | FC2 out tile [NMAX][256] bf16
constexpr int ACT_PITCH = 136;                    // act tile row pitch (bf16): 272 B rows, 16 B apart in banks (no 4-way conflicts)
constexpr int OUT_PITCH = 264;                    // FC2 out tile row pitch (bf16): 528 B rows
constexpr int OUT_TILE = NMAX * OUT_PITCH * 2;    // 16896
constexpr int STG_BYTES = 2 * OUT_TILE;           // FC2: two out tiles (consumers fill one while the store warps drain the other)
static_assert(16384 + NMAX * ACT_PITCH * 2 <= STG_BYTES, "FC1 up exchange + act tile fit the staging area");
constexpr int OFF_TAB = OFF_STG + STG_BYTES;
constexpr int TAB_IDS = 0;                             // uint16 [MAXM][TOPK_MAX] expert of (t, k) (row = two 16-B loads), 0xFFFF = invalid
constexpr int TAB_ENT = TAB_IDS + MAXE * 2;            // uint16 [MAXE] token of entry
constexpr int TAB_UEXP = TAB_ENT + MAXE * 2;           // uint16 [NEXP] union expert of slot
constexpr int TAB_OFF = TAB_UEXP + NEXP * 2;           // uint16 [NEXP + 8] entry offset of slot (+ total)
constexpr int TAB_PID = TAB_OFF + (NEXP + 8) * 2;      // uint16 [NEXP + 8] first panel of slot (+ total)
constexpr int TAB_P2U = TAB_PID + (NEXP + 8) * 2;      // uint16 [MAXP] slot of panel
constexpr int TAB_KM = TAB_P2U + MAXP * 2;             // uint16 [MAXE] top-k slots (bit mask) of entry
constexpr int TAB_E2P = TAB_KM + MAXE * 2;             // uint16 [MAXE] panel of entry
constexpr int TAB_FS = TAB_E2P + MAXE * 2;             // float [NMAX] per-token epilogue factor
constexpr int TAB_AM = TAB_FS + NMAX * 4;              // float [NMAX][16] amax partials (per warp)
constexpr int TAB_MISC = TAB_AM + NMAX * 16 * 4;       // int [32]: 0 U, 1 entries, 2 panels, 3 queued quantizations
constexpr int TAB_CCUM = TAB_MISC + 32 * 4;            // uint32 [MAXP + 1] exclusive FC1 cost prefix over panels
constexpr int TAB_PROW = TAB_CCUM + (MAXP + 8) * 4;    // uint32 [MAXP + 1] exclusive prefix of padded panel widths (xg rows)
constexpr int TAB_PINF = TAB_PROW + (MAXP + 8) * 4;    // uint32 [MAXP] panel info: e | n << 8 | ebase << 14
constexpr int MAXG = 160;
constexpr int TAB_BND = TAB_PINF + MAXP * 4;           // int [2][MAXG + 1] FC1 stream-K range starts of every CTA, per panel group
constexpr int TAB_QQ = TAB_BND + (2 * (MAXG + 1) + 6) * 4;   // (unused since v3j)
constexpr int TAB_UNIT = TAB_QQ + 64 * 4;              // int2 [8] FC2 unit ring (producer -> consumers), int [NRT] units done per row tile, int4 [2] out tile info
constexpr int NRTP = (NRT + 3) & ~3;                   // NRT padded to a multiple of 4 ints (16-B alignment of the out tile info)
constexpr int TAB_BAR = TAB_UNIT + (16 + NRTP + 8) * 4; // uint64 full[NST], empty[NST], epi[4], hdone, gbar, fullo[2], freeo[2], sdone
constexpr int TAB_DBG = TAB_BAR + (2 * NST + 12) * 8;   // (+ xsb: dense fused-quant token scales in TAB_XS)  // TP2_STAMPS: u64 [32] per-CTA stamps / accumulators (flushed at exit)
constexpr int TAB_GRW = TAB_DBG + 48 * 8;              // float [128] fused gate, small tables: routing weight of (t, k)
constexpr int TAB_XS = TAB_GRW + 128 * 4;               // float [NMAX] dense + fused quantization: per-token x scale (warp 19, xsb)
constexpr int TAB_BYTES = TAB_XS + NMAX * 4;
#ifndef TP2_COST
#define TP2_COST 100, 104, 109, 132                    // relative FC1 k-tile cost by panel width class N8 / N16 / N24 / N32 (measured, tp4moe2 v3d)
#endif
#define TP2_A0(a, b, c, d) a
#define TP2_A1(a, b, c, d) b
#define TP2_A2(a, b, c, d) c
#define TP2_A3(a, b, c, d) d
#define TP2_APPLY(m, x) m(x)
constexpr int k_cost0 = TP2_APPLY(TP2_A0, TP2_COST), k_cost1 = TP2_APPLY(TP2_A1, TP2_COST), k_cost2 = TP2_APPLY(TP2_A2, TP2_COST), k_cost3 = TP2_APPLY(TP2_A3, TP2_COST);
#ifndef TP2_EPI_COST
#define TP2_EPI_COST 250, 300, 400, 500                // owner epilogue cost of one FC1 item by width class (same units: 100 = one N8 k-tile)
#endif
#ifndef TP2_EPI_SCALE
#define TP2_EPI_SCALE 300   // percent; compact panels only (dense mode: 0 -- measured best at M1 / M8, 300 best at M32 / M64)
#endif
constexpr int k_epi0 = TP2_APPLY(TP2_A0, TP2_EPI_COST), k_epi1 = TP2_APPLY(TP2_A1, TP2_EPI_COST), k_epi2 = TP2_APPLY(TP2_A2, TP2_EPI_COST), k_epi3 = TP2_APPLY(TP2_A3, TP2_EPI_COST);
__device__ __forceinline__ uint32_t sel4(int cl, int a, int b, int c, int d) { return (uint32_t)(cl == 0 ? a : (cl == 1 ? b : (cl == 2 ? c : d))); }   // immediates, no constant-bank load
__device__ __forceinline__ uint32_t c_cost(int cl) { return sel4(cl, k_cost0, k_cost1, k_cost2, k_cost3); }
// stream-K cost of one panel of width class cl: 4 items x (owner epilogue + 32 k-tiles); a position x (cost units from the panel
// start) maps to the item-major k-tile whose range it falls in (the epilogue cost sits before an item's k-tile 0: its owner pays it)
__device__ __forceinline__ uint32_t c_epi(int cl, bool dense) {
    return dense ? 0u : sel4(cl, k_epi0 * TP2_EPI_SCALE / 100, k_epi1 * TP2_EPI_SCALE / 100, k_epi2 * TP2_EPI_SCALE / 100, k_epi3 * TP2_EPI_SCALE / 100);
}
#ifndef TP2_SHARED
#define TP2_SHARED 0   // K3: the shared expert is full-width (7168), not fused here (compiled out)
#endif
#ifndef TP2_SH_DEFER
#define TP2_SH_DEFER 4   // FC2 producer: shared expert units deferred while their panel is not ready (0: wait in place, as v20)
#endif
#ifndef TP2_SH_COST
#define TP2_SH_COST 170   // percent: a fused shared expert k-block (2 x 16 KB FP8 weights + x) vs a routed k-tile (8 KB MXFP4) of its width class
#endif
__device__ __forceinline__ uint32_t c_cost_x(int cl, bool sh) { return sh ? c_cost(cl) * TP2_SH_COST / 100u : c_cost(cl); }
__device__ __forceinline__ uint32_t panel_cost(int cl, bool dense, bool sh = false) { return (uint32_t)ITEMS * (c_epi(cl, dense) + (uint32_t)NKT1 * c_cost_x(cl, sh)); }
__device__ __forceinline__ int panel_kt(uint32_t x, int cl, bool dense, bool sh = false) {
    const uint32_t e = c_epi(cl, dense), cc = c_cost_x(cl, sh), ic = e + (uint32_t)NKT1 * cc;
    const uint32_t q = min((uint32_t)ITEMS - 1u, x / ic), y = x - q * ic;
    return (int)(q * (uint32_t)NKT1 + (y < e ? 0u : min((uint32_t)NKT1 - 1u, (y - e) / cc))) & ~(KPS - 1);   // (KPS-aligned: a ring stage never spans two pieces)
}
constexpr int SMEM_TOTAL = OFF_TAB + TAB_BYTES;
constexpr int SMEM_ALLOC = SMEM_TOTAL + 1024;
static_assert(STAGE % 1024 == 0 && OFF_STG % 1024 == 0, "swizzle atoms");
static_assert(SMEM_ALLOC <= 232448, "smem budget");
static_assert(TAB_BAR % 8 == 0 && (TAB_UNIT + (16 + NRTP) * 4) % 16 == 0, "mbarrier / int4 alignment");
// prologue scratch (inside the ring, dead before the first TMA)
constexpr int PRO_BITS = 0;                            // uint32 [NEXP][16] token bits
constexpr int PRO_PRE = PRO_BITS + NEXP * 16 * 4;      // uint16 [NEXP][16] exclusive word prefix
constexpr int PRO_CNT = PRO_PRE + NEXP * 16 * 2;       // int [NEXP]
constexpr int PRO_OFFE = PRO_CNT + NEXP * 4;           // int [NEXP] entry offset by expert
constexpr int PRO_WS = PRO_OFFE + NEXP * 4;            // int [72] warp sums
constexpr int PRO_PIDE = PRO_WS + 72 * 4;              // int [NEXP] first panel of expert
constexpr int PRO_GW = PRO_PIDE + NEXP * 4;            // (fused router GEMM weights: unused here)
static_assert(PRO_GW + 4 * 128 * 64 <= OFF_STG, "prologue scratch fits in the ring");
static_assert((unsigned long long)MAXP * ITEMS * (600 + NKT1 * 200) * 160 < (1ull << 32), "32-bit stream-K cost math (k-tile cost <= 200, epilogue <= 600, <= 160 CTAs)");
constexpr int NSTAMP = 48;   // + 24 last FC1 piece done (consumers) 25 hdone 26 quant done 27 first FC2 unit start; 28 sum fc2 first-stage waits; 29 x gathered   // 0 start 1 tables 2 fc1 done 3 fc2 done 4 T1 5 T2 6 end 7 first tile | ns sums: 8 fc1 k-loop 9 fc1 partial 10 fc1 epi 11 fc2 wait 12 fc2 B 13 fc2 k-loop 14 fc2 epi 15 fc2 units

struct alignas(64) Params {
    CUtensorMap xmap;                        // dense mode: x [M][4096] e4m3, box {128 B, md rows}, 128B swizzle (OOB rows zero-filled)
    CUtensorMap xrow;                        // compact panels of <= TP2_ROWX rows: x [M][4096] e4m3, box {128 B, 1 row}, 128B swizzle
    const CUtensorMap* tmaps;                // [0] w13 5-D, [1] s13 4-D, [2] w2 3-D, [3] s2 3-D
    const float* fc1_s2;
    const float* fc2_s2;
    const uint8_t* xq;                       // [M][4096] e4m3
    const float* xs;                         // per-token scale at xs[t * xs_stride]
    int xs_stride;
    const __nv_bfloat16* xb;                 // fused input quantization: BF16 x [M][4096] (xq / xs then point at the workspace), else null
    const float* glog;                       // fused gate: router logits fp32 [M][256] (ids / rw then point at the workspace), else null
    const float* gbias;                      // fused gate: score correction bias [256]
    unsigned* gflag;                         // [MAXM] fused gate (block-wide tables): token t's ids / weights written (tag hseq)
    unsigned* gcnt;                          // [2] fused router GEMM: CTAs whose logits are written, per epoch parity
    const __nv_bfloat16* gw;                 // fused router GEMM (M <= 16): router weight bf16 [256][4096] (glog then points at the workspace)
    unsigned* xflag;                         // [MAXM] fused quantization: token t quantized (tag hseq)
    unsigned hseq;                           // host launch sequence number (fused quantization flags)
    const int* ids;                          // [M][topk]
    const float* rw;                         // [M][topk]
    int topk;
    const __nv_bfloat16* shared;             // [M][4096] or null
    float factor;
    __nv_bfloat16* act;                      // [MAXE][512]
    float* amax;                             // [MAXE][4]
    float* part;                             // [grid][NCONS][NMAX/2]
    unsigned* act_flags;                     // [MAXP][4] epoch tags (unused since v2; kept for the workspace layout)
    uint8_t* hq;                             // [MAXP][4 k-tiles][NMAX][128] quantized FC2 input (128B-swizzled rows)
    uint8_t* xg;                             // [rows][32 k-tiles] x gathered per panel: panel p at row prow[p], [kt][Npad][128] swizzled
    unsigned* xg_cnt;                        // [2][MAXP] CTAs done gathering each panel's x rows (epoch parity)
    unsigned* fc2_next;                      // [2] FC2 unit queue head (epoch parity)
    float* hs;                               // [MAXP][NMAX] per-token FC2 input scale
    unsigned* h_flags;                       // [MAXP] epoch tag: panel quantized
    unsigned* pan_cnt;                       // [2][MAXP] finished FC1 items per panel (epoch parity)
    unsigned* part_flags;                    // [grid] epoch tags
    unsigned* rt_cnt;                        // [2][NRT] FC2 row-tile completion counters (epoch parity)
    unsigned* work;                          // [0] epoch, [1 + parity] CTA done counters
    __nv_bfloat16* outp;                     // [MAXM][TOPK_MAX][DIM] per-slot expert outputs
    uint4* rs[NDEV];                         // peer one-shot buffers [2 epoch parity][NDEV src][NRT rtile][OS_MAXM][43]
    uint4* ag[NDEV];                         // (RS + AG mode: not used by this kernel -- one-shot only)
    __nv_bfloat16* out;                      // [M][DIM] latent sum over the ranks (BF16)
    int M, n_exp, my_rank, ndev;
    int dense, md;                           // dense mode (small M): FC1 B operand = x rows 0..md-1 for every expert, no gather
    int lquant;                              // dense + fused quantization, CTA-local (TP2_LOCAL_QUANT): xmap rows b * 16 .., scales in TAB_XS (xsb)
    int groups;                              // FC1 panel groups (1 or 2, see TP2_GROUPS): host-chosen by M (TP2_GROUPS_MINM)
    int oneshot;                             // TP8 all-reduce: 1 = one-shot push to all ranks (always, M <= OS_MAXM)
    int pdl_early;                           // PDL launch: 1 = routing tables (ids only) + first weight loads before griddepcontrol.wait
    // fused shared expert (shw): production DeepseekV2MLP FP8 weights, used in place
    int shw;
    CUtensorMap w1s, w2s, xgs;               // 2D uint8 maps, 128B swizzle: gate_up [1024][4096] / down [4096][512] box {128 B, 128 rows};
                                             // xqg [M][4096] box {128 B, 32 rows} (OOB rows zero)
    const float* sb1;                        // [8][32] gate_up block scales (weight_scale_inv)
    const float* sb2;                        // [32][4] down block scales
    uint8_t* xqg;                            // [MAXM][4096] e4m3: x per (token, 128-group) (sglang per_token_group_quant v2)
    float* xsT;                              // [32][MAXM] its scales, group-major (one 128-B bulk copy per (k-block, panel))
    float* hs4;                              // [MAXP][4][NMAX] shared FC2 input group scales
    __nv_bfloat16* shout;                    // [MAXM][4096] shared expert output (BF16; T1 adds it as the external shared output)
    unsigned long long* dbg;                 // [grid][NSTAMP] globaltimer stamps (TP2_STAMPS builds)
    unsigned long long* tl;                  // [grid][16 units][8] clock64 timeline of the first FC2 units (TP2_STAMPS builds)
};

// ------------------------------------------------------------------------------------------------ device helpers
#ifdef TP2_STAMPS
__shared__ unsigned long long* g_dbg_smem;
__device__ unsigned long long* dbg_smem() { return g_dbg_smem; }
#endif
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
#ifdef TP2_STAMPS
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ unsigned long long* dbg_smem();
#define STAMPT(P, k) do { if ((P).dbg) dbg_smem()[(k)] = gtimer(); } while (0)   // any (single) thread
#define CKM(P, k) do { if ((P).dbg && threadIdx.x == 0) dbg_smem()[(k)] = clock64(); } while (0)
#define STAMP(P, k) do { if ((P).dbg && threadIdx.x == 0) dbg_smem()[(k)] = gtimer(); } while (0)
#ifdef TP2_LIGHT_STAMPS   // phase stamps only (no accumulators / per-unit timeline): least distortion
#define TSTART(v) do {} while (0)
#define TACC(P, k, v) do {} while (0)
#define TL(P, j, k) do {} while (0)
#else
#define TSTART(v) const unsigned long long v = clock64()
#define TACC(P, k, v) do { if ((P).dbg && threadIdx.x == 0) dbg_smem()[(k)] += clock64() - (v); } while (0)
#define TL(P, j, k) do { if ((P).tl && threadIdx.x == 0 && (j) < 16) (P).tl[(blockIdx.x * 16 + (j)) * 8 + (k)] = clock64(); } while (0)
#endif
#else
#define STAMP(P, k) do {} while (0)
#define STAMPT(P, k) do {} while (0)
#define CKM(P, k) do {} while (0)
#define TSTART(v) do {} while (0)
#define TACC(P, k, v) do {} while (0)
#define TL(P, j, k) do {} while (0)
#endif
#if defined(TP2_STAMPS) && defined(TP2_TL_FC1)   // FC1 stage timeline: tl[cta][stage < 16][k] (globaltimer ns); light-stamps builds only
__shared__ unsigned long long* g_tl_ptr;
__shared__ int g_tl_j, g_tl_fc1;
#define TL1(j, k) do { if (g_tl_ptr && (j) < 16) g_tl_ptr[(blockIdx.x * 16 + (j)) * 8 + (k)] = gtimer(); } while (0)
#else
#define TL1(j, k) do {} while (0)
#endif
#if defined(TP2_STAMPS) && defined(TP2_FC2_ACC)   // FC2 per-phase cycle sums (thread 0): slot 23 running clock, slots 9..13 sums
#define FC2A_SET(P) do { if ((P).dbg && threadIdx.x == 0) dbg_smem()[23] = clock64(); } while (0)
#define FC2A(P, k) do { if ((P).dbg && threadIdx.x == 0) { const unsigned long long n_ = clock64(); dbg_smem()[(k)] += n_ - dbg_smem()[23]; dbg_smem()[23] = n_; } } while (0)
#else
#define FC2A_SET(P) do {} while (0)
#define FC2A(P, k) do {} while (0)
#endif
__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count)); }
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, unsigned bytes) {
#if TP2_DBG_KLOOP == 2
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory"); (void)bytes;
#else
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
#endif
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory"); }
__device__ __forceinline__ void mbar_arrive_cnt(uint64_t* bar, unsigned n) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
    asm volatile("{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_acquire_map(const CUtensorMap* map) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(reinterpret_cast<uint64_t>(map)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
__device__ __forceinline__ void tma_load_2d_row(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
    if (TP2_DBG_KLOOP == 2) return;
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* bar) {
    if (TP2_DBG_KLOOP == 2) return;
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_3d(void* dst, const CUtensorMap* map, int c0, int c1, int c2, uint64_t* bar) {
    if (TP2_DBG_KLOOP == 2) return;
    asm volatile("{ .reg .b64 policy;\n createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n"
                 "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], policy;\n}"
                 :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_4d(void* dst, const CUtensorMap* map, int c0, int c1, int c2, int c3, uint64_t* bar) {
    if (TP2_DBG_KLOOP == 2) return;
    asm volatile("{ .reg .b64 policy;\n createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n"
                 "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5}], [%6], policy;\n}"
                 :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_5d(void* dst, const CUtensorMap* map, int c0, int c1, int c2, int c3, int c4, uint64_t* bar) {
    if (TP2_DBG_KLOOP == 2) return;
    asm volatile("{ .reg .b64 policy;\n createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n"
                 "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], policy;\n}"
                 :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_prefetch_3d(const CUtensorMap* map, int c0, int c1, int c2) {
    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
#ifndef TP2_PROD_REGS
#define TP2_PROD_REGS 64   // producer warpgroup registers after setmaxnreg (512 x CONS + 128 x PROD = 61440 = the launch allocation)
#endif
#ifndef TP2_CONS_REGS
#define TP2_CONS_REGS 104
#endif
static_assert(512 * TP2_CONS_REGS + 128 * TP2_PROD_REGS <= 61440, "register pool");
#ifndef TP2_CLASS_STATS
#define TP2_CLASS_STATS 0   // stamps build: per panel-width k-tile counts / times in slots 16..23 (else: prologue sub-stamps 16..21)
#endif
#ifndef TP2_FAST_SILU
#define TP2_FAST_SILU 1   // SiLU with __expf / __fdividef (its result is BF16-rounded right after)
#endif
#ifndef TP2_GROUPS
#define TP2_GROUPS 1   // 2: FC1 in 2 panel groups (measured worse at M6: +2 partial exchanges per CTA); 1 = one stream-K range (each stream-K over all CTAs): group 0's h is quantized while group 1's FC1 runs
#endif
#ifndef TP2_PIN_LOADS
#define TP2_PIN_LOADS 1   // epilogue scale loads pinned early (asm volatile) instead of sunk next to their use
#endif
#ifndef TP2_DBG_FC2_NOEPI
#define TP2_DBG_FC2_NOEPI 0   // timing diagnostic: FC2 units skip the out-tile conversion / smem stores (wrong results)
#endif
#ifndef TP2_DBG_FC2_NOSTORE
#define TP2_DBG_FC2_NOSTORE 0   // timing diagnostic: the store warps skip the global stores (wrong results)
#endif
#ifndef TP2_DBG_FC2_ONLY
#define TP2_DBG_FC2_ONLY 0   // timing diagnostic: skip FC1 and the h readiness waits (wrong results)
#endif
#ifndef TP2_SMALL_TABLES
#define TP2_SMALL_TABLES 2   // M * topk <= 128: one-warp table build (2: expert token bitmaps, 1: bitonic sort) instead of the block-wide chain
#endif
#ifndef TP2_ONE_N
#define TP2_ONE_N 0   // diagnostic: every panel runs the N32 instantiation (code-size / i-cache probe)
#endif
#ifndef TP2_FC2_L2PF
#define TP2_FC2_L2PF 4   // FC2 units per CTA whose weights the TMA lane L2-prefetches right after issuing its last FC1 tile
#endif
__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint64_t* bar) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(smem_u32(bar)) : "memory"); }
// programmatic dependent launch: everything that reads the previous kernels' outputs (x, xs, shared, routing weights) or this
// kernel's own global workspace (epoch, counters -- the previous launch of this kernel may be the prerequisite grid) follows it
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned ld_relaxed_gpu(const unsigned* p) { unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void fence_acquire_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
// spin with relaxed loads (an ld.acquire per poll invalidates L1 each time: CCTL.IVALL), one acquire fence once satisfied
__device__ __forceinline__ void wait_eq(const unsigned* p, unsigned v) { while (ld_relaxed_gpu(p) != v) {} (void)ld_acquire_gpu(p); }
__device__ __forceinline__ void wait_ge(const unsigned* p, unsigned v) { while (ld_relaxed_gpu(p) < v) {} (void)ld_acquire_gpu(p); }
// loads pinned where they are written (asm volatile keeps program order against the k-step asm blocks): issued early, waited on only at
// their use -- the compiler otherwise sinks them next to the use and exposes the L2 latency
__device__ __forceinline__ float ldv_cg_f32(const float* p) { float v; asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p)); return v; }
__device__ __forceinline__ float ldv_nc_f32(const float* p) { float v; asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p)); return v; }
__device__ __forceinline__ void st_release_gpu(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void red_release_gpu_add(unsigned* p, unsigned v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ unsigned atom_add_acq_rel_gpu(unsigned* p, unsigned v) {
    unsigned r; asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r;
}
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void st_ll(uint4* p, uint4 v) { asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ uint4 ld_ll(const uint4* p) { uint4 v; asm volatile("ld.relaxed.sys.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint16_t pack_e4m3x2(float lo, float hi) { uint16_t r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
// Kimi K3 SiTU-GLU (modeling_kimi_linear.py SituAndMul): b1 tanh(g / b1) sigmoid(g) * b2 tanh(u / b2), fp32, b1 = activation_situ_beta = 4,
// b2 = activation_situ_linear_beta = 25 (the result is BF16-rounded by the caller)
constexpr float SITU_B1 = 4.0f, SITU_B2 = 25.0f;
__device__ __forceinline__ float situ_glu(float g, float u) {
    const float sa = SITU_B1 * tanhf(g * (1.0f / SITU_B1)) * __fdividef(1.0f, 1.0f + __expf(-g));
    return sa * (SITU_B2 * tanhf(u * (1.0f / SITU_B2)));
}
__device__ __forceinline__ uint32_t pack_bf16x2_rn(float lo, float hi) { __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&r); }
template <int N> __device__ __forceinline__ void fence_operand(float (&r)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t a) {   // K-major 128B-swizzled tile: 8-row atoms of 1 KB (SBO 1024)
    return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N)); }
template <int N> __device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N)); }

// ---- MXFP4 -> e4m3 register dequant (Humming fused-e8m0 layout; = optimize_14_codex RAW_PRMT form) ----
__device__ __forceinline__ void make_pools(uint32_t o, uint32_t& lo, uint32_t& hi) {
    lo = 0x0C080000u + o * 0x08080800u;   // codes 0..3 -> {0, 0.5, 1, 1.5} * 2^(o-6) as e4m3 bytes
    hi = 0x1C181410u + o * 0x08080808u;   // codes 4..7 -> {2, 3, 4, 6} * 2^(o-6)
}
__device__ __forceinline__ void dequant_word(uint32_t q, uint32_t lo, uint32_t hi, uint32_t& w_lo, uint32_t& w_hi) {
    uint32_t mag_lo, mag_hi;
    asm volatile("{ .reg .b32 sel, sel_hi, ml, mh;\n"
                 "and.b32 sel, %2, 0x77777777;\n"
                 "shr.u32 sel_hi, sel, 16;\n"
                 "prmt.b32 ml, %3, %4, sel;\n"
                 "prmt.b32 mh, %3, %4, sel_hi;\n"
                 "mov.b32 %0, ml; mov.b32 %1, mh; }"
                 : "=r"(mag_lo), "=r"(mag_hi) : "r"(q), "r"(lo), "r"(hi));
    w_lo = ((q << 4) & 0x80808080u) | mag_lo;
    w_hi = (q & 0x80808080u) | mag_hi;
}
// 4 k32 slices of this warpgroup's 64-row block: A[s] = m64k32 e4m3 A fragment of slice s (slices 4 KB / 256 B apart)
__device__ __forceinline__ void dequant_slices4(const uint8_t* frag, const uint8_t* scl, uint32_t (&A)[4][4]) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const uint32_t o_a = scl[s * 256], o_b = scl[s * 256 + 1];
        uint32_t lo_a, hi_a, lo_b, hi_b;
        make_pools(o_a, lo_a, hi_a);
        make_pools(o_b, lo_b, hi_b);
        const uint2 q = *reinterpret_cast<const uint2*>(frag + s * 4096);
        dequant_word(q.x, lo_a, hi_a, A[s][0], A[s][2]);
        dequant_word(q.y, lo_b, hi_b, A[s][1], A[s][3]);
    }
}

// One k-tile: wait the stage, dequant into A, retire the previous chain (D += C, release its stage), issue this chain.
#ifndef TP2_ROWX
#define TP2_ROWX 0   // (8 measured much slower: small TMAs clog the TMA unit) compact panels of <= this many rows load x by per-row TMA instead of the gathered xg
#endif
#ifndef TP2_FC2_DEFER
#define TP2_FC2_DEFER 0   // (1 measured 1-2 us slower) FC2 epilogue of unit j stored after unit j + 1's first k-step (fc2_flush)
#endif
#ifndef TP2_DBG_TWICE
#define TP2_DBG_TWICE 0
#endif
#ifndef TP2_HF_PREFETCH
#define TP2_HF_PREFETCH 0   // (1: no wait reduction, M64 +3 us) FC2 producer: acquire-load the next unit's panel flag early (see flag_prefetch)
#endif
#ifndef TP2_T1_DYN
#define TP2_T1_DYN 1   // T1 items (row tile, 8 tokens) from a queue in row-tile order (rt_cnt counted per unit by the store warps): the
                       // CTAs leaving FC2 first reduce the row tiles already complete instead of all waiting for the last one
#endif
#ifndef TP2_EARLY_XS
#define TP2_EARLY_XS 1   // dense + fused quantization: warp 19 acquires the token flags (+ stages the owners' scales) during the prologue, so
                         // the producer issues stage 0's x right behind its weights (no flag round trip after the tables)
#endif
#ifndef TP2_EARLY_XQ
#define TP2_EARLY_XQ 1   // compact panels + fused quantization: the producer warpgroup acquires every token flag during the prologue (after
                         // its own quantization), so the consumers start the x gather right after the __syncthreads (no flag round trip)
#endif
#ifndef TP2_EARLY_XQ_MAXM
#define TP2_EARLY_XQ_MAXM 64   // (M 128: +1.5..2.5 us -- the late flags hold the __syncthreads and with it the pass-A weight loads)
#endif
#ifndef TP2_FC2_STAGGER
#define TP2_FC2_STAGGER 0   // cycles: consumer warpgroups 2, 3 enter the FC2 unit loop this much later than 0, 1 (once, after the first unit's
                            // stage landed) so one pair's per-unit drain / epilogue / restart overlaps the other pair's k-steps
#endif
#ifndef TP2_FC2_STAGGER_MINM
#define TP2_FC2_STAGGER_MINM 16
#endif
#ifndef TP2_STREAM_TAIL
#define TP2_STREAM_TAIL 0   // (1 measured much slower at M >= 32: warp-18 ALU slows the RS-wgmma k-loops) T1 by warp 18 during FC2 (row tile by row tile, rt_cnt counted per unit by the store warps)
#endif
#ifndef TP2_REL_FENCES
#define TP2_REL_FENCES 0   // 1: the release helper's explicit fence.acq_rel around the panel count / h flag (v3r form)
#endif
#ifndef TP2_MAP_FENCE
#define TP2_MAP_FENCE 0   // 1: fence.proxy.tensormap acquire before the first use of each weight map
#endif
#ifndef TP2_KPIPE
#define TP2_KPIPE 1   // k-steps: dequant of k32 slice j + 1 under the wgmma of slice j (kpipe.cuh, one asm block; bit-identical)
#endif
template <int N, bool FIRST>
__device__ __forceinline__ void kstep(uint8_t* smem, uint64_t* full, uint64_t* empty, int& s, unsigned& ph, int& sprev,
                                      uint32_t (&A)[4][4], float (&C)[N / 2], float (&D)[N / 2], uint64_t desc,
                                      int frag_off, int scl_off, int lane, bool gather_fence) {
#if defined(TP2_STAMPS) && defined(TP2_TL_FC1)
    const int tl_j = g_tl_j; const bool tl_on = threadIdx.x == 0 && g_tl_fc1;
    if (tl_on) TL1(tl_j, 0);
#endif
    const bool sub0 = KPS == 1 || (s & (KPS - 1)) == 0;   // first k-tile of its ring stage: wait for the stage, release the previous one
    if (sub0) mbar_wait(&full[s / KPS], ph);
#if defined(TP2_STAMPS) && defined(TP2_TL_FC1)
    if (tl_on) TL1(tl_j, 1);
#endif
    if (TP2_GATHER_FENCE && gather_fence) fence_proxy_async_smem();
    const uint8_t* st = smem + s * STAGE;
    if constexpr (TP2_DBG_KLOOP == 1) {
        if (!FIRST && lane == 0 && sub0) mbar_arrive(&empty[sprev / KPS]);
        sprev = s;
        if (++s == NST) { s = 0; ph ^= 1; }
        return;
    }
    if constexpr (TP2_KPIPE) {   // slice-pipelined dequant + wgmma in one asm block (kpipe.cuh)
        kpipe::KPipe<N, FIRST>::run(C, D, smem_u32(st + frag_off), smem_u32(st + scl_off), desc, smem_u32(&empty[sprev / KPS]), lane == 0 && sub0 ? 1u : 0u);
#if defined(TP2_STAMPS) && defined(TP2_TL_FC1)
        if (tl_on) { TL1(tl_j, 2); g_tl_j = tl_j + 1; }
#endif
        sprev = s;
        if (++s == NST) { s = 0; ph ^= 1; }
        return;
    }
    dequant_slices4(st + frag_off, st + scl_off, A);
    wg_wait0();
    fence_operand(C);
    if constexpr (!FIRST) {
        if (lane == 0 && sub0) mbar_arrive(&empty[sprev / KPS]);
#pragma unroll
        for (int i = 0; i < N / 2; ++i) D[i] += C[i];
    }
    fence_operand(D);
    fence_operand(C);
    wg_fence();
    chain::Chain<N>::run(C, A, desc);
    wg_commit();
    sprev = s;
    if (++s == NST) { s = 0; ph ^= 1; }
}
template <int N>
__device__ __forceinline__ void kfinish(uint64_t* empty, int sprev, float (&C)[N / 2], float (&D)[N / 2], int lane) {
    wg_wait0();
    fence_operand(C);
    if (lane == 0) mbar_arrive(&empty[sprev / KPS]);   // (pieces / units end on a stage's last k-tile)
#pragma unroll
    for (int i = 0; i < N / 2; ++i) D[i] += C[i];
    fence_operand(D);
}

// ------------------------------------------------------------------------------------------------ routing tables
struct Tabs {
    uint16_t* ids; uint16_t* ent; uint16_t* uexp; uint16_t* off; uint16_t* pid; uint16_t* p2u; float* fs; float* am; int* misc;
    uint32_t* ccum; int* qq; uint32_t* prow; uint16_t* km; uint32_t* pinf; int* bnd; uint16_t* e2p; float* grw;
};
__device__ __forceinline__ Tabs tabs(uint8_t* smem) {
    uint8_t* t = smem + OFF_TAB;
    return Tabs{reinterpret_cast<uint16_t*>(t + TAB_IDS), reinterpret_cast<uint16_t*>(t + TAB_ENT), reinterpret_cast<uint16_t*>(t + TAB_UEXP),
                reinterpret_cast<uint16_t*>(t + TAB_OFF), reinterpret_cast<uint16_t*>(t + TAB_PID), reinterpret_cast<uint16_t*>(t + TAB_P2U),
                reinterpret_cast<float*>(t + TAB_FS), reinterpret_cast<float*>(t + TAB_AM), reinterpret_cast<int*>(t + TAB_MISC),
                reinterpret_cast<uint32_t*>(t + TAB_CCUM), reinterpret_cast<int*>(t + TAB_QQ), reinterpret_cast<uint32_t*>(t + TAB_PROW),
                reinterpret_cast<uint16_t*>(t + TAB_KM), reinterpret_cast<uint32_t*>(t + TAB_PINF), reinterpret_cast<int*>(t + TAB_BND),
                reinterpret_cast<uint16_t*>(t + TAB_E2P), reinterpret_cast<float*>(t + TAB_GRW)};
}
// Every CTA: union = present experts in ascending id; entries of an expert = its tokens in ascending order (a token listing the
// same expert in several slots is ONE entry; FC2 writes its output to every such slot); panels of <= NMAX entries.
// Small-M tables (M * topk <= 128, top-k <= 8, M <= 128): warp 0 alone, warp-synchronous -- bitonic sort of the (expert, token,
// slot) keys (4 per lane), then prefix counts by ballot.  Produces exactly the tables of build_tables (one panel per expert:
// n <= M <= NMAX here).  The caller's __syncthreads publishes them.
__device__ __forceinline__ void build_tables_small(const Params& P, uint8_t* smem, const Tabs& T) {
    const int lane = threadIdx.x & 31;
    const int M = P.M, topk = P.topk, MT = M * topk;
    uint32_t v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int p = lane + 32 * q;
        uint32_t key = 0xFFFFFFFFu;
        if (p < MT) {
            const int e = __ldcg(P.ids + p), t = p / topk, k = p - t * topk;
            const bool ok = (unsigned)e < (unsigned)P.n_exp;
            T.ids[t * TOPK_MAX + k] = ok ? (uint16_t)e : (uint16_t)0xFFFF;
            if (ok) key = ((uint32_t)e << 11) | ((uint32_t)t << 4) | (uint32_t)k;
        }
        v[q] = key;
    }
    STAMP(P, 16);
    for (int i = lane; i < M * TOPK_MAX; i += 32) if ((i & (TOPK_MAX - 1)) >= topk) T.ids[i] = (uint16_t)0xFFFF;   // row padding
    for (int i = lane; i < (MT + 1) / 2; i += 32) reinterpret_cast<unsigned*>(T.km)[i] = 0u;
    __syncwarp();
    STAMP(P, 17);
    // bitonic sort, element index i = lane + 32 q, ascending
#pragma unroll
    for (int k = 2; k <= 128; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (j >= 32) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int qq = q ^ (j >> 5);
                    if (qq > q) {
                        const int i = lane + 32 * q;
                        const uint32_t a = v[q], b2 = v[qq], lo = min(a, b2), hi = max(a, b2);
                        const bool asc = (i & k) == 0;
                        v[q] = asc ? lo : hi; v[qq] = asc ? hi : lo;
                    }
                }
            } else {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int i = lane + 32 * q;
                    const uint32_t o = __shfl_xor_sync(0xffffffffu, v[q], j);
                    const bool asc = (i & k) == 0, lower = (i & j) == 0;
                    v[q] = (asc == lower) ? min(v[q], o) : max(v[q], o);
                }
            }
        }
    }
    STAMP(P, 18);
    // run structure: previous element (i - 1) of each element
    uint32_t prev[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint32_t up = __shfl_up_sync(0xffffffffu, v[q], 1);
        const uint32_t last_prev_q = __shfl_sync(0xffffffffu, q ? v[q - 1] : 0xFFFFFFFFu, 31);
        prev[q] = lane ? up : (q ? last_prev_q : 0xFFFFFFFFu);
    }
    unsigned ent_before = 0, head_before = 0;   // exclusive prefixes over elements of earlier q blocks
    int ent_idx[4], slot_idx[4];
    bool isent[4], ishead[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const bool valid = v[q] != 0xFFFFFFFFu;
        const bool samet = valid && prev[q] != 0xFFFFFFFFu && (prev[q] >> 4) == (v[q] >> 4);   // same (expert, token): duplicate slot
        const bool newe = valid && (prev[q] == 0xFFFFFFFFu || (prev[q] >> 11) != (v[q] >> 11));
        isent[q] = valid && !samet; ishead[q] = newe;
        const unsigned be = __ballot_sync(0xffffffffu, isent[q]), bh = __ballot_sync(0xffffffffu, newe);
        const unsigned below = (1u << lane) - 1u;
        ent_idx[q] = (int)(ent_before + __popc(be & below)) - (isent[q] ? 0 : 1);   // a duplicate maps to its run's entry
        slot_idx[q] = (int)(head_before + __popc(bh & below)) + (newe ? 0 : -1);
        ent_before += __popc(be); head_before += __popc(bh);
    }
    const int E_tot = (int)ent_before, U = (int)head_before;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        if (v[q] == 0xFFFFFFFFu) continue;
        const int e = (int)(v[q] >> 11), t = (int)((v[q] >> 4) & 127u), k = (int)(v[q] & 15u);
        if (isent[q]) { T.ent[ent_idx[q]] = (uint16_t)t; T.e2p[ent_idx[q]] = (uint16_t)slot_idx[q]; }
        atomicOr(reinterpret_cast<unsigned*>(T.km + (ent_idx[q] & ~1)), (1u << k) << ((ent_idx[q] & 1) * 16));
        if (ishead[q]) { T.uexp[slot_idx[q]] = (uint16_t)e; T.off[slot_idx[q]] = (uint16_t)ent_idx[q]; T.pid[slot_idx[q]] = (uint16_t)slot_idx[q]; T.p2u[slot_idx[q]] = (uint16_t)slot_idx[q]; }
    }
    if (lane == 0) { T.off[U] = (uint16_t)E_tot; T.pid[U] = (uint16_t)U; T.misc[0] = U; T.misc[1] = E_tot; T.misc[2] = U; T.misc[4] = U; T.misc[5] = U; }
    __syncwarp();
    STAMP(P, 19);
    // panels (= slots): info, cost / padded-row prefixes (4 per lane), totals at index U
    uint32_t c[4], np[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int u = lane * 4 + q;   // consecutive panels per lane -> one warp scan of lane sums
        c[q] = 0; np[q] = 0;
        if (u < U) {
            const int o = T.off[u], n = (int)T.off[u + 1] - o;
            T.pinf[u] = (uint32_t)T.uexp[u] | ((uint32_t)n << 10) | ((uint32_t)o << 16);
            const int nb = P.dense ? P.md : n;
            c[q] = panel_cost((nb + 7) / 8 - 1, P.dense);
            np[q] = (uint32_t)((n + 7) & ~7);
        }
    }
    uint32_t sc = c[0] + c[1] + c[2] + c[3], sn = np[0] + np[1] + np[2] + np[3], ic = sc, in = sn;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, ic, o), z = __shfl_up_sync(0xffffffffu, in, o);
        if (lane >= o) { ic += y; in += z; }
    }
    uint32_t bc = ic - sc, bn = in - sn;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int u = lane * 4 + q;
        if (u <= U) { T.ccum[u] = bc; T.prow[u] = bn; }
        bc += c[q]; bn += np[q];
    }
    if (lane == 31) { T.ccum[U] = ic; T.prow[U] = in; }   // totals (also for U == 128)
    __syncwarp();
    STAMP(P, 20);
    // stream-K range starts (group 0 = all panels; group 1 empty): only the window this CTA reads -- its own range and the
    // starts of the CTAs whose partials it may add (an item spans <= 32 CTAs: every CTA has >= 1 k-tile of the >= 128)
    {
        const unsigned G = gridDim.x, total = T.ccum[U];
        const int ncl = P.dense ? (P.md + 7) / 8 - 1 : -1;
        for (unsigned b = blockIdx.x + lane; b <= min(G, blockIdx.x + 33u); b += 32) {
            int r;
            if (b == 0) r = 0;
            else if (b == G || U == 0) r = U * ITEMS * NKT1;
            else {
                const unsigned bcst = (total * b) / G;
                int lo = 0, hi = U - 1;
                while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (T.ccum[mid] <= bcst) lo = mid; else hi = mid - 1; }
                const int cl = ncl >= 0 ? ncl : (int)((((T.pinf[lo] >> 10) & 63u) + 7u) / 8u) - 1;
                r = lo * ITEMS * NKT1 + panel_kt(bcst - T.ccum[lo], cl, P.dense);
            }
            T.bnd[b] = r;
            T.bnd[MAXG + 1 + b] = U * ITEMS * NKT1;
        }
    }
    STAMP(P, 21);
}

// Small-M tables, bitmap form (TP2_SMALL_TABLES 2): warp 0 alone.  Per expert a token bitmask (M <= 32) built with shared
// atomicOr; a warp scan over the 256 experts (8 per lane, ascending) gives union slots, entry offsets and the cost prefix; entries
// are the set bits in ascending token order.  Same tables as build_tables_small (one panel per expert).
__device__ __forceinline__ void build_tables_small2(const Params& P, uint8_t* smem, const Tabs& T) {
    const int lane = threadIdx.x & 31;
    const int M = P.M, topk = P.topk, MT = M * topk;
    uint32_t* tm = reinterpret_cast<uint32_t*>(smem + PRO_BITS);    // [NEXP] token mask per expert
    int* eoff = reinterpret_cast<int*>(smem + PRO_OFFE);             // [NEXP] entry offset of expert (valid if present)
    int* eslot = reinterpret_cast<int*>(smem + PRO_PIDE);            // [NEXP] union slot (= panel) of expert
#pragma unroll
    for (int i = 0; i < NEXP / 32; ++i) tm[lane + 32 * i] = 0u;
    int ev[4];
    const int* gids = reinterpret_cast<const int*>(smem + PRO_PRE);   // fused gate: this CTA's ids
#pragma unroll
    for (int q = 0; q < 4; ++q) { const int p = lane + 32 * q; ev[q] = p < MT ? (P.glog ? gids[p] : __ldcg(P.ids + p)) : -1; }
    STAMP(P, 16);
    for (int i = lane; i < M * TOPK_MAX; i += 32) if ((i & (TOPK_MAX - 1)) >= topk) T.ids[i] = (uint16_t)0xFFFF;   // row padding
    for (int i = lane; i < (MT + (P.shw ? M : 0) + 1) / 2; i += 32) reinterpret_cast<unsigned*>(T.km)[i] = 0u;   // (+ shared entries: no slots)
    __syncwarp();
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int p = lane + 32 * q;
        if (p < MT) {
            const int e = ev[q], t = p / topk, k = p - t * topk;
            const bool ok = (unsigned)e < (unsigned)P.n_exp;
            T.ids[t * TOPK_MAX + k] = ok ? (uint16_t)e : (uint16_t)0xFFFF;
            if (ok) atomicOr(tm + e, 1u << t);
            else ev[q] = -1;
        }
    }
    __syncwarp();
    STAMP(P, 17);
    CKM(P, 32);
    // experts lane * 8 .. + 7: counts, presence, panel costs / padded widths
    constexpr int EPL = NEXP / 32;   // experts per lane
    uint32_t m[EPL];
    int cnt = 0, pres = 0;
    uint32_t cst = 0, npd = 0;
    const uint32_t dpc = P.dense ? panel_cost((P.md + 7) / 8 - 1, true) : 0u;   // dense: every panel costs the same
#pragma unroll
    for (int i = 0; i < EPL; ++i) m[i] = tm[lane * EPL + i];
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
        const int n = __popc(m[i]);
        cnt += n; pres += n > 0;
        if (n) { cst += P.dense ? dpc : panel_cost((n + 7) / 8 - 1, false); npd += (uint32_t)((n + 7) & ~7); }
    }
    int ic = cnt, ip = pres;
    uint32_t icst = cst, inp = npd;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y0 = __shfl_up_sync(0xffffffffu, ic, o), y1 = __shfl_up_sync(0xffffffffu, ip, o);
        const uint32_t y2 = __shfl_up_sync(0xffffffffu, icst, o), y3 = __shfl_up_sync(0xffffffffu, inp, o);
        if (lane >= o) { ic += y0; ip += y1; icst += y2; inp += y3; }
    }
    CKM(P, 33);
    int E = ic - cnt, u = ip - pres;
    uint32_t cc = icst - cst, nn = inp - npd;
    const int E_tot = __shfl_sync(0xffffffffu, ic, 31), U = __shfl_sync(0xffffffffu, ip, 31);
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
        const int n = __popc(m[i]);
        if (!n) continue;
        const int e = lane * EPL + i;
        T.uexp[u] = (uint16_t)e; T.off[u] = (uint16_t)E; T.pid[u] = (uint16_t)u; T.p2u[u] = (uint16_t)u;
        T.pinf[u] = (uint32_t)e | ((uint32_t)n << 10) | ((uint32_t)E << 16);
        T.ccum[u] = cc; T.prow[u] = nn;
        eoff[e] = E; eslot[e] = u;
        cc += P.dense ? dpc : panel_cost((n + 7) / 8 - 1, false); nn += (uint32_t)((n + 7) & ~7);
        E += n; ++u;
    }
    CKM(P, 34);
    const int PT = U + (P.shw ? 1 : 0);   // + the shared expert's panel (M <= 32: one): tokens 0..M-1 as entries E_tot..
    if (lane == 31) {
        T.off[U] = (uint16_t)E_tot; T.pid[U] = (uint16_t)U; T.misc[0] = U; T.misc[1] = E_tot; T.misc[2] = PT; T.misc[4] = PT; T.misc[5] = U;
        T.ccum[U] = cc; T.prow[U] = nn;
        if (P.shw) {
            T.pinf[U] = ((uint32_t)M << 10) | ((uint32_t)E_tot << 16);
            T.ccum[U + 1] = cc + panel_cost(((P.dense ? P.md : M) + 7) / 8 - 1, P.dense, true); T.prow[U + 1] = nn;
        }
    }
    if (P.shw) for (int i = lane; i < M; i += 32) { T.ent[E_tot + i] = (uint16_t)i; T.e2p[E_tot + i] = (uint16_t)U; }
    __syncwarp();
    // entries (rank of the token in its expert's mask) and their slot masks, one id per lane-step (a duplicate expert in several
    // slots of a token: one entry -- written twice with the same values -- and several km bits)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int p = lane + 32 * q;
        if (p < MT && ev[q] >= 0) {
            const int e = ev[q], t = p / topk, k = p - t * topk;
            const int ent = eoff[e] + __popc(tm[e] & ((1u << t) - 1u));
            T.ent[ent] = (uint16_t)t; T.e2p[ent] = (uint16_t)eslot[e];
            atomicOr(reinterpret_cast<unsigned*>(T.km + (ent & ~1)), (1u << k) << ((ent & 1) * 16));
        }
    }
    __syncwarp();
    CKM(P, 35);
    STAMP(P, 20);
    if (P.dense && !P.shw) {   // uniform panel cost, no epilogue term: CTA b starts at k-tile floor(U * 128 * b / G) (= the general form)
        const unsigned G = gridDim.x;
        for (unsigned b = blockIdx.x + lane; b <= min(G, blockIdx.x + 33u); b += 32) {
            T.bnd[b] = (int)(((unsigned)PT * (unsigned)ITEMS * NKT1 * b) / G) & ~(KPS - 1);
            T.bnd[MAXG + 1 + b] = PT * ITEMS * NKT1;
        }
    } else {   // stream-K range starts: this CTA's window (as build_tables_small)
        const unsigned G = gridDim.x, total = T.ccum[PT];
        const int ncl = P.dense ? (P.md + 7) / 8 - 1 : -1;
        for (unsigned b = blockIdx.x + lane; b <= min(G, blockIdx.x + 33u); b += 32) {
            int r;
            if (b == 0) r = 0;
            else if (b == G || PT == 0) r = PT * ITEMS * NKT1;
            else {
                const unsigned bcst = (total * b) / G;
                int lo = 0, hi = PT - 1;
                while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (T.ccum[mid] <= bcst) lo = mid; else hi = mid - 1; }
                const int cl = ncl >= 0 ? ncl : (int)((((T.pinf[lo] >> 10) & 63u) + 7u) / 8u) - 1;
                r = lo * ITEMS * NKT1 + panel_kt(bcst - T.ccum[lo], cl, P.dense, lo >= U);
            }
            T.bnd[b] = r;
            T.bnd[MAXG + 1 + b] = PT * ITEMS * NKT1;
        }
    }
    CKM(P, 36);
    STAMP(P, 21);
}

// Dense small-M tables (P.dense: M <= 16, M * topk <= 128): warps 0..7 (named barrier 2), one expert / one routing entry per thread,
// few distinct instructions (the prologue runs from a cold instruction cache: unrolled per-lane loops cost more than they save).
// Same tables as build_tables_small2.
#ifndef TP2_DENSE_TABLES
#define TP2_DENSE_TABLES 0   // (dense tables assume one expert per thread of 256: not adapted to 896 experts)
#endif
__device__ __forceinline__ void build_tables_dense(const Params& P, uint8_t* smem, const Tabs& T) {
    constexpr int BT = 256;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int M = P.M, topk = P.topk, MT = M * topk;
    uint32_t* tm = reinterpret_cast<uint32_t*>(smem + PRO_BITS);    // [NEXP] token mask per expert
    int* eoff = reinterpret_cast<int*>(smem + PRO_OFFE);
    int* eslot = reinterpret_cast<int*>(smem + PRO_PIDE);
    int* ws = reinterpret_cast<int*>(smem + PRO_WS);                 // [8] presence, [8] entries, [8] padded rows per warp
    int e = -1, t = 0, k = 0;
    if (tid < MT) { e = P.glog ? reinterpret_cast<const int*>(smem + PRO_PRE)[tid] : __ldcg(P.ids + tid); t = tid / topk; k = tid - t * topk; }
    tm[tid] = 0u;
    if (tid < M * TOPK_MAX && (tid & (TOPK_MAX - 1)) >= topk) T.ids[tid] = (uint16_t)0xFFFF;
    if (tid < (MT + (P.shw ? M : 0) + 1) / 2) reinterpret_cast<unsigned*>(T.km)[tid] = 0u;   // (+ shared entries: no slots)
    const bool ok = tid < MT && (unsigned)e < (unsigned)P.n_exp;
    if (tid < MT) T.ids[t * TOPK_MAX + k] = ok ? (uint16_t)e : (uint16_t)0xFFFF;
    named_bar_sync(2, BT);
    if (ok) atomicOr(tm + e, 1u << t);
    named_bar_sync(2, BT);
    const uint32_t m = tm[tid];
    const int n = __popc(m), np = (n + 7) & ~7;
    const unsigned bal = __ballot_sync(0xffffffffu, n > 0);
    int c = n, d = np;   // inclusive warp scans: entries, padded rows
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, c, o), z = __shfl_up_sync(0xffffffffu, d, o);
        if (lane >= o) { c += y; d += z; }
    }
    if (lane == 31) { ws[warp] = __popc(bal); ws[8 + warp] = c; ws[16 + warp] = d; }
    named_bar_sync(2, BT);
    int ub = 0, eb = 0, db = 0, U = 0, Et = 0, Dt = 0;
#pragma unroll 1
    for (int w = 0; w < BT / 32; ++w) {
        const int a = ws[w], b = ws[8 + w], g = ws[16 + w];
        if (w < warp) { ub += a; eb += b; db += g; }
        U += a; Et += b; Dt += g;
    }
    const uint32_t dpc = panel_cost((P.md + 7) / 8 - 1, true);   // dense: every panel costs the same
    if (n) {
        const int u = ub + __popc(bal & ((1u << lane) - 1u)), E = eb + c - n;
        T.uexp[u] = (uint16_t)tid; T.off[u] = (uint16_t)E; T.pid[u] = (uint16_t)u; T.p2u[u] = (uint16_t)u;
        T.pinf[u] = (uint32_t)tid | ((uint32_t)n << 10) | ((uint32_t)E << 16);
        T.ccum[u] = (uint32_t)u * dpc; T.prow[u] = (uint32_t)(db + d - np);
        eoff[tid] = E; eslot[tid] = u;
    }
    // + the shared expert's panel (P.shw) at U, cost-uniform with the dense panels: tokens 0..M-1 as entries Et..
    const int PT = U + (P.shw ? 1 : 0);
    // panel groups (TP2_GROUPS 2: group 0 = panels starting before the cost midpoint, as build_tables; uniform costs: floor(PT / 2) + 1)
    const int pg = (P.groups == 2 && PT > 1 && !P.shw) ? min(PT - 1, max(1, PT / 2 + 1)) : PT;   // (shared expert: one group)
    if (tid == BT - 1) {
        T.off[U] = (uint16_t)Et; T.pid[U] = (uint16_t)U; T.misc[0] = U; T.misc[1] = Et; T.misc[2] = PT; T.misc[4] = pg; T.misc[5] = U;
        T.ccum[U] = (uint32_t)U * dpc; T.prow[U] = (uint32_t)Dt;
        if (P.shw) { T.pinf[U] = ((uint32_t)M << 10) | ((uint32_t)Et << 16); T.ccum[U + 1] = (uint32_t)U * dpc + panel_cost((P.md + 7) / 8 - 1, true, true); T.prow[U + 1] = (uint32_t)Dt; }
    }
    if (P.shw && tid < M) { T.ent[Et + tid] = (uint16_t)tid; T.e2p[Et + tid] = (uint16_t)U; }
    {   // stream-K range starts of this CTA's window per group (uniform panel cost, no epilogue term): p0 * 128 + floor((p1 - p0) * 128 * b / G)
        const unsigned G = gridDim.x, b = blockIdx.x + (unsigned)tid;
        if (tid < 34 && b <= G && !P.shw) {
            T.bnd[b] = (int)(((unsigned)pg * (unsigned)ITEMS * NKT1 * b) / G) & ~(KPS - 1);
            T.bnd[MAXG + 1 + b] = pg * ITEMS * NKT1 + ((int)(((unsigned)(PT - pg) * (unsigned)ITEMS * NKT1 * b) / G) & ~(KPS - 1));
        } else if (tid < 34 && b <= G) {   // shared expert: uniform routed panels then the shared panel (its own cost), one group
            const uint32_t sc = panel_cost((P.md + 7) / 8 - 1, true, true), tot = (uint32_t)U * dpc + sc;
            const uint32_t x = (uint32_t)(((unsigned long long)tot * b) / G);
            int r;
            if (b == G) r = PT * ITEMS * NKT1;
            else if (x < (uint32_t)U * dpc) { const int pnl = (int)(x / dpc); r = pnl * ITEMS * NKT1 + panel_kt(x - (uint32_t)pnl * dpc, (P.md + 7) / 8 - 1, true); }
            else r = U * ITEMS * NKT1 + panel_kt(x - (uint32_t)U * dpc, (P.md + 7) / 8 - 1, true, true);
            T.bnd[b] = r;
            T.bnd[MAXG + 1 + b] = PT * ITEMS * NKT1;
        }
    }
    named_bar_sync(2, BT);
    if (ok) {   // entries: rank of the token in its expert's mask (a duplicate expert of a token: one entry, several km bits)
        const int ent = eoff[e] + __popc(tm[e] & ((1u << t) - 1u));
        T.ent[ent] = (uint16_t)t; T.e2p[ent] = (uint16_t)eslot[e];
        atomicOr(reinterpret_cast<unsigned*>(T.km + (ent & ~1)), (1u << k) << ((ent & 1) * 16));
    }
}

// block-wide tables: the consumer warps only (BT threads, named barrier 1) -- the producer warpgroup quantizes x meanwhile (fused)
__device__ __forceinline__ void build_tables(const Params& P, uint8_t* smem, const Tabs& T) {
    constexpr int BT = NCONS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int M = P.M, topk = P.topk, MT = M * topk, W = (M + 31) >> 5;
    uint32_t* bits = reinterpret_cast<uint32_t*>(smem + PRO_BITS);
    uint16_t* pre = reinterpret_cast<uint16_t*>(smem + PRO_PRE);
    int* offe = reinterpret_cast<int*>(smem + PRO_OFFE);
    int* ws = reinterpret_cast<int*>(smem + PRO_WS);
    int* pide = reinterpret_cast<int*>(smem + PRO_PIDE);
    constexpr int IDR = (MAXM * TOPK_MAX + BT - 1) / BT;   // routing values per thread
    int idr[IDR];
#pragma unroll
    for (int i = 0; i < IDR; ++i) { const int p = tid + i * BT; idr[i] = p < MT ? __ldcg(P.ids + p) : -1; }   // in flight during the zeroing
    STAMP(P, 16);
    for (int i = tid; i < NEXP * W; i += BT) bits[(i / W) * 16 + (i % W)] = 0u;
    named_bar_sync(1, BT);
    STAMP(P, 17);
    for (int i = tid; i < M * TOPK_MAX; i += BT) if ((i & (TOPK_MAX - 1)) >= topk) T.ids[i] = (uint16_t)0xFFFF;   // row padding
    auto put = [&](int p, int e) {
        const bool ok = (unsigned)e < (unsigned)P.n_exp;
        const int t = p / topk;
        T.ids[t * TOPK_MAX + (p - t * topk)] = ok ? (uint16_t)e : (uint16_t)0xFFFF;
        if (ok) atomicOr(&bits[e * 16 + (t >> 5)], 1u << (t & 31));
    };
#pragma unroll
    for (int i = 0; i < IDR; ++i) { const int p = tid + i * BT; if (p < MT) put(p, idr[i]); }
    for (int p = tid + IDR * BT; p < MT; p += BT) put(p, __ldcg(P.ids + p));   // top-k 7..8 at large M
    named_bar_sync(1, BT);
    STAMP(P, 18);
    // expert scan in rounds of BT experts (NEXP 896 > BT): per round inclusive warp scans of (present, count, panels), the warp
    // sums combined across warps and carried over the rounds
    int base_a = 0, base_b = 0, base_d = 0;
#pragma unroll 1
    for (int r = 0; r < (NEXP + BT - 1) / BT; ++r) {
        const int e = tid + r * BT;
        const bool ve = e < NEXP;
        int c = 0, pres = 0, npan = 0;
        if (ve) {
            for (int w = 0; w < W; ++w) { pre[e * 16 + w] = (uint16_t)c; c += __popc(bits[e * 16 + w]); }
            pres = c > 0; npan = (c + NMAX - 1) / NMAX;
        }
        int a = pres, b = c, d = npan;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int ya = __shfl_up_sync(0xffffffffu, a, o), yb = __shfl_up_sync(0xffffffffu, b, o), yd = __shfl_up_sync(0xffffffffu, d, o);
            if (lane >= o) { a += ya; b += yb; d += yd; }
        }
        if (lane == 31) { ws[warp] = a; ws[16 + warp] = b; ws[32 + warp] = d; }
        named_bar_sync(1, BT);
        int xa = base_a + a - pres, xb = base_b + b - c, xd = base_d + d - npan, ta = 0, tb = 0, td = 0;
#pragma unroll
        for (int w = 0; w < BT / 32; ++w) {
            const int sa = ws[w], sb = ws[16 + w], sd = ws[32 + w];   // independent loads
            if (w < warp) { xa += sa; xb += sb; xd += sd; }
            ta += sa; tb += sb; td += sd;
        }
        if (ve) {
            offe[e] = xb; pide[e] = xd;
            if (pres) {
                T.uexp[xa] = (uint16_t)e; T.off[xa] = (uint16_t)xb; T.pid[xa] = (uint16_t)xd;
                for (int p = 0; p < npan; ++p) {
                    T.p2u[xd + p] = (uint16_t)xa;
                    T.pinf[xd + p] = (uint32_t)e | ((uint32_t)min(NMAX, c - p * NMAX) << 10) | ((uint32_t)(xb + p * NMAX) << 16);
                }
            }
        }
        base_a += ta; base_b += tb; base_d += td;
        named_bar_sync(1, BT);   // (ws reused by the next round; tables of this round visible)
    }
    if (tid == 0) {
        const int U = base_a;
        T.off[U] = (uint16_t)base_b; T.pid[U] = (uint16_t)base_d;
        T.misc[0] = U; T.misc[1] = base_b; T.misc[2] = base_d;
    }
    named_bar_sync(1, BT);
    {   // + the shared expert's panels (P.shw): tokens 32 sp .. as entries E_tot + 32 sp .., after the routed panels
        const int P_r = T.misc[2], E_tot = T.misc[1], Ps = P.shw ? (M + NMAX - 1) / NMAX : 0;
        if (tid < Ps) T.pinf[P_r + tid] = ((uint32_t)min(NMAX, M - tid * NMAX) << 10) | ((uint32_t)(E_tot + tid * NMAX) << 16);
        if (Ps) for (int i = tid; i < M; i += BT) { T.ent[E_tot + i] = (uint16_t)i; T.km[E_tot + i] = 0; T.e2p[E_tot + i] = (uint16_t)(P_r + i / NMAX); }
        named_bar_sync(1, BT);
        if (tid == 0) { T.misc[5] = P_r; T.misc[2] = P_r + Ps; }
        named_bar_sync(1, BT);
    }
    STAMP(P, 19);
    const int P_tot = T.misc[2];
    for (int p = tid; p < MT; p += BT) {
        const int t = p / topk, k = p - t * topk;
        const uint4 row0 = *reinterpret_cast<const uint4*>(T.ids + t * TOPK_MAX), row1 = *reinterpret_cast<const uint4*>(T.ids + t * TOPK_MAX + 8);   // the token's 16 slots
        const uint32_t rw[8] = {row0.x, row0.y, row0.z, row0.w, row1.x, row1.y, row1.z, row1.w};
        const int e = (int)((rw[k >> 1] >> ((k & 1) * 16)) & 0xFFFFu);
        if (e == 0xFFFF) continue;
        unsigned km = 0;
#pragma unroll
        for (int k2 = 0; k2 < TOPK_MAX; ++k2) km |= (((rw[k2 >> 1] >> ((k2 & 1) * 16)) & 0xFFFFu) == (unsigned)e ? 1u : 0u) << k2;
        if (km & ((1u << k) - 1u)) continue;   // an earlier slot of this token has the same expert: one entry per (token, expert)
        const int w = t >> 5;
        const int idx = offe[e] + pre[e * 16 + w] + __popc(bits[e * 16 + w] & ((1u << (t & 31)) - 1u));
        T.ent[idx] = (uint16_t)t;
        T.km[idx] = (uint16_t)km;
        T.e2p[idx] = (uint16_t)(pide[e] + (idx - offe[e]) / NMAX);
    }
    // FC1 cost prefix over panels (stream-K balances cost, not k-tiles): panel cost = 128 k-tiles x c(width class)
    {
        uint32_t c = 0, np = 0;
        if (tid < P_tot) {
            const int n = P.dense ? P.md : (int)((T.pinf[tid] >> 10) & 63u);   // B operand width of the panel's FC1
            c = panel_cost((n + 7) / 8 - 1, P.dense, tid >= T.misc[5]);
            np = (uint32_t)((n + 7) & ~7);
        }
        uint32_t a = c, d = np;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffffu, a, o), z = __shfl_up_sync(0xffffffffu, d, o);
            if (lane >= o) { a += y; d += z; }
        }
        if (lane == 31) { ws[24 + warp] = (int)a; ws[48 + warp] = (int)d; }
        named_bar_sync(1, BT);   // also publishes T.ent / T.km
        STAMP(P, 20);
        uint32_t base = 0, based = 0;
#pragma unroll
        for (int w = 0; w < BT / 32; ++w) if (w < warp) { base += (uint32_t)ws[24 + w]; based += (uint32_t)ws[48 + w]; }
        if (tid <= P_tot) { T.ccum[tid] = base + a - c; T.prow[tid] = based + d - np; }   // tid == P_tot: c == np == 0 -> totals
        named_bar_sync(1, BT);
    }
    // panel groups: group 0 = panels starting before the cost midpoint (TP2_GROUPS 2), group 1 = the rest
    {
        const unsigned total = T.ccum[P_tot];
        if (tid == 0 && (P.groups == 1 || P_tot <= 1)) T.misc[4] = P_tot;   // else exactly one panel thread writes the split below
        if (P.groups == 2 && P_tot > 1 && tid < P_tot) {
            const unsigned half = total / 2;
            if (T.ccum[tid] <= half && half < T.ccum[tid + 1]) T.misc[4] = min(P_tot - 1, max(1, tid + 1));
        }
        named_bar_sync(1, BT);
    }
    // stream-K range starts per group and CTA: panel thread pid writes the CTAs b whose group cost boundary falls in its range
    // (32-bit: total <= MAXP * 128 * 200 < 2^24, total * G < 2^32)
    {
        const unsigned G = gridDim.x;
        const int pg = T.misc[4];
        if (tid < 2) { const int p0 = tid ? pg : 0, p1 = tid ? P_tot : pg; T.bnd[tid * (MAXG + 1)] = p0 * ITEMS * NKT1; T.bnd[tid * (MAXG + 1) + G] = p1 * ITEMS * NKT1; }
        if (tid < P_tot) {
            const int g = tid < pg ? 0 : 1;
            const unsigned C0 = T.ccum[g ? pg : 0], C1 = T.ccum[g ? P_tot : pg], tot = C1 - C0;
            const unsigned c0 = T.ccum[tid] - C0, c1 = T.ccum[tid + 1] - C0;
            const int n = P.dense ? P.md : (int)((T.pinf[tid] >> 10) & 63u), cl = (n + 7) / 8 - 1;
            const unsigned b0 = max(1u, (c0 * G + tot - 1) / tot), b1 = min(G, (c1 * G + tot - 1) / tot);   // c0 <= tot*b/G < c1
            for (unsigned b = b0; b < b1; ++b) T.bnd[g * (MAXG + 1) + b] = tid * ITEMS * NKT1 + panel_kt((tot * b) / G - c0, cl, P.dense, tid >= T.misc[5]);
        }
        if (tid > 0 && tid < (int)G) {   // an empty group: every CTA's range is empty
            if (pg == 0) T.bnd[tid] = 0;
            if (pg == P_tot) T.bnd[MAXG + 1 + tid] = P_tot * ITEMS * NKT1;
        }
    }
    STAMP(P, 21);
}

// Fused gate of one token by one warp: sglang's Triton moe_fused_gate (_router_triton_kernel: sqrtsoftplus, top-6, renormalize,
// no output scaling -- the production router since the AOT kernels were retired), bit for bit.  Its PTX: act = sqrt.approx.ftz(x > 20 ?
// x : logf(ex2.approx(x * log2e) + 1)) with libdevice's ftz logf; biased = act + bias (NaN -> -1e30); top-k by repeated max, ties to
// the lowest expert id, taken -> -inf; weights = div.full(act, routed_sum), routed_sum = (((R0 + R1) + R2) + R3) + (R4 + R5).
// Expert e = lane + 32 w here (the selection does not depend on the layout); keys = order-preserving uints (0 = taken).
__device__ __forceinline__ float tri_act(float x) {
    float r;
    asm("{\n\t"
        ".reg .pred ps, pf, pz, pg;\n\t"
        ".reg .b32 a, e, y, y2, ys, eb, t, mb, fe, f, p, q, lg;\n\t"
        "mul.f32 a, %1, 0f3FB8AA3B;\n\t"
        "ex2.approx.f32 e, a;\n\t"
        "add.f32 y, e, 0f3F800000;\n\t"
        "setp.lt.f32 ps, y, 0f00800000;\n\t"
        "mul.f32 y2, y, 0f4B000000;\n\t"
        "selp.f32 ys, y2, y, ps;\n\t"
        "selp.f32 eb, 0fC1B80000, 0f00000000, ps;\n\t"
        "add.s32 t, ys, -1059760811;\n\t"
        "and.b32 t, t, -8388608;\n\t"
        "sub.s32 mb, ys, t;\n\t"
        "cvt.rn.f32.s32 fe, t;\n\t"
        "fma.rn.ftz.f32 fe, fe, 0f34000000, eb;\n\t"
        "add.f32 f, mb, 0fBF800000;\n\t"
        "fma.rn.ftz.f32 p, 0fBE055027, f, 0f3E1039F6;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0fBDF8CDCC;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0f3E0F2955;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0fBE2AD8B9;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0f3E4CED0B;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0fBE7FFF22;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0f3EAAAA78;\n\t"
        "fma.rn.ftz.f32 p, p, f, 0fBF000000;\n\t"
        "mul.f32 q, f, p;\n\t"
        "fma.rn.ftz.f32 q, q, f, f;\n\t"
        "fma.rn.ftz.f32 lg, fe, 0f3F317218, q;\n\t"
        "setp.lt.u32 pf, ys, 2139095040;\n\t"
        "@!pf fma.rn.ftz.f32 lg, ys, 0f7F800000, 0f7F800000;\n\t"
        "setp.eq.f32 pz, ys, 0f00000000;\n\t"
        "selp.f32 lg, 0fFF800000, lg, pz;\n\t"
        "setp.gt.f32 pg, %1, 0f41A00000;\n\t"
        "selp.f32 lg, %1, lg, pg;\n\t"
        "sqrt.approx.ftz.f32 %0, lg;\n\t"
        "}" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ float div_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ uint32_t gate_key(float v) {   // order-preserving, > 0 for every finite value (taken = 0)
    const uint32_t b = __float_as_uint(v == 0.0f ? 0.0f : v);   // -0 == +0
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
constexpr int GATE_TOPK = 6;
__device__ __forceinline__ void gate_cas(uint64_t& a, uint64_t& b) { const uint64_t x = a > b ? a : b, y = a > b ? b : a; a = x; b = y; }
// candidates c = key << 32 | ~id (larger = better: higher score, then lower expert id); each lane sorts its 8 (19-comparator network),
// round k takes the warp's best head (redux max of the keys, then of ~id among the equal keys) and the owner lane pops its head
__device__ __forceinline__ void gate_token(const Params& P, const float* lg, int lane, int& id, float& wt) {
    float sc[8];
    uint64_t h[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const float a = tri_act(__ldcg(lg + lane + 32 * w));
        float v = __fadd_rn(a, __ldg(P.gbias + lane + 32 * w));
        v = v == v ? v : -1e30f;
        sc[w] = a;
        h[w] = ((uint64_t)gate_key(v) << 32) | (uint64_t)(0xFFFFFFFFu - (uint32_t)(lane + 32 * w));
    }
    gate_cas(h[0], h[2]); gate_cas(h[1], h[3]); gate_cas(h[4], h[6]); gate_cas(h[5], h[7]);
    gate_cas(h[0], h[4]); gate_cas(h[1], h[5]); gate_cas(h[2], h[6]); gate_cas(h[3], h[7]);
    gate_cas(h[0], h[1]); gate_cas(h[2], h[3]); gate_cas(h[4], h[5]); gate_cas(h[6], h[7]);
    gate_cas(h[2], h[4]); gate_cas(h[3], h[5]);
    gate_cas(h[1], h[4]); gate_cas(h[3], h[6]);
    gate_cas(h[1], h[2]); gate_cas(h[3], h[4]); gate_cas(h[5], h[6]);
    int ex[GATE_TOPK];
#pragma unroll
    for (int k = 0; k < GATE_TOPK; ++k) {
        const uint32_t hk = (uint32_t)(h[0] >> 32), hl = (uint32_t)h[0];
        const uint32_t gm = __reduce_max_sync(0xffffffffu, hk);
        const uint32_t gl = __reduce_max_sync(0xffffffffu, hk == gm ? hl : 0u);
        ex[k] = (int)(0xFFFFFFFFu - gl);
        if (hk == gm && hl == gl) {
#pragma unroll
            for (int i = 0; i < 7; ++i) h[i] = h[i + 1];
            h[7] = 0ull;
        }
    }
    float R[GATE_TOPK];   // the selected experts' activations (owner lane ex & 31, slot ex >> 5)
#pragma unroll
    for (int k = 0; k < GATE_TOPK; ++k) {
        const int ew = ex[k] >> 5;
        float sv = sc[0];
#pragma unroll
        for (int w = 1; w < 8; ++w) sv = ew == w ? sc[w] : sv;
        R[k] = __shfl_sync(0xffffffffu, sv, ex[k] & 31);
    }
    const float sum = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(R[0], R[1]), R[2]), R[3]), __fadd_rn(R[4], R[5]));
    const float norm = sum > 0.0f ? sum : 1.0f;
    float r = R[0];
    id = ex[0];
#pragma unroll
    for (int k = 1; k < GATE_TOPK; ++k) { r = lane == k ? R[k] : r; id = lane == k ? ex[k] : id; }
    wt = div_full(r, norm);
}

// dsv3_router_gemm<kNumTokens = M <= 16> bit for bit (128 threads, 8 bf16 per thread and k-iteration, 4 k-iterations of 1024): thread
// j of the expert's warpgroup chains fma over k = 8 j + 1024 ki + i (ki outer, i inner), then 32-lane xor-tree sums -> red [m][warp];
// the caller adds 0 + warp 0 + .. + warp 3.  wq: thread j's own weights (cp.async before the PDL wait)
__device__ __forceinline__ void router_partial(const Params& P, int j, const uint4* wq, float* red) {
    const int M = P.M, lane = j & 31;
    asm volatile("cp.async.wait_all;" ::: "memory");
    const __nv_bfloat16* xj = P.xb + j * 8;
#pragma unroll 1
    for (int mc = 0; mc < M; mc += 2) {   // 2 tokens at a time (each token's chain is k-ordered whatever the grouping)
        float acc[2] = {0.f, 0.f};
#pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            const uint4 wv = wq[ki];
            const uint32_t w4[4] = {wv.x, wv.y, wv.z, wv.w};
            uint4 a[2];
#pragma unroll
            for (int u = 0; u < 2; ++u) a[u] = mc + u < M ? __ldg(reinterpret_cast<const uint4*>(xj + (size_t)(mc + u) * DIM + ki * 1024)) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const uint32_t a4[4] = {a[u].x, a[u].y, a[u].z, a[u].w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[u] = __fmaf_rn(bf16lo(a4[i]), bf16lo(w4[i]), acc[u]);
                    acc[u] = __fmaf_rn(bf16hi(a4[i]), bf16hi(w4[i]), acc[u]);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            float v = acc[u];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && mc + u < M) red[(mc + u) * 4 + (j >> 5)] = v;
        }
    }
}

// Fused input quantization of token t by threads [t0, t0 + 256) (named barrier 7): Humming quant_input float8e4m3, bit for bit --
// absmax = max(max|x|, 1e-30), scale = absmax * fp32(1/448) (Triton's "/ 448"), inv = 1 / scale (div.rn), q = e4m3 RN satfinite(x * inv)
template <int NQ>   // NQ threads (named barrier 7), ceil(DIM / 8 / NQ) 16-B chunks each (chunk index < DIM / 8: DIM / 8 = 448 is not a multiple of NQ)
__device__ __forceinline__ void quant_token(const Params& P, int t, int lt, float* red) {
    constexpr int NCH = DIM / 8, CH = (NCH + NQ - 1) / NQ;
    const uint4* src = reinterpret_cast<const uint4*>(P.xb + (size_t)t * DIM);
    uint32_t w[CH][4];
#pragma unroll
    for (int c = 0; c < CH; ++c) {
        const uint4 v = lt + c * NQ < NCH ? __ldg(src + lt + c * NQ) : make_uint4(0u, 0u, 0u, 0u);
        w[c][0] = v.x; w[c][1] = v.y; w[c][2] = v.z; w[c][3] = v.w;
    }
    float m = 0.f;
#pragma unroll
    for (int c = 0; c < CH; ++c)
#pragma unroll
        for (int i = 0; i < 4; ++i) m = fmaxf(m, fmaxf(fabsf(bf16lo(w[c][i])), fabsf(bf16hi(w[c][i]))));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((lt & 31) == 0) red[lt >> 5] = m;
    named_bar_sync(7, NQ);
    m = red[0];
#pragma unroll
    for (int i = 1; i < NQ / 32; ++i) m = fmaxf(m, red[i]);
    const float scale = __fmul_rn(fmaxf(m, 1e-30f), 0.0022321429569274187f), inv = __fdiv_rn(1.0f, scale);   // absmax / 448 compiles to x * (1/448)
    uint8_t* dst = const_cast<uint8_t*>(P.xq) + (size_t)t * DIM;
#pragma unroll
    for (int c = 0; c < CH; ++c) {
        if (lt + c * NQ >= NCH) break;
        uint32_t q[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t a = w[c][2 * i], b = w[c][2 * i + 1];
            const uint32_t lo = pack_e4m3x2(__fmul_rn(bf16lo(a), inv), __fmul_rn(bf16hi(a), inv));
            const uint32_t hi = pack_e4m3x2(__fmul_rn(bf16lo(b), inv), __fmul_rn(bf16hi(b), inv));
            q[i] = lo | (hi << 16);
        }
        *reinterpret_cast<uint2*>(dst + (lt + c * NQ) * 8) = make_uint2(q[0], q[1]);
    }
    if (lt == 0) const_cast<float*>(P.xs)[t] = scale;
    if (TP2_SHARED && P.shw) {   // + the shared expert's input: per (token, 128-group) quantization, chunk j = lt + c NQ of group j >> 4 (16 lanes)
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            const int j = lt + c * NQ;
            float f[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) { f[2 * i] = bf16lo(w[c][i]); f[2 * i + 1] = bf16hi(w[c][i]); }
            float gm = shnum::gq_amax_init();
#pragma unroll
            for (int i = 0; i < 8; ++i) gm = shnum::absmax_ftz(gm, f[i]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) gm = shnum::max_ftz(gm, __shfl_xor_sync(0xffffffffu, gm, o));
            const float mult = shnum::gq_mult(gm);
            const uint32_t q0 = (uint32_t)shnum::gq_pack2(f[0], f[1], mult) | ((uint32_t)shnum::gq_pack2(f[2], f[3], mult) << 16);
            const uint32_t q1 = (uint32_t)shnum::gq_pack2(f[4], f[5], mult) | ((uint32_t)shnum::gq_pack2(f[6], f[7], mult) << 16);
            *reinterpret_cast<uint2*>(P.xqg + (size_t)t * DIM + j * 8) = make_uint2(q0, q1);
            if ((j & 15) == 0) P.xsT[(j >> 4) * MAXM + t] = shnum::gq_scale(gm);
        }
    }
    asm volatile("fence.proxy.async.global;" ::: "memory");   // x is read by TMA / bulk copies (async proxy)
    named_bar_sync(7, NQ);
    if (lt == 0) { fence_acq_rel_gpu(); st_release_gpu(P.xflag + t, P.hseq); }
}
// CTA-local dense quantization (P.lquant): warp w8 = warp - 8 (warps 8..15) quantizes tokens w8, w8 + 8 (< md) into this CTA's private
// rows xq[b * 16 + t] (rows M..md-1 zero: the MMA's padded columns), scale -> TAB_XS[t]; Humming quant_input bit for bit (as quant_token:
// amax over the 4096 values, scale = absmax * fp32(1/448), inv = 1 / scale (div.rn), RN satfinite e4m3).  In the prologue: the
// producer's x TMAs and the owners' TAB_XS reads follow its __syncthreads (no flag, no mbarrier)
__device__ __forceinline__ void quant_local(const Params& P, uint8_t* smem, int w8, int lane) {
    for (int t = w8; t < P.md; t += 8) {
        uint8_t* dst = const_cast<uint8_t*>(P.xq) + ((size_t)blockIdx.x * 16 + t) * DIM;
        if (t >= P.M) {
#pragma unroll 1
            for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(dst + i * 512 + lane * 16) = make_uint4(0u, 0u, 0u, 0u);
            continue;
        }
        const uint4* src = reinterpret_cast<const uint4*>(P.xb + (size_t)t * DIM);
        float m = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {   // pass 1: absmax, all 16 loads in flight (they stay in L1 for pass 2)
            const uint4 a = __ldg(src + i * 64 + lane * 2), b = __ldg(src + i * 64 + lane * 2 + 1);
            const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
            for (int j = 0; j < 8; ++j) m = fmaxf(m, fmaxf(fabsf(bf16lo(w[j])), fabsf(bf16hi(w[j]))));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float scale = __fmul_rn(fmaxf(m, 1e-30f), 0.0022321429569274187f), inv = __fdiv_rn(1.0f, scale);
#pragma unroll
        for (int i = 0; i < 8; ++i) {   // pass 2: 16 values per lane -> 16 e4m3 bytes
            const uint4 a = __ldg(src + i * 64 + lane * 2), b = __ldg(src + i * 64 + lane * 2 + 1);
            const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            uint32_t q[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                q[j] = (uint32_t)pack_e4m3x2(__fmul_rn(bf16lo(w[2 * j]), inv), __fmul_rn(bf16hi(w[2 * j]), inv)) |
                       ((uint32_t)pack_e4m3x2(__fmul_rn(bf16lo(w[2 * j + 1]), inv), __fmul_rn(bf16hi(w[2 * j + 1]), inv)) << 16);
            *reinterpret_cast<uint4*>(dst + i * 512 + lane * 16) = make_uint4(q[0], q[1], q[2], q[3]);
        }
        if (lane == 0) reinterpret_cast<float*>(smem + OFF_TAB + TAB_XS)[t] = scale;
    }
    asm volatile("fence.proxy.async.global;" ::: "memory");   // read back by this CTA's x TMA (async proxy)
}

// wait until tokens [0, M) are quantized: thread lt of nthr polls tokens lt, lt + nthr, .. with all its loads in flight together
// (the caller orders the other threads after it)
__device__ __forceinline__ void wait_tokens(const Params& P, int lt, int nthr) {
    for (;;) {
        bool all = true;
        for (int t0 = lt; t0 < P.M; t0 += 16 * nthr) {
            unsigned v[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) { const int t = t0 + i * nthr; v[i] = t < P.M ? ld_relaxed_gpu(P.xflag + t) : P.hseq; }
#pragma unroll
            for (int i = 0; i < 16; ++i) all &= v[i] == P.hseq;
        }
        if (all) break;
    }
    fence_acq_rel_gpu();
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

// fused routing, consumer warps (tid < NCONS; named barrier 1).  gw (M <= 16): router GEMM of experts b + g G (consumer warpgroup g)
// for all tokens -> P.glog workspace, grid barrier (P.gcnt).  Small tables: every CTA gates all tokens (warp w: w, w + 16, ..) -> ids at
// PRO_PRE, weights at T.grw (CTA 0 also -> P.ids / P.rw).  Block-wide: token t by CTA t % G -> P.ids / P.rw + flag, all CTAs wait for all.
__device__ __forceinline__ void route_prologue(const Params& P, uint8_t* smem, const Tabs& T, bool small, int nthr, int bar) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, M = P.M, topk = P.topk;
    if (P.gw) {
        const int re = blockIdx.x + (tid >> 7) * gridDim.x, j = tid & 127;   // expert of consumer warpgroup tid >> 7
        float* red = reinterpret_cast<float*>(smem + PRO_CNT) + (tid >> 7) * 64;
        if (re < NEXP) router_partial(P, j, reinterpret_cast<const uint4*>(smem + PRO_GW) + tid * 4, red);
        STAMP(P, 37);
        named_bar_sync(1, NCONS);
        if (re < NEXP && j < M) {
            float f = 0.0f;
#pragma unroll
            for (int w = 0; w < 4; ++w) f += red[j * 4 + w];
            const_cast<float*>(P.glog)[(size_t)j * NEXP + re] = f;
        }
        named_bar_sync(1, NCONS);
        if (tid == 0) {
            unsigned* c = P.gcnt + ((__ldcg(P.work) + 1u) & 1u);
            fence_acq_rel_gpu();
            atomicAdd(c, 1u);
            while (ld_relaxed_gpu(c) < gridDim.x) {}
            fence_acq_rel_gpu();
        }
        named_bar_sync(1, NCONS);
        STAMP(P, 38);
    }
    if (small) {
        int* gids = reinterpret_cast<int*>(smem + PRO_PRE);
        for (int t = warp; t < M; t += nthr / 32) {
            int id; float w;
            gate_token(P, P.glog + (size_t)t * NEXP, lane, id, w);
            if (lane < topk) {
                gids[t * topk + lane] = id; T.grw[t * topk + lane] = w;
                if (blockIdx.x == 0) { const_cast<int*>(P.ids)[t * topk + lane] = id; const_cast<float*>(P.rw)[t * topk + lane] = w; }
            }
        }
    } else {
        // (gated-token counter P.gcnt[par] -- the router-GEMM barrier counter, unused here -- and ONE poller per CTA: thousands of
        // spinning pollers slow the very stores / atomics they wait for)
        unsigned* c = P.gcnt + ((__ldcg(P.work) + 1u) & 1u);
        for (int t = blockIdx.x + warp * gridDim.x; t < M; t += (NCONS / 32) * gridDim.x) {
            int id; float w;
            gate_token(P, P.glog + (size_t)t * NEXP, lane, id, w);
            if (lane < topk) { const_cast<int*>(P.ids)[t * topk + lane] = id; const_cast<float*>(P.rw)[t * topk + lane] = w; }
            __syncwarp();
            if (lane == 0) { fence_acq_rel_gpu(); atomicAdd(c, 1u); }
        }
        if (tid == 0) {
            while (ld_relaxed_gpu(c) < (unsigned)M) {}
            fence_acq_rel_gpu();
        }
    }
    named_bar_sync(bar, nthr);
}

// ------------------------------------------------------------------------------------------------ work lists
struct Item { int pid, q, u, e, ebase, n; };
__device__ __forceinline__ Item item_of(const Tabs& T, int pid, int q) {
    Item r;
    const uint32_t v = T.pinf[pid];
    r.pid = pid; r.q = q; r.u = 0; r.e = (int)(v & 1023u); r.n = (int)((v >> 10) & 63u); r.ebase = (int)(v >> 16);
    return r;
}
constexpr int PART_F = NMAX / 2;   // fp32 partial floats per consumer thread and slot
// fp32 partial slot (stream-K): float4 k of consumer thread tid at [(slot * PART_F / 4 + k) * NCONS + tid] (warp accesses coalesced)
__device__ __forceinline__ float4* part_f4(const Params& P, int slot, int k, int tid) {
    return reinterpret_cast<float4*>(P.part) + ((size_t)slot * (PART_F / 4) + k) * NCONS + tid;
}
// k-tile index (global, item-major) where CTA b's FC1 range starts (precomputed in the prologue: cost boundary total*b/G)
__device__ __forceinline__ int fc1_lo(const Tabs& T, int g, int b) { return T.bnd[g * (MAXG + 1) + b]; }
// x gather: C = entries * NKT1 * 8 16-B chunks (entry-major: [kt NKT1][chunk 8]) split evenly: CTA b gathers [b C / G, (b + 1) C / G)
__device__ __forceinline__ unsigned gather_owner(unsigned j, unsigned C, unsigned G) { return ((j + 1) * G + C - 1) / C - 1; }
__device__ __forceinline__ unsigned gather_contrib(const Item& I, unsigned C, unsigned G) {   // CTAs that gather panel I
    const unsigned s0 = (unsigned)I.ebase * (unsigned)(NKT1 * 8), s1 = (unsigned)(I.ebase + I.n) * (unsigned)(NKT1 * 8);
    return gather_owner(s1 - 1, C, G) - gather_owner(s0, C, G) + 1u;
}
// FC2 unit v -> (row tile, panel): group 0's panels first (their h is ready earliest), row tile major inside a group
__device__ __forceinline__ void unit_of(const Tabs& T, int v, int& rtile, int& pid) {
    const int pg = T.misc[4], P_tot = T.misc[2];
    if (v < pg * NRT) { rtile = v / pg; pid = v - rtile * pg; }
    else { const int n1 = P_tot - pg, v2 = v - pg * NRT; rtile = v2 / n1; pid = pg + v2 - rtile * n1; }
}

// The producer lane (warp 16, lane 0): per stage one expect_tx for weights + B operand, the w13/w2 weight + offset TMAs and
// one bulk copy of the B operand (FC1: the panel's gathered x rows of k-tile kt from xg; FC2: the panel's quantized h rows).
// FC1: the first stages' weights are issued before the grid-wide x gather is waited for.  FC2 units are taken from a global
// queue (work stealing); the unit id travels to the consumers through s_unit (written before the unit's first stage arrive).
__device__ __forceinline__ void producer(const Params& P, uint8_t* smem, uint64_t* full, uint64_t* empty, const Tabs& T, int* s_unit,
                                         uint64_t* gbar, bool early_xs) {
    const int G = gridDim.x, b = blockIdx.x;
    const int P_tot = T.misc[2];
    int s = 0; unsigned ph = 0;
    // the maps were written by a host copy before this launch: no tensormap-proxy acquire fence needed, just the prefetch
    for (int i = 0; i < 4; ++i) {
        if (TP2_MAP_FENCE) tma_acquire_map(P.tmaps + i);
        else asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(P.tmaps + i)) : "memory");
    }
    // pass A: weights of the first <= NST FC1 stages (no dependency on the x gather / the previous kernels)
    int nA = 0;
    int kts[NST];
    const int P_r = T.misc[5];   // first shared-expert panel (P_tot without the shared expert)
    bool stopA = false;
    for (int g = 0; g < (TP2_DBG_FC2_ONLY ? 0 : 2) && !stopA; ++g) {
    const int lo = fc1_lo(T, g, b), hi = fc1_lo(T, g, b + 1);
    for (int gk = lo; gk < hi && nA < NST;) {
        const int it = gk / NKT1, kt0 = gk % NKT1, kt1 = min(NKT1, hi - (it * NKT1));
        const Item I = item_of(T, it / ITEMS, it % ITEMS);
        if (TP2_SHARED && I.pid >= P_r) { stopA = true; break; }   // shared expert: issued whole in pass B
        const int np = P.dense ? P.md : (I.n <= TP2_ROWX ? I.n : (I.n + 7) & ~7);   // row mode: only the panel's n rows are loaded
        for (int kt = kt0; kt < kt1 && nA < NST; ++kt, ++nA) {
            uint8_t* st = smem + nA * STAGE;
            TL1(nA, 3);
            if (nA == 0) STAMPT(P, 38);   // (stamps) stage 0 weights issued
            if ((nA & (KPS - 1)) == 0) mbar_arrive_expect_tx(&full[nA / KPS], (unsigned)(KPS * (STAGE_WS + np * 128)));   // (a stage = KPS k-tiles of one item)
            const int row = I.e * (NKT1 * 4) + kt * 4;
            if (P.dense && nA == 1) {   // the TMA unit works in issue order: stage 0's x goes right behind its weights
                griddep_wait();          // (PDL: the previous kernel -- shorter than the table build -- has usually finished)
                if (P.xb && !P.lquant) {   // (local quantization: done before the prologue's __syncthreads)
                    if (early_xs) asm volatile("fence.proxy.async.global;" ::: "memory");   // (flags acquired by warp 19 before the __syncthreads)
                    else wait_tokens(P, 0, 1);
                }
                STAMPT(P, 39);   // (stamps) dense: x of stage 0 issued
                TL1(0, 5); tma_load_2d(smem + STAGE_WS, &P.xmap, kts[0] * 128, P.lquant ? blockIdx.x * 16 : 0, &full[0]);
            }
            tma_load_5d(st, P.tmaps + 0, 0, 0, I.q, 0, row, &full[nA / KPS]);
            tma_load_4d(st + STAGE_W, P.tmaps + 1, 0, I.q, 0, row, &full[nA / KPS]);
            kts[nA] = kt;
            if (P.dense && nA > 0) { TL1(nA, 5); tma_load_2d(st + STAGE_WS, &P.xmap, kt * 128, P.lquant ? blockIdx.x * 16 : 0, &full[nA / KPS]); }
        }
        gk = (it + 1) * NKT1;
    }
    }
    griddep_wait();   // pass A read weights only (dense: and x after stage 0); the workspace from here on
    if (P.xb && P.dense && nA <= 1 && !P.lquant) {
        if (early_xs) asm volatile("fence.proxy.async.global;" ::: "memory");
        else wait_tokens(P, 0, 1);
    }   // (dense with nA >= 2 waited above; compact panels read xg -- gathered after the wait)
    if (P.dense && nA == 1) { TL1(0, 5); tma_load_2d(smem + STAGE_WS, &P.xmap, kts[0] * 128, P.lquant ? blockIdx.x * 16 : 0, &full[0]); }
    const unsigned epoch = __ldcg(P.work) + 1u, par = epoch & 1u;
    int n = 0, ready_pid = -1;
    bool rows_ok = false;
    const unsigned C = (unsigned)T.misc[1] * (unsigned)(NKT1 * 8);
    for (int g = 0; g < (TP2_DBG_FC2_ONLY ? 0 : 2); ++g) {
    const int lo = fc1_lo(T, g, b), hi = fc1_lo(T, g, b + 1);
    for (int gk = lo; gk < hi;) {
        const int it = gk / NKT1, kt0 = gk % NKT1, kt1 = min(NKT1, hi - (it * NKT1));
        const Item I = item_of(T, it / ITEMS, it % ITEMS);
        if (TP2_SHARED && I.pid >= P_r) {   // shared expert item (gate rows 128 q.., up rows 512 + 128 q..): per k-block two stages (gate, up), each
                              // 128 weight rows x K128 + the panel's group-quantized x tile + its 32 k-block scales
            if (P.xb && !rows_ok) { wait_tokens(P, 0, 1); rows_ok = true; }
            const int t0 = (I.pid - P_r) * NMAX;
            for (int kt = kt0; kt < kt1; ++kt) {   // slot 0: gate rows + x tile + scales, slot 1: up rows
                const int s0 = s; mbar_wait(&empty[s], ph ^ 1); if (++s == NST) { s = 0; ph ^= 1; }
                const int s1 = s; mbar_wait(&empty[s], ph ^ 1); if (++s == NST) { s = 0; ph ^= 1; }
                uint8_t* st0 = smem + s0 * STAGE;
                mbar_arrive_expect_tx(&full[s0], (unsigned)(STAGE_W + NMAX * 128 + NMAX * 4));
                mbar_arrive_expect_tx(&full[s1], (unsigned)STAGE_W);
                tma_load_2d(st0, &P.w1s, kt * 128, I.q * 128, &full[s0]);
                tma_load_2d(smem + s1 * STAGE, &P.w1s, kt * 128, INTER + I.q * 128, &full[s1]);
                tma_load_2d(st0 + STAGE_WS, &P.xgs, kt * 128, t0, &full[s0]);
                bulk_g2s(st0 + STAGE_W, P.xsT + (size_t)kt * MAXM + t0, (unsigned)(NMAX * 4), &full[s0]);
                ++n;
            }
            gk = (it + 1) * NKT1;
            continue;
        }
        const bool rowx = !P.dense && I.n <= TP2_ROWX;   // x rows straight from xq by per-row TMA (no gather)
        const int np = P.dense ? P.md : (rowx ? I.n : (I.n + 7) & ~7);
        const uint8_t* xsrc = P.xg + (size_t)T.prow[I.pid] * DIM;
        if (rowx && !rows_ok) {   // fused quantization: x is written in this launch
            if (P.xb) wait_tokens(P, 0, 1);
            rows_ok = true;
        }
        if (!P.dense && !rowx && I.pid != ready_pid) {   // the CTAs that gather this panel's x rows are done
            const unsigned need = gather_contrib(I, C, (unsigned)G);
            while (ld_relaxed_gpu(P.xg_cnt + par * MAXP + I.pid) < need) {}
            (void)ld_acquire_gpu(P.xg_cnt + par * MAXP + I.pid);
            asm volatile("fence.proxy.async.global;" ::: "memory");
            ready_pid = I.pid;
        }
        for (int kt = kt0; kt < kt1; ++kt, ++n) {
            uint8_t* st = smem + s * STAGE;
            uint64_t* fb = &full[s / KPS];
            if (n >= nA) {   // pass B: full stage
                TL1(n, 4);
                if ((s & (KPS - 1)) == 0) {
                    mbar_wait(&empty[s / KPS], ph ^ 1);
                    TL1(n, 3);
                    mbar_arrive_expect_tx(fb, (unsigned)(KPS * (STAGE_WS + np * 128)));
                }
                const int row = I.e * (NKT1 * 4) + kt * 4;
                tma_load_5d(st, P.tmaps + 0, 0, 0, I.q, 0, row, fb);
                tma_load_4d(st + STAGE_W, P.tmaps + 1, 0, I.q, 0, row, fb);
            }
            if (P.dense) { if (n >= nA) { TL1(n, 5); tma_load_2d(st + STAGE_WS, &P.xmap, kt * 128, P.lquant ? blockIdx.x * 16 : 0, fb); } }
            else if (rowx) { for (int r = 0; r < I.n; ++r) tma_load_2d_row(st + STAGE_WS + r * 128, &P.xrow, kt * 128, T.ent[I.ebase + r], fb); }
            else bulk_g2s(st + STAGE_WS, xsrc + (size_t)kt * np * 128, (unsigned)(np * 128), fb);
            if (++s == NST) { s = 0; ph ^= 1; }
        }
        gk = (it + 1) * NKT1;
    }
    }
    const int V2 = P_tot * NRT;
    if constexpr (TP2_FC2_L2PF > 0) {   // keep HBM busy across the FC1 -> FC2 transition: the units of the static schedule
        for (int k = 0, v = b; v < V2 && k < TP2_FC2_L2PF; v += G, ++k) {
            int rtile, upid;
            unit_of(T, v, rtile, upid);
            if (upid >= P_r) continue;   // (shared expert units: other weights)
            const Item I = item_of(T, upid, 0);
            for (int kt = 0; kt < NKT2; ++kt) {
                const int sl = I.e * (INTER / 32) + kt * 4;
                tma_prefetch_3d(P.tmaps + 2, 0, rtile * 4, sl);
                tma_prefetch_3d(P.tmaps + 3, 0, rtile, sl);
            }
        }
    }
    unsigned* pready = reinterpret_cast<unsigned*>(T.qq);   // panel-ready bit set (smem, producer lane only)
    for (int i = 0; i < MAXP / 32; ++i) pready[i] = 0u;
    int vnext = (int)atomicAdd(P.fc2_next + par, 1u);
    // the next unit's panel flag is loaded (acquire) right after this unit's loads are issued, so its round trip overlaps the
    // producer's empty-slot waits instead of delaying the next unit's first TMA (most units are a new panel for this CTA)
    unsigned hf_next = 0u;
    auto flag_prefetch = [&]() {
        hf_next = 0u;
        if (!TP2_HF_PREFETCH || vnext >= V2) return;
        int rt2, p2;
        unit_of(T, vnext, rt2, p2);
        if (!((pready[p2 >> 5] >> (p2 & 31)) & 1u)) hf_next = ld_acquire_gpu(P.h_flags + p2);
    };
    // shared expert units whose panel is not quantized yet are deferred (not waited for: up to 2 x NRT units could otherwise idle their
    // SMs until the shared FC1 -- the last panels of the stream-K order -- completes) and issued as soon as their panel is ready
    auto sh_ready = [&](int pid) -> bool {
        if ((pready[pid >> 5] >> (pid & 31)) & 1u) return true;
        if (ld_relaxed_gpu(P.h_flags + pid) != epoch) return false;
        (void)ld_acquire_gpu(P.h_flags + pid);
        asm volatile("fence.proxy.async.global;" ::: "memory");
        pready[pid >> 5] |= 1u << (pid & 31);
        return true;
    };
    int dq[TP2_SH_DEFER > 0 ? TP2_SH_DEFER : 1], nd = 0;
    for (int j = 0;; ++j) {
        int v = -1;
        for (int i = 0; i < nd; ++i) {   // a deferred unit that became ready goes first
            int rt2, p2;
            unit_of(T, dq[i], rt2, p2);
            if (sh_ready(p2)) { v = dq[i]; dq[i] = dq[--nd]; break; }
        }
        while (v < 0 && vnext < V2) {
            const int vv = vnext;
            vnext = (int)atomicAdd(P.fc2_next + par, 1u);   // the next unit's grab is in flight during this one
            int rt2, p2;
            unit_of(T, vv, rt2, p2);
            if (TP2_SHARED && p2 >= P_r && nd < TP2_SH_DEFER && !sh_ready(p2)) dq[nd++] = vv;
            else v = vv;
        }
        if (v < 0 && nd) v = dq[--nd];   // queue exhausted: the remaining deferred units (wait for their panel below)
        const unsigned hf = hf_next;
        int2* su = reinterpret_cast<int2*>(s_unit) + (j & 7);
        if (v < 0) {   // sentinel: consumers read -1 at the next stage
            mbar_wait(&empty[s / KPS], ph ^ 1);   // (s is stage-aligned: every piece / unit is)
            *reinterpret_cast<volatile int*>(&su->x) = -1;
            mbar_arrive(&full[s / KPS]);
            break;
        }
        int rtile, upid;
        unit_of(T, v, rtile, upid);
        const Item I = item_of(T, upid, 0);
        const int np = (I.n + 7) & ~7;
        if (TP2_SHARED && upid >= P_r) {   // shared expert unit (256 rows of down): 8 stages (k-block, row half), each 128 rows x K128 + the panel's
                             // quantized activation tile + its 32 scales.  Waits for the panel first (8 stages > the ring: no weights-first pass)
            if (!((pready[I.pid >> 5] >> (I.pid & 31)) & 1u)) {
                if (!(hf == epoch)) while (ld_relaxed_gpu(P.h_flags + I.pid) != epoch) {}
                (void)ld_acquire_gpu(P.h_flags + I.pid);
                asm volatile("fence.proxy.async.global;" ::: "memory");
                pready[I.pid >> 5] |= 1u << (I.pid & 31);
            }
            if (j == 0) STAMPT(P, 43);
            const uint8_t* hsrc = P.hq + (size_t)I.pid * NKT2 * NMAX * 128;
            for (int kb = 0; kb < NKT2; ++kb) {   // slot 0: rows 0..127 + B tile + scales, slot 1: rows 128..255
                const int s0 = s; mbar_wait(&empty[s], ph ^ 1); if (++s == NST) { s = 0; ph ^= 1; }
                if (kb == 0) {
                    *reinterpret_cast<volatile int*>(&su->y) = I.ebase;
                    *reinterpret_cast<volatile int*>(&su->x) = rtile | (I.n << 4) | (I.e << 10) | (I.pid << 20);
                }
                const int s1 = s; mbar_wait(&empty[s], ph ^ 1); if (++s == NST) { s = 0; ph ^= 1; }
                uint8_t* st0 = smem + s0 * STAGE;
                mbar_arrive_expect_tx(&full[s0], (unsigned)(STAGE_W + np * 128 + NMAX * 4));
                mbar_arrive_expect_tx(&full[s1], (unsigned)STAGE_W);
                tma_load_2d(st0, &P.w2s, kb * 128, rtile * 256, &full[s0]);
                tma_load_2d(smem + s1 * STAGE, &P.w2s, kb * 128, rtile * 256 + 128, &full[s1]);
                bulk_g2s(st0 + STAGE_WS, hsrc + (size_t)kb * NMAX * 128, (unsigned)(np * 128), &full[s0]);
                bulk_g2s(st0 + STAGE_W, P.hs4 + ((size_t)I.pid * NKT2 + kb) * NMAX, (unsigned)(NMAX * 4), &full[s0]);
            }
            flag_prefetch();
            continue;
        }
        // panel readiness is cached in a CTA-local bit set: only the first unit of a panel pays an acquire + proxy fence
        bool ready = TP2_DBG_FC2_ONLY || ((pready[I.pid >> 5] >> (I.pid & 31)) & 1u);
        if (j == 0 && ready) STAMPT(P, 43);
        if (!ready && (hf == epoch || ld_acquire_gpu(P.h_flags + I.pid) == epoch)) {
            if (j == 0) STAMPT(P, 43);
            asm volatile("fence.proxy.async.global;" ::: "memory");   // hq (generic writes, other CTAs) -> async-proxy bulk reads
            pready[I.pid >> 5] |= 1u << (I.pid & 31);
            ready = true;
        }
        const uint8_t* hsrc = P.hq + (size_t)I.pid * NKT2 * NMAX * 128;
        if (ready) {   // one pass: weights + B operand per stage
            for (int kt = 0; kt < NKT2; ++kt) {
                uint64_t* fb = &full[s / KPS];
                if ((kt & (KPS - 1)) == 0) {
                    mbar_wait(&empty[s / KPS], ph ^ 1);
                    if (kt == 0) {
                        *reinterpret_cast<volatile int*>(&su->y) = I.ebase;
                        *reinterpret_cast<volatile int*>(&su->x) = rtile | (I.n << 4) | (I.e << 10) | (I.pid << 20);
                    }
                    mbar_arrive_expect_tx(fb, (unsigned)(KPS * (STAGE_WS + np * 128)));
                }
                uint8_t* st = smem + s * STAGE;
                const int sl = I.e * (INTER / 32) + kt * 4;
                tma_load_3d(st, P.tmaps + 2, 0, rtile * 4, sl, fb);
                tma_load_3d(st + STAGE_W, P.tmaps + 3, 0, rtile, sl, fb);
                bulk_g2s(st + STAGE_WS, hsrc + (size_t)kt * NMAX * 128, (unsigned)(np * 128), fb);
                if (++s == NST) { s = 0; ph ^= 1; }
            }
            flag_prefetch();
            continue;
        }
        int s0 = s; unsigned ph0 = ph;
        for (int kt = 0; kt < NKT2; ++kt) {   // pass A: the unit's weights (h not published yet)
            uint64_t* fb = &full[s / KPS];
            if ((kt & (KPS - 1)) == 0) {
                mbar_wait(&empty[s / KPS], ph ^ 1);
                if (kt == 0) {
                    *reinterpret_cast<volatile int*>(&su->y) = I.ebase;
                    *reinterpret_cast<volatile int*>(&su->x) = rtile | (I.n << 4) | (I.e << 10) | (I.pid << 20);
                }
                mbar_arrive_expect_tx(fb, (unsigned)(KPS * (STAGE_WS + np * 128)));
            }
            uint8_t* st = smem + s * STAGE;
            const int sl = I.e * (INTER / 32) + kt * 4;
            tma_load_3d(st, P.tmaps + 2, 0, rtile * 4, sl, fb);
            tma_load_3d(st + STAGE_W, P.tmaps + 3, 0, rtile, sl, fb);
            if (++s == NST) { s = 0; ph ^= 1; }
        }
        while (ld_relaxed_gpu(P.h_flags + I.pid) != epoch) {}
        if (j == 0) STAMPT(P, 43);
        (void)ld_acquire_gpu(P.h_flags + I.pid);
        asm volatile("fence.proxy.async.global;" ::: "memory");
        pready[I.pid >> 5] |= 1u << (I.pid & 31);
        for (int kt = 0; kt < NKT2; ++kt) {   // pass B: the unit's B operand
            bulk_g2s(smem + s0 * STAGE + STAGE_WS, hsrc + (size_t)kt * NMAX * 128, (unsigned)(np * 128), &full[s0 / KPS]);
            if (++s0 == NST) { s0 = 0; ph0 ^= 1; }
        }
        flag_prefetch();
    }
}

// Release helper (warp 18): per owner FC1 item of this CTA, in order: wait for the consumers' stores (epi[j & 3]), fence + count
// the item on its panel; when it is the panel's 4th, the warp quantizes the panel (per token over 512 columns) and publishes it.
__device__ __forceinline__ void quantize_panel(const Params& P, const Tabs& T, int pid, int t0, int nt);
__device__ __forceinline__ void quantize_panel_sh(const Params& P, const Tabs& T, int pid, int t0, int nt);
__device__ __forceinline__ void release_helper(const Params& P, uint64_t* epi, const Tabs& T, unsigned epoch, unsigned par) {
    const int b = blockIdx.x, lane = threadIdx.x & 31;
    int j = 0;
    for (int g = 0; g < (TP2_DBG_FC2_ONLY ? 0 : 2); ++g) {
        const int lo = fc1_lo(T, g, b), hi = fc1_lo(T, g, b + 1);
        for (int gk = lo; gk < hi;) {
            const int it = gk / NKT1, kt0 = gk % NKT1;
            if (kt0 == 0) {
                mbar_wait(&epi[j & 3], (unsigned)(j >> 2) & 1u);
                if (lane == 0) STAMPT(P, 42);
                ++j;
                const int pid = it / ITEMS;
                unsigned old = 0;
                // one acq_rel atomic: releases this item's act / amax (consumer stores ordered by the epi mbarrier), and the panel's
                // 4th item acquires the other 3 (the __shfl_sync orders the other lanes' loads after lane 0's acquire)
                if (lane == 0) { if (TP2_REL_FENCES) { fence_acq_rel_gpu(); old = atomicAdd(P.pan_cnt + par * MAXP + pid, 1u); } else old = atom_add_acq_rel_gpu(P.pan_cnt + par * MAXP + pid, 1u); STAMPT(P, 46); }
                old = __shfl_sync(0xffffffffu, old, 0);
                if (old == (unsigned)ITEMS - 1u) {
                    if (TP2_REL_FENCES) fence_acq_rel_gpu();
                    if (lane == 0) STAMPT(P, 47);
                    if (TP2_SHARED && pid >= T.misc[5]) quantize_panel_sh(P, T, pid, lane, 32);
                    else quantize_panel(P, T, pid, lane, 32);
                    if (lane == 0) STAMPT(P, 44);
                    asm volatile("fence.proxy.async.global;" ::: "memory");   // hq is read by bulk copies (async proxy)
                    __syncwarp();
                    if (lane == 0) { if (TP2_REL_FENCES) fence_acq_rel_gpu(); st_release_gpu(P.h_flags + pid, epoch); STAMPT(P, 41); }
                }
            }
            gk = (it + 1) * NKT1;
        }
    }
}

// FC2 store warps (warps 17, 19): drain each finished out tile to every (token, slot) of its expert with 16-B stores.
__device__ __forceinline__ void store_helper(const Params& P, uint8_t* smem, const Tabs& T, uint64_t* fullo, uint64_t* freeo, uint64_t* sdone, int half,
                                             unsigned par) {
    const int lt = half * 32 + (threadIdx.x & 31);
    for (int j = 0;; ++j) {
        const int bo = j & 1;
        mbar_wait(&fullo[bo], (unsigned)(j >> 1) & 1u);
        const volatile int* oi = reinterpret_cast<const volatile int*>(smem + OFF_TAB + TAB_UNIT + (16 + NRTP) * 4) + bo * 4;
        const int n = oi[2];
        if (n < 0) break;
        const int rtile = oi[0], ebase = oi[1];
        const __nv_bfloat16* out_t = reinterpret_cast<const __nv_bfloat16*>(smem + OFF_STG + bo * OUT_TILE);
        const bool shu = TP2_SHARED && ebase >= T.misc[1];   // shared expert unit: its token rows of the shared output
        for (int x = lt; x < n * 32; x += 64) {
            const int tk = x >> 5, c = x & 31, t = T.ent[ebase + tk];
            unsigned km = T.km[ebase + tk];
            const uint4 v = *reinterpret_cast<const uint4*>(out_t + tk * OUT_PITCH + c * 8);
            if (shu) { *reinterpret_cast<uint4*>(P.shout + (size_t)t * DIM + rtile * 256 + c * 8) = v; continue; }
            __nv_bfloat16* o = P.outp + (size_t)t * TOPK_MAX * DIM + rtile * 256 + c * 8;
            while (km && !TP2_DBG_FC2_NOSTORE) { const int k = __ffs(km) - 1; km &= km - 1; *reinterpret_cast<uint4*>(o + (size_t)k * DIM) = v; }
        }
        __syncwarp();
        if ((threadIdx.x & 31) == 0) mbar_arrive(&freeo[bo]);   // the tile's smem reads are done
        if (TP2_STREAM_TAIL || TP2_T1_DYN) {   // both store warps' global stores of this unit done -> count it on its row tile
            named_bar_sync(5, 64);
            if (half == 0 && (threadIdx.x & 31) == 0) { fence_acq_rel_gpu(); atomicAdd(P.rt_cnt + par * NRT + rtile, 1u); }
        }
    }
    __syncwarp();
    if ((threadIdx.x & 31) == 0) mbar_arrive(sdone);
}

// ------------------------------------------------------------------------------------------------ consumer roles
struct Lane { int tid, w, wq, lane, g, tig, Lb, frag_off, scl_off; };
__device__ __forceinline__ Lane lane_of() {
    Lane L;
    L.tid = threadIdx.x; L.lane = L.tid & 31; const int warp = L.tid >> 5;
    L.w = warp >> 2; L.wq = warp & 3; L.g = L.lane >> 2; L.tig = L.lane & 3;
    const int sel = ((L.g >> 1) + (L.wq & 1)) & 1;
    L.Lb = (L.wq >> 1) * 32 + sel * 16 + L.g;   // logical row (within the 64-row block) of accumulator rows g; + 8 for rows g + 8
    L.frag_off = L.w * HLB + ((L.wq >> 1) * 32 + L.lane) * 16 + sel * 8;
    L.scl_off = STAGE_W + L.w * 64 + 8 * L.g + 4 * (L.wq >> 1) + 2 * sel;
    return L;
}

template <int N>
__device__ __forceinline__ void fc1_piece(const Params& P, uint8_t* smem, uint64_t* full, uint64_t* empty, const Tabs& T, const Lane& L,
                                          int& s, unsigned& ph, const Item& I, int grp, int it, int kt0, int kt1, unsigned epoch, uint64_t* epi_bar) {
    constexpr int R = N / 2;
    const int G = gridDim.x, b = blockIdx.x, tid = L.tid;
    float C[R], D[R];
#pragma unroll
    for (int i = 0; i < R; ++i) { C[i] = 0.f; D[i] = 0.f; }
    uint32_t A0[4][4], A1[4][4];
    const uint64_t desc0 = make_desc_sw128(smem_u32(smem + STAGE_WS));
    // owner: the epilogue's per-column factor -- loads issued now, consumed (stored to smem) after the k-loop
    // (no T.am zeroing here: the previous owner epilogue's tid < n threads may still be reading T.am -- stage 2 writes every
    //  (token, warp) cell of the N tokens, 0 where the warp has no element of that token)
    // (fused quantization: block-wide -- the consumers acquired every token before FC1; dense -- warp 19 stages the scales in
    //  TAB_XS (xsb), read after the k-loop: no consumer warp polls a flag ahead of its first k-step)
    const bool fs_mine = kt0 == 0 && tid < (P.dense ? P.md : I.n);
    const int fs_t = fs_mine ? (P.dense ? tid : (int)T.ent[I.ebase + tid]) : P.M;   // dense: column = token
    float fs_x = 0.f, fs_s2 = 0.f;
    if (fs_t < P.M) {
        if (!P.xb) fs_x = __ldg(P.xs + (size_t)fs_t * P.xs_stride);
        else if (!P.dense) fs_x = __ldcg(P.xs + fs_t);
        fs_s2 = __ldg(P.fc1_s2 + I.e);
    }
    int sprev = 0, kt = kt0;
    TSTART(t_loop);
    kstep<N, true>(smem, full, empty, s, ph, sprev, A0, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)), L.frag_off, L.scl_off, L.lane, true);
#ifdef TP2_STAMPS
    if (P.dbg && tid == 0 && dbg_smem()[37] == 0ull) dbg_smem()[37] = gtimer();   // first FC1 stage consumed (light-stamps safe)
#endif
    ++kt;
    for (; kt + 1 < kt1; kt += 2) {
        kstep<N, false>(smem, full, empty, s, ph, sprev, A1, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)), L.frag_off, L.scl_off, L.lane, true);
        kstep<N, false>(smem, full, empty, s, ph, sprev, A0, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)), L.frag_off, L.scl_off, L.lane, true);
    }
    if (kt < kt1) kstep<N, false>(smem, full, empty, s, ph, sprev, A1, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)), L.frag_off, L.scl_off, L.lane, true);
    kfinish<N>(empty, sprev, C, D, L.lane);
    if (fs_mine) {
        if (P.xb && P.dense && fs_t < P.M) {
            if (!P.lquant) mbar_wait(reinterpret_cast<uint64_t*>(smem + OFF_TAB + TAB_BAR) + 2 * NST + 11, 0u);
            fs_x = reinterpret_cast<const float*>(smem + OFF_TAB + TAB_XS)[fs_t];
        }
        T.fs[tid] = fs_x * (fs_s2 * 64.0f);   // published by the epilogue's first barrier
    }
    TACC(P, 8, t_loop);
#if defined(TP2_STAMPS) && !defined(TP2_LIGHT_STAMPS)
    if (P.dbg && tid == 0 && TP2_CLASS_STATS) { dbg_smem()[16 + (N / 8 - 1)] += (kt1 - kt0); dbg_smem()[20 + (N / 8 - 1)] += clock64() - t_loop; }
#endif
    TSTART(t_part);

    if (kt0 != 0) {   // not the owner: publish the fp32 partial (thread-private slots) for the CTA that owns k-tile 0
#pragma unroll
        for (int i = 0; i < R / 4; ++i) *part_f4(P, grp * G + b, i, tid) = make_float4(D[4 * i], D[4 * i + 1], D[4 * i + 2], D[4 * i + 3]);
        named_bar_sync(1, NCONS);
        if (tid == 0) { fence_acq_rel_gpu(); st_release_gpu(P.part_flags + grp * G + b, epoch); }
        TACC(P, 9, t_part);
        return;
    }
    if (kt1 < NKT1) {   // owner of a split item: add the partials of CTAs b+1.. (their ranges start inside this item), CTA order
        const int iend = (it + 1) * NKT1;
        int nsrc = 0;
        while (b + 1 + nsrc < G && fc1_lo(T, grp, b + 1 + nsrc) < iend) ++nsrc;
        if (tid < nsrc && (KPS == 1 || fc1_lo(T, grp, b + 1 + tid) < fc1_lo(T, grp, b + 2 + tid))) wait_eq(P.part_flags + grp * G + b + 1 + tid, epoch);   // all sources polled in parallel
        named_bar_sync(1, NCONS);
#pragma unroll 2
        for (int k = 0; k < nsrc; ++k) {   // added in CTA order
            if (KPS > 1 && fc1_lo(T, grp, b + 1 + k) == fc1_lo(T, grp, b + 2 + k)) continue;   // (empty range: no partial)
            float4 v[R / 4];
#pragma unroll
            for (int i = 0; i < R / 4; ++i) v[i] = __ldcg(part_f4(P, grp * G + b + 1 + k, i, tid));
#pragma unroll
            for (int i = 0; i < R / 4; ++i) { D[4 * i] += v[i].x; D[4 * i + 1] += v[i].y; D[4 * i + 2] += v[i].z; D[4 * i + 3] += v[i].w; }
        }
    }
    TACC(P, 9, t_part);
    TSTART(t_epi);
    // ---- epilogue: gu = BF16(D * xs * s2 * 64); SiTU-GLU (Kimi K3): act = BF16((b1 tanh(g / b1) sigmoid(g)) * (b2 tanh(u / b2))), b1 = 4, b2 = 25
    TSTART(t_e0);
    named_bar_sync(1, NCONS);   // T.fs (written at piece start) visible; partial reads done
    TACC(P, 32, t_e0);
    TSTART(t_e1);
    // stage 1: every warpgroup writes its BF16 gate / up values, lane-contiguous: gx[blk][i][128]
    __nv_bfloat16* gx = reinterpret_cast<__nv_bfloat16*>(smem + OFF_STG);
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const int tk = 8 * (i >> 2) + 2 * L.tig + (i & 1);
        gx[(L.w * (NMAX / 2) + i) * 128 + (tid & 127)] = __float2bfloat16_rn(D[i] * T.fs[tk]);
    }
    named_bar_sync(1, NCONS);
    TACC(P, 33, t_e1);
    TSTART(t_e2);
    // stage 2: all 512 threads: (row half rh = w & 1, element half = w >> 1) of the (gate, up) pairs of lane (tid & 127)
    {
        const int rh = L.w & 1, h0 = (L.w >> 1) * (R / 2);
        __nv_bfloat16* act_t = reinterpret_cast<__nv_bfloat16*>(smem + OFF_STG + 16384) + rh * 64 + L.Lb;   // [NMAX][ACT_PITCH]
        float am[R / 2];   // by global token slot j = 2 (i >> 2) + (i & 1) (compile-time); slots outside this half stay 0
#pragma unroll
        for (int j = 0; j < R / 2; ++j) am[j] = 0.f;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if (i < h0 || i >= h0 + R / 2) continue;   // the other element half belongs to warpgroup w ^ 2
            const int tk = 8 * (i >> 2) + 2 * L.tig + (i & 1), r = (i >> 1) & 1;
            const float gate = __bfloat162float(gx[(rh * (NMAX / 2) + i) * 128 + (tid & 127)]);
            const float up = __bfloat162float(gx[((2 + rh) * (NMAX / 2) + i) * 128 + (tid & 127)]);
            const float a = bf16r(situ_glu(gate, up));
            act_t[tk * ACT_PITCH + r * 8] = __float2bfloat16_rn(a);
            if (P.dense || tk < I.n) am[2 * (i >> 2) + (i & 1)] = fmaxf(am[2 * (i >> 2) + (i & 1)], fabsf(a));
        }
#pragma unroll
        for (int j = 0; j < R / 2; ++j) {
            float v = am[j];
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
            const int tk = 8 * (j >> 1) + 2 * L.tig + (j & 1);
            if (L.g == 0) T.am[tk * 16 + L.w * 4 + L.wq] = v;   // (token, warp) cells are private: untouched slots write 0
        }
    }
    named_bar_sync(1, NCONS);
    TACC(P, 34, t_e2);
    TSTART(t_e3);
    if (tid < I.n) {   // entry tid of the panel: its column is tid (compact) or its token (dense)
        const int col = P.dense ? (int)T.ent[I.ebase + tid] : tid;
        float m = T.am[col * 16];
#pragma unroll
        for (int j = 1; j < 16; ++j) m = fmaxf(m, T.am[col * 16 + j]);
        P.amax[(size_t)(I.ebase + tid) * 4 + I.q] = m;
    }
    {
        const __nv_bfloat16* act_t = reinterpret_cast<const __nv_bfloat16*>(smem + OFF_STG + 16384);
        for (int c = tid; c < I.n * 16; c += NCONS) {
            const int tk = c >> 4, ch = c & 15, col = P.dense ? (int)T.ent[I.ebase + tk] : tk;
            *reinterpret_cast<uint4*>(P.act + (size_t)(I.ebase + tk) * INTER + I.q * 128 + ch * 8) = *reinterpret_cast<const uint4*>(act_t + col * ACT_PITCH + ch * 8);
        }
    }
    __syncwarp();
    if (L.lane == 0) mbar_arrive(epi_bar);   // the release helper (warp 18) fences and counts this item
    STAMP(P, 45);
    TACC(P, 35, t_e3);
#if defined(TP2_STAMPS) && !defined(TP2_LIGHT_STAMPS)
    if (P.dbg && tid == 0) dbg_smem()[36] += 1;
#endif
    TACC(P, 10, t_epi);
}

// Quantize panel pid (all 4 items finished): per token amax over 512 columns -> hs, e4m3 rows [4 k-tiles][NMAX][128] + hs.
__device__ __forceinline__ void quantize_panel(const Params& P, const Tabs& T, int pid, int t0, int nt) {
    const Item I = item_of(T, pid, 0);
    constexpr int CPT = INTER / 16;   // 16-B chunks per token row
    for (int ch = t0; ch < I.n * CPT; ch += nt) {
        const int tk = ch / CPT, c16 = ch - tk * CPT, kb = c16 >> 3, ci = c16 & 7, en = I.ebase + tk;
        const float4 a4 = __ldcg(reinterpret_cast<const float4*>(P.amax + (size_t)en * 4));
        float amax = a4.x;
        if (ITEMS > 1) amax = fmaxf(amax, a4.y);
        if (ITEMS > 2) amax = fmaxf(amax, a4.z);
        if (ITEMS > 3) amax = fmaxf(amax, a4.w);
        const float hs = __fmul_rn(fmaxf(amax, 1e-30f), 0.0022321429569274187f);
        float inv;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(hs));
        const uint4* src = reinterpret_cast<const uint4*>(P.act + (size_t)en * INTER + c16 * 16);
        const uint4 v0 = __ldcg(src), v1 = __ldcg(src + 1);
        const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        uint32_t q4[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            q4[j] = (uint32_t)pack_e4m3x2(bf16lo(w[2 * j]) * inv, bf16hi(w[2 * j]) * inv) |
                    ((uint32_t)pack_e4m3x2(bf16lo(w[2 * j + 1]) * inv, bf16hi(w[2 * j + 1]) * inv) << 16);
        *reinterpret_cast<uint4*>(P.hq + ((size_t)(pid * NKT2 + kb) * NMAX + tk) * 128 + ((ci ^ (tk & 7)) << 4)) = make_uint4(q4[0], q4[1], q4[2], q4[3]);
        if (c16 == 0) P.hs[pid * NMAX + tk] = hs;
    }
}

// ------------------------------------------------------------------------------------------------ fused shared expert
static_assert(KPS == 1, "the fused shared expert assumes one k-tile per ring stage");
// D = fma(sa[t] * sb, C, D) per accumulator column (token t = 8 j + 2 tig + h): deep_gemm's per-K128 promotion (scale_a * scale_b, FFMA)
template <int N>
__device__ __forceinline__ void promote_sh(const float* sa, float sb, const float (&C)[N / 2], float (&D)[N / 2], const Lane& L) {
#pragma unroll
    for (int j = 0; j < N / 8; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float sc = __fmul_rn(sa[8 * j + 2 * L.tig + h], sb);
            D[4 * j + h] = __fmaf_rn(sc, C[4 * j + h], D[4 * j + h]);
            D[4 * j + 2 + h] = __fmaf_rn(sc, C[4 * j + 2 + h], D[4 * j + 2 + h]);
        }
    }
}
// One k-block of the shared expert = two ring slots (a logical stage): slot s0 = rows 0..127 (warpgroups 0, 1) + the B tile + the
// B columns' K128 scales, slot s1 = rows 128..255 (warpgroups 2, 3).  Every warpgroup runs the same code (no divergent wgmma: the
// slot is a data select): retire the previous chain (D = fma(sa sb, C, D) with its scales, release its two slots), issue this one.
__device__ __forceinline__ void adv_slot(int& s, unsigned& ph) { if (++s == NST) { s = 0; ph ^= 1; } }
template <int N, bool FIRST>
__device__ __forceinline__ void kstep_sh(uint8_t* smem, uint64_t* full, uint64_t* empty, int& s, unsigned& ph, int& p0, int& p1,
                                         float& sbp, float (&C)[N / 2], float (&D)[N / 2], float sb, const Lane& L, uint64_t descB0) {
    const int s0 = s;
    mbar_wait(&full[s], ph);
    adv_slot(s, ph);
    const int s1 = s;
    mbar_wait(&full[s], ph);
    adv_slot(s, ph);
    if constexpr (!FIRST) {
        wg_wait0();
        fence_operand(C);
        promote_sh<N>(reinterpret_cast<const float*>(smem + p0 * STAGE + STAGE_W), sbp, C, D, L);
        if (L.lane == 0) { mbar_arrive(&empty[p0]); mbar_arrive(&empty[p1]); }
    }
    const int sw = (L.w >> 1) ? s1 : s0;
    const uint64_t da = make_desc_sw128(smem_u32(smem + sw * STAGE)) + (uint64_t)((L.w & 1) * ((64 * 128) >> 4));
    fence_operand(D);
    fence_operand(C);
    wg_fence();
    chain_ss::ChainSS<N>::run(C, da, descB0 + (uint64_t)(s0 * (STAGE >> 4)));
    wg_commit();
    p0 = s0; p1 = s1; sbp = sb;
}
template <int N>
__device__ __forceinline__ void kfinish_sh(uint8_t* smem, uint64_t* empty, int p0, int p1, float sbp, float (&C)[N / 2], float (&D)[N / 2], const Lane& L) {
    wg_wait0();
    fence_operand(C);
    promote_sh<N>(reinterpret_cast<const float*>(smem + p0 * STAGE + STAGE_W), sbp, C, D, L);
    if (L.lane == 0) { mbar_arrive(&empty[p0]); mbar_arrive(&empty[p1]); }
    fence_operand(D);
}

// Shared expert FC1 piece: item q = act columns 128 q .. (gate rows 128 q.. on warpgroups 0, 1; up rows 512 + 128 q.. on 2, 3), k-blocks
// [kt0, kt1).  Stream-K partials / owner merge as fc1_piece; the owner's epilogue: gu = BF16(D) (deep_gemm), gate min 10 / up clamp
// +-10 in BF16, act = BF16(up * silu(gate)) with the production fast-math SiLU, per-token amax of the item's 128 columns (= its group).
template <int N>
__device__ __forceinline__ void fc1_piece_sh(const Params& P, uint8_t* smem, uint64_t* full, uint64_t* empty, const Tabs& T, const Lane& L,
                                             int& s, unsigned& ph, const Item& I, int grp, int it, int kt0, int kt1, unsigned epoch, uint64_t* epi_bar) {
    constexpr int R = N / 2;
    const int G = gridDim.x, b = blockIdx.x, tid = L.tid;
    float C[R], D[R];
#pragma unroll
    for (int i = 0; i < R; ++i) { C[i] = 0.f; D[i] = 0.f; }
    const uint64_t descB0 = make_desc_sw128(smem_u32(smem + STAGE_WS));
    const int nb = (L.w >> 1) * 4 + I.q;   // 128-row scale block of this warpgroup: gate q (warpgroups 0, 1) / up 4 + q (2, 3)
    int p0 = 0, p1 = 0; float sbp = 0.f;
    TSTART(t_loop);
    kstep_sh<N, true>(smem, full, empty, s, ph, p0, p1, sbp, C, D, __ldg(P.sb1 + nb * NKT1 + kt0), L, descB0);
    for (int kt = kt0 + 1; kt < kt1; ++kt) kstep_sh<N, false>(smem, full, empty, s, ph, p0, p1, sbp, C, D, __ldg(P.sb1 + nb * NKT1 + kt), L, descB0);
    kfinish_sh<N>(smem, empty, p0, p1, sbp, C, D, L);
    TACC(P, 8, t_loop);
    if (kt0 != 0) {   // not the owner: publish the fp32 partial for the CTA that owns k-block 0
#pragma unroll
        for (int i = 0; i < R / 4; ++i) *part_f4(P, grp * G + b, i, tid) = make_float4(D[4 * i], D[4 * i + 1], D[4 * i + 2], D[4 * i + 3]);
        named_bar_sync(1, NCONS);
        if (tid == 0) { fence_acq_rel_gpu(); st_release_gpu(P.part_flags + grp * G + b, epoch); }
        return;
    }
    if (kt1 < NKT1) {   // owner of a split item: add the partials of CTAs b+1.. in CTA order
        const int iend = (it + 1) * NKT1;
        int nsrc = 0;
        while (b + 1 + nsrc < G && fc1_lo(T, grp, b + 1 + nsrc) < iend) ++nsrc;
        if (tid < nsrc) wait_eq(P.part_flags + grp * G + b + 1 + tid, epoch);
        named_bar_sync(1, NCONS);
#pragma unroll 2
        for (int k = 0; k < nsrc; ++k) {
            float4 v[R / 4];
#pragma unroll
            for (int i = 0; i < R / 4; ++i) v[i] = __ldcg(part_f4(P, grp * G + b + 1 + k, i, tid));
#pragma unroll
            for (int i = 0; i < R / 4; ++i) { D[4 * i] += v[i].x; D[4 * i + 1] += v[i].y; D[4 * i + 2] += v[i].z; D[4 * i + 3] += v[i].w; }
        }
    }
    named_bar_sync(1, NCONS);   // partial reads done (the staging area below is shared with them only through the ring: none)
    // stage 1: BF16 gate (min 10) / up (clamp +-10), lane-contiguous gx[blk][i][128] (as fc1_piece)
    __nv_bfloat16* gx = reinterpret_cast<__nv_bfloat16*>(smem + OFF_STG);
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const __nv_bfloat16 v = __float2bfloat16_rn(D[i]);
        const __nv_bfloat16 lim = __float2bfloat16_rn(10.f), nlim = __float2bfloat16_rn(-10.f);
        gx[(L.w * (NMAX / 2) + i) * 128 + (tid & 127)] = L.w < 2 ? __hmin(v, lim) : __hmin(__hmax(v, nlim), lim);
    }
    named_bar_sync(1, NCONS);
    // stage 2: (row half rh = w & 1, element half = w >> 1) of the (gate, up) pairs of lane (tid & 127); natural accumulator rows
    {
        const int rh = L.w & 1, h0 = (L.w >> 1) * (R / 2), Lrow = 16 * L.wq + L.g;
        __nv_bfloat16* act_t = reinterpret_cast<__nv_bfloat16*>(smem + OFF_STG + 16384) + rh * 64 + Lrow;   // [NMAX][ACT_PITCH]
        float am[R / 2];
#pragma unroll
        for (int j = 0; j < R / 2; ++j) am[j] = 0.f;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if (i < h0 || i >= h0 + R / 2) continue;
            const int tk = 8 * (i >> 2) + 2 * L.tig + (i & 1), r = (i >> 1) & 1;
            const float gate = __bfloat162float(gx[(rh * (NMAX / 2) + i) * 128 + (tid & 127)]);
            const float up = __bfloat162float(gx[((2 + rh) * (NMAX / 2) + i) * 128 + (tid & 127)]);
            const float sg = shnum::mul_ftz(gate, shnum::rcp_approx_ftz(shnum::add_ftz(shnum::ex2_approx_ftz(shnum::mul_ftz(gate, -1.4426950216293334961f)), 1.0f)));
            const float a = bf16r(shnum::mul_ftz(up, sg));
            act_t[tk * ACT_PITCH + r * 8] = __float2bfloat16_rn(a);
            if (P.dense || tk < I.n) am[2 * (i >> 2) + (i & 1)] = fmaxf(am[2 * (i >> 2) + (i & 1)], fabsf(a));
        }
#pragma unroll
        for (int j = 0; j < R / 2; ++j) {
            float v = am[j];
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
            const int tk = 8 * (j >> 1) + 2 * L.tig + (j & 1);
            if (L.g == 0) T.am[tk * 16 + L.w * 4 + L.wq] = v;
        }
    }
    named_bar_sync(1, NCONS);
    if (tid < I.n) {   // entry tid = token t0 + tid: this item's group amax
        const int col = P.dense ? (int)T.ent[I.ebase + tid] : tid;
        float m = T.am[col * 16];
#pragma unroll
        for (int j = 1; j < 16; ++j) m = fmaxf(m, T.am[col * 16 + j]);
        P.amax[(size_t)(I.ebase + tid) * 4 + I.q] = m;
    }
    {
        const __nv_bfloat16* act_t = reinterpret_cast<const __nv_bfloat16*>(smem + OFF_STG + 16384);
        for (int c = tid; c < I.n * 16; c += NCONS) {
            const int tk = c >> 4, ch = c & 15, col = P.dense ? (int)T.ent[I.ebase + tk] : tk;
            *reinterpret_cast<uint4*>(P.act + (size_t)(I.ebase + tk) * INTER + I.q * 128 + ch * 8) = *reinterpret_cast<const uint4*>(act_t + col * ACT_PITCH + ch * 8);
        }
    }
    __syncwarp();
    if (L.lane == 0) mbar_arrive(epi_bar);   // the release helper counts the item; the panel's 4th quantizes it (quantize_panel_sh)
}

// Quantize shared panel pid: per (token, 128-group = k-block) as sglang per_token_group_quant v2 (amax = max(1e-10, group amax),
// x * (448 * rcp(amax)), clamp, RN satfinite e4m3; dequant scale amax * (1/448)) -> hq rows [4][NMAX][128] + hs4.
__device__ __forceinline__ void quantize_panel_sh(const Params& P, const Tabs& T, int pid, int t0, int nt) {
    const Item I = item_of(T, pid, 0);
    for (int ch = t0; ch < I.n * 32; ch += nt) {
        const int tk = ch >> 5, c16 = ch & 31, kb = c16 >> 3, ci = c16 & 7, en = I.ebase + tk;
        const float amax = shnum::max_ftz(shnum::gq_amax_init(), __ldcg(P.amax + (size_t)en * 4 + kb));
        const float mult = shnum::gq_mult(amax);
        const uint4* src = reinterpret_cast<const uint4*>(P.act + (size_t)en * INTER + c16 * 16);
        const uint4 v0 = __ldcg(src), v1 = __ldcg(src + 1);
        const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        uint32_t q4[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            q4[j] = (uint32_t)shnum::gq_pack2(bf16lo(w[2 * j]), bf16hi(w[2 * j]), mult) |
                    ((uint32_t)shnum::gq_pack2(bf16lo(w[2 * j + 1]), bf16hi(w[2 * j + 1]), mult) << 16);
        *reinterpret_cast<uint4*>(P.hq + ((size_t)(pid * NKT2 + kb) * NMAX + tk) * 128 + ((ci ^ (tk & 7)) << 4)) = make_uint4(q4[0], q4[1], q4[2], q4[3]);
        if (ci == 0) P.hs4[((size_t)pid * NKT2 + kb) * NMAX + tk] = shnum::gq_scale(amax);
    }
}

// Shared expert FC2 unit (256 down rows of row tile rtile): stages (k-block kb, row half hh), half hh on warpgroups 2 hh, 2 hh + 1;
// out = BF16(D) into the out tile (natural rows), the store warps write it to the tokens' shared output rows.
template <int N>
__device__ __forceinline__ void fc2_unit_sh(const Params& P, uint8_t* smem, uint64_t* full, uint64_t* empty, const Tabs& T, const Lane& L,
                                            int& s, unsigned& ph, const Item& I, int rtile, int j, uint64_t* fullo, uint64_t* freeo) {
    constexpr int R = N / 2;
    float C[R], D[R];
#pragma unroll
    for (int i = 0; i < R; ++i) { C[i] = 0.f; D[i] = 0.f; }
    const uint64_t descB0 = make_desc_sw128(smem_u32(smem + STAGE_WS));
    const float* sbr = P.sb2 + (rtile * 2 + (L.w >> 1)) * NKT2;   // 128-row scale block of this warpgroup
    int p0 = 0, p1 = 0; float sbp = 0.f;
    kstep_sh<N, true>(smem, full, empty, s, ph, p0, p1, sbp, C, D, __ldg(sbr), L, descB0);
#pragma unroll 1
    for (int kb = 1; kb < NKT2; ++kb) kstep_sh<N, false>(smem, full, empty, s, ph, p0, p1, sbp, C, D, __ldg(sbr + kb), L, descB0);
    kfinish_sh<N>(smem, empty, p0, p1, sbp, C, D, L);
    const int bo = j & 1;
    if (j >= 2) mbar_wait(&freeo[bo], (unsigned)((j >> 1) - 1) & 1u);   // tile drained (unit j - 2)
    __nv_bfloat16* out_t = reinterpret_cast<__nv_bfloat16*>(smem + OFF_STG + bo * OUT_TILE);
    const int Lrow = 16 * L.wq + L.g;
#pragma unroll
    for (int j2 = 0; j2 < R / 2; ++j2) {
        const int tk = 8 * (j2 >> 1) + 2 * L.tig + (j2 & 1);
        const int i0 = ((j2 >> 1) << 2) | (j2 & 1);
        __nv_bfloat16* o = out_t + tk * OUT_PITCH + L.w * 64 + Lrow;
        o[0] = __float2bfloat16_rn(D[i0]);
        o[8] = __float2bfloat16_rn(D[i0 + 2]);
    }
    if (L.tid == 0) {
        volatile int* oi = reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_UNIT + (16 + NRTP) * 4) + bo * 4;
        oi[0] = rtile; oi[1] = I.ebase; oi[2] = I.n;
    }
    __syncwarp();
    if (L.lane == 0) mbar_arrive(&fullo[bo]);
}

// FC2 epilogue deferred past the next unit's first k-step (TP2_FC2_DEFER): the unit's BF16 outputs wait in registers, packed (row g,
// row g + 8) per token slot, so the out-tile stores + store-warp handoff overlap the next unit's first MMA
struct Fc2Defer { uint32_t pk[8]; int nh, j, rtile, ebase, n; bool pend; };
__device__ __forceinline__ void fc2_flush(uint8_t* smem, const Lane& L, Fc2Defer& d, uint64_t* fullo, uint64_t* freeo) {
    if (!d.pend) return;
    const int bo = d.j & 1;
    if (d.j >= 2) mbar_wait(&freeo[bo], (unsigned)((d.j >> 1) - 1) & 1u);   // tile drained (unit j - 2)
    __nv_bfloat16* out_t = reinterpret_cast<__nv_bfloat16*>(smem + OFF_STG + bo * OUT_TILE);
#pragma unroll
    for (int j2 = 0; j2 < 8; ++j2) {
        if (j2 >= d.nh) continue;
        const int tk = 8 * (j2 >> 1) + 2 * L.tig + (j2 & 1);
        uint16_t* o = reinterpret_cast<uint16_t*>(out_t + tk * OUT_PITCH + L.w * 64 + L.Lb);
        o[0] = (uint16_t)(d.pk[j2] & 0xFFFFu);
        o[8] = (uint16_t)(d.pk[j2] >> 16);
    }
    if (L.tid == 0) {
        volatile int* oi = reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_UNIT + (16 + NRTP) * 4) + bo * 4;
        oi[0] = d.rtile; oi[1] = d.ebase; oi[2] = d.n;
    }
    __syncwarp();
    if (L.lane == 0) mbar_arrive(&fullo[bo]);
    d.pend = false;
}

template <int N>
__device__ __forceinline__ void fc2_unit(const Params& P, uint8_t* smem, uint64_t* full, uint64_t* empty, const Tabs& T, const Lane& L,
                                         int& s, unsigned& ph, const Item& I, int rtile, int j, uint64_t* fullo, uint64_t* freeo, Fc2Defer& df) {
    constexpr int R = N / 2;
    TSTART(t_k2);
#ifdef TP2_STAMPS
    const int jj = P.dbg ? (int)dbg_smem()[15] : 99;
#endif
    TL(P, jj, 3);
    FC2A(P, 10);   // decode + unit setup
    const float f2r = TP2_PIN_LOADS ? ldv_nc_f32(P.fc2_s2 + I.e) : __ldg(P.fc2_s2 + I.e);
    float C[R], D[R];
#pragma unroll
    for (int i = 0; i < R; ++i) { C[i] = 0.f; D[i] = 0.f; }
    uint32_t A0[4][4], A1[4][4];
    const uint64_t desc0 = make_desc_sw128(smem_u32(smem + STAGE_WS));
    int sprev = 0;
    kstep<N, true>(smem, full, empty, s, ph, sprev, A0, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)), L.frag_off, L.scl_off, L.lane, true);
    float hsr[R / 2];   // the stage landed => the gather lanes saw the panel's h flag: its scales are readable
#pragma unroll
    for (int j = 0; j < R / 2; ++j) hsr[j] = TP2_PIN_LOADS ? ldv_cg_f32(P.hs + I.pid * NMAX + 8 * (j >> 1) + 2 * L.tig + (j & 1))
                                                             : __ldcg(P.hs + I.pid * NMAX + 8 * (j >> 1) + 2 * L.tig + (j & 1));
    if (TP2_FC2_DEFER) fc2_flush(smem, L, df, fullo, freeo);   // the previous unit's epilogue, under this unit's first MMA
#pragma unroll
    for (int kt = 1; kt < NKT2; ++kt) {
        if (kt & 1) kstep<N, false>(smem, full, empty, s, ph, sprev, A1, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)), L.frag_off, L.scl_off, L.lane, true);
        else kstep<N, false>(smem, full, empty, s, ph, sprev, A0, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)), L.frag_off, L.scl_off, L.lane, true);
    }
    kfinish<N>(empty, sprev, C, D, L.lane);
    {
        const float f2 = f2r * 64.0f;
#pragma unroll
        for (int j = 0; j < R / 2; ++j) hsr[j] *= f2;   // (same product as before: hs * (s2 * 64))
    }
    TL(P, jj, 4);
    FC2A(P, 11);   // 4 k-steps + finish
    TACC(P, 13, t_k2);
    TSTART(t_e2);
    // ---- epilogue: BF16(D * hs * s2 * 64) -> out tile (j & 1); the store warps write it to every (token, slot) of this expert
    if (TP2_FC2_DEFER) {
#pragma unroll
        for (int j2 = 0; j2 < R / 2; ++j2) {
            const int i0 = ((j2 >> 1) << 2) | (j2 & 1);
            df.pk[j2] = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(D[i0] * hsr[j2])) |
                        ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(D[i0 + 2] * hsr[j2])) << 16);
        }
        df.nh = R / 2; df.j = j; df.rtile = rtile; df.ebase = I.ebase; df.n = I.n; df.pend = true;
        FC2A(P, 12);
        return;
    }
    const int bo = j & 1;
    if (j >= 2) mbar_wait(&freeo[bo], (unsigned)((j >> 1) - 1) & 1u);   // tile drained (unit j - 2)
    __nv_bfloat16* out_t = reinterpret_cast<__nv_bfloat16*>(smem + OFF_STG + bo * OUT_TILE);   // [NMAX][OUT_PITCH]
    if (TP2_DBG_FC2_NOEPI == 1) {   // keep D live (the k-steps' fp32 promotes stay), skip only the conversions / smem stores
        float sum = 0.f;   // (an impossible-branch use ptxas could prove dead turned the QGMMAs into dummy HGMMAs: compare to a runtime NaN)
#pragma unroll
        for (int i = 0; i < R; ++i) sum += D[i];
        if (sum == __int_as_float((int)(P.hseq | 0x7fc00000u))) out_t[0] = __float2bfloat16_rn(sum);
    }
    if (TP2_DBG_FC2_NOEPI == 2) {   // conversions, no smem stores
        unsigned acc = 0u;
#pragma unroll
        for (int j2 = 0; j2 < R / 2; ++j2) {
            const int i0 = ((j2 >> 1) << 2) | (j2 & 1);
            acc ^= (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(D[i0] * hsr[j2])) ^ ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(D[i0 + 2] * hsr[j2])) << 16);
        }
        if (acc == 0x12345678u && I.n < 0) out_t[0] = __float2bfloat16_rn(1.f);
    }
    if (TP2_DBG_FC2_NOEPI == 3) {   // smem stores of D-independent values (D kept live via an impossible branch)
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < R; ++i) sum += D[i];
#pragma unroll
        for (int j2 = 0; j2 < R / 2; ++j2) {
            const int tk = 8 * (j2 >> 1) + 2 * L.tig + (j2 & 1);
            __nv_bfloat16* o = out_t + tk * OUT_PITCH + L.w * 64 + L.Lb;
            o[0] = __float2bfloat16_rn(hsr[j2]);
            o[8] = __float2bfloat16_rn(hsr[j2]);
        }
        if (sum == 1234.5f && I.n < 0) out_t[1] = __float2bfloat16_rn(sum);
    }
#pragma unroll
    for (int j2 = 0; j2 < (TP2_DBG_FC2_NOEPI ? 0 : R / 2); ++j2) {
        const int tk = 8 * (j2 >> 1) + 2 * L.tig + (j2 & 1);
        const int i0 = ((j2 >> 1) << 2) | (j2 & 1);   // accumulator index of (token slot j2, row g); + 2 for row g + 8
        __nv_bfloat16* o = out_t + tk * OUT_PITCH + L.w * 64 + L.Lb;
        o[0] = __float2bfloat16_rn(D[i0] * hsr[j2]);
        o[8] = __float2bfloat16_rn(D[i0 + 2] * hsr[j2]);
    }
    if (L.tid == 0) {
        volatile int* oi = reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_UNIT + (16 + NRTP) * 4) + bo * 4;
        oi[0] = rtile; oi[1] = I.ebase; oi[2] = I.n;
    }
    __syncwarp();
    if (L.lane == 0) mbar_arrive(&fullo[bo]);
    TL(P, jj, 5);
    FC2A(P, 12);   // epilogue
    TACC(P, 14, t_e2);
#ifdef TP2_STAMPS
    if (P.dbg && L.tid == 0) dbg_smem()[15] += 1;
#endif
}

// ---- tail.  Message m of (row tile, token): rows rtile*256 + 6m .. +5 (m = 42: 4 rows), 3 bf16x2 words + epoch tag.
__device__ __forceinline__ size_t rs_idx(int src, int lrt, int t, int m) { return ((size_t)((src * 4 + lrt) * MAXM + t)) * NMSG + m; }
__device__ __forceinline__ size_t ag_idx(int rtile, int t, int m) { return ((size_t)(rtile * MAXM + t)) * NMSG + m; }
// one-shot mode (M <= OS_MAXM): every rank pushes all its messages to all NDEV ranks (in the RS buffers), each rank reduces locally
constexpr int OS_MAXM = MAXM;
__device__ __forceinline__ size_t os_idx(int src, int rtile, int t, int m) { return ((size_t)((src * NRT + rtile) * OS_MAXM + t)) * NMSG + m; }
// Both buffers are double-buffered by epoch parity (host: RankState.RS_BYTES / AG_BYTES hold two halves).  A rank that has
// pushed its messages but not yet read its peers' (preempted between T1 and the reduce) does not hold up its peers, which can
// finish this call and push the next one into its buffer: with a single buffer the pending messages would be overwritten (the
// epoch tags never match again, the rank spins forever).  No rank can get two calls ahead, so two halves suffice.
constexpr size_t RS_PAR = (size_t)NDEV * NRT * OS_MAXM * NMSG;   // uint4 per parity half (one-shot layout)
constexpr size_t AG_PAR = (size_t)NRT * MAXM * NMSG;

// T1 message m of (row tile, token t): slot-ordered fp32 sum x routing weight, x factor -> BF16, + shared -> BF16, then (ndev 1) the
// output, (one-shot) a copy to every rank, (RS) the row tile's owner rank
// T1 in two phases (all of a message's loads in flight, then the slot-ordered sum) so a thread can overlap two messages' loads
struct T1Ld { uint32_t sv[TOPK_MAX][3]; float wk[TOPK_MAX]; uint32_t sh[3]; uint32_t on; };
__device__ __forceinline__ void t1_load(const Params& P, const Tabs& T, int rtile, int t, int m, T1Ld& L1) {
    const int topk = P.topk;
    const bool rws = P.glog && P.M * topk <= 128 && P.M <= NMAX && TP2_SMALL_TABLES;   // fused gate, small tables: weights in smem
    const int r0 = rtile * 256 + 6 * m;
    const bool full6 = m < NMSG - 1;
    const uint4 idrow0 = *reinterpret_cast<const uint4*>(T.ids + t * TOPK_MAX), idrow1 = *reinterpret_cast<const uint4*>(T.ids + t * TOPK_MAX + 8);
    const uint32_t idw[8] = {idrow0.x, idrow0.y, idrow0.z, idrow0.w, idrow1.x, idrow1.y, idrow1.z, idrow1.w};
    L1.on = 0u;
#pragma unroll
    for (int k = 0; k < TOPK_MAX; ++k) {
        const bool on = k < topk && ((idw[k >> 1] >> ((k & 1) * 16)) & 0xFFFFu) != 0xFFFFu;
        if (on) {
            L1.on |= 1u << k;
            L1.wk[k] = rws ? T.grw[t * topk + k] : __ldcg(P.rw + t * topk + k);
            const uint32_t* src = reinterpret_cast<const uint32_t*>(P.outp + ((size_t)t * TOPK_MAX + k) * DIM + r0);
            L1.sv[k][0] = __ldcg(src); L1.sv[k][1] = __ldcg(src + 1); L1.sv[k][2] = full6 ? __ldcg(src + 2) : 0u;
        }
    }
    if (P.shared) {
        const uint32_t* sh = reinterpret_cast<const uint32_t*>(P.shared + (size_t)t * DIM + r0);
        L1.sh[0] = __ldcg(sh); L1.sh[1] = __ldcg(sh + 1); L1.sh[2] = full6 ? __ldcg(sh + 2) : 0u;   // (fused: written in this launch)
    }
}
__device__ __forceinline__ void t1_finish(const Params& P, int rtile, int t, int m, unsigned epoch, const T1Ld& L1) {
    const int r0 = rtile * 256 + 6 * m;
    const bool full6 = m < NMSG - 1;
    float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < TOPK_MAX; ++k) {
        if (!((L1.on >> k) & 1u)) continue;
        const float wk = L1.wk[k];
        const uint32_t v0 = L1.sv[k][0], v1 = L1.sv[k][1], v2 = L1.sv[k][2];
        acc[0] = __fadd_rn(acc[0], __fmul_rn(bf16lo(v0), wk)); acc[1] = __fadd_rn(acc[1], __fmul_rn(bf16hi(v0), wk));
        acc[2] = __fadd_rn(acc[2], __fmul_rn(bf16lo(v1), wk)); acc[3] = __fadd_rn(acc[3], __fmul_rn(bf16hi(v1), wk));
        acc[4] = __fadd_rn(acc[4], __fmul_rn(bf16lo(v2), wk)); acc[5] = __fadd_rn(acc[5], __fmul_rn(bf16hi(v2), wk));
    }
    float y[6];
#pragma unroll
    for (int q = 0; q < 6; ++q) y[q] = bf16r(__fmul_rn(acc[q], P.factor));
    if (P.shared) {
        y[0] = bf16r(__fadd_rn(y[0], bf16lo(L1.sh[0]))); y[1] = bf16r(__fadd_rn(y[1], bf16hi(L1.sh[0])));
        y[2] = bf16r(__fadd_rn(y[2], bf16lo(L1.sh[1]))); y[3] = bf16r(__fadd_rn(y[3], bf16hi(L1.sh[1])));
        y[4] = bf16r(__fadd_rn(y[4], bf16lo(L1.sh[2]))); y[5] = bf16r(__fadd_rn(y[5], bf16hi(L1.sh[2])));
    }
    const uint4 msg = make_uint4(pack_bf16x2_rn(y[0], y[1]), pack_bf16x2_rn(y[2], y[3]), pack_bf16x2_rn(y[4], y[5]), epoch);
    if (P.ndev == 1) {
        uint32_t* o = reinterpret_cast<uint32_t*>(P.out + (size_t)t * DIM + r0);
        o[0] = msg.x; o[1] = msg.y;
        if (full6) o[2] = msg.z;
    } else if (P.oneshot) {
        const size_t pb = (epoch & 1u) * RS_PAR;
#pragma unroll
        for (int d = 0; d < NDEV; ++d) st_ll(P.rs[(P.my_rank + d) % NDEV] + pb + os_idx(P.my_rank, rtile, t, m), msg);   // own copy first
    } else {
        st_ll(P.rs[rtile >> 2] + (epoch & 1u) * RS_PAR + rs_idx(P.my_rank, rtile & 3, t, m), msg);
    }
}
__device__ __forceinline__ void t1_msg(const Params& P, const Tabs& T, int rtile, int t, int m, unsigned epoch) {
    T1Ld L1;
    t1_load(P, T, rtile, t, m, L1);
    t1_finish(P, rtile, t, m, epoch, L1);
}

// warp 18 during FC2: this CTA's T1 pairs in row-tile order (pairs j = b + pi G: row tile j / M is non-decreasing), each as soon as
// its row tile's FC2 units are all stored (rt_cnt, published per unit by the store warps); lane 0 polls, the warp shares it
__device__ __forceinline__ void t1_stream(const Params& P, const Tabs& T, unsigned epoch, unsigned par) {
    const int G = gridDim.x, b = blockIdx.x, M = P.M, lane = threadIdx.x & 31;
    const unsigned need = (unsigned)T.misc[2];
    const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
    int ready = -1;
    for (int pi = 0; pi < mine; ++pi) {
        const int j = b + pi * G, rtile = j / M, t = j - rtile * M;
        if (rtile > ready) {
            if (lane == 0) wait_ge(P.rt_cnt + par * NRT + rtile, need);
            __syncwarp();
            ready = rtile;
        }
        for (int m = lane; m < NMSG; m += 32) t1_msg(P, T, rtile, t, m, epoch);
    }
}

__device__ __forceinline__ void tail(const Params& P, const Tabs& T, int tid, unsigned epoch, unsigned par) {
    const int G = gridDim.x, b = blockIdx.x, M = P.M, topk = P.topk;
    const unsigned need = (unsigned)T.misc[2];
    // T1 (TP2_STREAM_TAIL: done by warp 18 during FC2, row tile by row tile -- see t1_stream)
    if (TP2_T1_DYN && !TP2_STREAM_TAIL) {
        constexpr int TCH = 8;   // tokens per item
        const int ntc = (M + TCH - 1) / TCH, nitem = NRT * ntc;
        volatile int* s_item = reinterpret_cast<volatile int*>(const_cast<int*>(T.misc) + 16);
        unsigned done_rt = 0u;   // row tiles known complete (thread 0)
        for (;;) {
            if (tid == 0) {
                const int it = (int)atomicAdd(P.fc2_next + 2 + par, 1u);
                if (it < nitem) {
                    const int rt = it / ntc;
                    if (!((done_rt >> rt) & 1u)) { wait_ge(P.rt_cnt + par * NRT + rt, need); done_rt |= 1u << rt; }
                }
                *s_item = it;
            }
            named_bar_sync(1, NCONS);
            const int it = *s_item;
            named_bar_sync(1, NCONS);   // (s_item reread by thread 0 next round only after everyone read it)
            if (it >= nitem) break;
            const int rtile = it / ntc, t0 = (it - rtile * ntc) * TCH, nt = min(TCH, M - t0);
            for (int x = tid; x < nt * NMSG; x += NCONS) {
                const int tt = x / NMSG, m = x - tt * NMSG;
                t1_msg(P, T, rtile, t0 + tt, m, epoch);
            }
        }
    } else if (!TP2_STREAM_TAIL) {
        const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
        // one poller per row tile this CTA's pairs need (thousands of threads spinning on the 16 counters slowed both the polls
        // and the counters' own atomic updates), then the consumers' barrier releases everyone
        if (tid < NRT) {
            bool needed = false;
            for (int pi = 0; pi < mine && !needed; ++pi) needed = (b + pi * G) / M == tid;
            if (needed) wait_ge(P.rt_cnt + par * NRT + tid, need);
        }
        named_bar_sync(1, NCONS);
#if defined(TP2_STAMPS)
        if (tid == 0 && P.dbg) dbg_smem()[30] = gtimer();   // row-tile waits done (T1)
#endif
        for (int x = tid; x < mine * NMSG; x += NCONS) {   // (two messages per thread per round measured no better)
            const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / M, t = j - rtile * M;
            t1_msg(P, T, rtile, t, m, epoch);
        }
    }
    if (P.ndev == 1) return;
    STAMP(P, 4);
    (void)need;
    if (P.oneshot) {   // every (row tile, token) of this CTA: the 4 ranks' messages, summed in rank order (fp32), one BF16 RN
        const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
        const uint4* rs_me = P.rs[P.my_rank] + par * RS_PAR;
        for (int x = tid; x < mine * NMSG; x += NCONS) {
            const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / M, t = j - rtile * M;
            uint4 v[NDEV];
            unsigned pending = (1u << NDEV) - 1u;
            do {
#pragma unroll
                for (int d = 0; d < NDEV; ++d)
                    if (pending & (1u << d)) { v[d] = ld_ll(rs_me + os_idx(d, rtile, t, m)); if (v[d].w == epoch) pending &= ~(1u << d); }
            } while (pending);
            float a[6] = {bf16lo(v[0].x), bf16hi(v[0].x), bf16lo(v[0].y), bf16hi(v[0].y), bf16lo(v[0].z), bf16hi(v[0].z)};
#pragma unroll
            for (int d = 1; d < NDEV; ++d) {
                a[0] = __fadd_rn(a[0], bf16lo(v[d].x)); a[1] = __fadd_rn(a[1], bf16hi(v[d].x));
                a[2] = __fadd_rn(a[2], bf16lo(v[d].y)); a[3] = __fadd_rn(a[3], bf16hi(v[d].y));
                a[4] = __fadd_rn(a[4], bf16lo(v[d].z)); a[5] = __fadd_rn(a[5], bf16hi(v[d].z));
            }
            uint32_t* o = reinterpret_cast<uint32_t*>(P.out + (size_t)t * DIM + rtile * 256 + 6 * m);
            o[0] = pack_bf16x2_rn(a[0], a[1]); o[1] = pack_bf16x2_rn(a[2], a[3]);
            if (m < NMSG - 1) o[2] = pack_bf16x2_rn(a[4], a[5]);
        }
        STAMP(P, 5);
        return;
    }
    // T2: owner reduce in rank order (fp32 from rank 0's value, one BF16 RN), all-gather push
    {
        const int npair = 4 * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
        const uint4* rs_me = P.rs[P.my_rank] + par * RS_PAR;
        for (int x = tid; x < mine * NMSG; x += NCONS) {
            const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, lrt = j / M, t = j - lrt * M;
            uint4 v[4];
            unsigned pending = 0xfu;
            do {
#pragma unroll
                for (int d = 0; d < 4; ++d)
                    if (pending & (1u << d)) { v[d] = ld_ll(rs_me + rs_idx(d, lrt, t, m)); if (v[d].w == epoch) pending &= ~(1u << d); }
            } while (pending);
            float a[6] = {bf16lo(v[0].x), bf16hi(v[0].x), bf16lo(v[0].y), bf16hi(v[0].y), bf16lo(v[0].z), bf16hi(v[0].z)};
#pragma unroll
            for (int d = 1; d < 4; ++d) {
                a[0] = __fadd_rn(a[0], bf16lo(v[d].x)); a[1] = __fadd_rn(a[1], bf16hi(v[d].x));
                a[2] = __fadd_rn(a[2], bf16lo(v[d].y)); a[3] = __fadd_rn(a[3], bf16hi(v[d].y));
                a[4] = __fadd_rn(a[4], bf16lo(v[d].z)); a[5] = __fadd_rn(a[5], bf16hi(v[d].z));
            }
            const uint4 mo = make_uint4(pack_bf16x2_rn(a[0], a[1]), pack_bf16x2_rn(a[2], a[3]), pack_bf16x2_rn(a[4], a[5]), epoch);
            const int rtile = P.my_rank * 4 + lrt;
#pragma unroll
            for (int d = 0; d < 4; ++d) st_ll(P.ag[d] + par * AG_PAR + ag_idx(rtile, t, m), mo);
        }
    }
    STAMP(P, 5);
    // T3: pull the reduced rows
    {
        const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
        const uint4* ag_me = P.ag[P.my_rank] + par * AG_PAR;
        for (int x = tid; x < mine * NMSG; x += NCONS) {
            const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / M, t = j - rtile * M;
            uint4 v;
            do { v = ld_ll(ag_me + ag_idx(rtile, t, m)); } while (v.w != epoch);
            uint32_t* o = reinterpret_cast<uint32_t*>(P.out + (size_t)t * DIM + rtile * 256 + 6 * m);
            o[0] = v.x; o[1] = v.y;
            if (m < NMSG - 1) o[2] = v.z;
        }
    }
}

// ------------------------------------------------------------------------------------------------ kernel
__global__ void __launch_bounds__(NTHREADS, 1) k3tp8moe_kernel(const __grid_constant__ Params P) {
    extern __shared__ uint8_t smem_raw[];
    const uint32_t raw = smem_u32(smem_raw);
    uint8_t* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);
    const Tabs T = tabs(smem);
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + OFF_TAB + TAB_BAR);
    uint64_t* empty = full + NST;
    uint64_t* epi = full + 2 * NST;
    uint64_t* hdone = epi + 4;
    uint64_t* gbar = hdone + 1;
    uint64_t* fullo = gbar + 1;
    uint64_t* freeo = fullo + 2;
    uint64_t* sdone = freeo + 2;
    uint64_t* xsb = sdone + 1;
    int* s_unit = reinterpret_cast<int*>(smem + OFF_TAB + TAB_UNIT);
    int* s_rtc = s_unit + 16;   // after the int2 [8] unit ring
    const int tid = threadIdx.x, warp = tid >> 5;
#ifdef TP2_STAMPS
    if (tid == 0) { g_dbg_smem = reinterpret_cast<unsigned long long*>(smem + OFF_TAB + TAB_DBG); for (int i = 0; i < 48; ++i) g_dbg_smem[i] = 0ull; g_dbg_smem[26] = clock64(); }
#endif
#if defined(TP2_STAMPS) && defined(TP2_TL_FC1)
    if (tid == 0) { g_tl_ptr = P.tl; g_tl_j = 0; g_tl_fc1 = 1; }
#endif
    STAMP(P, 0);
    if (TP2_DBG_EXIT == 1) return;
    if (P.gw && tid < NCONS) {   // fused router GEMM: consumer warpgroup g's expert b + g G, thread j's weights (static: before the PDL wait)
        const int re = blockIdx.x + (tid >> 7) * gridDim.x;
        if (re < NEXP) {
#pragma unroll
            for (int ki = 0; ki < 4; ++ki) cp_async_16(smem + PRO_GW + (tid * 4 + ki) * 16, P.gw + (size_t)re * DIM + ki * 1024 + (tid & 127) * 8);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    const bool small_tab = P.M * P.topk <= 128 && P.M <= NMAX && TP2_SMALL_TABLES;
    // fused quantization threads (small tables: warps 8..15; block-wide / fused routing: the producer warpgroup).  x is complete
    // before the routing kernels (they read it) and the previous launch (same workspace) completed before them: with a PDL launch the
    // quantization starts before griddepcontrol.wait, overlapping the routing kernel's tail; every other thread waits first
    const bool gsplit = P.glog && small_tab && !P.gw;   // fused gate, small tables: gate on warps 0..7, quantization on warps 8..15
    const bool qthr = P.xb && (tid < NCONS ? ((!P.glog || gsplit) && small_tab && tid >= 256) : (!small_tab || (P.glog && !gsplit)));
    const bool qpre = TP2_QUANT_PRE && P.pdl_early && qthr;
    const bool early_xs = TP2_EARLY_XS && P.dense && P.xb && !P.lquant && small_tab && !(P.glog && !gsplit);   // (the producer warpgroup is idle)
    const bool pwg_q = P.xb && (!small_tab || (P.glog && !gsplit));   // the producer warpgroup quantizes (else it is idle in the prologue)
    const bool early_xq = TP2_EARLY_XQ && P.xb && !P.dense && !TP2_DBG_FC2_ONLY && P.M <= TP2_EARLY_XQ_MAXM;
    if (!qpre && (!P.pdl_early || P.xb)) griddep_wait();   // (no-op when not PDL-launched; fused quantization: the routing is the previous kernel)
    STAMP(P, 35);   // (stamps) past the PDL wait: > start when launched before the previous kernel completed
    if (tid < NST / KPS) mbar_init(&full[tid], 1u);   // the producer lane's expect_tx arrive (+ tx bytes)
    else if (tid >= 96 && tid < 96 + NRT) s_rtc[tid - 96] = 0;
    else if (tid >= 32 && tid < 32 + NST / KPS) mbar_init(&empty[tid - 32], (unsigned)(NCONS / 32));
    else if (tid >= 64 && tid < 68) mbar_init(&epi[tid - 64], (unsigned)(NCONS / 32));
    else if (tid == 68) mbar_init(hdone, 1u);
    else if (tid == 69) mbar_init(gbar, (unsigned)(NCONS / 32));
    else if (tid == 70 || tid == 71) mbar_init(&fullo[tid - 70], (unsigned)(NCONS / 32));
    else if (tid == 72 || tid == 73) mbar_init(&freeo[tid - 72], 2u);
    else if (tid == 74) mbar_init(sdone, 2u);
    else if (tid == 75) mbar_init(xsb, 1u);
    if (tid == 0) asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (tid < NCONS) {   // (exclusive branches: the router weights are dead in the quantizing producer warpgroup)
        if (gsplit && tid < 256) { route_prologue(P, smem, T, true, 256, 2); STAMP(P, 39); }   // (warps 8..15 quantize meanwhile)
        else if (P.glog && !gsplit) { route_prologue(P, smem, T, small_tab, NCONS, 1); STAMP(P, 39); }   // (the producer warpgroup quantizes)
        else if (P.xb && small_tab && tid >= 256) {   // fused input quantization of tokens t = b, b + G, .. (small: warps 8..15)
            if (P.lquant) quant_local(P, smem, warp - 8, tid & 31);
            else {
                float* red = reinterpret_cast<float*>(smem + PRO_WS) + 32;
                for (int t = blockIdx.x; t < P.M; t += gridDim.x) quant_token<256>(P, t, tid - 256, red);
            }
            if (qpre) griddep_wait();
        }
    } else if (early_xs && warp == GATHER_WARP + 2) {   // dense: token flags acquired + owners' scales staged
        const int l = tid & 31;                                                                   // while the tables are built (xsb arrive after the sync)
        float v = 0.f;
        if (l < P.M) { while (ld_relaxed_gpu(P.xflag + l) != P.hseq) __nanosleep(32); (void)ld_acquire_gpu(P.xflag + l); v = __ldcg(P.xs + l); }
        if (l < NMAX) reinterpret_cast<float*>(smem + OFF_TAB + TAB_XS)[l] = v;
    } else if (P.xb && (!small_tab || (P.glog && !gsplit))) {   // (block-wide tables / fused routing with the router GEMM: the producer warpgroup)
        float* red = reinterpret_cast<float*>(smem + OFF_TAB + TAB_QQ);   // (unused scratch, 64 words)
        for (int t = blockIdx.x; t < P.M; t += gridDim.x) quant_token<NTHREADS - NCONS>(P, t, tid - NCONS, red);
        if (qpre) griddep_wait();
        if (tid == NCONS) STAMPT(P, 40);
    }
    if (small_tab && P.dense && TP2_DENSE_TABLES) {
        if (tid < 256) {
            if (TP2_DBG_TWICE) {
#pragma unroll 1
                for (int r = 0; r < 2; ++r) { build_tables_dense(P, smem, T); named_bar_sync(2, 256); }
            } else build_tables_dense(P, smem, T);
        }
    } else if (small_tab) {
        if (warp == 0) {
            if (TP2_DBG_TWICE) {   // diagnostic: the same code twice (the second pass, which the marks keep, runs with a warm i-cache)
#pragma unroll 1
                for (int r = 0; r < 2; ++r) { build_tables_small2(P, smem, T); __syncwarp(); }
            } else if (TP2_SMALL_TABLES == 2) build_tables_small2(P, smem, T); else build_tables_small(P, smem, T);
        }
    }
    else if (tid < NCONS) {   // consumer warps (named barrier 1); the producer warpgroup quantizes meanwhile
        if (TP2_DBG_TWICE) {
#pragma unroll 1
            for (int r = 0; r < 2; ++r) { build_tables(P, smem, T); named_bar_sync(1, NCONS); }
        } else build_tables(P, smem, T);
    }
    if (early_xq && tid >= NCONS) wait_tokens(P, tid - NCONS, NTHREADS - NCONS);   // every token's flag: the consumers' gather follows the sync
    fence_proxy_async_smem();       // prologue scratch (generic writes) precedes the ring's async-proxy writes
    __syncthreads();
    STAMP(P, 1);
    if (TP2_DBG_EXIT == 2) return;
    if (warp >= PROD_WARP) {
        setmaxnreg_dec<TP2_PROD_REGS>();
        if (warp == PROD_WARP) { if ((tid & 31) == 0) producer(P, smem, full, empty, T, s_unit, gbar, early_xs); return; }   // waits after pass A
        griddep_wait();
        const unsigned epoch = __ldcg(P.work) + 1u, par = epoch & 1u;
        if (blockIdx.x == 0) {   // next launch's parity set (unused by this launch; the previous launch has completed): warps 17..19
            const int t2 = tid - NCONS - 32;
            if (t2 < NRT) P.rt_cnt[(par ^ 1u) * NRT + t2] = 0u;
            if (t2 == NRT) P.work[1 + (par ^ 1u)] = 0u;
            for (int i = t2; i < MAXP; i += NTHREADS - NCONS - 32) { P.pan_cnt[(par ^ 1u) * MAXP + i] = 0u; P.xg_cnt[(par ^ 1u) * MAXP + i] = 0u; }
            if (t2 == NRT + 2) { P.fc2_next[par ^ 1u] = 0u; P.fc2_next[2 + (par ^ 1u)] = 0u; }
            if (t2 == NRT + 3 && P.gcnt) P.gcnt[par ^ 1u] = 0u;
        }
        if (warp == GATHER_WARP + 2 && P.dense && P.xb && !P.lquant) {   // the owners' token scales: acquired + staged here, not polled by consumer warps
            const int l = tid & 31;
            if (!early_xs) {   // (else done in the prologue)
                float v = 0.f;
                if (l < P.M) { while (ld_relaxed_gpu(P.xflag + l) != P.hseq) {} (void)ld_acquire_gpu(P.xflag + l); v = __ldcg(P.xs + l); }
                if (l < NMAX) reinterpret_cast<float*>(smem + OFF_TAB + TAB_XS)[l] = v;
            }
            __syncwarp();
            if (l == 0) mbar_arrive(xsb);
        }
        if (warp == GATHER_WARP || warp == GATHER_WARP + 2) store_helper(P, smem, T, fullo, freeo, sdone, warp == GATHER_WARP ? 0 : 1, par);
        else if (warp == GATHER_WARP + 1) {
            release_helper(P, epi, T, epoch, par);
            if (TP2_STREAM_TAIL) { t1_stream(P, T, epoch, par); __syncwarp(); named_bar_arrive(6, NCONS + 32); }
        }
        return;
    }
    setmaxnreg_inc<TP2_CONS_REGS>();
    griddep_wait();
    const unsigned epoch = __ldcg(P.work) + 1u, par = epoch & 1u;
    const Lane L = lane_of();
    const int G = gridDim.x, b = blockIdx.x;
    const int P_tot = T.misc[2];
    if (!P.dense && !TP2_DBG_FC2_ONLY && P.xb && !early_xq) {   // fused quantization: the gathered rows are written in this launch
        wait_tokens(P, tid, NCONS);
        named_bar_sync(1, NCONS);
    }
    if (!P.dense && !TP2_DBG_FC2_ONLY) {   // this CTA's share of every panel's x rows into xg (per panel [kt][Npad][128], 128B-swizzled)
        const unsigned C = (unsigned)T.misc[1] * (unsigned)(NKT1 * 8), j0 = (unsigned)(((unsigned long long)C * b) / G), j1 = (unsigned)(((unsigned long long)C * (b + 1)) / G);
        for (unsigned jb = j0 + tid; jb < j1; jb += 8 * NCONS) {   // 8 independent loads in flight per thread, then the stores
            uint4 v[8];
            int off[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const unsigned j = jb + u * NCONS;
                off[u] = -1;
                if (j < j1) {
                    const int e = (int)(j / (unsigned)(NKT1 * 8)), kt = (int)((j >> 3) % (unsigned)NKT1), c = (int)(j & 7);
                    const int pid = T.e2p[e];
                    const uint32_t pv = T.pinf[pid];
                    if ((int)((pv >> 10) & 63u) <= TP2_ROWX) continue;   // row-mode panel: loaded by per-row TMA, not gathered
                    const int r = e - (int)(pv >> 16), np = ((int)((pv >> 10) & 63u) + 7) & ~7;
                    const uint4* xp = reinterpret_cast<const uint4*>(P.xq + (size_t)T.ent[e] * DIM + kt * 128 + c * 16);
                    v[u] = P.xb ? __ldcg(xp) : __ldg(xp);
                    off[u] = (int)(T.prow[pid] * NKT1 * 128u) + (kt * np + r) * 128 + ((c ^ (r & 7)) << 4);
                }
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) if (off[u] >= 0) *reinterpret_cast<uint4*>(P.xg + off[u]) = v[u];
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");   // read back by bulk copies (async proxy)
        named_bar_sync(1, NCONS);
        if (tid == 0 && j0 < j1) {
            fence_acq_rel_gpu();
            const int p0 = T.e2p[j0 / (unsigned)(NKT1 * 8)], p1 = T.e2p[(j1 - 1) / (unsigned)(NKT1 * 8)];
            for (int pid = p0; pid <= p1; ++pid) atomicAdd(P.xg_cnt + par * MAXP + pid, 1u);
        }
        STAMP(P, 29);
    }
    int s = 0; unsigned ph = 0;
    STAMP(P, 36);   // (stamps) consumers reach FC1
    {
        int j = 0;
        for (int g = 0; g < (TP2_DBG_FC2_ONLY ? 0 : 2); ++g) {
            const int lo = fc1_lo(T, g, b), hi = fc1_lo(T, g, b + 1);
            for (int gk = lo; gk < hi;) {
                const int it = gk / NKT1, kt0 = gk % NKT1, kt1 = min(NKT1, hi - (it * NKT1));
                const Item I = item_of(T, it / ITEMS, it % ITEMS);
                uint64_t* eb = &epi[j & 3];
                if (kt0 == 0) ++j;
                if (TP2_SHARED && I.pid >= T.misc[5]) {   // fused shared expert: one N32 instantiation (code size: the routed loops'
                    fc1_piece_sh<32>(P, smem, full, empty, T, L, s, ph, I, g, it, kt0, kt1, epoch, eb);   // i-cache locality)
                    gk = (it + 1) * NKT1;
                    continue;
                }
                switch (TP2_ONE_N ? 4 : P.dense ? P.md >> 3 : (I.n + 7) >> 3) {
                    case 1: fc1_piece<8>(P, smem, full, empty, T, L, s, ph, I, g, it, kt0, kt1, epoch, eb); break;
                    case 2: fc1_piece<16>(P, smem, full, empty, T, L, s, ph, I, g, it, kt0, kt1, epoch, eb); break;
                    case 3: fc1_piece<24>(P, smem, full, empty, T, L, s, ph, I, g, it, kt0, kt1, epoch, eb); break;
                    default: fc1_piece<32>(P, smem, full, empty, T, L, s, ph, I, g, it, kt0, kt1, epoch, eb); break;
                }
                gk = (it + 1) * NKT1;
            }
        }
        STAMP(P, 24);
    }
    STAMP(P, 2);
#if defined(TP2_STAMPS) && defined(TP2_TL_FC1)
    if (tid == 0) g_tl_fc1 = 0;
#endif
    {
        TSTART(t_fc2all);
        int j = 0;
        Fc2Defer df;
        df.pend = false;
        for (;; ++j) {   // units from the producer's queue grabs
            TSTART(t_w);
            TL(P, j, 0);
            if (j == 0) FC2A_SET(P); else FC2A(P, 13);   // iteration tail
            mbar_wait(&full[s / KPS], ph);
            if (TP2_FC2_STAGGER && j == 0 && (tid >> 7) >= 2 && P.M >= TP2_FC2_STAGGER_MINM) {
                const long long t0 = clock64();
                while (clock64() - t0 < TP2_FC2_STAGGER) {}
            }
            TL(P, j, 1);
            FC2A(P, 9);   // first-stage wait
            TACC(P, 28, t_w);
            if (j == 0) STAMP(P, 27);
            TSTART(t_sw);
            const int2 uv = make_int2(reinterpret_cast<volatile int*>(s_unit)[2 * (j & 7)], reinterpret_cast<volatile int*>(s_unit)[2 * (j & 7) + 1]);
            if (uv.x < 0) { fc2_flush(smem, L, df, fullo, freeo); break; }
            const int rtile = uv.x & 15;
            Item I;
            I.n = (uv.x >> 4) & 63; I.e = (uv.x >> 10) & 1023; I.pid = (uv.x >> 20) & 1023; I.ebase = uv.y; I.q = 0; I.u = 0;
            TACC(P, 11, t_sw);
            TL(P, j, 2);
            if (TP2_SHARED && I.pid >= T.misc[5]) {   // fused shared expert: one N32 instantiation (stale B rows -> unused columns)
                fc2_unit_sh<32>(P, smem, full, empty, T, L, s, ph, I, rtile, j, fullo, freeo);
                if (tid == 0) ++s_rtc[rtile];
                continue;
            }
            switch (TP2_ONE_N ? 4 : (I.n + 7) >> 3) {
                case 1: fc2_unit<8>(P, smem, full, empty, T, L, s, ph, I, rtile, j, fullo, freeo, df); break;
                case 2: fc2_unit<16>(P, smem, full, empty, T, L, s, ph, I, rtile, j, fullo, freeo, df); break;
                case 3: fc2_unit<24>(P, smem, full, empty, T, L, s, ph, I, rtile, j, fullo, freeo, df); break;
                default: fc2_unit<32>(P, smem, full, empty, T, L, s, ph, I, rtile, j, fullo, freeo, df); break;
            }
            if (tid == 0) ++s_rtc[rtile];
            TL(P, j, 6);
            TACC(P, 12, t_w);
        }
        {   // end marker for the store warps (in the next out tile), then wait until they drained everything
            const int bo = j & 1;
            if (j >= 2) mbar_wait(&freeo[bo], (unsigned)((j >> 1) - 1) & 1u);
            if (tid == 0) reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_UNIT + (16 + NRTP) * 4)[bo * 4 + 2] = -1;
            __syncwarp();
            if (L.lane == 0) mbar_arrive(&fullo[bo]);
            mbar_wait(sdone, 0u);
        }
        TACC(P, 30, t_fc2all);
        TSTART(t_rel);
        if (tid == 0 && !TP2_STREAM_TAIL && !TP2_T1_DYN) {
            fence_acq_rel_gpu();
            for (int r = 0; r < NRT; ++r) if (s_rtc[r]) atomicAdd(P.rt_cnt + par * NRT + r, (unsigned)s_rtc[r]);   // ordered by the one fence above
        }
        TACC(P, 31, t_rel);
    }
    STAMP(P, 3);
    tail(P, T, tid, epoch, par);
    STAMP(P, 6);
    if (TP2_STREAM_TAIL) named_bar_sync(6, NCONS + 32);   // + warp 18 (its streamed T1 is done)
    else named_bar_sync(1, NCONS);
    if (tid == 0) {
        fence_acq_rel_gpu();
        if (atomicAdd(P.work + 1 + par, 1u) == (unsigned)G - 1u) atomicAdd(P.work, 1u);   // last CTA out: next epoch
#ifdef TP2_STAMPS
        if (P.dbg) { dbg_smem()[7] = gtimer(); dbg_smem()[26] = clock64() - dbg_smem()[26]; for (int i = 0; i < 48; ++i) P.dbg[blockIdx.x * NSTAMP + i] = dbg_smem()[i]; }
#endif
    }
}

// ------------------------------------------------------------------------------------------------ host
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                  const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn encode_fn() {
    static EncodeTiledFn fn = nullptr;
    if (!fn) {
        void* p = nullptr; cudaDriverEntryPointQueryResult q;
        TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) == cudaSuccess && p,
                    "cuTensorMapEncodeTiled not available");
        fn = reinterpret_cast<EncodeTiledFn>(p);
    }
    return fn;
}
#ifndef TP2_W_L2PROMO
#define TP2_W_L2PROMO CU_TENSOR_MAP_L2_PROMOTION_L2_128B   // weight / scale maps (streamed once per call)
#endif
static void encode_map(CUtensorMap* m, const void* base, CUtensorMapDataType dt, uint32_t rank, const cuuint64_t* dims, const cuuint64_t* strides,
                       const cuuint32_t* box) {
    const cuuint32_t estr[5] = {1, 1, 1, 1, 1};
    CUresult r = encode_fn()(m, dt, rank, const_cast<void*>(base), dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             TP2_W_L2PROMO, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
}
constexpr int64_t SL1 = (int64_t)2 * ITEMS * 2 * HLB;   // FC1 k32 slice: [w1|w3][item][64-row block][1 KB]
constexpr int64_t SL2 = (int64_t)(DIM / 64) * HLB;      // FC2 k32 slice: [DIM/64 blocks][1 KB]
constexpr int64_t FC1_W_EXPERT = (int64_t)NKT1 * 4 * SL1, FC1_S_EXPERT = (int64_t)NKT1 * 4 * 2 * INTER;
constexpr int64_t FC2_W_EXPERT = (int64_t)(INTER / 32) * SL2, FC2_S_EXPERT = (int64_t)(INTER / 32) * DIM;
// weights: (w13 words, w13 offsets, w13 factor [E], w2 words, w2 offsets, w2 factor [E]) in Humming's layout (used in place)
static int set_weights(Params& P, torch::Tensor w1, torch::Tensor s1, torch::Tensor f1, torch::Tensor w2, torch::Tensor s2, torch::Tensor f2) {
    for (const auto& t : {w1, s1, f1, w2, s2, f2}) TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "weights must be contiguous CUDA tensors");
    TORCH_CHECK(f1.scalar_type() == torch::kFloat32 && f2.scalar_type() == torch::kFloat32, "factors must be fp32");
    const int64_t n1 = w1.numel() * w1.element_size() / FC1_W_EXPERT, n1s = s1.numel() * s1.element_size() / FC1_S_EXPERT;
    const int64_t n2 = w2.numel() * w2.element_size() / FC2_W_EXPERT, n2s = s2.numel() * s2.element_size() / FC2_S_EXPERT;
    const int64_t n = std::min(std::min(std::min(n1, n1s), std::min(n2, n2s)), std::min(f1.numel(), f2.numel()));
    TORCH_CHECK(n >= 1 && n <= NEXP, "expert count out of range");
    const void* p1 = w1.data_ptr(); const void* q1 = s1.data_ptr(); const void* p2 = w2.data_ptr(); const void* q2 = s2.data_ptr();
    TORCH_CHECK(((reinterpret_cast<uint64_t>(p1) | reinterpret_cast<uint64_t>(q1) | reinterpret_cast<uint64_t>(p2) | reinterpret_cast<uint64_t>(q2)) & 15) == 0, "16-B alignment");
    static std::unordered_map<std::string, torch::Tensor> cache;
    const std::string key = std::to_string(reinterpret_cast<uint64_t>(p1)) + "/" + std::to_string(reinterpret_cast<uint64_t>(q1)) + "/" +
                            std::to_string(reinterpret_cast<uint64_t>(p2)) + "/" + std::to_string(reinterpret_cast<uint64_t>(q2)) + "/" + std::to_string(n);
    auto it = cache.find(key);
    if (it == cache.end()) {
        alignas(64) CUtensorMap maps[4];
        const cuuint64_t d_w13[5] = {256, 2, INTER / 128, 2, (cuuint64_t)n * NKT1 * 4};
        const cuuint64_t s_w13[4] = {HLB, 2 * HLB, (INTER / 64) * HLB, (cuuint64_t)SL1};
        const cuuint32_t b_w13[5] = {256, 2, 1, 2, 4};
        const cuuint64_t d_s13[4] = {128, INTER / 128, 2, (cuuint64_t)n * NKT1 * 4}, s_s13[3] = {128, INTER, 2 * INTER};
        const cuuint32_t b_s13[4] = {128, 1, 2, 4};
        const cuuint64_t d_w2[3] = {256, DIM / 64, (cuuint64_t)n * (INTER / 32)}, s_w2[2] = {HLB, (cuuint64_t)SL2};
        const cuuint32_t b_w2[3] = {256, 4, 4};
        const cuuint64_t d_s2[3] = {256, DIM / 256, (cuuint64_t)n * (INTER / 32)}, s_s2[2] = {256, DIM};
        const cuuint32_t b_s2[3] = {256, 1, 4};
        encode_map(&maps[0], p1, CU_TENSOR_MAP_DATA_TYPE_INT32, 5, d_w13, s_w13, b_w13);
        encode_map(&maps[1], q1, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, d_s13, s_s13, b_s13);
        encode_map(&maps[2], p2, CU_TENSOR_MAP_DATA_TYPE_INT32, 3, d_w2, s_w2, b_w2);
        encode_map(&maps[3], q2, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, d_s2, s_s2, b_s2);
        torch::Tensor buf = torch::empty({(int64_t)sizeof(maps)}, torch::TensorOptions().dtype(torch::kUInt8).device(w1.device()));
        auto stream = at::cuda::getCurrentCUDAStream();
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        TORCH_CHECK(cudaStreamIsCapturing(stream, &cs) == cudaSuccess && cs == cudaStreamCaptureStatusNone, "warm up each weight set before CUDA Graph capture");
        TORCH_CHECK(cudaMemcpyAsync(buf.data_ptr(), maps, sizeof(maps), cudaMemcpyHostToDevice, stream) == cudaSuccess, "map upload");
        TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "map upload sync");
        it = cache.emplace(key, buf).first;
    }
    P.tmaps = reinterpret_cast<const CUtensorMap*>(it->second.data_ptr());
    P.fc1_s2 = f1.data_ptr<float>(); P.fc2_s2 = f2.data_ptr<float>();
    return (int)n;
}

// ws: dict-like list of workspace tensors (see tp4moe2.py RankState): act, amax, part, act_flags, part_flags, rt_cnt, work, outp.
void run(torch::Tensor xq, torch::Tensor xs, torch::Tensor w1, torch::Tensor s1, torch::Tensor f1, torch::Tensor w2, torch::Tensor s2, torch::Tensor f2,
         torch::Tensor ids, torch::Tensor rw, torch::Tensor shared, double factor,
         torch::Tensor act, torch::Tensor amax, torch::Tensor part, torch::Tensor act_flags, torch::Tensor part_flags, torch::Tensor rt_cnt,
         torch::Tensor work, torch::Tensor outp, torch::Tensor hq, torch::Tensor hs, torch::Tensor h_flags, torch::Tensor pan_cnt,
         torch::Tensor xg, torch::Tensor xg_cnt, torch::Tensor fc2_next,
         std::vector<int64_t> rs_ptrs, std::vector<int64_t> ag_ptrs, torch::Tensor out,
         int64_t my_rank, int64_t ndev, torch::Tensor dbg, torch::Tensor tl,
         torch::Tensor xb, torch::Tensor xflag, int64_t hseq,
         torch::Tensor glog, torch::Tensor gbias, torch::Tensor gw, torch::Tensor gws,
         torch::Tensor shw1, torch::Tensor shs1, torch::Tensor shw2, torch::Tensor shs2, torch::Tensor xqg, torch::Tensor xsT, torch::Tensor hs4,
         torch::Tensor shout) {
    // fused input quantization when xb (BF16 [M,4096]) is given: xq / xs are then the workspace ([MAXM,4096] fp8 bytes, [MAXM] fp32)
    const bool fused = xb.numel() > 0;
    if (fused) {
        TORCH_CHECK(xb.is_cuda() && xb.scalar_type() == torch::kBFloat16 && xb.dim() == 2 && xb.size(1) == DIM && xb.is_contiguous() &&
                    xq.numel() >= (int64_t)MAXM * DIM && xs.numel() >= MAXM && xflag.numel() >= MAXM && hseq > 0, "fused x");
    } else {
        TORCH_CHECK(xq.is_cuda() && xq.dim() == 2 && xq.size(1) == DIM && xq.is_contiguous() && xq.element_size() == 1, "x must be [M,4096] fp8 bytes");
    }
    const int M = (int)(fused ? xb.size(0) : xq.size(0));
    TORCH_CHECK(M >= 1 && M <= MAXM, "M out of range");
    if (!fused) {
        TORCH_CHECK(xs.scalar_type() == torch::kFloat32 && xs.size(0) == M && (xs.dim() == 1 || xs.stride(1) == 1), "x scale must be fp32 [M] or [M,k]");
    }
    TORCH_CHECK(ids.scalar_type() == torch::kInt32 && rw.scalar_type() == torch::kFloat32 && ids.is_contiguous() && rw.is_contiguous() &&
                ids.dim() == 2 && ids.size(0) == M && ids.sizes() == rw.sizes() && ids.size(1) >= 1 && ids.size(1) <= TOPK_MAX, "routing [M,k<=8]");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.is_contiguous() && out.numel() == (int64_t)M * DIM, "out must be bf16 [M,4096]");
    TORCH_CHECK(shared.numel() == 0 || (shared.scalar_type() == torch::kBFloat16 && shared.is_contiguous() && shared.numel() == (int64_t)M * DIM), "shared");
    TORCH_CHECK(ndev == 1 || (ndev == NDEV && (int64_t)rs_ptrs.size() == NDEV && my_rank >= 0 && my_rank < NDEV), "ndev 1 or 8");
    TORCH_CHECK(M * ids.size(1) <= MAXP, "M * topk <= 512 (one block-wide table thread per panel)");
    int dev = 0; cudaGetDevice(&dev);
    static int sms[64] = {0};
    static bool attr[64] = {false};
    if (!sms[dev]) cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev);
    const int grid = sms[dev];
    TORCH_CHECK(grid <= MAXG, "more SMs than the stream-K bound table");
    TORCH_CHECK(act.numel() >= (int64_t)MAXE * INTER && amax.numel() >= (int64_t)MAXE * 4 && part.numel() >= (int64_t)2 * grid * NCONS * (NMAX / 2) &&
                act_flags.numel() >= (int64_t)MAXP * 4 && part_flags.numel() >= 2 * grid && rt_cnt.numel() >= 2 * NRT && work.numel() >= 4 &&
                outp.numel() >= (int64_t)MAXM * TOPK_MAX * DIM && hq.numel() >= (int64_t)MAXP * NKT2 * NMAX * 128 && hs.numel() >= (int64_t)MAXP * NMAX &&
                h_flags.numel() >= MAXP && pan_cnt.numel() >= 2 * MAXP && xg.numel() >= (int64_t)(MAXE + 8 * MAXP) * DIM && xg_cnt.numel() >= 2 * MAXP && fc2_next.numel() >= 4,
                "workspace too small");
    Params P{};
    P.n_exp = set_weights(P, w1, s1, f1, w2, s2, f2);
    P.xq = xq.data_ptr<uint8_t>(); P.xs = xs.data_ptr<float>(); P.xs_stride = (fused || xs.dim() == 1) ? 1 : (int)xs.stride(0);
    P.xb = fused ? reinterpret_cast<const __nv_bfloat16*>(xb.data_ptr<at::BFloat16>()) : nullptr;
    P.xflag = fused ? reinterpret_cast<unsigned*>(xflag.data_ptr<int>()) : nullptr;
    P.hseq = (unsigned)hseq;
    TORCH_CHECK((reinterpret_cast<uint64_t>(P.xq) & 15) == 0, "x must be 16-B aligned");
    {
        static const int dense_maxm = getenv("TP2_DENSE_MAXM") ? atoi(getenv("TP2_DENSE_MAXM")) : 0;   // (dense tables not adapted to 896 experts yet)
        P.dense = M <= dense_maxm ? 1 : 0;
        TORCH_CHECK(M <= OS_MAXM, "one-shot all-reduce only: M <= OS_MAXM");
        P.oneshot = 1;
        static const int groups_minm = getenv("TP2_GROUPS_MINM") ? atoi(getenv("TP2_GROUPS_MINM")) : 16;
        P.groups = (TP2_GROUPS == 2 || M >= groups_minm) ? 2 : 1;   // 2 groups: group 0's FC1 -> FC2 handoff hides under group 1's FC1
        P.md = M <= 8 ? 8 : (M <= 16 ? 16 : (M <= 24 ? 24 : 32));
        if (!P.dense) {   // per-row x map for the compact panels of <= TP2_ROWX rows
            const cuuint64_t dims[2] = {DIM, (cuuint64_t)M}, strides[1] = {DIM};
            const cuuint32_t box[2] = {128, 1}, estr[2] = {1, 1};
            const CUresult r = encode_fn()(&P.xrow, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<uint8_t*>(P.xq), dims, strides, box, estr,
                                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            TORCH_CHECK(r == CUDA_SUCCESS, "x row tensor map: ", (int)r);
        }
        P.lquant = (TP2_LOCAL_QUANT && fused && P.dense && P.md <= 16) ? 1 : 0;
        if (P.lquant) TORCH_CHECK(xq.numel() >= (int64_t)grid * 16 * DIM, "local quantization workspace: grid x 16 rows");
        if (P.dense) {
            TORCH_CHECK(M <= NMAX, "dense mode needs M <= 32");
            const cuuint64_t dims[2] = {DIM, P.lquant ? (cuuint64_t)grid * 16 : (cuuint64_t)M}, strides[1] = {DIM};   // lquant: CTA b's rows b * 16 ..
            const cuuint32_t box[2] = {128, (cuuint32_t)P.md}, estr[2] = {1, 1};
            const CUresult r = encode_fn()(&P.xmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<uint8_t*>(P.xq), dims, strides, box, estr,
                                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            TORCH_CHECK(r == CUDA_SUCCESS, "x tensor map: ", (int)r);
        }
    }
    P.ids = ids.data_ptr<int>(); P.rw = rw.data_ptr<float>(); P.topk = (int)ids.size(1);
    if (gws.numel()) {   // every launch resets the next parity's router-GEMM barrier counter (whatever its mode)
        TORCH_CHECK(gws.scalar_type() == torch::kInt32 && gws.numel() >= (int64_t)MAXM * NEXP + MAXM + 8, "gate workspace");
        P.gcnt = reinterpret_cast<unsigned*>(gws.data_ptr<int>()) + (size_t)MAXM * NEXP + MAXM;
    }
    if (gbias.numel()) {   // fused routing: ids / rw are outputs; gw given (M <= 16): router GEMM + gate, else gate of the logits glog
        TORCH_CHECK(false, "fused routing (sqrtsoftplus top-6 gate) is not the K3 router: pass ids / rw");
        TORCH_CHECK(fused && P.n_exp == NEXP && P.topk == GATE_TOPK, "fused routing: fused quantization, 256 experts, top-6");
        TORCH_CHECK(gbias.scalar_type() == torch::kFloat32 && gbias.numel() == NEXP && gbias.is_contiguous(), "gate bias fp32 [256]");
        TORCH_CHECK(P.gcnt, "gate workspace");
        P.gbias = gbias.data_ptr<float>();
        P.gflag = reinterpret_cast<unsigned*>(gws.data_ptr<int>()) + (size_t)MAXM * NEXP;
        if (gw.numel()) {
            TORCH_CHECK(M <= 16 && grid * 4 >= NEXP, "fused router GEMM: M <= 16, >= 64 CTAs");
            TORCH_CHECK(gw.scalar_type() == torch::kBFloat16 && gw.dim() == 2 && gw.size(0) == NEXP && gw.size(1) == DIM && gw.is_contiguous(), "router weight bf16 [256,4096]");
            P.gw = reinterpret_cast<const __nv_bfloat16*>(gw.data_ptr<at::BFloat16>());
            P.glog = reinterpret_cast<const float*>(gws.data_ptr<int>());
        } else {
            TORCH_CHECK(glog.scalar_type() == torch::kFloat32 && glog.dim() == 2 && glog.size(0) == M && glog.size(1) == NEXP && glog.is_contiguous(), "logits fp32 [M,256]");
            P.glog = glog.data_ptr<float>();
        }
    }
    P.shared = shared.numel() ? reinterpret_cast<const __nv_bfloat16*>(shared.data_ptr<at::BFloat16>()) : nullptr;
    if (shw1.numel()) {   // fused shared expert: production FP8 gate_up [1024][4096] + [8][32] scales, down [4096][512] + [32][4] scales
        TORCH_CHECK(TP2_SHARED, "built without the fused shared expert (TP2_SHARED=0)");
        TORCH_CHECK(fused && !P.lquant && shared.numel() == 0, "fused shared expert: fused quantization, no external shared input");
        TORCH_CHECK(shw1.is_cuda() && shw1.element_size() == 1 && shw1.dim() == 2 && shw1.size(0) == 2 * INTER && shw1.size(1) == DIM && shw1.is_contiguous(), "shared gate_up fp8 [1024,4096]");
        TORCH_CHECK(shw2.is_cuda() && shw2.element_size() == 1 && shw2.dim() == 2 && shw2.size(0) == DIM && shw2.size(1) == INTER && shw2.is_contiguous(), "shared down fp8 [4096,512]");
        TORCH_CHECK(shs1.scalar_type() == torch::kFloat32 && shs1.is_contiguous() && shs1.numel() == 8 * NKT1, "shared gate_up scales fp32 [8,32]");
        TORCH_CHECK(shs2.scalar_type() == torch::kFloat32 && shs2.is_contiguous() && shs2.numel() == (DIM / 128) * NKT2, "shared down scales fp32 [32,4]");
        TORCH_CHECK(xqg.numel() >= (int64_t)MAXM * DIM && xqg.element_size() == 1 && xsT.numel() >= (int64_t)NKT1 * MAXM && xsT.scalar_type() == torch::kFloat32 &&
                    hs4.numel() >= (int64_t)MAXP * NKT2 * NMAX && hs4.scalar_type() == torch::kFloat32 &&
                    shout.numel() >= (int64_t)M * DIM && shout.scalar_type() == torch::kBFloat16 && shout.is_contiguous(), "shared workspace");
        P.shw = 1;
        P.sb1 = shs1.data_ptr<float>(); P.sb2 = shs2.data_ptr<float>();
        P.xqg = reinterpret_cast<uint8_t*>(xqg.data_ptr()); P.xsT = xsT.data_ptr<float>(); P.hs4 = hs4.data_ptr<float>();
        P.shout = reinterpret_cast<__nv_bfloat16*>(shout.data_ptr<at::BFloat16>());
        P.shared = P.shout;
        auto map2d = [](CUtensorMap* m, const void* base, uint64_t cols, uint64_t rows, uint32_t box_rows) {
            const cuuint64_t dims[2] = {cols, rows}, strides[1] = {cols};
            const cuuint32_t box[2] = {128, box_rows}, estr[2] = {1, 1};
            const CUresult r = encode_fn()(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box, estr,
                                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            TORCH_CHECK(r == CUDA_SUCCESS, "shared tensor map: ", (int)r);
        };
        static std::unordered_map<uint64_t, CUtensorMap> wmaps;   // weight maps by base pointer (encoded once per layer)
        for (int i = 0; i < 2; ++i) {
            const void* base = i ? shw2.data_ptr() : shw1.data_ptr();
            auto mit = wmaps.find(reinterpret_cast<uint64_t>(base));
            if (mit == wmaps.end()) {
                CUtensorMap mm;
                if (i) map2d(&mm, base, INTER, DIM, 128); else map2d(&mm, base, DIM, 2 * INTER, 128);
                mit = wmaps.emplace(reinterpret_cast<uint64_t>(base), mm).first;
            }
            (i ? P.w2s : P.w1s) = mit->second;
        }
        map2d(&P.xgs, P.xqg, DIM, (uint64_t)M, NMAX);
    }
    P.factor = (float)factor;
    P.act = reinterpret_cast<__nv_bfloat16*>(act.data_ptr<at::BFloat16>()); P.amax = amax.data_ptr<float>(); P.part = part.data_ptr<float>();
    P.act_flags = reinterpret_cast<unsigned*>(act_flags.data_ptr<int>()); P.part_flags = reinterpret_cast<unsigned*>(part_flags.data_ptr<int>());
    P.rt_cnt = reinterpret_cast<unsigned*>(rt_cnt.data_ptr<int>()); P.work = reinterpret_cast<unsigned*>(work.data_ptr<int>());
    P.outp = reinterpret_cast<__nv_bfloat16*>(outp.data_ptr<at::BFloat16>());
    P.hq = hq.data_ptr<uint8_t>(); P.hs = hs.data_ptr<float>();
    P.h_flags = reinterpret_cast<unsigned*>(h_flags.data_ptr<int>()); P.pan_cnt = reinterpret_cast<unsigned*>(pan_cnt.data_ptr<int>());
    P.xg = xg.data_ptr<uint8_t>(); P.xg_cnt = reinterpret_cast<unsigned*>(xg_cnt.data_ptr<int>());
    P.fc2_next = reinterpret_cast<unsigned*>(fc2_next.data_ptr<int>());
    for (int d = 0; d < NDEV; ++d) {
        P.rs[d] = ndev == NDEV ? reinterpret_cast<uint4*>(rs_ptrs[d]) : nullptr;
        P.ag[d] = (ndev == NDEV && (int64_t)ag_ptrs.size() == NDEV) ? reinterpret_cast<uint4*>(ag_ptrs[d]) : nullptr;
    }
    P.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    P.M = M; P.my_rank = (int)my_rank; P.ndev = (int)ndev;
    P.dbg = dbg.numel() ? reinterpret_cast<unsigned long long*>(dbg.data_ptr<int64_t>()) : nullptr;
    P.tl = tl.numel() ? reinterpret_cast<unsigned long long*>(tl.data_ptr<int64_t>()) : nullptr;
#ifndef TP2_STAMPS
    TORCH_CHECK(!P.dbg, "stamps need a -DTP2_STAMPS build");
#else
    if (P.dbg) TORCH_CHECK(dbg.numel() >= (int64_t)grid * NSTAMP, "dbg too small");
#endif
    if (!attr[dev]) {
        TORCH_CHECK(cudaFuncSetAttribute(k3tp8moe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_ALLOC) == cudaSuccess, "smem attribute");
        attr[dev] = true;
    }
    static const int pdl = getenv("TP2_PDL") ? atoi(getenv("TP2_PDL")) : 0;   // 0 off, 1 tables before the wait, 2 wait first
    P.pdl_early = pdl == 1 ? 1 : 0;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid); cfg.blockDim = dim3(NTHREADS); cfg.dynamicSmemBytes = SMEM_ALLOC; cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, k3tp8moe_kernel, P) == cudaSuccess, "launch");
    const cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "launch failed: ", cudaGetErrorString(e));
}


// ------------------------------------------------------------------------------------------------ router GEMM (M <= 16)
// dsv3_router_gemm<kNumTokens = M> bit for bit (logits fp32 [M][256] = x bf16 [M][4096] . W bf16 [256][4096]^T): per expert 128
// threads, thread j chains fma over k = 8 j + 1024 ki + i (ki outer, i inner), 32-lane xor-tree sums, then 0 + warp 0 + .. + warp 3.
// RX_E experts per block (one 128-thread group each) share x staged once in smem (dsv3 reads x from L2 once per expert).
#ifndef TP2_RX_E
#define TP2_RX_E 2
#endif
constexpr int RX_E = TP2_RX_E;
__global__ void __launch_bounds__(RX_E * 128) router16_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                              float* __restrict__ out, int M) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // (the MoE kernel's wait covers this grid's completion)
    extern __shared__ uint4 rx_smem[];   // x chunk [M][1024] bf16 (k-iteration ki), then float red [RX_E][16][4]
    const int tid = threadIdx.x, g = tid >> 7, j = tid & 127, lane = tid & 31, wq = j >> 5;
    const int e = blockIdx.x * RX_E + g;
    uint4 wv[4];
#pragma unroll
    for (int ki = 0; ki < 4; ++ki) wv[ki] = __ldg(reinterpret_cast<const uint4*>(w + (size_t)e * DIM + ki * 1024 + j * 8));   // static: before the wait
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float acc[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) acc[m] = 0.f;
#pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
        if (ki) __syncthreads();   // previous chunk consumed
        for (int i = tid; i < M * 128; i += RX_E * 128) {   // chunk ki: x[m][ki * 1024 .. + 1024] -> rx_smem[m * 128 + c]
            const int m = i >> 7, c = i & 127;
            rx_smem[i] = __ldg(reinterpret_cast<const uint4*>(x + (size_t)m * DIM + ki * 1024) + c);
        }
        __syncthreads();
        const uint32_t w4[4] = {wv[ki].x, wv[ki].y, wv[ki].z, wv[ki].w};
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            if (m < M) {
                const uint4 a = rx_smem[m * 128 + j];
                const uint32_t a4[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[m] = __fmaf_rn(bf16lo(a4[i]), bf16lo(w4[i]), acc[m]);
                    acc[m] = __fmaf_rn(bf16hi(a4[i]), bf16hi(w4[i]), acc[m]);
                }
            }
        }
    }
    float* red = reinterpret_cast<float*>(rx_smem + 16 * 128) + g * 64;
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        if (m < M) {
            float v = acc[m];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) red[m * 4 + wq] = v;
        }
    }
    __syncthreads();
    if (j < M) {
        float f = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) f += red[j * 4 + q];
        out[(size_t)j * NEXP + e] = f;
    }
}

// routerx: the same per-(expert, token) arithmetic as router16 / dsv3 (thread j chains fma over k = 8 j + 1024 ki + i, xor-tree,
// 0 + warp 0 + .. + warp 3), but one 128-thread block computes EB experts and each thread reuses its x lanes for all EB of them
// (dsv3 re-reads x per expert: M x 16 B per thread per k-iteration from L1 / L2).  Weights loaded before the PDL wait.
template <int MT, int EB>
__global__ void __launch_bounds__(128) routerx_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                      float* __restrict__ out) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // (the MoE kernel's wait covers this grid's completion)
    __shared__ float red[EB][MT][4];
    const int j = threadIdx.x, lane = j & 31, wq = j >> 5;
    const int e0 = blockIdx.x * EB;
    uint4 wv[EB][4];
#pragma unroll
    for (int e = 0; e < EB; ++e)
#pragma unroll
        for (int ki = 0; ki < 4; ++ki) wv[e][ki] = __ldg(reinterpret_cast<const uint4*>(w + (size_t)(e0 + e) * DIM + ki * 1024 + j * 8));
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float acc[EB][MT];
#pragma unroll
    for (int e = 0; e < EB; ++e)
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[e][m] = 0.f;
    uint4 xa[MT], xb[MT];   // x lanes of k-iterations ki (current) and ki + 1 (in flight): one L2 round trip exposed, not four
#pragma unroll
    for (int m = 0; m < MT; ++m) xa[m] = __ldg(reinterpret_cast<const uint4*>(x + (size_t)m * DIM + j * 8));
#pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
        uint4 (&xv)[MT] = (ki & 1) ? xb : xa;
        uint4 (&xn)[MT] = (ki & 1) ? xa : xb;
        if (ki < 3) {
#pragma unroll
            for (int m = 0; m < MT; ++m) xn[m] = __ldg(reinterpret_cast<const uint4*>(x + (size_t)m * DIM + (ki + 1) * 1024 + j * 8));
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const uint32_t a4[4] = {xv[m].x, xv[m].y, xv[m].z, xv[m].w};
#pragma unroll
            for (int e = 0; e < EB; ++e) {
                const uint32_t w4[4] = {wv[e][ki].x, wv[e][ki].y, wv[e][ki].z, wv[e][ki].w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[e][m] = __fmaf_rn(bf16lo(a4[i]), bf16lo(w4[i]), acc[e][m]);
                    acc[e][m] = __fmaf_rn(bf16hi(a4[i]), bf16hi(w4[i]), acc[e][m]);
                }
            }
        }
    }
#pragma unroll
    for (int e = 0; e < EB; ++e)
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float v = acc[e][m];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) red[e][m][wq] = v;
        }
    __syncthreads();
    if (j < EB * MT) {
        const int e = j / MT, m = j - e * MT;
        float f = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) f += red[e][m][q];
        out[(size_t)m * NEXP + e0 + e] = f;
    }
}
template <int MT>
static void routerx_launch(int eb, const __nv_bfloat16* x, const __nv_bfloat16* w, float* out, int pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.blockDim = dim3(128); cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
    cudaError_t r;
    if (eb == 1) { cfg.gridDim = dim3(NEXP); r = cudaLaunchKernelEx(&cfg, routerx_kernel<MT, 1>, x, w, out); }
    else if (eb == 2) { cfg.gridDim = dim3(NEXP / 2); r = cudaLaunchKernelEx(&cfg, routerx_kernel<MT, 2>, x, w, out); }
    else { cfg.gridDim = dim3(NEXP / 4); r = cudaLaunchKernelEx(&cfg, routerx_kernel<MT, 4>, x, w, out); }
    TORCH_CHECK(r == cudaSuccess, "routerx launch");
}
template <int MT>
static void routerx_dispatch(int M, int eb, const __nv_bfloat16* x, const __nv_bfloat16* w, float* out, int pdl) {
    if constexpr (MT > 16) { TORCH_CHECK(false, "routerx: M <= 16"); }
    else if (M == MT) routerx_launch<MT>(eb, x, w, out, pdl);
    else routerx_dispatch<MT + 1>(M, eb, x, w, out, pdl);
}
void routerx(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t eb) {
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.size(1) == DIM && x.is_contiguous() && x.size(0) >= 1 && x.size(0) <= 16, "x bf16 [M <= 16, 4096]");
    TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.dim() == 2 && w.size(0) == NEXP && w.size(1) == DIM && w.is_contiguous(), "router weight bf16 [256, 4096]");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32 && out.is_contiguous() && out.numel() == x.size(0) * NEXP, "logits fp32 [M, 256]");
    static const int pdl = getenv("TP2_PDL") ? atoi(getenv("TP2_PDL")) : 0;
    routerx_dispatch<1>((int)x.size(0), (int)eb, (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(), (const __nv_bfloat16*)w.data_ptr<at::BFloat16>(),
                        out.data_ptr<float>(), pdl);
}

void router16(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.size(1) == DIM && x.is_contiguous() && x.size(0) >= 1 && x.size(0) <= 16, "x bf16 [M <= 16, 4096]");
    TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.dim() == 2 && w.size(0) == NEXP && w.size(1) == DIM && w.is_contiguous(), "router weight bf16 [256, 4096]");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32 && out.is_contiguous() && out.numel() == x.size(0) * NEXP, "logits fp32 [M, 256]");
    const int M = (int)x.size(0);
    const size_t smem = 16 * 128 * 16 + RX_E * 64 * 4;   // one 1024-column chunk of x (M <= 16) + the cross-warp sums
    static const int pdl = getenv("TP2_PDL") ? atoi(getenv("TP2_PDL")) : 0;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(NEXP / RX_E); cfg.blockDim = dim3(RX_E * 128); cfg.dynamicSmemBytes = smem; cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, router16_kernel, (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(), (const __nv_bfloat16*)w.data_ptr<at::BFloat16>(),
                                   out.data_ptr<float>(), M) == cudaSuccess, "router launch");
}

}  // namespace TP2_NS

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &TP2_NS::run);
    m.def("router16", &TP2_NS::router16);
    m.def("routerx", &TP2_NS::routerx);
    m.attr("MAXM") = TP2_NS::MAXM;
    m.attr("DIM") = TP2_NS::DIM;
    m.attr("INTER") = TP2_NS::INTER;
    m.attr("NEXP") = TP2_NS::NEXP;
    m.attr("NDEV") = TP2_NS::NDEV;
    m.attr("ITEMS") = TP2_NS::ITEMS;
    m.attr("NKT1") = TP2_NS::NKT1;
    m.attr("NKT2") = TP2_NS::NKT2;
    m.attr("OS_MAXM") = TP2_NS::OS_MAXM;
    m.attr("MAXE") = TP2_NS::MAXE;
    m.attr("MAXP") = TP2_NS::MAXP;
    m.attr("NCONS") = TP2_NS::NCONS;
    m.attr("NMAX") = TP2_NS::NMAX;
    m.attr("NRT") = TP2_NS::NRT;
    m.attr("NMSG") = TP2_NS::NMSG;
    m.attr("TOPK_MAX") = TP2_NS::TOPK_MAX;
    m.attr("NSTAMP") = TP2_NS::NSTAMP;
    m.attr("SMEM_ALLOC") = TP2_NS::SMEM_ALLOC;
    m.attr("GATE") = 1;   // run() takes the fused-routing tensors
}
