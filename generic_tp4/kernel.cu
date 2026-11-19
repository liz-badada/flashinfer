// SM90a block128 projection. Tiny PTX helpers below follow DeepGEMM (MIT,
// Copyright 2025 DeepSeek) and CUTLASS/CUTE (BSD-3-Clause, Copyright 2017-2025
// NVIDIA CORPORATION & AFFILIATES). Full notices are in licenses/.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#ifndef TILE_GROUPS
#define TILE_GROUPS 2
#endif
#ifndef N_WIDTH
#error N_WIDTH must be supplied
#endif
#ifndef K_WIDTH
#error K_WIDTH must be supplied
#endif
#ifndef PIPE_STAGES
#define PIPE_STAGES 3
#endif
#ifndef CACHE_POLICY
#define CACHE_POLICY 0
#endif
#ifndef WEIGHTED
#define WEIGHTED 0
#endif

#ifndef TOP_K
#error TOP_K must be supplied
#endif
static_assert(TOP_K > 0);
namespace {
__device__ __forceinline__ uint32_t shared_address(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
struct Barrier {
    uint64_t state;
    __device__ __forceinline__ void init(uint32_t count) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
                     "r"(shared_address(this)), "r"(count) : "memory");
    }
    __device__ __forceinline__ void wait(uint32_t phase) const {
        asm volatile("{\n.reg .pred p;\nwait_loop:\n"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n"
                     "@!p bra wait_loop;\n}"
                     :: "r"(shared_address(this)), "r"(phase), "r"(0x989680)
                     : "memory");
    }
    __device__ __forceinline__ void arrive() {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::
                     "r"(shared_address(this)) : "memory");
    }
    __device__ __forceinline__ void expect(uint32_t bytes) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
                     "r"(shared_address(this)), "r"(bytes) : "memory");
    }
};
static_assert(sizeof(Barrier) == 8);

__device__ __forceinline__ void load_tma(const CUtensorMap* map, Barrier* full,
                                       void* destination, int k, int row) {
    uint64_t policy;
#if CACHE_POLICY == 1
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 0f3F800000;" : "=l"(policy));
#elif CACHE_POLICY == 2
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 0f3F800000;" : "=l"(policy));
#else
    policy = 0x1000000000000000ULL;
#endif
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global."
                 "mbarrier::complete_tx::bytes.L2::cache_hint "
                 "[%0], [%1, {%3, %4}], [%2], %5;"
                 :: "r"(shared_address(destination)), "l"(map),
                    "r"(shared_address(full)), "r"(k), "r"(row),
                    "l"(policy) : "memory");
}
__device__ __forceinline__ uint64_t smem_descriptor(const void* p) {
    // 128-byte swizzle, K-major, leading offset 0, stride offset 1024 bytes.
    return ((shared_address(p) >> 4) & 0x3fffULL) |
           (64ULL << 32) | (1ULL << 62);
}
__device__ __forceinline__ void operand_fence(float& x) {
    asm volatile("" : "+f"(x) :: "memory");
}
__device__ __forceinline__ void mma(uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
 uint64_t b,float* d,bool accumulate) {
    asm volatile("{.reg .pred p;setp.ne.b32 p,%9,0;"
                 "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7},%8,p,1,1,0;}"
                 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
                 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"l"(b),"r"(int(accumulate)));
}
__device__ __forceinline__ void register_fence(uint32_t& a) {
    asm volatile("":"+r"(a)::"memory");
}
constexpr int STAGE_BYTES=(8*TILE_GROUPS+2)*1024;
constexpr int WRAW=0,X16=8*TILE_GROUPS*1024;
__device__ __forceinline__ int sw128(int address) {
    return address ^ ((address>>3)&0x70);
}
__device__ __forceinline__ uint32_t fp8pair_to_f16(uint16_t raw) {
    const __half2_raw pair=__nv_cvt_fp8x2_to_halfraw2(raw,__NV_E4M3);
    return uint32_t(pair.x)|(uint32_t(pair.y)<<16);
}
template<bool Weighted>
__device__ __forceinline__ void project(const CUtensorMap& weight_map,
 const uint8_t* __restrict__ x,const float* __restrict__ xs,
 const float* __restrict__ ws,const int32_t* __restrict__ slots,
 const int32_t* __restrict__ experts,const float* __restrict__ rw,
 nv_bfloat16* __restrict__ out) {
    constexpr int N=N_WIDTH,K=K_WIDTH,KB=K/128,S=PIPE_STAGES;
    static_assert(N%128==0 && K%128==0);
    const int panel=blockIdx.y,tile0=blockIdx.x*(64*TILE_GROUPS);
    const int n0=tile0+(threadIdx.x/128)*64;
    const int expert=experts[panel];
    if(expert<0) return;
    extern __shared__ __align__(1024) unsigned char smem[];
    auto* raw=reinterpret_cast<Barrier*>(smem+S*STAGE_BYTES);
    auto* ready=raw+S;
    auto* empty=ready+S;
    const int tid=threadIdx.x,lane=tid%32;
    for(int i=tid;i<S*512;i+=128*TILE_GROUPS+32) {
        const int st=i/512,word=i%512;
        reinterpret_cast<uint32_t*>(smem+st*STAGE_BYTES+X16)[word]=0;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if(tid==0) {
        #pragma unroll
        for(int i=0;i<S;++i) {
            raw[i].init(1);
            ready[i].init(1);
            empty[i].init(4*TILE_GROUPS);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if(tid>=128*TILE_GROUPS) {
        #pragma unroll 1
        for(int kb=0;kb<KB;++kb) {
            const int stage=kb%S,phase=(kb/S)&1;
            if(lane==0) {
                empty[stage].wait(phase^1);
                load_tma(&weight_map,raw+stage,smem+stage*STAGE_BYTES+WRAW,
                         kb*128,expert*N+tile0);
                raw[stage].expect(64*TILE_GROUPS*128);
            }
            __syncwarp();
            auto* base=smem+stage*STAGE_BYTES;
            #pragma unroll
            for(int m=0;m<8;++m) {
                const int slot=slots[panel*8+m];
                if(slot>=0) {
                    const int row=Weighted?slot:slot/TOP_K;
                    #pragma unroll
                    for(int part=0;part<2;++part) {
                        const int k=lane*2+part*64;
                        const uint16_t value=*reinterpret_cast<const uint16_t*>(x+static_cast<int64_t>(row)*K+kb*128+k);
                        const int to=sw128((k/64)*8*128+m*128+(k%64)*2);
                        *reinterpret_cast<uint32_t*>(base+X16+to)=fp8pair_to_f16(value);
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if(lane==0) ready[stage].arrive();
        }
        return;
    }
    const int warp=(tid/32)%4,row0=warp*16+lane/4,row1=row0+8;
    const int m0=(lane%4)*2,m1=m0+1;
    const int slot0=slots[panel*8+m0],slot1=slots[panel*8+m1];
    const int input0=Weighted?slot0:slot0/TOP_K;
    const int input1=Weighted?slot1:slot1/TOP_K;
    float final[4]={0,0,0,0};
    #pragma unroll 1
    for(int kb=0;kb<KB;++kb) {
        const int stage=kb%S,phase=(kb/S)&1;
        const float sx0=slot0>=0?xs[static_cast<int64_t>(input0)*KB+kb]:0.0f;
        const float sx1=slot1>=0?xs[static_cast<int64_t>(input1)*KB+kb]:0.0f;
        const float sw=ws[(static_cast<int64_t>(expert)*(N/128)+n0/128)*KB+kb];
        ready[stage].wait(phase);
        raw[stage].wait(phase);
        auto* base=smem+stage*STAGE_BYTES;
        const uint64_t bdesc=smem_descriptor(base+X16);
        uint32_t registers[2][4];
        float partial[4]={0,0,0,0};
        #pragma unroll
        for(int i=0;i<4;++i) operand_fence(partial[i]);
        const int start=sw128(((tid/128)*64+row0)*128+(lane%4)*2);
        #pragma unroll
        for(int kk=0;kk<8;++kk) {
            const int current=kk%2;
            const int from=start^(kk*16);
            registers[current][0]=fp8pair_to_f16(*reinterpret_cast<const uint16_t*>(base+WRAW+from));
            registers[current][1]=fp8pair_to_f16(*reinterpret_cast<const uint16_t*>(base+WRAW+from+1024));
            registers[current][2]=fp8pair_to_f16(*reinterpret_cast<const uint16_t*>(base+WRAW+from+8));
            registers[current][3]=fp8pair_to_f16(*reinterpret_cast<const uint16_t*>(base+WRAW+from+1032));
            #pragma unroll
            for(int i=0;i<4;++i) register_fence(registers[current][i]);
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
            mma(registers[current][0],registers[current][1],registers[current][2],registers[current][3],
                bdesc+(kk/4)*64+(kk%4)*2,partial,kk!=0);
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            if(kk>0) {
                asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                #pragma unroll
                for(int i=0;i<4;++i) register_fence(registers[1-current][i]);
            }
        }
        #pragma unroll
        for(int i=0;i<4;++i) operand_fence(partial[i]);
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        #pragma unroll
        for(int i=0;i<4;++i) register_fence(registers[1][i]);
        if(lane==0) empty[stage].arrive();
        final[0]=fmaf(__fmul_rn(partial[0],sx0),sw,final[0]);
        final[1]=fmaf(__fmul_rn(partial[1],sx1),sw,final[1]);
        final[2]=fmaf(__fmul_rn(partial[2],sx0),sw,final[2]);
        final[3]=fmaf(__fmul_rn(partial[3],sx1),sw,final[3]);
    }
    if(slot0>=0) {
        const float factor=Weighted?rw[slot0]:1.0f;
        out[static_cast<int64_t>(slot0)*N+n0+row0]=__float2bfloat16_rn(final[0]*factor);
        out[static_cast<int64_t>(slot0)*N+n0+row1]=__float2bfloat16_rn(final[2]*factor);
    }
    if(slot1>=0) {
        const float factor=Weighted?rw[slot1]:1.0f;
        out[static_cast<int64_t>(slot1)*N+n0+row0]=__float2bfloat16_rn(final[1]*factor);
        out[static_cast<int64_t>(slot1)*N+n0+row1]=__float2bfloat16_rn(final[3]*factor);
    }
}
}

extern "C" __global__ __launch_bounds__(128*TILE_GROUPS+32)
void panel_project(const __grid_constant__ CUtensorMap w,const uint8_t* x,
 const float* xs,const float* ws,const int32_t* slots,const int32_t* experts,
 const float* rw,nv_bfloat16* out) {
    project<WEIGHTED!=0>(w,x,xs,ws,slots,experts,rw,out);
}
