// glm53_sparse_mla -- sparse MLA (NoPE, tail_dim == 0) for sm_120 / sm_121.
//
// Serves GLM-5.3-Flash's MLA geometry, which no vLLM backend supports on
// capability 12.x: kv_lora_rank=512, qk_rope_head_dim=0 (so pe_dim==0),
// v_head_dim=256, index_topk=2048, 32 q-heads per rank at TP=2.
//
// This is the single-pass kernel used for BOTH prefill and decode. That is not a
// simplification: causality is the indexer's job (the reference computes
// max_kv_i = q_i and never reads it), so prefill and decode tokens are
// indistinguishable to the kernel and a mixed batch runs as one grid.
//
// Derived from the M5 milestone -- see README.md for the measured evolution and
// for the two defects this fixes in the TileLang kernel it replaces.
//
//
// With M4 the gather is fully hidden -- ablation at T=2048: gather alone 1.01 ms
// of a 3.27 ms kernel, and the parts now sum to MORE than the whole, so the
// pipeline genuinely overlaps. What is left is compute-bound, and the cost is
// not the mma instructions but feeding them: GEMM1 issued six 32-bit shared
// loads per mma, and GEMM2's B operand -- which needs KV transposed -- was
// assembled by hand from four 16-bit loads per register, eight per mma.
//
// ldmatrix replaces those: one .x4 for the four A registers, one .x2 for the two
// B registers, one .x2.trans for GEMM2's transposed B.
//
// kPad = 8 halves keeps all three conflict-free: rows sit 520 halves = 260 words
// apart and 260 % 32 = 4, so the eight rows an ldmatrix addresses land on eight
// distinct bank groups and the 32 lanes cover 32 banks.
//
// M4 header follows.
//
// After M3 the gather was still 58% of the kernel (1.94 ms of 3.35 ms at
// T=2048) -- and it is a LATENCY problem, not a bandwidth one: the KV address
// depends on an index that is itself a global load, and with only 8 chunks per
// thread there is no ILP to hide it behind.
//
// Shared-memory double buffering is the textbook fix and does not fit: a second
// KV stage costs 33280 B and takes the block to 107392 B against the 101376 B
// sm_12x ceiling. So the pipeline lives in REGISTERS instead -- tile i+1 is
// loaded global->register while tile i computes, then stored register->smem at
// the top of the next iteration. Costs 8 uint4 = 32 registers/lane (M3 used 128
// of 255, so there is room) and zero extra shared memory.
//
// Chunk mapping is deliberately left as M3's strided one (thread t owns chunks
// t, t+256, t+512, ...). Giving each thread 8 consecutive chunks of one row was
// tried first -- it halves the index loads and looks better coalesced -- but it
// puts all 8 threads of a row 128 B apart, i.e. exactly 32 words, so every
// shared-memory store lands on the same four banks and the tile write serialises
// 8 ways. Measured 1.35x SLOWER than M3. The strided mapping also keeps 8
// independent index loads in flight per thread, which is memory-level
// parallelism the single-index version gives up.
//
// M3 header follows.
//
// Ablation at T=2048 said the tensor cores were never the problem: GEMM1 cost
// 0.89 ms and GEMM2 0.60 ms of a 5.72 ms kernel, while gather + softmax alone
// cost 3.94 ms (69%). The gather moved KV two bytes at a time and re-read the
// index from global for every one of the 64 elements a thread copied. Here it
// moves 16 bytes at a time (uint4) with one index load per 8-element chunk, and
// the epilogue packs adjacent d-pairs into 32-bit stores.
//
// M2 header follows.
// sparse MLA (NoPE, tail_dim == 0) for sm_120/121 -- tensor cores.
//
// Same structure and same numerics as M1; the two GEMMs now run on bf16 tensor
// cores via mma.sync.m16n8k16. sm_120/121 are consumer/embedded Blackwell: no
// tcgen05, no TMEM, no TMA, no wgmma, so the warp-level mma.sync family is the
// tensor-core path available. (TileLang concedes the same point by disabling
// TMA lowering and warp specialization for this target.)
//
// Layout, per block: one token, 32 heads, 256 threads = 8 warps.
//   GEMM1  S[32h, 32i] = Q[32h, 512d] . KV[32i, 512d]^T
//          M=32 -> 2 m-tiles, N=32 -> 4 n-tiles  =>  8 warp-tiles for 8 warps,
//          each reducing the full K=512 itself, so no cross-warp reduction.
//   GEMM2  acc[32h, 512d] += P[32h, 32i] . KV[32i, 512d]
//          warp w owns m-tile w/4 and 16 of the 64 d-tiles => 64 acc regs/lane.
//
// Shared memory (padded, single KV stage):
//   Q  32 x 520 bf16 = 33280      KV 32 x 520 bf16 = 33280
//   S  32 x  36 f32  =  4608      P  32 x  40 bf16 =  2560
//   + 3 x 32 f32 scratch          => 69632 B, comfortably under the 101376 ceiling.
//
// The kPad = 8 halves (4 words) is load-bearing, not decoration: fragment loads
// address rows (lane>>2) at a stride of 520 halves = 260 words, and 260 % 32 = 4,
// so the 8 rows of a fragment land on 8 distinct bank groups and the 32 lanes
// cover 32 distinct banks. Remove the pad and every fragment load conflicts.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/all.h>

// c10 rather than ATen/cuda/CUDAContext.h: the latter pulls in
// CUDAContextLight.h -> cusparse.h, and the deployed vLLM image ships a CUDA
// toolkit without the cuSPARSE headers, so including it fails the build there.
// All we need is the current stream and the launch check.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {

// Heads per rank is a template parameter so one kernel serves several TP
// degrees for a 64-head model: TP=2 gives 32, TP=4 gives 16, TP=8 gives 8.
//
// mma.m16n8k16 has a minimum M of 16, so fewer than 16 heads are PADDED to 16
// and the padding rows are computed and discarded. That wastes half the tensor
// core at 8 heads, which is the price of running this model at TP=8 at all.
constexpr int kDim = 512;
constexpr int kBI = 32;

constexpr int padded_heads(int h) { return h >= 16 ? h : 16; }
constexpr int m_tiles(int h) { return padded_heads(h) / 16; }
constexpr int n_tiles = kBI / 8;                       // 4
constexpr int threads_for(int h) { return m_tiles(h) * n_tiles * 32; }
constexpr int kQS = kDim + 8;     // Q row stride, halves
constexpr int kKS = kDim + 8;     // KV row stride, halves
constexpr int kSS = kBI + 4;      // S row stride, floats
constexpr int kPS = kBI + 8;      // P row stride, halves
constexpr float kLog2e = 1.44269504f;
constexpr float kNegBig = -(1 << 30);

__device__ __forceinline__ void mma_m16n8k16(float& d0, float& d1, float& d2, float& d3,
                                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                            uint32_t& r3, const __nv_bfloat16* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr(p)));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1,
                                            const __nv_bfloat16* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1) : "r"(smem_addr(p)));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1,
                                                  const __nv_bfloat16* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1) : "r"(smem_addr(p)));
}

__device__ __forceinline__ uint32_t ld32(const __nv_bfloat16* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  uint32_t r;
  uint16_t l = *reinterpret_cast<const uint16_t*>(&lo);
  uint16_t h = *reinterpret_cast<const uint16_t*>(&hi);
  r = (uint32_t)l | ((uint32_t)h << 16);
  return r;
}

// The KV cache is read either as bf16 (2 B/element) or as fp8_e4m3 (1 B), so a
// 16-byte gather chunk covers 8 or 16 elements respectively. fp8 halves the
// global traffic of the gather, which the ablation showed is the dominant cost,
// and doubles the KV capacity for a given pool. Both variants land the SAME
// bf16 tile in shared memory, so the MMA path below is untouched.
template <typename KVT>
struct KvTraits;

template <>
struct KvTraits<__nv_bfloat16> {
  static constexpr int kElemsPerChunk = 8;         // 16 B / 2 B
  static constexpr bool kQuantized = false;
};

template <>
struct KvTraits<__nv_fp8_e4m3> {
  static constexpr int kElemsPerChunk = 16;        // 16 B / 1 B
  static constexpr bool kQuantized = true;
};

template <typename KVT, int THREADS>
__device__ __forceinline__ void load_tile(
    uint4 (&pf)[kBI * (kDim / KvTraits<KVT>::kElemsPerChunk) / THREADS],
    const KVT* __restrict__ KV, const int32_t* __restrict__ idx_row,
    int tile, int tid) {
  constexpr int CPR = kDim / KvTraits<KVT>::kElemsPerChunk;
  constexpr int NPF = kBI * CPR / THREADS;
#pragma unroll
  for (int i = 0; i < NPF; ++i) {
    const int c = tid + i * THREADS;
    const int32_t raw = idx_row[tile * kBI + c / CPR];
    const int64_t row = raw < 0 ? 0 : (int64_t)raw;   // clamp: KV[-1] is OOB
    pf[i] = *reinterpret_cast<const uint4*>(
        KV + row * kDim + (c % CPR) * KvTraits<KVT>::kElemsPerChunk);
  }
}

// Publish a prefetched tile into the bf16 shared tile, dequantizing on the way
// when the cache is fp8. kv_scale is the per-tensor scale vLLM stores alongside
// the cache; it is 1.0f on the bf16 path.
template <typename KVT, int THREADS>
__device__ __forceinline__ void store_tile(
    const uint4 (&pf)[kBI * (kDim / KvTraits<KVT>::kElemsPerChunk) / THREADS],
    __nv_bfloat16* kv_s, int tid, float kv_scale) {
  constexpr int CPR = kDim / KvTraits<KVT>::kElemsPerChunk;
  constexpr int NPF = kBI * CPR / THREADS;
  constexpr int EPC = KvTraits<KVT>::kElemsPerChunk;
#pragma unroll
  for (int i = 0; i < NPF; ++i) {
    const int c = tid + i * THREADS;
    __nv_bfloat16* dst = kv_s + (c / CPR) * kKS + (c % CPR) * EPC;
    if constexpr (!KvTraits<KVT>::kQuantized) {
      *reinterpret_cast<uint4*>(dst) = pf[i];
    } else {
      // Scale here rather than folding it into sm_scale and the epilogue.
      // Folding was tried: it removes one multiply per element, measured ZERO
      // speedup (the path is not multiply-bound), and pushed two cases past the
      // correctness tolerance because acc_o then accumulates unscaled values up
      // to fp8 max. Not worth it.
      const __nv_fp8_e4m3* src = reinterpret_cast<const __nv_fp8_e4m3*>(&pf[i]);
#pragma unroll
      for (int e = 0; e < EPC; ++e)
        dst[e] = __float2bfloat16(static_cast<float>(src[e]) * kv_scale);
    }
  }
}

template <int kHeads, typename KVT>
__global__ __launch_bounds__(threads_for(kHeads)) void sparse_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const KVT* __restrict__ KV,
    const int32_t* __restrict__ Indices,
    __nv_bfloat16* __restrict__ O,
    int topk,
    float sm_scale,
    float kv_scale) {
  constexpr int kPH = padded_heads(kHeads);      // rows the MMA actually covers
  constexpr int kThreads = threads_for(kHeads);

  extern __shared__ char smem_raw[];
  __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* kv_s = q_s + kPH * kQS;
  float* s_s = reinterpret_cast<float*>(kv_s + kBI * kKS);
  __nv_bfloat16* p_s = reinterpret_cast<__nv_bfloat16*>(s_s + kPH * kSS);
  float* alpha = reinterpret_cast<float*>(p_s + kPH * kPS);
  float* m_state = alpha + kPH;
  float* sum_state = m_state + kPH;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int token = blockIdx.x;

  const int mt = warp / n_tiles;     // GEMM1/GEMM2 m-tile: heads [mt*16, +16)
  const int nt = warp % n_tiles;     // GEMM1 n-tile: indices [nt*8, +8)
  const int lr = lane >> 2;          // fragment row within an 8-row group
  const int lc = (lane & 3) * 2;     // fragment column pair

  // ---- stage Q once; it is reused by every index tile ---------------------
  {
    constexpr int kQChunks = kDim / 8;
    const __nv_bfloat16* q_base = Q + (int64_t)token * kHeads * kDim;
#pragma unroll
    for (int c = tid; c < kPH * kQChunks; c += kThreads) {
      const int h = c / kQChunks, dc = c % kQChunks;
      uint4* dst = reinterpret_cast<uint4*>(q_s + h * kQS + dc * 8);
      if (h < kHeads) {
        *dst = *reinterpret_cast<const uint4*>(q_base + (int64_t)h * kDim + dc * 8);
      } else {
        // Padding rows must be ZERO, not stale smem: the MMA reads all kPH rows
        // and their scores would otherwise be garbage that perturbs nothing we
        // keep, but could produce inf/NaN in the discarded lanes.
        *dst = make_uint4(0, 0, 0, 0);
      }
    }
  }
  if (tid < kPH) {
    m_state[tid] = kNegBig;
    sum_state[tid] = 0.f;
    alpha[tid] = 1.f;
  }

  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

  const int32_t* idx_row = Indices + (int64_t)token * topk;
  const int num_tiles = topk / kBI;

  constexpr int kCPR = kDim / KvTraits<KVT>::kElemsPerChunk;
  constexpr int kNPF = kBI * kCPR / kThreads;
  uint4 pf[kNPF];
  load_tile<KVT, kThreads>(pf, KV, idx_row, 0, tid);        // prime the pipeline

  for (int tile = 0; tile < num_tiles; ++tile) {
    // ---- gather the KV tile ----------------------------------------------
    // publish the tile prefetched during the previous iteration
    __syncthreads();                     // everyone finished reading kv_s
    store_tile<KVT, kThreads>(pf, kv_s, tid, kv_scale);
    __syncthreads();

    // issue the next tile's global loads now; GEMM1 + softmax + GEMM2 below hide
    // the index->KV dependent latency.
    if (tile + 1 < num_tiles) load_tile<KVT, kThreads>(pf, KV, idx_row, tile + 1, tid);

    // ---- GEMM1 ------------------------------------------------------------
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    {
      // ldmatrix.x4: lane L supplies row (L & 15) of the 16x16 A tile, column
      // block (L >> 4) * 8, which returns r0..r3 in exactly the a0..a3 order
      // mma.m16n8k16 expects (rows 0-7/8-15 x cols 0-7/8-15).
      const __nv_bfloat16* qa = q_s + (mt * 16 + (lane & 15)) * kQS + (lane >> 4) * 8;
      // B is [n = index][k = dim], already the fragment's (row, col) order, so
      // plain non-trans .x2 works: lanes 0-7 give k 0-7 and lanes 8-15 k 8-15
      // for the same eight index rows.
      const __nv_bfloat16* kb = kv_s + (nt * 8 + (lane & 7)) * kKS + ((lane >> 3) & 1) * 8;
#pragma unroll 4
      for (int k = 0; k < kDim; k += 16) {
        uint32_t a0, a1, a2, a3, b0, b1;
        ldmatrix_x4(a0, a1, a2, a3, qa + k);
        ldmatrix_x2(b0, b1, kb + k);
        mma_m16n8k16(s0, s1, s2, s3, a0, a1, a2, a3, b0, b1);
      }
    }
    s_s[(mt * 16 + lr) * kSS + nt * 8 + lc] = s0;
    s_s[(mt * 16 + lr) * kSS + nt * 8 + lc + 1] = s1;
    s_s[(mt * 16 + lr + 8) * kSS + nt * 8 + lc] = s2;
    s_s[(mt * 16 + lr + 8) * kSS + nt * 8 + lc + 1] = s3;
    __syncthreads();

    // ---- online softmax, one owner thread per head ------------------------
    // Identical arithmetic to M1: exp2(x*c - m*c), two multiplies, matching the
    // reference (tilelang_kernel.py:405-407) rather than exp2((x-m)*c).
    if (tid < kPH) {
      const int h = tid;
      const float m_prev = m_state[h];
      float m_cur = m_prev;
      float row[kBI];
#pragma unroll
      for (int i = 0; i < kBI; ++i) {
        const bool valid = idx_row[tile * kBI + i] >= 0;
        row[i] = valid ? s_s[h * kSS + i] : -INFINITY;
        m_cur = fmaxf(m_cur, row[i]);   // reduce_max(clear=False)
      }
      const float c = sm_scale * kLog2e;
      const float a = exp2f((m_prev - m_cur) * c);
      float ssum = 0.f;
#pragma unroll
      for (int i = 0; i < kBI; ++i) {
        const float p = isinf(row[i]) ? 0.f : exp2f(row[i] * c - m_cur * c);
        p_s[h * kPS + i] = __float2bfloat16(p);
        ssum += p;
      }
      alpha[h] = a;
      m_state[h] = m_cur;
      sum_state[h] = sum_state[h] * a + ssum;
    }
    __syncthreads();

    // ---- rescale, then GEMM2 ---------------------------------------------
    const float a_lo = alpha[mt * 16 + lr];
    const float a_hi = alpha[mt * 16 + lr + 8];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      acc[j][0] *= a_lo; acc[j][1] *= a_lo;
      acc[j][2] *= a_hi; acc[j][3] *= a_hi;
    }

    {
      const __nv_bfloat16* pa = p_s + (mt * 16 + (lane & 15)) * kPS + (lane >> 4) * 8;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const int i0 = ks * 16;
        uint32_t a0, a1, a2, a3;
        ldmatrix_x4(a0, a1, a2, a3, pa + i0);
        // B wants KV[k = index][n = dim] with n as the fragment row -- the
        // transpose of the [index][dim] smem layout, which is exactly what
        // ldmatrix.trans does. One instruction replaces eight 16-bit loads.
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int d0 = ((warp & 3) * 16 + j) * 8;
          uint32_t b0, b1;
          ldmatrix_x2_trans(b0, b1, kv_s + (i0 + (lane & 15)) * kKS + d0);
          mma_m16n8k16(acc[j][0], acc[j][1], acc[j][2], acc[j][3],
                       a0, a1, a2, a3, b0, b1);
        }
      }
    }
  }

  // ---- epilogue ----------------------------------------------------------
  // Fully masked token -> sumexp == 0 -> return exactly zero, not 0/0 = NaN.
  const float dlo = sum_state[mt * 16 + lr];
  const float dhi = sum_state[mt * 16 + lr + 8];
  const float ilo = dlo > 0.f ? 1.f / dlo : 0.f;
  const float ihi = dhi > 0.f ? 1.f / dhi : 0.f;
  const bool row_lo_valid = (mt * 16 + lr) < kHeads;
  const bool row_hi_valid = (mt * 16 + lr + 8) < kHeads;
  __nv_bfloat16* o_base = O + (int64_t)token * kHeads * kDim;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int d = ((warp & 3) * 16 + j) * 8 + lc;
    __nv_bfloat16* olo = o_base + (int64_t)(mt * 16 + lr) * kDim;
    __nv_bfloat16* ohi = o_base + (int64_t)(mt * 16 + lr + 8) * kDim;
    // acc[j][0]/[1] and [2]/[3] are adjacent in d, so pack each pair into one
    // 32-bit store instead of four 16-bit ones.
    if (row_lo_valid)
      *reinterpret_cast<uint32_t*>(olo + d) =
          pack2(__float2bfloat16(acc[j][0] * ilo), __float2bfloat16(acc[j][1] * ilo));
    if (row_hi_valid)
      *reinterpret_cast<uint32_t*>(ohi + d) =
          pack2(__float2bfloat16(acc[j][2] * ihi), __float2bfloat16(acc[j][3] * ihi));
  }
}

}  // namespace

// One launch helper per KV dtype. The shared-memory opt-in is raised once per
// specialization via a function-local static: cudaFuncSetAttribute is not
// stream-ordered and must not be issued inside a CUDA graph capture.
template <int H>
static constexpr size_t smem_bytes() {
  constexpr int PH = padded_heads(H);
  return sizeof(__nv_bfloat16) * (PH * kQS + kBI * kKS + PH * kPS)
       + sizeof(float) * (PH * kSS + 3 * PH);
}

template <int H, typename KVT>
static void launch(const torch::Tensor& q, const torch::Tensor& kv,
                   const torch::Tensor& indices, torch::Tensor& out,
                   int T, int topk, float sm_scale, float kv_scale) {
  constexpr size_t smem = smem_bytes<H>();
  static_assert(smem <= 101376, "tile exceeds the sm_12x opt-in shared-memory ceiling");
  static const bool smem_configured = [&] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        sparse_fwd_kernel<H, KVT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    return true;
  }();
  (void)smem_configured;
  sparse_fwd_kernel<H, KVT><<<T, threads_for(H), smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const KVT*>(kv.data_ptr()),
      indices.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      topk, sm_scale, kv_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_fwd(torch::Tensor q, torch::Tensor kv, torch::Tensor indices,
                double sm_scale, double kv_scale, torch::Tensor out) {
  TORCH_CHECK(q.is_cuda() && kv.is_cuda() && indices.is_cuda() && out.is_cuda());
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 2 && indices.dim() == 2);
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(indices.scalar_type() == torch::kInt);
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && indices.is_contiguous());
  // Heads per rank is dispatched below. 64 heads (TP=1) is not built: it would
  // need 114944 B of shared memory against the 101376 B sm_12x ceiling.
  TORCH_CHECK(q.size(2) == kDim && kv.size(1) == kDim,
              "expected kv_lora_rank ", kDim, " with qk_rope_head_dim 0 (NoPE)");

  const int T = q.size(0);
  const int topk = indices.size(1);
  TORCH_CHECK(topk % kBI == 0);
  TORCH_CHECK(indices.size(0) == T && out.sizes() == q.sizes());

  if (T == 0) return;

#define GLM53_DISPATCH(H)                                                       \
  do {                                                                          \
    if (kv.scalar_type() == torch::kBFloat16)                                    \
      launch<H, __nv_bfloat16>(q, kv, indices, out, T, topk, (float)sm_scale,    \
                               1.0f);                                            \
    else if (kv.scalar_type() == torch::kFloat8_e4m3fn)                          \
      launch<H, __nv_fp8_e4m3>(q, kv, indices, out, T, topk, (float)sm_scale,    \
                               (float)kv_scale);                                 \
    else                                                                         \
      TORCH_CHECK(false, "kv must be bfloat16 or float8_e4m3fn, got ",           \
                  kv.scalar_type());                                             \
    return;                                                                      \
  } while (0)

  switch ((int)q.size(1)) {
    case 32: GLM53_DISPATCH(32);   // TP=2 for a 64-head model
    case 16: GLM53_DISPATCH(16);   // TP=4
    case 8:  GLM53_DISPATCH(8);    // TP=8, half the MMA rows are padding
    default:
      TORCH_CHECK(false,
                  "unsupported heads-per-rank ", q.size(1),
                  "; built for 8, 16 or 32 (TP=8/4/2 on a 64-head model)");
  }
#undef GLM53_DISPATCH
}

TORCH_LIBRARY(glm53_sparse_mla, m) {
  m.def("sparse_fwd(Tensor q, Tensor kv, Tensor indices, float sm_scale, float kv_scale, Tensor(a!) out) -> ()");
}
TORCH_LIBRARY_IMPL(glm53_sparse_mla, CUDA, m) {
  m.impl("sparse_fwd", &sparse_fwd);
}
