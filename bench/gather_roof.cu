// What is the achievable roof for the indexed KV gather alone, in isolation?
// If the roof is below ~2900 GB/s, no amount of overlap makes our kernel 1.3x
// faster than TileLang, because TileLang already runs its whole kernel at an
// effective ~2100 GB/s of gathered bytes.
#include <cuda_bf16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

namespace {
constexpr int kDim = 512;
constexpr int kBI = 32;

template <int THREADS>
__global__ __launch_bounds__(THREADS) void gather_only(
    const __nv_bfloat16* __restrict__ KV, const int32_t* __restrict__ Indices,
    __nv_bfloat16* __restrict__ sink, int topk) {
  extern __shared__ char raw[];
  __nv_bfloat16* kv_s = reinterpret_cast<__nv_bfloat16*>(raw);
  constexpr int KS = kDim + 8;
  constexpr int CPR = kDim / 8;
  const int tid = threadIdx.x;
  const int32_t* idx_row = Indices + (int64_t)blockIdx.x * topk;
  float acc = 0.f;
  for (int tile = 0; tile < topk / kBI; ++tile) {
    for (int c = tid; c < kBI * CPR; c += THREADS) {
      const int32_t r = idx_row[tile * kBI + c / CPR];
      const int64_t row = r < 0 ? 0 : (int64_t)r;
      *reinterpret_cast<uint4*>(kv_s + (c / CPR) * KS + (c % CPR) * 8) =
          *reinterpret_cast<const uint4*>(KV + row * kDim + (c % CPR) * 8);
    }
    __syncthreads();
    acc += __bfloat162float(kv_s[tid]);   // keep the store alive
    __syncthreads();
  }
  if (acc == 12345.678f) sink[0] = __float2bfloat16(acc);
}
}  // namespace

void gather_roof(torch::Tensor kv, torch::Tensor indices, torch::Tensor sink, int64_t threads) {
  const int T = indices.size(0), topk = indices.size(1);
  const size_t smem = sizeof(__nv_bfloat16) * kBI * (kDim + 8);
  auto st = at::cuda::getCurrentCUDAStream();
  auto* K = reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr());
  auto* S = reinterpret_cast<__nv_bfloat16*>(sink.data_ptr());
  auto* I = indices.data_ptr<int32_t>();
  if (threads == 128)      gather_only<128><<<T, 128, smem, st>>>(K, I, S, topk);
  else if (threads == 256) gather_only<256><<<T, 256, smem, st>>>(K, I, S, topk);
  else                     gather_only<512><<<T, 512, smem, st>>>(K, I, S, topk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(gather_roof_lib, m) {
  m.def("gather_roof(Tensor kv, Tensor indices, Tensor(a!) sink, int threads) -> ()");
}
TORCH_LIBRARY_IMPL(gather_roof_lib, CUDA, m) { m.impl("gather_roof", &gather_roof); }
