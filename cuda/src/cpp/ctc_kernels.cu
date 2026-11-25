#include <cfloat>
#include <climits>

#include "common.h"
#include "ctc_kernels.h"

namespace indra_cuda {

namespace {

constexpr int kWarp = 32;
constexpr int kBlock = 256;  // 8 warps: one frame each
constexpr unsigned kAll = 0xffffffffu;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(kAll, v, o));
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(kAll, v, o);
  return v;
}

// One warp per frame: the row's max, then the sum of exp(x - max); lanes read
// consecutive classes (coalesced).  The row stays in L1/L2 between the passes.
__global__ void __launch_bounds__(kBlock) frame_conf_k(const float* __restrict__ logits,
                                                       const int64_t* __restrict__ lengths, int B, int T,
                                                       int C, float* __restrict__ conf) {
  const int f = (blockIdx.x * kBlock + threadIdx.x) / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (f >= B * T) return;
  const int b = f / T, t = f % T;
  if (t >= lengths[b]) {
    if (lane == 0) conf[f] = 0.f;
    return;
  }
  const float* x = logits + static_cast<size_t>(f) * C;
  float m = -FLT_MAX;
  for (int c = lane; c < C; c += kWarp) m = fmaxf(m, x[c]);
  m = warp_max(m);
  float s = 0.f;
  for (int c = lane; c < C; c += kWarp) s += expf(x[c] - m);
  s = warp_sum(s);
  if (lane == 0) conf[f] = 1.f / s;
}

struct Cand {
  float v;
  int i;
};

// a before b: the higher value, or the lower class at equal values
__device__ __forceinline__ bool before(float av, int ai, float bv, int bi) {
  return av > bv || (av == bv && ai < bi);
}

// One warp per (line, frame).  Each lane keeps its K best classes (sorted, in
// registers), then K rounds of a warp-wide pick of the best head.
template <int K>
__global__ void __launch_bounds__(kBlock) lm_inputs_k(const float* __restrict__ logits,
                                                      const int32_t* __restrict__ rows,
                                                      const int32_t* __restrict__ n, int R, int T, int C,
                                                      int k, float* __restrict__ blank,
                                                      float* __restrict__ topv, int64_t* __restrict__ topi) {
  const int f = (blockIdx.x * kBlock + threadIdx.x) / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (f >= R * T) return;
  const int r = f / T, t = f % T;
  float* bo = blank + f;
  float* vo = topv + static_cast<size_t>(f) * k;
  int64_t* io = topi + static_cast<size_t>(f) * k;
  if (t >= n[r]) {
    if (lane == 0) *bo = 0.f;
    if (lane < k) vo[lane] = -INFINITY, io[lane] = 0;
    return;
  }
  const float* x = logits + (static_cast<size_t>(rows[r]) * T + t) * C;
  float m = -FLT_MAX;
  for (int c = lane; c < C; c += kWarp) m = fmaxf(m, x[c]);
  m = warp_max(m);
  float s = 0.f;
  for (int c = lane; c < C; c += kWarp) s += expf(x[c] - m);
  s = warp_sum(s);
  const float logs = logf(s);
  if (lane == 0) *bo = (x[0] - m) - logs;
  float v[K];
  int ix[K];
#pragma unroll
  for (int j = 0; j < K; ++j) v[j] = -INFINITY, ix[j] = INT_MAX;
  // classes in increasing order per lane: a later equal value never moves up
  for (int c = lane; c < C; c += kWarp) {
    const float val = x[c];
    if (!(val > v[K - 1])) continue;
    int p = K - 1;
#pragma unroll
    for (int j = K - 1; j > 0; --j) {
      if (p == j && val > v[j - 1]) {
        v[j] = v[j - 1], ix[j] = ix[j - 1];
        p = j - 1;
      }
    }
    v[p] = val, ix[p] = c;
  }
  for (int j = 0; j < k; ++j) {
    float bv = v[0];
    int bi = ix[0];
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor_sync(kAll, bv, o);
      const int oi = __shfl_xor_sync(kAll, bi, o);
      if (before(ov, oi, bv, bi)) bv = ov, bi = oi;
    }
    if (ix[0] == bi) {  // the lane whose head won: pop it
#pragma unroll
      for (int q = 0; q < K - 1; ++q) v[q] = v[q + 1], ix[q] = ix[q + 1];
      v[K - 1] = -INFINITY, ix[K - 1] = INT_MAX;
    }
    if (lane == 0) {
      const bool found = bi != INT_MAX;
      vo[j] = found ? (bv - m) - logs : -INFINITY;
      io[j] = found ? bi : 0;
    }
  }
}

// One block per kept row, 4-byte words (rows are float32 / int64 tensors).
__global__ void gather_rows_k(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst,
                              const int32_t* __restrict__ rows, size_t words) {
  const uint32_t* s = src + static_cast<size_t>(rows[blockIdx.x]) * words;
  uint32_t* d = dst + static_cast<size_t>(blockIdx.x) * words;
  for (size_t i = threadIdx.x; i < words; i += blockDim.x) d[i] = s[i];
}

}  // namespace

void gather_rows(const void* src, void* dst, const int32_t* rows, int R, size_t row_bytes, cudaStream_t stream) {
  if (R == 0 || row_bytes == 0) return;
  if (row_bytes % 4) throw std::invalid_argument("indra-ocr-cuda: gather_rows takes rows of whole 4-byte words");
  gather_rows_k<<<R, 256, 0, stream>>>(static_cast<const uint32_t*>(src), static_cast<uint32_t*>(dst), rows,
                                       row_bytes / 4);
  INDRA_CK(cudaGetLastError());
}

void frame_conf(const float* logits, const int64_t* lengths, int B, int T, int C, float* conf,
                cudaStream_t stream) {
  const long frames = static_cast<long>(B) * T;
  if (frames == 0) return;
  const int blocks = static_cast<int>((frames * kWarp + kBlock - 1) / kBlock);
  frame_conf_k<<<blocks, kBlock, 0, stream>>>(logits, lengths, B, T, C, conf);
  INDRA_CK(cudaGetLastError());
}

void lm_inputs(const float* logits, const int32_t* rows, const int32_t* n, int R, int T, int C, int k,
               float* blank, float* topv, int64_t* topi, cudaStream_t stream) {
  const long frames = static_cast<long>(R) * T;
  if (frames == 0) return;
  if (k < 1 || k > 8) throw std::invalid_argument("indra-ocr-cuda: lm_inputs takes k from 1 to 8");
  const int blocks = static_cast<int>((frames * kWarp + kBlock - 1) / kBlock);
  lm_inputs_k<8><<<blocks, kBlock, 0, stream>>>(logits, rows, n, R, T, C, k, blank, topv, topi);
  INDRA_CK(cudaGetLastError());
}

}  // namespace indra_cuda
