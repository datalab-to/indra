// The recognizers' CTC head on their logits [B, V, T] (the graph cut at the
// head conv, see indra_ocr/onnx_cut.py).  The ONNX graph runs eight ops over
// a [B, T, V] tensor (transpose, max, argmax, exp, sum, log-softmax, top-k);
// here one pass finds each step's best class and a second sums exp(x - max),
// a tile of steps at a time so the tile's logits stay in cache (tile_stats,
// in ctc_hwy.cpp: SIMD for the CPU at hand).  One thread: the kernel runs in
// short bursts between ONNX Runtime calls, whose own thread pool uses the
// cores (OpenMP threads here made pages slower, measured in #13).

#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels.h"

namespace indra {

namespace {
constexpr int TILE = 32;  // steps per tile: V x TILE floats stay in L2
}  // namespace

void ctc_greedy(const float* logits, int B, int V, int T, const int64_t* lengths,
                int64_t* ids, float* conf) {
  std::fill(ids, ids + static_cast<size_t>(B) * T, 0);
  std::fill(conf, conf + static_cast<size_t>(B) * T, 0.0f);
  float m[TILE], s[TILE];
  int32_t am[TILE];
  for (int b = 0; b < B; ++b) {
    int n = static_cast<int>(std::min<int64_t>(T, std::max<int64_t>(lengths[b], 0)));
    const float* L = logits + static_cast<size_t>(b) * V * T;
    for (int t0 = 0; t0 < n; t0 += TILE) {
      int w = std::min(TILE, n - t0);
      tile_stats(L + t0, V, T, w, m, am, s);
      for (int t = 0; t < w; ++t) {
        ids[static_cast<size_t>(b) * T + t0 + t] = am[t];
        conf[static_cast<size_t>(b) * T + t0 + t] = 1.0f / s[t];
      }
    }
  }
}

void ctc_topk(const float* logits, int V, int T, int n, int K, float* blank, float* topv,
              int64_t* topi) {
  float m[TILE], s[TILE];
  int32_t am[TILE];
  std::vector<float> val(static_cast<size_t>(TILE) * K);
  std::vector<int32_t> idx(static_cast<size_t>(TILE) * K);
  std::vector<int> cnt(TILE);
  n = std::min(n, T);
  for (int t0 = 0; t0 < n; t0 += TILE) {
    int w = std::min(TILE, n - t0);
    const float* L = logits + t0;
    tile_stats(L, V, T, w, m, am, s);
    std::fill(cnt.begin(), cnt.end(), 0);
    for (int v = 0; v < V; ++v) {
      const float* row = L + static_cast<size_t>(v) * T;
      for (int t = 0; t < w; ++t) {
        float x = row[t];
        float* tv = &val[static_cast<size_t>(t) * K];
        int32_t* ti = &idx[static_cast<size_t>(t) * K];
        int c = cnt[t];
        if (c == K && !(x > tv[K - 1])) continue;
        // after every kept value >= x: an equal value keeps its lower class first
        int p = c < K ? c : K - 1;
        while (p > 0 && tv[p - 1] < x) {
          tv[p] = tv[p - 1], ti[p] = ti[p - 1];
          --p;
        }
        tv[p] = x, ti[p] = v;
        if (c < K) cnt[t] = c + 1;
      }
    }
    for (int t = 0; t < w; ++t) {
      float ls = std::log(s[t]);
      blank[t0 + t] = (L[t] - m[t]) - ls;
      for (int j = 0; j < K; ++j) {
        size_t o = static_cast<size_t>(t0 + t) * K + j;
        bool ok = j < cnt[t];
        topv[o] = ok ? (val[static_cast<size_t>(t) * K + j] - m[t]) - ls : -INFINITY;
        topi[o] = ok ? idx[static_cast<size_t>(t) * K + j] : 0;
      }
    }
  }
}

}  // namespace indra
