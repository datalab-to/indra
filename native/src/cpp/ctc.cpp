// The recognizers' CTC head on their logits [B, V, T] (the graph cut at the
// head conv, see indra_ocr/onnx_cut.py).  The ONNX graph runs eight ops over
// a [B, T, V] tensor (transpose, max, argmax, exp, sum, log-softmax, top-k);
// here one pass finds each step's best class and a second sums exp(x - max),
// a tile of steps at a time so the tile's logits stay in cache.

#include <algorithm>
#include <cmath>
#include <cstring>

#include "kernels.h"

#if defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h>
#define INDRA_SSE2 1
#endif

namespace indra {

namespace {

constexpr int TILE = 32;  // steps per tile: V x TILE floats stay in L2
// exp(x - max) under this is skipped: V terms below e^-26 change the sum by
// less than float32 rounding (V < 2e4).
constexpr float SKIP = -26.0f;

// exp of x <= 0 (Cephes polynomial, about 1 ulp), written so compilers can
// vectorize a loop over it.
inline float exp_neg(float x) {
  x = std::max(x, -87.0f);
  float fx = x * 1.44269504088896341f + 0.5f;
  float t = static_cast<float>(static_cast<int>(fx));
  fx = t - (t > fx ? 1.0f : 0.0f);
  x = x - fx * 0.693359375f - fx * -2.12194440e-4f;
  float z = x * x;
  float y = 1.9875691500e-4f;
  y = y * x + 1.3981999507e-3f;
  y = y * x + 8.3334519073e-3f;
  y = y * x + 4.1665795894e-2f;
  y = y * x + 1.6666665459e-1f;
  y = y * x + 5.0000001201e-1f;
  y = y * z + x + 1.0f;
  int32_t e = (static_cast<int32_t>(fx) + 127) << 23;
  float p;
  std::memcpy(&p, &e, sizeof p);
  return y * p;
}

#ifdef INDRA_SSE2
inline __m128 exp_neg4(__m128 x) {
  x = _mm_max_ps(x, _mm_set1_ps(-87.0f));
  __m128 fx = _mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(1.44269504088896341f)),
                         _mm_set1_ps(0.5f));
  __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(fx));
  fx = _mm_sub_ps(t, _mm_and_ps(_mm_cmpgt_ps(t, fx), _mm_set1_ps(1.0f)));
  x = _mm_sub_ps(x, _mm_mul_ps(fx, _mm_set1_ps(0.693359375f)));
  x = _mm_sub_ps(x, _mm_mul_ps(fx, _mm_set1_ps(-2.12194440e-4f)));
  __m128 z = _mm_mul_ps(x, x);
  __m128 y = _mm_set1_ps(1.9875691500e-4f);
  y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(1.3981999507e-3f));
  y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(8.3334519073e-3f));
  y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(4.1665795894e-2f));
  y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(1.6666665459e-1f));
  y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(5.0000001201e-1f));
  y = _mm_add_ps(_mm_add_ps(_mm_mul_ps(y, z), x), _mm_set1_ps(1.0f));
  __m128i e = _mm_slli_epi32(
      _mm_add_epi32(_mm_cvttps_epi32(fx), _mm_set1_epi32(127)), 23);
  return _mm_mul_ps(y, _mm_castsi128_ps(e));
}
#endif

// Per step t < w of a tile starting at row pointer L (rows of stride T):
// the max logit m, its class am (first on ties), and s = sum exp(x - m).
void tile_stats(const float* L, int V, int T, int w, float* m, int32_t* am,
                float* s) {
  for (int t = 0; t < w; ++t) m[t] = L[t], am[t] = 0, s[t] = 0.0f;
  for (int v = 1; v < V; ++v) {
    const float* row = L + static_cast<size_t>(v) * T;
    int t = 0;
#ifdef INDRA_SSE2
    __m128i vv = _mm_set1_epi32(v);
    for (; t + 4 <= w; t += 4) {
      __m128 x = _mm_loadu_ps(row + t), mm = _mm_loadu_ps(m + t);
      __m128 gt = _mm_cmpgt_ps(x, mm);
      _mm_storeu_ps(m + t, _mm_or_ps(_mm_and_ps(gt, x), _mm_andnot_ps(gt, mm)));
      __m128i g = _mm_castps_si128(gt);
      __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(am + t));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(am + t),
                       _mm_or_si128(_mm_and_si128(g, vv), _mm_andnot_si128(g, a)));
    }
#endif
    for (; t < w; ++t) {
      bool gt = row[t] > m[t];
      m[t] = gt ? row[t] : m[t];
      am[t] = gt ? v : am[t];
    }
  }
  for (int v = 0; v < V; ++v) {
    const float* row = L + static_cast<size_t>(v) * T;
    int t = 0;
#ifdef INDRA_SSE2
    __m128 skip = _mm_set1_ps(SKIP);
    for (; t + 4 <= w; t += 4) {
      __m128 d = _mm_sub_ps(_mm_loadu_ps(row + t), _mm_loadu_ps(m + t));
      if (_mm_movemask_ps(_mm_cmpgt_ps(d, skip)) == 0) continue;
      _mm_storeu_ps(s + t, _mm_add_ps(_mm_loadu_ps(s + t), exp_neg4(d)));
    }
#endif
    for (; t < w; ++t) s[t] += exp_neg(row[t] - m[t]);
  }
}

}  // namespace

void ctc_greedy(const float* logits, int B, int V, int T,
                const int64_t* lengths, int64_t* ids, float* conf) {
  float m[TILE], s[TILE];
  int32_t am[TILE];
  std::fill(ids, ids + static_cast<size_t>(B) * T, 0);
  std::fill(conf, conf + static_cast<size_t>(B) * T, 0.0f);
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

void ctc_topk(const float* logits, int V, int T, int n, int K, float* blank,
              float* topv, int64_t* topi) {
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
