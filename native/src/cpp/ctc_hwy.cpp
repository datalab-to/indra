// The CTC head's tile kernel, compiled by Google Highway for every SIMD target
// it supports (SSE2 to AVX-512 on x86, NEON and SVE on Arm) and dispatched at
// run time to the best one the CPU has.  Per step of a tile: the best class
// (the first on ties: an exact comparison on every target) and the sum of
// exp(x - max) (Highway's Exp, 1 ulp; a target with FMA rounds a little
// differently, well inside float32 summation noise).

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ctc_hwy.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// foreach_target re-includes this file once per target, then once more.
#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernels.h"

HWY_BEFORE_NAMESPACE();
namespace indra {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// exp(x - max) under this is skipped: V terms below e^-26 change the sum by
// less than float32 rounding (V < 2e4).
constexpr float kSkipBelowMax = -26.0f;

// One vector of steps [t, t + n): full vectors load and store whole, the tail
// with LoadN / StoreN (n lanes).
template <bool kFull, class DF, class DI, class VI>
HWY_INLINE void MaxStep(DF df, DI di, const float* row, float* m, int32_t* am, VI vv,
                        size_t n) {
  const auto x = kFull ? hn::LoadU(df, row) : hn::LoadN(df, row, n);
  const auto mm = kFull ? hn::LoadU(df, m) : hn::LoadN(df, m, n);
  const auto gt = hn::Gt(x, mm);  // strictly greater: the first maximum stays
  const auto a = kFull ? hn::LoadU(di, am) : hn::LoadN(di, am, n);
  const auto keep_m = hn::IfThenElse(gt, x, mm);
  const auto keep_a = hn::IfThenElse(hn::RebindMask(di, gt), vv, a);
  if (kFull) {
    hn::StoreU(keep_m, df, m);
    hn::StoreU(keep_a, di, am);
  } else {
    hn::StoreN(keep_m, df, m, n);
    hn::StoreN(keep_a, di, am, n);
  }
}

template <bool kFull, class DF>
HWY_INLINE void SumStep(DF df, const float* row, const float* m, float* s, size_t n) {
  const auto d = kFull ? hn::Sub(hn::LoadU(df, row), hn::LoadU(df, m))
                       : hn::Sub(hn::LoadN(df, row, n), hn::LoadN(df, m, n));
  auto live = hn::Gt(d, hn::Set(df, kSkipBelowMax));
  if (!kFull) live = hn::And(live, hn::FirstN(df, n));
  if (hn::AllFalse(df, live)) return;
  if (kFull) {
    hn::StoreU(hn::Add(hn::LoadU(df, s), hn::Exp(df, d)), df, s);
  } else {
    hn::StoreN(hn::Add(hn::LoadN(df, s, n), hn::Exp(df, d)), df, s, n);
  }
}

void TileStats(const float* HWY_RESTRICT L, int V, int T, int w, float* HWY_RESTRICT m,
               int32_t* HWY_RESTRICT am, float* HWY_RESTRICT s) {
  const hn::ScalableTag<float> df;
  const hn::RebindToSigned<decltype(df)> di;
  const int N = static_cast<int>(hn::Lanes(df));
  for (int t = 0; t < w; ++t) m[t] = L[t], am[t] = 0, s[t] = 0.0f;
  for (int v = 1; v < V; ++v) {
    const float* row = L + static_cast<size_t>(v) * T;
    const auto vv = hn::Set(di, v);
    int t = 0;
    for (; t + N <= w; t += N) MaxStep<true>(df, di, row + t, m + t, am + t, vv, 0);
    if (t < w) MaxStep<false>(df, di, row + t, m + t, am + t, vv, w - t);
  }
  for (int v = 0; v < V; ++v) {
    const float* row = L + static_cast<size_t>(v) * T;
    int t = 0;
    for (; t + N <= w; t += N) SumStep<true>(df, row + t, m + t, s + t, 0);
    if (t < w) SumStep<false>(df, row + t, m + t, s + t, w - t);
  }
}

const char* TargetName() { return hwy::TargetName(HWY_TARGET); }

}  // namespace HWY_NAMESPACE
}  // namespace indra
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace indra {

HWY_EXPORT(TileStats);
HWY_EXPORT(TargetName);

void tile_stats(const float* L, int V, int T, int w, float* m, int32_t* am, float* s) {
  HWY_DYNAMIC_DISPATCH(TileStats)(L, V, T, w, m, am, s);
}

std::string ctc_isa() { return HWY_DYNAMIC_DISPATCH(TargetName)(); }

std::vector<std::string> ctc_isas() {
  std::vector<std::string> out;
  for (int64_t t : hwy::SupportedAndGeneratedTargets()) out.emplace_back(hwy::TargetName(t));
  return out;
}

void set_ctc_isa(const std::string& name) {
  hwy::SetSupportedTargetsForTest(0);  // back to the CPU's own targets, to choose from
  if (name == "best") return;
  for (int64_t t : hwy::SupportedAndGeneratedTargets()) {
    if (name == hwy::TargetName(t)) {
      hwy::SetSupportedTargetsForTest(t);
      return;
    }
  }
  throw std::invalid_argument("set_ctc_isa: " + name + " is not compiled in or not on this CPU");
}

}  // namespace indra
#endif  // HWY_ONCE
