// order_post._cluster and the cut points of pipeline.split_line.

#include <algorithm>
#include <limits>

#include "kernels.h"

namespace indra {

std::vector<std::vector<int>> cluster(const double* logits, int n) {
  // Same steps as the numpy code, element by element (each element of S and
  // M sees the same float64 operations), and argmax keeps the first maximum.
  const double NEG_INF = -std::numeric_limits<double>::infinity();
  std::vector<std::vector<int>> out;
  if (n == 0) return out;
  const size_t N = static_cast<size_t>(n);
  std::vector<double> S(logits, logits + N * N);
  for (size_t i = 0; i < N; ++i) S[i * N + i] = 0.0;
  std::vector<double> size(N, 1.0);
  std::vector<char> alive(N, 1);
  std::vector<std::vector<int>> members(N);
  for (int i = 0; i < n; ++i) members[i] = {i};
  std::vector<double> M(S);
  for (size_t i = 0; i < N; ++i) M[i * N + i] = NEG_INF;
  auto argmax = [&](const double* r) {
    size_t k = 0;
    for (size_t j = 1; j < N; ++j)
      if (r[j] > r[k]) k = j;
    return k;
  };
  std::vector<size_t> best(N);
  std::vector<double> best_v(N);
  for (size_t i = 0; i < N; ++i) {
    best[i] = argmax(&M[i * N]);
    best_v[i] = M[i * N + best[i]];
  }
  std::vector<double> row(N);
  for (int it = 0; it < n - 1; ++it) {
    size_t a = 0;
    for (size_t j = 1; j < N; ++j)
      if (best_v[j] > best_v[a]) a = j;
    size_t b = best[a];
    if (best_v[a] <= 0) break;
    if (a > b) std::swap(a, b);
    for (size_t j = 0; j < N; ++j) S[a * N + j] += S[b * N + j];
    for (size_t i = 0; i < N; ++i) S[i * N + a] += S[i * N + b];
    S[a * N + a] = 0.0;
    size[a] += size[b];
    alive[b] = 0;
    members[a].insert(members[a].end(), members[b].begin(), members[b].end());
    for (size_t j = 0; j < N; ++j) {
      row[j] = S[a * N + j] / (size[a] * size[j]);
      if (!alive[j]) row[j] = NEG_INF;
    }
    row[a] = NEG_INF;
    for (size_t j = 0; j < N; ++j) M[a * N + j] = row[j], M[j * N + a] = row[j];
    for (size_t j = 0; j < N; ++j) M[b * N + j] = NEG_INF, M[j * N + b] = NEG_INF;
    best_v[b] = NEG_INF;
    best[a] = argmax(row.data());
    best_v[a] = row[best[a]];
    // other rows: (j, a) changed and (j, b) is gone
    for (size_t j = 0; j < N; ++j) {
      if (j == a || !alive[j] || !(best[j] == a || best[j] == b)) continue;
      best[j] = argmax(&M[j * N]);
      best_v[j] = M[j * N + best[j]];
    }
    for (size_t j = 0; j < N; ++j) {
      if (j == a || !alive[j] || !(row[j] > best_v[j])) continue;
      best[j] = a;
      best_v[j] = row[j];
    }
  }
  for (size_t i = 0; i < N; ++i) {
    if (!alive[i]) continue;
    std::sort(members[i].begin(), members[i].end());
    out.push_back(members[i]);
  }
  return out;
}

std::vector<int> split_points(const double* xs, const uint8_t* space, int n,
                              double limit, const double* at, int n_at) {
  std::vector<int> cuts;
  bool have_prev = false;
  double prev = 0;
  for (int k = 0; k < n; ++k) {
    if (space[k]) continue;
    double x = xs[k];
    if (have_prev) {
      bool cut = x - prev > limit;
      for (int i = 0; !cut && i < n_at; ++i) cut = prev < at[i] && at[i] < x;
      if (cut) cuts.push_back(k);
    }
    have_prev = true;
    prev = x;
  }
  return cuts;
}

}  // namespace indra
