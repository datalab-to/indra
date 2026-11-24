// lm.NgramLM.logp_keys and lm.beam_search.  The results are the Python ones
// bit for bit: the same float64 operations in the same order, and the same
// tie-breaking (Python dicts keep insertion order; sorted() is stable).

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "kernels.h"

namespace indra {

namespace {
constexpr int TAG_SHIFT = 61;
constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

// lm._lse
inline double lse(double a, double b) {
  if (a == NEG_INF) return b;
  if (b == NEG_INF) return a;
  double m = std::max(a, b);
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

// value at key in a sorted table (numpy searchsorted, then an equality test)
inline bool find(const uint64_t* keys, const double* vals, size_t n,
                 uint64_t key, double& out) {
  const uint64_t* p = std::lower_bound(keys, keys + n, key);
  size_t j = std::min(static_cast<size_t>(p - keys), n - 1);
  if (n && keys[j] == key) {
    out = vals[j];
    return true;
  }
  out = 0.0;
  return false;
}
}  // namespace

NgramLM::NgramLM(int order_, int bits_, int64_t vocab_size, const uint64_t* p_keys,
                 const double* p_vals, size_t n_p, const uint64_t* h_keys,
                 const double* h_vals, size_t n_h, double max_logp_)
    : order(order_), bits(bits_), max_logp(max_logp_), vocab_size_(vocab_size),
      p_keys_(p_keys), h_keys_(h_keys), p_vals_(p_vals), h_vals_(h_vals),
      n_p_(n_p), n_h_(n_h) {}

double NgramLM::logp_key(uint64_t key) {
  auto hit = cache_.find(key);
  if (hit != cache_.end()) return hit->second;
  const int N = order;
  const uint64_t b = static_cast<uint64_t>(bits);
  const uint64_t ct = key & ((uint64_t(1) << b) - 1);
  std::vector<double> hb(N), pv(N);
  std::vector<bool> h_ok(N), p_ok(N);
  for (int n = 0; n < N; ++n) {
    uint64_t tag = static_cast<uint64_t>(n) << TAG_SHIFT;
    uint64_t mask = (uint64_t(1) << (b * static_cast<uint64_t>(n))) - 1;
    uint64_t hq = (key >> b) & mask;
    uint64_t pq = (hq << b) | ct | tag;
    h_ok[n] = find(h_keys_, h_vals_, n_h_, hq | tag, hb[n]);
    p_ok[n] = find(p_keys_, p_vals_, n_p_, pq, pv[n]);
  }
  h_ok[0] = true;
  // m: the longest history length the model has seen
  int m = 0;
  for (int n = 1; n < N && h_ok[n]; ++n) m = n;
  int chosen = -1;
  for (int n = 0; n <= m; ++n)
    if (p_ok[n]) chosen = n;
  // suffix[n] = backoffs of the seen histories n..N-1, summed from the end
  // (numpy's cumsum of the reversed row)
  std::vector<double> suffix(N + 1, 0.0);
  double acc = 0.0;
  for (int n = N - 1; n >= 0; --n) {
    double term = hb[n] * (n <= m ? 1.0 : 0.0);
    acc = (n == N - 1) ? term : acc + term;
    suffix[n] = acc;
  }
  double res = chosen >= 0
                   ? suffix[chosen + 1] + pv[chosen]
                   : suffix[0] - std::log(static_cast<double>(vocab_size_));
  if (cache_.size() < 1000000) cache_.emplace(key, res);
  return res;
}

namespace {

struct Entry {
  int prefix;  // trie node
  double b, nb, s;
  int steps;  // step-list node
  uint64_t hk;
};

struct Trie {
  std::vector<int> parent{-1};
  std::vector<int64_t> token{-1};
  std::unordered_map<uint64_t, int> child;
  static uint64_t key(int p, int64_t c) {
    return (static_cast<uint64_t>(p) << 32) ^ static_cast<uint64_t>(c);
  }
  int find(int p, int64_t c) const {
    auto it = child.find(key(p, c));
    return it == child.end() ? -1 : it->second;
  }
  int get(int p, int64_t c) {
    auto it = child.find(key(p, c));
    if (it != child.end()) return it->second;
    int id = static_cast<int>(parent.size());
    parent.push_back(p);
    token.push_back(c);
    child.emplace(key(p, c), id);
    return id;
  }
};

// nxt: a dict keyed by prefix that keeps insertion order
struct Beams {
  std::vector<Entry> items;
  std::unordered_map<int, size_t> pos;
  void clear() {
    items.clear();
    pos.clear();
  }
  bool has(int prefix) const { return pos.count(prefix) > 0; }
  void add(int prefix, double b, double nb, double s, int steps, uint64_t hk) {
    auto it = pos.find(prefix);
    if (it == pos.end()) {
      pos.emplace(prefix, items.size());
      items.push_back({prefix, b, nb, s, steps, hk});
    } else {
      Entry& e = items[it->second];
      e.b = lse(e.b, b);
      e.nb = lse(e.nb, nb);
    }
  }
};

}  // namespace

void beam_search(const float* blank, const float* topv, const int64_t* topi,
                 int T, int K, NgramLM& lm, double alpha, double beta, int beam,
                 double blank_skip, double cand_floor, std::vector<int64_t>& ids,
                 std::vector<int64_t>& steps_out) {
  const uint64_t bits = static_cast<uint64_t>(lm.bits);
  const uint64_t hmask = (uint64_t(1) << (bits * (lm.order - 1))) - 1;
  Trie trie;
  std::vector<int> step_prev{-1};  // step-list nodes: previous node, step
  std::vector<int64_t> step_t{-1};
  std::vector<Entry> beams{{0, 0.0, NEG_INF, 0.0, 0, 0}};
  Beams nxt;
  const double log_skip = std::log(blank_skip);
  const double log_floor = std::log(cand_floor);
  std::vector<std::pair<int64_t, double>> cands;
  struct Ext {
    int prefix;
    int64_t c;
    double new_from, s;
    int steps;
    uint64_t hk;
  };
  std::vector<Ext> ext;
  for (int t = 0; t < T; ++t) {
    double b0 = static_cast<double>(blank[t]);
    if (b0 > log_skip) {
      for (Entry& e : beams) {
        e.b = lse(e.b, e.nb) + b0;
        e.nb = NEG_INF;
      }
      continue;
    }
    nxt.clear();
    cands.clear();
    for (int k = 0; k < K; ++k) {
      int64_t c = topi[static_cast<size_t>(t) * K + k];
      double v = static_cast<double>(topv[static_cast<size_t>(t) * K + k]);
      if (c != 0 && v >= log_floor) cands.emplace_back(c, v);
    }
    // every beam's own entries first (the blank, and a repeat that collapses)
    for (const Entry& e : beams) {
      nxt.add(e.prefix, lse(e.b, e.nb) + b0, NEG_INF, e.s, e.steps, e.hk);
      int64_t last = e.prefix ? trie.token[e.prefix] : -1;
      for (auto& [c, lp] : cands)
        if (c == last) nxt.add(e.prefix, NEG_INF, e.nb + lp, e.s, e.steps, e.hk);
    }
    double floor = NEG_INF;
    if (static_cast<int>(beams.size()) >= beam) {
      floor = std::numeric_limits<double>::infinity();
      for (const Entry& e : nxt.items) floor = std::min(floor, lse(e.b, e.nb) + e.s);
    }
    double reach = beta + alpha * lm.max_logp;
    ext.clear();
    for (const Entry& e : beams) {
      double tot = lse(e.b, e.nb);
      int64_t last = e.prefix ? trie.token[e.prefix] : -1;
      for (auto& [c, lp] : cands) {
        // after a blank, the same class is a new token
        double new_from = c == last ? e.b + lp : tot + lp;
        if (new_from == NEG_INF) continue;
        int q = trie.find(e.prefix, c);
        if (q >= 0 && nxt.has(q)) {  // an existing beam: merges, its LM score stays
          nxt.add(q, NEG_INF, new_from, e.s, e.steps, e.hk);
        } else if (new_from + e.s + reach >= floor - 1e-9) {
          ext.push_back({e.prefix, c, new_from, e.s, e.steps, e.hk});
        }
      }
    }
    for (const Ext& x : ext) {
      uint64_t key = (x.hk << bits) | static_cast<uint64_t>(x.c + 1);
      double lmp = lm.logp_key(key);
      step_prev.push_back(x.steps);
      step_t.push_back(t);
      nxt.add(trie.get(x.prefix, x.c), NEG_INF, x.new_from,
              x.s + alpha * lmp + beta, static_cast<int>(step_prev.size()) - 1,
              key & hmask);
    }
    // the best ``beam`` prefixes, by score (stable: equal scores keep their order)
    std::vector<std::pair<double, size_t>> order(nxt.items.size());
    for (size_t i = 0; i < order.size(); ++i) {
      const Entry& e = nxt.items[i];
      order[i] = {-(lse(e.b, e.nb) + e.s), i};
    }
    std::stable_sort(order.begin(), order.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    beams.clear();
    for (size_t i = 0; i < order.size() && static_cast<int>(i) < beam; ++i)
      beams.push_back(nxt.items[order[i].second]);
  }
  // the best beam (the first on ties, as max() keeps)
  const Entry* best = nullptr;
  double best_v = 0;
  for (const Entry& e : beams) {
    double v = lse(e.b, e.nb) + e.s;
    if (!best || v > best_v) best = &e, best_v = v;
  }
  if (best == nullptr) throw std::logic_error("beam_search: no beam left");
  ids.clear();
  steps_out.clear();
  for (int p = best->prefix; p > 0; p = trie.parent[p]) ids.push_back(trie.token[p]);
  for (int sn = best->steps; sn > 0; sn = step_prev[sn]) steps_out.push_back(step_t[sn]);
  std::reverse(ids.begin(), ids.end());
  std::reverse(steps_out.begin(), steps_out.end());
}

}  // namespace indra
