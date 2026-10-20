// Kernels behind indra_ocr_native: plain C++ on raw buffers.  module.cpp
// wraps them for Python.  Each one mirrors a function of indra_ocr (named in
// its comment) and gives the same result.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace indra {

// ---------------------------------------------------------------- text layer

// Python's str.isspace for one code point.
bool is_space(uint32_t c);

// Per character of a pdfium text page: its UTF-16 code unit, its angle, its
// box (left, bottom, right, top) and whether pdfium gave the box.  The function
// pointers are pdfium's own FPDFText_GetUnicode, FPDFText_GetCharAngle and
// FPDFText_GetCharBox, from the library pypdfium2 loaded.
void pdf_char_data(uintptr_t textpage, int n, uintptr_t get_unicode,
                   uintptr_t get_angle, uintptr_t get_box, uint32_t* codes,
                   float* angles, double* boxes, uint8_t* box_ok);

struct CharLoop {
  std::vector<uint32_t> chars;    // characters kept, in page order
  std::vector<int8_t> qs;         // each kept character's quarter turn
  std::vector<int32_t> src;       // each kept character's pdfium index
  std::vector<int32_t> breaks;    // runs end before these positions
  std::vector<uint32_t> counted;  // non-space characters (bad-character count)
  int n_chars = 0, n_rot = 0;
  int quarters[4] = {0, 0, 0, 0};
  int error = 0;  // 1: a code point above U+10FFFF; 2: pdfium gave no box
};

// The character loop of textlayer.extract_runs.  q_of: per pdfium character,
// the quarter turn of its angle (0..3) or -1 when the angle is not a quarter
// turn.
void pdf_char_loop(const uint32_t* codes, const int8_t* q_of,
                   const uint8_t* box_ok, int n, int turn, CharLoop& out);

// document.assign_runs: for each character (boxes in points, k pixels per
// point), the detected line it goes to (-1: none) and its x.  run_start: the
// first character of each run, then the total.
int assign_chars(const uint32_t* codes, const double* cboxes,
                 const int64_t* run_start, int n_runs, const float* boxes,
                 int n_boxes, double k, int32_t* line_of, double* x_of);

// ----------------------------------------------------------------------- CTC

// The recognizers' greedy CTC head on logits [B, V, T]: per step, the best
// class and its softmax probability; steps from lengths[b] on are zero.
void ctc_greedy(const float* logits, int B, int V, int T,
                const int64_t* lengths, int64_t* ids, float* conf);

// The language-model outputs for the first n steps of one line's logits
// [V, T]: blank log-probability [n], the K best log-probabilities [n, K] and
// their classes [n, K] (best first; equal values: lower class first).
void ctc_topk(const float* logits, int V, int T, int n, int K, float* blank,
              float* topv, int64_t* topi);

// ------------------------------------------------------------ language model

// lm.NgramLM over its sorted tables (owned by the Python object).
class NgramLM {
 public:
  NgramLM(int order, int bits, int64_t vocab_size, const uint64_t* p_keys,
          const double* p_vals, size_t n_p, const uint64_t* h_keys,
          const double* h_vals, size_t n_h, double max_logp);
  double logp_key(uint64_t key);
  int order, bits;
  double max_logp;

 private:
  int64_t vocab_size_;
  const uint64_t *p_keys_, *h_keys_;
  const double *p_vals_, *h_vals_;
  size_t n_p_, n_h_;
  std::unordered_map<uint64_t, double> cache_;
};

// lm.beam_search: CTC prefix beam search with the language model.
void beam_search(const float* blank, const float* topv, const int64_t* topi,
                 int T, int K, NgramLM& lm, double alpha, double beta,
                 int beam, double blank_skip, double cand_floor,
                 std::vector<int64_t>& ids, std::vector<int64_t>& steps);

// --------------------------------------------------------------- ordering

// order_post._cluster: average-linkage clustering of pairwise logits [n, n]
// (float32 or float64 values given as double).  Returns each group's members,
// ascending, in the order of their first member's slot.
std::vector<std::vector<int>> cluster(const double* logits, int n);

// The cut positions of pipeline.split_line: indices k where a new segment
// starts at token k.
std::vector<int> split_points(const double* xs, const uint8_t* space, int n,
                              double limit, const double* at, int n_at);

}  // namespace indra
