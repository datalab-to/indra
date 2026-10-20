// The per-character loops of the PDF text layer (textlayer.extract_runs and
// document.assign_runs).

#include <algorithm>
#include <cmath>

#include "kernels.h"

namespace indra {

bool is_space(uint32_t c) {
  // Python's str.isspace: bidi class WS, B or S, or category Zs.
  if (c <= 0x20) return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C);
  switch (c) {
    case 0x85: case 0xA0: case 0x1680: case 0x2028: case 0x2029:
    case 0x202F: case 0x205F: case 0x3000:
      return true;
    default:
      return c >= 0x2000 && c <= 0x200A;
  }
}

using GetUnicodeFn = unsigned int (*)(void*, int);
using GetAngleFn = float (*)(void*, int);
using GetBoxFn = int (*)(void*, int, double*, double*, double*, double*);

void pdf_char_data(uintptr_t textpage, int n, uintptr_t get_unicode,
                   uintptr_t get_angle, uintptr_t get_box, uint32_t* codes,
                   float* angles, double* boxes, uint8_t* box_ok) {
  auto tp = reinterpret_cast<void*>(textpage);
  auto uni = reinterpret_cast<GetUnicodeFn>(get_unicode);
  auto ang = reinterpret_cast<GetAngleFn>(get_angle);
  auto box = reinterpret_cast<GetBoxFn>(get_box);
  for (int i = 0; i < n; ++i) {
    codes[i] = uni(tp, i);
    angles[i] = ang(tp, i);
    double l = 0, r = 0, b = 0, t = 0;
    box_ok[i] = box(tp, i, &l, &r, &b, &t) ? 1 : 0;
    double* o = boxes + 4 * static_cast<size_t>(i);
    o[0] = l, o[1] = b, o[2] = r, o[3] = t;  // as extract_runs keeps them
  }
}

void pdf_char_loop(const uint32_t* codes, const int8_t* q_of,
                   const uint8_t* box_ok, int n, int turn, CharLoop& out) {
  int cur_q = 0;
  bool low = false;  // this index is the second half of a surrogate pair
  for (int i = 0; i < n; ++i) {
    if (low) {
      low = false;
      continue;
    }
    // textlayer.code_point
    uint32_t u = codes[i];
    uint32_t u2 = (u >= 0xD800 && u <= 0xDBFF && i + 1 < n) ? codes[i + 1] : 0;
    uint32_t c;
    if (u >= 0xD800 && u <= 0xDBFF && u2 >= 0xDC00 && u2 <= 0xDFFF) {
      c = 0x10000 + ((u - 0xD800) << 10) + (u2 - 0xDC00);
      low = true;
    } else if (u >= 0xD800 && u <= 0xDFFF) {
      c = 0xFFFD;
    } else {
      c = u;
    }
    if (c > 0x10FFFF) {  // chr() raises
      out.error = 1;
      return;
    }
    if (c == '\r' || c == '\n') {
      out.breaks.push_back(static_cast<int32_t>(out.chars.size()));
      continue;
    }
    if (c == 0x02) c = '-';  // pdfium's mark for a hyphen at a line break
    if (!is_space(c)) {
      out.n_chars += 1;
      out.counted.push_back(c);
      int q = q_of[i];
      if (q >= 0) out.quarters[q] += 1;
      if (q != turn) out.n_rot += 1;
      if (q < 0) {
        out.breaks.push_back(static_cast<int32_t>(out.chars.size()));
        continue;
      }
      cur_q = ((q - turn) % 4 + 4) % 4;
    }
    out.chars.push_back(c);
    out.qs.push_back(static_cast<int8_t>(cur_q));
    if (!box_ok[i]) {
      out.error = 2;
      return;
    }
    out.src.push_back(i);
  }
}

static int row_of(double y, double y_lo, double bin, int n_bins) {
  double r = std::floor((y - y_lo) / bin);
  return static_cast<int>(std::min(std::max(r, 0.0), n_bins - 1.0));
}

int assign_chars(const uint32_t* codes, const double* cboxes,
                 const int64_t* run_start, int n_runs, const float* boxes,
                 int n_boxes, double k, int32_t* line_of, double* x_of) {
  // The boxes that hold a character's centre are found through rows of
  // height ``bin``: each box is listed in the rows it spans, in index order.
  // Containment and the smallest area are tested exactly as numpy does
  // (float32 box sides against float64 centres, float32 areas, the first
  // box on ties), so the rows only narrow the search.
  std::vector<float> area(n_boxes);
  double y_lo = INFINITY, y_hi = -INFINITY, h_sum = 0;
  int n_valid = 0;
  for (int j = 0; j < n_boxes; ++j) {
    const float* b = boxes + 4 * static_cast<size_t>(j);
    area[j] = (b[2] - b[0]) * (b[3] - b[1]);
    if (b[3] >= b[1]) {
      y_lo = std::min(y_lo, static_cast<double>(b[1]));
      y_hi = std::max(y_hi, static_cast<double>(b[3]));
      h_sum += static_cast<double>(b[3]) - b[1];
      n_valid += 1;
    }
  }
  int n_bins = 0;
  double bin = 1.0;
  std::vector<std::vector<int>> rows;
  if (n_valid) {
    bin = std::max(h_sum / n_valid, 1.0);
    n_bins = static_cast<int>(std::min((y_hi - y_lo) / bin, 1e6)) + 1;
    rows.resize(n_bins);
    for (int j = 0; j < n_boxes; ++j) {
      const float* b = boxes + 4 * static_cast<size_t>(j);
      if (!(b[3] >= b[1])) continue;  // holds no centre
      // Row of y: floor((y - y_lo) / bin), clamped; it never decreases as y
      // grows, so a centre inside the box falls in one of the box's rows.
      int r0 = row_of(b[1], y_lo, bin, n_bins), r1 = row_of(b[3], y_lo, bin, n_bins);
      for (int r = r0; r <= r1; ++r) rows[r].push_back(j);
    }
  }
  int lost = 0;
  for (int r = 0; r < n_runs; ++r) {
    int j = -1;  // the line of the run's last character
    double x_prev = 0;
    for (int64_t i = run_start[r]; i < run_start[r + 1]; ++i) {
      if (!is_space(codes[i])) {
        const double* b = cboxes + 4 * i;
        double cx = (b[0] + b[2]) / 2 * k, cy = (b[1] + b[3]) / 2 * k;
        int best = -1;
        if (n_bins && cy >= y_lo && cy <= y_hi) {
          for (int c : rows[row_of(cy, y_lo, bin, n_bins)]) {
            const float* q = boxes + 4 * static_cast<size_t>(c);
            if (q[0] <= cx && cx <= q[2] && q[1] <= cy && cy <= q[3] &&
                (best < 0 || area[c] < area[best]))
              best = c;
          }
        }
        if (best < 0) {
          lost += 1;
          j = -1;
          line_of[i] = -1;
          continue;
        }
        j = best;
        x_prev = cx;
        line_of[i] = j;
        x_of[i] = cx;
      } else if (j >= 0) {  // a space follows the character before it
        line_of[i] = j;
        x_of[i] = x_prev;
      } else {
        line_of[i] = -1;
      }
    }
  }
  return lost;
}

}  // namespace indra
