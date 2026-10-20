// Python bindings of the kernels (indra_ocr_native._core).
//
// Every binding checks its inputs (shapes, lengths, ranges, handles) before a
// kernel touches a buffer, and raises ValueError instead: a kernel never reads
// out of bounds on bad input.  indra_ocr catches any exception from a kernel
// and runs its Python code instead (indra_ocr/native.py).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "kernels.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Bump when a kernel's signature or meaning changes; indra_ocr checks it.
constexpr int API = 1;

template <class T>
using In = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

// A numpy array that owns a vector's buffer.
template <class T>
nb::ndarray<nb::numpy, T> owned(std::vector<T>&& v, std::initializer_list<size_t> shape) {
  if (v.empty()) v.reserve(1);
  auto* p = new std::vector<T>(std::move(v));
  nb::capsule owner(p, [](void* q) noexcept { delete static_cast<std::vector<T>*>(q); });
  return nb::ndarray<nb::numpy, T>(p->data(), shape, owner);
}

[[noreturn]] void bad(const std::string& what) { throw std::invalid_argument(what); }

void need(bool ok, const char* what) {
  if (!ok) bad(what);
}

int as_int(size_t v, const char* what) {
  if (v > static_cast<size_t>(INT32_MAX)) bad(std::string(what) + ": too large");
  return static_cast<int>(v);
}

struct PyNgramLM {
  In<uint64_t> pk, hk;
  In<double> pv, hv;
  std::unique_ptr<indra::NgramLM> lm;
};

}  // namespace

NB_MODULE(_core, m) {
  m.attr("API") = API;

  m.def(
      "is_space", [](uint32_t c) { return indra::is_space(c); }, "c"_a,
      "Python's str.isspace of one code point.");

  m.def(
      "pdf_char_data",
      [](uintptr_t textpage, int n, uintptr_t get_unicode, uintptr_t get_angle,
         uintptr_t get_box) {
        need(textpage && get_unicode && get_angle && get_box,
             "pdf_char_data: a null text page or pdfium function");
        need(n >= 0, "pdf_char_data: a negative character count");
        size_t N = static_cast<size_t>(n);
        std::vector<uint32_t> codes(N);
        std::vector<float> angles(N);
        std::vector<double> boxes(4 * N);
        std::vector<uint8_t> ok(N);
        // pdfium is not thread-safe: the GIL stays held, as for pypdfium2's calls
        indra::pdf_char_data(textpage, n, get_unicode, get_angle, get_box, codes.data(),
                             angles.data(), boxes.data(), ok.data());
        return nb::make_tuple(owned(std::move(codes), {N}), owned(std::move(angles), {N}),
                              owned(std::move(boxes), {N, 4}), owned(std::move(ok), {N}));
      },
      "textpage"_a, "n"_a, "get_unicode"_a, "get_angle"_a, "get_box"_a,
      "Per character of a pdfium text page: code unit, angle, box (l, b, r, t), box ok.");

  m.def(
      "pdf_char_loop",
      [](In<uint32_t> codes, In<int8_t> q_of, In<uint8_t> box_ok, int turn) {
        need(q_of.size() == codes.size() && box_ok.size() == codes.size(),
             "pdf_char_loop: codes, q_of and box_ok differ in length");
        int n = as_int(codes.size(), "pdf_char_loop");
        for (size_t i = 0; i < q_of.size(); ++i)
          need(q_of.data()[i] >= -1 && q_of.data()[i] <= 3,
               "pdf_char_loop: a quarter turn outside -1..3");
        indra::CharLoop r;
        indra::pdf_char_loop(codes.data(), q_of.data(), box_ok.data(), n, turn, r);
        size_t k = r.chars.size(), c = r.counted.size(), s = r.src.size();
        return nb::make_tuple(
            owned(std::move(r.chars), {k}), owned(std::move(r.qs), {k}),
            owned(std::move(r.src), {s}), r.breaks, owned(std::move(r.counted), {c}),
            r.n_chars, r.n_rot, std::vector<int>(r.quarters, r.quarters + 4), r.error);
      },
      "codes"_a, "q_of"_a, "box_ok"_a, "turn"_a,
      "The character loop of textlayer.extract_runs (error: 1 code point out of range, "
      "2 no box).");

  m.def(
      "assign_chars",
      [](In<uint32_t> codes, In<double> cboxes, In<int64_t> run_start, In<float> boxes,
         double k) {
        size_t n_chars = codes.size();
        need(cboxes.size() == 4 * n_chars, "assign_chars: cboxes must be [len(codes), 4]");
        need(boxes.size() % 4 == 0, "assign_chars: boxes must be [N, 4]");
        need(run_start.size() >= 1, "assign_chars: run_start needs at least one entry");
        const int64_t* rs = run_start.data();
        need(rs[0] == 0 && rs[run_start.size() - 1] == static_cast<int64_t>(n_chars),
             "assign_chars: run_start must go from 0 to len(codes)");
        for (size_t i = 1; i < run_start.size(); ++i)
          need(rs[i] >= rs[i - 1], "assign_chars: run_start must not decrease");
        need(std::isfinite(k), "assign_chars: k must be finite");
        int n_runs = as_int(run_start.size() - 1, "assign_chars");
        int n_boxes = as_int(boxes.size() / 4, "assign_chars");
        as_int(n_chars, "assign_chars");
        std::vector<int32_t> line_of(n_chars, -1);
        std::vector<double> x_of(n_chars, 0.0);
        int lost;
        {
          nb::gil_scoped_release release;
          lost = indra::assign_chars(codes.data(), cboxes.data(), rs, n_runs, boxes.data(),
                                     n_boxes, k, line_of.data(), x_of.data());
        }
        return nb::make_tuple(owned(std::move(line_of), {n_chars}),
                              owned(std::move(x_of), {n_chars}), lost);
      },
      "codes"_a, "cboxes"_a, "run_start"_a, "boxes"_a, "k"_a,
      "document.assign_runs: each character's line (-1: none) and x, and the lost count.");

  m.def(
      "ctc_greedy",
      [](nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu> logits,
         In<int64_t> lengths) {
        int B = as_int(logits.shape(0), "ctc_greedy"), V = as_int(logits.shape(1), "ctc_greedy"),
            T = as_int(logits.shape(2), "ctc_greedy");
        need(V >= 1, "ctc_greedy: logits need at least one class");
        need(static_cast<int>(lengths.size()) == B && lengths.ndim() == 1,
             "ctc_greedy: lengths must be [B]");
        std::vector<int64_t> ids(static_cast<size_t>(B) * T);
        std::vector<float> conf(static_cast<size_t>(B) * T);
        {
          nb::gil_scoped_release release;
          indra::ctc_greedy(logits.data(), B, V, T, lengths.data(), ids.data(), conf.data());
        }
        size_t b = B, t = T;
        return nb::make_tuple(owned(std::move(ids), {b, t}), owned(std::move(conf), {b, t}));
      },
      "logits"_a, "lengths"_a,
      "Greedy CTC on logits [B, V, T]: best class and its probability per step.");

  m.def(
      "ctc_topk",
      [](nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> logits, int n,
         int k) {
        int V = as_int(logits.shape(0), "ctc_topk"), T = as_int(logits.shape(1), "ctc_topk");
        need(V >= 1, "ctc_topk: logits need at least one class");
        need(k >= 1 && k <= 4096, "ctc_topk: k must be 1..4096");
        n = std::max(0, std::min(n, T));
        std::vector<float> blank(n), topv(static_cast<size_t>(n) * k);
        std::vector<int64_t> topi(static_cast<size_t>(n) * k);
        {
          nb::gil_scoped_release release;
          indra::ctc_topk(logits.data(), V, T, n, k, blank.data(), topv.data(), topi.data());
        }
        size_t N = n, K = k;
        return nb::make_tuple(owned(std::move(blank), {N}), owned(std::move(topv), {N, K}),
                              owned(std::move(topi), {N, K}));
      },
      "logits"_a, "n"_a, "k"_a,
      "One line's logits [V, T]: blank log-prob, top-k log-probs and classes, first n steps.");

  nb::class_<PyNgramLM>(m, "NgramLM")
      .def(
          "__init__",
          [](PyNgramLM* self, int order, int bits, int64_t vocab_size, In<uint64_t> p_keys,
             In<double> p_vals, In<uint64_t> h_keys, In<double> h_vals, double max_logp) {
            need(p_keys.size() == p_vals.size() && h_keys.size() == h_vals.size(),
                 "NgramLM: keys and values differ in length");
            need(p_keys.ndim() == 1 && h_keys.ndim() == 1, "NgramLM: tables must be 1-D");
            need(order >= 1 && bits >= 1 && order * bits <= 61,
                 "NgramLM: order x bits must fit 61 bits");
            need(vocab_size >= 1, "NgramLM: vocab_size must be positive");
            const uint64_t* pk = p_keys.data();
            const uint64_t* hk = h_keys.data();
            for (size_t i = 1; i < p_keys.size(); ++i)
              need(pk[i - 1] <= pk[i], "NgramLM: p_keys must be sorted");
            for (size_t i = 1; i < h_keys.size(); ++i)
              need(hk[i - 1] <= hk[i], "NgramLM: h_keys must be sorted");
            new (self) PyNgramLM{p_keys, h_keys, p_vals, h_vals, nullptr};
            self->lm = std::make_unique<indra::NgramLM>(
                order, bits, vocab_size, pk, p_vals.data(), p_keys.size(), hk, h_vals.data(),
                h_keys.size(), max_logp);
          },
          "order"_a, "bits"_a, "vocab_size"_a, "p_keys"_a, "p_vals"_a, "h_keys"_a,
          "h_vals"_a, "max_logp"_a)
      .def(
          "logp_keys",
          [](PyNgramLM& self, const std::vector<uint64_t>& keys) {
            std::vector<double> out(keys.size());
            for (size_t i = 0; i < keys.size(); ++i) out[i] = self.lm->logp_key(keys[i]);
            return out;
          },
          "keys"_a);

  m.def(
      "beam_search",
      [](In<float> blank, nb::ndarray<const float, nb::ndim<2>, nb::c_contig> topv,
         nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig> topi, PyNgramLM& lm,
         double alpha, double beta, int beam, double blank_skip, double cand_floor) {
        need(blank.ndim() == 1, "beam_search: blank must be [T]");
        int T = as_int(blank.size(), "beam_search"), K = as_int(topv.shape(1), "beam_search");
        need(static_cast<int>(topv.shape(0)) == T && topi.shape(0) == topv.shape(0) &&
                 topi.shape(1) == topv.shape(1),
             "beam_search: blank, topv and topi differ in shape");
        need(beam >= 1, "beam_search: beam must be at least 1");
        need(blank_skip > 0 && cand_floor > 0, "beam_search: blank_skip and cand_floor must be positive");
        need(lm.lm != nullptr, "beam_search: an uninitialized NgramLM");
        const int64_t* ti = topi.data();
        for (size_t i = 0; i < topi.size(); ++i)
          need(ti[i] >= 0 && ti[i] < (int64_t(1) << lm.lm->bits) - 1,
               "beam_search: a class id outside the LM's token range");
        std::vector<int64_t> ids, steps;
        // The GIL stays held: the LM's lookup cache is shared state.
        indra::beam_search(blank.data(), topv.data(), ti, T, K, *lm.lm, alpha, beta, beam,
                           blank_skip, cand_floor, ids, steps);
        return std::make_pair(ids, steps);
      },
      "blank"_a, "topv"_a, "topi"_a, "lm"_a, "alpha"_a, "beta"_a, "beam"_a,
      "blank_skip"_a, "cand_floor"_a, "lm.beam_search: (token ids, step of each token).");

  m.def(
      "cluster",
      [](nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu> logits) {
        need(logits.shape(0) == logits.shape(1), "cluster: logits must be square");
        int n = as_int(logits.shape(0), "cluster");
        need(static_cast<uint64_t>(n) * n <= (uint64_t(1) << 31), "cluster: too many lines");
        nb::gil_scoped_release release;
        return indra::cluster(logits.data(), n);
      },
      "logits"_a, "order_post._cluster: average-linkage groups (each sorted).");

  m.def(
      "split_points",
      [](nb::list toks, double limit, std::vector<double> at, nb::object space_id) {
        // split_line's tokens: (str or vocab id, x, ...); a space is " " or space_id
        size_t n = nb::len(toks);
        as_int(n, "split_points");
        bool has_id = !space_id.is_none();
        long long sid = has_id ? nb::cast<long long>(space_id) : 0;
        std::vector<double> xs(n);
        std::vector<uint8_t> space(n);
        for (size_t k = 0; k < n; ++k) {
          nb::handle tk = PyList_GetItem(toks.ptr(), static_cast<Py_ssize_t>(k));
          if (!tk.is_valid()) throw nb::python_error();
          nb::object tok = nb::steal(PySequence_GetItem(tk.ptr(), 0));
          if (!tok.is_valid()) throw nb::python_error();
          bool sp = false;
          if (nb::isinstance<nb::str>(tok)) {
            sp = PyUnicode_GetLength(tok.ptr()) == 1 && PyUnicode_ReadChar(tok.ptr(), 0) == ' ';
          } else if (has_id && PyLong_Check(tok.ptr())) {
            int overflow = 0;
            long long v = PyLong_AsLongLongAndOverflow(tok.ptr(), &overflow);
            if (v == -1 && PyErr_Occurred()) throw nb::python_error();
            sp = !overflow && v == sid;
          }
          space[k] = sp;
          if (!sp) {
            nb::object xo = nb::steal(PySequence_GetItem(tk.ptr(), 1));
            if (!xo.is_valid()) throw nb::python_error();
            double x = PyFloat_AsDouble(xo.ptr());
            if (x == -1.0 && PyErr_Occurred()) throw nb::python_error();
            xs[k] = x;
          }
        }
        return indra::split_points(xs.data(), space.data(), static_cast<int>(n), limit,
                                   at.data(), static_cast<int>(at.size()));
      },
      "toks"_a, "limit"_a, "at"_a, "space_id"_a.none(),
      "pipeline.split_line: indices of the tokens that start a new segment.");
}
