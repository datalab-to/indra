// Python bindings of the GPU engine (nanobind).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstring>

#include "ctc_kernels.h"
#include "trt_model.h"

namespace nb = nanobind;
using namespace nb::literals;
using indra_cuda::DeviceBuffer;
using indra_cuda::DType;
using indra_cuda::Model;
using indra_cuda::Profile;
using indra_cuda::Shape;
using indra_cuda::Tensor;

namespace {

constexpr int API = 1;  // the indra_ocr_cuda.API indra_ocr's gpu.py calls

using HostArray = nb::ndarray<nb::numpy, nb::device::cpu, nb::c_contig>;

DType dtype_of(const nb::dlpack::dtype& d) {
  if (d == nb::dtype<float>()) return DType::f32;
  if (d == nb::dtype<int64_t>()) return DType::i64;
  if (d == nb::dtype<int32_t>()) return DType::i32;
  if (d == nb::dtype<bool>()) return DType::b8;
  throw nb::type_error("indra-ocr-cuda: inputs are float32, int64, int32 or bool arrays");
}

template <class T>
nb::object new_array(const Shape& shape, T** data) {
  size_t n = 1;
  for (auto v : shape) n *= static_cast<size_t>(v);
  T* p = new T[n ? n : 1];
  nb::capsule owner(p, [](void* q) noexcept { delete[] static_cast<T*>(q); });
  std::vector<size_t> shp(shape.begin(), shape.end());
  *data = p;
  return nb::cast(nb::ndarray<nb::numpy, T>(p, shp.size(), shp.data(), owner));
}

nb::object new_array(DType t, const Shape& shape, void** data) {
  switch (t) {
    case DType::f32: return new_array<float>(shape, reinterpret_cast<float**>(data));
    case DType::i64: return new_array<int64_t>(shape, reinterpret_cast<int64_t**>(data));
    case DType::i32: return new_array<int32_t>(shape, reinterpret_cast<int32_t**>(data));
    case DType::b8: return new_array<bool>(shape, reinterpret_cast<bool**>(data));
    case DType::f16: return new_array<uint16_t>(shape, reinterpret_cast<uint16_t**>(data));
  }
  throw std::runtime_error("indra-ocr-cuda: unknown type");
}

// A model on the GPU, with the recognizer helpers that read its outputs in place.
class PyModel {
 public:
  PyModel(nb::bytes onnx, const std::map<std::string, std::tuple<Shape, Shape, Shape>>& profiles,
          const std::string& cache, size_t workspace, int device, int opt_level) {
    std::map<std::string, Profile> p;
    for (auto& [name, mm] : profiles) p[name] = Profile{std::get<0>(mm), std::get<1>(mm), std::get<2>(mm)};
    std::string bytes(onnx.c_str(), onnx.size());
    nb::gil_scoped_release nogil;
    m_ = std::make_unique<Model>(bytes, p, cache, workspace, device, opt_level);
  }

  std::vector<std::string> names(bool inputs) const {
    std::vector<std::string> out;
    for (auto& t : m_->tensors())
      if (t.input == inputs) out.push_back(t.name);
    return out;
  }

  // Copy the inputs up, run, and copy the fetched outputs down; the others
  // stay on the device until the next run.
  nb::list run(nb::dict feed, const std::vector<std::string>& fetch) {
    struct Up {
      Tensor* t;
      const void* src;
      size_t bytes;
    };
    std::vector<Up> ups;
    for (auto [k, v] : feed) {
      std::string name = nb::cast<std::string>(k);
      HostArray a = nb::cast<HostArray>(v);
      Tensor& t = m_->tensor(name);
      if (dtype_of(a.dtype()) != t.dtype) throw nb::type_error(("indra-ocr-cuda: wrong type for " + name).c_str());
      Shape s(a.shape_ptr(), a.shape_ptr() + a.ndim());
      m_->set_input_shape(name, s);
      ups.push_back({&t, a.data(), a.nbytes()});
    }
    cudaStream_t st = m_->stream();
    {
      nb::gil_scoped_release nogil;
      for (auto& u : ups) INDRA_CK(cudaMemcpyAsync(u.t->buf.ptr, u.src, u.bytes, cudaMemcpyHostToDevice, st));
      m_->enqueue();
    }
    // the outputs' shapes are known once the run is enqueued
    std::vector<std::pair<Tensor*, void*>> downs;
    nb::list out;
    for (auto& name : fetch) {
      Tensor& t = m_->tensor(name);
      void* data = nullptr;
      out.append(new_array(t.dtype, t.shape, &data));
      downs.push_back({&t, data});
    }
    {
      nb::gil_scoped_release nogil;
      for (auto& [t, data] : downs)
        INDRA_CK(cudaMemcpyAsync(data, t->buf.ptr, t->nbytes(), cudaMemcpyDeviceToHost, st));
      INDRA_CK(cudaStreamSynchronize(st));
    }
    return out;
  }

  // The last run's peak probability per frame, from its logits and lengths
  // outputs on the device: [B, T] float32.
  nb::object frame_conf(const std::string& logits, const std::string& lengths) {
    Tensor& L = m_->tensor(logits);
    Tensor& N = m_->tensor(lengths);
    check_logits(L, N);
    const int B = static_cast<int>(L.shape[0]), T = static_cast<int>(L.shape[1]), C = static_cast<int>(L.shape[2]);
    float* host = nullptr;
    nb::object arr = new_array<float>(Shape{B, T}, &host);
    {
      nb::gil_scoped_release nogil;
      conf_.reserve(static_cast<size_t>(B) * T * sizeof(float) + 4);
      indra_cuda::frame_conf(L.buf.as<float>(), N.buf.as<int64_t>(), B, T, C, conf_.as<float>(), m_->stream());
      INDRA_CK(cudaMemcpyAsync(host, conf_.ptr, static_cast<size_t>(B) * T * sizeof(float),
                               cudaMemcpyDeviceToHost, m_->stream()));
      INDRA_CK(cudaStreamSynchronize(m_->stream()));
    }
    return arr;
  }

  // The beam search's inputs for some lines of the last run: (blank [R, T],
  // topv [R, T, k], topi [R, T, k]) for the lines ``rows``, their first n[r]
  // frames (the rest: blank 0, topv -inf, topi 0).
  nb::tuple lm_inputs(const std::string& logits, const std::vector<int32_t>& rows, const std::vector<int32_t>& n,
                      int k) {
    Tensor& L = m_->tensor(logits);
    if (L.dtype != DType::f32 || L.shape.size() != 3) throw nb::value_error("indra-ocr-cuda: logits [B, T, C] float32");
    if (rows.size() != n.size()) throw nb::value_error("indra-ocr-cuda: rows and n differ in length");
    const int B = static_cast<int>(L.shape[0]), T = static_cast<int>(L.shape[1]), C = static_cast<int>(L.shape[2]);
    for (auto r : rows)
      if (r < 0 || r >= B) throw nb::index_error("indra-ocr-cuda: a row outside the batch");
    const int R = static_cast<int>(rows.size());
    float *hb = nullptr, *hv = nullptr;
    int64_t* hi = nullptr;
    nb::object ab = new_array<float>(Shape{R, T}, &hb);
    nb::object av = new_array<float>(Shape{R, T, k}, &hv);
    nb::object ai = new_array<int64_t>(Shape{R, T, k}, &hi);
    {
      nb::gil_scoped_release nogil;
      cudaStream_t st = m_->stream();
      // the rows' indexes go up in the same stream as the kernel: the host
      // vectors outlive the synchronize below
      idx_.reserve(2 * R * sizeof(int32_t) + 8);
      INDRA_CK(cudaMemcpyAsync(idx_.as<int32_t>(), rows.data(), R * sizeof(int32_t), cudaMemcpyHostToDevice, st));
      INDRA_CK(cudaMemcpyAsync(idx_.as<int32_t>() + R, n.data(), R * sizeof(int32_t), cudaMemcpyHostToDevice, st));
      const size_t f = static_cast<size_t>(R) * T;
      blank_.reserve(f * sizeof(float) + 4);
      topv_.reserve(f * k * sizeof(float) + 4);
      topi_.reserve(f * k * sizeof(int64_t) + 8);
      indra_cuda::lm_inputs(L.buf.as<float>(), idx_.as<int32_t>(), idx_.as<int32_t>() + R, R, T, C, k,
                            blank_.as<float>(), topv_.as<float>(), topi_.as<int64_t>(), st);
      INDRA_CK(cudaMemcpyAsync(hb, blank_.ptr, f * sizeof(float), cudaMemcpyDeviceToHost, st));
      INDRA_CK(cudaMemcpyAsync(hv, topv_.ptr, f * k * sizeof(float), cudaMemcpyDeviceToHost, st));
      INDRA_CK(cudaMemcpyAsync(hi, topi_.ptr, f * k * sizeof(int64_t), cudaMemcpyDeviceToHost, st));
      INDRA_CK(cudaStreamSynchronize(st));
    }
    return nb::make_tuple(ab, av, ai);
  }

  // This model's input ``name`` from ``src``'s tensor ``src_name`` (an output
  // of its last run, or an input), copied on the device; with ``index`` >= 0,
  // the index-th slice of its first axis (a per-layer cache of a stacked one).
  void bind(const std::string& name, PyModel& src, const std::string& src_name, int index) {
    Tensor& s = src.m_->tensor(src_name);
    Shape shape = s.shape;
    size_t bytes = s.nbytes(), offset = 0;
    if (index >= 0) {
      if (shape.empty() || index >= shape[0]) throw nb::index_error("indra-ocr-cuda: bind index outside the tensor");
      bytes /= static_cast<size_t>(shape[0]);
      offset = bytes * static_cast<size_t>(index);
      shape.erase(shape.begin());
    }
    Tensor& d = m_->tensor(name);
    if (d.dtype != s.dtype) throw nb::type_error(("indra-ocr-cuda: bind " + name + ": the types differ").c_str());
    nb::gil_scoped_release nogil;
    // src's last run was synchronized (run() and bind() end with a synchronize)
    m_->set_input_shape(name, shape);
    if (bytes)
      INDRA_CK(cudaMemcpyAsync(d.buf.ptr, static_cast<const char*>(s.buf.ptr) + offset, bytes,
                               cudaMemcpyDeviceToDevice, m_->stream()));
    INDRA_CK(cudaStreamSynchronize(m_->stream()));
  }

  // For each (output, input) pair: the last run's output becomes the input of
  // the next run (a decoder's grown caches), on the device.
  void chain(const std::vector<std::pair<std::string, std::string>>& pairs) {
    nb::gil_scoped_release nogil;
    for (auto& [out, in] : pairs) {
      Tensor& s = m_->tensor(out);
      Tensor& d = m_->tensor(in);
      m_->set_input_shape(in, s.shape);
      if (s.nbytes())
        INDRA_CK(cudaMemcpyAsync(d.buf.ptr, s.buf.ptr, s.nbytes(), cudaMemcpyDeviceToDevice, m_->stream()));
    }
    INDRA_CK(cudaStreamSynchronize(m_->stream()));
  }

  // Keep only the rows ``rows`` (increasing) of the first axis of these inputs,
  // in that order, on the device.
  void keep_rows(const std::vector<std::string>& names, const std::vector<int32_t>& rows) {
    nb::gil_scoped_release nogil;
    cudaStream_t st = m_->stream();
    const int R = static_cast<int>(rows.size());
    idx_.reserve(R * sizeof(int32_t) + 4);
    INDRA_CK(cudaMemcpyAsync(idx_.ptr, rows.data(), R * sizeof(int32_t), cudaMemcpyHostToDevice, st));
    for (auto& name : names) {
      Tensor& t = m_->tensor(name);
      if (t.shape.empty()) throw std::runtime_error("indra-ocr-cuda: keep_rows on a scalar");
      const int64_t B = t.shape[0];
      for (auto r : rows)
        if (r < 0 || r >= B) throw std::out_of_range("indra-ocr-cuda: keep_rows: a row outside the tensor");
      const size_t row = B ? t.nbytes() / static_cast<size_t>(B) : 0;
      Shape s = t.shape;
      s[0] = R;
      if (row && R) {
        gather_.reserve(row * R);
        indra_cuda::gather_rows(t.buf.ptr, gather_.ptr, idx_.as<int32_t>(), R, row, st);
        INDRA_CK(cudaMemcpyAsync(t.buf.ptr, gather_.ptr, row * R, cudaMemcpyDeviceToDevice, st));
      }
      m_->set_input_shape(name, s);
    }
    INDRA_CK(cudaStreamSynchronize(st));
  }

  bool built() const { return m_->built(); }
  double build_seconds() const { return m_->build_seconds(); }
  int device() const { return m_->device(); }

 private:
  static void check_logits(const Tensor& L, const Tensor& N) {
    if (L.dtype != DType::f32 || L.shape.size() != 3) throw nb::value_error("indra-ocr-cuda: logits [B, T, C] float32");
    if (N.dtype != DType::i64 || N.shape.size() != 1 || N.shape[0] != L.shape[0])
      throw nb::value_error("indra-ocr-cuda: lengths [B] int64");
  }

  std::unique_ptr<Model> m_;
  DeviceBuffer conf_, idx_, blank_, topv_, topi_, gather_;
};

}  // namespace

NB_MODULE(_core, m) {
  m.attr("API") = API;
  m.def("device_count", [] {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0;
  });
  m.def("device_name", [](int d) {
    cudaDeviceProp p{};
    INDRA_CK(cudaGetDeviceProperties(&p, d));
    return std::string(p.name) + " (sm_" + std::to_string(p.major) + std::to_string(p.minor) + ")";
  });
  nb::class_<PyModel>(m, "Model")
      .def(nb::init<nb::bytes, const std::map<std::string, std::tuple<Shape, Shape, Shape>>&, const std::string&, size_t,
                    int, int>(),
           "onnx"_a, "profiles"_a, "cache"_a = "", "workspace"_a = size_t(1) << 30, "device"_a = 0, "opt_level"_a = 3,
           "An ONNX model as a TensorRT engine (built, or loaded from ``cache``).")
      .def_prop_ro("inputs", [](const PyModel& p) { return p.names(true); })
      .def_prop_ro("outputs", [](const PyModel& p) { return p.names(false); })
      .def_prop_ro("built", &PyModel::built)
      .def_prop_ro("build_seconds", &PyModel::build_seconds)
      .def_prop_ro("device", &PyModel::device)
      .def("run", &PyModel::run, "feed"_a, "fetch"_a)
      .def("frame_conf", &PyModel::frame_conf, "logits"_a = "logits", "lengths"_a = "lengths")
      .def("lm_inputs", &PyModel::lm_inputs, "logits"_a, "rows"_a, "n"_a, "k"_a = 8)
      .def("bind", &PyModel::bind, "name"_a, "src"_a, "src_name"_a, "index"_a = -1)
      .def("chain", &PyModel::chain, "pairs"_a)
      .def("keep_rows", &PyModel::keep_rows, "names"_a, "rows"_a);
}
