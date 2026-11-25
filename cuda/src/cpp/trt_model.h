// One ONNX model as a TensorRT engine, with its inputs and outputs in device
// buffers at fixed addresses (sized for the profile's largest shapes when the
// engine loads).  The engine is built once and cached on disk.
#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common.h"

namespace indra_cuda {

using Shape = std::vector<int64_t>;

struct Profile {
  Shape min, opt, max;
};

// numpy-style type codes of the tensors the models use
enum class DType { f32, f16, i64, i32, b8 };
size_t dtype_size(DType t);

struct Tensor {
  std::string name;
  bool input = false;
  DType dtype = DType::f32;
  DeviceBuffer buf;  // fixed address
  Shape shape;       // the last run's shape
  size_t count() const;
  size_t nbytes() const { return count() * dtype_size(dtype); }
};

class Model {
 public:
  // onnx: the model file's bytes.  profiles: each dynamic input's (min, opt,
  // max) shape.  cache: the engine file ("" for none); a cache that does not
  // load is built again.  workspace: the builder's workspace limit, bytes.
  Model(const std::string& onnx, const std::map<std::string, Profile>& profiles,
        const std::string& cache, size_t workspace, int device, int opt_level);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  std::vector<Tensor>& tensors() { return io_; }
  Tensor& tensor(const std::string& name);
  const std::map<std::string, Profile>& profiles() const { return profiles_; }
  cudaStream_t stream() const { return stream_; }
  int device() const { return device_; }
  bool built() const { return built_; }  // built in this process (no cache)
  double build_seconds() const { return build_s_; }

  // An input's shape for the next run (within its profile).
  void set_input_shape(const std::string& name, const Shape& shape);
  // Run on the stream; outputs' shapes are known after this.
  void enqueue();

 private:
  void load_or_build(const std::string& onnx, const std::string& cache, size_t workspace, int opt_level);
  void allocate();

  std::map<std::string, Profile> profiles_;
  std::vector<Tensor> io_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
  DeviceBuffer scratch_;
  bool scratch_set_ = false;
  cudaStream_t stream_ = nullptr;
  int device_ = 0;
  bool built_ = false;
  double build_s_ = 0;
};

}  // namespace indra_cuda
