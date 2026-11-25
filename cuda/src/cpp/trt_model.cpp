#include "trt_model.h"

#include <NvOnnxParser.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <sstream>

namespace indra_cuda {

namespace {

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity s, const char* msg) noexcept override {
    if (s > Severity::kWARNING) return;
    std::string m(msg);
    // the int64 I/O of the models (widths, ids, lengths) is intended
    if (m.find("Int64 binding") != std::string::npos) return;
    // engines are cached per GPU model: another card of the same model is fine
    if (m.find("across different models of devices") != std::string::npos) return;
    std::fprintf(stderr, "[indra-ocr-cuda] TensorRT: %s\n", msg);
  }
};

Logger& logger() {
  static Logger l;
  return l;
}

DType from_trt(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return DType::f32;
    case nvinfer1::DataType::kHALF: return DType::f16;
    case nvinfer1::DataType::kINT64: return DType::i64;
    case nvinfer1::DataType::kINT32: return DType::i32;
    case nvinfer1::DataType::kBOOL: return DType::b8;
    default: throw std::runtime_error("indra-ocr-cuda: a tensor type this engine does not handle");
  }
}

nvinfer1::Dims dims(const Shape& s) {
  nvinfer1::Dims d{};
  d.nbDims = static_cast<int32_t>(s.size());
  for (size_t i = 0; i < s.size(); ++i) d.d[i] = s[i];
  return d;
}

Shape shape(const nvinfer1::Dims& d) { return Shape(d.d, d.d + d.nbDims); }

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), {});
}

}  // namespace

size_t dtype_size(DType t) {
  switch (t) {
    case DType::f32: return 4;
    case DType::f16: return 2;
    case DType::i64: return 8;
    case DType::i32: return 4;
    case DType::b8: return 1;
  }
  return 0;
}

size_t Tensor::count() const {
  size_t n = 1;
  for (auto v : shape) n *= static_cast<size_t>(v < 0 ? 0 : v);
  return n;
}

Model::Model(const std::string& onnx, const std::map<std::string, Profile>& profiles,
             const std::string& cache, size_t workspace, int device, int opt_level)
    : profiles_(profiles), device_(device) {
  INDRA_CK(cudaSetDevice(device_));
  INDRA_CK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  load_or_build(onnx, cache, workspace, opt_level);
  allocate();
}

Model::~Model() {
  ctx_.reset();
  engine_.reset();
  runtime_.reset();
  if (stream_) cudaStreamDestroy(stream_);
}

void Model::load_or_build(const std::string& onnx, const std::string& cache, size_t workspace, int opt_level) {
  runtime_.reset(nvinfer1::createInferRuntime(logger()));
  if (!cache.empty() && std::filesystem::exists(cache)) {
    std::string plan = read_file(cache);
    engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (engine_) return;
    std::fprintf(stderr, "[indra-ocr-cuda] the cached engine %s does not load: building it again\n", cache.c_str());
  }
  auto t0 = std::chrono::steady_clock::now();
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger()));
  std::unique_ptr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(0));
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, logger()));
  if (!parser->parse(onnx.data(), onnx.size())) {
    std::ostringstream err;
    err << "indra-ocr-cuda: TensorRT cannot parse the model:";
    for (int i = 0; i < parser->getNbErrors(); ++i) err << " " << parser->getError(i)->desc();
    throw std::runtime_error(err.str());
  }
  std::unique_ptr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
  cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace);
  // FP32 files run in FP32: TF32 would change the numbers (10-bit mantissa).
  cfg->clearFlag(nvinfer1::BuilderFlag::kTF32);
  cfg->setBuilderOptimizationLevel(opt_level);
  auto* prof = builder->createOptimizationProfile();
  for (int i = 0; i < net->getNbInputs(); ++i) {
    auto* in = net->getInput(i);
    auto it = profiles_.find(in->getName());
    if (it == profiles_.end()) {
      // a fixed-shape input needs no profile
      bool dynamic = false;
      auto d = in->getDimensions();
      for (int k = 0; k < d.nbDims; ++k) dynamic |= d.d[k] < 0;
      if (dynamic) throw std::runtime_error(std::string("indra-ocr-cuda: no profile for input ") + in->getName());
      continue;
    }
    prof->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kMIN, dims(it->second.min));
    prof->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kOPT, dims(it->second.opt));
    prof->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kMAX, dims(it->second.max));
  }
  cfg->addOptimizationProfile(prof);
  std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*net, *cfg));
  if (!plan) throw std::runtime_error("indra-ocr-cuda: TensorRT could not build the engine");
  engine_.reset(runtime_->deserializeCudaEngine(plan->data(), plan->size()));
  if (!engine_) throw std::runtime_error("indra-ocr-cuda: the built engine does not load");
  built_ = true;
  build_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (!cache.empty()) {
    // written to a temporary name first: a reader never sees half a file
    std::filesystem::create_directories(std::filesystem::path(cache).parent_path());
    std::string tmp = cache + ".tmp" + std::to_string(::getpid());
    {
      std::ofstream f(tmp, std::ios::binary);
      f.write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
    }
    std::filesystem::rename(tmp, cache);
  }
}

void Model::allocate() {
  // The context's activation memory and the I/O buffers follow the shapes the
  // runs use (grown x1.5 when a run needs more), not the profile's largest:
  // rec_cjk's logits at 64 lines of 4,096 px would hold 3 GB.
  ctx_.reset(engine_->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  if (!ctx_) throw std::runtime_error("indra-ocr-cuda: no execution context");
  int n = engine_->getNbIOTensors();
  io_.clear();
  io_.reserve(n);
  for (int i = 0; i < n; ++i) {
    Tensor t;
    t.name = engine_->getIOTensorName(i);
    t.input = engine_->getTensorIOMode(t.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    t.dtype = from_trt(engine_->getTensorDataType(t.name.c_str()));
    io_.push_back(std::move(t));
  }
}

Tensor& Model::tensor(const std::string& name) {
  for (auto& t : io_)
    if (t.name == name) return t;
  throw std::runtime_error("indra-ocr-cuda: no tensor " + name);
}

void Model::set_input_shape(const std::string& name, const Shape& s) {
  auto& t = tensor(name);
  if (!t.input) throw std::runtime_error("indra-ocr-cuda: " + name + " is not an input");
  auto it = profiles_.find(name);
  if (it != profiles_.end()) {
    const auto& p = it->second;
    if (s.size() != p.max.size()) throw std::runtime_error("indra-ocr-cuda: " + name + ": wrong rank");
    for (size_t i = 0; i < s.size(); ++i)
      if (s[i] < p.min[i] || s[i] > p.max[i])
        throw std::out_of_range("indra-ocr-cuda: " + name + ": shape outside the engine's profile");
  }
  t.shape = s;
  if (!ctx_->setInputShape(name.c_str(), dims(s)))
    throw std::runtime_error("indra-ocr-cuda: " + name + ": the engine refused the shape");
  if (t.buf.reserve(std::max<size_t>(t.nbytes(), 1))) ctx_->setTensorAddress(name.c_str(), t.buf.ptr);
}

void Model::enqueue() {
  for (auto& t : io_) {
    if (t.input) {
      if (!t.buf.ptr) throw std::runtime_error("indra-ocr-cuda: no shape for input " + t.name);
      continue;
    }
    t.shape = shape(ctx_->getTensorShape(t.name.c_str()));
    if (t.buf.reserve(std::max<size_t>(t.nbytes(), 1))) ctx_->setTensorAddress(t.name.c_str(), t.buf.ptr);
  }
  const size_t need = std::max<size_t>(ctx_->updateDeviceMemorySizeForShapes(), 1);
  if (scratch_.reserve(need) || !scratch_set_) {
    ctx_->setDeviceMemoryV2(scratch_.ptr, static_cast<int64_t>(scratch_.bytes));
    scratch_set_ = true;
  }
  if (!ctx_->enqueueV3(stream_)) throw std::runtime_error("indra-ocr-cuda: the engine run failed");
}

}  // namespace indra_cuda
