// Shared helpers of the GPU engine: CUDA error checks and device buffers.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace indra_cuda {

inline void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

#define INDRA_CK(x) ::indra_cuda::check((x), #x)

// Device memory at a fixed address: sized once, grown (x1.5) only when a run
// needs more, so the addresses an engine or a CUDA graph holds stay valid
// between runs.  cudaFree synchronizes the device, so growth is rare by design.
struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), bytes(o.bytes) { o.ptr = nullptr, o.bytes = 0; }
  ~DeviceBuffer() {
    if (ptr) cudaFree(ptr);
  }
  // true when the address changed
  bool reserve(size_t n) {
    if (n <= bytes) return false;
    size_t m = bytes ? (bytes + bytes / 2 > n ? bytes + bytes / 2 : n) : n;
    if (ptr) INDRA_CK(cudaFree(ptr));
    ptr = nullptr;
    INDRA_CK(cudaMalloc(&ptr, m));
    bytes = m;
    return true;
  }
  template <class T>
  T* as() const {
    return static_cast<T*>(ptr);
  }
};

}  // namespace indra_cuda
