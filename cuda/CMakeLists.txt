cmake_minimum_required(VERSION 3.24...3.31)
project(indra_ocr_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# The GPUs to compile for: INDRA_CUDA_ARCHS (e.g. "80;86;89;90;120"), else the
# GPU of the machine that builds.
if (DEFINED ENV{INDRA_CUDA_ARCHS})
  set(CMAKE_CUDA_ARCHITECTURES "$ENV{INDRA_CUDA_ARCHS}")
elseif (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit 12.8 REQUIRED)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
             OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# TensorRT 11: TENSORRT_ROOT (the SDK's include/ and lib/).
set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT SDK (include/, lib/)")
find_path(TRT_INCLUDE NvInfer.h HINTS ${TENSORRT_ROOT}/include REQUIRED)
find_library(TRT_NVINFER nvinfer HINTS ${TENSORRT_ROOT}/lib REQUIRED)
find_library(TRT_ONNXPARSER nvonnxparser HINTS ${TENSORRT_ROOT}/lib REQUIRED)
get_filename_component(TRT_LIBDIR ${TRT_NVINFER} DIRECTORY)

nanobind_add_module(_core STABLE_ABI NB_STATIC
  src/cpp/module.cpp
  src/cpp/trt_model.cpp
  src/cpp/ctc_kernels.cu)
target_include_directories(_core PRIVATE src/cpp ${TRT_INCLUDE})
# The CUDA runtime linked statically: the process may hold another libcudart
# (onnxruntime-gpu's wheels); the primary context is shared either way.
target_link_libraries(_core PRIVATE ${TRT_NVINFER} ${TRT_ONNXPARSER} CUDA::cudart_static)
set_target_properties(_core PROPERTIES
  BUILD_RPATH "${TRT_LIBDIR}" INSTALL_RPATH "${TRT_LIBDIR}")
target_compile_options(_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CXX>:-O2>)

install(TARGETS _core LIBRARY DESTINATION indra_ocr_cuda)
