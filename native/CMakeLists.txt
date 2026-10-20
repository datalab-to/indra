cmake_minimum_required(VERSION 3.24...3.31)
project(indra_ocr_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
             OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Google Highway: portable SIMD with run-time dispatch (SSE2 to AVX-512 on x86,
# NEON and SVE on Arm).  An installed Highway 1.4+ is used when CMake finds
# one; otherwise the pinned release is downloaded (or taken from
# FETCHCONTENT_SOURCE_DIR_HIGHWAY for an offline build).  Static, no tests.
include(FetchContent)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)  # contrib/math is header-only
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_FORCE_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  highway
  URL https://github.com/google/highway/archive/refs/tags/1.4.0.tar.gz
  URL_HASH SHA256=e72241ac9524bb653ae52ced768b508045d4438726a303f10181a38f764a453c
  FIND_PACKAGE_ARGS 1.4 NAMES hwy)
FetchContent_MakeAvailable(highway)
foreach (extra hwy_list_targets hwy_test)  # Highway's own tools: not needed
  if (TARGET ${extra})
    set_target_properties(${extra} PROPERTIES EXCLUDE_FROM_ALL ON)
  endif()
endforeach()
if (TARGET hwy::hwy)
  set(INDRA_HWY hwy::hwy)
else()
  set(INDRA_HWY hwy)
endif()

nanobind_add_module(_core STABLE_ABI NB_STATIC LTO
  src/cpp/module.cpp
  src/cpp/textlayer.cpp
  src/cpp/ctc.cpp
  src/cpp/ctc_hwy.cpp
  src/cpp/lm.cpp
  src/cpp/order.cpp)
target_link_libraries(_core PRIVATE ${INDRA_HWY})

# The kernels reproduce numpy's results bit for bit wherever the Python code
# compares or sorts them: IEEE arithmetic as written, no fused multiply-adds,
# no reassociation (no fast-math).  (The CTC head's SIMD code calls Highway's
# own operations, FMA where the target has it: its sums match to float32
# rounding, as the ONNX graph's do.)
if (MSVC)
  target_compile_options(_core PRIVATE /O2 /fp:precise /permissive- /Zc:__cplusplus)
else()
  target_compile_options(_core PRIVATE -O3 -ffp-contract=off -fno-fast-math)
endif()

install(TARGETS _core LIBRARY DESTINATION indra_ocr_native)
