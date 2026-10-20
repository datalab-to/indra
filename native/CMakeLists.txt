cmake_minimum_required(VERSION 3.18...3.30)
project(indra_ocr_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
             OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core STABLE_ABI NB_STATIC
  src/cpp/module.cpp
  src/cpp/textlayer.cpp
  src/cpp/ctc.cpp
  src/cpp/lm.cpp
  src/cpp/order.cpp)

# The kernels reproduce numpy's results bit for bit wherever the Python code
# compares or sorts them: IEEE arithmetic as written, no fused multiply-adds,
# no reassociation (no fast-math).
if (MSVC)
  target_compile_options(_core PRIVATE /O2 /fp:precise)
else()
  target_compile_options(_core PRIVATE -O3 -ffp-contract=off -fno-fast-math)
endif()

install(TARGETS _core LIBRARY DESTINATION indra_ocr_native)
