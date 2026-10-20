# indra-ocr-native

Optional C++ kernels for [Indra OCR](https://github.com/datalab-to/indra).

```shell
pip install "indra-ocr[native]"
```

Indra OCR runs the same without this package: each step falls back to its
Python code. With it, the hot loops run in C++:

- the recognizers' CTC head (one pass over the logits instead of eight graph
  ops; the recognizer graphs are cut at the head when they load),
- the beam search with the n-gram language model,
- the per-character loops of the PDF text layer (pdfium calls, characters to
  lines),
- line splitting and the reading-order clustering.

`INDRA_OCR_NATIVE=0` turns the kernels off without uninstalling them, and
`python -c "import indra_ocr.native as n; print(n.status())"` says whether they
are on, and the CTC head's SIMD target.

**SIMD.** The CTC head is written once with [Google Highway](https://github.com/google/highway)
and compiled for every target it supports: SSE2 to AVX-512 on x86 (AVX2 with
MSVC), NEON and SVE on Arm. The best one the CPU has is picked at run time.
On an Intel i7-13620H the AVX2 head is 1.4-1.5x faster than the hand-written
SSE2 code of 0.1; it is a small part of a page, so pages read 1.02-1.03x
faster on one core.

**Threads.** The kernels run on one thread. The CTC head runs in short bursts
between ONNX Runtime calls, whose own thread pool uses the cores: spreading it
over OpenMP threads made pages 3-14% slower with `indra_ocr.load(threads=4)`,
whether the idle threads spun or slept.

Building from source needs a C++20 compiler (GCC 10+, Clang 12+, MSVC 2019
16.11+), CMake 3.24+ and the network (for Highway 1.4, unless CMake finds an
installed one, or `-DFETCHCONTENT_SOURCE_DIR_HIGHWAY=<folder>` points at a
copy).

```shell
pip install ./native
```
