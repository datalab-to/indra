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

`INDRA_OCR_NATIVE=0` turns the kernels off without uninstalling them.

Building from source needs a C++17 compiler and CMake:

```shell
pip install ./native
```
