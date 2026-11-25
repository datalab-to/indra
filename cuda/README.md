# indra-ocr-cuda

An optional GPU engine for [Indra OCR](https://github.com/datalab-to/indra): the
models run as TensorRT engines on fixed device buffers, and the recognizers'
logits stay in VRAM, where CUDA kernels read the confidences and the beam
search's inputs only for the frames and lines that use them.

```shell
TENSORRT_ROOT=/path/to/TensorRT-11 pip install ./cuda
```

```python
import indra_ocr
conv = indra_ocr.load(device="cuda")
```

Indra OCR runs the same without this package: each model the engine does not
build runs on ONNX Runtime's CUDA provider, or on the CPU.

**Precision.** The engines are FP32 (TF32 off) by default, within 1e-4 of ONNX
Runtime on the CPU.  `INDRA_GPU_PRECISION=fp16` builds FP16 engines (the model
files converted with their inputs and outputs kept FP32).

**Engines** are built the first time a model loads (10 to 40 seconds each) and
cached in `~/.cache/indra-ocr/trt` (`INDRA_TRT_CACHE`), keyed by the model file,
the GPU, TensorRT's version, the precision and the shapes.

Requires an NVIDIA GPU (sm_80 or newer), CUDA 13 and TensorRT 11.
