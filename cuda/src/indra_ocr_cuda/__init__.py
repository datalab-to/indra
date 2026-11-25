"""Optional GPU engine for Indra OCR: the models on TensorRT, with their data kept in VRAM.

indra_ocr uses it with ``load(device="cuda")`` when this package imports and its
API matches (indra_ocr/gpu.py); every model the engine cannot build runs on ONNX
Runtime's CUDA provider instead, and without the package on ONNX Runtime alone.
"""

from __future__ import annotations

import ctypes
import os


def _preload_tensorrt() -> None:
    """Load TensorRT's libraries from the tensorrt-cu13 wheels when they are
    installed (the module also finds them on its build-time rpath or the library
    path)."""
    try:
        import tensorrt_libs  # the tensorrt-cu13-libs wheel
    except ImportError:
        return
    d = os.path.dirname(tensorrt_libs.__file__)
    for name in ("libnvinfer.so.11", "libnvonnxparser.so.11"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)


_preload_tensorrt()

from ._core import API, Model, device_count, device_name  # noqa: E402,F401

__version__ = "0.1.0"
