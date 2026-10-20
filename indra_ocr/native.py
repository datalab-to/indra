"""Optional C++ kernels: the indra-ocr-native package (pip install "indra-ocr[native]").

``lib`` is its kernel module, or None when the package is not installed, does
not load, does not match this version of indra_ocr, or when the environment
sets INDRA_OCR_NATIVE=0.  Every caller keeps its Python code for the None case,
and the kernels give the same results as that code:

- textlayer.extract_runs: pdfium's per-character calls and the character loop;
- document.assign_runs: each text-layer character to its detected line;
- OrtOCR.recognize: the recognizers' CTC head (the graphs are cut at the head,
  see onnx_cut; confidences agree to float32 rounding);
- lm.beam_search and the n-gram language model;
- OrtOCR.split_line: where a line splits at wide gaps;
- order_post._cluster: the average-linkage clustering of blocks and cells.

Reliability: the kernels check their inputs and raise rather than read out of
bounds, and every caller catches an exception from a kernel, reports it once
(``failed``) and runs its Python code for that call.  A recognizer whose cut
graph fails reads with its full graph.  When convert_many's worker processes
die (a crash in a kernel would show up so), the remaining pages are read in
the calling process with the kernels off (``disabled``).
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager

API = 1  # the indra_ocr_native.API this version calls


def _load():
    if os.environ.get("INDRA_OCR_NATIVE", "").strip() == "0":
        return None
    try:
        import indra_ocr_native as m
    except ImportError:  # not installed
        return None
    except Exception as e:  # noqa: BLE001  (installed but broken: run the Python code)
        warnings.warn(
            f"indra-ocr-native does not load ({type(e).__name__}: {e}); "
            "indra_ocr runs its Python code",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if getattr(m, "API", None) != API:
        warnings.warn(
            f"indra-ocr-native {getattr(m, '__version__', '?')} has API "
            f"{getattr(m, 'API', None)}; this indra-ocr needs API {API}, so it runs "
            'its Python code (reinstall with pip install "indra-ocr[native]")',
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return m


lib = _load()
_reported: set[str] = set()


def failed(kernel: str, err: BaseException) -> None:
    """A kernel raised: the caller runs its Python code instead.  Warns once per
    kernel and process (a kernel that raises on real input is a bug)."""
    if kernel in _reported:
        return
    _reported.add(kernel)
    warnings.warn(
        f"indra_ocr native kernel {kernel} failed ({type(err).__name__}: {err}); "
        "that step runs as Python code. Please report this at "
        "https://github.com/datalab-to/indra/issues",
        RuntimeWarning,
        stacklevel=3,
    )


@contextmanager
def disabled():
    """The kernels off inside the block: every step runs its Python code."""
    global lib
    saved, lib = lib, None
    try:
        yield
    finally:
        lib = saved


def status() -> str:
    """Whether the kernels are on, and why not (one line)."""
    if lib is not None:
        extras = [f"indra-ocr-native {getattr(lib, '__version__', '?')}"]
        if hasattr(lib, "ctc_isa"):
            extras.append(lib.ctc_isa())  # the SIMD target of the CTC head
        return f"native kernels on ({', '.join(extras)})"
    if os.environ.get("INDRA_OCR_NATIVE", "").strip() == "0":
        return "native kernels off (INDRA_OCR_NATIVE=0)"
    return 'native kernels off (pip install "indra-ocr[native]")'
