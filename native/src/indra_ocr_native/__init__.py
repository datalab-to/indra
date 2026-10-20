"""Optional C++ kernels for Indra OCR.

indra_ocr uses them when this package imports and its API matches; without it
(or with INDRA_OCR_NATIVE=0) every step runs as plain Python.  The kernels give
the same results as the Python code they replace (see indra_ocr/native.py).
"""

from ._core import (  # noqa: F401
    API,
    assign_chars,
    beam_search,
    cluster,
    ctc_greedy,
    ctc_topk,
    is_space,
    pdf_char_data,
    pdf_char_loop,
    NgramLM,
    split_points,
)

__version__ = "0.1.0"
