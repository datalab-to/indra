"""Tiled detection for large pages (OrtOCR._detect_tiled): a fake detector that marks ink as line core, and a page with
one long text line that crosses a tile seam. The tiles must give back ONE box over the whole line (core ownership +
joining the pieces at the seam), in page pixels."""
import numpy as np

from indra_ocr import pipeline as P
from indra_ocr.det_post import OUT_STRIDE


class _FakeDet:
    """det.onnx stand-in: core logit high where the input has ink, top/bottom distances to the ink band of the row."""

    def run(self, _names, feed):
        x = feed["x"][0, 0]
        ink = x[::OUT_STRIDE, ::OUT_STRIDE] > 0.5
        out = np.zeros((3,) + ink.shape, np.float32)
        out[0] = np.where(ink, 8.0, -8.0)
        out[1] = 3.0  # half the line height above the core centre (input px)
        out[2] = 3.0
        return [out[None]]


def _ocr_with_page(h, w, band):
    o = P.OrtOCR.__new__(P.OrtOCR)
    o.det = _FakeDet()
    region = np.full((h // 8 // OUT_STRIDE + 1, w // 8 // OUT_STRIDE + 1), 5.0, np.float32)
    o._prob = (region, 1 / 8)  # every part of the page holds text
    gray = np.full((h, w), 255, np.uint8)
    y0, y1, x0, x1 = band
    gray[y0:y1, x0:x1] = 0
    return o, gray


def test_line_across_a_seam_comes_back_whole():
    h, w = 1600, 6000
    band = (800, 806, 300, 5700)  # one thin line across several tiles
    o, gray = _ocr_with_page(h, w, band)
    boxes = o._detect_tiled(gray)
    on_line = boxes[(boxes[:, 1] < band[1]) & (boxes[:, 3] > band[0])]
    assert len(on_line) == 1
    x0, _, x1, _, _ = on_line[0]
    assert x0 < band[2] + 20 and x1 > band[3] - 20


def test_blank_page_runs_no_tile():
    o, gray = _ocr_with_page(1600, 6000, (0, 0, 0, 0))
    o._prob = (np.full_like(o._prob[0], -5.0), o._prob[1])  # no text anywhere
    assert len(o._detect_tiled(gray)) == 0
