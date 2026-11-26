"""The detector's input, written in place, is the expression it replaces, to the bit."""

import numpy as np

from indra_ocr.det_post import page_to_input


def _reference(gray, long_side):
    import math

    import cv2

    h, w = gray.shape
    scale = long_side / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    if (nh, nw) != (h, w):
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    out = np.zeros((math.ceil(nh / 32) * 32, math.ceil(nw / 32) * 32), np.float32)
    out[:nh, :nw] = 1.0 - gray.astype(np.float32) / 255.0
    return out, scale


def test_page_to_input_is_one_minus_gray_over_255():
    rng = np.random.default_rng(0)
    every = np.arange(256, dtype=np.uint8).reshape(16, 16)
    for gray, long_side in ((every, 16), (every, 40), (rng.integers(0, 256, (300, 211), dtype=np.uint8), 256)):
        a, sa = page_to_input(gray, long_side)
        b, sb = _reference(gray, long_side)
        assert sa == sb and a.dtype == b.dtype and a.tobytes() == b.tobytes()
