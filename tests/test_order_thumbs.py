"""Order models with line thumbnails: the package cuts them and feeds them."""

import numpy as np

from indra_ocr import pipeline as pl


class _Sess:
    def __init__(self):
        self.feed = None

    def run(self, _, feed):
        self.feed = feed
        n = feed["feats"].shape[1]
        return [np.zeros((1, n, n), np.float32)]


def test_reading_order_feeds_thumbs_from_the_image_boxes():
    eng = object.__new__(pl.OrtOCR)
    eng.order = _Sess()
    eng._order_thumbs = True
    g = np.full((100, 200), 255, np.uint8)
    g[10:20, 10:110] = 0
    real = np.array([[10, 10, 110, 20, 1], [10, 40, 110, 50, 1]], np.float32)
    mirrored = real.copy()
    mirrored[:, [0, 2]] = 200 - real[:, [2, 0]]
    eng.reading_order(mirrored, ["a", "b"], (200, 100), g, real)
    th = eng.order.feed["thumbs"]
    assert th.shape == (1, 2, 1, pl.THUMB_H, pl.THUMB_W)
    assert th[0, 0].max() > 0.9 and th[0, 1].max() == 0.0  # ink only where the first line is
    assert eng.order.feed["has_img"].tolist() == [[1.0, 1.0]]
    eng.reading_order(real, ["a", "b"], (200, 100))  # no image: has_img 0
    assert eng.order.feed["has_img"].tolist() == [[0.0, 0.0]]


def test_reading_order_uses_the_model_thumb_size_and_mode():
    eng = object.__new__(pl.OrtOCR)
    eng.order = _Sess()
    eng._order_thumbs = True
    eng._thumb = (12, 48, "linear")
    g = np.full((100, 200), 255, np.uint8)
    b = np.array([[10, 10, 110, 20, 1], [10, 40, 110, 50, 1]], np.float32)
    eng.reading_order(b, ["a", "b"], (200, 100), g, b)
    assert eng.order.feed["thumbs"].shape == (1, 2, 1, 12, 48)
    eng.reading_order(b, ["a", "b"], (200, 100))
    assert eng.order.feed["thumbs"].shape == (1, 2, 1, 12, 48)
