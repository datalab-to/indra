"""OrtOCR.read_turn: an unsure orientation is settled by reading lines each way."""

import numpy as np

from indra_ocr import pipeline as pl


def _engine(up: float, ccw: float, cw: float, boxes=True):
    eng = object.__new__(pl.OrtOCR)
    # The top-left pixel tells the sides apart: 100 on the page as it stands, 0 on
    # the page turned counter-clockwise, 255 turned clockwise (half round from it).
    pages = {}
    for k, v in ((0, 100), (3, 0)):
        pages[k] = np.full((100, 200), 255, np.uint8)
        pages[k][0, 0] = v
    eng._turn = lambda gray, k, a, small=None: (pages[k], k, 0.0)
    eng.detect = lambda img: (
        np.array([[10, 10, 150, 30, 0.9], [10, 40, 120, 60, 0.9]], np.float32)
        if boxes
        else np.zeros((0, 5), np.float32)
    )

    def recognize(img, bx, **kw):
        c = {100: up, 0: ccw, 255: cw}[int(img[0, 0])]
        return [("x", c, [])] * len(bx)

    eng.recognize = recognize
    return eng


def _probs(p3: float, pv: float) -> np.ndarray:
    p = np.array([0.02, 0.0, 0.0, p3, pv], np.float32)
    p[1] = 1.0 - p.sum()
    return p


def test_surer_clockwise_reading_turns_the_page_once():
    eng = _engine(up=0.3, ccw=0.4, cw=0.9)
    assert eng.read_turn(None, 0.0, None, _probs(0.3, 0.5)) == 1


def test_counter_clockwise_reading_keeps_orient_choice_of_vertical_or_turn():
    eng = _engine(up=0.3, ccw=0.9, cw=0.4)
    assert eng.read_turn(None, 0.0, None, _probs(0.2, 0.6)) == pl.VERTICAL
    assert eng.read_turn(None, 0.0, None, _probs(0.6, 0.2)) == 3


def test_surer_upright_reading_keeps_the_page():
    # orient.onnx unsure on an upright page (seen on Hebrew): the lines decide
    eng = _engine(up=0.94, ccw=0.6, cw=0.65)
    assert eng.read_turn(None, 0.0, None, _probs(0.77, 0.02)) == 0


def test_no_lines_reads_upright():
    eng = _engine(up=0.9, ccw=0.9, cw=0.4, boxes=False)
    assert eng.read_turn(None, 0.0, None, _probs(0.2, 0.6)) == 0


def _half_engine(up: float, down: float, boxes=True):
    eng = object.__new__(pl.OrtOCR)
    # The top-left pixel tells the page as it stands (100) from the page turned
    # half round (0).
    pages = {}
    for k, v in ((0, 100), (2, 0)):
        pages[k] = np.full((100, 200), 255, np.uint8)
        pages[k][0, 0] = v
    eng._turn = lambda gray, k, a, small=None: (pages[k], k, 0.0)
    eng.detect = lambda img: (
        np.array([[10, 10, 150, 30, 0.9]], np.float32)
        if boxes
        else np.zeros((0, 5), np.float32)
    )
    eng.recognize = lambda img, bx, **kw: [("x", {100: up, 0: down}[int(img[0, 0])], [])] * len(bx)
    return eng


def test_half_turn_is_kept_when_the_turned_lines_read_surer():
    assert _half_engine(up=0.4, down=0.9).read_half_turn(None, 0.0, None) == 2


def test_half_turn_is_undone_when_the_upright_lines_read_surer():
    # an upright book scan that orient.onnx saw upside down (issue #32)
    assert _half_engine(up=0.9, down=0.5).read_half_turn(None, 0.0, None) == 0
    assert _half_engine(up=0.9, down=0.9, boxes=False).read_half_turn(None, 0.0, None) == 0
