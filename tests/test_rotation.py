"""A known page rotation skips orient.onnx; orient_min sets when its turn is taken."""

import numpy as np
import pytest

from indra_ocr import pipeline as pl
from indra_ocr.document import Converter
from indra_ocr.pipeline import Page, quarter_turns
from test_inputs import StubOCR, _pdf


def test_quarter_turns():
    assert [quarter_turns(d) for d in (0, 90, 180, 270, 360, -90, np.int64(450))] == [
        0, 1, 2, 3, 0, 3, 1,
    ]
    for bad in (45, 1.5, "90", True):
        with pytest.raises(ValueError):
            quarter_turns(bad)


def _engine(probs, orient_min=pl.ORIENT_MIN):
    """An OrtOCR whose orient.onnx gives ``probs``; _read returns the turn it got."""
    eng = object.__new__(pl.OrtOCR)
    eng.orient_min = orient_min
    eng.asked = 0

    def orientation_probs(gray):
        eng.asked += 1
        return np.array(probs, np.float32)

    eng.orientation_probs = orientation_probs
    eng.read_turn = lambda gray, a, small, p: "read_turn"
    eng.read_half_turn = lambda gray, a, small: "read_half_turn"
    eng._read = lambda gray, k, a, small=None: Page([], {"orient_ms": 0.0}, note=k)
    return eng


PAGE = np.full((400, 300), 255, np.uint8)


def test_known_rotation_skips_the_orientation_model():
    eng = _engine([0.99, 0.0, 0.01, 0.0, 0.0])
    assert eng(PAGE, rotation=270).note == 3 and eng.asked == 0
    assert eng(PAGE, turn=False, rotation=90).note == 1 and eng.asked == 0
    assert eng(PAGE).note == 0 and eng.asked == 1


def test_orient_min_sets_when_a_turn_is_taken():
    probs = [0.01, 0.9, 0.0, 0.09, 0.0]
    # Below the default 0.95, a page seen as sideways is settled by reading lines.
    assert _engine(probs)(PAGE).note == "read_turn"
    assert _engine(probs).orientation(PAGE) == 0
    assert _engine(probs, orient_min=0.85)(PAGE).note == 1
    assert _engine(probs, orient_min=0.85).orientation(PAGE) == 1


def test_orient_min_out_of_range():
    with pytest.raises(ValueError):
        pl.OrtOCR("/nonexistent", orient_min=1.5)


def test_convert_passes_rotation_per_page(tmp_path):
    pdf = _pdf(tmp_path / "three.pdf", [(200, 300)] * 3)
    for rotation, want in (
        (None, [None] * 3),
        (180, [180] * 3),
        ({1: 90}, [None, 90, None]),
    ):
        ocr = StubOCR()
        list(Converter(ocr, mode="ocr").convert(pdf, rotation=rotation))
        assert ocr.rotations == want
    with pytest.raises(ValueError):
        next(Converter(StubOCR(), mode="ocr").convert(pdf, rotation={0: 45}))


def test_a_sure_half_turn_is_checked_by_reading_lines():
    assert _engine([0.01, 0.0, 0.98, 0.01, 0.0])(PAGE).note == "read_half_turn"
    assert _engine([0.01, 0.98, 0.0, 0.01, 0.0])(PAGE).note == 1
