"""The read gate (math_gate.json): every flagged line is read without it; with it, a logistic gate on gate_features
skips lines whose reading would not be used."""
from indra_ocr import math as M


class _Reader:
    gate = M.MathReader.gate

    def __init__(self, head=None):
        self.gate_head = head


def test_without_gate_every_flagged_line_is_read():
    assert _Reader().gate([0.6, 0.9], ["x = 1", "Total $ 1,200"], 5, "ocr") == [True, True]


def test_gate_uses_the_probability():
    n = len(M.gate_features(0.5, "", 1, 0.5, 1, "ocr"))
    head = {"w": [10.0] + [0.0] * (n - 1), "b": -7.0, "z_min": 0.0}  # keep when prob >= 0.7
    assert _Reader(head).gate([0.6, 0.9], ["a", "b"], 2, "text") == [False, True]


def test_features_of_a_number_line():
    f = M.gate_features(0.8, "$ 1,200.00", 3, 0.9, 10, "ocr")
    assert f[0] == 0.8 and f[2] > 0.5  # digit share
    assert f[5] > 0  # currency share
    assert f[-2] == 1.0 and f[-1] == 0.0  # source: ocr


def test_gate_file_is_downloaded_with_math():
    from indra_ocr.models import MATH_FILES, download_patterns

    allow, ignore = download_patterns(math=True)
    assert "math_gate.json" in allow and "math_gate.json" not in MATH_FILES
    assert "math_gate.json" in download_patterns(math=False)[1]
