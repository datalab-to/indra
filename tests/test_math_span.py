from types import SimpleNamespace

import numpy as np

from indra_ocr.document import _line_chars
from indra_ocr.math import has_structure
from indra_ocr.math_span import SpanTagger


class _Sess:
    def __init__(self, p):
        self.p = np.array([p], np.float32)

    def run(self, _, feeds):
        assert feeds["ids"].shape == feeds["geo"].shape[:2]
        B, T = feeds["ids"].shape
        return [np.repeat(self.p[:, :T], B, 0)]


def test_second_head_drops_runs_mathrec_would_reject():
    # head 1 marks two runs; head 2 keeps only the second
    p = [[0.1, 0.0], [0.9, 0.2], [0.8, 0.1], [0.2, 0.0], [0.7, 0.9]]
    assert _tagger(p).spans([_chars(5)]) == [[(4, 5)]]


def _tagger(p):
    t = SpanTagger.__new__(SpanTagger)
    t.sess = _Sess(p)
    t.read_threshold = 0.5
    t.pack_gap = 0
    t.stoi = {"<cat:?>": 0, "<cat:Ll>": 1, "x": 2}
    t.unknown = 0
    t.threshold = 0.5
    return t


def _chars(n):
    return [(chr(97 + i), 10.0 * i, 0.0, 10.0 * i + 8, 10.0) for i in range(n)]


def test_spans_are_runs_over_threshold():
    t = _tagger([0.1, 0.9, 0.8, 0.2, 0.7])
    assert t.spans([_chars(5)]) == [[(1, 3), (4, 5)]]
    # a shorter line in the batch: its padding is not read
    assert t.spans([_chars(3), [], _chars(5)]) == [[(1, 3)], [], [(1, 3), (4, 5)]]
    assert t.spans([]) == []


def test_encode_ids_and_geometry():
    t = _tagger([0.0, 0.0])
    ids, geo = t.encode([("x", 0, 0, 8, 10), ("q", 12, -5, 20, 5)])
    assert list(ids) == [2, 1]  # known char, then its unicode category
    assert geo.shape == (2, 4)
    assert geo[1, 3] == (12 - 8) / 10  # gap to the previous char / median height


def test_line_chars_nearest_box_and_missing():
    run = SimpleNamespace(chars=[("a", (0, 0, 4, 2)), ("a", (20, 2, 24, 0))])
    toks = [("a", 44.0, 0), (" ", 50.0, 0)]
    out = _line_chars(toks, {0: run}, 2.0)
    assert out == [("a", 40.0, 0.0, 48.0, 4.0)]  # nearest center, y sorted, scaled
    assert _line_chars([("b", 0.0, 0)], {0: run}, 1.0) is None
    assert _line_chars([("a", 0.0, 1)], {0: run}, 1.0) is None


def test_has_structure():
    assert has_structure("$x ^ { 2 }$") and has_structure("$a _ i$")
    assert has_structure("$\\frac { 1 } { 2 }$") and has_structure("$\\sqrt { x }$")
    # symbols a text layer has as characters, and styling, add nothing
    assert not has_structure("$\\triangle 1 , 4 4 0$")
    assert not has_structure("$\\alpha \\in \\mathbb { Z }$")
    assert not has_structure("$\\mathrm { T }$") and not has_structure("no math")


class _ByChar:
    """A model that calls the char with id 2 ("x") math, wherever it is."""

    def run(self, _, feeds):
        return [(feeds["ids"] == 2).astype(np.float32)]


def test_packed_lines_get_their_own_spans():
    t = _tagger([0.0])
    t.sess = _ByChar()
    lines = [
        [(c, 10.0 * i, 0.0, 10.0 * i + 8, 10.0) for i, c in enumerate(w)]
        for w in ("axxb", "xq", "", "qqqqqx")
    ]
    want = [[(1, 3)], [(0, 1)], [], [(5, 6)]]
    t.pack_gap = 0
    assert t.spans(lines) == want
    t.pack_gap = 16
    assert t.spans(lines) == want


def test_median_matches_numpy():
    from indra_ocr.math_span import _median

    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 4, 7, 10):
        v = rng.uniform(0, 50, n)
        assert _median(v.tolist(), np.float64) == float(np.median(v))
        h = v.astype(np.float32)
        assert _median(h.tolist(), np.float32) == float(np.median(h))
