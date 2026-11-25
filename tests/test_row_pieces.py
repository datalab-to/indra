"""Line models in pieces (ort_session.RowPieces) and the layer check's one read per crop."""

import numpy as np

from indra_ocr.ort_session import RowPieces, is_line_model, run_pieces


class _Arg:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class FakeLineModel:
    """Per row: its pixel sum and the padded width, so a row's outputs depend on
    the row and the batch's width only; records the batch sizes it ran."""

    def __init__(self):
        self.calls = []

    def get_inputs(self):
        return [_Arg("x", ["B", 1, 32, "W"]), _Arg("widths", ["B"])]

    def run(self, names, feed, run_options=None):
        x, w = feed["x"], feed["widths"]
        self.calls.append(len(x))
        s = x.reshape(len(x), -1).sum(1)
        return [s * x.shape[3], w * 2]


def _batch(B, W=100):
    rng = np.random.default_rng(0)
    return {"x": rng.random((B, 1, 32, W), dtype=np.float32), "widths": np.arange(B, dtype=np.int64)}


def test_pieces_give_the_whole_batch():
    feed = _batch(10)
    whole = FakeLineModel().run(None, feed)
    m = FakeLineModel()
    out = RowPieces(m, pixels=3 * 32 * 100).run(None, feed)
    assert m.calls == [3, 3, 3, 1]
    for a, b in zip(whole, out):
        assert np.array_equal(a, b)


def test_pieces_yield_row_ranges():
    m = FakeLineModel()
    got = [(lo, hi, len(o[0])) for lo, hi, o in RowPieces(m, pixels=4 * 32 * 100).pieces(None, _batch(9))]
    assert got == [(0, 4, 4), (4, 8, 4), (8, 9, 1)]


def test_a_wide_batch_runs_a_row_at_a_time():
    m = FakeLineModel()
    RowPieces(m, pixels=1000).run(None, _batch(3, W=500))
    assert m.calls == [1, 1, 1]


def test_run_pieces_on_a_plain_session_is_one_run():
    m = FakeLineModel()
    assert [(lo, hi) for lo, hi, _ in run_pieces(m, None, _batch(5))] == [(0, 5)]
    assert m.calls == [5]


class FakeRecognizer:
    """A recognizer ending at its logits: frame t of a row is class (t + row sum) % V,
    so a row's outputs depend on its crop only; records the rows it read."""

    V = 12  # at least the top-k's 8

    def __init__(self):
        self.rows = []

    def get_outputs(self):
        return [_Arg(n, None) for n in ("ids", "lengths", "logits")]

    def run(self, names, feed, run_options=None):
        x = feed["x"]
        B, T = len(x), x.shape[3] // 4
        self.rows.append(B)
        cls = (np.arange(T)[None] + np.round(x.reshape(B, -1).sum(1))[:, None].astype(int)) % self.V
        logits = np.full((B, T, self.V), -4.0, np.float32)
        np.put_along_axis(logits, cls[..., None], 4.0, -1)
        return [cls.astype(np.int64), np.full(B, T, np.int64), logits]


def _ocr(rec):
    from indra_ocr.pipeline import OrtOCR
    from indra_ocr.vocab import Vocab

    ocr = OrtOCR.__new__(OrtOCR)
    ocr.rec, ocr.vocab = rec, Vocab(list("abcde"))
    ocr.groups, ocr.script, ocr.mask, ocr.window, ocr.height = {}, None, None, 1, 32
    return ocr


def test_layer_gaps_reads_a_box_once():
    gray = np.full((60, 400), 255, np.uint8)
    gray[10:30, 20:120] = 0
    gray[35:55, 150:330] = 80
    a, b = [10, 10, 130, 30], [140, 35, 340, 55]
    toks = [[(c, 0.0, 0) for c in t] for t in ("abc", "abd", "cab", "abe", "dd")]
    rec1, rec2 = FakeRecognizer(), FakeRecognizer()
    once = _ocr(rec1).layer_gaps(gray, np.array([a, a, b, a, b], np.float32), toks)
    assert rec1.rows == [2]  # two boxes, one row each
    # the same gaps as reading every copy: each box on its own
    each = [
        _ocr(rec2).layer_gaps(gray, np.array([box, b if box == a else a], np.float32), [t, t])[0]
        for box, t in zip([a, a, b, a, b], toks)
    ]
    assert once == each


def test_is_line_model():
    assert is_line_model(FakeLineModel())

    class Det:
        def get_inputs(self):
            return [_Arg("x", [1, 1, "H", "W"])]

    assert not is_line_model(Det())
