"""Reliability of the native kernels: bad input raises (never reads out of
bounds), and when a kernel raises, each caller runs its Python code with the
same result and one warning.  Skipped when indra-ocr-native is not installed."""

import warnings

import numpy as np
import pytest

nat = pytest.importorskip("indra_ocr_native")

from indra_ocr import native  # noqa: E402

pytestmark = pytest.mark.skipif(
    native.lib is None, reason="native kernels off (INDRA_OCR_NATIVE=0)"
)

# ------------------------------------------------- bad input raises, no crash


def test_ctc_kernels_check_their_input():
    with pytest.raises(ValueError):
        nat.ctc_greedy(np.zeros((2, 0, 5), np.float32), np.array([5, 5]))  # no classes
    with pytest.raises(ValueError):
        nat.ctc_greedy(
            np.zeros((2, 3, 5), np.float32), np.array([5])
        )  # one length short
    with pytest.raises(ValueError):
        nat.ctc_topk(np.zeros((3, 5), np.float32), 5, 0)  # k = 0
    with pytest.raises(ValueError):
        nat.ctc_topk(np.zeros((0, 5), np.float32), 5, 2)
    with pytest.raises(TypeError):
        nat.ctc_greedy(np.zeros((2, 3), np.float32), np.array([1, 1]))  # not 3-D
    # lengths out of range are clamped, not followed
    ids, conf = nat.ctc_greedy(np.ones((2, 3, 4), np.float32), np.array([-7, 1 << 40]))
    assert ids.shape == (2, 4) and (conf[0] == 0).all() and (conf[1] > 0).all()


def _lm_tables():
    keys = np.array([1, 5, 9], np.uint64)
    return keys, np.zeros(3), keys.copy(), np.zeros(3)


def test_language_model_checks_its_tables():
    pk, pv, hk, hv = _lm_tables()
    with pytest.raises(ValueError):
        nat.NgramLM(0, 11, 10, pk, pv, hk, hv, 0.0)  # order 0
    with pytest.raises(ValueError):
        nat.NgramLM(7, 11, 10, pk, pv, hk, hv, 0.0)  # 77 bits
    with pytest.raises(ValueError):
        nat.NgramLM(2, 11, 10, pk[::-1].copy(), pv, hk, hv, 0.0)  # unsorted
    with pytest.raises(ValueError):
        nat.NgramLM(2, 11, 10, pk, pv[:2].copy(), hk, hv, 0.0)  # lengths differ
    with pytest.raises(ValueError):
        nat.NgramLM(2, 11, 0, pk, pv, hk, hv, 0.0)  # no vocabulary


def test_beam_search_checks_its_input():
    lm = nat.NgramLM(2, 11, 10, *_lm_tables(), 0.0)
    blank = np.full(3, -1e-5, np.float32)  # sure blanks: every step skips
    topv = np.full((3, 2), -1.0, np.float32)
    topi = np.array([[0, 1]] * 3, np.int64)
    assert nat.beam_search(blank, topv, topi, lm, 0.5, 1.5, 8, 0.999, 1e-3) == ([], [])
    with pytest.raises(ValueError):
        nat.beam_search(blank, topv, topi, lm, 0.5, 1.5, 0, 0.999, 1e-3)  # beam 0
    with pytest.raises(ValueError):
        nat.beam_search(blank[:2].copy(), topv, topi, lm, 0.5, 1.5, 8, 0.999, 1e-3)
    with pytest.raises(ValueError):
        bad = topi.copy()
        bad[1, 1] = 1 << 20  # past the LM's 11-bit tokens
        nat.beam_search(blank, topv, bad, lm, 0.5, 1.5, 8, 0.999, 1e-3)
    with pytest.raises(ValueError):
        nat.beam_search(blank, topv, -topi, lm, 0.5, 1.5, 8, 0.999, 1e-3)
    with pytest.raises(ValueError):
        nat.beam_search(blank, topv, topi, lm, 0.5, 1.5, 8, 0.0, 1e-3)  # log(0)


def test_text_layer_kernels_check_their_input():
    with pytest.raises(ValueError):
        nat.pdf_char_data(0, 3, 1, 1, 1)  # null text page
    with pytest.raises(ValueError):
        nat.pdf_char_data(1, -1, 1, 1, 1)
    codes = np.array([65, 66], np.uint32)
    ok = np.ones(2, np.uint8)
    with pytest.raises(ValueError):
        nat.pdf_char_loop(codes, np.array([0, 4], np.int8), ok, 0)  # no quarter turn 4
    with pytest.raises(ValueError):
        nat.pdf_char_loop(codes, np.array([0], np.int8), ok, 0)
    boxes = np.zeros((1, 4), np.float32)
    cb = np.zeros((2, 4))
    for rs in ([0, 3], [1, 2], [0, 2, 1, 2], []):
        with pytest.raises(ValueError):
            nat.assign_chars(codes, cb, np.array(rs, np.int64), boxes, 1.0)
    with pytest.raises(ValueError):
        nat.assign_chars(codes, cb[:1].copy(), np.array([0, 2], np.int64), boxes, 1.0)
    with pytest.raises(ValueError):
        nat.assign_chars(codes, cb, np.array([0, 2], np.int64), boxes, float("nan"))


def test_ordering_kernels_check_their_input():
    with pytest.raises(ValueError):
        nat.cluster(np.zeros((3, 4)))
    with pytest.raises(TypeError):
        nat.split_points([("a", 1.0), 5], 1.0, [], None)  # a token that is no tuple
    with pytest.raises(TypeError):
        nat.split_points([("a", "x")], 1.0, [], None)  # an x that is no number
    with pytest.raises(IndexError):
        nat.split_points([("a",)], 1.0, [], None)  # no x


# ------------------------------------------------- a failing kernel falls back


class _Broken:
    """The kernel module with some kernels raising."""

    def __init__(self, *names):
        self._names = set(names)

    def __getattr__(self, name):
        if name in self._names:

            def fail(*a, **kw):
                raise RuntimeError(f"{name} broke")

            return fail
        return getattr(nat, name)


@pytest.fixture
def broken(monkeypatch):
    """broken("k1", ...): those kernels raise; returns the warnings they give."""

    def make(*names):
        monkeypatch.setattr(native, "lib", _Broken(*names))
        monkeypatch.setattr(native, "_reported", set())

    return make


def _python(monkeypatch, f, *a):
    with monkeypatch.context() as m:
        m.setattr(native, "lib", None)
        return f(*a)


def test_a_failing_kernel_warns_once(broken):
    from indra_ocr.order_post import _cluster

    broken("cluster")
    a = np.random.default_rng(0).standard_normal((20, 20)).astype(np.float32)
    with pytest.warns(RuntimeWarning, match="cluster failed"):
        first = _cluster(a)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no second warning
        assert _cluster(a) == first


def test_failing_kernels_give_the_python_results(broken, monkeypatch):
    from indra_ocr.document import assign_runs
    from indra_ocr.lm import beam_search
    from indra_ocr.order_post import _cluster
    from indra_ocr.pipeline import OrtOCR
    from indra_ocr.vocab import Vocab
    from test_native import _ctc_steps, _random_lm, _runs

    rng = np.random.default_rng(1)
    a = rng.standard_normal((30, 30)).astype(np.float32)
    runs = _runs(rng, 12)
    boxes = np.stack([rng.uniform(0, 300, 20), rng.uniform(0, 600, 20)], 1)
    boxes = np.concatenate([boxes, boxes + rng.uniform(5, 200, (20, 2))], 1).astype(
        np.float32
    )
    ocr = OrtOCR.__new__(OrtOCR)
    ocr.vocab = Vocab.latin()
    toks = [
        (chr(97 + i % 26), float(x))
        for i, x in enumerate(np.cumsum(rng.exponential(9, 30)))
    ]
    want = {
        "cluster": _python(monkeypatch, _cluster, a),
        "assign": _python(monkeypatch, assign_runs, runs, boxes, 1.0),
        "split": _python(monkeypatch, ocr.split_line, boxes[0], toks, 3.0),
    }
    broken("cluster", "assign_chars", "split_points", "beam_search")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert _cluster(a) == want["cluster"]
        assert assign_runs(runs, boxes, 1.0) == want["assign"]
        assert ocr.split_line(boxes[0], toks, 3.0) == want["split"]
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        lm = _random_lm(Path(d), rng)
        steps = _ctc_steps(rng, 40, 12, 8)
        monkeypatch.setattr(native, "lib", None)
        expect = beam_search(*steps, lm)
        broken("beam_search")
        with pytest.warns(RuntimeWarning, match="beam_search failed"):
            assert beam_search(*steps, lm) == expect


def test_assign_runs_starts_over_when_the_kernel_fails_midway(broken, monkeypatch):
    """A kernel that fails after filling some lines leaves no tokens behind."""
    from indra_ocr import document
    from indra_ocr.document import assign_runs
    from test_native import _runs

    rng = np.random.default_rng(2)
    runs = _runs(rng, 10)
    boxes = np.array([[0, 0, 1000, 1000]], np.float32)
    want = _python(monkeypatch, assign_runs, runs, boxes, 1.0)
    real = document._assign_native

    def half(runs, boxes, k, parts):
        real(runs, boxes, k, parts)  # fills the lines, then fails
        raise RuntimeError("midway")

    monkeypatch.setattr(document, "_assign_native", half)
    with pytest.warns(RuntimeWarning, match="assign_chars failed"):
        assert assign_runs(runs, boxes, 1.0) == want


def test_extract_runs_falls_back_when_pdfium_calls_fail(tmp_path, broken, monkeypatch):
    import pypdfium2 as pdfium

    from indra_ocr.textlayer import extract_runs
    from test_native import _pdf

    _pdf(tmp_path / "p.pdf")
    page = pdfium.PdfDocument(str(tmp_path / "p.pdf"))[0]
    want = _python(monkeypatch, extract_runs, page)
    broken("pdf_char_data")
    with pytest.warns(RuntimeWarning, match="pdf_chars failed"):
        got = extract_runs(page)
    assert got[1] == want[1]
    assert [(r.text, r.chars) for r in got[0]] == [(r.text, r.chars) for r in want[0]]


class _Session:
    def __init__(self, outs):
        self.outs, self.calls = outs, 0

    def run(self, names, feed):
        self.calls += 1
        return self.outs if names is None else [self.outs[n] for n in names]


def test_a_cut_recognizer_reads_with_its_full_graph_when_the_head_fails(broken):
    from indra_ocr.pipeline import OUT_GREEDY, OUT_LM, CutGraph, OrtOCR

    rng = np.random.default_rng(3)
    B, V, T = 2, 7, 5
    logits = rng.standard_normal((B, V, T)).astype(np.float32)
    lengths = np.array([5, 3], np.int64)
    full = _Session(
        dict(
            zip(
                OUT_GREEDY + OUT_LM,
                [
                    logits.argmax(1),
                    rng.random((B, T)).astype(np.float32),
                    lengths,
                    rng.standard_normal((B, T)).astype(np.float32),
                    rng.standard_normal((B, T, 8)).astype(np.float32),
                    rng.integers(0, V, (B, T, 8)),
                ],
            )
        )
    )
    cut = _Session([logits, lengths])
    ocr = OrtOCR.__new__(OrtOCR)
    ocr._cut = {id(cut): CutGraph(set(OUT_GREEDY + OUT_LM), 8, lambda: full)}
    feed = (np.zeros((B, 1, 32, 20), np.float32), np.array([20, 12]))

    # kernels working: the logits go to ctc_greedy, the full graph is never made
    best, pmax, n, lm_steps = ocr._read_batch(cut, *feed, True)
    assert (best[0, :5] == logits[0].argmax(0)).all() and full.calls == 0
    assert lm_steps(1, 3)[2].shape == (3, 8) and full.calls == 0

    broken("ctc_topk")  # the LM outputs come from the full graph, once per batch
    best, pmax, n, lm_steps = ocr._read_batch(cut, *feed, True)
    with pytest.warns(RuntimeWarning, match="ctc_topk failed"):
        b, v, i = lm_steps(1, 3)
    assert (i == full.outs["topi"][1, :3]).all()
    lm_steps(0, 5)
    assert full.calls == 1

    broken("ctc_greedy")  # the batch reads with the full graph
    with pytest.warns(RuntimeWarning, match="ctc_greedy failed"):
        best, pmax, n, lm_steps = ocr._read_batch(cut, *feed, True)
    assert best is full.outs["ids"] and pmax is full.outs["conf"]
    assert (lm_steps(1, 3)[1] == full.outs["topv"][1, :3]).all()


def test_the_python_code_runs_when_the_kernels_are_off(monkeypatch):
    from indra_ocr.pipeline import OUT_GREEDY, OUT_LM, CutGraph, OrtOCR

    monkeypatch.setattr(native, "lib", None)
    outs = [np.zeros((1, 4), np.int64), np.ones((1, 4), np.float32), np.array([4])]
    full = _Session(dict(zip(OUT_GREEDY + OUT_LM, outs + [None] * 3)))
    cut = _Session(None)
    ocr = OrtOCR.__new__(OrtOCR)
    ocr._cut = {id(cut): CutGraph(set(OUT_GREEDY), 8, lambda: full)}
    best, *_ = ocr._read_batch(
        cut, np.zeros((1, 1, 32, 8), np.float32), np.array([8]), False
    )
    assert best is outs[0] and cut.calls == 0
    assert "off" in native.status()


def test_disabled_turns_the_kernels_off_and_back_on():
    assert native.lib is not None
    with native.disabled():
        assert native.lib is None
    assert native.lib is not None


def test_a_language_model_with_its_native_twin_pickles(tmp_path):
    import pickle

    from indra_ocr.lm import beam_search
    from test_native import _ctc_steps, _random_lm

    rng = np.random.default_rng(4)
    lm = _random_lm(tmp_path, rng)
    steps = _ctc_steps(rng, 30, 12, 8)
    first = beam_search(*steps, lm)  # makes the native twin
    assert lm._native is not None
    copy = pickle.loads(pickle.dumps(lm))
    assert copy._native is None and beam_search(*steps, copy) == first
