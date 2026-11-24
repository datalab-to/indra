"""The CTC outputs from a recognizer's logits (ctc.py) are the older graphs' outputs."""

import numpy as np

from indra_ocr.ctc import lm_inputs, peak_prob
from indra_ocr.pipeline import OrtOCR
from indra_ocr.vocab import Vocab


def _reference(logits, k=8):
    """The older graph's tail: softmax peak, log-softmax, blank column, top-k
    (best first, equal values lower class first)."""
    m = logits.max(-1, keepdims=True)
    e = np.exp(logits - m)
    conf = 1.0 / e.sum(-1)
    lsm = (logits - m) - np.log(e.sum(-1))[:, None]
    topi = np.argsort(-lsm, axis=-1, kind="stable")[:, :k]
    return conf, lsm[:, 0], np.take_along_axis(lsm, topi, -1), topi


def test_lm_inputs_match_a_full_sort():
    rng = np.random.default_rng(0)
    for V in (5, 9, 300):
        L = rng.normal(0, 3, (40, V)).astype(np.float32)
        L[::3, 2] = L[::3, 1]  # equal values: the lower class comes first
        L[::5, : min(V, 12)] = 0.0  # more equal values than k
        conf, blank, topv, topi = _reference(L, min(8, V))
        b, v, i = lm_inputs(L)
        assert np.array_equal(i, topi)
        assert np.allclose(v, topv, atol=1e-6) and np.allclose(b, blank, atol=1e-6)
        assert np.allclose(peak_prob(L), conf, rtol=1e-6)
        assert peak_prob(L).dtype == np.float32


class _Out:
    def __init__(self, name):
        self.name = name


def _recognizer(logits, lengths, new):
    """A recognizer double: the older outputs (conf, blank, topv, topi) or the
    newer ones (logits), from the same logits."""

    class Rec:
        def get_outputs(self):
            names = (
                ["ids", "lengths", "logits"]
                if new
                else ["ids", "conf", "lengths", "blank", "topv", "topi"]
            )
            return [_Out(n) for n in names]

        def run(self, names, feed):
            B = len(feed["widths"])
            L = logits[:B]
            out = {"ids": L.argmax(-1), "lengths": lengths[:B], "logits": L}
            conf, blank, topv, topi = zip(*(_reference(x) for x in L))
            out.update(
                conf=np.stack(conf),
                blank=np.stack(blank),
                topv=np.stack(topv),
                topi=np.stack(topi),
            )
            return [out[n] for n in names]

    return Rec()


def test_a_logits_recognizer_reads_like_the_older_graph(tmp_path):
    from indra_ocr.lm import NgramLM, _pack

    rng = np.random.default_rng(1)
    V, T = 6, 30
    logits = rng.normal(0, 1, (3, T, V)).astype(np.float32)
    logits[:, ::2, 0] += 4  # blanks between tokens
    logits[:, 1::2, 1:] += (
        3 * np.eye(V - 1, dtype=np.float32)[rng.integers(0, V - 1, (3, T // 2))]
    )
    lengths = np.array([T, 20, 25])
    np.savez(
        tmp_path / "lm.npz",
        order=np.int64(1),
        vocab_size=np.int64(V),
        k0=np.array([_pack((t,)) for t in range(1, V)], np.uint64),
        v0=np.log(rng.dirichlet(np.ones(V - 1))).astype(np.float32),
        hk0=np.array([0], np.uint64),
        hb0=np.array([0.0], np.float32),
    )
    gray = np.full((100, 400), 255, np.uint8)
    gray[10:30, 20:300] = 0
    gray[40:60, 20:200] = 0
    gray[70:90, 20:250] = 0
    boxes = np.array(
        [[20, 10, 300, 30, 1], [20, 40, 200, 60, 1], [20, 70, 250, 90, 1]], np.float32
    )
    reads = []
    for new in (False, True):
        ocr = OrtOCR.__new__(OrtOCR)
        ocr.rec = _recognizer(logits, lengths, new)
        ocr.vocab = Vocab(list("abcde"))
        ocr.height, ocr.groups, ocr.script, ocr.mask = 32, {}, None, None
        # every line with a reading goes through the beam search (gate over 1)
        ocr.lms = {id(ocr.rec): (NgramLM(tmp_path / "lm.npz"), 0.5, 1.5, 1.1)}
        reads.append(ocr.recognize(gray, boxes))
    for (t0, c0, k0), (t1, c1, k1) in zip(*reads):
        assert c0 >= 0.5 and t0  # the line has a reading
        assert t0 == t1 and k0 == k1
        assert abs(c0 - c1) < 1e-6


def test_lm_inputs_over_a_floor_give_the_beam_the_same_candidates():
    rng = np.random.default_rng(2)
    L = rng.normal(0, 3, (60, 400)).astype(np.float32)
    L[::4, 5:20] = L[::4, 5:6]  # equal values
    full = lm_inputs(L)
    cut = lm_inputs(L, floor=3e-3)
    assert np.array_equal(full[0], cut[0])
    lf = np.log(3e-3)
    for t in range(len(L)):
        a = [(c, v) for c, v in zip(full[2][t], full[1][t]) if c != 0 and v >= lf]
        b = [(c, v) for c, v in zip(cut[2][t], cut[1][t]) if c != 0 and v >= lf]
        assert a == b
