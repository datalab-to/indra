"""The native kernels (indra-ocr-native) give the Python code's results.

Each test runs a function with the kernels and with indra_ocr.native.lib set to
None (the Python code), on random and edge-case inputs.  Skipped when the
package is not installed."""

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

nat = pytest.importorskip("indra_ocr_native")

from indra_ocr import native  # noqa: E402

pytestmark = pytest.mark.skipif(
    native.lib is None, reason="native kernels off (INDRA_OCR_NATIVE=0)"
)


@pytest.fixture
def python(monkeypatch):
    """Call f(*a, **kw) with the kernels off."""

    def call(f, *a, **kw):
        with monkeypatch.context() as m:
            m.setattr(native, "lib", None)
            return f(*a, **kw)

    return call


def test_kernels_are_on():
    assert native.lib is not None and native.lib.API == native.API


def test_is_space_is_str_isspace():
    assert [c for c in range(0x110000) if nat.is_space(c) != chr(c).isspace()] == []


# ------------------------------------------------------------------ CTC head


def _greedy_ref(logits, lengths):
    B, V, T = logits.shape
    ids = np.zeros((B, T), np.int64)
    conf = np.zeros((B, T), np.float32)
    for b in range(B):
        n = int(lengths[b])
        x = logits[b, :, :n].astype(np.float64)
        ids[b, :n] = x.argmax(0)  # first maximum
        conf[b, :n] = 1.0 / np.exp(x - x.max(0)).sum(0)
    return ids, conf


def test_ctc_greedy_matches_softmax_argmax():
    rng = np.random.default_rng(0)
    for V, T in [(5, 3), (97, 41), (1698, 70)]:
        logits = (rng.standard_normal((4, V, T)) * 6).astype(np.float32)
        logits[1] = np.round(logits[1])  # ties: the first class wins
        lengths = np.array([T, T, T // 2, 0], np.int64)
        ids, conf = nat.ctc_greedy(logits, lengths)
        rids, rconf = _greedy_ref(logits, lengths)
        assert (ids == rids).all()
        # float32 sums over V classes, as the graph's ReduceSum
        assert np.allclose(conf, rconf, rtol=2e-5, atol=1e-7)


def test_ctc_topk_matches_log_softmax_topk():
    rng = np.random.default_rng(1)
    V, T, n, K = 300, 50, 37, 8
    logits = np.round(rng.standard_normal((V, T)) * 4).astype(np.float32)  # many ties
    blank, topv, topi = nat.ctc_topk(logits, n, K)
    x = logits[:, :n].astype(np.float64)
    lp = x - x.max(0) - np.log(np.exp(x - x.max(0)).sum(0))
    order = np.argsort(-lp, axis=0, kind="stable")[
        :K
    ].T  # equal values: lower class first
    assert (topi == order).all()
    assert np.allclose(topv, np.take_along_axis(lp.T, order, 1), atol=2e-5)
    assert np.allclose(blank, lp[0], atol=2e-5)


# ------------------------------------------------------------ language model


def _random_lm(tmp_path, rng, order=3, vocab=12, bits=11):
    from indra_ocr.lm import BOS, NgramLM, _pack

    arrays = {"order": np.int64(order), "vocab_size": np.int64(vocab)}
    for n in range(order):
        hists = {tuple(rng.integers(0, vocab, n)) for _ in range(40)}
        if n:
            hists |= {(BOS,) + tuple(rng.integers(0, vocab, n - 1)) for _ in range(5)}
        ents = {h + (int(rng.integers(0, vocab)),) for h in hists for _ in range(3)}
        k = sorted(_pack(e, bits) for e in ents)
        hk = sorted({_pack(h, bits) for h in hists} | ({0} if n == 0 else set()))
        arrays[f"k{n}"] = np.array(k, np.uint64)
        arrays[f"v{n}"] = rng.uniform(-6, -0.1, len(k)).astype(np.float32)
        arrays[f"hk{n}"] = np.array(hk, np.uint64)
        arrays[f"hb{n}"] = rng.uniform(-2, 0.5, len(hk)).astype(np.float32)
    path = tmp_path / f"lm{rng.integers(1 << 30)}.npz"
    np.savez(path, **arrays)
    return NgramLM(path)


def test_lm_lookups_match(tmp_path):
    rng = np.random.default_rng(2)
    lm = _random_lm(tmp_path, rng)
    keys = [int(k) for k in rng.integers(0, 1 << 33, 3000)] + [
        int(k) for k in lm.p_keys[:500] & ((1 << 33) - 1)
    ]
    assert lm.native_lm().logp_keys(keys) == lm.logp_keys(keys)


def _ctc_steps(rng, T, V, K):
    logits = rng.standard_normal((T, V)) * 3
    logits[:, 0] += rng.uniform(-2, 4, T)  # blanks between tokens
    lp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    topi = np.argsort(-lp, 1, kind="stable")[:, :K]
    return (
        lp[:, 0].astype(np.float32),
        np.take_along_axis(lp, topi, 1).astype(np.float32),
        topi.astype(np.int64),
    )


def test_beam_search_matches(tmp_path, python):
    from indra_ocr.lm import beam_search

    rng = np.random.default_rng(3)
    for trial in range(60):
        lm = _random_lm(tmp_path, rng, order=int(rng.integers(1, 5)))
        steps = _ctc_steps(rng, int(rng.integers(0, 60)), 12, 8)
        kw = dict(
            alpha=float(rng.uniform(0, 1)),
            beta=float(rng.uniform(0, 2)),
            beam=int(rng.integers(1, 10)),
            cand_floor=1e-3,
        )
        assert beam_search(*steps, lm, **kw) == python(beam_search, *steps, lm, **kw), (
            trial
        )


# ----------------------------------------------------------- ordering, lines


def test_cluster_matches(python):
    from indra_ocr.order_post import _cluster

    rng = np.random.default_rng(4)
    for n in [0, 1, 2, 5, 30, 120]:
        for kind in range(3):
            a = rng.standard_normal((n, n)) * 3
            if kind == 1:
                a = np.round(a)  # ties
            a = (a + a.T) / 2 if kind != 2 else a
            a = a.astype(np.float32)
            assert _cluster(a) == python(_cluster, a)


def test_split_line_matches(python):
    from indra_ocr.pipeline import OrtOCR
    from indra_ocr.vocab import Vocab

    ocr = OrtOCR.__new__(OrtOCR)
    ocr.vocab = Vocab.latin()
    space = ocr.vocab.index[" "]
    rng = np.random.default_rng(5)
    for trial in range(300):
        n = int(rng.integers(0, 40))
        xs = np.cumsum(
            rng.exponential(4, n) * (rng.random(n) < 0.9) + (rng.random(n) < 0.1) * 40
        )
        if trial % 2:
            toks = [
                (space if rng.random() < 0.2 else int(rng.integers(1, 90)), float(x))
                for x in xs
            ]
        else:
            toks = [
                (
                    " " if rng.random() < 0.2 else chr(97 + int(rng.integers(0, 26))),
                    np.float64(x),
                    3,
                )
                for x in xs
            ]
        box = np.array([0, 10, 400, 20], np.float32)
        at = list(rng.uniform(0, xs[-1] if n else 1, int(rng.integers(0, 3))))
        assert ocr.split_line(box, toks, 3.0, at) == python(
            ocr.split_line, box, toks, 3.0, at
        )


# ---------------------------------------------------------------- text layer


def _runs(rng, n_runs, page=600.0):
    from indra_ocr.textlayer import TextLine

    runs = []
    for s in range(n_runs):
        n = int(rng.integers(1, 30))
        x = np.cumsum(rng.uniform(2, 7, n)) + rng.uniform(0, page * 0.6)
        y = rng.uniform(0, page)
        b = np.stack([x, np.full(n, y), x + 4, np.full(n, y + 9)], 1)
        if s % 3 == 0:  # centres exactly on box sides
            b = np.round(b)
        chars = [
            " " if rng.random() < 0.15 else chr(0x41 + int(rng.integers(0, 50)))
            for _ in range(n)
        ]
        seg = [(c, tuple(bb)) for c, bb in zip(chars, b)]
        runs.append(TextLine((0, 0, 1, 1), "".join(chars), seg, 0, s, b))
    return runs


def test_assign_runs_matches(python):
    from indra_ocr.document import assign_runs

    rng = np.random.default_rng(6)
    for trial in range(80):
        runs = _runs(rng, int(rng.integers(0, 25)))
        k = float(rng.choice([1.0, 200 / 72]))
        m = int(rng.integers(0, 40))
        y0 = rng.uniform(0, 600 * k, m)
        x0 = rng.uniform(0, 400 * k, m)
        boxes = np.stack(
            [x0, y0, x0 + rng.uniform(5, 300 * k, m), y0 + rng.uniform(4, 40, m)], 1
        )
        if trial % 2:
            boxes = np.round(boxes / 4) * 4  # equal areas and shared sides
        boxes = np.concatenate([boxes, np.ones((m, 1))], 1).astype(np.float32)
        got = assign_runs(runs, boxes[:, :4], k)
        want = python(assign_runs, runs, boxes[:, :4], k)
        assert got == want
        assert [type(t[1]) for tk in got[0] for t in tk] == [
            type(t[1]) for tk in want[0] for t in tk
        ]


def _pdf(path):
    import pypdfium2 as pdfium
    import pypdfium2.raw as R

    doc = pdfium.PdfDocument.new()
    page = doc.new_page(612, 792)
    texts = [
        ("Upright body text, two  spaces", (1, 0, 0, 1, 72, 700)),
        ("Second line \U0001d400\U0001d401 math letters", (1, 0, 0, 1, 72, 680)),
        ("Axis label", (0, 1, -1, 0, 40, 300)),
        ("Upside down", (-1, 0, 0, -1, 400, 200)),
        ("Tilted forty-five", (0.7071, 0.7071, -0.7071, 0.7071, 300, 400)),
        (" nbsp　ideographic em", (1, 0, 0, 1, 72, 600)),
    ]
    for t, mtx in texts:
        obj = R.FPDFPageObj_NewTextObj(doc.raw, b"Helvetica", ctypes.c_float(12.0))
        units = list(t.encode("utf-16-le"))
        u16 = [units[i] | units[i + 1] << 8 for i in range(0, len(units), 2)]
        R.FPDFText_SetText(obj, (ctypes.c_ushort * (len(u16) + 1))(*u16, 0))
        R.FPDFPageObj_Transform(obj, *mtx)
        R.FPDFPage_InsertObject(page.raw, obj)
    R.FPDFPage_GenerateContent(page.raw)
    doc.save(str(path))


def test_extract_runs_matches(tmp_path, python):
    import pypdfium2 as pdfium

    from indra_ocr.textlayer import extract_runs

    _pdf(tmp_path / "p.pdf")
    page = pdfium.PdfDocument(str(tmp_path / "p.pdf"))[0]
    for turn, frame in [(0, None), (1, None), (2, None), (3, None), (0, 3)]:
        got, gst = extract_runs(page, turn, frame)
        want, wst = python(extract_runs, page, turn, frame)
        assert gst == wst
        assert [(r.bbox, r.text, r.chars, r.quarter, r.seq) for r in got] == [
            (r.bbox, r.text, r.chars, r.quarter, r.seq) for r in want
        ]
        assert all((a.boxes == b.boxes).all() for a, b in zip(got, want))


# ------------------------------------------------------------ the graph cut


def _models():
    from indra_ocr.models import HF_REPO

    p = os.environ.get("INDRA_OCR_MODEL_DIR")
    if p and (Path(p) / "rec.onnx").exists():
        return Path(p)
    try:
        from huggingface_hub import snapshot_download

        from indra_ocr.models import HF_REVISION

        return Path(
            snapshot_download(
                HF_REPO,
                revision=HF_REVISION,
                local_files_only=True,
                allow_patterns=["*.onnx", "meta.json"],
            )
        )
    except Exception:  # noqa: BLE001  (no cached models: skip)
        return None


@pytest.mark.skipif(
    _models() is None, reason="needs the models (INDRA_OCR_MODEL_DIR or the cache)"
)
def test_cut_recognizer_reads_as_the_full_graph():
    import onnxruntime as ort

    from indra_ocr.onnx_cut import cut_ctc_head

    rng = np.random.default_rng(7)
    for name in ["rec.onnx", "rec_cjk.onnx", "rec_world.onnx", "rec_rtl.onnx"]:
        path = _models() / name
        if not path.exists():
            continue
        cut = cut_ctc_head(path.read_bytes())
        assert cut is not None and cut.topk == 8
        full = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        part = ort.InferenceSession(cut.model, providers=["CPUExecutionProvider"])
        h = full.get_inputs()[0].shape[2]
        x = (rng.random((3, 1, h, 160)) < 0.15).astype(np.float32)
        feed = {"x": x, "widths": np.array([160, 120, 64], np.int64)}
        ids, conf, lengths, blank, topv, topi = full.run(None, feed)
        logits, lengths2 = part.run(None, feed)
        assert (lengths == lengths2).all()
        gids, gconf = nat.ctc_greedy(logits, lengths2)
        for k, n in enumerate(lengths.tolist()):
            assert (gids[k, :n] == ids[k, :n]).all()
            assert np.allclose(gconf[k, :n], conf[k, :n], atol=1e-4)
            b, v, i = nat.ctc_topk(logits[k], n, 8)
            assert (i == topi[k, :n]).all()
            assert np.allclose(v, topv[k, :n], atol=1e-4) and np.allclose(
                b, blank[k, :n], atol=1e-4
            )
    assert cut_ctc_head(b"not a model") is None


def test_a_graph_without_a_ctc_head_is_not_cut():
    from indra_ocr.onnx_cut import cut_ctc_head

    models = _models()
    if models is None or not (models / "det.onnx").exists():
        pytest.skip("needs the models")
    assert cut_ctc_head((models / "det.onnx").read_bytes()) is None
