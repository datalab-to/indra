"""The image math span tagger (math_span_img.py): column runs, page positions, the
splice of a formula into a line's tokens, and the formula kept whole through the
right-to-left flip.  Without the model files nothing loads."""
from pathlib import Path

import numpy as np

from indra_ocr.math_span_img import Formula, SpanImgTagger, page_x, splice, unspliced
from indra_ocr.pipeline import OrtOCR
from indra_ocr.rec_preproc import PAD_X


def _tagger(threshold=0.5, min_cols=2, merge_cols=1):
    t = SpanImgTagger.__new__(SpanImgTagger)
    t.threshold, t.min_cols, t.merge_cols, t.stride = threshold, min_cols, merge_cols, 4
    return t


def test_no_model_files_no_tagger(tmp_path: Path):
    assert SpanImgTagger.load(tmp_path, None, None) is None


def test_runs_join_short_gaps_and_drop_short_runs():
    p = np.array([0, 0.9, 0.9, 0.2, 0.8, 0.9, 0, 0, 0, 0.7, 0, 0, 0.6, 0.6, 0.6])
    # [1, 3) + [4, 6) join over a one-column gap; [9, 10) is too short; [12, 15) stays.
    assert _tagger().runs(p) == [(1, 6), (12, 15)]


def test_page_x_inverts_the_crop_padding_and_scale():
    box = [100.0, 50.0, 300.0, 70.0]  # 20 px high
    h = 20.0
    ix0 = np.floor(100 - PAD_X * h)
    cw = np.ceil(300 + PAD_X * h) - ix0
    crop_w = int(cw * 32 / (h * 1.0))  # any width: page_x only uses the ratio
    x0, x1 = page_x(box, crop_w, (0, crop_w))
    assert abs(x0 - ix0) < 1e-6 and abs(x1 - (ix0 + cw)) < 1e-6


def test_splice_replaces_the_characters_inside_the_span():
    toks = [(c, i * 10.0) for i, c in enumerate("ab cd ef")]
    out = splice(toks, 25, 45, "$x^2$")
    assert "".join(t[0] for t in out) == "ab $x^2$ ef"
    assert [t[1] for t in out if t[0] == ""] == [30.0, 40.0]  # c and d keep their x (gaps unchanged)
    assert sum(isinstance(t[0], Formula) for t in out) == 1


def test_splice_adds_spaces_when_the_neighbours_have_none():
    toks = [(c, i * 10.0) for i, c in enumerate("例如在中")]
    out = splice(toks, 15, 25, "$a+b$")
    assert "".join(t[0] for t in out) == "例如 $a+b$ 中"


class _OCR:
    _decode = OrtOCR._decode
    _page_rtl = True


def test_formula_stays_whole_through_the_rtl_flip():
    # Visual order (left to right) of a right-to-left line: "في $\frac{a}{b}$ المعادلة".
    vis = [(c, float(i)) for i, c in enumerate("ةلداعملا ")]
    vis += [(Formula("$\\frac { a } { b } = 2$"), 20.0), (" ", 30.0)]
    vis += [(c, 40.0 + i) for i, c in enumerate("يف")]
    text = _OCR()._decode(vis)
    assert "$\\frac { a } { b } = 2$" in text
    assert text.startswith("في") and text.endswith("المعادلة")


def test_plain_tokens_decode_as_before():
    vis = [(c, float(i)) for i, c in enumerate("$x$ ab")]
    o = _OCR()
    o._page_rtl = False
    assert o._decode(vis) == "$x$ ab"


def test_splice_keeps_the_gaps_that_cut_cells():
    toks = [(c, x) for c, x in zip("abcdef", [0.0, 10.0, 20.0, 30.0, 200.0, 210.0])]
    out = splice(toks, 15, 35, "$x$")
    xs = sorted(t[1] for t in out if t[0] != " ")
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    assert max(gaps) == 170.0  # the cell gap (30 -> 200) is still the only wide one


def test_unspliced_gives_the_recognizer_tokens_back():
    toks = [(c, i * 10.0) for i, c in enumerate("ab cd ef")]
    assert unspliced(splice(toks, 25, 45, "$x^2$")) == toks
    assert unspliced(toks) is toks


def test_span_files_are_downloaded_with_math_only():
    from indra_ocr.models import download_patterns

    allow, _ = download_patterns(math=True)
    assert "math_span_img.onnx" in allow and "math_span_img.json" in allow
    _, ignore = download_patterns(math=False)
    assert "math_span_img.onnx" in ignore and "math_span_img.json" in ignore
