import numpy as np

from indra_ocr.textlayer import bad_char_count


def test_bad_chars_count_replacement_and_private_use():
    assert bad_char_count("ok") == 0
    assert bad_char_count("a�") == 2


def _chars(text, x0=0.0, w=5.0, gaps=None):
    out, x = [], x0
    for i, ch in enumerate(text):
        x += (gaps or {}).get(i, 0.0)
        out.append({"char": ch, "bbox": [x, 0.0, x + w, 10.0]})
        x += w
    return out


def test_cli_source_flags():
    import argparse

    from indra_ocr.cli import add_source_args, source_options

    ap = argparse.ArgumentParser()
    add_source_args(ap)
    assert source_options(ap.parse_args([])) == {
        "mode": "auto",
        "image_ocr": True,
    }
    assert source_options(ap.parse_args(["--mode", "ocr"]))["mode"] == "ocr"
    o = source_options(ap.parse_args(["--no-image-ocr"]))
    assert o["image_ocr"] is False


def test_runs_go_to_the_smallest_containing_line_in_x_order():
    from indra_ocr.document import assign_runs
    from indra_ocr.textlayer import TextLine

    def run(text, x0, y0, y1, w=4.0):
        chars = [
            (c, (x0 + i * w, y0, x0 + (i + 1) * w, y1)) for i, c in enumerate(text)
        ]
        return TextLine((x0, y0, x0 + w * len(text), y1), text, chars)

    boxes = np.array([[0, 0, 100, 20], [70, 2, 95, 18], [0, 30, 100, 50]], np.float32)
    runs = [
        run("ab c", 10, 5, 15),
        run("\u03b1", 2, 3, 13),
        run("xy", 75, 5, 15),
        run("far", 500, 5, 15),
    ]
    toks, lost = assign_runs(runs, boxes, 1.0)
    assert "".join(t[0] for t in toks[0]) == "\u03b1 ab c"  # runs by x, own order kept
    assert [t[2] for t in toks[0]][0] == 1  # content index of the run
    assert "".join(t[0] for t in toks[1]) == "xy"  # inside both boxes: the smaller wins
    assert toks[2] == [] and lost == 3


def test_a_run_across_two_cells_splits_between_them():
    from indra_ocr.document import assign_runs
    from indra_ocr.textlayer import TextLine

    text = "12 34"
    chars = [(c, (i * 10.0, 0.0, i * 10.0 + 8, 10.0)) for i, c in enumerate(text)]
    run = TextLine((0, 0, 48, 10), text, chars)
    boxes = np.array([[0, 0, 19, 10], [29, 0, 50, 10]], np.float32)  # two cells
    toks, lost = assign_runs([run], boxes, 1.0)
    assert ["".join(t[0] for t in tk).strip() for tk in toks] == [
        "12",
        "34",
    ] and lost == 0


def test_runs_of_a_right_to_left_line_go_right_to_left():
    from indra_ocr.document import assign_runs
    from indra_ocr.textlayer import TextLine

    def run(text, x0):  # logical order, drawn right to left from x0
        chars = [
            (c, (x0 - (i + 1) * 4.0, 5, x0 - i * 4.0, 15)) for i, c in enumerate(text)
        ]
        return TextLine((x0 - 4.0 * len(text), 5, x0, 15), text, chars)

    boxes = np.array([[0, 0, 100, 20]], np.float32)
    # two table cells on one row: "שלום" on the right, "עולם" on the left
    toks, _ = assign_runs([run("עולם", 30), run("שלום", 90)], boxes, 1.0)
    assert "".join(t[0] for t in toks[0]) == "שלום עולם"


def test_ocr_only_reads_scripts_in_its_vocab():
    from types import SimpleNamespace

    from indra_ocr.document import readable

    ocr = SimpleNamespace(
        vocab=SimpleNamespace(index={c: 1 for c in "abcdefghijklmnopqrstuvwxyz"}),
        groups={},
    )
    line = lambda t: SimpleNamespace(text=t)  # noqa: E731
    assert readable(ocr, [line("hello world 123")])
    assert not readable(ocr, [line("مرحبا بالعالم"), line("ok")])
    assert readable(ocr, [line("1234 ...")])  # no letters: nothing to rule out
    assert not readable(ocr, [line("中文文本")])
    ocr.groups = {"cjk": (None, SimpleNamespace(index={c: 1 for c in "中文文本"}))}
    assert readable(ocr, [line("中文文本")])


def test_lines_with_hanzi_go_to_the_cjk_recognizer():
    from indra_ocr.pipeline import OrtOCR
    from indra_ocr.vocab import Vocab

    class Rec:  # emits one fixed token id per line, at every other timestep
        def __init__(self, tok):
            self.tok = tok

        def run(self, _, feed, run_options=None):
            b, w = feed["x"].shape[0], feed["x"].shape[-1] // 4
            best = np.zeros((b, w), np.int64)
            best[:, ::2] = self.tok
            return best, np.ones((b, w), np.float32), np.full(b, w)

    class Script:  # lines wider than 300 px "hold hanzi"
        def run(self, _, feed, run_options=None):
            return [(feed["widths"] > 300).astype(np.float32)]

    ocr = OrtOCR.__new__(OrtOCR)
    ocr.vocab, ocr.groups = Vocab(list("ab")), {"cjk": (Rec(1), Vocab(list("中b")))}
    ocr.rec, ocr.script, ocr.mask, ocr.height = Rec(1), Script(), None, 32
    gray = np.full((200, 800), 255, np.uint8)
    boxes = np.array([[10, 10, 100, 42, 1], [10, 100, 700, 132, 1]], np.float32)
    (t0, _, k0), (t1, _, k1) = ocr.recognize(gray, boxes)
    assert set(t0) == {"a"} and isinstance(k0[0][0], int)  # main recognizer: ids
    assert set(t1) == {"中"} and k1[0][0] == "中"  # CJK recognizer: characters
    assert ocr._decode(k1) == t1


def test_a_turned_page_is_turned_upright_before_detection():
    from indra_ocr.pipeline import OrtOCR

    class Orient:  # says: content turned one quarter turn counter-clockwise
        def run(self, _, feed, run_options=None):
            assert feed["x"].shape == (1, 1, 768, 768)
            return [np.array([[0.01, 0.97, 0.01, 0.01]], np.float32)]

    seen = {}
    ocr = OrtOCR.__new__(OrtOCR)
    ocr.orient = Orient()
    ocr.detect = lambda g: (
        seen.setdefault("shape", g.shape) and np.zeros((0, 5), np.float32)
    )
    ocr.recognize = lambda g, b, **kw: []
    ocr.assemble = lambda wh, b, r, t, **kw: __import__(
        "indra_ocr.pipeline"
    ).pipeline.Page([], t)
    page = ocr(np.full((300, 500), 255, np.uint8))  # wide: a turned portrait page
    assert seen["shape"] == (500, 300) and page.rotation == 90
    ocr.orient = None
    assert ocr.orientation(np.full((300, 500), 255, np.uint8)) == 0


def test_fast_average_linkage_matches_the_brute_force():
    from indra_ocr.order_post import _cluster

    def brute(logits):
        n = logits.shape[0]
        S = logits.astype(np.float64).copy()
        np.fill_diagonal(S, 0.0)
        size, alive, members = np.ones(n), np.ones(n, bool), [[i] for i in range(n)]
        for _ in range(n - 1):
            mean = S / np.outer(size, size)
            mean[~alive] = -np.inf
            mean[:, ~alive] = -np.inf
            np.fill_diagonal(mean, -np.inf)
            a, b = np.unravel_index(int(np.argmax(mean)), mean.shape)
            if mean[a, b] <= 0:
                break
            S[a] += S[b]
            S[:, a] += S[:, b]
            S[a, a] = 0.0
            size[a] += size[b]
            alive[b] = False
            members[a] += members[b]
        return sorted(tuple(sorted(members[i])) for i in range(n) if alive[i])

    rng = np.random.default_rng(1)
    for _ in range(100):
        n = int(rng.integers(1, 40))
        g = rng.integers(0, max(1, n // 4), n)
        L = np.where(g[:, None] == g[None], 2.0, -2.0) + rng.normal(0, 1.5, (n, n))
        L = (L + L.T) / 2
        assert sorted(tuple(c) for c in _cluster(L)) == brute(L)


def test_skew_estimate_recovers_a_tilt():
    from indra_ocr.pipeline import estimate_skew, rotate_page

    rng = np.random.default_rng(0)
    page = np.full((2200, 1700), 255, np.uint8)
    for y in range(150, 2050, 45):  # text-like rows of ink blobs
        x = 150
        while x < 1550:
            w = int(rng.integers(20, 90))
            page[y : y + 18, x : x + w] = 30
            x += w + int(rng.integers(10, 25))
    assert abs(estimate_skew(page)) < 0.15
    for tilt in (2.0, -1.3, 0.6):
        est = estimate_skew(rotate_page(page, tilt))
        assert abs(est + tilt) <= 0.15, (tilt, est)  # the estimate undoes the tilt
    assert estimate_skew(np.full((800, 600), 255, np.uint8)) == 0.0


def test_cjk_lines_join_without_a_space_and_bare_bullets_start_items():
    from indra_ocr.layout_post import _ITEM, join_lines

    assert (
        join_lines(["或其中的", "任何部分，在 2020 年"])
        == "或其中的任何部分，在 2020 年"
    )
    assert join_lines(["説明します。", "次に"]) == "説明します。次に"
    assert (
        join_lines(["한국어 문장", "다음 줄"]) == "한국어 문장 다음 줄"
    )  # Korean keeps the space
    assert join_lines(["tumori-", "genesis is"]) == "tumorigenesis is"
    assert join_lines(["first line", "second"]) == "first line second"
    assert _ITEM.match("•") and _ITEM.match("• text") and _ITEM.match("12. text")
    assert not _ITEM.match("•text") and not _ITEM.match("12.5 kg")


def test_unrotate_box_maps_a_deskewed_box_back():
    from indra_ocr.pipeline import rotate_page, unrotate_box

    page = np.full((400, 600), 255, np.uint8)
    page[180:220, 100:500] = 0  # a long dark bar
    tilted = rotate_page(page, -3.0)
    ys, xs = np.nonzero(tilted < 128)
    got = unrotate_box([100, 180, 500, 220], 3.0, (600, 400))
    # the mapped box covers the tilted bar in the input page
    assert got[0] <= xs.min() + 2 and got[2] >= xs.max() - 2
    assert got[1] <= ys.min() + 2 and got[3] >= ys.max() - 2
    assert unrotate_box([10, 10, 20, 20], 0.0, (600, 400)) == [10, 10, 20, 20]


def test_math_tokens_back_to_text():
    from indra_ocr.math import detokenize

    toks = (
        list("we have ")
        + ["$", "x", "_", "{", "i", "}", "\\leq", "1", "$"]
        + list(" here")
    )
    assert detokenize(toks) == "we have $x _ { i } \\leq 1$ here"
    assert detokenize(["$", "\\alpha"]) == "$\\alpha$"  # an open span closes
    assert detokenize([" the", " bound"]) == "the bound"  # word tokens keep their space


def test_math_models_are_optional(tmp_path):
    from indra_ocr.math import MathReader

    (tmp_path / "math_enc.onnx").write_bytes(b"")  # only some of the files
    assert MathReader.load(tmp_path, None, None) is None


def test_a_vertical_page_reads_columns_as_lines_and_maps_boxes_back():
    from indra_ocr.pipeline import Line, OrtOCR, Page

    class Orient:  # five classes: the page is set in vertical columns
        def run(self, _, feed, run_options=None):
            return [np.array([[0.01, 0.0, 0.0, 0.01, 0.98]], np.float32)]

    seen = {}
    ocr = OrtOCR.__new__(OrtOCR)
    ocr.orient = Orient()

    def detect(g):
        seen["shape"] = g.shape
        # One column, as a line of the turned page (500 wide, 300 high).
        return np.array([[100, 20, 400, 40, 1.0]], np.float32)

    def recognize(g, b, **kw):
        seen["vertical"] = kw.get("vertical")
        return [("縦書き", 0.9, [])]

    ocr.detect, ocr.recognize = detect, recognize
    ocr.assemble = lambda wh, b, r, t, **kw: Page(
        [Line([float(v) for v in b[0, :4]], r[0][0], 0.9)], t
    )
    page = ocr(np.full((500, 300), 255, np.uint8))  # a tall page, 300 wide
    assert seen["shape"] == (300, 500) and seen["vertical"]  # turned counter-clockwise
    assert page.vertical and page.rotation == 0
    # (x, y) -> (W - y, x) with W = 300: the column runs down x 260-280, y 100-400.
    assert page.lines[0].box == [260.0, 100.0, 280.0, 400.0]
