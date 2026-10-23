"""Right-to-left text: visual order, routing to a group recognizer, mirrored pages."""

import numpy as np

from indra_ocr.rtl import flip, normalize

AR = "مرحبا"
HE = "שלום"


def test_flip_reverses_rtl_and_keeps_ltr_runs():
    assert flip(AR) == AR[::-1]
    v = flip(f"{AR} 2026-10-13 {HE} abc def")
    assert "abc def" in v and "2026-10-13" in v
    assert v.index("abc def") < v.index(HE[::-1]) < v.index("2026-10-13")
    assert flip(f"the word {HE} means peace") == f"the word {HE[::-1]} means peace"


def test_flip_is_its_own_inverse():
    rng = np.random.default_rng(0)
    alphabet = list(AR + HE + "abc 019()[].,:-") + ["َ", "١"]
    for _ in range(1000):
        s = normalize("".join(rng.choice(alphabet, rng.integers(1, 30))))
        assert flip(flip(s)) == s


def test_rtl_lines_read_in_logical_order_right_column_first():
    from indra_ocr.pipeline import OrtOCR
    from indra_ocr.vocab import Vocab

    vocab_rtl = Vocab(list("שלום "))  # ids 1..5

    class Rec:  # reads the letters of "שלום" left to right: ם ו ל ש
        def run(self, _, feed, run_options=None):
            b, w = feed["x"].shape[0], feed["x"].shape[-1] // 4
            best = np.zeros((b, w), np.int64)
            for i, t in enumerate([4, 3, 2, 1]):
                best[:, 2 * i + 1] = t
            return best, np.ones((b, w), np.float32), np.full(b, w)

    class Script:  # four groups; every line is right-to-left
        def run(self, _, feed, run_options=None):
            p = np.zeros((len(feed["widths"]), 4), np.float32)
            p[:, 2] = 1.0
            return [p]

    ocr = OrtOCR.__new__(OrtOCR)
    ocr.vocab, ocr.groups = Vocab(list("ab")), {"rtl": (Rec(), vocab_rtl)}
    ocr.rec, ocr.script, ocr.mask, ocr.height, ocr.order = (
        None,
        Script(),
        None,
        32,
        None,
    )
    gray = np.full((200, 800), 255, np.uint8)
    # Two columns, the left one first in detector order.
    boxes = np.array([[20, 10, 380, 42, 1], [420, 10, 780, 42, 1]], np.float32)
    rec = ocr.recognize(gray, boxes)
    assert rec[0][0] == HE[::-1]  # the recognizer's visual order
    page = ocr.assemble((800, 200), boxes[:, :4], rec, {})
    assert [ln.text for ln in page.lines] == [HE, HE]
    assert page.lines[0].box[0] > page.lines[1].box[0]  # the right column first


def test_two_way_classifier_routes_to_the_world_recognizer():
    from indra_ocr.pipeline import OrtOCR
    from indra_ocr.vocab import Vocab

    class Rec:  # one fixed token id per line
        def __init__(self, tok):
            self.tok = tok

        def run(self, _, feed, run_options=None):
            b, w = feed["x"].shape[0], feed["x"].shape[-1] // 4
            best = np.zeros((b, w), np.int64)
            best[:, 1] = self.tok
            return best, np.ones((b, w), np.float32), np.full(b, w)

    class Script:  # lines wider than 300 px are "world"
        def run(self, _, feed, run_options=None):
            wide = feed["widths"] > 300
            return [np.stack([~wide, wide], 1).astype(np.float32)]

    ocr = OrtOCR.__new__(OrtOCR)
    ocr.vocab, ocr.groups = Vocab(list("ab")), {"world": (Rec(1), Vocab(list("न")))}
    ocr.script_groups = ("base", "world")
    ocr.rec, ocr.script, ocr.mask, ocr.height = Rec(1), Script(), None, 32
    gray = np.full((200, 800), 255, np.uint8)
    boxes = np.array([[10, 10, 100, 42, 1], [10, 100, 700, 132, 1]], np.float32)
    (t0, _, _), (t1, _, _) = ocr.recognize(gray, boxes)
    assert t0 == "a" and t1 == "न"


def test_unsure_line_keeps_the_more_confident_reading():
    from indra_ocr.pipeline import OrtOCR
    from indra_ocr.vocab import Vocab

    class Rec:  # one token per line, at a fixed confidence
        def __init__(self, tok, conf):
            self.tok, self.conf = tok, conf

        def run(self, _, feed, run_options=None):
            b, w = feed["x"].shape[0], feed["x"].shape[-1] // 4
            best = np.zeros((b, w), np.int64)
            best[:, 1] = self.tok
            return best, np.full((b, w), self.conf, np.float32), np.full(b, w)

    class Script:  # world, but only 60% sure
        def run(self, _, feed, run_options=None):
            n = len(feed["widths"])
            return [np.tile(np.array([[0.3, 0.1, 0.6]], np.float32), (n, 1))]

    ocr = OrtOCR.__new__(OrtOCR)
    ocr.vocab, ocr.script_groups = Vocab(list("ab")), ("base", "cjk", "world")
    ocr.groups = {"world": (Rec(1, 0.6), Vocab(list("न")))}
    ocr.rec, ocr.script, ocr.mask, ocr.height = Rec(1, 0.9), Script(), None, 32
    gray = np.full((200, 800), 255, np.uint8)
    ((text, conf, _),) = ocr.recognize(
        gray, np.array([[10, 10, 300, 42, 1]], np.float32)
    )
    assert text == "a" and conf > 0.85  # base read it with more confidence
