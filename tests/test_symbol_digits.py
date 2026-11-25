"""Digits that a font gives to symbol glyphs (a minus coded '2', a plus-minus coded
'6'): the word is read from the page and only those digits change, only to a symbol."""

import numpy as np

from indra_ocr.document import symbol_candidates, symbol_words


def _toks(text, x0=0.0, step=10.0):
    return [(c, x0 + step * i, 0) for i, c in enumerate(text)]


def test_a_digit_the_reading_has_as_a_symbol_is_a_candidate():
    assert symbol_candidates("x 20.20* y", "x -0.20* y", [2]) == {2: "-"}
    assert symbol_candidates("62.067.0", "62.0±7.0", [4]) == {4: "±"}


def test_no_candidate_from_letters_digits_or_unaligned_readings():
    assert symbol_candidates("2.5", "2.5", [0]) == {}  # the page shows the digit
    assert symbol_candidates("25", "Z5", [0]) == {}  # a letter is not a symbol
    assert symbol_candidates("20.20", "0.20", [0]) == {}  # no one-for-one place
    assert symbol_candidates("a 20 b", "a -0 b", [3]) == {}  # not a checked digit
    # light punctuation is what a cut or tiny glyph reads as, not a symbol
    assert symbol_candidates("16.92", "16.9.", [4]) == {}
    assert symbol_candidates("IC50", "IC_0", [2]) == {}
    assert symbol_candidates("Cases1", "Cases!", [5]) == {}


def test_words_with_a_digit_in_a_symbol_box():
    toks = _toks("a 20.2 b 7")
    sym = np.array([[18.0, 0.0, 22.0, 10.0]], np.float32)  # over the '2' at x=20
    box = [0.0, 0.0, 100.0, 10.0]
    assert symbol_words(toks, box, sym) == [(2, 6, [2])]
    # a box on another line (no vertical overlap) marks nothing
    assert symbol_words(toks, box, np.array([[18.0, 30.0, 22.0, 40.0]], np.float32)) == []


def test_only_ascii_digits_are_checked():
    # circled or other-script digits are real characters, not fallback codes
    toks = [("❶", 10.0, 0), (" ", 10.0, 0), ("२", 30.0, 0), ("८", 40.0, 0)]
    sym = np.array([[0.0, 0.0, 50.0, 10.0]], np.float32)
    assert symbol_words(toks, [0.0, 0.0, 60.0, 10.0], sym) == []


def test_the_fix_reads_its_lines_without_mathrec():
    """The candidate lines are read by the recognizers alone: MathRec's LaTeX
    reading would hide the symbol (math lines are read after the fix)."""
    from indra_ocr.document import Converter

    seen = []

    class OCR:
        def recognize(self, gray, boxes, batch=64, vertical=False, math=True):
            seen.append(math)
            return [("1", 0.9, [])] * len(boxes)

    conv = Converter.__new__(Converter)
    conv.ocr = OCR()
    toks = [[("1", 10.0, 0)]]
    boxes = np.array([[0, 0, 100, 20]], np.float32)
    sym = np.array([[5, 0, 15, 20]], np.float32)
    assert conv._fix_symbol_digits(np.zeros((40, 200), np.uint8), boxes, toks, sym) == 0
    assert seen == [False]
