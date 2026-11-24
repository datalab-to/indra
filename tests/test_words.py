"""Line.words (load(words=True)): words in reading order with boxes from their
characters' x positions (issue #7)."""

from indra_ocr.pipeline import OrtOCR, Word


def _ocr(rtl=False):
    o = OrtOCR.__new__(OrtOCR)
    o._page_rtl = rtl

    class V:
        index = {" ": 1}

    o.vocab = V()
    return o


def _toks(text, x0=0.0, step=10.0):
    return [(c, x0 + step * i) for i, c in enumerate(text)]


def test_words_follow_the_line_text_and_their_characters():
    o = _ocr()
    box = [0.0, 0.0, 200.0, 20.0]
    ws = o.line_words(box, ("ab cd", 0.9, _toks("ab cd")))
    assert [w.text for w in ws] == ["ab", "cd"]
    assert all(isinstance(w, Word) for w in ws)
    assert ws[0].box[0] < ws[1].box[0] and ws[0].box[2] < ws[1].box[2]  # left word first
    assert ws[0].box[1] == 0.0 and ws[0].box[3] == 20.0  # the line's height


def test_text_layer_rtl_line_keeps_reading_order():
    # a text layer gives content (logical) order: x falls from right to left
    o = _ocr(rtl=True)
    text = "שלום עולם"
    toks = [(c, 200.0 - 10.0 * i, i) for i, c in enumerate(text)]
    ws = o.line_words([0.0, 0.0, 220.0, 20.0], (text, 1.0, toks))
    assert [w.text for w in ws] == ["שלום", "עולם"]
    assert ws[0].box[0] > ws[1].box[0]  # the first word is the right one


def test_a_line_the_math_reader_replaced_is_one_word():
    o = _ocr()
    ws = o.line_words([0.0, 0.0, 100.0, 20.0], ("$x^2$", 0.8, _toks("x2")))
    assert [(w.text, w.box) for w in ws] == [("$x^2$", [0.0, 0.0, 100.0, 20.0])]
