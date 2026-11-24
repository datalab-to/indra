"""The text-layer check (agree.py): a layer whose characters are not what the recognizer
reads in the page image (a legacy font, a broken ToUnicode map) is not trusted, and OCR
reads the page."""

import math

import numpy as np
import pytest

from indra_ocr.agree import UNKNOWN, forced_gap, layer_text, snippet, target_ids
from indra_ocr.document import layer_matches
from indra_ocr.vocab import Vocab

V = Vocab(list("abcxyz "))


def frames(seq: str, k: int = 3):
    """Recognizer outputs (blank, top-k values, top-k ids) whose best path reads seq,
    each character in one frame followed by a blank frame."""
    peaks = []
    for c in seq:
        peaks += [V.index[c], 0]
    T, C = len(peaks), len(V.chars) + 1
    lp = np.full((T, C), math.log(0.02))
    for t, c in enumerate(peaks):
        lp[t, c] = math.log(0.9)
    top = np.argsort(-lp, axis=1)[:, :k]
    return lp[:, 0], np.take_along_axis(lp, top, 1), top


def test_the_layer_text_the_image_shows_has_no_gap():
    blank, topv, topi = frames("abc")
    assert forced_gap(blank, topv, topi, target_ids("abc", V, False)) < 1e-9


def test_other_letters_have_a_large_gap():
    blank, topv, topi = frames("abc")
    assert forced_gap(blank, topv, topi, target_ids("xyz", V, False)) > 3.0
    # A private-use code counts like a wrong letter.
    assert target_ids("a\ue000c", V, False) == [(V.index["a"],), UNKNOWN, (V.index["c"],)]
    assert forced_gap(blank, topv, topi, target_ids("a\ue000c", V, False)) > 1.0


def test_a_character_the_recognizer_cannot_write_counts_as_a_miss():
    # A broken letter map writes characters the recognizer cannot write
    # (a Cyrillic letter in a Telugu layer, modifier letters in an Arabic one).
    blank, topv, topi = frames("abc")
    assert target_ids("aQc", V, False) == [(V.index["a"],), UNKNOWN, (V.index["c"],)]
    assert forced_gap(blank, topv, topi, target_ids("aQc", V, False)) > 0.2


def test_case_does_not_count():
    cased = Vocab(list("abcABC "))
    peaks = [cased.index[c] for c in "ABC"]
    T, C = 2 * len(peaks), len(cased.chars) + 1
    lp = np.full((T, C), math.log(0.02))
    for t, c in enumerate(peaks):
        lp[2 * t, c] = math.log(0.9)
        lp[2 * t + 1, 0] = math.log(0.9)
    top = np.argsort(-lp, axis=1)[:, :3]
    args = (lp[:, 0], np.take_along_axis(lp, top, 1), top)
    assert forced_gap(*args, target_ids("abc", cased, False)) < 1e-9  # small capitals


def test_a_mark_the_recognizer_misses_is_no_evidence():
    accented = Vocab(list("abcċ "))
    blank, topv, topi = (lambda lp: (lp[:, 0], np.take_along_axis(lp, np.argsort(-lp, 1)[:, :3], 1), np.argsort(-lp, 1)[:, :3]))(
        _frames_lp(accented, "acb"))
    assert forced_gap(blank, topv, topi, target_ids("aċb", accented, False)) < 1e-9  # the image shows c
    blank, topv, topi = (lambda lp: (lp[:, 0], np.take_along_axis(lp, np.argsort(-lp, 1)[:, :3], 1), np.argsort(-lp, 1)[:, :3]))(
        _frames_lp(accented, "aċb"))
    assert forced_gap(blank, topv, topi, target_ids("acb", accented, False)) > 1.0  # the layer lacks the mark


def _frames_lp(vocab, seq):
    peaks = []
    for c in seq:
        peaks += [vocab.index[c], 0]
    lp = np.full((len(peaks), len(vocab.chars) + 1), math.log(0.02))
    for t, c in enumerate(peaks):
        lp[t, c] = math.log(0.9)
    return lp


def test_a_layer_longer_than_the_frames_cannot_fit():
    blank, topv, topi = frames("a")
    assert forced_gap(blank, topv, topi, target_ids("abcabc", V, False)) == float("inf")


def test_a_right_to_left_layer_is_read_in_visual_order():
    hebrew = Vocab(list("אבג "))
    assert target_ids("אבג", hebrew, True) == [(hebrew.index[c],) for c in "גבא"]
    assert target_ids("אבג", hebrew, False) == [(hebrew.index[c],) for c in "אבג"]


def test_hangul_is_read_as_jamo():
    jamo = Vocab(["ᄀ", "ᅡ", "ᆨ", " "])  # ㄱ ㅏ ㄱ (final)
    assert target_ids("각", jamo, False) == [(1,), (2,), (3,)]


def test_snippet_cuts_the_line_start_at_a_word_gap():
    toks = [(c, 10.0 * i, 0) for i, c in enumerate("abc xyz abc")]
    part, cut = snippet(toks, 2)  # 3 letters: the cut waits for the word gap
    assert (layer_text(part, False), cut) == ("abc", 30.0)
    part, cut = snippet(toks, 20)  # a short line: all of it
    assert (layer_text(part, False), cut) == ("abc xyz abc", None)


def test_snippet_keeps_a_combining_mark_with_its_letter():
    # Thai: the vowel sign above the last letter has an x after the letter's.
    toks = [(c, 10.0 * i, 0) for i, c in enumerate("\u0e2a\u0e32\u0e23\u0e30\u0e1b\u0e0f\u0e34\u0e23\u0e39")]
    part, cut = snippet(toks, 6)
    assert layer_text(part, False) == "\u0e2a\u0e32\u0e23\u0e30\u0e1b\u0e0f\u0e34"
    assert cut == 65.0


def test_runs_go_in_reading_order():
    # Two runs on one line, stored right run first (a page set right to left).
    toks = [(c, 60.0 + 10 * i, 1) for i, c in enumerate("JOURNAL")]
    toks += [(" ", 120.0, 2)] + [(c, 10.0 * i - 50, 2) for i, c in enumerate("ALGEMEINER")]
    assert layer_text(toks, False) == "ALGEMEINER JOURNAL"
    assert layer_text(toks, True) == "JOURNAL ALGEMEINER"


def test_words_go_in_line_order():
    # One run that stores the line's second half first.
    toks = [(c, 100.0 + 10 * i, 1) for i, c in enumerate("yi tambayan")]
    toks += [(" ", 95.0, 1)] + [(c, 10.0 * i, 1) for i, c in enumerate("rigakafi kira")]
    assert layer_text(toks, False) == "rigakafi kira yi tambayan"


def test_the_page_fails_when_many_lines_read_other_letters():
    def page(gaps, conf=0.97, group="world"):
        return [(g, conf, group) for g in gaps]

    assert layer_matches(page((0.0, 0.1, 0.0, 0.3, 0.0, 0.2)))  # a sound layer
    assert layer_matches(page((0.0, 0.0, 0.0, 0.0, 0.0, 9.0)))  # one line in an odd font
    assert not layer_matches(page((0.0, 2.0, 1.5, 3.0, 0.5, 2.5)))  # some letters wrong on most lines
    assert not layer_matches(page((12.0, 14.0, 0.0, 11.0, 15.0, 13.0), group="base"))  # a legacy font
    assert not layer_matches(page((float("inf"),) * 4 + (0.0, 0.0), group="base"))  # the layer does not fit
    assert layer_matches([])  # nothing checked: keep the layer
    assert layer_matches([(None, 0.0, "base"), (None, 0.0, "base"), (0.0, 0.97, "base")])  # math lines are left out
    assert layer_matches([(None, 0.0, "base"), (None, 0.0, "base")])


def test_an_unsure_recognizer_counts_less():
    gaps = [0.8, 1.2, 0.6, 1.1, 0.3, 0.9]
    assert not layer_matches([(g, 0.98, "world") for g in gaps])
    assert layer_matches([(g, 0.4, "world") for g in gaps])


def test_a_latin_layer_needs_more_evidence():
    # A Latin recognizer that misreads a rare letter on some lines (Maltese, Fon)
    # does not fail a Latin layer; the same gaps in another script do.
    gaps = [0.8, 1.2, 0.6, 1.1, 0.3, 0.9]
    assert layer_matches([(g, 0.98, "base") for g in gaps])
    assert not layer_matches([(g, 0.98, "world") for g in gaps])


def test_a_sure_latin_recognizer_fails_a_broken_spacing_layer():
    # A sure recognizer that reads most lines otherwise (a layer with broken word
    # spacing: "Ú Z EMNÍPLÁ NROHOZ NICE" for "ÚZEMNÍ PLÁN ROHOZNICE") fails a Latin layer.
    assert not layer_matches([(g, 0.985, "base") for g in (2.16, 2.37, 0.72, 0.0, 1.02, 1.75)])


def test_a_less_sure_recognizer_keeps_a_latin_layer():
    # Larger gaps from a less sure recognizer (rare Fon letters) do not fail a Latin layer.
    fon = ((1.0, 0.95), (3.1, 0.92), (2.31, 0.94), (2.04, 0.9), (0.61, 0.92), (2.01, 0.89))
    assert layer_matches([(g, c, "base") for g, c in fon])


def test_top_k_from_logits_is_the_log_softmax_top_k():
    from indra_ocr.pipeline import _top_k

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2, 7, 30)).astype(np.float32)
    blank, topv, topi = _top_k(logits, 5, k=4)
    lp = logits[:, :5] - np.log(np.exp(logits[:, :5]).sum(-1, keepdims=True))
    want = np.argsort(-lp, axis=-1)[..., :4]
    assert (topi == want).all()
    assert np.allclose(topv, np.take_along_axis(lp, want, -1), atol=1e-5)
    assert np.allclose(blank, lp[..., 0], atol=1e-5)
