"""Shortcuts for ASCII text give what the per-character code gives (latin_line, has_rtl,
display_len), and the math decoder's per-step mean is np.mean of each line's probabilities."""

import unicodedata

import numpy as np

from indra_ocr.math import latin_line
from indra_ocr.order_post import display_len
from indra_ocr.rtl import has_rtl, is_rtl

ASCII = "".join(map(chr, range(128)))


def test_no_ascii_character_needs_the_slow_paths():
    for c in ASCII:
        assert not unicodedata.category(c).startswith("M")
        assert not (c.isalpha() and not unicodedata.name(c, "").startswith("LATIN"))
        assert not is_rtl(c)
        assert unicodedata.east_asian_width(c) not in "WF"


def test_ascii_shortcuts_agree_with_the_character_loops():
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = "".join(rng.choice(list(ASCII), int(rng.integers(0, 40))))
        assert latin_line(t) is True
        assert has_rtl(t) is False
        assert display_len(t) == len(t)
    # the slow paths still run for other text
    assert not latin_line("Привет мир")
    assert has_rtl("abc שלום")
    assert display_len("中文") == 4


def test_row_means_are_np_mean_of_each_list():
    rng = np.random.default_rng(1)
    for n in (1, 7, 8, 9, 63, 128, 129, 300):
        P = rng.random((6, n)).astype(np.float32).astype(np.float64)
        rows = np.array([0, 2, 5])
        m = P[rows, :n].mean(1)
        for i, r in enumerate(rows):
            assert m[i] == np.mean(P[r].tolist())
