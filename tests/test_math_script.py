"""MathRec reads Latin script only: which lines may go to it."""

import pytest

from indra_ocr.math import has_math, latin_line


@pytest.mark.parametrize(
    "text",
    [
        "we have x = 2",
        "α + β = x",  # Greek letters on their own are math symbols
        "Ω = (−2, 2)2",
        "πx",
        "frå eit avløpsanlegg",  # Latin letters with diacritics
        "200",  # no letters
    ],
)
def test_latin_lines_may_go_to_math(text):
    assert latin_line(text)


@pytest.mark.parametrize(
    "text",
    [
        "Η εργασία αυτή",  # Greek prose
        "Это текст x",  # Cyrillic prose
        "中文 x=1",
        "مرحبا x",
        "हिन्दी x",
    ],
)
def test_lines_of_other_scripts_keep_their_reading(text):
    assert not latin_line(text)


@pytest.mark.parametrize(
    "text",
    [
        "$2 0 0$",
        "Total $0 . 0 \\%$",
        "$\\# 9 1$",
        "$\\mathbf { 1 0 1 5 }$",
        "$\\mathsf { F }$",  # a styled letter
        "2) Form $L o g \\ B o o k$",  # a word in math italic
        "$( n = 5 0 )$",
        "$6 6 \\pm 5$",  # symbols plain text has as well
        "$\\downarrow \\mathrm { E C P }$",
        "no math",
    ],
)
def test_plain_text_in_math_mode_keeps_the_reading(text):
    assert not has_math(text)


@pytest.mark.parametrize(
    "text", ["$R ^ { 2 }$", "see $^ { 2 1 }$", "$\\frac { a } { b }$", "$x _ { i } \\leq 1$", "$9 \\times 1 0 ^ { - 4 }$"]
)
def test_math_replaces_the_reading(text):
    assert has_math(text)


def test_accents_come_back_and_math_stays():
    from indra_ocr.math import restore_accents

    reading = "Z divodu volime nyni $\\mathrm { R _ { 2 } }$ mensi a to hodnotu"
    draft = "Z důvodu volíme nyní R2 menší a to hodnotu"
    assert restore_accents(reading, draft) == "Z důvodu volíme nyní $\\mathrm { R _ { 2 } }$ menší a to hodnotu"


def test_without_a_plain_reading_the_math_reading_stays():
    from indra_ocr.math import restore_accents

    assert restore_accents("$x _ { i } \\leq 1$", "") == "$x _ { i } \\leq 1$"


def test_an_accent_written_as_math_is_not_math():
    from indra_ocr.math import accept_math

    assert accept_math("Dodatki do $\\bar { z }$ ywnosci $-$ cel", "Dodatki do żywności – cel") is None
    assert accept_math("A satisfagao dos consumidores $\\hat { \\epsilon }$ um", "A satisfação dos consumidores é um") is None


def test_latex_outside_the_spans_is_a_malformed_reading():
    from indra_ocr.math import accept_math

    assert accept_math("b\\delta la da tim $\\bar { c o }$ ten", "bố là đã tìm ra được vị") is None


def test_a_line_of_one_symbol_keeps_its_reading():
    from indra_ocr.math import accept_math

    assert accept_math("$\\rightarrow$", "-") is None
    assert accept_math("$\\sim$", "~") is None


def test_real_math_with_accented_prose_is_accepted():
    from indra_ocr.math import accept_math

    out = accept_math("Vypocet rezistoru $\\mathrm { R } _ { 3 }$ pro", "Výpočet rezistoru R3 pro")
    assert out == "Výpočet rezistoru $\\mathrm { R } _ { 3 }$ pro"


def test_a_symbol_plain_text_has_is_not_math():
    from indra_ocr.math import accept_math

    assert accept_math("Kitap Değerlendirmesi $\\mid$ Book Review", "Kitap Değerlendirmesi | Book Review") is None


def _reader(tokens):
    from indra_ocr.math import MathReader

    r = MathReader.__new__(MathReader)  # no models: writes() needs only the vocab
    r.chars = frozenset(t for t in tokens if len(t) == 1)
    return r


def test_math_writes_the_scripts_of_its_vocab():
    latin = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    r = _reader(latin + ["\\frac", "\\text"])
    assert r.writes("let x be a number")
    assert not r.writes("设 x 为正数")  # Chinese prose: a Latin-only MathRec does not read it
    zh = _reader(latin + list("设为正数"))
    assert zh.writes("设 x 为正数")
    assert not zh.writes("")  # no letters: latin_line decides
