"""math_rescore: the character signatures that compare a beam's LaTeX reading with the line's draft."""
from indra_ocr import math as M


def test_signatures_match_reading_and_draft():
    reading = r"$x _ { i } ^ 2 + \alpha = \frac { 1 } { 2 }$"
    assert M._sig_reading(reading) == "xi2+=12"
    # The draft (text layer) writes the same line in Unicode; Greek and fraction
    # bars drop out of both signatures.
    assert M._sig_draft("xᵢ² + α = 1 2") == "xi2+=12"


def test_rescore_off_by_default():
    assert M.MathReader.rescore is False
