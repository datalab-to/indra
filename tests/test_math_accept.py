"""The learned accept head (math_accept.json): the old rule without it, the head's decision with it, and the table
feature that sends a reading on a table line back to the plain line."""
from indra_ocr import math as M


class _Reader:
    """Only what MathReader.accept needs."""

    accept = M.MathReader.accept
    _spliced = M.MathReader._spliced

    def __init__(self, head=None):
        self.accept_head = head


def _head(table_weight=0.0):
    # features: conf, rconf, accept_math_ok, share_in_math, draft_agree, log_len, ok_x_conf, in_table
    w = [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, table_weight]
    return {"w": w, "mu": [0.9] + [0.0] * 7, "sd": [0.05] + [1.0] * 7, "tau": 0.5}


def test_old_rule_without_head():
    r = _Reader()
    assert r.accept("$x ^ 2$", 0.95, "x2") is not None
    assert r.accept("$x ^ 2$", 0.85, "x2") is None


def test_head_decides_on_confidence():
    r = _Reader(_head())
    assert r.accept("$x ^ 2$", 0.95, "x2") is not None
    assert r.accept("$x ^ 2$", 0.85, "x2") is None


def test_head_keeps_plain_writable_math():
    # accept_math rejects math that plain text can write; the head may keep the reading
    r = _Reader(_head())
    assert M.accept_math("$z = g ' x '$", "z = g a") is None
    assert r.accept("$z = g ' x '$", 0.95, "z = g a") == "$z = g ' x '$"


def test_malformed_reading_never_accepted():
    r = _Reader(_head())
    assert r.accept("x ^ 2 $y$", 0.99, "x2 y") is None


def test_table_feature_rejects():
    r = _Reader(_head(table_weight=-20.0))
    assert r.accept("$x ^ 2$", 0.95, "x2") is not None
    assert r.accept("$x ^ 2$", 0.95, "x2", 1.0, in_table=1.0) is None



def test_download_gets_the_head_with_math():
    from indra_ocr.models import MATH_FILES, download_patterns

    allow, ignore = download_patterns(math=True)
    assert "math_accept.json" in allow and "math_accept.json" not in ignore
    allow, ignore = download_patterns(math=False)
    assert "math_accept.json" not in allow and "math_accept.json" in ignore
    # optional: a model folder without the head still loads
    assert "math_accept.json" not in MATH_FILES
