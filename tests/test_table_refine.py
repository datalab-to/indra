"""tsr_refine.onnx in read_table: without a refiner the output is unchanged, and the
refiner can change only the word tags (header, outside the table)."""

from types import SimpleNamespace

import numpy as np

from indra_ocr.table_rel import N_REFINE_TEXT, read_table, refine_text_features

# A 3 x 2 table, one word per cell: (text, x, row).
CELLS = [("Name", 0, 0), ("Total", 120, 0), ("a", 0, 1), ("12", 120, 1), ("b", 0, 2), ("3.5", 120, 2)]


def _lines():
    # One line per cell; a token = (text, x).  No spaces: one word per line.
    return [([x, 20.0 * r, x + 30, 20.0 * r + 10], [(t, x)]) for t, x, r in CELLS]


class FakeRel:
    """A tsr_rel session: same cell = same line, same row = same y, same column = same x.
    ``head`` is the header tag logit of the first row."""

    def __init__(self, head: float):
        self.head = head

    def get_inputs(self):
        return [SimpleNamespace(shape=[1, None, 33]), SimpleNamespace(shape=[1, None, 4])]

    def run(self, _names, inp):
        n = inp["feat"].shape[1]
        assert n == len(CELLS)
        pair = np.full((1, 3, n, n), -5.0, np.float32)
        for i, (_, xi, ri) in enumerate(CELLS):
            for j, (_, xj, rj) in enumerate(CELLS):
                pair[0, 0, i, j] = 5.0 if i == j else -5.0
                pair[0, 1, i, j] = 5.0 if ri == rj else -5.0
                pair[0, 2, i, j] = 5.0 if xi == xj else -5.0
        tags = np.full((1, n, 2), -5.0, np.float32)
        tags[0, :2, 0] = self.head
        return [pair, tags]


class FakeRefine:
    """A tsr_refine session that makes the first row a header and records its inputs."""

    def __init__(self):
        self.calls = []

    def run(self, names, inp):
        self.calls.append((names, inp))
        tags = inp["tag"].copy()
        tags[0, :2, 0] = 5.0
        return [tags]


def _read(rel, refine=None):
    gray = np.full((80, 200), 255, np.uint8)
    return read_table(rel, gray, _lines(), lambda t: t == " ", lambda toks: "".join(t[0] for t in toks), refine_sess=refine)


def test_no_refiner_keeps_the_table():
    a = _read(FakeRel(-5.0))
    b = read_table(FakeRel(-5.0), np.full((80, 200), 255, np.uint8), _lines(), lambda t: t == " ", lambda toks: "".join(t[0] for t in toks))
    assert a == b
    assert "<th>" not in a and "<td>Name</td>" in a


def test_refiner_changes_only_the_header():
    ref = FakeRefine()
    base = _read(FakeRel(-5.0))
    out = _read(FakeRel(-5.0), ref)
    assert "<tr><th>Name</th><th>Total</th></tr>" in out
    # Same rows and cells: only the header markup differs.
    assert out.replace("th>", "td>").replace("<thead>", "").replace("</thead>", "").replace(
        "<tbody>", ""
    ).replace("</tbody>", "") == base.replace("<thead>", "").replace("</thead>", "").replace(
        "<tbody>", ""
    ).replace("</tbody>", "")
    (names, inp), = ref.calls
    assert names == ["tag_out"]
    n = len(CELLS)
    assert inp["pair"].shape == (1, 3, n, n) and inp["tag"].shape == (1, n, 2)
    assert inp["box"].shape == (1, n, 4) and inp["feat"].shape[:2] == (1, n)
    assert inp["tfeat"].shape == (1, n, N_REFINE_TEXT) and inp["tfeat"].dtype == np.float32


def test_refine_text_features():
    f = refine_text_features(["Total:", "12.5%", "$1,200", "n/a", "*", "", "ABC"])
    assert f.shape == (7, N_REFINE_TEXT)
    assert f[0, 7] == 1 and f[0, 4] == 1 and f[0, 0] == 0  # ends with ':', upper first, no digit
    assert f[1, 3] == 1 and f[1, 8] == 1 and f[1, 0] == 1  # number form, '%', digit
    assert f[2, 3] == 1 and f[2, 9] == 1  # number form, currency
    assert f[3, 11] == 1  # n/a
    assert f[4, 12] == 1  # footnote mark
    assert not f[5].any()  # empty word
    assert f[6, 5] == 1 and np.isclose(f[6, 6], np.log1p(3) / 3.0)  # all upper, length
