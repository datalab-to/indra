import json

import numpy as np

from indra_ocr.layout_post import TABLE, page_markdown
from indra_ocr.order_post import BLOCK_CLASSES
from indra_ocr.table_verify import TableVerifier, _any_run, block_features, parse_table

TEXT = BLOCK_CLASSES.index("Text")


def _page():
    """Two blocks the layout calls Table: a 3 x 2 grid of short cells, and one paragraph of three long lines."""
    boxes = np.array(
        [
            [100, 100, 200, 120], [400, 100, 500, 120],
            [100, 130, 200, 150], [400, 130, 500, 150],
            [100, 160, 200, 180], [400, 160, 500, 180],
            [100, 300, 900, 320], [100, 330, 900, 350], [100, 360, 700, 380],
        ],
        np.float32,
    )  # fmt: skip
    texts = ["Name", "Value", "a", "1", "b", "2"] + ["A long sentence of running text that fills a line."] * 3
    order = list(range(9))
    cls = np.full((9, len(BLOCK_CLASSES)), -2.0, np.float32)
    cls[:, TABLE] = 2.0
    cls[:, TEXT] = 1.0
    same = np.full((9, 9), -1.0, np.float32)
    same[:6, :6] = 1.0
    same[6:, 6:] = 1.0
    return boxes, texts, order, cls, same


def _model(path, names, weight_of):
    w = [weight_of.get(k, 0.0) for k in names]
    m = {"features": names, "mean": [0.0] * len(names), "scale": [1.0] * len(names), "weights": w, "bias": 0.0,
         "threshold": 0.5}
    path.write_text(json.dumps(m))
    return TableVerifier.load(path)[0]


def test_parse_table_spans():
    rows = parse_table('<table><tr><th colspan="2">A</th></tr><tr><td>x</td><td></td></tr></table>')
    assert rows == [["A", ""], ["x", ""]]


def test_features_have_fixed_names():
    boxes, texts, order, cls, same = _page()
    f = block_features(list(range(6)), boxes, texts, cls, "<table><tr><td>a</td></tr></table>", None, 2, True)
    g = block_features(list(range(6, 9)), boxes, texts, cls, "", None, 2, False)
    assert sorted(f) == sorted(g)
    assert f["g_multi_rows"] == 1.0 and g["g_multi_rows"] == 0.0


def test_veto_relabels_only_the_false_table(tmp_path):
    boxes, texts, order, cls, same = _page()
    names = sorted(block_features(list(range(6)), boxes, texts, cls, "", None, 2, True)) + ["tsr"]
    # P(false) > .5 exactly when lines per row < 1.5: the paragraph (1 line per row), not the grid (2 per row).
    v = _model(tmp_path / "tv.json", names, {"g_lines_row": -10.0})
    v.b = 15.0
    cls_v, tables = v.apply(boxes, texts, order, cls, same, None, None, None, None)
    assert cls_v is not cls
    assert cls_v[:6].sum(0).argmax() == TABLE
    assert cls_v[6:].sum(0).argmax() == TEXT
    md = page_markdown(boxes, texts, order, cls_v, same)
    assert md.count("<table>") == 1 and "A long sentence" in md.split("</table>")[1]


def test_no_veto_returns_inputs(tmp_path):
    boxes, texts, order, cls, same = _page()
    names = sorted(block_features(list(range(6)), boxes, texts, cls, "", None, 2, True)) + ["tsr"]
    v = _model(tmp_path / "tv.json", names, {})
    v.b = -5.0
    tabs = {(0, 1, 2, 3, 4, 5): "<table></table>"}
    cls_v, t2 = v.apply(boxes, texts, order, cls, same, None, None, tabs, None)
    assert cls_v is cls and t2 is tabs


def test_extreme_scores_do_not_overflow(tmp_path):
    names = ["a", "b"]
    v = _model(tmp_path / "tv.json", names, {"a": 1e6})
    assert v.p_false({"a": -1e6, "b": float("nan")}) == 1.0 / (1.0 + np.exp(50.0))
    assert v.p_false({"a": 1e6, "b": 0.0}) > 0.999


def test_any_run_matches_opencv_open():
    import cv2

    rng = np.random.default_rng(0)
    for _ in range(500):
        h, w = int(rng.integers(4, 40)), int(rng.integers(4, 120))
        ink = (rng.random((h, w)) < rng.uniform(0.3, 0.98)).astype(np.uint8)
        n = int(rng.integers(1, w + 3))
        ref = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (n, 1))).any(1)
        assert np.array_equal(ref, _any_run(ink, n))


def test_ruling_dark_crop():
    from indra_ocr.table_verify import ruling

    g = np.full((60, 200), 10, np.uint8)  # median 10: the ink threshold would be -20
    g[30, 5:195] = 0
    assert ruling(g, (10, 10, 190, 50)) == (0.0, 0.0)


def test_download_includes_table_verify():
    from indra_ocr.models import download_patterns

    for math in (False, True):
        allow, ignore = download_patterns(math=math)
        assert "table_verify.json" in allow and "table_verify.json" not in ignore


def test_decided_matches_full_score(tmp_path):
    names = ["a", "rule_h", "rule_v"]
    v = _model(tmp_path / "tv.json", names, {"a": 1.0, "rule_h": 2.0, "rule_v": -1.0})
    rng = np.random.default_rng(2)
    for _ in range(300):
        f = {"a": float(rng.normal() * 3)}
        d = v.decided(f)
        full = [v.p_false({**f, "rule_h": h, "rule_v": u}) >= 0.5 for h in np.linspace(0, 1, 5) for u in np.linspace(0, 1, 5)]
        if d is not None:
            assert all(x == d for x in full)


def test_load_layer_set(tmp_path):
    base = {"features": ["a"], "mean": [0.0], "scale": [1.0], "weights": [1.0], "bias": 0.0, "threshold": 0.5}
    (tmp_path / "one.json").write_text(json.dumps(base))
    ocr, layer = TableVerifier.load(tmp_path / "one.json")
    assert layer is None and ocr.threshold == 0.5
    (tmp_path / "two.json").write_text(json.dumps({**base, "layer": {**base, "weights": [-1.0], "threshold": 0.7}}))
    ocr, layer = TableVerifier.load(tmp_path / "two.json")
    assert ocr.w[0] == 1.0 and layer.w[0] == -1.0 and layer.threshold == 0.7
