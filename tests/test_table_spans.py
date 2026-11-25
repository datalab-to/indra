import numpy as np

from indra_ocr.layout_post import table_html, table_html_heads

X = [(0, 100), (100, 300), (300, 460), (460, 620), (620, 800), (800, 1000)]


def _header():
    """Six columns, a two-row header: the first, second, fifth and sixth cells
    span both header rows, the third spans two columns in the first row, and a
    first-column cell wraps over two lines beside the body row."""
    boxes, texts = [], []

    def add(x0, x1, y0, y1, t):
        boxes.append([x0, y0, x1, y1])
        texts.append(t)

    add(*X[0], 0, 40, "主要会计数据")
    add(*X[1], 0, 40, "2025年")
    add(300, 620, 0, 20, "2024年")
    add(*X[4], 0, 40, "本期比上年同期增减(%)")
    add(*X[5], 0, 40, "2023年")
    add(*X[2], 20, 40, "调整后")
    add(*X[3], 20, 40, "调整前")
    add(*X[0], 50, 70, "归属于上市公司股")
    add(*X[0], 71, 91, "东的净利润")
    for c, (x0, x1) in enumerate(X[1:], 1):
        add(x0, x1, 71, 91, f"v{c}")
    return np.array(boxes, np.float32), texts


def test_merged_header_cells_keep_their_spans():
    boxes, texts = _header()
    html = table_html(boxes, texts)
    assert '<th rowspan="2">主要会计数据</th>' in html
    assert '<th rowspan="2">2025年</th>' in html
    assert '<th colspan="2">2024年</th>' in html
    assert '<th rowspan="2">2023年</th>' in html
    assert html.count("<tr>") == 3  # two header rows, one body row
    assert "<td>归属于上市公司股东的净利润</td>" in html


def test_heads_merged_header_cells_keep_their_spans():
    boxes, texts = _header()
    n = len(boxes)
    tab = np.full((3, n, n), -5.0, np.float32)

    def same(k, ids):
        for i in ids:
            for j in ids:
                if i != j:
                    tab[k, i, j] = 5.0

    same(0, [0, 2, 3, 4])  # first header row
    same(0, [5, 6])  # second header row
    same(0, list(range(7, n)))  # body row
    same(2, [7, 8])  # one wrapped cell over two lines
    for col in ([0, 7, 8], [1, 9], [2, 5, 10], [3, 6, 11], [4, 12], [5, 13]):
        same(1, col)  # same column
    tags = np.full((n, 2), -5.0, np.float32)
    tags[:7, 0] = 5.0  # header words

    html = table_html_heads(boxes, texts, tab)
    assert '<th rowspan="2">主要会计数据</th>' in html
    assert '<th colspan="2">2024年</th>' in html
    assert "<td>归属于上市公司股东的净利润</td>" in html
    assert html.count("<tr>") == 3
