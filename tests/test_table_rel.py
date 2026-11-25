import numpy as np

from indra_ocr.table_rel import cells_html, decode, norm_boxes, table_html, word_features, words_of


def _words():
    # A 3 x 2 table: a header row, then two body rows.  One word per cell, except
    # the header's first cell ("Sample size", two words).
    rows = [[("Sample", 0), ("size", 40), ("Value", 120)], [("a", 0), ("1", 120)], [("b", 0), ("2", 120)]]
    words, cell, k = [], [], 0
    for r, row in enumerate(rows):
        for t, x in row:
            words.append({"box": [x, 20.0 * r, x + 30, 20.0 * r + 10], "text": t, "line": r, "pos": 0, "n": 1})
    cell = [0, 0, 1, 2, 3, 4, 5]
    return words, cell, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]


def test_decode_and_html():
    words, cell, pos = _words()
    n = len(words)
    rc = [pos[c] for c in cell]
    pair = np.full((3, n, n), -5.0, np.float32)
    for i in range(n):
        for j in range(n):
            if cell[i] == cell[j]:
                pair[0, i, j] = 5
            if rc[i][0] == rc[j][0]:
                pair[1, i, j] = 5
            if rc[i][1] == rc[j][1]:
                pair[2, i, j] = 5
    tags = np.full((n, 2), -5.0, np.float32)
    tags[:3, 0] = 5  # header words
    st = decode(pair, tags, np.array([w["box"] for w in words], np.float32))
    assert st["n_rows"] == 3 and st["n_cols"] == 2 and st["n_head"] == 1
    html = table_html(st, words)
    assert "<tr><th>Sample size</th><th>Value</th></tr>" in html
    assert "<tr><td>a</td><td>1</td></tr>" in html


def test_features_and_boxes():
    lines = [([0, 0, 100, 10], [("a", 5), ("b", 12), (" ", 20), ("c", 60)])]
    words = words_of(lines, lambda t: t == " ", lambda toks: "".join(t[0] for t in toks))
    assert [w["text"] for w in words] == ["ab", "c"]
    f = word_features(words, np.full((20, 120), 255, np.uint8), (0, 0, 100, 10))
    assert f.shape == (2, 33) and f[0, 17] == 1 and f[1, 18] == 1 and f[0, 32] == 1
    b = norm_boxes(np.array([w["box"] for w in words], np.float32))
    assert b[:, 0].min() == 0 and np.isclose(np.median(b[:, 3] - b[:, 1]), 1.0)


def test_group_header_spans():
    # "Group" (narrow, centred) heads the columns "n" and "%": the same-column
    # relations put it over both; its geometry covers neither column by half.
    words = [
        {"box": [55.0, 0.0, 85.0, 10.0], "text": "Group", "line": 0, "pos": 0, "n": 1},
        {"box": [0.0, 20.0, 30.0, 30.0], "text": "n", "line": 1, "pos": 0, "n": 1},
        {"box": [110.0, 20.0, 140.0, 30.0], "text": "%", "line": 2, "pos": 0, "n": 1},
        {"box": [0.0, 40.0, 30.0, 50.0], "text": "1", "line": 3, "pos": 0, "n": 1},
        {"box": [110.0, 40.0, 140.0, 50.0], "text": "2", "line": 4, "pos": 0, "n": 1},
    ]
    row = [0, 1, 1, 2, 2]
    col = [{0, 1}, {0}, {1}, {0}, {1}]
    n = len(words)
    pair = np.full((3, n, n), -5.0, np.float32)
    for i in range(n):
        pair[0, i, i] = 5
        for j in range(n):
            if row[i] == row[j]:
                pair[1, i, j] = 5
            if col[i] & col[j]:
                pair[2, i, j] = 5
    tags = np.full((n, 2), -5.0, np.float32)
    tags[:3, 0] = 5
    boxes = np.array([w["box"] for w in words], np.float32)
    geo = decode(pair, tags, boxes)
    rel = decode(pair, tags, boxes, "rel")
    span = lambda st: [c[3] - c[2] + 1 for c in st["cells"] if words[c[4][0]]["text"] == "Group"][0]  # noqa: E731
    assert span(geo) == 1 and span(rel) == 2
    assert '<th colspan="2">Group</th>' in table_html(rel, words)


def test_stub_header_row_spans():
    # "Item" (one line, centred) heads the two header rows "Group" / "n %": the
    # same-row relations put it over both; its geometry covers neither row by half.
    words = [
        {"box": [0.0, 12.0, 30.0, 22.0], "text": "Item", "line": 0, "pos": 0, "n": 1},
        {"box": [60.0, 0.0, 120.0, 10.0], "text": "Group", "line": 1, "pos": 0, "n": 1},
        {"box": [60.0, 24.0, 80.0, 34.0], "text": "n", "line": 2, "pos": 0, "n": 1},
        {"box": [100.0, 24.0, 120.0, 34.0], "text": "%", "line": 3, "pos": 0, "n": 1},
        {"box": [0.0, 44.0, 30.0, 54.0], "text": "a", "line": 4, "pos": 0, "n": 1},
        {"box": [60.0, 44.0, 80.0, 54.0], "text": "1", "line": 5, "pos": 0, "n": 1},
        {"box": [100.0, 44.0, 120.0, 54.0], "text": "2", "line": 6, "pos": 0, "n": 1},
    ]
    row = [{0, 1}, {0}, {1}, {1}, {2}, {2}, {2}]
    col = [{0}, {1, 2}, {1}, {2}, {0}, {1}, {2}]
    n = len(words)
    pair = np.full((3, n, n), -5.0, np.float32)
    for i in range(n):
        pair[0, i, i] = 5
        for j in range(n):
            if row[i] & row[j]:
                pair[1, i, j] = 5
            if col[i] & col[j]:
                pair[2, i, j] = 5
    tags = np.full((n, 2), -5.0, np.float32)
    tags[:4, 0] = 5
    boxes = np.array([w["box"] for w in words], np.float32)
    rel = decode(pair, tags, boxes, "rel")
    rel2 = decode(pair, tags, boxes, "rel2")
    rows = lambda st: [c[1] - c[0] + 1 for c in st["cells"] if words[c[4][0]]["text"] == "Item"][0]  # noqa: E731
    assert rows(rel) == 1 and rows(rel2) == 2
    html = table_html(rel2, words)
    assert '<th rowspan="2">Item</th>' in html and '<th colspan="2">Group</th>' in html


def test_rule_extent():
    # A word "Group" with a rule under it that runs well past the word to both sides.
    gray = np.full((60, 300), 255, np.uint8)
    gray[22:24, 20:280] = 0  # the rule
    words = [{"box": [120.0, 5.0, 170.0, 20.0], "text": "Group", "line": 0, "pos": 0, "n": 1},
             {"box": [20.0, 30.0, 60.0, 45.0], "text": "n", "line": 1, "pos": 0, "n": 1}]
    f = word_features(words, gray, (20, 5, 280, 45))
    assert f[0, 26] == 1 and f[0, 27] > 0.3 and f[0, 28] > 0.3


def test_word_ids():
    # The model expects exactly this hash: fixed values.
    from indra_ocr.table_rel import word_ids

    ids = word_ids(["Age", "2019", "n (%)", "Ｍedian", "序號", "of", "Пенсия", "Öl"], 4096)
    assert ids.tolist() == [1109, 3643, 150, 279, 0, 2268, 0, 2334]  # non-Latin words: 0
    assert word_ids(["2019"], 4096)[0] == word_ids(["1987"], 4096)[0]  # digits are 0


def test_cluster_sparse_equals_cluster():
    # Components + the 2-item shortcut must give exactly _cluster's groups (same order up to sorting).
    from indra_ocr.order_post import _cluster
    from indra_ocr.table_rel import _cluster_sparse

    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 30, 120):
        for p in (0.02, 0.1, 0.5):
            L = np.where(rng.random((n, n)) < p, rng.normal(1, 1, (n, n)), -rng.random((n, n)) - 0.1)
            L = ((L + L.T) / 2).astype(np.float32)
            a = sorted(sorted(g) for g in _cluster_sparse(L))
            b = sorted(sorted(g) for g in _cluster(L))
            assert a == b, (n, p)


def _grid(cells_rc, texts, head_rows=1):
    """Words (one per cell) at (row, col) positions and logits that group them as given."""
    words = [
        {"box": [100.0 * c, 20.0 * r, 100.0 * c + 40, 20.0 * r + 10], "text": t, "line": r, "pos": 0, "n": 1}
        for (r, c), t in zip(cells_rc, texts)
    ]
    n = len(words)
    pair = np.full((3, n, n), -5.0, np.float32)
    for i in range(n):
        for j in range(n):
            pair[0, i, j] = 5 if i == j else -5
            pair[1, i, j] = 5 if cells_rc[i][0] == cells_rc[j][0] else -5
            pair[2, i, j] = 5 if cells_rc[i][1] == cells_rc[j][1] else -5
    tags = np.full((n, 2), -5.0, np.float32)
    for i, (r, _) in enumerate(cells_rc):
        if r < head_rows:
            tags[i, 0] = 5
    return words, pair, tags


def test_a_lone_unit_row_spans_the_table():
    # "(In millions)" alone in a body row, over the value columns: a section row
    rc = [(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)]
    texts = ["Item", "2023", "2024", "(In millions)", "Sales", "10", "12"]
    words, pair, tags = _grid(rc, texts)
    boxes = np.array([w["box"] for w in words], np.float32)
    html = table_html(decode(pair, tags, boxes, texts=texts), words)
    assert '<td colspan="3">(In millions)</td>' in html


def test_a_lone_label_or_number_keeps_its_column():
    # a lone row label in the first column stays a row label; a lone number stays in its column
    rc = [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2), (3, 0), (3, 1), (3, 2)]
    texts = ["Item", "2023", "2024", "Assets", "7", "Cash", "10", "12"]
    words, pair, tags = _grid(rc, texts)
    boxes = np.array([w["box"] for w in words], np.float32)
    html = table_html(decode(pair, tags, boxes, texts=texts), words)
    assert "colspan" not in html


def test_a_wrapped_label_belongs_to_its_own_row():
    # A label running to two lines right under the header belongs to its own row,
    # not to the header row above it
    boxes = np.array(
        [
            [0, 0, 60, 10],  # header
            [120, 0, 180, 10],  # header
            [0, 16, 90, 36],  # two-line label, just under the header
            [120, 22, 150, 32],  # its row's value
            [0, 45, 60, 55],  # next row
            [120, 45, 150, 55],
        ],
        np.float64,
    )
    texts = ["Item", "2025", "net profit of the parent company", "1", "Total assets", "2"]
    html = cells_html([[0], [1], [2], [3], [4], [5]], boxes, texts)
    assert "<tr><td>net profit of the parent company</td><td>1</td></tr>" in html
    assert "<tr><td>Total assets</td><td>2</td></tr>" in html


def test_a_wrapped_label_ending_at_a_column_start_is_two_cells():
    # A label that ends right at the next column's left edge, its neighbour on the
    # same line: two cells the model merged
    boxes = np.array(
        [
            [0, 0, 60, 10],
            [130, 0, 190, 10],
            [250, 0, 330, 10],
            [0, 20, 120, 30],  # the long first-column label
            [130, 20, 200, 30],  # its right neighbour, on 'names's left edge
            [250, 20, 330, 30],
            [130, 40, 200, 50],
            [250, 40, 330, 50],
        ],
        np.float64,
    )
    texts = ["Sponsor", "Name", "ABC Securities", "term of the sponsor's", "Address", "Beijing", "Signature", "Yang"]
    html = cells_html([[0], [1], [2], [3, 4], [5], [6], [7]], boxes, texts)
    assert "<td>term of the sponsor's</td><td>Address</td>" in html


def test_a_wide_label_does_not_swallow_its_neighbour():
    # A long label's box reaches into the next column's band (a header there can
    # be wider than the band its text needs): the label must not take a colspan
    # over its neighbour, or the writer skips that column and drops the value
    boxes = np.array(
        [
            [0, 0, 100, 10],  # header, column 0
            [180, 0, 280, 10],  # header, column 1
            [0, 20, 200, 30],  # long label, over column 1's band
            [210, 20, 280, 30],  # its value
        ],
        np.float64,
    )
    texts = ["Item", "Value", "a label too long for one column", "42"]
    html = cells_html([[0], [1], [2], [3]], boxes, texts)
    assert "<td>42</td>" in html


def test_leader_dots_end_no_cell():
    rc = [(0, 0), (0, 1), (1, 0), (1, 1)]
    texts = ["Item", "2024", "Total assets . . .", "12"]
    words, pair, tags = _grid(rc, texts)
    boxes = np.array([w["box"] for w in words], np.float32)
    html = table_html(decode(pair, tags, boxes, texts=texts), words)
    assert "<td>Total assets</td>" in html and "Inc." not in html


def test_split_features():
    from indra_ocr.table_rel import N_SPLIT, split_features

    # Map px = page px / 2, crop origin (0, 0).  A row separator at map rows 20-21
    # (page y 40-43) and a column separator at map columns 30-31 (page x 60-63).
    prob = np.zeros((3, 50, 60), np.float32)
    prob[0, 20:22, :] = 1.0
    prob[1, :, 30:32] = 1.0
    prob[2, :10, :] = 1.0  # header rows: page y < 20
    maps = (prob, 0.5, (0.0, 0.0))
    boxes = np.array(
        [
            [10, 28, 50, 38],  # above the row separator, left of the column separator
            [70, 46, 100, 56],  # below the row separator, right of the column separator
            [10, 2, 40, 12],  # a header word
        ],
        np.float32,
    )
    f = split_features(boxes, maps)
    assert f.shape == (3, N_SPLIT)
    assert f[0, 0] > 0.9 and f[0, 1] < 0.1  # separator below, none above
    assert f[0, 3] > 0.9 and f[0, 2] < 0.1  # separator right, none left
    assert f[1, 1] > 0.9 and f[1, 0] < 0.1
    assert f[1, 2] > 0.9 and f[1, 3] < 0.1
    assert (f[:2, 4] < 0.1).all() and (f[:2, 5] < 0.1).all()  # nothing through the words
    assert f[2, 6] > 0.9 and f[0, 6] < 0.1
    assert (f[:, 7] == 1).all()
    assert (split_features(boxes, None) == 0).all()  # no split model: absent
