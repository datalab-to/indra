"""Table structure from word relations (tsr_rel.onnx).

A table block's lines are cut into words (at the recognizer's space tokens).
Each word gets a feature vector (``word_features``): its box relative to the
table, simple text classes, the gaps to its neighbours on its line, and the ink
in thin strips just outside its box (ruling lines).  A small transformer over
the words predicts, for every pair, logits for "same cell", "same row" and
"same column", and per word logits for "header" and "outside the table".

With tsr_split.onnx (the TableNet separator model) and a tsr_rel.onnx that takes
more than ``N_FEAT_BASE`` features, each word also gets ``split_features``: how
strongly the model's row / column separator maps run next to and through the word,
and its header map.  tsr_rel can follow or overrule them.

``decode`` groups words into cells (average linkage on "same cell"), cells into
rows and columns (average linkage on the mean pair logits), orders them by
position, and gives a cell every row (column) whose words it overlaps
vertically (horizontally) by more than half: a spanning cell.
"""

from __future__ import annotations

import functools
import html
import math
import re
import unicodedata
import zlib

import cv2
import numpy as np

from .order_post import _cluster

N_FEAT_BASE = 33  # 26 word features + 7 ruling-line extents (a model that takes 26 reads the first 26)
N_SPLIT = 8  # split features (split_features), the last one = 1 when they are present
N_FEAT = N_FEAT_BASE + N_SPLIT
SAME_CELL, SAME_ROW, SAME_COL = 0, 1, 2
TAG_HEAD, TAG_OUT = 0, 1


def words_of(lines: list, is_space, decode) -> list[dict]:
    """Words from lines [(box [x0, y0, x1, y1], tokens)]; tokens are (token, x, ...).
    A word: {"box", "text", "line", "pos" (index in its line), "n" (words in its line)}."""
    out = []
    for li, (box, toks) in enumerate(lines):
        h = max(1.0, float(box[3] - box[1]))
        words, cur = [], []
        for tk in toks:
            if is_space(tk[0]):
                if cur:
                    words.append(cur)
                    cur = []
            else:
                cur.append(tk)
        if cur:
            words.append(cur)
        ws = []
        for w in words:
            xs = [float(t[1]) for t in w]
            text = decode(w).strip()
            if not text:
                continue
            x0 = max(float(box[0]), min(xs) - 0.35 * h)
            x1 = min(float(box[2]), max(xs) + 0.35 * h)
            if x1 <= x0:
                x0, x1 = min(xs) - 0.3 * h, max(xs) + 0.3 * h
            ws.append({"box": [x0, float(box[1]), x1, float(box[3])], "text": text, "line": li})
        for k, w in enumerate(ws):
            w["pos"], w["n"] = k, len(ws)
        out += ws
    return out


@functools.lru_cache(maxsize=8192)
def _char_class(c: str) -> tuple:
    """(digit, alpha, upper, punctuation, symbol, wide script) for one character."""
    cat = unicodedata.category(c)
    return (c.isdigit(), c.isalpha(), c.isupper(), cat[0] == "P", cat[0] == "S", ord(c) > 0x2E7F)


def _text_features(texts: list[str]) -> np.ndarray:
    """[N, 6]: shares of digits, letters, capitals, punctuation and symbols, and
    whether any character is of a wide script, per text (all texts in one pass)."""
    lens = np.array([max(len(t), 1) for t in texts])
    chars = [_char_class(c) for t in texts for c in (t or " ")]
    cls = np.array(chars, np.float32).reshape(-1, 6)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    tot = np.add.reduceat(cls, starts, axis=0)
    out = tot / lens[:, None]
    out[:, 5] = tot[:, 5] > 0
    return out


def word_features(words: list[dict], gray: np.ndarray, bbox) -> np.ndarray:
    """[N, N_FEAT_BASE] float32 features for the words of one table (``bbox`` = the
    table's box on ``gray``, the page)."""
    n = len(words)
    f = np.zeros((n, N_FEAT_BASE), np.float32)
    if n == 0:
        return f
    boxes = np.array([w["box"] for w in words], np.float32)
    x0, y0, x1, y1 = boxes.T
    hs = np.maximum(y1 - y0, 1.0)
    hm = float(np.median(y1 - y0))
    tx0, ty0, tx1, ty1 = (float(v) for v in bbox)
    tw, th = max(tx1 - tx0, 1.0), max(ty1 - ty0, 1.0)
    f[:, 0] = (x0 - tx0) / tw
    f[:, 1] = (y0 - ty0) / th
    f[:, 2] = (x1 - tx0) / tw
    f[:, 3] = (y1 - ty0) / th
    f[:, 4] = (x0 - tx0) / hm / 50.0
    f[:, 5] = (y0 - ty0) / hm / 50.0
    f[:, 6] = (x1 - x0) / hm / 20.0
    f[:, 7] = hs / hm
    for i, w in enumerate(words):
        t = w["text"]
        f[i, 8] = np.log1p(len(t)) / 4.0
        f[i, 14] = float(t[:1].isdigit())
        f[i, 15] = float(t.endswith(":"))
        f[i, 16] = w["pos"] / max(w["n"] - 1, 1)
        f[i, 17] = float(w["pos"] == 0)
        f[i, 18] = float(w["pos"] == w["n"] - 1)
    # digits, letters, capitals, punctuation, symbols; any wide-script (CJK...) character
    tf = _text_features([w["text"] for w in words])
    f[:, 9:14] = tf[:, :5]
    f[:, 25] = tf[:, 5]
    by_line: dict = {}
    for i, w in enumerate(words):
        by_line.setdefault(w["line"], []).append(i)
    for line in by_line.values():
        for k, i in enumerate(line):
            h = hs[i]
            f[i, 19] = min((x0[i] - x1[line[k - 1]]) / h, 10.0) / 10.0 if k > 0 else 1.0
            f[i, 20] = min((x0[line[k + 1]] - x1[i]) / h, 10.0) / 10.0 if k + 1 < len(line) else 1.0
    # Ink in thin strips just outside each word (ruling lines, cell borders): an
    # integral image of the table's region of the page.
    H, W = gray.shape
    m = float(hs.max())  # the strips reach 0.5 h beyond a word
    ox, oy = max(int(tx0 - m), 0), max(int(ty0 - m), 0)
    ex, ey = min(int(np.ceil(tx1 + m)), W), min(int(np.ceil(ty1 + m)), H)
    if ex > ox and ey > oy:
        ink = (gray[oy:ey, ox:ex] < 128).astype(np.uint8)
        ii = cv2.integral(ink)  # [h + 1, w + 1] int32
        rh, rw = ink.shape

        def strip(a0, b0, a1, b1):  # x0, y0, x1, y1 arrays in page px
            c0 = np.clip(np.floor(a0 - ox), 0, rw).astype(int)
            r0 = np.clip(np.floor(b0 - oy), 0, rh).astype(int)
            c1 = np.clip(np.ceil(a1 - ox), 0, rw).astype(int)
            r1 = np.clip(np.ceil(b1 - oy), 0, rh).astype(int)
            area = (c1 - c0) * (r1 - r0)
            tot = ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]
            return np.where(area > 0, tot / np.maximum(area, 1), 0.0)

        f[:, 21] = strip(x0, y0 - 0.45 * hs, x1, y0 - 0.1 * hs)
        f[:, 22] = strip(x0, y1 + 0.1 * hs, x1, y1 + 0.45 * hs)
        f[:, 23] = strip(x0 - 0.5 * hs, y0, x0 - 0.1 * hs, y1)
        f[:, 24] = strip(x1 + 0.1 * hs, y0, x1 + 0.5 * hs, y1)
        # A horizontal rule right under (over) a word, and how far it reaches past
        # the word to the left and right: a group header's rule spans the columns
        # it heads.  [26] below, [27] its left reach, [28] its right reach (table
        # widths); [29..31] the same above; [32] = 1: these features are present.
        f[:, 32] = 1.0
        for k, (lo, hi) in enumerate(((y1 + 0.05 * hs, y1 + 0.8 * hs), (y0 - 0.8 * hs, y0 - 0.05 * hs))):
            for i in range(n):
                r = _rule(ink, ox, oy, float(x0[i]), float(x1[i]), float(lo[i]), float(hi[i]))
                if r is not None:
                    f[i, 26 + 3 * k] = 1.0
                    f[i, 27 + 3 * k] = min((float(x0[i]) - r[0]) / tw, 1.0)
                    f[i, 28 + 3 * k] = min((r[1] - float(x1[i])) / tw, 1.0)
    return f


SPLIT_TEXT_PX = 12.0  # tsr_split.onnx's line height (it expects text 8-16 px high)
SPLIT_MAX_SIDE = 1024
SPLIT_PAD_TO = 32
SPLIT_STRIDE = 2  # map px per crop px: 1/2


def split_maps(sess, gray: np.ndarray, bbox, line_h: float):
    """tsr_split.onnx on the table at ``bbox`` (page px) of ``gray``: (prob [3, h, w] =
    row separator, column separator, header at 1/2 of the scaled crop; the map px per
    page px; the crop origin (x, y) in page px), or None for a crop too small to read."""
    H, W = gray.shape
    m = 1.5 * line_h  # context around the lines (the model expects 0.3-2 text heights)
    x0, y0 = max(int(bbox[0] - m), 0), max(int(bbox[1] - m), 0)
    x1, y1 = min(int(np.ceil(bbox[2] + m)), W), min(int(np.ceil(bbox[3] + m)), H)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    s = SPLIT_TEXT_PX / max(line_h, 1.0)
    if max(crop.shape) * s > SPLIT_MAX_SIDE:
        s = SPLIT_MAX_SIDE / max(crop.shape)
    nh, nw = max(1, round(crop.shape[0] * s)), max(1, round(crop.shape[1] * s))
    if min(nh, nw) < 16:
        return None
    small = cv2.resize(
        np.ascontiguousarray(crop), (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    )
    ph, pw = -(-nh // SPLIT_PAD_TO) * SPLIT_PAD_TO, -(-nw // SPLIT_PAD_TO) * SPLIT_PAD_TO
    x = np.zeros((1, 1, ph, pw), np.float32)
    x[0, 0, :nh, :nw] = 1.0 - small.astype(np.float32) / 255.0
    logits = sess.run(None, {"x": x})[0][0]
    vh, vw = -(-nh // SPLIT_STRIDE), -(-nw // SPLIT_STRIDE)
    prob = 1.0 / (1.0 + np.exp(-logits[:, :vh, :vw]))
    return prob.astype(np.float32), s / SPLIT_STRIDE, (float(x0), float(y0))


def _band(P: np.ndarray, a, b, lo, hi, k: int = 8) -> np.ndarray:
    """Per word: the highest mean of ``P`` [h, w] over columns [lo, hi) on any of
    ``k`` rows spread over [a, b] (map px arrays); 0 where the area is off the map."""
    h, w = P.shape
    cs = np.concatenate([np.zeros((h, 1), np.float32), np.cumsum(P, 1, dtype=np.float32)], 1)
    c0 = np.clip(np.floor(lo), 0, w).astype(int)
    c1 = np.clip(np.ceil(hi), 0, w).astype(int)
    c1 = np.maximum(c1, np.minimum(c0 + 1, w))
    t = (np.arange(k, dtype=np.float32) + 0.5) / k
    r = np.floor(a[:, None] + (b - a)[:, None] * t[None])  # [N, k]
    ok = (r >= 0) & (r < h) & (c1 > c0)[:, None]
    r = np.clip(r, 0, h - 1).astype(int)
    v = (cs[r, c1[:, None]] - cs[r, c0[:, None]]) / np.maximum(c1 - c0, 1)[:, None]
    return np.where(ok, v, 0.0).max(1)


def split_features(boxes: np.ndarray, maps) -> np.ndarray:
    """[N, N_SPLIT] features of word ``boxes`` [N, 4] (page px) from ``split_maps``:
    row separator just below / above the word, column separator just left / right
    of it, a row (column) separator through it, its mean header probability, and 1
    (present).  Zeros (absent) when ``maps`` is None."""
    f = np.zeros((len(boxes), N_SPLIT), np.float32)
    if maps is None or len(boxes) == 0:
        return f
    prob, s, (ox, oy) = maps
    pr, pc, ph = prob
    x0, x1 = (boxes[:, 0] - ox) * s, (boxes[:, 2] - ox) * s
    y0, y1 = (boxes[:, 1] - oy) * s, (boxes[:, 3] - oy) * s
    h = np.maximum(y1 - y0, 1.0)
    f[:, 0] = _band(pr, y1 - 0.1 * h, y1 + 0.8 * h, x0, x1)
    f[:, 1] = _band(pr, y0 - 0.8 * h, y0 + 0.1 * h, x0, x1)
    f[:, 2] = _band(pc.T, x0 - 2.0 * h, x0 + 0.1 * h, y0, y1)
    f[:, 3] = _band(pc.T, x1 - 0.1 * h, x1 + 2.0 * h, y0, y1)
    f[:, 4] = _band(pr, y0 + 0.3 * h, y1 - 0.3 * h, x0, x1)
    f[:, 5] = _band(pc.T, x0 + 0.2 * h, x1 - 0.2 * h, y0, y1)
    f[:, 6] = _band(ph, y0, y1, x0, x1)
    f[:, 7] = 1.0
    return f


def _rule(ink: np.ndarray, ox: int, oy: int, x0: float, x1: float, y0: float, y1: float):
    """The horizontal ink rule in rows [y0, y1) (page px) that covers at least 70%
    of the word's span [x0, x1): its ink run's (left, right) in page px, or None.
    ``ink`` is the 0/1 crop of the page from (ox, oy)."""
    rh, rw = ink.shape
    r0, r1 = max(int(y0) - oy, 0), min(int(np.ceil(y1)) - oy, rh)
    c0, c1 = max(int(x0) - ox, 0), min(int(np.ceil(x1)) - ox, rw)
    if r1 <= r0 or c1 - c0 < 2:
        return None
    cover = ink[r0:r1, c0:c1].mean(1)
    j = int(cover.argmax())
    if cover[j] < 0.7:
        return None
    row = ink[max(r0 + j - 1, 0) : r0 + j + 2].max(0)  # a 3-row band: rules can be 1-2 px thick or slanted
    mid = (c0 + c1) // 2
    if not row[mid]:
        return None
    gaps = np.flatnonzero(row == 0)
    left = gaps[gaps < mid]
    right = gaps[gaps > mid]
    a = (int(left[-1]) + 1) if len(left) else 0
    b = int(right[0]) if len(right) else rw
    return float(a + ox), float(b + ox)


def norm_boxes(boxes: np.ndarray) -> np.ndarray:
    """Word boxes [N, 4] (px) -> median line heights from the top-left word corner
    (the model's box input)."""
    if len(boxes) == 0:
        return boxes.astype(np.float32)
    hm = max(float(np.median(boxes[:, 3] - boxes[:, 1])), 1.0)
    o = np.array([boxes[:, 0].min(), boxes[:, 1].min()] * 2, np.float32)
    return ((boxes - o) / hm).astype(np.float32)


def read_table(
    sess,
    gray: np.ndarray,
    lines: list,
    is_space,
    decode_text,
    mirror: bool = False,
    col_spans: str = "geo",
    split_sess=None,
    refine_sess=None,
) -> str | None:
    """HTML for one table block from tsr_rel.onnx.  ``lines``: [(box, tokens)]
    on ``gray``; ``mirror``: a right-to-left page (columns from the right);
    ``col_spans``: how a cell's columns are found (``decode``); ``split_sess``:
    tsr_split.onnx, run only for a tsr_rel.onnx that takes the split features;
    ``refine_sess``: tsr_refine.onnx, which refines the word tags of tsr_rel."""
    words = words_of(lines, is_space, decode_text)
    if len(words) < 2:
        return None
    if mirror:
        # A right-to-left table is read mirrored, as a left-to-right one: the page
        # and the word boxes are flipped before the features (positions, gaps,
        # ink strips, rules), so features and boxes agree.
        W = float(gray.shape[1])
        gray = gray[:, ::-1]
        for w in words:
            x0, y0, x1, y1 = w["box"]
            w["box"] = [W - x1, y0, W - x0, y1]
        words.sort(key=lambda w: (w["line"], w["box"][0]))
        by_line: dict = {}
        for w in words:
            by_line.setdefault(w["line"], []).append(w)
        for ws in by_line.values():
            for k, w in enumerate(ws):
                w["pos"], w["n"] = k, len(ws)
    boxes = np.array([w["box"] for w in words], np.float32)
    bbox = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
    feat = word_features(words, gray, bbox)
    nf = sess.get_inputs()[0].shape[-1]
    if isinstance(nf, int) and nf > N_FEAT_BASE:
        line_h = float(np.median(boxes[:, 3] - boxes[:, 1]))
        maps = split_maps(split_sess, gray, bbox, line_h) if split_sess is not None else None
        feat = np.concatenate([feat, split_features(boxes, maps)], 1)
    if isinstance(nf, int) and nf < feat.shape[1]:  # a model that takes fewer features
        feat = feat[:, :nf]
    inp = {"feat": feat[None], "box": norm_boxes(boxes)[None]}
    n_vocab = _n_vocab(sess)
    if n_vocab:  # a model with a word-text embedding
        inp["wid"] = word_ids([w["text"] for w in words], n_vocab)[None]
    pair, tags = sess.run(None, inp)
    if refine_sess is not None:
        # tsr_refine.onnx: a small relational model over the tsr_rel logits.  It changes
        # only the word tags (header, outside the table).  Rows, columns and cells
        # keep the tsr_rel relations.
        tags = refine_sess.run(
            ["tag_out"],
            {
                "pair": pair,
                "tag": tags,
                "box": inp["box"],
                "feat": inp["feat"],
                "tfeat": refine_text_features([w["text"] for w in words])[None],
            },
        )[0]
    struct = decode(pair[0], tags[0], boxes, col_spans, [w["text"] for w in words])
    return table_html(struct, words) if struct else None


def word_ids(texts: list[str], n_vocab: int) -> np.ndarray:
    """Hashed word ids in [1, n_vocab) (lower case NFKC, every digit 0).  The model
    expects exactly this hash.  A word with a letter outside the Latin script gets
    id 0 (no word embedding)."""
    out = np.zeros(len(texts), np.int64)
    for i, t in enumerate(texts):
        s = re.sub(r"\d", "0", unicodedata.normalize("NFKC", t).lower())
        if any(c.isalpha() and not unicodedata.name(c, "").startswith("LATIN") for c in s):
            continue
        out[i] = zlib.crc32(s.encode("utf-8")) % (n_vocab - 1) + 1
    return out


def _n_vocab(sess) -> int:
    """The word-id buckets of a model with a "wid" input (its "n_vocab" metadata), else 0."""
    if len(sess.get_inputs()) < 3:
        return 0
    return int(sess.get_modelmeta().custom_metadata_map.get("n_vocab", 0))


def _cluster_sparse(L: np.ndarray) -> list[list[int]]:
    """``_cluster`` (average linkage, merge while the mean logit is positive) run on
    each connected component of the positive pairs: two groups with no positive
    pair between them never merge, so the result is the same, at a fraction of
    the cost when positives are sparse (same cell)."""
    n = L.shape[0]
    ii, jj = np.nonzero(np.triu(L > 0, 1))
    # Connected components: each item takes the smallest label of its neighbours
    # until nothing changes, so a component's label is its smallest member.
    lab = np.arange(n)
    while len(ii):
        new = lab.copy()
        np.minimum.at(new, ii, lab[jj])
        np.minimum.at(new, jj, lab[ii])
        new = new[new]
        if (new == lab).all():
            break
        lab = new
    out = []
    for c in np.unique(lab):  # components in the order of their smallest member
        members = np.nonzero(lab == c)[0].tolist()
        if len(members) <= 2:  # one item, or two joined by a positive pair: one group
            out.append(members)
            continue
        sub = L[np.ix_(members, members)]
        out += [[members[k] for k in g] for g in _cluster(sub)]
    return out


def _mean_logit(L: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """Mean logit between every two groups of items, by segment sums (no BLAS:
    a matrix product here starts a thread pool in every worker process)."""
    order = np.concatenate([np.asarray(g, np.int64) for g in groups])
    size = np.array([len(g) for g in groups], np.float64)
    starts = np.concatenate([[0], np.cumsum(size)[:-1]]).astype(np.int64)
    S = L.astype(np.float64)[np.ix_(order, order)]
    S = np.add.reduceat(np.add.reduceat(S, starts, axis=0), starts, axis=1)
    out = (S / np.outer(size, size)).astype(np.float32)
    np.fill_diagonal(out, 0.0)
    return out


def decode(
    pair: np.ndarray, tags: np.ndarray, boxes: np.ndarray, col_spans: str = "geo", texts: list | None = None
) -> dict | None:
    """Cells, rows and columns from the model's outputs for N words.
    pair [3, N, N] logits, tags [N, 2] logits, boxes [N, 4].
    col_spans "geo": a cell spans the columns whose band its words cover by more
    than half; "rel": the run of columns around its own with which its words
    have a positive mean "same column" logit (a narrow group header over the
    columns it heads); "rel2": "rel" for columns, and the same for rows with the
    "same row" logits (a stub header over the header rows it spans).
    Returns {"cells": [(r0, r1, c0, c1, [word ids])], "n_head", "outside": [word ids]}."""
    n = len(boxes)
    if n == 0:
        return None
    outside = [i for i in range(n) if tags[i, TAG_OUT] > 0]
    keep = [i for i in range(n) if tags[i, TAG_OUT] <= 0]
    if len(keep) < 2:
        return None
    sub = np.ix_(keep, keep)
    cells = [[keep[k] for k in c] for c in _cluster_sparse(pair[SAME_CELL][sub])]
    m = len(cells)
    cb = np.array([[boxes[c, 0].min(), boxes[c, 1].min(), boxes[c, 2].max(), boxes[c, 3].max()] for c in cells])
    rows = sorted(_cluster_sparse(_mean_logit(pair[SAME_ROW], cells)), key=lambda g: float(np.median((cb[g, 1] + cb[g, 3]) / 2)))
    cols = sorted(_cluster_sparse(_mean_logit(pair[SAME_COL], cells)), key=lambda g: float(np.median((cb[g, 0] + cb[g, 2]) / 2)))
    # Each row's (column's) band: the median extent of its cells.
    rband = [(float(np.median(cb[g, 1])), float(np.median(cb[g, 3]))) for g in rows]
    cband = [(float(np.median(cb[g, 0])), float(np.median(cb[g, 2]))) for g in cols]
    row_of = {c: r for r, g in enumerate(rows) for c in g}
    col_of = {c: k for k, g in enumerate(cols) for c in g}

    def span(lo, hi, bands, own):
        """Bands the extent [lo, hi] covers over half of, always including its own."""
        out = [k for k, (a, b) in enumerate(bands) if min(hi, b) - max(lo, a) > 0.5 * (b - a)]
        out = sorted(set(out) | {own})
        return out[0], out[-1]

    def positive(ml, groups):
        """[cells, groups]: True where a cell's mean logit with a group's cells is
        positive (one mean per group over all cells at once)."""
        return np.stack([ml[:, g].mean(axis=1) for g in groups], axis=1) > 0

    if col_spans in ("rel", "rel2"):
        pc = positive(_mean_logit(pair[SAME_COL], cells), cols)
    if col_spans == "rel2":
        pr = positive(_mean_logit(pair[SAME_ROW], cells), rows)

    def rel_span(c, own, P):
        """The run of groups (columns or rows) around ``own`` with a positive mean logit with cell c."""
        pos = P[c].copy()
        pos[own] = True  # its own group (the only one that holds c)
        lo = hi = own
        while lo > 0 and pos[lo - 1]:
            lo -= 1
        while hi + 1 < len(pos) and pos[hi + 1]:
            hi += 1
        return lo, hi

    out = []
    for c in range(m):
        if col_spans == "rel2":
            r0, r1 = rel_span(c, row_of[c], pr)
        else:
            r0, r1 = span(cb[c, 1], cb[c, 3], rband, row_of[c])
        if col_spans in ("rel", "rel2"):
            c0, c1 = rel_span(c, col_of[c], pc)
        else:
            c0, c1 = span(cb[c, 0], cb[c, 2], cband, col_of[c])
        out.append((r0, r1, c0, c1, cells[c]))
    head = np.zeros(len(rows))
    for c, (r0, r1, _, _, ws) in enumerate(out):
        head[r0] = max(head[r0], float(tags[ws, TAG_HEAD].mean()))
    n_head = 0
    while n_head < len(rows) - 1 and head[n_head] > 0:
        n_head += 1
    # A body row that holds one cell with words, away from the first column (a unit
    # row such as "(In millions)" or a centered section title), spans the table's
    # width, as a section row does; a lone label in the first column stays a row
    # label, and a lone number stays in its column.  ``texts``: the words' texts.
    C = len(cols)
    in_row: dict = {}
    for k, (r0, r1, _, _, _) in enumerate(out):
        for r in range(r0, r1 + 1):
            in_row.setdefault(r, []).append(k)
    for r, ks in in_row.items():
        if r < n_head or len(ks) != 1 or C < 3:
            continue
        r0, r1, c0, _, ws = out[ks[0]]
        if r0 == r1 and c0 > 0 and (texts is None or any(ch.isalpha() for i in ws for ch in texts[i])):
            out[ks[0]] = (r0, r1, 0, C - 1, ws)
    return {"cells": out, "n_head": n_head, "outside": outside, "n_rows": len(rows), "n_cols": len(cols)}


LEADER_DOTS = re.compile(r"(\s+[.·…]+)+$")  # space-separated dot runs that end a cell


def _cell_text(g: list[int], boxes: np.ndarray, texts: list[str]) -> str:
    t = " ".join(texts[i] for i in sorted(g, key=lambda i: (boxes[i, 1], boxes[i, 0])))
    return LEADER_DOTS.sub("", t) or t


def cells_html(
    groups: list[list[int]], boxes: np.ndarray, texts: list[str], n_head: int = 1
) -> str:
    """An HTML table rebuilt from the cells the model gives as word groups.

    tsr_rel tells which words share a cell; on a PDF text layer its rows and
    columns are less reliable (a merged header label is centred between two
    rows and gets a row of its own, a wrapped label is split).  So the rows and
    columns come from the geometry here: the body rows give the column bands,
    the single-line cells give the rows, and a row squeezed between two rows
    whose columns are empty above and below is the centred label of a header
    cell that spans those two rows.  A cell keeps a rowspan or colspan when its
    box covers several rows or columns."""
    if not groups:
        return "<table>\n</table>"
    boxes = np.asarray(boxes, np.float64)
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    # Split a cell whose words are far apart in x: two cells the model merged.
    split = []
    for g in groups:
        g = sorted(g, key=lambda i: (boxes[i, 0], boxes[i, 1]))
        cur = [g[0]]
        for i in g[1:]:
            if boxes[i, 0] - boxes[cur[-1], 2] > 2.0 * h:
                split.append(cur)
                cur = [i]
            else:
                cur.append(i)
        split.append(cur)

    def boxes_of(groups):
        return np.array(
            [
                [boxes[g, 0].min(), boxes[g, 1].min(), boxes[g, 2].max(), boxes[g, 3].max()]
                for g in groups
            ]
        )

    b = boxes_of(split)
    # A word on the same line that starts on another cell's left edge is another
    # cell: a wrapped label ends right at a column start, and the model merged it
    # with its right neighbour.
    tol = 0.15 * h
    starts = b[:, 0].tolist()
    merged = []
    for k, g in enumerate(split):
        g = sorted(g, key=lambda i: (boxes[i, 0], boxes[i, 1]))
        cur = [g[0]]
        for i in g[1:]:
            other = [v for j, v in enumerate(starts) if j != k]
            same_line = abs((boxes[i, 1] + boxes[i, 3]) - (boxes[cur[-1], 1] + boxes[cur[-1], 3])) < h
            if same_line and other and min(abs(boxes[i, 0] - v) for v in other) <= tol:
                merged.append(cur)
                cur = [i]
            else:
                cur.append(i)
        merged.append(cur)
    split = merged
    b = boxes_of(split)
    n = len(split)
    hh = b[:, 3] - b[:, 1]
    tall = hh > 1.5 * h
    cy = (b[:, 1] + b[:, 3]) / 2
    # Rows from the single-line cells; a merged label centres between rows.
    rows: list[list[int]] = []
    for i in np.argsort(cy):
        if tall[i]:
            continue
        if rows and abs(cy[i] - np.mean(cy[rows[-1]])) < 0.5 * h:
            rows[-1].append(int(i))
        else:
            rows.append([int(i)])
    row_of = {i: r for r, g in enumerate(rows) for i in g}
    rbands = [(b[g, 1].min(), b[g, 3].max()) for g in rows]
    for i in np.nonzero(tall)[0]:
        # A cell taller than a line whose box sits within a single row's band is a
        # wrapped label of that row, however far its lines run: it belongs to the
        # row it is centred on, not to the row above it.  One that covers two rows
        # (a header cell split over two lines) is joined to the cell above it.
        cov = [k for k, (a, z) in enumerate(rbands) if min(b[i, 3], z) - max(b[i, 1], a) > 0]
        if len(cov) == 1:
            row_of[int(i)] = cov[0]
            continue
        best, bestd = None, 0.0
        for j, r in row_of.items():
            if tall[j] or cy[j] >= cy[i]:
                continue
            ov = min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0])
            gap = max(b[i, 1], b[j, 1]) - min(b[i, 3], b[j, 3])
            if ov > 0.5 * min(b[i, 2] - b[i, 0], b[j, 2] - b[j, 0]) and gap < 0.75 * h:
                if best is None or cy[i] - cy[j] < bestd:
                    best, bestd = r, cy[i] - cy[j]
        if best is None:
            ov = [(min(b[i, 3], z) - max(b[i, 1], a)) / max(z - a, 1.0) for a, z in rbands]
            best = int(np.argmax(ov))
        row_of[int(i)] = best
    # Column bands from the row with the most cells (the body rows).
    body = max(
        range(len(rows)),
        key=lambda r: (len(rows[r]), -np.median(b[rows[r], 2] - b[rows[r], 0])),
    )
    band = sorted(rows[body], key=lambda i: b[i, 0])
    C = len(band)

    def cols_of(i: int) -> tuple[int, int]:
        cover = [
            k
            for k in range(C)
            if min(b[i, 2], b[band[k], 2]) - max(b[i, 0], b[band[k], 0]) > 0
        ]
        if cover:
            return cover[0], cover[-1]
        # No band under the cell: a header centred over the gap between two
        # columns (a group header) spans them both.
        left = [k for k in range(C) if b[band[k], 2] <= b[i, 0]]
        right = [k for k in range(C) if b[band[k], 0] >= b[i, 2]]
        if left and right:
            return left[-1], right[0]
        k = int(
            np.argmin(
                [
                    abs((b[i, 0] + b[i, 2]) / 2 - (b[band[k], 0] + b[band[k], 2]) / 2)
                    for k in range(C)
                ]
            )
        )
        return k, k

    cell_cols = [cols_of(i) for i in range(n)]
    # A body row with one cell away from the first column (a unit row such as
    # "(In millions)") spans the table's width.
    for r, g in enumerate(rows):
        if (
            r >= n_head
            and len(g) == 1
            and cell_cols[g[0]][0] > 0
            and any(ch.isalpha() for i in g for ch in texts[i])
        ):
            cell_cols[g[0]] = (0, C - 1)
    # A row squeezed between two rows, its columns empty above and below: the
    # centred labels of header cells that span those two rows.
    phantom = set()
    for r in range(1, len(rows) - 1):
        members = [i for i in range(n) if row_of[i] == r]
        if not members:
            continue
        up = {
            c
            for i in range(n)
            if row_of[i] == r - 1
            for c in range(cell_cols[i][0], cell_cols[i][1] + 1)
        }
        dn = {
            c
            for i in range(n)
            if row_of[i] == r + 1
            for c in range(cell_cols[i][0], cell_cols[i][1] + 1)
        }
        if (
            rbands[r][0] - rbands[r - 1][1] < 0.5 * h
            and rbands[r + 1][0] - rbands[r][1] < 0.5 * h
            and all(
                c not in up and c not in dn
                for i in members
                for c in range(cell_cols[i][0], cell_cols[i][1] + 1)
            )
        ):
            phantom.add(r)
    # A row holding one lone label under the same column's line: the tail of a
    # wrapped cell (its first line is in the row above).
    snap = dict(row_of)
    lonely = set()
    for r in range(1, len(rows)):
        members = [i for i in range(n) if snap[i] == r and not tall[i]]
        if len(members) != 1 or cell_cols[members[0]][0] != 0 or r in phantom:
            continue
        i = members[0]
        if rbands[r][0] - rbands[r - 1][1] >= 0.75 * h:
            continue
        if any(
            snap[j] == r - 1
            and min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0])
            > 0.5 * min(b[i, 2] - b[i, 0], b[j, 2] - b[j, 0])
            for j in range(n)
        ):
            lonely.add(r)
    for r in sorted(lonely):
        for i in range(n):
            if snap[i] == r:
                row_of[i] = r - 1
    keep = [r for r in range(len(rows)) if r not in phantom and r not in lonely]
    new = {r: k for k, r in enumerate(keep)}
    grid: dict[tuple[int, int], tuple[int, int, str]] = {}
    order = sorted(
        range(n), key=lambda i: (new.get(row_of[i], 0), cell_cols[i][0], b[i, 1])
    )
    for i in order:
        r = row_of[i]
        cov = [
            k
            for k in range(len(rbands))
            if min(b[i, 3], rbands[k][1]) - max(b[i, 1], rbands[k][0]) > 0
        ]
        if r in phantom:
            r0, r1 = r - 1, r + 1
        elif tall[i] and cov:
            r0, r1 = r, max(r, cov[-1])
        else:
            r0 = r1 = r
        c0, c1 = cell_cols[i]
        covered_rows = [new[x] for x in range(r0, r1 + 1) if x in new]
        rs = covered_rows[-1] - covered_rows[0] + 1 if covered_rows else 1
        key = (new.get(r0, 0), c0)
        text = _cell_text(split[i], boxes, texts)
        if key in grid:
            prev = grid[key]
            grid[key] = (max(prev[0], rs), max(prev[1], c1 - c0 + 1), prev[2] + " " + text)
        else:
            grid[key] = (rs, c1 - c0 + 1, text)
    # A colspan must not swallow a cell of the same row: the writer skips the
    # columns a cell spans, so a cell it covers would never be written.
    for (r, c0), (rs, cs, text) in list(grid.items()):
        for c in range(c0 + 1, c0 + cs):
            if (r, c) in grid:
                grid[(r, c0)] = (rs, c - c0, text)
                break
    nrows = len(keep)
    covered = np.zeros((nrows, C), bool)
    for (r, c0), (rs, cs, _) in grid.items():
        covered[r : r + rs, c0 : c0 + cs] = True
    out = ["<table>"]
    for r in range(nrows):
        tag = "th" if r < n_head else "td"
        tds, c = [], 0
        while c < C:
            item = grid.get((r, c))
            if item is not None:
                rs, cs, text = item
                attr = (f' rowspan="{rs}"' if rs > 1 else "") + (
                    f' colspan="{cs}"' if cs > 1 else ""
                )
                tds.append(f"<{tag}{attr}>{html.escape(text, quote=False)}</{tag}>")
                c += cs
            else:
                if not covered[r, c]:
                    tds.append(f"<{tag}></{tag}>")
                c += 1
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def table_html(struct: dict, words: list[dict]) -> str | None:
    """HTML for a decoded table; words outside it come before it as text.

    The model's cells (which words share a cell) are laid out by their
    geometry (``cells_html``), so merged header cells keep their rowspan and
    colspan on a PDF text layer."""
    R, C = struct["n_rows"], struct["n_cols"]
    if R < 2 and C < 2:
        return None
    boxes = np.array([w["box"] for w in words], np.float32)
    texts = [w["text"] for w in words]
    body = cells_html([list(c[4]) for c in struct["cells"]], boxes, texts, struct["n_head"])
    pre = _cell_text(struct["outside"], boxes, texts) if struct["outside"] else ""
    return (pre + "\n\n" if pre else "") + body


# Word text features of the tsr_refine.onnx refiner.  The model expects exactly
# these features.
N_REFINE_TEXT = 13
_REF_NUM = re.compile(r"^[(\[]?[-+−–]?[$€£¥]?[(\[]?\d[\d,.\s]*[%)\]]*\*{0,3}$")
_REF_DASH = {"-", "–", "—", "−", "n/a", "na", "nd", "n.a.", "nr", "…", "...", "·"}
_REF_MARK = {"*", "**", "***", "†", "‡", "§", "¶", "#"}


def refine_text_features(texts: list[str]) -> np.ndarray:
    """[N, N_REFINE_TEXT] content features of each word for tsr_refine.onnx:
    has a digit, digit share, letter share, number form, first letter upper, all upper,
    length, ends with ':', has '%', has a currency sign, has a bracket, a dash or n/a,
    a footnote mark."""
    out = np.zeros((len(texts), N_REFINE_TEXT), np.float32)
    for i, t in enumerate(texts):
        s = unicodedata.normalize("NFKC", t).strip()
        if not s:
            continue
        n = len(s)
        nd = sum(c.isdigit() for c in s)
        na = sum(c.isalpha() for c in s)
        letters = [c for c in s if c.isalpha()]
        out[i, 0] = float(nd > 0)
        out[i, 1] = nd / n
        out[i, 2] = na / n
        out[i, 3] = float(bool(_REF_NUM.match(s)))
        out[i, 4] = float(bool(letters) and letters[0].isupper())
        out[i, 5] = float(len(letters) >= 2 and all(c.isupper() for c in letters))
        out[i, 6] = math.log1p(n) / 3.0
        out[i, 7] = float(s.endswith(":"))
        out[i, 8] = float("%" in s)
        out[i, 9] = float(any(c in "$€£¥" for c in s))
        out[i, 10] = float(any(c in "()[]" for c in s))
        out[i, 11] = float(s.lower() in _REF_DASH)
        out[i, 12] = float(s in _REF_MARK or (n == 1 and s.isalpha() and s.islower()))
    return out
