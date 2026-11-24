"""False-table verifier: a logistic model (table_verify.json in the models folder, optional) on features of every
block the layout step classes Table.  A block the model calls a false table (P >= the model's threshold) gets its best
non-Table class instead, so it is written as text, a header, a list and so on.

Features (``block_features``): the table as written (rows, columns, cells, fill, words per cell, numeric cells, a
two-column key-value pattern), the block text (math tokens, whole-formula lines, sentence ends, characters per line),
the line geometry (height spread, rows by vertical centre, lines per row, gaps, distinct left / right edges, width
spread), the block size on the page, the Table blocks on the page, ruling-line pixels in the block crop, the order
model's class probabilities, whether the table heads were plausible and whether tsr_rel.onnx read the table.
"""


from __future__ import annotations

import html as _html
import math
import re

import numpy as np

_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_TD = re.compile(r"<t[dh]([^>]*)>(.*?)</t[dh]>", re.S)
_SPAN = re.compile(r'colspan="(\d+)"')
_TAG = re.compile(r"<[^>]+>")
_NUM = re.compile(r"^[\s$€£¥%()+\-–.,:/0-9]+$")
_MATH = re.compile(r"\$|\\[A-Za-z]+|[=+×÷±∑∫√∞≤≥≠≈^_{}]")
_WHOLE_MATH = re.compile(r"^\$[^$]+\$$")
_SENT = re.compile(r"[.!?。！？]$")


def parse_table(t: str) -> list[list[str]]:
    rows = []
    for r in _TR.findall(t):
        cells = []
        for attr, body in _TD.findall(r):
            span = int(_SPAN.search(attr).group(1)) if _SPAN.search(attr) else 1
            txt = _html.unescape(_TAG.sub(" ", body)).strip()
            cells += [txt] + [""] * (span - 1)
        rows.append(cells)
    return rows


def _any_run(ink: np.ndarray, length: int) -> np.ndarray:
    """Per row: does a morphological open of ``ink`` with a (length x 1) rectangle keep a pixel?  The same result as
    cv2.morphologyEx(MORPH_OPEN) with OpenCV's defaults (anchor at the centre, the border counts as ink for the
    erosion): a row keeps a pixel when ``length`` consecutive pixels are ink, when its first b + 1 or last a + 1
    pixels are ink (a = length // 2, b = length - 1 - a), or when the whole row is ink."""
    H, W = ink.shape
    a = length // 2
    b = length - 1 - a
    x = ink.view(np.uint8) if ink.dtype == bool else (ink > 0).astype(np.uint8)
    out = x.all(1)
    if b + 1 <= W:
        out |= x[:, : b + 1].all(1)
    if a + 1 <= W:
        out |= x[:, W - a - 1 :].all(1)
    if length <= W:
        cs = np.zeros((H, W + 1), np.int32)
        np.cumsum(x, axis=1, out=cs[:, 1:])
        out |= ((cs[:, length:] - cs[:, :-length]) == length).any(1)
    return out


def ruling(gray: np.ndarray | None, box) -> tuple[float, float]:
    """Share of the block's height (width) covered by long horizontal (vertical) dark strokes."""
    if gray is None:
        return 0.0, 0.0
    H, W = gray.shape
    x0, y0, x1, y1 = (int(max(0, box[0] - 4)), int(max(0, box[1] - 4)), int(min(W, box[2] + 4)), int(min(H, box[3] + 4)))
    c = gray[y0:y1, x0:x1]
    if c.size == 0 or min(c.shape) < 4:
        return 0.0, 0.0
    # c < t with t < 0 is all False; clamp t to 0 so the comparison stays inside uint8 (a negative Python int
    # against a uint8 array crashed numpy 2.2.0 on an x86 host).
    ink = c < np.uint8(max(0, min(160, int(np.percentile(c, 50)) - 30)))
    h = _any_run(ink, max(8, c.shape[1] // 3))
    v = _any_run(ink.T, max(8, c.shape[0] // 3))
    return float(h.mean()), float(v.mean())


def alignment(b: np.ndarray) -> dict:
    """Line geometry inside the block: rows by vertical centre (overlap > half a line height), lines per row, the gap
    between neighbours in a row, and how many distinct left / right edges the lines use (within half a line height)."""
    h = float(np.median(b[:, 3] - b[:, 1])) or 1.0
    order = np.argsort((b[:, 1] + b[:, 3]) / 2)
    rows: list[list[int]] = []
    for i in order:
        cy = (b[i, 1] + b[i, 3]) / 2
        if rows and abs(cy - np.mean([(b[j, 1] + b[j, 3]) / 2 for j in rows[-1]])) < 0.5 * h:
            rows[-1].append(int(i))
        else:
            rows.append([int(i)])
    gaps = []
    for r in rows:
        xs = sorted(r, key=lambda j: b[j, 0])
        gaps += [(b[k, 0] - b[j, 2]) / h for j, k in zip(xs, xs[1:])]

    def edges(v):
        v = np.sort(v)
        return 1 + int(np.sum(np.diff(v) > 0.5 * h)) if len(v) else 0

    n = len(b)
    multi = [r for r in rows if len(r) >= 2]
    return {
        "g_rows": math.log1p(len(rows)),
        "g_lines_row": n / max(1, len(rows)),
        "g_multi_rows": len(multi) / max(1, len(rows)),
        "g_gap": math.log1p(max(0.0, float(np.median(gaps)))) if gaps else 0.0,
        "g_gap_cv": float(np.std(gaps) / max(1e-6, np.mean(np.abs(gaps)))) if len(gaps) > 1 else 0.0,
        "g_left_edges": edges(b[:, 0]) / max(1, n),
        "g_right_edges": edges(b[:, 2]) / max(1, n),
        "g_width_cv": float(np.std(b[:, 2] - b[:, 0]) / max(1e-6, np.mean(b[:, 2] - b[:, 0]))),
        "g_wide_lines": float(np.mean((b[:, 2] - b[:, 0]) > 0.6 * (b[:, 2].max() - b[:, 0].min()))),
    }


def block_features(blk, boxes, texts, cls_logits, html_table, gray, n_tables, heads_ok, with_ruling=True) -> dict:
    b = np.asarray(boxes)[blk, :4]
    allb = np.asarray(boxes)[:, :4]
    H = float(gray.shape[0]) if gray is not None else float(allb[:, 3].max())
    W = float(gray.shape[1]) if gray is not None else float(allb[:, 2].max())
    bb = (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())
    hts = b[:, 3] - b[:, 1]
    page_h = float(np.median(allb[:, 3] - allb[:, 1])) or 1.0
    rows = parse_table(html_table)
    ncol = [len(r) for r in rows] or [0]
    maxc = max(ncol)
    cells = [c for r in rows for c in r]
    full = [c for c in cells if c]
    lt = [texts[i].strip() for i in blk]
    alltxt = " ".join(lt)
    kv = 0.0
    if maxc == 2 and rows:
        kv = sum(1 for r in rows if len(r) == 2 and r[0] and r[1] and (r[0].endswith(":") or (len(r[0]) <= 40 and not _NUM.match(r[0])))) / len(rows)
    z = np.asarray(cls_logits)[blk].sum(0)
    p = np.exp(z - z.max())
    p = p / p.sum()
    hr, vr = ruling(gray, bb) if with_ruling else (0.0, 0.0)
    f = {
        "lines": math.log1p(len(blk)),
        "rows": math.log1p(len(rows)),
        "cols": math.log1p(float(np.median(ncol))),
        "maxcols": math.log1p(maxc),
        "cells": math.log1p(len(cells)),
        "rows_le3": float(len(rows) <= 3),
        "cols_2": float(maxc == 2),
        "fill": len(full) / max(1, len(rows) * maxc),
        "words_cell": math.log1p(float(np.mean([len(c.split()) for c in full])) if full else 0.0),
        "num_cells": sum(1 for c in full if _NUM.match(c)) / max(1, len(full)),
        "kv2": kv,
        "math_tok": len(_MATH.findall(alltxt)) / max(1, len(alltxt.split())),
        "whole_math": sum(1 for t in lt if _WHOLE_MATH.match(t)) / max(1, len(lt)),
        "sent_end": sum(1 for t in lt if _SENT.search(t)) / max(1, len(lt)),
        "chars_line": math.log1p(float(np.mean([len(t) for t in lt])) if lt else 0.0),
        "lh_cv": float(hts.std() / max(1e-6, hts.mean())),
        "lh_rel": float(np.median(hts) / page_h),
        "area": float((bb[2] - bb[0]) * (bb[3] - bb[1]) / max(1.0, W * H)),
        "width": float((bb[2] - bb[0]) / max(1.0, W)),
        "tables_page": math.log1p(n_tables),
        "rule_h": hr,
        "rule_v": vr,
        "heads_ok": float(bool(heads_ok)),
    }
    for k, v in enumerate(p):
        f[f"cls_{k}"] = float(v)
    f.update(alignment(b))
    return f


class TableVerifier:
    """One weight set: {"features": [...], "mean": [...], "scale": [...], "weights": [...], "bias": b, "threshold": t}.
    table_verify.json holds the set for pages read by OCR at its top level and, optionally, a set for pages read
    from a PDF text layer under "layer" (``load``)."""

    def __init__(self, m: dict):
        self.names = m["features"]
        self.mean = np.array(m["mean"], np.float64)
        self.scale = np.array(m["scale"], np.float64)
        self.w = np.array(m["weights"], np.float64)
        self.b = float(m["bias"])
        self.threshold = float(m["threshold"])

    @staticmethod
    def load(path) -> tuple[TableVerifier, TableVerifier | None]:
        """(the OCR-page verifier, the text-layer verifier or None) from table_verify.json."""
        import json

        m = json.loads(open(path, encoding="utf-8").read())
        return TableVerifier(m), TableVerifier(m["layer"]) if "layer" in m else None

    def decided(self, f: dict) -> bool | None:
        """The decision (P(false) >= threshold) when it holds for every value of the ruling features (each in
        [0, 1]), else None: ruling() then has to run.  The same decisions as computing everything."""
        z = {}
        for v in (0.0, 1.0):
            for u in (0.0, 1.0):
                z[(v, u)] = self.p_false({**f, "rule_h": v, "rule_v": u}) >= self.threshold
        out = set(z.values())
        return out.pop() if len(out) == 1 else None

    def p_false(self, f: dict) -> float:
        x = (np.nan_to_num(np.array([f[k] for k in self.names], np.float64)) - self.mean) / self.scale
        z = float(np.clip(x @ self.w + self.b, -50.0, 50.0))  # a large score must not overflow exp
        return 1.0 / (1.0 + math.exp(-z))

    def apply(self, boxes, texts, order, cls_logits, same, tab, cells, tables, gray):
        """(cls_logits, tables) with every false table relabelled: its lines lose the Table class and its
        tsr_rel table is dropped.  The inputs are returned unchanged when no block is relabelled."""
        from .layout_post import TABLE, _block_part, heads_plausible, split_blocks

        blocks = [b for b in split_blocks(order, same) if int(cls_logits[b].sum(0).argmax()) == TABLE]
        out_cls, out_tab = cls_logits, tables
        for blk in blocks:
            key = tuple(blk)
            if tables and key in tables:
                html, tsr = tables[key], 1.0
            else:
                parts: list = []
                _block_part(parts, blk, boxes, texts, cls_logits, True, cells, tab)
                html, tsr = (parts[0] if parts else ""), 0.0
            ok = len(blk) > 1 and tab is not None and heads_plausible(boxes[blk, :4], tab[:, blk][:, :, blk])
            f = block_features(blk, boxes, texts, cls_logits, html, gray, len(blocks), ok, with_ruling=False)
            f["tsr"] = tsr
            # The ruling features cost most of the time: compute them only when they can change the decision
            # (the score is linear in each, so its extremes are at the corners of [0, 1] x [0, 1]).
            false = self.decided(f)
            if false is None:
                b = boxes[blk, :4]
                f["rule_h"], f["rule_v"] = ruling(gray, (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()))
                false = self.p_false(f) >= self.threshold
            if false:
                if out_cls is cls_logits:
                    out_cls = cls_logits.copy()
                    out_tab = dict(tables) if tables else tables
                out_cls[blk, TABLE] = -1e4
                if out_tab and key in out_tab:
                    del out_tab[key]
        return out_cls, out_tab
