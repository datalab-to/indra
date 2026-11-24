"""Rotated text lines on an upright page, with the upright detector and recognizer as they are.

The detector finds no line in text turned a quarter turn (an axis label, a margin stamp on a drawing, a vertical table
header).  After the upright read, ink that no readable upright line and no picture covers is grouped (rules removed,
glyphs joined top to bottom); a group much taller than wide is one candidate turned line.  The recognizer reads every
candidate on the page turned 90 and 270 degrees (one batch per turn).  A candidate becomes a line when one turn reads
at least ROT_MIN_CHARS letters or digits at ROT_MIN_CONF or more and the other turn does not (text reads in one direction
only; a rule, a dash or an 'o' reads the same both ways), so the recognizer makes the decision.  Accepted lines join the
page as sideways lines, the way text-layer sideways lines do (document.py): box flag 1 in column 4 and one x for all
tokens, so reading order and markdown need nothing new.

A page without a candidate group is read exactly as before.
"""

from __future__ import annotations

import math

import numpy as np

# A candidate group: at least ROT_ASPECT times taller than wide and at least ROT_MIN_LEN median line heights tall.
# At most ROT_MAX groups per page (a bound on the cost; rotcls.onnx decides which are text).
ROT_ASPECT = 2.5
ROT_MIN_LEN = 2.0
ROT_MAX = 64
# A turned reading must beat the crop read upright, and its lines need this recognizer confidence: marks, bullets
# and rules read as text sit below it.
ROT_MIN_CONF = 0.9
ROT_MIN_CHARS = 3  # letters or digits in an accepted turned line: a turned 1-2 character reading is as likely a mark
MASK_LINE_H = 8.0  # the ink mask is made at a scale where the median line is this many pixels tall
RULE_LEN = 3.0
# rotcls.onnx (optional): P(the column is turned text worth a recognizer read); columns under ROT_CLS_MIN are skipped
ROT_CLS_MIN = 0.3
CLS_H, CLS_W = 16, 64  # its input: the column (+0.2 line heights) turned np.rot90(crop, 1), resized, ink = 1 - gray / 255  # straight ink runs this many line heights long are rules (table grids, borders), not text


def candidates(gray: np.ndarray, boxes: np.ndarray, rec: list, pictures: np.ndarray, med_h: float, min_conf: float,
               det_in=None):
    """Page boxes [x0, y0, x1, y1] (pixels) of tall, narrow ink groups that no readable line covers.  ``det_in``: the
    detector's input of this page (ink map, scale), which saves resizing the page again."""
    import cv2

    f = min(1.0, MASK_LINE_H / max(med_h, 1.0))
    if det_in is not None and det_in[1] >= f:
        x, s = det_in
        H, W = gray.shape
        small = (255.0 * (1.0 - x[: max(1, round(H * s)), : max(1, round(W * s))])).astype(np.uint8)
        if s > 1.5 * f:  # lines much taller than MASK_LINE_H: shrink the small map (cheap) instead
            small = cv2.resize(small, None, fx=f / s, fy=f / s, interpolation=cv2.INTER_AREA)
        else:
            f = s
        f = small.shape[1] / W
    else:
        small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1.0 else gray
    _, ink = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h = med_h * f
    pad = 0.25 * h
    for b, r in zip(boxes, rec):
        if r[0].strip() and r[1] >= min_conf:
            x0, y0, x1, y1 = (int(v) for v in (b[0] * f - pad, b[1] * f - pad, b[2] * f + pad + 1, b[3] * f + pad + 1))
            ink[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = 0
    for b in pictures:
        x0, y0, x1, y1 = (int(v * f) for v in b[:4])
        ink[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1] = 0
    if not ink.any():
        return []
    # Rules join rotated header text to the whole table grid: remove them first.
    n_rule = max(3, round(RULE_LEN * h))
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (n_rule, 1)))
    rules |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, n_rule)))
    ink &= 1 - rules
    # Join the glyphs of a turned line (stacked top to bottom), not neighbouring columns.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, round(1.5 * h))))  # 1.5: a word gap does not cut the line
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, k), connectivity=8)
    cols = [
        [x, y, x + w, y + hh]
        for x, y, w, hh, _ in stats[1:]
        if hh >= ROT_ASPECT * w and hh >= ROT_MIN_LEN * h
    ]
    cols.sort(key=lambda c: -(c[2] - c[0]) * (c[3] - c[1]))
    return [(c[0] / f, c[1] / f, c[2] / f, c[3] / f) for c in cols[:ROT_MAX]]


def column_inputs(gray: np.ndarray, cols: list, med_h: float) -> np.ndarray:
    """[N, 1, CLS_H, CLS_W] float32 inputs of rotcls.onnx."""
    import cv2

    H, W = gray.shape
    m = 0.2 * med_h
    out = np.zeros((len(cols), 1, CLS_H, CLS_W), np.float32)
    for i, b in enumerate(cols):
        x0, y0, x1, y1 = int(max(0, b[0] - m)), int(max(0, b[1] - m)), int(min(W, b[2] + m)), int(min(H, b[3] + m))
        c = gray[y0:y1, x0:x1]
        if c.size == 0 or min(c.shape) < 2:
            continue
        r = cv2.resize(np.ascontiguousarray(np.rot90(c, 1)), (CLS_W, CLS_H), interpolation=cv2.INTER_AREA)
        out[i, 0] = 1.0 - r.astype(np.float32) / 255.0
    return out


def _turn_box(b, k: int, H: int, W: int):
    """A page box (H x W page) on the page turned by np.rot90(page, k), k = 1 or 3."""
    x0, y0, x1, y1 = b
    if k == 1:  # turned[r, c] = page[c, W - 1 - r]: page (x, y) -> turned (y, W - x)
        return (y0, W - x1, y1, W - x0)
    return (H - y1, x0, H - y0, x1)  # k == 3: turned[r, c] = page[H - 1 - c, r]: page (x, y) -> turned (H - y, x)


def read_rotated(ocr, gray: np.ndarray, boxes: np.ndarray, rec: list, pictures: np.ndarray, min_conf: float,
                 cls=None, det_in=None):
    """(boxes, rec) with the rotated lines of the page added and the upright fragments they replace removed.
    Each candidate column is one turned line: the recognizer reads it on the page turned 90 and 270 degrees (one
    batch per turn), and the surer reading is kept when its confidence is ROT_MIN_CONF or more."""
    if len(boxes) == 0:
        return boxes, rec
    med_h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    cols = candidates(gray, boxes, rec, pictures, med_h, min_conf, det_in)
    if cols and cls is not None:  # the column classifier skips columns not worth a recognizer read
        logit = cls.run(None, {"x": column_inputs(gray, cols, med_h)})[0]
        cols = [c for c, z in zip(cols, logit) if 1.0 / (1.0 + np.exp(-float(z))) >= ROT_CLS_MIN]
    if not cols:
        return boxes, rec
    H, W = gray.shape
    reads = {}
    for k in (1, 3):
        t = np.ascontiguousarray(np.rot90(gray, k))
        tb = np.zeros((len(cols), boxes.shape[1]), np.float32)
        tb[:, :4] = [_turn_box(c, k, H, W) for c in cols]
        reads[k] = ocr.recognize(t, tb)
    centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    drop: set = set()
    new_boxes, new_rec = [], []
    for j, c in enumerate(cols):
        r, other = sorted((reads[1][j], reads[3][j]), key=lambda x: -x[1])
        # Text reads in one direction only (upside down it is unsure); a rule, a dash or an 'o' reads the same both ways.
        if sum(ch.isalnum() for ch in r[0]) < ROT_MIN_CHARS or r[1] < ROT_MIN_CONF or other[1] >= ROT_MIN_CONF:
            continue
        x0, y0, x1, y1 = c
        drop.update(
            int(i)
            for i in np.where((centres[:, 0] >= x0) & (centres[:, 0] < x1) & (centres[:, 1] >= y0) & (centres[:, 1] < y1))[0]
            if not (rec[i][0].strip() and rec[i][1] >= min_conf)
        )
        cx = float(x0 + x1) / 2  # one x: a sideways line is never cut
        row = np.zeros(boxes.shape[1], np.float32)
        row[:4] = c
        if boxes.shape[1] > 4:
            row[4] = 1.0
        new_boxes.append(row)
        new_rec.append((r[0], r[1], [(tk[0], cx) for tk in r[2]]))
    if not new_boxes:
        return boxes, rec
    keep = [i for i in range(len(boxes)) if i not in drop]
    out_boxes = np.concatenate([boxes[keep].reshape(-1, boxes.shape[1]), np.array(new_boxes, np.float32)])
    return out_boxes, [rec[i] for i in keep] + new_rec
