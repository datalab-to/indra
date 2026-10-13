"""Text-layer runs from a PDF page (pdfium), and checks that decide whether to
trust them.

A digital PDF already holds its text.  When the text layer is sound, reading it
is faster than OCR and has no recognition errors.  The checks follow the rules
that select the recognizer's own training labels:

- invisible text (render mode 3) marks an OCR layer over a scan;
- replacement, private-use or control characters mark a broken font map;
- a text run with no ink under it marks hidden or misplaced text.

A run is pdfium's own text segment between the line breaks it generates.
Coordinates are PDF points in the frame that pypdfium2 renders (crop box and
page rotation applied), with a top-left origin.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field

import numpy as np

MAX_INVISIBLE_FRAC = 0.05
MAX_BAD_CHAR_FRAC = 0.005
MAX_NO_INK_LINE_FRAC = 0.03
DARK = 160  # grey level below which a pixel counts as ink


def bad_char_count(text: str) -> int:
    n = 0
    for c in text:
        cat = unicodedata.category(c)
        if c == "�" or cat in ("Co", "Cs", "Cn") or (cat == "Cc" and c != "\t"):
            n += 1
    return n


BROKEN_FONT_MAP = "broken font map"  # TextLayer.reason: the layer's codes are not text


@dataclass
class TextLine:
    """One text-layer run: its box, text, and characters with boxes."""

    bbox: tuple[float, float, float, float]
    text: str
    chars: list = field(default_factory=list)  # (char, bbox) in PDF order


@dataclass
class TextLayer:
    lines: list[TextLine]  # runs in PDF content order
    size: tuple[float, float]  # page width, height in points (as rendered)
    trusted: bool
    reason: str = ""  # why the layer is not trusted
    stats: dict = field(default_factory=dict)


def _device_affine(page) -> np.ndarray:
    """[2, 3] map from PDF page space to the rendered frame (points, top-left
    origin), from pdfium's own FPDF_PageToDevice."""
    import ctypes

    import pypdfium2.raw as R

    W, H = page.get_size()
    S = 1000  # device units per point: keeps the integer output precise

    def dev(x, y):
        dx, dy = ctypes.c_int(), ctypes.c_int()
        R.FPDF_PageToDevice(page.raw, 0, 0, round(W * S), round(H * S), 0, x, y, dx, dy)
        return np.array([dx.value, dy.value], float) / S

    o, ex, ey = dev(0.0, 0.0), dev(1.0, 0.0), dev(0.0, 1.0)
    return np.stack([ex - o, ey - o, o], 1)


def extract_runs(page) -> tuple[list[TextLine], dict]:
    """Upright text runs of a pypdfium2 page, in content order."""
    import ctypes

    import pypdfium2
    import pypdfium2.raw as R

    tp = page.get_textpage()
    try:
        n = tp.count_chars()
        # A character is upright on the rendered page when its baseline, mapped
        # through the page-to-render transform, runs left to right (any /Rotate,
        # any mirroring).
        L = _device_affine(page)[:, :2]
        # The loop below runs once per character: raw pdfium calls with reused
        # buffers, and the upright and font-map tests cached per angle and per
        # character (a page has few distinct ones).
        get_unicode, get_angle = R.FPDFText_GetUnicode, R.FPDFText_GetCharAngle
        get_box, raw_tp = R.FPDFText_GetCharBox, tp.raw
        bl, br, bb, bt = (ctypes.c_double() for _ in range(4))
        pl, pr, pb, pt = (ctypes.byref(v) for v in (bl, br, bb, bt))
        upright: dict = {}
        bad: dict = {}
        chars, raw, breaks = [], [], []  # breaks: run ends after these positions
        n_chars = n_rot = n_bad = 0
        for i in range(n):
            c = chr(get_unicode(raw_tp, i))
            if c in "\r\n":
                breaks.append(len(chars))
                continue
            if c == "\x02":  # pdfium's mark for a hyphen at a line break
                c = "-"
            if not c.isspace():
                n_chars += 1
                k = bad.get(c)
                if k is None:
                    k = bad[c] = bad_char_count(c)
                n_bad += k
                a = get_angle(raw_tp, i)
                up = upright.get(a)
                if up is None:
                    d = L @ np.array([math.cos(a), math.sin(a)])
                    up = upright[a] = not (
                        d[0] <= 0 or abs(d[1]) > 1e-3 * max(abs(d[0]), 1e-9)
                    )
                if not up:
                    n_rot += 1  # not upright on the rendered page
                    breaks.append(len(chars))
                    continue
            chars.append(c)
            if not get_box(raw_tp, i, pl, pr, pb, pt):  # yes, l r b t
                raise pypdfium2.PdfiumError("Failed to get charbox.")
            raw.append((bl.value, bb.value, br.value, bt.value))  # l, b, r, t
    finally:
        tp.close()
    stats = {"chars": n_chars, "rotated": n_rot, "bad": n_bad}
    if not chars:
        return [], stats
    # Page space -> rendered frame, all characters at once.
    A = _device_affine(page)
    b = np.asarray(raw, float)
    xs = b[:, [0, 2, 0, 2]]
    ys = b[:, [1, 1, 3, 3]]
    px = A[0, 0] * xs + A[0, 1] * ys + A[0, 2]
    py = A[1, 0] * xs + A[1, 1] * ys + A[1, 2]
    boxes = np.stack([px.min(1), py.min(1), px.max(1), py.max(1)], 1)
    # Characters outside the rendered frame (beyond a crop box) are not on the
    # page: they are dropped, not read.
    W, H = page.get_size()
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    visible = (cx >= 0) & (cx <= W) & (cy >= 0) & (cy <= H)
    stats["off_page"] = int((~visible).sum())
    runs: list[TextLine] = []
    cuts = sorted(set(breaks + [len(chars)]))
    start = 0
    for end in cuts:
        seg = [(chars[i], tuple(boxes[i])) for i in range(start, end) if visible[i]]
        start = end
        ink = [bx for c, bx in seg if not c.isspace()]
        if not ink:
            continue
        box = (
            min(v[0] for v in ink),
            min(v[1] for v in ink),
            max(v[2] for v in ink),
            max(v[3] for v in ink),
        )
        runs.append(TextLine(box, " ".join("".join(c for c, _ in seg).split()), seg))
    return runs, stats


def invisible_text_frac(page) -> float:
    """Share of text objects drawn invisibly (an OCR layer) on a pypdfium2 page."""
    import pypdfium2.raw as R

    n = inv = 0
    for obj in page.get_objects(max_depth=4):
        if obj.type == R.FPDF_PAGEOBJ_TEXT:
            n += 1
            inv += (
                R.FPDFTextObj_GetTextRenderMode(obj.raw)
                == R.FPDF_TEXTRENDERMODE_INVISIBLE
            )
    return inv / n if n else 0.0


def no_ink_frac(gray: np.ndarray, runs: list[TextLine], scale: float) -> float:
    """Share of runs whose box holds no dark pixel in ``gray`` (``scale`` px/pt)."""
    ink = gray < DARK
    H, W = ink.shape
    empty = 0
    for ln in runs:
        x0, y0, x1, y1 = ln.bbox
        r = slice(max(0, int(y0 * scale)), min(H, int(y1 * scale) + 1))
        c = slice(max(0, int(x0 * scale)), min(W, int(x1 * scale) + 1))
        if not ink[r, c].any():
            empty += 1
    return empty / max(len(runs), 1)


def read_text_layer(page, gray: np.ndarray, scale: float) -> TextLayer:
    """Text-layer runs of a page and the verdict on them.

    ``page`` is the open pypdfium2 page, ``gray`` a render of it at ``scale``
    pixels per point (a low resolution is enough for the ink check)."""
    size = page.get_size()
    inv = invisible_text_frac(page)
    if inv > MAX_INVISIBLE_FRAC:
        return TextLayer([], size, False, f"invisible text {inv:.0%} (OCR layer)")
    runs, st = extract_runs(page)
    if not runs:
        return TextLayer([], size, False, "no text layer", st)
    if st["bad"] > MAX_BAD_CHAR_FRAC * max(st["chars"], 1):
        return TextLayer(runs, size, False, BROKEN_FONT_MAP, st)
    empty = no_ink_frac(gray, runs, scale)
    st["no_ink"] = round(empty, 4)
    if empty > MAX_NO_INK_LINE_FRAC:
        return TextLayer(runs, size, False, f"{empty:.0%} of text runs have no ink", st)
    return TextLayer(runs, size, True, "", st)


def order_backtracks(boxes: np.ndarray) -> int:
    """Moves upward in content order that do not go to a column on the right.

    Reading goes down a column, then up to the top of the next column on the
    right.  Any other upward move means the content order is not reading order."""
    if len(boxes) < 3:
        return 0
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    bt = 0
    for a, b in zip(boxes[:-1], boxes[1:]):
        if b[1] < a[1] - 0.5 * h and b[0] < a[2] - h:
            bt += 1
    return bt


MIN_MODEL_AGREEMENT = 0.85


def pdf_order_coherent(
    boxes: np.ndarray, pair_logits: np.ndarray | None = None
) -> bool:
    """Is the content order a reading order?

    Few upward moves that are not column changes, and (when the order model's
    pairwise logits are given, ``a[i, j] > 0``: i before j) the model agrees with
    most consecutive pairs.  The second check catches two columns written line by
    line across the gutter, which has no upward move at all."""
    if order_backtracks(boxes) > max(1, int(0.02 * len(boxes))):
        return False
    if pair_logits is None or len(boxes) < 3:
        return True
    i = np.arange(len(boxes) - 1)
    return float((pair_logits[i, i + 1] > 0).mean()) >= MIN_MODEL_AGREEMENT
