"""Text-layer runs from a PDF page (pdfium), and checks that decide whether to
trust them.

A digital PDF already holds its text.  When the text layer is sound, reading it
is faster than OCR and has no recognition errors.  The checks:

- invisible text (render mode 3) marks an OCR layer over a scan;
- replacement, private-use or control characters mark a broken font map;
- a text run with no ink under it marks hidden or misplaced text;
- right-to-left letters stored in visual order (left to right on the page, as
  some PDF producers write Arabic script) cannot be put back into reading order
  safely, so OCR reads the page instead.

A run is pdfium's own text segment between the line breaks it generates.
Coordinates are PDF points in the frame that pypdfium2 renders (crop box and
page rotation applied), with a top-left origin.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from . import native

MAX_INVISIBLE_FRAC = 0.05
MAX_BAD_CHAR_FRAC = 0.005
MAX_NO_INK_LINE_FRAC = 0.03
# Share of neighbouring right-to-left letters in a run that step left to right.
# Logical order steps right to left (sound layers: 0-1%); a layer in visual
# order steps left to right (Chrome-printed Arabic script: 52-94%).
MAX_VISUAL_RTL_FRAC = 0.25
MIN_RTL_PAIRS = 20
VISUAL_RTL = "right-to-left text in visual order"  # TextLayer.reason
DARK = 224  # grey level below which a pixel counts as ink (a thin glyph at the 72 dpi check render is anti-aliased grey, not black)


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
    quarter: int = 0  # quarter turns from upright in the (turned) frame
    seq: int | None = None  # position in the page's content order (all runs)
    boxes: np.ndarray | None = None  # the chars' boxes [n, 4] (for the native kernels)


@dataclass
class TextLayer:
    lines: list[TextLine]  # upright runs in PDF content order
    size: tuple[float, float]  # page width, height in points (as rendered)
    trusted: bool
    reason: str = ""  # why the layer is not trusted
    stats: dict = field(default_factory=dict)
    rotated: list[TextLine] = field(default_factory=list)  # runs at other quarter turns
    rejected: list[TextLine] = field(default_factory=list)  # bad-code runs to read from pixels
    # boxes (x0, y0, x1, y1, points, in the runs' frame) of digits that may be
    # symbol glyphs: symbol_digit_boxes
    symbol_boxes: np.ndarray | None = None


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


def _quarter(d) -> int | None:
    """Quarter turns (counter-clockwise, as ``np.rot90``) that a baseline running
    in direction d (rendered frame, y down) is turned from left to right; None
    when it is not a quarter turn."""
    q = math.atan2(d[1], d[0]) / (math.pi / 2)  # pdfium's char angle runs clockwise
    k = round(q)
    return k % 4 if abs(q - k) < 0.01 else None


def _turn_frame(px, py, W: float, H: float, turn: int):
    """Rendered-frame points in the frame of the page turned upright again
    (``np.rot90(gray, -turn)``)."""
    if turn == 1:
        return H - py, px
    if turn == 2:
        return W - px, H - py
    if turn == 3:
        return py, W - px
    return px, py


def vertical_layout(stats: dict) -> bool:
    """Whether a page's upright characters run top to bottom (vertical writing):
    most steps between neighbouring characters go down, not right."""
    down, right = stats.get("steps", (0, 0))
    return down >= 30 and down > 2 * right


def page_turn(stats: dict) -> int:
    """The quarter turn that holds most of the page's characters (a table set
    sideways on a portrait page), or 0."""
    q = stats.get("quarters", [0, 0, 0, 0])
    k = int(np.argmax(q[1:])) + 1
    return k if q[k] > 0.5 * max(stats.get("chars", 0), 1) else 0


def code_point(u: int, u2: int = 0) -> tuple[str, bool]:
    """The character of pdfium's UTF-16 code unit ``u`` (``u2``: the next unit),
    and whether it used ``u2``.  A character above U+FFFF (math italic letters,
    emoji, rare CJK) comes as two surrogates; a lone surrogate becomes U+FFFD."""
    if 0xD800 <= u <= 0xDBFF and 0xDC00 <= u2 <= 0xDFFF:
        return chr(0x10000 + ((u - 0xD800) << 10) + (u2 - 0xDC00)), True
    if 0xD800 <= u <= 0xDFFF:
        return "\ufffd", False
    return chr(u), False


def extract_runs(
    page, turn: int = 0, frame: int | None = None
) -> tuple[list[TextLine], dict]:
    """Text runs of a pypdfium2 page that read left to right once the page is
    turned upright by ``turn`` quarter turns (see ``page_turn``), in content
    order; boxes are in the frame turned by ``frame`` (default ``turn``): a
    vertical page has upright characters (turn 0) read in a turned frame (3)."""
    frame = turn if frame is None else frame
    import ctypes

    import pypdfium2
    import pypdfium2.raw as R

    tp = page.get_textpage()
    try:
        n = tp.count_chars()
        # A character is upright when its baseline, mapped through the
        # page-to-render transform, runs left to right after ``turn`` (any
        # /Rotate, any mirroring).
        # Note: pdfium's char angle turns clockwise with y up, and _quarter counts
        # the other way with y down; F L F (F flips y) applies both.  They cancel
        # when L is diagonal (/Rotate 0, 180) but not when it swaps x and y
        # (/Rotate 90, 270), where L alone turns the page 180 degrees the wrong way.
        F = np.diag([1.0, -1.0])
        L = F @ _device_affine(page)[:, :2] @ F
        # The loop below runs once per character: raw pdfium calls with reused
        # buffers, and the upright and font-map tests cached per angle and per
        # character (a page has few distinct ones).
        get_unicode, get_angle = R.FPDFText_GetUnicode, R.FPDFText_GetCharAngle
        get_box, raw_tp = R.FPDFText_GetCharBox, tp.raw
        fast = None
        if native.lib is not None:
            try:
                fast = _native_chars(tp, n, L, turn)
            except Exception as e:  # noqa: BLE001  (the loop below runs instead)
                native.failed("pdf_chars", e)
        if fast is not None:
            chars, raw, breaks, qs, n_chars, n_rot, n_bad, quarters = fast
            n = 0  # the loop below has nothing left to do
        else:
            chars, raw, breaks = [], [], []  # breaks: run ends after these positions
            qs: list = []  # each character's quarter turn relative to ``turn``
            n_chars = n_rot = n_bad = 0
            quarters = [0, 0, 0, 0]
        bl, br, bb, bt = (ctypes.c_double() for _ in range(4))
        pl, pr, pb, pt = (ctypes.byref(v) for v in (bl, br, bb, bt))
        upright: dict = {}
        bad: dict = {}
        cur_q = 0
        low = False  # this index is the second half of a surrogate pair
        for i in range(n):
            if low:
                low = False
                continue
            u = get_unicode(raw_tp, i)
            u2 = (
                get_unicode(raw_tp, i + 1) if 0xD800 <= u <= 0xDBFF and i + 1 < n else 0
            )
            c, low = code_point(u, u2)
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
                q = upright.get(a, -1)
                if q == -1:
                    q = upright[a] = _quarter(L @ np.array([math.cos(a), math.sin(a)]))
                if q is not None:
                    quarters[q] += 1
                if q != turn:
                    n_rot += 1  # not upright on the rendered page
                if q is None:  # not a quarter turn: not read
                    breaks.append(len(chars))
                    continue
                cur_q = (q - turn) % 4
            chars.append(c)
            qs.append(cur_q)
            if not get_box(raw_tp, i, pl, pr, pb, pt):  # yes, l r b t
                raise pypdfium2.PdfiumError("Failed to get charbox.")
            raw.append((bl.value, bb.value, br.value, bt.value))  # l, b, r, t
    finally:
        tp.close()
    stats = {"chars": n_chars, "rotated": n_rot, "bad": n_bad, "quarters": quarters}
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
    # Steps between neighbouring upright characters of a run: down (vertical
    # writing) or right (horizontal), in the rendered frame.
    ok = np.ones(max(len(chars) - 1, 0), bool)
    ok[[b - 1 for b in breaks if 0 < b < len(chars)]] = False
    q = np.asarray(qs) != 0
    sp = np.array([c.isspace() for c in chars], bool)
    ok &= ~(q[:-1] | q[1:] | sp[:-1] | sp[1:])
    h = np.maximum(boxes[:-1, 3] - boxes[:-1, 1], 1e-3)
    dx, dy = np.diff(cx), np.diff(cy)
    ok &= (np.abs(dx) <= 3 * h) & (
        np.abs(dy) <= 3 * h
    )  # a jump to another run is not a step
    down = int((ok & (dy > 0.3 * h) & (np.abs(dx) < 0.3 * h)).sum())
    # Horizontal steps go right (left-to-right scripts) or left (right to left).
    right = int((ok & (np.abs(dx) > 0.3 * h) & (np.abs(dy) < 0.3 * h)).sum())
    stats["steps"] = [down, right]
    if frame:
        tx, ty = _turn_frame(px, py, W, H, frame)
        boxes = np.stack([tx.min(1), ty.min(1), tx.max(1), ty.max(1)], 1)
    runs: list[TextLine] = []
    # A run also ends where the quarter turn changes (a sideways label in an
    # upright line of characters).
    turns = [i for i in range(1, len(chars)) if qs[i] != qs[i - 1]]
    cuts = sorted(set(breaks + turns + [len(chars)]))
    start = 0
    for end in cuts:
        sel = [i for i in range(start, end) if visible[i]]
        seg = [(chars[i], tuple(boxes[i])) for i in sel]
        q = qs[start] if start < len(qs) else 0
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
        text = " ".join("".join(c for c, _ in seg).split())
        if frame != turn and not q and len(ink) > 1:
            # A vertical page read turned: a horizontal run (a header, a
            # caption) runs top to bottom here, so it is a sideways line.
            c0 = [((v[0] + v[2]) / 2, (v[1] + v[3]) / 2) for v in ink]
            dx = abs(c0[-1][0] - c0[0][0])
            dy = abs(c0[-1][1] - c0[0][1])
            if dy > 2 * dx:
                q = 1
        runs.append(TextLine(box, text, seg, q, len(runs), boxes[sel]))
    return runs, stats


def _native_chars(tp, n: int, L: np.ndarray, turn: int):
    """extract_runs' character loop with the native kernels: pdfium's calls and
    the loop run in C++, and the upright test runs here once per distinct angle.
    Returns (chars, boxes [k, 4] as l, b, r, t, breaks, quarter turns, characters,
    turned characters, bad characters, characters per quarter turn), or None
    when anything is off: the Python loop then runs (and raises as it does)."""
    import ctypes

    import pypdfium2.raw as R

    def addr(obj) -> int:
        return ctypes.cast(obj, ctypes.c_void_p).value or 0

    codes, angles, cbox, ok = native.lib.pdf_char_data(
        addr(tp.raw),
        n,
        addr(R.FPDFText_GetUnicode),
        addr(R.FPDFText_GetCharAngle),
        addr(R.FPDFText_GetCharBox),
    )
    ua, inv = np.unique(angles, return_inverse=True)
    qmap = np.empty(len(ua), np.int8)
    for j, a in enumerate(ua.tolist()):
        try:
            q = _quarter(L @ np.array([math.cos(a), math.sin(a)]))
        except (ValueError, OverflowError):  # a NaN angle
            return None
        qmap[j] = -1 if q is None else q
    chars, qs, src, breaks, counted, n_chars, n_rot, quarters, err = (
        native.lib.pdf_char_loop(codes, qmap[inv.reshape(-1)], ok, turn)
    )
    if err:
        return None
    u, cnt = np.unique(counted, return_counts=True)
    n_bad = sum(bad_char_count(chr(c)) * k for c, k in zip(u.tolist(), cnt.tolist()))
    text = chars.astype("<u4").tobytes().decode("utf-32-le")
    return list(text), cbox[src], breaks, qs.tolist(), n_chars, n_rot, n_bad, quarters


def visual_rtl_frac(runs: list[TextLine]) -> tuple[int, float]:
    """(pairs, share): neighbouring right-to-left letters inside a run, and the
    share of them that step left to right on the page."""
    from .rtl import is_rtl

    pairs = right = 0
    for r in runs:
        for (a, ba), (b, bb) in zip(r.chars, r.chars[1:]):
            if is_rtl(a) and is_rtl(b):
                pairs += 1
                right += bb[0] + bb[2] > ba[0] + ba[2]
    return pairs, right / max(pairs, 1)


def invisible_text_frac(page, max_depth: int = 4) -> float:
    """Share of text objects drawn invisibly (an OCR layer) on a pypdfium2 page,
    form objects included down to ``max_depth`` levels (as
    ``page.get_objects(max_depth=4)`` walks them, with raw pdfium calls)."""
    import pypdfium2
    import pypdfium2.raw as R

    TEXT, FORM = R.FPDF_PAGEOBJ_TEXT, R.FPDF_PAGEOBJ_FORM
    INVISIBLE, get_type = R.FPDF_TEXTRENDERMODE_INVISIBLE, R.FPDFPageObj_GetType
    get_mode = R.FPDFTextObj_GetTextRenderMode
    n = inv = 0
    todo = [(None, 0)]  # (form object or None for the page, level)
    while todo:
        form, level = todo.pop()
        if form is None:
            parent, get = page.raw, R.FPDFPage_GetObject
            count = R.FPDFPage_CountObjects(parent)
        else:
            parent, get = form, R.FPDFFormObj_GetObject
            count = R.FPDFFormObj_CountObjects(parent)
        if count < 0:
            raise pypdfium2.PdfiumError("Failed to get number of pageobjects.")
        for i in range(count):
            obj = get(parent, i)
            if not obj:
                raise pypdfium2.PdfiumError("Failed to get pageobject.")
            t = get_type(obj)
            if t == TEXT:
                n += 1
                inv += get_mode(obj) == INVISIBLE
            elif t == FORM and level < max_depth - 1:
                todo.append((obj, level + 1))
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


def symbol_digit_boxes(page, frame: int = 0) -> np.ndarray:
    """Boxes [n, 4] (x0, y0, x1, y1, points, in the frame turned by ``frame``) of
    the digits that may be symbol glyphs: an ASCII digit (0-9) in a symbolic font
    (PDF font flag 4), or an ASCII digit whose font differs from the fonts of the characters on
    both sides of it in its word.  Some fonts give a symbol glyph a digit's code
    and no Unicode map (one publisher's minus comes out '2', its plus-minus '6'),
    drawn from a font other than the number's digits."""
    import ctypes

    import pypdfium2.raw as R

    tp = page.get_textpage()
    try:
        n = R.FPDFText_CountChars(tp.raw)
        uni = [R.FPDFText_GetUnicode(tp.raw, i) for i in range(n)]
        buf = ctypes.create_string_buffer(128)
        flags = ctypes.c_int()
        fonts: dict = {}

        def font(i):
            f = fonts.get(i)
            if f is None:
                R.FPDFText_GetFontInfo(tp.raw, i, buf, len(buf), ctypes.byref(flags))
                f = fonts[i] = (buf.value, flags.value)
            return f

        loose = R.FS_RECTF()

        def height(i):
            # the character's loose box: its drawn size, whatever the glyph's shape
            R.FPDFText_GetLooseCharBox(tp.raw, i, ctypes.byref(loose))
            return loose.top - loose.bottom

        def seen(i, step, reach: int = 3):
            # the nearest character that is not a space, within ``reach`` places
            for d in range(1, reach + 1):
                j = i + step * d
                if not 0 <= j < n:
                    return None
                if not chr(uni[j]).isspace():
                    return j
            return None

        def side(i, step):
            j = i + step
            if 0 <= j < n and not chr(uni[j]).isspace():
                return j
            return None

        raw = []
        x0, x1, y0, y1 = (ctypes.c_double() for _ in range(4))
        for i, u in enumerate(uni):
            if not 0x30 <= u <= 0x39:  # an ASCII digit code (what such a font falls back to)
                continue
            name, fl = font(i)
            near = [j for j in (side(i, -1), side(i, 1)) if j is not None]
            # a superscript or subscript digit (smaller than its neighbours) is a
            # real digit set small, and too small to check against the page
            # the size test looks past spaces (an exponent set apart: 'x 6')
            sized = [j for j in (seen(i, -1), seen(i, 1)) if j is not None]
            if sized and height(i) < 0.85 * max(height(j) for j in sized):
                continue
            if fl & 4 or (near and all(font(j)[0] != name for j in near)):
                if R.FPDFText_GetCharBox(
                    tp.raw, i, ctypes.byref(x0), ctypes.byref(x1), ctypes.byref(y0), ctypes.byref(y1)
                ):
                    raw.append((x0.value, y0.value, x1.value, y1.value))
    finally:
        tp.close()
    if not raw:
        return np.zeros((0, 4), np.float32)
    A = _device_affine(page)
    bb = np.asarray(raw, float)
    xs, ys = bb[:, [0, 2, 0, 2]], bb[:, [1, 1, 3, 3]]
    px = A[0, 0] * xs + A[0, 1] * ys + A[0, 2]
    py = A[1, 0] * xs + A[1, 1] * ys + A[1, 2]
    if frame:
        W, H = page.get_size()
        px, py = _turn_frame(px, py, W, H, frame)
    return np.stack([px.min(1), py.min(1), px.max(1), py.max(1)], 1).astype(np.float32)

def read_text_layer(
    page, gray: np.ndarray, scale: float, turn: int = 0, frame: int | None = None
) -> TextLayer:
    """Text-layer runs of a page and the verdict on them.

    ``page`` is the open pypdfium2 page, ``gray`` a render of it at ``scale``
    pixels per point (a low resolution is enough for the ink check), turned
    upright by ``turn`` quarter turns like the runs (``np.rot90(gray, -turn)``)."""
    frame = turn if frame is None else frame
    W, H = page.get_size()
    size = (H, W) if frame % 2 else (W, H)
    inv = invisible_text_frac(page)
    if inv > MAX_INVISIBLE_FRAC:
        return TextLayer([], size, False, f"invisible text {inv:.0%} (OCR layer)")
    runs, st = extract_runs(page, turn, frame)
    st["turn"], st["frame"] = turn, frame
    rotated = [r for r in runs if r.quarter]
    runs = [r for r in runs if not r.quarter]
    if not runs:
        return TextLayer([], size, False, "no text layer", st, rotated)
    rejected = []
    if st["bad"] > MAX_BAD_CHAR_FRAC * max(st["chars"], 1):
        # An isolated non-text run (e.g. a math extension font's bracket
        # pieces) can use control/private-use codes. Keep its geometry for
        # OCR, then let the recognizer check judge the remaining layer.
        rejected, sound = [], []
        for run in runs:
            (rejected if bad_char_count(run.text) else sound).append(run)
        st["bad_runs"] = len(rejected)
        # Mixed word text and bad codes indicate a broken text encoding, not
        # an isolated non-text symbol. Do not trust the remaining font runs:
        # their ordinary Unicode characters can be wrong too (legacy fonts).
        if not sound or any(c.isalpha() for r in rejected for c in r.text):
            return TextLayer(runs, size, False, BROKEN_FONT_MAP, st, rotated)
        runs = sound
    # Only upright text has a reading direction on the page (a vertical page read
    # turned does not).
    pairs, visual = visual_rtl_frac(runs) if frame == turn else (0, 0.0)
    if pairs >= MIN_RTL_PAIRS and visual > MAX_VISUAL_RTL_FRAC:
        st["visual_rtl"] = round(visual, 3)
        return TextLayer(runs, size, False, VISUAL_RTL, st, rotated)
    empty = no_ink_frac(gray, runs, scale)
    st["no_ink"] = round(empty, 4)
    if empty > MAX_NO_INK_LINE_FRAC:
        return TextLayer(
            runs, size, False, f"{empty:.0%} of text runs have no ink", st, rotated
        )
    out = TextLayer(runs, size, True, "", st, rotated, rejected)
    try:
        out.symbol_boxes = symbol_digit_boxes(page, frame)
    except Exception:  # noqa: BLE001  (an odd font or object: no symbol check, the layer as is)
        out.symbol_boxes = None
    st["symbol_boxes"] = 0 if out.symbol_boxes is None else len(out.symbol_boxes)
    return out
