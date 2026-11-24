"""Whole documents: the PDF text layer where it can be trusted, OCR elsewhere.

For each PDF page (``mode="auto"``):

1. Read the text layer with pdfium and check it (``textlayer.read_text_layer``).  A page
   without a sound layer (a scan, an OCR layer, a broken font map) goes to OCR.
2. The detector finds the visual lines, as in OCR.  Each text-layer character
   goes to the detected line that contains its center.  Inside a line, the
   characters of each text-layer run (pdfium's segment between line breaks) keep the PDF order
   and spaces, and runs are ordered by x, right to left in a right-to-left line.  So inline math, sub- and
   superscripts and footnote marks land in their line, and a run across two
   table cells splits between them.  A detected line with no text (text inside
   an image) is OCR'd when the recognizer reads the page's script.
3. The rest is the OCR path: lines split at wide gaps, reading order, layout
   classes and markdown.  The PDF content order is kept when it is coherent
   (few upward jumps that are not column changes, and the order model agrees
   with at least 85% of consecutive pairs).

``mode="ocr"`` never reads the text layer; ``mode="text"`` uses it even when
the checks fail (and OCR only on pages with no text at all).
"""

from __future__ import annotations

import ctypes
import time
import unicodedata
from pathlib import Path

import numpy as np

from . import native
from .agree import snippet
from .pipeline import InputError, OrtOCR, Page, is_pdf, quarter_turns, read_images
from .textlayer import BROKEN_FONT_MAP, page_turn, read_text_layer, vertical_layout

CHECK_DPI = 72.0  # resolution of the render for the text-layer ink checks
MAX_PAGE_PIXELS = 100e6  # a page larger than this at the set dpi renders at a lower dpi (large scans reach ~65 MP)

READABLE_MIN = 0.9  # share of the layer's letters the recognizer must know
CHECK_LINES = 6  # text-layer lines whose start is read from the page image (agree.py)
CHECK_CHARS = 24  # about this many characters from the start of each, cut at a word gap
GAP_CAP = 10.0  # a line's gap counts at most this much (one odd line cannot fail a page)
EASY_GAP = 0.25  # the first line read at most this far off: the layer matches, the rest is not read
SURE_CONF = 0.97  # a recognizer this sure of its own reading: a line it reads off the layer is strong evidence
# The page score over the checked lines: a logistic model.  Inputs, in order: the trimmed mean gap, the median gap, the share of lines with a gap
# over 0.5, 1, 2 and 4, the recognizers' mean confidence in their own readings,
# per group recognizer (LAYER_GROUPS; the other lines are the base recognizer's)
# the share of lines it read times the share of lines over 0.5, and the share of
# lines over 1 that their recognizer reads with a confidence of at least
# SURE_CONF.  So lines that do not match count for more in a script whose layers
# are often broken (Indic, Arabic) than in Latin, and a line that a sure
# recognizer reads otherwise (broken word spacing, a wrong letter map) counts for
# more than one it is unsure of (a rare letter, as in Fon, that it misreads).
# Over 0, OCR reads the page.
LAYER_GROUPS = ("world", "rtl", "cjk")
LAYER_WEIGHTS = (0.06462, 0.07808, -0.1493, 0.3135, 1.514, 1.05, 7.785, 2.993, 3.216, 5.219, 3.738)
LAYER_BIAS = -9.334
# A digit that may be a symbol glyph (textlayer.symbol_digit_boxes: a font that
# gives a symbol a digit's code, as a minus read '2') is checked against the page:
# its line is read, a digit that the reading has as a symbol is a candidate, and
# the symbol replaces the digit when the recognizer finds the line with the symbol
# at least SYMBOL_MARGIN nats more likely than with the digit (agree.forced_gap).
SYMBOL_MARGIN = 2.0
NOT_THE_PAGE = "text layer does not match the page"  # Page.note when the check sends a page to OCR


def layer_features(reads: list[tuple]) -> list[float] | None:
    """The page score's inputs from the checked lines' (gap, conf, group)
    (``OrtOCR.layer_gaps``; a math line, gap None, is left out); None when no
    line counts.  Gaps are capped at GAP_CAP; the trimmed mean leaves out the
    largest (one line of overlapping or odd text cannot fail a page)."""
    lines = [(min(r[0], GAP_CAP), r[1], r[2] if len(r) > 2 else "base") for r in reads if r[0] is not None]
    if not lines:
        return None
    gaps = sorted(g for g, *_ in lines)
    kept = gaps[:-1] if len(gaps) > 2 else gaps
    n = len(gaps)
    over = sum(g > 0.5 for g in gaps) / n
    return [
        sum(kept) / len(kept),
        float(np.median(gaps)),
        *(sum(g > t for g in gaps) / n for t in (0.5, 1.0, 2.0, 4.0)),
        sum(c for _, c, _ in lines) / n,
        *(over * sum(grp == name for *_, grp in lines) / n for name in LAYER_GROUPS),
        sum(g > 1.0 and c >= SURE_CONF for g, c, _ in lines) / n,
    ]


def layer_matches(reads: list[tuple]) -> bool:
    """Whether a text layer is its page's text: the page score (LAYER_WEIGHTS)
    over the checked lines is at most 0.  A legacy font or a broken ToUnicode map
    gives most lines a gap of 2 to 20; a map that breaks only some letters (Thai
    tone marks, Arabic lam-alef, Indic vowel signs) gives some lines a gap of 0.5
    to 3; a sound layer gives about 0, also where the recognizer is unsure."""
    x = layer_features(reads)
    if x is None:
        return True
    return LAYER_BIAS + sum(w * v for w, v in zip(LAYER_WEIGHTS, x)) <= 0.0


def symbol_words(toks: list, box, sym: np.ndarray) -> list[tuple[int, int, list[int]]]:
    """Words of one detected line (tokens (char, x, ...), box in pixels) that hold
    a digit inside a symbolic-font box (``sym``, pixels): (start, end, positions
    of those digits)."""
    if not len(sym) or not toks:
        return []
    y0, y1 = float(box[1]), float(box[3])
    over = np.minimum(sym[:, 3], y1) - np.maximum(sym[:, 1], y0)
    rows = sym[over > 0.5 * np.minimum(sym[:, 3] - sym[:, 1], y1 - y0)]
    if not len(rows):
        return []
    out, start = [], None
    for j in range(len(toks) + 1):
        space = j == len(toks) or toks[j][0] == " "
        if not space and start is None:
            start = j
        if space and start is not None:
            hits = [
                p
                for p in range(start, j)
                if isinstance(toks[p][0], str)
                and "0" <= toks[p][0] <= "9"
                and ((rows[:, 0] - 1 <= toks[p][1]) & (toks[p][1] <= rows[:, 2] + 1)).any()
            ]
            if hits:
                out.append((start, j, hits))
            start = None
    return out


def symbol_candidates(layer: str, read: str, hits: list[int]) -> dict[int, str]:
    """Positions (into ``layer``, a line's text from its tokens) of the digits at
    ``hits`` that the recognizer's reading ``read`` of the line has as a symbol or a
    dash (Unicode Sm, So, Sk, Sc, Pd), one character for one, and that character."""
    import difflib

    out = {}
    hit = set(hits)
    sm = difflib.SequenceMatcher(None, layer, read, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "replace" or i2 - i1 != j2 - j1:
            continue
        for p in range(i1, i2):
            c = read[j1 + p - i1]
            # a symbol or a dash: what a font that codes glyphs as digits holds (a
            # cut or tiny glyph reads as light punctuation: '.', ':', '_')
            if p in hit and unicodedata.category(c) in ("Sm", "So", "Sk", "Sc", "Pd"):
                out[p] = c
    return out


def readable(ocr: OrtOCR, runs: list) -> bool:
    """Whether the recognizer can read the script of these text-layer runs."""
    letters = [c for r in runs for c in r.text if c.isalpha()]
    if not letters:
        return True
    vocabs = [ocr.vocab.index] + [v.index for _, v in ocr.groups.values()]
    # A recognizer that reads Hangul as jamo knows a syllable by its jamo.
    known = sum(
        any(
            c in v or all(j in v for j in unicodedata.normalize("NFD", c))
            for v in vocabs
        )
        for c in letters
    )
    return known >= READABLE_MIN * len(letters)


def right_to_left(tokens: list) -> bool:
    """Unicode bidi base direction: most strong characters are right-to-left."""
    rtl = ltr = 0
    for t in tokens:
        b = unicodedata.bidirectional(t[0]) if len(t[0]) == 1 else ""
        rtl += b in ("R", "AL")
        ltr += b == "L"
    return rtl > ltr


def _covered(box, boxes: np.ndarray, frac: float = 0.5) -> bool:
    """Whether more than ``frac`` of ``box`` lies inside one of ``boxes``."""
    ix = np.clip(
        np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None
    )
    iy = np.clip(
        np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None
    )
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    return bool(((ix * iy) / area > frac).any())


def _page_rotations(rotation):
    """Page index -> known rotation (None: not known), from convert's ``rotation``;
    raises ValueError for a value that is not a multiple of 90 degrees."""
    if rotation is None:
        return lambda i: None
    if isinstance(rotation, dict):
        for v in rotation.values():
            quarter_turns(v)
        return rotation.get
    quarter_turns(rotation)
    return lambda i: rotation


def _render(page, scale: float) -> np.ndarray:
    import pypdfium2 as pdfium
    import pypdfium2.internal as pdfium_i

    def empty_bitmap(width, height, format, rev_byteorder=False):
        # The render fills every pixel, so the buffer needs no zeroing first.
        size = width * height * pdfium_i.BitmapTypeToNChannels[format]
        buffer = (ctypes.c_ubyte * size).from_buffer(np.empty(size, np.uint8))
        return pdfium.PdfBitmap.new_native(
            width, height, format, rev_byteorder, buffer=buffer
        )

    arr = page.render(
        scale=scale, grayscale=True, bitmap_maker=empty_bitmap
    ).to_numpy()
    return np.ascontiguousarray(arr[..., 0] if arr.ndim == 3 else arr)


def _smallest_box(
    ink: list, boxes: np.ndarray, area: np.ndarray, k: float
) -> np.ndarray:
    """Per character box in ``ink`` (points; l, b, r, t as the text layer gives
    them), the index of the smallest detected box (pixels, ``k`` per point)
    that holds the character's center, the first one on a tie; -1 for none.
    Same result as testing one character at a time against ``boxes``."""
    out = np.full(len(ink), -1, np.int64)
    if not len(ink) or not len(boxes):
        return out
    # One character's center is a scalar of the box values' type; numpy
    # compares it with ``boxes`` in the dtype its promotion rules give.
    cmp = np.result_type(boxes.dtype, type(ink[0][0])(1.0))
    chars = np.asarray(ink, float).reshape(-1, 4)
    cx = (chars[:, 0] + chars[:, 2]) / 2 * k
    cy = (chars[:, 1] + chars[:, 3]) / 2 * k
    step = max(1, 4_000_000 // len(boxes))  # bounds the [chars, boxes] masks
    for s in range(0, len(chars), step):
        x = cx[s : s + step, None].astype(cmp)
        y = cy[s : s + step, None].astype(cmp)
        inside = (
            (boxes[None, :, 0] <= x)
            & (x <= boxes[None, :, 2])
            & (boxes[None, :, 1] <= y)
            & (y <= boxes[None, :, 3])
        )
        a = np.where(inside, area[None, :], np.inf)
        j = a.argmin(1)
        out[s : s + step] = np.where(inside[np.arange(len(j)), j], j, -1)
    return out


def _line_chars(toks: list, by: dict, k: float, index: dict | None = None) -> list | None:
    """(char, x0, y0, x1, y1) in pixels for the non-space text-layer tokens of one
    line, in token order: each token's box is the box of the character of its run
    with the same character and the nearest center (a run may be split between
    lines).  None when a token has no such character.  index: a cache of
    _run_index per run, shared by the lines of a page."""
    index = {} if index is None else index
    out = []
    for c, cx, ix in toks:
        if c.isspace():
            continue
        run = by.get(ix)
        if run is None:
            return None
        found = index.get(ix)
        if found is None:
            found = index[ix] = _run_index(run, k)
        exact, near = found
        b = exact.get((c, cx))
        if b is None:  # the token's x is not exactly a char center: search
            boxes = near.get(c)
            if boxes is None:
                return None
            boxes = np.asarray(boxes, np.float64)
            b = boxes[int(np.argmin(np.abs((boxes[:, 0] + boxes[:, 2]) / 2 * k - cx)))]
        out.append((c, b[0] * k, min(b[1], b[3]) * k, b[2] * k, max(b[1], b[3]) * k))
    return out


def _run_index(run, k: float) -> tuple[dict, dict]:
    """A run's chars by (char, center x in pixels), as assign_runs writes a token's
    x (the first char on a tie), and char -> its boxes in run order."""
    exact: dict = {}
    by_char: dict = {}
    for rc, rb in run.chars:
        exact.setdefault((rc, (rb[0] + rb[2]) / 2 * k), rb)
        by_char.setdefault(rc, []).append(rb)
    return exact, by_char


def assign_runs(runs: list, boxes: np.ndarray, k: float) -> tuple[list[list], int]:
    """Tokens (char, x, content index) per detected line.

    A run is one text-layer line piece (``textlayer.TextLine``), in content
    order.  Each character goes to the smallest detected box that contains its
    center (a space follows the character before it), so a run that spans two
    table cells splits between them.  Inside a detected line, each run's part
    keeps the PDF character order, and the parts are ordered by x.  ``boxes``
    are pixels, ``k`` pixels per point.  Returns (tokens per detected line,
    characters outside every box)."""
    parts: list[dict] = [{} for _ in range(len(boxes))]  # line -> run idx -> tokens
    lost = 0
    if not len(boxes):
        return [], sum(not c.isspace() for r in runs for c, _ in r.chars)
    if native.lib is not None:
        try:
            n_lost = _assign_native(runs, boxes, k, parts)
        except Exception as e:  # noqa: BLE001  (the loop below assigns them)
            native.failed("assign_chars", e)
            n_lost = None
            for p in parts:  # start over from empty lines
                p.clear()
        if n_lost is not None:
            lost, runs = n_lost, []  # assigned
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # Every character's box at once: its center, and the smallest detected box
    # holding it (the first such box on a tie), or -1.
    ink = [b for run in runs for c, b in run.chars if not c.isspace()]
    home = _smallest_box(ink, boxes, area, k)
    n = 0
    for pos, run in enumerate(runs):
        idx = run.seq if run.seq is not None else pos
        j = None
        for c, b in run.chars:
            if not c.isspace():
                j = int(home[n])
                cx = (b[0] + b[2]) / 2 * k
                n += 1
                if j < 0:
                    lost += 1
                    j = None
                    continue
                parts[j].setdefault(idx, []).append((c, cx, idx))
            elif j is not None:
                prev = parts[j][idx][-1]
                parts[j][idx].append((c, prev[1], idx))
    toks: list[list] = []
    for line in parts:
        tk: list = []
        rtl = right_to_left([t for part in line.values() for t in part])
        order = (
            sorted(line, key=lambda i: -max(t[1] for t in line[i]))
            if rtl
            else sorted(line, key=lambda i: min(t[1] for t in line[i]))
        )
        for idx in order:
            if tk:
                tk.append((" ", tk[-1][1], idx))  # parts of different runs
            tk += line[idx]
        toks.append(tk)
    return toks, lost


def _assign_native(runs: list, boxes: np.ndarray, k: float, parts: list) -> int | None:
    """The loop of assign_runs with the native kernels: each character's line
    and x come from C++ (the same tests in float64, the same smallest box), and
    the tokens go into ``parts`` as the loop puts them.  Returns the lost count,
    or None (``parts`` untouched) for runs the kernel does not take: runs
    without their boxes array, or tokens that are not one character."""
    if any(r.boxes is None or len(r.boxes) != len(r.chars) for r in runs):
        return None
    chars = [c for r in runs for c, _ in r.chars]
    if not chars:
        return 0
    codes = np.frombuffer("".join(chars).encode("utf-32-le"), "<u4")
    if len(codes) != len(chars):
        return None
    start = np.cumsum([0] + [len(r.chars) for r in runs], dtype=np.int64)
    line_of, x_of, lost = native.lib.assign_chars(
        codes,
        np.ascontiguousarray(np.concatenate([r.boxes for r in runs]), np.float64),
        start,
        np.ascontiguousarray(boxes[:, :4], np.float32),
        float(k),
    )
    line_of = line_of.tolist()
    xs = list(x_of)  # float64 scalars, as the loop's centres are
    for pos, run in enumerate(runs):
        idx = run.seq if run.seq is not None else pos
        for i in range(start[pos], start[pos + 1]):
            j = line_of[i]
            if j >= 0:
                parts[j].setdefault(idx, []).append((chars[i], xs[i], idx))
    return lost


class Converter:
    def __init__(
        self,
        ocr: OrtOCR,
        dpi: float = 200.0,
        mode: str = "auto",
        image_ocr: bool = True,
    ):
        """mode: auto (text layer where sound, OCR elsewhere), ocr (always OCR,
        never read the text layer), text (text layer even when the checks fail).
        image_ocr: OCR detected lines that
        hold no text-layer characters (False: drop them)."""
        if mode not in ("auto", "ocr", "text"):
            raise ValueError(f"mode must be auto, ocr or text, not {mode!r}")
        self.ocr, self.dpi, self.mode = ocr, dpi, mode
        self.image_ocr = image_ocr

    def page_scale(self, page) -> float:
        """Pixels per point for this page: the set dpi, lowered for pages so
        large that the render would pass MAX_PAGE_PIXELS (posters, plans)."""
        w, h = page.get_size()
        k = self.dpi / 72.0
        return min(k, (MAX_PAGE_PIXELS / max(w * h, 1.0)) ** 0.5)

    def pdf_page(
        self,
        pdf_path: str,
        doc,
        index: int,
        password: str | None = None,
        rotation: int | None = None,
    ) -> Page:
        """Page ``index`` of the open PDF ``doc``.  rotation: as OrtOCR's, for the
        page if it is OCR'd (a page read from its text layer takes the turn of
        its characters)."""
        page = doc[index]
        if self.mode == "ocr":
            return self._pdf_page(page, None, 0.0, rotation)
        import pypdfium2 as pdfium

        # pdfium gives a character the box of its glyph outline only in a document
        # opened after it rendered that glyph, and the box of the font's metrics
        # before that.  Render the page, then read its text layer from a new
        # document, so the first read of a page gives the same boxes as later reads.
        t0 = time.perf_counter()
        check = _render(page, CHECK_DPI / 72.0)
        fresh = pdfium.PdfDocument(pdf_path, password=password)
        try:
            return self._pdf_page(fresh[index], check, t0, rotation)
        finally:
            fresh.close()

    def _pdf_page(
        self, page, check: np.ndarray | None, t0: float, rotation: int | None
    ) -> Page:
        """The page read from its text layer (check: its image at CHECK_DPI), or
        by OCR (check None, or the layer is not usable)."""
        t: dict = {}
        note = "mode=ocr"
        if check is not None:
            s = CHECK_DPI / 72.0
            tl = read_text_layer(page, check, s)
            # A table set sideways: most characters share one quarter turn.  Read
            # the page turned upright, as the OCR path does with orient.onnx.
            k = page_turn(tl.stats)
            rot = page.get_rotation()
            if k and (rot + 90 * k) % 360 == 0:
                # The page's /Rotate alone turns the text: read the page without
                # it.  pdfium builds its text runs in the displayed frame, and on
                # a turned page it can join a line's font runs out of order.
                page.set_rotation(0)
                upright = _render(page, s)
                tl = read_text_layer(page, upright, s)
                if tl.lines and (tl.trusted or self.mode == "text"):
                    tl.stats["turn"] = k
                    t["text_ms"] = 1000 * (time.perf_counter() - t0)
                    out = self._text_page(page, tl, t, check=self.mode == "auto")
                    if out is not None:
                        return out
                page.set_rotation(rot)
                tl = read_text_layer(
                    page, np.ascontiguousarray(np.rot90(check, -k)), s, k
                )
            elif k:
                tl = read_text_layer(
                    page, np.ascontiguousarray(np.rot90(check, -k)), s, k
                )
            elif vertical_layout(tl.stats):
                # Vertical writing: upright characters, read in the page turned a
                # quarter turn counter-clockwise (columns become lines).
                tl = read_text_layer(
                    page, np.ascontiguousarray(np.rot90(check, 1)), s, 0, 3
                )
                tl.stats["vertical"] = True
            t["text_ms"] = 1000 * (time.perf_counter() - t0)
            # A layer the recognizer cannot read beats OCR, unless its codes are junk.
            unreadable = tl.reason != BROKEN_FONT_MAP and not readable(
                self.ocr, tl.lines
            )
            note = tl.reason
            if tl.lines and (tl.trusted or self.mode == "text" or unreadable):
                # A trusted layer in a script the recognizer reads is checked
                # against what the page shows (legacy fonts store other codes).
                out = self._text_page(
                    page, tl, t, check=self.mode == "auto" and tl.trusted and not unreadable
                )
                if out is not None:
                    return out
                note = NOT_THE_PAGE
            # A layer with a broken font map still places its glyphs: when most
            # run upright, the page is upright (orient.onnx is not asked).
            st = tl.stats
            turn = not (
                tl.reason == BROKEN_FONT_MAP and st["rotated"] < 0.5 * st["chars"]
            )
        else:
            turn = True
        t0 = time.perf_counter()
        gray = _render(page, self.page_scale(page))
        t["render_ms"] = 1000 * (time.perf_counter() - t0)
        out = self.ocr(gray, turn, rotation)
        out.timings = {**t, **out.timings}
        out.timings["total_ms"] = sum(
            v for k, v in out.timings.items() if k != "total_ms"
        )
        out.note = f"{note}; {out.note}"
        return out

    def _text_page(self, page, tl, t: dict, check: bool = False) -> Page | None:
        """The page from its text layer.  check: first read the start of up to
        CHECK_LINES of the layer's lines from the page image (``agree.py``); when
        the layer is not what the recognizers see there (layer_matches), return
        None, so OCR reads the page.  A vertical page is not checked."""
        k = self.page_scale(page)
        turn = tl.stats.get("turn", 0)
        frame = tl.stats.get("frame", turn)
        vertical = bool(tl.stats.get("vertical"))
        t0 = time.perf_counter()
        gray = _render(page, k)
        if frame:
            gray = np.ascontiguousarray(np.rot90(gray, -frame))
        t["render_ms"] = 1000 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        boxes = self.ocr.detect(gray)
        t["det_ms"] = 1000 * (time.perf_counter() - t0)
        toks, lost = assign_runs(tl.lines, boxes[:, :4], k)
        # A rejected font run can share a detected line with sound runs. OCR
        # the whole affected line: keeping its remaining tokens would silently
        # omit words and prevent the empty-line fallback from reading them.
        rejected = getattr(tl, "rejected", [])
        repair = set()
        if rejected:
            bad, _ = assign_runs(rejected, boxes[:, :4], k)
            repair = {i for i, tk in enumerate(bad) if tk}
            if repair and (not self.image_ocr or not readable(self.ocr, tl.lines)):
                return None
        sym = getattr(tl, "symbol_boxes", None)
        if sym is not None and len(sym) and self.image_ocr and not vertical:
            t0 = time.perf_counter()
            fixed = self._fix_symbol_digits(gray, boxes, toks, sym * k, repair)
            t["symbol_ms"] = 1000 * (time.perf_counter() - t0)
            tl.stats["symbol_fixed"] = fixed
        rec: list = [None] * len(boxes)
        empty = []
        for i, tk in enumerate(toks):
            text = self.ocr._decode(tk) if tk and i not in repair else ""
            if text:
                rec[i] = (text, 1.0, tk)
            else:
                empty.append(i)
        if check and not vertical:
            t0 = time.perf_counter()
            full = [i for i, r in enumerate(rec) if r and sum(c.isalpha() for c in r[0]) >= 4]
            pick = full[:: max(1, len(full) // CHECK_LINES)][:CHECK_LINES]
            if pick:
                # The line halfway down first; the others only when it does not
                # clearly match, or when the layer dropped runs with bad codes
                # (a sign of a font map that may break other runs too).
                first = [len(pick) // 2]
                reads = self._layer_gaps(gray, boxes, toks, [pick[j] for j in first])
                if tl.stats.get("bad_runs") or any(
                    r[0] is not None and r[0] > EASY_GAP for r in reads
                ):
                    rest = [i for j, i in enumerate(pick) if j not in first]
                    reads += self._layer_gaps(gray, boxes, toks, rest)
                    t["check_ms"] = 1000 * (time.perf_counter() - t0)
                    if not layer_matches(reads):
                        return None
                else:
                    t["check_ms"] = 1000 * (time.perf_counter() - t0)
        source = "text"
        # Sideways runs (a rotated table header, a chart's axis label) come from
        # the layer as their own lines; detected lines on them are not OCR'd.
        side = [r for r in tl.rotated if r.text.strip()]
        side_boxes = np.array([[v * k for v in r.bbox] for r in side], np.float32)
        if len(side) and empty:
            empty = [i for i in empty if i in repair or not _covered(boxes[i, :4], side_boxes)]
        if empty and self.image_ocr and readable(self.ocr, tl.lines):
            t0 = time.perf_counter()
            for i, r in zip(
                empty, self.ocr.recognize(gray, boxes[empty], vertical=vertical)
            ):
                if r[0].strip():
                    rec[i] = r
                    source = "hybrid"
            t["rec_ms"] = 1000 * (time.perf_counter() - t0)
        if getattr(self.ocr, "math", None) is not None:
            t0 = time.perf_counter()
            self._page_source = source  # the read gate weighs text-layer and hybrid pages apart
            self._read_math(gray, boxes, rec, tl.lines, k)
            t["math_ms"] = 1000 * (time.perf_counter() - t0)
        keep = [i for i, r in enumerate(rec) if r is not None]
        kept_boxes = boxes[keep].reshape(-1, boxes.shape[1])
        kept_rec = [rec[i] for i in keep]
        if len(side):
            extra = np.zeros((len(side), boxes.shape[1]), np.float32)
            extra[:, :4] = side_boxes
            if boxes.shape[1] > 4:
                extra[:, 4] = 1.0
            kept_boxes = np.concatenate([kept_boxes, extra])
            for r, b in zip(side, side_boxes):
                cx = float(b[0] + b[2]) / 2  # one x: a sideways line is never cut
                kept_rec.append((r.text, 1.0, [(c, cx, r.seq) for c in r.text]))
        out = self.ocr.assemble(
            (gray.shape[1], gray.shape[0]),
            kept_boxes,
            kept_rec,
            t,
            vertical=vertical,
            gray=gray,
            from_layer=True,
        )
        out.source = source
        out.rotation = 90 * turn
        if vertical:  # back on the page as it is: (x, y) -> (W - y, x)
            W = gray.shape[0]

            def back(b):
                return [W - b[3], b[0], W - b[1], b[2]]

            for ln in out.lines:
                ln.box = back(ln.box)
                for w in ln.words or ():
                    w.box = back(w.box)
            for b in out.blocks:
                b["bbox"] = back(b["bbox"])
            out.vertical = True
        out.note += f"; {lost} chars outside detected lines" if lost else ""
        return out

    def _fix_symbol_digits(self, gray, boxes, toks: list, sym: np.ndarray, skip=()) -> int:
        """Check the layer digits that lie in a possible symbol glyph's box (``sym``,
        pixels) against the page and put the symbol in place of a digit where the
        recognizer supports it (SYMBOL_MARGIN); ``toks`` changes in place.  Returns
        the number of digits replaced."""
        lines = {}
        for i, tk in enumerate(toks):
            if tk and i not in skip:
                hits = [p for _, _, hs in symbol_words(tk, boxes[i, :4], sym) for p in hs]
                if hits:
                    lines[i] = hits
        if not lines:
            return 0
        idx = sorted(lines)
        # Each line is read in a box that holds the whole of every checked glyph: a
        # detected box can stop inside a line's last glyph, and a cut digit reads as
        # a symbol ('2' as '.').
        rbox = boxes[:, :4].astype(np.float32).copy()
        for i in idx:
            b = rbox[i]
            y0, y1 = float(b[1]), float(b[3])
            over = np.minimum(sym[:, 3], y1) - np.maximum(sym[:, 1], y0)
            near = sym[over > 0.5 * np.minimum(sym[:, 3] - sym[:, 1], y1 - y0)]
            xs = np.array([toks[i][p][1] for p in lines[i]], np.float32)
            near = near[((near[:, None, 0] - 1 <= xs) & (xs <= near[:, None, 2] + 1)).any(1)]
            if len(near):
                pad = 0.3 * (y1 - y0)
                b[0] = min(b[0], float(near[:, 0].min()) - pad)
                b[2] = max(b[2], float(near[:, 2].max()) + pad)
        rbox[:, [0, 2]] = np.clip(rbox[:, [0, 2]], 0, gray.shape[1] - 1)
        reads = self.ocr.recognize(gray, rbox[idx])
        cand_boxes, cand_toks, owners = [], [], []
        for i, (text, _, _) in zip(idx, reads):
            layer = "".join(t[0] if isinstance(t[0], str) else " " for t in toks[i])
            for p, c in symbol_candidates(layer, text, lines[i]).items():
                alt = list(toks[i])
                alt[p] = (c, *alt[p][1:])
                cand_boxes += [rbox[i], rbox[i]]
                cand_toks += [toks[i], alt]
                owners.append((i, p, c, len(layer.replace(" ", ""))))
        if not owners:
            return 0
        gaps = self.ocr.layer_gaps(gray, np.array(cand_boxes, np.float32), cand_toks)
        n = 0
        for k, (i, p, c, length) in enumerate(owners):
            g_layer, g_alt = gaps[2 * k][0], gaps[2 * k + 1][0]
            if g_layer is None or g_alt is None:
                continue
            if (g_layer - g_alt) * max(length, 1) >= SYMBOL_MARGIN:
                toks[i][p] = (c, *toks[i][p][1:])
                n += 1
        return n

    def _layer_gaps(self, gray, boxes, toks: list, lines: list[int]) -> list:
        """OrtOCR.layer_gaps, (gap, conf) per line, for the start (CHECK_CHARS) of
        these detected lines."""
        if not lines:
            return []
        cut = [snippet(toks[i], CHECK_CHARS) for i in lines]
        sb = boxes[lines, :4].copy()
        for b, (_, x) in zip(sb, cut):
            if x is not None:
                b[2] = min(b[2], x)
        return self.ocr.layer_gaps(gray, sb, [c[0] for c in cut])

    def _read_math(self, gray, boxes, rec: list, runs=None, k: float = 1.0) -> None:
        """Text-layer lines that hold math (the classifier, on their crops) are read
        by MathRec instead: the layer has the symbols but not the LaTeX.  The line
        keeps its place in the PDF content order (its first character's index).
        With math_span.onnx, inline math in the other lines is found too
        (_read_math_spans)."""
        from .line_mask import crop_lines
        from .math import latin_line

        idx = [
            i
            for i, r in enumerate(rec)
            if r is not None
            and r[2]
            and isinstance(r[2][0][0], str)
            and len(r[2][0]) > 2
            and latin_line(r[0])
        ]
        if not idx:
            return
        crops = crop_lines(gray, boxes[idx, :4], self.ocr.height, None)
        flags = self.ocr.math.flags(crops)
        math = [i for i, f in zip(idx, flags) if f]
        if math and self.ocr.math.gate_head is not None:
            probs = [p for p, f in zip(self.ocr.math.last_probs, flags) if f]
            keep = self.ocr.math.gate(probs, [rec[i][0] for i in math], len(idx), getattr(self, "_page_source", "text"))
            math = [i for i, k in zip(math, keep) if k]
        drafts = [rec[i][0] for i in math]
        read = self.ocr.math.read(gray, boxes[math], drafts=drafts) if math else []
        done = set()
        for i, (text, conf), draft in zip(math, read, drafts):
            # A reading without math, or not confident, leaves the layer's text;
            # one with math takes the layer's accents in its prose.
            merged = self.ocr.math.accept(text, conf, draft, 1.0)
            if merged is not None:
                self.ocr.math.accepted[merged] = (rec[i], text, conf, draft, 1.0)
                first = min(tk[2] for tk in rec[i][2])
                xc = float(boxes[i, 0] + boxes[i, 2]) / 2
                rec[i] = (merged, 1.0, [(merged, xc, first)])
                done.add(i)
        self._read_math_spans(gray, boxes, rec, [i for i in idx if i not in done], runs, k)

    def _read_math_spans(self, gray, boxes, rec: list, idx: list, runs, k: float) -> None:
        """Inline math in text-layer lines: the span tagger (math_span.py) marks the
        characters of math in each line, MathRec reads each marked run's crop, and a
        confident reading with math replaces the run's characters in the line's
        text.  Each character's box comes from its text-layer run."""
        tagger = getattr(self.ocr, "math_span", None)
        if tagger is None or runs is None or not idx:
            return
        from .math import MATH_MIN_CONF, accept_math, has_structure

        by = {(r.seq if r.seq is not None else pos): r for pos, r in enumerate(runs)}
        jobs = []  # (line, span start, span end, crop box, draft)
        lines = {}
        index: dict = {}
        for i in idx:
            chars = _line_chars(rec[i][2], by, k, index)
            if not chars:
                continue
            text = rec[i][0]
            pos = [p for p, c in enumerate(text) if not c.isspace()]
            if len(pos) == len(chars):
                lines[i] = (text, pos, chars)
        for i, found in zip(lines, tagger.spans([v[2] for v in lines.values()])):
            chars = lines[i][2]
            for a, b in found:
                # 3 px of margin, but not past the middle of the gap to the next
                # character: MathRec would read it too, and the line keeps it.
                x0 = min(c[1] for c in chars[a:b]) - 3
                x1 = max(c[3] for c in chars[a:b]) + 3
                if a > 0:
                    x0 = max(x0, (chars[a - 1][3] + chars[a][1]) / 2)
                if b < len(chars):
                    x1 = min(x1, (chars[b - 1][3] + chars[b][1]) / 2)
                draft = "".join(c[0] for c in chars[a:b])
                jobs.append((i, a, b, [max(0.0, x0), float(boxes[i, 1]), x1, float(boxes[i, 3])], draft))
        if not jobs:
            return
        read = self.ocr.math.read(
            gray, np.array([j[3] for j in jobs], np.float32), drafts=[j[4] for j in jobs]
        )
        repl: dict = {}
        for (i, a, b, _, draft), (text, conf) in zip(jobs, read):
            merged = accept_math(text, draft) if conf >= MATH_MIN_CONF else None
            # The layer already has the run's symbols: only structure is worth it.
            if merged is not None and has_structure(merged):
                repl.setdefault(i, []).append((a, b, merged))
        for i, spans in repl.items():
            text, pos, _ = lines[i]
            for a, b, merged in sorted(spans, reverse=True):
                text = text[: pos[a]] + merged + text[pos[b - 1] + 1 :]
            first = min(tk[2] for tk in rec[i][2])
            xc = float(boxes[i, 0] + boxes[i, 2]) / 2
            rec[i] = (text, 1.0, [(text, xc, first)])

    def convert(
        self,
        path: str | Path,
        max_pages: int = 0,
        password: str | None = None,
        rotation: int | dict[int, int] | None = None,
    ):
        """Yield (page index, Page) for a PDF or an image file (the file's bytes
        decide which).  A page that fails yields an empty Page whose ``source``
        is "error" and whose ``note`` says why; a file that cannot be opened
        raises InputError.  max_pages: 0 reads every page; a negative value
        raises ValueError.  rotation: the known turn of the pages' content, in
        degrees counter-clockwise (0, 90, 180, 270), for every page or as
        {page index: degrees}; an OCR'd page with a known turn skips orient.onnx."""
        if max_pages < 0:
            raise ValueError(
                f"max_pages must be 0 (all pages) or more, not {max_pages}"
            )
        rot = _page_rotations(rotation)
        path = Path(path)
        if not path.is_file():
            raise InputError(f"{path}: no such file")
        if not is_pdf(path):
            for i, gray in enumerate(read_images(path)[: max_pages or None]):
                yield i, self._safe(lambda: self.ocr(gray, rotation=rot(i)))
            return
        import pypdfium2 as pdfium

        try:
            doc = pdfium.PdfDocument(str(path), password=password)
        except pdfium.PdfiumError as e:
            locked = "password" in str(e).lower()
            raise InputError(
                f"{path}: password-protected PDF (pass password=, or --password)"
                if locked
                else f"{path}: cannot open the PDF ({e})"
            ) from e
        try:
            n = len(doc) if not max_pages else min(len(doc), max_pages)
            for i in range(n):
                yield i, self._safe(
                    lambda: self.pdf_page(str(path), doc, i, password, rot(i))
                )
        finally:
            doc.close()

    @staticmethod
    def _safe(read) -> Page:
        try:
            return read()
        except Exception as e:  # noqa: BLE001  (one bad page must not end the document)
            return Page(
                [],
                {"total_ms": 0.0},
                markdown="",
                source="error",
                note=f"{type(e).__name__}: {e}",
            )
