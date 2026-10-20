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

import time
import unicodedata
from pathlib import Path

import numpy as np

from . import native
from .pipeline import InputError, OrtOCR, Page, is_pdf, read_images
from .textlayer import BROKEN_FONT_MAP, page_turn, read_text_layer, vertical_layout

CHECK_DPI = 72.0  # resolution of the render for the text-layer ink checks
MAX_PAGE_PIXELS = 100e6  # a page larger than this at the set dpi renders at a lower dpi (large scans reach ~65 MP)

READABLE_MIN = 0.9  # share of the layer's letters the recognizer must know
AGREE_LINES = 6  # text-layer lines whose script is checked against the page image
AGREE_SHARE = 0.25  # under this share of agreeing lines, the layer is not the page's text
LATIN_CODES = 0.9  # a layer line this Latin while its image is another script: legacy font codes
NOT_THE_PAGE = "text layer does not match the page"  # Page.note when the check sends a page to OCR


def latin_share(text: str) -> float:
    """Share of the letters in text that are basic Latin (a-z, A-Z)."""
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / max(len(letters), 1)


def layer_agrees(agree: list) -> bool:
    """Whether a text layer matches its page: at least AGREE_SHARE of the checked
    lines agree.  A legacy font (Kruti Dev and DevLys for Hindi, Bijoy for Bangla)
    stores Latin codes for another script, so almost every line disagrees."""
    return not agree or sum(agree) >= AGREE_SHARE * len(agree)


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


def _render(page, scale: float) -> np.ndarray:
    arr = page.render(scale=scale, grayscale=True).to_numpy()
    return np.ascontiguousarray(arr[..., 0] if arr.ndim == 3 else arr)


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
    for pos, run in enumerate(runs):
        idx = run.seq if run.seq is not None else pos
        j = None
        for c, b in run.chars:
            if not c.isspace():
                cx, cy = (b[0] + b[2]) / 2 * k, (b[1] + b[3]) / 2 * k
                inside = (
                    (boxes[:, 0] <= cx)
                    & (cx <= boxes[:, 2])
                    & (boxes[:, 1] <= cy)
                    & (cy <= boxes[:, 3])
                )
                if not inside.any():
                    lost += 1
                    j = None
                    continue
                j = int(np.flatnonzero(inside)[np.argmin(area[inside])])
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

    def pdf_page(self, pdf_path: str, doc, index: int) -> Page:
        page = doc[index]
        t: dict = {}
        note = "mode=ocr"
        if self.mode != "ocr":
            t0 = time.perf_counter()
            s = CHECK_DPI / 72.0
            check = _render(page, s)
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
        out = self.ocr(gray, turn)
        out.timings = {**t, **out.timings}
        out.timings["total_ms"] = sum(
            v for k, v in out.timings.items() if k != "total_ms"
        )
        out.note = f"{note}; {out.note}"
        return out

    def _text_page(self, page, tl, t: dict, check: bool = False) -> Page | None:
        """The page from its text layer.  check: first compare the script of up to
        AGREE_LINES of the layer's lines with the script the line classifier sees
        in their images; when they do not match (layer_agrees), return None, so OCR
        reads the page."""
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
        rec: list = [None] * len(boxes)
        empty = []
        for i, tk in enumerate(toks):
            text = self.ocr._decode(tk) if tk else ""
            if text:
                rec[i] = (text, 1.0, tk)
            else:
                empty.append(i)
        if check:
            t0 = time.perf_counter()
            full = [i for i, r in enumerate(rec) if r and sum(c.isalpha() for c in r[0]) >= 4]
            pick = full[:: max(1, len(full) // AGREE_LINES)][:AGREE_LINES]
            if pick:
                # A line agrees unless its layer text is Latin while the router is
                # sure its image is another script.
                groups, sure = self.ocr.box_groups(gray, boxes[pick])
                agree = [
                    not (latin_share(rec[i][0]) >= LATIN_CODES and g != "base" and s)
                    for i, g, s in zip(pick, groups, sure)
                ]
                t["check_ms"] = 1000 * (time.perf_counter() - t0)
                if not layer_agrees(agree):
                    return None
        source = "text"
        # Sideways runs (a rotated table header, a chart's axis label) come from
        # the layer as their own lines; detected lines on them are not OCR'd.
        side = [r for r in tl.rotated if r.text.strip()]
        side_boxes = np.array([[v * k for v in r.bbox] for r in side], np.float32)
        if len(side) and empty:
            empty = [i for i in empty if not _covered(boxes[i, :4], side_boxes)]
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
            self._read_math(gray, boxes, rec)
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
        )
        out.source = source
        out.rotation = 90 * turn
        if vertical:  # back on the page as it is: (x, y) -> (W - y, x)
            W = gray.shape[0]

            def back(b):
                return [W - b[3], b[0], W - b[1], b[2]]

            for ln in out.lines:
                ln.box = back(ln.box)
            for b in out.blocks:
                b["bbox"] = back(b["bbox"])
            out.vertical = True
        out.note += f"; {lost} chars outside detected lines" if lost else ""
        return out

    def _read_math(self, gray, boxes, rec: list) -> None:
        """Text-layer lines that hold math (the classifier, on their crops) are read
        by MathRec instead: the layer has the symbols but not the LaTeX.  The line
        keeps its place in the PDF content order (its first character's index)."""
        from .line_mask import crop_lines

        idx = [
            i
            for i, r in enumerate(rec)
            if r is not None
            and r[2]
            and isinstance(r[2][0][0], str)
            and len(r[2][0]) > 2
        ]
        if not idx:
            return
        crops = crop_lines(gray, boxes[idx, :4], self.ocr.height, None)
        math = [i for i, f in zip(idx, self.ocr.math.flags(crops)) if f]
        if not math:
            return
        from .math import MATH_MIN_CONF

        for i, (text, conf) in zip(math, self.ocr.math.read(gray, boxes[math])):
            # A reading without math, or not confident, leaves the layer's text.
            if "$" in text and conf >= MATH_MIN_CONF:
                first = min(tk[2] for tk in rec[i][2])
                xc = float(boxes[i, 0] + boxes[i, 2]) / 2
                rec[i] = (text, 1.0, [(text, xc, first)])

    def convert(
        self, path: str | Path, max_pages: int = 0, password: str | None = None
    ):
        """Yield (page index, Page) for a PDF or an image file (the file's bytes
        decide which).  A page that fails yields an empty Page whose ``source``
        is "error" and whose ``note`` says why; a file that cannot be opened
        raises InputError.  max_pages: 0 reads every page; a negative value
        raises ValueError."""
        if max_pages < 0:
            raise ValueError(
                f"max_pages must be 0 (all pages) or more, not {max_pages}"
            )
        path = Path(path)
        if not path.is_file():
            raise InputError(f"{path}: no such file")
        if not is_pdf(path):
            for i, gray in enumerate(read_images(path)[: max_pages or None]):
                yield i, self._safe(lambda: self.ocr(gray))
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
                yield i, self._safe(lambda: self.pdf_page(str(path), doc, i))
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
