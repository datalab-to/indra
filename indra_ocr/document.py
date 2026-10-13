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

from .pipeline import InputError, OrtOCR, Page, is_pdf, read_images
from .textlayer import BROKEN_FONT_MAP, read_text_layer

CHECK_DPI = 72.0  # resolution of the render for the text-layer ink checks
MAX_PAGE_PIXELS = 100e6  # a page larger than this at the set dpi renders at a lower dpi (large scans reach ~65 MP)

READABLE_MIN = 0.9  # share of the layer's letters the recognizer must know


def readable(ocr: OrtOCR, runs: list) -> bool:
    """Whether the recognizer can read the script of these text-layer runs."""
    letters = [c for r in runs for c in r.text if c.isalpha()]
    if not letters:
        return True
    cjk = ocr.vocab_cjk.index if ocr.vocab_cjk is not None else {}
    known = sum(c in ocr.vocab.index or c in cjk for c in letters)
    return known >= READABLE_MIN * len(letters)


def right_to_left(tokens: list) -> bool:
    """Unicode bidi base direction: most strong characters are right-to-left."""
    rtl = ltr = 0
    for t in tokens:
        b = unicodedata.bidirectional(t[0]) if len(t[0]) == 1 else ""
        rtl += b in ("R", "AL")
        ltr += b == "L"
    return rtl > ltr


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
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for idx, run in enumerate(runs):
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


class Converter:
    def __init__(
        self,
        ocr: OrtOCR,
        dpi: float = 200.0,
        mode: str = "auto",
        pdf_order: bool = True,
        image_ocr: bool = True,
    ):
        """mode: auto (text layer where sound, OCR elsewhere), ocr (always OCR,
        never read the text layer), text (text layer even when the checks fail).
        pdf_order: keep the PDF content order when it is coherent (False: the
        order model always orders the lines).  image_ocr: OCR detected lines that
        hold no text-layer characters (False: drop them)."""
        if mode not in ("auto", "ocr", "text"):
            raise ValueError(f"mode must be auto, ocr or text, not {mode!r}")
        self.ocr, self.dpi, self.mode = ocr, dpi, mode
        self.pdf_order, self.image_ocr = pdf_order, image_ocr

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
            tl = read_text_layer(page, _render(page, s), s)
            t["text_ms"] = 1000 * (time.perf_counter() - t0)
            # A layer the recognizer cannot read beats OCR, unless its codes are junk.
            unreadable = tl.reason != BROKEN_FONT_MAP and not readable(
                self.ocr, tl.lines
            )
            if tl.lines and (tl.trusted or self.mode == "text" or unreadable):
                return self._text_page(page, tl, t)
            note = tl.reason
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

    def _text_page(self, page, tl, t: dict) -> Page:
        k = self.page_scale(page)
        t0 = time.perf_counter()
        gray = _render(page, k)
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
        source = "text"
        if empty and self.image_ocr and readable(self.ocr, tl.lines):
            t0 = time.perf_counter()
            for i, r in zip(empty, self.ocr.recognize(gray, boxes[empty])):
                if r[0].strip():
                    rec[i] = r
                    source = "hybrid"
            t["rec_ms"] = 1000 * (time.perf_counter() - t0)
        if getattr(self.ocr, "math", None) is not None:
            t0 = time.perf_counter()
            self._read_math(gray, boxes, rec)
            t["math_ms"] = 1000 * (time.perf_counter() - t0)
        keep = [i for i, r in enumerate(rec) if r is not None]
        out = self.ocr.assemble(
            (gray.shape[1], gray.shape[0]),
            boxes[keep].reshape(-1, boxes.shape[1]),
            [rec[i] for i in keep],
            t,
            content_order=self.pdf_order,
        )
        out.source = source
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
        raises InputError."""
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
