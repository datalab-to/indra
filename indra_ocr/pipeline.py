"""The OCR pipeline on ONNX Runtime: line detection, CTC line recognition,
reading order and layout classes.  CPU only, no PyTorch.

The model folder holds det.onnx, rec.onnx (or rec.int8.onnx), order.onnx and
meta.json (see ``indra_ocr.models``).  With script.onnx and rec_cjk.onnx, each
line goes to the recognizer for its script: lines with Chinese characters to
rec_cjk.onnx (which also reads Latin), the rest to rec.onnx.  With orient.onnx,
a page turned by a quarter or half turn is turned upright before detection
(``Page.rotation``; boxes are in the upright page).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .det_post import decode, page_to_input
from .layout_post import page_blocks, page_markdown
from .line_mask import crop_lines
from .math import MATH_MIN_CONF, MathReader
from .order_post import (
    BLOCK_CLASSES,
    decode_blocks_order,
    decode_order,
    line_features,
    order_xycut,
)
from .rec_preproc import PAD_X, pad_batch, width_batches
from .textlayer import pdf_order_coherent
from .vocab import Vocab

ADAPTIVE_MIN_H = (
    9.0  # median line height (input px) that triggers a second detection pass
)
ADAPTIVE_TARGET_H = 12.0
# Gaps between CTC character positions (centres), in line heights.  The label
# rules (2.5 and 1.0 line heights) measure glyph edges; centres add about half a
# line height of glyph width.
SPLIT_GAP = 3.0  # cut a detected line at a wider gap (label rule outside tables)
CELL_GAP = 1.8  # cut a table line into cells at a wider gap (label rule in tables)
GUTTER_GAP = 1.6  # cut at a smaller gap when other lines' edges line up with it (a column gutter)
GUTTER_EDGES = 3  # how many other line edges must line up (within half a line height)
ADAPTIVE_MAX_SIDE = 2048
ORIENT_SIDE = 768  # orient.onnx input: page thumbnail, long side, padded square
# A turn is applied only above this probability: real pages are almost all upright,
# and a wrong turn loses the page.  Measured (orient_v1): 2 of 1,000 upright eval
# pages turned at 0.95 (13 at 0.5), 101 of the 164 Omni Parse Bench pages with
# rotated tests turned (128 at 0.5).
ORIENT_MIN = 0.95


# A recognized line under this confidence (mean CTC peak probability) gets no
# text: on empty form fields, checkboxes and rules the recognizer reads "-", "I"
# or "1" at low confidence.  The line's box stays (layout and reading order).
MIN_CONF = 0.5

DESKEW_MAX = 5.0  # degrees of tilt searched
DESKEW_MIN = 0.3  # smaller tilts are left alone (the search step is 0.1)


def estimate_skew(gray: np.ndarray, side: int = 800) -> float:
    """Degrees (counter-clockwise) that straighten the text rows of a page: the
    angle whose horizontal ink profile changes most sharply row to row, on a
    page thumbnail, searched in 0.5 degree steps and then 0.1 degree steps."""
    h, w = gray.shape
    s = side / max(h, w)
    small = cv2.resize(
        gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA
    )
    ink = (255.0 - small.astype(np.float32)) / 255.0
    if ink.mean() < 0.003:  # a blank page has no rows to straighten
        return 0.0
    size = (small.shape[1], small.shape[0])
    c = (size[0] / 2, size[1] / 2)

    def sharpness(a: float) -> float:
        r = cv2.warpAffine(ink, cv2.getRotationMatrix2D(c, a, 1.0), size)
        p = r.sum(1)
        return float(((p[1:] - p[:-1]) ** 2).sum())

    best = max(np.arange(-DESKEW_MAX, DESKEW_MAX + 1e-6, 0.5), key=sharpness)
    return float(max(np.arange(best - 0.5, best + 0.5 + 1e-6, 0.1), key=sharpness))


def rotate_page(gray: np.ndarray, degrees: float) -> np.ndarray:
    """The page rotated by ``degrees`` counter-clockwise about its centre (same size)."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def unrotate_box(box, degrees: float, wh: tuple[int, int]) -> list[float]:
    """A box of the page that rotate_page(.., degrees) made, mapped back to the
    page before it: the bounding box of the four mapped corners, clipped to it."""
    w, h = wh
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -degrees, 1.0)
    x0, y0, x1, y1 = box[:4]
    c = M @ np.array([[x0, x1, x0, x1], [y0, y0, y1, y1], [1, 1, 1, 1]], float)
    return [
        round(float(np.clip(c[0].min(), 0, w)), 1),
        round(float(np.clip(c[1].min(), 0, h)), 1),
        round(float(np.clip(c[0].max(), 0, w)), 1),
        round(float(np.clip(c[1].max(), 0, h)), 1),
    ]


def orient_input(gray: np.ndarray, side: int = ORIENT_SIDE) -> np.ndarray:
    """[side, side] float32 ink map (1 = ink): long side scaled to ``side``, padded."""
    h, w = gray.shape
    s = side / max(h, w)
    small = cv2.resize(
        gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA
    )
    out = np.zeros((side, side), np.float32)
    out[: small.shape[0], : small.shape[1]] = 1.0 - small.astype(np.float32) / 255.0
    return out


@dataclass
class Line:
    box: list[float]  # x0, y0, x1, y1 in image pixels
    text: str
    conf: float


@dataclass
class Page:
    lines: list[Line]  # reading order
    timings: dict = field(default_factory=dict)
    markdown: str | None = None  # with an order model that has layout heads
    source: str = (
        "ocr"  # ocr | text (PDF text layer) | hybrid (text layer + OCR) | error
    )
    note: str = ""  # why the page took this path
    # layout blocks in reading order: {"label", "bbox" (pixels), "text"}, with an
    # order model that has layout heads
    blocks: list = field(default_factory=list)
    # degrees counter-clockwise the input was turned; boxes are upright
    rotation: int = 0
    # degrees of small tilt undone after the turn (OCR reads the straightened
    # page; boxes are mapped back to the turned page)
    skew: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)


class OrtOCR:
    def __init__(
        self,
        onnx_dir: str | Path,
        threads: int = 4,
        rec_int8: bool | None = None,
        adaptive: bool = True,
        table_heads: bool = False,
        providers: list | None = None,
        furniture: bool = False,
        math: bool = False,
    ):
        """table_heads: build tables from the order model's row / column / cell
        heads (when it has them) instead of from geometry.  furniture: keep page
        headers and footers in the markdown (dropped by default).  math: read lines
        with math as LaTeX (math.py; needs the math models; off by default).  providers: ONNX
        Runtime execution providers (default CPU; e.g. ["CUDAExecutionProvider",
        "CPUExecutionProvider"] with onnxruntime-gpu)."""
        self.table_heads = table_heads
        self.furniture = furniture
        import onnxruntime as ort

        onnx_dir = Path(onnx_dir)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # No CPU memory arena: with line batches of changing shape it kept growing
        # (onnxruntime 1.31: rec.onnx +1.4 GB over 10 identical runs; a 40-page
        # scan peaked at 4.9 GB), and it bought no measurable speed.
        so.enable_cpu_mem_arena = False
        prov = providers or ["CPUExecutionProvider"]

        def need(name: str) -> str:
            p = onnx_dir / name
            if not p.exists():
                raise FileNotFoundError(p)
            return str(p)

        if rec_int8 is None:
            rec_int8 = (onnx_dir / "rec.int8.onnx").exists()
        self.det = ort.InferenceSession(need("det.onnx"), so, providers=prov)
        self.rec = ort.InferenceSession(
            need("rec.int8.onnx" if rec_int8 else "rec.onnx"), so, providers=prov
        )
        self.mask = (
            ort.InferenceSession(str(onnx_dir / "mask.onnx"), so, providers=prov)
            if (onnx_dir / "mask.onnx").exists()
            else None
        )
        self.order = (
            ort.InferenceSession(str(onnx_dir / "order.onnx"), so, providers=prov)
            if (onnx_dir / "order.onnx").exists()
            else None
        )
        self.orient = (
            ort.InferenceSession(str(onnx_dir / "orient.onnx"), so, providers=prov)
            if (onnx_dir / "orient.onnx").exists()
            else None
        )
        meta = json.loads(Path(need("meta.json")).read_text())
        self.vocab = Vocab.from_json(meta["vocab"])
        # Chinese: a line classifier routes lines with hanzi to their own recognizer.
        self.script = self.rec_cjk = self.vocab_cjk = None
        if (onnx_dir / "script.onnx").exists() and (onnx_dir / "rec_cjk.onnx").exists():
            self.script = ort.InferenceSession(
                str(onnx_dir / "script.onnx"), so, providers=prov
            )
            self.rec_cjk = ort.InferenceSession(
                str(onnx_dir / "rec_cjk.onnx"), so, providers=prov
            )
            self.vocab_cjk = Vocab.from_json(meta["vocab_cjk"])
        # Math: a line classifier routes lines with math to MathRec (math.py).
        self.math = MathReader.load(onnx_dir, so, prov) if math else None
        if math and self.math is None:
            raise FileNotFoundError(f"math=True needs the math models in {onnx_dir}")
        self.long_side = meta["det_long_side"]
        self.height = meta["rec_height"]
        self.adaptive = adaptive

    def _detect_at(self, gray: np.ndarray, long_side: int):
        x, s = page_to_input(gray, long_side)
        return decode(self.det.run(None, {"x": x[None, None]})[0][0]), s

    def detect(self, gray: np.ndarray) -> np.ndarray:
        boxes, s = self._detect_at(gray, self.long_side)
        if self.adaptive and len(boxes) >= 8:
            med_h = float(np.median(boxes[:, 3] - boxes[:, 1]))
            if med_h < ADAPTIVE_MIN_H:
                ls = (
                    int(
                        min(
                            ADAPTIVE_MAX_SIDE,
                            self.long_side * ADAPTIVE_TARGET_H / max(med_h, 1.0),
                        )
                    )
                    // 32
                    * 32
                )
                if ls > self.long_side:
                    boxes, s = self._detect_at(gray, ls)
        if len(boxes):
            boxes[:, :4] /= s
            h, w = gray.shape
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes

    def recognize(
        self, gray: np.ndarray, boxes: np.ndarray, batch: int = 64
    ) -> list[tuple[str, float, list]]:
        """(text, confidence, tokens) per line; tokens = [(token id, page x)],
        x from the CTC peak's timestep mapped back through the crop.  A line
        under MIN_CONF comes back empty."""
        # CenterMask first when the model folder has mask.onnx (same step as training)
        crops = crop_lines(gray, boxes[:, :4], self.height, self.mask)
        out: list = [("", 0.0, [])] * len(crops)
        groups = [(self.rec, self.vocab, list(range(len(crops))), False)]
        if self.script is not None and crops:
            hanzi = self.hanzi_lines(crops, batch)
            groups = [
                (
                    self.rec,
                    self.vocab,
                    [j for j, h in enumerate(hanzi) if not h],
                    False,
                ),
                (
                    self.rec_cjk,
                    self.vocab_cjk,
                    [j for j, h in enumerate(hanzi) if h],
                    True,
                ),
            ]
        for sess, vocab, sel, as_chars in groups:
            for part in width_batches([crops[j].shape[1] for j in sel], batch):
                idx = [sel[i] for i in part]
                arr, widths = pad_batch([crops[j] for j in idx])
                best, pmax, lengths = sess.run(None, {"x": arr, "widths": widths})
                stride = arr.shape[-1] / max(best.shape[1], 1)
                for k, j in enumerate(idx):
                    n = int(lengths[k])
                    seq, pm = best[k, :n], pmax[k, :n]
                    keep = (seq != 0) & np.concatenate([[True], seq[1:] != seq[:-1]])
                    x0, y0, x1, y1 = (float(v) for v in boxes[j, :4])
                    h = max(1.0, y1 - y0)
                    ix0 = np.floor(x0 - PAD_X * h)
                    cw = np.ceil(x1 + PAD_X * h) - ix0
                    f = cw / crops[j].shape[1]
                    t = np.nonzero(keep)[0]
                    xs = (ix0 + (t + 0.5) * stride * f).tolist()
                    ids = seq[t].tolist()
                    # The CJK recognizer's ids mean nothing in self.vocab: its tokens
                    # are characters, like the text-layer tokens.
                    toks = (
                        list(zip([vocab.chars[i - 1] for i in ids], xs))
                        if as_chars
                        else list(zip(ids, xs))
                    )
                    conf = float(pm[keep].mean()) if keep.any() else 0.0
                    out[j] = (
                        (vocab.decode(seq[keep].tolist()), conf, toks)
                        if conf >= MIN_CONF
                        else ("", conf, [])
                    )
        if getattr(self, "math", None) is not None and crops:
            math = [j for j, f in enumerate(self.math.flags(crops)) if f]
            if math:
                for j, (text, conf) in zip(math, self.math.read(gray, boxes[math])):
                    # Only a confident reading with math replaces the line (one
                    # token holds the whole line: it is never split).
                    if "$" in text and conf >= MATH_MIN_CONF:
                        xc = float(boxes[j, 0] + boxes[j, 2]) / 2
                        out[j] = (text, 1.0, [(text, xc)])
        return out

    def hanzi_lines(self, crops: list, batch: int = 64) -> list[bool]:
        """Per line crop: does it hold Chinese characters (script.onnx)?"""
        out = [False] * len(crops)
        for idx in width_batches([c.shape[1] for c in crops], batch):
            arr, widths = pad_batch([crops[j] for j in idx])
            p = self.script.run(None, {"x": arr, "widths": widths})[0]
            for k, j in enumerate(idx):
                out[j] = bool(p[k] > 0.5)
        return out

    def gutter_cuts(self, boxes: np.ndarray, rec: list) -> list[list[float]]:
        """Per line, the x positions of gaps that sit on a column gutter: a gap over
        GUTTER_GAP line heights where at least GUTTER_EDGES other lines end at its
        left side or start at its right side."""
        lefts, rights = boxes[:, 0], boxes[:, 2]
        cuts: list[list[float]] = []
        for i, (b, (_, _, toks)) in enumerate(zip(boxes, rec)):
            h = max(1.0, float(b[3] - b[1]))
            xs = [tk[1] for tk in toks if not self._is_space(tk[0])]
            mine = []
            for a, c in zip(xs[:-1], xs[1:]):
                if c - a <= GUTTER_GAP * h:
                    continue
                gl, gr = a + 0.35 * h, c - 0.35 * h  # gap edges
                near = (np.abs(rights - gl) < 0.5 * h) | (np.abs(lefts - gr) < 0.5 * h)
                near[i] = False
                if int(near.sum()) >= GUTTER_EDGES:
                    mine.append((a + c) / 2)
            cuts.append(mine)
        return cuts

    def _is_space(self, tok) -> bool:
        return tok == " " or tok == self.vocab.index.get(" ")

    def _decode(self, toks: list) -> str:
        """Text of tokens: (vocab id, x) from the recognizer, or (char, x, content
        index) from a PDF text layer."""
        if toks and isinstance(toks[0][0], str):
            return " ".join("".join(tk[0] for tk in toks).split())
        return self.vocab.decode([tk[0] for tk in toks])

    def split_line(
        self, box, toks: list, gap: float, at: list | None = None
    ) -> list[tuple[list, str, list]]:
        """Cut a line where two neighbouring characters are more than ``gap``
        line heights apart (the detector merged two cells or columns).
        Returns [(box, text, tokens)]; one item when nothing is cut."""
        h = max(1.0, float(box[3] - box[1]))
        segs: list[list] = [[]]
        prev = None
        at = sorted(at or [])
        for tk in toks:
            x = tk[1]
            if not self._is_space(tk[0]):
                cut = prev is not None and (
                    x - prev > gap * h or any(prev < c < x for c in at)
                )
                if cut:
                    segs.append([])
                prev = x
            segs[-1].append(tk)
        if len(segs) == 1:
            return [(list(box[:4]), self._decode(toks), toks)]
        out = []
        for sg in segs:
            text = self._decode(sg).strip()
            xs = [tk[1] for tk in sg if not self._is_space(tk[0])]
            if text and xs:
                bx = [
                    max(box[0], min(xs) - 0.35 * h),
                    box[1],
                    min(box[2], max(xs) + 0.35 * h),
                    box[3],
                ]
                out.append((bx, text, sg))
        return out

    def reading_order(
        self, boxes: np.ndarray, texts: list[str], wh: tuple[int, int]
    ) -> tuple[list[int], tuple | None, np.ndarray | None]:
        """Reading order, (cls logits, same-block logits, table logits) when the
        order model has layout heads, and the pairwise order logits."""
        if len(boxes) < 2 or self.order is None:
            order = order_xycut(boxes[:, :4]) if len(boxes) else []
            return order, None, None
        f = line_features(boxes[:, :4], wh, texts)[None]
        outs = self.order.run(
            None, {"feats": f, "mask": np.ones((1, len(boxes)), bool)}
        )
        if len(outs) >= 3:
            # Layout heads: blocks from the same-block head, lines inside a block by
            # geometry, blocks by the order head (exact page order 0.68 -> 0.81 on eval).
            cls, same = outs[1][0], outs[2][0]
            # same row / column / cell logits, when asked for
            tab = outs[3][0] if len(outs) > 3 and self.table_heads else None
            order = decode_blocks_order(outs[0][0], same, boxes[:, :4], cls)
            return order, (cls, same, tab), outs[0][0]
        return decode_order(outs[0][0]), None, outs[0][0]

    def orientation(self, gray: np.ndarray) -> int:
        """Quarter turns (counter-clockwise) the page content is turned by, from
        orient.onnx: 0 without the model or when no turn is more likely than ORIENT_MIN."""
        if self.orient is None:
            return 0
        p = self.orient.run(None, {"x": orient_input(gray)[None, None]})[0][0]
        k = int(p.argmax())
        return k if p[k] > ORIENT_MIN else 0

    def upright(
        self, gray: np.ndarray, turn: bool = True
    ) -> tuple[np.ndarray, int, float]:
        """The page turned upright (orient.onnx) and straightened (estimate_skew):
        (page, quarter turns undone, degrees of skew undone)."""
        k = self.orientation(gray) if turn else 0
        if k:
            gray = np.ascontiguousarray(np.rot90(gray, -k))
        a = estimate_skew(gray)
        if abs(a) >= DESKEW_MIN:
            gray = rotate_page(gray, a)
        else:
            a = 0.0
        return gray, k, a

    def __call__(self, gray: np.ndarray, turn: bool = True) -> Page:
        """OCR of a page image.  turn=False: the page is known to be upright (no
        orient.onnx)."""
        t = {}
        t0 = time.perf_counter()
        gray, k, skew = self.upright(gray, turn)
        t["orient_ms"] = 1000 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        boxes = self.detect(gray)
        t["det_ms"] = 1000 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        rec = self.recognize(gray, boxes)
        t["rec_ms"] = 1000 * (time.perf_counter() - t0)
        page = self.assemble((gray.shape[1], gray.shape[0]), boxes, rec, t)
        page.rotation, page.skew = 90 * k, skew
        if skew:  # boxes back on the page as turned, before the deskew
            wh = (gray.shape[1], gray.shape[0])
            for ln in page.lines:
                ln.box = unrotate_box(ln.box, skew, wh)
            for b in page.blocks:
                b["bbox"] = unrotate_box(b["bbox"], skew, wh)
        return page

    def assemble(
        self, wh, boxes: np.ndarray, rec: list, t: dict, content_order: bool = False
    ) -> Page:
        """Lines -> split at wide gaps -> reading order -> markdown.

        ``rec`` holds (text, confidence, tokens) per detected line.  With
        ``content_order`` and text-layer tokens (char, x, content index) on every
        line, the PDF content order is kept when it is coherent."""
        t0 = time.perf_counter()
        # Text-layer labels split a line at wide gaps; do the same (SPLIT_GAP).
        nb, nrec = [], []
        cuts = self.gutter_cuts(boxes, rec) if len(boxes) else []
        for b, (text, conf, toks), at in zip(boxes, rec, cuts):
            for bx, tx, tk in self.split_line(b, toks, SPLIT_GAP, at):
                nb.append(bx)
                nrec.append((tx, conf, tk))
        boxes = np.array(nb, np.float32).reshape(-1, 4)
        rec = nrec
        texts = [r[0] for r in rec]
        order, layout, logits = self.reading_order(boxes, texts, wh)
        if layout is not None and layout[2] is not None:
            # The table heads learned one line per cell, but the detector can join
            # cells: split table lines into cells, then run the order model again.
            table_cls = BLOCK_CLASSES.index("Table")
            is_table = layout[0].argmax(-1) == table_cls
            if is_table.any():
                nb, nrec = [], []
                for i, (b, r) in enumerate(zip(boxes, rec)):
                    parts = (
                        self.split_line(b, r[2], CELL_GAP)
                        if is_table[i]
                        else [(b, r[0], r[2])]
                    )
                    for bx, tx, tk in parts:
                        nb.append(bx)
                        nrec.append((tx, r[1], tk))
                if len(nb) != len(boxes):
                    boxes = np.array(nb, np.float32).reshape(-1, 4)
                    rec = nrec
                    texts = [r[0] for r in rec]
                    order, layout, logits = self.reading_order(boxes, texts, wh)
        note = "order=model"
        if content_order and len(boxes) > 1:
            keys = [
                min((tk[2] for tk in r[2] if len(tk) > 2), default=None) for r in rec
            ]
            if all(k is not None for k in keys):
                pdf = sorted(range(len(keys)), key=lambda i: keys[i])
                ok = pdf_order_coherent(
                    boxes[pdf, :4],
                    None if logits is None else logits[np.ix_(pdf, pdf)],
                )
                if ok:
                    order, note = pdf, "order=pdf"
        t["order_ms"] = 1000 * (time.perf_counter() - t0)
        t["total_ms"] = sum(v for k, v in t.items() if k != "total_ms")
        lines = [
            Line(
                [round(float(v), 1) for v in boxes[i, :4]],
                rec[i][0],
                round(rec[i][1], 4),
            )
            for i in order
        ]
        md, blocks = None, []
        if layout is not None:
            blocks = page_blocks(boxes[:, :4], texts, order, layout[0], layout[1])
            # Inside tables, labels split cells at smaller gaps (CELL_GAP).
            cells = {
                i: [
                    (bx, tx)
                    for bx, tx, _ in self.split_line(boxes[i], rec[i][2], CELL_GAP)
                ]
                for i in range(len(boxes))
            }
            md = page_markdown(
                boxes[:, :4],
                texts,
                order,
                layout[0],
                layout[1],
                keep_furniture=self.furniture,
                cells=cells,
                tab=layout[2],
            )
        elif len(boxes) < 2:
            md = "\n".join(texts)
        return Page(lines, t, md, note=note, blocks=blocks)


class InputError(Exception):
    """A file that cannot be read: not a PDF or an image, corrupt, or locked."""


def is_pdf(path: Path) -> bool:
    """The file's own bytes say PDF (the header may follow a little junk)."""
    with open(path, "rb") as f:
        return b"%PDF-" in f.read(1024)


def read_images(path: Path) -> list[np.ndarray]:
    """Grayscale pages of an image file; every frame of a multi-page TIFF."""
    ok, frames = cv2.imreadmulti(str(path), flags=cv2.IMREAD_GRAYSCALE)
    if not ok or not frames:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        frames = [img] if img is not None else []
    if not frames:
        raise InputError(f"{path}: not a PDF or a readable image")
    return [np.ascontiguousarray(f) for f in frames]


def load_pages(path: Path, dpi: float = 200.0):
    """Yield grayscale pages of a PDF (rendered at ``dpi``) or an image file
    (every frame of a multi-page TIFF); the file's bytes decide which."""
    path = Path(path)
    if is_pdf(path):
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(path))
        try:
            for i in range(len(doc)):
                arr = doc[i].render(scale=dpi / 72.0, grayscale=True).to_numpy()
                yield i, np.ascontiguousarray(arr[..., 0] if arr.ndim == 3 else arr)
        finally:
            doc.close()
    else:
        yield from enumerate(read_images(path))
