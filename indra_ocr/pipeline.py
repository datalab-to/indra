"""The OCR pipeline on ONNX Runtime: line detection, CTC line recognition,
reading order and layout classes.  CPU only, no PyTorch.

The model folder holds det.onnx, rec.onnx (or rec.int8.onnx), order.onnx and
meta.json (see ``indra_ocr.models``).  With script.onnx and a group recognizer
(rec_cjk.onnx, rec_rtl.onnx, rec_other.onnx), each line goes to the recognizer
of its script group (GROUPS); every recognizer also reads Latin, and a group
without its recognizer is read by rec.onnx.  A right-to-left recognizer reads
in visual order: ``rtl.flip`` turns its text into logical order.  With orient.onnx,
a page turned by a quarter or half turn is turned upright before detection
(``Page.rotation``; boxes are in the upright page).
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import native
from .ctc import lm_inputs, peak_prob
from .det_post import OUT_STRIDE, decode, page_to_input
from .layout_post import TABLE, page_blocks, page_markdown, split_blocks
from .line_mask import crop_lines, line_darkness
from .math import MathReader, latin_line
from .math_span_img import Formula, unspliced
from .order_post import (
    BLOCK_CLASSES,
    GRID_LABELS,
    decode_blocks_order,
    decode_order,
    line_features,
    order_xycut,
    xycut_blocks,
)
from .table_rel import read_table, words_of
from .table_verify import TableVerifier
from .rec_preproc import PAD_X, pad_batch, widen, width_batches
from .rot_gate import read_rotated
from .rtl import flip, has_rtl, is_rtl
from .vocab import Vocab
from .ort_session import session

ADAPTIVE_MIN_H = (
    9.0  # median line height (input px) that triggers a second detection pass
)
ADAPTIVE_TARGET_H = 12.0
# Gaps between CTC character positions (centres), in line heights.  The cut
# rules (2.5 and 1.0 line heights) measure glyph edges; centres add about half a
# line height of glyph width.
SPLIT_GAP = 3.0  # cut a detected line at a wider gap (outside tables)
CELL_GAP = 1.8  # cut a table line into cells at a wider gap (in tables)
GUTTER_GAP = 1.6  # cut at a smaller gap when other lines' edges line up with it (a column gutter)
GUTTER_EDGES = 3  # how many other line edges must line up (within half a line height)
ADAPTIVE_MAX_SIDE = 2048
# Large pages (plans, posters): when the lines of a page whose long side is over
# TILE_MIN_SIDE px are still under ADAPTIVE_MIN_H after the adaptive pass, the page
# is detected again in overlapping tiles at TILE_SCALE (a 200 dpi render -> 150
# dpi), where the whole-page pass sees small text at 30-60 dpi.  Only tiles over
# text-bearing regions of the whole-page pass run (its text probability over
# TILE_REGION_P).
TILE_MIN_SIDE = 4096
TILE_SCALE = 0.75
TILE = 1024  # detector input side per tile
TILE_OVERLAP = 128  # detector input px shared by neighbouring tiles
TILE_REGION_P = 0.15
ORIENT_SIDE = 768  # orient.onnx input: page thumbnail, long side, padded square
# A turn is applied only above this probability: real pages are almost all upright,
# and a wrong turn loses the page.
ORIENT_MIN = 0.95
VERTICAL = 4  # orient.onnx class (5-class models): the page is set in vertical columns
# When orient.onnx is unsure but puts at least this much probability on "text runs top to
# bottom" (a quarter turn either way, or vertical columns), a few lines are read as the page
# stands and turned each way, and the reading the recognizers are surest of picks the turn.  At a thumbnail, vertical CJK
# columns and a page turned a quarter turn look alike: only the characters tell them apart.
SIDEWAYS_MASS = 0.8
SIDEWAYS_LINES = 8  # lines read per candidate turn
# A half turn is checked the same way before it is made: the longest lines are read
# as the page stands and turned half round, and the reading the recognizers are surer
# of wins.  orient.onnx can turn an upright book scan half round above
# orient_min (issue #32); the recognizers tell the two apart, since a line read upside down gets a low confidence.


# A recognized line under this confidence (mean CTC peak probability) gets no
# text: on empty form fields, checkboxes and rules the recognizer reads "-", "I"
# or "1" at low confidence.  The line's box stays (layout and reading order).
MIN_CONF = 0.5
# hw.onnx: a base line goes to rec_hw at this handwritten probability or more.
HW_ROUTE_MIN = 0.5

# Beam search with a language model (lm.npz for the base recognizer, optional
# lm_<group>.npz for a group) reads a recognizer's lines under this confidence;
# the rest keep the greedy reading.  alpha weighs the LM, beta rewards each token
# (meta.json "lm_params" overrides them per group).
LM_GATE = 0.9
LM_ALPHA = 0.5
LM_BETA = 1.5
LM_BEAM = 8
# A class under this probability at a step is not tried by the beam search.
LM_CAND_FLOOR = 3e-3
OUT_GREEDY = ["ids", "conf", "lengths"]
OUT_LM = ["blank", "topv", "topi"]
# A newer recognizer ends at these: the package computes the confidence and the
# LM inputs from the logits, only where it uses them (ctc.py).
OUT_LOGITS = ["ids", "lengths", "logits"]

# Recognizer groups, in script.onnx's class order unless meta.json names them
# ("script_groups", e.g. base and world).  base is rec.onnx.  An older
# script.onnx has one output: the chance that the line holds Chinese characters.
GROUPS = ("base", "cjk", "rtl", "other")

# A line goes to one recognizer when the classifier is surer than this; otherwise
# base reads it as well and the more confident reading wins.
ROUTE_SURE = 0.95

# det.onnx picture channel: a region is a picture above this probability and
# this share of the page area.
PICTURE_MIN = 0.5
PICTURE_MIN_AREA = 0.004
DESKEW_MAX = 45.0  # degrees of tilt searched (quarter turns cover the rest)
DESKEW_MIN = 0.3  # smaller tilts are left alone (the search step is 0.1)


THUMB_H, THUMB_W = 16, 64  # order.onnx line thumbnails


def line_thumbs(
    gray: np.ndarray,
    boxes,
    ctx: float = 0.25,
    th: int = THUMB_H,
    tw: int = THUMB_W,
    interp: str = "area",
) -> np.ndarray:
    """[N, 1, th, tw] float32 thumbnails of line boxes (pixels of ``gray``): the box with ``ctx``
    line heights above and below, scaled to th rows with its aspect kept (cut at tw), ink = 1;
    interp "area" (cv2.INTER_AREA) or "linear".  The order model expects exactly these thumbnails."""
    import cv2

    mode = cv2.INTER_LINEAR if interp == "linear" else cv2.INTER_AREA
    out = np.zeros((len(boxes), 1, th, tw), np.float32)
    H, W = gray.shape[:2]
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = (float(v) for v in b[:4])
        h = max(2.0, y1 - y0)
        a, z = int(max(0, y0 - ctx * h)), int(min(H, y1 + ctx * h))
        c, d = int(max(0, x0)), int(min(W, x1))
        if z - a < 2 or d - c < 1:
            continue
        crop = gray[a:z, c:d]
        w = max(1, min(tw, round(crop.shape[1] * th / crop.shape[0])))
        r = cv2.resize(crop, (w, th), interpolation=mode)
        out[i, 0, :, :w] = (255.0 - r) / 255.0
    return out


def _has_logits(sess) -> bool:
    """Whether a recognizer ends at its logits (OUT_LOGITS), not at the CTC outputs."""
    outs = getattr(sess, "get_outputs", None)  # test doubles may have no outputs list
    return outs is not None and any(o.name == "logits" for o in outs())


def _lm_ready(sess) -> bool:
    """Whether a recognizer gives what the beam search reads: the top-k outputs
    (OUT_LM) or the logits."""
    outs = {o.name for o in sess.get_outputs()}
    return set(OUT_LM) <= outs or "logits" in outs


def _ink_thumb(gray: np.ndarray, side: int) -> np.ndarray:
    """uint8 ink map (255 = ink) of the page shrunk to ``side`` pixels: 8-bit
    images rotate faster than float ones."""
    h, w = gray.shape
    s = side / max(h, w)
    small = cv2.resize(
        gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA
    )
    return 255 - small


def _row_sharpness(ink: np.ndarray, a: float) -> float:
    """How sharply the horizontal ink profile changes row to row, the page
    turned by ``a`` degrees: highest when the text rows are level."""
    size = (ink.shape[1], ink.shape[0])
    r = cv2.warpAffine(
        ink, cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), a, 1.0), size
    )
    p = r.sum(1, dtype=np.int64).astype(np.float64)
    return float(((p[1:] - p[:-1]) ** 2).sum())


def estimate_skew(gray: np.ndarray, side: int = 800, pool=None) -> float:
    """Degrees (counter-clockwise) that straighten the text rows of a page: the
    angle whose horizontal ink profile changes most sharply row to row.  Searched
    over +-DESKEW_MAX in 2 degree steps on a small thumbnail (3/16 of ``side``: on
    895 test pages, upright and tilted up to 44 degrees, the same angle as a
    half-size one but on one page, which it reads better); the two best
    angles and 0 are then refined in 0.5 and 0.1 degree steps on the full
    thumbnail (``side`` pixels), where small text rows are still resolved.
    pool: a thread pool that scores the angles in parallel (same result)."""
    run = pool.map if pool is not None else map
    fine = _ink_thumb(gray, side)
    if fine.mean() < 0.003 * 255:  # a blank page has no rows to straighten
        return 0.0
    coarse = _ink_thumb(gray, side * 3 // 16)
    grid = np.arange(-DESKEW_MAX, DESKEW_MAX + 1e-6, 2.0)
    scores = list(run(lambda a: _row_sharpness(coarse, a), grid))
    starts = {0.0, *(float(grid[i]) for i in np.argsort(scores)[-2:])}
    near = [a for a0 in starts for a in np.arange(a0 - 2.0, a0 + 2.0 + 1e-6, 0.5)]
    # The 0.5 degree ranges of nearby starts overlap: each angle is scored once.
    seen = dict.fromkeys(near)
    seen.update(zip(seen, run(lambda a: _row_sharpness(fine, a), list(seen))))
    a1 = max(near, key=seen.__getitem__)
    last = np.arange(a1 - 0.5, a1 + 0.5 + 1e-6, 0.1)
    vals = list(run(lambda a: _row_sharpness(fine, a), last))
    return float(max(zip(last, vals), key=lambda av: av[1])[0])


PREVIEW_SIDE = 1600  # the skew search and orient.onnx read thumbnails of at most 800


def preview(gray: np.ndarray, side: int = PREVIEW_SIDE) -> np.ndarray:
    """The page shrunk once to at most ``side`` pixels a side (area averaging):
    the skew search and the orientation model read it instead of the full scan."""
    h, w = gray.shape
    s = side / max(h, w)
    if s >= 1:
        return gray
    return cv2.resize(
        gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA
    )


def picture_boxes(logit: np.ndarray, s: float, page_hw: tuple) -> np.ndarray:
    """Picture regions [P, 4] in page pixels from det.onnx's picture channel (at
    half the detector's input scale ``s``): connected regions above PICTURE_MIN,
    at least PICTURE_MIN_AREA of the page."""
    mask = (logit > np.log(PICTURE_MIN / (1 - PICTURE_MIN))).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    f = 2.0 / s
    min_area = PICTURE_MIN_AREA * page_hw[0] * page_hw[1]
    out = []
    for x, y, w, h, _ in stats[1:n]:
        b = [x * f, y * f, min((x + w) * f, page_hw[1]), min((y + h) * f, page_hw[0])]
        if (b[2] - b[0]) * (b[3] - b[1]) >= min_area:
            out.append(b)
    return np.array(out, np.float32).reshape(-1, 4)


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
class Word:
    box: list[float]  # x0, y0, x1, y1 in image pixels (the line's height)
    text: str


@dataclass
class Line:
    box: list[float]  # x0, y0, x1, y1 in image pixels
    text: str
    conf: float
    # with load(words=True): the line's words in reading order; a word's x
    # extent comes from where the recognizer (or text layer) placed its
    # characters, its y extent is the line's
    words: list[Word] | None = None


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
    # set in vertical columns (Chinese, Japanese): read turned a quarter turn,
    # columns as lines; boxes are on the page as it is (rotation stays 0)
    vertical: bool = False

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)

    @property
    def pictures(self) -> list[list[float]]:
        """Picture regions (figures, photos, diagrams) on the page, in reading
        order: [x0, y0, x1, y1] pixels; the markdown names them picture-<k>."""
        return [b["bbox"] for b in self.blocks if b["label"] == "Picture"]


def quarter_turns(rotation: int) -> int:
    """Quarter turns for ``rotation`` degrees (a multiple of 90; Page.rotation's
    sense: counter-clockwise turn of the content, undone before reading)."""
    if isinstance(rotation, bool) or not isinstance(rotation, (int, np.integer)) or rotation % 90:
        raise ValueError(f"rotation must be a multiple of 90 degrees, not {rotation!r}")
    return int(rotation) // 90 % 4


def _top_k(logits: np.ndarray, n: int, k: int = 8):
    """The blank's log-probability and the k most likely classes (log-probabilities,
    best first) per frame, from a recognizer's logits (B, T, C); frames after n
    are left out."""
    x = logits[:, :n].astype(np.float32)
    m = x.max(-1, keepdims=True)
    lse = m + np.log(np.exp(x - m).sum(-1, keepdims=True))
    top = np.argpartition(-x, k - 1, axis=-1)[..., :k]
    v = np.take_along_axis(x, top, axis=-1)
    order = np.argsort(-v, axis=-1)
    return (
        x[..., 0] - lse[..., 0],
        np.take_along_axis(v, order, axis=-1) - lse,
        np.take_along_axis(top, order, axis=-1),
    )


class OrtOCR:
    orient_min = ORIENT_MIN

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
        math_spans: bool = False,
        math_rescore: bool = False,
        orient_min: float = ORIENT_MIN,
        rot_gate: bool | None = None,
        words: bool = False,
    ):
        """table_heads: build tables from the order model's row / column / cell
        heads (when it has them) instead of from geometry.  furniture: keep page
        headers and footers in the markdown (dropped by default).  math: read lines
        with math as LaTeX (math.py; needs the math models; off by default).  math_spans:
        with math, also inline math inside text-layer lines (math_span.py, when the
        model folder has math_span.onnx; off by default).  math_rescore: with math, a
        beam search re-reads lines MathRec is unsure of (math.py; off by default).  providers: ONNX
        Runtime execution providers (default CPU; e.g. ["CUDAExecutionProvider",
        "CPUExecutionProvider"] with onnxruntime-gpu).  orient_min: the orient.onnx
        probability above which a page is turned without reading lines each way
        (read_turn).  rot_gate: read quarter-turned text lines on OCR pages (rot_gate.py;
        off by default; None follows INDRA_ROT_GATE=1).  words: give each line its
        words with boxes (Line.words; off by default)."""
        if not 0.0 <= orient_min <= 1.0:
            raise ValueError(f"orient_min must be in [0, 1], not {orient_min!r}")
        self.orient_min = orient_min
        self.words = words
        self.table_heads = table_heads
        self.furniture = furniture
        import onnxruntime as ort

        onnx_dir = Path(onnx_dir)
        # With more than one thread, the skew search and the line crops (OpenCV
        # releases the GIL) run on a small thread pool; one-thread workers
        # (parallel.py) keep everything on their one core.
        self.pool = None
        if threads > 1:
            from concurrent.futures import ThreadPoolExecutor

            self.pool = ThreadPoolExecutor(threads)
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
        self.det = session(need("det.onnx"), so, providers=prov)
        self._pic = None  # the last page's picture channel, (logit map, scale)
        self.rec = session(need("rec.int8.onnx" if rec_int8 else "rec.onnx"), so, providers=prov)
        self.mask = (
            session(str(onnx_dir / "mask.onnx"), so, providers=prov)
            if (onnx_dir / "mask.onnx").exists()
            else None
        )
        self.order = (
            session(str(onnx_dir / "order.onnx"), so, providers=prov)
            if (onnx_dir / "order.onnx").exists()
            else None
        )
        # an order model that also reads line thumbnails
        self._order_thumbs = self.order is not None and "thumbs" in {
            i.name for i in self.order.get_inputs()
        }
        self.orient = (
            session(str(onnx_dir / "orient.onnx"), so, providers=prov)
            if (onnx_dir / "orient.onnx").exists()
            else None
        )
        meta = json.loads(Path(need("meta.json")).read_text(encoding="utf-8"))
        # line thumbnails for the order model: size from its "thumbs" input, resize mode
        # from meta.json (absent: area)
        self._thumb = (THUMB_H, THUMB_W, meta.get("order_thumbs_interp", "area"))
        if self._order_thumbs:
            shp = next(i.shape for i in self.order.get_inputs() if i.name == "thumbs")
            if isinstance(shp[-2], int) and isinstance(shp[-1], int):
                self._thumb = (shp[-2], shp[-1], self._thumb[2])
        self.vocab = Vocab.from_json(meta["vocab"])
        # Scripts: a line classifier routes each line to its group's recognizer
        # (rec_<group>.onnx with meta["vocab_<group>"]), in script.onnx's class
        # order meta["script_groups"] (default GROUPS).
        self.script, self.groups = None, {}
        self.script_groups = tuple(meta.get("script_groups", GROUPS))
        for g in [k[len("vocab_") :] for k in meta if k.startswith("vocab_")]:
            if (onnx_dir / f"rec_{g}.onnx").exists():
                self.groups[g] = (
                    session(str(onnx_dir / f"rec_{g}.onnx"), so, providers=prov),
                    Vocab.from_json(meta[f"vocab_{g}"]),
                )
        # A group recognizer that reads widened lines gets its crops widened by
        # meta["rec_xscale_<group>"] (rec_preproc.widen), keyed by its session.
        self.xscale = {
            id(sess): float(meta.get(f"rec_xscale_{g}", 1.0))
            for g, (sess, _) in self.groups.items()
        }
        if self.groups and (onnx_dir / "script.onnx").exists():
            self.script = session(str(onnx_dir / "script.onnx"), so, providers=prov)
        # Math: a line classifier routes lines with math to MathRec (math.py).
        self.math = MathReader.load(onnx_dir, so, prov) if math else None
        # Math spans inside text-layer lines (math_span.py), optional.
        from .math_span import SpanTagger

        self.math_span = SpanTagger.load(onnx_dir, so, prov) if math and math_spans else None
        # Inline math in lines MathRec does not read whole (math_span_img.py), optional.
        from .math_span_img import SpanImgTagger

        self.math_span_img = SpanImgTagger.load(onnx_dir, so, prov) if math else None
        if self.math is not None:
            self.math.rescore = math_rescore
        if math and self.math is None:
            raise FileNotFoundError(f"math=True needs the math models in {onnx_dir}")
        self.long_side = meta["det_long_side"]
        self.height = meta["rec_height"]
        # rec.onnx may read a tall window (the line and the page around it);
        # the classifiers and group recognizers read the line band only.
        self.window = meta.get("rec_window", 1)
        # Handwriting (optional): rec_hw.onnx, a larger recognizer with
        # the base vocab and its own input height and window, reads the base lines
        # that hw.onnx (a printed/handwritten classifier on the line band) calls
        # handwritten.  INDRA_HW_ROUTE=all sends every base line to it, =none none.
        self.rec_hw = (
            session(str(onnx_dir / "rec_hw.onnx"), so, providers=prov)
            if (onnx_dir / "rec_hw.onnx").exists()
            else None
        )
        self.hw_height = meta.get("rec_hw_height", self.height)
        self.hw_window = meta.get("rec_hw_window", self.window)
        self.hw_cls = (
            session(str(onnx_dir / "hw.onnx"), so, providers=prov)
            if self.rec_hw is not None and (onnx_dir / "hw.onnx").exists()
            else None
        )
        self.hw_route_min = meta.get("hw_route_min", HW_ROUTE_MIN)
        # Table structure (optional): tsr_rel.onnx gives, for the words of a table
        # block, which share a cell, a row or a column (table_rel.py).  meta.json
        # "tsr_rel_layer": false keeps the geometric tables for PDF text layers.
        # INDRA_TSR=0 keeps the geometric tables everywhere.
        self.tsr_rel_layer = bool(meta.get("tsr_rel_layer", True))
        # "tsr_rel_span": "rel" takes a cell's columns from the model's same-column
        # relations (group headers over several columns) instead of its geometry;
        # "rel2" does the same for rows (stub headers over several header rows).
        self.tsr_rel_span = str(meta.get("tsr_rel_span", "geo"))
        self.tsr_rel = (
            session(str(onnx_dir / "tsr_rel.onnx"), so, providers=prov)
            if (onnx_dir / "tsr_rel.onnx").exists() and os.environ.get("INDRA_TSR", "1") != "0"
            else None
        )
        # tsr_split.onnx (optional): row / column separator maps of a table crop, read
        # as extra word features by a tsr_rel.onnx that takes them (table_rel.py).
        self.tsr_split = (
            ort.InferenceSession(str(onnx_dir / "tsr_split.onnx"), so, providers=prov)
            if self.tsr_rel is not None and (onnx_dir / "tsr_split.onnx").exists()
            else None
        )
        # tsr_refine.onnx (optional): refines the tsr_rel word tags (table_rel.py).
        # Without the file, tables are as before.
        self.tsr_refine = (
            ort.InferenceSession(str(onnx_dir / "tsr_refine.onnx"), so, providers=prov)
            if self.tsr_rel is not None and (onnx_dir / "tsr_refine.onnx").exists()
            else None
        )
        # False-table verifier (optional): table_verify.json relabels Table blocks it
        # calls false tables (table_verify.py).  INDRA_TABLE_VERIFY=0 turns it off.
        # The file holds a weight set for OCR-read pages and, optionally, one for text-layer pages.
        self.table_verify, self.table_verify_layer = (
            TableVerifier.load(onnx_dir / "table_verify.json")
            if (onnx_dir / "table_verify.json").exists() and os.environ.get("INDRA_TABLE_VERIFY", "1") != "0"
            else (None, None)
        )
        # Rotated lines on upright pages (rot_gate.py): opt-in (load(rot_gate=True) or INDRA_ROT_GATE=1).
        self.rot_gate = os.environ.get("INDRA_ROT_GATE", "0") == "1" if rot_gate is None else bool(rot_gate)
        # rotcls.onnx (optional): the column classifier in front of the turned reads
        self.rot_cls = (
            session(str(onnx_dir / "rotcls.onnx"), so, providers=prov)
            if (onnx_dir / "rotcls.onnx").exists()
            else None
        )
        self.hw_conf_gate = meta.get("hw_conf_gate")
        self.hw_min = meta.get("hw_min", 0.0)  # script.onnx hw class threshold
        # Language models: lm.npz for rec.onnx, lm_<group>.npz for a group's
        # recognizer.  Each needs its recognizer's top-k or logits outputs.
        # meta.json "lm_params" may set a group's alpha, beta and gate; the
        # defaults are the base recognizer's.
        self.lms: dict = {}  # id(session) -> (NgramLM, alpha, beta, gate)
        params = meta.get("lm_params", {})
        for name, sess in [("base", self.rec)] + [
            (g, s) for g, (s, _) in self.groups.items()
        ]:
            f = onnx_dir / ("lm.npz" if name == "base" else f"lm_{name}.npz")
            if f.exists() and _lm_ready(sess):
                from .lm import NgramLM

                p = params.get(name, {})
                self.lms[id(sess)] = (
                    NgramLM(f),
                    p.get("alpha", LM_ALPHA),
                    p.get("beta", LM_BETA),
                    p.get("gate", LM_GATE),
                )
        if self.rec_hw is not None and id(self.rec) in self.lms:
            if _lm_ready(self.rec_hw):
                self.lms[id(self.rec_hw)] = self.lms[id(self.rec)]
        self.lm = self.lms.get(id(self.rec), (None,))[0]
        self.adaptive = adaptive

    def _detect_at(self, gray: np.ndarray, long_side: int):
        x, s = page_to_input(gray, long_side)
        self._det_in = (x, s)  # the ink map at the detector's scale (rot_gate reuses it)
        out = self.det.run(None, {"x": x[None, None]})[0][0]
        self._pic = (out[3], s) if out.shape[0] > 3 else None
        self._prob = (out[0], s)  # text logit at OUT_STRIDE of the detector input
        return decode(out), s

    def _detect_tiled(self, gray: np.ndarray) -> np.ndarray:
        """Line boxes [N, 5] in page pixels from overlapping TILE x TILE detector
        inputs at TILE_SCALE.  Each tile keeps the boxes whose centre lies in its
        own core (the tile minus half the overlap on every inner side); a line cut
        by a core edge comes back in pieces from two tiles, and pieces from
        different tiles that share a text row and touch are joined."""
        logit, s = self._prob
        region = cv2.dilate(
            (logit > np.log(TILE_REGION_P / (1 - TILE_REGION_P))).astype(np.uint8),
            np.ones((5, 5), np.uint8),
        )
        h, w = gray.shape
        k = TILE_SCALE
        step = TILE - TILE_OVERLAP
        span = TILE / k  # page px per tile
        sh, sw = int(np.ceil(h * k)), int(np.ceil(w * k))
        ys = list(range(0, max(1, sh - TILE_OVERLAP), step))
        xs = list(range(0, max(1, sw - TILE_OVERLAP), step))
        found = []
        for ty in ys:
            for tx in xs:
                # page box of this tile, and the same box on the region map
                py0, px0 = ty / k, tx / k
                py1, px1 = min(h, py0 + span), min(w, px0 + span)
                r = region[
                    int(py0 * s / OUT_STRIDE) : int(np.ceil(py1 * s / OUT_STRIDE)) + 1,
                    int(px0 * s / OUT_STRIDE) : int(np.ceil(px1 * s / OUT_STRIDE)) + 1,
                ]
                if not r.any():
                    continue
                crop = gray[int(py0) : int(np.ceil(py1)), int(px0) : int(np.ceil(px1))]
                x, cs = page_to_input(crop, int(round(max(crop.shape) * k)))
                out = self.det.run(None, {"x": x[None, None]})[0][0]
                b = decode(out)
                if not len(b):
                    continue
                b[:, :4] = b[:, :4] / cs
                b[:, [0, 2]] += px0
                b[:, [1, 3]] += py0
                # the core: half the overlap in from every edge that has a neighbour
                m = TILE_OVERLAP / 2 / k
                cx0 = px0 + (m if tx > 0 else -1e9)
                cy0 = py0 + (m if ty > 0 else -1e9)
                cx1 = px1 - (m if tx + step < sw - TILE_OVERLAP else -1e9)
                cy1 = py1 - (m if ty + step < sh - TILE_OVERLAP else -1e9)
                cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
                keep = (cx >= cx0) & (cx < cx1) & (cy >= cy0) & (cy < cy1)
                for row in b[keep]:
                    lh = row[3] - row[1]
                    # a piece that reaches a core edge may continue in the next tile
                    edge = (
                        row[0] < cx0 + lh
                        or row[2] > cx1 - lh
                        or row[1] < cy0 + lh
                        or row[3] > cy1 - lh
                    )
                    found.append(((ty, tx), list(row), bool(edge)))
        if not found:
            return np.zeros((0, 5), np.float32)
        tiles = [t for t, _, _ in found]
        boxes = [b for _, b, _ in found]
        alive = [True] * len(boxes)
        # join pieces of one line from two tiles: a shared text row (vertical
        # overlap over half the lower box) and a horizontal gap under one line height
        edge = [i for i, (_, _, e) in enumerate(found) if e]
        for a in edge:
            if not alive[a]:
                continue
            for c in edge:
                if c == a or not alive[c] or tiles[c] == tiles[a]:
                    continue
                A, C = boxes[a], boxes[c]
                hgt = min(A[3] - A[1], C[3] - C[1])
                vov = min(A[3], C[3]) - max(A[1], C[1])
                gap = max(A[0], C[0]) - min(A[2], C[2])
                if hgt > 0 and vov > 0.5 * hgt and gap < hgt:
                    boxes[a] = [
                        min(A[0], C[0]),
                        min(A[1], C[1]),
                        max(A[2], C[2]),
                        max(A[3], C[3]),
                        max(A[4], C[4]),
                    ]
                    alive[c] = False
        out = np.array([b for b, ok in zip(boxes, alive) if ok], np.float32)
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
        return out

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
        if (
            max(gray.shape) > TILE_MIN_SIDE
            and len(boxes) >= 8
            and float(np.median(boxes[:, 3] - boxes[:, 1])) < ADAPTIVE_MIN_H
        ):
            # the lines are still too small for the detector after the adaptive
            # pass (a plan sheet): detect again in tiles at a higher scale
            return self._detect_tiled(gray)
        if len(boxes):
            boxes[:, :4] /= s
            h, w = gray.shape
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes

    def detect_all(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Text-line boxes and picture boxes [P, 4] (pixels; none from a
        det.onnx without the picture channel)."""
        self._pic = None
        boxes = self.detect(gray)
        pic = getattr(self, "_pic", None)
        if pic is None:
            return boxes, np.zeros((0, 4), np.float32)
        return boxes, picture_boxes(*pic, gray.shape)

    def recognize(
        self,
        gray: np.ndarray,
        boxes: np.ndarray,
        batch: int = 64,
        vertical: bool = False,
        math: bool = True,
    ) -> list[tuple[str, float, list]]:
        """(text, confidence, tokens) per line; tokens = [(token id, page x)],
        x from the CTC peak's timestep mapped back through the crop.  A line
        under MIN_CONF comes back empty.  vertical: the page is a vertical one
        turned a quarter turn (every line to the CJK recognizer).  math=False:
        the recognizers' readings only, MathRec not asked (for callers that
        compare the readings with a text layer)."""
        # CenterMask first when the model folder has mask.onnx
        pool = getattr(self, "pool", None)
        window = getattr(self, "window", 1)
        # Light-on-dark is decided once per line box, for both crops.
        dark = line_darkness(gray, boxes[:, :4], pool) if window > 1 else None
        crops = crop_lines(
            gray, boxes[:, :4], self.height, self.mask, pool=pool, dark=dark
        )
        tall = (
            crop_lines(
                gray, boxes[:, :4], self.height, window=window, pool=pool, dark=dark
            )
            if window > 1
            else crops
        )
        out: list = [("", 0.0, [])] * len(crops)
        groups = [(self.rec, self.vocab, list(range(len(crops))), False)]
        script_hw: list = []  # lines script.onnx calls handwritten (an "hw" class)
        if vertical and "cjk" in self.groups:
            # A vertical page's lines are CJK columns, characters on their side;
            # the CJK recognizer reads those (and the Latin set across them).
            sess, vocab = self.groups["cjk"]
            groups = [(sess, vocab, list(range(len(crops))), True)]
        elif self.script is not None and crops:
            names, sure = self.line_groups(crops, batch)
            script_hw = [j for j, g in enumerate(names) if g == "hw"]
            # A line the classifier is not sure about is read by base as well;
            # the reading with the higher confidence wins (below).
            groups = [
                (
                    self.rec,
                    self.vocab,
                    [j for j, g in enumerate(names) if g == "base" or not sure[j]],
                    False,
                )
            ] + [
                (sess, vocab, [j for j, g in enumerate(names) if g == name], True)
                for name, (sess, vocab) in self.groups.items()
            ]
        # Handwritten base lines go to rec_hw, with crops at its own height.  With
        # meta.json hw_conf_gate, only base lines read under that confidence are
        # classified (after the base read), so confident printed lines cost nothing.
        hw_crops = hw_tall = None
        rec_hw = getattr(self, "rec_hw", None)
        gate = getattr(self, "hw_conf_gate", None)

        def hw_crops_for(sel):
            line = [None] * len(crops)
            tl = [None] * len(crops)
            sub = boxes[sel, :4]
            sel_dark = None if dark is None else [dark[j] for j in sel]
            for j, c in zip(
                sel,
                crop_lines(
                    gray, sub, self.hw_height, self.mask, pool=pool, dark=sel_dark
                ),
            ):
                line[j] = c
            if self.hw_window > 1:
                for j, c in zip(
                    sel,
                    crop_lines(
                        gray,
                        sub,
                        self.hw_height,
                        window=self.hw_window,
                        pool=pool,
                        dark=sel_dark,
                    ),
                ):
                    tl[j] = c
            else:
                tl = line
            return line, tl

        if rec_hw is not None and gate is None and not vertical and crops:
            hw = self.hw_lines(crops, batch)
            base = groups[0]
            moved = [j for j in base[2] if hw[j]]
            if moved:
                groups[0] = (
                    base[0],
                    base[1],
                    [j for j in base[2] if not hw[j]],
                    base[3],
                )
                groups.append((rec_hw, self.vocab, moved, False))
                hw_crops, hw_tall = hw_crops_for(moved)
        if script_hw and rec_hw is not None and hw_crops is None:
            groups.append((rec_hw, self.vocab, script_hw, False))
            hw_crops, hw_tall = hw_crops_for(script_hw)
        best_conf = [-1.0] * len(crops)

        def read(groups, force=False):
            for sess, vocab, sel, as_chars in groups:
                is_hw = rec_hw is not None and sess is rec_hw
                line = hw_crops if is_hw else crops
                for part in width_batches([line[j].shape[1] for j in sel], batch):
                    idx = [sel[i] for i in part]
                    src = hw_tall if is_hw else tall if sess is self.rec else crops
                    xscale = getattr(self, "xscale", {}).get(id(sess), 1.0)
                    arr, widths = pad_batch(widen([src[j] for j in idx], xscale))
                    lm = getattr(self, "lms", {}).get(id(sess))
                    use_lm = lm is not None
                    feed = {"x": arr, "widths": widths}
                    logits = None
                    if _has_logits(sess):
                        best, lengths, logits = sess.run(OUT_LOGITS, feed)
                    else:
                        res = sess.run(OUT_GREEDY + (OUT_LM if use_lm else []), feed)
                        best, pmax, lengths = res[:3]
                    stride = arr.shape[-1] / max(best.shape[1], 1)
                    for k, j in enumerate(idx):
                        n = int(lengths[k])
                        seq = best[k, :n]
                        keep = (seq != 0) & np.concatenate(
                            [[True], seq[1:] != seq[:-1]]
                        )
                        if logits is None:
                            pm = pmax[k, :n][keep]
                        else:
                            pm = peak_prob(logits[k, :n][keep])
                        x0, y0, x1, y1 = (float(v) for v in boxes[j, :4])
                        h = max(1.0, y1 - y0)
                        ix0 = np.floor(x0 - PAD_X * h)
                        cw = np.ceil(x1 + PAD_X * h) - ix0
                        f = cw / (line[j].shape[1] * xscale)
                        t = np.nonzero(keep)[0]
                        ids = seq[t].tolist()
                        conf = float(pm.mean()) if keep.any() else 0.0
                        if use_lm and MIN_CONF <= conf < lm[3]:
                            from .lm import beam_search

                            lmin = (
                                lm_inputs(logits[k, :n], floor=LM_CAND_FLOOR)
                                if logits is not None
                                else (res[3][k, :n], res[4][k, :n], res[5][k, :n])
                            )
                            ids, steps = beam_search(
                                *lmin,
                                lm[0],
                                lm[1],
                                lm[2],
                                LM_BEAM,
                                cand_floor=LM_CAND_FLOOR,
                            )
                            t = np.array(steps, np.int64)
                        xs = (ix0 + (t + 0.5) * stride * f).tolist()
                        # A group recognizer's ids mean nothing in self.vocab: its
                        # tokens are characters, like the text-layer tokens.
                        toks = (
                            list(zip([vocab.chars[i - 1] for i in ids], xs))
                            if as_chars
                            else list(zip(ids, xs))
                        )
                        if conf <= best_conf[j] and not force:
                            continue
                        best_conf[j] = conf
                        out[j] = (
                            (vocab.decode(ids), conf, toks)
                            if conf >= MIN_CONF
                            else ("", conf, [])
                        )

        read(groups)
        if rec_hw is not None and gate is not None and not vertical and crops:
            # Second pass: unsure base lines that the classifier calls handwritten.
            unsure = [j for j in groups[0][2] if best_conf[j] < gate]
            if unsure:
                hw = self.hw_lines([crops[j] for j in unsure], batch)
                moved = [j for j, h in zip(unsure, hw) if h]
                if moved:
                    hw_crops, hw_tall = hw_crops_for(moved)
                    read([(rec_hw, self.vocab, moved, False)], force=True)
        if math and getattr(self, "math", None) is not None and crops:
            # MathRec reads a line whose reading is mostly Latin (a formula line may
            # have gone to a group recognizer: Greek letters to rec_world), or a line
            # whose letters its own vocabulary writes (a MathRec whose vocabulary has
            # Chinese reads Chinese lines); any other line keeps its reading.  The
            # classifier decides which of these hold math.
            cand = [
                j
                for j in range(len(crops))
                if out[j] is not None
                and (latin_line(out[j][0]) or self.math.writes(out[j][0]))
            ]
            flags = self.math.flags([crops[j] for j in cand]) if cand else []
            math = [j for j, f in zip(cand, flags) if f]
            if math and self.math.gate_head is not None:
                probs = [p for p, f in zip(self.math.last_probs, flags) if f]
                drafts0 = [out[j][0] if out[j] is not None else "" for j in math]
                keep = self.math.gate(probs, drafts0, len(cand), "ocr")
                math = [j for j, k in zip(math, keep) if k]
            if math:
                drafts = [out[j][0] if out[j] is not None else "" for j in math]
                read = self.math.read(gray, boxes[math], drafts=drafts)
                for j, (text, conf), draft in zip(math, read, drafts):
                    # Only a confident reading with math replaces the line, with
                    # the recognizer's accents in its prose (one token holds the
                    # whole line: it is never split).
                    rconf = out[j][1] if out[j] is not None else 0.0
                    merged = self.math.accept(text, conf, draft, rconf)
                    if merged is not None:
                        self.math.accepted[merged] = (out[j], text, conf, draft, rconf)
                        xc = float(boxes[j, 0] + boxes[j, 2]) / 2
                        out[j] = (merged, 1.0, [(merged, xc)])
            if getattr(self, "math_span_img", None) is not None and not vertical:
                seen = set(cand)
                self._read_span_img(gray, boxes, out, [j for j in range(len(crops)) if j not in seen])
        return out

    def _read_span_img(self, gray: np.ndarray, boxes: np.ndarray, out: list, idx: list) -> None:
        """Inline math in the lines idx (lines that MathRec does not read whole):
        the image tagger marks math columns, MathRec reads each run's crop, and a
        confident reading with math replaces the recognizer's characters there."""
        from .math import MATH_MIN_CONF, has_math
        from .math_span_img import page_x, splice

        idx = [j for j in idx if out[j] is not None and out[j][2] and any(c.isalpha() for c in out[j][0])]
        if not idx:
            return
        tagger = self.math_span_img
        crops = crop_lines(gray, boxes[idx, :4], tagger.height)
        jobs = []  # (line, page x0, page x1)
        for j, c, found in zip(idx, crops, tagger.spans(crops)):
            jobs += [(j, *page_x(boxes[j], c.shape[1], xs)) for xs in found]
        if not jobs:
            return
        sb = np.array([[x0, boxes[j, 1], x1, boxes[j, 3]] for j, x0, x1 in jobs], np.float32)
        for (j, x0, x1), (text, conf) in zip(jobs, self.math.read(gray, sb)):
            t = text.strip()
            # A run's reading must be one formula: the line keeps its own prose.
            if conf < MATH_MIN_CONF or t.count("$") != 2 or not (t.startswith("$") and t.endswith("$")):
                continue
            if not (tagger.accept_plain or has_math(t)):
                continue
            toks = out[j][2]
            if not isinstance(toks[0][0], str):  # main recognizer: vocab ids -> characters
                toks = [(self.vocab.decode([tk[0]]) or " ", tk[1]) for tk in toks]
            toks = splice(toks, x0, x1, t)
            out[j] = ("".join(tk[0] for tk in toks), out[j][1], toks)

    def layer_gaps(
        self, gray: np.ndarray, boxes: np.ndarray, tokens: list[list]
    ) -> list[tuple[float | None, float, str]]:
        """Per line box and its text-layer tokens: (gap, conf, group).  gap: the
        log-probability per token by which the box's recognizer (script.onnx
        routes the crop) finds the text less likely than its own best reading
        (``agree.forced_gap``).  A line the router is not sure of is also read by
        each other recognizer that writes the layer's letters (``agree.writes``),
        and the smallest gap counts.  conf: that recognizer's confidence in its
        own reading (mean peak probability of its characters, as ``recognize``);
        group: that recognizer ("base" or a model folder group: "world", "rtl"...).
        gap None for a line read as math: the recognizer writes LaTeX where the
        layer holds Unicode symbols, so the two do not compare."""
        from .agree import forced_gaps, layer_text, target_ids, writes

        if not len(boxes):
            return []
        pool = getattr(self, "pool", None)
        window = getattr(self, "window", 1)
        dark = line_darkness(gray, boxes[:, :4], pool) if window > 1 else None
        crops = crop_lines(gray, boxes[:, :4], self.height, self.mask, pool=pool, dark=dark)
        tall = (
            crop_lines(gray, boxes[:, :4], self.height, window=window, pool=pool, dark=dark)
            if window > 1
            else crops
        )
        recs = {"base": (self.rec, self.vocab), **self.groups}
        if self.script is not None:
            names, sure = self.line_groups(crops)
        else:
            names, sure = ["base"] * len(crops), [True] * len(crops)
        texts = {name: [layer_text(t, name == "rtl") for t in tokens] for name in recs}
        cands = [
            {g} | ({n for n, (_, v) in recs.items() if writes(texts[n][j], v)} if not s else set())
            for j, (g, s) in enumerate(zip(names, sure))
        ]
        math_lines: set = set()
        todo: list = []
        for name, (sess, vocab) in recs.items():
            sel = [j for j, c in enumerate(cands) if name in c]
            if not sel:
                continue
            xscale = getattr(self, "xscale", {}).get(id(sess), 1.0)
            arr, widths = pad_batch(
                widen([(tall if sess is self.rec else crops)[j] for j in sel], xscale)
            )
            if any(o.name == "logits" for o in sess.get_outputs()):
                best, lengths, logits = sess.run(["ids", "lengths", "logits"], {"x": arr, "widths": widths})
                blank, topv, topi = _top_k(logits, int(lengths.max()))
            else:
                best, _, lengths, blank, topv, topi = sess.run(
                    OUT_GREEDY + OUT_LM, {"x": arr, "widths": widths}
                )
            dollar = vocab.index.get("$")  # a math span opens: the reading is LaTeX
            for k, j in enumerate(sel):
                n = int(lengths[k])
                if dollar is not None and (best[k, :n] == dollar).any():
                    math_lines.add(j)
                    continue
                ids = target_ids(texts[name][j], vocab, visual=name == "rtl")
                seq = best[k, :n]
                keep = (seq != 0) & np.concatenate([[True], seq[1:] != seq[:-1]])
                peak = np.exp(np.maximum(blank[k, :n], topv[k, :n, 0]))
                conf = float(peak[keep].mean()) if keep.any() else 0.0
                todo.append((j, conf, name, (blank[k, :n], topv[k, :n], topi[k, :n], ids)))
        out: list[tuple[float | None, float, str]] = [(float("inf"), 0.0, g) for g in names]
        for (j, conf, name, _), g in zip(todo, forced_gaps([it for *_, it in todo])):
            if g < out[j][0]:
                out[j] = (g, conf, name)
        for j in math_lines:
            out[j] = (None, 0.0, names[j])
        return out

    def hw_lines(self, crops: list, batch: int = 64) -> list[bool]:
        """Per line crop (line band, base height), whether rec_hw reads it:
        hw.onnx's handwritten probability at least meta.json hw_route_min
        (default HW_ROUTE_MIN), or INDRA_HW_ROUTE."""
        mode = os.environ.get("INDRA_HW_ROUTE", "")
        if mode in ("all", "none"):
            return [mode == "all"] * len(crops)
        if self.hw_cls is None:
            return [False] * len(crops)
        out = [False] * len(crops)
        for idx in width_batches([c.shape[1] for c in crops], batch):
            arr, widths = pad_batch([crops[j] for j in idx])
            p = self.hw_cls.run(None, {"x": arr, "widths": widths})[0]
            for k, j in enumerate(idx):
                out[j] = bool(p[k] >= self.hw_route_min)
        return out

    def line_groups(self, crops: list, batch: int = 64) -> tuple[list[str], list[bool]]:
        """Per line crop, the group whose recognizer reads it (script.onnx), and
        whether the classifier is sure of it (probability over ROUTE_SURE); a
        group without its recognizer goes to base.  A script.onnx with an "hw"
        class (handwritten base-script lines) sends those to rec_hw.onnx."""
        out = ["base"] * len(crops)
        sure = [True] * len(crops)
        for idx in width_batches([c.shape[1] for c in crops], batch):
            arr, widths = pad_batch([crops[j] for j in idx])
            p = self.script.run(None, {"x": arr, "widths": widths})[0]
            for k, j in enumerate(idx):
                names = getattr(self, "script_groups", GROUPS)
                pk_row = p[k]
                if p.ndim > 1 and "hw" in names:
                    # The hw class wins only at hw_min or more; otherwise the
                    # line keeps the best of the script classes.
                    h = names.index("hw")
                    if pk_row[h] < getattr(self, "hw_min", 0.0):
                        pk_row = pk_row.copy()
                        pk_row[h] = -1.0
                g = (
                    ("cjk" if p[k] > 0.5 else "base")
                    if p.ndim == 1
                    else names[int(pk_row.argmax())]
                )
                pk = (
                    max(float(p[k]), 1.0 - float(p[k]))
                    if p.ndim == 1
                    else float(pk_row.max())
                )
                keep = g in self.groups or (
                    g == "hw" and getattr(self, "rec_hw", None) is not None
                )
                out[j] = g if keep else "base"
                sure[j] = pk > ROUTE_SURE
        return out, sure

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

    _page_rtl = False  # set by assemble: the page reads right to left

    def _is_space(self, tok) -> bool:
        return tok == " " or tok == self.vocab.index.get(" ")

    def _order_texts(self, rec: list, texts: list) -> list:
        """The texts the order and layout model reads: a line with a spliced formula
        (math_span_img) as the recognizer read it, so blocks and tables stay as
        without the formula."""
        return [
            self._decode(unspliced(r[2])) if r[2] and any(isinstance(t[0], Formula) for t in r[2]) else tx
            for r, tx in zip(rec, texts)
        ]

    def _decode(self, toks: list) -> str:
        """Text of tokens: (vocab id, x) from the main recognizer, (char, x) from a
        group recognizer (visual order: right-to-left text is flipped by the
        page direction), or (char, x, content index) from a PDF text layer."""
        if toks and isinstance(toks[0][0], str):
            # A formula token (math_span_img) stays one left-to-right unit: a
            # private-use character (strong LTR) holds its place through the flip.
            forms = [tk[0] for tk in toks if isinstance(tk[0], Formula)]
            if forms:
                k = iter(range(len(forms)))
                toks = [(chr(0xE000 + next(k)), *tk[1:]) if isinstance(tk[0], Formula) else tk for tk in toks]
            text = " ".join("".join(tk[0] for tk in toks).split())
            if len(toks[0]) == 2:
                text = unicodedata.normalize("NFC", text)  # Hangul jamo -> syllables
                if has_rtl(text):
                    text = flip(text, base_rtl=self._page_rtl or None)
            for i, f in enumerate(forms):
                text = text.replace(chr(0xE000 + i), f)
            return text
        return self.vocab.decode([tk[0] for tk in toks])

    def table_blocks(self, gray, boxes, rec, order, layout, geo, from_layer=False) -> dict:
        """HTML per table block (key: the block's line indices as a tuple) from
        tsr_rel.onnx, plus "ms": the time spent.  A block the model cannot read
        (fewer than two rows or columns) is left out: it gets the geometric table."""
        t0 = time.perf_counter()
        out: dict = {}
        if from_layer and not self.tsr_rel_layer:
            out["ms"] = 0.0
            return out
        mirror = bool(self._page_rtl)
        for blk in split_blocks(order, layout[1]):
            if len(blk) < 2 or int(layout[0][blk].sum(0).argmax()) != TABLE:
                continue
            html = read_table(
                self.tsr_rel,
                gray,
                [(boxes[i, :4], unspliced(rec[i][2])) for i in blk],
                self._is_space,
                self._decode,
                mirror,
                self.tsr_rel_span,
                self.tsr_split,
                self.tsr_refine,
            )
            if html:
                out[tuple(blk)] = html
        out["ms"] = 1000 * (time.perf_counter() - t0)
        return out

    def line_words(self, box, rec) -> list[Word]:
        """The words of one line (text, conf, tokens), in reading order.  A word is
        a run of tokens between spaces; its box spans its characters' x positions
        (as split_line cuts cells) at the line's height.  A line whose text is not
        its tokens' text (the math reader replaced it) is one word."""
        text, _, toks = rec
        box = [round(float(v), 1) for v in box[:4]]
        if not text:
            return []
        if not toks or self._decode(toks) != text:
            return [Word(box, text)]
        ws = [
            Word([round(float(v), 1) for v in w["box"]], w["text"])
            for w in words_of([(box, toks)], self._is_space, self._decode)
        ]
        # Reading order is the line text's order (tokens are in visual order from
        # the recognizers and in content order from a text layer): each word of
        # the text takes the first unused box with its text, left to right, or
        # right to left on a right-to-left line.
        strong = next((c for c in text if c.isalpha()), "")
        if strong and is_rtl(strong):
            ws.sort(key=lambda w: -w.box[2])
        else:
            ws.sort(key=lambda w: w.box[0])
        out, used = [], [False] * len(ws)
        for t in text.split():
            k = next((j for j, w in enumerate(ws) if not used[j] and w.text == t), None)
            if k is None:
                return ws
            used[k] = True
            out.append(ws[k])
        return out if all(used) else ws

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
        cuts = None
        if native.lib is not None and isinstance(toks, list):
            # the same cuts, found in C++: the tokens that start a new segment
            try:
                space = self.vocab.index.get(" ")
                cuts = native.lib.split_points(toks, gap * h, at, space)
            except Exception as e:  # noqa: BLE001  (the loop below finds them)
                native.failed("split_points", e)
        if cuts is not None:
            bounds = [0, *cuts, len(toks)]
            segs = [toks[a:b] for a, b in zip(bounds, bounds[1:])]
        else:
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
        self,
        boxes: np.ndarray,
        texts: list[str],
        wh: tuple[int, int],
        gray: np.ndarray | None = None,
        img_boxes: np.ndarray | None = None,
    ) -> tuple[list[int], tuple | None, np.ndarray | None]:
        """Reading order, (cls logits, same-block logits, table logits) when the
        order model has layout heads, and the pairwise order logits.  An order
        model with line thumbnails (input "thumbs") gets them from ``gray`` at
        ``img_boxes`` (the boxes on the page image; ``boxes`` may be mirrored for
        an RTL page); without ``gray`` its lines have no image (has_img 0)."""
        if len(boxes) < 2 or self.order is None:
            order = order_xycut(boxes[:, :4]) if len(boxes) else []
            return order, None, None
        f = line_features(boxes[:, :4], wh, texts)[None]
        feed = {"feats": f, "mask": np.ones((1, len(boxes)), bool)}
        if getattr(self, "_order_thumbs", False):
            n = len(boxes)
            if gray is not None:
                src = boxes if img_boxes is None else img_boxes
                th, tw, interp = getattr(self, "_thumb", (THUMB_H, THUMB_W, "area"))
                feed["thumbs"] = line_thumbs(gray, src[:, :4], th=th, tw=tw, interp=interp)[None]
                feed["has_img"] = np.ones((1, n), np.float32)
            else:
                th, tw, _ = getattr(self, "_thumb", (THUMB_H, THUMB_W, "area"))
                feed["thumbs"] = np.zeros((1, n, 1, th, tw), np.float32)
                feed["has_img"] = np.zeros((1, n), np.float32)
        outs = self.order.run(None, feed)
        if len(outs) >= 3:
            # Layout heads: blocks from the same-block head, lines inside a block by
            # geometry, blocks by the order head.
            cls, same = outs[1][0], outs[2][0]
            # same row / column / cell logits, when asked for
            tab = outs[3][0] if len(outs) > 3 and self.table_heads else None
            order = decode_blocks_order(outs[0][0], same, boxes[:, :4], cls)
            return order, (cls, same, tab), outs[0][0]
        return decode_order(outs[0][0]), None, outs[0][0]

    def orientation_probs(self, gray: np.ndarray) -> np.ndarray | None:
        """orient.onnx class probabilities for the page (None without the model)."""
        if self.orient is None:
            return None
        return self.orient.run(None, {"x": orient_input(gray)[None, None]})[0][0]

    def orientation(self, gray: np.ndarray) -> int:
        """Quarter turns (counter-clockwise) the page content is turned by, from
        orient.onnx: 0 without the model or when no turn is more likely than
        orient_min; VERTICAL for a page set in vertical columns."""
        p = self.orientation_probs(gray)
        if p is None:
            return 0
        k = int(p.argmax())
        return k if p[k] > self.orient_min else 0

    def upright(
        self, gray: np.ndarray, turn: bool = True, k: int | None = None
    ) -> tuple[np.ndarray, int, float]:
        """The page turned upright (``k`` quarter turns, or orient.onnx's choice)
        and straightened (estimate_skew): (page, quarter turns undone, degrees of
        skew undone)."""
        a = estimate_skew(gray, pool=getattr(self, "pool", None))
        if k is None:
            k = (
                self.orientation(rotate_page(gray, a) if abs(a) >= DESKEW_MIN else gray)
                if turn
                else 0
            )
        return self._turn(gray, k, a)

    def _turn(
        self, gray: np.ndarray, k: int, a: float, small: np.ndarray | None = None
    ) -> tuple[np.ndarray, int, float]:
        """``small``: the page's preview (``preview``), for measuring the angle again."""
        # A half turn keeps the rows, so the angle stands; after a quarter turn the
        # rows were columns, so the angle is measured again on the turned page.
        q = 1 if k == VERTICAL else -k
        if k == VERTICAL:
            # A quarter turn counter-clockwise: columns become lines, the first
            # (right) column on top, each read left to right.
            gray = np.ascontiguousarray(np.rot90(gray, 1))
        elif k:
            gray = np.ascontiguousarray(np.rot90(gray, -k))
        if k in (1, 3, VERTICAL):
            a = estimate_skew(
                np.ascontiguousarray(np.rot90(small, q)) if small is not None else gray,
                pool=getattr(self, "pool", None),
            )
        if abs(a) >= DESKEW_MIN:
            gray = rotate_page(gray, a)
        else:
            a = 0.0
        return gray, k, a

    def __call__(
        self, gray: np.ndarray, turn: bool = True, rotation: int | None = None
    ) -> Page:
        """OCR of a page image.  turn=False: the page is known to be upright (no
        orient.onnx).  rotation: the page's content is known to be turned by this
        many degrees counter-clockwise (0, 90, 180, 270; as Page.rotation reports
        it): Indra turns it back and does not ask orient.onnx."""
        t0 = time.perf_counter()
        small = preview(gray)
        a = estimate_skew(small, pool=getattr(self, "pool", None))
        # The orientation model reads the page straightened by its row angle (a
        # tilted page could read as turned).
        k = 0
        if rotation is not None:
            k = quarter_turns(rotation)
        elif turn:
            p = self.orientation_probs(
                rotate_page(small, a) if abs(a) >= DESKEW_MIN else small
            )
            if p is not None:
                k = int(p.argmax())
                if p[k] <= self.orient_min:
                    sideways = float(p[1] + p[3] + (p[VERTICAL] if len(p) > VERTICAL else 0))
                    k = self.read_turn(gray, a, small, p) if sideways >= SIDEWAYS_MASS else 0
                elif k == 2:
                    k = self.read_half_turn(gray, a, small)
        orient_ms = 1000 * (time.perf_counter() - t0)
        page = self._read(gray, k, a, small)
        page.timings["orient_ms"] += orient_ms
        return page

    def read_turn(
        self, gray: np.ndarray, a: float, small: np.ndarray | None, p: np.ndarray
    ) -> int:
        """The turn of a page orient.onnx is unsure of but sees as likely sideways
        (probabilities ``p``): up to SIDEWAYS_LINES of the longest lines are read as the
        page stands, turned counter-clockwise, and turned clockwise (the counter-clockwise
        lines turned half round); the reading the recognizers are surest of picks the
        side.  On the counter-clockwise side, a quarter turn (3) and vertical columns
        (VERTICAL) give the same crops, so orient.onnx, which sees the characters'
        shape, picks between them.  The lines are read by the recognizers alone
        (recognize(math=False)): only their confidences are kept."""

        def longest(img):
            bx = self.detect(img)
            return bx[np.argsort(-(bx[:, 2] - bx[:, 0]))[:SIDEWAYS_LINES]]

        def conf(img, bx):
            if not len(bx):
                return 0.0
            return float(np.mean([c for _, c, _ in self.recognize(img, bx, math=False)]))

        up, _, _ = self._turn(gray, 0, a, small)
        ccw, _, _ = self._turn(gray, 3, a, small)
        sel = longest(ccw)
        h, w = ccw.shape[:2]
        cw = np.ascontiguousarray(ccw[::-1, ::-1])  # the page turned clockwise
        sel_cw = sel.copy()
        sel_cw[:, :4] = np.stack(
            [w - sel[:, 2], h - sel[:, 3], w - sel[:, 0], h - sel[:, 1]], 1
        )
        side = VERTICAL if len(p) > VERTICAL and p[VERTICAL] > p[3] else 3
        scores = {0: conf(up, longest(up)), 1: conf(cw, sel_cw), side: conf(ccw, sel)}
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 0

    def read_half_turn(self, gray: np.ndarray, a: float, small: np.ndarray | None) -> int:
        """2 when up to SIDEWAYS_LINES of the longest lines read surer with the page
        turned half round than as it stands, else 0."""

        def conf(img):
            bx = self.detect(img)
            bx = bx[np.argsort(-(bx[:, 2] - bx[:, 0]))[:SIDEWAYS_LINES]]
            return float(np.mean([c for _, c, _ in self.recognize(img, bx, math=False)])) if len(bx) else 0.0

        up, _, _ = self._turn(gray, 0, a, small)
        down, _, _ = self._turn(gray, 2, a, small)
        return 2 if conf(down) > conf(up) else 0

    def _read(
        self, gray: np.ndarray, k: int, a: float, small: np.ndarray | None = None
    ) -> Page:
        t = {}
        t0 = time.perf_counter()
        gray, k, skew = self._turn(gray, k, a, small)
        t["orient_ms"] = 1000 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        boxes, pictures = self.detect_all(gray)
        t["det_ms"] = 1000 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        vertical = k == VERTICAL
        rec = self.recognize(gray, boxes, vertical=vertical)
        t["rec_ms"] = 1000 * (time.perf_counter() - t0)
        if getattr(self, "rot_gate", False) and not vertical and len(boxes):
            # Rotated lines the upright detector cannot find (rot_gate.py); a page
            # without a tall, narrow group of uncovered ink is read as before.
            t0 = time.perf_counter()
            boxes, rec = read_rotated(
                self, gray, boxes, rec, pictures, MIN_CONF, cls=getattr(self, "rot_cls", None), det_in=getattr(self, "_det_in", None)
            )
            t["rot_ms"] = 1000 * (time.perf_counter() - t0)
        page = self.assemble(
            (gray.shape[1], gray.shape[0]),
            boxes,
            rec,
            t,
            vertical=vertical,
            pictures=None if vertical else pictures,
            gray=gray,
        )
        page.rotation, page.skew = (0 if vertical else 90 * k), round(skew, 2)
        page.vertical = vertical
        if skew:  # boxes back on the page as turned, before the deskew
            wh = (gray.shape[1], gray.shape[0])
            for ln in page.lines:
                ln.box = unrotate_box(ln.box, skew, wh)
                for w in ln.words or ():
                    w.box = unrotate_box(w.box, skew, wh)
            for b in page.blocks:
                b["bbox"] = unrotate_box(b["bbox"], skew, wh)
        if vertical:  # back on the page as it is: (x, y) -> (W - y, x)
            W = gray.shape[0]

            def back(b):
                return [W - b[3], b[0], W - b[1], b[2]]

            for ln in page.lines:
                ln.box = back(ln.box)
                for w in ln.words or ():
                    w.box = back(w.box)
            for b in page.blocks:
                b["bbox"] = back(b["bbox"])
        return page

    def assemble(
        self,
        wh,
        boxes: np.ndarray,
        rec: list,
        t: dict,
        vertical: bool = False,
        pictures: np.ndarray | None = None,
        gray: np.ndarray | None = None,
        from_layer: bool = False,
    ) -> Page:
        """Lines -> split at wide gaps -> reading order -> markdown.

        ``rec`` holds (text, confidence, tokens) per detected line.  ``vertical``:
        the page is a vertical page turned a quarter turn (see ``upright``).
        ``gray``: the page the boxes are on; with tsr_rel.onnx, tables are read
        with it (``table_blocks``).  ``from_layer``: the lines come from a PDF
        text layer (meta.json "tsr_rel_layer": false leaves those tables to the
        geometric assembly)."""
        t0 = time.perf_counter()
        # A right-to-left page: lines read right to left (_decode), and reading
        # order and markdown work on the page mirrored left to right, so the
        # first column is the right one.  Letter counts do not depend on order.
        letters = [c for r in rec for c in r[0] if c.isalpha()]
        n_rtl = sum(is_rtl(c) for c in letters)
        self._page_rtl = n_rtl > len(letters) - n_rtl
        width = float(wh[0])

        def geo(b: np.ndarray) -> np.ndarray:
            if not self._page_rtl:
                return b
            m = np.array(b, np.float32).copy()
            m[..., 0], m[..., 2] = width - b[..., 2], width - b[..., 0]
            return m

        # Split each line at wide gaps (SPLIT_GAP).
        nb, nrec = [], []
        cuts = self.gutter_cuts(boxes, rec) if len(boxes) else []
        for b, (text, conf, toks), at in zip(boxes, rec, cuts):
            for bx, tx, tk in self.split_line(b, toks, SPLIT_GAP, at):
                nb.append(bx)
                nrec.append((tx, conf, tk))
        boxes, rec = tidy_lines(
            np.array(nb, np.float32).reshape(-1, 4),
            nrec,
            self._page_rtl,
            self.vocab.index.get(" "),
        )
        texts = [r[0] for r in rec]
        order, layout, _ = self.reading_order(geo(boxes), self._order_texts(rec, texts), wh, gray, boxes)
        pend = getattr(getattr(self, "math", None), "accepted", None)
        if pend and layout is not None and self.math.accept_head is not None:
            # The head sees the table class only now: a reading on a table line goes
            # back to the plain line when the head rejects it there.
            rec = list(rec)
            in_table = layout[0].argmax(-1) == BLOCK_CLASSES.index("Table")
            back = False
            for i in np.flatnonzero(in_table):
                q = pend.get(rec[i][0])
                if q is not None and q[0] is not None and self.math.accept(*q[1:], in_table=1.0) is None:
                    rec[i] = q[0]
                    back = True
            if back:
                texts = [r[0] for r in rec]
                order, layout, _ = self.reading_order(geo(boxes), self._order_texts(rec, texts), wh, gray, boxes)
        if pend is not None:
            pend.clear()
        if vertical and layout is not None:
            # The order model does not read vertical pages: turned, their columns of
            # short lines look like one table read row by row across tiers.
            # Blocks and order come from XY-cut instead, without grid classes.
            blocks = xycut_blocks(boxes[:, :4])
            order = [i for b in blocks for i in b]
            same = np.full((len(boxes), len(boxes)), -1.0, np.float32)
            for b in blocks:
                same[np.ix_(b, b)] = 1.0
            cls = layout[0].copy()
            for c in GRID_LABELS:
                cls[:, BLOCK_CLASSES.index(c)] = -1e4
            layout = (cls, same, None)
        elif layout is not None and layout[2] is not None:
            # The table heads expect one line per cell, but the detector can join
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
                    order, layout, _ = self.reading_order(geo(boxes), self._order_texts(rec, texts), wh, gray, boxes)
        note = "order=xycut" if vertical else "order=model"
        t["order_ms"] = 1000 * (time.perf_counter() - t0)
        t["total_ms"] = sum(v for k, v in t.items() if k != "total_ms")
        lines = [
            Line(
                [round(float(v), 1) for v in boxes[i, :4]],
                rec[i][0],
                round(rec[i][1], 4),
                self.line_words(boxes[i], rec[i]) if getattr(self, "words", False) else None,
            )
            for i in order
        ]
        md, blocks = None, []
        if layout is not None:
            blocks = page_blocks(geo(boxes[:, :4]), texts, order, layout[0], layout[1])
            for b in blocks:  # back on the page as it is
                b["bbox"] = [round(float(v), 1) for v in geo(np.array(b["bbox"]))]
            # Inside tables, lines split into cells at smaller gaps (CELL_GAP).
            cells = {
                i: [
                    (list(geo(np.array(bx, np.float32))), tx)
                    for bx, tx, _ in self.split_line(boxes[i], rec[i][2], CELL_GAP)
                ]
                for i in range(len(boxes))
            }
            has_pics = pictures is not None and len(pictures)
            tables = (
                self.table_blocks(gray, boxes, rec, order, layout, geo, from_layer)
                if self.tsr_rel is not None and gray is not None and not vertical
                else None
            )
            if tables:
                t["table_ms"] = tables.pop("ms")
            # Text-layer lines (from_layer) have other line geometry than OCR lines: they get their
            # own weight set, and no check when the file has none (the OCR set rejected real tables there).
            verifier = self.table_verify_layer if from_layer else self.table_verify
            if verifier is not None and not vertical:
                # The verifier reads line and cell texts: lines with a spliced formula
                # as the recognizer read them (the formula is not a table feature).
                vtexts = self._order_texts(rec, texts)
                vcells = cells
                if vtexts is not texts and any(a != b for a, b in zip(vtexts, texts)):
                    vcells = {
                        i: [(list(geo(np.array(bx, np.float32))), tx) for bx, tx, _ in self.split_line(boxes[i], unspliced(rec[i][2]), CELL_GAP)]
                        if vtexts[i] != texts[i] else cells[i]
                        for i in range(len(boxes))
                    }
                cls_v, tables = verifier.apply(
                    geo(boxes[:, :4]), vtexts, order, layout[0], layout[1], layout[2], vcells, tables, gray
                )
                if cls_v is not layout[0]:  # some Table blocks were false tables: relabel them
                    layout = (cls_v, layout[1], layout[2])
                    blocks = page_blocks(geo(boxes[:, :4]), texts, order, layout[0], layout[1])
                    for b in blocks:
                        b["bbox"] = [round(float(v), 1) for v in geo(np.array(b["bbox"]))]
            md = page_markdown(
                geo(boxes[:, :4]),
                texts,
                order,
                layout[0],
                layout[1],
                keep_furniture=self.furniture,
                cells=cells,
                tab=layout[2],
                pictures=geo(pictures) if has_pics else None,
                tables=tables,
            )
            if (
                has_pics
            ):  # picture blocks, in reading order by the same rule as the markdown
                for p in pictures:
                    cy = (p[1] + p[3]) / 2
                    at = next(
                        (
                            j
                            for j, b in enumerate(blocks)
                            if (b["bbox"][1] + b["bbox"][3]) / 2 > cy
                            and min(p[2], b["bbox"][2]) > max(p[0], b["bbox"][0])
                        ),
                        len(blocks),
                    )
                    blocks.insert(
                        at,
                        {
                            "label": "Picture",
                            "bbox": [round(float(v), 1) for v in p[:4]],
                            "text": "",
                        },
                    )
        elif len(boxes) < 2:
            md = "\n".join(texts)
        return Page(lines, t, md, note=note, blocks=blocks)


LEADER_CHARS = set(".·…‥_-–—=")  # dot leaders, fill-in rules
DASHES = set("-–—")  # a lone short dash can be a table cell; other lone marks are not
BULLETS = set("•◦▪▫●○■□‣⁃∙*")  # list markers (not dashes: a "-" can be a table cell)
# A numbered list marker: "1.", "12)", "a.", "B)", "(3)", "(b)", "(iv)".
ENUMERATOR = re.compile(
    r"(\d{1,3}|[A-Za-z])[.)]|\((\d{1,3}|[A-Za-z]|[ivxlcIVXLC]{1,5})\)"
)
# A list marker and its item's text are at most this many line heights apart
# (the cut rule outside tables: a line is cut only at wider gaps).
MARKER_GAP = 2.5


def tidy_lines(
    boxes: np.ndarray, rec: list, rtl: bool = False, space=None
) -> tuple[np.ndarray, list]:
    """Two kinds of lines carry no text of their own:

    - a line of leader characters only (form fill-in dots and rules, specks)
      is dropped: 3 or more of them, a box 3 or more times wider than tall
      (CTC reads a dotted rule as one "."), or any dot or underscore alone.  A
      short lone dash stays: it can be an empty table cell;
    - a lone list marker (a bullet, "1.", "(a)") joins the line that starts
      within MARKER_GAP line heights after it on the same row (on its left on a
      right-to-left page): the item keeps its marker.  A column gutter cut
      (``gutter_cuts``) splits the markers off a hanging-indent list, and a
      column of markers beside a column of items reads as a table.  On a
      left-to-right page the marker's tokens join too (``space``: the space
      token), so a table line still splits into cells at its gaps.

    Other lines without letters or digits ("-", "/ /", "( )") stay: they can be
    table cells or form content."""
    keep = [True] * len(rec)
    rec = list(rec)
    h = boxes[:, 3] - boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    for i, (text, conf, toks) in enumerate(rec):
        core = text.replace(" ", "")
        if core and set(core) <= LEADER_CHARS:
            wide = w[i] >= 3 * h[i]
            if len(core) >= 3 or wide or not set(core) & DASHES:
                keep[i] = False
    for i, (text, conf, toks) in enumerate(rec):
        t = text.strip()
        if not keep[i] or not (t in BULLETS or ENUMERATOR.fullmatch(t)):
            continue
        best, best_gap = None, None
        for j in range(len(rec)):
            if j == i or not keep[j] or not rec[j][0].strip():
                continue
            ov = min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1])
            if ov < 0.5 * min(h[i], h[j]):
                continue
            gap = boxes[i, 0] - boxes[j, 2] if rtl else boxes[j, 0] - boxes[i, 2]
            if -0.3 * h[i] <= gap <= MARKER_GAP * max(h[i], h[j]) and (
                best_gap is None or gap < best_gap
            ):
                best, best_gap = j, gap
        if best is None:
            continue
        tj, cj, kj = rec[best]
        chars = bool(kj) and isinstance(kj[0][0], str)
        sp = " " if chars else space
        # same kind of tokens (text-layer chars or recognizer ids) only
        if not rtl and toks and kj and sp is not None:
            if isinstance(toks[0][0], str) == chars:
                x = float(boxes[i, 2] + boxes[best, 0]) / 2
                kj = [*toks, (sp, x, *toks[-1][2:])] + list(kj)
        rec[best] = (f"{t} {tj.strip()}", cj, kj)
        boxes[best] = [
            min(boxes[i, 0], boxes[best, 0]),
            min(boxes[i, 1], boxes[best, 1]),
            max(boxes[i, 2], boxes[best, 2]),
            max(boxes[i, 3], boxes[best, 3]),
        ]
        keep[i] = False
    idx = [i for i, k in enumerate(keep) if k]
    return boxes[idx].reshape(-1, 4), [rec[i] for i in idx]


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
