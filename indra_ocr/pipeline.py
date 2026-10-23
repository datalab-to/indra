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
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .det_post import decode, page_to_input
from .layout_post import page_blocks, page_markdown
from .line_mask import crop_lines, line_darkness
from .math import MATH_MIN_CONF, MathReader
from .order_post import (
    BLOCK_CLASSES,
    GRID_LABELS,
    decode_blocks_order,
    decode_order,
    line_features,
    order_xycut,
    xycut_blocks,
)
from .rec_preproc import PAD_X, pad_batch, width_batches
from .rtl import flip, has_rtl, is_rtl
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
VERTICAL = 4  # orient.onnx class (5-class models): the page is set in vertical columns


# A recognized line under this confidence (mean CTC peak probability) gets no
# text: on empty form fields, checkboxes and rules the recognizer reads "-", "I"
# or "1" at low confidence.  The line's box stays (layout and reading order).
MIN_CONF = 0.5

# Beam search with a language model (lm.npz for the base recognizer, optional
# lm_<group>.npz for a group) reads a recognizer's lines under this confidence;
# the rest keep the greedy reading.  alpha weighs the LM, beta rewards each token
# (tuned on held-out handwriting; meta.json "lm_params" overrides them per group).
LM_GATE = 0.9
LM_ALPHA = 0.5
LM_BETA = 1.5
LM_BEAM = 8
# A class under this probability at a step is not tried by the beam search
# (1e-2 cost 0.2 OPB handwriting points; 3e-3 changes no OPB score).
LM_CAND_FLOOR = 3e-3
OUT_GREEDY = ["ids", "conf", "lengths"]
OUT_LM = ["blank", "topv", "topi"]

# Recognizer groups, in script.onnx's class order unless meta.json names them
# ("script_groups", e.g. base and world).  base is rec.onnx.  An older
# script.onnx has one output: the chance that the line holds Chinese characters.
GROUPS = ("base", "cjk", "rtl", "other")

# A line goes to one recognizer when the classifier is surer than this; otherwise
# base reads it as well and the more confident reading wins.  Measured
# (script_w2, eval lines): every Latin line sent elsewhere was under 0.95, and
# 3.4% of the right non-base lines are read twice.
ROUTE_SURE = 0.95

# det.onnx picture channel: a region is a picture above this probability and
# this share of the page area.
PICTURE_MIN = 0.5
PICTURE_MIN_AREA = 0.004
DESKEW_MAX = 45.0  # degrees of tilt searched (quarter turns cover the rest)
DESKEW_MIN = 0.3  # smaller tilts are left alone (the search step is 0.1)


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
    over +-DESKEW_MAX in 2 degree steps on a half-size thumbnail; the two best
    angles and 0 are then refined in 0.5 and 0.1 degree steps on the full
    thumbnail (``side`` pixels), where small text rows are still resolved.
    pool: a thread pool that scores the angles in parallel (same result)."""
    run = pool.map if pool is not None else map
    fine = _ink_thumb(gray, side)
    if fine.mean() < 0.003 * 255:  # a blank page has no rows to straighten
        return 0.0
    coarse = _ink_thumb(gray, side // 2)
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


def gpu_run_options(providers: list):
    """RunOptions for sessions on a GPU: give the arena's free memory back after
    every run.  None on CPU.

    ONNX Runtime's CUDA provider keeps the largest allocation each session ever
    needed (its BFC arena never shrinks on its own), and the line batches change
    shape with every page, so a process grows until the device is full.  119
    mixed pages on an A100 (onnxruntime-gpu 1.19.2): 0.57 GB after loading,
    2.1 GB after the run and held for good; in a service with 8 workers on a
    16 GB T4, one worker reached 5.2 GB and pages failed with CUDA OOM.  With
    ``memory.enable_memory_arena_shrinkage`` on each run the same 119 pages
    idle at 0.66 GB (peak 1.7 GB while a page runs), at the same speed (4.7
    vs 4.5 pages/s).  The CPU arena is off instead (enable_cpu_mem_arena
    above), which this does not touch.  The device is the provider's
    ``device_id`` (0 by default)."""
    for p in providers:
        name, opts = (p, {}) if isinstance(p, str) else (p[0], p[1] or {})
        if name == "CUDAExecutionProvider":
            import onnxruntime as ort

            ro = ort.RunOptions()
            ro.add_run_config_entry(
                "memory.enable_memory_arena_shrinkage", f"gpu:{opts.get('device_id', 0)}"
            )
            return ro
    return None


class OrtOCR:
    ro = None  # RunOptions for every run: arena shrinkage on a GPU (gpu_run_options)

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
        self.ro = gpu_run_options(prov)  # passed to every run (None on CPU)

        def need(name: str) -> str:
            p = onnx_dir / name
            if not p.exists():
                raise FileNotFoundError(p)
            return str(p)

        if rec_int8 is None:
            rec_int8 = (onnx_dir / "rec.int8.onnx").exists()
        self.det = ort.InferenceSession(need("det.onnx"), so, providers=prov)
        self._pic = None  # the last page's picture channel, (logit map, scale)
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
        # Scripts: a line classifier routes each line to its group's recognizer
        # (rec_<group>.onnx with meta["vocab_<group>"]), in script.onnx's class
        # order meta["script_groups"] (default GROUPS).
        self.script, self.groups = None, {}
        self.script_groups = tuple(meta.get("script_groups", GROUPS))
        for g in [k[len("vocab_") :] for k in meta if k.startswith("vocab_")]:
            if (onnx_dir / f"rec_{g}.onnx").exists():
                self.groups[g] = (
                    ort.InferenceSession(
                        str(onnx_dir / f"rec_{g}.onnx"), so, providers=prov
                    ),
                    Vocab.from_json(meta[f"vocab_{g}"]),
                )
        if self.groups and (onnx_dir / "script.onnx").exists():
            self.script = ort.InferenceSession(
                str(onnx_dir / "script.onnx"), so, providers=prov
            )
        # Math: a line classifier routes lines with math to MathRec (math.py).
        self.math = MathReader.load(onnx_dir, so, prov, self.ro) if math else None
        if math and self.math is None:
            raise FileNotFoundError(f"math=True needs the math models in {onnx_dir}")
        self.long_side = meta["det_long_side"]
        self.height = meta["rec_height"]
        # rec.onnx may read a tall window (the line and the page around it);
        # the classifiers and group recognizers read the line band only.
        self.window = meta.get("rec_window", 1)
        # Language models: lm.npz for rec.onnx, lm_<group>.npz for a group's
        # recognizer.  Each needs its recognizer's top-k outputs (newer exports).
        # meta.json "lm_params" may set a group's alpha, beta and gate (tuned
        # with the model); the defaults are the base recognizer's.
        self.lms: dict = {}  # id(session) -> (NgramLM, alpha, beta, gate)
        params = meta.get("lm_params", {})
        for name, sess in [("base", self.rec)] + [
            (g, s) for g, (s, _) in self.groups.items()
        ]:
            f = onnx_dir / ("lm.npz" if name == "base" else f"lm_{name}.npz")
            if f.exists() and set(OUT_LM) <= {o.name for o in sess.get_outputs()}:
                from .lm import NgramLM

                p = params.get(name, {})
                self.lms[id(sess)] = (
                    NgramLM(f),
                    p.get("alpha", LM_ALPHA),
                    p.get("beta", LM_BETA),
                    p.get("gate", LM_GATE),
                )
        self.lm = self.lms.get(id(self.rec), (None,))[0]
        self.adaptive = adaptive

    def _detect_at(self, gray: np.ndarray, long_side: int):
        x, s = page_to_input(gray, long_side)
        out = self.det.run(None, {"x": x[None, None]}, self.ro)[0][0]
        self._pic = (out[3], s) if out.shape[0] > 3 else None
        return decode(out), s

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
    ) -> list[tuple[str, float, list]]:
        """(text, confidence, tokens) per line; tokens = [(token id, page x)],
        x from the CTC peak's timestep mapped back through the crop.  A line
        under MIN_CONF comes back empty.  vertical: the page is a vertical one
        turned a quarter turn (every line to the CJK recognizer)."""
        # CenterMask first when the model folder has mask.onnx (same step as training)
        pool = getattr(self, "pool", None)
        window = getattr(self, "window", 1)
        # Light-on-dark is decided once per line box, for both crops.
        dark = line_darkness(gray, boxes[:, :4], pool) if window > 1 else None
        crops = crop_lines(
            gray, boxes[:, :4], self.height, self.mask, pool=pool, dark=dark
        )
        tall = (
            crop_lines(gray, boxes[:, :4], self.height, window=window, pool=pool, dark=dark)
            if window > 1
            else crops
        )
        out: list = [("", 0.0, [])] * len(crops)
        groups = [(self.rec, self.vocab, list(range(len(crops))), False)]
        if vertical and "cjk" in self.groups:
            # A vertical page's lines are CJK columns, characters on their side;
            # the CJK recognizer reads those (and the Latin set across them).
            sess, vocab = self.groups["cjk"]
            groups = [(sess, vocab, list(range(len(crops))), True)]
        elif self.script is not None and crops:
            names, sure = self.line_groups(crops, batch)
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
        best_conf = [-1.0] * len(crops)
        for sess, vocab, sel, as_chars in groups:
            for part in width_batches([crops[j].shape[1] for j in sel], batch):
                idx = [sel[i] for i in part]
                src = tall if sess is self.rec else crops
                arr, widths = pad_batch([src[j] for j in idx])
                lm = getattr(self, "lms", {}).get(id(sess))
                use_lm = lm is not None
                res = sess.run(
                    OUT_GREEDY + (OUT_LM if use_lm else []),
                    {"x": arr, "widths": widths},
                    self.ro,
                )
                best, pmax, lengths = res[:3]
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
                    ids = seq[t].tolist()
                    conf = float(pm[keep].mean()) if keep.any() else 0.0
                    if use_lm and MIN_CONF <= conf < lm[3]:
                        from .lm import beam_search

                        ids, steps = beam_search(
                            res[3][k, :n],
                            res[4][k, :n],
                            res[5][k, :n],
                            lm[0],
                            lm[1],
                            lm[2],
                            LM_BEAM,
                            cand_floor=LM_CAND_FLOOR,
                        )
                        t = np.array(steps, np.int64)
                    xs = (ix0 + (t + 0.5) * stride * f).tolist()
                    # A group recognizer's ids mean nothing in self.vocab: its tokens
                    # are characters, like the text-layer tokens.
                    toks = (
                        list(zip([vocab.chars[i - 1] for i in ids], xs))
                        if as_chars
                        else list(zip(ids, xs))
                    )
                    if conf <= best_conf[j]:
                        continue
                    best_conf[j] = conf
                    out[j] = (
                        (vocab.decode(ids), conf, toks)
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

    def box_groups(self, gray: np.ndarray, boxes: np.ndarray) -> tuple[list[str], list[bool]]:
        """line_groups for line boxes on the page (crops as recognize makes them);
        every line is "base" when the model folder has no script.onnx."""
        if self.script is None or not len(boxes):
            return ["base"] * len(boxes), [False] * len(boxes)
        pool = getattr(self, "pool", None)
        window = getattr(self, "window", 1)
        dark = line_darkness(gray, boxes[:, :4], pool) if window > 1 else None
        crops = crop_lines(gray, boxes[:, :4], self.height, self.mask, pool=pool, dark=dark)
        return self.line_groups(crops)

    def line_groups(self, crops: list, batch: int = 64) -> tuple[list[str], list[bool]]:
        """Per line crop, the group whose recognizer reads it (script.onnx), and
        whether the classifier is sure of it (probability over ROUTE_SURE); a
        group without its recognizer goes to base."""
        out = ["base"] * len(crops)
        sure = [True] * len(crops)
        for idx in width_batches([c.shape[1] for c in crops], batch):
            arr, widths = pad_batch([crops[j] for j in idx])
            p = self.script.run(None, {"x": arr, "widths": widths}, self.ro)[0]
            for k, j in enumerate(idx):
                g = (
                    ("cjk" if p[k] > 0.5 else "base")
                    if p.ndim == 1
                    else getattr(self, "script_groups", GROUPS)[int(p[k].argmax())]
                )
                pk = (
                    max(float(p[k]), 1.0 - float(p[k]))
                    if p.ndim == 1
                    else float(p[k].max())
                )
                out[j] = g if g in self.groups else "base"
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

    def _decode(self, toks: list) -> str:
        """Text of tokens: (vocab id, x) from the main recognizer, (char, x) from a
        group recognizer (visual order: right-to-left text is flipped by the
        page direction), or (char, x, content index) from a PDF text layer."""
        if toks and isinstance(toks[0][0], str):
            text = " ".join("".join(tk[0] for tk in toks).split())
            if len(toks[0]) == 2:
                text = unicodedata.normalize("NFC", text)  # Hangul jamo -> syllables
                if has_rtl(text):
                    text = flip(text, base_rtl=self._page_rtl or None)
            return text
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
            None, {"feats": f, "mask": np.ones((1, len(boxes)), bool)}, self.ro
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

    def orientation_probs(self, gray: np.ndarray) -> np.ndarray | None:
        """orient.onnx class probabilities for the page (None without the model)."""
        if self.orient is None:
            return None
        return self.orient.run(None, {"x": orient_input(gray)[None, None]}, self.ro)[0][0]

    def orientation(self, gray: np.ndarray) -> int:
        """Quarter turns (counter-clockwise) the page content is turned by, from
        orient.onnx: 0 without the model or when no turn is more likely than
        ORIENT_MIN; VERTICAL for a page set in vertical columns."""
        p = self.orientation_probs(gray)
        if p is None:
            return 0
        k = int(p.argmax())
        return k if p[k] > ORIENT_MIN else 0

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

    def __call__(self, gray: np.ndarray, turn: bool = True) -> Page:
        """OCR of a page image.  turn=False: the page is known to be upright (no
        orient.onnx)."""
        t0 = time.perf_counter()
        small = preview(gray)
        a = estimate_skew(small, pool=getattr(self, "pool", None))
        # The orientation model reads the page straightened by its row angle (a
        # tilted page could read as turned).
        k = (
            self.orientation(rotate_page(small, a) if abs(a) >= DESKEW_MIN else small)
            if turn
            else 0
        )
        orient_ms = 1000 * (time.perf_counter() - t0)
        page = self._read(gray, k, a, small)
        page.timings["orient_ms"] += orient_ms
        return page

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
        page = self.assemble(
            (gray.shape[1], gray.shape[0]),
            boxes,
            rec,
            t,
            vertical=vertical,
            pictures=None if vertical else pictures,
        )
        page.rotation, page.skew = (0 if vertical else 90 * k), round(skew, 2)
        page.vertical = vertical
        if skew:  # boxes back on the page as turned, before the deskew
            wh = (gray.shape[1], gray.shape[0])
            for ln in page.lines:
                ln.box = unrotate_box(ln.box, skew, wh)
            for b in page.blocks:
                b["bbox"] = unrotate_box(b["bbox"], skew, wh)
        if vertical:  # back on the page as it is: (x, y) -> (W - y, x)
            W = gray.shape[0]

            def back(b):
                return [W - b[3], b[0], W - b[1], b[2]]

            for ln in page.lines:
                ln.box = back(ln.box)
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
    ) -> Page:
        """Lines -> split at wide gaps -> reading order -> markdown.

        ``rec`` holds (text, confidence, tokens) per detected line.  ``vertical``:
        the page is a vertical page turned a quarter turn (see ``upright``)."""
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

        # Text-layer labels split a line at wide gaps; do the same (SPLIT_GAP).
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
        order, layout, _ = self.reading_order(geo(boxes), texts, wh)
        if vertical and layout is not None:
            # The order model saw no vertical pages: turned, their columns of
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
                    order, layout, _ = self.reading_order(geo(boxes), texts, wh)
        note = "order=xycut" if vertical else "order=model"
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
            blocks = page_blocks(geo(boxes[:, :4]), texts, order, layout[0], layout[1])
            for b in blocks:  # back on the page as it is
                b["bbox"] = [round(float(v), 1) for v in geo(np.array(b["bbox"]))]
            # Inside tables, labels split cells at smaller gaps (CELL_GAP).
            cells = {
                i: [
                    (list(geo(np.array(bx, np.float32))), tx)
                    for bx, tx, _ in self.split_line(boxes[i], rec[i][2], CELL_GAP)
                ]
                for i in range(len(boxes))
            }
            has_pics = pictures is not None and len(pictures)
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
# A list marker and its item's text are at most this many line heights apart:
# the label rule (training labels cut a text-layer line only at wider gaps).
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
