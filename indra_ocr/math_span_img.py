"""Inline math inside lines that MathRec does not read whole.

MathRec writes LaTeX and the letters of its own vocabulary: a Cyrillic, Korean,
Arabic or Devanagari line with inline math stays with its script recognizer,
which reads the formula as plain characters.  A small image tagger marks the
columns of such a line that hold math; MathRec reads each marked run's crop, and
a confident reading replaces the characters that the recognizer read there.

Model (optional; without it nothing changes): math_span_img.onnx (ink maps
x [B, 1, H, W] + widths [B] -> probability [B, W / stride] per column) and
math_span_img.json (height, stride, threshold, min_cols, merge_cols, accept_plain).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .rec_preproc import PAD_X, pad_batch, width_batches
from .ort_session import session


class Formula(str):
    """A formula token spliced into a line (one LaTeX run, kept whole through the
    right-to-left flip of the line's text).  held: the tokens it replaced."""

    held: list = []


class Pad(str):
    """A token the splice added (a space around a formula, or an empty token that
    keeps a replaced character's x)."""


class SpanImgTagger:
    def __init__(self, onnx_dir: Path, so, providers):
        self.sess = session(str(onnx_dir / "math_span_img.onnx"), so, providers=providers)
        meta = json.loads((onnx_dir / "math_span_img.json").read_text(encoding="utf-8"))
        self.height = int(meta["height"])
        self.stride = int(meta["stride"])
        self.threshold = float(meta.get("threshold", 0.5))
        self.min_cols = int(meta.get("min_cols", 2))  # a shorter run is noise
        self.merge_cols = int(meta.get("merge_cols", 1))  # a shorter gap joins two runs
        # accept_plain: a confident reading without structure ("$a = 0$", "$ABC$") also
        # replaces the run: the script recognizer reads Latin letters as its own look-alikes.
        self.accept_plain = bool(meta.get("accept_plain", False))

    @classmethod
    def load(cls, onnx_dir: Path, so, providers) -> "SpanImgTagger | None":
        if (onnx_dir / "math_span_img.onnx").exists() and (
            onnx_dir / "math_span_img.json"
        ).exists():
            return cls(onnx_dir, so, providers)
        return None

    def runs(self, prob: np.ndarray) -> list[tuple[int, int]]:
        """Column runs [a, b) at or above the threshold, short gaps joined."""
        out: list[list[int]] = []
        a = None
        for i, p in enumerate(list(prob) + [0.0]):
            if p >= self.threshold and a is None:
                a = i
            elif p < self.threshold and a is not None:
                if out and a - out[-1][1] <= self.merge_cols:
                    out[-1][1] = i
                else:
                    out.append([a, i])
                a = None
        return [(a, b) for a, b in out if b - a >= self.min_cols]

    def spans(self, crops: list, batch: int = 64) -> list[list[tuple[int, int]]]:
        """Per ink map [height, W]: column runs that hold math, in crop pixels."""
        out: list = [[] for _ in crops]
        for idx in width_batches([c.shape[1] for c in crops], batch):
            arr, widths = pad_batch([crops[j] for j in idx])
            prob = self.sess.run(None, {"x": arr, "widths": widths})[0]
            for k, j in enumerate(idx):
                n = crops[j].shape[1] // self.stride
                out[j] = [
                    (a * self.stride, b * self.stride) for a, b in self.runs(prob[k][:n])
                ]
        return out


def page_x(box, crop_w: int, xs: tuple[int, int]) -> tuple[float, float]:
    """Crop pixel columns -> page x (crop_line's padding and scale, as the
    recognizer's character positions)."""
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    h = max(1.0, y1 - y0)
    ix0 = np.floor(x0 - PAD_X * h)
    cw = np.ceil(x1 + PAD_X * h) - ix0
    f = cw / crop_w
    return float(ix0 + xs[0] * f), float(ix0 + xs[1] * f)


def splice(toks: list, x0: float, x1: float, latex: str) -> list:
    """The line's (char, x) tokens with those inside [x0, x1] replaced by one
    formula token; a space token on each side keeps the formula a word.  Each
    replaced character stays as an empty token at its x: the gaps between
    characters (cell and column cuts) are the same as before."""
    xc = (x0 + x1) / 2
    left = [t for t in toks if t[1] < x0]
    right = [t for t in toks if t[1] > x1]
    inside = [t for t in toks if x0 <= t[1] <= x1]
    f = Formula(latex)
    f.held = inside
    held = [(Pad(""), t[1]) for t in inside if t[0] != " "]
    mid = [(Pad(" "), x0), *sorted([*held, (f, xc)], key=lambda t: t[1]), (Pad(" "), x1)]
    if left and left[-1][0] == " ":
        mid = mid[1:]
    if right and right[0][0] == " ":
        mid = mid[:-1]
    return left + mid + right


def unspliced(toks: list) -> list:
    """The tokens as the recognizer read them, before any splice (the table model
    reads words and their characters: a formula word changes its structure)."""
    if not any(isinstance(t[0], Formula) for t in toks):
        return toks
    out = [t for t in toks if not isinstance(t[0], (Formula, Pad))]
    out += [h for t in toks if isinstance(t[0], Formula) for h in t[0].held]
    return sorted(out, key=lambda t: t[1])
