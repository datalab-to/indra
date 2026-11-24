"""Math spans inside text-layer lines.

A PDF text layer writes inline math as plain characters ("O(2M · K)" for
O(2^{M \\cdot K})): the layer has the symbols but not the structure.  A small
tagger reads the characters of one line (each character and its box: size,
vertical offset, width and gap, over the line's median height) and marks the
characters that belong to math.  Each run of marked characters is cropped from
the page and read by MathRec, and a confident reading with math replaces the run.

Model (optional; without it nothing changes): math_span.onnx (ids [B, T],
geo [B, T, 4] -> probability [B, T], or [B, T, 2] with a second head that
predicts whether MathRec accepts a run's reading) and math_span.json (its vocabulary and
threshold).
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np

from .ort_session import session

BATCH = 32  # lines per model run


def _median(v: list, dtype) -> float:
    """np.median of a short list (the mean of the two middle values in dtype), without
    NumPy's per-call cost: encode runs once per line."""
    v = sorted(v)
    m = len(v) // 2
    if len(v) % 2:
        return v[m]
    return float((dtype(v[m - 1]) + dtype(v[m])) / dtype(2))


class SpanTagger:
    def __init__(self, onnx_dir: Path, so, providers):
        self.sess = session(str(onnx_dir / "math_span.onnx"), so, providers=providers)
        meta = json.loads((onnx_dir / "math_span.json").read_text(encoding="utf-8"))
        self.stoi = {t: i for i, t in enumerate(meta["itos"])}
        self.threshold = float(meta.get("threshold", 0.5))
        # A model with a second output also says, per char, whether MathRec will
        # accept a reading of its run: a run is read only when its mean is this high.
        self.read_threshold = float(meta.get("read_threshold", 0.5))
        # pack_gap: the model zeroes its padding at every layer and reaches at most
        # this many cells, so all lines of a page can go through it as one row with
        # pack_gap padding cells between them (one run, the same outputs).
        self.pack_gap = int(meta.get("pack_gap", 0))
        self.unknown = self.stoi["<cat:?>"]

    def _packed(self, todo: list, enc: dict) -> dict:
        """All lines in one row, pack_gap padding cells apart: line -> its outputs."""
        g = self.pack_gap
        T = sum(len(enc[j][0]) for j in todo) + g * (len(todo) + 1)
        ids = np.zeros((1, T), np.int64)
        geo = np.zeros((1, T, enc[todo[0]][1].shape[1]), np.float32)
        at, pos = {}, g
        for j in todo:
            a, x = enc[j]
            ids[0, pos : pos + len(a)], geo[0, pos : pos + len(a)] = a, x
            at[j] = pos
            pos += len(a) + g
        prob = self.sess.run(None, {"ids": ids, "geo": geo})[0][0]
        return {j: prob[at[j] : at[j] + len(enc[j][0])] for j in todo}

    @classmethod
    def load(cls, onnx_dir: Path, so, providers) -> "SpanTagger | None":
        if (onnx_dir / "math_span.onnx").exists() and (
            onnx_dir / "math_span.json"
        ).exists():
            return cls(onnx_dir, so, providers)
        return None

    def encode(self, chars: list) -> tuple[np.ndarray, np.ndarray]:
        """chars: [(c, x0, y0, x1, y1), ...] of one line, in reading order."""
        b = np.array([c[1:5] for c in chars], np.float64)
        h = (b[:, 3] - b[:, 1]).astype(np.float32)
        mh = float(_median(h.tolist(), np.float32)) or 1.0
        mid = (b[:, 1] + b[:, 3]) / 2
        mc = float(_median(mid.tolist(), np.float64))
        ids = [
            self.stoi.get(c[0])
            or self.stoi.get("<cat:" + unicodedata.category(c[0]) + ">", self.unknown)
            for c in chars
        ]
        gap = np.zeros(len(chars))
        gap[1:] = b[1:, 0] - b[:-1, 2]
        geo = np.stack(
            [b[:, 3] - b[:, 1], mid - mc, b[:, 2] - b[:, 0], gap], 1
        ) / mh
        return np.array(ids, np.int64), np.clip(geo.astype(np.float32), -5, 5)

    def spans(self, lines: list) -> list[list[tuple[int, int]]]:
        """Per line (chars as in encode), the [start, end) runs of chars the tagger
        calls math.  Lines are padded at the end (id 0, zero geometry); the model
        must give a line the same output in any batch."""
        out: list = [[] for _ in lines]
        # Lines of similar length go through the model together (BATCH at a time),
        # so little of each batch is padding; the model must not depend on it.
        todo = sorted((j for j, ch in enumerate(lines) if ch), key=lambda j: len(lines[j]))
        enc = {j: self.encode(lines[j]) for j in todo}
        probs = self._packed(todo, enc) if self.pack_gap and todo else {}
        for s in range(0, len(todo) if not probs else 0, BATCH):
            part = todo[s : s + BATCH]
            T = max(len(enc[j][0]) for j in part)
            ids = np.zeros((len(part), T), np.int64)
            geo = np.zeros((len(part), T, enc[part[0]][1].shape[1]), np.float32)
            for r, j in enumerate(part):
                a, g = enc[j]
                ids[r, : len(a)], geo[r, : len(a)] = a, g
            prob = self.sess.run(None, {"ids": ids, "geo": geo})[0]
            for r, j in enumerate(part):
                probs[j] = prob[r, : len(enc[j][0])]
        for j in todo:
            p = probs[j]
            read = p[:, 1] if p.ndim == 2 else None
            p = p[:, 0] if p.ndim == 2 else p
            start = None
            for i, v in enumerate(list(p) + [0.0]):
                if v >= self.threshold and start is None:
                    start = i
                elif v < self.threshold and start is not None:
                    if read is None or read[start:i].mean() >= self.read_threshold:
                        out[j].append((start, i))
                    start = None
        return out
