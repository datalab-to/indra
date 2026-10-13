"""Torch-free pre- and post-processing for LineNet (shared by training and inference).

Inference with ONNX Runtime needs only numpy and cv2, so these functions live
apart from the PyTorch model in det_model.py, which re-exports them.
"""

from __future__ import annotations

import math

import numpy as np

# Target geometry, shared by training (encode) and inference (decode).
CORE_FRAC = 0.4  # core band height as a fraction of the line ink height
CORE_X_SHRINK = 0.15  # the core is shorter than the line by this fraction of its height, at each end
OUT_STRIDE = 2
INPUT_MULTIPLE = 32  # H and W must divide by this


# ---------------------------------------------------------------------------
# Input preprocessing (shared by training and inference)
# ---------------------------------------------------------------------------


def page_to_input(gray: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    """Resize a uint8 grayscale page so its long side is ``long_side`` and pad
    to a multiple of 32 with white.  Returns (float32 [H, W] ink map, scale)."""
    import cv2

    h, w = gray.shape
    scale = long_side / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    if (nh, nw) != (h, w):
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        gray = cv2.resize(gray, (nw, nh), interpolation=interp)
    ph = math.ceil(nh / INPUT_MULTIPLE) * INPUT_MULTIPLE
    pw = math.ceil(nw / INPUT_MULTIPLE) * INPUT_MULTIPLE
    out = np.zeros((ph, pw), np.float32)  # ink map: 0 = white
    out[:nh, :nw] = 1.0 - gray.astype(np.float32) / 255.0
    return out, scale


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def encode_targets(
    boxes: np.ndarray,
    in_hw: tuple[int, int],
    valid_hw: tuple[int, int],
    ignore: np.ndarray | None = None,
):
    """Build training targets at 1/2 resolution.

    boxes:   [N, 4] line ink boxes in input pixels (x0, y0, x1, y1)
    in_hw:   padded input size; valid_hw: size of the real page inside it
    ignore:  [M, 4] regions with no loss, in input pixels

    Returns core [h, w] float32, dist [2, h, w] float32, weight [h, w] float32
    (0 on padding and ignore regions), and reg_mask [h, w] bool (core pixels).
    """
    s = OUT_STRIDE
    h, w = in_hw[0] // s, in_hw[1] // s
    core = np.zeros((h, w), np.float32)
    dist = np.zeros((2, h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    weight[: math.ceil(valid_hw[0] / s), : math.ceil(valid_hw[1] / s)] = 1.0
    if ignore is not None:
        for x0, y0, x1, y1 in ignore:
            weight[
                max(0, int(y0 // s)) : max(0, math.ceil(y1 / s)),
                max(0, int(x0 // s)) : max(0, math.ceil(x1 / s)),
            ] = 0.0
    # Larger lines first, so a small line drawn later keeps its own band.
    order = np.argsort(-(boxes[:, 3] - boxes[:, 1])) if len(boxes) else []
    ys = (np.arange(h, dtype=np.float32) + 0.5) * s  # pixel centres in input px
    for i in order:
        x0, y0, x1, y1 = boxes[i]
        lh = y1 - y0
        if lh < 2 or x1 - x0 < 2:
            continue
        cy = 0.5 * (y0 + y1)
        half = 0.5 * CORE_FRAC * lh
        r0 = max(0, int(math.floor((cy - half) / s)))
        r1 = min(h, max(r0 + 1, int(math.ceil((cy + half) / s))))
        sx = min(CORE_X_SHRINK * lh, 0.25 * (x1 - x0))
        c0 = max(0, int(math.floor((x0 + sx) / s)))
        c1 = min(w, max(c0 + 1, int(math.ceil((x1 - sx) / s))))
        if r0 >= h or c0 >= w:
            continue
        core[r0:r1, c0:c1] = 1.0
        dist[0, r0:r1, c0:c1] = (ys[r0:r1] - y0)[:, None]
        dist[1, r0:r1, c0:c1] = (y1 - ys[r0:r1])[:, None]
        # A line inside an ignore region is still a positive.
        weight[r0:r1, c0:c1] = np.maximum(weight[r0:r1, c0:c1], 1.0)
    return core, dist, weight, core > 0.5


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode(
    out: np.ndarray,
    thresh: float = 0.5,
    min_area: int = 2,
    grow_y: float = 0.0,
    grow_x: float = 0.0,
) -> np.ndarray:
    """Decode one page.  out: [3, h, w] (core logit, top dist, bottom dist).

    ``grow_y`` / ``grow_x`` widen every box by that fraction of its height
    (each side), to absorb the +-1 px error of the 1/2-resolution output.

    Returns [N, 5] boxes in input pixels: x0, y0, x1, y1, score."""
    import cv2

    prob = 1.0 / (1.0 + np.exp(-out[0]))
    mask = (prob > thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if n <= 1:
        return np.zeros((0, 5), np.float32)
    s = OUT_STRIDE
    flat = lab.ravel()
    keep = flat > 0
    idx = flat[keep]
    ys = (np.indices(lab.shape)[0].ravel()[keep].astype(np.float32) + 0.5) * s
    top = ys - out[1].ravel()[keep]
    bot = ys + out[2].ravel()[keep]
    sc = prob.ravel()[keep]
    cnt = np.bincount(idx, minlength=n).astype(np.float32)
    # Mean is a robust enough summary here and is far cheaper than a median.
    y0 = np.bincount(idx, weights=top, minlength=n) / np.maximum(cnt, 1)
    y1 = np.bincount(idx, weights=bot, minlength=n) / np.maximum(cnt, 1)
    score = np.bincount(idx, weights=sc, minlength=n) / np.maximum(cnt, 1)
    boxes = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        lh = max(1.0, y1[i] - y0[i])
        ex = min(CORE_X_SHRINK * lh, 0.5 * lh)
        x0 = stats[i, cv2.CC_STAT_LEFT] * s - ex
        x1 = (stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) * s + ex
        boxes.append(
            (
                x0 - grow_x * lh,
                y0[i] - grow_y * lh,
                x1 + grow_x * lh,
                y1[i] + grow_y * lh,
                score[i],
            )
        )
    return np.asarray(boxes, np.float32).reshape(-1, 5)
