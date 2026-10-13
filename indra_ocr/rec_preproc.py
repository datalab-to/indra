"""Line crop rule shared by recognition training and inference.

A recognizer input is made from a line box (the ink box from the detector, or
the ink box from a synthetic or real label) in three steps:
  1. expand the box by PAD_Y x its height on top and bottom and PAD_X x its
     height on left and right,
  2. crop that region from the page (white outside the page),
  3. resize the crop to LINE_HEIGHT pixels high, keeping the aspect ratio.

The result is an ink map (0 = paper, 1 = ink) of shape [LINE_HEIGHT, W].
Training and inference must both call ``crop_line``, so the text scale and the
margins match.
"""

from __future__ import annotations

import math

import numpy as np

LINE_HEIGHT = 32
PAD_Y = 0.2
PAD_X = 0.3
WIDTH_MULTIPLE = 8  # widths are padded to a multiple of this (timestep stride)


def crop_line(
    gray: np.ndarray,
    box,
    height: int = LINE_HEIGHT,
    pad_y: float = PAD_Y,
    pad_x: float = PAD_X,
    max_width: int = 4096,
) -> np.ndarray:
    """Crop one line from a uint8 grayscale page and return a float32 ink map [height, W]."""
    import cv2

    x0, y0, x1, y1 = (float(v) for v in box[:4])
    h = max(1.0, y1 - y0)
    X0, X1 = x0 - pad_x * h, x1 + pad_x * h
    Y0, Y1 = y0 - pad_y * h, y1 + pad_y * h
    H, W = gray.shape
    ix0, iy0 = int(math.floor(X0)), int(math.floor(Y0))
    ix1, iy1 = int(math.ceil(X1)), int(math.ceil(Y1))
    cw, ch = max(1, ix1 - ix0), max(1, iy1 - iy0)
    crop = np.full((ch, cw), 255, np.uint8)
    sx0, sy0, sx1, sy1 = max(0, ix0), max(0, iy0), min(W, ix1), min(H, iy1)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - iy0 : sy1 - iy0, sx0 - ix0 : sx1 - ix0] = gray[sy0:sy1, sx0:sx1]
    out_w = max(WIDTH_MULTIPLE, min(max_width, round(cw * height / ch)))
    interp = cv2.INTER_AREA if ch > height else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (out_w, height), interpolation=interp)
    return 1.0 - resized.astype(np.float32) / 255.0


def pad_batch(lines: list[np.ndarray], multiple: int = WIDTH_MULTIPLE):
    """Right-pad ink maps to one width.  Returns (array [B, 1, H, W], widths [B])."""
    H = lines[0].shape[0]
    Wm = max(x.shape[1] for x in lines)
    Wm = int(math.ceil(Wm / multiple) * multiple)
    out = np.zeros((len(lines), 1, H, Wm), np.float32)
    widths = np.zeros(len(lines), np.int64)
    for i, x in enumerate(lines):
        out[i, 0, :, : x.shape[1]] = x
        widths[i] = x.shape[1]
    return out, widths


def width_batches(
    widths: list[int],
    max_batch: int = 64,
    max_ratio: float = 1.3,
    min_width: int = 64,
    max_area: int = 64 * 640,
) -> list[list[int]]:
    """Group line indices into batches of similar width, so little compute goes to padding.

    Lines are taken narrowest first.  A batch closes at ``max_batch`` lines, when the
    next line is wider than ``max_ratio`` times the batch's first line (widths below
    ``min_width`` count as ``min_width``, so short lines do not fragment into tiny batches),
    or when lines x width would pass ``max_area`` pixel columns: ONNX Runtime keeps its
    largest allocation, and 64 lines of 4,096 px held 5 GB on large scanned pages.
    """
    order = sorted(range(len(widths)), key=lambda i: widths[i])
    batches: list[list[int]] = []
    for i in order:
        if (
            batches
            and len(batches[-1]) < max_batch
            and max(widths[i], min_width)
            <= max_ratio * max(widths[batches[-1][0]], min_width)
            and (len(batches[-1]) + 1) * max(widths[i], min_width) <= max_area
        ):
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches
