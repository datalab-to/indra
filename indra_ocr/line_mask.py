"""Mask a line window with CenterMask before the recognizer crops the line.

One function for training and inference: the recognizer must only ever see
crops that went through the same exported mask model (mask.onnx).

    sess = load_mask("mask.onnx")
    win, box_in_win = masked_window(gray, box, sess)
    img = crop_line(win, box_in_win, height)
"""

from __future__ import annotations

import numpy as np

WINDOW_H = 96  # window height in pixels; the centre line is about a third of it
WINDOW_ABOVE = 1.0  # box heights above and below the line box
WINDOW_SIDE = 0.5  # box heights left and right


def window_box(box, w_page: float, h_page: float) -> tuple[float, float, float, float]:
    """The window around a line box (page pixels), clipped to the page."""
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    h = max(1.0, y1 - y0)
    return (
        max(0.0, x0 - WINDOW_SIDE * h),
        max(0.0, y0 - WINDOW_ABOVE * h),
        min(w_page, x1 + WINDOW_SIDE * h),
        min(h_page, y1 + WINDOW_ABOVE * h),
    )


KEEP = 0.5  # pixels with a lower centre-line probability become paper


def resize_window(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Scale a window to ``size`` (w, h): linear down to half scale, area below
    (anti-aliasing).  INTER_AREA at a non-integer scale costs 20x more."""
    import cv2

    shrink = size[1] / max(1, a.shape[0])
    interp = cv2.INTER_LINEAR if shrink >= 0.5 else cv2.INTER_AREA
    return cv2.resize(a, size, interpolation=interp)


def load_mask(path: str, threads: int = 1):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _window(gray: np.ndarray, box) -> tuple[np.ndarray, tuple[float, float]]:
    H, W = gray.shape
    x0, y0, x1, y1 = (int(round(v)) for v in window_box(box, W, H))
    return gray[y0:y1, x0:x1], (x0, y0)


def mask_windows(sess, wins: list[np.ndarray]) -> list[np.ndarray]:
    """Centre-line keep masks (bool, window resolution) for grayscale windows.

    The probabilities are scaled back bilinearly before the threshold: a
    threshold at mask resolution with a nearest-neighbour upscale breaks the
    stroke edges (surya OCR 2 CER on M804 lines 0.30 -> 0.36)."""
    import cv2

    if not wins:
        return []
    xs, sizes = [], []
    for w in wins:
        h, wd = w.shape
        W = max(32, int(round(wd * WINDOW_H / max(h, 1) / 8)) * 8)
        ink = 1.0 - resize_window(w, (W, WINDOW_H)).astype(np.float32) / 255.0
        xs.append(ink)
        sizes.append((wd, h))
    out: list = [None] * len(wins)
    order = np.argsort([x.shape[1] for x in xs])
    for k in range(0, len(order), 16):  # similar widths together: less padding
        idx = order[k : k + 16]
        Wm = max(xs[i].shape[1] for i in idx)
        batch = np.zeros((len(idx), 1, WINDOW_H, Wm), np.float32)
        for j, i in enumerate(idx):
            batch[j, 0, :, : xs[i].shape[1]] = xs[i]
        p = sess.run(None, {"x": batch})[0]
        for j, i in enumerate(idx):
            pm = p[j, 0, :, : xs[i].shape[1]]
            out[i] = cv2.resize(pm, sizes[i], interpolation=cv2.INTER_LINEAR) >= KEEP
    return out


def masked_windows(
    gray: np.ndarray, boxes, sess
) -> list[tuple[np.ndarray, list[float]]]:
    """For each line box: (window with foreign ink whitened, the box inside it)."""
    wins, offs = zip(*[_window(gray, b) for b in boxes]) if len(boxes) else ((), ())
    out = []
    for w, (ox, oy), keep, b in zip(wins, offs, mask_windows(sess, list(wins)), boxes):
        m = w.copy()
        # paper level: 90th percentile grey on a 1/16 subsample of the window
        m[~keep] = np.uint8(np.percentile(w[::4, ::4], 90)) if w.size else 255
        out.append((m, [b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy]))
    return out


def masked_window(gray: np.ndarray, box, sess) -> tuple[np.ndarray, list[float]]:
    return masked_windows(gray, [box], sess)[0]


def crop_lines(gray: np.ndarray, boxes, height: int, sess=None) -> list[np.ndarray]:
    """Recognizer inputs for the line boxes of one page: ``crop_line`` of each
    line, after CenterMask when ``sess`` is given (one batched call per page)."""
    from .rec_preproc import crop_line

    if sess is None:
        return [crop_line(gray, b, height) for b in boxes]
    return [crop_line(w, b, height) for w, b in masked_windows(gray, list(boxes), sess)]


_SESSIONS: dict = {}


def session_for(path: str | None):
    """One CPU session per process (dataloader workers call this lazily)."""
    if not path:
        return None
    if path not in _SESSIONS:
        _SESSIONS[path] = load_mask(path, threads=1)
    return _SESSIONS[path]
