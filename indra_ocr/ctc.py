"""The recognizer's CTC outputs from its logits, in numpy.

A recognizer exported with a "logits" output ([B, T, V]) ends at the greedy
ids, the lengths and the logits.  ONNX Runtime runs every node of a graph
whatever the caller fetches, so the softmax, the top-k and the blank column
over all V classes cost time on every line (42% of rec_cjk's time, V = 11,750).
The package computes them here, only for the frames and lines that use them:
the confidence for the kept frames of each line, the LM inputs for the lines
under the LM gate.  The values are the older graphs' outputs (the same ops in
float32), so a line reads the same.
"""

from __future__ import annotations

import numpy as np

TOPK = 8  # classes per step for the beam search (the older graphs' TopK)


def peak_prob(logits: np.ndarray) -> np.ndarray:
    """[N, V] logits -> [N] the probability of the best class at each step."""
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return (np.float32(1.0) / e.sum(-1)).astype(np.float32)


def lm_inputs(logits: np.ndarray, k: int = TOPK, floor: float = 0.0):
    """[T, V] logits -> (blank [T], topv [T, k], topi [T, k]): the blank
    log-probability and the k best classes' log-probabilities, best first.
    Equal values keep the lower class first, as ONNX Runtime's TopK does.

    With ``floor`` (a probability), only classes at or over it are found; the
    other places hold class 0 at -inf.  beam_search with cand_floor >= floor
    reads the same candidates either way (it skips class 0 and classes under
    its floor), and this skips a sort over all V classes."""
    m = logits.max(-1, keepdims=True)
    d = logits - m
    logs = np.log(np.exp(d).sum(-1))
    blank = d[:, 0] - logs
    k = min(k, logits.shape[1])
    if floor > 0:
        # Classes near or over the floor (slack for float rounding), then the
        # exact test on their log-probabilities.
        lf = np.float64(np.log(floor))  # beam_search compares in float64
        rows, cols = np.nonzero(d >= (logs + np.float32(lf - 1e-3))[:, None])
        vals = d[rows, cols] - logs[rows]
        ok = vals >= lf
        rows, cols, vals = rows[ok], cols[ok], vals[ok]
    else:
        lsm = d - logs[:, None]
        # The k-th best value per step; every class at or over it is a
        # candidate (k or more per step).
        kth = -np.partition(-lsm, k - 1, axis=-1)[:, k - 1]
        rows, cols = np.nonzero(lsm >= kth[:, None])
        vals = lsm[rows, cols]
    # Sorted by step, value (best first) and class; the first k of each step.
    o = np.lexsort((cols, -vals, rows))
    rows, cols, vals = rows[o], cols[o], vals[o]
    first = np.searchsorted(rows, np.arange(len(logits)))
    pos = np.arange(len(rows)) - first[rows]
    take = pos < k
    topi = np.zeros((len(logits), k), np.int64)
    topv = np.full((len(logits), k), -np.inf, np.float32)
    topi[rows[take], pos[take]] = cols[take]
    topv[rows[take], pos[take]] = vals[take]
    return blank, topv, topi
