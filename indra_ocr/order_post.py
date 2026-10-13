"""Torch-free parts of the reading-order stage: line features, decoding,
baselines, metrics, and the ground-truth order of a labelled page.

order_model.py (the PyTorch OrderNet) re-exports all of these.
"""

from __future__ import annotations

import math

import numpy as np

N_GEOM = 10
N_TEXT = 6
N_FEATS = N_GEOM + N_TEXT


def line_features(
    boxes: np.ndarray, page_wh: tuple[float, float], texts: list[str] | None = None
) -> np.ndarray:
    """[N, 4] boxes (x0, y0, x1, y1) -> [N, N_FEATS] float32 features."""
    W, H = page_wh
    b = boxes.astype(np.float32)
    n = len(b)
    if n == 0:
        return np.zeros((0, N_FEATS), np.float32)
    w = np.maximum(b[:, 2] - b[:, 0], 1e-3)
    h = np.maximum(b[:, 3] - b[:, 1], 1e-3)
    med_h = float(np.median(h))
    geom = np.stack(
        [
            b[:, 0] / W,
            b[:, 1] / H,
            b[:, 2] / W,
            b[:, 3] / H,
            (b[:, 0] + b[:, 2]) / (2 * W),
            (b[:, 1] + b[:, 3]) / (2 * H),
            w / W,
            np.log(h / med_h),
            np.log(w / h) / 4.0,
            np.full(n, med_h / H * 20.0),
        ],
        1,
    )
    txt = np.zeros((n, N_TEXT), np.float32)
    if texts is not None:
        for i, t in enumerate(texts):
            t = t.strip()
            if not t:
                continue
            txt[i, 0] = math.log1p(len(t)) / 5.0
            txt[i, 1] = sum(c.isdigit() for c in t) / len(t)
            txt[i, 2] = float(
                t[0] in "•-–▪◦*·→" or (t[:3].rstrip(".)").isdigit() and len(t) > 3)
            )
            txt[i, 3] = float(t.endswith("-"))
            txt[i, 4] = float(t[-1] in ".!?:;")
            txt[i, 5] = float(t[0].isupper())
    return np.concatenate([geom, txt], 1).astype(np.float32)


def decode_order(a: np.ndarray, sweeps: int = 4) -> list[int]:
    """Order from a pairwise logit matrix [N, N] (a[i, j] > 0: i before j).

    Start from the row-sum ranking, then local search with insertion moves:
    take each line out and put it back at the position that most increases
    the summed log-probability of the pairwise relations.  Unlike adjacent
    swaps, an insertion can move a line far (e.g. a short last line of a
    column that the ranking put after the next column).  Vectorized: each
    move costs a few numpy calls, O(N) work."""
    n = a.shape[0]
    if n <= 1:
        return list(range(n))
    lp = -np.logaddexp(0.0, -a)  # log P(i before j)
    order = np.argsort(-lp.sum(1) + lp.sum(0))
    for _ in range(sweeps):
        improved = False
        for x in order.copy():
            rest = order[order != x]
            before = lp[rest, x]  # x placed after these
            after = lp[x, rest]  # x placed before these
            gains = np.concatenate(([0.0], np.cumsum(before))) + np.concatenate(
                (np.cumsum(after[::-1])[::-1], [0.0])
            )
            k = int(np.argmax(gains))
            cur = int(np.nonzero(order == x)[0][0])
            if gains[k] > gains[cur] + 1e-6:
                order = np.insert(rest, k, x)
                improved = True
        if not improved:
            break
    return [int(v) for v in order]


# ---------------------------------------------------------------------------
# Baselines and metrics
# ---------------------------------------------------------------------------


def order_top_down(boxes: np.ndarray) -> list[int]:
    """Rows top to bottom (centres within half a line height share a row), left to right."""
    if len(boxes) == 0:
        return []
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    h = np.median(boxes[:, 3] - boxes[:, 1])
    idx = sorted(
        range(len(boxes)),
        key=lambda i: (round(cy[i] / max(h * 0.5, 1e-3)), boxes[i, 0]),
    )
    return idx


def order_xycut(boxes: np.ndarray) -> list[int]:
    """Recursive XY-cut: split at the widest whitespace gap on either axis.

    A horizontal cut (between rows) puts the upper part first; a vertical cut
    (between columns) puts the left part first.  Without a gap, fall back to
    top-down order."""

    def best_cut(b, axis):
        lo, hi = (b[:, 1], b[:, 3]) if axis == 1 else (b[:, 0], b[:, 2])
        o = np.argsort(lo)
        end = hi[o[0]]
        best_gap, best_k = 0.0, -1
        for k in range(1, len(o)):
            gap = lo[o[k]] - end
            if gap > best_gap:
                best_gap, best_k = gap, k
            end = max(end, hi[o[k]])
        return best_gap, best_k, o

    def rec(ids):
        if len(ids) <= 1:
            return list(ids)
        b = boxes[ids]
        gy, ky, oy = best_cut(b, 1)
        gx, kx, ox = best_cut(b, 0)
        if ky < 0 and kx < 0:
            return [ids[i] for i in order_top_down(b)]
        k, o = (ky, oy) if gy >= gx else (kx, ox)
        return rec([ids[i] for i in o[:k]]) + rec([ids[i] for i in o[k:]])

    return rec(list(range(len(boxes))))


def kendall_tau(pred: list[int], n: int) -> float:
    """Kendall tau between a predicted order and the identity order 0..n-1."""
    if n < 2:
        return 1.0
    pos = np.empty(n, int)
    pos[np.asarray(pred)] = np.arange(n)
    conc = 0
    for i in range(n):
        conc += int((pos[i] < pos[i + 1 :]).sum())
    total = n * (n - 1) // 2
    return (2 * conc - total) / total


def successor_acc(pred: list[int], n: int) -> float:
    """Fraction of true consecutive pairs (i, i+1) that are also consecutive in the prediction."""
    if n < 2:
        return 1.0
    nxt = {pred[k]: pred[k + 1] for k in range(len(pred) - 1)}
    return sum(nxt.get(i) == i + 1 for i in range(n - 1)) / (n - 1)


# ---------------------------------------------------------------------------
# Ground-truth order from page labels
# ---------------------------------------------------------------------------

GRID_LABELS = {"Table", "Form", "Table-Of-Contents"}

# Block classes the layout heads predict for each line (chandra block labels).
BLOCK_CLASSES = [
    "Text",
    "Section-Header",
    "Page-Header",
    "Page-Footer",
    "List-Group",
    "Table",
    "Caption",
    "Footnote",
    "Table-Of-Contents",
    "Form",
]
_CLASS_ID = {c: i for i, c in enumerate(BLOCK_CLASSES)}


def block_class(label: str) -> int:
    """Class id of a chandra block label (unknown labels count as Text)."""
    return _CLASS_ID.get(label, 0)


def label_order(lab: dict) -> list[int] | None:
    """True reading order of a labelled page (indices into lab["lines"]).

    Blocks come in chandra's order.  Inside a table / form / table-of-contents
    block, lines go row by row.  Inside any other block, lines go by XY-cut,
    so a list or text block that spans two columns reads column by column
    (row order interleaved the columns).  Returns None when a line has no block."""
    lines = lab["lines"]
    if any(ln.get("block") is None for ln in lines):
        return None
    blocks = lab["blocks"]
    by_block: dict[int, list[int]] = {}
    for i, ln in enumerate(lines):
        by_block.setdefault(ln["block"], []).append(i)
    out: list[int] = []
    for b in sorted(by_block):
        idx = by_block[b]
        boxes = np.array(
            [(lines[i]["ink"] or lines[i]["bbox"]) for i in idx], np.float32
        ).reshape(-1, 4)
        label = blocks[b][0] if b < len(blocks) else "Text"
        local = order_top_down(boxes) if label in GRID_LABELS else order_xycut(boxes)
        out += [idx[k] for k in local]
    return out


def _cluster(logits: np.ndarray) -> list[list[int]]:
    """Average-linkage clustering of pairwise "same group" logits: merge the two
    groups with the highest mean logit between them while that mean is positive
    (probability above one half).  Unlike connected components, one wrong pair
    cannot join two groups that the other pairs keep apart.

    A merge changes only the merged group's row and column of the mean matrix,
    and each row keeps its best partner, so a page of n lines costs about n^2
    instead of n^3 (an 800-line page: 1.5 s -> tens of ms)."""
    n = logits.shape[0]
    if n == 0:
        return []
    S = logits.astype(np.float64).copy()
    np.fill_diagonal(S, 0.0)
    size = np.ones(n)
    alive = np.ones(n, bool)
    members = [[i] for i in range(n)]
    M = S.copy()  # mean logit between groups; -inf on the diagonal and dead groups
    np.fill_diagonal(M, -np.inf)
    best = M.argmax(1)
    best_v = M[np.arange(n), best]
    for _ in range(n - 1):
        a = int(np.argmax(best_v))
        b = int(best[a])
        if best_v[a] <= 0:
            break
        a, b = min(a, b), max(a, b)
        S[a] += S[b]
        S[:, a] += S[:, b]
        S[a, a] = 0.0
        size[a] += size[b]
        alive[b] = False
        members[a] += members[b]
        row = S[a] / (size[a] * size)
        row[~alive] = -np.inf
        row[a] = -np.inf
        M[a], M[:, a] = row, row
        M[b], M[:, b] = -np.inf, -np.inf
        best_v[b] = -np.inf
        best[a] = int(row.argmax())
        best_v[a] = row[best[a]]
        # other rows: (j, a) changed and (j, b) is gone
        stale = alive & ((best == a) | (best == b))
        stale[a] = False
        for j in np.flatnonzero(stale):
            best[j] = int(M[j].argmax())
            best_v[j] = M[j, best[j]]
        up = alive & (row > best_v)
        up[a] = False
        best[up], best_v[up] = a, row[up]
    return [sorted(members[i]) for i in range(n) if alive[i]]


def decode_blocks_order(
    a: np.ndarray,
    same: np.ndarray,
    boxes: np.ndarray,
    cls: np.ndarray | None = None,
) -> list[int]:
    """Reading order with the layout heads: blocks first, then lines inside blocks.

    Blocks come from average-linkage clustering of the same-block logits
    (independent of the order).  Inside a block, lines go by XY-cut, or
    row by row in a table / form / table-of-contents block (``cls`` ids of
    BLOCK_CLASSES).  Blocks go by the mean pairwise order logit between them."""
    n = a.shape[0]
    if n <= 1:
        return list(range(n))
    blocks = _cluster(same)
    grid = {_CLASS_ID[c] for c in GRID_LABELS if c in _CLASS_ID}
    inner = []
    for b in blocks:
        bx = boxes[b, :4]
        if cls is not None and int(cls[b].sum(0).argmax()) in grid:
            local = order_top_down(bx)
        else:
            local = order_xycut(bx)
        inner.append([b[k] for k in local])
    # Mean order logit between blocks p < q, antisymmetric: G a G^T with G the
    # block-membership matrix scaled by 1 / block size (one matrix product
    # instead of a loop over block pairs).
    m = len(blocks)
    G = np.zeros((m, n), np.float64)
    for k, b in enumerate(blocks):
        G[k, b] = 1.0 / len(b)
    up = np.triu(G @ a.astype(np.float64) @ G.T, 1)
    ab = (up - up.T).astype(np.float32)
    return [i for k in decode_order(ab) for i in inner[k]]
