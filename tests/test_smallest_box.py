import numpy as np

from indra_ocr.document import _smallest_box


def _one_at_a_time(ink, boxes, area, k):
    """The per-character loop that _smallest_box replaces."""
    out = []
    for b in ink:
        cx, cy = (b[0] + b[2]) / 2 * k, (b[1] + b[3]) / 2 * k
        inside = (
            (boxes[:, 0] <= cx)
            & (cx <= boxes[:, 2])
            & (boxes[:, 1] <= cy)
            & (cy <= boxes[:, 3])
        )
        out.append(
            int(np.flatnonzero(inside)[np.argmin(area[inside])]) if inside.any() else -1
        )
    return out


def test_smallest_box_matches_the_loop():
    rng = np.random.default_rng(0)
    for dtype in (np.float32, np.float64):
        boxes = rng.uniform(0, 500, (60, 4)).astype(dtype)
        boxes[:, 2:] = boxes[:, :2] + rng.uniform(5, 200, (60, 2)).astype(dtype)
        boxes[7] = boxes[3]  # a tie: the first box wins
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        k = 200 / 72
        pts = rng.uniform(0, 250, (500, 4))
        pts[:, 2:] = pts[:, :2] + rng.uniform(0, 8, (500, 2))
        # centers exactly on box edges
        pts[:10, 0] = pts[:10, 2] = boxes[:10, 0] / k
        for conv in (lambda v: tuple(np.float64(x) for x in v), lambda v: tuple(float(x) for x in v)):
            ink = [conv(v) for v in pts]
            assert _smallest_box(ink, boxes, area, k).tolist() == _one_at_a_time(
                ink, boxes, area, k
            )


def test_smallest_box_empty():
    boxes = np.zeros((0, 4), np.float32)
    assert _smallest_box([(1.0, 2.0, 3.0, 4.0)], boxes, boxes[:, 0], 1.0).tolist() == [-1]
