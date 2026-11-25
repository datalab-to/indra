"""Page text from the ordered lines and the OrderNet layout heads (torch-free).

OrderNet with ``layout`` gives every line a block class and every pair of lines
a "same block" logit.  Walking the reading order, a new block starts wherever
the same-block logit between consecutive lines is negative.  Each block is then
rendered by its class:

  Section-Header        "## " + joined text
  Text, Caption, ...    one paragraph; a word split by a hyphen at a line end is joined
  List-Group            one line per item (a new item starts at a bullet or number)
  Table                 an HTML table assembled from the cell lines (rows by vertical
                        centre, columns anchored on the fullest rows)
  Page-Header / Footer  left out unless ``keep_furniture``
"""

from __future__ import annotations

import html
import re

import numpy as np

from .order_post import BLOCK_CLASSES, _cluster

HEADER, FOOTER = BLOCK_CLASSES.index("Page-Header"), BLOCK_CLASSES.index("Page-Footer")
SECTION = BLOCK_CLASSES.index("Section-Header")
LIST = BLOCK_CLASSES.index("List-Group")
TABLE = BLOCK_CLASSES.index("Table")

_WHOLE_MATH = re.compile(r"^\$[^$]+\$$")  # a line that is one formula
# A list marker, followed by text or alone on its line (the detector often cuts
# a bullet off as a line of its own; the next line then joins its item).
_ITEM = re.compile(r"^\s*([•\-–▪◦*·→]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$)")


def _cjk(c: str) -> bool:
    """Han, kana, CJK punctuation and fullwidth forms: written without spaces
    (Korean Hangul is not: Korean separates words with spaces)."""
    o = ord(c)
    return (
        0x3000 <= o <= 0x30FF
        or 0x3400 <= o <= 0x9FFF
        or 0xF900 <= o <= 0xFAFF
        or 0xFF00 <= o <= 0xFFEF
        or 0x20000 <= o <= 0x2FA1F
    )


def join_lines(texts: list[str]) -> str:
    """Join the lines of one paragraph.  'tumori-' + 'genesis' -> 'tumorigenesis';
    a line break between two CJK characters takes no space."""
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if (
            len(out) > 1
            and out.endswith("-")
            and out[-2].isalpha()
            and t[0].isalpha()
            and t[0].islower()
        ):
            out = out[:-1] + t
        elif out and _cjk(out[-1]) and _cjk(t[0]):
            out += t
        else:
            out = f"{out} {t}" if out else t
    return out


def split_blocks(order: list[int], same: np.ndarray) -> list[list[int]]:
    """Runs of the reading order with a positive same-block logit between neighbours."""
    blocks: list[list[int]] = []
    for i in order:
        if blocks and same[blocks[-1][-1], i] > 0:
            blocks[-1].append(i)
        else:
            blocks.append([i])
    return blocks


def _rows(boxes: np.ndarray) -> list[list[int]]:
    """Group cell lines into rows by vertical centre, top to bottom, left to right."""
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    rows: list[list[int]] = []
    for i in np.argsort(cy):
        if rows and abs(cy[i] - np.mean(cy[rows[-1]])) < 0.5 * h:
            rows[-1].append(int(i))
        else:
            rows.append([int(i)])
    return [sorted(r, key=lambda k: boxes[k, 0]) for r in rows]


def _same_columns(a: list[int], b: list[int], boxes: np.ndarray) -> bool:
    """Every line of ``a`` lies over a line of ``b``: one wrapped cell's first
    lines above the rest of that cell."""
    for i in a:
        x0, x1 = boxes[i, 0], boxes[i, 2]
        if not any(
            min(x1, boxes[j, 2]) - max(x0, boxes[j, 0]) > 0.5 * (x1 - x0) for j in b
        ):
            return False
    return True


def _merge_wrapped(
    rows: list[list[int]],
    boxes: np.ndarray,
    skip: set[int] | frozenset[int] = frozenset(),
) -> list[list[int]]:
    """Join the line rows of wrapped cells into one table row.

    Inside a wrapped cell the gap to the next line is the font's line gap; a new
    table row adds cell padding.  A row joins the row above when its gap is under
    half the table's median row gap and under 0.6 line heights, and either it has
    fewer lines than the row above (a continuation fills only the wrapped
    columns) or the row above has fewer lines, each over a line of this row (a
    wrapped first column beside a row of data).  ``skip``: rows that hold a cell
    spanning several rows (a merged header cell) never join their neighbours."""
    if len(rows) < 3:
        return rows
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    gaps = [
        float(boxes[nxt, 1].min() - boxes[cur, 3].max())
        for cur, nxt in zip(rows[:-1], rows[1:])
    ]
    # Only real gaps count: a cell spanning two rows overlaps the rows above and
    # below it, and those negative gaps would drag the median down.
    positive = [g for g in gaps if g > 0]
    med = float(np.median(positive)) if positive else 0.0
    out = [list(rows[0])]
    for i, g in enumerate(gaps, 1):
        r, prev = rows[i], rows[i - 1]
        wrapped = len(r) < len(prev) or (
            len(prev) < len(r) and _same_columns(prev, r, boxes)
        )
        if i not in skip and i - 1 not in skip and g < 0.5 * med and g < 0.6 * h and wrapped:
            out[-1] += r
        else:
            out.append(list(r))
    return out


def _row_spans(boxes: np.ndarray, rows: list[list[int]]) -> dict[int, tuple[int, int]]:
    """{cell: (first row, last row)} for cells that cover two or more rows.

    ``_rows`` puts a cell tall enough to centre between two rows (a merged header
    cell) in a row of its own.  Its box still covers the bands of the rows above
    and below by more than half, so it is rendered once, with a rowspan, in the
    first of them."""
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    bands = [(float(boxes[r, 1].min()), float(boxes[r, 3].max())) for r in rows]
    out = {}
    for r, cells in enumerate(rows):
        for i in cells:
            y0, y1 = float(boxes[i, 1]), float(boxes[i, 3])
            if y1 - y0 <= 1.5 * h:
                continue
            cover = [
                k
                for k in range(len(rows))
                if k != r
                and min(y1, bands[k][1]) - max(y0, bands[k][0])
                > 0.5 * (bands[k][1] - bands[k][0])
            ]
            if len(cover) >= 2:
                out[i] = (cover[0], cover[-1])
    return out


def table_html(boxes: np.ndarray, texts: list[str]) -> str:
    """An HTML table from cell lines (one text line per cell, as born-digital PDFs give).

    A first row that is one line across most of the table is a title: it goes
    before the table as text.  The first remaining row is the header (<th>).  A
    cell that covers several rows or columns keeps them: rowspan and colspan."""
    rows = _rows(boxes)
    span_x = boxes[:, 2].max() - boxes[:, 0].min()
    if (
        len(rows) > 2
        and len(rows[0]) == 1
        and max(len(r) for r in rows) > 1
        and boxes[rows[0][0], 2] - boxes[rows[0][0], 0] > 0.6 * span_x
    ):
        keep = [k for k in range(len(texts)) if k != rows[0][0]]
        return (
            texts[rows[0][0]]
            + "\n\n"
            + table_html(boxes[keep], [texts[k] for k in keep])
        )
    width = max(len(r) for r in rows)
    full = [r for r in rows if len(r) == width]
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    tall = {i for r in rows for i in r if boxes[i, 3] - boxes[i, 1] > 1.5 * h}
    rows = _merge_wrapped(
        rows, boxes, {r for r, cells in enumerate(rows) if any(i in tall for i in cells)}
    )
    # A cell covering several rows (a merged header cell) is rendered once, in
    # the first row it covers, with a rowspan.
    spans = _row_spans(boxes, rows)
    extra: dict[int, list[int]] = {r: [] for r in range(len(rows))}
    for r, cells in enumerate(rows):
        for i in cells:
            if i in spans and spans[i][0] != r:
                extra[spans[i][0]].append(i)
    drop = {
        r
        for r, cells in enumerate(rows)
        if cells and all(i in spans and spans[i][0] != r for i in cells)
    }
    old2new: dict[int, int] = {}
    kept: list[list[int]] = []
    for r, cells in enumerate(rows):
        if r in drop:
            continue
        old2new[r] = len(kept)
        kept.append(
            [i for i in cells if not (i in spans and spans[i][0] != r)] + extra[r]
        )
    rowspan: dict[int, int] = {}
    for i, (a, b) in spans.items():
        if a not in old2new:
            continue
        covered_rows = [old2new[k] for k in range(a, b + 1) if k in old2new]
        if len(covered_rows) >= 2:
            rowspan[i] = covered_rows[-1] - old2new[a] + 1
    rows = kept
    # Column bands from the rows with the most cells: those rows rarely span.
    bands = np.array(
        [
            [
                np.median([boxes[r[c], 0] for r in full]),
                np.median([boxes[r[c], 2] for r in full]),
            ]
            for c in range(width)
        ]
    )
    # Widen each band to the midpoints between neighbours, so every x falls in one band.
    edges = (
        [-np.inf]
        + [(bands[c, 1] + bands[c + 1, 0]) / 2 for c in range(width - 1)]
        + [np.inf]
    )
    placed: dict[int, dict[int, tuple[int, int, str]]] = {}
    for r, cells in enumerate(rows):
        slots: dict[int, tuple[int, int, str]] = {}
        for k in sorted(cells, key=lambda k: (boxes[k, 1], boxes[k, 0])):
            x0, x1 = boxes[k, 0], boxes[k, 2]
            first = int(np.searchsorted(edges, x0, side="right") - 1)
            last = int(np.searchsorted(edges, x1, side="right") - 1)
            # A cell spans every band it covers by more than half of the band.
            span = [
                c
                for c in range(max(first, 0), min(last, width - 1) + 1)
                if min(x1, bands[c, 1]) - max(x0, bands[c, 0])
                > 0.5 * (bands[c, 1] - bands[c, 0])
            ] or [min(max(first, 0), width - 1)]
            c0 = span[0]
            prev = slots.get(c0)
            rs = rowspan.get(k, 1)
            if prev is None:
                slots[c0] = (rs, len(span), texts[k])
            else:
                slots[c0] = (
                    max(prev[0], rs),
                    max(prev[1], len(span)),
                    join_lines([prev[2], texts[k]]),
                )
        placed[r] = slots
    n = len(rows)
    covered = np.zeros((n, width), bool)
    for r in range(n):
        for c0, (rs, cs, _) in placed[r].items():
            covered[r : r + rs, c0 : c0 + cs] = True
    out = ["<table>"]
    for r in range(n):
        tag = "th" if r == 0 and n > 1 else "td"
        tds, c = [], 0
        while c < width:
            item = placed[r].get(c)
            if item is not None:
                rs, cs, text = item
                attr = (f' rowspan="{rs}"' if rs > 1 else "") + (
                    f' colspan="{cs}"' if cs > 1 else ""
                )
                tds.append(f"<{tag}{attr}>{html.escape(text, quote=False)}</{tag}>")
                c += cs
            else:
                if not covered[r, c]:  # covered by a rowspan above: no cell here
                    tds.append(f"<{tag}></{tag}>")
                c += 1
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def heads_plausible(boxes: np.ndarray, tab: np.ndarray) -> bool:
    """The lines of one cell are stacked (they overlap horizontally).  When the
    cell head joins side-by-side lines in more than 10% of its cells, it has
    failed on this table and the geometric assembly is used instead."""
    cells = _cluster(tab[2])
    bad = 0
    for c in cells:
        if len(c) < 2:
            continue
        x0, x1 = boxes[c, 0], boxes[c, 2]
        if (np.minimum.outer(x1, x1) - np.maximum.outer(x0, x0)).min() < 0:
            bad += 1
    return bad <= 0.1 * len(cells)


def table_html_heads(boxes: np.ndarray, texts: list[str], tab: np.ndarray) -> str:
    """An HTML table from the OrderNet table heads.  tab [3, N, N] = logits
    same row, same column, same cell, for the N lines of one table block.  A cell
    that covers several rows or columns keeps them: rowspan and colspan."""
    cells = _cluster(tab[2])
    cb = np.array(
        [
            [boxes[c, 0].min(), boxes[c, 1].min(), boxes[c, 2].max(), boxes[c, 3].max()]
            for c in cells
        ]
    )
    ctext = [
        join_lines([texts[k] for k in sorted(c, key=lambda k: (boxes[k, 1], boxes[k, 0]))])
        for c in cells
    ]
    m = len(cells)

    def cell_logits(k: int) -> np.ndarray:
        out = np.zeros((m, m), np.float32)
        for a in range(m):
            for b in range(a + 1, m):
                out[a, b] = out[b, a] = tab[k][np.ix_(cells[a], cells[b])].mean()
        return out

    rows = sorted(_cluster(cell_logits(0)), key=lambda g: cb[g, 1].min())
    cols = sorted(_cluster(cell_logits(1)), key=lambda g: cb[g, 0].min())
    row_of = {c: r for r, g in enumerate(rows) for c in g}
    col_of = {c: k for k, g in enumerate(cols) for c in g}
    centers = [float(np.median((cb[g, 0] + cb[g, 2]) / 2)) for g in cols]
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))

    def band_span(lo: float, hi: float, bands: list) -> list[int]:
        return [
            k
            for k, (a, b) in enumerate(bands)
            if min(hi, b) - max(lo, a) > 0.5 * (b - a)
        ]

    # A cell taller than one line and covering two or more row bands (a merged
    # header cell) starts at the first row it covers, with a rowspan.
    rbands = [(float(cb[g, 1].min()), float(cb[g, 3].max())) for g in rows]
    spans: dict[int, tuple[int, int]] = {}
    for c in range(m):
        if cb[c, 3] - cb[c, 1] > 1.5 * h:
            cover = band_span(cb[c, 1], cb[c, 3], rbands)
            if len(cover) >= 2:
                spans[c] = (cover[0], cover[-1])
    first_row = {c: spans.get(c, (row_of[c], row_of[c]))[0] for c in range(m)}
    drop = {
        r
        for r, g in enumerate(rows)
        if g and all(first_row[c] != r for c in g)
    }
    old2new: dict[int, int] = {}
    kept = []
    for r in range(len(rows)):
        if r in drop:
            continue
        old2new[r] = len(kept)
        kept.append(r)
    rowspan: dict[int, int] = {}
    for c, (a, b) in spans.items():
        covered_rows = [old2new[k] for k in range(a, b + 1) if k in old2new]
        if len(covered_rows) >= 2:
            rowspan[c] = covered_rows[-1] - old2new[a] + 1
    n = len(kept)
    grid: dict[tuple[int, int], tuple[int, int, str]] = {}
    for c in range(m):
        c0 = col_of[c]
        span = 1
        while (
            c0 + span < len(cols) and cb[c, 0] - h <= centers[c0 + span] <= cb[c, 2] + h
        ):
            span += 1
        key = (old2new[first_row[c]], c0)
        rs = rowspan.get(c, 1)
        if key in grid:  # two cells on one slot: keep both texts
            grid[key] = (max(grid[key][0], rs), max(grid[key][1], span), join_lines([grid[key][2], ctext[c]]))
        else:
            grid[key] = (rs, span, ctext[c])
    covered = np.zeros((n, len(cols)), bool)
    for (r, c0), (rs, cs, _) in grid.items():
        covered[r : r + rs, c0 : c0 + cs] = True
    out = ["<table>"]
    for r in range(n):
        tag = "th" if r == 0 and n > 1 else "td"
        tds, c = [], 0
        while c < len(cols):
            if (r, c) in grid:
                rs, span, text = grid[(r, c)]
                attr = (f' rowspan="{rs}"' if rs > 1 else "") + (
                    f' colspan="{span}"' if span > 1 else ""
                )
                tds.append(f"<{tag}{attr}>{html.escape(text, quote=False)}</{tag}>")
                c += span
            else:
                if not covered[r, c]:
                    tds.append(f"<{tag}></{tag}>")
                c += 1
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def page_blocks(
    boxes: np.ndarray,
    texts: list[str],
    order: list[int],
    cls_logits: np.ndarray,
    same: np.ndarray,
) -> list[dict]:
    """The layout blocks behind ``page_markdown``, in reading order:
    {"label" (a BLOCK_CLASSES name), "bbox" (pixels, the union of its lines),
    "text" (its lines, newline-joined)}."""
    out = []
    for blk in split_blocks(order, same):
        b = boxes[blk, :4]
        out.append(
            {
                "label": BLOCK_CLASSES[int(cls_logits[blk].sum(0).argmax())],
                "bbox": [
                    round(float(b[:, 0].min()), 1),
                    round(float(b[:, 1].min()), 1),
                    round(float(b[:, 2].max()), 1),
                    round(float(b[:, 3].max()), 1),
                ],
                "text": "\n".join(texts[i] for i in blk),
            }
        )
    return out


def page_markdown(
    boxes: np.ndarray,
    texts: list[str],
    order: list[int],
    cls_logits: np.ndarray,
    same: np.ndarray,
    keep_furniture: bool = False,
    cells: dict | None = None,
    tab: np.ndarray | None = None,
    pictures: np.ndarray | None = None,
    tables: dict | None = None,
) -> str:
    """Markdown for one page.  boxes [N, 4], texts [N], order = reading order,
    cls_logits [N, C], same [N, N] (logit "same block").  ``cells`` maps a line
    to its table cells [(box, text)] when the line holds several.  ``tab``
    [3, N, N] (same row / column / cell logits) builds tables from the heads.
    ``pictures`` [P, 4]: picture regions, each written as ``![](picture-<k>)``
    before the first block below it that it overlaps horizontally.  ``tables``
    maps a table block (its line indices as a tuple) to HTML read by tsr_rel.onnx."""
    parts: list[str] = []
    part_boxes: list = []
    for blk in split_blocks(order, same):
        n_parts = len(parts)
        if tables and tuple(blk) in tables:
            parts.append(tables[tuple(blk)])
        else:
            _block_part(parts, blk, boxes, texts, cls_logits, keep_furniture, cells, tab)
        if len(parts) > n_parts:
            b = boxes[blk, :4]
            part_boxes.append(
                (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())
            )
    if pictures is not None and len(pictures):
        slots: dict[int, list[str]] = {}
        for k, (x0, y0, x1, y1) in enumerate(np.asarray(pictures)[:, :4]):
            cy = (y0 + y1) / 2
            at = next(
                (
                    j
                    for j, (bx0, by0, bx1, by1) in enumerate(part_boxes)
                    if (by0 + by1) / 2 > cy and min(x1, bx1) > max(x0, bx0)
                ),
                len(parts),
            )
            slots.setdefault(at, []).append(f"![](picture-{k})")
        merged = []
        for j in range(len(parts) + 1):
            merged.extend(slots.get(j, []))
            if j < len(parts):
                merged.append(parts[j])
        parts = merged
    return "\n\n".join(p for p in parts if p)


def _block_part(parts, blk, boxes, texts, cls_logits, keep_furniture, cells, tab):
    """Append one block's markdown to ``parts`` (nothing for dropped furniture)."""
    cls = int(cls_logits[blk].sum(0).argmax())
    texts_b = [texts[i] for i in blk]
    if cls in (HEADER, FOOTER) and not keep_furniture:
        return
    if all(_WHOLE_MATH.match(t.strip()) for t in texts_b):  # display equations
        parts.append("\n".join("$$" + t.strip()[1:-1] + "$$" for t in texts_b))
        return
    if cls == SECTION:
        parts.append("## " + join_lines(texts_b))
    elif (
        cls == TABLE
        and len(blk) > 1
        and tab is not None
        and heads_plausible(boxes[blk, :4], tab[:, blk][:, :, blk])
    ):
        parts.append(table_html_heads(boxes[blk, :4], texts_b, tab[:, blk][:, :, blk]))
    elif cls == TABLE and len(blk) > 1:
        cb, ct = [], []
        for i in blk:
            for bx, tx in (cells or {}).get(i) or [(boxes[i, :4], texts[i])]:
                cb.append(bx)
                ct.append(tx)
        parts.append(table_html(np.array(cb, np.float32).reshape(-1, 4), ct))
    elif cls == LIST:
        items: list[list[str]] = []
        for t in texts_b:
            if not items or _ITEM.match(t):
                items.append([t])
            else:
                items[-1].append(t)
        parts.append("\n".join(join_lines(it) for it in items))
    else:
        parts.append(join_lines(texts_b))
