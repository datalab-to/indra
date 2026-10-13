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


_PLAIN_WORD = re.compile(r" [A-Za-z]{3,}\b(?![_^{(])")


def balance_math(text: str) -> str:
    """Close $ math spans the recognizer left open.

    Left to right: a $ opens math only when LaTeX (\\ _ ^ {) or a closing $
    follows within 15 characters, so prices ("$5 per item") stay text.  Inside
    math, a following plain word of 3+ letters closes the span; an open span
    closes at the end of the line."""
    if "$" not in text:
        return text
    out, in_math, i = [], False, 0
    while i < len(text):
        c = text[i]
        if c == "$":
            if in_math:
                in_math = False
            else:
                ahead = text[i + 1 : i + 16]
                in_math = bool(re.search(r"[\\_^{]", ahead)) or "$" in ahead
            out.append(c)
            i += 1
            continue
        if in_math and c == " " and _PLAIN_WORD.match(text, i):
            out.append("$")
            in_math = False
        out.append(c)
        i += 1
    if in_math:
        out.append("$")
    return "".join(out)


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


def _merge_wrapped(rows: list[list[int]], boxes: np.ndarray) -> list[list[int]]:
    """Join the line rows of wrapped cells into one table row.

    Inside a wrapped cell the gap to the next line is the font's line gap; a new
    table row adds cell padding.  A row joins the row above when its gap is under
    half the table's median row gap and under 0.6 line heights, and it has fewer
    lines than the row above (a continuation fills only the wrapped columns)."""
    if len(rows) < 3:
        return rows
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    gaps = [
        float(boxes[nxt, 1].min() - boxes[cur, 3].max())
        for cur, nxt in zip(rows[:-1], rows[1:])
    ]
    med = float(np.median(gaps))
    out = [list(rows[0])]
    prev = rows[0]
    for g, r in zip(gaps, rows[1:]):
        # A continuation fills only the wrapped columns: fewer lines than the row above.
        if g < 0.5 * med and g < 0.6 * h and len(r) < len(prev):
            out[-1] += r
        else:
            out.append(list(r))
        prev = r
    return out


def table_html(boxes: np.ndarray, texts: list[str]) -> str:
    """An HTML table from cell lines (one text line per cell, as born-digital PDFs give).

    A first row that is one line across most of the table is a title: it goes
    before the table as text.  The first remaining row is the header (<th>)."""
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
    rows = _merge_wrapped(rows, boxes)
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
    out = ["<table>"]
    for ri, r in enumerate(rows):
        tag = "th" if ri == 0 and len(rows) > 1 else "td"
        cells: dict[int, tuple[int, str]] = {}
        for k in sorted(r, key=lambda k: (boxes[k, 1], boxes[k, 0])):
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
            prev = cells.get(c0)
            text = texts[k] if prev is None else prev[1] + " " + texts[k]
            cells[c0] = (len(span) if prev is None else max(prev[0], len(span)), text)
        tds, c = [], 0
        while c < width:
            if c in cells:
                n, text = cells[c]
                attr = f' colspan="{n}"' if n > 1 else ""
                tds.append(f"<{tag}{attr}>{html.escape(text, quote=False)}</{tag}>")
                c += n
            else:
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
    same row, same column, same cell, for the N lines of one table block."""
    cells = _cluster(tab[2])
    cb = np.array(
        [
            [boxes[c, 0].min(), boxes[c, 1].min(), boxes[c, 2].max(), boxes[c, 3].max()]
            for c in cells
        ]
    )
    ctext = [
        " ".join(texts[k] for k in sorted(c, key=lambda k: (boxes[k, 1], boxes[k, 0])))
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
    grid: dict[tuple[int, int], tuple[int, str]] = {}
    for c in range(m):
        c0 = col_of[c]
        span = 1
        while (
            c0 + span < len(cols) and cb[c, 0] - h <= centers[c0 + span] <= cb[c, 2] + h
        ):
            span += 1
        key = (row_of[c], c0)
        if key in grid:  # two cells on one slot: keep both texts
            grid[key] = (max(grid[key][0], span), grid[key][1] + " " + ctext[c])
        else:
            grid[key] = (span, ctext[c])
    out = ["<table>"]
    for r in range(len(rows)):
        tag = "th" if r == 0 and len(rows) > 1 else "td"
        tds, c = [], 0
        while c < len(cols):
            if (r, c) in grid:
                span, text = grid[(r, c)]
                attr = f' colspan="{span}"' if span > 1 else ""
                tds.append(f"<{tag}{attr}>{html.escape(text, quote=False)}</{tag}>")
                c += span
            else:
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
) -> str:
    """Markdown for one page.  boxes [N, 4], texts [N], order = reading order,
    cls_logits [N, C], same [N, N] (logit "same block").  ``cells`` maps a line
    to its table cells [(box, text)] when the line holds several.  ``tab``
    [3, N, N] (same row / column / cell logits) builds tables from the heads."""
    parts = []
    for blk in split_blocks(order, same):
        cls = int(cls_logits[blk].sum(0).argmax())
        texts_b = [
            texts[i] for i in blk
        ]  # balance_math measured worse on olmOCR arxiv_math
        if cls in (HEADER, FOOTER) and not keep_furniture:
            continue
        if all(_WHOLE_MATH.match(t.strip()) for t in texts_b):  # display equations
            parts.append("\n".join("$$" + t.strip()[1:-1] + "$$" for t in texts_b))
            continue
        if cls == SECTION:
            parts.append("## " + join_lines(texts_b))
        elif (
            cls == TABLE
            and len(blk) > 1
            and tab is not None
            and heads_plausible(boxes[blk, :4], tab[:, blk][:, :, blk])
        ):
            parts.append(
                table_html_heads(boxes[blk, :4], texts_b, tab[:, blk][:, :, blk])
            )
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
    return "\n\n".join(p for p in parts if p)
