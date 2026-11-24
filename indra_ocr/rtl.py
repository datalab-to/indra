"""Right-to-left text for a recognizer that reads every line left to right.

A CTC recognizer emits characters in the order it sees them, left to right, so
its reading of a line with Arabic or Hebrew is the characters in visual order.
``flip`` turns logical order (how text is stored and typed) into visual order,
and the same call turns visual order back into logical order:

- RTL base line (more RTL letters than LTR letters): the line reads right to
  left, so the order of its units reverses.  A unit is a run of LTR text
  (Latin words, digits, and the spaces and punctuation between them), which
  keeps its own order, or one other character cluster.
- LTR base line: the line keeps its order, and each run of RTL text reverses.

A cluster is a base character with its combining marks (Arabic harakat, Hebrew
niqqud), so marks stay after their base.  A mirrored character (brackets) at
right-to-left level changes to its mirror image, as a renderer draws it.

This is a small subset of the Unicode bidi algorithm (UAX #9): it has no
embeddings or isolates, and it resolves neutrals by runs.  It is its own
inverse, so text flipped to visual order and back is unchanged.
"""

from __future__ import annotations

import unicodedata

_MIRROR = dict(zip("()[]{}<>«»‹›", ")(][}{><»«›‹"))
# Presentation forms (glyph variants: initial, medial, final, ligatures) map to
# base letters.  Tatweel stretches a word and carries no letter.
_PRESENTATION = ((0xFB1D, 0xFDFF), (0xFE70, 0xFEFF))
_TATWEEL = "ـ"


def is_rtl(c: str) -> bool:
    return unicodedata.bidirectional(c) in ("R", "AL")


def _is_ltr(c: str) -> bool:
    # Digits (European and Arabic-Indic) read left to right inside RTL text.
    return unicodedata.bidirectional(c) in ("L", "EN", "AN")


def has_rtl(text: str) -> bool:
    return any(is_rtl(c) for c in text)


def rtl_base(text: str) -> bool:
    """Whether the line reads right to left: more RTL letters than LTR letters.
    The counts do not depend on the order, so logical and visual text agree."""
    r = sum(is_rtl(c) for c in text)
    lt = sum(unicodedata.bidirectional(c) == "L" for c in text)
    return r > lt


def normalize(text: str) -> str:
    """Presentation forms to base letters (NFKC on those blocks only), no
    tatweel, no combining mark without a base character."""
    out = []
    for c in text:
        o = ord(c)
        if any(a <= o <= b for a, b in _PRESENTATION):
            out.append(unicodedata.normalize("NFKC", c))
        elif c != _TATWEEL and (out or unicodedata.category(c) not in ("Mn", "Me")):
            out.append(c)
    return "".join(out)


def clusters(text: str) -> list[str]:
    out: list[str] = []
    for c in text:
        if out and unicodedata.category(c) in ("Mn", "Me"):
            out[-1] += c
        else:
            out.append(c)
    return out


def _kind(cl: str) -> str:
    c = cl[0]
    return "R" if is_rtl(c) else "L" if _is_ltr(c) else "N"


def _runs(kinds: list[str], strong: str, stop: set[str]) -> list[tuple[int, int]]:
    """[start, end) spans from a ``strong`` cluster to the last ``strong`` cluster
    before a cluster of a kind in ``stop``."""
    spans, start, last = [], None, None
    for i, k in enumerate(kinds + ["#"]):
        if k == strong:
            if start is None:
                start = i
            last = i
        elif k in stop or k == "#":
            if start is not None:
                spans.append((start, last + 1))
            start = last = None
    return spans


def _mirror(cl: str) -> str:
    return _MIRROR.get(cl[0], cl[0]) + cl[1:]


def flip(text: str, mirror: bool = True, base_rtl: bool | None = None) -> str:
    """Logical order to visual order, and visual order back to logical order.
    base_rtl: the paragraph direction, when known (a renderer lays out a line
    by its paragraph's direction); None decides by ``rtl_base``."""
    if not has_rtl(text):
        return text
    cls = clusters(text)
    kinds = [_kind(c) for c in cls]
    if rtl_base(text) if base_rtl is None else base_rtl:
        # Units: LTR runs kept whole, every other cluster alone; reverse the units.
        units: list[str] = []
        spans = _runs(kinds, "L", {"R"})
        i = 0
        for a, b in spans + [(len(cls), len(cls))]:
            for j in range(i, a):
                units.append(_mirror(cls[j]) if mirror else cls[j])
            if a < b:
                units.append("".join(cls[a:b]))
            i = b
        return "".join(reversed(units))
    # LTR base: reverse each RTL run in place.
    out = list(cls)
    for a, b in _runs(kinds, "R", {"L"}):
        seg = cls[a:b][::-1]
        out[a:b] = [_mirror(c) for c in seg] if mirror else seg
    return "".join(out)


def visual_label(logical: str, base_rtl: bool | None = None) -> str:
    """A line stored in logical order, in the visual order the recognizer writes."""
    return flip(normalize(logical), base_rtl=base_rtl)


def label_from_visual(raw_visual: str, shapes: bool = False) -> str:
    """The recognizer's visual-order form of characters already in visual order, as stored (a PDF text
    layer read by glyph position): normalize in logical order, then mirror.
    shapes: the text layer stores each bracket as the shape drawn (see
    ``brackets_as_shapes``), so it is already mirrored."""
    return visual_label(flip(raw_visual, mirror=shapes))


def brackets_as_shapes(raw_visual_lines: list[str]) -> bool:
    """Whether a page's text layer stores RTL brackets as drawn shapes.

    Producers differ: most store the logical character, so the leftmost bracket
    of an RTL pair reads ")"; some store the drawn shape, so it reads "(".  A
    page uses one convention, so its RTL lines without Latin letters vote by
    their first bracket."""
    shapes = logical = 0
    for t in raw_visual_lines:
        if not rtl_base(t) or any(unicodedata.bidirectional(c) == "L" for c in t):
            continue
        for c in t:
            if c in "([{":
                shapes += 1
                break
            if c in ")]}":
                logical += 1
                break
    return shapes > logical
