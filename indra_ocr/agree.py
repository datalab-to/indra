"""Does a PDF text layer say what its page shows?

A layer can hold the wrong codes for its glyphs: a legacy font (Kruti Dev, Chanakya,
Bijoy, VNI) stores Latin codes for another script, a broken ToUnicode map writes
other letters, and a layer can hold syllables in visual order.  The ink checks in
``textlayer`` do not see this: the glyphs are drawn where the layer says.

The check reads a few short pieces of layer lines from the page image with the
recognizers, and asks how likely each recognizer finds the layer's text (a forced
CTC alignment) compared with its own best reading.  The gap is near 0 where the
layer is the page's text, also where the recognizer itself is unsure (both
readings are about as likely), and large where the layer's characters are not
in the image.  The same measure works for every script and recognizer.
"""

from __future__ import annotations

import unicodedata

import numpy as np

from .rtl import flip, has_rtl
from .rtl import normalize as rtl_normalize

UNKNOWN: tuple = ()  # a layer character the recognizer cannot write


def snippet(tokens: list, n: int) -> tuple[list, float | None]:
    """The start (leftmost by x) of a detected line's layer tokens (char, x,
    content index): about n non-space characters, cut at the first word gap
    after them when one comes within 2n characters.  Returns those tokens and
    the cut x (None: the whole line)."""
    chars = sorted((t[1], t[0]) for t in tokens if not t[0].isspace())
    xs = [x for x, _ in chars]
    if len(xs) <= n:
        return list(tokens), None
    spaces = [t[1] for t in tokens if t[0].isspace()]
    k = next(
        (k for k in range(n, min(len(xs), 2 * n)) if any(xs[k - 1] <= s < xs[k] for s in spaces)),
        n,
    )
    # Never cut before a combining mark: a vowel sign or tone mark is drawn on
    # the letter before it, so the image piece shows it.
    while k < len(xs) and unicodedata.category(chars[k][1][:1] or " ").startswith("M"):
        k += 1
    if k >= len(xs):
        return list(tokens), None
    cut = (xs[k - 1] + xs[k]) / 2
    return [t for t in tokens if t[1] < cut], cut


def layer_text(tokens: list, rtl: bool) -> str:
    """The text of layer tokens as a recognizer reads the line: each word (a
    text-layer run's characters between spaces) keeps the layer's order, and
    the words go left to right on the line (right to left for a recognizer of
    right-to-left text, which reads the logical order flipped).  A layer can
    store a line's words in another order than they stand, so the order of the
    layer's words is no evidence against it."""
    words: list[list] = []
    for t in tokens:
        if t[0].isspace():
            if words and words[-1]:
                words.append([])
            continue
        if not words or (words[-1] and words[-1][0][2] != t[2]):
            words.append([])
        words[-1].append(t)
    words = [w for w in words if w]
    words.sort(key=lambda w: -max(t[1] for t in w) if rtl else min(t[1] for t in w))
    return " ".join("".join(t[0] for t in w) for w in words)


def writes(text: str, vocab, share: float = 0.9) -> bool:
    """Whether a recognizer can write at least share of the text's letters
    (Hangul by its jamo when the vocab holds jamo)."""
    letters = [c for c in vocab.normalize(text) if c.isalpha()]
    if not letters:
        return True
    known = sum(
        c in vocab.index
        or (vocab.jamo and all(j in vocab.index for j in unicodedata.normalize("NFD", c)))
        for c in letters
    )
    return known >= share * len(letters)


def base_letter(c: str) -> str | None:
    """The letter a precomposed character is built on (its canonical
    decomposition is that letter and combining marks), else None."""
    d = unicodedata.normalize("NFD", c)
    if len(d) > 1 and d[0].isalpha() and all(unicodedata.category(m).startswith("M") for m in d[1:]):
        return d[0]
    return None


def target_ids(text: str, vocab, visual: bool) -> list[tuple]:
    """The recognizer's class ids for a layer text: per token, the class ids that
    write it (the character in either case: small capitals and text set in
    capitals by the PDF draw another case than the layer stores; and a
    precomposed letter's base letter: a recognizer that misses a mark the
    layer has is no evidence against the layer), UNKNOWN for a character the
    recognizer cannot write.  visual: the recognizer reads right-to-left text
    in visual order."""
    if has_rtl(text):
        text = rtl_normalize(text)
    text = vocab.normalize(text)
    if visual:
        text = flip(text)
    ids: list[tuple] = []
    for c in vocab.tokenize(text):
        forms = [c, c.upper(), c.lower()]
        b = base_letter(c)
        if b:
            forms += [b, b.upper(), b.lower()]
        alts = tuple(dict.fromkeys(vocab.index[x] for x in forms if x in vocab.index))
        if alts:
            ids.append(alts)
        elif vocab.jamo and all(j in vocab.index for j in unicodedata.normalize("NFD", c)):
            ids += [(vocab.index[j],) for j in unicodedata.normalize("NFD", c)]  # Hangul as jamo
        else:
            ids.append(UNKNOWN)
    return ids


def forced_gaps(items: list[tuple]) -> list[float]:
    """Per item (blank, topv, topi, ids): the log-probability per token by which
    the best CTC alignment of ids falls below the recognizer's best path, from
    its per-frame outputs: the blank's log-probability (T,) and the top-k
    classes (T, k).  A class outside the top k counts 1 nat under the k-th; so
    does an UNKNOWN token (a character the recognizer cannot write: a broken
    letter map often writes such characters, e.g. modifier letters or Cyrillic
    in a Telugu or Arabic layer).  0 for no ids, inf when ids cannot fit the
    frames.
    One Viterbi pass over all items."""
    out = [0.0] * len(items)
    run = [i for i, it in enumerate(items) if it[3] and len(it[0])]
    if not run:
        return out
    B = len(run)
    T = max(len(items[i][0]) for i in run)
    L = max(len(items[i][3]) for i in run)
    K = items[run[0]][1].shape[1]
    A = max(len(alt) for i in run for alt in items[i][3]) or 1
    S = 2 * L + 1
    blank = np.full((B, T), -np.inf)
    topv = np.full((B, T, K), -np.inf)
    topi = np.full((B, T, K), -1, np.int64)
    alts = np.full((B, L, A), -2, np.int64)  # -2 matches no class
    ends = np.zeros(B, np.int64)
    lens = np.zeros(B, np.int64)
    skip = np.zeros((B, S), bool)
    for b, i in enumerate(run):
        bl, tv, ti, ids = items[i]
        n = len(bl)
        blank[b, :n], topv[b, :n], topi[b, :n] = bl, tv, ti
        ends[b], lens[b] = n - 1, len(ids)
        for j, alt in enumerate(ids):
            alts[b, j, : len(alt)] = alt
        # A token may follow the one before it directly (no blank between)
        # unless the two are the same class.
        skip[b, 3 : 2 * len(ids) : 2] = [not set(x) & set(y) for x, y in zip(ids[1:], ids[:-1])]
    hit = (topi[:, :, None, :, None] == alts[:, None, :, None, :]).any(-1)  # B, T, L, K
    tok = np.where(hit, topv[:, :, None, :], -np.inf).max(-1)
    tok = np.where(hit.any(-1), tok, topv[:, :, -1:] - 1.0)
    lp = np.full((B, T, S), -np.inf)
    lp[:, :, 0::2] = blank[:, :, None]
    lp[:, :, 1::2] = tok
    a = np.full((B, S), -np.inf)
    a[:, :2] = lp[:, 0, :2]
    best = np.full(B, -np.inf)

    def finish(t):
        for b in np.flatnonzero(ends == t):
            best[b] = max(a[b, 2 * lens[b]], a[b, 2 * lens[b] - 1])

    finish(0)
    for t in range(1, T):
        nxt = a.copy()
        nxt[:, 1:] = np.maximum(nxt[:, 1:], a[:, :-1])
        nxt[:, 2:] = np.where(skip[:, 2:], np.maximum(nxt[:, 2:], a[:, :-2]), nxt[:, 2:])
        a = nxt + lp[:, t]
        finish(t)
    path = np.where(np.isfinite(topv[:, :, 0]), topv[:, :, 0], 0.0).sum(1)
    for b, i in enumerate(run):
        out[i] = float((path[b] - best[b]) / lens[b]) if np.isfinite(best[b]) else float("inf")
    return out


def forced_gap(blank, topv, topi, ids: list[tuple]) -> float:
    """``forced_gaps`` for one line."""
    return forced_gaps([(blank, topv, topi, ids)])[0]
