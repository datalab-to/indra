"""Math lines: a classifier finds lines that hold math, and MathRec reads them as
text with inline LaTeX ("we have $x _ { i } \\leq 1$").

The CTC recognizer reads one symbol per column, so it cannot write a fraction or
a sum with limits.  MathRec reads a taller crop of the line with an attention
decoder, one token per step, with the decoder's keys and values cached.

Models (all optional; the pipeline skips math without them): mathcls.onnx,
math_enc.onnx, math_dec_init.onnx, math_meta.json and a decoder step graph.
The step graph is math_dec_step_kv.onnx (one cache input and output per layer),
or in an older model folder math_dec_step.onnx (stacked caches).  Both give the
same tokens.  tools/math_dec_step_kv.py builds the first from the second.  With
both in the folder, the decoder runs math_dec_step_kv.onnx.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from .line_mask import crop_lines
from .rec_preproc import pad_batch, width_batches
from .ort_session import session

MATH_MIN = 0.5  # classifier probability above which a line is read as math
# A math reading replaces a line only at this mean token probability or more: on
# some print (old scans) it writes confident-looking LaTeX that is wrong, at low
# probability.
MATH_MIN_CONF = 0.9
# A line whose mean token probability is this far below MATH_MIN_CONF after 8
# tokens stops decoding: it would be rejected anyway (garbage on such print).
EARLY_STOP_MARGIN = 0.1
BOS, EOS, N_SPECIAL = 1, 2, 4
FILES = ("mathcls.onnx", "math_enc.onnx", "math_dec_init.onnx")
# The decoder step with per-layer caches (preferred), and the older stacked step.
STEP_KV = "math_dec_step_kv.onnx"
STEP_STACKED = "math_dec_step.onnx"


def step_file(onnx_dir: Path) -> str | None:
    """The decoder step graph in ``onnx_dir`` that MathReader runs, or None."""
    for f in (STEP_KV, STEP_STACKED):
        if (onnx_dir / f).exists():
            return f
    return None


# Commands that style or space text, and escaped symbols (\\%): no math of their own.
_STYLE = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|mathcal|mathbb|boldsymbol|textbf|textit"
    r"|textrm|text|operatorname)\b|\\[,;:! %#&$]|\\q?quad\b|[{}~\s]"
)


# Commands that write one symbol plain text has as well (± × ≤ → ...): a span of
# digits, letters and these is plain text ("66 ± 5").
_SYMBOL = re.compile(
    r"\\(?:pm|mp|times|cdot|div|leq|geq|le|ge|neq|ne|approx|sim|simeq|equiv|"
    r"rightarrow|leftarrow|to|uparrow|downarrow|longrightarrow|circ|degree|ldots|cdots|"
    r"dots|prime|bullet|star|ast|mid|vert|Rightarrow|Leftarrow|Leftrightarrow|leftrightarrow)(?![A-Za-z])"
)


def has_math(text: str) -> bool:
    """A MathRec reading holds math: a $...$ span with a command (\\frac, \\alpha,
    \\leq), a superscript or a subscript once styling and spacing are gone.  A span
    of plain characters (a number, a word, "x = 2") writes the same as plain text,
    and the line keeps the recognizer's reading."""
    for span in text.split("$")[1::2]:
        s = _STYLE.sub("", _SYMBOL.sub("", span))
        if "\\" in s or "^" in s or "_" in s:
            return True
    return False



def has_structure(text: str) -> bool:
    """A reading's $...$ spans hold structure a text layer cannot write: a
    superscript, a subscript, or a command with an argument (\\frac{, \\sqrt{,
    \\hat{).  A text layer has every symbol (alpha, triangle, the element sign) as a
    character, so a span of symbols and plain characters adds nothing to it."""
    for span in text.split("$")[1::2]:
        s = re.sub(r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|mathcal|mathbb|boldsymbol|"
                   r"textbf|textit|textrm|text|operatorname)\s*\{([^{}]*)\}", r"\1", span)
        if "^" in s or "_" in s or re.search(r"\\[A-Za-z]+\s*\{", s):
            return True
    return False

# An accent written as math over one letter: MathRec has no accented letters, so
# "é" may come out as \hat { \epsilon } and "ż" as \bar { z }.
_ACCENT = re.compile(
    r"\s*\\(?:hat|bar|dot|ddot|tilde|acute|grave|breve|check)\s*\{\s*\\?[A-Za-z]{1,8}\s*\}\s*"
)


def accept_math(reading: str, draft: str) -> str | None:
    """The line's text when MathRec's reading replaces the plain reading (the
    recognizer's or the text layer's), else None: the reading must hold math
    (has_math) once accents written as math are left out (when the plain reading
    has accented letters), it must not write LaTeX outside its $...$ spans (a
    malformed reading).  The plain reading's accents go back into the prose
    (restore_accents)."""
    parts = reading.split("$")
    if any(re.search(r"\\[A-Za-z]", p) for p in parts[0::2]):
        return None
    accented = any(c.isalpha() and _fold(c) != c.lower() for c in draft)
    spans = [sp for sp in parts[1::2] if not (accented and _ACCENT.fullmatch(sp))]
    if not has_math("".join(f"${sp}$" for sp in spans)):
        return None
    return restore_accents(reading, draft)


def _fold(c: str) -> str:
    """A character without its accents, lower case (one character out)."""
    d = unicodedata.normalize("NFKD", c)
    base = "".join(x for x in d if not unicodedata.combining(x))
    return (base[:1] or c).lower()


def restore_accents(reading: str, draft: str) -> str:
    """MathRec's reading with the accents of the plain reading (the recognizer's
    or the text layer's) put back in its prose: MathRec writes Latin letters
    without accents ("důvodu" -> "duvodu").  Only prose letters that match a
    letter of the plain reading change; the $...$ spans stay as MathRec wrote them."""
    import difflib

    if not draft.strip():
        return reading
    parts = reading.split("$")
    # (part index, char index) of every prose letter, and its folded form
    pos = [(k, i) for k in range(0, len(parts), 2) for i, c in enumerate(parts[k]) if c.isalpha()]
    b = [c for c in draft if c.isalpha()]
    if not pos or not b:
        return reading
    sm = difflib.SequenceMatcher(
        None, [_fold(parts[k][i]) for k, i in pos], [_fold(c) for c in b], autojunk=False
    )
    chars = [list(p) for p in parts]
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        # letters that match, and short one-for-one swaps (MathRec read "ů" as "i")
        if op == "equal" or (op == "replace" and a1 - a0 == b1 - b0 <= 2):
            for t in range(a1 - a0):
                k, i = pos[a0 + t]
                c = b[b0 + t]
                if _fold(c) != c.lower():  # the plain reading has an accent here
                    chars[k][i] = c
    return "$".join("".join(p) for p in chars)


# Prose splice (splice_prose): letters that look alike across Latin, Cyrillic and
# Greek, and Latin letters without a decomposition, map to one Latin letter, so a
# MathRec reading ("Toni", "pá", "molumdur") aligns with the plain reading ("Toдi",
# "þá", "məlumdur").  Only the alignment uses this map; no text is written with it.
_LOOK = str.maketrans(dict(zip(
    "авсеһіјкмнорѕтухьԁԛԝгєαβεηικνορτυχωϲϳıȷłđðħŧøœæßþƿəɛɔ",
    "abcehijkmhopstyxbdqwreabenikvoptuxwcjijlddhtooabppeeo",
)))
# A splice replaces a prose segment only when this share of its characters (spaces
# left out) align with equal characters of the plain reading, and the line's prose
# as a whole reaches SPLICE_MIN_LINE: else the line keeps today's reading.
SPLICE_MIN_SEG = 0.5
SPLICE_MIN_LINE = 0.6
_GREEK = re.compile(r"\\(?:var)?([a-z]+)$")


def _look(c: str) -> str:
    """One character for the alignment: no accents, lower case, look-alike letters
    of Latin, Cyrillic and Greek as one Latin letter."""
    return _fold(c).translate(_LOOK)


def _units(text: str) -> list[tuple[str, int, int]]:
    """Alignment units of a text, spaces left out: (folded unit, start, end).  A
    LaTeX command is one unit (a Greek letter as its look-alike), and the
    structure of a $...$ span (braces, ^, _, styling commands) writes none, so a
    span compares with the plain characters it stands for.  A plain reading may
    hold commands as well ("Burada \\sigma ana")."""
    out = []
    for m in re.finditer(r"\\[A-Za-z]+|\S", text):
        tok = m.group()
        if tok.startswith("\\"):
            if _STYLE.fullmatch(tok):
                continue
            g = _GREEK.match(tok)
            try:
                tok = _look(unicodedata.lookup("GREEK SMALL LETTER " + g.group(1).upper())) if g else tok
            except KeyError:
                pass
        elif tok in "{}^_~\\":
            continue
        else:
            tok = _look(tok)
        out.append((tok, m.start(), m.end()))
    return out


def _letters(s: str):
    import collections

    return collections.Counter(c for c in s if c.isalpha() and ord(c) > 127)


def splice_prose(reading: str, draft: str, today: str | None = None) -> str:
    """MathRec's reading with its prose (the text outside $...$) taken from the
    plain reading (the recognizer's or the text layer's): MathRec writes LaTeX and
    Latin letters, so prose around a formula loses accents and letters of other
    scripts ("məlumdur" -> "molumdur", "þá" -> "pá", "Тоді" -> "Toni").

    The reading's prose characters and its spans' plain characters (_math_sig) are
    aligned with the plain reading on a folded form (_look).  Each prose segment
    maps to the plain characters between its first and last aligned position, and
    that substring replaces the segment when it holds non-ASCII letters that the
    segment lacks.  A span that only writes an accent over one letter ("é" read as
    \\dot { \\epsilon }) counts as prose when the plain reading has accents.

    today: the text without the splice (default: restore_accents).  It comes back
    unchanged when nothing is lost, when the line is all math, or when the
    alignment is not sure (SPLICE_MIN_SEG, SPLICE_MIN_LINE)."""
    import difflib

    if today is None:
        today = restore_accents(reading, draft)
    if not draft.strip() or not _letters(draft) or "$" not in reading:
        return today
    parts, tparts = reading.split("$"), today.split("$")
    if len(parts) != len(tparts) or len(parts) % 2 == 0:
        return today  # an odd number of $: no clean segments
    accented = any(c.isalpha() and _fold(c) != c.lower() for c in draft)
    # segments [is prose, reading text, today's text]; an accent span joins the prose
    segs: list = []
    for k, (p, t) in enumerate(zip(parts, tparts)):
        if k % 2 == 1 and not (accented and _ACCENT.fullmatch(p)):
            segs.append([False, p, t])
            continue
        if k % 2 == 1:
            p, t = f"${p}$", f"${t}$"
        if segs and segs[-1][0]:
            segs[-1][1] += p
            segs[-1][2] += t
        else:
            segs.append([True, p, t])
    if not any(s[0] and any(c.isalpha() for c in s[1]) for s in segs):
        return today  # all math
    # Alignment units of the reading: (folded unit, segment); a prose unit that
    # aligns with an equal plain unit is a hit.
    units: list = []
    for n, (prose, text, _) in enumerate(segs):
        if not prose:
            units += [(u, n) for u, _, _ in _units(text)]
            continue
        for piece in re.split(r"(\$[^$]*\$)", text):
            if len(piece) > 1 and piece[0] == piece[-1] == "$":
                units.append(("\0", n))  # an accent span: one letter of the plain reading
            else:
                units += [(_look(c), n) for c in piece if not c.isspace()]
    du = _units(draft)
    sm = difflib.SequenceMatcher(None, [u[0] for u in units], [u[0] for u in du], autojunk=False)
    # per unit: the plain units [start, end) it stands for
    start, end, hit = [0] * len(units), [0] * len(units), [False] * len(units)
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            for i in range(a0, a1):
                start[i], end[i], hit[i] = b0 + i - a0, b0 + i - a0 + 1, True
            continue
        kinds = [segs[units[i][1]][0] for i in range(a0, a1)]
        cut = next((i for i in range(1, len(kinds)) if kinds[i] != kinds[0]), None)
        if cut is not None and all(k == kinds[cut] for k in kinds[cut:]):
            # one prose run and one formula run: a misread letter is one letter
            # (o for ə), so the prose run takes one plain unit per unit at its edge
            # and the formula the rest
            n_p = cut if kinds[0] else len(kinds) - cut
            take = min(n_p, b1 - b0)
            mid = b0 + take if kinds[0] else b1 - take
        else:
            mid = None
        for i in range(a0, a1):
            r = i - a0
            if mid is None:  # spread the plain units over the units
                start[i] = b0 + r * (b1 - b0) // (a1 - a0)
                end[i] = b0 + (r + 1) * (b1 - b0) // (a1 - a0)
            elif kinds[0]:  # prose first
                start[i], end[i] = (min(b0 + r, mid), min(b0 + r + 1, mid)) if r < cut else (mid, b1)
            else:  # formula first
                q = r - cut
                start[i], end[i] = (b0, mid) if r < cut else (min(mid + q, b1), min(mid + q + 1, b1))
    prose_ids = [i for i, u in enumerate(units) if segs[u[1]][0]]
    if not prose_ids or sum(hit[i] for i in prose_ids) < SPLICE_MIN_LINE * len(prose_ids):
        return today
    out = []
    for n, (prose, text, t) in enumerate(segs):
        if not prose:
            out.append(f"${t}$")
            continue
        ids = [i for i in prose_ids if units[i][1] == n]
        lo, hi = (start[ids[0]], end[ids[-1]]) if ids else (0, 0)
        sub = draft[du[lo][1] : du[hi - 1][2]] if hi > lo else ""
        if not (_letters(sub) - _letters(t)):
            out.append(t)  # nothing lost here: today's text, byte for byte
            continue
        if sum(hit[i] for i in ids) < SPLICE_MIN_SEG * len(ids) or not 0.5 <= (hi - lo) / len(ids) <= 2.0:
            return today
        lead = text[: len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()) :]
        out.append(lead + sub + trail)
    return "".join(out)


def latin_line(text: str) -> bool:
    """Whether MathRec may read a line with this text (the recognizer's or the text
    layer's reading).  MathRec writes Latin letters only: a line whose letters are
    mostly of another script (Greek or Cyrillic prose, any other script) stays with
    its own reading.  A letter of another script on its own (a Greek variable) is a
    math symbol, not text: only runs of 2 or more such letters count.  A line
    without such text (numbers, symbols) may go."""
    if text.isascii():  # no marks, and every letter is Latin
        return True
    latin = other = run = 0
    for c in text + " ":
        if unicodedata.category(c).startswith("M"):
            run += run > 0  # a combining mark (Indic vowel sign) continues a word
            continue
        if c.isalpha() and not unicodedata.name(c, "").startswith("LATIN"):
            run += 1
            continue
        if run >= 2:
            other += run
        run = 0
        if c.isalpha():
            latin += 1
    return latin > other or other == 0


def detokenize(tokens: list[str]) -> str:
    """Tokens -> text; inside math (between "$" tokens) they are joined by spaces."""
    out, math, in_math = [], [], False
    for t in tokens:
        if t == "$":
            if in_math:
                out.append("$" + " ".join(math) + "$")
                math = []
            in_math = not in_math
        elif in_math:
            math.append(t)
        else:
            out.append(t)
    if in_math and math:
        out.append("$" + " ".join(math) + "$")
    return " ".join("".join(out).split())


# Rescore (math_rescore): a line whose greedy reading has a mean token probability
# under RESCORE_BELOW is read again by a beam search, and the beam that agrees best
# with the line's draft wins (score = mean log-probability + RESCORE_W x the
# character agreement of reading and draft).  The draft holds the line's characters
# (text layer or recognizer) but not their structure; the reading holds structure.
RESCORE_BELOW = 0.96
RESCORE_BEAMS = 5
RESCORE_W = 2.0
_SIG_KEEP = frozenset("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=<>()/")
_COMMAND = re.compile(r"\\[A-Za-z]+")


def _sig_reading(text: str) -> str:
    """Letters, digits and a few operators of a LaTeX reading (commands dropped)."""
    return "".join(c for c in _COMMAND.sub(" ", text) if c in _SIG_KEEP)


def _sig_draft(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text) if c in _SIG_KEEP)


_GATE_OPS = frozenset("=+-*/<>^_∑∫√≤≥±×÷∂∞≈≠∈∀∃→←|")


def gate_features(prob: float, draft: str, n_flagged: int, pmax: float, n_cand: int, source: str) -> list[float]:
    """Features of a flagged line for the read gate (MathReader.gate): its
    math-classifier probability, what its draft is made of, and page counts."""
    d = draft or ""
    n = max(1, len(d))
    return [
        prob,
        float(np.log1p(len(d))),
        sum(c.isdigit() for c in d) / n,
        sum(c.isalpha() for c in d) / n,
        sum(c in _GATE_OPS for c in d) / n,
        sum(c in "$€£¥%" for c in d) / n,
        d.count(" ") / n,
        float(len(d.split())),
        sum(0x36F < ord(c) < 0x400 for c in d) / n,
        float("=" in d),
        sum(c.isupper() for c in d) / n,
        float(np.log1p(n_flagged)),
        pmax,
        float(np.log1p(n_cand)),
        float(source == "ocr"),
        float(source == "text"),
    ]


def _own(x: np.ndarray) -> np.ndarray:
    """x, or a copy of it if it cannot be written in place."""
    return x if x.flags.writeable else x.copy()


class MathReader:
    rescore = False  # math_rescore: beam search + draft agreement on unsure lines
    # prose_splice: an accepted whole-line reading keeps the plain reading's prose
    # around its $...$ spans (splice_prose).
    prose_splice = True
    # move_rows: finished lines leave the decoder's batch by moving live rows into
    # their places (_front), not by taking the live rows (_take); same tokens.
    move_rows = True

    def __init__(self, onnx_dir: Path, so, providers):
        # The stacked step graph copies the constant cross-attention caches (and
        # re-stacks the self-attention caches) on every token; the per-layer graph
        # (STEP_KV) does not.  Load only the one the decoder runs.
        step = step_file(onnx_dir)
        self.split_kv = step == STEP_KV
        s = {
            f: session(str(onnx_dir / f), so, providers=providers)
            for f in (*FILES, step)
        }
        self.cls, self.enc = s["mathcls.onnx"], s["math_enc.onnx"]
        self.init, self.step = s["math_dec_init.onnx"], s[step]
        meta = json.loads((onnx_dir / "math_meta.json").read_text(encoding="utf-8"))
        cfg = meta["config"]
        self.height = cfg["height"]
        self.max_tokens = cfg["max_tokens"]
        self.layers, self.heads = cfg["dec_layers"], cfg["dec_heads"]
        self.head_dim = cfg["channels"] // cfg["dec_heads"]
        self.itos = ["", "", "", ""] + meta["tokens"]
        # Single-character tokens: the letters MathRec can write (writes()).
        self.chars = frozenset(t for t in meta["tokens"] if len(t) == 1)
        # Exports drop unused inputs (the init graph does not read the mask).
        self.inputs = {f: {i.name for i in v.get_inputs()} for f, v in s.items()}
        self.step_in = self.inputs[step]
        acc = onnx_dir / "math_accept.json"
        self.accept_head = json.loads(acc.read_text(encoding="utf-8")) if acc.exists() else None
        gate = onnx_dir / "math_gate.json"
        self.gate_head = json.loads(gate.read_text(encoding="utf-8")) if gate.exists() else None
        self.last_probs: list[float] = []
        # Lines the head let MathRec's reading replace on this page: reading -> (the
        # plain line, reading, conf, draft, recognizer conf); assemble() asks the head
        # again for lines the layout puts in a table.
        self.accepted: dict = {}

    def accept(self, text: str, conf: float, draft: str, rconf: float = 1.0, in_table: float = 0.0) -> str | None:
        """The line's text when MathRec's reading replaces the plain reading, else None.
        With math_accept.json in the model folder, a learned logistic head decides
        (features of the reading, its confidence, the recognizer's confidence and the
        agreement with the draft); else the reading needs conf >= MATH_MIN_CONF and
        accept_math.  A malformed reading (LaTeX outside its $...$ spans) never
        replaces the line.  With prose_splice, the accepted text keeps the plain
        reading's prose (splice_prose)."""
        h = self.accept_head
        if h is None:
            out = accept_math(text, draft) if conf >= MATH_MIN_CONF else None
            return self._spliced(text, draft, out)
        import difflib

        parts = text.split("$")
        if not text or any(re.search(r"\\[A-Za-z]|[\^_{}]", p) for p in parts[0::2]):
            return None
        kept = accept_math(text, draft)
        inside = sum(len(p) for p in parts[1::2]) / max(1, len(text.replace("$", "")))
        agree = difflib.SequenceMatcher(None, _sig_draft(draft), _sig_reading(text), autojunk=False).ratio()
        ok = float(kept is not None)
        x = [conf, rconf, ok, inside, agree, float(np.log1p(len(text.split()))), ok * conf]
        if len(h["mu"]) > len(x):  # the head also knows whether the layout put the line in a table
            x.append(in_table)
        z = h["w"][0] + sum(w * (v - m) / s for w, v, m, s in zip(h["w"][1:], x, h["mu"], h["sd"]))
        if 1.0 / (1.0 + np.exp(-z)) < h["tau"]:
            return None
        return self._spliced(text, draft, kept if kept is not None else (text if "$" in text else None))

    def _spliced(self, text: str, draft: str, out: str | None) -> str | None:
        if out is None or not getattr(self, "prose_splice", True):
            return out
        return splice_prose(text, draft, out)

    def writes(self, text: str, share: float = 0.9) -> bool:
        """Whether MathRec's vocabulary writes this line's letters (the recognizer's
        or the text layer's reading): at least ``share`` of them are its tokens.
        A MathRec whose vocabulary holds another script's letters may then read
        lines of that script; one with Latin letters only writes no such line, and
        latin_line alone decides."""
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and sum(c in self.chars for c in letters) >= share * len(letters)

    @classmethod
    def load(cls, onnx_dir: Path, so, providers) -> "MathReader | None":
        names = (*FILES, "math_meta.json")
        if all((onnx_dir / f).exists() for f in names) and step_file(onnx_dir):
            return cls(onnx_dir, so, providers)
        return None

    def flags(self, crops: list, batch: int = 64) -> list[bool]:
        """Per line crop (the recognizer's crop): does it hold math?  The
        probabilities stay in ``last_probs`` (for the read gate)."""
        out = [False] * len(crops)
        probs = [0.0] * len(crops)
        for idx in width_batches([c.shape[1] for c in crops], batch):
            arr, widths = pad_batch([crops[j] for j in idx])
            p = self.cls.run(None, {"x": arr, "widths": widths})[0]
            for k, j in enumerate(idx):
                probs[j] = float(np.asarray(p[k]).reshape(-1)[0])
                out[j] = bool(p[k] > MATH_MIN)
        self.last_probs = probs
        return out

    def gate(self, probs: list[float], drafts: list[str], n_cand: int, source: str) -> list[bool]:
        """Which flagged lines MathRec reads.  With math_gate.json in the model
        folder, a learned logistic gate skips lines whose reading would not be
        used (features: gate_features); without it every flagged line is read."""
        h = self.gate_head
        if h is None or not probs:
            return [True] * len(probs)
        pmax = max(probs)
        keep = []
        for p, d in zip(probs, drafts):
            x = gate_features(p, d, len(probs), pmax, n_cand, source)
            z = h["b"] + sum(w * v for w, v in zip(h["w"], x))
            keep.append(z >= h["z_min"])
        return keep

    def _caches(self, ck: np.ndarray, cv: np.ndarray, B: int):
        """(self_k, self_v, cross_k, cross_v) at the first step, from the init
        graph's cross keys and values ([layers, B, heads, T, d]).  Per-layer
        lists for the per-layer step graph, else the stacked arrays."""
        empty = np.zeros((self.layers, B, self.heads, 0, self.head_dim), np.float32)
        if self.split_kv:
            return list(empty), list(empty), list(ck), list(cv)
        return empty, empty, ck, cv

    def _take(self, c, rows):
        """The cache c for decoder rows ``rows`` (a mask or row indexes)."""
        return [x[rows] for x in c] if self.split_kv else c[:, rows]

    @staticmethod
    def _front(live: np.ndarray, arrays: list) -> int:
        """Moves the rows still decoding (``live``, a mask over the first rows) to
        the front of each array in place -> how many there are.  A finished row
        in the front takes a live row from the back, so each finished line costs
        one row's copy, where taking the live rows copies all of them."""
        n = int(live.sum())
        dst = np.flatnonzero(~live[:n])
        src = n + np.flatnonzero(live[n:])
        for x in arrays:
            x[dst] = x[src]
        return n

    def _step(self, tok, t: int, sk, sv, ck, cv, mask):
        """One decoder step -> (logits [B, V], self_k, self_v) with this token's
        keys and values appended."""
        feed = {"token": tok, "pos": np.array(t, np.int64), "mask": mask}
        if self.split_kv:
            for i in range(self.layers):
                feed[f"self_k{i}"], feed[f"self_v{i}"] = sk[i], sv[i]
                feed[f"cross_k{i}"], feed[f"cross_v{i}"] = ck[i], cv[i]
            out = self.step.run(None, {k: v for k, v in feed.items() if k in self.step_in})
            L = self.layers
            return out[0], out[1 : 1 + L], out[1 + L : 1 + 2 * L]
        feed.update(self_k=sk, self_v=sv, cross_k=ck, cross_v=cv)
        logits, sk, sv = self.step.run(None, {k: v for k, v in feed.items() if k in self.step_in})
        return logits, sk, sv

    def read(
        self,
        gray: np.ndarray,
        boxes: np.ndarray,
        batch: int = 64,
        drafts: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """(text with inline LaTeX, mean token probability) for each box.
        drafts: each line's text as the recognizer or the text layer read it."""
        crops = crop_lines(gray, boxes[:, :4], self.height, None)
        out = [("", 0.0)] * len(crops)
        # Few, wide batches: each batch is one decoding loop of up to max_tokens
        # steps, and a step costs about the same for 8 lines as for 40.
        widths = [c.shape[1] for c in crops]
        for idx in width_batches(widths, batch, max_ratio=4.0, max_area=16 * 64 * 640):
            arr, widths = pad_batch([crops[j] for j in idx])
            mem, mask = self.enc.run(None, {"x": arr, "widths": widths})
            feed = {"memory": mem, "mask": mask}
            names = self.inputs["math_dec_init.onnx"]
            ck, cv = self.init.run(None, {k: v for k, v in feed.items() if k in names})
            B = len(idx)
            sk, sv, ck, cv = self._caches(ck, cv, B)
            tok = np.full(B, BOS, np.int64)
            seqs: list[list[int]] = [[] for _ in range(B)]
            probs: list[list[float]] = [[] for _ in range(B)]
            rows = np.arange(B)  # the batch's line in each decoder row
            # With per-layer caches, finished lines leave the batch by moving live
            # rows into their places (_front): the step reads the first n rows of
            # the cross caches and the mask.  A line's outputs do not depend on the
            # other rows or their order, and each step runs the same lines as
            # taking the live rows would, so the batch sizes are the same too (a
            # step of one line is not bitwise one row of a larger step).
            front = self.split_kv and self.move_rows
            if front:
                ck, cv = [_own(x) for x in ck], [_own(x) for x in cv]
                mask = _own(mask)
            n = B
            in_math = [False] * B  # inside a $...$ span
            text_like = [False] * B  # the reading already fails accept_math
            # probs as one array: every line still decoding has a probability for
            # each step so far, so one mean per step gives each line's (a row's
            # mean is np.mean of its list, to the bit)
            P = np.zeros((B, self.max_tokens), np.float64)
            for t in range(self.max_tokens - 1):
                if front:
                    rk, rv = [x[:n] for x in ck], [x[:n] for x in cv]
                    logits, sk, sv = self._step(tok, t, sk, sv, rk, rv, mask[:n])
                else:
                    logits, sk, sv = self._step(tok, t, sk, sv, ck, cv, mask)
                tok = logits.argmax(-1).astype(np.int64)
                e = np.exp(logits - logits.max(-1, keepdims=True))
                top = 1.0 / e.sum(-1)  # softmax probability of the argmax token
                P[rows, t] = top
                low = (
                    P[rows, : t + 1].mean(1) < MATH_MIN_CONF - EARLY_STOP_MARGIN
                    if t + 1 >= 8
                    else np.zeros(len(rows), bool)
                )
                live = np.ones(len(rows), bool)
                for r, b in enumerate(rows):
                    probs[b].append(float(top[r]))
                    if tok[r] == EOS:
                        live[r] = False
                    elif low[r]:
                        live[r] = False  # rejected: keeps a low mean probability
                    elif tok[r] >= N_SPECIAL:
                        seqs[b].append(int(tok[r]))
                        if self._text_like(b, self.itos[int(tok[r])], in_math):
                            text_like[b] = True
                            live[r] = False  # accept_math would reject it: stop here
                if not live.any():
                    break
                if not live.all() and front:
                    sk, sv = [_own(x) for x in sk], [_own(x) for x in sv]
                    n = self._front(live, [rows, tok, *sk, *sv, *ck, *cv, mask])
                    rows, tok = rows[:n], tok[:n]
                    sk, sv = [x[:n] for x in sk], [x[:n] for x in sv]
                elif not live.all():
                    # Finished lines leave the batch: a step costs about its
                    # number of lines, and the longest line sets the step count.
                    rows, tok, mask = rows[live], tok[live], mask[live]
                    sk, sv = self._take(sk, live), self._take(sv, live)
                    ck, cv = self._take(ck, live), self._take(cv, live)
            for k, j in enumerate(idx):
                if text_like[k]:
                    continue  # stays ("", 0.0): not math
                text = detokenize([self.itos[i] for i in seqs[k]])
                out[j] = (text, float(np.mean(probs[k])) if probs[k] else 0.0)
        if self.rescore and drafts:
            unsure = [j for j, (text, conf) in enumerate(out) if text and conf < RESCORE_BELOW and drafts[j]]
            if unsure:
                for j, r in zip(unsure, self._rescore([crops[j] for j in unsure], [drafts[j] for j in unsure], batch)):
                    out[j] = r
        return out

    def _rescore(self, crops: list, drafts: list[str], batch: int = 64) -> list[tuple[str, float]]:
        """Beam search on the lines (RESCORE_BEAMS beams each, all lines of a width
        batch in one decoding loop); per line the beam with the best mean
        log-probability + RESCORE_W x agreement with its draft -> (text, mean token
        probability)."""
        import difflib

        K = RESCORE_BEAMS
        res = [("", 0.0)] * len(crops)
        names = self.inputs["math_dec_init.onnx"]
        widths = [c.shape[1] for c in crops]
        for idx in width_batches(widths, max(1, batch // K), max_ratio=4.0, max_area=16 * 64 * 640):
            arr, w = pad_batch([crops[j] for j in idx])
            mem, mask = self.enc.run(None, {"x": arr, "widths": w})
            ck, cv = self.init.run(None, {k: v for k, v in {"memory": mem, "mask": mask}.items() if k in names})
            n = len(idx)
            beams = [[([], [], 0.0)] for _ in range(n)]  # per line: (tokens, probabilities, log-prob sum)
            done = [[] for _ in range(n)]
            live = list(range(n))  # lines still decoding
            rows = np.arange(n)  # line of each decoder row
            sk, sv, ck, cv = self._caches(ck, cv, n)
            rck, rcv, rmask, key = ck, cv, mask, rows.tobytes()
            for t in range(self.max_tokens - 1):
                if rows.tobytes() != key:
                    rck, rcv = self._take(ck, rows), self._take(cv, rows)
                    rmask, key = mask[rows], rows.tobytes()
                tok = np.array([b[0][-1] if b[0] else BOS for line in live for b in beams[line]], np.int64)
                logits, sk, sv = self._step(tok, t, sk, sv, rck, rcv, rmask)
                lp = logits - logits.max(-1, keepdims=True)
                lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
                top = np.argsort(-lp, -1)[:, :K]
                parents, nrows, nlive, r0 = [], [], [], 0
                for line in live:
                    cand = []
                    for b, (ids, pr, sc) in enumerate(beams[line]):
                        r = r0 + b
                        for i in top[r]:
                            cand.append((sc + float(lp[r, i]), r, b, int(i)))
                    r0 += len(beams[line])
                    cand.sort(key=lambda z: -z[0])
                    nxt = []
                    for sc, r, b, i in cand:
                        ids, pr, _ = beams[line][b]
                        p = float(np.exp(lp[r, i]))
                        if i == EOS:
                            done[line].append((ids, pr, sc))
                        elif i >= N_SPECIAL:
                            nxt.append((ids + [i], pr + [p], sc)); parents.append(r)
                        if len(nxt) == K:
                            break
                    if len(done[line]) >= K or not nxt:
                        del parents[len(parents) - len(nxt):]
                        if not done[line]:
                            done[line] = nxt or beams[line]
                        continue
                    beams[line] = nxt; nlive.append(line); nrows += [line] * len(nxt)
                live = nlive
                if not live:
                    break
                rows = np.array(nrows)
                sk, sv = self._take(sk, parents), self._take(sv, parents)
            for line in live:
                done[line] += beams[line]
            for line, j in enumerate(idx):
                ds = _sig_draft(drafts[j])

                def score(b):
                    text = detokenize([self.itos[i] for i in b[0]])
                    agree = difflib.SequenceMatcher(None, ds, _sig_reading(text), autojunk=False).ratio()
                    return b[2] / max(1, len(b[0])) + RESCORE_W * agree

                best = max(done[line], key=score)
                res[j] = (detokenize([self.itos[i] for i in best[0]]), float(np.mean(best[1])) if best[1] else 0.0)
        return res

    @staticmethod
    def _text_like(b: int, t: str, in_math: list) -> bool:
        """Track one emitted token; True when the reading can no longer pass
        accept_math: a LaTeX command outside the $...$ spans."""
        if t == "$":
            in_math[b] = not in_math[b]
            return False
        return not in_math[b] and len(t) > 1 and t[0] == "\\" and t[1].isalpha()
