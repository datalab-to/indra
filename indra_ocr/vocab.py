"""Token vocabulary for the line recognizer.

Class 0 is the CTC blank.  Class i + 1 is ``tokens[i]``.  A token is one
character, or one LaTeX command (``\\frac``, ``\\alpha``) inside math.  Labels
are NFC normalized before encoding.  A training line with a token outside the
vocabulary is dropped (``encode`` returns None), never silently altered.

``Vocab.latin()`` is the first recognizer's character set.  ``Vocab.multi()``
adds Cyrillic, full Greek, Devanagari, Bengali and LaTeX command tokens; math
is written ``$...$`` in labels.
"""

from __future__ import annotations

import json
import re
import unicodedata

# Label canonicalization.  Each rule maps characters or runs that the image
# cannot tell apart to one form, so neither training targets nor the eval
# punish a reading that is visually correct.
_CANON = str.maketrans(
    {
        # Romanian comma-below and Turkish/legacy cedilla look the same in most fonts.
        "\u0218": "\u015e",  # Ș -> Ş
        "\u0219": "\u015f",  # ș -> ş
        "\u021a": "\u0162",  # Ț -> Ţ
        "\u021b": "\u0163",  # ț -> ţ
        # Bullet look-alikes (the font decides the drawn shape).
        "\u25aa": "\u2022",  # ▪
        "\u25cf": "\u2022",  # ●
        "\u2219": "\u2022",  # ∙
        "\u2023": "\u2022",  # ‣
        "\u2043": "\u2022",  # ⁃
    }
)
# Runs of 4+ leader / rule characters: the count cannot be read from the image.
_RUNS = re.compile(r"([._\-=·*~])\1{3,}")
_DOUBLE_SINGLE = re.compile(r"''|``")


def canonical(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_CANON)
    text = _DOUBLE_SINGLE.sub('"', text)
    return _RUNS.sub(lambda m: m.group(1) * 3, text)


def _range(a: int, b: int) -> list[str]:
    return [chr(c) for c in range(a, b + 1)]


def _printable(chars) -> list[str]:
    out = []
    for c in chars:
        cat = unicodedata.category(c)
        if cat.startswith(("C", "Z")) and c != " ":
            continue  # control, format, unassigned, private, separators other than space
        if cat == "Mn":
            continue  # combining marks: labels are NFC, marks come precomposed
        out.append(c)
    return out


def _script_chars(a: int, b: int) -> list[str]:
    """Assigned letters, marks and digits of a script block (combining marks kept:
    Indic vowel signs and virama are separate codepoints even after NFC)."""
    out = []
    for c in _range(a, b):
        cat = unicodedata.category(c)
        if cat == "Cn" or cat.startswith(("C", "Z")):
            continue
        out.append(c)
    return out


# LaTeX commands the recognizer emits as single tokens inside $...$.
LATEX_COMMANDS = (
    "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi "
    "pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega "
    "Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega "
    "frac dfrac tfrac sqrt sum prod int iint oint lim limsup liminf sup inf max min arg det "
    "sin cos tan cot sec csc arcsin arccos arctan sinh cosh tanh log ln exp "
    "left right big Big bigg Bigg langle rangle lfloor rfloor lceil rceil "
    "cdot cdots ldots dots vdots ddots times div pm mp ast star circ bullet "
    "leq geq neq le ge ne ll gg approx sim simeq cong equiv propto "
    "in notin ni subset supset subseteq supseteq cup cap cap setminus emptyset varnothing "
    "forall exists neg land lor wedge vee oplus otimes "
    "to rightarrow leftarrow Rightarrow Leftarrow leftrightarrow Leftrightarrow mapsto "
    "implies iff uparrow downarrow longrightarrow "
    "infty partial nabla hbar ell prime perp parallel mid top angle "
    "hat bar tilde vec dot ddot widehat widetilde overline underline "
    "mathbb mathcal mathbf mathrm mathit mathsf mathfrak boldsymbol operatorname text "
    "quad qquad "
).split()
LATEX_TOKENS = ["\\" + c for c in LATEX_COMMANDS] + [
    "\\{",
    "\\}",
    "\\|",
    "\\,",
    "\\;",
    "\\!",
    "\\ ",
]
_TOKEN_RE = re.compile(r"\\[A-Za-z]+|\\[{}|,;! ]|.", re.S)
# $...$ is math only when it holds LaTeX, so prices ("$5 or $10") stay text.
_MATH_SPAN = re.compile(r"\$[^$]*[\\^_{][^$]*\$")


def build_latin_chars() -> list[str]:
    chars: list[str] = []
    chars += _range(0x20, 0x7E)  # ASCII printable, with space
    chars += _range(
        0x00A1, 0x00FF
    )  # Latin-1 Supplement (letters, ¢ £ ¥ § © ° ± µ ¹²³ ½ ¼ ¾ × ÷ ...)
    chars += _range(0x0100, 0x017F)  # Latin Extended-A
    chars += _range(
        0x0180, 0x024F
    )  # Latin Extended-B (Ș ș Ț ț, Vietnamese Ơ ơ Ư ư, pinyin)
    chars += _range(
        0x1E00, 0x1EFF
    )  # Latin Extended Additional (Vietnamese precomposed)
    chars += _range(0x0391, 0x03A9) + _range(0x03B1, 0x03C9)  # basic Greek
    chars += list("‐‑‒–—―‖‘’‚‛“”„‟†‡•‣…‰′″‹›⁄※")
    chars += list("⁰⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉")
    chars += list("€₹₽₩₺₪₱₫₴₦₡₸")
    chars += list("™℃℉№ℓ℮⅓⅔⅛")
    chars += list("←↑→↓↔⇒⇔↵")
    chars += list("−∓∗∘∙√∞∝∠∧∨∩∪∫≈≠≡≤≥⊂⊃⊆⊇⊕⊗⊥⋅∂∆∇∈∉∋∏∑∀∃∅")
    chars += list("■□▪▫▲△▶►▼▽◀◄◆◇○◦●★☆✓✔✗✘❖➢➤")
    seen, out = set(), []
    for c in _printable(chars):
        if c not in seen and c != "­":  # soft hyphen is invisible
            seen.add(c)
            out.append(c)
    return out


def build_multi_tokens() -> list[str]:
    out = build_latin_chars()
    seen = set(out)
    extra = (
        _script_chars(0x0370, 0x03FF)  # Greek and Coptic (accented Greek)
        + _script_chars(0x0400, 0x045F)  # Cyrillic (Russian, Ukrainian, Serbian, ...)
        + _script_chars(
            0x0490, 0x04FF
        )  # Cyrillic extended letters (Kazakh, Uzbek, ...)
        + _script_chars(0x0900, 0x097F)  # Devanagari (Hindi, Marathi, Nepali)
        + _script_chars(0x0980, 0x09FF)  # Bengali
        + list("।॥₹")
        + LATEX_TOKENS
    )
    for t in extra:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


class Vocab:
    def __init__(self, chars: list[str]):
        self.chars = list(chars)  # tokens: characters, or LaTeX commands
        self.index = {c: i + 1 for i, c in enumerate(self.chars)}
        self.has_commands = any(len(c) > 1 for c in self.chars)

    @classmethod
    def latin(cls) -> "Vocab":
        return cls(build_latin_chars())

    @classmethod
    def multi(cls) -> "Vocab":
        return cls(build_multi_tokens())

    @classmethod
    def named(cls, name: str) -> "Vocab":
        if name not in ("latin", "multi"):
            raise ValueError(f"unknown vocab {name!r}")
        return cls.multi() if name == "multi" else cls.latin()

    def tokenize(self, text: str) -> list[str]:
        """Characters, with a LaTeX command as one token when the vocab has it.

        Inside a $...$ math span spaces are dropped: they are not visible in
        rendered math.  ``decode`` puts back the one space LaTeX needs, between a
        command and a following letter."""
        if not self.has_commands:
            return list(text)
        out: list[str] = []
        pos = 0
        for m in list(_MATH_SPAN.finditer(text)) + [None]:
            end = m.start() if m else len(text)
            out += self._tokens(text[pos:end], math=False)
            if m:
                out += self._tokens(m.group(0), math=True)
                pos = m.end()
        return out

    def _tokens(self, text: str, math: bool) -> list[str]:
        out = []
        for t in _TOKEN_RE.findall(text):
            if math and t.isspace():
                continue
            out.extend([t] if t in self.index or len(t) == 1 else list(t))
        return out

    @property
    def n_classes(self) -> int:
        return len(self.chars) + 1  # + blank

    def normalize(self, text: str) -> str:
        return canonical(text)

    def encode(self, text: str) -> list[int] | None:
        text = self.normalize(text)
        ids = [self.index.get(c) for c in self.tokenize(text)]
        if not ids or any(i is None for i in ids):
            return None
        return ids

    def oov(self, text: str) -> set[str]:
        return {c for c in self.tokenize(self.normalize(text)) if c not in self.index}

    def decode(self, ids: list[int]) -> str:
        toks = [self.chars[i - 1] for i in ids if 0 < i <= len(self.chars)]
        if not self.has_commands:
            return "".join(toks)
        out = []
        for k, t in enumerate(toks):
            out.append(t)
            # "\\alpha" + "x" must not read back as "\\alphax"
            nxt = toks[k + 1] if k + 1 < len(toks) else ""
            if len(t) > 2 and t[0] == "\\" and t[1:].isalpha() and nxt[:1].isalpha():
                out.append(" ")
        return "".join(out)

    def to_json(self) -> str:
        return json.dumps({"blank": 0, "chars": self.chars}, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Vocab":
        d = json.loads(s)
        if d.get("blank", 0) != 0:
            raise ValueError("only blank=0 vocabularies are supported")
        return cls(d["chars"])


# Indic syllable structure.  A broken font mapping (common in Nepali and Hindi
# PDFs converted from legacy fonts) leaves vowel signs with no consonant before
# them ("िामथ", "का ा") or two vowel signs in a row ("णािी"); valid text never does.
_INDIC_BLOCKS = ((0x0900, 0x097F), (0x0980, 0x09FF))
_VIRAMA = {0x094D, 0x09CD}
_NUKTA = {0x093C, 0x09BC}
_MODIFIERS = {0x0900, 0x0901, 0x0902, 0x0903, 0x0981, 0x0982, 0x0983}


def _indic(c: str) -> bool:
    o = ord(c)
    return any(a <= o <= b for a, b in _INDIC_BLOCKS)


def indic_valid(text: str) -> bool:
    """False when a Devanagari or Bengali mark has no base to attach to."""
    prev = ""
    for c in text:
        if _indic(c) and unicodedata.category(c) in ("Mn", "Mc"):
            o = ord(c)
            p_ok = bool(prev) and _indic(prev)
            p_cat = unicodedata.category(prev) if prev else ""
            if (
                o in _MODIFIERS
            ):  # anusvara, candrabindu, visarga: after a letter or a vowel sign
                if not p_ok:
                    return False
            elif o in _VIRAMA or o in _NUKTA:  # after a consonant (or nukta)
                if not (p_ok and (p_cat == "Lo" or ord(prev) in _NUKTA)):
                    return False
            else:  # vowel sign: after a consonant, nukta, or nothing else
                if not (p_ok and (p_cat == "Lo" or ord(prev) in _NUKTA)):
                    return False
        prev = c
    return True


def indic_page_ok(texts: list[str], max_bad: float = 0.05) -> bool:
    """A page whose Indic lines break syllable structure more than ``max_bad`` of
    the time has a broken font mapping; its other lines are suspect too."""
    ind = [t for t in texts if any(_indic(c) for c in t)]
    if not ind:
        return True
    return sum(not indic_valid(t) for t in ind) <= max_bad * len(ind)
