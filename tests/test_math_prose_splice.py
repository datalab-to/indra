"""splice_prose: an accepted whole-line MathRec reading keeps the plain reading's
prose (accents, other scripts) around its $...$ spans; its LaTeX stays as is."""
from indra_ocr import math as M


def _spans(text):
    return text.split("$")[1::2]


def _check(draft, reading, want):
    got = M.splice_prose(reading, draft)
    assert got == want
    # the formulas stay exactly as MathRec wrote them (accent spans aside)
    assert [s for s in _spans(got)] == [s for s in _spans(reading) if not M._ACCENT.fullmatch(s)]


# Examples from real math lines: (plain reading, MathRec reading, spliced)
AUDIT = [
    (  # az: ə read as o
        "2. \\sigma məlum deyil (David R. Anderson, 2008). Burada \\sigma ana kütlədə kəmiyyət",
        "2. $\\sigma$ molum deyil (David R. Anderson, 2008). Burada $\\sigma$ ana kütlado komiyyat",
        "2. $\\sigma$ məlum deyil (David R. Anderson, 2008). Burada $\\sigma$ ana kütlədə kəmiyyət",
    ),
    (  # tr: dotless ı, ü
        "böyükdür (tnt > tkr) , onda bu parametr verilmiş etibarlılıq əmsalı, sərbəstlik",
        "böyükdir $( t _ { a l } > t _ { k r } )$ , onda bu parametr verilmiş etibarliliq omsali, sorbastlik",
        "böyükdür $( t _ { a l } > t _ { k r } )$ , onda bu parametr verilmiş etibarlılıq əmsalı, sərbəstlik",
    ),
    (  # tr layer line
        "şeklinde istatistiksel çapandır. Γ <<E0 için tesir kesiti E=E0 olduğunda maksimumdadır ve",
        "şeklinde istatistiksel çapandir. $\\, \\Gamma \\ll \\mathrm { E } _ { 0 }$ için tesir kesiti "
        "$\\mathrm { E = E } _ { 0 }$ olduğunda maksimumdadir ve",
        "şeklinde istatistiksel çapandır. $\\, \\Gamma \\ll \\mathrm { E } _ { 0 }$ için tesir kesiti "
        "$\\mathrm { E = E } _ { 0 }$ olduğunda maksimumdadır ve",
    ),
    (  # hu: ö read as 6, é read as 6
        "b) Jelölje v_2 a C test kezdősebességét. Ha az A és a B test egyszerre ér talajt, akkor velük",
        "$b )$ Jelölje $v _ { 2 }$ a $C$ test kezd6sebességét. Ha az $A$ és a $B$ test egyszerre "
        "$\\operatorname { e r }$ talajt, akkor velük",
        "$b )$ Jelölje $v _ { 2 }$ a $C$ test kezdősebességét. Ha az $A$ és a $B$ test egyszerre "
        "$\\operatorname { e r }$ talajt, akkor velük",
    ),
    (
        "vezessük be az x = 2 jelölést.",
        "Vezessiik be az $x = \\frac { y _ 2 } { \\dots }$ jel6lést!",
        "vezessük be az $x = \\frac { y _ 2 } { \\dots }$ jelölést.",
    ),
    (  # is: þ read as p, æ as a
        "1 + 2 +... + 2^n = 2^1 - 1 og þá fæst",
        "$1 + 2 + \\dots + 2 ^ n = 2 ^ { n + 1 } - 1$ og pá fast",
        "$1 + 2 + \\dots + 2 ^ n = 2 ^ { n + 1 } - 1$ og þá fæst",
    ),
    (
        "= 2 k.m, þar sem m er oddatala. Hér er k \\geq 1 því n er slétt,",
        "$= 2 ^ { k } . m ,$ par sem $m$ er oddatala. Hér er $k \\geq 1$ pví $n$ er slétt,",
        "$= 2 ^ { k } . m ,$ þar sem $m$ er oddatala. Hér er $k \\geq 1$ því $n$ er slétt,",
    ),
    (  # pl: ł read as h
        "fluorescencji w stanie stałym Ф = 0.94.",
        "fluorescencji $w$ stanie stahym $\\Phi = 0 . 9 4$.",
        "fluorescencji $w$ stanie stałym $\\Phi = 0 . 9 4$.",
    ),
    (  # uk: Cyrillic read as Latin look-alikes
        "Toдi 2ơ=ED84 - ED16 = 5,28 - 0,43 = 4,85 мr/",
        "Toni $2 \\sigma$ =ED84 - ED16 = 5,28 - $0 , 4 3 = 4 , 8 5$ Mr/",
        "Toдi $2 \\sigma$ =ED84 - ED16 = 5,28 - $0 , 4 3 = 4 , 8 5$ мr/",
    ),
]


def test_audit_examples_keep_the_plain_prose():
    for draft, reading, want in AUDIT:
        _check(draft, reading, want)


def test_accent_written_as_math_goes_back_to_the_letter():
    # pt: é read as \dot { \epsilon }
    draft = "é igual a E_n „ 1 = 13,6 eV, a constante de Planck é igual a"
    reading = (
        "$\\dot { \\epsilon }$ igual a $E _ { n = 1 } = - 1 3 , 6 \\ \\mathrm { e V } , \\ a$ "
        "constante de Planck $\\dot { \\epsilon }$ igual a"
    )
    want = "é igual a $E _ { n = 1 } = - 1 3 , 6 \\ \\mathrm { e V } , \\ a$ constante de Planck é igual a"
    _check(draft, reading, want)


def test_all_math_line_unchanged():
    draft = "cospi = မုန် အ cosp ="
    reading = "$\\cos \\beta = \\frac { \\log } { A D } \\Rightarrow \\cos \\beta = \\frac { a - b } { 2 d }$"
    assert M.splice_prose(reading, draft) == M.restore_accents(reading, draft) == reading


def test_unsure_alignment_keeps_todays_text():
    # the plain reading is another text (a misrouted crop): no splice
    draft = "Произвольный текст совсем другой строки здесь"
    reading = "Let $x _ 1$ be the first root of the polynomial"
    assert M.splice_prose(reading, draft) == M.restore_accents(reading, draft)
    # most prose letters disagree: no splice
    draft = "Ωζξ ψφ ğüş $ 2 çöğ ıüş"
    reading = "abc de $x ^ 2$ fgh ijk"
    assert M.splice_prose(reading, draft, "TODAY") == "TODAY"


def test_plain_latin_prose_is_byte_identical():
    cases = [
        ("trục này kí hiệu là L1 , theo hệ quả 2.3 ta có:", "trục này kí hiệu là $L _ 1$, theo hệ quả 2.3 ta có:"),
        ("(354 kun × 12 yil+384 kun × 7 yil+4 kun=6940 kun).",
         "(354 kun $\\times 1 2$ yil+384 kun $\\times 7$ yil+4 $\\mathrm { k u n } = 6 9 4 0$ kun)."),
        ("where x2 + y2 = r2 holds for all points", "where $x ^ 2 + y ^ 2 = r ^ 2$ holds for all points"),
        ("Let f(x) = ax + b , then", "Let $f ( x ) = a x + b$, then"),
        ("the sum E an converges", "the sum $\\sum a _ { n }$ converges"),
        ("", "$x ^ 2$"),
    ]
    for draft, reading in cases:
        assert M.splice_prose(reading, draft) == M.restore_accents(reading, draft)


def test_prose_inside_text_command_is_left_alone():
    # prose that MathRec wrote inside a span is math to the splice: today's text
    draft = "Klīniskais robežpunkts: J\\leq0,016 mg/l; R > 0,016 mg/l."
    reading = "Kliniskais robezpunkts: $\\mathrm { ~ J \\le 0 , 0 1 6 ~ m g / l ; ~ R > 0 , 0 1 6 ~ m g / l . }$"
    assert M.splice_prose(reading, draft) == M.restore_accents(reading, draft)


class _Reader:
    accept = M.MathReader.accept
    _spliced = M.MathReader._spliced
    prose_splice = True

    def __init__(self):
        self.accept_head = None


def test_accept_splices_and_the_flag_turns_it_off():
    draft, reading, want = AUDIT[5]
    r = _Reader()
    assert r.accept(reading, 0.95, draft) == want
    r.prose_splice = False
    assert r.accept(reading, 0.95, draft) == M.accept_math(reading, draft)
    assert r.accept(reading, 0.5, draft) is None
