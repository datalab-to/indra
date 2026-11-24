"""Indra OCR: fast CPU OCR (line detection, CTC recognition, reading order, markdown)."""

from .document import Converter
from .models import model_dir
from .parallel import convert_many, usable_cores
from .models import ModelError
from .pipeline import ORIENT_MIN, InputError, Line, OrtOCR, Page, Word, load_pages

__all__ = [
    "Converter",
    "InputError",
    "ModelError",
    "Line",
    "OrtOCR",
    "Page",
    "Word",
    "convert_many",
    "usable_cores",
    "load_pages",
    "load",
    "model_dir",
]
__version__ = "1.2.0"


def load(
    path=None,
    threads: int = 4,
    dpi: float = 200.0,
    mode: str = "auto",
    image_ocr: bool = True,
    providers: list | None = None,
    furniture: bool = False,
    math: bool = True,
    math_spans: bool = False,
    math_rescore: bool = False,
    orient_min: float = ORIENT_MIN,
    rot_gate: bool | None = None,
    words: bool = False,
) -> Converter:
    """A ready Converter (see ``Converter`` for the options).  mode="ocr" never
    reads the PDF text layer.  furniture: keep page headers and footers in the
    markdown.  math: write math as LaTeX (on by default; math=False skips the math
    models and their download).  math_spans: also read inline math inside text-layer
    lines (math_span.py; off by default: it adds time on text-layer pages).
    math_rescore: read lines MathRec is unsure of again with a beam search and keep
    the beam that agrees best with the line's draft (math.py; off by default: it
    adds time on pages with math).
    orient_min: the orient.onnx probability above which a page is turned at once
    (below it, a page orient.onnx sees as sideways is settled by reading lines
    each way).  rot_gate: on OCR pages, also read text turned a quarter turn
    (axis labels, margin labels on drawings, vertical table headers) from ink that no
    upright line covers (rot_gate.py; off by default: it adds time on pages with
    such ink; None follows INDRA_ROT_GATE=1).  words: give each line its words with
    boxes (Line.words: text and [x0, y0, x1, y1]; off by default)."""
    return Converter(
        OrtOCR(
            model_dir(path, math=math),
            threads=threads,
            providers=providers,
            furniture=furniture,
            math=math,
            math_spans=math_spans,
            math_rescore=math_rescore,
            orient_min=orient_min,
            rot_gate=rot_gate,
            words=words,
        ),
        dpi=dpi,
        mode=mode,
        image_ocr=image_ocr,
    )
