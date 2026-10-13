"""Indra OCR: fast CPU OCR (line detection, CTC recognition, reading order, markdown)."""

from .document import Converter
from .models import model_dir
from .parallel import convert_many, usable_cores
from .pipeline import InputError, Line, OrtOCR, Page, load_pages

__all__ = [
    "Converter",
    "InputError",
    "Line",
    "OrtOCR",
    "Page",
    "convert_many",
    "usable_cores",
    "load_pages",
    "load",
    "model_dir",
]
__version__ = "0.3.0"


def load(
    path=None,
    threads: int = 4,
    dpi: float = 200.0,
    mode: str = "auto",
    pdf_order: bool = True,
    image_ocr: bool = True,
    providers: list | None = None,
    furniture: bool = False,
    math: bool = False,
) -> Converter:
    """A ready Converter (see ``Converter`` for the options).  mode="ocr" never
    reads the PDF text layer.  furniture: keep page headers and footers in the
    markdown.  math: write math as LaTeX (slower on math pages; off by default)."""
    return Converter(
        OrtOCR(
            model_dir(path, math=math),
            threads=threads,
            providers=providers,
            furniture=furniture,
            math=math,
        ),
        dpi=dpi,
        mode=mode,
        pdf_order=pdf_order,
        image_ocr=image_ocr,
    )
