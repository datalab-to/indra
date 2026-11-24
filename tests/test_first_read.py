"""A digital page reads the same on its first read in a process as on later reads.

pdfium gives a character of a substituted font (not embedded, here
HelveticaNeue-Light set at 1 pt and scaled by Tm) the box of its font metrics
until the process renders the glyph, and the box of its outline in documents
opened after that.  Each check runs in a new process, as the state is global."""

import subprocess
import sys

import pytest

# The page's Widths for character codes 32..126 (from a real HelveticaNeue-Light).
WIDTHS = (
    "278 241 370 556 556 889 611 278 241 241 352 600 278 370 278 333 556 556 556 556 "
    "556 556 556 556 556 556 278 278 600 600 600 537 800 630 667 704 685 593 537 741 "
    "704 222 500 648 537 833 704 741 630 741 667 630 556 685 593 907 574 611 574 241 "
    "333 241 600 500 185 519 574 519 574 519 259 556 537 185 185 500 185 833 537 556 "
    "574 574 315 481 296 537 463 741 481 463 463 333 222 333 600"
)


def _substituted_font_pdf(path):
    text = (
        b"BT /F1 1 Tf 10 0 0 10 54 612 Tm 0.0065 Tc "
        b"(has received an Initial Notification of fifty) Tj 0 -1.2 TD "
        b"(affine offices waffle) Tj ET"
    )
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(text) + text + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /HelveticaNeue-Light "
        b"/Encoding /WinAnsiEncoding /FirstChar 32 /LastChar 126 "
        b"/Widths [" + WIDTHS.encode() + b"] /FontDescriptor 6 0 R >>",
        b"<< /Type /FontDescriptor /FontName /HelveticaNeue-Light /Flags 32 "
        b"/FontBBox [-166 -214 1050 967] /ItalicAngle 0 /Ascent 714 /Descent -191 "
        b"/CapHeight 714 /StemV 63 >>",
    ]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offs)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1,
        xref,
    )
    path.write_bytes(out)
    return path


# Prints one line per read: the page's character boxes.
SCRIPT = """
import ctypes, sys
import pypdfium2 as pdfium
import pypdfium2.raw as R
import indra_ocr.document as D

path, how = sys.argv[1], sys.argv[2]


def boxes(page):
    tp = page.get_textpage()
    out = []
    for i in range(tp.count_chars()):
        v = [ctypes.c_double() for _ in range(4)]
        R.FPDFText_GetCharBox(tp.raw, i, *v)
        out.append(tuple(round(x.value, 3) for x in v))
    tp.close()
    return out


class Read(Exception):
    pass


def read_text_layer(page, *a):  # stop at the text layer: print the boxes it gets
    raise Read(boxes(page))


D.read_text_layer = read_text_layer
conv = D.Converter(None)
for _ in range(3):
    doc = pdfium.PdfDocument(path)
    if how == "pdfium":  # render, then read the same document (1.1.2 and earlier)
        doc[0].render(scale=1.0)
        print(boxes(doc[0]))
    else:
        try:
            conv.pdf_page(path, doc, 0)
        except Read as e:
            print(e.args[0])
    doc.close()
"""


def _reads(path, how):
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(path), how],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert len(out) == 3
    return out


def test_first_read_equals_later_reads(tmp_path):
    pdf = _substituted_font_pdf(tmp_path / "p.pdf")
    plain = _reads(pdf, "pdfium")
    if plain[0] == plain[1]:
        pytest.skip("this pdfium build gives the same boxes before and after a render")
    assert plain[1] == plain[2]
    reads = _reads(pdf, "converter")
    assert reads[0] == reads[1] == reads[2]
    assert reads[0] == plain[1]  # the boxes of the glyph outlines
