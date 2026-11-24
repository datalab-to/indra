"""Command line: PDFs or images in, markdown (or text, or JSON lines) out.

indra-ocr paper.pdf                    # markdown to stdout
indra-ocr *.pdf --out results/         # one .md per input
indra-ocr scan.png --format json       # lines with boxes, confidences, timings

Pages run in parallel on all the cores the process may use (--cores to limit).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from . import InputError
from .parallel import convert_many, usable_cores


def add_source_args(ap: argparse.ArgumentParser) -> None:
    """Options that choose between the PDF text layer and OCR."""
    g = ap.add_argument_group("text layer or OCR (PDFs)")
    g.add_argument(
        "--mode",
        choices=["auto", "ocr", "text"],
        default="auto",
        help="auto: text layer where it passes the checks, OCR elsewhere (default); "
        "ocr: never read the text layer; text: use the text layer even when the "
        "checks fail",
    )
    g.add_argument(
        "--no-image-ocr",
        action="store_true",
        help="on text-layer pages, do not OCR text inside images",
    )


def source_options(a) -> dict:
    return {
        "mode": a.mode,
        "image_ocr": not a.no_image_ocr,
    }


def _page_count(v: str) -> int:
    """argparse type for --max-pages: 0 (all pages) or a positive number."""
    n = int(v)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be 0 (all pages) or more, not {n}")
    return n


def line_json(ln) -> dict:
    """A line as JSON: words only when they were asked for (load(words=True))."""
    d = asdict(ln)
    if d.get("words") is None:
        d.pop("words", None)
    return d


def main(argv=None):
    """Exit code 0, or 1 when an input failed or the models are not available."""
    from .models import ModelError

    try:
        return _main(argv)
    except ModelError as e:
        print(f"indra-ocr: {e}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        # The reader stopped early (indra-ocr x.pdf | head): no traceback, and
        # no second error when Python flushes stdout at exit.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1


def _main(argv=None):
    from . import __version__

    ap = argparse.ArgumentParser(
        prog="indra-ocr",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=f"indra-ocr {__version__}")
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument(
        "--out", type=Path, default=None, help="write one file per input here"
    )
    ap.add_argument("--format", choices=["md", "text", "json"], default="md")
    add_source_args(ap)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument(
        "--cores",
        "--threads",
        dest="cores",
        type=int,
        default=0,
        help=f"cores to use (default: all this process may run on, here {usable_cores()})",
    )
    ap.add_argument("--dpi", type=float, default=200.0)
    ap.add_argument(
        "--max-pages",
        type=_page_count,
        default=0,
        help="read at most this many pages of each file (0, the default: all)",
    )
    ap.add_argument("--password", default=None, help="password of encrypted PDFs")
    ap.add_argument(
        "--math",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write math as LaTeX ($...$) (on by default; --no-math skips the math models)",
    )
    ap.add_argument(
        "--math-spans",
        action="store_true",
        help="with --math, also read inline math inside text-layer lines (adds time on text-layer pages)",
    )
    ap.add_argument(
        "--math-rescore",
        action="store_true",
        help="with --math, re-read lines MathRec is unsure of with a beam search (adds time on pages with math)",
    )
    ap.add_argument(
        "--words",
        action="store_true",
        help="with --format json, give each line its words with boxes (lines[].words)",
    )
    ap.add_argument(
        "--keep-furniture",
        action="store_true",
        help="keep page headers and footers (page numbers, running titles) in the markdown",
    )
    ap.add_argument(
        "--timings", action="store_true", help="print per-page timings to stderr"
    )
    a = ap.parse_args(argv)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
    ext = {"md": ".md", "text": ".txt", "json": ".jsonl"}[a.format]
    failed, names = 0, set()
    results = convert_many(
        a.inputs,
        cores=a.cores,
        max_pages=a.max_pages,
        password=a.password,
        path=a.model_dir,
        dpi=a.dpi,
        furniture=a.keep_furniture,
        math=a.math,
        math_spans=a.math_spans,
        math_rescore=a.math_rescore,
        words=a.words,
        **source_options(a),
    )
    for path, result in results:
        path = Path(path)
        chunks = []
        if isinstance(result, InputError):
            print(f"indra-ocr: {result}", file=sys.stderr)
            failed += 1
            continue
        for i, page in result:
            if page.source == "error":
                print(f"indra-ocr: {path} page {i + 1}: {page.note}", file=sys.stderr)
            if a.format == "json":
                chunks.append(
                    json.dumps(
                        {
                            "file": str(path),
                            "page": i,
                            "source": page.source,
                            "rotation": page.rotation,
                            "skew": page.skew,
                            "vertical": page.vertical,
                            "note": page.note,
                            "lines": [line_json(ln) for ln in page.lines],
                            "markdown": page.markdown,
                            "blocks": page.blocks,
                            "timings": page.timings,
                        },
                        ensure_ascii=False,
                    )
                )
            elif a.format == "md":
                chunks.append(page.markdown if page.markdown is not None else page.text)
            else:
                chunks.append(page.text)
            if a.timings:
                t = page.timings
                print(
                    f"{path} page {i + 1}: {page.source}, {t.get('total_ms', 0):.0f} ms {page.note}",
                    file=sys.stderr,
                )
        sep = "\n" if a.format == "json" else "\n\n"
        body = sep.join(chunks) + "\n"
        if a.out:
            name, k = path.stem, 1
            while name in names:  # two inputs with one name: keep both outputs
                k += 1
                name = f"{path.stem}_{k}"
            names.add(name)
            (a.out / (name + ext)).write_text(body, encoding="utf-8")
        else:
            sys.stdout.write(body)
    if failed:
        print(f"indra-ocr: {failed} of {len(a.inputs)} inputs failed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
