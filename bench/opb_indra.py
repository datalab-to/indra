"""Score Indra OCR on Omni Parse Bench (github.com/datalab-to/omni_parse_bench).

    pip install 'omni-parse-bench[benchmark]'
    python bench/opb_indra.py --out runs/opb_auto --mode auto --procs 8

Each page goes through the Indra OCR Converter (one thread per process, page
headers and footers kept: the benchmark asks for everything on the page).  The
markdown becomes HTML through the benchmark's own converter, as every vendor's
markdown does, and the layout blocks map onto the benchmark's labels.  Latency
is the converter's wall time per page, page loading included.

Writes summary.json (the benchmark's summarize) and pages.jsonl (per page:
latency, source, verdict counts).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Indra OCR block class -> Omni Parse Bench label (views.LABELS)
LABELS = {
    "Text": "text",
    "Section-Header": "heading",
    "Page-Header": "page_furniture",
    "Page-Footer": "page_furniture",
    "List-Group": "list",
    "Table": "table",
    "Caption": "caption",
    "Footnote": "footnote",
    "Table-Of-Contents": "text",
    "Form": "form",
}

_CONV = None
_ARGS: dict = {}


def init(args: dict) -> None:
    global _CONV, _ARGS
    import indra_ocr

    _ARGS = args
    _CONV = indra_ocr.load(
        args["models"],
        threads=1,
        mode=args["mode"],
        furniture=True,
        math=args.get("math", False),
    )


def outputs_of(page, w: float, h: float) -> dict:
    from omni_parse_bench import markdown

    md = page.markdown if page.markdown is not None else page.text
    clip = lambda v: min(1.0, max(0.0, v))  # noqa: E731
    blocks = [
        {
            "label": LABELS.get(b["label"], "text"),
            "bbox": [
                clip(b["bbox"][0] / w),
                clip(b["bbox"][1] / h),
                clip(b["bbox"][2] / w),
                clip(b["bbox"][3] / h),
            ],
            "text": b["text"],
        }
        for b in page.blocks
    ]
    return {"html": markdown.to_html(md), "blocks": blocks}


def run_page(item: tuple[str, str]) -> dict:
    import cv2
    import pypdfium2 as pdfium
    from omni_parse_bench import benchmark, score

    page_id, doc_path = item
    root, _, by_page = benchmark.load(_ARGS["dataset"])
    path = benchmark.page_file(_ARGS["dataset"], root, doc_path)
    tests = by_page[page_id]
    t0 = time.perf_counter()
    try:
        if str(path).lower().endswith(".pdf"):
            doc = pdfium.PdfDocument(str(path))
            page = _CONV.pdf_page(str(path), doc, 0)
            pw, ph = doc[0].get_size()
            k = _CONV.dpi / 72.0
            w, h = pw * k, ph * k
            doc.close()
        else:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            page = _CONV.ocr(gray)
            h, w = gray.shape
        ms = 1000 * (time.perf_counter() - t0)
        verdicts = score(outputs_of(page, w, h), tests)
        source = page.source
    except Exception as e:  # noqa: BLE001  (a failed page fails its tests, it does not stop the run)
        ms, source = (
            1000 * (time.perf_counter() - t0),
            f"error: {type(e).__name__}: {e}",
        )
        verdicts = score({}, tests)
    return {"page_id": page_id, "ms": ms, "source": source, "verdicts": verdicts}


def main():
    import multiprocessing as mp

    from omni_parse_bench import benchmark, summarize

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default=benchmark.DEFAULT_DATASET)
    ap.add_argument("--models", default=None, help="model folder (default: download)")
    ap.add_argument("--mode", choices=["auto", "ocr", "text"], default="auto")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--suites", default="", help="comma-separated suites (default: all)"
    )
    ap.add_argument("--math", action="store_true", help="write math as LaTeX")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    root, pages, by_page = benchmark.load(a.dataset)
    keep = set(a.suites.split(",")) if a.suites else None
    pages = [p for p in pages if keep is None or p.suite in keep][: a.limit or None]
    items = [(p.page_id, p.doc_path) for p in pages]
    args = {"models": a.models, "mode": a.mode, "dataset": a.dataset, "math": a.math}
    t0 = time.time()
    rows, verdicts = [], []
    with mp.get_context("spawn").Pool(
        a.procs, initializer=init, initargs=(args,)
    ) as pool:
        for k, r in enumerate(pool.imap_unordered(run_page, items, chunksize=4), 1):
            verdicts += r["verdicts"]
            rows.append(
                {
                    "page_id": r["page_id"],
                    "ms": round(r["ms"], 1),
                    "source": r["source"],
                    "pass": sum(v.result == "pass" for v in r["verdicts"]),
                    "tests": len(r["verdicts"]),
                }
            )
            if k % 200 == 0:
                print(f"{k}/{len(items)} pages, {time.time() - t0:.0f} s", flush=True)
    tests = [t for p in pages for t in by_page[p.page_id]]
    summary = summarize(verdicts, tests, pages)
    ms = sorted(r["ms"] for r in rows)
    summary["indra_ocr"] = {
        "mode": a.mode,
        "pages": len(rows),
        "ms_median": ms[len(ms) // 2],
        "ms_p90": ms[int(0.9 * (len(ms) - 1))],
        "sources": {
            s: sum(r["source"] == s for r in rows)
            for s in {r["source"][:20] for r in rows}
        },
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    with open(a.out / "pages.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(
        json.dumps(
            {"headline": summary["headline"], **summary["indra_ocr"]},
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
