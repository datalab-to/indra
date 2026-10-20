"""Latency with and without the native kernels, paired page by page.

    pip install 'omni-parse-bench[benchmark]' "indra-ocr[native]"
    python bench/ab_latency.py --out runs/ab_auto --mode auto --procs 8

Each worker loads two converters, one with the kernels and one without
(INDRA_OCR_NATIVE=0 behaviour), and reads every page with both, back to back,
the order alternating from page to page.  Two separate benchmark runs drift
apart on a busy or thermally limited machine (a page's time can move 30%
between runs); paired times do not.  A page's time is what bench/opb_indra.py
measures: the PDF opened and the page converted, single-threaded.

Writes pairs.jsonl (per page: ms without and with, whether the markdown and
lines agree) and summary.json, and prints medians, means and the per-page
speedup by page source.  --pin 0,2,4 pins worker k to CPU k (needs psutil).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

_CONV: dict = {}
_LIB = None


def init(args: dict, slot) -> None:
    global _LIB
    import cv2

    import indra_ocr
    from indra_ocr import native

    if args["pin"]:
        import psutil

        with slot.get_lock():
            k = slot.value
            slot.value += 1
        psutil.Process().cpu_affinity([args["pin"][k % len(args["pin"])]])
    cv2.setNumThreads(1)
    _LIB = native.lib
    if _LIB is None:
        raise RuntimeError(f"ab_latency needs the native kernels: {native.status()}")
    _CONV[True] = indra_ocr.load(args["models"], threads=1, mode=args["mode"])
    with native.disabled():
        _CONV[False] = indra_ocr.load(args["models"], threads=1, mode=args["mode"])


def _read(on: bool, path: str):
    import cv2
    import pypdfium2 as pdfium

    from indra_ocr import native

    native.lib = _LIB if on else None
    try:
        t0 = time.perf_counter()
        if path.lower().endswith(".pdf"):
            doc = pdfium.PdfDocument(path)
            try:
                page = _CONV[on].pdf_page(path, doc, 0)
            finally:
                doc.close()
        else:
            page = _CONV[on].ocr(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        return 1000 * (time.perf_counter() - t0), page
    finally:
        native.lib = _LIB


def run_pair(item: tuple[int, str, str]) -> dict:
    k, page_id, path = item
    first = k % 2 == 0  # alternate which configuration goes first
    out = {}
    for on in (first, not first):
        out[on] = _read(on, path)
    (ms_off, a), (ms_on, b) = out[False], out[True]
    same = a.markdown == b.markdown and [ln.text for ln in a.lines] == [
        ln.text for ln in b.lines
    ]
    return {
        "page_id": page_id,
        "source": a.source,
        "ms_off": round(ms_off, 1),
        "ms_on": round(ms_on, 1),
        "same": same,
    }


def _stats(rows: list) -> dict:
    off = [r["ms_off"] for r in rows]
    on = [r["ms_on"] for r in rows]
    ratio = sorted(r["ms_off"] / max(r["ms_on"], 1e-3) for r in rows)
    return {
        "pages": len(rows),
        "ms_off_median": statistics.median(off),
        "ms_on_median": statistics.median(on),
        "ms_off_mean": round(statistics.mean(off), 1),
        "ms_on_mean": round(statistics.mean(on), 1),
        "speedup_of_means": round(statistics.mean(off) / statistics.mean(on), 3),
        "speedup_median_page": round(statistics.median(ratio), 3),
        "speedup_p10_page": round(ratio[len(ratio) // 10], 3),
        "speedup_p90_page": round(ratio[9 * len(ratio) // 10], 3),
        "same_output": sum(r["same"] for r in rows),
    }


def main():
    import multiprocessing as mp

    from omni_parse_bench import benchmark

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["auto", "ocr", "text"], default="auto")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--pages", type=int, default=0, help="spread evenly (0: all)")
    ap.add_argument("--dataset", default=benchmark.DEFAULT_DATASET)
    ap.add_argument("--models", default=None, help="model folder (default: download)")
    ap.add_argument("--pin", default="", help="comma-separated CPUs, worker k on CPU k")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    root, pages, _ = benchmark.load(a.dataset)
    if a.pages:
        pages = pages[:: max(1, len(pages) // a.pages)][: a.pages]
    items = [
        (k, p.page_id, str(benchmark.page_file(a.dataset, root, p.doc_path)))
        for k, p in enumerate(pages)
    ]
    args = {
        "models": a.models,
        "mode": a.mode,
        "pin": [int(c) for c in a.pin.split(",")] if a.pin else [],
    }
    ctx = mp.get_context("spawn")
    slot = ctx.Value("i", 0)  # the next worker's place in --pin
    rows = []
    with ctx.Pool(a.procs, initializer=init, initargs=(args, slot)) as pool:
        for k, r in enumerate(pool.imap_unordered(run_pair, items, chunksize=2), 1):
            rows.append(r)
            if k % 200 == 0:
                print(f"{k}/{len(items)} pages", flush=True)
    with open(a.out / "pairs.jsonl", "w") as f:
        for r in sorted(rows, key=lambda r: r["page_id"]):
            f.write(json.dumps(r) + "\n")
    summary = {"mode": a.mode, "all": _stats(rows)}
    for src in sorted({r["source"] for r in rows}):
        summary[src] = _stats([r for r in rows if r["source"] == src])
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
