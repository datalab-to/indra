"""Pages per second on Omni Parse Bench pages, one single-threaded worker
process per core (the README's throughput table).

    pip install 'omni-parse-bench[benchmark]'
    python bench/throughput.py --cores 4 8 16 32 --pages 300 --mode ocr

The pages are spread evenly over the benchmark (every k-th page).  Each worker
loads the models and reads one warm-up page before the clock starts; the
clock covers reading every page once (PDFs are read as indra_ocr reads them:
the text layer in auto mode, rendered and OCR'd in ocr mode).  --pin 0,2,4
pins worker k to CPU k (needs psutil).
"""

from __future__ import annotations

import argparse
import json
import time

_CONV = None


def init(args: dict, slot, start) -> None:
    global _CONV
    import cv2

    import indra_ocr

    if args["pin"]:
        import psutil

        with slot.get_lock():
            k = slot.value
            slot.value += 1
        psutil.Process().cpu_affinity([args["pin"][k % len(args["pin"])]])
    cv2.setNumThreads(1)
    _CONV = indra_ocr.load(args["models"], threads=1, mode=args["mode"])
    read(args["warmup"])
    start.wait(timeout=1800)  # every worker loaded and warmed up: the clock starts


def read(path: str) -> str:
    import cv2
    import pypdfium2 as pdfium

    if path.lower().endswith(".pdf"):
        doc = pdfium.PdfDocument(path)
        try:
            return _CONV.pdf_page(path, doc, 0).source
        finally:
            doc.close()
    return _CONV.ocr(cv2.imread(path, cv2.IMREAD_GRAYSCALE)).source


def main():
    import multiprocessing as mp

    from omni_parse_bench import benchmark

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cores", type=int, nargs="+", default=[4])
    ap.add_argument("--pages", type=int, default=300)
    ap.add_argument("--mode", choices=["auto", "ocr", "text"], default="ocr")
    ap.add_argument("--dataset", default=benchmark.DEFAULT_DATASET)
    ap.add_argument("--models", default=None, help="model folder (default: download)")
    ap.add_argument("--pin", default="", help="comma-separated CPUs, worker k on CPU k")
    ap.add_argument(
        "--repeat", type=int, default=1, help="runs per core count (best kept)"
    )
    a = ap.parse_args()
    root, pages, _ = benchmark.load(a.dataset)
    step = max(1, len(pages) // a.pages)
    paths = [
        str(benchmark.page_file(a.dataset, root, p.doc_path)) for p in pages[::step]
    ][: a.pages]
    args = {
        "models": a.models,
        "mode": a.mode,
        "warmup": paths[0],
        "pin": [int(c) for c in a.pin.split(",")] if a.pin else [],
    }
    ctx = mp.get_context("spawn")
    rows = []
    for n in a.cores:
        best = None
        for _ in range(a.repeat):
            start = ctx.Barrier(n + 1)
            with ctx.Pool(
                n, initializer=init, initargs=(args, ctx.Value("i", 0), start)
            ) as pool:
                start.wait(timeout=1800)  # a worker that cannot load breaks it
                t0 = time.perf_counter()
                pool.map(read, paths, chunksize=1)
                dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        rows.append(
            {
                "cores": n,
                "pages": len(paths),
                "seconds": round(best, 2),
                "pages_per_s": round(len(paths) / best, 2),
            }
        )
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
