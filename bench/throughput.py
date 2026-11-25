"""Pages per second on Omni Parse Bench pages, one single-threaded worker
process per core (the README's throughput table), on the CPU or on the GPU.

    pip install 'omni-parse-bench[benchmark]'
    python bench/throughput.py --cores 4 8 16 32 --pages 300 --mode ocr
    python bench/throughput.py --cores 8 16 --device engine --gpus 0,1

The pages are spread evenly over the benchmark (every k-th page).  Each worker
loads the models and reads one warm-up page before the clock starts; the
clock covers reading every page once (PDFs are read as indra_ocr reads them:
the text layer in auto mode, rendered and OCR'd in ocr mode).  --pin 0-15 (or
0,2,4) pins worker k to the k-th CPU.  --device engine reads with
load(device="cuda") (the indra-ocr-cuda engine where installed), worker k on
the GPU --gpus[k % n].
"""

from __future__ import annotations

import argparse
import json
import os
import time

_CONV = None


def cpus(spec: str) -> list[int]:
    out = []
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def init(args: dict, slot, start) -> None:
    global _CONV
    with slot.get_lock():
        k = slot.value
        slot.value += 1
    if args["pin"]:
        os.sched_setaffinity(0, [args["pin"][k % len(args["pin"])]])
    if args["device"] != "cpu":
        # before onnxruntime or the engine sees the GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpus"][k % len(args["gpus"])])
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    import cv2

    import indra_ocr

    cv2.setNumThreads(1)
    kw = {"device": "cuda"} if args["device"] != "cpu" else {}
    _CONV = indra_ocr.load(args["models"], threads=1, mode=args["mode"], math=not args["no_math"], **kw)
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

    from omni_parse_bench import data

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cores", type=int, nargs="+", default=[4])
    ap.add_argument("--pages", type=int, default=300)
    ap.add_argument("--mode", choices=["auto", "ocr", "text"], default="ocr")
    ap.add_argument("--no-math", action="store_true")
    ap.add_argument("--dataset", default=data.DEFAULT_DATASET)
    ap.add_argument("--models", default=None, help="model folder (default: download)")
    ap.add_argument("--pin", default="", help="CPUs, e.g. 0-15 or 0,2,4: worker k on the k-th")
    ap.add_argument("--device", choices=["cpu", "engine"], default="cpu")
    ap.add_argument("--gpus", default="0", help="with --device engine: GPU ids, worker k on the k-th")
    ap.add_argument(
        "--repeat", type=int, default=1, help="runs per core count (best kept)"
    )
    a = ap.parse_args()
    _, pages = data.read_manifest(a.dataset)
    step = max(1, len(pages) // a.pages)
    paths = [str(p.doc_path) for p in pages[::step]][: a.pages]
    args = {
        "models": a.models,
        "mode": a.mode,
        "no_math": a.no_math,
        "warmup": paths[0],
        "pin": cpus(a.pin),
        "device": a.device,
        "gpus": [int(g) for g in a.gpus.split(",")],
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
                "device": a.device,
                "workers": n,
                "pages": len(paths),
                "seconds": round(best, 2),
                "pages_per_s": round(len(paths) / best, 2),
            }
        )
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
