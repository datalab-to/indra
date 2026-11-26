"""Score Indra OCR on Omni Parse Bench (github.com/datalab-to/omni_parse_bench), 1.8 or newer.

    pip install 'omni-parse-bench[benchmark]'
    python bench/opb_indra.py --out runs/opb_auto --mode auto --procs 8

Each page goes through the Indra OCR Converter in a pool of single-threaded
worker processes, as the benchmark's own Indra OCR server reads it (math as
LaTeX, page headers and footers kept: the benchmark asks for everything on the
page).  The outputs are mapped by the benchmark's own Indra OCR adapter
(omni_parse_bench.harness.providers.indra_ocr.parse: the markdown through
markdown.to_html, the blocks onto its labels, turned back onto the page as
sent), written as predictions/<page_id>.json ({"pages": [its outputs]}), and graded by the benchmark's
grader (benchmark.default().grade: the folder then holds its scores).  Latency
is the converter's wall time per page, page loading included.

Writes predictions/, the grader's folders, pages.jsonl (per page: latency,
source) and summary.json (the grade, and latency by source).  --dump also writes
outputs.jsonl (per page: markdown, lines, blocks, rotation, skew), to compare two
runs page by page (bench/compare_outputs.py).  --pin 0-7 (or 0,2,4) pins worker
k to the k-th CPU.  --device cuda reads with load(device="cuda") (the GPU engine
where installed), worker k on the GPU --gpus[k % n].  --suites or --limit read part of the benchmark; its other
pages count as failed in the headline (the grader's partial mode).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

_CONV = None
_ARGS: dict = {}


def cpus(spec: str) -> list[int]:
    out = []
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def init(args: dict, slot) -> None:
    global _CONV, _ARGS
    _ARGS = args
    with slot.get_lock():
        k = slot.value
        slot.value += 1
    if args["pin"]:
        os.sched_setaffinity(0, [args["pin"][k % len(args["pin"])]])
    kw = {}
    if args["device"] != "cpu":
        # before onnxruntime or the GPU engine sees the GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpus"][k % len(args["gpus"])])
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        kw = {"device": "cuda"}
    import cv2

    import indra_ocr

    cv2.setNumThreads(1)
    _CONV = indra_ocr.load(
        args["models"], threads=1, mode=args["mode"], furniture=True, math=args["math"], **kw
    )


def _image_size(path: Path) -> list[int]:
    """The page as the Converter reads it, [width, height] in pixels (the
    benchmark's Indra OCR server's ``size``)."""
    import pypdfium2 as pdfium

    from indra_ocr import pipeline

    if not pipeline.is_pdf(path):
        h, w = pipeline.read_images(path)[0].shape[:2]
        return [w, h]
    doc = pdfium.PdfDocument(str(path))
    try:
        page = doc[0]
        k = _CONV.page_scale(page)
        w, h = page.get_size()
        return [round(w * k), round(h * k)]
    finally:
        doc.close()


def run_page(item: tuple) -> dict:
    from omni_parse_bench.harness import contract
    from omni_parse_bench.harness.providers import indra_ocr as adapter

    page_id, path, wants = item
    t0 = time.perf_counter()
    out = None
    try:
        _, page = next(_CONV.convert(path, max_pages=1))
        ms = 1000 * (time.perf_counter() - t0)
        if page.source == "error":
            raise RuntimeError(page.note)
        raw = {
            "markdown": page.markdown,
            "text": page.text,
            "blocks": page.blocks,
            "rotation": page.rotation,
            "image_size": _image_size(Path(path)),
        }
        outputs, errs = adapter.parse(raw, contract.Request(frozenset(wants) & adapter.SUPPORTS, {}))
        source = page.source
        if _ARGS.get("dump"):
            out = {
                "markdown": page.markdown,
                "lines": [[ln.box, ln.text, ln.conf] for ln in page.lines],
                "blocks": page.blocks,
                "rotation": page.rotation,
                "skew": page.skew,
                "source": page.source,
            }
    except Exception as e:  # noqa: BLE001  (a failed page fails its tests, it does not stop the run)
        ms = 1000 * (time.perf_counter() - t0)
        source, outputs, errs = f"error: {type(e).__name__}: {e}", {}, {}
        out = {"error": source}
    return {"page_id": page_id, "ms": ms, "source": source, "outputs": outputs, "errors": errs, "out": out}


def main():
    import multiprocessing as mp

    from omni_parse_bench import benchmark, data

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["auto", "ocr", "text"], default="auto")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--models", default=None, help="model folder (default: download)")
    ap.add_argument("--dataset", default=data.DEFAULT_DATASET)
    ap.add_argument("--limit", type=int, default=0, help="first N pages only")
    ap.add_argument("--suites", default="", help="comma-separated suites (default: all)")
    ap.add_argument(
        "--no-math", action="store_true", help="load(math=False): math read as plain text"
    )
    ap.add_argument("--dump", action="store_true", help="also write outputs.jsonl (page outputs)")
    ap.add_argument("--pin", default="", help="CPUs, e.g. 0-7 or 0,2,4: worker k on the k-th")
    ap.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="cuda: load(device='cuda'), worker k on --gpus[k]"
    )
    ap.add_argument("--gpus", default="0", help="with --device cuda: GPU ids")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    _, pages = data.read_manifest(a.dataset)
    keep = set(a.suites.split(",")) if a.suites else None
    chosen = [p for p in pages if keep is None or p.suite in keep][: a.limit or None]
    items = [(p.page_id, str(p.doc_path), tuple(p.outputs)) for p in chosen]
    args = {
        "models": a.models,
        "mode": a.mode,
        "math": not a.no_math,
        "dump": a.dump,
        "pin": cpus(a.pin),
        "device": a.device,
        "gpus": [int(g) for g in a.gpus.split(",")],
    }
    ctx = mp.get_context("spawn")
    pred = a.out / "predictions"
    pred.mkdir(exist_ok=True)
    rows, dumps = [], []
    t0 = time.time()
    with ctx.Pool(a.procs, initializer=init, initargs=(args, ctx.Value("i", 0))) as pool:
        for k, r in enumerate(pool.imap_unordered(run_page, items, chunksize=2), 1):
            (pred / f"{r['page_id']}.json").write_text(json.dumps({"pages": [r["outputs"]]}, ensure_ascii=False), encoding="utf-8")
            rows.append({"page_id": r["page_id"], "ms": round(r["ms"], 1), "source": r["source"]})
            if a.dump and r["out"] is not None:
                dumps.append({"page_id": r["page_id"], **r["out"]})
            if k % 200 == 0:
                print(f"{k}/{len(items)} pages, {time.time() - t0:.0f} s", flush=True)
    wall = time.time() - t0
    with open(a.out / "pages.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if a.dump:
        with open(a.out / "outputs.jsonl", "w", encoding="utf-8") as f:
            for d in sorted(dumps, key=lambda d: d["page_id"]):
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    partial = len(chosen) < len(pages)
    graded = benchmark.default(a.dataset).grade(a.out, partial=partial)
    by: dict = {}
    for r in rows:
        src = r["source"] if not r["source"].startswith("error") else "error"
        by.setdefault(src, []).append(r["ms"])
        by.setdefault("all", []).append(r["ms"])

    def stats(ms):
        ms = sorted(ms)
        return {
            "pages": len(ms),
            "ms_mean": round(statistics.fmean(ms), 1),
            "ms_median": round(statistics.median(ms), 1),
            "ms_p90": round(ms[int(0.9 * (len(ms) - 1))], 1),
        }

    summary = {
        "grade": graded,
        "partial": partial,
        "indra_ocr": {**stats(by["all"]), "wall_s": round(wall, 1), "procs": a.procs, "mode": a.mode,
                      "device": a.device,
                      "math": not a.no_math, "by_source": {k: stats(v) for k, v in by.items() if k != "all"}},
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary["indra_ocr"], indent=1))


if __name__ == "__main__":
    main()
