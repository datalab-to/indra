"""Two behaviours of the pipeline, paired page by page: outputs and time.

    pip install 'omni-parse-bench[benchmark]' psutil
    python bench/ab_variant.py --variant native --out runs/ab_native --procs 16 --pin 0-15

A variant names a change: "a" is the behaviour before it, "b" the code as it
is.  Each worker reads every page twice, once each way, the order alternating
from page to page, and records both times and whether the outputs (markdown,
lines, blocks, rotation, skew, source) are the same.  Writes pairs.jsonl and
summary.json, and prints the pages that differ and the speedup by page source.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

_CONV = None
_MODE = ["b"]  # the behaviour the next read runs


def _variant_native():
    """a: the native kernels off (indra_ocr.native.disabled), b: on."""
    from contextlib import nullcontext

    from indra_ocr import native

    if native.lib is None:
        raise SystemExit(f"the native kernels are not on: {native.status()}")
    return lambda mode: native.disabled() if mode == "a" else nullcontext()


def _variant_recognizer_math():
    """a: every OrtOCR.recognize call reads with MathRec too, as in 1.2.0 (the
    symbol-digit fix and the turn checks included); b: the code as it is."""
    import inspect
    from contextlib import nullcontext

    from indra_ocr import pipeline

    rec0 = pipeline.OrtOCR.recognize
    if "math" not in inspect.signature(rec0).parameters:
        raise SystemExit("recognizer-math needs OrtOCR.recognize(math=...)")

    def recognize(self, gray, boxes, batch=64, vertical=False, math=True):
        return rec0(self, gray, boxes, batch, vertical, math or _MODE[0] == "a")

    pipeline.OrtOCR.recognize = recognize
    return lambda mode: nullcontext()


def _variant_pieces():
    """a: every line model runs each batch whole, as in 1.2.0; b: in pieces of
    rows (ort_session.RowPieces), the code as it is."""
    from contextlib import nullcontext

    from indra_ocr import ort_session

    if not hasattr(ort_session, "RowPieces"):
        raise SystemExit("pieces needs ort_session.RowPieces")
    rows0 = ort_session.RowPieces.rows

    def rows(self, x):
        return len(x) if _MODE[0] == "a" else rows0(self, x)

    ort_session.RowPieces.rows = rows
    return lambda mode: nullcontext()


# A variant patches the pipeline once in each worker and returns, for a mode
# ("a" or "b"), the context the page is read in.
VARIANTS = {
    "native": _variant_native,
    "recognizer-math": _variant_recognizer_math,
    "pieces": _variant_pieces,
}
_CTX = None


def init(args: dict, slot) -> None:
    global _CONV
    import cv2

    import indra_ocr

    if args["pin"]:
        import os

        with slot.get_lock():
            k = slot.value
            slot.value += 1
        os.sched_setaffinity(0, [args["pin"][k % len(args["pin"])]])
    cv2.setNumThreads(1)
    global _CTX
    _CTX = VARIANTS[args["variant"]]()
    _CONV = indra_ocr.load(
        args["models"], threads=1, mode=args["mode"], furniture=True, math=not args["no_math"]
    )


def _read(path: str):
    import cv2
    import pypdfium2 as pdfium

    if path.lower().endswith(".pdf"):
        doc = pdfium.PdfDocument(path)
        try:
            return _CONV.pdf_page(path, doc, 0)
        finally:
            doc.close()
    return _CONV.ocr(cv2.imread(path, cv2.IMREAD_GRAYSCALE))


def _out(page) -> dict:
    return {
        "markdown": page.markdown,
        "lines": [[ln.box, ln.text, ln.conf] for ln in page.lines],
        "blocks": page.blocks,
        "rotation": page.rotation,
        "skew": page.skew,
        "source": page.source,
    }


def pair(item: tuple) -> dict:
    k, page_id, path = item
    res = {}
    for mode in ("a", "b") if k % 2 == 0 else ("b", "a"):
        _MODE[0] = mode
        t0 = time.perf_counter()
        try:
            with _CTX(mode):
                out = _out(_read(path))
        except Exception as e:  # noqa: BLE001
            out = {"error": f"{type(e).__name__}: {e}"}
        res[mode] = (1000 * (time.perf_counter() - t0), out)
    (ms_a, a), (ms_b, b) = res["a"], res["b"]
    differ = [key for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)]
    r = {"page_id": page_id, "ms_a": ms_a, "ms_b": ms_b, "source": b.get("source", "error"), "differ": differ}
    if differ:
        r["a"], r["b"] = a, b
    return r


def cpus(spec: str) -> list[int]:
    out = []
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main():
    import multiprocessing as mp

    from omni_parse_bench import data

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default=data.DEFAULT_DATASET)
    ap.add_argument("--models", default=None)
    ap.add_argument("--mode", choices=["auto", "ocr", "text"], default="auto")
    ap.add_argument("--no-math", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--pin", default="", help="CPUs, e.g. 16-31 or 0,2,4: worker k on the k-th")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="every k-th page (spread over the benchmark)")
    ap.add_argument("--pdf-only", action="store_true", help="only the PDF pages")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    _, pages = data.read_manifest(a.dataset)
    items = [(p.page_id, str(p.doc_path)) for p in pages if not a.pdf_only or p.doc_path.suffix == ".pdf"]
    items = [(k, *it) for k, it in enumerate(items[:: max(1, a.every)][: a.limit or None])]
    args = {"variant": a.variant, "models": a.models, "mode": a.mode, "no_math": a.no_math, "pin": cpus(a.pin)}
    ctx = mp.get_context("spawn")
    rows = []
    t0 = time.time()
    with ctx.Pool(a.procs, initializer=init, initargs=(args, ctx.Value("i", 0))) as pool, open(
        a.out / "pairs.jsonl", "w", encoding="utf-8"
    ) as f:
        for n, r in enumerate(pool.imap_unordered(pair, items, chunksize=2), 1):
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                print(f"{n}/{len(items)} pages, {time.time() - t0:.0f} s", flush=True)
    by: dict = {}
    for r in rows:
        by.setdefault("all", []).append(r)
        by.setdefault(r["source"], []).append(r)
    summary = {"variant": a.variant, "pages": len(rows), "differ": sum(1 for r in rows if r["differ"])}
    for g, rs in by.items():
        sa, sb = sum(r["ms_a"] for r in rs), sum(r["ms_b"] for r in rs)
        summary[g] = {
            "pages": len(rs),
            "differ": sum(1 for r in rs if r["differ"]),
            "ms_a_mean": round(sa / len(rs), 1),
            "ms_b_mean": round(sb / len(rs), 1),
            "speedup_mean": round(sa / max(sb, 1e-9), 3),
            "speedup_median": round(statistics.median(r["ms_a"] / max(r["ms_b"], 1e-9) for r in rs), 3),
        }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    for r in [r for r in rows if r["differ"]][:20]:
        print(f"  differs: {r['page_id']} ({', '.join(r['differ'])})")


if __name__ == "__main__":
    main()
