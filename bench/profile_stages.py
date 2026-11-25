"""Where a page's time goes: every ONNX model and every pipeline step, per page.

    pip install 'omni-parse-bench[benchmark]'
    python bench/profile_stages.py --out runs/prof_cpu --pages 300 --pin 2
    python bench/profile_stages.py --out runs/prof_gpu --pages 300 --device cuda

One process reads the pages one at a time (as one worker of bench/opb_indra.py
does), after one warm-up page.  Every InferenceSession.run is timed by its
model file, and the pipeline's own steps by the Page timings it records
(render, text layer, orient, det, rec, math, order, tables).  The pages are
spread evenly over the benchmark (every k-th page).

Writes pages.jsonl (per page: wall ms, source, the page's timings, ms and calls
per model) and summary.json, and prints the share of the wall time per model and
per step, overall and by page source (ocr, text, hybrid).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

_T: dict = defaultdict(float)  # model file -> seconds on this page
_N: dict = defaultdict(int)  # model file -> runs on this page


def patch_ort() -> None:
    import onnxruntime as ort

    run0 = ort.InferenceSession.run

    def run(self, output_names, input_feed, run_options=None):
        name = Path(str(getattr(self, "_model_path", "?"))).name
        t0 = time.perf_counter()
        try:
            return run0(self, output_names, input_feed, run_options)
        finally:
            _T[name] += time.perf_counter() - t0
            _N[name] += 1

    ort.InferenceSession.run = run


def patch_engine() -> None:
    """Time the GPU engine's sessions (indra_ocr/gpu.py) by model file too."""
    from indra_ocr import gpu

    for cls in (gpu.TrtSession, gpu.TrtRecognizer):
        run0 = cls.__dict__["run"]

        def run(self, output_names, input_feed, run_options=None, _run0=run0):
            name = "trt:" + Path(str(self._model_path)).name
            t0 = time.perf_counter()
            try:
                return _run0(self, output_names, input_feed, run_options)
            finally:
                _T[name] += time.perf_counter() - t0
                _N[name] += 1

        cls.run = run


def read(conv, path: str):
    import cv2
    import pypdfium2 as pdfium

    if path.lower().endswith(".pdf"):
        doc = pdfium.PdfDocument(path)
        try:
            return conv.pdf_page(path, doc, 0)
        finally:
            doc.close()
    return conv.ocr(cv2.imread(path, cv2.IMREAD_GRAYSCALE))


def pages_of(dataset: str) -> list:
    """(page id, suite, page file) of every page, in the manifest's order
    (omni-parse-bench 1.8: data.read_manifest on a dataset folder or hf:// path)."""
    from omni_parse_bench import data

    _, pages = data.read_manifest(dataset)
    return [(p.page_id, p.suite, str(p.doc_path)) for p in pages]


def main():
    from omni_parse_bench import data

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pages", type=int, default=300)
    ap.add_argument("--mode", choices=["auto", "ocr", "text"], default="auto")
    ap.add_argument("--dataset", default=data.DEFAULT_DATASET)
    ap.add_argument("--models", default=None, help="model folder (default: download)")
    ap.add_argument("--no-math", action="store_true", help="load(math=False)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument(
        "--device",
        choices=["cpu", "cuda", "engine"],
        default="cpu",
        help="cuda: ONNX Runtime's CUDA provider; engine: load(device='cuda'), the indra-ocr-cuda engine first",
    )
    ap.add_argument("--pin", default="", help="comma-separated CPUs for this process")
    ap.add_argument(
        "--dump", action="store_true", help="also write outputs.jsonl (bench/compare_outputs.py)"
    )
    ap.add_argument(
        "--cprofile",
        action="store_true",
        help="also run the pages under cProfile: cprofile.txt (by own time and by cumulative time)",
    )
    ap.add_argument(
        "--cuda-opts", default="", help="CUDA provider options, e.g. use_tf32=0,device_id=1"
    )
    ap.add_argument(
        "--cpu-buckets",
        action="store_true",
        help="on the CPU, pad line batches as the GPU does (ort_session._BucketedSession): "
        "how much the padding alone changes",
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.pin:
        import os

        os.sched_setaffinity(0, [int(c) for c in a.pin.split(",")])
    import cv2

    cv2.setNumThreads(a.threads)
    if a.device in ("cuda", "engine"):
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # the CUDA and cuDNN libraries of the nvidia-* wheels
    patch_ort()
    import indra_ocr

    if a.cpu_buckets:
        from indra_ocr import ort_session

        ort_session._on_cuda = lambda providers: True
        ort_session._preload = lambda: None
        ort_session.gpu_run_options = lambda providers: None
    opts = dict(kv.split("=", 1) for kv in a.cuda_opts.split(",") if kv)
    providers = (
        [("CUDAExecutionProvider", opts), "CPUExecutionProvider"] if a.device == "cuda" else None
    )
    if a.device == "engine":
        try:
            from indra_ocr import gpu
        except ImportError:
            raise SystemExit("--device engine needs indra_ocr.gpu and the indra-ocr-cuda engine") from None

        if not gpu.available():
            raise SystemExit("indra-ocr-cuda is not installed or sees no GPU")
        patch_engine()
        providers = indra_ocr.gpu_providers("cuda")
    conv = indra_ocr.load(
        a.models,
        threads=a.threads,
        mode=a.mode,
        furniture=True,
        math=not a.no_math,
        providers=providers,
    )
    pages = pages_of(a.dataset)
    step = max(1, len(pages) // a.pages)
    chosen = pages[::step][: a.pages]
    read(conv, chosen[0][2])  # warm-up
    rows = []
    dump = open(a.out / "outputs.jsonl", "w", encoding="utf-8") if a.dump else None
    prof = None
    if a.cprofile:
        import cProfile

        prof = cProfile.Profile()
        prof.enable()
    with open(a.out / "pages.jsonl", "w") as f:
        for page_id, suite, path in chosen:
            _T.clear()
            _N.clear()
            t0 = time.perf_counter()
            page = None
            try:
                page = read(conv, path)
                source, timings = page.source, dict(page.timings)
            except Exception as e:  # noqa: BLE001
                source, timings = f"error: {type(e).__name__}: {e}", {}
            ms = 1000 * (time.perf_counter() - t0)
            if dump is not None:
                out = (
                    {
                        "markdown": page.markdown,
                        "lines": [[ln.box, ln.text, ln.conf] for ln in page.lines],
                        "blocks": page.blocks,
                        "rotation": page.rotation,
                        "skew": page.skew,
                        "source": page.source,
                    }
                    if page is not None
                    else {"error": source}
                )
                dump.write(json.dumps({"page_id": page_id, **out}, ensure_ascii=False) + "\n")
            r = {
                "page_id": page_id,
                "suite": suite,
                "ms": ms,
                "source": source,
                "timings": timings,
                "models": {k: round(1000 * v, 3) for k, v in _T.items()},
                "calls": dict(_N),
            }
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
    if dump is not None:
        dump.close()
    if prof is not None:
        import io
        import pstats

        prof.disable()
        buf = io.StringIO()
        st = pstats.Stats(prof, stream=buf)
        st.sort_stats("tottime").print_stats(60)
        st.sort_stats("cumulative").print_stats(80)
        (a.out / "cprofile.txt").write_text(buf.getvalue())
    summary = summarize(rows)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    show(summary)


def summarize(rows: list) -> dict:
    out = {}
    groups = {"all": rows}
    for r in rows:
        groups.setdefault(r["source"], []).append(r)
    for g, rs in groups.items():
        wall = sum(r["ms"] for r in rs)
        models: dict = defaultdict(float)
        calls: dict = defaultdict(int)
        steps: dict = defaultdict(float)
        for r in rs:
            for k, v in r["models"].items():
                models[k] += v
            for k, v in r["calls"].items():
                calls[k] += v
            for k, v in r["timings"].items():
                if k.endswith("_ms") and isinstance(v, (int, float)):
                    steps[k] += v
        ort_ms = sum(models.values())
        rs_ms = sorted(r["ms"] for r in rs)
        out[g] = {
            "pages": len(rs),
            "wall_ms": round(wall, 1),
            "mean_ms": round(wall / max(1, len(rs)), 1),
            "median_ms": round(rs_ms[len(rs_ms) // 2], 1) if rs_ms else 0.0,
            "pages_per_s": round(1000 * len(rs) / max(wall, 1e-9), 2),
            "ort_share": round(ort_ms / max(wall, 1e-9), 3),
            "models": {
                k: {
                    "ms": round(v, 1),
                    "share": round(v / max(wall, 1e-9), 3),
                    "calls": calls[k],
                    "ms_per_call": round(v / max(1, calls[k]), 3),
                }
                for k, v in sorted(models.items(), key=lambda kv: -kv[1])
            },
            "steps": {
                k: {"ms": round(v, 1), "share": round(v / max(wall, 1e-9), 3)}
                for k, v in sorted(steps.items(), key=lambda kv: -kv[1])
            },
        }
    return out


def show(summary: dict) -> None:
    for g, s in summary.items():
        print(
            f"\n== {g}: {s['pages']} pages, mean {s['mean_ms']} ms, median {s['median_ms']} ms, "
            f"{s['pages_per_s']} pages/s, ONNX Runtime {100 * s['ort_share']:.1f}%"
        )
        for k, m in s["models"].items():
            print(
                f"  {k:28s} {m['ms']:10.1f} ms  {100 * m['share']:5.1f}%  "
                f"{m['calls']:6d} runs  {m['ms_per_call']:8.3f} ms/run"
            )
        print("  steps (inclusive, from Page.timings):")
        for k, m in s["steps"].items():
            print(f"  {k:28s} {m['ms']:10.1f} ms  {100 * m['share']:5.1f}%")


if __name__ == "__main__":
    main()
