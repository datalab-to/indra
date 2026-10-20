"""Two builds of Indra OCR on the same Omni Parse Bench pages, interleaved.

    python bench/ab_builds.py --out runs/ab_builds \\
        --a "PR 12=/envs/a/bin/python;/src/a" --b "PR 13=/envs/b/bin/python;/src/b" \\
        --cpus 0,2,4,6,8,10 --threads 1 --mode auto

A build is a Python interpreter (its own indra-ocr-native, say) and the source
folder put first on its path.  Two builds of the native package cannot share
a process, so each reads a block of pages in its own worker process: the
pages are split into one stream per CPU group, and each stream reads its pages
block by block, A then B, then B then A, so drift of the machine (heat, other
load) falls on both alike.  Each worker loads the models and reads its block's
first page once before timing.

Per page: the time of each build (what bench/opb_indra.py measures: the PDF
opened and the page converted) and whether the outputs agree (markdown, line
text and boxes, blocks, rotation and skew; not confidences).  Writes
pairs.jsonl and summary.json.  --threads N gives each worker N threads
(indra_ocr.load(threads=N)) on a group of N CPUs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


def worker(args) -> None:
    """Read the pages in args.pages (one path per line) and print, per page,
    {"i", "ms", "hash", "source"} as JSON."""
    import psutil

    psutil.Process().cpu_affinity([int(c) for c in args.cpus.split(",")])
    import cv2
    import pypdfium2 as pdfium

    import indra_ocr
    from indra_ocr import native

    cv2.setNumThreads(args.threads)
    conv = indra_ocr.load(threads=args.threads, mode=args.mode, furniture=True)
    paths = Path(args.pages).read_text(encoding="utf-8").splitlines()

    def read(path):
        if path.lower().endswith(".pdf"):
            doc = pdfium.PdfDocument(path)
            try:
                return conv.pdf_page(path, doc, 0)
            finally:
                doc.close()
        return conv.ocr(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

    read(paths[0])  # warm-up, not timed
    print(json.dumps({"status": native.status()}), flush=True)
    for i, path in enumerate(paths):
        t0 = time.perf_counter()
        page = read(path)
        ms = 1000 * (time.perf_counter() - t0)
        out = [
            page.markdown,
            [(ln.box, ln.text) for ln in page.lines],
            page.blocks,
            page.rotation,
            page.skew,
        ]
        h = hashlib.md5(json.dumps(out, ensure_ascii=False).encode()).hexdigest()
        print(
            json.dumps({"i": i, "ms": ms, "hash": h, "source": page.source}), flush=True
        )


def run_block(build: dict, cpus: list[int], block: list, args, tmp: Path) -> list:
    pages = tmp / f"block_{os.getpid()}_{threading.get_ident()}.txt"
    pages.write_text("\n".join(path for _, path in block), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": build["src"], "PYTHONUTF8": "1"}
    cmd = [
        build["python"],
        str(Path(__file__).resolve()),
        "--worker",
        "--pages",
        str(pages),
    ]
    cmd += ["--cpus", ",".join(map(str, cpus)), "--threads", str(args.threads)]
    cmd += ["--mode", args.mode]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8")
    rows = [
        json.loads(line) for line in res.stdout.splitlines() if line.startswith("{")
    ]
    status = [r["status"] for r in rows if "status" in r]
    rows = [r for r in rows if "i" in r]
    if res.returncode or len(rows) != len(block):
        raise RuntimeError(f"{build['label']} failed on a block:\n{res.stderr[-3000:]}")
    build.setdefault("status", status[0] if status else "?")
    return [{**r, "page_id": block[r["i"]][0]} for r in rows]


def stats(rows: list) -> dict:
    a = [r["ms_a"] for r in rows]
    b = [r["ms_b"] for r in rows]
    ratio = sorted(x / max(y, 1e-3) for x, y in zip(a, b))
    return {
        "pages": len(rows),
        "a_median_ms": round(statistics.median(a), 1),
        "b_median_ms": round(statistics.median(b), 1),
        "a_mean_ms": round(statistics.mean(a), 1),
        "b_mean_ms": round(statistics.mean(b), 1),
        "speedup_of_means": round(statistics.mean(a) / statistics.mean(b), 4),
        "speedup_median_page": round(statistics.median(ratio), 4),
        "speedup_p10_page": round(ratio[len(ratio) // 10], 4),
        "speedup_p90_page": round(ratio[9 * len(ratio) // 10], 4),
        "same_output": sum(r["same"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--pages", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--a", help='"label=python;source folder"')
    ap.add_argument("--b", help='"label=python;source folder"')
    ap.add_argument("--cpus", default="0", help="comma-separated CPUs to use")
    ap.add_argument("--threads", type=int, default=1, help="threads per worker")
    ap.add_argument("--mode", choices=["auto", "ocr", "text"], default="auto")
    ap.add_argument("--n", type=int, default=0, help="pages, spread evenly (0: all)")
    ap.add_argument("--ids", type=Path, help="read only these page ids (one per line)")
    ap.add_argument("--block", type=int, default=50, help="pages per worker process")
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    if args.worker:
        return worker(args)

    from omni_parse_bench import benchmark

    dataset = args.dataset or benchmark.DEFAULT_DATASET
    root, pages, _ = benchmark.load(dataset)
    if args.ids:
        wanted = set(args.ids.read_text(encoding="utf-8").split())
        pages = [p for p in pages if p.page_id in wanted]
        if not pages:
            raise SystemExit(f"no page of {args.ids} is in {dataset}")
    if args.n:
        pages = pages[:: max(1, len(pages) // args.n)][: args.n]
    items = [
        (p.page_id, str(benchmark.page_file(dataset, root, p.doc_path))) for p in pages
    ]
    builds = []
    for spec in (args.a, args.b):
        label, rest = spec.split("=", 1)
        python, src = rest.split(";", 1)
        builds.append({"label": label, "python": python, "src": src})
    cpus = [int(c) for c in args.cpus.split(",")]
    groups = [cpus[k : k + args.threads] for k in range(0, len(cpus), args.threads)]
    groups = [g for g in groups if len(g) == args.threads]
    streams = [items[k :: len(groups)] for k in range(len(groups))]
    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out / "tmp"
    tmp.mkdir(exist_ok=True)
    results: dict = {0: {}, 1: {}}
    lock = threading.Lock()
    errors: list = []

    def stream(g: list[int], mine: list) -> None:
        try:
            blocks = [mine[k : k + args.block] for k in range(0, len(mine), args.block)]
            for n, block in enumerate(blocks):
                for which in (0, 1) if n % 2 == 0 else (1, 0):
                    rows = run_block(builds[which], g, block, args, tmp)
                    with lock:
                        for r in rows:
                            results[which][r["page_id"]] = r
                        done = len(results[0]) + len(results[1])
                    print(f"{done}/{2 * len(items)} page reads", flush=True)
        except Exception as e:  # noqa: BLE001  (reported after the other streams)
            errors.append(e)

    t0 = time.time()
    ts = [threading.Thread(target=stream, args=(g, s)) for g, s in zip(groups, streams)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errors:
        raise errors[0]
    rows = []
    for pid, _ in items:
        a, b = results[0][pid], results[1][pid]
        rows.append(
            {
                "page_id": pid,
                "source": a["source"],
                "ms_a": round(a["ms"], 1),
                "ms_b": round(b["ms"], 1),
                "same": a["hash"] == b["hash"],
            }
        )
    with open(args.out / "pairs.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {
        "a": {k: builds[0][k] for k in ("label", "status")},
        "b": {k: builds[1][k] for k in ("label", "status")},
        "mode": args.mode,
        "threads": args.threads,
        "cpu_groups": groups,
        "minutes": round((time.time() - t0) / 60, 1),
        "all": stats(rows),
    }
    for src in sorted({r["source"] for r in rows}):
        summary[src] = stats([r for r in rows if r["source"] == src])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
