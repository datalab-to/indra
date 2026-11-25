"""indra-ocr-serve: an HTTP server that keeps the models loaded.

    indra-ocr-serve --port 8001                       # one worker per usable core
    indra-ocr-serve --workers 16 --device cuda --gpus 0,1

Every page is read by one single-threaded worker process of a pool, as
``convert_many`` reads them; the pages of one file are read in parallel.  Each
worker loads the models once per settings, so the first page with new settings
in each worker pays the load.  Routes:

    GET  /health    200 {"ok": true} once the workers are up
    GET  /served    {"weights", "revision"}: the models the package names
    GET  /metrics   pages read, pages failed, page latency (mean, p50, p90, ms),
                    workers, uptime
    POST /convert   the file (a PDF or an image) as the request body; query
                    parameters mode (auto, ocr, text), math, furniture, words
                    (0 or 1), dpi, max_pages (0: all), password
                    -> {"pages": [{"index", "markdown", "text", "blocks", "lines",
                        "source", "note", "rotation", "skew", "vertical",
                        "timings", "image_size"}], "ms"}
    POST /parse     the Omni Parse Bench Indra OCR adapter's request: {"page":
                    base64 file, "suffix", "settings": {"mode", "math", "dpi",
                    "furniture"}} -> the first page's markdown, text, blocks,
                    source, note, rotation, skew, timings, image_size, vendor and
                    served (the benchmark's own server answers the same)

A bearer token (--key, or INDRA_SERVE_KEY) is required on every POST when set.
The server listens on 127.0.0.1 unless --host says otherwise.
"""

from __future__ import annotations

import argparse
import base64
import importlib.metadata
import json
import math
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DEFAULTS = {"mode": "auto", "math": True, "dpi": 200.0, "furniture": False, "words": False}
MAX_BODY = 200 << 20  # bytes in one request (--max-mb)

_converters: dict = {}  # in a worker: settings -> Converter
_docs: dict = {}  # in a worker: path -> open PdfDocument (its last file)
_load_kwargs: dict = {}  # in a worker: what every load() gets (device or providers)


def _init(counter, device: str, gpus: list[int], pin: list[int]) -> None:
    """A worker process: its CPU and GPU, one thread for OpenCV."""
    with counter.get_lock():
        k = counter.value
        counter.value += 1
    if pin:
        try:
            os.sched_setaffinity(0, [pin[k % len(pin)]])
        except (AttributeError, OSError):  # macOS, Windows
            pass
    if device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k % len(gpus)]) if gpus else "0"
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        import inspect

        import indra_ocr

        if "device" in inspect.signature(indra_ocr.load).parameters:
            _load_kwargs["device"] = "cuda"
        else:
            _load_kwargs["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    import cv2

    cv2.setNumThreads(1)


def _converter(s: dict):
    import indra_ocr

    key = tuple(sorted(s.items()))
    if key not in _converters:
        _converters[key] = indra_ocr.load(
            threads=1,
            dpi=float(s["dpi"]),
            mode=s["mode"],
            furniture=bool(s["furniture"]),
            math=bool(s["math"]),
            words=bool(s["words"]),
            **_load_kwargs,
        )
    return _converters[key]


def _image_size(conv, path: str, index: int) -> list[int]:
    """The page as the Converter reads it, [width, height] in pixels, before
    any turn: a PDF page at the Converter's page_scale, an image as it is."""
    from . import pipeline

    if not pipeline.is_pdf(path):
        h, w = pipeline.read_images(Path(path))[index].shape[:2]
        return [w, h]
    page = _doc(path)[index]
    k = conv.page_scale(page)
    w, h = page.get_size()
    return [round(w * k), round(h * k)]


def _doc(path: str, password: str | None = None):
    import pypdfium2 as pdfium

    if path not in _docs:
        for d in _docs.values():
            d.close()
        _docs.clear()
        _docs[path] = pdfium.PdfDocument(path, password=password)
    return _docs[path]


def page_json(index: int, page, size: list[int]) -> dict:
    lines = []
    for ln in page.lines:
        d = {"box": ln.box, "text": ln.text, "conf": ln.conf}
        if getattr(ln, "words", None) is not None:
            d["words"] = [{"text": w.text, "box": w.box} for w in ln.words]
        lines.append(d)
    return {
        "index": index,
        "markdown": page.markdown,
        "text": page.text,
        "blocks": page.blocks,
        "lines": lines,
        "source": page.source,
        "note": page.note,
        "rotation": page.rotation,
        "skew": page.skew,
        "vertical": page.vertical,
        "timings": page.timings,
        "image_size": size,
    }


def read_unit(path: str, index: int | None, s: dict, password: str | None = None) -> list[dict]:
    """In a worker: page ``index`` of a PDF, or every page of an image file
    (index None).  Each page as page_json."""
    from . import pipeline

    conv = _converter(s)
    t0 = time.perf_counter()
    if index is None:
        out = []
        for i, gray in enumerate(pipeline.read_images(Path(path))):
            page = conv._safe(lambda g=gray: conv.ocr(g))
            out.append(page_json(i, page, [gray.shape[1], gray.shape[0]]))
        return out
    doc = _doc(path, password)
    page = conv._safe(lambda: conv.pdf_page(path, doc, index, password))
    d = page_json(index, page, _image_size(conv, path, index))
    d["timings"] = {**d["timings"], "worker_ms": round(1000 * (time.perf_counter() - t0), 1)}
    return [d]


def page_count(path: str, password: str | None) -> int | None:
    """Pages of a PDF, None for an image file (read whole by one worker)."""
    from . import pipeline

    if not pipeline.is_pdf(path):
        return None
    import pypdfium2 as pdfium

    from . import InputError

    with _PDFIUM:  # pdfium is not thread-safe; the server's threads share it
        try:
            doc = pdfium.PdfDocument(path, password=password)
        except pdfium.PdfiumError as e:
            locked = "password" in str(e).lower()
            raise InputError(
                "a password-protected PDF (pass password=)" if locked else f"cannot open the PDF ({e})"
            ) from e
        try:
            return len(doc)
        finally:
            doc.close()


_PDFIUM = threading.Lock()


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.pages = self.failed = self.requests = 0
        self.ms: list[float] = []
        self.started = time.time()

    def add(self, pages: list[dict], ms: float) -> None:
        with self.lock:
            self.requests += 1
            self.pages += len(pages)
            self.failed += sum(1 for p in pages if p["source"] == "error")
            self.ms.extend(p["timings"].get("worker_ms", ms / max(1, len(pages))) for p in pages)
            del self.ms[:-10000]  # the last 10,000 pages

    def json(self, workers: int) -> dict:
        with self.lock:
            ms = sorted(self.ms)

            def q(f):
                return round(ms[min(len(ms) - 1, int(f * (len(ms) - 1)))], 1) if ms else None

            return {
                "requests": self.requests,
                "pages": self.pages,
                "pages_failed": self.failed,
                "page_ms": {"mean": round(sum(ms) / len(ms), 1) if ms else None, "p50": q(0.5), "p90": q(0.9)},
                "workers": workers,
                "uptime_s": round(time.time() - self.started, 1),
            }


def _flag(v: str) -> bool:
    if v.lower() in ("1", "true", "yes", "on"):
        return True
    if v.lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"not a yes/no value: {v!r}")


def query_settings(q: dict) -> tuple[dict, int, str | None]:
    """(settings, max_pages, password) from /convert's query parameters."""
    s = dict(DEFAULTS)
    one = {k: v[-1] for k, v in q.items()}
    unknown = set(one) - {*DEFAULTS, "max_pages", "password"}
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    if "mode" in one:
        if one["mode"] not in ("auto", "ocr", "text"):
            raise ValueError("mode must be auto, ocr or text")
        s["mode"] = one["mode"]
    for k in ("math", "furniture", "words"):
        if k in one:
            s[k] = _flag(one[k])
    if "dpi" in one:
        s["dpi"] = float(one["dpi"])
        if not 36 <= s["dpi"] <= 600 or math.isnan(s["dpi"]):
            raise ValueError("dpi must be between 36 and 600")
    n = int(one.get("max_pages", 0))
    if n < 0:
        raise ValueError("max_pages must be 0 (all pages) or more")
    return s, n, one.get("password")


class Handler(BaseHTTPRequestHandler):
    pool: Executor
    workers: int
    key: str | None = None
    vendor: dict = {}
    served: dict = {}
    metrics: Metrics
    tmp: Path
    max_body: int = MAX_BODY
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # noqa: A002  (quiet: /metrics has the counts)
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            return self._send(200, {"ok": True})
        if path == "/served":
            return self._send(200, self.served)
        if path == "/metrics":
            return self._send(200, self.metrics.json(self.workers))
        self._send(404, {"error": f"no route {path}"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path not in ("/convert", "/parse"):
            return self._send(404, {"error": f"no route {u.path}"})
        if self.key and self.headers.get("Authorization") != f"Bearer {self.key}":
            return self._send(401, {"error": "a bearer token is needed (Authorization: Bearer ...)"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return self._send(400, {"error": "an empty request: send the file as the body"})
        if n > self.max_body:
            return self._send(413, {"error": f"the request is over {self.max_body >> 20} MB"})
        body = self.rfile.read(n)
        try:
            if u.path == "/parse":
                return self._parse(body)
            return self._convert(body, parse_qs(u.query))
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return self._send(400, {"error": f"{type(e).__name__}: {e}"[:500]})
        except BrokenProcessPool as e:
            # a worker died (out of memory, say): every later page would fail
            self._send(500, {"error": f"indra-ocr failed: {type(e).__name__}: {e}"[:500]})
            os._exit(1)
        except Exception as e:  # noqa: BLE001  (every failure goes back as an answer)
            self._send(500, {"error": f"indra-ocr failed: {type(e).__name__}: {e}"[:500]})

    def _read(self, data: bytes, suffix: str, s: dict, max_pages: int, password: str | None) -> list[dict]:
        fd, name = tempfile.mkstemp(suffix=suffix, dir=self.tmp)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            n = page_count(name, password)
            if n is None:
                futs = [self.pool.submit(read_unit, name, None, s)]
            else:
                n = min(n, max_pages) if max_pages else n
                futs = [self.pool.submit(read_unit, name, i, s, password) for i in range(n)]
            pages = [p for f in futs for p in f.result()]
            return pages[:max_pages] if max_pages else pages
        finally:
            try:
                os.unlink(name)
            except OSError:  # still open in a worker (Windows): removed with the folder at exit
                pass

    def _convert(self, body: bytes, q: dict):
        s, max_pages, password = query_settings(q)
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        suffix = ".pdf" if ctype == "application/pdf" or body[:5] == b"%PDF-" else ".img"
        t0 = time.perf_counter()
        from . import InputError

        try:
            pages = self._read(body, suffix, s, max_pages, password)
        except InputError as e:
            return self._send(422, {"error": str(e)[:500]})
        ms = 1000 * (time.perf_counter() - t0)
        self.metrics.add(pages, ms)
        self._send(200, {"pages": pages, "ms": round(ms, 1)})

    def _parse(self, body: bytes):
        req = json.loads(body)
        page = base64.b64decode(req["page"])
        got = req["settings"]
        if set(got) != {"mode", "math", "dpi", "furniture"}:
            raise ValueError("settings must name exactly dpi, furniture, math and mode")
        s = {**DEFAULTS, **{k: got[k] for k in ("mode", "math", "dpi", "furniture")}}
        t0 = time.perf_counter()
        pages = self._read(page, str(req.get("suffix") or ".pdf"), s, 1, None)
        ms = 1000 * (time.perf_counter() - t0)
        self.metrics.add(pages[:1], ms)
        p = pages[0]
        if p["source"] == "error":
            return self._send(500, {"error": f"indra-ocr failed: the page couldn't be read: {p['note']}"[:500]})
        keep = ("markdown", "text", "blocks", "source", "note", "rotation", "skew", "timings", "image_size")
        self._send(200, {**{k: p[k] for k in keep}, "vendor": self.vendor, "served": self.served})

    def _send(self, status: int, body: dict):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(
    host: str, port: int, pool: Executor, workers: int, key: str | None = None, max_body: int = MAX_BODY
) -> ThreadingHTTPServer:
    """The HTTP server on (host, port), reading pages with ``pool`` (workers:
    how many it runs)."""
    from . import models

    tmp = Path(tempfile.mkdtemp(prefix="indra-serve-"))
    attrs = {
        "pool": pool,
        "workers": workers,
        "key": key,
        "metrics": Metrics(),
        "tmp": tmp,
        "max_body": max_body,
        "vendor": {"indra-ocr": _version(), "models": f"{models.HF_REPO}@{models.HF_REVISION}"},
        "served": {"weights": models.HF_REPO, "revision": models.HF_REVISION},
    }
    handler = type("IndraHandler", (Handler,), attrs)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.indra_tmp = tmp
    return server


def _version() -> str:
    try:
        return importlib.metadata.version("indra-ocr")
    except importlib.metadata.PackageNotFoundError:
        from . import __version__

        return __version__


def _cpus(spec: str) -> list[int]:
    out = []
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main(argv: list[str] | None = None) -> None:
    from .parallel import usable_cores

    ap = argparse.ArgumentParser(
        prog="indra-ocr-serve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--workers", type=int, default=0, help="worker processes (default: one per usable core)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--gpus", default="0", help="with --device cuda: GPU ids, worker k on the k-th")
    ap.add_argument("--pin", default="", help="CPUs, e.g. 0-15 or 0,2,4: worker k on the k-th")
    ap.add_argument("--key", default=os.environ.get("INDRA_SERVE_KEY"), help="bearer token (INDRA_SERVE_KEY)")
    ap.add_argument("--max-mb", type=int, default=MAX_BODY >> 20, help="largest request, MB")
    ap.add_argument("--no-math-models", action="store_true", help="do not fetch the math models at start")
    a = ap.parse_args(argv)
    from . import model_dir

    # every model file fetched before the first request, so no page waits for a download
    model_dir(math=not a.no_math_models)
    workers = a.workers or usable_cores()
    gpus = [int(g) for g in a.gpus.split(",") if g != ""]
    counter = multiprocessing.get_context("spawn").Value("i", 0)
    pool = ProcessPoolExecutor(
        workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init,
        initargs=(counter, a.device, gpus, _cpus(a.pin)),
    )
    server = make_server(a.host, a.port, pool, workers, key=a.key, max_body=a.max_mb << 20)
    print(f"indra-ocr-serve on http://{a.host}:{server.server_port} ({workers} workers, {a.device})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        pool.shutdown(cancel_futures=True)
        shutil.rmtree(server.indra_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
