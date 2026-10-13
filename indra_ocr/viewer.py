"""Local viewer: drop in a PDF or an image, see what Indra OCR reads.

    indra-ocr-viewer                      # models from the Hugging Face repo
    open http://localhost:8765

Each page shows its lines, layout blocks and reading order over the page (each
can be shown or hidden), the rendered markdown, the markdown source, the lines
with their confidences, the JSON the CLI writes, where the text came from (the
PDF text layer, OCR, or both) and the time of each stage.  The pipeline options
(text layer or OCR, math, page headers and footers, reading order, OCR of text
inside images, dpi) are set in the page and apply when the document is read
again.  Pages are read one at a time and appear as they finish.  With
--tesseract (needs the tesseract binary and pytesseract) the page can also run
Tesseract on each page.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import __version__
from .cli import add_source_args, source_options
from .document import Converter
from .models import model_dir
from .pipeline import OrtOCR

MAX_SIDE = 1600  # longest side of the page image sent to the browser
TIMING_KEYS = (
    "text_ms",
    "det_ms",
    "rec_ms",
    "ocr_ms",
    "math_ms",
    "order_ms",
    "total_ms",
)
MODES = ("auto", "ocr", "text")


def decode_image(data: bytes):
    import cv2

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("not a PDF or a readable image")
    yield 0, img, 0.0


class Engine:
    """Holds uploaded documents and reads one page per request, with the options
    the page sends, so the browser shows each page as soon as it is done."""

    def __init__(
        self,
        model_path: Path | None,
        threads: int,
        tesseract: bool,
        tess_lang: str,
        defaults: dict,
    ):
        self.model_path, self.threads, self.defaults = model_path, threads, defaults
        self.ocrs: dict[tuple, OrtOCR] = {}  # (math, furniture) -> recognizer
        self.tmp = tempfile.TemporaryDirectory(prefix="indra_ocr_viewer_")
        self.lock = threading.Lock()
        self.docs: dict[str, bytes] = {}
        self.tesseract = None
        if tesseract:
            import pytesseract

            os.environ["OMP_THREAD_LIMIT"] = str(threads)
            self.tesseract = (pytesseract, tess_lang)
        self._ocr(bool(defaults["math"]), bool(defaults["furniture"]))

    def _ocr(self, math: bool, furniture: bool) -> OrtOCR:
        """The recognizer for these options; the math models download on first use."""
        key = (math, furniture)
        if key not in self.ocrs:
            onnx = model_dir(self.model_path, math=math)
            ocr = OrtOCR(
                onnx,
                threads=self.threads,
                rec_int8=False,
                math=math,
                furniture=furniture,
            )
            ocr(
                np.full((200, 200), 255, np.uint8)
            )  # warm-up: first-call allocation is not timed
            self.ocrs[key] = ocr
        return self.ocrs[key]

    def upload(self, data: bytes) -> dict:
        import uuid

        if data[:5] == b"%PDF-":
            import pypdfium2 as pdfium

            doc = pdfium.PdfDocument(data)
            n = len(doc)
            doc.close()
        else:
            next(decode_image(data))  # raises on an unreadable image
            n = 1
        did = uuid.uuid4().hex[:12]
        self.docs[did] = data
        if data[:5] == b"%PDF-":  # the converter opens the PDF from a file
            with open(os.path.join(self.tmp.name, did + ".pdf"), "wb") as f:
                f.write(data)
        while len(self.docs) > 5:  # keep the last few uploads only
            old = next(iter(self.docs))
            self.docs.pop(old)
            try:
                os.remove(os.path.join(self.tmp.name, old + ".pdf"))
            except OSError:
                pass
        return {"id": did, "pages": n}

    def page(self, did: str, n: int, opts: dict) -> dict:
        import cv2

        data = self.docs[did]
        mode = opts.get("mode", "auto")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        math, furniture = bool(opts.get("math")), bool(opts.get("furniture"))
        dpi = float(opts.get("dpi") or 200.0)
        with self.lock:
            ocr = self._ocr(math, furniture)
            conv = Converter(
                ocr,
                dpi=dpi,
                mode=mode,
                pdf_order=not opts.get("model_order"),
                image_ocr=opts.get("image_ocr", True),
            )
            if data[:5] == b"%PDF-":
                import pypdfium2 as pdfium

                path = os.path.join(self.tmp.name, did + ".pdf")
                doc = pdfium.PdfDocument(path)
                try:
                    t0 = time.perf_counter()
                    # the converter's own scale, so the boxes match the picture
                    arr = (
                        doc[n]
                        .render(scale=conv.page_scale(doc[n]), grayscale=True)
                        .to_numpy()
                    )
                    render_ms = 1000 * (time.perf_counter() - t0)
                    page = conv.pdf_page(path, doc, n)
                finally:
                    doc.close()
                gray = np.ascontiguousarray(arr[..., 0] if arr.ndim == 3 else arr)
            else:
                _, gray, render_ms = next(decode_image(data))
                page = ocr(gray)
            if page.rotation:  # the boxes are in the upright page: show it
                gray = np.ascontiguousarray(np.rot90(gray, -(page.rotation // 90)))
            h, w = gray.shape
            s = min(1.0, MAX_SIDE / max(h, w))
            small = (
                cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                if s < 1
                else gray
            )
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
            scaled = lambda b: [round(v * s, 1) for v in b[:4]]  # noqa: E731
            rec = {
                "page": n + 1,
                "w": small.shape[1],
                "h": small.shape[0],
                "img": "data:image/jpeg;base64,"
                + base64.b64encode(jpg.tobytes()).decode(),
                "lines": [
                    {"box": scaled(ln.box), "text": ln.text, "conf": ln.conf}
                    for ln in page.lines
                ],
                "blocks": [
                    {
                        "label": b["label"],
                        "box": scaled(b["bbox"]),
                        "text": b.get("text", ""),
                    }
                    for b in page.blocks
                ],
                "markdown": page.markdown if page.markdown is not None else page.text,
                "source": page.source,
                "note": page.note,
                "rotation": page.rotation,
                "skew": page.skew,
                "json": {
                    "page": n,
                    "source": page.source,
                    "rotation": page.rotation,
                    "skew": page.skew,
                    "note": page.note,
                    "lines": [asdict(ln) for ln in page.lines],
                    "markdown": page.markdown,
                    "blocks": page.blocks,
                    "timings": page.timings,
                },
                "timings": {
                    "render_ms": round(render_ms, 1),
                    **{k: 0.0 for k in TIMING_KEYS},
                    **{k: round(v, 1) for k, v in page.timings.items()},
                },
            }
            if opts.get("tesseract") and self.tesseract:
                pt, lang = self.tesseract
                t0 = time.perf_counter()
                try:
                    rec["tesseract"] = pt.image_to_string(
                        gray, lang=lang, config="--psm 3"
                    )
                except Exception as e:  # noqa: BLE001  (show the reason in the page)
                    rec["tesseract"] = f"[tesseract failed: {e}]"
                rec["timings"]["tesseract_ms"] = round(
                    1000 * (time.perf_counter() - t0), 1
                )
        t = rec["timings"]
        print(
            f"page {n + 1}: {page.source}, {len(page.lines)} lines, {len(page.blocks)} blocks, "
            f"{t['total_ms']:.0f} ms ({mode}{', math' if math else ''})",
            flush=True,
        )
        return rec


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(
                    200,
                    PAGE.replace("__LOGO__", LOGO).encode(),
                    "text/html; charset=utf-8",
                )
            elif self.path == "/info":
                info = {
                    "tesseract": engine.tesseract is not None,
                    "version": __version__,
                    "defaults": engine.defaults,
                }
                self._send(200, json.dumps(info).encode())
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n)
            try:
                if self.path == "/upload":
                    res = engine.upload(body)
                elif self.path == "/page":
                    q = json.loads(body)
                    res = engine.page(q["id"], int(q["n"]), q.get("opts", {}))
                else:
                    return self._send(404, b"not found", "text/plain")
                self._send(200, json.dumps(res).encode())
            except Exception as e:  # noqa: BLE001  (report the error to the page)
                print(f"error: {e}", flush=True)
                self._send(400, json.dumps({"error": str(e)}).encode())

    return Handler


# The Datalab logo (assets/datalab-logo.png), for the page header.
LOGO = (
    "iVBORw0KGgoAAAANSUhEUgAAAwAAAAMACAYAAACTgQCOAAAX4ElEQVR4Ae3dwVFcx9rH4dZXWkMAQALCexfsLSWA"
    "CUAl7Y0JwFIAiABcUgCIACwH4LH2lhNABGAS0KfGl3uvL0gzwMw5p/v/PFUuubzyBtG/6bffefDpswIAAET4vwIA"
    "AMQQAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQR"
    "AAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAA"
    "EEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQA"
    "AABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABA"
    "EAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEA"
    "AABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABB"
    "BAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAA"
    "AAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQR"
    "AAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAA"
    "EEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQA"
    "AABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABA"
    "EAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEA"
    "AABBHhYAuKOzs7Mym80K1+3u7pbNzc0CMDUCAIA7q4f/g4ODwnXHx8dlf3+/AEyNESAAAAgiAAAAIIgAAACAIAIA"
    "AACCCAAAAAgiAAAAIIgAAACAIAIAAACCCAAAAAgiAAAAIIgAAACAIAIAAACCCAAAAAgiAAAAIIgAAACAIAIAAACC"
    "CAAAAAgiAAAAIIgAAACAIAIAAACCCAAAAAgiAAAAIIgAAACAIAIAAACCCAAAAAgiAAAAIIgAAACAIAIAAACCCAAA"
    "AAgiAAAAIIgAAACAIAIAAACCCAAAAAgiAAAAIIgAAACAIAIAAACCCAAAAAgiAAAAIIgAAACAIAIAAACCCAAAAAgi"
    "AAAAIIgAAACAIAIAAACCCAAAAAgiAAAAIIgAAACAIAIAAACCCAAAAAgiAAAAIIgAAACAIAIAAACCCAAAAAgiAAAA"
    "IIgAAACAIAIAAACCCAAAAAgiAAAAIIgAAACAIAIAAACCCAAAAAgiAAAAIIgAAACAIAIAAACCCAAAAAgiAAAAIIgA"
    "AACAIAIAAACCCAAAAAjysAAw197eXuG6ra2tcnp6Wrhue3u7AEyRAABYwGw2K1xXA2BnZ6cA0A4jQAAAEEQAAABA"
    "EAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEA"
    "AABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABB"
    "BAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAA"
    "AAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQR"
    "AAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAAEEQAAABAEAEAAABBBAAAAAQRAAAA"
    "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"
    "GwAivHr1qhwdHRWYKvP+VN4DtMsbAFoiAIhRf7HWEKiPhGEqrlbYGvnhSv1Sw/rlhrRFANASI0DEqAet+jj4999/"
    "L0+ePCkwNvP+3MR7AGDV3AAQ6+pGoH7aVv8dhlQPeHXmG76k3gLUv59ogxsAWiIA4LN3796VX375RQwwCPP+LMJ7"
    "gLYIAFoiAOB/1F+6f/75Z/njjz/Kx48f/x0E9d/hPtbW1i7H0Iz8sKj6oYQvNmxDHS+FVggAAAAI4hEwAAAEEQAA"
    "ABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBE"
    "AAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAA"
    "QBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBAB"
    "AAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAA"
    "QQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQA"
    "AAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAE"
    "EQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAA"
    "ABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBE"
    "AAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAA"
    "QBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBAB"
    "AAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAA"
    "QQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQAAAAEEQAAABBEAAAAQBABAAAAQQQA"
    "AAAEEQAAABDk/wH98CgW8PO2cwAAAABJRU5ErkJggg=="
)

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Indra OCR viewer · Datalab</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
 --font-body:"Geist",ui-sans-serif,system-ui,sans-serif;--font-mono:"Geist Mono",ui-monospace,"SF Mono",Menlo,monospace;
 --page:oklch(98.5% .004 263);--surface:oklch(96.5% .006 263);--surface-2:oklch(94% .007 263);
 --rule:oklch(86% .008 263);--rule-strong:oklch(78% .01 263);
 --ink:oklch(18% .012 263);--ink-2:oklch(36% .012 263);--ink-3:oklch(48% .014 263);--ink-4:oklch(70% .014 263);
 --accent:oklch(38% .24 263);--accent-soft:oklch(96% .04 263);--accent-strong:oklch(30% .26 263);--accent-ink:oklch(99% 0 0);
 --ok:oklch(58% .15 145);--warn:oklch(72% .16 75);--err:oklch(56% .2 25);--focus:oklch(38% .24 263 / .35);
 --hairline:1px;--ease:cubic-bezier(.25,1,.5,1)}
@media (min-resolution:2dppx){:root{--hairline:.5px}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 var(--font-body);-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.mono{font-family:var(--font-mono)}
.label{font:500 10.5px/1 var(--font-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
/* top bar */
.top{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:18px;height:56px;padding:0 24px;background:color-mix(in oklch,var(--page) 88%,transparent);backdrop-filter:blur(8px);border-bottom:var(--hairline) solid var(--rule)}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;letter-spacing:-.01em}
.mark{width:30px;height:30px;margin:0 -4px}
.brand .sep{color:var(--ink-4);font-weight:400}.brand .prod{font-weight:500}
.ver{font:11px var(--font-mono);color:var(--ink-3);border:var(--hairline) solid var(--rule);border-radius:999px;padding:2px 8px}
.top .sp{flex:1}.top nav{display:flex;gap:18px;font-size:13px}.top nav a{color:var(--ink-2)}
.cta{background:var(--accent);color:var(--accent-ink)!important;padding:7px 12px;border-radius:6px;font-weight:500}.cta:hover{background:var(--accent-strong);text-decoration:none}
/* layout */
.wrap{display:grid;grid-template-columns:300px minmax(0,1fr);min-height:calc(100vh - 56px)}
aside{border-right:var(--hairline) solid var(--rule);padding:20px;display:flex;flex-direction:column;gap:22px;position:sticky;top:56px;height:calc(100vh - 56px);overflow:auto;background:var(--page)}
main{padding:24px;display:flex;flex-direction:column;gap:24px;min-width:0}
@media(max-width:900px){.wrap{grid-template-columns:1fr}aside{position:static;height:auto;border-right:0;border-bottom:var(--hairline) solid var(--rule)}}
/* drop zone */
#drop{border:1px dashed var(--rule-strong);border-radius:10px;padding:18px 14px;text-align:center;cursor:pointer;background:var(--surface);transition:border-color var(--ease) .12s,background .12s}
#drop:hover,#drop.over{border-color:var(--accent);background:var(--accent-soft)}
#drop b{display:block;font-weight:500;margin-bottom:2px}#drop span{color:var(--ink-3);font-size:12.5px}
#fname{margin-top:8px;font:12px var(--font-mono);color:var(--ink-2);word-break:break-all}
/* controls */
.group{display:flex;flex-direction:column;gap:10px}
.seg{display:grid;grid-template-columns:repeat(3,1fr);border:var(--hairline) solid var(--rule-strong);border-radius:7px;overflow:hidden}
.seg button{border:0;background:transparent;padding:7px 4px;font:500 12.5px var(--font-body);color:var(--ink-2);cursor:pointer}
.seg button+button{border-left:var(--hairline) solid var(--rule)}.seg button.on{background:var(--ink);color:var(--page)}
.hint{font-size:12px;color:var(--ink-3);line-height:1.4}
.sw{display:flex;align-items:center;justify-content:space-between;gap:10px;font-size:13px;color:var(--ink-2);cursor:pointer;user-select:none}
.sw input{appearance:none;width:30px;height:18px;border-radius:999px;background:var(--rule-strong);position:relative;cursor:pointer;transition:background .15s;flex:none}
.sw input::after{content:"";position:absolute;top:2px;left:2px;width:14px;height:14px;border-radius:50%;background:#fff;transition:transform .15s var(--ease)}
.sw input:checked{background:var(--accent)}.sw input:checked::after{transform:translateX(12px)}
.sw input:focus-visible{outline:3px solid var(--focus)}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.field{display:flex;flex-direction:column;gap:5px}.field select,.field input{font:13px var(--font-body);padding:6px 8px;border:var(--hairline) solid var(--rule-strong);border-radius:6px;background:#fff;color:var(--ink)}
.btn{border:0;background:var(--accent);color:var(--accent-ink);font:500 13px var(--font-body);padding:9px 12px;border-radius:7px;cursor:pointer}
.btn:hover{background:var(--accent-strong)}.btn:disabled{background:var(--rule-strong);cursor:default}
.dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:0}
#stats{font:12px/1.7 var(--font-mono);color:var(--ink-2)}#stats b{color:var(--ink);font-weight:500}
#status{font:12px var(--font-mono);color:var(--ink-3);min-height:18px}
.callout{border:var(--hairline) solid var(--rule);border-radius:10px;padding:12px;background:var(--surface);font-size:12.5px;color:var(--ink-2)}
/* empty state */
.empty{border:1px dashed var(--rule-strong);border-radius:14px;padding:64px 24px;text-align:center;color:var(--ink-3)}
.empty h2{font-weight:500;color:var(--ink);letter-spacing:-.015em;margin:0 0 6px;font-size:22px}
/* page card */
.card{background:#fff;border:var(--hairline) solid var(--rule);border-radius:12px;overflow:hidden}
.ch{display:flex;align-items:center;flex-wrap:wrap;gap:10px 16px;padding:12px 16px;border-bottom:var(--hairline) solid var(--rule)}
.ch h3{margin:0;font-size:15px;font-weight:600;letter-spacing:-.01em}
.chip{font:500 10.5px/1 var(--font-mono);letter-spacing:.06em;text-transform:uppercase;padding:5px 7px;border-radius:5px;background:var(--surface-2);color:var(--ink-2)}
.chip.text{background:oklch(95% .04 145);color:oklch(38% .12 145)}.chip.ocr{background:var(--accent-soft);color:var(--accent)}.chip.hybrid{background:oklch(95% .05 75);color:oklch(42% .12 75)}.chip.error{background:oklch(95% .05 25);color:var(--err)}
.meta{font:12px var(--font-mono);color:var(--ink-3)}.ch .sp{flex:1}
.tbar{display:flex;height:6px;border-radius:3px;overflow:hidden;background:var(--surface-2);width:180px}.tbar i{display:block;height:100%}
.body{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);min-height:420px}
@media(max-width:1200px){.body{grid-template-columns:1fr}}
.pageview{padding:16px;background:var(--surface);border-right:var(--hairline) solid var(--rule)}
svg.pg{width:100%;height:auto;display:block;background:#fff;box-shadow:0 1px 2px oklch(18% .012 263 / .08),0 0 0 var(--hairline) var(--rule)}
.right{display:flex;flex-direction:column;min-width:0}
.tabs{display:flex;gap:2px;padding:10px 12px 0;border-bottom:var(--hairline) solid var(--rule)}
.tabs button{border:0;background:transparent;padding:8px 10px;font:500 12.5px var(--font-body);color:var(--ink-3);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px}
.tabs button.on{color:var(--ink);border-bottom-color:var(--accent)}
.pane{flex:1;overflow:auto;max-height:78vh;padding:14px 16px}
.pane.src{white-space:pre-wrap;font:12.5px/1.6 var(--font-mono);color:var(--ink-2)}
.preview{font-size:14px;line-height:1.6}.preview h1,.preview h2,.preview h3{letter-spacing:-.015em;line-height:1.25;margin:1.1em 0 .4em}.preview h1{font-size:20px}.preview h2{font-size:17px}.preview h3{font-size:15px}
.preview p{margin:.5em 0}.preview ul,.preview ol{padding-left:1.4em}.preview table{border-collapse:collapse;margin:10px 0;font-size:12.5px}
.preview td,.preview th{border:var(--hairline) solid var(--rule-strong);padding:4px 7px;vertical-align:top}.preview th{background:var(--surface)}
.preview code.m{font:12.5px var(--font-mono);background:var(--accent-soft);color:var(--accent-strong);padding:1px 4px;border-radius:4px}
.ln-row{display:flex;gap:10px;padding:3px 6px;border-radius:5px;font-size:13px;cursor:default}.ln-row:hover,.ln-row.on{background:var(--accent-soft)}
.ln-row .k{font:11px var(--font-mono);color:var(--ink-4);min-width:28px;text-align:right;padding-top:2px}.ln-row .t{flex:1;min-width:0;word-break:break-word}
.ln-row .c{font:11px var(--font-mono);color:var(--ink-3);padding-top:2px}.ln-row .c.low{color:var(--err)}
/* overlays */
rect.ln{fill:transparent;stroke:oklch(55% .15 263);stroke-width:1;vector-effect:non-scaling-stroke;opacity:.55}
rect.ln.on{fill:oklch(55% .2 263 / .18);stroke-width:2.5;opacity:1}
rect.bk{fill:transparent;stroke-width:2;vector-effect:non-scaling-stroke}rect.bk.on{stroke-width:3.5}
text.bl{font-family:var(--font-mono);font-weight:600;paint-order:stroke;stroke:#fff}
path.ord{fill:none;stroke:oklch(60% .2 30);stroke-width:1.4;vector-effect:non-scaling-stroke;opacity:.8}
circle.on-n{fill:oklch(60% .2 30)}text.n{font-family:var(--font-mono);font-weight:600;fill:#fff;text-anchor:middle;dominant-baseline:central}
.hide-lines rect.ln,.hide-blocks rect.bk,.hide-blocks text.bl,.hide-labels text.bl,.hide-order .ord,.hide-order circle.on-n,.hide-order text.n{display:none}
footer{padding:18px 24px;border-top:var(--hairline) solid var(--rule);display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--ink-3)}
</style></head><body>
<div class="top">
 <div class="brand"><img class="mark" src="data:image/png;base64,__LOGO__" alt="Datalab">Datalab<span class="sep">/</span><span class="prod">Indra OCR</span></div><span class="ver" id="ver"></span>
 <div class="sp"></div>
 <nav><a href="https://github.com/datalab-to/indra" target="_blank" rel="noopener">GitHub</a><a href="https://huggingface.co/datalab-to/indra-ocr" target="_blank" rel="noopener">Models</a><a class="cta" href="https://www.datalab.to/?utm_source=indra-viewer" target="_blank" rel="noopener">Try Chandra on Datalab</a></nav>
</div>
<div class="wrap">
<aside>
 <div><div id="drop"><b>Drop a PDF or an image</b><span>or click to choose a file</span><input id="file" type="file" accept=".pdf,image/*" hidden></div><div id="fname"></div></div>
 <div class="group"><div class="label">Source</div>
  <div class="seg" id="mode"><button data-v="auto">Auto</button><button data-v="ocr">OCR</button><button data-v="text">Text layer</button></div>
  <div class="hint" id="modehint"></div></div>
 <div class="group"><div class="label">Pipeline</div>
  <label class="sw">Math as LaTeX<input type="checkbox" id="math"></label>
  <label class="sw">Keep headers and footers<input type="checkbox" id="furniture"></label>
  <label class="sw">PDF reading order<input type="checkbox" id="pdforder" checked></label>
  <label class="sw">OCR text inside images<input type="checkbox" id="imageocr" checked></label>
  <label class="sw" id="tesslab" hidden>Compare with Tesseract<input type="checkbox" id="tess"></label>
  <div class="row2"><div class="field"><span class="label">DPI</span><select id="dpi"><option>150</option><option selected>200</option><option>300</option></select></div>
  <div class="field"><span class="label">Max pages</span><input id="maxp" type="number" min="1" value="20"></div></div>
  <button class="btn" id="rerun" disabled>Read again with these options</button></div>
 <div class="group"><div class="label">Show</div>
  <label class="sw">Lines<input type="checkbox" class="disp" data-c="lines" checked></label>
  <label class="sw">Layout blocks<input type="checkbox" class="disp" data-c="blocks" checked></label>
  <label class="sw">Block labels<input type="checkbox" class="disp" data-c="labels" checked></label>
  <label class="sw">Reading order<input type="checkbox" class="disp" data-c="order"></label>
  <div id="legend" class="hint"></div></div>
 <div class="group"><div class="label">Run</div><div id="status"></div><div id="stats"></div></div>
 <div class="callout">Need handwriting, math and tables at the highest accuracy? <a href="https://www.datalab.to/?utm_source=indra-viewer" target="_blank" rel="noopener">Chandra on Datalab</a> reads them, with $5 in free credits to start.</div>
</aside>
<main id="out"><div class="empty"><h2>Read a document</h2>Drop a PDF or an image on the left. Pages appear as they finish, with lines, layout blocks and reading order drawn over each page.</div></main>
</div>
<footer><span>Indra OCR · fast CPU OCR by <a href="https://www.datalab.to" target="_blank" rel="noopener">Datalab</a></span><span class="mono">Runs locally: your files stay on this machine.</span></footer>
<script>
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]);
const BLOCK={"Text":"oklch(55% .02 263)","Section-Header":"oklch(45% .22 263)","Page-Header":"oklch(65% .03 263)","Page-Footer":"oklch(65% .03 263)","List-Group":"oklch(58% .15 145)","Table":"oklch(55% .18 320)","Caption":"oklch(62% .14 200)","Footnote":"oklch(60% .12 60)","Table-Of-Contents":"oklch(55% .12 180)","Form":"oklch(50% .16 25)"};
const STAGES=[["text_ms","Text layer","oklch(55% .15 300)"],["det_ms","Detect","oklch(45% .22 263)"],["rec_ms","Recognize","oklch(58% .15 145)"],["ocr_ms","OCR in images","oklch(62% .12 200)"],["math_ms","Math","oklch(55% .18 320)"],["order_ms","Order","oklch(68% .15 75)"]];
const HINT={auto:"Reads the PDF text layer where it passes the checks, and OCRs the rest.",ocr:"Ignores the text layer and OCRs every page.",text:"Uses the text layer even when the checks fail."};
let mode="auto",doc=null,stats=[],runId=0;
function setMode(v){mode=v;$$('#mode button').forEach(b=>b.classList.toggle('on',b.dataset.v===v));$('#modehint').textContent=HINT[v]}
$$('#mode button').forEach(b=>b.onclick=()=>setMode(b.dataset.v));
$('#legend').innerHTML=Object.entries(BLOCK).filter(([k])=>k!=="Page-Footer").map(([k,c])=>`<span style="white-space:nowrap;margin-right:8px"><span class="dot" style="background:${c}"></span>${k==='Page-Header'?'Header/Footer':k.replace(/-/g,' ')}</span>`).join(' ');
function applyDisplay(){const o=$('#out');$$('.disp').forEach(i=>o.classList.toggle('hide-'+i.dataset.c,!i.checked))}
$$('.disp').forEach(i=>i.onchange=applyDisplay);applyDisplay();
const drop=$('#drop'),file=$('#file');
drop.onclick=()=>file.click();file.onchange=()=>file.files[0]&&start(file.files[0]);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');e.dataTransfer.files[0]&&start(e.dataTransfer.files[0])};
$('#rerun').onclick=()=>doc&&run();
fetch('/info').then(r=>r.json()).then(i=>{$('#ver').textContent='v'+i.version;if(i.tesseract)$('#tesslab').hidden=false;const d=i.defaults||{};
 setMode(d.mode||'auto');$('#math').checked=!!d.math;$('#furniture').checked=!!d.furniture;$('#pdforder').checked=d.pdf_order!==false;$('#imageocr').checked=d.image_ocr!==false;
 if(d.dpi)$('#dpi').value=String(Math.round(d.dpi));if(d.max_pages)$('#maxp').value=d.max_pages});
setMode('auto');
async function start(f){$('#fname').textContent=f.name+' · '+(f.size/1024).toFixed(0)+' KB';$('#status').textContent='Uploading ...';
 const u=await (await fetch('/upload',{method:'POST',body:await f.arrayBuffer()})).json();
 if(u.error){$('#status').textContent='Error: '+u.error;return}doc={...u,name:f.name};$('#rerun').disabled=false;run()}
function opts(){return{mode,math:$('#math').checked,furniture:$('#furniture').checked,model_order:!$('#pdforder').checked,image_ocr:$('#imageocr').checked,dpi:+$('#dpi').value,tesseract:$('#tess').checked}}
async function run(){const id=++runId,o=opts(),N=Math.min(doc.pages,Math.max(1,+$('#maxp').value||1));
 $('#out').innerHTML='';stats=[];summary();const t0=performance.now();
 for(let n=0;n<N;n++){if(id!==runId)return;$('#status').textContent=`Reading page ${n+1} of ${N}`+(doc.pages>N?` (first ${N} of ${doc.pages})`:'')+(o.math&&n===0?' · math models load on first use':'')+' ...';
  const p=await (await fetch('/page',{method:'POST',body:JSON.stringify({id:doc.id,n,opts:o})})).json();if(id!==runId)return;
  if(p.error){$('#status').textContent=`Error on page ${n+1}: ${p.error}`;return}render(p);stats.push(p.timings);summary()}
 $('#status').textContent=`${N} page${N>1?'s':''} in ${(performance.now()-t0).toFixed(0)} ms (including display)`}
function summary(){const el=$('#stats');if(!stats.length){el.innerHTML='';return}const n=stats.length,s=k=>stats.reduce((a,t)=>a+(t[k]||0),0),tot=s('total_ms');
 let h=`<b>${(tot/n).toFixed(0)} ms</b> per page · <b>${(1000*n/tot).toFixed(1)}</b> pages/s<br>`+STAGES.filter(([k])=>s(k)>0).map(([k,l])=>`${l} ${(s(k)/n).toFixed(0)} ms`).join(' · ');
 const tt=stats.filter(t=>t.tesseract_ms!=null);if(tt.length){const tm=tt.reduce((a,t)=>a+t.tesseract_ms,0)/tt.length;h+=`<br>Tesseract ${tm.toFixed(0)} ms/page (${(tm/(tot/n)).toFixed(1)}x)`}
 el.innerHTML=h}
function md(src){// a small markdown renderer: headings, lists, paragraphs, HTML tables, $math$
 const lines=(src||'').split('\n'),out=[];let list=null,para=[];
 const inline=t=>esc(t).replace(/\$\$([^$]+)\$\$|\$([^$\n]+)\$/g,(m,a,b)=>`<code class="m">${a||b}</code>`).replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>');
 const flush=()=>{if(para.length){out.push('<p>'+para.map(inline).join(' ')+'</p>');para=[]}if(list){out.push(`</${list}>`);list=null}};
 for(const raw of lines){const l=raw.trimEnd();
  if(/^<table/.test(l)||/^<\/?t(r|d|h|able|head|body)/.test(l)){flush();out.push(l.replace(/<(?!\/?(table|thead|tbody|tr|td|th)\b)[^>]*>/g,''));continue}
  const h=l.match(/^(#{1,6})\s+(.*)/);if(h){flush();const k=Math.min(3,h[1].length);out.push(`<h${k}>${inline(h[2])}</h${k}>`);continue}
  const li=l.match(/^\s*(?:[-*•]|(\d+)[.)])\s+(.*)/);if(li){if(para.length){out.push('<p>'+para.map(inline).join(' ')+'</p>');para=[]}const t=li[1]?'ol':'ul';if(list!==t){if(list)out.push(`</${list}>`);out.push(`<${t}>`);list=t}out.push(`<li>${inline(li[2])}</li>`);continue}
  if(!l.trim()){flush();continue}if(list){out.push(`</${list}>`);list=null}para.push(l)}
 flush();return out.join('\n')}
function render(p){const c=document.createElement('div');c.className='card';const t=p.timings,tot=t.total_ms||1;
 const u=Math.max(1,p.w/640);// page units per screen pixel, about: keeps labels and markers readable
 const src={text:'Text layer',ocr:'OCR',hybrid:'Text layer + OCR',error:'Error'}[p.source]||p.source;
 const bar=STAGES.map(([k,l,col])=>`<i style="width:${100*(t[k]||0)/tot}%;background:${col}" title="${l} ${t[k]||0} ms"></i>`).join('');
 const extra=(p.rotation?` · turned ${p.rotation}°`:'')+(p.skew?` · straightened ${(+p.skew).toFixed(1)}°`:'');
 const blocks=p.blocks.map((b,k)=>{const col=BLOCK[b.label]||'oklch(55% .02 263)',[x0,y0,x1,y1]=b.box;return `<rect class="bk" data-b="${k}" x="${x0-3}" y="${y0-3}" width="${x1-x0+6}" height="${y1-y0+6}" stroke="${col}"/><text class="bl" x="${x0-2}" y="${Math.max(11*u,y0-5*u)}" fill="${col}" font-size="${11*u}" stroke-width="${3*u}">${esc(b.label)}</text>`}).join('');
 const lines=p.lines.map((l,k)=>`<rect class="ln" data-k="${k}" x="${l.box[0]}" y="${l.box[1]}" width="${l.box[2]-l.box[0]}" height="${l.box[3]-l.box[1]}"/>`).join('');
 const cen=p.lines.map(l=>[(l.box[0]+l.box[2])/2,(l.box[1]+l.box[3])/2]);
 const ord=cen.length>1?`<path class="ord" d="M${cen.map(c=>c.join(',')).join(' L')}"/>`:'';
 const nums=cen.map((c,k)=>`<circle class="on-n" cx="${c[0]}" cy="${c[1]}" r="${7*u}"/><text class="n" x="${c[0]}" y="${c[1]}" font-size="${8*u}">${k+1}</text>`).join('');
 const rows=p.lines.map((l,k)=>`<div class="ln-row" data-k="${k}"><span class="k">${k+1}</span><span class="t">${esc(l.text)||'<span style=\"color:var(--ink-4)\">(no text)</span>'}</span><span class="c ${l.conf<.8?'low':''}">${l.conf.toFixed(2)}</span></div>`).join('');
 const tabs=['Preview','Markdown','Lines','JSON'].concat(p.tesseract!=null?['Tesseract']:[]);
 c.innerHTML=`<div class="ch"><h3>Page ${p.page}</h3><span class="chip ${p.source}">${src}</span><span class="meta">${p.lines.length} lines · ${p.blocks.length} blocks${extra}</span><span class="sp"></span><div class="tbar">${bar}</div><span class="meta">${t.total_ms} ms</span></div>
 ${p.note?`<div class="meta" style="padding:6px 16px;border-bottom:var(--hairline) solid var(--rule)">${esc(p.note)}</div>`:''}
 <div class="body"><div class="pageview"><svg class="pg" viewBox="0 0 ${p.w} ${p.h}"><image href="${p.img}" width="${p.w}" height="${p.h}"/>${blocks}${lines}${ord}${nums}</svg></div>
 <div class="right"><div class="tabs">${tabs.map((n,i)=>`<button data-tab="${n}" class="${i?'':'on'}">${n}</button>`).join('')}</div>
 <div class="pane preview" data-pane="Preview">${md(p.markdown)}</div>
 <div class="pane src" data-pane="Markdown" hidden>${esc(p.markdown||'')}</div>
 <div class="pane" data-pane="Lines" hidden>${rows}</div>
 <div class="pane src" data-pane="JSON" hidden>${esc(JSON.stringify(p.json,null,1))}</div>
 ${p.tesseract!=null?`<div class="pane src" data-pane="Tesseract" hidden>${esc(p.tesseract)}</div>`:''}</div></div>`;
 c.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{c.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x===b));c.querySelectorAll('.pane').forEach(x=>x.hidden=x.dataset.pane!==b.dataset.tab)});
 const hl=k=>c.querySelectorAll('[data-k]').forEach(el=>el.classList.toggle('on',el.dataset.k==k));
 c.querySelectorAll('.ln-row').forEach(el=>{el.onmouseenter=()=>hl(el.dataset.k);el.onmouseleave=()=>hl(-1)});
 c.querySelectorAll('rect.ln').forEach(el=>{el.onmouseenter=()=>{hl(el.dataset.k);const r=c.querySelector(`.ln-row[data-k="${el.dataset.k}"]`);r&&r.scrollIntoView({block:'nearest'})};el.onmouseleave=()=>hl(-1)});
 c.querySelectorAll('rect.bk').forEach(el=>{el.onmouseenter=()=>el.classList.add('on');el.onmouseleave=()=>el.classList.remove('on')});
 $('#out').appendChild(c)}
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="folder with the models (default: download from Hugging Face)",
    )
    add_source_args(ap)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--dpi", type=float, default=200.0, help="initial PDF render resolution"
    )
    ap.add_argument(
        "--max-pages", type=int, default=20, help="initial page limit per document"
    )
    ap.add_argument(
        "--keep-furniture",
        action="store_true",
        help="initially keep page headers and footers",
    )
    ap.add_argument(
        "--tesseract", action="store_true", help="allow comparing with Tesseract"
    )
    ap.add_argument("--tesseract-lang", default="eng")
    ap.add_argument("--math", action="store_true", help="initially write math as LaTeX")
    a = ap.parse_args()
    src = source_options(a)
    defaults = {
        "mode": src["mode"],
        "pdf_order": src["pdf_order"],
        "image_ocr": src["image_ocr"],
        "math": a.math,
        "furniture": a.keep_furniture,
        "dpi": a.dpi,
        "max_pages": a.max_pages,
    }
    engine = Engine(a.model_dir, a.threads, a.tesseract, a.tesseract_lang, defaults)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(engine))
    print(
        f"Indra OCR viewer on http://localhost:{a.port}  ({a.threads} threads)",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
