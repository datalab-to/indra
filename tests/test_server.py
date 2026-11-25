"""indra-ocr-serve: its routes, with a stand-in Converter (no models) and a thread pool."""

import base64
import io
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

from indra_ocr import server
from indra_ocr.document import Converter
from indra_ocr.pipeline import Line, Page


class FakeConverter:
    _safe = staticmethod(Converter._safe)

    def page_scale(self, page):
        return 2.0

    def pdf_page(self, path, doc, index, password=None):
        return Page([Line([0, 0, 10, 10], f"page {index}", 0.9)], {"total_ms": 1.0}, markdown=f"# page {index}",
                    source="text")

    def ocr(self, gray):
        return Page([], {"total_ms": 1.0}, markdown="", source="ocr")


def pdf_bytes(pages: int) -> bytes:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(200, 100)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture()
def url(monkeypatch):
    monkeypatch.setattr(server, "_converter", lambda s: FakeConverter())
    server._docs.clear()
    srv = server.make_server("127.0.0.1", 0, ThreadPoolExecutor(2), 2, key="secret")
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def call(url, path, data=None, key="secret", ctype="application/pdf"):
    req = urllib.request.Request(url + path, data=data, method="POST" if data is not None else "GET")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_served_and_routes(url):
    assert call(url, "/health") == (200, {"ok": True})
    code, served = call(url, "/served")
    assert code == 200 and served["weights"] == "datalab-to/indra-ocr"
    assert call(url, "/nowhere")[0] == 404


def test_a_token_is_needed(url):
    assert call(url, "/convert", pdf_bytes(1), key=None)[0] == 401
    assert call(url, "/convert", pdf_bytes(1), key="wrong")[0] == 401


def test_every_page_of_a_pdf_in_order(url):
    code, out = call(url, "/convert", pdf_bytes(3))
    assert code == 200
    assert [p["index"] for p in out["pages"]] == [0, 1, 2]
    assert out["pages"][1]["markdown"] == "# page 1" and out["pages"][1]["lines"][0]["text"] == "page 1"
    assert out["pages"][0]["image_size"] == [400, 200]  # 200 x 100 pt at page_scale 2


def test_max_pages_and_bad_parameters(url):
    code, out = call(url, "/convert?max_pages=2&mode=ocr&math=0", pdf_bytes(5))
    assert code == 200 and len(out["pages"]) == 2
    assert call(url, "/convert?mode=fast", pdf_bytes(1))[0] == 400
    assert call(url, "/convert?colour=1", pdf_bytes(1))[0] == 400


def test_a_file_that_is_not_a_pdf_or_image(url):
    code, out = call(url, "/convert", b"not a document at all", ctype="application/octet-stream")
    assert code == 422 and "readable image" in out["error"]


def test_parse_is_the_benchmark_adapter_contract(url):
    body = {"page": base64.b64encode(pdf_bytes(2)).decode(), "suffix": ".pdf",
            "settings": {"mode": "auto", "math": True, "dpi": 200, "furniture": True}}
    code, out = call(url, "/parse", json.dumps(body).encode(), ctype="application/json")
    assert code == 200 and out["markdown"] == "# page 0" and out["image_size"] == [400, 200]
    assert set(out) >= {"text", "blocks", "source", "rotation", "skew", "timings", "vendor", "served"}
    body["settings"].pop("dpi")
    assert call(url, "/parse", json.dumps(body).encode(), ctype="application/json")[0] == 400


def test_metrics_count_the_pages(url):
    call(url, "/convert", pdf_bytes(2))
    code, m = call(url, "/metrics")
    assert code == 200 and m["pages"] == 2 and m["requests"] == 1 and m["workers"] == 2


def test_query_settings():
    s, n, pw = server.query_settings({"math": ["false"], "dpi": ["300"], "max_pages": ["4"]})
    assert s["math"] is False and s["dpi"] == 300.0 and n == 4 and pw is None
    with pytest.raises(ValueError):
        server.query_settings({"dpi": ["9000"]})
