"""gpu.py without a GPU: the providers load(device=...) gives, the engine's shapes per model, and the
recognizer session's lazy LM rows."""

import numpy as np
import pytest

import indra_ocr
from indra_ocr import gpu


def test_device_strings(monkeypatch):
    monkeypatch.setattr(gpu, "available", lambda: False)
    assert indra_ocr.gpu_providers("cuda") == [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    assert indra_ocr.gpu_providers("cuda:1")[0] == ("CUDAExecutionProvider", {"device_id": 1})
    monkeypatch.setattr(gpu, "available", lambda: True)
    assert indra_ocr.gpu_providers("cuda")[0] == gpu.ENGINE
    for bad in ("gpu", "cuda:x", "cuda:"):
        if bad == "cuda:":
            continue  # "cuda:" is cuda:0
        with pytest.raises(ValueError):
            indra_ocr.gpu_providers(bad)


def test_device_and_providers_together():
    with pytest.raises(ValueError):
        indra_ocr.load(device="cuda", providers=["CPUExecutionProvider"])


def test_engine_shapes():
    I = gpu._Info  # noqa: E741
    line = gpu._profiles("rec.onnx", [I("x", ["B", 1, 96, "W"]), I("widths", ["B"])])
    assert line["x"][2] == [gpu.LINE_MAX_BATCH, 1, 96, gpu.LINE_MAX_WIDTH]
    assert line["widths"][2] == [gpu.LINE_MAX_BATCH]
    assert gpu._profiles("det.onnx", [I("x", [1, 1, "H", "W"])])["x"][2] == [1, 1, 2048, 2048]
    assert gpu._profiles("orient.onnx", [I("x", ["B", 1, 768, 768])])["x"][0] == [1, 1, 768, 768]
    assert gpu._profiles("order.onnx", [I("feats", [1, "N", 16])]) is None


def test_lm_rows_are_made_once_and_not_after_the_next_run():
    class Model:
        calls: list = []

        def lm_inputs(self, logits, rows, n, k):
            Model.calls.append((rows, n))
            r = len(rows)
            return np.zeros((r, 5)), np.ones((r, 5, k)), np.full((r, 5, k), 7)

    s = gpu.TrtRecognizer.__new__(gpu.TrtRecognizer)
    s._m, s._gen, s._lengths, s._rows = Model(), 1, np.array([5, 3]), {}
    topv = gpu._LM(s, 1, 1)
    topi = gpu._LM(s, 2, 1)
    assert topv[1, :3].shape == (3, 8) and topv[1, :3, 0].tolist() == [1, 1, 1]
    assert topi[1, :3, 0].tolist() == [7, 7, 7]
    assert Model.calls == [([1], [3])]  # one kernel call for the row
    s._gen = 2
    with pytest.raises(RuntimeError):
        topv[0, :5]


def test_engine_feed_keeps_scalars():
    feed = gpu.TrtSession._feed({"pos": np.array(3, np.int64), "x": np.ones((2, 3), np.float32)[:, ::2]})
    assert feed["pos"].shape == () and feed["x"].flags.c_contiguous and feed["x"].shape == (2, 2)
