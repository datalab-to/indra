"""indra-ocr-cuda against ONNX Runtime on the CPU (needs the package, a GPU and the models):
the recognizers' CTC outputs from logits left in VRAM, and the math decoder with its caches on the
device against the host decoder, token for token.

    TENSORRT_ROOT=... pip install ./cuda && pytest cuda/tests
"""

import numpy as np
import pytest

ic = pytest.importorskip("indra_ocr_cuda")
if ic.device_count() < 1:
    pytest.skip("no GPU", allow_module_level=True)

import onnxruntime as ort  # noqa: E402

import indra_ocr  # noqa: E402
from indra_ocr import gpu  # noqa: E402
from indra_ocr.ctc import lm_inputs, peak_prob  # noqa: E402


@pytest.fixture(scope="module")
def model_dir():
    return indra_ocr.model_dir(math=True)


def _batch(rng, B, H, W):
    x = np.zeros((B, 1, H, W), np.float32)
    widths = (rng.integers(max(1, W // 24), W // 8 + 1, B) * 8).astype(np.int64)
    widths[0] = W
    for i, w in enumerate(widths):
        x[i, 0, :, :w] = rng.random((H, w), dtype=np.float32) ** 3
    return x, widths


@pytest.mark.parametrize("name", ["rec.onnx", "rec_cjk.onnx", "rec_world.onnx"])
def test_recognizer_outputs_match_onnx_runtime(model_dir, tmp_path, monkeypatch, name):
    monkeypatch.setenv("INDRA_TRT_CACHE", str(tmp_path))
    path = str(model_dir / name)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = gpu.session_for(path, so, ["CPUExecutionProvider"])
    assert isinstance(s, gpu.TrtRecognizer)
    ref = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    H = ref.get_inputs()[0].shape[2]
    x, widths = _batch(np.random.default_rng(0), 12, H, 512)
    ids, conf, lengths, blank, topv, topi = s.run(None, {"x": x, "widths": widths})
    rids, rlen, rlog = ref.run(["ids", "lengths", "logits"], {"x": x, "widths": widths})
    assert np.array_equal(lengths, rlen)
    assert (ids == rids).mean() > 0.999
    for k in range(len(x)):
        n = int(rlen[k])
        assert np.abs(conf[k, :n] - peak_prob(rlog[k, :n])).max() < 1e-4
        rb, rv, ri = lm_inputs(rlog[k, :n])
        assert np.abs(blank[k, :n] - rb).max() < 1e-3
        assert (topi[k, :n, 0] == ri[:, 0]).mean() > 0.999


def test_device_decoder_matches_the_host_decoder(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("INDRA_TRT_CACHE", str(tmp_path))
    conv = indra_ocr.load(device="cuda", threads=1)
    reader = conv.ocr.math
    assert reader.on_engine
    from indra_ocr.math import BOS, _HostDecoder

    x, widths = _batch(np.random.default_rng(1), 6, reader.height, 320)
    live = np.array([True, False, True, True, False, True])  # two lines leave after step 5

    def decode(dec):
        # the two decoders share the step engine: each runs to its end alone
        tok, out = np.full(len(x), BOS, np.int64), []
        for t in range(12):
            logits = dec.step(tok, t)
            out.append(logits)
            tok = logits.argmax(-1).astype(np.int64)
            if t == 5:
                dec.take(live)
                tok = tok[live]
        return out

    dev = decode(gpu.DeviceDecoder(reader, x, widths, len(x)))
    mem, mask = reader.enc.run(None, {"x": x, "widths": widths})
    names = reader.inputs["math_dec_init.onnx"]
    ck, cv = reader.init.run(None, {k: v for k, v in {"memory": mem, "mask": mask}.items() if k in names})
    host = decode(_HostDecoder(reader, ck, cv, mask, len(x)))
    for a, b in zip(dev, host):
        assert a.shape == b.shape and np.abs(a - b).max() < 1e-3
