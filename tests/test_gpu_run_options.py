"""gpu_run_options: arena shrinkage on a GPU session's runs, nothing on CPU (PR #23)."""

from indra_ocr.ort_session import _ShrinkingSession, gpu_run_options


def test_cpu_has_no_run_options():
    assert gpu_run_options(["CPUExecutionProvider"]) is None


def test_cuda_provider_shrinks_its_device_arena():
    ro = gpu_run_options(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert ro.get_run_config_entry("memory.enable_memory_arena_shrinkage") == "gpu:0"
    ro = gpu_run_options([("CUDAExecutionProvider", {"device_id": 2}), "CPUExecutionProvider"])
    assert ro.get_run_config_entry("memory.enable_memory_arena_shrinkage") == "gpu:2"


def test_wrapped_session_passes_its_run_options():
    seen = []

    class Sess:
        def run(self, names, feed, run_options=None):
            seen.append(run_options)
            return [feed["x"]]

        def get_inputs(self):
            return ["x"]

    s = _ShrinkingSession(Sess(), "RO")
    assert s.run(None, {"x": 1}) == [1] and seen == ["RO"]
    assert s.get_inputs() == ["x"]


def test_buckets():
    from indra_ocr.ort_session import count_bucket, width_bucket

    assert [count_bucket(b) for b in (1, 2, 3, 5, 33, 64, 65, 130)] == [1, 2, 4, 8, 64, 64, 128, 192]
    assert [width_bucket(w) for w in (8, 32, 40, 640, 4096, 8192, 8200)] == [32, 32, 48, 768, 4096, 8192, 10240]


def test_bucketed_session_pads_and_cuts_back():
    import numpy as np

    from indra_ocr.ort_session import _BucketedSession

    class Out:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape

    class Sess:
        """A line model: ids [B, T] with T = W / 4, p [B, 4]; records the padded feed."""

        feeds: list = []

        def run(self, names, feed, run_options=None):
            Sess.feeds.append(feed)
            x = feed["x"]
            b, w = x.shape[0], x.shape[3]
            ids = np.tile(np.arange(w // 4), (b, 1)) + 100 * np.arange(b)[:, None]
            return [ids, np.zeros((b, 4), np.float32) + feed["widths"][:, None]]

        def get_outputs(self):
            return [Out("ids", ["B", "T"]), Out("p", ["B", 4])]

    s = _BucketedSession(Sess(), "m.onnx", None, ["CUDAExecutionProvider"], "RO")
    x = np.ones((3, 1, 32, 600), np.float32)
    ids, p = s.run(None, {"x": x, "widths": np.array([600, 520, 400])})
    fed = Sess.feeds[-1]
    assert fed["x"].shape == (4, 1, 32, 768) and fed["x"][:3, :, :, 600:].sum() == 0
    assert fed["widths"].tolist() == [600, 520, 400, 8]
    assert ids.shape == (3, 150) and ids[2, 149] == 200 + 149
    assert p.shape == (3, 4) and p[:, 0].tolist() == [600, 520, 400]


def test_cuda_provider_gets_tf32_off_unless_set():
    from indra_ocr.ort_session import fp32_providers

    assert fp32_providers(["CUDAExecutionProvider", "CPUExecutionProvider"]) == [
        ("CUDAExecutionProvider", {"use_tf32": "0"}),
        "CPUExecutionProvider",
    ]
    mine = [("CUDAExecutionProvider", {"device_id": 1, "use_tf32": "1"})]
    assert fp32_providers(mine) == [("CUDAExecutionProvider", {"device_id": 1, "use_tf32": "1"})]
    assert fp32_providers(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
