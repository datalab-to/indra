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
