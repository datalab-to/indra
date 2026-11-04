"""gpu_run_options: arena shrinkage on a GPU session's runs, nothing on CPU."""

from indra_ocr.pipeline import gpu_run_options


def test_cpu_has_no_run_options():
    assert gpu_run_options(["CPUExecutionProvider"]) is None


def test_cuda_provider_shrinks_its_device_arena():
    ro = gpu_run_options(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert ro.get_run_config_entry("memory.enable_memory_arena_shrinkage") == "gpu:0"
    ro = gpu_run_options([("CUDAExecutionProvider", {"device_id": 2}), "CPUExecutionProvider"])
    assert ro.get_run_config_entry("memory.enable_memory_arena_shrinkage") == "gpu:2"
