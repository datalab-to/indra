"""ONNX Runtime sessions that give a GPU arena's free memory back after every run.

ONNX Runtime's CUDA provider keeps the largest allocation each session ever
needed (its BFC arena never shrinks on its own), and the line batches change
shape with every page, so a process grows until the device is full.  119 mixed
pages on an A100 (onnxruntime-gpu 1.19.2): 0.57 GB after loading, 2.1 GB after
the run and held for good; in a service with 8 workers on a 16 GB T4, one worker
reached 5.2 GB and pages failed with CUDA OOM.  With
``memory.enable_memory_arena_shrinkage`` on each run the same 119 pages idle at
0.66 GB (peak 1.7 GB while a page runs), at the same speed (4.7 vs 4.5 pages/s).
The CPU arena is off instead (``enable_cpu_mem_arena``), which this does not
touch.  PR #23 by wuxuedaifu (Apache-2.0).
"""

from __future__ import annotations


def gpu_run_options(providers: list):
    """RunOptions for sessions on a GPU: give the arena's free memory back after
    every run.  None on CPU.  The device is the provider's ``device_id`` (0 by
    default)."""
    for p in providers:
        name, opts = (p, {}) if isinstance(p, str) else (p[0], p[1] or {})
        if name == "CUDAExecutionProvider":
            import onnxruntime as ort

            ro = ort.RunOptions()
            ro.add_run_config_entry(
                "memory.enable_memory_arena_shrinkage", f"gpu:{opts.get('device_id', 0)}"
            )
            return ro
    return None


class _ShrinkingSession:
    """An InferenceSession whose runs carry the arena-shrinkage RunOptions."""

    def __init__(self, sess, run_options):
        self._sess, self._ro = sess, run_options

    def run(self, output_names, input_feed, run_options=None):
        return self._sess.run(output_names, input_feed, run_options or self._ro)

    def __getattr__(self, name):
        return getattr(self._sess, name)


def session(path, so, providers):
    """ort.InferenceSession(path, so, providers=providers); on a GPU, wrapped so
    that every run gives the arena's free memory back."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), so, providers=providers)
    ro = gpu_run_options(providers)
    return sess if ro is None else _ShrinkingSession(sess, ro)
