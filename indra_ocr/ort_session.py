"""ONNX Runtime sessions for the GPU: the arena's free memory given back after
every run, and line batches padded to a few shapes.

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

The CUDA provider also sets up its convolutions again for every input shape it
has not seen recently: a session keeps about 8 shapes warm (onnxruntime-gpu
1.32, cuDNN 9.28, RTX A6000).  ``rec.onnx`` on 16 lines of 480 to 1,100 px runs
in 2.7 to 3.4 ms a batch while 8 shapes or fewer cycle, and in 40 to 43 ms once
12 or more do; the first run of a shape costs 45 to 310 ms.  Line batches have a
new width and line count almost every time, so nearly every recognizer and
classifier run paid that.  The limit is per session (4 sessions cycling 6 shapes
each: 3.9 ms a batch), so a line model on the GPU pads its batch to a width
bucket (WIDTH_BUCKETS) and a power-of-two line count, and keeps one session per
width bucket.  Padding changes a line's numbers a little (the models are not
padding-invariant, on the CPU too), so this is done on the GPU only, where the
numbers differ from the CPU's anyway.
"""

from __future__ import annotations

import bisect

import numpy as np

# Line widths a GPU batch is padded to: about x1.41 apart, multiples of 8.
WIDTH_BUCKETS = (
    32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
)
WIDTH_BEYOND = 2048  # wider batches: padded to a multiple of this
PAD_WIDTH = 8  # the ``widths`` of a padding line

_PRELOADED = False


def _preload() -> None:
    """Load the CUDA and cuDNN libraries of the nvidia-* wheels that
    ``pip install onnxruntime-gpu[cuda,cudnn]`` installs: without this ONNX
    Runtime does not find them ("libcudnn.so: cannot open shared object file")
    unless they are on the library path."""
    global _PRELOADED
    if _PRELOADED:
        return
    _PRELOADED = True
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:  # noqa: BLE001  (the libraries may be on the path already)
            pass


def _on_cuda(providers: list) -> bool:
    return any((p if isinstance(p, str) else p[0]) == "CUDAExecutionProvider" for p in providers)


def fp32_providers(providers: list) -> list:
    """The providers with TF32 off on the CUDA provider, unless the caller set
    ``use_tf32``.  ONNX Runtime runs FP32 convolutions and matrix products in TF32
    on Ampere and later GPUs by default (10 bits of mantissa): on 300
    OmniParseBench pages that changed 0.55% of the characters against the CPU,
    and 0.002% with TF32 off (onnxruntime-gpu 1.32, RTX A6000)."""
    out = []
    for p in providers:
        name, opts = (p, {}) if isinstance(p, str) else (p[0], dict(p[1] or {}))
        if name == "CUDAExecutionProvider":
            opts.setdefault("use_tf32", "0")
            out.append((name, opts))
        else:
            out.append(p)
    return out


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


def width_bucket(w: int) -> int:
    """The bucket a batch of width w is padded to."""
    i = bisect.bisect_left(WIDTH_BUCKETS, w)
    if i < len(WIDTH_BUCKETS):
        return WIDTH_BUCKETS[i]
    return -(-w // WIDTH_BEYOND) * WIDTH_BEYOND


def count_bucket(b: int) -> int:
    """The line count a batch of b lines is padded to: a power of two up to 64,
    then a multiple of 64."""
    if b <= 64:
        return 1 << max(0, (b - 1).bit_length())
    return -(-b // 64) * 64


def is_line_model(sess) -> bool:
    """A model fed a line batch: x [B, 1, H, W] and widths [B], with outputs whose
    first axis is the batch."""
    ins = {i.name: i for i in sess.get_inputs()}
    if set(ins) != {"x", "widths"} or len(ins["x"].shape) != 4:
        return False
    b, w = ins["x"].shape[0], ins["x"].shape[3]
    return isinstance(b, str) and isinstance(w, str)


class _BucketedSession:
    """A line model on the GPU: each batch padded to (count_bucket, width_bucket),
    one session per width bucket (made when first needed), outputs cut back to
    the batch's lines and, on a time axis (a symbolic second axis, T = W / stride),
    to its width."""

    def __init__(self, first, path, so, providers, run_options):
        self._args = (str(path), so, providers)
        self._ro = run_options
        self._first = first
        self._by_width: dict = {}
        self._timed = [
            len(o.shape) >= 2 and isinstance(o.shape[1], str) for o in first.get_outputs()
        ]
        self._names = [o.name for o in first.get_outputs()]

    def _session(self, wb: int):
        s = self._by_width.get(wb)
        if s is None:
            if not self._by_width:
                s = self._first
            else:
                import onnxruntime as ort

                s = ort.InferenceSession(self._args[0], self._args[1], providers=self._args[2])
            self._by_width[wb] = s
        return s

    def run(self, output_names, input_feed, run_options=None):
        x, widths = input_feed["x"], input_feed["widths"]
        b, _, h, w = x.shape
        wb, bb = width_bucket(w), count_bucket(b)
        if (b, w) != (bb, wb):
            xp = np.zeros((bb, 1, h, wb), x.dtype)
            xp[:b, :, :, :w] = x
            wp = np.full(bb, PAD_WIDTH, widths.dtype)
            wp[:b] = widths
            x, widths = xp, wp
        out = self._session(wb).run(output_names, {"x": x, "widths": widths}, run_options or self._ro)
        names = output_names or self._names
        res = []
        for name, o in zip(names, out):
            o = o[:b]
            if self._timed[self._names.index(name)] and wb != w:
                t = o.shape[1] * w // wb
                o = o[:, :t]
            res.append(np.ascontiguousarray(o))
        return res

    def __getattr__(self, name):
        return getattr(self._first, name)


def session(path, so, providers):
    """ort.InferenceSession(path, so, providers=providers).  On a GPU: wrapped so
    that every run gives the arena's free memory back, and, for a line model,
    with its batches padded to a few shapes (``_BucketedSession``)."""
    import onnxruntime as ort

    if not _on_cuda(providers):
        return ort.InferenceSession(str(path), so, providers=providers)
    _preload()
    providers = fp32_providers(providers)
    ro = gpu_run_options(providers)
    sess = ort.InferenceSession(str(path), so, providers=providers)
    if is_line_model(sess):
        return _BucketedSession(sess, path, so, providers, ro)
    return _ShrinkingSession(sess, ro)
