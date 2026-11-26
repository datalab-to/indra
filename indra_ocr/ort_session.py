"""ONNX Runtime sessions: line models run in pieces on the CPU; on a GPU, the
arena's free memory is given back after every run.

**Line models in pieces (CPU).**  A line model (inputs ``x`` [B, 1, H, W] and
``widths`` [B]: the recognizers, the line classifiers, the math encoder) reads
each line of a batch on its own: a line's outputs are bitwise the same in any
batch padded to the same width (checked for every line model).  So a batch of up
to 64 lines (``width_batches``) runs as pieces of at most PIECE_PIXELS input
pixels, with the same padding, and gives the same outputs.  A whole batch's
activations are tens of MB (``rec.onnx``, 64 lines of 640 px: 63 MB for one
layer's output), far more than any core's cache, so every layer goes to memory
and back; worker processes on cores that share a cache and memory bandwidth slow
each other down.  A piece's are a few MB (2 lines of 640 px: about 2 MB), which
fit in one core's share of the last-level cache on current CPUs, while a run of
that size still costs well over its fixed overhead.  Measured with
bench/ab_variant.py (variant "pieces").  ``pieces`` gives each piece's outputs as
it runs, so a caller can read a piece's logits while they are in the cache.

**GPU arena.**  ONNX Runtime's CUDA provider keeps the largest allocation each session ever
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

import os

import numpy as np

# input pixels per run of a line model on the CPU; INDRA_PIECE_PIXELS=0 runs every batch whole
PIECE_PIXELS = int(os.environ.get("INDRA_PIECE_PIXELS", 2 * 96 * 640))


def is_line_model(sess) -> bool:
    """Whether a session reads a batch of line crops: inputs x [B, 1, H, W] and widths [B]."""
    ins = {i.name: i.shape for i in sess.get_inputs()}
    return set(ins) == {"x", "widths"} and len(ins["x"]) == 4


class RowPieces:
    """A line model's session run in pieces of rows (module docstring): ``run``
    gives the outputs of the whole batch, ``pieces`` yields (first row, end row,
    outputs) per piece."""

    def __init__(self, sess, pixels: int = PIECE_PIXELS):
        self._sess, self.pixels = sess, pixels

    def rows(self, x) -> int:
        """Rows per piece for a batch x [B, 1, H, W]."""
        return max(1, self.pixels // max(1, x.shape[2] * x.shape[3]))

    def pieces(self, output_names, input_feed, run_options=None):
        x, widths = input_feed["x"], input_feed["widths"]
        B, n = len(x), self.rows(x)
        if n >= B:
            yield 0, B, self._sess.run(output_names, input_feed, run_options)
            return
        for lo in range(0, B, n):
            hi = min(B, lo + n)
            yield lo, hi, self._sess.run(
                output_names, {"x": x[lo:hi], "widths": widths[lo:hi]}, run_options
            )

    def run(self, output_names, input_feed, run_options=None):
        parts = [out for _, _, out in self.pieces(output_names, input_feed, run_options)]
        if len(parts) == 1:
            return parts[0]
        return [np.concatenate([p[i] for p in parts]) for i in range(len(parts[0]))]

    def __getattr__(self, name):
        return getattr(self._sess, name)


def run_pieces(sess, output_names, input_feed):
    """(first row, end row, outputs) per piece of a batch: RowPieces.pieces, or one
    piece for any other session."""
    pieces = getattr(sess, "pieces", None)
    if pieces is not None:
        yield from pieces(output_names, input_feed)
    else:
        yield 0, len(input_feed["x"]), sess.run(output_names, input_feed)


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
    that every run gives the arena's free memory back; on the CPU, a line model
    runs in pieces (RowPieces)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), so, providers=providers)
    ro = gpu_run_options(providers)
    if ro is not None:
        return _ShrinkingSession(sess, ro)
    if PIECE_PIXELS and is_line_model(sess):
        return RowPieces(sess)
    return sess
