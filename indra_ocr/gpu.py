"""The optional GPU engine (the indra-ocr-cuda package, ``cuda/``) behind ``load(device="cuda")``.

Each model the engine knows (the recognizers and line classifiers, the detector,
the orientation and table-separator models) runs as a TensorRT engine on fixed
device buffers.  The recognizers' logits stay on the GPU: the engine session
gives the CTC outputs the pipeline reads (``ids``, ``conf``, ``lengths``, and
``blank``, ``topv``, ``topi`` for the lines the beam search reads), computed on
the device, as the older recognizer graphs gave them (ctc.py: the same values).
A model the engine does not know, or a batch outside an engine's shapes, runs on
ONNX Runtime's CUDA provider (ort_session.py).

Engines are FP32 with TF32 off by default (within 1e-4 of ONNX Runtime on the
CPU); INDRA_GPU_PRECISION=fp16 builds FP16 ones.  They are built when a model
first loads and cached in INDRA_TRT_CACHE (default ~/.cache/indra-ocr/trt).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ENGINE = "IndraCudaEngine"  # the pseudo-provider load(device="cuda") puts first
API = 1  # the indra_ocr_cuda.API this module calls
LINE_MAX_BATCH = 64
LINE_MAX_WIDTH = 4096
LINE_OPT = (32, 768)
MATH_MAX_WIDTH = 2048
DEC_MAX_FRAMES = 512  # math decoder: encoder frames (W / 8) and tokens it attends over
DEC_MAX_TOKENS = 320  # math_meta.json max_tokens
WORKSPACE = 2 << 30  # the TensorRT builder's workspace, bytes

_lib = None
_tried = False


def lib():
    """The indra_ocr_cuda module, or None when it is not installed, does not
    match, or sees no GPU."""
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    if os.environ.get("INDRA_OCR_CUDA", "").strip() == "0":
        return None
    try:
        import indra_ocr_cuda as m
    except ImportError:
        return None
    if getattr(m, "API", None) != API or m.device_count() < 1:
        return None
    _lib = m
    return _lib


def available() -> bool:
    return lib() is not None


class _Info:
    """An input or output's name and shape, as InferenceSession.get_inputs gives them."""

    def __init__(self, name: str, shape: list):
        self.name, self.shape = name, shape


def _io(path: str) -> tuple[list[_Info], list[_Info]]:
    import onnx

    m = onnx.load(path, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}

    def shp(v):
        return [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]

    ins = [_Info(v.name, shp(v)) for v in m.graph.input if v.name not in inits]
    outs = [_Info(v.name, shp(v)) for v in m.graph.output]
    return ins, outs


def _profiles(name: str, ins: list[_Info]) -> dict | None:
    """The engine's shapes for a model file, or None for a model the engine does
    not run."""
    by = {i.name: i for i in ins}
    if set(by) == {"x", "widths"} and len(by["x"].shape) == 4 and isinstance(by["x"].shape[2], int):
        h = by["x"].shape[2]
        b, w = LINE_OPT
        # the math encoder attends over the whole line: its largest shape is
        # smaller (TensorRT found no tactic within the workspace at 64 x 4096)
        mw = MATH_MAX_WIDTH if name == "math_enc.onnx" else LINE_MAX_WIDTH
        return {
            "x": ([1, 1, h, 8], [b, 1, h, min(w, mw)], [LINE_MAX_BATCH, 1, h, mw]),
            "widths": ([1], [b], [LINE_MAX_BATCH]),
        }
    if name == "math_dec_init.onnx" and "memory" in by:
        c = by["memory"].shape[2]
        return {"memory": ([1, 1, c], [16, 120, c], [LINE_MAX_BATCH, DEC_MAX_FRAMES, c])}
    if name == "math_dec_step_kv.onnx":
        prof = {}
        for i in ins:
            s = i.shape
            if i.name == "token":
                prof[i.name] = ([1], [16], [LINE_MAX_BATCH])
            elif i.name == "mask":
                prof[i.name] = ([1, 1], [16, 120], [LINE_MAX_BATCH, DEC_MAX_FRAMES])
            elif i.name.startswith("self_"):
                prof[i.name] = ([1, s[1], 0, s[3]], [16, s[1], 60, s[3]], [LINE_MAX_BATCH, s[1], DEC_MAX_TOKENS, s[3]])
            elif i.name.startswith("cross_"):
                prof[i.name] = ([1, s[1], 1, s[3]], [16, s[1], 120, s[3]], [LINE_MAX_BATCH, s[1], DEC_MAX_FRAMES, s[3]])
        return prof
    if name == "det.onnx":
        return {"x": ([1, 1, 32, 32], [1, 1, 1024, 800], [1, 1, 2048, 2048])}
    if name == "orient.onnx":
        s = by["x"].shape
        return {"x": ([1, 1, s[2], s[3]], [1, 1, s[2], s[3]], [8, 1, s[2], s[3]])}
    if name == "tsr_split.onnx":
        return {"x": ([1, 1, 32, 32], [1, 1, 512, 768], [1, 1, 1056, 1056])}
    return None


def _precision() -> str:
    p = os.environ.get("INDRA_GPU_PRECISION", "fp32").strip().lower()
    if p not in ("fp32", "fp16"):
        raise ValueError(f"INDRA_GPU_PRECISION must be fp32 or fp16, not {p!r}")
    return p


def fp16_model(data: bytes) -> bytes:
    """The model with FP16 weights and compute, its inputs and outputs kept FP32,
    in a form TensorRT 11 (strongly typed) builds: the casts the conversion
    leaves to FLOAT inside the graph made FLOAT16, and the float constants too,
    so no node mixes the two."""
    import onnx
    from onnx import TensorProto, numpy_helper
    from onnxconverter_common import float16

    m = float16.convert_float_to_float16(onnx.load_from_string(data), keep_io_types=True)
    g = m.graph
    outs = {o.name for o in g.output}
    for n in g.node:
        if n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to" and a.i == TensorProto.FLOAT and not set(n.output) & outs:
                    a.i = TensorProto.FLOAT16
        elif n.op_type in ("Constant", "ConstantOfShape"):
            for a in n.attribute:
                if a.name == "value" and a.t.data_type == TensorProto.FLOAT:
                    a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float16), a.t.name))
    del g.value_info[:]
    return m.SerializeToString()


def _cache_dir() -> Path:
    d = os.environ.get("INDRA_TRT_CACHE")
    return Path(d) if d else Path.home() / ".cache" / "indra-ocr" / "trt"


def _device(providers: list) -> int:
    for p in providers:
        if not isinstance(p, str) and p[0] == "CUDAExecutionProvider":
            return int((p[1] or {}).get("device_id", 0))
    return 0


class TrtSession:
    """A model on the engine, with InferenceSession's run, get_inputs and get_outputs."""

    def __init__(self, path: str, profiles: dict, device: int, fallback):
        m = lib()
        self._model_path = path
        self._ins, self._outs = _io(path)
        self._fallback = fallback  # (path, so, providers) for ONNX Runtime
        self._fb = None
        data = Path(path).read_bytes()
        prec = _precision()
        if prec == "fp16":
            data = fp16_model(data)
        key = hashlib.sha256(
            data
            + json.dumps(profiles, sort_keys=True).encode()
            + m.device_name(device).encode()
            + f"{prec} api{API} {getattr(m, '__version__', '')}".encode()
        ).hexdigest()[:24]
        cache = _cache_dir() / f"{Path(path).stem}-{prec}-{key}.engine"
        self._limits = {k: v[2] for k, v in profiles.items()}
        self._mins = {k: v[0] for k, v in profiles.items()}
        self._m = m.Model(data, profiles, str(cache), WORKSPACE, device)

    def get_inputs(self):
        return self._ins

    def get_outputs(self):
        return self._outs

    def _fits(self, feed: dict) -> bool:
        for k, v in feed.items():
            lim, lo = self._limits.get(k), self._mins.get(k)
            if lim is not None and any(a > b for a, b in zip(v.shape, lim)):
                return False
            if lo is not None and any(a < b for a, b in zip(v.shape, lo)):
                return False
        return True

    def _ort(self):
        if self._fb is None:
            from .ort_session import session

            path, so, providers = self._fallback
            self._fb = session(path, so, providers)
        return self._fb

    @staticmethod
    def _feed(feed: dict) -> dict:
        # np.ascontiguousarray would make a 0-d input (the decoder's pos) 1-d
        return {k: v if isinstance(v, np.ndarray) and v.flags.c_contiguous else np.require(v, requirements="C")
                for k, v in feed.items()}

    def run(self, output_names, input_feed, run_options=None):
        feed = self._feed(input_feed)
        if not self._fits(feed):
            return self._ort().run(output_names, input_feed)
        names = list(output_names) if output_names else [o.name for o in self._outs]
        return self._m.run(feed, names)


class _LM:
    """One of blank, topv, topi of a recognizer run, computed on the device for
    a line when the pipeline first reads that line (out[k, ...])."""

    def __init__(self, sess, which: int, gen: int):
        self._s, self._which, self._gen = sess, which, gen

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        row = self._s._lm_row(int(idx[0]), self._gen)[self._which]
        return row[idx[1:]] if len(idx) > 1 else row


class TrtRecognizer(TrtSession):
    """A recognizer whose graph ends at its logits: on the engine it gives the
    older graphs' CTC outputs instead (``ids``, ``conf``, ``lengths``, ``blank``,
    ``topv``, ``topi``), so the logits stay on the GPU.  Batches of more than
    LINE_MAX_BATCH lines run in parts."""

    OUTS = ("ids", "conf", "lengths", "blank", "topv", "topi")

    def __init__(self, path, profiles, device, fallback):
        super().__init__(path, profiles, device, fallback)
        self._gen = 0
        self._lengths = None
        self._rows: dict = {}
        shp = {o.name: o.shape for o in self._outs}
        self._outs = [
            _Info("ids", shp["ids"]),
            _Info("conf", shp["ids"]),
            _Info("lengths", shp["lengths"]),
            _Info("blank", shp["ids"]),
            _Info("topv", shp["ids"] + [8]),
            _Info("topi", shp["ids"] + [8]),
        ]

    def _lm_row(self, k: int, gen: int):
        if gen != self._gen:
            raise RuntimeError("indra-ocr-cuda: the LM inputs of a recognizer run read after its next run")
        if k not in self._rows:
            n = int(self._lengths[k])
            b, v, i = self._m.lm_inputs("logits", [k], [n], 8)
            self._rows[k] = (b[0], v[0], i[0])
        return self._rows[k]

    def run(self, output_names, input_feed, run_options=None):
        feed = self._feed(input_feed)
        names = list(output_names) if output_names else list(self.OUTS)
        x, widths = feed["x"], feed["widths"]
        if x.shape[3] > LINE_MAX_WIDTH or x.shape[3] < 8:
            return self._run_ort(names, feed)
        if x.shape[0] > LINE_MAX_BATCH:
            # in parts; the LM inputs of every line of a part are made before the
            # next part overwrites the logits
            lm = any(n in ("blank", "topv", "topi") for n in names)
            parts = []
            for i in range(0, len(x), LINE_MAX_BATCH):
                ids, lengths = self._m.run(
                    {"x": x[i : i + LINE_MAX_BATCH], "widths": widths[i : i + LINE_MAX_BATCH]}, ["ids", "lengths"]
                )
                got = {"ids": ids, "lengths": lengths, "conf": self._m.frame_conf("logits", "lengths")}
                if lm:
                    rows = list(range(len(ids)))
                    got["blank"], got["topv"], got["topi"] = self._m.lm_inputs(
                        "logits", rows, [int(v) for v in lengths], 8
                    )
                parts.append(got)
            self._gen += 1
            return [np.concatenate([p[n] for p in parts]) for n in names]
        ids, lengths = self._m.run(feed, ["ids", "lengths"])
        self._gen += 1
        self._lengths, self._rows = lengths, {}
        conf = self._m.frame_conf("logits", "lengths") if "conf" in names else None
        got = {"ids": ids, "lengths": lengths, "conf": conf}
        out = []
        for n in names:
            if n in got:
                out.append(got[n])
            elif n in ("blank", "topv", "topi"):
                out.append(_LM(self, ("blank", "topv", "topi").index(n), self._gen))
            else:
                raise KeyError(f"indra-ocr-cuda: a recognizer gives {self.OUTS}, not {n!r}")
        return out

    def _run_ort(self, names, feed):
        """The batch on ONNX Runtime: its logits output, turned into the CTC
        outputs on the host (ctc.py)."""
        from .ctc import lm_inputs, peak_prob

        ids, lengths, logits = self._ort().run(["ids", "lengths", "logits"], feed)
        B, T = ids.shape
        conf = np.zeros((B, T), np.float32)
        lm = None
        if any(n in ("blank", "topv", "topi") for n in names):
            lm = (np.zeros((B, T), np.float32), np.full((B, T, 8), -np.inf, np.float32), np.zeros((B, T, 8), np.int64))
        for b in range(B):
            n = int(lengths[b])
            conf[b, :n] = peak_prob(logits[b, :n])
            if lm is not None:
                bl, tv, ti = lm_inputs(logits[b, :n])
                lm[0][b, :n], lm[1][b, :n], lm[2][b, :n] = bl, tv, ti
        got = {"ids": ids, "lengths": lengths, "conf": conf}
        if lm is not None:
            got.update(blank=lm[0], topv=lm[1], topi=lm[2])
        return [got[n] for n in names]


class DeviceDecoder:
    """A batch's math decoder on the GPU engine: the encoder's memory, the
    cross-attention caches and the growing self-attention caches stay in VRAM;
    a step sends the batch's tokens up and brings the logits ([B, V]) back, and
    lines that finish leave the caches on the device.  math.py's decoding loop
    (the token rules, early stops, rows leaving the batch) runs as on the CPU."""

    def __init__(self, reader, arr: np.ndarray, widths: np.ndarray, B: int):
        enc, init, step = reader.enc._m, reader.init._m, reader.step._m
        enc.run({"x": np.ascontiguousarray(arr), "widths": np.ascontiguousarray(widths)}, [])
        init.bind("memory", enc, "memory")
        if "mask" in init.inputs:
            init.bind("mask", enc, "mask")
        init.run({}, [])
        L = reader.layers
        for i in range(L):
            step.bind(f"cross_k{i}", init, "cross_k", i)
            step.bind(f"cross_v{i}", init, "cross_v", i)
        if "mask" in step.inputs:
            step.bind("mask", enc, "mask")
        empty = np.zeros((B, reader.heads, 0, reader.head_dim), np.float32)
        self._first = {f"self_{kv}{i}": empty for kv in "kv" for i in range(L)}
        self._rows = [f"{n}{i}" for n in ("self_k", "self_v", "cross_k", "cross_v") for i in range(L)]
        if "mask" in step.inputs:
            self._rows.append("mask")
        self._pairs = [(f"new_{kv}{i}", f"self_{kv}{i}") for kv in "kv" for i in range(L)]
        self._s = step
        self._in = set(step.inputs)

    @staticmethod
    def fits(reader, arr: np.ndarray, widths: np.ndarray) -> bool:
        feed = {"x": arr, "widths": widths}
        return (
            reader.enc._fits(feed)
            and arr.shape[0] <= LINE_MAX_BATCH
            and arr.shape[3] // 8 <= DEC_MAX_FRAMES
            and reader.max_tokens <= DEC_MAX_TOKENS
        )

    def step(self, tok: np.ndarray, t: int) -> np.ndarray:
        feed = {"token": np.ascontiguousarray(tok, np.int64), "pos": np.array(t, np.int64)}
        if self._first is not None:
            feed.update(self._first)
            self._first = None
        (logits,) = self._s.run({k: v for k, v in feed.items() if k in self._in}, ["logits"])
        self._s.chain(self._pairs)
        return logits

    def take(self, live: np.ndarray) -> None:
        self._s.keep_rows(self._rows, [int(r) for r in np.flatnonzero(live)])


def session_for(path, so, providers):
    """The engine's session for this model file, or None for ONNX Runtime.
    providers: the ONNX Runtime providers a fallback uses."""
    if lib() is None:
        return None
    path = str(path)
    name = Path(path).name
    ins, outs = _io(path)
    prof = _profiles(name, ins)
    if prof is None:
        return None
    device = _device(providers)
    fallback = (path, so, providers)
    try:
        if {o.name for o in outs} >= {"ids", "lengths", "logits"}:
            return TrtRecognizer(path, prof, device, fallback)
        return TrtSession(path, prof, device, fallback)
    except Exception as e:  # noqa: BLE001  (a model TensorRT cannot build runs on ONNX Runtime)
        import warnings

        warnings.warn(f"indra-ocr-cuda: {name} runs on ONNX Runtime ({e})", RuntimeWarning, stacklevel=2)
        return None
