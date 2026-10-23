"""Math lines: a classifier finds lines that hold math, and MathRec reads them as
text with inline LaTeX ("we have $x _ { i } \\leq 1$").

The CTC recognizer reads one symbol per column, so it cannot write a fraction or
a sum with limits.  MathRec reads a taller crop of the line with an attention
decoder, one token per step, with the decoder's keys and values cached.

Models (all optional; the pipeline skips math without them): mathcls.onnx,
math_enc.onnx, math_dec_init.onnx, math_dec_step.onnx and math_meta.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .line_mask import crop_lines
from .rec_preproc import pad_batch, width_batches

MATH_MIN = 0.5  # classifier probability above which a line is read as math
# A math reading replaces a line only at this mean token probability or more: on
# print the model has not seen (old scans) it writes confident-looking LaTeX
# that is wrong, at low probability.
MATH_MIN_CONF = 0.9
# A line whose mean token probability is this far below MATH_MIN_CONF after 8
# tokens stops decoding: it would be rejected anyway (garbage on unseen print).
EARLY_STOP_MARGIN = 0.1
BOS, EOS, N_SPECIAL = 1, 2, 4
FILES = ("mathcls.onnx", "math_enc.onnx", "math_dec_init.onnx", "math_dec_step.onnx")


def detokenize(tokens: list[str]) -> str:
    """Tokens -> text; inside math (between "$" tokens) they are joined by spaces."""
    out, math, in_math = [], [], False
    for t in tokens:
        if t == "$":
            if in_math:
                out.append("$" + " ".join(math) + "$")
                math = []
            in_math = not in_math
        elif in_math:
            math.append(t)
        else:
            out.append(t)
    if in_math and math:
        out.append("$" + " ".join(math) + "$")
    return " ".join("".join(out).split())


class MathReader:
    ro = None  # RunOptions for every run (pipeline.gpu_run_options)

    def __init__(self, onnx_dir: Path, so, providers, run_options=None):
        import onnxruntime as ort

        self.ro = run_options

        s = {
            f: ort.InferenceSession(str(onnx_dir / f), so, providers=providers)
            for f in FILES
        }
        self.cls, self.enc = s["mathcls.onnx"], s["math_enc.onnx"]
        self.init, self.step = s["math_dec_init.onnx"], s["math_dec_step.onnx"]
        meta = json.loads((onnx_dir / "math_meta.json").read_text())
        cfg = meta["config"]
        self.height = cfg["height"]
        self.max_tokens = cfg["max_tokens"]
        self.layers, self.heads = cfg["dec_layers"], cfg["dec_heads"]
        self.head_dim = cfg["channels"] // cfg["dec_heads"]
        self.itos = ["", "", "", ""] + meta["tokens"]
        # Exports drop unused inputs (the init graph does not read the mask).
        self.inputs = {f: {i.name for i in v.get_inputs()} for f, v in s.items()}

    @classmethod
    def load(cls, onnx_dir: Path, so, providers, run_options=None) -> "MathReader | None":
        names = (*FILES, "math_meta.json")
        if all((onnx_dir / f).exists() for f in names):
            return cls(onnx_dir, so, providers, run_options)
        return None

    def flags(self, crops: list, batch: int = 64) -> list[bool]:
        """Per line crop (the recognizer's crop): does it hold math?"""
        out = [False] * len(crops)
        for idx in width_batches([c.shape[1] for c in crops], batch):
            arr, widths = pad_batch([crops[j] for j in idx])
            p = self.cls.run(None, {"x": arr, "widths": widths}, self.ro)[0]
            for k, j in enumerate(idx):
                out[j] = bool(p[k] > MATH_MIN)
        return out

    def read(
        self, gray: np.ndarray, boxes: np.ndarray, batch: int = 64
    ) -> list[tuple[str, float]]:
        """(text with inline LaTeX, mean token probability) for each box."""
        crops = crop_lines(gray, boxes[:, :4], self.height, None)
        out = [("", 0.0)] * len(crops)
        # Few, wide batches: each batch is one decoding loop of up to max_tokens
        # steps, and a step costs about the same for 8 lines as for 40.
        widths = [c.shape[1] for c in crops]
        for idx in width_batches(widths, batch, max_ratio=4.0, max_area=16 * 64 * 640):
            arr, widths = pad_batch([crops[j] for j in idx])
            mem, mask = self.enc.run(None, {"x": arr, "widths": widths}, self.ro)
            feed = {"memory": mem, "mask": mask}
            names = self.inputs["math_dec_init.onnx"]
            ck, cv = self.init.run(None, {k: v for k, v in feed.items() if k in names}, self.ro)
            B = len(idx)
            sk = np.zeros((self.layers, B, self.heads, 0, self.head_dim), np.float32)
            sv = sk
            tok = np.full(B, BOS, np.int64)
            step_in = self.inputs["math_dec_step.onnx"]
            seqs: list[list[int]] = [[] for _ in range(B)]
            probs: list[list[float]] = [[] for _ in range(B)]
            done = np.zeros(B, bool)
            for t in range(self.max_tokens - 1):
                feed = {
                    "token": tok,
                    "pos": np.array(t, np.int64),
                    "self_k": sk,
                    "self_v": sv,
                    "cross_k": ck,
                    "cross_v": cv,
                    "mask": mask,
                }
                logits, sk, sv = self.step.run(
                    None, {k: v for k, v in feed.items() if k in step_in}, self.ro
                )
                tok = logits.argmax(-1).astype(np.int64)
                e = np.exp(logits - logits.max(-1, keepdims=True))
                top = 1.0 / e.sum(-1)  # softmax probability of the argmax token
                for b in range(B):
                    if not done[b]:
                        probs[b].append(float(top[b]))
                        if tok[b] == EOS:
                            done[b] = True
                        elif (
                            len(probs[b]) >= 8
                            and np.mean(probs[b]) < MATH_MIN_CONF - EARLY_STOP_MARGIN
                        ):
                            done[b] = True  # rejected: keeps a low mean probability
                        elif tok[b] >= N_SPECIAL:
                            seqs[b].append(int(tok[b]))
                if done.all():
                    break
            for k, j in enumerate(idx):
                text = detokenize([self.itos[i] for i in seqs[k]])
                out[j] = (text, float(np.mean(probs[k])) if probs[k] else 0.0)
        return out
