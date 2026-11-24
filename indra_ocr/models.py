"""Model files: a local folder, or a download from the Hugging Face repo."""

from __future__ import annotations

import os
from pathlib import Path

# The Hugging Face repo and the tag the package downloads.
HF_REPO = "datalab-to/indra-ocr"
HF_REVISION = "v1.2.0"
FILES = ("det.onnx", "rec.onnx", "order.onnx", "meta.json")


class ModelError(OSError):
    """The models are not available: a download failed (no network, no access)
    or a local model folder lacks files."""


MATH_FILES = (
    "mathcls.onnx",
    "math_enc.onnx",
    "math_dec_init.onnx",
    "math_dec_step_kv.onnx",
    "math_meta.json",
)
# The older MathRec decoder step (stacked caches, same tokens as
# math_dec_step_kv.onnx).  Downloads skip it.  A local model folder from before
# math_dec_step_kv.onnx works with it in place of that file.
MATH_STEP_STACKED = "math_dec_step.onnx"


# Optional math files: downloaded with the math models, never required (an older
# model folder without them keeps the default behavior).
# the accept head and the read gate (math.MathReader) and the image span tagger (math_span_img.py)
MATH_OPTIONAL = (
    "math_accept.json",
    "math_gate.json",
    "math_span_img.onnx",
    "math_span_img.json",
)


def download_patterns(math: bool = False) -> tuple[list[str], list[str]]:
    """(allow_patterns, ignore_patterns) for the Hub download of the models."""
    allow = [
        "*.onnx",
        "meta.json",
        "math_span.json",  # the opt-in text-layer math span tagger's vocabulary
        "table_verify.json",  # the optional false-table verifier (table_verify.py)
        "lm.npz",
        "lm_*.npz",
        *(MATH_FILES + MATH_OPTIONAL if math else ()),
    ]
    ignore = [MATH_STEP_STACKED] if math else [*MATH_FILES, *MATH_OPTIONAL, MATH_STEP_STACKED]
    return allow, ignore


def model_dir(
    path: str | Path | None = None, revision: str | None = None, math: bool = False
) -> Path:
    """Folder with det.onnx, rec.onnx, meta.json and usually order.onnx (and,
    with ``math``, the math models, which are downloaded only then).  A local
    folder without order.onnx works: reading order then falls back to
    XY-cut, with no layout labels.

    Order of lookup: ``path``, the INDRA_OCR_MODEL_DIR environment variable, then
    a download of ``HF_REPO``.  Downloads are cached by huggingface_hub."""
    path = path or os.environ.get("INDRA_OCR_MODEL_DIR")
    if path:
        p = Path(path).expanduser()
        missing = [f for f in FILES[:2] + FILES[3:] if not (p / f).exists()]
        if math:
            missing += [
                f
                for f in MATH_FILES
                if not (p / f).exists()
                and not (f == "math_dec_step_kv.onnx" and (p / MATH_STEP_STACKED).exists())
            ]
        if missing:
            raise ModelError(f"the model folder {p} lacks {', '.join(missing)}")
        return p
    from huggingface_hub import snapshot_download

    rev = revision or HF_REVISION
    allow, ignore = download_patterns(math)
    kw = dict(revision=rev, allow_patterns=allow, ignore_patterns=ignore)
    # The files this call must leave behind.  A cached snapshot can be partial:
    # a first run without math downloads no math models, and huggingface_hub
    # only notices on its own when the cache carries a tree index (it does not
    # for a cache built by an older version), so check the folder here.
    need = (*FILES[:2], "meta.json", *(MATH_FILES if math else ()))
    try:
        # A tag's files never change: a cached copy needs no network (and no
        # progress bar on every run).
        cached = Path(snapshot_download(HF_REPO, local_files_only=True, **kw))
        if all((cached / f).exists() for f in need):
            return cached
    except Exception:  # noqa: BLE001  (not cached yet)
        pass
    try:
        p = Path(snapshot_download(HF_REPO, **kw))
    except Exception as e:  # noqa: BLE001  (network, access, disk: one clear message)
        reason = (str(e).strip().splitlines() or [type(e).__name__])[0]
        raise ModelError(
            f"could not download the models from Hugging Face ({HF_REPO}, {rev}): "
            f"{type(e).__name__}: {reason[:200]}. Check the network connection, or "
            "point at a local model folder with --model-dir or INDRA_OCR_MODEL_DIR."
        ) from e
    missing = [f for f in need if not (p / f).exists()]
    if missing:
        raise ModelError(f"the model folder {p} lacks {', '.join(missing)}")
    return p
