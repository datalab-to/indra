"""Model files: a local folder, or a download from the Hugging Face repo."""

from __future__ import annotations

import os
from pathlib import Path

HF_REPO = "datalab-to/indra-ocr"
# Model set this package version was built and tested against (a tag in HF_REPO).
HF_REVISION = "v1.1.0"
FILES = ("det.onnx", "rec.onnx", "order.onnx", "meta.json")


class ModelError(OSError):
    """The models are not available: a download failed (no network, no access)
    or a local model folder lacks files."""


MATH_FILES = (
    "mathcls.onnx",
    "math_enc.onnx",
    "math_dec_init.onnx",
    "math_dec_step.onnx",
    "math_meta.json",
)


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
            missing += [f for f in MATH_FILES if not (p / f).exists()]
        if missing:
            raise ModelError(f"the model folder {p} lacks {', '.join(missing)}")
        return p
    from huggingface_hub import snapshot_download

    rev = revision or HF_REVISION
    kw = dict(
        revision=rev,
        allow_patterns=[
            "*.onnx",
            "meta.json",
            "lm.npz",
            "lm_*.npz",
            *(MATH_FILES if math else ()),
        ],
        ignore_patterns=[] if math else list(MATH_FILES),
    )
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
