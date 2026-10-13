"""Model files: a local folder, or a download from the Hugging Face repo."""

from __future__ import annotations

import os
from pathlib import Path

HF_REPO = "datalab-to/indra-ocr"
# Model set this package version was built and tested against (a tag in HF_REPO).
HF_REVISION = "v3.1"
FILES = ("det.onnx", "rec.onnx", "order.onnx", "meta.json")


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
    """Folder with det.onnx, rec.onnx, order.onnx and meta.json (and, with
    ``math``, the math models, which are downloaded only then).

    Order of lookup: ``path``, the INDRA_OCR_MODEL_DIR environment variable, then
    a download of ``HF_REPO``.  Downloads are cached by huggingface_hub."""
    path = path or os.environ.get("INDRA_OCR_MODEL_DIR")
    if path:
        p = Path(path).expanduser()
        missing = [f for f in FILES[:2] + FILES[3:] if not (p / f).exists()]
        if missing:
            raise FileNotFoundError(f"{p} lacks {missing}")
        return p
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            HF_REPO,
            revision=revision or HF_REVISION,
            allow_patterns=["*.onnx", "meta.json", *(MATH_FILES if math else ())],
            ignore_patterns=[] if math else list(MATH_FILES),
        )
    )
