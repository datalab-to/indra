"""The MathRec decoder step file: math_dec_step_kv.onnx is the one a download gets.
A local folder with only the older math_dec_step.onnx still reads math."""

import fnmatch
import json

import pytest

from indra_ocr import math as M
from indra_ocr.models import MATH_FILES, MATH_STEP_STACKED, download_patterns, model_dir

META = {"config": {"height": 32, "max_tokens": 8, "dec_layers": 2, "dec_heads": 2, "channels": 8}, "tokens": ["x", "\\frac"]}


def _folder(tmp_path, *steps):
    for f in (*M.FILES, *steps):
        (tmp_path / f).touch()
    (tmp_path / "math_meta.json").write_text(json.dumps(META))
    return tmp_path


class _Sess:
    def __init__(self, path):
        self.path = path

    def get_inputs(self):
        return []


@pytest.fixture
def fake_session(monkeypatch):
    monkeypatch.setattr(M, "session", lambda path, so, providers=None: _Sess(path))


@pytest.mark.parametrize(
    "steps, used",
    [
        ((M.STEP_KV, M.STEP_STACKED), M.STEP_KV),
        ((M.STEP_KV,), M.STEP_KV),
        ((M.STEP_STACKED,), M.STEP_STACKED),  # a folder from before the kv graph
    ],
)
def test_reader_runs_the_step_it_finds(tmp_path, fake_session, steps, used):
    r = M.MathReader.load(_folder(tmp_path, *steps), None, None)
    assert r is not None
    assert r.step.path.endswith(used) and r.split_kv == (used == M.STEP_KV)


def test_no_step_graph_no_math(tmp_path, fake_session):
    assert M.step_file(_folder(tmp_path)) is None
    assert M.MathReader.load(tmp_path, None, None) is None


def _downloaded(names, math):
    allow, ignore = download_patterns(math)
    return [n for n in names if any(fnmatch.fnmatch(n, p) for p in allow) and not any(fnmatch.fnmatch(n, p) for p in ignore)]


def test_download_skips_the_stacked_step():
    repo = [*MATH_FILES, MATH_STEP_STACKED, "det.onnx", "rec.onnx", "meta.json", "tsr_rel.onnx"]
    got = _downloaded(repo, math=True)
    assert MATH_STEP_STACKED not in got and M.STEP_KV in got
    assert set(MATH_FILES) <= set(got)
    got = _downloaded(repo, math=False)
    assert not set(got) & {*MATH_FILES, MATH_STEP_STACKED}
    assert {"det.onnx", "rec.onnx", "meta.json", "tsr_rel.onnx"} <= set(got)


@pytest.mark.parametrize("step", [M.STEP_KV, M.STEP_STACKED])
def test_local_folder_with_either_step(tmp_path, monkeypatch, step):
    for f in ("det.onnx", "rec.onnx", "meta.json", *M.FILES, "math_meta.json", step):
        (tmp_path / f).touch()
    monkeypatch.delenv("INDRA_OCR_MODEL_DIR", raising=False)
    assert model_dir(tmp_path, math=True) == tmp_path


def test_local_folder_without_a_step(tmp_path, monkeypatch):
    from indra_ocr import ModelError

    for f in ("det.onnx", "rec.onnx", "meta.json", *M.FILES, "math_meta.json"):
        (tmp_path / f).touch()
    with pytest.raises(ModelError, match="lacks math_dec_step_kv.onnx"):
        model_dir(tmp_path, math=True)
