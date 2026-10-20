"""indra_ocr.native loads the optional kernels, or runs the Python code: when
the package is missing, broken, of another API version, or turned off."""

import sys
import types

import pytest

from indra_ocr import native


@pytest.fixture
def fake_module(monkeypatch):
    """fake_module(m): ``import indra_ocr_native`` gives m (None: ImportError)."""

    def put(m):
        monkeypatch.setitem(sys.modules, "indra_ocr_native", m)

    monkeypatch.delenv("INDRA_OCR_NATIVE", raising=False)
    return put


def test_a_missing_package_means_python_quietly(fake_module, recwarn):
    fake_module(None)  # import raises ImportError
    assert native._load() is None
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_a_broken_package_means_python_with_a_warning(monkeypatch):
    class Finder:
        def find_spec(self, name, path=None, target=None):
            if name == "indra_ocr_native":
                raise OSError("DLL load failed")

    monkeypatch.delitem(sys.modules, "indra_ocr_native", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Finder(), *sys.meta_path])
    monkeypatch.delenv("INDRA_OCR_NATIVE", raising=False)
    with pytest.warns(RuntimeWarning, match="does not load"):
        assert native._load() is None


def test_another_api_version_means_python_with_a_warning(fake_module):
    fake_module(types.SimpleNamespace(API=native.API + 1, __version__="9.9"))
    with pytest.warns(RuntimeWarning, match="needs API"):
        assert native._load() is None


def test_the_matching_api_loads(fake_module):
    m = types.SimpleNamespace(API=native.API)
    fake_module(m)
    assert native._load() is m


def test_the_environment_turns_the_kernels_off(fake_module, monkeypatch):
    fake_module(types.SimpleNamespace(API=native.API))
    monkeypatch.setenv("INDRA_OCR_NATIVE", "0")
    assert native._load() is None
    monkeypatch.setattr(native, "lib", None)
    assert "INDRA_OCR_NATIVE=0" in native.status()


def test_disabled_restores_the_kernels_after_an_error(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(native, "lib", sentinel)
    with pytest.raises(ValueError):
        with native.disabled():
            assert native.lib is None
            raise ValueError
    assert native.lib is sentinel
