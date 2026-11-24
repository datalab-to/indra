"""rec_preproc.widen: a group recognizer that reads widened lines (meta.json rec_xscale_<group>)."""

import numpy as np

from indra_ocr.rec_preproc import pad_batch, widen


def test_widen_scales_width_only():
    x = np.random.rand(32, 100).astype(np.float32)
    assert widen([x], 1.5)[0].shape == (32, 150)
    assert widen([x], 1.0)[0] is x


def test_widened_batch_pads_to_the_stride():
    x = np.random.rand(32, 101).astype(np.float32)
    arr, widths = pad_batch(widen([x], 1.5))
    assert widths.tolist() == [152] and arr.shape[-1] % 8 == 0
