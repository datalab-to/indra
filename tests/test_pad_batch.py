"""pad_batch pads a batch to at least MIN_BATCH_WIDTH columns: a lone crop
narrower than the classifiers' pooling (22 px) used to pool to zero columns
and fail the page in onnxruntime (#24).  The widths keep the true widths."""

import numpy as np

from indra_ocr.rec_preproc import MIN_BATCH_WIDTH, WIDTH_MULTIPLE, pad_batch


def test_narrow_batch_is_padded_to_the_minimum_width():
    arr, widths = pad_batch([np.ones((32, 22), np.float32)])
    assert arr.shape == (1, 1, 32, MIN_BATCH_WIDTH)
    assert widths.tolist() == [22]
    assert arr[0, 0, :, 22:].sum() == 0


def test_wide_batch_is_padded_to_the_multiple_only():
    arr, widths = pad_batch([np.ones((32, 100), np.float32), np.ones((32, 90), np.float32)])
    assert arr.shape[-1] == 104 and arr.shape[-1] % WIDTH_MULTIPLE == 0
    assert widths.tolist() == [100, 90]
