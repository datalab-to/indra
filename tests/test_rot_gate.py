import numpy as np

from indra_ocr import rot_gate


def test_turn_box_maps_points_like_rot90():
    H, W = 30, 50
    page = np.zeros((H, W), np.uint8)
    page[5:9, 10:20] = 255  # box (10, 5, 20, 9)
    for k in (1, 3):
        t = np.rot90(page, k)
        ys, xs = np.nonzero(t)
        got = rot_gate._turn_box((10, 5, 20, 9), k, H, W)
        assert got == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def _page_with_turned_word():
    import cv2

    g = np.full((400, 600), 255, np.uint8)
    word = np.full((30, 160), 255, np.uint8)
    cv2.putText(word, "Exam score", (2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    g[100:260, 300:330] = np.rot90(word, 1)  # a vertical label
    return g


def test_no_leftover_ink_reads_nothing():
    g = np.full((200, 300), 255, np.uint8)
    boxes = np.array([[10, 10, 100, 30, 1.0]], np.float32)
    assert rot_gate.candidates(g, boxes, [("text", 0.99, [])], np.zeros((0, 4)), 20.0, 0.5) == []


def test_turned_word_is_a_candidate():
    g = _page_with_turned_word()
    boxes = np.array([[20, 20, 200, 44, 1.0]], np.float32)
    cols = rot_gate.candidates(g, boxes, [("x", 0.99, [])], np.zeros((0, 4)), 24.0, 0.5)
    ys, xs = np.nonzero(g < 128)  # the word's ink
    assert any(c[0] <= xs.min() and c[2] >= xs.max() and c[1] <= ys.min() and c[3] >= ys.max() for c in cols)


def test_column_inputs_shape():
    g = _page_with_turned_word()
    x = rot_gate.column_inputs(g, [(300, 100, 330, 260)], 24.0)
    assert x.shape == (1, 1, rot_gate.CLS_H, rot_gate.CLS_W) and x.max() > 0.5


def test_read_rotated_keeps_page_without_candidates():
    g = np.full((200, 300), 255, np.uint8)
    boxes = np.array([[10, 10, 100, 30, 1.0]], np.float32)
    rec = [("text", 0.99, [(1, 20.0)])]
    b2, r2 = rot_gate.read_rotated(None, g, boxes, rec, np.zeros((0, 4)), 0.5)
    assert b2 is boxes and r2 is rec


def test_rot_gate_is_off_by_default(monkeypatch):
    import inspect

    from indra_ocr import pipeline

    monkeypatch.delenv("INDRA_ROT_GATE", raising=False)
    src = inspect.getsource(pipeline.OrtOCR.__init__)
    assert 'os.environ.get("INDRA_ROT_GATE", "0") == "1"' in src  # default off unless asked


def test_download_includes_rotcls():
    from indra_ocr.models import download_patterns

    for math in (False, True):
        allow, ignore = download_patterns(math=math)
        assert "rotcls.onnx" in allow or "*.onnx" in allow
