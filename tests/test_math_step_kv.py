"""MathReader decodes the same tokens with the per-layer step graph
(math_dec_step_kv.onnx) as with the stacked one (math_dec_step.onnx).

The sessions here are numpy stand-ins: a step whose logits depend on the token,
the position and every cache row, so a cache in the wrong order or on the wrong
row changes the output.  Rows stop at different lengths, so the finished-row
drops (read) and the beam reorders (_rescore) both run."""

import numpy as np
import pytest

from indra_ocr import math as M

L, H, D, V = 2, 2, 3, 16


class _In:
    def __init__(self, name):
        self.name = name


class FakeEnc:
    def get_inputs(self):
        return [_In("x"), _In("widths")]

    def run(self, _, feed):
        w = feed["widths"].astype(np.float32)
        T = 5
        mem = np.repeat(w[:, None, None], T, 1) * np.ones((1, 1, 4), np.float32)
        mask = np.ones((len(w), T), np.float32)
        mask[:, -1] = (w % 2).astype(np.float32)  # some rows have a padded column
        return mem, mask


class FakeInit:
    def get_inputs(self):
        return [_In("memory")]

    def run(self, _, feed):
        mem = feed["memory"]
        B, T = mem.shape[:2]
        base = mem[:, :, 0]  # [B, T]
        ck = np.stack([np.broadcast_to((base + i)[:, None, :, None], (B, H, T, D)) for i in range(L)])
        cv = np.stack([np.broadcast_to((base * (i + 2))[:, None, :, None], (B, H, T, D)) for i in range(L)])
        return np.ascontiguousarray(ck, np.float32), np.ascontiguousarray(cv, np.float32)


class FakeStep:
    def __init__(self, split: bool):
        self.split = split
        if split:
            names = [f"{c}{i}" for c in ("self_k", "self_v", "cross_k", "cross_v") for i in range(L)]
        else:
            names = ["self_k", "self_v", "cross_k", "cross_v"]
        self.names = ["token", "pos", *names, "mask"]

    def get_inputs(self):
        return [_In(n) for n in self.names]

    def run(self, _, feed):
        if self.split:
            sk, sv, ck, cv = ([feed[f"{c}{i}"] for i in range(L)] for c in ("self_k", "self_v", "cross_k", "cross_v"))
        else:
            sk, sv, ck, cv = (list(feed[c]) for c in ("self_k", "self_v", "cross_k", "cross_v"))
        tok, pos, mask = feed["token"], int(feed["pos"]), feed["mask"]
        B = len(tok)
        new_k, new_v = [], []
        for i in range(L):
            add = np.broadcast_to((tok + i)[:, None, None, None], (B, H, 1, D)).astype(np.float32)
            new_k.append(np.concatenate([sk[i], add], 2))
            new_v.append(np.concatenate([sv[i], 2 * add], 2))
        hist = sum(k.sum((1, 2, 3)) + 3 * v.sum((1, 2, 3)) for k, v in zip(new_k, new_v))
        cross = sum((c * (j + 1)).sum((1, 2, 3)) for j, c in enumerate(ck + cv))
        width = cross / (L * H * D * 5)  # about the line width
        length = width.astype(np.int64) % 5 + 2  # each row stops at its own length
        pick = M.N_SPECIAL + (hist.astype(np.int64) + 7 * pos + mask.sum(1).astype(np.int64)) % (V - M.N_SPECIAL)
        logits = np.zeros((B, V), np.float32)
        logits[np.arange(B), pick] = 20.0
        logits[np.arange(B), (pick + 1 - M.N_SPECIAL) % (V - M.N_SPECIAL) + M.N_SPECIAL] = 18.0 + (hist % 3)
        logits[pos >= length, M.EOS] = 40.0
        out = [logits]
        if self.split:
            return out + new_k + new_v
        return out + [np.stack(new_k), np.stack(new_v)]


def reader(split: bool) -> M.MathReader:
    r = object.__new__(M.MathReader)
    r.split_kv = split
    r.enc, r.init, r.step = FakeEnc(), FakeInit(), FakeStep(split)
    r.inputs = {"math_dec_init.onnx": {"memory"}}
    r.step_in = set(r.step.names)
    r.layers, r.heads, r.head_dim = L, H, D
    r.height, r.max_tokens = 8, 12
    r.itos = ["", "", "", ""] + list("abcdefghijkl")
    r.accepted, r.last_probs = {}, []
    return r


WIDTHS = [37, 64, 90, 41, 118, 77, 52, 66, 101]


@pytest.fixture
def crops(monkeypatch):
    monkeypatch.setattr(M, "crop_lines", lambda gray, boxes, h, _: [np.full((h, w), 0.5, np.float32) for w in WIDTHS])
    return [np.full((8, w), 0.5, np.float32) for w in WIDTHS]


def test_read_same_with_per_layer_caches(crops, monkeypatch):
    boxes = np.zeros((len(WIDTHS), 4), np.float32)
    old = reader(False).read(np.zeros((8, 8), np.uint8), boxes, batch=4)
    new = reader(True).read(np.zeros((8, 8), np.uint8), boxes, batch=4)
    assert old == new
    lengths = {len(t) for t, _ in old}
    assert len(lengths) > 1  # rows stopped at different steps: the drops ran


def test_rescore_same_with_per_layer_caches(crops, monkeypatch):
    drafts = ["abc", "bcd", "x", "efg", "aaa", "b", "kl", "jj", "abcd"]
    old = reader(False)._rescore(crops, drafts, batch=8)
    new = reader(True)._rescore(crops, drafts, batch=8)
    assert old == new
    assert any(t for t, _ in old)


def test_front_moves_live_rows_into_finished_rows():
    live = np.array([False, True, False, True, True])
    a = np.arange(5) * 10
    rows = np.arange(16).reshape(8, 2)  # more rows than the step ran: only the first 5 are read
    n = M.MathReader._front(live, [a, rows])
    assert n == 3
    assert a[:n].tolist() == [30, 10, 40]
    assert rows[:n].tolist() == [[6, 7], [2, 3], [8, 9]]
    assert rows[5:].tolist() == [[10, 11], [12, 13], [14, 15]]


def test_read_moves_rows_and_reads_the_same(crops, monkeypatch):
    """One batch of all lines, finishing in an order that leaves finished rows in
    front of live ones: rows move, and the readings are the stacked graph's."""
    moved = []
    front = M.MathReader._front

    def spy(live, arrays):
        moved.append(int((~live[: int(live.sum())]).sum()))
        return front(live, arrays)

    monkeypatch.setattr(M.MathReader, "_front", staticmethod(spy))
    boxes = np.zeros((len(WIDTHS), 4), np.float32)
    old = reader(False).read(np.zeros((8, 8), np.uint8), boxes, batch=len(WIDTHS))
    new = reader(True).read(np.zeros((8, 8), np.uint8), boxes, batch=len(WIDTHS))
    assert old == new
    assert sum(moved) > 0
