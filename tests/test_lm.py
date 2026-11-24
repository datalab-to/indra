import math

import numpy as np

from indra_ocr.lm import NgramLM, _pack, beam_search


def _lm(tmp_path, probs):
    """A unigram model (order 1) with these token probabilities."""
    toks = sorted(probs)
    keys = np.array([_pack((t,)) for t in toks], np.uint64)
    path = tmp_path / "lm.npz"
    np.savez(
        path,
        order=np.int64(1),
        vocab_size=np.int64(4),
        k0=keys,
        v0=np.array([math.log(probs[t]) for t in toks], np.float32),
        hk0=np.array([0], np.uint64),
        hb0=np.array([math.log(0.01)], np.float32),
    )
    return NgramLM(path)


def _steps(rows):
    lp = np.log(np.array(rows, np.float64) / np.sum(rows, 1, keepdims=True))
    topi = np.argsort(-lp, 1)[:, :3]
    return lp[:, 0], np.take_along_axis(lp, topi, 1), topi


def test_lm_breaks_a_close_call(tmp_path):
    # second token: class 2 (0.48) against class 3 (0.47)
    blank, topv, topi = _steps(
        [[0.05, 0.9, 0.03, 0.02], [0.05, 0.0, 0.48, 0.47], [0.97, 0.01, 0.01, 0.01]]
    )
    flat = _lm(tmp_path, {1: 0.34, 2: 0.33, 3: 0.33})
    assert beam_search(blank, topv, topi, flat, alpha=1.0, beta=0.0)[0] == [1, 2]
    likes3 = _lm(tmp_path, {1: 0.45, 2: 0.05, 3: 0.5})
    ids, steps = beam_search(blank, topv, topi, likes3, alpha=1.0, beta=0.0)
    assert ids == [1, 3] and steps == [0, 1]


def test_batched_lookup_matches_single(tmp_path):
    rng = np.random.default_rng(0)
    # an order-3 table: some bigram / trigram entries and histories
    toks = range(1, 4)
    ent = {n: {} for n in range(3)}
    hist = {n: {} for n in range(3)}
    hist[0][()] = math.log(0.2)
    for t in toks:
        ent[0][(t,)] = math.log(0.3)
    for a in toks:
        hist[1][(a,)] = math.log(0.3)
        ent[1][(a, 1)] = math.log(0.5)
        for b in (1, 2):
            hist[2][(a, b)] = math.log(0.4)
            ent[2][(a, b, 3)] = math.log(0.6)
    arrays = {"order": np.int64(3), "vocab_size": np.int64(4)}
    for n in range(3):
        for name, d in (("", ent[n]), ("h", hist[n])):
            keys = np.array([_pack(k) for k in d], np.uint64)
            o = np.argsort(keys)
            arrays[f"{name}k{n}"] = keys[o]
            arrays[f"{name}{'b' if name else 'v'}{n}"] = np.array(
                list(d.values()), np.float32
            )[o]
    np.savez(tmp_path / "lm.npz", **arrays)
    lm = NgramLM(tmp_path / "lm.npz")
    pairs = [
        (tuple(int(x) for x in rng.integers(-1, 4, 2)), int(rng.integers(1, 4)))
        for _ in range(200)
    ]
    single = [lm.logp(h, t) for h, t in pairs]
    keys = [(_pack(h) << 11) | (t + 1) for h, t in pairs]
    assert np.allclose(lm.logp_keys(keys), single)
    # a few new keys at a time take the scalar path: the same numbers, bit for bit
    batched = lm.logp_keys(keys)
    lm.ikcache.clear()
    assert [v for i in range(0, len(keys), 2) for v in lm.logp_keys(keys[i : i + 2])] == batched


def _beam_reference(blank, topv, topi, lm, alpha, beta, beam, blank_skip=0.999, cand_floor=1e-3):
    """beam_search without the floor that skips hopeless prefixes (the older search)."""
    from indra_ocr.lm import _lse

    bits = lm.bits
    hmask = (1 << (bits * (lm.order - 1))) - 1
    beams = {(): (0.0, -math.inf, 0.0, (), 0)}
    for t in range(len(blank)):
        b0 = float(blank[t])
        if b0 > math.log(blank_skip):
            beams = {p: (_lse(e[0], e[1]) + b0, -math.inf, *e[2:]) for p, e in beams.items()}
            continue
        nxt = {}

        def add(prefix, b, nb, s, st, hk):
            old = nxt.get(prefix)
            nxt[prefix] = (
                (b, nb, s, st, hk)
                if old is None
                else (_lse(old[0], b), _lse(old[1], nb), *old[2:])
            )

        cands = [(int(c), float(v)) for c, v in zip(topi[t], topv[t]) if c != 0 and v >= math.log(cand_floor)]
        ext = []
        for prefix, (b, nb, s, st, hk) in beams.items():
            tot = _lse(b, nb)
            add(prefix, tot + b0, -math.inf, s, st, hk)
            last = prefix[-1] if prefix else None
            for c, lp in cands:
                if c == last:
                    add(prefix, -math.inf, nb + lp, s, st, hk)
                    new_from = b + lp
                else:
                    new_from = tot + lp
                if new_from != -math.inf:
                    ext.append((prefix, c, new_from, s, st, hk))
        lps = lm.logp_keys([(e[5] << bits) | (e[1] + 1) for e in ext])
        for (prefix, c, new_from, s, st, hk), lmp in zip(ext, lps):
            add(prefix + (c,), -math.inf, new_from, s + alpha * lmp + beta, st + (t,), ((hk << bits) | (c + 1)) & hmask)
        beams = dict(sorted(nxt.items(), key=lambda kv: -(_lse(kv[1][0], kv[1][1]) + kv[1][2]))[:beam])
    best = max(beams.items(), key=lambda kv: _lse(kv[1][0], kv[1][1]) + kv[1][2])
    return list(best[0]), list(best[1][3])


def test_skipping_hopeless_prefixes_keeps_the_result(tmp_path):
    rng = np.random.default_rng(1)
    V = 9
    toks = list(range(V - 1))
    keys = np.array([_pack((t,)) for t in toks], np.uint64)
    p = rng.dirichlet(np.ones(len(toks)))
    np.savez(
        tmp_path / "lm.npz",
        order=np.int64(1),
        vocab_size=np.int64(V),
        k0=keys,
        v0=np.log(p).astype(np.float32),
        hk0=np.array([0], np.uint64),
        hb0=np.array([0.0], np.float32),
    )
    lm = NgramLM(tmp_path / "lm.npz")
    for trial in range(200):
        T = int(rng.integers(3, 30))
        rows = rng.dirichlet(np.full(V, 0.3), size=T)
        lp = np.log(rows)
        topi = np.argsort(-lp, 1)[:, :6]
        topv = np.take_along_axis(lp, topi, 1)
        for beam in (1, 2, 4):
            args = (lp[:, 0], topv, topi, lm, 0.8, 0.5, beam)
            assert beam_search(*args) == _beam_reference(*args), (trial, beam)
