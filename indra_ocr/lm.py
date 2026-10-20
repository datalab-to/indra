"""Beam search with a token n-gram language model, for lines the recognizer is
unsure of.

The recognizer reads greedily: one best class per step.  On a line whose
confidence is under the gate, a CTC prefix beam search keeps several readings
and the language model breaks close calls between them:
score = log P_ctc + alpha * log P_lm + beta * tokens.  The recognizer's own
probabilities still decide, so a clearly written odd word survives.

lm.npz holds the model as a backoff table (no pickle): per history length n,
sorted keys of the (history, token) pairs seen in training with their
log-probabilities, and sorted history keys with their backoff weights.  A key
packs tokens ``bits`` bits each (0 = line start, t + 1 = token t): 11 unless
the file says more (a larger vocab), with order x bits at most TAG_SHIFT.
"""

from __future__ import annotations

import math

import numpy as np

from . import native

BITS = 11  # per token, unless lm.npz holds "bits"
TAG_SHIFT = 61  # the history length goes in the top 3 bits
BOS = -1


def _pack(tokens, bits: int = BITS) -> int:
    k = 0
    for t in tokens:
        k = (k << bits) | (0 if t == BOS else t + 1)
    return k


class NgramLM:
    def __init__(self, path):
        z = np.load(path)
        self.order = int(z["order"])
        self.vocab_size = int(z["vocab_size"])
        self.bits = int(z["bits"]) if "bits" in z else BITS
        if self.order * self.bits > TAG_SHIFT:
            raise ValueError(f"order {self.order} x {self.bits} bits do not fit a key")
        self.k = [z[f"k{n}"] for n in range(self.order)]
        self.v = [z[f"v{n}"] for n in range(self.order)]
        self.hk = [z[f"hk{n}"] for n in range(self.order)]
        self.hb = [z[f"hb{n}"] for n in range(self.order)]
        self.cache: dict = {}
        self.ikcache: dict = {}
        # All history lengths in one sorted table, the length in the top bits
        # (keys use at most order x bits), so a step needs one search per table.
        tags = [np.uint64(n) << np.uint64(TAG_SHIFT) for n in range(self.order)]
        for name, keys, vals in (("p", self.k, self.v), ("h", self.hk, self.hb)):
            allk = np.concatenate([k | t for k, t in zip(keys, tags)])
            allv = np.concatenate(vals).astype(np.float64)
            o = np.argsort(allk)
            setattr(self, f"{name}_keys", allk[o])
            setattr(self, f"{name}_vals", allv[o])
        n = np.arange(self.order, dtype=np.uint64)
        self._tags = (n << np.uint64(TAG_SHIFT))[None, :]
        self._masks = ((np.uint64(1) << (np.uint64(self.bits) * n)) - np.uint64(1))[
            None, :
        ]
        self._native = None  # the C++ model over the same tables, made on first use

    def __getstate__(self) -> dict:
        # the native model points into this process's tables: rebuilt on demand
        return {**self.__dict__, "_native": None}

    def native_lm(self):
        """The native kernels' NgramLM over these tables (same values as logp_keys)."""
        if self._native is None:
            self._native = native.lib.NgramLM(
                self.order,
                self.bits,
                self.vocab_size,
                self.p_keys,
                self.p_vals,
                self.h_keys,
                self.h_vals,
                self.max_logp,
            )
        return self._native

    @property
    def max_logp(self) -> float:
        """An upper bound of every value logp and logp_keys return: the largest
        log-probability plus every positive backoff weight a back-off can add."""
        if not hasattr(self, "_max_logp"):
            top = max(float(self.p_vals.max()), -math.log(self.vocab_size))
            bo = sum(max(0.0, float(h.max())) for h in self.hb if len(h))
            self._max_logp = top + bo
        return self._max_logp

    @staticmethod
    def _find(keys: np.ndarray, key: int) -> int:
        i = int(np.searchsorted(keys, np.uint64(key)))
        return i if i < len(keys) and int(keys[i]) == key else -1

    def logp(self, hist: tuple, t: int) -> float:
        ck = (hist, t)
        v = self.cache.get(ck)
        if v is not None:
            return v
        top = min(len(hist), self.order - 1)
        hs = [tuple(hist[len(hist) - n :]) if n else () for n in range(top + 1)]
        m = 0  # the longest history the model has seen
        for n in range(1, top + 1):
            if self._find(self.hk[n], _pack(hs[n], self.bits)) < 0:
                break
            m = n
        acc, v = 0.0, None
        for n in range(m, -1, -1):
            i = self._find(self.k[n], _pack(hs[n] + (t,), self.bits))
            if i >= 0:
                v = acc + float(self.v[n][i])
                break
            acc += float(self.hb[n][self._find(self.hk[n], _pack(hs[n], self.bits))])
        if v is None:
            v = acc - math.log(self.vocab_size)
        if len(self.cache) < 1_000_000:
            self.cache[ck] = v
        return v

    def logp_keys(self, keys: list[int]) -> list[float]:
        """logp for many (history, token) pairs, each given as one key: the
        last order - 1 history tokens and then the token, packed (``_pack``).
        Shorter histories are the key's low bits, so every history length
        is one vectorized search over all the pairs."""
        out = [self.ikcache.get(k) for k in keys]
        miss = [i for i, v in enumerate(out) if v is None]
        if not miss:
            return out
        N, P = self.order, len(miss)
        full = np.array([keys[i] for i in miss], np.uint64)[:, None]
        b = np.uint64(self.bits)
        ct = full & np.uint64((1 << self.bits) - 1)
        hq = (full >> b) & self._masks
        pq = (hq << b) | ct | self._tags
        hq = hq | self._tags
        found = []
        for table, vals, q in (
            (self.h_keys, self.h_vals, hq),
            (self.p_keys, self.p_vals, pq),
        ):
            j = np.minimum(np.searchsorted(table, q), len(table) - 1)
            hit = table[j] == q
            found.append((hit, np.where(hit, vals[j], 0.0)))
        (h_ok, hb), (p_ok, pv) = found
        h_ok[:, 0] = True
        # m: the longest history length the model has seen, per pair.  The
        # value comes from the longest seen history n <= m that saw the token,
        # plus the backoff weights of the longer seen histories (n, m]; with
        # no such n, all backoffs and a uniform floor.
        m = np.cumprod(h_ok[:, 1:], axis=1).sum(1) if N > 1 else np.zeros(P, int)
        valid = np.arange(N)[None, :] <= m[:, None]
        hitm = valid & p_ok
        has = hitm.any(1)
        chosen = N - 1 - np.argmax(hitm[:, ::-1], axis=1)
        suffix = np.zeros((P, N + 1))
        suffix[:, :N] = np.cumsum((hb * valid)[:, ::-1], axis=1)[:, ::-1]
        rows = np.arange(P)
        res = np.where(
            has,
            suffix[rows, chosen + 1] + pv[rows, chosen],
            suffix[:, 0] - math.log(self.vocab_size),
        )
        for r, i in enumerate(miss):
            out[i] = float(res[r])
            if len(self.ikcache) < 1_000_000:
                self.ikcache[keys[i]] = out[i]
        return out


def _lse(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_search(
    blank: np.ndarray,
    topv: np.ndarray,
    topi: np.ndarray,
    lm: NgramLM,
    alpha: float = 0.5,
    beta: float = 1.5,
    beam: int = 8,
    blank_skip: float = 0.999,
    cand_floor: float = 1e-3,
) -> tuple[list[int], list[int]]:
    """CTC prefix beam search over a line's steps.  blank: [T] blank
    log-probabilities; topv / topi: [T, k] the k best classes per step (blank
    included) and their log-probabilities; a class under ``cand_floor``
    probability at a step is not tried there.  Returns (token ids, the step
    where each token was first emitted).  With the native kernels the same
    search runs in C++ (same result: the same float64 steps and tie order)."""
    if native.lib is not None and isinstance(lm, NgramLM):
        try:
            return native.lib.beam_search(
                np.ascontiguousarray(blank, np.float32),
                np.ascontiguousarray(topv, np.float32),
                np.ascontiguousarray(topi, np.int64),
                lm.native_lm(),
                alpha,
                beta,
                beam,
                blank_skip,
                cand_floor,
            )
        except Exception as e:  # noqa: BLE001  (the search below runs instead)
            native.failed("beam_search", e)
    bits = lm.bits
    hmask = (1 << (bits * (lm.order - 1))) - 1
    # prefix -> (log p ending in blank, log p ending in non-blank, lm score,
    # steps, the prefix's last order - 1 tokens packed)
    beams: dict = {(): (0.0, -math.inf, 0.0, (), 0)}
    log_skip = math.log(blank_skip)
    log_floor = math.log(cand_floor)
    for t in range(len(blank)):
        b0 = float(blank[t])
        if b0 > log_skip:
            beams = {
                p: (_lse(e[0], e[1]) + b0, -math.inf, e[2], e[3], e[4])
                for p, e in beams.items()
            }
            continue
        nxt: dict = {}

        def add(prefix, b, nb, s, st, hk):
            old = nxt.get(prefix)
            if old is None:
                nxt[prefix] = (b, nb, s, st, hk)
            else:
                nxt[prefix] = (_lse(old[0], b), _lse(old[1], nb), *old[2:])

        cands = [
            (int(c), float(v))
            for c, v in zip(topi[t], topv[t])
            if c != 0 and v >= log_floor
        ]
        # Every beam's own entries first (the blank, and a repeat that collapses):
        # their scores only grow below, so with a full beam the lowest of them is
        # a floor; a new prefix that cannot reach it never enters the beam, and
        # its LM score is not looked up (the result is the same).
        for prefix, (b, nb, s, st, hk) in beams.items():
            add(prefix, _lse(b, nb) + b0, -math.inf, s, st, hk)
            last = prefix[-1] if prefix else None
            for c, lp in cands:
                if c == last:
                    add(prefix, -math.inf, nb + lp, s, st, hk)
        floor = (
            min(_lse(e[0], e[1]) + e[2] for e in nxt.values())
            if len(beams) >= beam
            else -math.inf
        )
        reach = beta + alpha * lm.max_logp
        ext = []  # (prefix, class, log p of the new token, lm score, steps, history)
        for prefix, (b, nb, s, st, hk) in beams.items():
            tot = _lse(b, nb)
            last = prefix[-1] if prefix else None
            for c, lp in cands:
                # after a blank, the same class is a new token
                new_from = b + lp if c == last else tot + lp
                if new_from == -math.inf:
                    continue
                q = prefix + (c,)
                if q in nxt:  # an existing beam: merges, its LM score stays
                    add(q, -math.inf, new_from, s, st, hk)
                elif new_from + s + reach >= floor - 1e-9:  # float slack
                    ext.append((prefix, c, new_from, s, st, hk))
        lps = lm.logp_keys([(e[5] << bits) | (e[1] + 1) for e in ext]) if ext else []
        for (prefix, c, new_from, s, st, hk), lmp in zip(ext, lps):
            add(
                prefix + (c,),
                -math.inf,
                new_from,
                s + alpha * lmp + beta,
                st + (t,),
                ((hk << bits) | (c + 1)) & hmask,
            )
        beams = dict(
            sorted(nxt.items(), key=lambda kv: -(_lse(kv[1][0], kv[1][1]) + kv[1][2]))[
                :beam
            ]
        )
    best = max(beams.items(), key=lambda kv: _lse(kv[1][0], kv[1][1]) + kv[1][2])
    return list(best[0]), list(best[1][3])
