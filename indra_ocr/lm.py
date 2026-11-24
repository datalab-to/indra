"""Beam search with a token n-gram language model, for lines the recognizer is
unsure of.

The recognizer reads greedily: one best class per step.  On a line whose
confidence is under the gate, a CTC prefix beam search keeps several readings
and the language model breaks close calls between them:
score = log P_ctc + alpha * log P_lm + beta * tokens.  The recognizer's own
probabilities still decide, so a clearly written odd word survives.

lm.npz holds the model as a backoff table (no pickle): per history length n,
sorted keys of the (history, token) pairs the model holds, with their
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
SCALAR_KEYS = 4  # logp_keys looks up at most this many new keys one at a time


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

    def _logp_key(self, key: int) -> float:
        """logp_keys for one key, without the array set-up (the same table
        searches, and the same float additions in the same order)."""
        if not hasattr(self, "_hb0"):
            i = self._find(self.h_keys, 0)  # the empty history
            self._hb0 = float(self.h_vals[i]) if i >= 0 else 0.0
            self._logv = math.log(self.vocab_size)
        v = self.ikcache.get(key)
        if v is not None:
            return v
        b, N = self.bits, self.order
        ct = key & ((1 << b) - 1)
        hist = key >> b
        hkeys, pkeys = self.h_keys, self.p_keys
        # x[n]: the backoff weight of the length-n history, for n <= m (the
        # longest history the model has seen); m counts the seen lengths from 1.
        x = [self._hb0]
        hs = [0]
        for n in range(1, N):
            h = hist & ((1 << (b * n)) - 1)
            q = np.uint64(h | (n << TAG_SHIFT))
            j = int(hkeys.searchsorted(q))
            if j >= len(hkeys) or hkeys[j] != q:
                break
            x.append(float(self.h_vals[j]))
            hs.append(h)
        m = len(x) - 1
        acc = 0.0
        for n in range(m, -1, -1):
            q = np.uint64((hs[n] << b) | ct | (n << TAG_SHIFT))
            j = int(pkeys.searchsorted(q))
            if j < len(pkeys) and pkeys[j] == q:
                v = acc + float(self.p_vals[j])
                break
            acc += x[n]
        else:
            v = acc - self._logv
        if len(self.ikcache) < 1_000_000:
            self.ikcache[key] = v
        return v

    def logp_keys(self, keys: list[int]) -> list[float]:
        """logp for many (history, token) pairs, each given as one key: the
        last order - 1 history tokens and then the token, packed (``_pack``).
        Shorter histories are the key's low bits, so every history length
        is one vectorized search over all the pairs."""
        get = self.ikcache.get
        out = [get(k) for k in keys]
        miss = [i for i, v in enumerate(out) if v is None]
        if not miss:
            return out
        if len(miss) <= SCALAR_KEYS:
            for i in miss:
                out[i] = self._logp_key(keys[i])
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
    inf = math.inf
    exp, log = math.exp, math.log
    bits = lm.bits
    hmask = (1 << (bits * (lm.order - 1))) - 1
    # A prefix is a node of a token trie: node 0 is the empty prefix, node k
    # extends node parent[k] by token[k]; trie maps (parent, token) -> node.
    # (A tuple of tokens costs its length to hash and to extend.)
    trie: dict = {}
    parent, token = [-1], [-1]
    # Beams in score order: (node, log p ending in blank, log p ending in
    # non-blank, lm score, steps (linked: (step, earlier steps)), the prefix's
    # last order - 1 tokens packed, log p of the prefix).  The log-sums are
    # _lse, inlined.
    beams = [(0, 0.0, -inf, 0.0, None, 0, 0.0)]
    log_skip = log(blank_skip)
    log_floor = log(cand_floor)
    reach = beta + alpha * lm.max_logp
    blank, topv, topi = blank.tolist(), topv.tolist(), topi.tolist()
    for t, b0 in enumerate(blank):
        if b0 > log_skip:
            beams = [
                (p, tot + b0, -inf, s, st, hk, tot + b0)
                for p, b, nb, s, st, hk, tot in beams
            ]
            continue
        cands = [(c, v) for c, v in zip(topi[t], topv[t]) if c != 0 and v >= log_floor]
        # node -> [log p ending in blank, in non-blank, lm score, steps, history]
        nxt: dict = {}
        # Every beam's own entries first (the blank, and a repeat that collapses):
        # their scores only grow below, so with a full beam the lowest of them is
        # a floor; a new prefix that cannot reach it never enters the beam, and
        # its LM score is not looked up (the result is the same).
        for p, b, nb, s, st, hk, tot in beams:
            last = token[p]
            rep = -inf
            for c, lp in cands:
                if c == last:
                    rep = nb + lp
            nxt[p] = [tot + b0, rep, s, st, hk]
        floor = -inf
        if len(beams) >= beam:
            for e in nxt.values():
                x, y = e[0], e[1]
                if y == -inf:
                    v = x + e[2]
                elif x == -inf:
                    v = y + e[2]
                else:
                    m = x if x > y else y
                    v = m + log(exp(x - m) + exp(y - m)) + e[2]
                if v < floor or floor == -inf:
                    floor = v
        floor -= 1e-9  # float slack
        ext = []  # (parent node, class, log p of the new token, lm score, steps, history)
        for p, b, nb, s, st, hk, tot in beams:
            last = token[p]
            for c, lp in cands:
                # after a blank, the same class is a new token
                new_from = b + lp if c == last else tot + lp
                if new_from == -inf:
                    continue
                q = trie.get((p, c))
                old = nxt.get(q) if q is not None else None
                if old is not None:  # an existing beam: merges, its LM score stays
                    y = old[1]
                    if y == -inf:
                        old[1] = new_from
                    else:
                        m = y if y > new_from else new_from
                        old[1] = m + log(exp(y - m) + exp(new_from - m))
                elif new_from + s + reach >= floor:
                    ext.append((p, c, new_from, s, st, hk))
        if ext:
            lps = lm.logp_keys([(e[5] << bits) | (e[1] + 1) for e in ext])
            for (p, c, new_from, s, st, hk), lmp in zip(ext, lps):
                q = trie.get((p, c))
                if q is None:
                    q = trie[(p, c)] = len(parent)
                    parent.append(p)
                    token.append(c)
                nxt[q] = [
                    -inf,
                    new_from,
                    s + alpha * lmp + beta,
                    (t, st),
                    ((hk << bits) | (c + 1)) & hmask,
                ]
        scored = []
        for q, (x, y, s, st, hk) in nxt.items():
            if y == -inf:
                tot = x
            elif x == -inf:
                tot = y
            else:
                m = x if x > y else y
                tot = m + log(exp(x - m) + exp(y - m))
            scored.append((-(tot + s), (q, x, y, s, st, hk, tot)))
        scored.sort(key=_first)
        beams = [e for _, e in scored[:beam]]
    best = max(beams, key=_score)
    ids, steps = [], []
    q, st = best[0], best[4]
    while q > 0:
        ids.append(token[q])
        q = parent[q]
    while st is not None:
        steps.append(st[0])
        st = st[1]
    return ids[::-1], steps[::-1]


def _first(kv):
    return kv[0]


def _score(e):
    return e[6] + e[3]
