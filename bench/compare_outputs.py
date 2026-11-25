"""Compare two runs of bench/opb_indra.py --dump (or bench/profile_stages.py --dump) page by page.

    python bench/compare_outputs.py runs/opb_auto_python runs/opb_auto_native

Reports how many pages give the same markdown, the same lines (text, and
boxes and confidences within a tolerance), the same blocks, rotation and
skew, lists the pages that differ, and compares the scores and latency of the
two runs (their summary.json and pages.jsonl).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(run: Path) -> dict:
    with open(run / "outputs.jsonl", encoding="utf-8") as f:
        return {d["page_id"]: d for d in map(json.loads, f)}


def close(a, b, tol: float) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def diff(a: dict, b: dict, box_tol: float, conf_tol: float) -> list[str]:
    """What differs between two outputs of one page."""
    if "error" in a or "error" in b:
        return [] if a.get("error") == b.get("error") else ["error"]
    out = []
    if a["markdown"] != b["markdown"]:
        out.append("markdown")
    la, lb = a["lines"], b["lines"]
    if [ln[1] for ln in la] != [ln[1] for ln in lb]:
        out.append("line text")
    elif not all(close(x[0], y[0], box_tol) for x, y in zip(la, lb)):
        out.append("line boxes")
    elif not all(abs(x[2] - y[2]) <= conf_tol for x, y in zip(la, lb)):
        out.append("line conf")
    ba, bb = a["blocks"], b["blocks"]
    if [(x["label"], x["text"]) for x in ba] != [
        (y["label"], y["text"]) for y in bb
    ] or not all(close(x["bbox"], y["bbox"], box_tol) for x, y in zip(ba, bb)):
        out.append("blocks")
    for k in ("rotation", "skew", "source"):
        if a.get(k) != b.get(k):
            out.append(k)
    return out


def changed_chars(a: dict, b: dict) -> tuple[int, int]:
    """(characters of a's line texts, characters changed to give b's): the
    differing spans of the two pages' texts, lines joined in reading order."""
    import difflib

    ta = "\n".join(ln[1] for ln in a.get("lines", []))
    tb = "\n".join(ln[1] for ln in b.get("lines", []))
    if ta == tb:
        return len(ta), 0
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    return len(ta), sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def summary(run: Path) -> dict | None:
    s = json.loads((run / "summary.json").read_text())
    if "grade" not in s:  # a bench/profile_stages.py run: no scores
        return None
    u = s["grade"]["scorers"]["unit_tests"]  # bench/opb_indra.py: the benchmark's grade
    return {
        "headline": 100 * u["headline"],
        **{k: 100 * v["score"] for k, v in u["families"].items()},
        "ms_median": s["indra_ocr"]["ms_median"],
        "ms_p90": s["indra_ocr"]["ms_p90"],
    }


def mean_ms(run: Path) -> float:
    with open(run / "pages.jsonl") as f:
        ms = [json.loads(line)["ms"] for line in f]
    return sum(ms) / max(len(ms), 1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("a", type=Path, help="the reference run")
    ap.add_argument("b", type=Path, help="the run to compare")
    ap.add_argument("--box-tol", type=float, default=0.0, help="pixels")
    ap.add_argument("--conf-tol", type=float, default=0.0)
    ap.add_argument("--show", type=int, default=20, help="pages to list")
    ap.add_argument(
        "--chars", action="store_true", help="also count the characters that differ (GPU and other gated runs)"
    )
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    common = sorted(set(a) & set(b))
    tol = (args.box_tol, args.conf_tol)
    bad = {p: d for p in common if (d := diff(a[p], b[p], *tol))}
    kinds: dict[str, int] = {}
    for d in bad.values():
        for k in d:
            kinds[k] = kinds.get(k, 0) + 1
    only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
    print(f"pages compared: {len(common)} (only in a: {only_a}, only in b: {only_b})")
    print(f"pages identical: {len(common) - len(bad)}; differing: {len(bad)} {kinds}")
    for p in list(bad)[: args.show]:
        print(f"  {p}: {', '.join(bad[p])}")
    if args.chars:
        n = c = 0
        worst = []
        for p in common:
            k, d = changed_chars(a[p], b[p])
            n, c = n + k, c + d
            if d:
                worst.append((d / max(k, 1), d, p))
        print(f"characters changed: {c} of {n} ({100 * c / max(n, 1):.3f}%)")
        for r, d, p in sorted(worst, reverse=True)[: args.show]:
            print(f"  {100 * r:6.2f}% ({d} chars) {p}")
    sa, sb = summary(args.a), summary(args.b)
    if sa and sb:
        print(f"{'':12s} {'a':>9s} {'b':>9s}")
        for k in sa:
            print(
                f"{k:12s} {sa[k]:9.4f} {sb[k]:9.4f}"
                if k.startswith(("head", "text", "tab", "lay"))
                else f"{k:12s} {sa[k]:9.1f} {sb[k]:9.1f}"
            )
    ma, mb = mean_ms(args.a), mean_ms(args.b)
    print(f"{'ms_mean':12s} {ma:9.1f} {mb:9.1f}")
    if sa and sb:
        median, p90 = sa["ms_median"] / sb["ms_median"], sa["ms_p90"] / sb["ms_p90"]
        print(f"speed: median {median:.2f}x, mean {ma / mb:.2f}x, p90 {p90:.2f}x")
    else:
        print(f"speed: mean {ma / mb:.2f}x")


if __name__ == "__main__":
    main()
