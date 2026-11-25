"""Audit a document's tables end to end: does the markdown keep every character
and number the text layer holds, and is every table structurally sound?

    pip install pymupdf pdfplumber          # the two independent readers
    python bench/audit_tables.py report.pdf
    python bench/audit_tables.py report.pdf --json run1.json
    python bench/audit_tables.py report.pdf --compare run1.json

The reference is PyMuPDF's own read of the PDF text layer, so nothing here needs
hand labels.  Three layers:

  content   characters of the text layer against the markdown's (whitespace and
            markup removed), and the same for number tokens and digit characters
            -- a finance table that loses one digit is a wrong number, not a
            formatting nit
  structure every table's cells on a grid: a colspan or rowspan that swallows a
            neighbour, a row whose width disagrees with the table's
  detection Table blocks against pdfplumber's ruled-line table finder (page level:
            a finder that splits one ruled table into two is still right that the
            page has a table)

With ``--baseline`` (the default) the document is also read with INDRA_TSR=0, the
geometric tables, and both losses are reported: the difference tells whether a
loss comes from the table model or from the page pipeline.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

DASH = {"\u2014": "-", "\u2013": "-", "\u2212": "-", "\u2015": "-"}
QUOTES = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
NUM = re.compile(r"\d[\d,]*\.?\d*")


def norm(s: str) -> str:
    """Comparable characters: NFKC, one dash and one quote, no whitespace."""
    s = unicodedata.normalize("NFKC", s)
    for a, b in (DASH | QUOTES).items():
        s = s.replace(a, b)
    return "".join(c for c in s if not c.isspace())


def strip_md(md: str) -> str:
    s = re.sub(r"<[^>]+>", "", md)
    s = html.unescape(s)
    s = re.sub(r"(?m)^#{1,6}\s*", "", s)
    s = re.sub(r"(?m)^\s*[-+*]\s+", "", s)
    return s.replace("|", "").replace("`", "")


def plain(md: str) -> str:
    """Markdown with tags and entities gone, whitespace kept: number tokens need
    their boundaries."""
    s = re.sub(r"<[^>]+>", " ", md)
    s = html.unescape(s)
    s = re.sub(r"(?m)^#{1,6}\s*", "", s)
    return unicodedata.normalize("NFKC", s)


def numbers(s: str) -> Counter:
    """Number tokens with their comma groups kept: 1,594,099,177.35 is one."""
    return Counter(re.sub(r"[,\s]", "", t) for t in NUM.findall(plain(s)))


def digits(s: str) -> Counter:
    """Just the digit characters: a number split over two cells still counts."""
    return Counter(c for c in norm(s) if c.isdigit())


class TableParser(HTMLParser):
    """Tables as rows of (text, rowspan, colspan)."""

    def __init__(self):
        super().__init__()
        self.tables: list[list[list]] = []
        self.cur: list[list] | None = None
        self.row: list | None = None
        self.buf: list[str] | None = None
        self.rs = self.cs = 1

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self.cur = []
        elif tag == "tr" and self.cur is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.buf = []
            self.rs = int(a.get("rowspan") or 1)
            self.cs = int(a.get("colspan") or 1)
        elif tag == "br" and self.buf is not None:
            self.buf.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.buf is not None:
            self.row.append(("".join(self.buf).strip(), self.rs, self.cs))
            self.buf = None
        elif tag == "tr" and self.row is not None:
            self.cur.append(self.row)
            self.row = None
        elif tag == "table" and self.cur is not None:
            self.tables.append(self.cur)
            self.cur = None

    def handle_data(self, data):
        if self.buf is not None:
            self.buf.append(data)


def grid_of(rows: list[list]) -> list[list]:
    """Cells on a grid, a rowspan leaving a marker in the rows it covers."""
    out: list[list] = []
    pending: dict[int, int] = {}
    for row in rows:
        line: list = []
        c = 0
        cells = list(row)
        while cells:
            if pending.get(c, 0) > 0:
                line.append(None)
                pending[c] -= 1
                c += 1
                continue
            text, rs, cs = cells.pop(0)
            for k in range(cs):
                line.append(text if k == 0 else "")
                if rs > 1:
                    pending[c + k] = rs - 1
            c += cs
        out.append(line)
    return out


def table_stats(md: str) -> list[dict]:
    """Per table: rows, width, empty cells, ragged rows, spanning cells."""
    tp = TableParser()
    tp.feed(md)
    stats = []
    for rows in tp.tables:
        if not rows:
            continue
        grid = grid_of(rows)
        widths = [len(r) for r in grid]
        stats.append(
            {
                "rows": len(rows),
                "cols": max(widths) if widths else 0,
                "cells": sum(len(r) for r in rows),
                "empty": sum(1 for r in rows for (t, _, _) in r if not t),
                "span": sum(1 for r in rows for (_, rs, cs) in r if rs > 1 or cs > 1),
                "ragged": sum(1 for w in widths if w != max(widths)),
            }
        )
    return stats


def run_indra(pdf: Path, out: Path, dpi: float, env_extra: dict | None = None) -> list[dict]:
    """The document as one JSON record per page, from the CLI itself."""
    env = dict(os.environ)
    env.update(env_extra or {})
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "indra_ocr",
        str(pdf),
        "--format",
        "json",
        "--keep-furniture",
        "--dpi",
        str(dpi),
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
    path = next(out.glob("*.jsonl"))
    return [json.loads(ln) for ln in path.open()]


def audit(pdf: Path, pages: list[dict], geo: list[dict] | None, dpi: float) -> dict:
    import pymupdf

    doc = pymupdf.open(pdf)
    tot = Counter()
    per_page = []
    for rec in pages:
        i = rec["page"]
        truth = Counter(norm(doc[i].get_text("text")))
        read_raw = "\n".join(ln["text"] for ln in rec["lines"])
        read = Counter(norm(read_raw))
        md = Counter(norm(strip_md(rec["markdown"])))
        miss, extra = read - md, md - read
        tn, mn = numbers(read_raw), numbers(rec["markdown"])
        num_miss = tn - mn
        tdn, mdn = digits(read_raw), digits(rec["markdown"])
        dig_miss = tdn - mdn
        geo_num = geo_dig = 0
        if geo:
            g = Counter(norm(strip_md(geo[i]["markdown"])))
            geo_num = sum((tn - numbers(geo[i]["markdown"])).values())
            geo_dig = sum((tdn - digits(geo[i]["markdown"])).values())
        tabs = table_stats(rec["markdown"])
        n = sum(truth.values())
        tot.update(
            {
                "chars": n,
                "miss": sum(miss.values()),
                "extra": sum(extra.values()),
                "nums": sum(tn.values()),
                "num_miss": sum(num_miss.values()),
                "num_miss_geo": geo_num,
                "digs": sum(tdn.values()),
                "dig_miss": sum(dig_miss.values()),
                "dig_miss_geo": geo_dig,
                "tables": len(tabs),
                "t_cells": sum(s["cells"] for s in tabs),
                "t_empty": sum(s["empty"] for s in tabs),
                "t_span": sum(s["span"] for s in tabs),
                "t_ragged": sum(1 for s in tabs if s["ragged"]),
            }
        )
        tot["clean_chars"] += 0 if miss or extra else 1
        tot["clean_nums"] += 0 if num_miss else 1
        per_page.append(
            {
                "page": i + 1,
                "source": rec["source"],
                "chars": n,
                "miss": sum(miss.values()),
                "num": sum(tn.values()),
                "num_miss": sum(num_miss.values()),
                "num_miss_geo": geo_num,
                "num_miss_list": [x for x, _ in num_miss.most_common(6)],
                "tables": len(tabs),
                "t_empty": sum(s["empty"] for s in tabs),
                "t_cells": sum(s["cells"] for s in tabs),
            }
        )
    return {"summary": dict(tot), "pages": per_page}


def detection(pdf: Path, pages: list[dict], dpi: float) -> dict | None:
    """Page level: does each page have a table, indra vs pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        return None
    mine: set[int] = set()
    theirs: set[int] = set()
    with pdfplumber.open(pdf) as doc:
        for rec in pages:
            i = rec["page"]
            if any(b["label"] == "Table" for b in rec["blocks"]):
                mine.add(i + 1)
            try:
                if doc.pages[i].find_tables():
                    theirs.add(i + 1)
            except Exception:  # noqa: BLE001  (a page pdfplumber cannot read)
                continue
    return {
        "mine": len(mine),
        "theirs": len(theirs),
        "both": len(mine & theirs),
        "mine_only": sorted(mine - theirs),
        "theirs_only": sorted(theirs - mine),
    }


def report(res: dict, det: dict | None, pages: int, compare: dict | None) -> None:
    s = res["summary"]
    pc = lambda v, d: f"{v}  ({v / max(d, 1) * 100:.3f}%)"  # noqa: E731
    lines = [
        f"pages                          {pages}",
        f"chars in the text layer        {s['chars']}",
        f"  lost in markdown             {pc(s['miss'], s['chars'])}",
        f"  added in markdown            {s['extra']}",
        f"  pages with no loss           {s['clean_chars']}/{pages}",
        f"number tokens                  {s['nums']}",
        f"  lost in markdown             {pc(s['num_miss'], s['nums'])}",
        f"  lost, geometric tables       {pc(s['num_miss_geo'], s['nums'])}",
        f"digit characters               {s['digs']}",
        f"  lost in markdown             {pc(s['dig_miss'], s['digs'])}",
        f"  lost, geometric tables       {pc(s['dig_miss_geo'], s['digs'])}",
        f"  pages with no number lost    {s['clean_nums']}/{pages}",
        f"tables                         {s['tables']}",
        f"  cells                        {s['t_cells']}, empty {pc(s['t_empty'], s['t_cells'])}",
        f"  spanning cells               {s['t_span']}",
        f"  tables with a ragged row     {s['t_ragged']}/{s['tables']}",
    ]
    if det:
        lines += [
            f"table detection (page level)   indra {det['mine']}, pdfplumber {det['theirs']},"
            f" both {det['both']}",
            f"  indra only                   {det['mine_only']}",
            f"  pdfplumber only              {det['theirs_only']}",
        ]
    print("\n".join(lines))

    worst = sorted(res["pages"], key=lambda r: -r["num_miss"])[:10]
    if any(w["num_miss"] for w in worst):
        print("\ntop pages by lost numbers:")
        for w in worst:
            if not w["num_miss"]:
                break
            print(
                f"  p{w['page']:3d} {w['source']:6s} {w['num_miss']:3d}/{w['num']:3d} lost"
                f"  (geometric {w['num_miss_geo']:3d})  {w['num_miss_list']}"
            )
    if compare:
        old = compare["summary"]
        print("\nagainst the previous run:")
        for key, label in (
            ("miss", "chars lost"),
            ("num_miss", "number tokens lost"),
            ("dig_miss", "digit chars lost"),
            ("t_empty", "empty table cells"),
            ("t_ragged", "ragged tables"),
        ):
            a, b = old.get(key, 0), s.get(key, 0)
            print(f"  {label:24s} {a:6d} -> {b:6d}  ({b - a:+d})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--dpi", type=float, default=200.0, help="the render scale indra reads at (default 200)")
    ap.add_argument("--no-baseline", action="store_true", help="skip the INDRA_TSR=0 comparison run")
    ap.add_argument("--no-detect", action="store_true", help="skip the pdfplumber table detection check")
    ap.add_argument("--json", type=Path, help="write the per-page numbers here")
    ap.add_argument("--compare", type=Path, help="a --json file from an earlier run to diff against")
    a = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pages = run_indra(a.pdf, tmp / "tsr", a.dpi)
        geo = None if a.no_baseline else run_indra(a.pdf, tmp / "geo", a.dpi, {"INDRA_TSR": "0"})
        res = audit(a.pdf, pages, geo, a.dpi)
        det = None if a.no_detect else detection(a.pdf, pages, a.dpi)

    compare = json.loads(a.compare.read_text()) if a.compare else None
    report(res, det, len(pages), compare)
    if a.json:
        a.json.write_text(json.dumps({"summary": res["summary"], "pages": res["pages"], "detection": det}, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
