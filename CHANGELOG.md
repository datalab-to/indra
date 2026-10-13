# Changelog

## 0.3.0

Models v3 (Hugging Face `datalab-to/indra-ocr`, tag `v3`; `v3.1` adds the math models).

- **Viewer.** A Datalab-styled page with the pipeline options in the page
  (source, math, headers and footers, reading order, OCR inside images, dpi,
  page limit), show/hide for lines, blocks, block labels and reading order, and
  tabs for the rendered markdown, its source, the lines and the JSON.
- `convert_many` no longer hangs when its worker processes cannot start (with
  the spawn start method, code not under `if __name__ == "__main__":`, or code
  from stdin): it warns and reads the remaining inputs in the calling process.
- **Math (opt-in).** `--math` / `load(math=True)`: a line classifier
  (`mathcls.onnx`) and a math recognizer (`math_enc` / `math_dec_init` /
  `math_dec_step.onnx`, attention decoder with cached keys and values) write
  lines with math as text with inline LaTeX. A reading replaces a line only
  with math in it and a mean token probability of 0.9 or more. Omni Parse
  Bench: headline 62.1 -> 66.8, text 41.4 -> 57.5, math tests 14.3 -> 56.5,
  tables 56.4 -> 54.8; median page time 421 -> 873 ms (GH200, 1 thread).
  Off by default; the math models download only when it is on.

- **Orientation.** `orient.onnx` turns a page that is turned by a quarter or
  half turn upright before detection (`page.rotation`). A turn needs at least
  95% from the model. Omni Parse Bench: rotated content 3.5% -> 24.3%.
- **All cores.** The CLI and the new `indra_ocr.convert_many` run one
  single-threaded worker process per usable core, one page per task (a single
  PDF too); `--cores` limits them. 40 documents: 30.0 s on 1 core, 8.7 s on 4,
  2.8 s on 16 (one process with ORT threads gained at most 1.2x beyond 4
  cores). The CLI's `--threads` is now an alias of `--cores`.
- **Deskew.** A page tilted by up to 5 degrees is straightened before
  detection (`page.skew`, about 8 ms a page); boxes are mapped back to the
  page as turned. Omni Parse Bench rotation_skew suite: 60.0 -> 76.4.
- **Low-confidence lines get no text.** A recognized line under 0.5 mean
  confidence (empty form fields, checkboxes, rules read as "-", "I", "1")
  keeps its box but no text.
- **Orientation from a broken text layer.** A PDF page OCR'd for a broken font
  map is not turned when its glyphs run upright (orient.onnx turns about half
  of upright Arabic and Hebrew pages).
- **CJK markdown.** Lines of Chinese or Japanese join without a space; a
  bullet detected as its own line starts a list item.
- Table cells escape only `&`, `<` and `>` (no `&#x27;`).
- **Text inside diagrams and charts.** The detector finds 51% of the text
  inside pictures (12.5% before).
- **Right-to-left text layers.** Runs in a right-to-left line are ordered right
  to left.
- **OCR only where it can read.** A text layer in a script the recognizer
  cannot read is kept even when a trust check fails (not for broken font
  maps), and empty lines on such pages are not OCR'd into junk.
- **Layout blocks.** `page.blocks` lists the layout blocks (label, box, text)
  in reading order; the CLI's JSON output includes them.
- **`--keep-furniture`** keeps page headers and footers in the markdown.
- **Optional models:** `script.onnx` + `rec_cjk.onnx` route lines with Chinese
  characters to a Chinese recognizer; `mask.onnx` (CenterMask) masks
  neighbouring lines before recognition. Both run only when present.
- `bench/opb_indra.py`: Omni Parse Bench runner.
- **Robust inputs** (release test: 4,340 benchmark pages, 1,177 documents with
  16,017 pages, 37 edge cases): the file type comes from the file's bytes;
  multi-page TIFFs give every page; pages above 100 MP render at a lower dpi;
  unreadable, corrupt or password-protected files raise `InputError`
  (`password=` / `--password`); a page that fails gives an empty page with
  `source == "error"` instead of ending the document; the CLI continues past
  failed files and exits 1. Text outside a PDF's crop box is not read.
- **Faster on dense pages:** reading-order clustering is O(n^2) instead of
  O(n^3) (a 1,984-line page: 53 s -> 0.45 s).
- **Bounded memory:** ONNX Runtime's CPU arena is off (it grew without bound
  on changing batch shapes; a 40-page large scan: 4.9 GB -> 1.4 GB).

## 0.2.0

Models v2. Blocks and table heads from average-linkage
clustering.

## 0.1.0

First package: ONNX pipeline, PDF text-layer mode, CLI, viewer.
