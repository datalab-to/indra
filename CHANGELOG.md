# Changelog

## Unreleased

Same models (Hugging Face tag `v1.1.0`).

- **Optional C++ kernels** (`pip install "indra-ocr[native]"`, the new `indra-ocr-native` package in `native/`). The recognizers' CTC head, the language-model beam search, the PDF text layer's per-character work, line splitting and block clustering run in C++. On all 2,937 OmniParseBench pages the scores, markdown, lines, boxes and blocks are unchanged; pages read about 1.2x faster on one core (1.4x for Chinese, Japanese and Korean scans, 1.3x for the Indic and Southeast Asian scripts). The recognizer graphs are cut at the CTC head when they load, so the models are unchanged.
- **The Python code stays.** Without the package, or with `INDRA_OCR_NATIVE=0`, every step runs as before. A kernel that raises falls back to the Python code for that call (with one warning); a recognizer whose cut graph fails reads with its full graph; when `convert_many`'s workers die, the rest is read without the kernels.
- **indra-ocr-native 0.2.** The CTC head is written with Google Highway, which picks the CPU's best SIMD target at run time: SSE2 to AVX-512 on x86 (AVX2 with MSVC), NEON and SVE on Arm (which ran plain C++). C++20, link-time optimization. The head is 1.4-1.5x faster than 0.1's SSE2 code; pages read 1.02-1.03x faster on one core and at the same speed with a worker per core, with the same outputs. `indra_ocr.native.status()` names the SIMD target. Threads (OpenMP) for the head made pages slower next to ONNX Runtime's own threads, so the kernels stay single-threaded.
- `bench/`: `ab_builds.py` (two builds of the package on the same pages, in turns; `--ids` for a subset), `opb_indra.py --dump` (page outputs) and `--pin` (worker per CPU), `compare_outputs.py` (two runs page by page), `throughput.py` (the throughput table) and `ab_latency.py` (with and without the kernels, paired page by page).

## 1.1.2

Same models (Hugging Face tag `v1.1.0`).

- **PDFs with legacy-font text layers.** Many Hindi PDFs use legacy fonts (Kruti Dev, DevLys), whose text layer stores Latin codes in place of Devanagari, so Indra read them as Latin garbage. Before trusting a text layer, Indra now checks the script of a few of its lines against the line images; when the layer is Latin while the page shows another script, OCR reads the page. On such Hindi pages, word accuracy goes from 0.01-0.20 to 0.85-0.89. The same check covers other legacy fonts, such as Bijoy for Bangla. It adds about 5 ms per text-layer page.

## 1.1.1

Same models (Hugging Face tag `v1.1.0`).

- **Right-to-left text layers in visual order** ([#11](https://github.com/datalab-to/indra/issues/11)). Some PDF producers, Chrome among them, store Arabic-script letters partly in visual order, so their text layer read letter-reversed, with spaces inside words. Such a layer is no longer trusted, and OCR reads the page instead. On Chrome-printed Wikipedia pages, word recall goes from 0.17 to 1.00 (Arabic), 0.25 to 0.89 (Persian), 0.22 to 0.95 (Pashto) and 0.28 to 0.98 (Urdu). Hebrew and Yiddish layers in logical order still read as text.
- A right-to-left page is no longer mistaken for vertical writing.
- `examples/quickstart.ipynb` downloads two sample pages (`examples/samples/`) when you have no `scan.png` or `equations.png`, so it runs as is (also in Colab).

## 1.1.0

Models from Hugging Face tag `v1.1.0`. On OmniParseBench (the benchmark's own harness), the headline goes from 62.3 to 66.7, and from 67.0 to 71.4 with `--math`. Handwriting tests 8.3% -> 25.2%, rotated pages 25.3% -> 50.3%, tables 56.3 -> 61.1.

- **Handwriting.** The line detector and the Latin recognizer are trained on real handwritten and filled-form pages.
- **80+ languages on scans.** New recognizers for Chinese, Japanese and Korean; for right-to-left scripts (Arabic, Persian, Urdu, Pashto, Hebrew); and for other scripts (the Indic scripts, Thai, Lao, Khmer, Myanmar, Sinhala, Georgian, Armenian, Ethiopic). A line classifier sends each line to its recognizer, and n-gram language models re-read unsure lines. The Latin recognizer adds the letters of Hausa, Fula, Akan, Azerbaijani and Uzbek and the tone marks of Vietnamese and Yoruba.
- **Pages at any angle.** Tilts up to 45 degrees are straightened, and the orientation model is trained on real turned scans. A sideways table in a PDF's text layer reads upright, and a page turned by its PDF `/Rotate` reads its text layer in order.
- **Light text on dark ground** (slides, dark table headers) reads.
- **Chinese layout.** Chinese text blocks are no longer mistaken for tables.
- **Examples.** `examples/ocr_image.py` (one image file to markdown, text or JSON) and `examples/quickstart.ipynb`.
- `--max-pages` rejects a negative value.
- The default model download is about 76 MB (was 18 MB): 30 MB of networks and 46 MB of language models.
- Speed: about 44 pages per second on 32 x86 cores, and about 16% more time per page than 0.3.2, for the extra work on each OCR page.

## 0.3.2

- Models from Hugging Face tag `v3.1.1`: the same models as `v3.1`, plus
  `config.json` (the file the Hub counts downloads by) and `model.safetensors`
  (the recognizer's weights, for the Hub's parameter count; never downloaded).
- A cached model folder is checked against the Hub at most once a day
  (`CHECK_EVERY_S`); the other runs use the cache with no network, and a
  failed check (offline) keeps the cached models.

## 0.3.1

- A model download that fails (no network, no access) or a model folder with
  missing files raises `indra_ocr.ModelError` with one clear message; the CLI
  and the viewer print it as one line and exit with 1.
- `--version` for `indra-ocr` and `indra-ocr-viewer`.
- `page.skew` is rounded to 0.01 degree.
- Project links on PyPI, and absolute image URLs in the PyPI description, so
  its images show there.
- README: install from PyPI (`pip install indra-ocr`); a Limitations section
  that leads with complex tables, handwriting and scripts outside the OCR list.

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
  tables 56.4 -> 54.8; median page ~210 -> ~360 ms (x86, 4 cores, auto mode).
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
