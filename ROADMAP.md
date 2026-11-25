# Indra OCR: ways to make the next version faster and more accurate

This note collects what measurements of Indra OCR 1.2.0 suggest for the next version: where the time goes on the CPU and the GPU, which tests it fails and why, and for each, a change that could help, what it would need, and how to check it.

It sits beside the 1.2 pull requests that need no new models: the ONNX Runtime CUDA fixes, recognizer-only reads in the symbol-digit fix and the turn checks, the bench tools on omni-parse-bench 1.8, the optional GPU engine, and `indra-ocr-serve`. What is left here needs new models, or changes whose outputs move and need the gates below.

**How it was measured.** A shared workstation (other users ran jobs on it): an AMD Threadripper 3970X (32 cores, AVX2, no AVX-512 or VNNI) and two NVIDIA RTX A6000. The software was ONNX Runtime 1.32, TensorRT 11.4, CUDA 13 and omni-parse-bench 1.8. Every OmniParseBench number is from the benchmark's own grader on all 2,937 pages (`bench/opb_indra.py`), and 1.2.0 grades 78.83 there. Times come from paired runs or one worker on an idle core. Because the machine was shared, please measure again on your own machines. The commands are at the end.

## 1. Where the time goes

One CPU core, 300 pages spread over the benchmark (every 9th), auto mode, math on (`bench/profile_stages.py`). Mean per page: 737 ms. ONNX Runtime takes 70% of it. A sampling profile of 150 pages puts it at 80%, so the Python and OpenCV steps are a small share.

| model | ms per page | share | what would make it faster |
|---|---|---|---|
| `rec.onnx` (Latin, 96-px window) | 137 | 18.6% | the window (3.4), ReLU (3.3), int8 head (3.5) |
| `math_enc.onnx` | 111 | 15.1% | int8, narrower front (3.2) |
| `math_dec_step_kv.onnx`, one run per token | 106 | 14.4% | fewer steps (3.1), a smaller decoder (3.2) |
| `det.onnx` | 56 | 7.5% | its output head (3.6), ReLU (3.3) |
| `rec_cjk`, `rec_world`, `rec_hw`, `rec_rtl` | 34 | 4.6% | int8 (3.5) |
| `tsr_split.onnx` | 19 | 2.6% | |
| `mathcls`, `script`, `hw` (7,000 runs per 300 pages) | 30 | 4.0% | one model with three heads (3.7) |

On text-layer pages, math is 45% of the time. On the GPU engine (one page at a time, 248 ms), the models are a small part. What is left there is the work between them: crops, the CTC loop, renders, the skew search, the detector's decoding, and table and order features (section 4).

## 2. Where the tests are lost

1.2.0, all 2,937 pages: headline 78.83, text 67.91 (8,499 tests), tables 75.94 (5,386), layout 92.63 (2,403). 4,200 tests fail. The grader gives a reason for each:

| reason | failed tests |
|---|---|
| an equation not found | 1,169 |
| a text not found (the closest string in the output is close, but under the threshold) | 1,109 |
| a table cell found, but not under its headings / not in its row group / not in its section | 480 / 229 / 124 (+ 26 under both) |
| no table cell matching the text in any table | 308 |
| no table found on the page | 34 |
| a reading-order test whose text is not there | 352 |
| a block with the wrong label | 165 |

The weakest areas:

| suite or tag | score | tests |
|---|---|---|
| `frbench_handwriting` / `handwriting_samples` (tag handwriting) | 9.5 / 19.1 (35.0) | 84 / 324 (923) |
| `olmo_old_scans` / `olmo_old_scans_math` (tag degraded) | 30.4 / 40.7 (42.4) | 395 / 450 (655) |
| `multilingual` | 45.7 | 92 |
| tag math (mostly `olmo_arxiv_math`, 71.3) | 63.9 | 3,316 |
| tag rotated | 63.8 | 824 |
| tag header_footer | 61.7 | 227 |
| tag form | 65.6 | 1,028 |

The French tag (43.0, 279 tests) looks like a language problem, but it is not: its failures are handwriting (`frbench_handwriting`, `handwriting_samples`) and forms.

## 3. Speed: changes to the models

Each needs a model Datalab trains or exports. For each, the first step is the measurement that says whether the training is worth it.

### 3.1 Fewer decoder steps: speculative decoding from the recognizer's draft

The math decoder runs one ONNX Runtime call per token (21,468 calls on 300 pages, 1.49 ms each on one core). Every line it reads already has a draft, the recognizer's reading. That draft is passed to the accept head and spliced into the prose (`splice_prose`). Where a math reading's prose agrees with the draft, those tokens can be checked several at a time instead of generated one by one.

- **How:** a step graph that takes k draft tokens, returns k sets of logits, and keeps the tokens up to the first disagreement. This is the standard speculative check, and the output is the same as greedy decoding.
- **Needs:** an export of the step graph with a k-token input (the per-layer KV interface stays), and a draft tokenizer from the recognizer's reading into MathRec's vocabulary.
- **First measurement:** on the math lines of the benchmark, how many leading tokens of MathRec's greedy reading the draft predicts. That count bounds the saving.

### 3.2 A smaller math model

The encoder (111 ms per page, Conv 53%, MatMul 21%, Gelu 15% inside) and the decoder (3 layers, 8 heads, d = 192, FFN 768, a 3,819-token head) are now the largest CPU cost.

Candidates, each distilled from the 1.2.0 math models:
- 2 decoder layers;
- a narrower FFN;
- int8 for the encoder's front convolutions;
- a tanh-approximated Gelu (or ReLU) in the encoder.

Gate: the math tests (3,316) and the per-suite scores, plus the share of math lines whose reading changes.

### 3.3 ReLU in place of HardSwish

On the v1.1.0 models, swapping the activation (outputs wrong; an upper bound) made det 1.18× faster, `rec_world` 1.11× and `rec` 1.04× (issue #15). A student distilled from the 1.2.0 models would keep their outputs within the gates. det first: it runs on every page, and on text-layer pages it is most of the non-math time.

### 3.4 The Latin recognizer's window

`rec.onnx` reads a 96-px window (the line and the page above and below it), three times the pixels of the other recognizers' 32-px band. Before any training: blank the context above and below the line and grade, to see what the window buys and where (handwriting, tiny text). Then distill a 64-px or 32-px student if the context costs little. The backbone is about 80% of the recognizer.

### 3.5 int8

Post-training int8 on the v1.1.0 models:
- `rec_cjk`: 1.37×, 0.2% of tokens changed;
- `rec_world`: 1.61×, 13.7% changed, which needs quantization-aware training;
- `rec`: no faster.

The head convolutions (128 to 1,698 or 11,750 classes) are 13% of `rec.onnx` before the argmax, and the largest single layer of `rec_cjk`. An int8 head alone is the smallest step. On CPUs without VNNI the gain is smaller, so int8 models would be optional and chosen at load time.

### 3.6 The detector's output head

23% of `det.onnx` on the CPU is its last two ops: Softplus over the whole distance maps (14%) and a Transpose (9%). `decode` reads the distance maps only at line-core pixels. An export that ends before the Softplus, with `decode` applying it at those pixels, keeps the same boxes (same formula) at about a fifth less detector time.

### 3.7 One routing model

`script.onnx`, `hw.onnx` and `mathcls.onnx` read the same 32-px crops: 7,000 runs and 30 ms per page together. One backbone with three heads would read each line once.

## 4. Speed: engineering without new models

- **CPU, one process with threads.** Today it is one single-threaded process per core. One process of pinned worker threads would share the sessions, keep pages in flight, and split a single document's lines over idle cores (one page's latency). pdfium is not thread-safe, so rendering stays serialized.
- **CPU, the math decoder's cache copies.** Compacting the caches when lines finish (`MathReader._take`) is 2.3% of CPU time. A step graph that writes into a preallocated cache would remove the per-step concatenation, whose copies grow with the square of the length: 34% of a step at 32 lines and 100 tokens.
- **GPU engine, next steps:**
  - a fused CTC head (the head GEMM with an online max and argmax, so logits are never written);
  - line crops, page resampling and the detector's decoding on the device;
  - CUDA graphs for the fixed-shape engines;
  - pages in flight in one process, with lines batched across pages;
  - a second math-encoder engine for lines wider than 2,048 px;
  - FP16, once it passes the gates. It is 1.5 to 1.7× faster on the recognizers and the detector.
- **Rendering.** A text-layer page is rendered twice: at 72 dpi for the checks and at 200 dpi for the page, about 51 ms each. Rendering once and downsampling for the checks would change the checks' input, so it is a gated change.

## 5. Accuracy

### 5.1 Math (1,169 equations not found)

The largest single reason. Math is on by default, so the remaining failures are readings MathRec gets wrong, or lines the classifier and gate do not send to it.

**First measurement:** split the 1,169 into three groups:
- lines `mathcls` did not flag;
- lines the gate or the accept head rejected;
- readings that are wrong.

Each group points to a different fix: classifier recall, the gate's threshold, or the reader itself. The old scans with math (`olmo_old_scans_math`, 40.7) are mostly the degraded-scan problem of 5.3.

### 5.2 Handwriting (35.0 over 923 tests)

`frbench_handwriting` 9.5, `handwriting_samples` 19.1. The failures are near misses ("allée des Lilas" read "aller des ilas", 0.87 of the line) and cells missing from handwritten forms. Candidates:
- more handwriting in `rec_hw`'s training data;
- routing: `hw.onnx`'s threshold (`hw_route_min`) and `hw_conf_gate`, checked on these suites;
- the n-gram LM on handwritten lines with a wider beam.

### 5.3 Degraded and old scans (42.4 over 655 tests)

`olmo_old_scans` 30.4. Candidates: degradation in training (blur, bleed-through, low contrast, JPEG), and a contrast normalization of the line crops. The second can be tried without training, but it changes the recognizers' inputs, so it needs a graded run.

### 5.4 Tables (1,167 structure failures, 308 cells missing, 34 tables missed)

- **Headings, row groups and sections:** 480, 229 and 124 tests find the cell but not its structure. These are the grid, not the reading. `tsr_rel` and `tsr_refine` already give header tags and relations; the failures say which tables they miss.
- **Rotated tables:** 243 of the 298 failed rotated tests are sideways tables in `tables_derived`. The table models read the table turned upright, so the turn of the block and its words is the first thing to check.
- **No table found (34):** the false-table verifier's threshold, read against these pages.

### 5.5 Text turned a quarter turn: rot_gate

Indra already reads quarter-turned text on upright pages (`rot_gate`, off by default). On all pages with it on:
- `large_format` +10.5;
- header_footer tests +1.8 (vertical margin stamps);
- rotated tests +0.6;
- text family +0.06;
- but layout −0.20 (`tiny_plans` −1.1, `realdoc_pages` −0.45);
- headline 78.83 → 78.77, for 1.9% more time.

Its text helps, but the blocks its lines make cost layout tests. Keeping the turned lines' text while placing or labelling their blocks so they do not take over the layout (as page furniture, or inside the block they cross) could keep the text gain and lose the layout cost.

### 5.6 Headers, footers and labels (61.7 over 227; 165 wrong labels)

The furniture failures are text that is misread (vertical stamps; see 5.5) more than furniture that is missing. The label failures are mostly page furniture against text, and text against heading, on forms. Both are order-model heads; the failing blocks are a ready list for its next training set.

## 6. How to check each change

- **Without new models:** byte-identical pages on all 2,937 (`bench/compare_outputs.py`), paired times (`bench/ab_variant.py`).
- **With new models or moved outputs:**
  - the headline within 0.2 points of 1.2.0, no suite more than 1 point lower, and small suites read test by test;
  - the characters that change (`compare_outputs.py --chars`), with every changed line read;
  - speed per page type, paired, and in throughput.

Commands (each tool's docstring has its options):

```shell
pip install 'omni-parse-bench[benchmark]'
python bench/opb_indra.py --out runs/opb --procs 30 --pin 0-29 --dump          # grade (78.83 for 1.2.0)
python bench/opb_indra.py --out runs/opb_rot --procs 30 --pin 0-29             # with INDRA_ROT_GATE=1 set: section 5.5
python bench/compare_outputs.py runs/opb runs/opb_new --chars                  # pages, characters, scores
python bench/profile_stages.py --out runs/prof --pages 300 --pin 2             # section 1
python bench/ab_variant.py --variant recognizer-math --out runs/ab --procs 30  # paired A/B
python bench/throughput.py --cores 8 16 32 --mode auto --pin 0-31              # pages per second
```
