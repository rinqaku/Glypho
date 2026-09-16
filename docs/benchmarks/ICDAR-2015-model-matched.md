# ICDAR 2015: PP-OCRv6 scale-matched comparison

Controlled internal comparison of Glypho and `ppu-paddle-ocr` 6.6.0. This is
not an official Robust Reading Competition submission or leaderboard result.

## Protocol

- Dataset: all 500 ICDAR 2015 incidental scene-text test images, 2,077 care
  words.
- Hardware: Intel Core i5-12600KF, CPU-only, one engine at a time.
- Workload: one image per public SDK call after one warm-up image.
- Model pairing: PP-OCRv6 Tiny, Small, and Medium families at closely matched
  installed sizes. PPU uses optimized `.ort` artifacts and Glypho uses `.onnx`;
  these are scale/family matches, not byte-identical graphs.
- Both engines: English hint, recognition confidence threshold 0.8, detector
  maximum side 960/1280/2048 for Tiny/Small/Medium.
- PPU: public API, `per-box`, CPU, default crop padding 0.4 vertical and 0.6
  horizontal, cache disabled per call. `per-box` was selected on the training
  split because ICDAR evaluates word boxes; PPU's default `per-line` merges
  multiple reference words and scored worse in the calibration.
- Glypho: `fast`/`balanced`/`maximum`, CPU, public Python API. Glypho exposes
  word coordinates recovered from recognizer character positions.
- Detection: greedy one-to-one polygon matching at IoU >= 0.5.
- End-to-end: the same geometric match plus exact case-folded transcription
  after trimming surrounding ASCII punctuation.
- `###` regions and predictions overlapping them by more than 50% are ignored.
- p50/p95 include image decode and OCR after warm-up. Model initialization is
  reported separately as cold time and is not used for the speed ratio.

## Results

| Scale | Engine | Det. H | E2E H | Cold | p50 | p95 | Peak RSS | Models |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny | **Glypho Fast** | **0.6152** | **0.4525** | 275 ms | **60.2 ms** | **70.7 ms** | **211.5 MB** | **6.33 MB** |
| Tiny | PPU V6 Tiny | 0.2124 | 0.1276 | **190 ms** | 159.5 ms | 417.9 ms | 789.2 MB | 6.44 MB |
| Small | **Glypho Balanced** | **0.7129** | **0.5972** | **260 ms** | **171.9 ms** | **231.8 ms** | 1,250.3 MB | **31.27 MB** |
| Small | PPU V6 Small | 0.1032 | 0.0772 | **465 ms** | 405.1 ms | 597.3 ms | **1,068.0 MB** | 31.35 MB |
| Medium | **Glypho Maximum** | **0.7310** | **0.6160** | **918 ms** | **677.1 ms** | **908.5 ms** | **1,250.3 MB** | **138.81 MB** |
| Medium | PPU V6 Medium | 0.1261 | 0.1009 | **1.50 s** | 1,441.9 ms | 1,726.6 ms | 1,615.2 MB | 138.93 MB |

| Scale | Glypho E2E H advantage | Glypho p50 speedup | Glypho p95 speedup |
| --- | ---: | ---: | ---: |
| Tiny | 3.55x | 2.65x | 5.91x |
| Small | 7.73x | 2.36x | 2.58x |
| Medium | 6.11x | 2.13x | 1.90x |

## Interpretation

The Small pair is the strongest public comparison: installed model sizes are
within 0.3%, while Glypho is 2.36x faster at p50 and has 7.73x the end-to-end
H-mean in this protocol. Tiny demonstrates the low-latency profile; Medium shows
that the result is not limited to the default model size.

PPU starts faster at Tiny. Glypho starts faster at Small and Medium after detector
and primary-recognizer initialization was made concurrent on CPU. Cold time is
sensitive to the operating-system file cache and process order, so it still needs
repeated isolated measurement before a launch-time claim.

The quality gap measures the complete SDK pipelines, not only neural-network
weights. PPU's public result geometry is an axis-aligned rectangle and its
detector can return a whole multi-word line as one region. Glypho preserves
quadrilateral geometry and emits word boxes using character positions. Both are
real user-visible architecture choices, but they make this unsuitable as a
claim that Glypho's raw model weights alone are more accurate.

Before using leaderboard language, reproduce quality with the official ICDAR
RRC evaluator. A public chart should say "internal ICDAR 2015 CPU benchmark",
include the hardware, package versions, configurations, scorer link/code, and
date, and publish all rows rather than only the winning scale.

## Raw reports

- `benchmark/icdar2015/reports/glypho-fast-parallel-init.json`
- `benchmark/icdar2015/reports/ppu-v6-tiny-matched.json`
- `benchmark/icdar2015/reports/glypho-balanced-parallel-init-full.json`
- `benchmark/icdar2015/reports/ppu-v6-small-matched.json`
- `benchmark/icdar2015/reports/glypho-maximum-parallel-init-full.json`
- `benchmark/icdar2015/reports/ppu-v6-medium-matched.json`
- Aggregated metrics: `benchmark/icdar2015/reports/comparison.json`
