# ICDAR 2015 comparison

Preliminary internal comparison for Glypho development. This is not an official
Robust Reading Competition submission or leaderboard result.

## Method

- Dataset: ICDAR 2015 incidental scene text, all 500 test images.
- Device: CPU, Intel Core i5-12600KF; no usable NVIDIA provider.
- Workload: one image per public SDK call, after one warm-up image.
- Detection: greedy one-to-one polygon matching at IoU >= 0.5.
- End-to-end: the same geometric match plus exact case-folded transcription,
  after trimming surrounding ASCII punctuation.
- `###` regions are ignored, including predictions overlapping them.
- Timings include image decoding and OCR but exclude model initialization after
  warm-up. Cold initialization, p50, p95, peak RSS, and model bytes are reported
  separately.

The local evaluator is intentionally simple and shared by every engine. Before
publishing marketing claims, reproduce the numbers with the official RRC scorer
and state model versions, hardware, and configuration beside the table.

## Results

| Engine | Det. H | E2E H | Cold | p50 | p95 | Peak RSS | Models |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Glypho balanced | **0.7129** | **0.5972** | **260 ms** | **172 ms** | **232 ms** | 1.25 GB | 31.3 MB |
| ppu-paddle-ocr | 0.2025 | 0.0998 | 405 ms | 195 ms | 517 ms | 872 MB | **6.4 MB** |
| Tesseract 5 | 0.0697 | 0.0480 | **263 ms** | 262 ms | 589 ms | **87.8 MB** | 23.5 MB |
| docTR | 0.5739 | 0.4107 | 1.49 s | 728 ms | 1.08 s | 1.39 GB | 141.6 MB |
| EasyOCR | 0.6061 | 0.2188 | 3.09 s | 1.77 s | 2.02 s | 1.42 GB | 98.3 MB |
| Surya Classic | 0.1687 | 0.1271 | 12.39 s | 5.80 s | 14.33 s | 4.86 GB | 1.72 GB |

Bold values identify the strongest quality/latency result and the smallest
resource footprint separately; no single engine is best on every resource.
Glypho has the best detection and end-to-end H-mean in this internal scorer and
the lowest warmed p50/p95. Tesseract has by far the lowest peak RSS, while the
PP-OCRv6 Tiny package has the smallest model download.

## Engines

- Glypho 0.2.0-dev: `balanced`, English hint, CPU.
- ppu-paddle-ocr 6.6.0: PP-OCRv6 Tiny defaults, CPU.
- Tesseract 5.5.3: `eng` word boxes.
- docTR 1.1.0: default pretrained detector and recognizer, CPU.
- EasyOCR 1.7.2: English, CPU, default single-image API.
- Surya OCR 0.14.6: classic detector/recognizer pipeline, CPU. Current Surya
  releases use a substantially heavier VLM pipeline and are not substituted into
  this compact OCR SDK comparison.
