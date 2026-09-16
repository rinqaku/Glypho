# Benchmarks

Glypho's public benchmark claims are based on reproducible **internal SDK-level comparisons**. They are not official Robust Reading Competition leaderboard submissions.

## ICDAR 2015 — six-engine CPU comparison

The comparison uses all 500 ICDAR 2015 incidental scene-text test images on an Intel Core i5-12600KF, CPU-only. Every engine receives one image per public SDK call after one warm-up image. Detection uses greedy one-to-one polygon matching at IoU >= 0.5; end-to-end quality additionally requires exact case-folded transcription after trimming surrounding ASCII punctuation.

| Engine | Det. H ↑ | E2E H ↑ | p50 ↓ | p95 ↓ | Peak RSS | Models |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Glypho Balanced** | **0.7129** | **0.5972** | **172 ms** | **232 ms** | 1.25 GB | 31.3 MB |
| ppu-paddle-ocr | 0.2025 | 0.0998 | 195 ms | 517 ms | 872 MB | **6.4 MB** |
| Tesseract 5 | 0.0697 | 0.0480 | 262 ms | 589 ms | **87.8 MB** | 23.5 MB |
| docTR | 0.5739 | 0.4107 | 728 ms | 1.08 s | 1.39 GB | 141.6 MB |
| EasyOCR | 0.6061 | 0.2188 | 1.77 s | 2.02 s | 1.42 GB | 98.3 MB |
| Surya Classic | 0.1687 | 0.1271 | 5.80 s | 14.33 s | 4.86 GB | 1.72 GB |

Within this protocol, Glypho Balanced has the strongest detection H-mean, end-to-end H-mean, warmed p50 and warmed p95 of the six tested SDKs. Tesseract uses substantially less peak memory, while the PP-OCRv6 Tiny PPU package has the smallest model footprint.

Full protocol and engine versions: [`benchmarks/ICDAR-2015-comparison.md`](benchmarks/ICDAR-2015-comparison.md).

## Glypho vs PPU — model-scale matched

A second comparison pairs Glypho and `ppu-paddle-ocr` 6.6.0 at closely matched Tiny, Small and Medium installed model sizes.

| Scale | Engine | Det. H ↑ | E2E H ↑ | p50 ↓ | p95 ↓ | Models |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tiny | **Glypho Fast** | **0.6152** | **0.4525** | **60.2 ms** | **70.7 ms** | **6.33 MB** |
| Tiny | PPU V6 Tiny | 0.2124 | 0.1276 | 159.5 ms | 417.9 ms | 6.44 MB |
| Small | **Glypho Balanced** | **0.7129** | **0.5972** | **171.9 ms** | **231.8 ms** | **31.27 MB** |
| Small | PPU V6 Small | 0.1032 | 0.0772 | 405.1 ms | 597.3 ms | 31.35 MB |
| Medium | **Glypho Maximum** | **0.7310** | **0.6160** | **677.1 ms** | **908.5 ms** | **138.81 MB** |
| Medium | PPU V6 Medium | 0.1261 | 0.1009 | 1,441.9 ms | 1,726.6 ms | 138.93 MB |

Across these three pairs, Glypho is **2.13×–2.65× faster at p50** and reaches **3.55×–7.73× the end-to-end H-mean** in this protocol. The Small pair is the closest size match: 31.27 MB vs 31.35 MB.

Full protocol and interpretation: [`benchmarks/ICDAR-2015-model-matched.md`](benchmarks/ICDAR-2015-model-matched.md).

## What the numbers do and do not claim

These measurements compare complete public SDK pipelines, not raw neural-network weights in isolation. Geometry, word splitting, preprocessing, routing and recognition behavior are part of the result. The local evaluator is intentionally simple and shared by all engines.

For an official leaderboard claim, reproduce the quality numbers with the official ICDAR/RRC evaluator. Until then, public wording should stay explicit: **internal ICDAR 2015 CPU benchmark**.
