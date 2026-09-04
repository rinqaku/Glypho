<div align="center">

# Glypho for Rust

**Fast multilingual OCR with a native Rust API.**<br>
Local-first, ONNX Runtime powered, and built for screenshots, photos, UI text and embedded OCR workloads.

[![Crates.io](https://img.shields.io/crates/v/glypho-ocr?style=flat-square&logo=rust)](https://crates.io/crates/glypho-ocr)
[![docs.rs](https://img.shields.io/docsrs/glypho-ocr?style=flat-square&logo=docs.rs)](https://docs.rs/glypho-ocr)
[![GitHub](https://img.shields.io/badge/GitHub-rinqaku%2FGlypho-181717?style=flat-square&logo=github)](https://github.com/rinqaku/Glypho)
[![License](https://img.shields.io/badge/license-Apache--2.0-4c8bf5?style=flat-square)](https://github.com/rinqaku/Glypho/blob/main/LICENSE)

[🌐 **Try Glypho Web**](https://glypho.kaneki.cz) · [**GitHub**](https://github.com/rinqaku/Glypho)

</div>

Glypho provides the native OCR engine for the Rust, Python and Node.js packages. Glypho Web brings the same model family and routing approach to the browser through ONNX Runtime Web. Models are downloaded on demand, verified with SHA-256 and cached locally for reuse.

## 📦 Install

Library:

```bash
cargo add glypho-ocr
```

CLI:

```bash
cargo install glypho-ocr
```

The package name is `glypho-ocr`; the Rust crate is imported as `glypho`.

## 🚀 Quick start

```rust
use glypho::{Device, OnnxConfig, OnnxEngine, QualityMode, RecognitionOptions};

fn main() -> glypho::Result<()> {
    let mut config = OnnxConfig::default();
    config.quality = QualityMode::Balanced;
    config.device = Device::Auto;

    let engine = OnnxEngine::new(config)?;

    let languages = vec!["en".to_owned(), "ja".to_owned()];
    engine.warmup(&languages)?;

    let mut options = RecognitionOptions::default();
    options.languages = languages;

    let document = engine.recognize("screenshot.png", &options)?;
    println!("{}", document.text);

    Ok(())
}
```

Models are cached under:

```text
~/.glypho-ocr/models
```

Set `GLYPHO_HOME` or `GLYPHO_MODELS` to use another location.

## 🎯 Quality profiles

| Profile | Detector | Primary recognizer | Use case |
| --- | --- | --- | --- |
| `Fast` | PP-OCRv6 Tiny | PP-OCRv6 Tiny | minimum latency |
| `Balanced` | PP-OCRv5 Mobile | PP-OCRv6 Small | default OCR |
| `Accurate` | PP-OCRv6 Small | PP-OCRv6 Small | smaller / harder text |
| `Maximum` | PP-OCRv6 Medium | PP-OCRv6 Medium | accuracy-first workloads |

Glypho supports 55 canonical language identifiers across Latin, Eastern Slavic, Chinese, Japanese and Korean routes. Language hints are normalized from common BCP-47-style values such as `cs-CZ`, `de-DE`, `ko-KR`, `jpn` and `rus`.

## ⚡ Hardware acceleration

CPU works with the default build.

CUDA:

```bash
cargo add glypho-ocr --features cuda
```

CoreML:

```bash
cargo add glypho-ocr --features coreml
```

Or for the CLI:

```bash
cargo install glypho-ocr --features cuda
```

Available runtime targets are:

```text
auto | cpu | cuda | coreml | openvino
```

`auto` probes available accelerators and falls back to CPU when necessary. OpenVINO expects a caller-supplied OpenVINO-enabled ONNX Runtime build.

## 🧩 CLI

The crate also ships the `glypho` executable:

```bash
glypho screenshot.png \
  --language en,de \
  --quality accurate \
  --device auto
```

JSON output:

```bash
glypho screenshot.png --format json --output result.json
```

## 🔒 Local-first

OCR inference runs locally. Glypho does not require an OCR API account or API key.

The first run may download pinned model artifacts. They are verified before use and then reused from the local cache. After the models are present, OCR can run without sending the input image anywhere.

---

Full project: [github.com/rinqaku/Glypho](https://github.com/rinqaku/Glypho)<br>
Web preview: [glypho.kaneki.cz](https://glypho.kaneki.cz)<br>
License: [Apache-2.0](https://github.com/rinqaku/Glypho/blob/main/LICENSE)
