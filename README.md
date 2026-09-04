<p align="center">
  <strong>English</strong> ·
  <a href="README/README.ru.md">Русский</a> ·
  <a href="README/README.cs.md">Čeština</a> ·
  <a href="README/README.ja.md">日本語</a>
</p>

<div align="center">

# Glypho

**Fast multilingual OCR with a Rust core.**<br>
Local by default, easy to embed, and available from Rust, Python, Node.js and the browser.

[![Crates.io](https://img.shields.io/crates/v/glypho-ocr?style=flat-square&logo=rust)](https://crates.io/crates/glypho-ocr)
[![PyPI](https://img.shields.io/pypi/v/glypho-ocr?style=flat-square&logo=python)](https://pypi.org/project/glypho-ocr/)
[![npm](https://img.shields.io/npm/v/glypho-ocr?style=flat-square&logo=npm)](https://www.npmjs.com/package/glypho-ocr)
[![License](https://img.shields.io/badge/license-Apache--2.0-4c8bf5?style=flat-square)](LICENSE)

</div>

<p align="center">
  <img src="README/assets/glypho-demo-en.png" alt="Glypho Web recognizing text on a guitar" width="100%">
</p>

<p align="center">
  <a href="https://glypho.kaneki.cz"><strong>🌐 Try Glypho Web</strong></a>
</p>

Glypho is built for screenshots, photos, UI text and everyday OCR where you want **good accuracy without sending the image to a remote API**.

The native engine is written in Rust and uses ONNX Runtime with PP-OCR models. Models are downloaded only when needed, verified with SHA-256 and cached locally. Warm sessions stay in memory, so repeated recognition does not pay startup cost every time.

---

## ⚡ Why Glypho?

- **Rust-native core** — bounded image decoding, routing, batching, model storage and inference orchestration.
- **Multilingual** — 55 canonical BCP-47 language identifiers across Latin, East Slavic, Chinese, Japanese and Korean routes.
- **Local-first** — input images are processed locally; no OCR API account or key is required.
- **Hardware acceleration** — CPU, CUDA, CoreML and OpenVINO-aware builds with automatic fallback.
- **Warm inference** — reuse one engine and keep detector/recognizer sessions alive.
- **One package name** — `glypho-ocr` on Cargo, PyPI and npm.
- **Browser demo** — WebGPU + threaded WASM, no backend OCR server.

---

## 📦 Installation

### Python

```bash
pip install glypho-ocr
```

### Rust

```bash
cargo add glypho-ocr
```

CLI only:

```bash
cargo install glypho-ocr
```

### Node.js

```bash
npm install glypho-ocr
```

Published Python wheels and npm platform packages include the native runtime. You do **not** need Rust, Cargo or a manually installed `libglypho.so` to use them.

Models are cached in:

```text
~/.glypho-ocr/models
```

Use `GLYPHO_HOME` or `GLYPHO_MODELS` to change the location.

---

## 🚀 Quick start

### Python

```python
from glypho import Glypho

ocr = Glypho(
    languages=("en", "de"),
    quality="balanced",
    device="auto",
)

ocr.warmup()
document = ocr.recognize("screenshot.png")

print(document.text)
```

### Node.js

```js
import { Glypho } from "glypho-ocr";

const ocr = new Glypho({
  languages: ["ja", "en"],
  quality: "balanced",
  device: "auto",
});

await ocr.warmup();
const document = await ocr.recognize("screenshot.png");

console.log(document.text);
await ocr.close();
```

### CLI

```bash
glypho screenshot.png
```

More control:

```bash
glypho screenshot.png \
  --language en,de \
  --quality accurate \
  --device cuda
```

JSON output:

```bash
glypho screenshot.png --format json --output result.json
```

### Rust

```rust
use glypho::{Device, OnnxConfig, OnnxEngine, RecognitionOptions};

let mut config = OnnxConfig::default();
config.device = Device::Auto;

let ocr = OnnxEngine::new(config)?;
ocr.warmup(&["en".to_owned()])?;

let document = ocr.recognize(
    "screenshot.png",
    &RecognitionOptions::default(),
)?;

println!("{}", document.text);
# Ok::<(), Box<dyn std::error::Error>>(())
```

---

## 🌍 Multilingual OCR

Glypho exposes **55 canonical language identifiers** and routes them to recognizers whose dictionaries actually cover the requested script.

Current routes include:

- Latin-script languages such as English, Czech, German, French, Spanish, Polish, Turkish and Vietnamese;
- Russian, Ukrainian and Belarusian;
- Chinese;
- Japanese;
- Korean.

Common ISO aliases and region tags such as `de-DE`, `cs-CZ`, `ko-KR`, `eng`, `jpn` and `rus` are normalized automatically.

<p align="center">
  <img src="README/assets/glypho-demo-ja.png" alt="Glypho Web recognizing Japanese text" width="100%">
</p>

See [`docs/LANGUAGES.md`](docs/LANGUAGES.md) for the complete list and routing rules.

> `fast` intentionally does not expose Japanese because the Tiny recognizer dictionary does not contain kana.

---

## 🎯 Quality profiles

Glypho uses fixed model profiles instead of silently changing model combinations between releases.

| Profile | Detector | Primary recognizer | Best for |
| --- | --- | --- | --- |
| `fast` | PP-OCRv6 Tiny | PP-OCRv6 Tiny | lowest latency / weaker devices |
| `balanced` | PP-OCRv5 Mobile | PP-OCRv6 Small | default everyday OCR |
| `accurate` | PP-OCRv6 Small | PP-OCRv6 Small | small or difficult text |
| `maximum` | PP-OCRv6 Medium | PP-OCRv6 Medium | accuracy-first workloads |

Latin, Eastern Slavic and Korean requests can use compact PP-OCRv5 specialist recognizers. Mixed-script requests share one detector pass and reuse the same perspective crops.

---

## 🖥️ Hardware acceleration

Native APIs accept:

```text
auto | cpu | cuda | coreml | openvino
```

`auto` probes native providers in this order:

```text
NVIDIA / CUDA → Apple CoreML → Intel OpenVINO → CPU
```

If an accelerator is unavailable, Glypho falls back to CPU and reports the reason through `info()`.

For custom Rust builds:

```bash
cargo install glypho-ocr --features cuda
cargo install glypho-ocr --features coreml
```

OpenVINO uses a caller-supplied OpenVINO-enabled ONNX Runtime. See [`docs/HARDWARE.md`](docs/HARDWARE.md).

---

## 🌐 Glypho Web

`web/` contains the browser version of Glypho.

It uses:

- React + TypeScript;
- ONNX Runtime Web;
- WebGPU and threaded WASM;
- Web Workers for inference;
- browser model cache;
- SHA-256 verification for model artifacts.

The image itself never needs to leave the browser.

Run it locally:

```bash
cd web
npm install
npm run dev
```

Web auto mode prefetches lightweight language packs in parallel and creates specialist sessions lazily when they are actually useful.

---

## 🔒 Local-first by design

Glypho does not require a cloud OCR backend.

- Native input images stay on the machine.
- Browser images stay inside the page.
- Model downloads are size-limited and SHA-256 verified.
- `--offline` disables model downloads completely.
- The result format is versioned as `glypho.annotation.v1`.

---

## 📄 License

Glypho is licensed under **Apache-2.0**.

PP-OCR model artifacts are distributed by PaddlePaddle under Apache-2.0 and are downloaded separately on first use.
