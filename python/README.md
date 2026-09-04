<div align="center">

# Glypho for Python

**Fast multilingual OCR for Python with a bundled native runtime.**<br>
Local-first, Python 3.10+, and powered by the same Rust + ONNX Runtime engine as Glypho for Rust and Node.js.

[![PyPI](https://img.shields.io/pypi/v/glypho-ocr?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/glypho-ocr/)
[![Python](https://img.shields.io/pypi/pyversions/glypho-ocr?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/glypho-ocr/)
[![GitHub](https://img.shields.io/badge/GitHub-rinqaku%2FGlypho-181717?style=flat-square&logo=github)](https://github.com/rinqaku/Glypho)
[![License](https://img.shields.io/badge/license-Apache--2.0-4c8bf5?style=flat-square)](https://github.com/rinqaku/Glypho/blob/main/LICENSE)

[🌐 **Try Glypho Web**](https://glypho.kaneki.cz) · [**GitHub**](https://github.com/rinqaku/Glypho)

</div>

Glypho's Python package calls the bundled native Rust library directly through a stable C ABI. It does not start a new OCR subprocess for every image, so ONNX Runtime sessions and loaded models can stay warm inside the Python process.

No OCR cloud account, API key, Rust toolchain or separately installed `libglypho` is required when a prebuilt wheel is available for your platform.

## 📦 Install

```bash
pip install glypho-ocr
```

The package name is `glypho-ocr`; import it as `glypho`:

```python
from glypho import Glypho
```

Prebuilt wheels for `0.1.0` are published for:

| OS | Architectures |
| --- | --- |
| Linux | x64, ARM64 |
| macOS | Apple Silicon (ARM64) |
| Windows | x64, ARM64 |

Intel macOS is not included in the `0.1.0` prebuilt-wheel matrix. On targets without a matching wheel, `pip` may fall back to the source distribution, which requires Rust/Cargo and is not covered by the prebuilt release matrix.

## 🚀 Quick start

```python
from glypho import Glypho

ocr = Glypho(
    quality="balanced",
    device="auto",
)

document = ocr.recognize("screenshot.png")
print(document.text)
```

Leaving `languages` empty enables automatic native routing. If you already know what to expect, pass language hints:

```python
ocr = Glypho(
    languages=("en", "ja"),
    quality="accurate",
)

document = ocr.recognize("mixed.png")
```

Common BCP-47-style values such as `cs-CZ`, `de-DE`, `ko-KR`, `zh-Hans`, `jpn` and `rus` are normalized automatically.

## ⚙️ Options

```python
ocr = Glypho(
    quality="balanced",       # fast | balanced | accurate | maximum
    device="auto",            # auto | cpu | cuda | coreml | openvino
    languages=(),              # () = automatic routing
    threads=None,
    offline=False,
    models=None,
)
```

Per-request options can override the defaults:

```python
document = ocr.recognize(
    "image.png",
    languages=["en", "ru"],
    segmentation="sparse_text",
    min_confidence=0.8,
    timeout=30.0,
)
```

Segmentation modes:

```text
auto | single_block | single_line | sparse_text
```

## 🎯 Quality profiles

| Profile | Detector | Primary recognizer | Use case |
| --- | --- | --- | --- |
| `fast` | PP-OCRv6 Tiny | PP-OCRv6 Tiny | minimum latency |
| `balanced` | PP-OCRv5 Mobile | PP-OCRv6 Small | default OCR |
| `accurate` | PP-OCRv6 Small | PP-OCRv6 Small | smaller / harder text |
| `maximum` | PP-OCRv6 Medium | PP-OCRv6 Medium | accuracy-first workloads |

Glypho exposes 55 canonical language identifiers across Latin, Eastern Slavic, Chinese, Japanese and Korean routes.

## 🔥 Warm sessions

For repeated OCR, create one `Glypho` instance and warm it before processing requests:

```python
from glypho import Glypho

ocr = Glypho(languages=("en", "cs"))
ocr.warmup()

first = ocr.recognize("first.png")
second = ocr.recognize("second.png")

print(first.text)
print(second.text)
```

You can also warm the complete decode → detect → recognize path with a representative image:

```python
ocr.warmup(sample="sample.png", languages=["en"])
```

`info()` reports the configured and resolved runtime/device information:

```python
print(ocr.info())
```

## 📐 Structured results

`recognize()` returns an immutable `Document`, not just a string:

```python
document = ocr.recognize("image.png")

print(document.text)

for line in document.lines:
    print(line.text)
    print(line.confidence)
    print(line.language, line.script)
    print([(point.x, point.y) for point in line.quad.points])
```

Useful fields include:

- `document.text` — ordered plain text;
- `document.lines` — detected text lines;
- `line.quad.points` — source-image quadrilateral coordinates;
- `line.confidence` — recognition confidence;
- `line.language` / `line.script` — routed metadata when known;
- `line.alternatives` — rejected candidate when available;
- `document.to_dict()` — complete JSON-compatible result.

## ⚡ Hardware acceleration

Available device targets are:

```text
auto | cpu | cuda | coreml | openvino
```

Provider availability depends on the wheel, platform and host runtime. `device="auto"` probes available accelerators and falls back to CPU when necessary; `info()` exposes the resolved device and fallback information.

The `0.1.0` release builds are CUDA-aware on Linux/Windows x64 and CoreML-aware on macOS Apple Silicon. CPU remains the fallback path.

## 🧩 CLI

The Python wheel also installs the native `glypho` executable:

```bash
glypho screenshot.png \
  --language en,ja \
  --quality balanced \
  --device auto
```

Runtime information:

```bash
glypho info --pretty
```

JSON output:

```bash
glypho screenshot.png --format json --output result.json
```

## 💾 Models and offline mode

Missing model files are downloaded from pinned revisions, verified with SHA-256 and cached under:

```text
~/.glypho-ocr/models
```

Use another location with `GLYPHO_HOME`, `GLYPHO_MODELS` or the `models` argument.

To prohibit model downloads completely:

```python
ocr = Glypho(offline=True)
```

## 🔒 Local-first

OCR inference runs locally. Glypho does not upload input images to an OCR API.

The only network access normally needed is the first model download. Once the required models are cached, recognition can run offline.

---

Python guide: [docs/PYTHON.md](https://github.com/rinqaku/Glypho/blob/main/docs/PYTHON.md)<br>
Full project: [github.com/rinqaku/Glypho](https://github.com/rinqaku/Glypho)<br>
Web preview: [glypho.kaneki.cz](https://glypho.kaneki.cz)<br>
Rust package: [crates.io/crates/glypho-ocr](https://crates.io/crates/glypho-ocr)<br>
Node.js package: [npmjs.com/package/glypho-ocr](https://www.npmjs.com/package/glypho-ocr)<br>
License: [Apache-2.0](https://github.com/rinqaku/Glypho/blob/main/LICENSE)
