# Python guide

Glypho's Python package calls the native Rust library through `ctypes`. It does not start a new CLI process for every image, so matching ONNX sessions can stay warm inside the Python process.

## 📦 Install

```bash
pip install glypho-ocr
```

Platform wheels include the native library and `glypho` executable. Rust and Cargo are not required for normal use.

Models are downloaded on first use and cached under:

```text
~/.glypho-ocr/models
```

## 🚀 First OCR

```python
from glypho import Glypho

ocr = Glypho(
    languages=("en", "ru"),
    quality="balanced",
    device="auto",
)

ocr.warmup()
document = ocr.recognize("image.png")

print(document.text)
```

For latency-sensitive programs, create the client once and reuse it.

```python
from pathlib import Path
from glypho import Glypho

ocr = Glypho(quality="balanced", device="auto")
ocr.warmup(languages=["en", "cs"])

for path in Path("images").glob("*.png"):
    document = ocr.recognize(path, languages=["en", "cs"])
    print(path.name, document.text)
```

## Warmup

`warmup()` downloads and initializes the detector and recognizers required by the selected profile/languages.

```python
ocr.warmup(languages=["en", "ru"])
```

Passing a representative image also executes the full path once:

```python
ocr.warmup(sample="sample.png", languages=["en"])
```

This is useful for servers where you want first-use allocations to happen before requests arrive.

## 🎯 Quality and devices

```python
ocr = Glypho(
    quality="accurate",
    device="cuda",
    threads=4,
)
```

Quality profiles:

```text
fast | balanced | accurate | maximum
```

Devices:

```text
auto | cpu | cuda | coreml | openvino
```

If an accelerated provider is unavailable, Glypho falls back to CPU. Inspect the runtime with:

```python
print(ocr.info())
```

More details: [`HARDWARE.md`](HARDWARE.md).

## 🌍 Languages

Hints accept common BCP-47 forms:

```python
ocr.recognize("czech.png", languages=["cs-CZ"])
ocr.recognize("korean.png", languages=["ko-KR"])
ocr.recognize("mixed.png", languages=["ja", "en"])
```

See [`LANGUAGES.md`](LANGUAGES.md) for the complete list and routing behavior.

## Result objects

`recognize()` returns an immutable `Document`.

```python
document = ocr.recognize("image.png")

print(document.text)

for line in document.lines:
    print(line.text)
    print(line.confidence)
    print(line.language, line.script)
    print([(point.x, point.y) for point in line.quad.points])
```

Useful fields:

- `document.text` — ordered plain text;
- `document.lines` — detected lines;
- `line.quad.points` — source-image coordinates;
- `line.confidence` — recognition confidence;
- `line.language` / `line.script` — routed metadata when known;
- `line.alternatives` — rejected candidate when available;
- `document.to_dict()` — complete JSON-compatible result.

## Offline mode

Prevent every model download:

```python
ocr = Glypho(offline=True)
```

Custom model cache:

```python
ocr = Glypho(models="/srv/glypho-models")
```

You can also use `GLYPHO_HOME` or `GLYPHO_MODELS`.

## Errors

```python
from glypho import Glypho, GlyphoNotFoundError, RecognitionError

try:
    document = Glypho().recognize("image.png")
except FileNotFoundError:
    print("image not found")
except GlyphoNotFoundError as error:
    print(f"native runtime unavailable: {error}")
except RecognitionError as error:
    print(f"OCR failed: {error}")
```