# Glypho architecture

Glypho has one native OCR engine and several ways to call it. Rust is the source of truth for native inference; Python and Node reuse that runtime instead of maintaining separate OCR implementations.

## 🔁 Native pipeline

```text
image
  → bounded decode
  → text detector
  → quadrilateral regions
  → perspective crops
  → language/script routing
  → batched recognition
  → confidence filtering
  → reading order
  → glypho.annotation.v1
```

The default backend uses ONNX Runtime through Rust. Tesseract is kept as an explicit compatibility backend and is not mixed into the normal ONNX path.

### Quality profiles

| Profile | Detector | Primary recognizer |
| --- | --- | --- |
| `fast` | PP-OCRv6 Tiny | PP-OCRv6 Tiny |
| `balanced` | PP-OCRv5 Mobile | PP-OCRv6 Small |
| `accurate` | PP-OCRv6 Small | PP-OCRv6 Small |
| `maximum` | PP-OCRv6 Medium | PP-OCRv6 Medium |

Profiles are fixed on purpose. A new upstream model should not silently change the behavior of an existing Glypho profile.

## 🌍 Language routing

One detector pass is shared by all requested scripts.

- English uses the profile's unified recognizer.
- Latin-script languages can use the PP-OCRv5 Latin pack.
- Russian, Ukrainian and Belarusian use the Eastern Slavic pack.
- Korean uses the Korean pack.
- Chinese and Japanese use the unified recognizer.

Mixed-language requests reuse the same perspective crops. A specialist is only initialized when the route needs it, and a rejected candidate can be preserved as an alternative.

See [`LANGUAGES.md`](LANGUAGES.md) for the exact language list.

## 📦 Model lifecycle

Missing models are downloaded on first use from pinned revisions. Glypho checks the registered size and SHA-256 before installing an artifact into the model cache.

Default cache:

```text
~/.glypho-ocr/models
```

Useful controls:

```text
GLYPHO_HOME
GLYPHO_MODELS
--offline
```

`--offline` disables model downloads completely.

## 🔌 Bindings

| Interface | Native path |
| --- | --- |
| Rust | direct library API |
| C | stable C ABI |
| Python | `ctypes` → native shared library |
| Node.js | persistent local `glypho serve` worker |
| Browser | separate TypeScript + ONNX Runtime Web runtime |

Python keeps matching native engines in a bounded process-wide cache. Node keeps one local worker alive per `Glypho` instance, so ONNX sessions survive between calls.

The browser does not spawn the native binary. It uses the same model family and result shape, but inference runs through ONNX Runtime Web with WebGPU or threaded WASM.

## 🧠 Sessions and memory

The native runtime is designed for reuse:

```python
from glypho import Glypho

ocr = Glypho(quality="balanced", device="auto")
ocr.warmup()

for image in images:
    print(ocr.recognize(image).text)
```

Important limits are applied before large allocations: input file size, decoded pixels, detected regions and output size are bounded. FFI calls catch Rust panics and return owned buffers through the stable C boundary.

## 📄 Result contract

All native entry points return the versioned `glypho.annotation.v1` document.

A result keeps:

- ordered plain text;
- line quadrilaterals in source-image pixels;
- confidence;
- language/script metadata when the route is unambiguous;
- optional alternative recognition candidates;
- engine metadata and timing.

The schema lives in [`../schemas/annotation.v1.schema.json`](../schemas/annotation.v1.schema.json).