# Glypho for Python

```bash
pip install glypho-ocr
```

The platform wheel contains the native Rust library and the `glypho` executable. The Python API calls the stable C ABI directly, retaining ONNX sessions in a bounded process-wide cache.

```python
from glypho import Glypho

ocr = Glypho(
    languages=('en', 'ru'),
    quality='balanced',
    device='auto',
)
ocr.warmup()
document = ocr.recognize('image.png')
print(document.text)
print(ocr.info())
```

Missing models are downloaded from pinned revisions, verified, and cached under `~/.glypho-ocr/models`. Use `offline=True` to prohibit downloads. Ordered BCP-47 hints select only the required multilingual or specialist language packs.

The native runtime supports `auto`, `cpu`, `cuda`, `coreml`, and `openvino`. If an accelerated provider cannot initialize, it falls back to CPU and exposes the reason through `info()`.

The complete lifecycle, concurrency, results, language, and troubleshooting guide is in [`../docs/PYTHON.md`](../docs/PYTHON.md).