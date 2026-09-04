<div align="center">

# Glypho for Node.js

**Fast multilingual OCR for Node.js with a persistent native runtime.**<br>
Local-first, TypeScript-friendly, and powered by the same native Glypho engine as the Rust package.

[![npm](https://img.shields.io/npm/v/glypho-ocr?style=flat-square&logo=npm)](https://www.npmjs.com/package/glypho-ocr)
[![Node.js](https://img.shields.io/badge/Node.js-%3E%3D20-339933?style=flat-square&logo=nodedotjs&logoColor=white)](https://nodejs.org/)
[![GitHub](https://img.shields.io/badge/GitHub-rinqaku%2FGlypho-181717?style=flat-square&logo=github)](https://github.com/rinqaku/Glypho)
[![License](https://img.shields.io/badge/license-Apache--2.0-4c8bf5?style=flat-square)](https://github.com/rinqaku/Glypho/blob/main/LICENSE)

[🌐 **Try Glypho Web**](https://glypho.kaneki.cz) · [**GitHub**](https://github.com/rinqaku/Glypho)

</div>

Glypho ships a small ESM wrapper plus a prebuilt native runtime for your platform. No Rust toolchain, Python environment or external OCR API is required.

Each `Glypho` instance keeps one native worker alive, so models and ONNX sessions stay warm between recognition calls instead of starting from scratch every time.

## 📦 Install

```bash
npm install glypho-ocr
```

Requires Node.js 20 or newer.

Prebuilt runtime packages are available for:

| OS | Architectures |
| --- | --- |
| Linux | x64, ARM64 |
| macOS | x64, Apple Silicon |
| Windows | x64, ARM64 |

npm selects the matching `glypho-ocr-<os>-<arch>` package automatically.

## 🚀 Quick start

```js
import { Glypho } from 'glypho-ocr';

const ocr = new Glypho({
  quality: 'balanced',
  device: 'auto',
});

try {
  const document = await ocr.recognize('screenshot.png');
  console.log(document.text);
} finally {
  await ocr.close();
}
```

Leaving `languages` empty enables automatic language routing.

You can also give hints when you already know what to expect:

```js
const ocr = new Glypho({
  languages: ['en', 'ja'],
  quality: 'accurate',
});
```

Common BCP-47-style values such as `cs-CZ`, `de-DE`, `ko-KR`, `zh-Hans`, `jpn` and `rus` are normalized automatically.

## ⚙️ Options

```js
const ocr = new Glypho({
  quality: 'balanced',       // fast | balanced | accurate | maximum
  device: 'auto',            // auto | cpu | cuda | coreml | openvino
  languages: [],             // [] = automatic routing
  threads: 8,
  offline: false,
  timeoutMs: 30_000,
});
```

Per-request options:

```js
const document = await ocr.recognize('image.png', {
  languages: ['en', 'ru'],
  segmentation: 'sparse_text',
  minConfidence: 0.8,
  timeoutMs: 20_000,
});
```

Segmentation modes:

```text
auto | single_block | single_line | sparse_text
```

## 🔥 Warm sessions

For repeated OCR, warm the runtime once and reuse the same instance:

```js
const ocr = new Glypho({ languages: ['en'] });

await ocr.warmup();

const first = await ocr.recognize('first.png');
const second = await ocr.recognize('second.png');

console.log(first.text);
console.log(second.text);

await ocr.close();
```

`info()` reports the resolved runtime and device:

```js
console.log(await ocr.info());
```

## ⏱️ Abort and timeouts

```js
const controller = new AbortController();

const result = await ocr.recognize('large-image.png', {
  signal: controller.signal,
  timeoutMs: 60_000,
});
```

`Glypho` rejects timed-out or aborted requests without killing the persistent worker.

## 🧩 CLI

The npm package also exposes the native `glypho` CLI:

```bash
npx glypho screenshot.png \
  --language en,ja \
  --quality balanced \
  --device auto
```

Runtime information:

```bash
npx glypho info --pretty
```

## 💾 Models

Missing model files are downloaded from pinned revisions, verified with SHA-256 and cached under:

```text
~/.glypho-ocr/models
```

Use another location with `GLYPHO_HOME`, `GLYPHO_MODELS` or the `models` option.

To prohibit downloads completely:

```js
const ocr = new Glypho({ offline: true });
```

## 🔒 Local-first

Recognition runs locally. Glypho does not upload input images to an OCR API.

The only network access normally needed is the first model download. Once the required models are cached, OCR can run offline.

---

Full project: [github.com/rinqaku/Glypho](https://github.com/rinqaku/Glypho)<br>
Web preview: [glypho.kaneki.cz](https://glypho.kaneki.cz)<br>
Rust package: [crates.io/crates/glypho-ocr](https://crates.io/crates/glypho-ocr)<br>
License: [Apache-2.0](https://github.com/rinqaku/Glypho/blob/main/LICENSE)
