# Glypho for Node.js

```bash
npm install glypho-ocr
```

The wrapper installs the matching `glypho-ocr-<os>-<arch>` package automatically. It starts one local NDJSON worker per `Glypho` instance, so ONNX sessions remain in memory between calls. No shell is involved and image data never leaves the machine.

```js
import { Glypho } from 'glypho-ocr';

const ocr = new Glypho({
  languages: ['en', 'ru'],
  quality: 'balanced',
  device: 'auto',
});

await ocr.warmup();
const document = await ocr.recognize('image.png');
console.log(document.text);
console.log(await ocr.info());
await ocr.close();
```

Missing models are downloaded from pinned revisions, verified, and cached under `~/.glypho-ocr/models`. Set `offline: true` to prohibit downloads. `GLYPHO_BIN` or the `binary` option can select a trusted custom executable.

Language hints accept BCP-47-style values such as `ko-KR`, `zh-Hans`, `cs-CZ`, and `de-DE`. Device choices are `auto`, `cpu`, `cuda`, `coreml`, and `openvino`; `info()` reports the actual provider and any CPU fallback reason.
