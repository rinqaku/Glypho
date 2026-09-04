<p align="center">
  <a href="../README.md">English</a> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.cs.md">Čeština</a> ·
  <strong>日本語</strong>
</p>

<div align="center">

# Glypho

**Rust コアで動く、高速な多言語 OCR。**<br>
ローカル実行を前提に設計され、Rust / Python / Node.js / ブラウザから同じ OCR エンジンを扱えます。

[![Crates.io](https://img.shields.io/crates/v/glypho-ocr?style=flat-square&logo=rust)](https://crates.io/crates/glypho-ocr)
[![PyPI](https://img.shields.io/pypi/v/glypho-ocr?style=flat-square&logo=python)](https://pypi.org/project/glypho-ocr/)
[![npm](https://img.shields.io/npm/v/glypho-ocr?style=flat-square&logo=npm)](https://www.npmjs.com/package/glypho-ocr)
[![License](https://img.shields.io/badge/license-Apache--2.0-4c8bf5?style=flat-square)](../LICENSE)

</div>

<p align="center">
  <img src="assets/glypho-demo-en.png" alt="ギター写真の文字を認識する Glypho Web" width="100%">
</p>

<p align="center">
  <a href="https://glypho.kaneki.cz"><strong>🌐 Glypho Web を試す</strong></a>
</p>

Glypho はスクリーンショット、写真、UI テキストなど、**画像を外部 OCR API に送らずに高精度で読み取りたい場面**のために作られています。

ネイティブエンジンは Rust 製で、PP-OCR モデルを ONNX Runtime 上で実行します。必要なモデルだけを取得し、SHA-256 で検証してローカルにキャッシュします。ウォームアップ済みの detector / recognizer session はメモリに保持されるため、2 回目以降の OCR で毎回初期化し直す必要はありません。

---

## ⚡ Glypho の特徴

- **Rust ネイティブコア** — bounded decode、routing、batching、model store、inference orchestration を担当。
- **多言語対応** — Latin、East Slavic、中国語、日本語、韓国語を含む 55 個の canonical BCP-47 identifier。
- **Local-first** — 入力画像はローカルで処理。OCR API のアカウントや API key は不要。
- **ハードウェアアクセラレーション** — CPU / CUDA / CoreML / OpenVINO-aware build と CPU fallback。
- **Warm inference** — 1 つの engine を使い回し、session をメモリに保持。
- **1 つの package name** — Cargo / PyPI / npm すべて `glypho-ocr`。
- **ブラウザ版** — WebGPU + threaded WASM、OCR backend server 不要。

---

## 📦 インストール

### Python

```bash
pip install glypho-ocr
```

### Rust

```bash
cargo add glypho-ocr
```

CLI のみ:

```bash
cargo install glypho-ocr
```

### Node.js

```bash
npm install glypho-ocr
```

公開されている Python wheel と npm platform package にはネイティブ runtime が含まれています。利用者側で **Rust / Cargo / `libglypho.so` を別途用意する必要はありません**。

モデルの既定 cache:

```text
~/.glypho-ocr/models
```

`GLYPHO_HOME` または `GLYPHO_MODELS` で変更できます。

---

## 🚀 クイックスタート

### Python

```python
from glypho import Glypho

ocr = Glypho(
    languages=("ja", "en"),
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

オプション指定:

```bash
glypho screenshot.png \
  --language ja,en \
  --quality accurate \
  --device cuda
```

JSON 出力:

```bash
glypho screenshot.png --format json --output result.json
```

### Rust

```rust
use glypho::{Device, OnnxConfig, OnnxEngine, RecognitionOptions};

let mut config = OnnxConfig::default();
config.device = Device::Auto;

let ocr = OnnxEngine::new(config)?;
ocr.warmup(&["ja".to_owned(), "en".to_owned()])?;

let document = ocr.recognize(
    "screenshot.png",
    &RecognitionOptions::default(),
)?;

println!("{}", document.text);
# Ok::<(), Box<dyn std::error::Error>>(())
```

---

## 🌍 多言語 OCR

Glypho は **55 個の canonical language identifier** を公開し、実際にその文字体系を扱える dictionary を持つ recognizer に routing します。

現在の主な routes:

- English / Czech / German / French / Spanish / Polish / Turkish / Vietnamese などの Latin-script languages;
- Russian / Ukrainian / Belarusian;
- Chinese;
- Japanese;
- Korean.

`de-DE`、`cs-CZ`、`ko-KR` のような region tag や、`eng`、`jpn`、`rus` などの一般的な ISO alias は自動で正規化されます。

<p align="center">
  <img src="assets/glypho-demo-ja.png" alt="日本語テキストを認識する Glypho Web" width="100%">
</p>

完全な一覧と routing rules は [`../docs/LANGUAGES.md`](../docs/LANGUAGES.md) を参照してください。

> `fast` profile では日本語を意図的に無効にしています。Tiny recognizer の dictionary に kana が含まれていないためです。

---

## 🎯 Quality profiles

Glypho は release ごとに挙動が勝手に変わらないよう、固定された model profile を使います。

| Profile | Detector | Primary recognizer | 用途 |
| --- | --- | --- | --- |
| `fast` | PP-OCRv6 Tiny | PP-OCRv6 Tiny | 最低 latency / 軽量端末 |
| `balanced` | PP-OCRv5 Mobile | PP-OCRv6 Small | 通常用途・既定値 |
| `accurate` | PP-OCRv6 Small | PP-OCRv6 Small | 小さい文字・難しい画像 |
| `maximum` | PP-OCRv6 Medium | PP-OCRv6 Medium | accuracy-first |

Latin、East Slavic、Korean では軽量な PP-OCRv5 specialist recognizer を利用できます。Mixed-script request でも detector pass は 1 回だけで、同じ perspective crop を各 recognizer で再利用します。

---

## 🖥️ ハードウェアアクセラレーション

ネイティブ API が受け付ける `device`:

```text
auto | cpu | cuda | coreml | openvino
```

`auto` の probe order:

```text
NVIDIA / CUDA → Apple CoreML → Intel OpenVINO → CPU
```

accelerator が利用できない場合は CPU に fallback し、理由は `info()` から確認できます。

カスタム Rust build:

```bash
cargo install glypho-ocr --features cuda
cargo install glypho-ocr --features coreml
```

OpenVINO は OpenVINO provider を含む外部 ONNX Runtime を使用します: [`../docs/HARDWARE.md`](../docs/HARDWARE.md)。

---

## 🌐 Glypho Web

`web/` には Glypho のブラウザ版が入っています。

使用技術:

- React + TypeScript;
- ONNX Runtime Web;
- WebGPU + threaded WASM;
- inference 用 Web Worker;
- model の browser cache;
- model artifact の SHA-256 検証。

入力画像そのものをサーバーへ送る必要はありません。

```bash
cd web
npm install
npm run dev
```

Auto multilingual mode では軽量 language pack を並列で cache し、specialist session は必要になった時だけ lazy に生成します。

---

## 🔒 Local-first design

Glypho に cloud OCR backend は必要ありません。

- Native の入力画像は端末内に残ります。
- Web の入力画像はページ内に残ります。
- model download は size limit と SHA-256 検証付きです。
- `--offline` で model download を完全に禁止できます。
- 結果 schema は `glypho.annotation.v1` として versioning されています。

---

## 📄 ライセンス

Glypho は **Apache-2.0** で公開されています。

PP-OCR の model artifact は PaddlePaddle から Apache-2.0 で配布され、初回利用時に別途ダウンロードされます。
