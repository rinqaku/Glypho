<p align="center">
  <a href="../README.md">English</a> ·
  <strong>Русский</strong> ·
  <a href="README.cs.md">Čeština</a> ·
  <a href="README.ja.md">日本語</a>
</p>

<div align="center">

# Glypho

**Быстрый мультиязычный OCR с ядром на Rust.**<br>
Работает локально, легко встраивается и доступен из Rust, Python, Node.js и прямо в браузере.

[![Crates.io](https://img.shields.io/crates/v/glypho-ocr?style=flat-square&logo=rust)](https://crates.io/crates/glypho-ocr)
[![PyPI](https://img.shields.io/pypi/v/glypho-ocr?style=flat-square&logo=python)](https://pypi.org/project/glypho-ocr/)
[![npm](https://img.shields.io/npm/v/glypho-ocr?style=flat-square&logo=npm)](https://www.npmjs.com/package/glypho-ocr)
[![License](https://img.shields.io/badge/license-Apache--2.0-4c8bf5?style=flat-square)](../LICENSE)

</div>

<p align="center">
  <img src="assets/glypho-demo-en.png" alt="Glypho Web распознаёт текст на фотографии гитары" width="100%">
</p>

<p align="center">
  <a href="https://glypho.rinqaku.dev"><strong>🌐 Открыть Glypho Web</strong></a>
</p>

## 🏆 CPU-бенчмарк ICDAR 2015

> **№1 по качеству + №1 по прогретой задержке в нашем внутреннем сравнении 6 OCR SDK.**
> На всех 500 тестовых изображениях ICDAR 2015 Glypho Balanced получил **0.7129 Det. H**, **0.5972 E2E H**, **172 мс p50** и **232 мс p95** на Intel Core i5-12600KF (только CPU).

| Движок | Det. H ↑ | E2E H ↑ | p50 ↓ | p95 ↓ |
| --- | ---: | ---: | ---: | ---: |
| **Glypho Balanced** | **0.7129** | **0.5972** | **172 мс** | **232 мс** |
| ppu-paddle-ocr | 0.2025 | 0.0998 | 195 мс | 517 мс |
| Tesseract 5 | 0.0697 | 0.0480 | 262 мс | 589 мс |
| docTR | 0.5739 | 0.4107 | 728 мс | 1.08 с |
| EasyOCR | 0.6061 | 0.2188 | 1.77 с | 2.02 с |
| Surya Classic | 0.1687 | 0.1271 | 5.80 с | 14.33 с |

В отдельном сравнении Glypho и PPU с **сопоставимым размером моделей** Glypho оказался **в 2.13×–2.65× быстрее по p50** и достиг **3.55×–7.73× E2E H-mean** на парах Tiny, Small и Medium.

<sub>Внутренний SDK-бенчмарк, не официальный результат RRC leaderboard. Для всех движков использованы одинаковые dataset, машина и evaluator. [Методика и полные результаты →](../docs/BENCHMARKS.md)</sub>

Glypho сделан для скриншотов, фотографий, интерфейсов и обычного OCR, где хочется **хорошей точности без отправки картинки в чужой API**.

Нативное ядро написано на Rust и использует ONNX Runtime вместе с PP-OCR. Модели скачиваются только когда нужны, проверяются через SHA-256 и кешируются локально. Прогретые detector/recognizer sessions остаются в памяти, поэтому повторное распознавание не платит за инициализацию заново.

---

## ⚡ Зачем Glypho?

- **Нативное ядро на Rust** — decode, routing, batching, model store и inference orchestration.
- **Мультиязычность** — 55 canonical BCP-47 идентификаторов: латиница, East Slavic, китайский, японский и корейский.
- **Local-first** — картинки обрабатываются локально; API key и аккаунт OCR-сервиса не нужны.
- **Аппаратное ускорение** — CPU, CUDA, CoreML и OpenVINO-aware сборки с fallback на CPU.
- **Прогретые сессии** — один engine можно переиспользовать между изображениями.
- **Одно имя пакета** — `glypho-ocr` в Cargo, PyPI и npm.
- **Web-версия** — WebGPU + threaded WASM, без OCR backend-сервера.

---

## 📦 Установка

### CLI — одной командой

Linux / macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/rinqaku/Glypho/main/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/rinqaku/Glypho/main/install.ps1 | iex
```

Скрипт скачивает подходящий архив из последнего GitHub Release, проверяет SHA-256 и устанавливает CLI `glypho`. Готовые сборки есть для Linux x64/ARM64, macOS ARM64 и Windows x64/ARM64.

### Python

```bash
pip install glypho-ocr
```

### Rust

```bash
cargo add glypho-ocr
```

Только CLI:

```bash
cargo install glypho-ocr
```

### Node.js

```bash
npm install glypho-ocr
```

Python wheels и npm platform packages уже содержат нативный runtime. Пользователю **не нужен Rust, Cargo или отдельно установленный `libglypho.so`**.

Модели по умолчанию кешируются в:

```text
~/.glypho-ocr/models
```

Путь можно изменить через `GLYPHO_HOME` или `GLYPHO_MODELS`.

---

## 🚀 Быстрый старт

### Python

```python
from glypho import Glypho

ocr = Glypho(
    languages=("ru", "en"),
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
  languages: ["ru", "en"],
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

С настройками:

```bash
glypho screenshot.png \
  --language ru,en \
  --quality accurate \
  --device cuda
```

JSON:

```bash
glypho screenshot.png --format json --output result.json
```

### Rust

```rust
use glypho::{Device, OnnxConfig, OnnxEngine, RecognitionOptions};

let mut config = OnnxConfig::default();
config.device = Device::Auto;

let ocr = OnnxEngine::new(config)?;
ocr.warmup(&["ru".to_owned(), "en".to_owned()])?;

let document = ocr.recognize(
    "screenshot.png",
    &RecognitionOptions::default(),
)?;

println!("{}", document.text);
# Ok::<(), Box<dyn std::error::Error>>(())
```

---

## 🌍 Языки

Glypho поддерживает **55 canonical language identifiers** и направляет запрос в recognizer, словарь которого реально содержит нужный алфавит.

Сейчас есть маршруты для:

- языков на латинице — английский, чешский, немецкий, французский, испанский, польский, турецкий, вьетнамский и многие другие;
- русского, украинского и белорусского;
- китайского;
- японского;
- корейского.

Обычные ISO aliases и region tags вроде `de-DE`, `cs-CZ`, `ko-KR`, `eng`, `jpn`, `rus` нормализуются автоматически.

<p align="center">
  <img src="assets/glypho-demo-ja.png" alt="Glypho Web распознаёт японский текст" width="100%">
</p>

Полный список и routing: [`../docs/LANGUAGES.md`](../docs/LANGUAGES.md).

> Профиль `fast` специально не заявляет поддержку японского: словарь Tiny recognizer не содержит kana.

---

## 🎯 Профили качества

Glypho использует фиксированные model profiles, чтобы поведение не менялось неожиданно между релизами.

| Профиль | Detector | Основной recognizer | Для чего |
| --- | --- | --- | --- |
| `fast` | PP-OCRv6 Tiny | PP-OCRv6 Tiny | минимальная задержка / слабые устройства |
| `balanced` | PP-OCRv6 Small | PP-OCRv6 Small | обычный OCR, профиль по умолчанию |
| `accurate` | PP-OCRv6 Small | PP-OCRv6 Small | мелкий и сложный текст |
| `maximum` | PP-OCRv6 Medium | PP-OCRv6 Medium | когда точность важнее скорости |

Для латиницы, East Slavic и корейского Glypho может использовать компактные specialist recognizers PP-OCRv5. При mixed-script запросе detector запускается один раз, а perspective crops переиспользуются между recognizers.

---

## 🖥️ Ускорение

Нативные API принимают:

```text
auto | cpu | cuda | coreml | openvino
```

`auto` проверяет providers в таком порядке:

```text
NVIDIA / CUDA → Apple CoreML → Intel OpenVINO → CPU
```

Если ускоритель недоступен, Glypho уходит на CPU и показывает причину через `info()`.

Для собственной Rust-сборки:

```bash
cargo install glypho-ocr --features cuda
cargo install glypho-ocr --features coreml
```

OpenVINO использует отдельно предоставленный ONNX Runtime с этим provider: [`../docs/HARDWARE.md`](../docs/HARDWARE.md).

---

## 🌐 Glypho Web

В `web/` лежит браузерная версия Glypho.

Стек:

- React + TypeScript;
- ONNX Runtime Web;
- WebGPU и threaded WASM;
- Web Worker для inference;
- browser cache моделей;
- SHA-256 проверка model artifacts.

Сама картинка никуда с устройства не отправляется.

```bash
cd web
npm install
npm run dev
```

В auto multilingual режиме Web параллельно кеширует лёгкие language packs, а specialist sessions создаёт лениво, только когда они реально нужны.

---

## 🔒 Local-first по умолчанию

Glypho не требует облачного OCR backend.

- Native изображения остаются на машине.
- Web изображения остаются внутри страницы.
- Модели ограничиваются по размеру и проверяются SHA-256.
- `--offline` полностью запрещает загрузки.
- Результат версионирован как `glypho.annotation.v1`.

---

## 📄 Лицензия

Glypho распространяется под **Apache-2.0**.

PP-OCR model artifacts публикуются PaddlePaddle под Apache-2.0 и скачиваются отдельно при первом использовании.
