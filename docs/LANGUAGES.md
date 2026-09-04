# Language support

Glypho accepts BCP-47-style language hints and routes text to recognizers whose dictionaries actually cover the requested script.

## 🌍 Current routes

| Route | Hints | Recognizer |
| --- | --- | --- |
| Chinese | `zh`, `zh-Hans`, `zh-Hant` | profile unified recognizer |
| Japanese | `ja` | profile unified recognizer |
| Korean | `ko` | PP-OCRv5 Korean pack |
| East Slavic | `be`, `ru`, `uk` | PP-OCRv5 Eastern Slavic pack |
| Latin | list below | PP-OCRv5 Latin pack |
| English | `en` | profile unified recognizer |

The Latin route currently accepts:

```text
af az bs ca cs cy da de en es et eu fi fr ga gl hr hu id is it jv ku
la lb lt lv mi ms mt nl no oc pi pl pt qu rm ro sk sl sq sr-Latn sv sw
tl tr uz vi
```

Together with Chinese, Japanese, Korean and East Slavic routes, Glypho exposes **55 canonical language identifiers**.

> `fast` does not support Japanese because the Tiny recognizer dictionary does not contain kana.

## Aliases

Common ISO aliases and region tags are normalized automatically.

Examples:

```text
eng      → en
deu      → de
ger      → de
ces/cze  → cs
fra/fre  → fr
jpn      → ja
kor      → ko
rus      → ru
ukr      → uk
cs-CZ    → cs
de-DE    → de
ko-KR    → ko
```

`sr-Latn` keeps its script subtag because plain `sr` would be ambiguous.

## 🚀 Examples

CLI:

```bash
glypho image.png --language en,ru
```

Python:

```python
document = ocr.recognize("image.png", languages=["cs-CZ", "en"])
```

Node.js:

```js
const document = await ocr.recognize("image.png", {
  languages: ["ja", "en"],
});
```

Language hints select recognition routes. They do **not** translate text or apply spelling correction.

## No language hint

Rust, Python, Node.js, the CLI and Glypho Web all default to automatic multilingual mode when `languages`/`--language` is omitted. The profile recognizer handles English, Chinese and Japanese first; ambiguous or script-relevant crops are then routed through the Latin, Eastern-Slavic and Korean specialists. Explicit hints remain useful when the script is known because they reduce model downloads, memory use and latency.

In API results an empty `language_hints` list means auto mode. A line receives a language tag only when its script identifies one unambiguous supported language, such as Japanese or Korean.

The optional legacy Tesseract compatibility backend is the exception: Tesseract has no equivalent reliable language detector, so an empty hint selects its installed English pack. The default native ONNX backend remains automatic.

## Mixed scripts

A mixed request still performs only one detector pass:

```text
image
  → detector
  → shared perspective crops
  → unified / Latin / Cyrillic / Korean recognition as needed
  → candidate merge
```

This keeps multilingual OCR cheaper than running the full pipeline once per language.

## Not supported yet

Arabic-family scripts, Greek, Thai, broad Cyrillic beyond East Slavic, Devanagari, Tamil and Telugu are not currently exposed as supported routes.
