# Glypho native runtime

This is a platform-specific runtime package used by [`glypho-ocr`](https://www.npmjs.com/package/glypho-ocr).

You normally **should not install this package directly**:

```bash
npm install glypho-ocr
```

npm selects the matching native package automatically through `optionalDependencies`.

The package contains the native `glypho` executable and `libglypho` for one operating system and CPU architecture. OCR runs locally; required model files are downloaded separately on first use, verified and cached under `~/.glypho-ocr/models`.

- 🌐 [Glypho Web](https://glypho.kaneki.cz)
- GitHub: [rinqaku/Glypho](https://github.com/rinqaku/Glypho)
- License: Apache-2.0
