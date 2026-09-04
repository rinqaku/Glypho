# Hardware acceleration

Glypho uses the same `device` option in Rust, Python, Node.js and the CLI:

```text
auto | cpu | cuda | coreml | openvino
```

## ⚡ Auto mode

`auto` probes accelerated providers available in the current build and falls back to CPU when necessary.

Use `info()` to see what actually happened:

```bash
glypho info --device auto --pretty
```

The result includes both the requested device and the resolved device, plus a fallback reason when an accelerator could not be used.

## CUDA

CUDA is the main NVIDIA path.

Custom Rust build:

```bash
cargo install glypho-ocr --features cuda
```

Or from the repository:

```bash
cargo build --release --locked -p glypho-ocr --features cuda
```

Check it:

```bash
glypho info --device cuda --pretty
glypho image.png --device cuda
```

Small OCR models can be faster on CPU because GPU transfer and dispatch overhead may cost more than the inference itself. `maximum` is much more likely to benefit from CUDA than `fast` or `balanced`, so benchmark on the hardware you actually target.

## CoreML

CoreML is available on macOS builds with the `coreml` feature:

```bash
cargo install glypho-ocr --features coreml
```

Then:

```bash
glypho info --device coreml --pretty
```

Published macOS release artifacts are built with CoreML support and retain CPU fallback.

## OpenVINO

The official ONNX Runtime native archives used by Glypho do not include the OpenVINO execution provider. OpenVINO therefore uses a caller-supplied compatible ONNX Runtime.

Build Glypho:

```bash
cargo install glypho-ocr --features openvino
```

Point Glypho at an OpenVINO-enabled ONNX Runtime:

```bash
export ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
glypho info --device openvino --pretty
```

On Windows use `onnxruntime.dll`; on macOS use `libonnxruntime.dylib`.

The external runtime must match the ONNX Runtime API version expected by the current Glypho build.

## CPU

CPU is always the safe fallback and is often the best choice for the smaller profiles.

```bash
glypho image.png --device cpu --threads 4
```

`threads` controls ONNX CPU worker threads. More threads are not automatically faster; keep the setting stable when benchmarking.

## 🌐 Browser

Glypho Web uses a separate browser runtime:

- WebGPU for larger graphs when useful;
- threaded WASM for broad compatibility and small models;
- Web Worker inference so model initialization does not block React.

Try it directly at **https://glypho.kaneki.cz**.

For local development:

```bash
cd web
npm install
npm run dev
```