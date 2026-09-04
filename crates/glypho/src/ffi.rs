use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use serde::Deserialize;

use crate::{
    Device, Glypho, OnnxConfig, OnnxEngine, PageSegmentation, QualityMode, RecognitionOptions,
    TesseractConfig, default_models_dir,
};

const STATUS_OK: i32 = 0;
const STATUS_ERROR: i32 = 1;
const STATUS_INVALID_INPUT: i32 = 2;
const STATUS_PANIC: i32 = 255;
const MAX_CACHED_ONNX_ENGINES: usize = 8;

static ONNX_ENGINE_CACHE: OnceLock<Mutex<HashMap<OnnxCacheKey, Arc<OnnxEngine>>>> = OnceLock::new();

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct OnnxCacheKey {
    models_dir: PathBuf,
    quality: QualityMode,
    device: Device,
    auto_download: bool,
    threads: usize,
}

#[repr(C)]
pub struct GlyphoBuffer {
    pub data: *mut u8,
    pub len: usize,
    pub capacity: usize,
}

#[repr(C)]
pub struct GlyphoFfiResult {
    pub status: i32,
    pub body: GlyphoBuffer,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FfiOptions {
    backend: FfiBackend,
    languages: Vec<String>,
    models: Option<String>,
    quality: QualityMode,
    device: Device,
    offline: bool,
    segmentation: PageSegmentation,
    min_confidence: f32,
    tesseract: String,
    threads: Option<usize>,
    timeout_ms: u64,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FfiBackend {
    #[default]
    Auto,
    Onnx,
    Tesseract,
}

impl Default for FfiOptions {
    fn default() -> Self {
        Self {
            backend: FfiBackend::Auto,
            languages: Vec::new(),
            models: None,
            quality: QualityMode::default(),
            device: Device::Auto,
            offline: false,
            segmentation: PageSegmentation::default(),
            min_confidence: 0.8,
            tesseract: "tesseract".to_owned(),
            threads: None,
            timeout_ms: 30_000,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glypho_recognize_json(
    path_data: *const u8,
    path_len: usize,
    options_data: *const u8,
    options_len: usize,
) -> GlyphoFfiResult {
    match catch_unwind(AssertUnwindSafe(|| {
        recognize_json(path_data, path_len, options_data, options_len)
    })) {
        Ok(Ok(json)) => ffi_result(STATUS_OK, json),
        Ok(Err((status, message))) => ffi_result(status, message.into_bytes()),
        Err(_) => ffi_result(STATUS_PANIC, b"Glypho panicked".to_vec()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glypho_warmup_json(
    options_data: *const u8,
    options_len: usize,
) -> GlyphoFfiResult {
    match catch_unwind(AssertUnwindSafe(|| warmup_json(options_data, options_len))) {
        Ok(Ok(json)) => ffi_result(STATUS_OK, json),
        Ok(Err((status, message))) => ffi_result(status, message.into_bytes()),
        Err(_) => ffi_result(STATUS_PANIC, b"Glypho panicked".to_vec()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glypho_info_json(
    options_data: *const u8,
    options_len: usize,
) -> GlyphoFfiResult {
    match catch_unwind(AssertUnwindSafe(|| info_json(options_data, options_len))) {
        Ok(Ok(json)) => ffi_result(STATUS_OK, json),
        Ok(Err((status, message))) => ffi_result(status, message.into_bytes()),
        Err(_) => ffi_result(STATUS_PANIC, b"Glypho panicked".to_vec()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn glypho_buffer_free(buffer: GlyphoBuffer) {
    if buffer.data.is_null() {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(
            buffer.data,
            buffer.len,
            buffer.capacity,
        ));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn glypho_version() -> *const u8 {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr()
}

fn recognize_json(
    path_data: *const u8,
    path_len: usize,
    options_data: *const u8,
    options_len: usize,
) -> std::result::Result<Vec<u8>, (i32, String)> {
    let path_bytes = raw_bytes(path_data, path_len)?;
    let path = std::str::from_utf8(path_bytes)
        .map_err(|_| (STATUS_INVALID_INPUT, "path must be valid UTF-8".to_owned()))?;
    if path.is_empty() {
        return Err((STATUS_INVALID_INPUT, "path must not be empty".to_owned()));
    }

    let options = parse_options(options_data, options_len)?;

    let config = onnx_config(&options);
    let recognition = RecognitionOptions {
        languages: options.languages,
        segmentation: options.segmentation,
        min_confidence: options.min_confidence,
    };
    let document = if let Some(config) = config {
        recognize_with_cached_onnx(config, Path::new(path), &recognition)
    } else {
        Glypho::new(TesseractConfig {
            binary: PathBuf::from(options.tesseract),
            timeout: Duration::from_millis(options.timeout_ms),
        })
        .recognize(PathBuf::from(path), &recognition)
    }
    .map_err(|error| (STATUS_ERROR, error.to_string()))?;

    serde_json::to_vec(&document).map_err(|error| (STATUS_ERROR, error.to_string()))
}

fn warmup_json(
    options_data: *const u8,
    options_len: usize,
) -> std::result::Result<Vec<u8>, (i32, String)> {
    let options = parse_options(options_data, options_len)?;
    let Some(config) = onnx_config(&options) else {
        return Err((
            STATUS_INVALID_INPUT,
            "warmup requires the native ONNX backend".to_owned(),
        ));
    };
    let engine = cached_onnx_engine(config).map_err(|error| (STATUS_ERROR, error.to_string()))?;
    engine
        .warmup(&options.languages)
        .map_err(|error| (STATUS_ERROR, error.to_string()))?;
    Ok(br#"{"warmed":true}"#.to_vec())
}

fn info_json(
    options_data: *const u8,
    options_len: usize,
) -> std::result::Result<Vec<u8>, (i32, String)> {
    let options = parse_options(options_data, options_len)?;
    if let Some(config) = onnx_config(&options) {
        let engine =
            cached_onnx_engine(config).map_err(|error| (STATUS_ERROR, error.to_string()))?;
        return serde_json::to_vec(&engine.info())
            .map_err(|error| (STATUS_ERROR, error.to_string()));
    }
    let engine = Glypho::new(TesseractConfig {
        binary: PathBuf::from(options.tesseract),
        timeout: Duration::from_millis(options.timeout_ms),
    });
    let info = engine
        .info()
        .map_err(|error| (STATUS_ERROR, error.to_string()))?;
    serde_json::to_vec(&info).map_err(|error| (STATUS_ERROR, error.to_string()))
}

fn parse_options(
    options_data: *const u8,
    options_len: usize,
) -> std::result::Result<FfiOptions, (i32, String)> {
    let options = if options_len == 0 {
        FfiOptions::default()
    } else {
        let bytes = raw_bytes(options_data, options_len)?;
        serde_json::from_slice(bytes).map_err(|error| {
            (
                STATUS_INVALID_INPUT,
                format!("invalid options JSON: {error}"),
            )
        })?
    };
    if options.timeout_ms == 0 || options.timeout_ms > 300_000 {
        return Err((
            STATUS_INVALID_INPUT,
            "timeout_ms must be between 1 and 300000".to_owned(),
        ));
    }
    Ok(options)
}

fn onnx_config(options: &FfiOptions) -> Option<OnnxConfig> {
    let models = options
        .models
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(default_models_dir);
    let use_onnx = match options.backend {
        FfiBackend::Onnx => true,
        FfiBackend::Tesseract => false,
        FfiBackend::Auto => true,
    };
    if !use_onnx {
        return None;
    }

    let mut config = OnnxConfig::new(models);
    config.quality = options.quality;
    config.device = options.device;
    config.auto_download = !options.offline;
    if let Some(threads) = options.threads {
        config.threads = threads;
    }
    Some(config)
}

fn recognize_with_cached_onnx(
    config: OnnxConfig,
    path: &Path,
    recognition: &RecognitionOptions,
) -> crate::Result<crate::Document> {
    cached_onnx_engine(config)?.recognize(path, recognition)
}

fn cached_onnx_engine(mut config: OnnxConfig) -> crate::Result<Arc<OnnxEngine>> {
    if config.models_dir.is_relative() {
        let current_dir = std::env::current_dir().map_err(|error| crate::Error::io(".", error))?;
        config.models_dir = current_dir.join(&config.models_dir);
    }
    config.models_dir = config
        .models_dir
        .canonicalize()
        .unwrap_or_else(|_| config.models_dir.clone());
    let key = OnnxCacheKey {
        models_dir: config.models_dir.clone(),
        quality: config.quality,
        device: config.device,
        auto_download: config.auto_download,
        threads: config.threads,
    };
    let mut cache = engine_cache()?;
    if let Some(engine) = cache.get(&key).cloned() {
        return Ok(engine);
    }

    let engine = Arc::new(OnnxEngine::new(config)?);
    if cache.len() >= MAX_CACHED_ONNX_ENGINES
        && let Some(evicted) = cache.keys().next().cloned()
    {
        cache.remove(&evicted);
    }
    cache.insert(key, Arc::clone(&engine));
    Ok(engine)
}

fn engine_cache() -> crate::Result<MutexGuard<'static, HashMap<OnnxCacheKey, Arc<OnnxEngine>>>> {
    ONNX_ENGINE_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| crate::Error::Backend {
            backend: "onnxruntime",
            message: "ONNX engine cache lock was poisoned".to_owned(),
        })
}

fn raw_bytes<'a>(data: *const u8, len: usize) -> std::result::Result<&'a [u8], (i32, String)> {
    if len == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err((
            STATUS_INVALID_INPUT,
            "non-empty input has a null pointer".to_owned(),
        ));
    }
    Ok(unsafe { slice::from_raw_parts(data, len) })
}

fn ffi_result(status: i32, mut body: Vec<u8>) -> GlyphoFfiResult {
    if body.is_empty() {
        return GlyphoFfiResult {
            status,
            body: GlyphoBuffer {
                data: ptr::null_mut(),
                len: 0,
                capacity: 0,
            },
        };
    }

    let buffer = GlyphoBuffer {
        data: body.as_mut_ptr(),
        len: body.len(),
        capacity: body.capacity(),
    };
    std::mem::forget(body);
    GlyphoFfiResult {
        status,
        body: buffer,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_options_without_panicking() {
        let path = b"missing.png";
        let options = b"{";
        let result = unsafe {
            glypho_recognize_json(path.as_ptr(), path.len(), options.as_ptr(), options.len())
        };

        assert_eq!(result.status, STATUS_INVALID_INPUT);
        let message = unsafe { slice::from_raw_parts(result.body.data, result.body.len) };
        assert!(String::from_utf8_lossy(message).contains("invalid options JSON"));
        unsafe { glypho_buffer_free(result.body) };
    }

    #[test]
    fn rejects_unknown_options() {
        let path = b"missing.png";
        let options = br#"{"unknown":true}"#;
        let result = unsafe {
            glypho_recognize_json(path.as_ptr(), path.len(), options.as_ptr(), options.len())
        };

        assert_eq!(result.status, STATUS_INVALID_INPUT);
        unsafe { glypho_buffer_free(result.body) };
    }

    #[test]
    fn warmup_rejects_non_persistent_backend() {
        let options = br#"{"backend":"tesseract"}"#;
        let result = unsafe { glypho_warmup_json(options.as_ptr(), options.len()) };

        assert_eq!(result.status, STATUS_INVALID_INPUT);
        let message = unsafe { slice::from_raw_parts(result.body.data, result.body.len) };
        assert!(String::from_utf8_lossy(message).contains("native ONNX backend"));
        unsafe { glypho_buffer_free(result.body) };
    }
}
