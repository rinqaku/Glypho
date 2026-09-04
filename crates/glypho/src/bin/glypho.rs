use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand, ValueEnum};
use glypho::{
    Device, Document, Glypho, OnnxConfig, OnnxEngine, PageSegmentation, QualityMode,
    RecognitionOptions, TesseractConfig, default_models_dir, evaluate,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Parser)]
#[command(name = "glypho", version, about = "Fast, local-first OCR")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    /// Image to recognize. This is the short form of `glypho recognize IMAGE`.
    #[arg(value_name = "IMAGE")]
    input: Option<PathBuf>,
    #[arg(short = 'l', long = "language", value_delimiter = ',')]
    languages: Vec<String>,
    #[arg(long, value_enum, default_value = "auto")]
    device: DeviceArg,
    #[arg(long, value_enum, default_value = "balanced")]
    quality: QualityArg,
    #[arg(long, help = "Glypho model cache directory")]
    models: Option<PathBuf>,
    #[arg(long, help = "ONNX worker threads (default: up to 8)")]
    threads: Option<usize>,
    #[arg(long, default_value_t = 0.8)]
    min_confidence: f32,
    #[arg(long, value_enum, default_value = "sparse-text")]
    segmentation: SegmentationArg,
    #[arg(long, value_enum, default_value = "text")]
    format: OutputFormat,
    #[arg(short, long)]
    output: Option<PathBuf>,
    #[arg(long, help = "Disable model downloads and use the local cache only")]
    offline: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Recognize text and emit a Glypho annotation document.
    Recognize {
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(
            short = 'l',
            long = "language",
            value_delimiter = ',',
            help = "Common BCP-47 or exact Tesseract IDs, for example en,ru,ja"
        )]
        languages: Vec<String>,
        #[arg(long, value_enum, default_value = "sparse-text")]
        segmentation: SegmentationArg,
        #[arg(long, default_value_t = 0.8)]
        min_confidence: f32,
        #[arg(long, value_enum, default_value = "auto")]
        backend: BackendArg,
        #[arg(long, help = "Installed Glypho model directory")]
        models: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "balanced")]
        quality: QualityArg,
        #[arg(long, value_enum, default_value = "auto")]
        device: DeviceArg,
        #[arg(long, value_enum, default_value = "json")]
        format: OutputFormat,
        #[arg(long, help = "Disable model downloads and use the local cache only")]
        offline: bool,
        #[arg(long, help = "ONNX CPU worker threads (default: up to 8)")]
        threads: Option<usize>,
        #[arg(long, default_value = "tesseract")]
        tesseract: PathBuf,
        #[arg(long, default_value_t = 30)]
        timeout: u64,
        #[arg(long)]
        pretty: bool,
    },
    /// Show the runtime, resolved device, models and supported languages.
    #[command(visible_alias = "doctor")]
    Info {
        #[arg(long, value_enum, default_value = "auto")]
        backend: BackendArg,
        #[arg(long, help = "Installed Glypho model directory")]
        models: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "balanced")]
        quality: QualityArg,
        #[arg(long, value_enum, default_value = "auto")]
        device: DeviceArg,
        #[arg(long, help = "Disable model downloads and use the local cache only")]
        offline: bool,
        #[arg(long, help = "ONNX CPU worker threads (default: up to 8)")]
        threads: Option<usize>,
        #[arg(long, default_value = "tesseract")]
        tesseract: PathBuf,
        #[arg(long)]
        pretty: bool,
    },
    /// Download, verify and initialize the selected models.
    Warmup {
        #[arg(short = 'l', long = "language", value_delimiter = ',')]
        languages: Vec<String>,
        #[arg(long, help = "Glypho model cache directory")]
        models: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "balanced")]
        quality: QualityArg,
        #[arg(long, value_enum, default_value = "auto")]
        device: DeviceArg,
        #[arg(long, help = "ONNX worker threads (default: up to 8)")]
        threads: Option<usize>,
        #[arg(long, help = "Disable model downloads and use the local cache only")]
        offline: bool,
    },
    /// Run the local NDJSON worker used by language bindings.
    #[command(hide = true)]
    Serve {
        #[arg(long, help = "Glypho model cache directory")]
        models: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "balanced")]
        quality: QualityArg,
        #[arg(long, value_enum, default_value = "auto")]
        device: DeviceArg,
        #[arg(long, help = "ONNX worker threads (default: up to 8)")]
        threads: Option<usize>,
        #[arg(long, help = "Disable model downloads and use the local cache only")]
        offline: bool,
    },
    /// Validate a Glypho annotation document.
    Validate { document: PathBuf },
    /// Calculate CER and WER between two Glypho documents.
    Evaluate {
        reference: PathBuf,
        prediction: PathBuf,
        #[arg(long)]
        pretty: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendArg {
    Auto,
    Onnx,
    Tesseract,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum QualityArg {
    Fast,
    Balanced,
    Accurate,
    Maximum,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
    Coreml,
    Openvino,
}

impl From<DeviceArg> for Device {
    fn from(value: DeviceArg) -> Self {
        match value {
            DeviceArg::Auto => Self::Auto,
            DeviceArg::Cpu => Self::Cpu,
            DeviceArg::Cuda => Self::Cuda,
            DeviceArg::Coreml => Self::CoreMl,
            DeviceArg::Openvino => Self::OpenVino,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    JsonPretty,
}

impl From<QualityArg> for QualityMode {
    fn from(value: QualityArg) -> Self {
        match value {
            QualityArg::Fast => Self::Fast,
            QualityArg::Balanced => Self::Balanced,
            QualityArg::Accurate => Self::Accurate,
            QualityArg::Maximum => Self::Maximum,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SegmentationArg {
    Auto,
    SingleBlock,
    SingleLine,
    SparseText,
}

impl From<SegmentationArg> for PageSegmentation {
    fn from(value: SegmentationArg) -> Self {
        match value {
            SegmentationArg::Auto => Self::Auto,
            SegmentationArg::SingleBlock => Self::SingleBlock,
            SegmentationArg::SingleLine => Self::SingleLine,
            SegmentationArg::SparseText => Self::SparseText,
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("glypho: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let Cli {
        command,
        input,
        languages,
        device,
        quality,
        models,
        threads,
        min_confidence,
        segmentation,
        format,
        output,
        offline,
    } = cli;
    if let Some(input) = input {
        if command.is_some() {
            return Err("an image path cannot be combined with a subcommand".into());
        }
        let options = RecognitionOptions {
            languages,
            segmentation: segmentation.into(),
            min_confidence,
        };
        let document =
            match select_backend(BackendArg::Onnx, models, quality, device, threads, offline)? {
                SelectedBackend::Onnx(engine) => engine.recognize(&input, &options)?,
                SelectedBackend::Tesseract => unreachable!("short form always uses ONNX"),
            };
        write_document(&document, output.as_deref(), format)?;
        return Ok(());
    }

    let command = command.ok_or("pass an image path or a subcommand; try --help")?;
    match command {
        Command::Recognize {
            input,
            output,
            languages,
            segmentation,
            min_confidence,
            backend,
            models,
            quality,
            device,
            format,
            offline,
            threads,
            tesseract,
            timeout,
            pretty,
        } => {
            if let Some(output) = output.as_deref()
                && paths_refer_to_same_file(&input, output)?
            {
                return Err("output path must not overwrite the input image".into());
            }
            let options = RecognitionOptions {
                languages,
                segmentation: segmentation.into(),
                min_confidence,
            };
            let document = match select_backend(backend, models, quality, device, threads, offline)?
            {
                SelectedBackend::Onnx(engine) => engine.recognize(input, &options)?,
                SelectedBackend::Tesseract => {
                    let engine = Glypho::new(TesseractConfig {
                        binary: tesseract,
                        timeout: Duration::from_secs(timeout),
                    });
                    engine.recognize(input, &options)?
                }
            };
            let format = if pretty {
                OutputFormat::JsonPretty
            } else {
                format
            };
            write_document(&document, output.as_deref(), format)?;
        }
        Command::Info {
            backend,
            models,
            quality,
            device,
            offline,
            threads,
            tesseract,
            pretty,
        } => {
            let report = match select_backend(backend, models, quality, device, threads, offline)? {
                SelectedBackend::Onnx(engine) => {
                    let info = engine.info();
                    json!({
                        "glypho_version": env!("CARGO_PKG_VERSION"),
                        "backend": "onnxruntime",
                        "backend_version": info.runtime,
                        "model": info.model,
                        "quality": info.quality,
                        "languages": info.languages,
                        "models_dir": info.models_dir,
                        "requested_device": info.requested_device,
                        "device": info.device,
                        "fallback_reason": info.fallback_reason,
                    })
                }
                SelectedBackend::Tesseract => {
                    let engine = Glypho::new(TesseractConfig {
                        binary: tesseract,
                        ..TesseractConfig::default()
                    });
                    let info = engine.info()?;
                    json!({
                        "glypho_version": env!("CARGO_PKG_VERSION"),
                        "backend": "tesseract",
                        "backend_version": info.version,
                        "languages": info.languages,
                    })
                }
            };
            write_json(&report, None, pretty)?;
        }
        Command::Warmup {
            languages,
            models,
            quality,
            device,
            threads,
            offline,
        } => {
            let SelectedBackend::Onnx(engine) =
                select_backend(BackendArg::Onnx, models, quality, device, threads, offline)?
            else {
                unreachable!("warmup always uses ONNX")
            };
            engine.warmup(&languages)?;
            let info = engine.info();
            write_json(
                &json!({
                    "warmed": true,
                    "quality": info.quality,
                    "device": info.device,
                    "languages": if languages.is_empty() { vec!["auto".to_owned()] } else { languages },
                    "models_dir": info.models_dir,
                }),
                None,
                true,
            )?;
        }
        Command::Serve {
            models,
            quality,
            device,
            threads,
            offline,
        } => {
            let SelectedBackend::Onnx(engine) =
                select_backend(BackendArg::Onnx, models, quality, device, threads, offline)?
            else {
                unreachable!("worker always uses ONNX")
            };
            serve(&engine)?;
        }
        Command::Validate { document } => {
            let document = read_document(&document)?;
            document.validate()?;
            println!("valid: {}", document.image.file_name);
        }
        Command::Evaluate {
            reference,
            prediction,
            pretty,
        } => {
            let reference = read_document(&reference)?;
            let prediction = read_document(&prediction)?;
            let result = evaluate(&reference.text, &prediction.text);
            write_json(&result, None, pretty)?;
        }
    }

    Ok(())
}

enum SelectedBackend {
    Onnx(Box<OnnxEngine>),
    Tesseract,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerRequest {
    id: u64,
    method: WorkerMethod,
    #[serde(default)]
    image: Option<PathBuf>,
    #[serde(default)]
    languages: Vec<String>,
    #[serde(default)]
    segmentation: PageSegmentation,
    #[serde(default = "default_min_confidence")]
    min_confidence: f32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WorkerMethod {
    Recognize,
    Warmup,
    Info,
}

fn serve(engine: &OnnxEngine) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.len() > 1024 * 1024 {
            write_worker_error(&mut stdout, None, "request exceeds the 1 MiB limit")?;
            continue;
        }
        let request = match serde_json::from_str::<WorkerRequest>(&line) {
            Ok(request) => request,
            Err(error) => {
                write_worker_error(&mut stdout, None, &format!("invalid request: {error}"))?;
                continue;
            }
        };
        let id = request.id;
        let result = match request.method {
            WorkerMethod::Recognize => request
                .image
                .ok_or_else(|| "recognize requires image".to_owned())
                .and_then(|image| {
                    engine
                        .recognize(
                            image,
                            &RecognitionOptions {
                                languages: request.languages,
                                segmentation: request.segmentation,
                                min_confidence: request.min_confidence,
                            },
                        )
                        .map_err(|error| error.to_string())
                        .and_then(|document| {
                            serde_json::to_value(document).map_err(|error| error.to_string())
                        })
                }),
            WorkerMethod::Warmup => engine
                .warmup(&request.languages)
                .map(|()| json!({ "warmed": true }))
                .map_err(|error| error.to_string()),
            WorkerMethod::Info => {
                serde_json::to_value(engine.info()).map_err(|error| error.to_string())
            }
        };
        match result {
            Ok(value) => write_worker_value(
                &mut stdout,
                &json!({
                    "id": id,
                    "ok": true,
                    "result": value,
                }),
            )?,
            Err(error) => write_worker_error(&mut stdout, Some(id), &error)?,
        }
    }
    Ok(())
}

fn write_worker_error(
    output: &mut impl Write,
    id: Option<u64>,
    error: &str,
) -> Result<(), serde_json::Error> {
    write_worker_value(output, &json!({ "id": id, "ok": false, "error": error }))
}

fn write_worker_value(
    output: &mut impl Write,
    value: &serde_json::Value,
) -> Result<(), serde_json::Error> {
    serde_json::to_writer(&mut *output, value)?;
    output.write_all(b"\n").map_err(serde_json::Error::io)?;
    output.flush().map_err(serde_json::Error::io)
}

fn default_min_confidence() -> f32 {
    0.8
}

fn select_backend(
    backend: BackendArg,
    models: Option<PathBuf>,
    quality: QualityArg,
    device: DeviceArg,
    threads: Option<usize>,
    offline: bool,
) -> Result<SelectedBackend, Box<dyn std::error::Error>> {
    if matches!(backend, BackendArg::Tesseract) {
        return Ok(SelectedBackend::Tesseract);
    }

    let models_dir = models.unwrap_or_else(default_models_dir);

    let mut config = OnnxConfig::new(models_dir);
    config.quality = quality.into();
    config.device = device.into();
    config.auto_download = !offline;
    if let Some(threads) = threads {
        config.threads = threads;
    }
    Ok(SelectedBackend::Onnx(Box::new(OnnxEngine::new(config)?)))
}

fn paths_refer_to_same_file(left: &Path, right: &Path) -> io::Result<bool> {
    let left = fs::canonicalize(left)?;
    let right = match fs::canonicalize(right) {
        Ok(path) => path,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let parent = right.parent().unwrap_or_else(|| Path::new("."));
            let parent = fs::canonicalize(parent)?;
            match right.file_name() {
                Some(file_name) => parent.join(file_name),
                None => parent,
            }
        }
        Err(error) => return Err(error),
    };
    Ok(left == right)
}

fn read_document(path: &Path) -> Result<Document, Box<dyn std::error::Error>> {
    let input = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let document = serde_json::from_slice(&input)?;
    Ok(document)
}

fn write_document(
    document: &Document,
    output: Option<&Path>,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = match format {
        OutputFormat::Text => {
            let mut bytes = document.text.as_bytes().to_vec();
            bytes.push(b'\n');
            bytes
        }
        OutputFormat::Json => {
            let mut bytes = serde_json::to_vec(document)?;
            bytes.push(b'\n');
            bytes
        }
        OutputFormat::JsonPretty => {
            let mut bytes = serde_json::to_vec_pretty(document)?;
            bytes.push(b'\n');
            bytes
        }
    };
    write_bytes(&bytes, output)?;
    Ok(())
}

fn write_json<T: Serialize>(
    value: &T,
    output: Option<&Path>,
    pretty: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = if pretty {
        serde_json::to_vec_pretty(value)?
    } else {
        serde_json::to_vec(value)?
    };

    let mut bytes = bytes;
    bytes.push(b'\n');
    write_bytes(&bytes, output)?;
    Ok(())
}

fn write_bytes(bytes: &[u8], output: Option<&Path>) -> io::Result<()> {
    match output {
        Some(path) => atomic_write(path, bytes),
        None => io::stdout().lock().write_all(bytes),
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("glypho.json");
    let (temporary, mut file) = create_temporary(parent, file_name)?;

    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        replace_file(&temporary, path)
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn create_temporary(parent: &Path, file_name: &str) -> io::Result<(PathBuf, fs::File)> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    for _ in 0..100 {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".{file_name}.{}.{timestamp}.{counter}.tmp",
            std::process::id()
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }

    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not create a unique output file",
    ))
}

#[cfg(not(windows))]
fn replace_file(source: &Path, target: &Path) -> io::Result<()> {
    fs::rename(source, target)
}

#[cfg(windows)]
fn replace_file(source: &Path, target: &Path) -> io::Result<()> {
    match fs::rename(source, target) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            fs::remove_file(target)?;
            fs::rename(source, target)
        }
        Err(error) => Err(error),
    }
}
