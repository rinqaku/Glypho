use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Cursor, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use image::{DynamicImage, ImageDecoder, ImageReader, Limits, RgbImage};
#[cfg(feature = "coreml")]
use oar_ocr::core::config::{
    OrtCoreMLComputeUnits, OrtCoreMLConfig, OrtCoreMLModelFormat, OrtCoreMLSpecializationStrategy,
};
use oar_ocr::core::config::{OrtExecutionProvider, OrtGraphOptimizationLevel, OrtSessionConfig};
use oar_ocr::core::traits::OrtConfigurable;
use oar_ocr::core::traits::adapter::{AdapterBuilder, ModelAdapter};
use oar_ocr::core::traits::task::ImageTaskInput;
use oar_ocr::domain::adapters::{
    TextDetectionAdapter, TextDetectionAdapterBuilder, TextRecognitionAdapter,
    TextRecognitionAdapterBuilder,
};
use oar_ocr::domain::tasks::{TextDetectionConfig, TextRecognitionConfig};
use oar_ocr::oarocr::{EdgeProcessor, TextCroppingProcessor, TextRegion as OarTextRegion};
use oar_ocr::processors::{BoundingBox, LimitType, Point as OarPoint, sort_quad_boxes};
#[cfg(any(feature = "cuda", feature = "coreml", feature = "openvino"))]
use ort::ep::ExecutionProvider;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod catalog;
mod device;
mod geometry;
mod routing;

pub use catalog::default_models_dir;
use catalog::*;
use device::*;
use geometry::*;
use routing::*;

use crate::document::Metadata;
use crate::{
    Document, EngineInfo, Error, EvaluationPolicy, ImageInfo, Legibility, Point, Quad,
    RecognitionOptions, RegionSource, Result, TextAlternative, TextDirection, TextLine, TextWord,
};

const MAX_MODEL_BYTES: u64 = 128 * 1024 * 1024;
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(30 * 60);
static DOWNLOAD_COUNTER: AtomicU64 = AtomicU64::new(0);
const CYRILLIC_LANGUAGES: &[&str] = &["be", "ru", "uk"];
const CJK_LANGUAGES: &[&str] = &["ja", "zh"];
const KOREAN_LANGUAGES: &[&str] = &["ko"];
const LATIN_LANGUAGES: &[&str] = &[
    "af", "az", "bs", "ca", "cs", "cy", "da", "de", "en", "es", "et", "eu", "fi", "fr", "ga", "gl",
    "hr", "hu", "id", "is", "it", "jv", "ku", "la", "lb", "lt", "lv", "mi", "ms", "mt", "nl", "no",
    "oc", "pi", "pl", "pt", "qu", "rm", "ro", "sk", "sl", "sq", "sr-latn", "sv", "sw", "tl", "tr",
    "uz", "vi",
];
const SUPPORTED_LANGUAGES: &[&str] = &[
    "af", "az", "be", "bs", "ca", "cs", "cy", "da", "de", "en", "es", "et", "eu", "fi", "fr", "ga",
    "gl", "hr", "hu", "id", "is", "it", "ja", "jv", "ko", "ku", "la", "lb", "lt", "lv", "mi", "ms",
    "mt", "nl", "no", "oc", "pi", "pl", "pt", "qu", "rm", "ro", "ru", "sk", "sl", "sq", "sr-latn",
    "sv", "sw", "tl", "tr", "uk", "uz", "vi", "zh",
];

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityMode {
    Fast,
    #[default]
    Balanced,
    Accurate,
    Maximum,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    #[default]
    Auto,
    Cpu,
    Cuda,
    CoreMl,
    OpenVino,
}

impl Device {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::CoreMl => "coreml",
            Self::OpenVino => "openvino",
        }
    }
}

impl QualityMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Accurate => "accurate",
            Self::Maximum => "maximum",
        }
    }
}

#[derive(Clone, Debug)]
pub struct OnnxConfig {
    pub models_dir: PathBuf,
    pub quality: QualityMode,
    pub device: Device,
    pub auto_download: bool,
    pub threads: usize,
    pub max_file_bytes: u64,
    pub max_image_pixels: u64,
}

impl OnnxConfig {
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            models_dir: models_dir.into(),
            ..Self::default()
        }
    }
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            models_dir: default_models_dir(),
            quality: QualityMode::Balanced,
            device: Device::Auto,
            auto_download: true,
            threads: default_threads(),
            max_file_bytes: 256 * 1024 * 1024,
            max_image_pixels: 50_000_000,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OnnxInfo {
    pub runtime: String,
    pub quality: QualityMode,
    pub model: String,
    pub languages: Vec<String>,
    pub models_dir: PathBuf,
    pub requested_device: Device,
    pub device: Device,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

#[derive(Debug)]
pub struct OnnxEngine {
    config: OnnxConfig,
    profile: ModelProfile,
    detector: OnceLock<TextDetectionAdapter>,
    primary: OnceLock<TextRecognitionAdapter>,
    latin: OnceLock<TextRecognitionAdapter>,
    cyrillic: OnceLock<TextRecognitionAdapter>,
    korean: OnceLock<TextRecognitionAdapter>,
    initialization: InitializationLocks,
    model_name: String,
    device: Mutex<DeviceResolution>,
}

#[derive(Debug, Default)]
struct InitializationLocks {
    detector: Mutex<()>,
    primary: Mutex<()>,
    latin: Mutex<()>,
    cyrillic: Mutex<()>,
    korean: Mutex<()>,
}

impl OnnxEngine {
    pub fn new(config: OnnxConfig) -> Result<Self> {
        validate_config(&config)?;
        let profile = ModelProfile::for_quality(config.quality);
        let device = resolve_device(config.device);

        Ok(Self {
            config,
            profile,
            detector: OnceLock::new(),
            primary: OnceLock::new(),
            latin: OnceLock::new(),
            cyrillic: OnceLock::new(),
            korean: OnceLock::new(),
            initialization: InitializationLocks::default(),
            model_name: profile.name.to_owned(),
            device: Mutex::new(device),
        })
    }

    pub fn recognize(
        &self,
        image_path: impl AsRef<Path>,
        options: &RecognitionOptions,
    ) -> Result<Document> {
        self.recognize_inner(image_path.as_ref(), options)
    }

    pub fn recognize_bytes(
        &self,
        image_bytes: &[u8],
        file_name: impl Into<String>,
        options: &RecognitionOptions,
    ) -> Result<Document> {
        validate_min_confidence(options.min_confidence)?;
        let started = Instant::now();
        let file_name = file_name.into();
        if file_name.trim().is_empty() {
            return Err(Error::InvalidOption(
                "file_name must not be empty".to_owned(),
            ));
        }
        let image = self.load_image_bytes(image_bytes)?;
        let image_id = Path::new(&file_name)
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("image")
            .to_owned();
        self.recognize_image(image, file_name, image_id, options, started)
    }

    fn recognize_inner(&self, image_path: &Path, options: &RecognitionOptions) -> Result<Document> {
        validate_min_confidence(options.min_confidence)?;
        let started = Instant::now();
        let image = self.load_image(image_path)?;
        let file_name = image_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("image")
            .to_owned();
        let image_id = image_path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("image")
            .to_owned();
        self.recognize_image(image, file_name, image_id, options, started)
    }

    fn recognize_image(
        &self,
        image: RgbImage,
        file_name: String,
        image_id: String,
        options: &RecognitionOptions,
        started: Instant,
    ) -> Result<Document> {
        let width = image.width();
        let height = image.height();
        let languages = resolve_languages(&options.languages)?;
        validate_profile_languages(self.config.quality, &languages)?;
        let regions = self.recognize_regions(image, &languages, options.min_confidence)?;
        let elapsed_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;

        let mut document = Document::new(ImageInfo {
            id: image_id,
            file_name,
            width,
            height,
            sha256: None,
        });
        document.language_hints = languages.clone();

        for candidate in regions {
            let Some((text, confidence)) = candidate.region.text_with_confidence() else {
                continue;
            };
            let text = text.trim();
            if text.is_empty() || confidence < options.min_confidence {
                continue;
            }
            let Some(quad) = quad_from_box(&candidate.region.bounding_box, width, height) else {
                continue;
            };
            let line_index = document.lines.len() + 1;
            let line_id = format!("line-{line_index:04}");
            let words = build_words(&line_id, &candidate.region, text, confidence, width, height);
            let alternatives = candidate.alternatives;
            document.lines.push(TextLine {
                id: line_id,
                order: 0,
                quad,
                text: text.to_owned(),
                corrected_text: None,
                alternatives,
                confidence: Some(confidence.clamp(0.0, 1.0)),
                language: line_language(text, &languages),
                script: script_tag(text).map(str::to_owned),
                direction: TextDirection::Auto,
                legibility: Legibility::Clear,
                flags: Vec::new(),
                evaluation: EvaluationPolicy::default(),
                source: RegionSource::Model,
                words,
                ignored: false,
            });
        }

        document.sort_reading_order();
        document.metadata = Metadata {
            engine: Some(EngineInfo {
                name: "Glypho".to_owned(),
                version: env!("CARGO_PKG_VERSION").to_owned(),
                backend: "onnxruntime".to_owned(),
                model: Some(self.model_name.clone()),
                languages,
                elapsed_ms,
            }),
            ..Metadata::default()
        };
        document.validate()?;
        Ok(document)
    }

    pub fn info(&self) -> OnnxInfo {
        let device = self
            .device
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        OnnxInfo {
            runtime: "ONNX Runtime".to_owned(),
            quality: self.config.quality,
            model: self.model_name.clone(),
            languages: profile_languages(self.config.quality),
            models_dir: self.config.models_dir.clone(),
            requested_device: self.config.device,
            device: device.resolved,
            fallback_reason: device.fallback_reason,
        }
    }

    pub fn warmup(&self, requested_languages: &[String]) -> Result<()> {
        let languages = resolve_languages(requested_languages)?;
        validate_profile_languages(self.config.quality, &languages)?;
        let plan = recognizer_plan(&languages);
        if self.resolved_device() != Device::Cpu {
            return self.warmup_sequential(plan);
        }

        std::thread::scope(|scope| {
            let mut tasks = Vec::new();
            tasks.push(scope.spawn(|| self.detector().map(|_| ())));
            if plan.primary {
                tasks.push(scope.spawn(|| self.primary().map(|_| ())));
            }
            if plan.latin {
                tasks.push(scope.spawn(|| self.latin().map(|_| ())));
            }
            if plan.cyrillic {
                tasks.push(scope.spawn(|| self.cyrillic().map(|_| ())));
            }
            if plan.korean {
                tasks.push(scope.spawn(|| self.korean().map(|_| ())));
            }
            for task in tasks {
                task.join()
                    .map_err(|_| backend_error("model initialization thread panicked"))??;
            }
            Ok(())
        })
    }

    fn warmup_sequential(&self, plan: RecognizerPlan) -> Result<()> {
        self.detector()?;
        if plan.primary {
            self.primary()?;
        }
        if plan.latin {
            self.latin()?;
        }
        if plan.cyrillic {
            self.cyrillic()?;
        }
        if plan.korean {
            self.korean()?;
        }
        Ok(())
    }

    fn recognize_regions(
        &self,
        image: RgbImage,
        languages: &[String],
        min_confidence: f32,
    ) -> Result<Vec<MergedRegion>> {
        let auto_languages = languages.is_empty();
        let plan = recognizer_plan(languages);
        self.initialize_frontend(plan)?;
        let image = Arc::new(image);
        let boxes = self.detect_boxes(Arc::clone(&image))?;
        let cropped = TextCroppingProcessor::new(true)
            .process((image, boxes.clone()))
            .map_err(|error| backend_error(error.to_string()))?;
        let (boxes, crops): (Vec<_>, Vec<_>) = boxes
            .into_iter()
            .zip(cropped)
            .filter_map(|(box_, crop)| crop.map(|crop| (box_, crop)))
            .unzip();
        if crops.is_empty() {
            return Ok(Vec::new());
        }

        let primary = if plan.primary {
            Some(self.recognize_crops(self.primary()?, &crops)?)
        } else if plan.latin {
            Some(self.recognize_crops(self.latin()?, &crops)?)
        } else {
            None
        };
        let mut specialists = Vec::new();
        // Auto mode spends specialist passes only on uncertain or matching-script crops.
        if auto_languages {
            let latin = auto_specialist_crop_indices(
                primary.as_deref(),
                crops.len(),
                SpecialistScript::Latin,
            );
            if !latin.is_empty() {
                specialists.push((
                    SpecialistScript::Latin,
                    self.recognize_crop_indices(self.latin()?, &crops, &latin)?,
                ));
            }
            let cyrillic = auto_specialist_crop_indices(
                primary.as_deref(),
                crops.len(),
                SpecialistScript::Cyrillic,
            );
            if !cyrillic.is_empty() {
                specialists.push((
                    SpecialistScript::Cyrillic,
                    self.recognize_crop_indices(self.cyrillic()?, &crops, &cyrillic)?,
                ));
            }
            let korean = auto_specialist_crop_indices(
                primary.as_deref(),
                crops.len(),
                SpecialistScript::Korean,
            );
            if !korean.is_empty() {
                specialists.push((
                    SpecialistScript::Korean,
                    self.recognize_crop_indices(self.korean()?, &crops, &korean)?,
                ));
            }
        } else if plan.primary && plan.latin {
            specialists.push((
                SpecialistScript::Latin,
                self.recognize_crops(self.latin()?, &crops)?,
            ));
        }
        if !auto_languages && plan.cyrillic {
            let indices = specialist_crop_indices(primary.as_deref(), crops.len());
            if !indices.is_empty() {
                specialists.push((
                    SpecialistScript::Cyrillic,
                    self.recognize_crop_indices(self.cyrillic()?, &crops, &indices)?,
                ));
            }
        }
        if !auto_languages && plan.korean {
            specialists.push((
                SpecialistScript::Korean,
                self.recognize_crops(self.korean()?, &crops)?,
            ));
        }
        Ok(merge_recognized_regions(
            &boxes,
            primary,
            &specialists,
            auto_languages,
            min_confidence,
        ))
    }

    fn detect_boxes(&self, image: Arc<RgbImage>) -> Result<Vec<BoundingBox>> {
        if !should_tile_image(image.width(), image.height(), self.profile.max_side_len) {
            return self.detect_boxes_single(image);
        }
        let portrait = image.height() >= image.width();
        let long_side = if portrait {
            image.height()
        } else {
            image.width()
        };
        let mut boxes = Vec::new();
        // Overlap preserves text cut by a tile boundary; deduplication happens afterward.
        for offset in tile_offsets(long_side, self.profile.max_side_len) {
            let (x, y, width, height) = if portrait {
                (
                    0,
                    offset,
                    image.width(),
                    self.profile.max_side_len.min(image.height() - offset),
                )
            } else {
                (
                    offset,
                    0,
                    self.profile.max_side_len.min(image.width() - offset),
                    image.height(),
                )
            };
            let tile =
                Arc::new(image::imageops::crop_imm(image.as_ref(), x, y, width, height).to_image());
            boxes.extend(
                self.detect_boxes_single(tile)?
                    .into_iter()
                    .map(|box_| box_.translate(x as f32, y as f32)),
            );
        }
        Ok(deduplicate_boxes(boxes))
    }

    fn detect_boxes_single(&self, image: Arc<RgbImage>) -> Result<Vec<BoundingBox>> {
        let mut output = self
            .detector()?
            .execute(ImageTaskInput::from_arc_images(vec![image]), None)
            .map_err(|error| backend_error(error.to_string()))?;
        let detections = output.detections.pop().unwrap_or_default();
        let boxes = detections
            .into_iter()
            .map(|detection| detection.bbox)
            .collect::<Vec<_>>();
        Ok(sort_quad_boxes(&boxes))
    }

    fn recognize_crops(
        &self,
        recognizer: &TextRecognitionAdapter,
        crops: &[Arc<RgbImage>],
    ) -> Result<Vec<Option<RecognizedCandidate>>> {
        let indices = (0..crops.len()).collect::<Vec<_>>();
        self.recognize_crop_indices(recognizer, crops, &indices)
    }

    fn recognize_crop_indices(
        &self,
        recognizer: &TextRecognitionAdapter,
        crops: &[Arc<RgbImage>],
        indices: &[usize],
    ) -> Result<Vec<Option<RecognizedCandidate>>> {
        let mut order = indices.to_vec();
        // Similar widths share a batch to limit zero padding in recognizer tensors.
        order.sort_by(|left, right| {
            crop_ratio(&crops[*left]).total_cmp(&crop_ratio(&crops[*right]))
        });
        let mut candidates = vec![None; crops.len()];
        for chunk in order.chunks(self.profile.region_batch_size) {
            let batch_max_ratio = chunk
                .iter()
                .map(|index| crop_ratio(&crops[*index]))
                .fold(1.0_f32, f32::max);
            let input = ImageTaskInput::from_arc_images(
                chunk
                    .iter()
                    .map(|index| Arc::clone(&crops[*index]))
                    .collect(),
            );
            let output = recognizer
                .execute(input, None)
                .map_err(|error| backend_error(error.to_string()))?;
            for (offset, index) in chunk.iter().enumerate() {
                let text = output.texts.get(offset).map(String::as_str).unwrap_or("");
                let confidence = *output.scores.get(offset).unwrap_or(&0.0);
                if !text.trim().is_empty() {
                    candidates[*index] = Some(RecognizedCandidate {
                        text: text.to_owned(),
                        confidence,
                        char_positions: output
                            .char_positions
                            .get(offset)
                            .cloned()
                            .unwrap_or_default(),
                        char_columns: output
                            .char_col_indices
                            .get(offset)
                            .cloned()
                            .unwrap_or_default(),
                        sequence_length: *output.sequence_lengths.get(offset).unwrap_or(&0),
                        crop_ratio: crop_ratio(&crops[*index]),
                        batch_max_ratio,
                    });
                }
            }
        }
        Ok(candidates)
    }

    fn detector(&self) -> Result<&TextDetectionAdapter> {
        retryable_init(&self.detector, &self.initialization.detector, || {
            self.build_detector()
        })
    }

    fn initialize_frontend(&self, plan: RecognizerPlan) -> Result<()> {
        let recognizer_ready = if plan.primary {
            self.primary.get().is_some()
        } else if plan.latin {
            self.latin.get().is_some()
        } else {
            true
        };
        if self.detector.get().is_some() && recognizer_ready {
            return Ok(());
        }
        if self.resolved_device() != Device::Cpu {
            self.detector()?;
            if plan.primary {
                self.primary()?;
            } else if plan.latin {
                self.latin()?;
            }
            return Ok(());
        }
        std::thread::scope(|scope| {
            let detector = scope.spawn(|| self.detector().map(|_| ()));
            let recognizer = if plan.primary {
                Some(scope.spawn(|| self.primary().map(|_| ())))
            } else if plan.latin {
                Some(scope.spawn(|| self.latin().map(|_| ())))
            } else {
                None
            };
            detector
                .join()
                .map_err(|_| backend_error("detector initialization thread panicked"))??;
            if let Some(recognizer) = recognizer {
                recognizer
                    .join()
                    .map_err(|_| backend_error("recognizer initialization thread panicked"))??;
            }
            Ok(())
        })
    }

    fn primary(&self) -> Result<&TextRecognitionAdapter> {
        self.recognizer(
            &self.primary,
            &self.initialization.primary,
            self.profile.recognizer,
        )
    }

    fn latin(&self) -> Result<&TextRecognitionAdapter> {
        self.recognizer(&self.latin, &self.initialization.latin, LATIN_RECOGNIZER)
    }

    fn cyrillic(&self) -> Result<&TextRecognitionAdapter> {
        self.recognizer(
            &self.cyrillic,
            &self.initialization.cyrillic,
            ESLAV_RECOGNIZER,
        )
    }

    fn korean(&self) -> Result<&TextRecognitionAdapter> {
        self.recognizer(&self.korean, &self.initialization.korean, KOREAN_RECOGNIZER)
    }

    fn recognizer<'a>(
        &'a self,
        cache: &'a OnceLock<TextRecognitionAdapter>,
        initialization: &Mutex<()>,
        artifacts: RecognizerArtifacts,
    ) -> Result<&'a TextRecognitionAdapter> {
        retryable_init(cache, initialization, || self.build_recognizer(artifacts))
    }

    fn build_detector(&self) -> Result<TextDetectionAdapter> {
        self.profile
            .detector
            .ensure(&self.config.models_dir, self.config.auto_download)?;
        let device = self.resolved_device();
        match self.build_detector_session(device) {
            Ok(adapter) => Ok(adapter),
            Err(error) if device != Device::Cpu => {
                self.fallback_to_cpu(device, &error);
                self.build_detector_session(Device::Cpu)
            }
            Err(error) => Err(error),
        }
    }

    fn build_detector_session(&self, device: Device) -> Result<TextDetectionAdapter> {
        let config = TextDetectionConfig {
            score_threshold: self.profile.detector_threshold,
            box_threshold: self.profile.box_threshold,
            unclip_ratio: self.profile.unclip_ratio,
            max_candidates: 1_000,
            limit_side_len: Some(64),
            limit_type: Some(LimitType::Min),
            max_side_len: Some(self.profile.max_side_len),
        };
        TextDetectionAdapterBuilder::new()
            .with_ort_config(self.session_config(device))
            .with_config(config)
            .build(self.profile.detector.path(&self.config.models_dir))
            .map_err(|error| backend_error(error.to_string()))
    }

    fn build_recognizer(&self, artifacts: RecognizerArtifacts) -> Result<TextRecognitionAdapter> {
        artifacts.ensure(&self.config.models_dir, self.config.auto_download)?;
        let dictionary_path = artifacts.dictionary.path(&self.config.models_dir);
        let dictionary = fs::read_to_string(&dictionary_path)
            .map_err(|error| Error::io(&dictionary_path, error))?
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let device = self.resolved_device();
        match self.build_recognizer_session(artifacts, &dictionary, device) {
            Ok(adapter) => Ok(adapter),
            Err(error) if device != Device::Cpu => {
                self.fallback_to_cpu(device, &error);
                self.build_recognizer_session(artifacts, &dictionary, Device::Cpu)
            }
            Err(error) => Err(error),
        }
    }

    fn build_recognizer_session(
        &self,
        artifacts: RecognizerArtifacts,
        dictionary: &[String],
        device: Device,
    ) -> Result<TextRecognitionAdapter> {
        TextRecognitionAdapterBuilder::new()
            .with_ort_config(self.session_config(device))
            .with_config(TextRecognitionConfig {
                score_threshold: 0.0,
            })
            .character_dict(dictionary.to_vec())
            .return_word_box(true)
            .build(artifacts.model.path(&self.config.models_dir))
            .map_err(|error| backend_error(error.to_string()))
    }

    fn resolved_device(&self) -> Device {
        self.device
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .resolved
    }

    fn fallback_to_cpu(&self, device: Device, error: &Error) {
        let mut resolution = self
            .device
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if resolution.resolved == Device::Cpu {
            return;
        }
        resolution.resolved = Device::Cpu;
        let reason = format!("{} session initialization failed: {error}", device.as_str());
        resolution.fallback_reason = Some(match resolution.fallback_reason.take() {
            Some(previous) => format!("{previous}; {reason}"),
            None => reason,
        });
    }

    fn session_config(&self, device: Device) -> OrtSessionConfig {
        let mut config = OrtSessionConfig::new()
            .with_intra_threads(self.config.threads)
            .with_inter_threads(1)
            .with_parallel_execution(false)
            .with_optimization_level(OrtGraphOptimizationLevel::Level3)
            .with_memory_pattern(true)
            .with_log_severity_level(3);
        config = match device {
            Device::Cuda => config.with_execution_providers(vec![
                OrtExecutionProvider::CUDA {
                    device_id: Some(0),
                    gpu_mem_limit: None,
                    arena_extend_strategy: None,
                    cudnn_conv_algo_search: Some("default".to_owned()),
                    cudnn_conv_use_max_workspace: Some(true),
                },
                OrtExecutionProvider::CPU,
            ]),
            Device::OpenVino => config.with_execution_providers(vec![
                OrtExecutionProvider::OpenVINO {
                    device_type: Some("AUTO".to_owned()),
                    num_threads: Some(self.config.threads),
                },
                OrtExecutionProvider::CPU,
            ]),
            Device::CoreMl => {
                config = config.with_execution_providers(vec![
                    OrtExecutionProvider::CoreML {
                        ane_only: None,
                        subgraphs: Some(false),
                    },
                    OrtExecutionProvider::CPU,
                ]);
                #[cfg(feature = "coreml")]
                {
                    config = config.with_coreml_config(OrtCoreMLConfig {
                        compute_units: Some(OrtCoreMLComputeUnits::All),
                        model_format: Some(OrtCoreMLModelFormat::MLProgram),
                        static_input_shapes: Some(false),
                        specialization_strategy: Some(
                            OrtCoreMLSpecializationStrategy::FastPrediction,
                        ),
                        allow_low_precision_accumulation_on_gpu: Some(true),
                        profile_compute_plan: None,
                        model_cache_dir: Some(
                            self.config
                                .models_dir
                                .join("compiled/coreml")
                                .to_string_lossy()
                                .into_owned(),
                        ),
                    });
                }
                config
            }
            Device::Auto | Device::Cpu => {
                config.with_execution_providers(vec![OrtExecutionProvider::CPU])
            }
        };
        config
    }

    fn load_image(&self, path: &Path) -> Result<RgbImage> {
        let metadata = fs::metadata(path).map_err(|error| Error::io(path, error))?;
        if !metadata.is_file() {
            return Err(Error::InvalidOption(format!(
                "input is not a file: {}",
                path.display()
            )));
        }
        if metadata.len() > self.config.max_file_bytes {
            return Err(Error::InvalidOption(format!(
                "input exceeds the {} byte limit",
                self.config.max_file_bytes
            )));
        }

        let file = File::open(path).map_err(|error| Error::io(path, error))?;
        let reader = ImageReader::new(BufReader::new(file))
            .with_guessed_format()
            .map_err(|error| Error::io(path, error))?;
        self.decode_image(reader)
    }

    fn load_image_bytes(&self, bytes: &[u8]) -> Result<RgbImage> {
        if bytes.is_empty() || bytes.len() as u64 > self.config.max_file_bytes {
            return Err(Error::InvalidOption(format!(
                "encoded image must contain between 1 and {} bytes",
                self.config.max_file_bytes
            )));
        }
        let reader = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|error| backend_error(format!("could not inspect image: {error}")))?;
        self.decode_image(reader)
    }

    fn decode_image<R: std::io::BufRead + Seek>(
        &self,
        mut reader: ImageReader<R>,
    ) -> Result<RgbImage> {
        let mut limits = Limits::default();
        limits.max_alloc = Some(self.config.max_image_pixels.saturating_mul(4));
        reader.limits(limits);
        let mut decoder = reader
            .into_decoder()
            .map_err(|error| backend_error(format!("could not inspect image: {error}")))?;
        let (width, height) = decoder.dimensions();
        let pixels = u64::from(width) * u64::from(height);
        if pixels == 0 || pixels > self.config.max_image_pixels {
            return Err(Error::InvalidOption(format!(
                "decoded image has {pixels} pixels; limit is {}",
                self.config.max_image_pixels
            )));
        }
        let orientation = decoder
            .orientation()
            .unwrap_or(image::metadata::Orientation::NoTransforms);
        let mut image = DynamicImage::from_decoder(decoder)
            .map_err(|error| backend_error(format!("could not decode image: {error}")))?;
        image.apply_orientation(orientation);
        Ok(image.into_rgb8())
    }
}

fn retryable_init<'a, T>(
    cache: &'a OnceLock<T>,
    initialization: &Mutex<()>,
    build: impl FnOnce() -> Result<T>,
) -> Result<&'a T> {
    // Cache successful sessions only: a transient provider or download failure stays retryable.
    if let Some(value) = cache.get() {
        return Ok(value);
    }
    let _guard = initialization
        .lock()
        .map_err(|_| backend_error("model initialization lock was poisoned"))?;
    if let Some(value) = cache.get() {
        return Ok(value);
    }
    let value = build()?;
    let _ = cache.set(value);
    cache
        .get()
        .ok_or_else(|| backend_error("model initialization failed"))
}

fn backend_error(message: impl Into<String>) -> Error {
    Error::Backend {
        backend: "onnxruntime",
        message: message.into(),
    }
}

#[cfg(test)]
mod tests;
