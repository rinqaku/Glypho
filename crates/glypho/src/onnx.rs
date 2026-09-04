use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
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
use oar_ocr::processors::{BoundingBox, LimitType, sort_quad_boxes};
#[cfg(any(feature = "cuda", feature = "coreml", feature = "openvino"))]
use ort::ep::ExecutionProvider;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
    detector: OnceLock<std::result::Result<TextDetectionAdapter, String>>,
    primary: OnceLock<std::result::Result<TextRecognitionAdapter, String>>,
    latin: OnceLock<std::result::Result<TextRecognitionAdapter, String>>,
    cyrillic: OnceLock<std::result::Result<TextRecognitionAdapter, String>>,
    korean: OnceLock<std::result::Result<TextRecognitionAdapter, String>>,
    model_name: String,
    device: Mutex<DeviceResolution>,
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
            model_name: profile.name.to_owned(),
            device: Mutex::new(device),
        })
    }

    pub fn recognize(
        &self,
        image_path: impl AsRef<Path>,
        options: &RecognitionOptions,
    ) -> Result<Document> {
        validate_min_confidence(options.min_confidence)?;
        let image_path = image_path.as_ref();
        let image = self.load_image(image_path)?;
        let width = image.width();
        let height = image.height();
        let languages = resolve_languages(&options.languages)?;
        validate_profile_languages(self.config.quality, &languages)?;
        let started = Instant::now();
        let regions = self.recognize_regions(image, &languages)?;
        let elapsed_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;

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
            let alternatives = candidate
                .alternative
                .filter(|alternative| alternative.text != text)
                .into_iter()
                .collect();
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
    ) -> Result<Vec<MergedRegion>> {
        let auto_languages = languages.is_empty();
        let plan = recognizer_plan(languages);
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
        if plan.cyrillic {
            let indices = specialist_crop_indices(primary.as_deref(), crops.len());
            if !indices.is_empty() {
                specialists.push((
                    SpecialistScript::Cyrillic,
                    self.recognize_crop_indices(self.cyrillic()?, &crops, &indices)?,
                ));
            }
        }
        if plan.korean {
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
        ))
    }

    fn detect_boxes(&self, image: Arc<RgbImage>) -> Result<Vec<BoundingBox>> {
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
        order.sort_by(|left, right| {
            crop_ratio(&crops[*left]).total_cmp(&crop_ratio(&crops[*right]))
        });
        let mut candidates = vec![None; crops.len()];
        for chunk in order.chunks(self.profile.region_batch_size) {
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
                    });
                }
            }
        }
        Ok(candidates)
    }

    fn detector(&self) -> Result<&TextDetectionAdapter> {
        let result = self
            .detector
            .get_or_init(|| self.build_detector().map_err(|error| error.to_string()));
        result
            .as_ref()
            .map_err(|message| backend_error(message.clone()))
    }

    fn primary(&self) -> Result<&TextRecognitionAdapter> {
        self.recognizer(&self.primary, self.profile.recognizer)
    }

    fn latin(&self) -> Result<&TextRecognitionAdapter> {
        self.recognizer(&self.latin, LATIN_RECOGNIZER)
    }

    fn cyrillic(&self) -> Result<&TextRecognitionAdapter> {
        self.recognizer(&self.cyrillic, ESLAV_RECOGNIZER)
    }

    fn korean(&self) -> Result<&TextRecognitionAdapter> {
        self.recognizer(&self.korean, KOREAN_RECOGNIZER)
    }

    fn recognizer<'a>(
        &'a self,
        cache: &'a OnceLock<std::result::Result<TextRecognitionAdapter, String>>,
        artifacts: RecognizerArtifacts,
    ) -> Result<&'a TextRecognitionAdapter> {
        let result = cache.get_or_init(|| {
            self.build_recognizer(artifacts)
                .map_err(|error| error.to_string())
        });
        result
            .as_ref()
            .map_err(|message| backend_error(message.clone()))
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
        let mut reader = ImageReader::new(BufReader::new(file))
            .with_guessed_format()
            .map_err(|error| Error::io(path, error))?;
        let mut limits = Limits::default();
        limits.max_alloc = Some(self.config.max_image_pixels.saturating_mul(4));
        reader.limits(limits);
        let decoder = reader
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
        DynamicImage::from_decoder(decoder)
            .map(DynamicImage::into_rgb8)
            .map_err(|error| backend_error(format!("could not decode image: {error}")))
    }
}

#[derive(Clone, Debug)]
struct MergedRegion {
    region: OarTextRegion,
    alternative: Option<TextAlternative>,
}

#[derive(Clone, Debug)]
struct RecognizedCandidate {
    text: String,
    confidence: f32,
}

#[derive(Clone, Debug)]
struct DeviceResolution {
    resolved: Device,
    fallback_reason: Option<String>,
}

fn resolve_device(requested: Device) -> DeviceResolution {
    if requested == Device::Cpu {
        return DeviceResolution {
            resolved: Device::Cpu,
            fallback_reason: None,
        };
    }

    let candidates = match requested {
        Device::Auto => {
            #[cfg(target_os = "macos")]
            {
                vec![Device::Cuda, Device::CoreMl, Device::OpenVino]
            }
            #[cfg(not(target_os = "macos"))]
            {
                vec![Device::Cuda, Device::OpenVino]
            }
        }
        value => vec![value],
    };
    let mut failures = Vec::new();
    for candidate in candidates {
        match probe_device(candidate) {
            Ok(()) => {
                return DeviceResolution {
                    resolved: candidate,
                    fallback_reason: None,
                };
            }
            Err(reason) => failures.push(reason),
        }
    }

    DeviceResolution {
        resolved: Device::Cpu,
        fallback_reason: (!failures.is_empty()).then(|| failures.join("; ")),
    }
}

fn probe_device(device: Device) -> std::result::Result<(), String> {
    match device {
        Device::Auto | Device::Cpu => Ok(()),
        Device::Cuda => {
            #[cfg(feature = "cuda")]
            {
                probe_provider(ort::ep::CUDA::default(), "CUDA")
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("CUDA support is not included in this build".to_owned())
            }
        }
        Device::CoreMl => {
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            {
                probe_provider(ort::ep::CoreML::default(), "CoreML")
            }
            #[cfg(not(all(feature = "coreml", target_os = "macos")))]
            {
                Err("CoreML support is not included for this platform".to_owned())
            }
        }
        Device::OpenVino => {
            #[cfg(feature = "openvino")]
            {
                probe_provider(ort::ep::OpenVINO::default(), "OpenVINO")
            }
            #[cfg(not(feature = "openvino"))]
            {
                Err("OpenVINO support is not included in this build".to_owned())
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "coreml", feature = "openvino"))]
fn probe_provider(provider: impl ExecutionProvider, name: &str) -> std::result::Result<(), String> {
    if !provider
        .is_available()
        .map_err(|error| format!("could not inspect {name}: {error}"))?
    {
        return Err(format!("{name} is unavailable in ONNX Runtime"));
    }
    let mut builder = ort::session::Session::builder()
        .map_err(|error| format!("could not initialize ONNX Runtime for {name}: {error}"))?;
    provider
        .register(&mut builder)
        .map_err(|error| format!("{name} initialization failed: {error}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SpecialistScript {
    Cyrillic,
    Latin,
    Korean,
}

#[derive(Clone, Copy, Debug)]
struct Artifact {
    directory: &'static str,
    file: &'static str,
    sha256: &'static str,
}

impl Artifact {
    fn path(self, root: &Path) -> PathBuf {
        root.join(self.directory).join(self.file)
    }

    fn verify(self, root: &Path) -> Result<()> {
        let path = self.path(root);
        let metadata = fs::symlink_metadata(&path).map_err(|error| Error::io(&path, error))?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(backend_error(format!(
                "model artifact is not a regular file: {}",
                path.display()
            )));
        }
        if metadata.len() > MAX_MODEL_BYTES {
            return Err(backend_error(format!(
                "model artifact is unexpectedly large: {}",
                path.display()
            )));
        }
        let actual = file_sha256(&path)?;
        if actual != self.sha256 {
            return Err(backend_error(format!(
                "model artifact checksum mismatch: {}",
                path.display()
            )));
        }
        Ok(())
    }

    fn ensure(self, root: &Path, auto_download: bool) -> Result<()> {
        if self.verify(root).is_ok() {
            return Ok(());
        }
        if !auto_download {
            return self.verify(root);
        }
        install_model(self.directory, root)?;
        self.verify(root)
    }
}

#[derive(Clone, Copy, Debug)]
struct RecognizerArtifacts {
    model: Artifact,
    dictionary: Artifact,
}

impl RecognizerArtifacts {
    fn ensure(self, root: &Path, auto_download: bool) -> Result<()> {
        self.model.ensure(root, auto_download)?;
        self.dictionary.ensure(root, auto_download)
    }
}

#[derive(Clone, Copy, Debug)]
struct ModelProfile {
    name: &'static str,
    detector: Artifact,
    recognizer: RecognizerArtifacts,
    detector_threshold: f32,
    box_threshold: f32,
    unclip_ratio: f32,
    region_batch_size: usize,
    max_side_len: u32,
}

impl ModelProfile {
    fn for_quality(quality: QualityMode) -> Self {
        match quality {
            QualityMode::Fast => Self {
                name: "PP-OCRv6 Tiny + routed PP-OCRv5 language packs",
                detector: V6_TINY_DETECTOR,
                recognizer: V6_TINY_RECOGNIZER,
                detector_threshold: 0.2,
                box_threshold: 0.4,
                unclip_ratio: 1.4,
                region_batch_size: 16,
                max_side_len: 960,
            },
            QualityMode::Balanced => Self {
                name: "PP-OCRv5 Mobile detector + PP-OCRv6 Small recognizer + routed language packs",
                detector: V5_MOBILE_DETECTOR,
                recognizer: V6_SMALL_RECOGNIZER,
                detector_threshold: 0.3,
                box_threshold: 0.6,
                unclip_ratio: 1.5,
                region_batch_size: 8,
                max_side_len: 1_280,
            },
            QualityMode::Accurate => Self {
                name: "PP-OCRv6 Small + routed PP-OCRv5 language packs",
                detector: V6_SMALL_DETECTOR,
                recognizer: V6_SMALL_RECOGNIZER,
                detector_threshold: 0.2,
                box_threshold: 0.45,
                unclip_ratio: 1.4,
                region_batch_size: 8,
                max_side_len: 1_600,
            },
            QualityMode::Maximum => Self {
                name: "PP-OCRv6 Medium + routed PP-OCRv5 language packs",
                detector: V6_MEDIUM_DETECTOR,
                recognizer: V6_MEDIUM_RECOGNIZER,
                detector_threshold: 0.2,
                box_threshold: 0.45,
                unclip_ratio: 1.4,
                region_batch_size: 8,
                max_side_len: 2_048,
            },
        }
    }
}

const V6_TINY_DETECTOR: Artifact = Artifact {
    directory: "v6-tiny-det",
    file: "inference.onnx",
    sha256: "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8",
};
const V6_TINY_RECOGNIZER: RecognizerArtifacts = RecognizerArtifacts {
    model: Artifact {
        directory: "v6-tiny-rec",
        file: "inference.onnx",
        sha256: "9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6",
    },
    dictionary: Artifact {
        directory: "v6-tiny-rec",
        file: "dict.txt",
        sha256: "c5cbe34ef40c29c4df07ed012bf96569cb69a2d2a01a07027e9f13cb832bd9cd",
    },
};
const V5_MOBILE_DETECTOR: Artifact = Artifact {
    directory: "v5-mobile-det",
    file: "inference.onnx",
    sha256: "a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d",
};
const V6_SMALL_DETECTOR: Artifact = Artifact {
    directory: "v6-small-det",
    file: "inference.onnx",
    sha256: "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e",
};
const V6_SMALL_RECOGNIZER: RecognizerArtifacts = RecognizerArtifacts {
    model: Artifact {
        directory: "v6-small-rec",
        file: "inference.onnx",
        sha256: "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
    },
    dictionary: Artifact {
        directory: "v6-small-rec",
        file: "dict.txt",
        sha256: "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
    },
};
const V6_MEDIUM_DETECTOR: Artifact = Artifact {
    directory: "v6-medium-det",
    file: "inference.onnx",
    sha256: "eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1",
};
const V6_MEDIUM_RECOGNIZER: RecognizerArtifacts = RecognizerArtifacts {
    model: Artifact {
        directory: "v6-medium-rec",
        file: "inference.onnx",
        sha256: "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba",
    },
    dictionary: Artifact {
        directory: "v6-medium-rec",
        file: "dict.txt",
        sha256: "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
    },
};
const ESLAV_RECOGNIZER: RecognizerArtifacts = RecognizerArtifacts {
    model: Artifact {
        directory: "v5-eslav-rec",
        file: "inference.onnx",
        sha256: "b3018ef2b09a0250b6e0c8e871c927098363e5fd4df890cc68e8358eb0aaf1bd",
    },
    dictionary: Artifact {
        directory: "v5-eslav-rec",
        file: "dict.txt",
        sha256: "3e95f1581557162870cacdba5af91a4c6be2890710d395b0c3c7578e7ee5e6eb",
    },
};
const LATIN_RECOGNIZER: RecognizerArtifacts = RecognizerArtifacts {
    model: Artifact {
        directory: "v5-latin-rec",
        file: "inference.onnx",
        sha256: "7888113072263cb471b93f66dd5e2ad70548dc526fa1ace760d0d973dd121498",
    },
    dictionary: Artifact {
        directory: "v5-latin-rec",
        file: "dict.txt",
        sha256: "ccbcc45730b3fbbd9050c5bc74db6a99067141ef1035e3d14889a84a6b9b1aff",
    },
};
const KOREAN_RECOGNIZER: RecognizerArtifacts = RecognizerArtifacts {
    model: Artifact {
        directory: "v5-korean-rec",
        file: "inference.onnx",
        sha256: "92f0b7785e64fc9090106a241cf4c1eb97472824558272751b88a2a4476d3a08",
    },
    dictionary: Artifact {
        directory: "v5-korean-rec",
        file: "dict.txt",
        sha256: "a88071c68c01707489baa79ebe0405b7beb5cca229f4fc94cc3ef992328802d7",
    },
};

#[derive(Clone, Copy)]
struct DownloadSpec {
    repository: &'static str,
    revision: &'static str,
    model_bytes: u64,
    model_sha256: &'static str,
    config_bytes: Option<u64>,
    config_sha256: Option<&'static str>,
    dictionary_entries: Option<usize>,
    dictionary_bytes: Option<u64>,
    dictionary_sha256: Option<&'static str>,
}

pub fn default_models_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("GLYPHO_MODELS").filter(|path| !path.is_empty()) {
        return PathBuf::from(path);
    }
    if let Some(home) = std::env::var_os("GLYPHO_HOME").filter(|path| !path.is_empty()) {
        return PathBuf::from(home).join("models");
    }
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".glypho-ocr/models")
}

fn install_model(directory: &str, root: &Path) -> Result<()> {
    let spec = download_spec(directory).ok_or_else(|| {
        backend_error(format!(
            "model {directory} is not registered for automatic download"
        ))
    })?;
    fs::create_dir_all(root).map_err(|error| Error::io(root, error))?;
    reject_symlink(root)?;
    let destination = root.join(directory);
    fs::create_dir_all(&destination).map_err(|error| Error::io(&destination, error))?;
    reject_symlink(&destination)?;

    download_if_needed(
        spec,
        "inference.onnx",
        &destination.join("inference.onnx"),
        spec.model_bytes,
        spec.model_sha256,
    )?;
    if let (Some(config_bytes), Some(config_sha256)) = (spec.config_bytes, spec.config_sha256) {
        let config_path = destination.join("inference.yml");
        download_if_needed(
            spec,
            "inference.yml",
            &config_path,
            config_bytes,
            config_sha256,
        )?;
        if let (Some(entries), Some(bytes), Some(sha256)) = (
            spec.dictionary_entries,
            spec.dictionary_bytes,
            spec.dictionary_sha256,
        ) {
            let dictionary_path = destination.join("dict.txt");
            if !file_matches(&dictionary_path, bytes, sha256) {
                let dictionary = extract_dictionary(&config_path, entries)?;
                if dictionary.len() as u64 != bytes
                    || format!("{:x}", Sha256::digest(&dictionary)) != sha256
                {
                    return Err(backend_error(format!(
                        "generated dictionary failed verification: {}",
                        dictionary_path.display()
                    )));
                }
                atomic_model_write(&dictionary_path, &dictionary)?;
            }
        }
    }
    Ok(())
}

fn download_if_needed(
    spec: DownloadSpec,
    source: &str,
    target: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
) -> Result<()> {
    if file_matches(target, expected_bytes, expected_sha256) {
        return Ok(());
    }
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        spec.repository, spec.revision, source
    );
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(DOWNLOAD_TIMEOUT))
        .timeout_connect(Some(Duration::from_secs(30)))
        .build()
        .new_agent();
    let response = agent
        .get(&url)
        .header("User-Agent", concat!("Glypho/", env!("CARGO_PKG_VERSION")))
        .call()
        .map_err(|error| backend_error(format!("could not download {url}: {error}")))?;
    let mut input = response.into_body().into_reader();
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    let counter = DOWNLOAD_COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = target
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    let temporary = parent.join(format!(
        ".{name}.{}.{}.download",
        std::process::id(),
        counter
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut output = options
        .open(&temporary)
        .map_err(|error| Error::io(&temporary, error))?;
    let result = (|| {
        let mut digest = Sha256::new();
        let mut written = 0_u64;
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let read = input
                .read(&mut buffer)
                .map_err(|error| backend_error(format!("download failed: {error}")))?;
            if read == 0 {
                break;
            }
            written = written.saturating_add(read as u64);
            if written > expected_bytes {
                return Err(backend_error("download exceeded its registered size"));
            }
            output
                .write_all(&buffer[..read])
                .map_err(|error| Error::io(&temporary, error))?;
            digest.update(&buffer[..read]);
        }
        output
            .sync_all()
            .map_err(|error| Error::io(&temporary, error))?;
        let actual_sha256 = format!("{:x}", digest.finalize());
        if written != expected_bytes || actual_sha256 != expected_sha256 {
            return Err(backend_error(format!(
                "downloaded model failed verification: {}",
                target.display()
            )));
        }
        drop(output);
        if target.exists() {
            if file_matches(target, expected_bytes, expected_sha256) {
                return Ok(());
            }
            fs::remove_file(target).map_err(|error| Error::io(target, error))?;
        }
        fs::rename(&temporary, target).map_err(|error| Error::io(target, error))
    })();
    if result.is_err() || temporary.exists() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn atomic_model_write(target: &Path, bytes: &[u8]) -> Result<()> {
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    let counter = DOWNLOAD_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(".dict.{}.{}.tmp", std::process::id(), counter));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    let mut file = options
        .open(&temporary)
        .map_err(|error| Error::io(&temporary, error))?;
    let result = (|| {
        file.write_all(bytes)
            .map_err(|error| Error::io(&temporary, error))?;
        file.sync_all()
            .map_err(|error| Error::io(&temporary, error))?;
        drop(file);
        if target.exists() {
            fs::remove_file(target).map_err(|error| Error::io(target, error))?;
        }
        fs::rename(&temporary, target).map_err(|error| Error::io(target, error))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn extract_dictionary(config_path: &Path, expected_entries: usize) -> Result<Vec<u8>> {
    let config = fs::read_to_string(config_path).map_err(|error| Error::io(config_path, error))?;
    let mut lines = config.lines();
    for line in lines.by_ref() {
        if line == "  character_dict:" {
            break;
        }
    }
    let mut characters = Vec::new();
    for line in lines {
        let Some(value) = line.strip_prefix("  - ") else {
            break;
        };
        let value = if value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2 {
            value[1..value.len() - 1].replace("''", "'")
        } else if value.starts_with('"') && value.ends_with('"') {
            serde_json::from_str::<String>(value)
                .map_err(|error| backend_error(format!("invalid dictionary entry: {error}")))?
        } else if value.is_empty() {
            return Err(backend_error("dictionary contains an empty entry"));
        } else {
            value.to_owned()
        };
        if value.contains('\n') || value.contains('\r') {
            return Err(backend_error("dictionary entry contains a line break"));
        }
        characters.push(value);
    }
    if characters.len() != expected_entries {
        return Err(backend_error(format!(
            "dictionary contains {} entries; expected {expected_entries}",
            characters.len()
        )));
    }
    Ok(format!("{}\n", characters.join("\n")).into_bytes())
}

fn reject_symlink(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path).map_err(|error| Error::io(path, error))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(backend_error(format!(
            "model cache path is not a real directory: {}",
            path.display()
        )));
    }
    Ok(())
}

fn file_matches(path: &Path, expected_bytes: u64, expected_sha256: &str) -> bool {
    fs::symlink_metadata(path).is_ok_and(|metadata| {
        metadata.is_file()
            && !metadata.file_type().is_symlink()
            && metadata.len() == expected_bytes
            && file_sha256(path).is_ok_and(|sha256| sha256 == expected_sha256)
    })
}

fn download_spec(directory: &str) -> Option<DownloadSpec> {
    let spec = match directory {
        "v6-tiny-det" => DownloadSpec {
            repository: "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
            revision: "2ba1506c0380b8f0b03dd142459aac66d4421f6c",
            model_bytes: 1_780_590,
            model_sha256: "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8",
            config_bytes: None,
            config_sha256: None,
            dictionary_entries: None,
            dictionary_bytes: None,
            dictionary_sha256: None,
        },
        "v5-mobile-det" => DownloadSpec {
            repository: "PaddlePaddle/PP-OCRv5_mobile_det_onnx",
            revision: "e6f4fa85f00e168c862bc462aebca69eef9b3d3d",
            model_bytes: 4_826_518,
            model_sha256: "a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d",
            config_bytes: None,
            config_sha256: None,
            dictionary_entries: None,
            dictionary_bytes: None,
            dictionary_sha256: None,
        },
        "v6-small-det" => DownloadSpec {
            repository: "PaddlePaddle/PP-OCRv6_small_det_onnx",
            revision: "28fe5895c24fd108c19eb3e8479f4ab385fbfc62",
            model_bytes: 9_880_512,
            model_sha256: "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e",
            config_bytes: None,
            config_sha256: None,
            dictionary_entries: None,
            dictionary_bytes: None,
            dictionary_sha256: None,
        },
        "v6-medium-det" => DownloadSpec {
            repository: "PaddlePaddle/PP-OCRv6_medium_det_onnx",
            revision: "61323801669c338b7891481ec7bac61ce31b576a",
            model_bytes: 62_032_837,
            model_sha256: "eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1",
            config_bytes: None,
            config_sha256: None,
            dictionary_entries: None,
            dictionary_bytes: None,
            dictionary_sha256: None,
        },
        "v6-tiny-rec" => DownloadSpec {
            repository: "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
            revision: "2612ab37152ae0a677521bae4e1e3d4fb4cf7c30",
            model_bytes: 4_462_639,
            model_sha256: "9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6",
            config_bytes: Some(55_571),
            config_sha256: Some("66170210bad538e83fff3c4a3867e547d6bf20b50d64b20347c4b913f3034ea1"),
            dictionary_entries: Some(6_904),
            dictionary_bytes: Some(27_156),
            dictionary_sha256: Some(
                "c5cbe34ef40c29c4df07ed012bf96569cb69a2d2a01a07027e9f13cb832bd9cd",
            ),
        },
        "v6-small-rec" => DownloadSpec {
            repository: "PaddlePaddle/PP-OCRv6_small_rec_onnx",
            revision: "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
            model_bytes: 21_159_378,
            model_sha256: "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
            config_bytes: Some(150_579),
            config_sha256: Some("ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1"),
            dictionary_entries: Some(18_708),
            dictionary_bytes: Some(74_947),
            dictionary_sha256: Some(
                "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
            ),
        },
        "v6-medium-rec" => DownloadSpec {
            repository: "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
            revision: "50c7eacafc52fa7bcf4194e8cd08e46f8558504b",
            model_bytes: 76_554_979,
            model_sha256: "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba",
            config_bytes: Some(150_580),
            config_sha256: Some("991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129"),
            dictionary_entries: Some(18_708),
            dictionary_bytes: Some(74_947),
            dictionary_sha256: Some(
                "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
            ),
        },
        "v5-eslav-rec" => DownloadSpec {
            repository: "PaddlePaddle/eslav_PP-OCRv5_mobile_rec_onnx",
            revision: "9a32171fc5718746875e1a261818884517975013",
            model_bytes: 7_887_627,
            model_sha256: "b3018ef2b09a0250b6e0c8e871c927098363e5fd4df890cc68e8358eb0aaf1bd",
            config_bytes: Some(4_538),
            config_sha256: Some("025039bac23eb4a308efcefa4d58eab3af440767815c6ba6938468bf6353ee5a"),
            dictionary_entries: Some(517),
            dictionary_bytes: Some(1_663),
            dictionary_sha256: Some(
                "3e95f1581557162870cacdba5af91a4c6be2890710d395b0c3c7578e7ee5e6eb",
            ),
        },
        "v5-latin-rec" => DownloadSpec {
            repository: "PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx",
            revision: "89d3a50e2c27e2e7cceeab0e944c25c807d5db4f",
            model_bytes: 8_042_023,
            model_sha256: "7888113072263cb471b93f66dd5e2ad70548dc526fa1ace760d0d973dd121498",
            config_bytes: Some(6_817),
            config_sha256: Some("0bbe984570f597af3638e50bdf2e8276f3ab26a61966096538b3b0d1849f5c84"),
            dictionary_entries: Some(836),
            dictionary_bytes: Some(2_616),
            dictionary_sha256: Some(
                "ccbcc45730b3fbbd9050c5bc74db6a99067141ef1035e3d14889a84a6b9b1aff",
            ),
        },
        "v5-korean-rec" => DownloadSpec {
            repository: "PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx",
            revision: "5c6f574b8e2230adf4287b33e736d71b9fabd28e",
            model_bytes: 13_418_787,
            model_sha256: "92f0b7785e64fc9090106a241cf4c1eb97472824558272751b88a2a4476d3a08",
            config_bytes: Some(96_039),
            config_sha256: Some("f757fa1c40e99edcf27e9cce879b93eb2a51fa46f5ef39095689b8c37dd75998"),
            dictionary_entries: Some(11_945),
            dictionary_bytes: Some(47_451),
            dictionary_sha256: Some(
                "a88071c68c01707489baa79ebe0405b7beb5cca229f4fc94cc3ef992328802d7",
            ),
        },
        _ => return None,
    };
    Some(spec)
}

fn validate_config(config: &OnnxConfig) -> Result<()> {
    if config.threads == 0 || config.threads > 64 {
        return Err(Error::InvalidOption(
            "ONNX threads must be between 1 and 64".to_owned(),
        ));
    }
    if config.max_file_bytes == 0 || config.max_image_pixels == 0 {
        return Err(Error::InvalidOption(
            "image resource limits must be greater than zero".to_owned(),
        ));
    }
    Ok(())
}

fn validate_min_confidence(value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Error::InvalidOption(
            "min_confidence must be between 0 and 1".to_owned(),
        ));
    }
    Ok(())
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|threads| recommended_threads(threads.get()))
        .unwrap_or(1)
}

fn recommended_threads(parallelism: usize) -> usize {
    parallelism.clamp(1, 8)
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|error| Error::io(path, error))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| Error::io(path, error))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn resolve_languages(requested: &[String]) -> Result<Vec<String>> {
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let mut languages = Vec::new();
    let mut unsupported = Vec::new();
    for language in requested {
        let normalized = normalize_language(language);
        if !SUPPORTED_LANGUAGES.contains(&normalized.as_str()) {
            unsupported.push(language.clone());
        } else if !languages.contains(&normalized) {
            languages.push(normalized);
        }
    }
    if unsupported.is_empty() {
        Ok(languages)
    } else {
        Err(Error::UnsupportedLanguage {
            requested: unsupported,
            available: SUPPORTED_LANGUAGES
                .iter()
                .map(|language| (*language).to_owned())
                .collect(),
        })
    }
}

fn validate_profile_languages(quality: QualityMode, languages: &[String]) -> Result<()> {
    if quality == QualityMode::Fast && languages.iter().any(|language| language == "ja") {
        return Err(Error::UnsupportedLanguage {
            requested: vec!["ja".to_owned()],
            available: profile_languages(quality),
        });
    }
    Ok(())
}

fn profile_languages(quality: QualityMode) -> Vec<String> {
    SUPPORTED_LANGUAGES
        .iter()
        .filter(|language| quality != QualityMode::Fast || **language != "ja")
        .map(|language| (*language).to_owned())
        .collect()
}

fn normalize_language(language: &str) -> String {
    let language = language.trim().to_ascii_lowercase().replace('_', "-");
    match language.as_str() {
        "bel" => "be".to_owned(),
        "cat" => "ca".to_owned(),
        "ces" | "cze" => "cs".to_owned(),
        "dan" => "da".to_owned(),
        "deu" | "ger" | "german" => "de".to_owned(),
        "dut" | "nld" => "nl".to_owned(),
        "eng" => "en".to_owned(),
        "est" => "et".to_owned(),
        "eus" | "baq" => "eu".to_owned(),
        "fin" => "fi".to_owned(),
        "fra" | "fre" | "french" => "fr".to_owned(),
        "glg" => "gl".to_owned(),
        "hrv" => "hr".to_owned(),
        "hun" => "hu".to_owned(),
        "ind" => "id".to_owned(),
        "isl" | "ice" => "is".to_owned(),
        "ita" => "it".to_owned(),
        "jpn" => "ja".to_owned(),
        "kor" | "korean" => "ko".to_owned(),
        "lav" => "lv".to_owned(),
        "lit" => "lt".to_owned(),
        "nno" | "nob" | "nor" => "no".to_owned(),
        "pol" => "pl".to_owned(),
        "por" => "pt".to_owned(),
        "ron" | "rum" => "ro".to_owned(),
        "rus" => "ru".to_owned(),
        "slk" | "slo" => "sk".to_owned(),
        "slv" => "sl".to_owned(),
        "spa" => "es".to_owned(),
        "sr-latn" | "rs-latin" => "sr-latn".to_owned(),
        "swe" => "sv".to_owned(),
        "tur" => "tr".to_owned(),
        "ukr" => "uk".to_owned(),
        "vie" => "vi".to_owned(),
        "chi-sim" | "chi-sim-vert" | "zho" => "zh".to_owned(),
        value => value.split('-').next().unwrap_or(value).to_owned(),
    }
}

fn crop_ratio(image: &RgbImage) -> f32 {
    image.width() as f32 / image.height().max(1) as f32
}

fn wants_language_group(languages: &[String], group: &[&str]) -> bool {
    languages
        .iter()
        .any(|language| group.contains(&language.as_str()))
}

fn wants_specialized_latin(languages: &[String]) -> bool {
    languages
        .iter()
        .any(|language| language != "en" && LATIN_LANGUAGES.contains(&language.as_str()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RecognizerPlan {
    primary: bool,
    latin: bool,
    cyrillic: bool,
    korean: bool,
}

fn recognizer_plan(languages: &[String]) -> RecognizerPlan {
    if languages.is_empty() {
        return RecognizerPlan {
            primary: true,
            latin: true,
            cyrillic: true,
            korean: true,
        };
    }
    let cyrillic = wants_language_group(languages, CYRILLIC_LANGUAGES);
    let cjk = wants_language_group(languages, CJK_LANGUAGES);
    let korean = wants_language_group(languages, KOREAN_LANGUAGES);
    let latin_group = wants_language_group(languages, LATIN_LANGUAGES);
    let latin = wants_specialized_latin(languages);
    let script_groups = [cyrillic, cjk, korean, latin_group]
        .into_iter()
        .filter(|requested| *requested)
        .count();
    RecognizerPlan {
        primary: cjk || script_groups > 1 || (latin_group && !latin),
        latin,
        cyrillic,
        korean,
    }
}

fn specialist_crop_indices(
    primary: Option<&[Option<RecognizedCandidate>]>,
    crop_count: usize,
) -> Vec<usize> {
    let Some(primary) = primary else {
        return (0..crop_count).collect();
    };
    (0..crop_count)
        .filter(|index| {
            primary.get(*index).is_none_or(|candidate| {
                candidate.as_ref().is_none_or(|candidate| {
                    candidate.confidence < 0.82 || contains_cyrillic(&candidate.text)
                })
            })
        })
        .collect()
}

fn auto_specialist_crop_indices(
    primary: Option<&[Option<RecognizedCandidate>]>,
    crop_count: usize,
    script: SpecialistScript,
) -> Vec<usize> {
    let Some(primary) = primary else {
        return (0..crop_count).collect();
    };
    (0..crop_count)
        .filter(|index| {
            primary.get(*index).is_none_or(|candidate| {
                candidate.as_ref().is_none_or(|candidate| {
                    candidate.confidence < 0.99 || contains_script(&candidate.text, script)
                })
            })
        })
        .collect()
}

fn merge_recognized_regions(
    boxes: &[BoundingBox],
    primary: Option<Vec<Option<RecognizedCandidate>>>,
    specialists: &[(SpecialistScript, Vec<Option<RecognizedCandidate>>)],
    auto_languages: bool,
) -> Vec<MergedRegion> {
    boxes
        .iter()
        .enumerate()
        .filter_map(|(index, box_)| {
            let mut selected = candidate_at(primary.as_deref(), index)
                .map(|candidate| recognized_region(box_, candidate));
            for (script, candidates) in specialists {
                let specialist = candidate_at(Some(candidates), index)
                    .map(|candidate| recognized_region(box_, candidate));
                selected = match (selected, specialist) {
                    (Some(primary), Some(specialist)) => {
                        Some(select_region(primary, specialist, *script, auto_languages))
                    }
                    (Some(region), None) | (None, Some(region)) => Some(region),
                    (None, None) => None,
                };
            }
            selected
        })
        .collect()
}

fn candidate_at(
    candidates: Option<&[Option<RecognizedCandidate>]>,
    index: usize,
) -> Option<RecognizedCandidate> {
    candidates?.get(index)?.clone()
}

fn recognized_region(box_: &BoundingBox, candidate: RecognizedCandidate) -> MergedRegion {
    let mut region = OarTextRegion::with_recognition(
        box_.clone(),
        Some(Arc::<str>::from(candidate.text)),
        Some(candidate.confidence),
    );
    region.dt_poly = Some(box_.clone());
    region.rec_poly = Some(box_.clone());
    MergedRegion {
        region,
        alternative: None,
    }
}

fn select_region(
    primary: MergedRegion,
    specialist: MergedRegion,
    script: SpecialistScript,
    auto_languages: bool,
) -> MergedRegion {
    let primary_candidate = candidate(&primary.region);
    let specialist_candidate = candidate(&specialist.region);
    let use_specialist = match (&primary_candidate, &specialist_candidate) {
        (_, None) => false,
        (None, Some(_)) => true,
        (Some((_, primary_score)), Some((specialist_text, specialist_score))) => {
            if auto_languages
                && script != SpecialistScript::Latin
                && contains_script(specialist_text, script)
            {
                specialist_score >= &0.75
            } else if contains_script(specialist_text, script) {
                specialist_score >= &(primary_score - 0.18)
            } else {
                specialist_score > &(primary_score + 0.08)
            }
        }
    };
    let (selected, alternative) = if use_specialist {
        (specialist.region, primary_candidate)
    } else {
        (primary.region, specialist_candidate)
    };
    MergedRegion {
        region: selected,
        alternative: alternative.map(|(text, confidence)| TextAlternative {
            text: text.to_owned(),
            confidence: confidence.clamp(0.0, 1.0),
        }),
    }
}

fn contains_script(text: &str, script: SpecialistScript) -> bool {
    match script {
        SpecialistScript::Cyrillic => contains_cyrillic(text),
        SpecialistScript::Latin => contains_latin(text),
        SpecialistScript::Korean => contains_korean(text),
    }
}

fn candidate(region: &OarTextRegion) -> Option<(&str, f32)> {
    region
        .text_with_confidence()
        .filter(|(text, _)| !text.trim().is_empty())
}

fn contains_cyrillic(text: &str) -> bool {
    text.chars()
        .any(|character| ('\u{0400}'..='\u{052f}').contains(&character))
}

fn contains_latin(text: &str) -> bool {
    text.chars().any(|character| {
        character.is_ascii_alphabetic()
            || ('\u{00c0}'..='\u{024f}').contains(&character)
            || ('\u{1e00}'..='\u{1eff}').contains(&character)
    })
}

fn contains_korean(text: &str) -> bool {
    text.chars().any(|character| {
        ('\u{1100}'..='\u{11ff}').contains(&character)
            || ('\u{3130}'..='\u{318f}').contains(&character)
            || ('\u{ac00}'..='\u{d7af}').contains(&character)
    })
}

fn contains_kana(text: &str) -> bool {
    text.chars()
        .any(|character| ('\u{3040}'..='\u{30ff}').contains(&character))
}

fn contains_han(text: &str) -> bool {
    text.chars()
        .any(|character| ('\u{3400}'..='\u{9fff}').contains(&character))
}

fn script_tag(text: &str) -> Option<&'static str> {
    if contains_cyrillic(text) {
        Some("Cyrl")
    } else if contains_korean(text) {
        Some("Kore")
    } else if contains_kana(text) {
        Some("Jpan")
    } else if contains_han(text) {
        Some("Hani")
    } else if text.chars().any(|character| character.is_alphabetic()) {
        Some("Latn")
    } else {
        None
    }
}

fn line_language(text: &str, languages: &[String]) -> Option<String> {
    let source = if languages.is_empty() {
        SUPPORTED_LANGUAGES.to_vec()
    } else {
        languages.iter().map(String::as_str).collect()
    };
    let compatible = match script_tag(text) {
        Some("Cyrl") => source
            .iter()
            .filter(|language| CYRILLIC_LANGUAGES.contains(language))
            .collect::<Vec<_>>(),
        Some("Jpan") => source
            .iter()
            .filter(|language| **language == "ja")
            .collect::<Vec<_>>(),
        Some("Hani") => source
            .iter()
            .filter(|language| CJK_LANGUAGES.contains(language))
            .collect::<Vec<_>>(),
        Some("Kore") => source
            .iter()
            .filter(|language| **language == "ko")
            .collect::<Vec<_>>(),
        Some("Latn") => source
            .iter()
            .filter(|language| LATIN_LANGUAGES.contains(language))
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    (compatible.len() == 1).then(|| (*compatible[0]).to_owned())
}

fn quad_from_box(box_: &BoundingBox, width: u32, height: u32) -> Option<Quad> {
    if box_.points.len() >= 4 {
        let points = std::array::from_fn(|index| Point {
            x: box_.points[index].x.clamp(0.0, width as f32),
            y: box_.points[index].y.clamp(0.0, height as f32),
        });
        return Some(Quad { points });
    }
    if box_.points.is_empty() {
        return None;
    }
    let min_x = box_
        .points
        .iter()
        .map(|point| point.x)
        .fold(f32::INFINITY, f32::min)
        .clamp(0.0, width as f32);
    let min_y = box_
        .points
        .iter()
        .map(|point| point.y)
        .fold(f32::INFINITY, f32::min)
        .clamp(0.0, height as f32);
    let max_x = box_
        .points
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max)
        .clamp(0.0, width as f32);
    let max_y = box_
        .points
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .clamp(0.0, height as f32);
    (max_x > min_x && max_y > min_y)
        .then(|| Quad::from_rect(min_x, min_y, max_x - min_x, max_y - min_y))
}

fn build_words(
    line_id: &str,
    region: &OarTextRegion,
    text: &str,
    confidence: f32,
    width: u32,
    height: u32,
) -> Vec<TextWord> {
    let texts = text.split_whitespace().collect::<Vec<_>>();
    let Some(boxes) = region.word_boxes.as_ref() else {
        return Vec::new();
    };
    if boxes.len() != texts.len() {
        return Vec::new();
    }
    boxes
        .iter()
        .zip(texts)
        .enumerate()
        .filter_map(|(index, (box_, text))| {
            Some(TextWord {
                id: format!("{line_id}-word-{:04}", index + 1),
                quad: quad_from_box(box_, width, height)?,
                text: text.to_owned(),
                confidence: Some(confidence.clamp(0.0, 1.0)),
            })
        })
        .collect()
}

fn backend_error(message: impl Into<String>) -> Error {
    Error::Backend {
        backend: "onnxruntime",
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn normalizes_common_language_identifiers() {
        assert_eq!(normalize_language("ru-RU"), "ru");
        assert_eq!(normalize_language("jpn"), "ja");
        assert_eq!(normalize_language("chi_sim"), "zh");
        assert_eq!(normalize_language("ko-KR"), "ko");
        assert_eq!(normalize_language("kor"), "ko");
        assert_eq!(normalize_language("deu"), "de");
        assert_eq!(normalize_language("sr-Latn"), "sr-latn");
        assert!(
            resolve_languages(&[])
                .expect("auto language mode must resolve")
                .is_empty()
        );
    }

    #[test]
    fn classifies_supported_scripts_without_guessing() {
        assert_eq!(script_tag("한국어 2026"), Some("Kore"));
        assert_eq!(
            line_language("한국어", &["ko".to_owned()]),
            Some("ko".to_owned())
        );
        assert_eq!(script_tag("本地文字"), Some("Hani"));
        assert_eq!(script_tag("ひらがな漢字"), Some("Jpan"));
        assert!(contains_script("Příliš Straße", SpecialistScript::Latin));
        assert!(!contains_script("本地文字", SpecialistScript::Latin));
        assert!(profile_languages(QualityMode::Balanced).contains(&"ko".to_owned()));
        assert!(LATIN_LANGUAGES.contains(&"cs"));
        assert!(LATIN_LANGUAGES.contains(&"de"));
        assert!(!wants_specialized_latin(&[
            "en".to_owned(),
            "ru".to_owned(),
            "ja".to_owned(),
        ]));
        assert!(wants_specialized_latin(
            &["de".to_owned(), "zh".to_owned(),]
        ));
        assert_eq!(
            recognizer_plan(&["en".to_owned()]),
            RecognizerPlan {
                primary: true,
                latin: false,
                cyrillic: false,
                korean: false,
            }
        );
        assert_eq!(
            recognizer_plan(&[]),
            RecognizerPlan {
                primary: true,
                latin: true,
                cyrillic: true,
                korean: true,
            }
        );
        assert_eq!(line_language("日本語かな", &[]), Some("ja".to_owned()));
        assert_eq!(line_language("안녕하세요", &[]), Some("ko".to_owned()));
        assert_eq!(line_language("Hello", &[]), None);
        assert_eq!(
            recognizer_plan(&["de".to_owned(), "zh".to_owned()]),
            RecognizerPlan {
                primary: true,
                latin: true,
                cyrillic: false,
                korean: false,
            }
        );
    }

    #[test]
    fn fast_profile_rejects_japanese() {
        let error = validate_profile_languages(QualityMode::Fast, &["ja".to_owned()])
            .expect_err("Japanese must use a profile that supports it");

        assert!(matches!(error, Error::UnsupportedLanguage { .. }));
        assert!(!profile_languages(QualityMode::Fast).contains(&"ja".to_owned()));
        assert!(profile_languages(QualityMode::Balanced).contains(&"ja".to_owned()));
    }

    #[test]
    fn maximum_profile_uses_medium_models() {
        let profile = ModelProfile::for_quality(QualityMode::Maximum);

        assert_eq!(profile.detector.directory, "v6-medium-det");
        assert_eq!(profile.recognizer.model.directory, "v6-medium-rec");
    }

    #[test]
    fn balanced_profile_uses_conservative_hybrid() {
        let profile = ModelProfile::for_quality(QualityMode::Balanced);

        assert_eq!(profile.detector.directory, "v5-mobile-det");
        assert_eq!(profile.recognizer.model.directory, "v6-small-rec");
    }

    #[test]
    fn default_thread_policy_scales_without_oversubscribing_small_devices() {
        assert_eq!(recommended_threads(1), 1);
        assert_eq!(recommended_threads(4), 4);
        assert_eq!(recommended_threads(16), 8);
    }

    #[test]
    fn engine_construction_does_not_require_or_install_models() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after the Unix epoch")
            .as_nanos();
        let models_dir = std::env::temp_dir().join(format!("glypho-models-{nonce}"));
        let mut config = OnnxConfig::new(&models_dir);
        config.auto_download = false;

        let engine = OnnxEngine::new(config).expect("engine metadata must be lazy");

        assert_eq!(engine.info().models_dir, models_dir);
        assert!(!models_dir.exists());
    }

    #[cfg(not(feature = "openvino"))]
    #[test]
    fn unavailable_openvino_falls_back_to_cpu() {
        let resolution = resolve_device(Device::OpenVino);

        assert_eq!(resolution.resolved, Device::Cpu);
        assert!(
            resolution
                .fallback_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("not included"))
        );
    }
}
