mod document;
mod error;
mod evaluate;
mod ffi;
mod onnx;
mod process;
mod tesseract;

pub use document::{
    AnnotationStatus, CoordinateSystem, Document, EngineInfo, EvaluationPolicy, ImageInfo,
    Legibility, Point, Quad, RegionSource, SCHEMA_VERSION, TextAlternative, TextDirection,
    TextLine, TextWord,
};
pub use error::{Error, Result};
pub use evaluate::{ErrorRate, Evaluation, evaluate};
pub use onnx::{Device, OnnxConfig, OnnxEngine, OnnxInfo, QualityMode, default_models_dir};
pub use tesseract::{Glypho, PageSegmentation, RecognitionOptions, TesseractConfig, TesseractInfo};
