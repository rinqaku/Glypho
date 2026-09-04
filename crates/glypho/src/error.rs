use std::fmt;
use std::io;
use std::path::PathBuf;

#[derive(Debug)]
pub enum Error {
    Backend {
        backend: &'static str,
        message: String,
    },
    InvalidDocument(Vec<String>),
    InvalidOption(String),
    InvalidTsv(String),
    Io {
        path: Option<PathBuf>,
        source: io::Error,
    },
    ProcessFailed {
        program: String,
        status: Option<i32>,
        stderr: String,
    },
    ProcessTimedOut {
        program: String,
        seconds: u64,
    },
    ProcessOutputTooLarge {
        program: String,
        stream: String,
        limit_bytes: usize,
    },
    Serialization(serde_json::Error),
    UnsupportedLanguage {
        requested: Vec<String>,
        available: Vec<String>,
    },
}

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            path: Some(path.into()),
            source,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend { backend, message } => write!(formatter, "{backend}: {message}"),
            Self::InvalidDocument(errors) => {
                write!(formatter, "invalid Glypho document: {}", errors.join("; "))
            }
            Self::InvalidOption(message) => write!(formatter, "invalid option: {message}"),
            Self::InvalidTsv(message) => write!(formatter, "invalid Tesseract TSV: {message}"),
            Self::Io { path, source } => match path {
                Some(path) => write!(formatter, "{}: {source}", path.display()),
                None => write!(formatter, "I/O error: {source}"),
            },
            Self::ProcessFailed {
                program,
                status,
                stderr,
            } => {
                let status = status
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "signal".to_owned());
                let detail = stderr.trim();
                if detail.is_empty() {
                    write!(formatter, "{program} failed with status {status}")
                } else {
                    write!(formatter, "{program} failed with status {status}: {detail}")
                }
            }
            Self::ProcessTimedOut { program, seconds } => {
                write!(formatter, "{program} timed out after {seconds}s")
            }
            Self::ProcessOutputTooLarge {
                program,
                stream,
                limit_bytes,
            } => write!(
                formatter,
                "{program} produced more than {limit_bytes} bytes on {stream}"
            ),
            Self::Serialization(error) => write!(formatter, "invalid JSON: {error}"),
            Self::UnsupportedLanguage {
                requested,
                available,
            } => write!(
                formatter,
                "OCR languages are not installed: {}; available: {}",
                requested.join(", "),
                available.join(", ")
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Serialization(source) => Some(source),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
