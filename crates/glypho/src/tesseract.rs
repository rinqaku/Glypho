use std::collections::{BTreeMap, HashSet};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::document::Metadata;
use crate::process;
use crate::{
    Document, EngineInfo, Error, EvaluationPolicy, ImageInfo, Legibility, Quad, RegionSource,
    Result, TextDirection, TextLine, TextWord,
};

#[derive(Clone, Debug)]
pub struct Glypho {
    config: TesseractConfig,
}

impl Glypho {
    pub fn new(config: TesseractConfig) -> Self {
        Self { config }
    }

    pub fn recognize(
        &self,
        image_path: impl AsRef<Path>,
        options: &RecognitionOptions,
    ) -> Result<Document> {
        let image_path = image_path.as_ref();
        let metadata = fs::metadata(image_path).map_err(|error| Error::io(image_path, error))?;
        if !metadata.is_file() {
            return Err(Error::InvalidOption(format!(
                "input is not a file: {}",
                image_path.display()
            )));
        }
        if !options.min_confidence.is_finite() || !(0.0..=1.0).contains(&options.min_confidence) {
            return Err(Error::InvalidOption(
                "min_confidence must be between 0 and 1".to_owned(),
            ));
        }

        let info = self.info()?;
        let languages = resolve_languages(&options.languages, &info.languages)?;
        let language_arg = languages.join("+");
        let args = vec![
            image_path.as_os_str().to_owned(),
            OsString::from("stdout"),
            OsString::from("-l"),
            OsString::from(&language_arg),
            OsString::from("--psm"),
            OsString::from(options.segmentation.as_tesseract_value()),
            OsString::from("tsv"),
        ];

        let started = Instant::now();
        let output = process::run(&self.config.binary, &args, self.config.timeout)?;
        let elapsed_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
        let tsv = String::from_utf8_lossy(&output.stdout);
        let _diagnostics = String::from_utf8_lossy(&output.stderr);
        let mut document = parse_tsv(&tsv, image_path, &languages, options.min_confidence)?;
        document.metadata = Metadata {
            engine: Some(EngineInfo {
                name: "Glypho".to_owned(),
                version: env!("CARGO_PKG_VERSION").to_owned(),
                backend: "tesseract".to_owned(),
                model: Some(info.version.clone()),
                languages,
                elapsed_ms,
            }),
            ..Metadata::default()
        };
        document.validate()?;
        Ok(document)
    }

    pub fn info(&self) -> Result<TesseractInfo> {
        let version_output = process::run(
            &self.config.binary,
            &[OsString::from("--version")],
            self.config.timeout,
        )?;
        let version = String::from_utf8_lossy(&version_output.stdout)
            .lines()
            .next()
            .unwrap_or("tesseract")
            .trim()
            .to_owned();

        let languages_output = process::run(
            &self.config.binary,
            &[OsString::from("--list-langs")],
            self.config.timeout,
        )?;
        let languages = String::from_utf8_lossy(&languages_output.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with("List of available"))
            .map(str::to_owned)
            .collect();

        Ok(TesseractInfo { version, languages })
    }
}

impl Default for Glypho {
    fn default() -> Self {
        Self::new(TesseractConfig::default())
    }
}

#[derive(Clone, Debug)]
pub struct TesseractConfig {
    pub binary: PathBuf,
    pub timeout: Duration,
}

impl Default for TesseractConfig {
    fn default() -> Self {
        Self {
            binary: PathBuf::from("tesseract"),
            timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RecognitionOptions {
    pub languages: Vec<String>,
    pub segmentation: PageSegmentation,
    pub min_confidence: f32,
}

impl Default for RecognitionOptions {
    fn default() -> Self {
        Self {
            languages: Vec::new(),
            segmentation: PageSegmentation::SparseText,
            min_confidence: 0.8,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PageSegmentation {
    Auto,
    SingleBlock,
    SingleLine,
    #[default]
    SparseText,
}

impl PageSegmentation {
    fn as_tesseract_value(self) -> &'static str {
        match self {
            Self::Auto => "3",
            Self::SingleBlock => "6",
            Self::SingleLine => "7",
            Self::SparseText => "11",
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TesseractInfo {
    pub version: String,
    pub languages: Vec<String>,
}

#[derive(Debug)]
struct TsvWord {
    line_key: (u32, u32, u32, u32),
    left: f32,
    top: f32,
    width: f32,
    height: f32,
    confidence: f32,
    text: String,
}

fn parse_tsv(
    tsv: &str,
    image_path: &Path,
    languages: &[String],
    min_confidence: f32,
) -> Result<Document> {
    let mut dimensions = None;
    let mut words = Vec::new();

    for (line_index, line) in tsv.lines().enumerate() {
        if line_index == 0 || line.trim().is_empty() {
            continue;
        }
        let columns = line.splitn(12, '\t').collect::<Vec<_>>();
        if columns.len() != 12 {
            return Err(Error::InvalidTsv(format!(
                "line {} has {} columns, expected 12",
                line_index + 1,
                columns.len()
            )));
        }

        let level = parse_column::<u8>(&columns, 0, line_index)?;
        let left = parse_column::<f32>(&columns, 6, line_index)?;
        let top = parse_column::<f32>(&columns, 7, line_index)?;
        let width = parse_column::<f32>(&columns, 8, line_index)?;
        let height = parse_column::<f32>(&columns, 9, line_index)?;
        if level == 1 {
            dimensions = Some((width.max(0.0) as u32, height.max(0.0) as u32));
            continue;
        }
        if level != 5 {
            continue;
        }

        let confidence = parse_column::<f32>(&columns, 10, line_index)? / 100.0;
        let text = columns[11].trim();
        if text.is_empty() || confidence < min_confidence || confidence < 0.0 {
            continue;
        }
        words.push(TsvWord {
            line_key: (
                parse_column(&columns, 1, line_index)?,
                parse_column(&columns, 2, line_index)?,
                parse_column(&columns, 3, line_index)?,
                parse_column(&columns, 4, line_index)?,
            ),
            left,
            top,
            width,
            height,
            confidence: confidence.clamp(0.0, 1.0),
            text: text.to_owned(),
        });
    }

    let (width, height) = dimensions.ok_or_else(|| {
        Error::InvalidTsv("page dimensions are missing from Tesseract output".to_owned())
    })?;
    if width == 0 || height == 0 {
        return Err(Error::InvalidTsv(
            "Tesseract returned empty page dimensions".to_owned(),
        ));
    }

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
    document.language_hints = languages.to_vec();

    let mut grouped = BTreeMap::<(u32, u32, u32, u32), Vec<TsvWord>>::new();
    for word in words {
        grouped.entry(word.line_key).or_default().push(word);
    }

    for (line_index, (_, words)) in grouped.into_iter().enumerate() {
        let line_id = format!("line-{:04}", line_index + 1);
        let left = words
            .iter()
            .map(|word| word.left)
            .fold(f32::INFINITY, f32::min);
        let top = words
            .iter()
            .map(|word| word.top)
            .fold(f32::INFINITY, f32::min);
        let right = words
            .iter()
            .map(|word| word.left + word.width)
            .fold(f32::NEG_INFINITY, f32::max);
        let bottom = words
            .iter()
            .map(|word| word.top + word.height)
            .fold(f32::NEG_INFINITY, f32::max);
        let confidence = weighted_confidence(&words);
        let text = words
            .iter()
            .map(|word| word.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let line_words = words
            .into_iter()
            .enumerate()
            .map(|(word_index, word)| TextWord {
                id: format!("{line_id}-word-{:04}", word_index + 1),
                quad: Quad::from_rect(word.left, word.top, word.width, word.height),
                text: word.text,
                confidence: Some(word.confidence),
            })
            .collect();

        document.lines.push(TextLine {
            id: line_id,
            order: line_index as u32,
            quad: Quad::from_rect(left, top, right - left, bottom - top),
            text,
            corrected_text: None,
            alternatives: Vec::new(),
            confidence: Some(confidence),
            language: None,
            script: None,
            direction: TextDirection::Auto,
            legibility: Legibility::Clear,
            flags: Vec::new(),
            evaluation: EvaluationPolicy::default(),
            source: RegionSource::Model,
            words: line_words,
            ignored: false,
        });
    }

    document.sort_reading_order();
    Ok(document)
}

fn parse_column<T>(columns: &[&str], index: usize, line_index: usize) -> Result<T>
where
    T: std::str::FromStr,
{
    columns[index].parse().map_err(|_| {
        Error::InvalidTsv(format!(
            "line {}, column {} is not a number",
            line_index + 1,
            index + 1
        ))
    })
}

fn weighted_confidence(words: &[TsvWord]) -> f32 {
    let mut weight = 0usize;
    let mut total = 0.0;
    for word in words {
        let word_weight = word.text.chars().count().max(1);
        weight += word_weight;
        total += word.confidence * word_weight as f32;
    }
    total / weight.max(1) as f32
}

fn resolve_languages(requested: &[String], available: &[String]) -> Result<Vec<String>> {
    let requested = if requested.is_empty() {
        available
            .iter()
            .find(|language| language.as_str() == "eng")
            .or_else(|| available.iter().find(|language| language.as_str() != "osd"))
            .cloned()
            .into_iter()
            .collect::<Vec<_>>()
    } else {
        let mut seen = HashSet::new();
        requested
            .iter()
            .map(|language| normalize_language(language))
            .filter(|language| seen.insert(language.clone()))
            .collect()
    };

    let missing = requested
        .iter()
        .filter(|language| !available.contains(language))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(Error::UnsupportedLanguage {
            requested: missing,
            available: available.to_vec(),
        });
    }
    if requested.is_empty() {
        return Err(Error::UnsupportedLanguage {
            requested: vec!["at least one OCR language".to_owned()],
            available: available.to_vec(),
        });
    }
    Ok(requested)
}

fn normalize_language(language: &str) -> String {
    let language = language.trim().to_lowercase();
    match language.as_str() {
        "zh-hans" | "zh_cn" | "zh-cn" => return "chi_sim".to_owned(),
        "zh-hant" | "zh_tw" | "zh-tw" => return "chi_tra".to_owned(),
        _ => {}
    }
    let base = language.split(['-', '_']).next().unwrap_or(&language);
    match base {
        "ar" | "ara" => "ara".to_owned(),
        "bg" | "bul" => "bul".to_owned(),
        "bn" | "ben" => "ben".to_owned(),
        "cs" | "ces" | "cze" => "ces".to_owned(),
        "da" | "dan" => "dan".to_owned(),
        "de" | "deu" | "ger" => "deu".to_owned(),
        "el" | "ell" | "gre" => "ell".to_owned(),
        "en" | "eng" => "eng".to_owned(),
        "es" | "spa" => "spa".to_owned(),
        "et" | "est" => "est".to_owned(),
        "fa" | "fas" | "per" => "fas".to_owned(),
        "fi" | "fin" => "fin".to_owned(),
        "fr" | "fra" | "fre" => "fra".to_owned(),
        "he" | "heb" => "heb".to_owned(),
        "hi" | "hin" => "hin".to_owned(),
        "hr" | "hrv" => "hrv".to_owned(),
        "hu" | "hun" => "hun".to_owned(),
        "id" | "ind" => "ind".to_owned(),
        "it" | "ita" => "ita".to_owned(),
        "ja" | "jpn" => "jpn".to_owned(),
        "ko" | "kor" => "kor".to_owned(),
        "lt" | "lit" => "lit".to_owned(),
        "lv" | "lav" => "lav".to_owned(),
        "ms" | "msa" | "may" => "msa".to_owned(),
        "nl" | "nld" | "dut" => "nld".to_owned(),
        "no" | "nor" => "nor".to_owned(),
        "pl" | "pol" => "pol".to_owned(),
        "pt" | "por" => "por".to_owned(),
        "ro" | "ron" | "rum" => "ron".to_owned(),
        "ru" | "rus" => "rus".to_owned(),
        "sk" | "slk" | "slo" => "slk".to_owned(),
        "sl" | "slv" => "slv".to_owned(),
        "sr" | "srp" => "srp".to_owned(),
        "sv" | "swe" => "swe".to_owned(),
        "ta" | "tam" => "tam".to_owned(),
        "te" | "tel" => "tel".to_owned(),
        "th" | "tha" => "tha".to_owned(),
        "tr" | "tur" => "tur".to_owned(),
        "uk" | "ukr" => "ukr".to_owned(),
        "vi" | "vie" => "vie".to_owned(),
        _ => language,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TSV: &str = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
1\t1\t0\t0\t0\t0\t0\t0\t200\t100\t-1\t\n\
5\t1\t1\t1\t1\t1\t10\t20\t40\t10\t90.0\tHello\n\
5\t1\t1\t1\t1\t2\t55\t20\t35\t10\t80.0\tworld\n\
5\t1\t2\t1\t1\t1\t10\t50\t30\t12\t95.0\tПривет\n";

    #[test]
    fn parses_words_and_lines() {
        let document = parse_tsv(TSV, Path::new("sample.png"), &["eng".to_owned()], 0.0)
            .expect("TSV should parse");

        assert_eq!(document.image.width, 200);
        assert_eq!(document.lines.len(), 2);
        assert_eq!(document.lines[0].text, "Hello world");
        assert_eq!(document.lines[0].words.len(), 2);
        assert_eq!(document.text, "Hello world\nПривет");
        document.validate().expect("document should be valid");
    }

    #[test]
    fn filters_low_confidence_words() {
        let document = parse_tsv(TSV, Path::new("sample.png"), &["eng".to_owned()], 0.85)
            .expect("TSV should parse");

        assert_eq!(document.lines[0].text, "Hello");
    }

    #[test]
    fn normalizes_common_bcp_47_languages() {
        assert_eq!(normalize_language("en-US"), "eng");
        assert_eq!(normalize_language("RU"), "rus");
        assert_eq!(normalize_language("jpn"), "jpn");
        assert_eq!(normalize_language("de-DE"), "deu");
        assert_eq!(normalize_language("zh-Hant"), "chi_tra");
    }

    #[test]
    fn uses_english_as_the_default_when_available() {
        let languages =
            resolve_languages(&[], &["ces".to_owned(), "eng".to_owned(), "rus".to_owned()])
                .expect("default language should resolve");

        assert_eq!(languages, ["eng"]);
    }
}
