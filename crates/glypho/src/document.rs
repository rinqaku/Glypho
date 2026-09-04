use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

pub const SCHEMA_VERSION: &str = "glypho.annotation.v1";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Document {
    pub schema_version: String,
    pub coordinate_system: CoordinateSystem,
    pub image: ImageInfo,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corrected_text: Option<String>,
    pub lines: Vec<TextLine>,
    pub language_hints: Vec<String>,
    pub metadata: Metadata,
}

impl Document {
    pub fn new(image: ImageInfo) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_owned(),
            coordinate_system: CoordinateSystem::PixelTopLeft,
            image,
            text: String::new(),
            corrected_text: None,
            lines: Vec::new(),
            language_hints: Vec::new(),
            metadata: Metadata::default(),
        }
    }

    pub fn rebuild_text(&mut self) {
        self.lines.sort_by_key(|line| line.order);
        self.text = self
            .lines
            .iter()
            .filter(|line| !line.ignored && !line.text.trim().is_empty())
            .map(|line| line.text.trim())
            .collect::<Vec<_>>()
            .join("\n");
        let corrected = self
            .lines
            .iter()
            .filter(|line| !line.ignored && !line.text.trim().is_empty())
            .map(|line| line.corrected_text.as_deref().unwrap_or(&line.text).trim())
            .collect::<Vec<_>>()
            .join("\n");
        self.corrected_text = (corrected != self.text).then_some(corrected);
    }

    pub fn sort_reading_order(&mut self) {
        self.lines.sort_by(|left, right| {
            left.quad
                .bounds()
                .y
                .total_cmp(&right.quad.bounds().y)
                .then_with(|| left.quad.bounds().x.total_cmp(&right.quad.bounds().x))
        });

        let mut rows: Vec<Vec<TextLine>> = Vec::new();
        for line in self.lines.drain(..) {
            let bounds = line.quad.bounds();
            let same_row = rows.last().is_some_and(|row| {
                let reference = row.last().expect("a row is never empty").quad.bounds();
                let tolerance = bounds.height.min(reference.height) * 0.5;
                (bounds.center_y() - reference.center_y()).abs() <= tolerance
            });
            if same_row {
                rows.last_mut().expect("the row exists").push(line);
            } else {
                rows.push(vec![line]);
            }
        }

        for row in &mut rows {
            row.sort_by(|left, right| left.quad.bounds().x.total_cmp(&right.quad.bounds().x));
        }
        self.lines = rows.into_iter().flatten().collect();

        for (order, line) in self.lines.iter_mut().enumerate() {
            line.order = order as u32;
        }
        self.rebuild_text();
    }

    pub fn validate(&self) -> Result<()> {
        let mut errors = Vec::new();

        if self.schema_version != SCHEMA_VERSION {
            errors.push(format!(
                "schema_version must be {SCHEMA_VERSION}, got {}",
                self.schema_version
            ));
        }
        if self.image.id.trim().is_empty() {
            errors.push("image.id must not be empty".to_owned());
        }
        if self.image.file_name.trim().is_empty() {
            errors.push("image.file_name must not be empty".to_owned());
        }
        if self.image.width == 0 || self.image.height == 0 {
            errors.push("image dimensions must be greater than zero".to_owned());
        }
        if let Some(sha256) = &self.image.sha256
            && (sha256.len() != 64
                || !sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
        {
            errors.push("image.sha256 must be 64 lowercase hexadecimal characters".to_owned());
        }
        validate_non_empty_unique(&self.language_hints, "language_hints", &mut errors);
        validate_non_empty_unique(&self.metadata.tags, "metadata.tags", &mut errors);
        if self
            .metadata
            .group_id
            .as_ref()
            .is_some_and(|group_id| group_id.trim().is_empty())
        {
            errors.push("metadata.group_id must not be empty".to_owned());
        }

        let mut ids = HashSet::new();
        let mut orders = HashSet::new();
        for (index, line) in self.lines.iter().enumerate() {
            let path = format!("lines[{index}]");
            validate_id(&line.id, &path, &mut ids, &mut errors);
            if !orders.insert(line.order) {
                errors.push(format!("{path}.order {} is duplicated", line.order));
            }
            validate_quad(
                &line.quad,
                self.image.width,
                self.image.height,
                &format!("{path}.quad"),
                &mut errors,
            );
            validate_confidence(line.confidence, &path, &mut errors);
            if line
                .language
                .as_ref()
                .is_some_and(|language| language.trim().is_empty())
            {
                errors.push(format!("{path}.language must not be empty"));
            }
            if line
                .script
                .as_ref()
                .is_some_and(|script| script.trim().is_empty())
            {
                errors.push(format!("{path}.script must not be empty"));
            }
            validate_non_empty_unique(&line.flags, &format!("{path}.flags"), &mut errors);
            for (alternative_index, alternative) in line.alternatives.iter().enumerate() {
                validate_confidence(
                    Some(alternative.confidence),
                    &format!("{path}.alternatives[{alternative_index}]"),
                    &mut errors,
                );
            }

            if self.metadata.status.is_final()
                && !line.ignored
                && line.evaluation.recognition
                && line.text.trim().is_empty()
            {
                errors.push(format!(
                    "{path}.text must not be empty in a reviewed document"
                ));
            }

            for (word_index, word) in line.words.iter().enumerate() {
                let word_path = format!("{path}.words[{word_index}]");
                validate_id(&word.id, &word_path, &mut ids, &mut errors);
                validate_quad(
                    &word.quad,
                    self.image.width,
                    self.image.height,
                    &format!("{word_path}.quad"),
                    &mut errors,
                );
                validate_confidence(word.confidence, &word_path, &mut errors);
            }
        }

        if self.metadata.status.is_final() {
            let mut ordered = self.lines.iter().collect::<Vec<_>>();
            ordered.sort_by_key(|line| line.order);
            let expected = ordered
                .into_iter()
                .filter(|line| !line.ignored && !line.text.trim().is_empty())
                .map(|line| line.text.trim())
                .collect::<Vec<_>>()
                .join("\n");
            if self.text != expected {
                errors.push("text must match non-ignored lines in reading order".to_owned());
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(Error::InvalidDocument(errors))
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageInfo {
    pub id: String,
    pub file_name: String,
    pub width: u32,
    pub height: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TextLine {
    pub id: String,
    pub order: u32,
    pub quad: Quad,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corrected_text: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternatives: Vec<TextAlternative>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
    pub direction: TextDirection,
    pub legibility: Legibility,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flags: Vec<String>,
    pub evaluation: EvaluationPolicy,
    pub source: RegionSource,
    pub words: Vec<TextWord>,
    pub ignored: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TextWord {
    pub id: String,
    pub quad: Quad,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TextAlternative {
    pub text: String,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateSystem {
    #[default]
    PixelTopLeft,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Quad {
    pub points: [Point; 4],
}

impl Quad {
    pub fn from_rect(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            points: [
                Point { x, y },
                Point { x: x + width, y },
                Point {
                    x: x + width,
                    y: y + height,
                },
                Point { x, y: y + height },
            ],
        }
    }

    pub fn bounds(&self) -> Bounds {
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for point in self.points {
            min_x = min_x.min(point.x);
            min_y = min_y.min(point.y);
            max_x = max_x.max(point.x);
            max_y = max_y.max(point.y);
        }

        Bounds {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }

    fn signed_area(&self) -> f32 {
        let mut area = 0.0;
        for index in 0..self.points.len() {
            let current = self.points[index];
            let next = self.points[(index + 1) % self.points.len()];
            area += current.x * next.y - next.x * current.y;
        }
        area * 0.5
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Bounds {
    fn center_y(&self) -> f32 {
        self.y + self.height * 0.5
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionSource {
    #[default]
    Manual,
    Model,
    Imported,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnnotationStatus {
    #[default]
    Draft,
    Reviewed,
    Verified,
}

impl AnnotationStatus {
    fn is_final(self) -> bool {
        matches!(self, Self::Reviewed | Self::Verified)
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextDirection {
    #[default]
    Auto,
    LeftToRight,
    RightToLeft,
    Vertical,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Legibility {
    #[default]
    Clear,
    Ambiguous,
    Unreadable,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationPolicy {
    pub detection: bool,
    pub recognition: bool,
}

impl Default for EvaluationPolicy {
    fn default() -> Self {
        Self {
            detection: true,
            recognition: true,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub status: AnnotationStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotator: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<EngineInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub sensitive: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
}

fn is_false(value: &bool) -> bool {
    !value
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EngineInfo {
    pub name: String,
    pub version: String,
    pub backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub languages: Vec<String>,
    pub elapsed_ms: u64,
}

fn validate_id(id: &str, path: &str, ids: &mut HashSet<String>, errors: &mut Vec<String>) {
    if id.trim().is_empty() {
        errors.push(format!("{path}.id must not be empty"));
        return;
    }
    if !ids.insert(id.to_owned()) {
        errors.push(format!("{path}.id {id:?} is duplicated"));
    }
}

fn validate_confidence(value: Option<f32>, path: &str, errors: &mut Vec<String>) {
    if let Some(value) = value
        && (!value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        errors.push(format!("{path}.confidence must be between 0 and 1"));
    }
}

fn validate_non_empty_unique(values: &[String], path: &str, errors: &mut Vec<String>) {
    let mut seen = HashSet::new();
    for (index, value) in values.iter().enumerate() {
        if value.trim().is_empty() {
            errors.push(format!("{path}[{index}] must not be empty"));
        }
        if !seen.insert(value) {
            errors.push(format!("{path}[{index}] is duplicated"));
        }
    }
}

fn validate_quad(quad: &Quad, width: u32, height: u32, path: &str, errors: &mut Vec<String>) {
    for (index, point) in quad.points.iter().enumerate() {
        if !point.x.is_finite() || !point.y.is_finite() {
            errors.push(format!("{path}.points[{index}] must be finite"));
            continue;
        }
        if point.x < 0.0 || point.x > width as f32 {
            errors.push(format!("{path}.points[{index}].x is outside the image"));
        }
        if point.y < 0.0 || point.y > height as f32 {
            errors.push(format!("{path}.points[{index}].y is outside the image"));
        }
    }

    if quad.signed_area().abs() < 0.5 {
        errors.push(format!("{path} has no area"));
    }
}
