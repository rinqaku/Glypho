use super::*;

const AUTO_SPECIALIST_THRESHOLD: f32 = 0.99;

#[derive(Clone, Debug)]
pub(super) struct MergedRegion {
    pub(super) region: OarTextRegion,
    pub(super) alternatives: Vec<TextAlternative>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct RecognizedCandidate {
    pub(super) text: String,
    pub(super) confidence: f32,
    pub(super) char_positions: Vec<f32>,
    pub(super) char_columns: Vec<usize>,
    pub(super) sequence_length: usize,
    pub(super) crop_ratio: f32,
    pub(super) batch_max_ratio: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SpecialistScript {
    Cyrillic,
    Latin,
    Korean,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RecognizerPlan {
    pub(super) primary: bool,
    pub(super) latin: bool,
    pub(super) cyrillic: bool,
    pub(super) korean: bool,
}

pub(super) fn recognizer_plan(languages: &[String]) -> RecognizerPlan {
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

pub(super) fn specialist_crop_indices(
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

pub(super) fn auto_specialist_crop_indices(
    primary: Option<&[Option<RecognizedCandidate>]>,
    crop_count: usize,
    script: SpecialistScript,
) -> Vec<usize> {
    let Some(primary) = primary else {
        return (0..crop_count).collect();
    };
    (0..crop_count)
        .filter(|index| {
            // Confident crops challenge only the matching script; uncertain ones try every specialist.
            primary.get(*index).is_none_or(|candidate| {
                candidate.as_ref().is_none_or(|candidate| {
                    candidate.confidence < AUTO_SPECIALIST_THRESHOLD
                        || contains_script(&candidate.text, script)
                })
            })
        })
        .collect()
}

pub(super) fn merge_recognized_regions(
    boxes: &[BoundingBox],
    primary: Option<Vec<Option<RecognizedCandidate>>>,
    specialists: &[(SpecialistScript, Vec<Option<RecognizedCandidate>>)],
    auto_languages: bool,
    min_confidence: f32,
) -> Vec<MergedRegion> {
    boxes
        .iter()
        .enumerate()
        .filter_map(|(index, box_)| {
            let mut candidates = Vec::new();
            if let Some(candidate) = candidate_at(primary.as_deref(), index) {
                candidates.push(RoutedCandidate {
                    candidate,
                    specialist: None,
                });
            }
            for (script, specialist_candidates) in specialists {
                if let Some(candidate) = candidate_at(Some(specialist_candidates), index) {
                    candidates.push(RoutedCandidate {
                        candidate,
                        specialist: Some(*script),
                    });
                }
            }
            select_region(box_, candidates, auto_languages, min_confidence)
        })
        .collect()
}

pub(super) fn candidate_at(
    candidates: Option<&[Option<RecognizedCandidate>]>,
    index: usize,
) -> Option<RecognizedCandidate> {
    candidates?.get(index)?.clone()
}

#[derive(Clone, Debug)]
pub(super) struct RoutedCandidate {
    candidate: RecognizedCandidate,
    specialist: Option<SpecialistScript>,
}

pub(super) fn recognized_region(
    box_: &BoundingBox,
    candidate: RecognizedCandidate,
    alternatives: Vec<TextAlternative>,
) -> MergedRegion {
    let word_boxes = candidate_word_boxes(box_, &candidate);
    let mut region = OarTextRegion::with_recognition(
        box_.clone(),
        Some(Arc::<str>::from(candidate.text)),
        Some(candidate.confidence),
    );
    region.dt_poly = Some(box_.clone());
    region.rec_poly = Some(box_.clone());
    region.word_boxes = (!word_boxes.is_empty()).then_some(word_boxes);
    MergedRegion {
        region,
        alternatives,
    }
}

pub(super) fn select_region(
    box_: &BoundingBox,
    candidates: Vec<RoutedCandidate>,
    auto_languages: bool,
    min_confidence: f32,
) -> Option<MergedRegion> {
    let mut eligible = candidates
        .into_iter()
        .filter(|candidate| candidate.candidate.confidence >= min_confidence)
        .collect::<Vec<_>>();
    eligible.sort_by(|left, right| {
        candidate_rank(right, auto_languages)
            .total_cmp(&candidate_rank(left, auto_languages))
            .then_with(|| {
                right
                    .candidate
                    .confidence
                    .total_cmp(&left.candidate.confidence)
            })
            .then_with(|| candidate_priority(left).cmp(&candidate_priority(right)))
            .then_with(|| left.candidate.text.cmp(&right.candidate.text))
    });
    let selected = eligible.first()?.candidate.clone();
    let mut alternatives = Vec::new();
    for candidate in eligible.into_iter().skip(1) {
        if candidate.candidate.text != selected.text
            && !alternatives
                .iter()
                .any(|alternative: &TextAlternative| alternative.text == candidate.candidate.text)
        {
            alternatives.push(TextAlternative {
                text: candidate.candidate.text,
                confidence: candidate.candidate.confidence.clamp(0.0, 1.0),
            });
        }
    }
    Some(recognized_region(box_, selected, alternatives))
}

pub(super) fn candidate_rank(candidate: &RoutedCandidate, _auto_languages: bool) -> f32 {
    let confidence = candidate.candidate.confidence;
    let Some(script) = candidate.specialist else {
        return confidence;
    };
    let script_fraction = script_fraction(&candidate.candidate.text, script);
    // Prefer script agreement without letting the routing bonus override real confidence gaps.
    if script_fraction > 0.0 {
        confidence + 0.004 * script_fraction * script_fraction
    } else {
        confidence - 0.08
    }
}

pub(super) fn script_fraction(text: &str, expected: SpecialistScript) -> f32 {
    let mut expected_count = 0_u32;
    let mut alphabetic_count = 0_u32;
    for character in text.chars().filter(|character| character.is_alphabetic()) {
        alphabetic_count += 1;
        let matches = match expected {
            SpecialistScript::Cyrillic => ('\u{0400}'..='\u{052f}').contains(&character),
            SpecialistScript::Latin => {
                character.is_ascii_alphabetic()
                    || ('\u{00c0}'..='\u{024f}').contains(&character)
                    || ('\u{1e00}'..='\u{1eff}').contains(&character)
            }
            SpecialistScript::Korean => {
                ('\u{1100}'..='\u{11ff}').contains(&character)
                    || ('\u{3130}'..='\u{318f}').contains(&character)
                    || ('\u{ac00}'..='\u{d7af}').contains(&character)
            }
        };
        expected_count += u32::from(matches);
    }
    if alphabetic_count == 0 {
        0.0
    } else {
        expected_count as f32 / alphabetic_count as f32
    }
}

pub(super) fn candidate_priority(candidate: &RoutedCandidate) -> u8 {
    match candidate.specialist {
        None => 0,
        Some(SpecialistScript::Latin) => 1,
        Some(SpecialistScript::Cyrillic) => 2,
        Some(SpecialistScript::Korean) => 3,
    }
}

pub(super) fn contains_script(text: &str, script: SpecialistScript) -> bool {
    match script {
        SpecialistScript::Cyrillic => contains_cyrillic(text),
        SpecialistScript::Latin => contains_latin(text),
        SpecialistScript::Korean => contains_korean(text),
    }
}

pub(super) fn contains_cyrillic(text: &str) -> bool {
    text.chars()
        .any(|character| ('\u{0400}'..='\u{052f}').contains(&character))
}

pub(super) fn contains_latin(text: &str) -> bool {
    text.chars().any(|character| {
        character.is_ascii_alphabetic()
            || ('\u{00c0}'..='\u{024f}').contains(&character)
            || ('\u{1e00}'..='\u{1eff}').contains(&character)
    })
}

pub(super) fn contains_korean(text: &str) -> bool {
    text.chars().any(|character| {
        ('\u{1100}'..='\u{11ff}').contains(&character)
            || ('\u{3130}'..='\u{318f}').contains(&character)
            || ('\u{ac00}'..='\u{d7af}').contains(&character)
    })
}

pub(super) fn contains_kana(text: &str) -> bool {
    text.chars()
        .any(|character| ('\u{3040}'..='\u{30ff}').contains(&character))
}

pub(super) fn contains_han(text: &str) -> bool {
    text.chars()
        .any(|character| ('\u{3400}'..='\u{9fff}').contains(&character))
}

pub(super) fn script_tag(text: &str) -> Option<&'static str> {
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

pub(super) fn line_language(text: &str, languages: &[String]) -> Option<String> {
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
