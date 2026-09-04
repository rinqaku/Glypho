use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Evaluation {
    pub character_error_rate: ErrorRate,
    pub word_error_rate: ErrorRate,
    pub exact_match: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ErrorRate {
    pub edits: usize,
    pub reference_units: usize,
    pub rate: f64,
}

pub fn evaluate(reference: &str, prediction: &str) -> Evaluation {
    let reference_chars = reference.chars().collect::<Vec<_>>();
    let prediction_chars = prediction.chars().collect::<Vec<_>>();
    let reference_words = reference.split_whitespace().collect::<Vec<_>>();
    let prediction_words = prediction.split_whitespace().collect::<Vec<_>>();

    Evaluation {
        character_error_rate: error_rate(&reference_chars, &prediction_chars),
        word_error_rate: error_rate(&reference_words, &prediction_words),
        exact_match: reference == prediction,
    }
}

fn error_rate<T: Eq>(reference: &[T], prediction: &[T]) -> ErrorRate {
    let edits = edit_distance(reference, prediction);
    let rate = if reference.is_empty() {
        if prediction.is_empty() { 0.0 } else { 1.0 }
    } else {
        edits as f64 / reference.len() as f64
    };

    ErrorRate {
        edits,
        reference_units: reference.len(),
        rate,
    }
}

fn edit_distance<T: Eq>(left: &[T], right: &[T]) -> usize {
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }

    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    let mut current = vec![0; right.len() + 1];

    for (left_index, left_item) in left.iter().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_item) in right.iter().enumerate() {
            let substitution = previous[right_index] + usize::from(left_item != right_item);
            let insertion = current[right_index] + 1;
            let deletion = previous[right_index + 1] + 1;
            current[right_index + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_unicode_by_codepoint() {
        let result = evaluate("Привет мир", "Привет, мир");

        assert_eq!(result.character_error_rate.edits, 1);
        assert_eq!(result.word_error_rate.edits, 1);
        assert!(!result.exact_match);
    }

    #[test]
    fn handles_empty_reference() {
        assert_eq!(evaluate("", "").character_error_rate.rate, 0.0);
        assert_eq!(evaluate("", "text").character_error_rate.rate, 1.0);
    }
}
