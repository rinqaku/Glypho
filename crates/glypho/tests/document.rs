use glypho::{
    AnnotationStatus, Document, EvaluationPolicy, ImageInfo, Legibility, Quad, RegionSource,
    TextAlternative, TextDirection, TextLine,
};

fn document() -> Document {
    let mut document = Document::new(ImageInfo {
        id: "sample".to_owned(),
        file_name: "sample.png".to_owned(),
        width: 200,
        height: 100,
        sha256: None,
    });
    document.lines = vec![
        TextLine {
            id: "second".to_owned(),
            order: 1,
            quad: Quad::from_rect(10.0, 50.0, 100.0, 20.0),
            text: "Second".to_owned(),
            corrected_text: None,
            alternatives: Vec::new(),
            confidence: None,
            language: Some("en".to_owned()),
            script: Some("Latn".to_owned()),
            direction: TextDirection::LeftToRight,
            legibility: Legibility::Clear,
            flags: Vec::new(),
            evaluation: EvaluationPolicy::default(),
            source: RegionSource::Manual,
            words: Vec::new(),
            ignored: false,
        },
        TextLine {
            id: "first".to_owned(),
            order: 0,
            quad: Quad::from_rect(10.0, 10.0, 80.0, 20.0),
            text: "First".to_owned(),
            corrected_text: None,
            alternatives: Vec::new(),
            confidence: None,
            language: Some("en".to_owned()),
            script: Some("Latn".to_owned()),
            direction: TextDirection::LeftToRight,
            legibility: Legibility::Clear,
            flags: Vec::new(),
            evaluation: EvaluationPolicy::default(),
            source: RegionSource::Manual,
            words: Vec::new(),
            ignored: false,
        },
    ];
    document
}

#[test]
fn rebuilds_text_in_reading_order() {
    let mut document = document();
    document.rebuild_text();

    assert_eq!(document.text, "First\nSecond");
    document.validate().expect("document should be valid");
}

#[test]
fn rejects_empty_reviewed_lines() {
    let mut document = document();
    document.metadata.status = AnnotationStatus::Reviewed;
    document.lines[0].text.clear();
    document.rebuild_text();

    let error = document.validate().expect_err("document should be invalid");
    assert!(error.to_string().contains("must not be empty"));
}

#[test]
fn sorts_lines_by_geometry() {
    let mut document = document();
    document.lines[0].order = 0;
    document.lines[1].order = 1;
    document.sort_reading_order();

    assert_eq!(document.lines[0].id, "first");
    assert_eq!(document.lines[1].id, "second");
}

#[test]
fn rejects_schema_level_constraints() {
    let mut document = document();
    document.image.sha256 = Some("ABC".to_owned());
    document.language_hints = vec!["en".to_owned(), "en".to_owned()];
    document.lines[0].alternatives.push(TextAlternative {
        text: "Second?".to_owned(),
        confidence: 1.5,
    });

    let error = document.validate().expect_err("document should be invalid");
    let message = error.to_string();
    assert!(message.contains("image.sha256"));
    assert!(message.contains("language_hints"));
    assert!(message.contains("alternatives"));
}

#[test]
fn rejects_unknown_evaluation_fields_during_deserialization() {
    let value = serde_json::json!({
        "detection": true,
        "recognition": true,
        "unknown": true,
    });

    let error =
        serde_json::from_value::<EvaluationPolicy>(value).expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
}
