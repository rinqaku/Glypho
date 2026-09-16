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
    assert!(
        resolve_languages(&["AUTO".to_owned()])
            .expect("the explicit auto alias must resolve")
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
    assert_eq!(script_fraction("Нello", SpecialistScript::Cyrillic), 0.2);
    assert_eq!(script_fraction("Привет", SpecialistScript::Cyrillic), 1.0);
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
fn balanced_profile_limits_small_models_to_1280_pixels() {
    let profile = ModelProfile::for_quality(QualityMode::Balanced);

    assert_eq!(profile.detector.directory, "v6-small-det");
    assert_eq!(profile.recognizer.model.directory, "v6-small-rec");
    assert_eq!(profile.max_side_len, 1_280);
}

#[test]
fn default_thread_policy_scales_without_oversubscribing_small_devices() {
    assert_eq!(recommended_threads(1), 1);
    assert_eq!(recommended_threads(4), 4);
    assert_eq!(recommended_threads(16), 8);
}

#[test]
fn auto_routing_limits_confident_crops_to_their_script() {
    let primary = [Some(RecognizedCandidate {
        text: "Hello".to_owned(),
        confidence: 0.995,
        ..RecognizedCandidate::default()
    })];

    assert_eq!(
        auto_specialist_crop_indices(Some(&primary), 1, SpecialistScript::Latin),
        vec![0]
    );
    assert!(auto_specialist_crop_indices(Some(&primary), 1, SpecialistScript::Cyrillic).is_empty());
    assert!(auto_specialist_crop_indices(Some(&primary), 1, SpecialistScript::Korean).is_empty());
}

#[test]
fn auto_routing_sends_uncertain_crops_to_every_specialist() {
    let primary = [Some(RecognizedCandidate {
        text: "Hello".to_owned(),
        confidence: 0.98,
        ..RecognizedCandidate::default()
    })];

    for script in [
        SpecialistScript::Latin,
        SpecialistScript::Cyrillic,
        SpecialistScript::Korean,
    ] {
        assert_eq!(
            auto_specialist_crop_indices(Some(&primary), 1, script),
            vec![0]
        );
    }
}

#[test]
fn candidate_selection_is_order_independent_and_keeps_valid_primary() {
    let box_ = BoundingBox::from_coords(0.0, 0.0, 100.0, 20.0);
    let primary = Some(vec![Some(RecognizedCandidate {
        text: "Hello".to_owned(),
        confidence: 0.95,
        ..RecognizedCandidate::default()
    })]);
    let cyrillic = vec![Some(RecognizedCandidate {
        text: "Нello".to_owned(),
        confidence: 0.78,
        ..RecognizedCandidate::default()
    })];
    let korean = vec![Some(RecognizedCandidate {
        text: "Hello".to_owned(),
        confidence: 0.81,
        ..RecognizedCandidate::default()
    })];
    let first = merge_recognized_regions(
        std::slice::from_ref(&box_),
        primary.clone(),
        &[
            (SpecialistScript::Cyrillic, cyrillic.clone()),
            (SpecialistScript::Korean, korean.clone()),
        ],
        true,
        0.8,
    );
    let second = merge_recognized_regions(
        &[box_],
        primary,
        &[
            (SpecialistScript::Korean, korean),
            (SpecialistScript::Cyrillic, cyrillic),
        ],
        true,
        0.8,
    );

    assert_eq!(first.len(), 1);
    assert_eq!(second.len(), 1);
    assert_eq!(first[0].region.text_with_confidence().unwrap().0, "Hello");
    assert_eq!(second[0].region.text_with_confidence().unwrap().0, "Hello");
}

#[test]
fn candidate_selection_filters_before_ranking() {
    let regions = merge_recognized_regions(
        &[BoundingBox::from_coords(0.0, 0.0, 100.0, 20.0)],
        Some(vec![Some(RecognizedCandidate {
            text: "strong".to_owned(),
            confidence: 0.91,
            ..RecognizedCandidate::default()
        })]),
        &[(
            SpecialistScript::Cyrillic,
            vec![Some(RecognizedCandidate {
                text: "слабый".to_owned(),
                confidence: 0.79,
                ..RecognizedCandidate::default()
            })],
        )],
        true,
        0.8,
    );

    assert_eq!(
        regions[0].region.text_with_confidence().unwrap().0,
        "strong"
    );
}

#[test]
fn adaptive_tiles_cover_long_images_with_overlap() {
    assert!(!should_tile_image(1_280, 1_920, 1_280));
    assert!(should_tile_image(1_080, 2_400, 1_280));
    assert_eq!(tile_offsets(2_400, 1_280), vec![0, 1_120]);
    assert_eq!(tile_offsets(4_000, 1_280), vec![0, 906, 1_813, 2_720]);
}

#[test]
fn tiled_detection_deduplicates_overlap_without_merging_lines() {
    let boxes = deduplicate_boxes(vec![
        BoundingBox::from_coords(10.0, 10.0, 100.0, 30.0),
        BoundingBox::from_coords(11.0, 10.0, 101.0, 30.0),
        BoundingBox::from_coords(10.0, 35.0, 100.0, 55.0),
    ]);

    assert_eq!(boxes.len(), 2);
}

#[test]
fn failed_initialization_can_be_retried() {
    let cache = OnceLock::new();
    let initialization = Mutex::new(());
    let first = retryable_init(&cache, &initialization, || -> Result<u32> {
        Err(backend_error("temporary failure"))
    });
    assert!(first.is_err());

    let value = retryable_init(&cache, &initialization, || Ok(42))
        .expect("a later initialization attempt must succeed");

    assert_eq!(*value, 42);
}

#[test]
fn recognition_positions_produce_word_boxes() {
    let line_box = BoundingBox::new(vec![
        OarPoint::new(10.0, 20.0),
        OarPoint::new(210.0, 40.0),
        OarPoint::new(205.0, 80.0),
        OarPoint::new(5.0, 60.0),
    ]);
    let boxes = candidate_word_boxes(
        &line_box,
        &RecognizedCandidate {
            text: "two words".to_owned(),
            char_columns: (0..9).collect(),
            sequence_length: 10,
            crop_ratio: 4.0,
            batch_max_ratio: 4.0,
            ..RecognizedCandidate::default()
        },
    );

    assert_eq!(boxes.len(), 2);
    assert!(box_area(&boxes[0]) > 0.0);
    assert!(box_area(&boxes[1]) > 0.0);
    assert!(box_bounds(&boxes[0]).unwrap().2 <= box_bounds(&boxes[1]).unwrap().0);
    assert!(boxes[0].points[0].y < boxes[0].points[1].y);
}

#[test]
fn recognition_positions_drop_words_collapsed_by_batch_padding() {
    let line_box = BoundingBox::from_coords(0.0, 0.0, 100.0, 20.0);
    let boxes = candidate_word_boxes(
        &line_box,
        &RecognizedCandidate {
            text: "a b".to_owned(),
            char_columns: vec![0, 9, 10],
            sequence_length: 2,
            crop_ratio: 1.0,
            batch_max_ratio: 1.0,
            ..RecognizedCandidate::default()
        },
    );

    assert_eq!(boxes.len(), 1);
    assert!(box_area(&boxes[0]) > 0.0);
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
