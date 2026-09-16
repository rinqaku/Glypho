use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct Artifact {
    pub(super) directory: &'static str,
    file: &'static str,
    sha256: &'static str,
}

impl Artifact {
    pub(super) fn path(self, root: &Path) -> PathBuf {
        root.join(self.directory).join(self.file)
    }

    pub(super) fn verify(self, root: &Path) -> Result<()> {
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

    pub(super) fn ensure(self, root: &Path, auto_download: bool) -> Result<()> {
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
pub(super) struct RecognizerArtifacts {
    pub(super) model: Artifact,
    pub(super) dictionary: Artifact,
}

impl RecognizerArtifacts {
    pub(super) fn ensure(self, root: &Path, auto_download: bool) -> Result<()> {
        self.model.ensure(root, auto_download)?;
        self.dictionary.ensure(root, auto_download)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ModelProfile {
    pub(super) name: &'static str,
    pub(super) detector: Artifact,
    pub(super) recognizer: RecognizerArtifacts,
    pub(super) detector_threshold: f32,
    pub(super) box_threshold: f32,
    pub(super) unclip_ratio: f32,
    pub(super) region_batch_size: usize,
    pub(super) max_side_len: u32,
}

impl ModelProfile {
    pub(super) fn for_quality(quality: QualityMode) -> Self {
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
                name: "PP-OCRv6 Small + routed PP-OCRv5 language packs",
                detector: V6_SMALL_DETECTOR,
                recognizer: V6_SMALL_RECOGNIZER,
                detector_threshold: 0.2,
                box_threshold: 0.45,
                unclip_ratio: 1.4,
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

pub(super) const V6_TINY_DETECTOR: Artifact = Artifact {
    directory: "v6-tiny-det",
    file: "inference.onnx",
    sha256: "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8",
};
pub(super) const V6_TINY_RECOGNIZER: RecognizerArtifacts = RecognizerArtifacts {
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
pub(super) const V6_SMALL_DETECTOR: Artifact = Artifact {
    directory: "v6-small-det",
    file: "inference.onnx",
    sha256: "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e",
};
pub(super) const V6_SMALL_RECOGNIZER: RecognizerArtifacts = RecognizerArtifacts {
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
pub(super) const V6_MEDIUM_DETECTOR: Artifact = Artifact {
    directory: "v6-medium-det",
    file: "inference.onnx",
    sha256: "eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1",
};
pub(super) const V6_MEDIUM_RECOGNIZER: RecognizerArtifacts = RecognizerArtifacts {
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
pub(super) const ESLAV_RECOGNIZER: RecognizerArtifacts = RecognizerArtifacts {
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
pub(super) const LATIN_RECOGNIZER: RecognizerArtifacts = RecognizerArtifacts {
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
pub(super) const KOREAN_RECOGNIZER: RecognizerArtifacts = RecognizerArtifacts {
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
pub(super) struct DownloadSpec {
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

pub(super) fn install_model(directory: &str, root: &Path) -> Result<()> {
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
                    || hex_digest(&Sha256::digest(&dictionary)) != sha256
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

pub(super) fn download_if_needed(
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
    // Stream into a private temporary file and expose it only after full verification.
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
        let actual_sha256 = hex_digest(&digest.finalize());
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

pub(super) fn atomic_model_write(target: &Path, bytes: &[u8]) -> Result<()> {
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

pub(super) fn extract_dictionary(config_path: &Path, expected_entries: usize) -> Result<Vec<u8>> {
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

pub(super) fn reject_symlink(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path).map_err(|error| Error::io(path, error))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(backend_error(format!(
            "model cache path is not a real directory: {}",
            path.display()
        )));
    }
    Ok(())
}

pub(super) fn file_matches(path: &Path, expected_bytes: u64, expected_sha256: &str) -> bool {
    fs::symlink_metadata(path).is_ok_and(|metadata| {
        metadata.is_file()
            && !metadata.file_type().is_symlink()
            && metadata.len() == expected_bytes
            && file_sha256(path).is_ok_and(|sha256| sha256 == expected_sha256)
    })
}

pub(super) fn download_spec(directory: &str) -> Option<DownloadSpec> {
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

pub(super) fn validate_config(config: &OnnxConfig) -> Result<()> {
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

pub(super) fn validate_min_confidence(value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Error::InvalidOption(
            "min_confidence must be between 0 and 1".to_owned(),
        ));
    }
    Ok(())
}

pub(super) fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|threads| recommended_threads(threads.get()))
        .unwrap_or(1)
}

pub(super) fn recommended_threads(parallelism: usize) -> usize {
    parallelism.clamp(1, 8)
}

pub(super) fn file_sha256(path: &Path) -> Result<String> {
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
    Ok(hex_digest(&digest.finalize()))
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().fold(
        String::with_capacity(bytes.len() * 2),
        |mut output, byte| {
            use std::fmt::Write;
            let _ = write!(output, "{byte:02x}");
            output
        },
    )
}

pub(super) fn resolve_languages(requested: &[String]) -> Result<Vec<String>> {
    if requested.is_empty()
        || (requested.len() == 1 && requested[0].trim().eq_ignore_ascii_case("auto"))
    {
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

pub(super) fn validate_profile_languages(quality: QualityMode, languages: &[String]) -> Result<()> {
    if quality == QualityMode::Fast && languages.iter().any(|language| language == "ja") {
        return Err(Error::UnsupportedLanguage {
            requested: vec!["ja".to_owned()],
            available: profile_languages(quality),
        });
    }
    Ok(())
}

pub(super) fn profile_languages(quality: QualityMode) -> Vec<String> {
    SUPPORTED_LANGUAGES
        .iter()
        .filter(|language| quality != QualityMode::Fast || **language != "ja")
        .map(|language| (*language).to_owned())
        .collect()
}

pub(super) fn normalize_language(language: &str) -> String {
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
