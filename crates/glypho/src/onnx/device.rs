use super::*;

#[derive(Clone, Debug)]
pub(super) struct DeviceResolution {
    pub(super) resolved: Device,
    pub(super) fallback_reason: Option<String>,
}

pub(super) fn resolve_device(requested: Device) -> DeviceResolution {
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

pub(super) fn probe_device(device: Device) -> std::result::Result<(), String> {
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
pub(super) fn probe_provider(
    provider: impl ExecutionProvider,
    name: &str,
) -> std::result::Result<(), String> {
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
