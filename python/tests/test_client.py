import base64
import json
from pathlib import Path

import pytest

from glypho import Glypho, RecognitionOptions


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAUAAAABAAQAAAABDnJOzAAABR0lEQVR42u2VMU7DQBBF39iW7AIpLikQ2iNwg/hYKRBscjC06TgExUaiCJ0jUbiwPRSx4+wqEYgKIW81s/975s/fkSzKz07CTJyJf4oo1c4KshU2ZOwMKlKI2K3ZipwRO17BIlABmPcRqvZhxU+A6jkBDyj9gHRx6xqAF6AGevSaRr+0YMhoANrcWenzJtXHhVu66asM55oPyreCrAGaW+MGqHwIKwoFsB+dMuUJKy74WFB+6+PmqMAc87oq/ADV116mRyd/O+EQEw8ZJFS0gaapSVBRsKF2SNhfW4pwHqG9RFwDRiZ/U8X3EXExxi7sfR8peRpDBShdM84g9peLG22LH31uY2JVnedF7Ye8s0Fr1JN6Fmt1qmtNW1Dtc4UVqTo9nSRyL0HO3iYAboILORsvWrOUVTDM3RDkNlhcmf8zM3Em/jviF/NMZs4TGkYWAAAAAElFTkSuQmCC'
)


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / 'glypho-test.png'
    path.write_bytes(SAMPLE_PNG)
    return path


def local_runtime() -> dict[str, object]:
    models = ROOT / 'models' / 'installed'
    if models.is_dir():
        return {'models': models, 'offline': True}
    return {}


def test_recognizes_synthetic_image(sample_image: Path):
    engine = Glypho(languages=('en',), **local_runtime())
    document = engine.recognize(
        sample_image,
        segmentation='single_line',
    )

    assert document.image.width == 320
    assert 'GLYPHO' in document.text.upper()
    assert 'TEST' in document.text.upper()
    assert document.lines


def test_rejects_non_finite_timeout(sample_image: Path):
    engine = Glypho()

    with pytest.raises(ValueError, match='timeout'):
        engine.recognize(
            sample_image,
            timeout=float('inf'),
        )


def test_accepts_maximum_quality_profile():
    payload = RecognitionOptions(quality='maximum').to_json()

    assert b'"quality": "maximum"' in payload


def test_defaults_to_automatic_language_routing():
    options = json.loads(RecognitionOptions().to_json())

    assert options['languages'] == []


def test_warmup_uses_persistent_native_runtime(monkeypatch):
    calls = []

    class FakeNative:
        def __init__(self, _path=None):
            pass

        def warmup(self, options):
            calls.append(json.loads(options))

    monkeypatch.setattr('glypho.client.NativeLibrary', FakeNative)
    engine = Glypho(models='models/installed', quality='accurate', threads=2)

    assert engine.runtime == 'native'
    assert engine.warmup(languages=['de-DE', 'zh-Hans']) is None
    assert calls == [
        {
            'backend': 'onnx',
            'languages': ['de-DE', 'zh-Hans'],
            'models': 'models/installed',
            'quality': 'accurate',
            'device': 'auto',
            'offline': False,
            'segmentation': 'sparse_text',
            'min_confidence': 0.8,
            'tesseract': 'tesseract',
            'threads': 2,
            'timeout_ms': 30_000,
        }
    ]
