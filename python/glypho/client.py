from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from ._native import NativeLibrary
from .errors import GlyphoNotFoundError, RecognitionError
from .models import Document


Segmentation = Literal['auto', 'single_block', 'single_line', 'sparse_text']
Backend = Literal['auto', 'onnx', 'tesseract']
Quality = Literal['fast', 'balanced', 'accurate', 'maximum']
Device = Literal['auto', 'cpu', 'cuda', 'coreml', 'openvino']


@dataclass(frozen=True, slots=True)
class RecognitionOptions:
    backend: Backend = 'auto'
    languages: tuple[str, ...] = ()
    models: str | None = None
    quality: Quality = 'balanced'
    device: Device = 'auto'
    offline: bool = False
    segmentation: Segmentation = 'sparse_text'
    min_confidence: float = 0.8
    tesseract: str = 'tesseract'
    threads: int | None = None
    timeout_ms: int = 30_000

    def to_json(self) -> bytes:
        if self.backend not in {'auto', 'onnx', 'tesseract'}:
            raise ValueError(f'unsupported OCR backend: {self.backend}')
        if self.quality not in {'fast', 'balanced', 'accurate', 'maximum'}:
            raise ValueError(f'unsupported quality mode: {self.quality}')
        if self.device not in {'auto', 'cpu', 'cuda', 'coreml', 'openvino'}:
            raise ValueError(f'unsupported device: {self.device}')
        if self.segmentation not in {'auto', 'single_block', 'single_line', 'sparse_text'}:
            raise ValueError(f'unsupported segmentation mode: {self.segmentation}')
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError('min_confidence must be between 0 and 1')
        if not 1 <= self.timeout_ms <= 300_000:
            raise ValueError('timeout_ms must be between 1 and 300000')
        if self.threads is not None and not 1 <= self.threads <= 64:
            raise ValueError('threads must be between 1 and 64')
        return json.dumps(asdict(self), ensure_ascii=False).encode('utf-8')


@dataclass(slots=True)
class Glypho:
    library: str | os.PathLike[str] | None = None
    binary: str | os.PathLike[str] | None = None
    backend: Backend = 'auto'
    models: str | os.PathLike[str] | None = None
    quality: Quality = 'balanced'
    device: Device = 'auto'
    languages: tuple[str, ...] = ()
    offline: bool = False
    threads: int | None = None
    _native: NativeLibrary | None = field(init=False, default=None, repr=False)
    _binary: Path | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.binary:
            self._binary = _find_binary(self.binary)
            return

        try:
            self._native = NativeLibrary(self.library)
        except GlyphoNotFoundError:
            if self.library:
                raise
            self._binary = _find_binary(None)

    @property
    def runtime(self) -> Literal['native', 'cli']:
        return 'native' if self._native is not None else 'cli'

    def warmup(
        self,
        sample: str | os.PathLike[str] | None = None,
        *,
        languages: tuple[str, ...] | list[str] | None = None,
        models: str | os.PathLike[str] | None = None,
        quality: Quality | None = None,
        threads: int | None = None,
        min_confidence: float = 0.8,
    ) -> Document | None:
        if self._native is None:
            raise GlyphoNotFoundError(
                'warmup requires the native library; a CLI subprocess cannot retain models'
            )
        selected_models = models or self.models
        selected_threads = threads if threads is not None else self.threads
        options = RecognitionOptions(
            backend='onnx',
            languages=tuple(self.languages if languages is None else languages),
            models=os.fspath(selected_models) if selected_models else None,
            quality=quality or self.quality,
            device=self.device,
            offline=self.offline,
            min_confidence=min_confidence,
            threads=selected_threads,
        )
        self._native.warmup(options.to_json())
        if sample is None:
            return None
        return self.recognize(
            sample,
            backend='onnx',
            languages=self.languages if languages is None else languages,
            models=selected_models,
            quality=quality,
            min_confidence=min_confidence,
            threads=selected_threads,
        )

    def recognize(
        self,
        image: str | os.PathLike[str],
        *,
        backend: Backend | None = None,
        languages: tuple[str, ...] | list[str] | None = None,
        models: str | os.PathLike[str] | None = None,
        quality: Quality | None = None,
        device: Device | None = None,
        segmentation: Segmentation = 'sparse_text',
        min_confidence: float = 0.8,
        tesseract: str = 'tesseract',
        threads: int | None = None,
        timeout: float = 30.0,
    ) -> Document:
        path = Path(image).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if not math.isfinite(timeout) or not 0.0 < timeout <= 300.0:
            raise ValueError('timeout must be between 0 and 300 seconds')

        options = RecognitionOptions(
            backend=backend or self.backend,
            languages=tuple(self.languages if languages is None else languages),
            models=os.fspath(models or self.models) if models or self.models else None,
            quality=quality or self.quality,
            device=device or self.device,
            offline=self.offline,
            segmentation=segmentation,
            min_confidence=min_confidence,
            tesseract=tesseract,
            threads=threads if threads is not None else self.threads,
            timeout_ms=max(1, round(timeout * 1000)),
        )
        if self._native:
            payload = self._native.recognize(path, options.to_json())
        else:
            payload = self._recognize_with_cli(path, options, timeout)

        try:
            return Document.from_dict(json.loads(payload))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise RecognitionError(f'Glypho returned an invalid document: {error}') from error

    def info(self) -> dict[str, object]:
        options = RecognitionOptions(
            backend=self.backend,
            languages=self.languages,
            models=os.fspath(self.models) if self.models else None,
            quality=self.quality,
            device=self.device,
            offline=self.offline,
            threads=self.threads,
        )
        if self._native is not None:
            payload = self._native.info(options.to_json())
        elif self._binary is not None:
            command = [str(self._binary), 'info', '--backend', self.backend, '--quality', self.quality]
            command.extend(['--device', self.device])
            if self.models:
                command.extend(['--models', os.fspath(self.models)])
            if self.threads is not None:
                command.extend(['--threads', str(self.threads)])
            if self.offline:
                command.append('--offline')
            result = subprocess.run(command, capture_output=True, check=False, timeout=30)
            if result.returncode != 0:
                raise RecognitionError(result.stderr.decode('utf-8', errors='replace').strip())
            payload = result.stdout
        else:
            raise GlyphoNotFoundError('Glypho runtime is not configured')
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise RecognitionError(f'Glypho returned invalid runtime information: {error}') from error
        if not isinstance(value, dict):
            raise RecognitionError('Glypho runtime information is not an object')
        return value

    def _recognize_with_cli(
        self,
        path: Path,
        options: RecognitionOptions,
        timeout: float,
    ) -> bytes:
        if not self._binary:
            raise GlyphoNotFoundError('Glypho CLI is not configured')

        command = [
            str(self._binary),
            'recognize',
            str(path),
            '--backend',
            options.backend,
            '--quality',
            options.quality,
            '--device',
            options.device,
            '--segmentation',
            options.segmentation.replace('_', '-'),
            '--min-confidence',
            str(options.min_confidence),
            '--tesseract',
            options.tesseract,
            '--timeout',
            str(max(1, round(timeout))),
        ]
        if options.models:
            command.extend(['--models', options.models])
        if options.threads is not None:
            command.extend(['--threads', str(options.threads)])
        if options.languages:
            command.extend(['--language', ','.join(options.languages)])
        if options.offline:
            command.append('--offline')

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=timeout + 1,
            )
        except subprocess.TimeoutExpired as error:
            raise RecognitionError(f'Glypho timed out after {timeout:g}s') from error
        if result.returncode != 0:
            message = result.stderr.decode('utf-8', errors='replace').strip()
            raise RecognitionError(message or f'Glypho exited with {result.returncode}')
        return result.stdout


def _find_binary(configured: str | os.PathLike[str] | None) -> Path:
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path
        raise GlyphoNotFoundError(f'Glypho CLI not found: {path}')

    packaged_name = 'glypho.exe' if os.name == 'nt' else 'glypho'
    packaged = Path(__file__).resolve().parent / '_bin' / packaged_name
    if packaged.is_file():
        return packaged

    root = Path(__file__).resolve().parents[2]
    for profile in ('release', 'debug'):
        candidate = root / 'target' / profile / 'glypho'
        if candidate.is_file():
            return candidate

    system_binary = shutil.which('glypho')
    if system_binary:
        return Path(system_binary)
    raise GlyphoNotFoundError(
        'Glypho was not found; reinstall `glypho-ocr` or pass binary=...'
    )
