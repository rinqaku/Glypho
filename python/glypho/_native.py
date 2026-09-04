from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
from pathlib import Path

from .errors import GlyphoNotFoundError, RecognitionError


class _Buffer(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_uint8)),
        ('len', ctypes.c_size_t),
        ('capacity', ctypes.c_size_t),
    ]


class _Result(ctypes.Structure):
    _fields_ = [('status', ctypes.c_int32), ('body', _Buffer)]


class NativeLibrary:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        library_path = _find_library(path)
        self._library = ctypes.CDLL(str(library_path))
        self._library.glypho_recognize_json.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self._library.glypho_recognize_json.restype = _Result
        self._warmup_function = getattr(self._library, 'glypho_warmup_json', None)
        if self._warmup_function is not None:
            self._warmup_function.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
            ]
            self._warmup_function.restype = _Result
        self._info_function = getattr(self._library, 'glypho_info_json', None)
        if self._info_function is not None:
            self._info_function.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
            ]
            self._info_function.restype = _Result
        self._library.glypho_buffer_free.argtypes = [_Buffer]
        self._library.glypho_buffer_free.restype = None

    def recognize(self, path: Path, options_json: bytes) -> bytes:
        path_bytes = os.fsencode(path)
        path_buffer = _bytes_buffer(path_bytes)
        options_buffer = _bytes_buffer(options_json)
        result = self._library.glypho_recognize_json(
            path_buffer,
            len(path_bytes),
            options_buffer,
            len(options_json),
        )
        return self._read_result(result)

    def warmup(self, options_json: bytes) -> None:
        if self._warmup_function is None:
            raise GlyphoNotFoundError(
                'the Glypho native library is older than the Python package; rebuild it'
            )
        options_buffer = _bytes_buffer(options_json)
        result = self._warmup_function(options_buffer, len(options_json))
        self._read_result(result)

    def info(self, options_json: bytes) -> bytes:
        if self._info_function is None:
            raise GlyphoNotFoundError(
                'the Glypho native library is older than the Python package; reinstall it'
            )
        options_buffer = _bytes_buffer(options_json)
        result = self._info_function(options_buffer, len(options_json))
        return self._read_result(result)

    def _read_result(self, result: _Result) -> bytes:
        try:
            body = ctypes.string_at(result.body.data, result.body.len)
        finally:
            self._library.glypho_buffer_free(result.body)

        if result.status != 0:
            raise RecognitionError(body.decode('utf-8', errors='replace'))
        return body


def _find_library(path: str | os.PathLike[str] | None) -> Path:
    if path:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return candidate
        raise GlyphoNotFoundError(f'Glypho native library not found: {candidate}')

    configured = os.environ.get('GLYPHO_LIBRARY')
    if configured:
        return _find_library(configured)

    name = _library_name()
    bundled = Path(__file__).resolve().parent / '_libs' / name
    if bundled.is_file():
        return bundled

    root = Path(__file__).resolve().parents[2]
    for profile in ('release', 'debug'):
        candidate = root / 'target' / profile / name
        if candidate.is_file():
            return candidate

    system_library = ctypes.util.find_library('glypho')
    if system_library:
        return Path(system_library)

    raise GlyphoNotFoundError(
        'Glypho native library was not found; reinstall `glypho-ocr` '
        'or set GLYPHO_LIBRARY'
    )


def _library_name() -> str:
    system = platform.system()
    if system == 'Darwin':
        return 'libglypho.dylib'
    if system == 'Windows':
        return 'glypho.dll'
    return 'libglypho.so'


def _bytes_buffer(value: bytes) -> ctypes.Array[ctypes.c_uint8]:
    if not value:
        return (ctypes.c_uint8 * 0)()
    return (ctypes.c_uint8 * len(value)).from_buffer_copy(value)
