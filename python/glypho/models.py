from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Point:
        _check_keys(data, {'x', 'y'}, set(), 'point')
        x = _number(data['x'], 'point.x')
        y = _number(data['y'], 'point.y')
        if x < 0 or y < 0:
            raise ValueError('point coordinates must not be negative')
        return cls(x=x, y=y)


@dataclass(frozen=True, slots=True)
class Quad:
    points: tuple[Point, Point, Point, Point]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Quad:
        _check_keys(data, {'points'}, set(), 'quad')
        if not isinstance(data['points'], list):
            raise ValueError('quad.points must be an array')
        points = tuple(Point.from_dict(point) for point in data['points'])
        if len(points) != 4:
            raise ValueError('quad must contain exactly four points')
        return cls(points=(points[0], points[1], points[2], points[3]))


@dataclass(frozen=True, slots=True)
class TextWord:
    id: str
    quad: Quad
    text: str
    confidence: float | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextWord:
        _check_keys(data, {'id', 'quad', 'text'}, {'confidence'}, 'word')
        return cls(
            id=_string(data['id'], 'word.id', non_empty=True),
            quad=Quad.from_dict(data['quad']),
            text=_string(data['text'], 'word.text'),
            confidence=_optional_float(data.get('confidence')),
        )


@dataclass(frozen=True, slots=True)
class TextAlternative:
    text: str
    confidence: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextAlternative:
        _check_keys(data, {'text', 'confidence'}, set(), 'alternative')
        confidence = _optional_float(data['confidence'])
        if confidence is None:
            raise ValueError('alternative confidence must not be null')
        return cls(text=_string(data['text'], 'alternative.text'), confidence=confidence)


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    detection: bool
    recognition: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationPolicy:
        _check_keys(data, {'detection', 'recognition'}, set(), 'evaluation')
        return cls(
            detection=_boolean(data['detection'], 'evaluation.detection'),
            recognition=_boolean(data['recognition'], 'evaluation.recognition'),
        )


@dataclass(frozen=True, slots=True)
class TextLine:
    id: str
    order: int
    quad: Quad
    text: str
    corrected_text: str | None
    alternatives: tuple[TextAlternative, ...]
    confidence: float | None
    language: str | None
    script: str | None
    source: str
    words: tuple[TextWord, ...]
    ignored: bool
    direction: str = 'auto'
    legibility: str = 'clear'
    flags: tuple[str, ...] = ()
    evaluation: EvaluationPolicy = EvaluationPolicy(True, True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextLine:
        _check_keys(
            data,
            {
                'id', 'order', 'quad', 'text', 'direction', 'legibility',
                'evaluation', 'source', 'words', 'ignored',
            },
            {
                'corrected_text', 'alternatives', 'confidence', 'language',
                'script', 'flags',
            },
            'line',
        )
        order = _integer(data['order'], 'line.order')
        if not 0 <= order <= 4_294_967_295:
            raise ValueError('line.order must be a non-negative 32-bit integer')
        source = _string(data['source'], 'line.source')
        if source not in {'manual', 'model', 'imported'}:
            raise ValueError('line.source is invalid')
        direction = _string(data['direction'], 'line.direction')
        if direction not in {'auto', 'left_to_right', 'right_to_left', 'vertical'}:
            raise ValueError('line.direction is invalid')
        legibility = _string(data['legibility'], 'line.legibility')
        if legibility not in {'clear', 'ambiguous', 'unreadable'}:
            raise ValueError('line.legibility is invalid')
        alternatives = _array(data.get('alternatives', []), 'line.alternatives')
        words = _array(data['words'], 'line.words')
        flags = _string_list(data.get('flags', []), 'line.flags', non_empty=True, unique=True)
        return cls(
            id=_string(data['id'], 'line.id', non_empty=True),
            order=order,
            quad=Quad.from_dict(data['quad']),
            text=_string(data['text'], 'line.text'),
            corrected_text=_optional_string(data.get('corrected_text'), 'line.corrected_text'),
            alternatives=tuple(
                TextAlternative.from_dict(value)
                for value in alternatives
            ),
            confidence=_optional_float(data.get('confidence')),
            language=_optional_string(data.get('language'), 'line.language', non_empty=True),
            script=_optional_string(data.get('script'), 'line.script', non_empty=True),
            source=source,
            words=tuple(TextWord.from_dict(word) for word in words),
            ignored=_boolean(data['ignored'], 'line.ignored'),
            direction=direction,
            legibility=legibility,
            flags=tuple(flags),
            evaluation=EvaluationPolicy.from_dict(data['evaluation']),
        )


@dataclass(frozen=True, slots=True)
class ImageInfo:
    id: str
    file_name: str
    width: int
    height: int
    sha256: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageInfo:
        _check_keys(
            data,
            {'id', 'file_name', 'width', 'height'},
            {'sha256'},
            'image',
        )
        image_id = _string(data['id'], 'image.id', non_empty=True)
        file_name = _string(data['file_name'], 'image.file_name', non_empty=True)
        width = _integer(data['width'], 'image.width')
        height = _integer(data['height'], 'image.height')
        if width < 1 or height < 1:
            raise ValueError('image dimensions must be greater than zero')
        sha256 = _optional_string(data.get('sha256'), 'image.sha256')
        if sha256 is not None and (len(sha256) != 64 or any(character not in '0123456789abcdef' for character in sha256)):
            raise ValueError('image.sha256 must be 64 lowercase hexadecimal characters')
        return cls(
            id=image_id,
            file_name=file_name,
            width=width,
            height=height,
            sha256=sha256,
        )


@dataclass(frozen=True, slots=True)
class Document:
    schema_version: str
    coordinate_system: str
    image: ImageInfo
    text: str
    corrected_text: str | None
    lines: tuple[TextLine, ...]
    language_hints: tuple[str, ...]
    metadata: dict[str, Any]
    _raw: dict[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        _check_keys(
            data,
            {
                'schema_version', 'coordinate_system', 'image', 'text',
                'lines', 'language_hints', 'metadata',
            },
            {'corrected_text'},
            'document',
        )
        if data.get('schema_version') != 'glypho.annotation.v1':
            raise ValueError(f"unsupported schema: {data.get('schema_version')!r}")
        if data.get('coordinate_system') != 'pixel_top_left':
            raise ValueError(f"unsupported coordinate system: {data.get('coordinate_system')!r}")

        image = ImageInfo.from_dict(data['image'])
        lines = tuple(TextLine.from_dict(line) for line in _array(data['lines'], 'lines'))
        language_hints = tuple(
            _string_list(data['language_hints'], 'language_hints', non_empty=True, unique=True)
        )
        _validate_geometry_and_ids(image, lines)
        return cls(
            schema_version=data['schema_version'],
            coordinate_system=data['coordinate_system'],
            image=image,
            text=_string(data['text'], 'text'),
            corrected_text=_optional_string(data.get('corrected_text'), 'corrected_text'),
            lines=lines,
            language_hints=language_hints,
            metadata=_validate_metadata(data['metadata']),
            _raw=deepcopy(data),
        )

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._raw)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = _number(value, 'confidence')
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError('confidence must be between 0 and 1')
    return result


def _check_keys(
    data: dict[str, Any],
    required: set[str],
    optional: set[str],
    name: str,
) -> None:
    if not isinstance(data, dict):
        raise ValueError(f'{name} must be an object')
    missing = required - data.keys()
    if missing:
        raise ValueError(f'{name} is missing fields: {", ".join(sorted(missing))}')
    unknown = data.keys() - required - optional
    if unknown:
        raise ValueError(f'{name} has unknown fields: {", ".join(sorted(unknown))}')


def _array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f'{name} must be an array')
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f'{name} must be boolean')
    return value


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f'{name} must be an integer')
    return value


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f'{name} must be a number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def _string(value: Any, name: str, *, non_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f'{name} must be a string')
    if non_empty and not value:
        raise ValueError(f'{name} must not be empty')
    return value


def _optional_string(value: Any, name: str, *, non_empty: bool = False) -> str | None:
    if value is None:
        return None
    return _string(value, name, non_empty=non_empty)


def _string_list(
    value: Any,
    name: str,
    *,
    non_empty: bool = False,
    unique: bool = False,
) -> list[str]:
    values = _array(value, name)
    result = [_string(item, f'{name}[{index}]', non_empty=non_empty) for index, item in enumerate(values)]
    if unique and len(set(result)) != len(result):
        raise ValueError(f'{name} must not contain duplicates')
    return result


def _validate_geometry_and_ids(image: ImageInfo, lines: tuple[TextLine, ...]) -> None:
    ids: set[str] = set()
    orders: set[int] = set()
    for line in lines:
        if line.id in ids:
            raise ValueError(f'duplicate region id: {line.id}')
        if line.order in orders:
            raise ValueError(f'duplicate line order: {line.order}')
        ids.add(line.id)
        orders.add(line.order)
        _validate_quad_bounds(line.quad, image)
        for word in line.words:
            if word.id in ids:
                raise ValueError(f'duplicate region id: {word.id}')
            ids.add(word.id)
            _validate_quad_bounds(word.quad, image)


def _validate_quad_bounds(quad: Quad, image: ImageInfo) -> None:
    if any(point.x > image.width or point.y > image.height for point in quad.points):
        raise ValueError('quad is outside the image')


def _validate_metadata(value: Any) -> dict[str, Any]:
    required = {'status'}
    optional = {
        'annotator', 'created_at', 'updated_at', 'engine', 'notes',
        'sensitive', 'tags', 'group_id',
    }
    _check_keys(value, required, optional, 'metadata')
    if value['status'] not in {'draft', 'reviewed', 'verified'}:
        raise ValueError('metadata.status is invalid')
    for name in ('annotator', 'created_at', 'updated_at', 'notes'):
        _optional_string(value.get(name), f'metadata.{name}')
    if 'sensitive' in value:
        _boolean(value['sensitive'], 'metadata.sensitive')
    if 'tags' in value:
        _string_list(value['tags'], 'metadata.tags', non_empty=True, unique=True)
    _optional_string(value.get('group_id'), 'metadata.group_id', non_empty=True)
    if 'engine' in value:
        _validate_engine(value['engine'])
    return deepcopy(value)


def _validate_engine(value: Any) -> None:
    required = {'name', 'version', 'backend', 'languages', 'elapsed_ms'}
    _check_keys(value, required, {'model'}, 'metadata.engine')
    for name in ('name', 'version', 'backend'):
        _string(value[name], f'metadata.engine.{name}')
    _optional_string(value.get('model'), 'metadata.engine.model')
    _string_list(value['languages'], 'metadata.engine.languages')
    elapsed_ms = _integer(value['elapsed_ms'], 'metadata.engine.elapsed_ms')
    if elapsed_ms < 0:
        raise ValueError('metadata.engine.elapsed_ms must not be negative')