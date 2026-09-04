from glypho.models import Document


def test_document_parses_valid_schema():
    raw = {
        'schema_version': 'glypho.annotation.v1',
        'coordinate_system': 'pixel_top_left',
        'image': {
            'id': 'sample',
            'file_name': 'sample.png',
            'width': 20,
            'height': 10,
        },
        'text': 'Hi',
        'lines': [],
        'language_hints': ['en'],
        'metadata': {'status': 'draft'},
    }

    document = Document.from_dict(raw)

    assert document.text == 'Hi'
    assert document.image.width == 20
    assert document.to_dict() == raw


def test_document_rejects_unknown_fields():
    raw = {
        'schema_version': 'glypho.annotation.v1',
        'coordinate_system': 'pixel_top_left',
        'image': {
            'id': 'sample',
            'file_name': 'sample.png',
            'width': 20,
            'height': 10,
        },
        'text': '',
        'lines': [],
        'language_hints': [],
        'metadata': {'status': 'draft'},
        'unknown': True,
    }

    try:
        Document.from_dict(raw)
    except ValueError as error:
        assert 'unknown fields' in str(error)
    else:
        raise AssertionError('unknown field should fail')


def test_document_rejects_unknown_metadata_fields():
    raw = {
        'schema_version': 'glypho.annotation.v1',
        'coordinate_system': 'pixel_top_left',
        'image': {
            'id': 'sample',
            'file_name': 'sample.png',
            'width': 20,
            'height': 10,
        },
        'text': '',
        'lines': [],
        'language_hints': [],
        'metadata': {'status': 'draft', 'unknown': True},
    }

    try:
        Document.from_dict(raw)
    except ValueError as error:
        assert 'unknown fields' in str(error)
    else:
        raise AssertionError('unknown metadata field should fail')
