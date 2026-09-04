#!/usr/bin/env python3

import json
import os
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding='utf-8') as file:
        return json.load(file)


def main() -> int:
    with (ROOT / 'Cargo.toml').open('rb') as file:
        version = tomllib.load(file)['workspace']['package']['version']

    js_manifest = load_json(ROOT / 'js' / 'package.json')
    manifests = {
        'python/pyproject.toml': toml_version(ROOT / 'python' / 'pyproject.toml'),
        'js/package.json': js_manifest['version'],
    }
    native_packages: list[str] = []
    for path in sorted((ROOT / 'npm').glob('*/package.json')):
        manifest = load_json(path)
        manifests[path.relative_to(ROOT).as_posix()] = manifest['version']
        native_packages.append(str(manifest['name']))

    mismatches = [
        f'{path}: {value}'
        for path, value in manifests.items()
        if value != version
    ]
    unpublished_native_packages = {'glypho-ocr-darwin-x64'}
    expected_optional = {
        name: version
        for name in native_packages
        if name not in unpublished_native_packages
    }
    actual_optional = js_manifest.get('optionalDependencies', {})
    if actual_optional != expected_optional:
        mismatches.append(
            'js/package.json optionalDependencies do not exactly match '
            f'the native packages at version {version}'
        )
    tag = os.environ.get('GITHUB_REF_NAME', '')
    if tag.startswith('v') and tag[1:] != version:
        mismatches.append(f'git tag: {tag}')
    if mismatches:
        print(f'expected version {version}')
        print('\n'.join(mismatches))
        return 1
    print(f'all package versions match {version}')
    return 0


def toml_version(path: Path) -> str:
    with path.open('rb') as file:
        return tomllib.load(file)['project']['version']


if __name__ == '__main__':
    raise SystemExit(main())
