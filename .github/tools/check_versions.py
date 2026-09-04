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

    manifests = {
        'python/pyproject.toml': toml_version(ROOT / 'python' / 'pyproject.toml'),
        'js/package.json': load_json(ROOT / 'js' / 'package.json')['version'],
    }
    for path in sorted((ROOT / 'npm').glob('*/package.json')):
        manifests[path.relative_to(ROOT).as_posix()] = load_json(path)['version']

    mismatches = [
        f'{path}: {value}'
        for path, value in manifests.items()
        if value != version
    ]
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
