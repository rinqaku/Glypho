#!/usr/bin/env python3

import argparse
import hashlib
import os
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path


PLATFORMS = {
    'darwin-arm64': ('glypho', 'libglypho.dylib'),
    'linux-arm64': ('glypho', 'libglypho.so'),
    'linux-x64': ('glypho', 'libglypho.so'),
    'win32-arm64': ('glypho.exe', 'glypho.dll'),
    'win32-x64': ('glypho.exe', 'glypho.dll'),
}

RUNTIME_LIBRARIES = {
    'darwin-arm64': ('libonnxruntime_providers_shared.dylib',),
    'linux-arm64': ('libonnxruntime_providers_shared.so',),
    'linux-x64': (
        'libonnxruntime_providers_shared.so',
        'libonnxruntime_providers_cuda.so',
    ),
    'win32-arm64': ('onnxruntime_providers_shared.dll',),
    'win32-x64': (
        'onnxruntime_providers_shared.dll',
        'onnxruntime_providers_cuda.dll',
    ),
}


def stage(platform: str, target: Path, require_cuda: bool = False) -> Path:
    binary, library = PLATFORMS[platform]
    root = Path(__file__).resolve().parents[2]
    package = root / 'npm' / platform
    shutil.rmtree(package / 'bin', ignore_errors=True)
    shutil.rmtree(package / 'lib', ignore_errors=True)
    shutil.copy2(root / 'npm' / 'README.md', package / 'README.md')
    shutil.copy2(root / 'LICENSE', package / 'LICENSE')
    for name, directory in ((binary, package / 'bin'), (library, package / 'lib')):
        source = target / name
        if not source.is_file():
            raise FileNotFoundError(source)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / name
        shutil.copy2(source, destination)
        if os.name != 'nt':
            destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    copied = set()
    for name in RUNTIME_LIBRARIES[platform]:
        source = target / name
        if not source.is_file():
            continue
        shutil.copy2(source, package / 'bin' / name)
        copied.add(name)
    if require_cuda:
        required = {name for name in RUNTIME_LIBRARIES[platform] if 'shared' in name or 'cuda' in name}
        missing = sorted(required - copied)
        if missing:
            raise FileNotFoundError(f'CUDA runtime libraries are missing: {", ".join(missing)}')
    return package


def archive(platform: str, package: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    root = f'glypho-ocr-{platform}'
    if platform.startswith('win32'):
        path = output / f'{root}.zip'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for source in sorted(package.glob('*/*')):
                bundle.write(source, f'{root}/{source.relative_to(package)}')
    else:
        path = output / f'{root}.tar.gz'
        with tarfile.open(path, 'w:gz') as bundle:
            for source in sorted(package.glob('*/*')):
                bundle.add(source, arcname=f'{root}/{source.relative_to(package)}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(f'{path.suffix}.sha256').write_text(f'{digest}  {path.name}\n', encoding='ascii')
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('platform', choices=PLATFORMS)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--require-cuda', action='store_true')
    arguments = parser.parse_args()
    package = stage(arguments.platform, arguments.target, arguments.require_cuda)
    if arguments.archive:
        archive(arguments.platform, package, arguments.archive)


if __name__ == '__main__':
    main()
