import os
import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
from wheel.bdist_wheel import bdist_wheel


PACKAGE_ROOT = Path(__file__).resolve().parent


def _source_root() -> Path:
    repository = PACKAGE_ROOT.parent
    if (repository / 'Cargo.toml').is_file():
        return repository
    bundled = PACKAGE_ROOT / '_rust'
    if (bundled / 'Cargo.toml').is_file():
        return bundled
    raise RuntimeError('Glypho Rust sources are missing')


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True


class BuildPy(build_py):
    def run(self) -> None:
        self._build_native()
        super().run()
        package = Path(self.build_lib) / 'glypho'
        self._copy_artifact(_library_path(), package / '_libs')
        self._copy_artifact(_binary_path(), package / '_bin')

    def _build_native(self) -> None:
        if os.environ.get('GLYPHO_SKIP_NATIVE_BUILD') == '1':
            return
        command = ['cargo', 'build', '--release', '--locked', '-p', 'glypho-ocr']
        features = os.environ.get('GLYPHO_CARGO_FEATURES')
        if features:
            command.extend(['--features', features])
        subprocess.run(command, cwd=_source_root(), check=True)

    @staticmethod
    def _copy_artifact(source: Path, destination: Path) -> None:
        if not source.is_file():
            raise RuntimeError(f'native artifact is missing: {source}')
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / source.name
        shutil.copy2(source, target)
        if source == _binary_path() and platform.system() != 'Windows':
            target.chmod(0o755)


class PlatformWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform_tag = super().get_tag()
        return 'py3', 'none', platform_tag


class SourceDistribution(sdist):
    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        super().make_release_tree(base_dir, files)
        source = _source_root()
        destination = Path(base_dir) / '_rust'
        destination.mkdir()
        for name in ('Cargo.toml', 'Cargo.lock', 'README.md', 'LICENSE'):
            shutil.copy2(source / name, destination / name)
        shutil.copytree(source / 'crates' / 'glypho', destination / 'crates' / 'glypho')


def _target_directory() -> Path:
    configured = os.environ.get('CARGO_TARGET_DIR')
    return Path(configured) if configured else _source_root() / 'target'


def _library_path() -> Path:
    name = {
        'Darwin': 'libglypho.dylib',
        'Windows': 'glypho.dll',
    }.get(platform.system(), 'libglypho.so')
    return _target_directory() / 'release' / name


def _binary_path() -> Path:
    name = 'glypho.exe' if platform.system() == 'Windows' else 'glypho'
    return _target_directory() / 'release' / name


setup(
    cmdclass={
        'bdist_wheel': PlatformWheel,
        'build_py': BuildPy,
        'sdist': SourceDistribution,
    },
    distclass=BinaryDistribution,
    zip_safe=False,
)
