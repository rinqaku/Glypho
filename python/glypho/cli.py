import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    name = 'glypho.exe' if os.name == 'nt' else 'glypho'
    package = Path(__file__).resolve().parent
    binary = next(
        (
            candidate
            for candidate in (package / '_native' / name, package / '_bin' / name)
            if candidate.is_file()
        ),
        None,
    )
    if binary is None:
        raise SystemExit('glypho: packaged native executable is missing; reinstall glypho-ocr')
    try:
        result = subprocess.run([os.fspath(binary), *sys.argv[1:]], check=False)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    raise SystemExit(result.returncode)
