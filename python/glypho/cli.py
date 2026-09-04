import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    name = 'glypho.exe' if os.name == 'nt' else 'glypho'
    binary = Path(__file__).resolve().parent / '_bin' / name
    if not binary.is_file():
        raise SystemExit('glypho: packaged native executable is missing; reinstall glypho-ocr')
    try:
        result = subprocess.run([os.fspath(binary), *sys.argv[1:]], check=False)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    raise SystemExit(result.returncode)
