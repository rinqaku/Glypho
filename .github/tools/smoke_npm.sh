#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:?usage: smoke_npm.sh <platform> <package-dir>}"
PACKAGE_DIR="${2:?usage: smoke_npm.sh <platform> <package-dir>}"
ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
if command -v cygpath >/dev/null 2>&1; then
  ROOT="$(cygpath -u "$ROOT")"
fi
PACKAGE_DIR="$(cd "$PACKAGE_DIR" && pwd)"
VERSION="$(cd "$ROOT/js" && node -p "require('./package.json').version")"
PLATFORM_PACKAGE="$PACKAGE_DIR/glypho-ocr-$TARGET-$VERSION.tgz"

if [[ ! -f "$PLATFORM_PACKAGE" ]]; then
  echo "Missing platform package: $PLATFORM_PACKAGE" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d)"
WRAPPER_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$WORK_DIR" "$WRAPPER_DIR"
}
trap cleanup EXIT

npm pack "$ROOT/js" --pack-destination "$WRAPPER_DIR" >/dev/null
WRAPPER_PACKAGE="$WRAPPER_DIR/glypho-ocr-$VERSION.tgz"

cat > "$WORK_DIR/package.json" <<'JSON'
{
  "name": "glypho-package-smoke-test",
  "private": true,
  "type": "module"
}
JSON

cd "$WORK_DIR"
npm install \
  --offline \
  --omit=optional \
  --package-lock=false \
  --no-audit \
  --no-fund \
  "$PLATFORM_PACKAGE" \
  "$WRAPPER_PACKAGE"

npx --no-install glypho --version

cat > smoke.mjs <<'NODE'
import { Glypho } from 'glypho-ocr';
import path from 'node:path';

const ocr = new Glypho({
  languages: ['en'],
  quality: 'balanced',
  device: 'cpu',
});

try {
  const image = path.join(process.env.GITHUB_WORKSPACE, '.github', 'fixtures', 'english.png');
  const document = await ocr.recognize(image);
  const expected = 'The quick brown fox jumps over 13 lazy dogs.';
  if (document.text !== expected) {
    throw new Error(`Unexpected OCR text: ${JSON.stringify(document.text)}`);
  }
  console.log(document.text);
} finally {
  await ocr.close();
}
NODE

node smoke.mjs
