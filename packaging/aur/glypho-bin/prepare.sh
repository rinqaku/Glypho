#!/usr/bin/env bash
set -euo pipefail

version="${1:-}"
if [[ -z "$version" ]]; then
  echo "usage: $0 VERSION   (example: $0 0.2.1)" >&2
  exit 2
fi
version="${version#v}"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
  echo "glypho-bin: invalid version: $version" >&2
  exit 2
fi

repo='rinqaku/Glypho'
base="https://github.com/$repo/releases/download/v$version"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

digest() {
  local platform="$1"
  local checksum
  checksum="$(curl --fail --location --silent --show-error \
    "$base/glypho-ocr-$platform.tar.gz.sha256" | awk 'NR == 1 {print $1}')"
  if [[ ! "$checksum" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "glypho-bin: invalid $platform checksum" >&2
    return 1
  fi
  printf '%s\n' "${checksum,,}"
}

x64="$(digest linux-x64)"
arm64="$(digest linux-arm64)"

sed \
  -e "s/@VERSION@/$version/g" \
  -e "s/@SHA256_X86_64@/$x64/g" \
  -e "s/@SHA256_AARCH64@/$arm64/g" \
  "$root/PKGBUILD.in" > "$root/PKGBUILD"

if command -v makepkg >/dev/null 2>&1; then
  (cd "$root" && makepkg --printsrcinfo > .SRCINFO)
  echo "generated PKGBUILD and .SRCINFO for glypho-bin $version"
else
  echo "generated PKGBUILD for glypho-bin $version"
  echo "run 'makepkg --printsrcinfo > .SRCINFO' on Arch Linux before publishing to AUR"
fi
