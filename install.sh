#!/bin/sh
set -eu

repository=${GLYPHO_GITHUB_REPOSITORY:-rinqaku/Glypho}
version=${GLYPHO_VERSION:-latest}
install_dir=${GLYPHO_INSTALL_DIR:-"$HOME/.local/bin"}
asset_dir=${GLYPHO_ASSET_DIR:-}

case "$(uname -s)" in
  Linux) platform=linux ;;
  Darwin) platform=darwin ;;
  *)
    printf '%s\n' 'glypho: this installer supports Linux and macOS' >&2
    exit 1
    ;;
esac

case "$(uname -m)" in
  x86_64|amd64) architecture=x64 ;;
  arm64|aarch64) architecture=arm64 ;;
  *)
    printf 'glypho: unsupported architecture: %s\n' "$(uname -m)" >&2
    exit 1
    ;;
esac

asset="glypho-ocr-$platform-$architecture.tar.gz"
if [ "$version" = latest ]; then
  base_url="https://github.com/$repository/releases/latest/download"
else
  base_url="https://github.com/$repository/releases/download/$version"
fi

temporary=$(mktemp -d)
cleanup() {
  rm -rf -- "$temporary"
}
trap cleanup EXIT HUP INT TERM

if [ -n "$asset_dir" ]; then
  cp -- "$asset_dir/$asset" "$temporary/$asset"
  cp -- "$asset_dir/$asset.sha256" "$temporary/$asset.sha256"
else
  curl --fail --location --proto '=https' --retry 3 --connect-timeout 15 \
    --max-time 300 --silent --show-error \
    "$base_url/$asset" --output "$temporary/$asset"
  curl --fail --location --proto '=https' --retry 3 --connect-timeout 15 \
    --max-time 60 --silent --show-error \
    "$base_url/$asset.sha256" --output "$temporary/$asset.sha256"
fi

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$temporary" && sha256sum --check "$asset.sha256")
elif command -v shasum >/dev/null 2>&1; then
  expected=$(awk '{print $1}' "$temporary/$asset.sha256")
  actual=$(shasum -a 256 "$temporary/$asset" | awk '{print $1}')
  [ "$actual" = "$expected" ] || {
    printf '%s\n' 'glypho: checksum verification failed' >&2
    exit 1
  }
else
  printf '%s\n' 'glypho: sha256sum or shasum is required' >&2
  exit 1
fi

tar -xzf "$temporary/$asset" -C "$temporary"
mkdir -p -- "$install_dir"
install -m 0755 "$temporary/glypho-ocr-$platform-$architecture/bin/glypho" "$install_dir/glypho"

printf 'Installed glypho to %s/glypho\n' "$install_dir"
case ":$PATH:" in
  *":$install_dir:"*) ;;
  *) printf 'Add %s to PATH to run glypho from any directory.\n' "$install_dir" ;;
esac