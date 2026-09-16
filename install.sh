#!/bin/sh
set -eu

repository=${GLYPHO_GITHUB_REPOSITORY:-rinqaku/Glypho}
version=${GLYPHO_VERSION:-latest}
install_dir=${GLYPHO_INSTALL_DIR:-}
asset_dir=${GLYPHO_ASSET_DIR:-}

if [ -z "$install_dir" ]; then
  install_dir="$HOME/.local/bin"
  existing=$(command -v glypho 2>/dev/null || true)
  case "$existing" in
    "$HOME"/*)
      existing_dir=${existing%/*}
      existing_version=$("$existing" --version 2>/dev/null || true)
      # Keep an older user installation from shadowing the downloaded release.
      if [ -d "$existing_dir" ] && [ -w "$existing_dir" ] && [ ! -L "$existing" ] &&
        [ "${existing_version#glypho }" != "$existing_version" ]; then
        install_dir=$existing_dir
      fi
      ;;
  esac
fi

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

if [ "$platform-$architecture" = darwin-x64 ]; then
  printf '%s\n' 'glypho: macOS Intel is not currently distributed; build from source to use this platform' >&2
  exit 1
fi

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
for source in "$temporary/glypho-ocr-$platform-$architecture"/bin/*; do
  install -m 0755 "$source" "$install_dir/$(basename "$source")"
done

installed="$install_dir/glypho"
installed_version=$("$installed" --version 2>/dev/null || true)
if [ -n "$installed_version" ]; then
  printf 'Installed %s to %s\n' "$installed_version" "$installed"
else
  printf 'Installed glypho to %s\n' "$installed"
fi

resolved=$(command -v glypho 2>/dev/null || true)
if [ -n "$resolved" ] && [ "$resolved" != "$installed" ]; then
  printf 'Warning: %s shadows the new installation at %s.\n' "$resolved" "$installed" >&2
fi
case ":$PATH:" in
  *":$install_dir:"*) ;;
  *) printf 'Add %s to PATH to run glypho from any directory.\n' "$install_dir" ;;
esac
