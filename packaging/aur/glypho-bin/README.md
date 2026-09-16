# glypho-bin AUR packaging

This directory prepares the AUR package that installs Glypho's official prebuilt Linux CLI release.

The package is intentionally named `glypho-bin`: the AUR package consumes upstream prebuilt release artifacts rather than rebuilding the Rust/ONNX Runtime stack from source.

## Prepare a release

The GitHub Release must exist first because the PKGBUILD pins the SHA-256 of both Linux archives.

```bash
./packaging/aur/glypho-bin/prepare.sh 0.2.0
cd packaging/aur/glypho-bin
makepkg --cleanbuild
```

Then inspect `PKGBUILD` and `.SRCINFO` and publish those two files to the AUR Git repository:

```bash
git clone ssh://aur@aur.archlinux.org/glypho-bin.git /tmp/glypho-bin-aur
cp PKGBUILD .SRCINFO /tmp/glypho-bin-aur/
cd /tmp/glypho-bin-aur
git add PKGBUILD .SRCINFO
git commit -m 'glypho-bin 0.2.0-1'
git push
```

For later Glypho releases, rerun `prepare.sh` with the new version and update the AUR repository.
