param(
    [string]$Version = $(if ($env:GLYPHO_VERSION) { $env:GLYPHO_VERSION } else { 'latest' }),
    [string]$InstallDir = $(if ($env:GLYPHO_INSTALL_DIR) { $env:GLYPHO_INSTALL_DIR } else { "$env:LOCALAPPDATA\Glypho\bin" })
)

$ErrorActionPreference = 'Stop'
$repository = if ($env:GLYPHO_GITHUB_REPOSITORY) { $env:GLYPHO_GITHUB_REPOSITORY } else { 'rinqaku/Glypho' }
$architecture = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString().ToLowerInvariant()
$architecture = switch ($architecture) {
    'x64' { 'x64' }
    'arm64' { 'arm64' }
    default { throw "glypho: unsupported architecture: $architecture" }
}

$asset = "glypho-ocr-win32-$architecture.zip"
$baseUrl = if ($Version -eq 'latest') {
    "https://github.com/$repository/releases/latest/download"
} else {
    "https://github.com/$repository/releases/download/$Version"
}

$temporary = Join-Path ([System.IO.Path]::GetTempPath()) "glypho-$([System.Guid]::NewGuid())"
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    $archive = Join-Path $temporary $asset
    $checksum = "$archive.sha256"
    if ($env:GLYPHO_ASSET_DIR) {
        Copy-Item (Join-Path $env:GLYPHO_ASSET_DIR $asset) $archive
        Copy-Item (Join-Path $env:GLYPHO_ASSET_DIR "$asset.sha256") $checksum
    } else {
        Invoke-WebRequest -Uri "$baseUrl/$asset" -OutFile $archive -TimeoutSec 300
        Invoke-WebRequest -Uri "$baseUrl/$asset.sha256" -OutFile $checksum -TimeoutSec 60
    }

    $expected = ((Get-Content -Raw $checksum).Trim() -split '\s+')[0].ToLowerInvariant()
    $actual = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        throw 'glypho: checksum verification failed'
    }

    Expand-Archive -Path $archive -DestinationPath $temporary
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $source = Join-Path $temporary "glypho-ocr-win32-$architecture\bin\glypho.exe"
    Copy-Item -Force $source (Join-Path $InstallDir 'glypho.exe')

    if (-not $env:GLYPHO_SKIP_PATH_UPDATE) {
        $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        $parts = @($userPath -split ';' | Where-Object { $_ })
        if ($InstallDir -notin $parts) {
            [Environment]::SetEnvironmentVariable('Path', (($parts + $InstallDir) -join ';'), 'User')
        }
    }
    Write-Host "Installed glypho to $InstallDir\glypho.exe"
    Write-Host 'Open a new terminal, then run: glypho image.png'
} finally {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $temporary
}