param(
    [string]$InstallDir = "",
    [switch]$SkipSync
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = Join-Path $RepoRoot "external\Irodori-TTS"
}

Write-Host "JPVoxCPM WebUI - Irodori-TTS setup"
Write-Host "Install dir: $InstallDir"

$parent = Split-Path -Parent $InstallDir
if (-not (Test-Path $parent)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
}

if (-not (Test-Path $InstallDir)) {
    git clone https://github.com/Aratako/Irodori-TTS.git $InstallDir
}
else {
    Write-Host "Irodori-TTS directory already exists. Skipping clone."
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv command was not found. Install uv first, then run this script again. See https://docs.astral.sh/uv/"
}

if (-not $SkipSync) {
    Push-Location $InstallDir
    try {
        uv sync --extra cu128
    }
    finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "Irodori-TTS setup finished."
Write-Host "Restart JPVoxCPM WebUI, then select 'Irodori-TTS（日本語特化・実験）' in the audio engine selector."
