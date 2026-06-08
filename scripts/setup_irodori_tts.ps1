param(
    [string]$InstallDir = "",
    [switch]$SkipSync
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = Join-Path $RepoRoot "external\Irodori-TTS"
}

Write-Host "JP Voice Studio - Irodori-TTS setup"
Write-Host "Install dir: $InstallDir"

$env:GIT_SSL_BACKEND = "schannel"
$env:GIT_CONFIG_COUNT = "1"
$env:GIT_CONFIG_KEY_0 = "http.sslBackend"
$env:GIT_CONFIG_VALUE_0 = "schannel"
$env:UV_NATIVE_TLS = "true"

$parent = Split-Path -Parent $InstallDir
if (-not (Test-Path $parent)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
}

if (-not (Test-Path $InstallDir)) {
    git -c http.sslBackend=schannel clone https://github.com/Aratako/Irodori-TTS.git $InstallDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone Irodori-TTS repository."
    }
}
else {
    Write-Host "Irodori-TTS directory already exists. Skipping clone."
}

if (-not (Test-Path (Join-Path $InstallDir "infer.py"))) {
    throw "Irodori-TTS clone did not complete correctly. Missing infer.py in $InstallDir"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv command was not found. Install uv first, then run this script again. See https://docs.astral.sh/uv/"
}

if (-not $SkipSync) {
    Push-Location $InstallDir
    try {
        $VenvDir = Join-Path $InstallDir ".venv"
        $VenvPython = if ($IsWindows -or $env:OS -eq "Windows_NT") {
            Join-Path $VenvDir "Scripts\python.exe"
        }
        else {
            Join-Path $VenvDir "bin/python"
        }

        if (Test-Path $VenvPython) {
            & $VenvPython -c "import sys; print(sys.executable)" *> $null
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Existing Irodori-TTS .venv is broken. Recreating it..."
                Remove-Item -LiteralPath $VenvDir -Recurse -Force
            }
        }

        uv --native-tls sync --extra cu128
        if ($LASTEXITCODE -ne 0) {
            throw "uv sync failed."
        }
        uv --native-tls pip install truststore
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install truststore into the Irodori-TTS environment."
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "Irodori-TTS setup finished."
Write-Host "Restart JP Voice Studio, then select 'Irodori-TTS（日本語特化・実験）' in the audio engine selector."
