param(
    [switch]$WithFaster
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Write-Host "JP Voice Studio - Qwen3-TTS setup"
Write-Host "Repo root: $RepoRoot"

$env:GIT_SSL_BACKEND = "schannel"
$env:GIT_CONFIG_COUNT = "1"
$env:GIT_CONFIG_KEY_0 = "http.sslBackend"
$env:GIT_CONFIG_VALUE_0 = "schannel"
$env:UV_NATIVE_TLS = "true"

if (-not (Test-Path $Python)) {
    throw "Python environment was not found: $Python. Run scripts\setup_windows_cuda.ps1 first."
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv command was not found. Install uv first, then run this script again. See https://docs.astral.sh/uv/"
}

Push-Location $RepoRoot
try {
    uv --native-tls --cache-dir .uv-cache pip install --python $Python qwen-tts sentencepiece truststore
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install qwen-tts."
    }
    if ($WithFaster) {
        uv --native-tls --cache-dir .uv-cache pip install --python $Python faster-qwen3-tts
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install faster-qwen3-tts."
        }
    }
    & $Python -c "import qwen_tts; print('qwen_tts import ok')"
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Qwen3-TTS setup finished."
Write-Host "Restart JP Voice Studio, then select the VoiceDesignCloner/Qwen3-TTS integration engine."
