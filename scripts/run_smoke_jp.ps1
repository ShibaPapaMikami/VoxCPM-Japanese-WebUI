param(
    [string]$Text = "Konnichiwa. This is a VoxCPM2 smoke test.",
    [string]$Control = "calm Japanese male narration",
    [string]$Output = "outputs\smoke_jp.wav",
    [string]$Device = "cuda",
    [int]$InferenceTimesteps = 4
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VoxCpm = Join-Path $RepoRoot ".venv\Scripts\voxcpm.exe"

if (!(Test-Path $VoxCpm)) {
    throw "Missing .venv. Run scripts\setup_windows_cuda.ps1 first."
}

Push-Location $RepoRoot
try {
    & $VoxCpm design `
        --text $Text `
        --control $Control `
        --output $Output `
        --device $Device `
        --cache-dir "pretrained_models\hf-cache" `
        --no-denoiser `
        --no-optimize `
        --cfg-value 2.0 `
        --inference-timesteps $InferenceTimesteps
}
finally {
    Pop-Location
}
