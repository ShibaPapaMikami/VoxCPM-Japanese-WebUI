param(
    [int]$Port = 8808,
    [string]$Device = "cuda",
    [string]$ModelId = "",
    [switch]$NoBrowser,
    [switch]$LoadDenoiser,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$App = Join-Path $RepoRoot "app.py"

if (!(Test-Path $Python)) {
    throw "Missing Python environment: $Python. Run scripts\setup_windows_cuda.ps1 first."
}

if (!(Test-Path $App)) {
    throw "Missing app.py: $App"
}

if ([string]::IsNullOrWhiteSpace($ModelId)) {
    $SnapshotRoot = Join-Path $RepoRoot "pretrained_models\hf-cache\models--openbmb--VoxCPM2\snapshots"
    if (Test-Path $SnapshotRoot) {
        $Snapshot = Get-ChildItem $SnapshotRoot -Directory |
            Where-Object { Test-Path (Join-Path $_.FullName "config.json") } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($Snapshot) {
            $ModelId = $Snapshot.FullName
        }
    }
}

if ([string]::IsNullOrWhiteSpace($ModelId)) {
    $ModelId = "openbmb/VoxCPM2"
}

$Url = "http://127.0.0.1:$Port"
$AppArgs = @(
    $App,
    "--model-id", $ModelId,
    "--port", "$Port",
    "--device", $Device
)

if (!$LoadDenoiser) {
    $AppArgs += "--no-denoiser"
}

Write-Host "VoxCPM Web UI"
Write-Host "  URL:    $Url"
Write-Host "  Model:  $ModelId"
Write-Host "  Device: $Device"

if ($DryRun) {
    Write-Host "  Command: $Python $($AppArgs -join ' ')"
    exit 0
}

if (!$NoBrowser) {
    $BrowserCommand = "Start-Sleep -Seconds 15; Start-Process '$Url'"
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command", $BrowserCommand
    ) -WindowStyle Hidden | Out-Null
}

Push-Location $RepoRoot
try {
    & $Python @AppArgs
}
finally {
    Pop-Location
}
