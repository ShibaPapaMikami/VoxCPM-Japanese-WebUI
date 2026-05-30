param(
    [int]$Port = 8808,
    [string]$Device = "cuda",
    [switch]$SkipSetup,
    [switch]$NoBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$SetupScript = Join-Path $RepoRoot "scripts\setup_windows_cuda.ps1"
$LaunchScript = Join-Path $RepoRoot "scripts\launch_webui.ps1"

if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    throw @"
uv が見つかりません。
先に以下のいずれかで uv をインストールしてください。

  winget install --id Astral-sh.UV

または公式手順:
  https://docs.astral.sh/uv/getting-started/installation/
"@
}

if (!$SkipSetup) {
    & $SetupScript
}

$launchArgs = @{
    Port = $Port
    Device = $Device
}

if ($NoBrowser) {
    $launchArgs.NoBrowser = $true
}

if ($DryRun) {
    $launchArgs.DryRun = $true
}

& $LaunchScript @launchArgs
