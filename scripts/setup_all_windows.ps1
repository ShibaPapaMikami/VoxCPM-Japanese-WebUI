param(
    [int]$Port = 8808,
    [string]$HostAddress = "127.0.0.1",
    [string]$Device = "cuda",
    [switch]$SkipBaseSetup,
    [switch]$WithIrodori,
    [switch]$WithQwen3,
    [switch]$WithFasterQwen3,
    [switch]$AllEngines,
    [switch]$AllowFirewall,
    [switch]$NoLaunch,
    [switch]$NoBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$BaseSetupScript = Join-Path $RepoRoot "scripts\setup_windows_cuda.ps1"
$IrodoriSetupScript = Join-Path $RepoRoot "scripts\setup_irodori_tts.ps1"
$Qwen3SetupScript = Join-Path $RepoRoot "scripts\setup_qwen3_tts.ps1"
$FirewallScript = Join-Path $RepoRoot "scripts\allow_firewall_8808.ps1"
$LaunchScript = Join-Path $RepoRoot "scripts\launch_webui.ps1"

if ($AllEngines) {
    $WithIrodori = $true
    $WithQwen3 = $true
}

function Invoke-Step {
    param(
        [string]$Title,
        [scriptblock]$Action,
        [string]$DryRunCommand
    )

    Write-Host ""
    Write-Host "== $Title =="
    if ($DryRun) {
        if (![string]::IsNullOrWhiteSpace($DryRunCommand)) {
            Write-Host $DryRunCommand
        }
        return
    }
    & $Action
}

Write-Host "JP Voice Studio - Windows setup"
Write-Host "Repo: $RepoRoot"
Write-Host "Port: $Port"
Write-Host "Host: $HostAddress"
Write-Host "Device: $Device"
Write-Host "Irodori-TTS: $WithIrodori"
Write-Host "Qwen3-TTS: $WithQwen3"

if (!$DryRun -and !(Get-Command uv -ErrorAction SilentlyContinue)) {
    throw @"
uv was not found. Install uv first, then run this script again.

  winget install --id Astral-sh.UV

Official install guide:
  https://docs.astral.sh/uv/getting-started/installation/
"@
}

if (!$SkipBaseSetup) {
    Invoke-Step `
        -Title "Install VoxCPM2 WebUI dependencies" `
        -DryRunCommand "powershell -ExecutionPolicy Bypass -File `"$BaseSetupScript`"" `
        -Action { & $BaseSetupScript }
}
else {
    Write-Host ""
    Write-Host "== Install VoxCPM2 WebUI dependencies =="
    Write-Host "Skipped by -SkipBaseSetup."
}

if ($WithIrodori) {
    Invoke-Step `
        -Title "Install Irodori-TTS optional engine" `
        -DryRunCommand "powershell -ExecutionPolicy Bypass -File `"$IrodoriSetupScript`"" `
        -Action { & $IrodoriSetupScript }
}

if ($WithQwen3) {
    $qwen3DryRunSuffix = ""
    if ($WithFasterQwen3) {
        $qwen3DryRunSuffix = " -WithFaster"
    }
    Invoke-Step `
        -Title "Install Qwen3-TTS optional engine" `
        -DryRunCommand "powershell -ExecutionPolicy Bypass -File `"$Qwen3SetupScript`"$qwen3DryRunSuffix" `
        -Action {
            if ($WithFasterQwen3) {
                & $Qwen3SetupScript -WithFaster
            }
            else {
                & $Qwen3SetupScript
            }
        }
}

if ($AllowFirewall) {
    Invoke-Step `
        -Title "Allow Windows Firewall port 8808" `
        -DryRunCommand "powershell -ExecutionPolicy Bypass -File `"$FirewallScript`"" `
        -Action { & $FirewallScript }
}

if (!$NoLaunch) {
    $launchDryRunSuffix = ""
    if ($NoBrowser) {
        $launchDryRunSuffix = " -NoBrowser"
    }
    $launchArgs = @{
        Port = $Port
        HostAddress = $HostAddress
        Device = $Device
    }
    if ($NoBrowser) {
        $launchArgs.NoBrowser = $true
    }
    if ($DryRun) {
        $launchArgs.DryRun = $true
    }

    Invoke-Step `
        -Title "Launch JP Voice Studio" `
        -DryRunCommand "powershell -ExecutionPolicy Bypass -File `"$LaunchScript`" -Port $Port -HostAddress $HostAddress -Device $Device$launchDryRunSuffix" `
        -Action { & $LaunchScript @launchArgs }
}
else {
    Write-Host ""
    Write-Host "== Launch JP Voice Studio =="
    Write-Host "Skipped by -NoLaunch."
}

Write-Host ""
Write-Host "Setup flow finished."
