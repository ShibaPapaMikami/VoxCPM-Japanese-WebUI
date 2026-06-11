param(
    [string]$Output = "",
    [switch]$NoOverwrite
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Source = Join-Path $RepoRoot "scripts\VoxCPMWebUILauncher.cs"

if (-not $Output) {
    $Output = Join-Path $RepoRoot "VoxCPM_WebUI.exe"
}
elseif (-not [System.IO.Path]::IsPathRooted($Output)) {
    $Output = Join-Path $RepoRoot $Output
}

if (-not (Test-Path $Source)) {
    throw "Launcher source was not found: $Source"
}

if ($NoOverwrite -and (Test-Path $Output)) {
    throw "Output already exists: $Output"
}

$csc = Get-Command csc.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
if (-not $csc) {
    $candidates = @(
        (Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"),
        (Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe"),
        (Join-Path $env:WINDIR "Microsoft.NET\Framework64\v3.5\csc.exe"),
        (Join-Path $env:WINDIR "Microsoft.NET\Framework\v3.5\csc.exe")
    )
    $csc = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $csc) {
    throw "csc.exe was not found. Install .NET Framework developer tools or Visual Studio Build Tools."
}

$outputDir = Split-Path -Parent $Output
if ($outputDir) {
    New-Item -ItemType Directory -Force $outputDir | Out-Null
}

Write-Host "Compiler: $csc"
Write-Host "Source:   $Source"
Write-Host "Output:   $Output"

& $csc /nologo /target:exe /platform:anycpu /optimize+ "/out:$Output" "$Source"
if ($LASTEXITCODE -ne 0) {
    throw "Launcher build failed with exit code $LASTEXITCODE."
}

$built = Get-Item $Output
Write-Host "Built launcher: $($built.FullName) ($($built.Length) bytes)"
