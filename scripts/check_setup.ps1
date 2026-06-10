param(
    [int]$Port = 8808,
    [string]$HostAddress = "127.0.0.1",
    [switch]$Json
)

$ErrorActionPreference = "Continue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$IrodoriDir = Join-Path $RepoRoot "external\Irodori-TTS"
$IrodoriPython = Join-Path $IrodoriDir ".venv\Scripts\python.exe"
$VoxSnapshotRoot = Join-Path $RepoRoot "pretrained_models\hf-cache\models--openbmb--VoxCPM2\snapshots"

$script:Results = New-Object System.Collections.Generic.List[object]

function Add-Result {
    param(
        [string]$Name,
        [string]$Status,
        [string]$Detail = "",
        [string]$Fix = ""
    )
    $script:Results.Add([pscustomobject]@{
        name = $Name
        status = $Status
        detail = $Detail
        fix = $Fix
    }) | Out-Null
}

function Test-CommandAvailable {
    param(
        [string]$CommandName,
        [string]$Fix = ""
    )
    $cmd = Get-Command $CommandName -ErrorAction SilentlyContinue
    if ($cmd) {
        Add-Result $CommandName "PASS" $cmd.Source ""
    }
    else {
        Add-Result $CommandName "FAIL" "Command not found." $Fix
    }
}

function Invoke-PythonCheck {
    param(
        [string]$Name,
        [string]$Code,
        [string]$Fix = "",
        [string]$WorkingDirectory = $RepoRoot,
        [switch]$LastLineOnPass
    )
    if (-not (Test-Path $Python)) {
        Add-Result $Name "FAIL" "Python environment was not found: $Python" "Run scripts\setup_windows_cuda.ps1"
        return
    }
    try {
        $output = & $Python -c $Code 2>&1
        if ($LASTEXITCODE -eq 0) {
            $detail = ($output | Out-String).Trim()
            if ($LastLineOnPass) {
                $lines = @($output | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
                if ($lines.Count -gt 0) {
                    $detail = $lines[$lines.Count - 1]
                }
            }
            Add-Result $Name "PASS" $detail ""
        }
        else {
            Add-Result $Name "FAIL" (($output | Out-String).Trim()) $Fix
        }
    }
    catch {
        Add-Result $Name "FAIL" $_.Exception.Message $Fix
    }
}

function Test-PathCheck {
    param(
        [string]$Name,
        [string]$Path,
        [string]$Fix = "",
        [switch]$Directory,
        [switch]$File
    )
    $exists = Test-Path $Path
    if ($exists -and $Directory) {
        $exists = (Get-Item $Path -ErrorAction SilentlyContinue).PSIsContainer
    }
    if ($exists -and $File) {
        $exists = -not (Get-Item $Path -ErrorAction SilentlyContinue).PSIsContainer
    }
    if ($exists) {
        Add-Result $Name "PASS" $Path ""
    }
    else {
        Add-Result $Name "FAIL" "Missing: $Path" $Fix
    }
}

function Test-PythonExecutable {
    param(
        [string]$Name,
        [string]$Path,
        [string]$Fix = "",
        [string]$FailureStatus = "FAIL"
    )
    if (-not (Test-Path $Path)) {
        Add-Result $Name $FailureStatus "Missing: $Path" $Fix
        return
    }
    try {
        $output = & $Path -c "import sys; print(sys.version.split()[0]); print(sys.executable)" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Add-Result $Name "PASS" (($output | Out-String).Trim()) ""
        }
        else {
            $lines = @($output | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
            $detail = if ($lines.Count -gt 0) { $lines[0] } else { "Python command failed." }
            Add-Result $Name $FailureStatus $detail $Fix
        }
    }
    catch {
        Add-Result $Name $FailureStatus $_.Exception.Message $Fix
    }
}

function Test-Port {
    param(
        [int]$Port,
        [string]$HostAddress
    )
    try {
        $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($listeners) {
            $owners = ($listeners | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
            Add-Result "Port $Port listener" "PASS" "Already listening. PID(s): $owners" ""
            return
        }
    }
    catch {
        Add-Result "Port $Port listener" "WARN" "Could not query TCP listeners: $($_.Exception.Message)" ""
    }

    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $async = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if ($async.AsyncWaitHandle.WaitOne(1000, $false)) {
            $client.EndConnect($async)
            $client.Close()
            Add-Result "Port $Port reachability" "PASS" "A service is reachable on http://127.0.0.1:$Port/." ""
            return
        }
        $client.Close()
    }
    catch {
        try {
            if ($client) {
                $client.Close()
            }
        }
        catch {}
    }

    try {
        $address = [System.Net.IPAddress]::Parse("127.0.0.1")
        $listener = [System.Net.Sockets.TcpListener]::new($address, $Port)
        $listener.Start()
        $listener.Stop()
        Add-Result "Port $Port availability" "PASS" "Port is available on 127.0.0.1." ""
    }
    catch {
        Add-Result "Port $Port availability" "FAIL" $_.Exception.Message "Close the process using the port or choose another port."
    }

    if ($HostAddress -eq "0.0.0.0") {
        try {
            $rule = Get-NetFirewallRule -DisplayName "VoxCPM Web UI 8808" -ErrorAction SilentlyContinue
            if ($rule) {
                Add-Result "Windows Firewall rule" "PASS" "VoxCPM Web UI 8808 exists." ""
            }
            else {
                Add-Result "Windows Firewall rule" "WARN" "Firewall rule was not found." "Run scripts\allow_firewall_8808.ps1 as Administrator."
            }
        }
        catch {
            Add-Result "Windows Firewall rule" "WARN" "Could not query firewall rules: $($_.Exception.Message)" ""
        }
    }
}

if (!$Json) {
    Write-Host "JP Voice Studio setup check"
    Write-Host "Repo: $RepoRoot"
    Write-Host "Port: $Port"
    Write-Host "Host: $HostAddress"
    Write-Host ""
}

Test-PathCheck "Repository root" $RepoRoot -Directory
Test-PathCheck "app.py" (Join-Path $RepoRoot "app.py") -File
Test-PathCheck "launch script" (Join-Path $RepoRoot "scripts\launch_webui.ps1") -File
Test-CommandAvailable "git" "Install Git for Windows."
Test-CommandAvailable "uv" "Install uv: winget install --id Astral-sh.UV"

Test-PythonExecutable "WebUI Python" $Python "Run scripts\setup_windows_cuda.ps1"
Invoke-PythonCheck "Python packages: gradio/voxcpm/truststore" "import gradio, truststore; import voxcpm; print('imports ok')" "Run scripts\setup_windows_cuda.ps1"
Invoke-PythonCheck "PyTorch CUDA" "import torch; print(torch.__version__); print('cuda=' + str(torch.cuda.is_available())); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')" "Install CUDA PyTorch with scripts\setup_windows_cuda.ps1"

if (Test-Path $VoxSnapshotRoot) {
    $snapshot = Get-ChildItem $VoxSnapshotRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "config.json") } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($snapshot) {
        Add-Result "VoxCPM2 local model cache" "PASS" $snapshot.FullName ""
    }
    else {
        Add-Result "VoxCPM2 local model cache" "WARN" "Snapshot directory exists, but config.json was not found." "The model can still download on first launch."
    }
}
else {
    Add-Result "VoxCPM2 local model cache" "WARN" "Local snapshot cache was not found." "The model can download on first launch, or rerun setup."
}

Test-PathCheck "Irodori-TTS directory" $IrodoriDir "Run scripts\setup_irodori_tts.ps1" -Directory
Test-PathCheck "Irodori-TTS infer.py" (Join-Path $IrodoriDir "infer.py") "Run scripts\setup_irodori_tts.ps1" -File
Test-PathCheck "Irodori-TTS train.py" (Join-Path $IrodoriDir "train.py") "Run scripts\setup_irodori_tts.ps1" -File
Test-PythonExecutable "Irodori .venv Python" $IrodoriPython "Run scripts\setup_irodori_tts.ps1 to repair the Irodori environment. The WebUI can fall back to its own Python if Irodori imports pass." "WARN"
Invoke-PythonCheck "Irodori import from WebUI Python" "import sys, pathlib; root=pathlib.Path(r'$IrodoriDir'); sys.path.insert(0, str(root)); import irodori_tts; import peft; import soundfile; import dacvae; print('irodori deps ok')" "Run scripts\setup_irodori_tts.ps1"

$irodoriModelRoot = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--Aratako--Irodori-TTS-500M-v3\snapshots"
if (Test-Path $irodoriModelRoot) {
    $model = Get-ChildItem $irodoriModelRoot -Recurse -Filter "model.safetensors" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($model) {
        Add-Result "Irodori model cache" "PASS" $model.FullName ""
    }
    else {
        Add-Result "Irodori model cache" "WARN" "Cache directory exists, but model.safetensors was not found." "It can download on first Irodori use."
    }
}
else {
    Add-Result "Irodori model cache" "WARN" "Irodori model cache was not found." "It can download on first Irodori use."
}

$dacvaeRoot = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--Aratako--Semantic-DACVAE-Japanese-32dim\snapshots"
if (Test-Path $dacvaeRoot) {
    $weights = Get-ChildItem $dacvaeRoot -Recurse -Filter "weights.pth" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($weights) {
        Add-Result "Semantic DACVAE cache" "PASS" $weights.FullName ""
    }
    else {
        Add-Result "Semantic DACVAE cache" "WARN" "Cache directory exists, but weights.pth was not found." "It can download when LoRA data is encoded."
    }
}
else {
    Add-Result "Semantic DACVAE cache" "WARN" "Semantic DACVAE cache was not found." "It can download when LoRA data is encoded."
}

Invoke-PythonCheck "Qwen3-TTS package" "import qwen_tts; import sentencepiece; import truststore; print('qwen3 deps ok')" "Run scripts\setup_qwen3_tts.ps1" -LastLineOnPass

Test-Port -Port $Port -HostAddress $HostAddress

if ($Json) {
    $script:Results | ConvertTo-Json -Depth 4
}
else {
    $failCount = @($script:Results | Where-Object { $_.status -eq "FAIL" }).Count
    $warnCount = @($script:Results | Where-Object { $_.status -eq "WARN" }).Count
    $passCount = @($script:Results | Where-Object { $_.status -eq "PASS" }).Count
    foreach ($result in $script:Results) {
        $line = "[{0}] {1}" -f $result.status, $result.name
        Write-Host $line
        if (![string]::IsNullOrWhiteSpace($result.detail)) {
            Write-Host "  $($result.detail)"
        }
        if (![string]::IsNullOrWhiteSpace($result.fix)) {
            Write-Host "  fix: $($result.fix)"
        }
    }
    Write-Host ""
    Write-Host "Summary: FAIL=$failCount WARN=$warnCount PASS=$passCount"
}

if (@($script:Results | Where-Object { $_.status -eq "FAIL" }).Count -gt 0) {
    exit 1
}
exit 0
