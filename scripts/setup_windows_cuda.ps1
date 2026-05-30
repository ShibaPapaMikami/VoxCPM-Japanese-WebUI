param(
    [string]$TorchVersion = "2.7.0+cu128",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    Write-Host "Syncing VoxCPM dependencies into .venv..."
    uv --native-tls --cache-dir .uv-cache sync --no-dev --no-managed-python

    $Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

    Write-Host "Installing truststore for Windows/corporate TLS roots..."
    uv --native-tls --cache-dir .uv-cache pip install --python $Python truststore

    Write-Host "Installing CUDA PyTorch $TorchVersion..."
    uv --native-tls --cache-dir .uv-cache pip install --python $Python --force-reinstall `
        "torch==$TorchVersion" "torchaudio==$TorchVersion" --index-url $TorchIndexUrl

    Write-Host "Verifying CUDA..."
    & $Python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
}
finally {
    Pop-Location
}
