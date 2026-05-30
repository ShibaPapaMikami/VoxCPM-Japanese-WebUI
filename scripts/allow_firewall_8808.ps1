param(
    [string]$RuleName = "VoxCPM Web UI 8808",
    [int]$Port = 8808,
    [ValidateSet("Domain", "Private", "Public", "Any")]
    [string]$Profile = "Private"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$StatusPath = Join-Path $RepoRoot "outputs\firewall_8808_status.txt"
New-Item -ItemType Directory -Path (Split-Path -Parent $StatusPath) -Force | Out-Null

try {
    $existing = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
    if ($existing) {
        Set-NetFirewallRule -DisplayName $RuleName -Enabled True -Direction Inbound -Action Allow -Profile $Profile
        Set-NetFirewallPortFilter -AssociatedNetFirewallRule $existing -Protocol TCP -LocalPort $Port
        "Updated firewall rule: $RuleName TCP $Port inbound allow profile $Profile" | Set-Content -Path $StatusPath -Encoding UTF8
    }
    else {
        New-NetFirewallRule `
            -DisplayName $RuleName `
            -Direction Inbound `
            -Action Allow `
            -Protocol TCP `
            -LocalPort $Port `
            -Profile $Profile | Out-Null
        "Created firewall rule: $RuleName TCP $Port inbound allow profile $Profile" | Set-Content -Path $StatusPath -Encoding UTF8
    }
}
catch {
    "Failed firewall rule update: $($_.Exception.Message)" | Set-Content -Path $StatusPath -Encoding UTF8
    throw
}
