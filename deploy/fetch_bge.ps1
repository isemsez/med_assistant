# Fetch the pinned bge-m3 F16 GGUF (deploy/bge-pin.txt) into <repo>\models.
# Idempotent: skips the download when the file already matches the pin hash.
#
#   pwsh deploy/fetch_bge.ps1            # -> models\bge-m3-f16.gguf
#   pwsh deploy/fetch_bge.ps1 -Force     # re-download even if present
[CmdletBinding()]
param([switch]$Force)

$ErrorActionPreference = 'Stop'
try {
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Target = Join-Path $RepoRoot 'models\bge-m3-f16.gguf'

$PinFile = Join-Path $PSScriptRoot 'bge-pin.txt'
$pin = @{}
Get-Content -LiteralPath $PinFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#') -and $line -match '^([^=]+)=(.*)$') {
        $pin[$Matches[1].Trim()] = $Matches[2].Trim()
    }
}
foreach ($k in 'url', 'sha256') {
    if (-not $pin[$k]) { throw "pin file $PinFile is missing key '$k'" }
}

if ((Test-Path -LiteralPath $Target) -and -not $Force) {
    $hash = (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash.ToLower()
    if ($hash -eq $pin['sha256']) {
        Write-Host "models\bge-m3-f16.gguf already matches the pin - nothing to do."
        exit 0
    }
    Write-Warning 'existing models\bge-m3-f16.gguf does not match the pin - re-downloading.'
}

$TempRoot = Join-Path $env:TEMP 'kilo\med_assistant_bge'
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null
$Tmp = Join-Path $TempRoot 'bge-m3-f16.gguf'

if (-not (Test-Path -LiteralPath $Tmp)) {
    Write-Host "Downloading $($pin['url']) ..."
    Invoke-WebRequest -Uri $pin['url'] -OutFile $Tmp -UseBasicParsing
} else {
    Write-Host "Reusing cached $Tmp"
}

Write-Host 'Verifying sha256 ...'
$hash = (Get-FileHash -LiteralPath $Tmp -Algorithm SHA256).Hash.ToLower()
if ($hash -ne $pin['sha256']) {
    Remove-Item -LiteralPath $Tmp -Force
    throw "sha256 mismatch: got $hash, pin expects $($pin['sha256']). Deleted the bad download."
}

Move-Item -LiteralPath $Tmp -Destination $Target -Force
Write-Host "Done: $Target"
