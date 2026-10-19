param(
  [string]$OutputDir = (Join-Path $PSScriptRoot '..\engine\llamacpp'),
  [string]$PinFile  = (Join-Path $PSScriptRoot 'llamacpp-pin.txt'),
  [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Get-Value {
  param([string]$Content, [string]$Key)
  if ($Content -match "(?m)^\s*$Key\s*=\s*(.+)$") {
    return $matches[1].Trim().Trim("`"","'")
  }
  throw "Missing key: $Key in $PinFile"
}

$pin = Get-Content -LiteralPath $PinFile -Raw -Encoding UTF8
$url      = Get-Value $pin 'url'
$zipName  = Get-Value $pin 'zip_name'
$version  = Get-Value $pin 'version'
$tag      = Get-Value $pin 'tag'
$pinSha   = Get-Value $pin 'sha256'
$pinSize  = [int64]::Parse((Get-Value $pin 'size'))

$stampFile = Join-Path $OutputDir "STAMP.txt"
$staging   = Join-Path $env:TEMP "kilo\llamacpp_$version"
$zipPath   = Join-Path $staging $zipName

# --- idempotency / stamp check -------------------------------------------
if ((Test-Path -LiteralPath $OutputDir) -and (Test-Path -LiteralPath $stampFile) -and (-not $Force)) {
  $storedTag = (Get-Content -LiteralPath $stampFile -Raw -Encoding UTF8).Trim()
  if ($storedTag -eq $tag) {
    Write-Host "llama.cpp $tag already installed at $OutputDir (use -Force to re-download)"
    exit 0
  }
  Write-Host "Installed tag ($storedTag) does not match pin ($tag) — re-downloading."
}

# --- download -------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $staging | Out-Null
Write-Host "Downloading llama.cpp $version from $url"
Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing

# --- sha256 verification --------------------------------------------------
$actual = Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath | Select-Object -ExpandProperty Hash
if ($actual -ne $pinSha) {
  $err = "SHA256 mismatch for ${zipName}: expected $pinSha, got ${actual}"
  Write-Host $err
  Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
  throw $err
}
$actualSize = (Get-Item -LiteralPath $zipPath).Length
if ($actualSize -ne $pinSize) {
  $err = "Size mismatch for ${zipName}: expected $pinSize bytes, got $actualSize bytes"
  Write-Host $err
  Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
  throw $err
}
Write-Host "SHA256 + size verified OK."

# --- extract --------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Expand-Archive -LiteralPath $zipPath -DestinationPath $OutputDir -Force

# --- stamp write ----------------------------------------------------------
$tag | Out-File -FilePath $stampFile -Encoding UTF8

Write-Host "Installed llama.cpp $tag to $OutputDir"
Remove-Item -LiteralPath $staging -Recurse -Force
