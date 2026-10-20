<#
.SYNOPSIS
    Reproducibly builds the portable CPython embeddable runtime (python\) used by
    med_assistant's launcher (stdlib-only entry, but the runtime itself is a
    full embeddable CPython + freeze-pinned site-packages).

.DESCRIPTION
    Downloads the official CPython embeddable package, extracts it to -Target
    (default: <repo>\python), enables site-packages via python3*._pth,
    bootstraps pip via get-pip.py, then installs the pinned runtime
    requirements (deploy/requirements.txt). Use -IncludeDev to also install
    deploy/requirements-dev.txt.

    The stock embeddable ships WITHOUT pip, so get-pip.py is used to bootstrap
    it. Network access to python.org and PyPI is required. Target machine is
    offline by design — python\ is baked into the zip once here.

.PARAMETER Target
    Destination directory for the runtime. Defaults to <repo>\python. Refuses
    to clobber a non-empty existing directory unless -Force is given.

.PARAMETER PythonVersion
    Exact CPython version to fetch from python.org. Default: 3.13.5.

.PARAMETER IncludeDev
    Also install deploy/requirements-dev.txt (pytest). Not needed for the
    deploy zip; the embedded runtime ships runtime-only.

.PARAMETER Force
    Allow proceeding when -Target already exists and is non-empty.

.EXAMPLE
    pwsh deploy/bootstrap_runtime.ps1
    # Rebuilds <repo>\python from scratch (errors if python/ already populated; add -Force).

.EXAMPLE
    pwsh deploy/bootstrap_runtime.ps1 -Target "$env:TEMP\kilo\py_repro" -IncludeDev
    # Safe test build into a throwaway location without touching the live python/.
#>
[CmdletBinding()]
param(
    [string]$Target,
    [string]$PythonVersion = '3.13.5',
    [switch]$IncludeDev,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

# TLS: ensure modern protocols for downloads on older PowerShell
try {
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch {
    # PowerShell 7 / .NET Core: managed independently, ignore.
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Target) { $Target = Join-Path $RepoRoot 'python' }
$Requirements    = Join-Path $PSScriptRoot 'requirements.txt'
$RequirementsDev = Join-Path $PSScriptRoot 'requirements-dev.txt'

$EmbedUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$GetPipUrl = 'https://bootstrap.pypa.io/get-pip.py'
$TempRoot = Join-Path $env:TEMP 'kilo\med_assistant_bootstrap'
$EmbedZip = Join-Path $TempRoot "python-$PythonVersion-embed-amd64.zip"
$GetPipPy  = Join-Path $TempRoot 'get-pip.py'

Write-Host "Repo root     : $RepoRoot"
Write-Host "Target        : $Target"
Write-Host "Python version: $PythonVersion"
Write-Host "Include dev   : $IncludeDev"

# Safety: refuse to clobber a non-empty Target
if (Test-Path -LiteralPath $Target) {
    $existing = Get-ChildItem -LiteralPath $Target -Force -ErrorAction SilentlyContinue
    if ($existing -and $existing.Count -gt 0 -and -not $Force) {
        throw "Target '$Target' exists and is not empty. Pass -Force to overwrite."
    }
    if ($Force) {
        Write-Host "Cleaning existing Target (Force)..."
        Remove-Item -LiteralPath $Target -Recurse -Force
    }
}

New-Item -ItemType Directory -Force -Path $Target | Out-Null
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null

# 1. Download embeddable
if (-not (Test-Path -LiteralPath $EmbedZip)) {
    Write-Host "Downloading $EmbedUrl ..."
    Invoke-WebRequest -Uri $EmbedUrl -OutFile $EmbedZip -UseBasicParsing
} else {
    Write-Host "Reusing cached $EmbedZip"
}

# 2. Extract embeddable into Target
Write-Host "Extracting embeddable into $Target ..."
Expand-Archive -LiteralPath $EmbedZip -DestinationPath $Target -Force

# 3. Enable site-packages in python3*._pth (uncomment 'import site')
$PthFile = Get-ChildItem -LiteralPath $Target -Filter 'python*._pth' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $PthFile) { throw "No python*._pth found in $Target after extraction." }
$pthContent = Get-Content -LiteralPath $PthFile.FullName -Raw
if ($pthContent -match '(?m)^\s*#\s*import\s+site\s*$') {
    $pthContent = $pthContent -replace '(?m)^\s*#\s*import\s+site\s*$', 'import site'
    Set-Content -LiteralPath $PthFile.FullName -Value $pthContent -NoNewline
    Write-Host "Enabled site in $($PthFile.Name) (uncommented 'import site')."
} elseif ($pthContent -match '(?m)^\s*import\s+site\s*$') {
    Write-Host "$($PthFile.Name) already has 'import site' enabled."
} else {
    Add-Content -LiteralPath $PthFile.FullName -Value "`r`nimport site"
    Write-Host "Appended 'import site' to $($PthFile.Name)."
}

# 3b. Put the app root on sys.path. Embedded Python resolves sys.path from the
# ._pth file and ignores CWD/PYTHONPATH, so Start.bat's `-m medassistant.launcher`
# would fail without an explicit relative entry (.. = the extracted zip root).
$pthContent = Get-Content -LiteralPath $PthFile.FullName -Raw
if ($pthContent -notmatch '(?m)^\s*\.\.\s*$') {
    Add-Content -LiteralPath $PthFile.FullName -Value "`r`n.."
    Write-Host "Added '..' to $($PthFile.Name) (app root on sys.path)."
}

$PythonExe = Join-Path $Target 'python.exe'
if (-not (Test-Path -LiteralPath $PythonExe)) { throw "python.exe missing at $PythonExe after extraction." }

# 4. Bootstrap pip via get-pip.py (stock embeddable has none)
if (-not (Test-Path -LiteralPath $GetPipPy)) {
    Write-Host "Downloading $GetPipUrl ..."
    Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPipPy -UseBasicParsing
} else {
    Write-Host "Reusing cached $GetPipPy"
}
Write-Host "Bootstrapping pip ..."
& $PythonExe $GetPipPy --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "get-pip.py failed (exit $LASTEXITCODE)." }

# 5. Install runtime requirements (freeze-pinned; transitives must not float)
if (-not (Test-Path -LiteralPath $Requirements)) {
    throw "requirements file not found: $Requirements"
}
Write-Host "Installing runtime requirements ($Requirements) ..."
& $PythonExe -m pip install --no-warn-script-location -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "pip install (runtime) failed (exit $LASTEXITCODE)." }

# 6. Optional dev requirements
if ($IncludeDev) {
    if (-not (Test-Path -LiteralPath $RequirementsDev)) {
        throw "dev requirements file not found: $RequirementsDev"
    }
    Write-Host "Installing dev requirements ($RequirementsDev) ..."
    & $PythonExe -m pip install --no-warn-script-location -r $RequirementsDev
    if ($LASTEXITCODE -ne 0) { throw "pip install (dev) failed (exit $LASTEXITCODE)." }
}

# 7. Report
Write-Host ""
Write-Host "Versions installed in $Target :"
& $PythonExe -m pip freeze
& $PythonExe -c "import sys; print('python', sys.version.split()[0])"

Write-Host ""
Write-Host "Done. Runtime built at: $Target"
if (-not $IncludeDev) {
    Write-Host "(runtime only; pass -IncludeDev to also install test dependencies)"
}
