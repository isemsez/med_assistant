[CmdletBinding()]
param(
    [string]$StagingRoot = (Join-Path $env:TEMP 'kilo\med_assistant_deploy'),
    [string]$ZipName = 'medassistant.zip',
    [string]$ZipDir,
    [ValidateSet('Optimal', 'Fastest', 'NoCompression')]
    [string]$CompressionLevel = 'Fastest',
    [switch]$KeepStaging
)

$ErrorActionPreference = 'Stop'
$FileAttributeHidden = 0x2
$FileAttributeDirectory = 0x10

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PySrc  = Join-Path $RepoRoot 'python'
$EngSrc = Join-Path $RepoRoot 'engine'
$ModSrc = Join-Path $RepoRoot 'models'
$AppSrc = Join-Path $RepoRoot 'medassistant'
$DataSrc = Join-Path $RepoRoot 'data'

if (-not (Test-Path -LiteralPath (Join-Path $PySrc 'python.exe'))) {
    throw "python.exe not found at '$PySrc'. Run deploy/bootstrap_runtime.ps1 first."
}
if (-not (Test-Path -LiteralPath (Join-Path $EngSrc 'llamacpp\llama-server.exe'))) {
    throw "llama-server.exe not found under engine\. Run deploy/fetch_llamacpp.ps1 first."
}
foreach ($g in 'medgemma1.5-4b-q4km-text-only.gguf', 'bge-m3-f16.gguf') {
    if (-not (Test-Path -LiteralPath (Join-Path $ModSrc $g))) {
        throw "model GGUF not found: models\$g. Run deploy/fetch_bge.ps1 first."
    }
}

$StagingApp = Join-Path $StagingRoot 'med_assistant'
if (-not $ZipDir) { $ZipDir = $RepoRoot }
$ZipPath = Join-Path $ZipDir $ZipName

Write-Host "Repo root : $RepoRoot"
Write-Host "Zip target: $ZipPath"
Write-Host ""
Write-Host "Staging   : $StagingApp"

if (Test-Path -LiteralPath $StagingRoot) {
    Write-Host "Cleaning previous staging..."
    Remove-Item -LiteralPath $StagingRoot -Recurse -Force
}
Write-Host ""
New-Item -ItemType Directory -Force -Path $StagingApp | Out-Null

# --- data\ whitelist guard (owner decision 2026-10-18) -----------------------
# A stray screen recording already sits in data/; never ship anything but the
# whitelist. sessions.sqlite, backups\, logs\ are runtime artifacts.
$DataWhitelist = @(
    'docs_meta.json',
    'clinrec_minzdrav',
    'index',
    'drugs.sqlite',
    'abbr2term.yaml',
    'drug_aliases.yaml',
    'patient_cards'
)
# Local-only entries: runtime artifacts and build-time inputs that must exist
# in the dev tree but are never staged. Anything outside both lists is a stray
# file the whitelist copy would silently ignore — fail loudly instead.
$DataLocalOnly = @(
    'backups',
    'sessions.sqlite',
    'sessions.sqlite-shm',
    'sessions.sqlite-wal',
    'grls.xlsx',
    'inn_gloss.yaml',
    'inn_list_en-lat.txt',
    'Запись 2026-10-17 150627.mp4'
)
$DataTop = Get-ChildItem -LiteralPath $DataSrc -Force -ErrorAction SilentlyContinue
$Bad = $DataTop | Where-Object {
    -not ($_.Name -in $DataWhitelist) -and -not ($_.Name -in $DataLocalOnly)
}
if ($Bad) {
    $names = ($Bad | ForEach-Object { $_.Name }) -join ', '
    throw "data\ contains unexpected entries that the whitelist copy would ignore: $names. "
        + "Remove them, or add them to `$DataLocalOnly` if they are local-only."
}

# --- llama.cpp pin hash (for the dist-manifest) ------------------------------
$PinFile = Join-Path $PSScriptRoot 'llamacpp-pin.txt'
function Get-PinValue {
    param([string]$Content, [string]$Key)
    if ($Content -match "(?m)^\s*$Key\s*=\s*(.+)$") {
        return $matches[1].Trim().Trim("`"","'")
    }
    throw "Missing key: $Key in $PinFile"
}
$pin = Get-Content -LiteralPath $PinFile -Raw -Encoding UTF8
$LlamaCppSha = Get-PinValue $pin 'sha256'
$LlamaCppTag = Get-PinValue $pin 'tag'

# --- index freshness stamp ---------------------------------------------------
$IndexMeta = Join-Path $DataSrc 'index\meta.json'
if (-not (Test-Path -LiteralPath $IndexMeta)) {
    throw "data\index\meta.json missing — run python -m medassistant build-index first."
}
$IndexMetaJson = Get-Content -LiteralPath $IndexMeta -Raw -Encoding UTF8 | ConvertFrom-Json
$IndexStamp = if ($IndexMetaJson.PSObject.Properties.Name -contains 'docs_meta_sha1') {
    $IndexMetaJson.docs_meta_sha1
} else { $null }
if (-not $IndexStamp) {
    throw "data\index\meta.json has no docs_meta_sha1 stamp — rebuild the index."
}
# Verify the index was built against the current docs_meta (freshness gate).
# The stamp must be computed by the same canonical hash gindex.build_index
# writes (json.dumps sort_keys) — recompute here through the real package so
# the representations cannot drift. A stale index fails the build.
$HashScript = @'
import json, sys
sys.path.insert(0, r"""REPO_ROOT_PLACEHOLDER""")
from medassistant import config
from medassistant.gindex import docs_hash
docs = config.load_docs_meta()
print(docs_hash(docs))
'@
$HashScript = $HashScript.Replace('REPO_ROOT_PLACEHOLDER', $RepoRoot)
$ComputedStamp = (& python -c $HashScript).Trim()
if ($LASTEXITCODE -ne 0 -or -not $ComputedStamp) { throw "could not compute current docs_meta hash (need python + the repo package on PATH)." }
$global:LASTEXITCODE = 0
if ($ComputedStamp -ne $IndexStamp) {
    throw "index is STALE: index docs_meta_sha1=$IndexStamp but current docs_meta hashes to $ComputedStamp. "
        + "Re-run: python -m medassistant build-index (pointed at the llama-server bge endpoint :8083)."
}

# --- staging: python\ (selective robocopy, exclusions) -----------------------
Write-Host "Copying python\ (selective robocopy)..."
$PyDst = Join-Path $StagingApp 'python'
$SitePackagesSrc = Join-Path $PySrc 'Lib\site-packages'

# dev packages + pip + scripts never ship in the runtime zip
$DevPkgs = @('pytest','_pytest','pluggy','iniconfig','pygments','colorama','pip','setuptools','wheel')
$XdExclude = New-Object System.Collections.Generic.List[string]
$XdExclude.Add('__pycache__')
if (Test-Path -LiteralPath $SitePackagesSrc) {
    Get-ChildItem -LiteralPath $SitePackagesSrc -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $base = $_.Name -replace '-\d.*\.dist-info$',''
        if ($DevPkgs -icontains $_.Name -or $DevPkgs -icontains $base) {
            $XdExclude.Add($_.FullName)
        }
    }
}
$ScriptsSrc = Join-Path $PySrc 'Scripts'
if (Test-Path -LiteralPath $ScriptsSrc) { $XdExclude.Add($ScriptsSrc) }

# robocopy still emits one stray blank line when fully silenced (/NP /NFL /NDL
# /NJH /NJS) — capture and print only non-blank lines, so the log has no gaps
# while real error text (robocopy reports errors on stdout) stays visible.
$RcOut = robocopy $PySrc $PyDst /E /R:1 /W:1 /MT:16 /NFL /NDL /NJH /NJS /NP /XD $XdExclude /XF '*.pyc' '*.pyo'
$RcOut | Where-Object { $_ -match '\S' } | Write-Host
if ($LASTEXITCODE -ge 8) { throw "robocopy python\ failed: exit code $LASTEXITCODE" }
$global:LASTEXITCODE = 0

# keep python3*._pth byte-identical to the bootstrapped runtime (import site enabled)
# the launcher/hide script (Start.bat invokes it on first run for
# Explorer extracts; Total Commander/WinRAR restore Hidden from the zip)
New-Item -ItemType Directory -Force -Path (Join-Path $StagingApp 'launcher') | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'hide_coding_dirs.bat') `
    -Destination (Join-Path $StagingApp 'launcher\hide_coding_dirs.bat') -Force

# --- staging: engine\ (llama-server + dlls) ----------------------------------
Write-Host "Copying engine\ ..."
$EngDst = Join-Path $StagingApp 'engine'
$RcOut = robocopy $EngSrc $EngDst /E /R:1 /W:1 /MT:16 /NFL /NDL /NJH /NJS /NP /XF 'STAMP.txt'
$RcOut | Where-Object { $_ -match '\S' } | Write-Host
if ($LASTEXITCODE -ge 8) { throw "robocopy engine\ failed: exit code $LASTEXITCODE" }
$global:LASTEXITCODE = 0

# --- staging: models\ (GGUFs + jinja templates only) -------------------------
Write-Host "Copying models\ (whitelisted files)..."
$ModDst = Join-Path $StagingApp 'models'
New-Item -ItemType Directory -Force -Path $ModDst | Out-Null
foreach ($f in 'medgemma1.5-4b-q4km-text-only.gguf', 'bge-m3-f16.gguf',
             'answerer.jinja', 'labeler.jinja') {
    Copy-Item -LiteralPath (Join-Path $ModSrc $f) -Destination $ModDst -Force
}

# --- staging: medassistant\ package (.py + server\static) --------------------
Write-Host "Copying medassistant\ (.py + server\static)..."
$AppDst = Join-Path $StagingApp 'medassistant'
New-Item -ItemType Directory -Force -Path $AppDst | Out-Null
$AppModules = @(
    '__init__.py','__main__.py','abbr.py','appendix.py','audit.py','canary.py',
    'cards.py','cli.py','config.py','drugmap.py','gindex.py','health.py',
    'launcher.py','llm.py','loopguard.py','pipeline.py','prompts.py',
    'regression.py','stats.py','store.py'
)
foreach ($m in $AppModules) {
    Copy-Item -LiteralPath (Join-Path $AppSrc $m) -Destination $AppDst -Force
}
$ServerDst = Join-Path $AppDst 'server'
New-Item -ItemType Directory -Force -Path $ServerDst | Out-Null
Copy-Item -LiteralPath (Join-Path $AppSrc 'server\__init__.py') -Destination $ServerDst -Force
$StaticDst = Join-Path $AppDst 'server\static'
New-Item -ItemType Directory -Force -Path $StaticDst | Out-Null
$RcOut = robocopy (Join-Path $AppSrc 'server\static') $StaticDst /E /R:1 /W:1 /MT:16 /NFL /NDL /NJH /NJS /NP
$RcOut | Where-Object { $_ -match '\S' } | Write-Host
if ($LASTEXITCODE -ge 8) { throw "robocopy server\static failed: exit code $LASTEXITCODE" }
$global:LASTEXITCODE = 0

# --- staging: data\ by whitelist only ----------------------------------------
Write-Host "Copying data\ (whitelist only)..."
$DataDst = Join-Path $StagingApp 'data'
New-Item -ItemType Directory -Force -Path $DataDst | Out-Null
foreach ($w in $DataWhitelist) {
    $src = Join-Path $DataSrc $w
    $dst = Join-Path $DataDst $w
    if (-not (Test-Path -LiteralPath $src)) {
        throw "whitelisted data entry missing: data\$w"
    }
    if ((Get-Item -LiteralPath $src).PSIsContainer) {
        $RcOut = robocopy $src $dst /E /R:1 /W:1 /MT:16 /NFL /NDL /NJH /NJS /NP
        $RcOut | Where-Object { $_ -match '\S' } | Write-Host
        if ($LASTEXITCODE -ge 8) { throw "robocopy data\$w failed: exit code $LASTEXITCODE" }
        $global:LASTEXITCODE = 0
    } else {
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }
}

# --- .env + Start.bat + README.txt at zip root --------------------------------
Write-Host "Copying .env, Start.bat and README.txt ..."
Copy-Item -LiteralPath (Join-Path $RepoRoot '.env') -Destination (Join-Path $StagingApp '.env') -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot 'Start.bat') -Destination (Join-Path $StagingApp 'Start.bat') -Force
$Readme = Join-Path $PSScriptRoot 'README.txt'
if (Test-Path -LiteralPath $Readme) {
    Copy-Item -LiteralPath $Readme -Destination (Join-Path $StagingApp 'README.txt') -Force
}

# --- dist-manifest.json (pin hash + index stamp) -----------------------------
$Manifest = [ordered]@{
    llamacpp_tag     = $LlamaCppTag
    llamacpp_sha256  = $LlamaCppSha
    index_stamp      = $IndexStamp
    index_rows       = if ($IndexMetaJson.PSObject.Properties.Name -contains 'rows') {
        $IndexMetaJson.rows.Count
    } else { $null }
    index_dim        = if ($IndexMetaJson.PSObject.Properties.Name -contains 'dim') {
        $IndexMetaJson.dim
    } else { $null }
    built_at         = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssZ')
}
$Manifest | ConvertTo-Json -Depth 10 |
    Set-Content -LiteralPath (Join-Path $StagingApp 'dist-manifest.json') -Encoding UTF8 -NoNewline

# --- mark internals Hidden on staging ----------------------------------------
# Everything except Start.bat and README.txt (matching hide_coding_dirs.bat;
# logs\ is a runtime artifact and cannot be staged). archivers that restore
# attributes from the zip flag these directly; Explorer extracts rely on
# launcher\hide_coding_dirs.bat, invoked by Start.bat on every run.
Write-Host ""
Write-Host "Marking internals Hidden on staging..."
$HideDirs = @('python', 'engine', 'medassistant', 'data', 'launcher', 'models')
$HideFiles = @('.env', 'dist-manifest.json')
foreach ($h in $HideDirs) {
    $p = Join-Path $StagingApp $h
    if (Test-Path -LiteralPath $p) { & attrib +h "$p" /d }
}
foreach ($f in $HideFiles) {
    $p = Join-Path $StagingApp $f
    if (Test-Path -LiteralPath $p) { & attrib +h "$p" }
}

# --- create UTF-8 zip, top folder med_assistant\ ------------------------------
Write-Host "Creating zip (UTF-8, top folder `"med_assistant`", compression $CompressionLevel)..."
Add-Type -AssemblyName System.IO.Compression.FileSystem
if (Test-Path -LiteralPath $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }
$ZipSw = [System.Diagnostics.Stopwatch]::StartNew()
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $StagingApp, $ZipPath,
    [System.IO.Compression.CompressionLevel]$CompressionLevel,
    $true,
    [System.Text.Encoding]::UTF8)
$ZipSw.Stop()
Write-Host "Zip written in $([math]::Round($ZipSw.Elapsed.TotalSeconds, 1)) s."

# --- embed Hidden flag on the internal folder/file entries --------------------
# CreateFromDirectory writes no directory entries for non-empty dirs, so the
# per-dir entry is re-created if missing. .env/dist-manifest.json are file
# entries (no trailing slash, no Directory flag).
Write-Host "Embedding Hidden flag on internal entries ($($HideDirs -join ','), $($HideFiles -join ','))..."
$archive = [System.IO.Compression.ZipFile]::Open($ZipPath, [System.IO.Compression.ZipArchiveMode]::Update)
try {
    foreach ($name in ($HideDirs + $HideFiles)) {
        $p = Join-Path $StagingApp $name
        if (-not (Test-Path -LiteralPath $p)) { continue }
        $isDir = (Get-Item -LiteralPath $p -Force).PSIsContainer
        $rel = $p.Substring($StagingRoot.Length).TrimStart('\','/').Replace('\','/')
        if ($isDir) { $rel += '/' }
        $entry = $archive.GetEntry($rel)
        if (-not $entry) {
            $entry = $archive.CreateEntry($rel, [System.IO.Compression.CompressionLevel]::NoCompression)
        }
        $flags = $FileAttributeHidden
        if ($isDir) { $flags = $flags -bor $FileAttributeDirectory }
        $entry.ExternalAttributes = ([int]$entry.ExternalAttributes) -bor $flags
    }
} finally { $archive.Dispose() }

if (-not $KeepStaging) {
    Remove-Item -LiteralPath $StagingApp -Recurse -Force
}

$sizeMB = [math]::Round((Get-Item -LiteralPath $ZipPath).Length / 1MB, 2)
Write-Host ""
Write-Host "Done. Built: $ZipPath ($sizeMB MB)"
Write-Host "dist-manifest: llamacpp $LlamaCppTag ($LlamaCppSha), index stamp $IndexStamp"
Write-Host "Hidden internals: $(($HideDirs + $HideFiles) -join ', ') entries carry Hidden (logs\ hidden by hide_coding_dirs.bat at runtime)."
Write-Host "(Total Commander / WinRAR restore it on extract). For Explorer/.NET extracts,"
Write-Host "run hide_coding_dirs.bat once after unzip."
