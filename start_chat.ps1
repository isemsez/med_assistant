# Toggle launcher for the whole assistant stack: main Ollama (:11434,
# answerer + embeddings), labeler Ollama (:11435, iGPU) and the web app
# (uvicorn, :8866, background by default).
#
# TOGGLE: if ANY managed port is up, re-running STOPS every managed
# component; if all are down, it starts them all. "Safe to re-run" no
# longer means "no-op" - re-running a live stack stops it.
#   powershell -File start_chat.ps1                # start all / stop all
#   powershell -File start_chat.ps1 -SkipCanary    # CLI full start: no offer
#   powershell -File start_chat.ps1 -SkipWeb       # Ollama backends only
#   powershell -File start_chat.ps1 -Foreground    # start with uvicorn in
#                                                  # this console; Ctrl+C
#                                                  # stops only the web app
#   powershell -File start_chat.ps1 -Canary        # force the startup canary
#
# Launch modes (owner decision 2026-10-04): BARE (no arguments) = hand-click
# (Explorer / Total Commander double-click): a 5 s canary offer appears
# after the backends start (any key runs it, timeout skips) and the console
# lingers before closing on its own (10 s after a start, 2 s after a
# stop). ANY argument = deliberate CLI run: no offer, no linger.
# Start sequence: both Ollama instances -> (optional) startup canary ->
# web app -> open http://127.0.0.1:8866 in the default browser. The canary
# (~40 s: health checks + one E2E question; it also warms the models, so
# the first real question is faster) runs before the web app; on failure
# the stack still comes up and the script exits 1.
# Background components run windowless; every start writes per-boot logs
# (logs/ollama-<inst>-<stamp>.*.log, logs/web-<stamp>.*.log) and prunes old
# boots, keeping the 5 latest per component.
# Stopping is a forced Stop-Process (no graceful shutdown) - acceptable for a
# dev tool, the SQLite session store is crash-safe. Only python/uvicorn and
# ollama processes are ever killed; anything else on a managed port is
# reported and left alone.
# Requires (once, see AGENTS.md): setx OLLAMA_IGPU_ENABLE 1,
# setx OLLAMA_MAX_LOADED_MODELS 2, models pulled, .venv built.
param([switch]$Foreground, [switch]$SkipWeb, [switch]$Canary, [switch]$SkipCanary)
$ErrorActionPreference = "Stop"

# bare invocation = hand-clicked (canary offer + exit linger); any argument
# = deliberate CLI run (no offer, no linger)
$NoArgs = ($PSBoundParameters.Count -eq 0)

$root = $PSScriptRoot
$webPort = 8866
$ollamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
$py = "$root\.venv\Scripts\python.exe"

function Quit([int]$code = 0, [double]$pauseSec = 10) {
    # linger on hand-clicked (bare) runs so the messages can be read
    # (10 s after a start, 2 s after a stop), then close on its own;
    # CLI runs exit immediately
    if ($NoArgs -and $pauseSec -gt 0) {
        ""
        "closing in $($pauseSec)s..."
        Start-Sleep -Milliseconds ([int]($pauseSec * 1000))
    }
    exit $code
}

if (-not (Test-Path $ollamaExe)) { Write-Warning "ollama.exe not found: $ollamaExe"; Quit 1 }
if (-not (Test-Path "$root\.env")) { Write-Warning ".env missing - copy .env.example .env" }
if (-not (Test-Path "$root\logs")) { New-Item -ItemType Directory "$root\logs" | Out-Null }
$bootStamp = Get-Date -Format "yyyyMMdd-HHmmss"

# one-time cleanup: pre-toggle log names (ollama-main.out.log etc.) that the
# per-boot pruning filter no longer matches
"ollama-main.out.log", "ollama-main.err.log",
"ollama-labeler.out.log", "ollama-labeler.err.log" | ForEach-Object {
    Remove-Item "$root\logs\$_" -Force -ErrorAction SilentlyContinue
}

function Test-PortOpen([int]$port) {
    $c = [Net.Sockets.TcpClient]::new()
    try { return $c.ConnectAsync("127.0.0.1", $port).Wait(500) -and $c.Connected }
    finally { $c.Dispose() }
}
function Wait-PortOpen([int]$port, [int]$timeoutSec = 60) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-PortOpen $port) { return }
        Start-Sleep -Milliseconds 500
    }
    Write-Warning "port $port did not open within ${timeoutSec}s"; Quit 1
}

function Request-CanaryRun([int]$seconds = 4) {
    # bare-run only: brief opt-in window - any key runs the canary,
    # timeout (or a non-interactive console) skips it. NB: every message
    # must go to the HOST stream - a bare string would pollute the return
    # value (function output = array) and make "skip" truthy.
    Write-Host ""
    Write-Host "canary: press ANY KEY to run. Skip in ${seconds}sec"
    Write-Host ""
    $deadline = (Get-Date).AddSeconds($seconds)
    $run = $false
    try {
        while ((Get-Date) -lt $deadline) {
            if ([Console]::KeyAvailable) {
                $null = [Console]::ReadKey($true)
                $run = $true
                break
            }
            $left = [int][Math]::Ceiling(($deadline - (Get-Date)).TotalSeconds)
            Write-Host "`r  skipping the canary in ${left}s...   " -NoNewline
            Start-Sleep -Milliseconds 200
        }
    }
    catch { }   # input redirected / no real console: fall through to skip
    if ($run) { Write-Host "`r  canary will run                       " }
    else {
        Write-Host "`r  canary skipped                         `r"
        Write-Host ""
    }
    return $run
}

function Start-BrowserWhenUp([int]$port, [int]$timeoutSec = 30) {
    # -Foreground helper: this script blocks on uvicorn right after, so a
    # detached hidden shell polls the port and opens the default browser
    # only once it binds (never on timeout)
    $cmd = @"
`$deadline = (Get-Date).AddSeconds($timeoutSec)
`$up = `$false
while ((Get-Date) -lt `$deadline) {
    `$c = [Net.Sockets.TcpClient]::new()
    try {
        if (`$c.ConnectAsync('127.0.0.1', $port).Wait(300) -and `$c.Connected) { `$up = `$true; break }
    }
    catch { }
    finally { `$c.Dispose() }
    Start-Sleep -Milliseconds 300
}
if (`$up) { Start-Process 'http://127.0.0.1:$port' }
"@
    Start-Process (Get-Process -Id $PID).Path -WindowStyle Hidden `
        -ArgumentList '-NoProfile', '-Command', $cmd
}

function Start-OllamaHidden([string]$inst) {
    # windowless with logs: Windows Terminal (default host) ignores
    # -WindowStyle Minimized and pops a foreground window, so redirect
    # both streams to files instead (separate files: Start-Process
    # cannot share one for stdout+stderr)
    Start-Process $ollamaExe -ArgumentList "serve" -WindowStyle Hidden `
        -RedirectStandardOutput "$root\logs\ollama-$inst-$bootStamp.out.log" `
        -RedirectStandardError "$root\logs\ollama-$inst-$bootStamp.err.log"
    Remove-OldBootLogs "ollama-$inst"
}

function Remove-OldBootLogs([string]$prefix, [int]$keep = 5) {
    # logs are per-boot stamped; keep the $keep newest boots of this prefix
    $files = @(Get-ChildItem "$root\logs" -Filter "$prefix-*.log" -File -ErrorAction SilentlyContinue)
    $stamps = @($files | ForEach-Object {
            if ($_.Name -match "^$prefix-(.+)\.(out|err)\.log$") { $Matches[1] }
        } | Select-Object -Unique)
    $old = @($stamps | Sort-Object -Descending | Select-Object -Skip $keep)
    foreach ($s in $old) {
        Get-ChildItem "$root\logs" -Filter "$prefix-$s.*.log" -File | Remove-Item -Force
    }
}

function Stop-ByPort([int]$port, [string]$procPattern, [string]$label) {
    $conns = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
    $procIds = @($conns | Select-Object -ExpandProperty OwningProcess -Unique)
    if ($procIds.Count -eq 0) { "  ${label} :${port} already down"; return }
    foreach ($procId in $procIds) {
        if ($procId -le 0) { continue }
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($null -eq $p) { continue }
        if ($p.ProcessName -like $procPattern) {
            Stop-Process -Id $procId -Force
            "  stopped $($p.ProcessName) pid $procId (${label}, :${port})"
        }
        else {
            Write-Warning "  :${port} is held by '$($p.ProcessName)' pid $procId (not ${procPattern}) - left alone"
        }
    }
}

# Managed components (web excluded by -SkipWeb in both directions).
$components = @(
    @{ Port = $webPort; Label = "web app";        Proc = "python*" }
    @{ Port = 11434;   Label = "main Ollama";     Proc = "ollama*" }
    @{ Port = 11435;   Label = "labeler Ollama";  Proc = "ollama*" }
)
if ($SkipWeb) { $components = @($components | Where-Object { $_.Port -ne $webPort }) }

# --- toggle: if anything managed is up, stop everything managed ------------
$up = @($components | Where-Object { Test-PortOpen $_.Port })
if ($up.Count -gt 0) {
    if ($Foreground) {
        Write-Warning "-Foreground applies only to starting; stopping instead."
    }
    "Stopping assistant (re-run to start again):"
    foreach ($c in $components) { Stop-ByPort $c.Port $c.Proc $c.Label }
    Quit 0 2
}

# --- start everything -------------------------------------------------------
"Starting assistant:"
Write-Host ""

# 1) main instance (answerer medgemma-cpu + bge-m3), default host :11434.
#    MUST run before the labeler env vars are set - otherwise it would
#    inherit OLLAMA_HOST=:11435.
if (Test-PortOpen 11434) { "main Ollama :11434 already up" }
else {
    Start-OllamaHidden "main"
    Wait-PortOpen 11434; "main Ollama :11434 started"
}

# 2) labeler instance, iGPU/Vulkan, :11435. The web app does NOT read these
#    vars (Python config uses its own *_BASE_URL keys), so inheriting them
#    is harmless.
if (Test-PortOpen 11435) { "labeler Ollama :11435 already up" }
else {
    $env:OLLAMA_HOST = "127.0.0.1:11435"
    $env:OLLAMA_VULKAN = "true"
    $env:OLLAMA_IGPU_ENABLE = "1"
    Start-OllamaHidden "labeler"
    Wait-PortOpen 11435; "labeler Ollama :11435 started"
}

if (-not (Test-Path $py)) { Write-Warning ".venv not found - see AGENTS.md Setup"; Quit 1 }

# 3) startup canary (owner decision 2026-10-04). The 5 s offer appears only
#    on BARE runs (hand-click); any argument means CLI and skips it,
#    -Canary forces the run. Runs before the web app: the whole chain is
#    verified (and the models warmed) first; on failure the stack still
#    comes up and the script exits 1 at the very end.
$runCanary = [bool]$Canary
if (-not $runCanary -and -not $SkipCanary -and $NoArgs) { $runCanary = Request-CanaryRun 5 }
$canaryFailed = $false
if ($runCanary) {
    "running canary (health checks + one E2E question, ~40 s)..."
    Push-Location $root
    try { & $py -m medassistant canary }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) {
        $canaryFailed = $true
        Write-Warning ("CANARY FAILED - the stack still comes up; check the " +
                       "output above, /admin and logs/ollama-*.log")
    }
}

# 4) web app. Run from $root: the package is not installed, uvicorn imports
#    medassistant from the repo root via CWD on sys.path. Once the port
#    binds, the URL opens in the default browser.
if ($SkipWeb) {
    "skipping web app (-SkipWeb)"
    if ($canaryFailed) { Quit 1 }
    Quit 0
}
if ($Foreground) {
    "web app in foreground (Ctrl+C stops only the web app)"
    Start-BrowserWhenUp $webPort
    Push-Location $root
    try { & $py -m uvicorn medassistant.server:app --host 127.0.0.1 --port $webPort }
    finally { Pop-Location }
}
else {
    Start-Process $py `
        -ArgumentList "-m","uvicorn","medassistant.server:app","--host","127.0.0.1","--port",$webPort `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput "$root\logs\web-$bootStamp.out.log" `
        -RedirectStandardError "$root\logs\web-$bootStamp.err.log"
    Remove-OldBootLogs "web"
    Wait-PortOpen $webPort
    Start-Process "http://127.0.0.1:$webPort"
    "web app started -> http://127.0.0.1:$webPort (opened in the default browser;"
    "logs: logs\web-$bootStamp.out.log / .err.log)"
}
if ($canaryFailed) { Quit 1 }
Quit 0
