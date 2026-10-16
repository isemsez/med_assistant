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
# (Explorer / Total Commander double-click): a 4 s canary offer appears
# after the backends start (any key runs it, timeout skips); a SMOOTH run
# closes the console on its own after a short linger (10 s after a start,
# 2 s after a stop), on ANY error the window stays open until a keypress -
# no auto-close on error (owner decision 2026-10-07). ANY argument =
# deliberate CLI run: no offer, no linger.
# Start sequence: both Ollama instances -> (optional) startup canary ->
# web app -> open http://127.0.0.1:8866 in the default browser. The canary
# (~40 s: health checks + one E2E question; it also warms the models, so
# the first real question is faster) runs before the web app; on failure
# the stack still comes up and the script exits 1.
# Background components run windowless; every start writes per-boot logs
# (logs/ollama-<inst>-<stamp>.*.log, logs/web-<stamp>.*.log) and prunes old
# boots, keeping the 5 latest per component.
# Stopping is a forced Stop-Process (no graceful shutdown) - acceptable for a
# dev tool, the SQLite session store is crash-safe. A stop kills the whole
# process TREE rooted at each managed listener: ollama.exe serve runs every
# loaded model in a child llama-server.exe (GBs of RAM), and a parent-only
# kill orphans the runners. Descendants are only killed when their executable
# lives under the component's install dir (Ollama dir / .venv), and
# orphaned llama-server.exe runners (dead parent) are swept on every start
# and stop. Anything else on a managed port is reported and left alone.
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
    # exit-time behavior, BARE (hand-clicked) runs only; CLI runs exit
    # immediately. Smooth runs linger (10 s after a start, 2 s after a
    # stop) then close on their own; ERROR runs (non-zero exit) never
    # auto-close - they wait for a keypress (buffered keys are drained
    # first) so the output can be read; a non-interactive console (input
    # redirected) falls back to a bounded 60 s linger
    if ($NoArgs) {
        if ($code -ne 0) {
            ""
            Write-Host "ERROR - press ANY KEY to close..." -ForegroundColor Red
            try {
                while ([Console]::KeyAvailable) { $null = [Console]::ReadKey($true) }
                while (-not [Console]::KeyAvailable) { Start-Sleep -Milliseconds 200 }
                $null = [Console]::ReadKey($true)
            }
            catch { Start-Sleep -Seconds 60 }
        }
        elseif ($pauseSec -gt 0) {
            ""
            "closing in $($pauseSec)s..."
            Start-Sleep -Milliseconds ([int]($pauseSec * 1000))
        }
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
function Wait-PortOpen([int]$port, [int]$timeoutSec = 60, [switch]$NoQuit) {
    # returns $true once the port binds; on timeout warns and exits through
    # Quit unless -NoQuit (the caller then kills the process it started
    # instead of orphaning it)
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-PortOpen $port) { return $true }
        Start-Sleep -Milliseconds 500
    }
    Write-Warning "port $port did not open within ${timeoutSec}s"
    if ($NoQuit) { return $false }
    Quit 1
}

function Wait-PortClosed([int]$port, [int]$timeoutSec = 8) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (-not (Test-PortOpen $port)) { return $true }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

function Request-CanaryRun([int]$seconds = 4) {
    # bare-run only: brief opt-in window - any key runs the canary,
    # timeout (or a non-interactive console) skips it. NB: every message
    # must go to the HOST stream - a bare string would pollute the return
    # value (function output = array) and make "skip" truthy.
    Write-Host ""
    Write-Host "canary: press ANY KEY to run."
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
            Write-Host "`r  skipping in ${left}s...   " -NoNewline
            Start-Sleep -Milliseconds 200
        }
    }
    catch { }   # input redirected / no real console: fall through to skip
    if ($run) {
        # erase the countdown line - the next printed line (the canary banner)
        # takes its place; no extra "will run" announcement
        Write-Host ("`r" + (" " * 40) + "`r") -NoNewline
    }
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

function Start-DetachedHidden([string]$exe, [string]$psArgList, [string]$stdout, [string]$stderr, [string]$workdir) {
    # Start a long-lived server so it is NEVER a live child of this script:
    # a short-lived hidden helper pwsh Start-Process'es the real server
    # (hidden + file-redirected, env inherited through the helper) and
    # exits immediately with the server's PID as its exit code. Direct
    # Start-Process children of this script made agent/CI shells hang on
    # the start call or kill the "leaked" servers at cleanup
    # (2026-10-14); the helper breaks the parent link before this script
    # returns, so no live descendant remains to track. Returns the PID
    # (0 on failure).
    $q = { param($s) "'" + ($s -replace "'", "''") + "'" }
    $inner = @"
`$ErrorActionPreference = 'Stop'
try {
    `$p = Start-Process $($q.Invoke($exe)) -ArgumentList $($psArgList) -WindowStyle Hidden -PassThru -WorkingDirectory $($q.Invoke($workdir)) -RedirectStandardOutput $($q.Invoke($stdout)) -RedirectStandardError $($q.Invoke($stderr))
    exit `$p.Id
}
catch { exit 0 }
"@
    $b64 = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($inner))
    $helper = Start-Process (Get-Process -Id $PID).Path -WindowStyle Hidden `
        -ArgumentList "-NoProfile", "-EncodedCommand", $b64 -PassThru
    if (-not $helper.WaitForExit(15000)) {
        Write-Warning "detached launcher helper did not exit"
        Stop-Process -Id $helper.Id -Force -ErrorAction SilentlyContinue
        return 0
    }
    return [int]$helper.ExitCode
}

function Start-OllamaHidden([string]$inst) {
    # windowless with logs (separate files: Start-Process cannot share
    # one for stdout+stderr), fully detached from this script's tree
    $procId = Start-DetachedHidden $ollamaExe "@('serve')" `
        "$root\logs\ollama-$inst-$bootStamp.out.log" `
        "$root\logs\ollama-$inst-$bootStamp.err.log" "$root"
    if ($procId -gt 0) { Remove-OldBootLogs "ollama-$inst" }
    return $procId
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

function Stop-Children([int]$procId, [string]$exeDir) {
    # kill every descendant of $procId, deepest first. ollama.exe serve runs
    # each loaded model in a child llama-server.exe process; killing only the
    # listener orphans the runners with their GBs of model RAM. A descendant
    # is only killed when its executable lives under $exeDir, so a mixed
    # tree can never harm unrelated processes.
    $kids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $procId" -ErrorAction SilentlyContinue)
    foreach ($k in $kids) {
        Stop-Children ([int]$k.ProcessId) $exeDir
        if ($k.ExecutablePath -and
            $k.ExecutablePath.StartsWith($exeDir, [StringComparison]::OrdinalIgnoreCase)) {
            Stop-Process -Id $k.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
}

function Stop-OrphanRunners([string]$ollamaDir) {
    # sweep llama-server.exe model runners whose parent ollama.exe is already
    # dead - garbage left behind by earlier stops or crashes; matched by
    # executable path so nothing unrelated is ever touched
    $runners = @(Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.ExecutablePath -and
            $_.ExecutablePath.StartsWith($ollamaDir, [StringComparison]::OrdinalIgnoreCase) })
    foreach ($r in $runners) {
        $parentAlive = ($r.ParentProcessId -gt 0) -and
            [bool](Get-Process -Id $r.ParentProcessId -ErrorAction SilentlyContinue)
        if (-not $parentAlive) {
            Stop-Process -Id $r.ProcessId -Force -ErrorAction SilentlyContinue
            "  swept orphaned runner pid $($r.ProcessId) (no parent ollama)"
        }
    }
}

function Stop-ByPort([int]$port, [string]$procPattern, [string]$label, [string]$exeDir) {
    $conns = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
    $procIds = @($conns | Select-Object -ExpandProperty OwningProcess -Unique)
    if ($procIds.Count -eq 0) { "  ${label} :${port} already down"; return }
    foreach ($procId in $procIds) {
        if ($procId -le 0) { continue }
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($null -eq $p) { continue }
        if ($p.ProcessName -like $procPattern) {
            Stop-Children $procId $exeDir
            Stop-Process -Id $procId -Force
            "  stopped $($p.ProcessName) pid $procId + children (${label}, :${port})"
        }
        else {
            Write-Warning "  :${port} is held by '$($p.ProcessName)' pid $procId (not ${procPattern}) - left alone"
        }
    }
}

# --- main flow (try/catch: uncaught terminating errors must route through
# --- Quit, or a bare-run window would vanish instantly on them) -------------
try {
    # Managed components (web excluded by -SkipWeb in both directions). Dir =
    # install dir used to scope descendant kills to this stack's binaries.
    $ollamaDir = Split-Path $ollamaExe
    $venvDir = "$root\.venv"
    $components = @(
        @{ Port = $webPort; Label = "web app";        Proc = "python*"; Dir = $venvDir }
        @{ Port = 11434;   Label = "main Ollama";     Proc = "ollama*"; Dir = $ollamaDir }
        @{ Port = 11435;   Label = "labeler Ollama";  Proc = "ollama*"; Dir = $ollamaDir }
    )
    if ($SkipWeb) { $components = @($components | Where-Object { $_.Port -ne $webPort }) }

    # --- toggle: if anything managed is up, stop everything managed --------
    $up = @($components | Where-Object { Test-PortOpen $_.Port })
    if ($up.Count -gt 0) {
        if ($Foreground) {
            Write-Warning "-Foreground applies only to starting; stopping instead."
        }
        "Stopping assistant:"
        foreach ($c in $components) { Stop-ByPort $c.Port $c.Proc $c.Label $c.Dir }
        Stop-OrphanRunners $ollamaDir
        foreach ($c in $components) {
            if (-not (Wait-PortClosed $c.Port)) {
                Write-Warning "  :$($c.Port) ($($c.Label)) still open after stopping"
            }
        }
        Quit 0 2
    }

    # --- start everything ---------------------------------------------------
    "Starting assistant:"
    Write-Host ""

    # sweep runners orphaned by earlier stops/crashes before booting
    Stop-OrphanRunners $ollamaDir

    # 1) main instance (answerer medgemma-cpu + bge-m3), default host :11434.
    #    MUST run before the labeler env vars are set - otherwise it would
    #    inherit OLLAMA_HOST=:11435.
    if (Test-PortOpen 11434) { "main Ollama :11434 already up" }
    else {
        $procId = Start-OllamaHidden "main"
        if ($procId -le 0) { Quit 1 }
        if (-not (Wait-PortOpen 11434 -NoQuit)) {
            Stop-Children $procId $ollamaDir
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            Quit 1
        }
        "main Ollama    :11434 started"
    }

    # 2) labeler instance, iGPU/Vulkan, :11435. The web app does NOT read these
    #    vars (Python config uses its own *_BASE_URL keys), so inheriting them
    #    is harmless.
    if (Test-PortOpen 11435) { "labeler Ollama :11435 already up" }
    else {
        $env:OLLAMA_HOST = "127.0.0.1:11435"
        $env:OLLAMA_VULKAN = "true"
        $env:OLLAMA_IGPU_ENABLE = "1"
        $procId = Start-OllamaHidden "labeler"
        if ($procId -le 0) { Quit 1 }
        if (-not (Wait-PortOpen 11435 -NoQuit)) {
            Stop-Children $procId $ollamaDir
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            Quit 1
        }
        "labeler Ollama :11435 started"
    }

    if (-not (Test-Path $py)) { Write-Warning ".venv not found - see AGENTS.md Setup"; Quit 1 }

    # 3) startup canary (owner decision 2026-10-04). The 4 s offer appears only
    #    on BARE runs (hand-click); any argument means CLI and skips it,
    #    -Canary forces the run. Runs before the web app: the whole chain is
    #    verified (and the models warmed) first; on failure the stack still
    #    comes up and the script exits 1 at the very end.
    $runCanary = [bool]$Canary
    if (-not $runCanary -and -not $SkipCanary -and $NoArgs) { $runCanary = Request-CanaryRun 4 }
    $canaryFailed = $false
    if ($runCanary) {
        "running canary (health checks + one E2E question, ~40 s)..."
        ""
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
        $procId = Start-DetachedHidden $py `
            "@('-m','uvicorn','medassistant.server:app','--host','127.0.0.1','--port','$webPort')" `
            "$root\logs\web-$bootStamp.out.log" "$root\logs\web-$bootStamp.err.log" $root
        if ($procId -le 0) { Quit 1 }
        Remove-OldBootLogs "web"
        if (-not (Wait-PortOpen $webPort -NoQuit)) {
            Stop-Children $procId $venvDir
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            Quit 1
        }
        Start-Process "http://127.0.0.1:$webPort"
        "web app started -> http://127.0.0.1:$webPort (opened in the default browser;"
        "logs: logs\web-$bootStamp.out.log / .err.log)"
    }
    if ($canaryFailed) { Quit 1 }
    Quit 0
}
catch {
    # uncaught terminating error: report and exit through Quit so bare
    # runs wait for a keypress instead of the window closing instantly
    Write-Host ""
    Write-Host "UNEXPECTED ERROR: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.ScriptStackTrace) { Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray }
    Quit 1
}
