# One-shot launcher: main Ollama (:11434, answerer + embeddings), labeler
# Ollama (:11435, iGPU) and the web app (uvicorn, foreground).
# Safe to re-run: anything already listening is left alone.
# Ctrl+C stops only the web app; the Ollama instances keep running.
# Ollama instances run windowless; console output goes to
# logs/ollama-main.*.log / logs/ollama-labeler.*.log (truncated per start).
# Requires (once, see README): setx OLLAMA_IGPU_ENABLE 1,
# setx OLLAMA_MAX_LOADED_MODELS 2, models pulled, .venv built.
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$webPort = 8866
$ollamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollamaExe)) { Write-Error "ollama.exe not found: $ollamaExe"; exit 1 }
if (-not (Test-Path "$root\.env")) { Write-Warning ".env missing - copy .env.example .env" }
if (-not (Test-Path "$root\logs")) { New-Item -ItemType Directory "$root\logs" | Out-Null }

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
    Write-Error "port $port did not open within ${timeoutSec}s"; exit 1
}

function Start-OllamaHidden([string]$logPrefix) {
    # windowless with logs: Windows Terminal (default host) ignores
    # -WindowStyle Minimized and pops a foreground window, so redirect
    # both streams to files instead (separate files: Start-Process
    # cannot share one for stdout+stderr)
    Start-Process $ollamaExe -ArgumentList "serve" -WindowStyle Hidden `
        -RedirectStandardOutput "$root\logs\$logPrefix.out.log" `
        -RedirectStandardError "$root\logs\$logPrefix.err.log"
}

# 1) main instance (answerer medgemma-cpu + bge-m3), default host :11434.
#    MUST run before the labeler env vars are set - otherwise it would
#    inherit OLLAMA_HOST=:11435.
if (Test-PortOpen 11434) { "main Ollama :11434 already up" }
else {
    Start-OllamaHidden "ollama-main"
    Wait-PortOpen 11434; "main Ollama :11434 started"
}

# 2) labeler instance, iGPU/Vulkan, :11435 (env mirrors start_labeler_ollama.ps1)
if (Test-PortOpen 11435) { "labeler Ollama :11435 already up" }
else {
    $env:OLLAMA_HOST = "127.0.0.1:11435"
    $env:OLLAMA_VULKAN = "true"
    $env:OLLAMA_IGPU_ENABLE = "1"
    Start-OllamaHidden "ollama-labeler"
    Wait-PortOpen 11435; "labeler Ollama :11435 started"
}

# 3) web app (foreground). Run from $root: the package is not installed,
#    uvicorn imports medassistant from the repo root via CWD on sys.path.
$py = "$root\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error ".venv not found - see README Setup"; exit 1 }
Push-Location $root
try { & $py -m uvicorn medassistant.server:app --host 127.0.0.1 --port $webPort }
finally { Pop-Location }
