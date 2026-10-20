"""Toggle launcher for the whole assistant stack: three llama-server
processes (answerer :8081, labeler :8082, embeddings :8083) and the web app
(uvicorn, :8866, background by default). Owner decision 2026-10-18: the
Ollama pair is replaced by llama.cpp; this module is the policy-immune
Python port of the former start_chat.ps1 (no PowerShell in the shipped
entry path) and runs identically on the dev venv and the embedded python
(stdlib-only — the embedded runtime has no pip).

TOGGLE: if ANY managed port is up, re-running STOPS every managed
component; if all are down, it starts them all.

  python -m medassistant.launcher                 # start all / stop all
  python -m medassistant.launcher --skip-canary   # CLI full start: no offer
  python -m medassistant.launcher --skip-web      # llama-servers only
  python -m medassistant.launcher --foreground    # web app in this console;
                                                  # Ctrl+C stops only the web
  python -m medassistant.launcher --canary        # force the startup canary
  python -m medassistant.launcher --force-cpu     # -ngl 0 regardless of GPU
  python -m medassistant.launcher --force-gpu     # system default layers

Launch modes (as in start_chat.ps1, owner decision 2026-10-04): BARE (no
arguments) = hand-clicked: a 4 s canary offer appears after the backends
start (any key runs it, timeout skips); a smooth run closes the console
after a short linger (10 s after a start, 2 s after a stop); ANY error
keeps the window open until a keypress (owner decision 2026-10-07). ANY
argument = deliberate CLI run: no offer, no linger.

GPU policy (owner decision 2026-10-18): system default, dGPU-only. Display
adapter names come from the registry
(HKLM\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-...}\\*\\DriverDesc
— Win32_VideoController.AdapterRAM is uint32 and wraps above 4 GB). A
discrete card (NVIDIA/GeForce/RTX/Quadro/Radeon RX/Radeon Pro/Arc A#/B#)
runs with llama.cpp's default layer offload; anything else — including
Intel "Arc Graphics"/Iris iGPUs and Radeon 7xxM — runs with -ngl 0.
--force-cpu/--force-gpu override; the chosen backend is logged per boot.

Background components run windowless (CREATE_NO_WINDOW) with per-boot logs
(logs/llama-<role>-<stamp>.{out,err}.log, logs/web-<stamp>.{out,err}.log),
5 latest boots kept per component. llama-server has no child runners, so
stopping is a plain taskkill of the port owner; orphaned llama-server.exe
processes (not listening on a managed port) are swept on every start and
stop. Anything else on a managed port is reported and left alone.
"""

import argparse
import re
import socket
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
ENGINE = ROOT / "engine" / "llamacpp" / "llama-server.exe"
GGUF = ROOT / "models" / "medgemma1.5-4b-q4km-text-only.gguf"
BGE_GGUF = ROOT / "models" / "bge-m3-f16.gguf"
ANSWERER_TEMPLATE = ROOT / "models" / "answerer.jinja"
LABELER_TEMPLATE = ROOT / "models" / "labeler.jinja"

WEB_PORT = 8866
# (role, port) for the llama-server trio; the web app is managed separately
SERVERS = (
    ("answerer", 8081),
    ("labeler", 8082),
    ("embed", 8083),
)
MANAGED_PORTS = {port for _, port in SERVERS} | {WEB_PORT}
BOOT_KEEP = 5

# discrete-only patterns (review 2026-10-18): bare "Arc" matches Intel Core
# Ultra *integrated* "Arc Graphics" — require the A#/B# product letter; allow
# "(TM)"/other separators between "Arc" and the letter+digit
# ("Intel(R) Arc(TM) A770 Graphics" is discrete; "Arc Graphics" is iGPU).
_GPU_RE = re.compile(
    r"nvidia|geforce|\brtx\b|quadro|radeon\s+rx|radeon\s+pro|arc.*?[ab]\d",
    re.IGNORECASE)

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _say(msg):
    print(msg, flush=True)


def _warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)


# --- console helpers (BARE-run UX) ------------------------------------------

def _key_pressed():
    try:
        import msvcrt
        if msvcrt.kbhit():
            msvcrt.getwch()
            return True
    except (ImportError, OSError):
        pass
    return False


def _wait_for_key(prompt="ERROR - press ANY KEY to close..."):
    print()
    _say(prompt)
    try:
        import msvcrt
        while not msvcrt.kbhit():
            time.sleep(0.2)
        msvcrt.getwch()
    except (ImportError, OSError):
        time.sleep(60)  # non-interactive console: bounded linger


def _countdown(seconds, label="closing"):
    deadline = time.monotonic() + seconds
    while True:
        left = int(round(deadline - time.monotonic()))
        if left <= 0:
            break
        print(f"\r  {label} in {left}s...   ", end="", flush=True)
        time.sleep(0.2)
    print("\r" + " " * 40 + "\r", end="", flush=True)


def make_quit(bare):
    """Exit-time behavior: BARE (hand-clicked) runs only; CLI runs exit
    immediately. Smooth runs linger (10 s after a start, 2 s after a stop);
    ERROR runs never auto-close (owner decision 2026-10-07)."""

    def quit(code=0, pause=10.0):
        if bare:
            if code != 0:
                _wait_for_key()
            elif pause > 0:
                _countdown(pause)
        sys.exit(code)

    return quit


def request_canary(seconds=4):
    """BARE-run only: brief opt-in window — any key runs the canary, timeout
    (or a non-interactive console) skips it."""
    print()
    _say("canary: press ANY KEY to run.")
    print()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _key_pressed():
            # erase the countdown line; the canary banner takes its place
            print("\r" + " " * 40 + "\r", end="", flush=True)
            return True
        left = int(round(deadline - time.monotonic()))
        print(f"\r  skipping in {left}s...   ", end="", flush=True)
        time.sleep(0.2)
    print("\r  canary skipped                         ")
    print()
    return False


# --- ports, processes --------------------------------------------------------

def port_open(port, timeout=0.4):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_port(port, timeout, poll=0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_open(port):
            return True
        time.sleep(poll)
    return False


def wait_port_closed(port, timeout=8.0, poll=0.25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_open(port):
            return True
        time.sleep(poll)
    return False


def _listeners():
    """{pid: {port}} for TCP LISTENING sockets, and {port: {pid}}."""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "tcp"],
                             capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as e:
        _warn(f"netstat failed ({e}); port ownership checks degraded")
        return {}, {}
    by_pid, by_port = {}, {}
    for line in out.splitlines():
        parts = line.split()
        # TCP  <local> <remote> LISTENING <pid> — state word is fixed even on
        # localized Windows; anything else is not a listener row
        if len(parts) >= 5 and parts[0] == "TCP" and parts[3] == "LISTENING":
            try:
                port = int(parts[1].rsplit(":", 1)[1])
                pid = int(parts[4])
            except (ValueError, IndexError):
                continue
            by_pid.setdefault(pid, set()).add(port)
            by_port.setdefault(port, set()).add(pid)
    return by_pid, by_port


def _image_name(pid):
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        if out and not out.startswith("INFO:"):
            return out.split('","')[0].strip('"')
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _kill(pid, tree=True):
    cmd = ["taskkill", "/F", "/T", "/PID", str(pid)] if tree \
        else ["taskkill", "/F", "/PID", str(pid)]
    subprocess.run(cmd, capture_output=True, timeout=30)


def sweep_orphans():
    """Kill llama-server.exe processes listening on NO managed port —
    garbage from earlier stops or crashes. llama-server has no child
    runners, so a plain kill is enough (no tree walk)."""
    _, by_port = _listeners()
    healthy = {pid for port in MANAGED_PORTS for pid in by_port.get(port, ())}
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq llama-server.exe",
             "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        pid = int(parts[1])
        if pid not in healthy:
            _kill(pid)
            _say(f"  swept orphaned llama-server pid {pid} (no managed port)")


def stop_by_port(port, label, expect_images):
    """Stop the listener on a managed port. Only expected image names are
    killed; anything else is reported and left alone."""
    _, by_port = _listeners()
    pids = by_port.get(port, set())
    if not pids:
        _say(f"  {label} :{port} already down")
        return
    for pid in pids:
        name = _image_name(pid)
        if name and name.lower() in expect_images:
            _kill(pid)
            _say(f"  stopped {name} pid {pid} ({label}, :{port})")
        else:
            _warn(f"  :{port} is held by {name or 'an unknown process'} "
                  f"pid {pid} (not {'/'.join(expect_images)}) — left alone")


# --- GPU decision ------------------------------------------------------------

def adapter_names():
    """Display adapter DriverDesc values from the registry class key (the
    AdapterRAM WMI property is uint32 and wraps above 4 GB — unusable)."""
    names = []
    try:
        import winreg
    except ImportError:
        return names
    base = (r"SYSTEM\CurrentControlSet\Control\Class"
            r"\{4d36e968-e325-11ce-bfc1-08002be10318}")
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except OSError:
        return names
    with key:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(key, i)
            except OSError:
                break
            i += 1
            try:
                with winreg.OpenKey(key, sub) as sk:
                    val, _ = winreg.QueryValueEx(sk, "DriverDesc")
                    if val:
                        names.append(str(val))
            except OSError:
                continue
    return names


def is_discrete_gpu(name):
    """dGPU-only heuristic (owner decision 2026-10-18). iGPUs (Intel 'Arc
    Graphics', Iris, Radeon 7xxM, bare Radeon) deliberately do NOT match:
    they share system RAM and Vulkan allocation pools — the 2026-10-06
    ErrorOutOfDeviceMemory class of problems."""
    return bool(_GPU_RE.search(name))


def decide_backend(force_cpu=False, force_gpu=False):
    """'gpu' (system default offload) | 'cpu' (-ngl 0)."""
    if force_cpu and force_gpu:
        raise SystemExit("--force-cpu and --force-gpu are mutually exclusive")
    if force_cpu:
        return "cpu"
    if force_gpu:
        return "gpu"
    names = adapter_names()
    discrete = [n for n in names if is_discrete_gpu(n)]
    backend = "gpu" if discrete else "cpu"
    if names:
        _say(f"display adapters: {', '.join(names)}")
        _say(f"backend: {backend}"
             + (f" (dGPU: {', '.join(discrete)})" if discrete else
                " (no discrete GPU found - iGPU-only or none)"))
    else:
        _say("backend: cpu (no display adapters found)")
    print()
    return backend


# --- per-boot logs -----------------------------------------------------------

def boot_log(prefix, stamp, stream):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{prefix}-{stamp}.{stream}.log"


def prune_boot_logs(prefix, keep=BOOT_KEEP):
    stamps = sorted({p.name[len(prefix) + 1:-8]
                     for p in LOG_DIR.glob(f"{prefix}-*.log")
                     if p.name.startswith(f"{prefix}-")
                     and p.name.endswith((".out.log", ".err.log"))})
    for old in stamps[:-keep] if len(stamps) > keep else ():
        for p in LOG_DIR.glob(f"{prefix}-{old}.*.log"):
            try:
                p.unlink()
            except OSError:
                pass


# --- stack lifecycle ---------------------------------------------------------

def server_cmd(role, port, backend):
    """llama-server invocation for one role (ports/ctx per architecture.md)."""
    ngl = ["-ngl", "0"] if backend == "cpu" else []
    # --repeat-penalty 1.1 matches the default sampler: at temp 0 the
    # penalties run before greedy and keep a grammar-constrained labeler out
    # of degenerate id repetition (the 2026-10-18 select loop) and the
    # answerer closer to production behavior
    sam = ["--repeat-penalty", "1.1"]
    if role == "answerer":
        # num_ctx 12288: patient card (~2750 tok) + history + the 2048-token
        # thinking/answer budget overflowed the 4096 default (owner decision
        # 2026-10-09, carried over from the previous modelfile)
        return [str(ENGINE), "-m", str(GGUF), "--host", "127.0.0.1",
                "--port", str(port), "--alias", "medgemma-answerer",
                "-c", "12288", "--chat-template-file", str(ANSWERER_TEMPLATE),
                "--reasoning-format", "deepseek", "--temp", "0", *sam, *ngl]
    if role == "labeler":
        # same GGUF, think-suppressing template; 8192 ctx headroom over the
        # 4096 default (llama-server 400s on overflow instead of
        # silently truncating the prompt head)
        return [str(ENGINE), "-m", str(GGUF), "--host", "127.0.0.1",
                "--port", str(port), "--alias", "medgemma-labeler",
                "-c", "8192", "--chat-template-file", str(LABELER_TEMPLATE),
                "--reasoning-format", "deepseek", "--temp", "0", *sam, *ngl]
    # embeddings: bge-m3 F16, CLS pooling; 4 parallel sequences x 2048 ctx
    # for index-build batching
    return [str(ENGINE), "-m", str(BGE_GGUF), "--host", "127.0.0.1",
            "--port", str(port), "--alias", "bge-m3",
            "--embeddings", "--pooling", "cls", "-c", "8192", "-np", "4",
            *ngl]


def start_detached(cmd, out_path, err_path, cwd):
    """Windowless, detached from this script's lifetime: the child keeps
    running after the launcher exits; logs go to per-boot files."""
    with open(out_path, "ab", buffering=0) as out, \
            open(err_path, "ab", buffering=0) as err:
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=out, stderr=err,
                             stdin=subprocess.DEVNULL,
                             creationflags=_CREATE_NO_WINDOW)
    return p.pid


def start_servers(backend, stamp, quit):
    for role, port in SERVERS:
        if port_open(port):
            _say(f"llama-server {role} :{port} already up")
            continue
        cmd = server_cmd(role, port, backend)
        pid = start_detached(cmd, boot_log(f"llama-{role}", stamp, "out"),
                             boot_log(f"llama-{role}", stamp, "err"), ROOT)
        _say(f"llama-server {role:<9} :{port} starting (pid {pid})")
        if not wait_port(port, timeout=180):
            _warn(f"port {port} ({role}) did not open within 180s")
            _kill(pid)
            quit(1)
        prune_boot_logs(f"llama-{role}")
        _say(f"llama-server {role:<9} :{port} up")


def start_web(stamp, quit, foreground=False):
    cmd = [sys.executable, "-m", "uvicorn", "medassistant.server:app",
           "--host", "127.0.0.1", "--port", str(WEB_PORT)]
    if foreground:
        _say("web app in foreground (Ctrl+C stops only the web app)")
        url = f"http://127.0.0.1:{WEB_PORT}"

        def open_when_up():
            if wait_port(WEB_PORT, timeout=60):
                webbrowser.open(url)

        import threading
        threading.Thread(target=open_when_up, daemon=True).start()
        raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))
    pid = start_detached(cmd, boot_log("web", stamp, "out"),
                         boot_log("web", stamp, "err"), ROOT)
    if not wait_port(WEB_PORT, timeout=60):
        _warn(f"web port {WEB_PORT} did not open within 60s")
        _kill(pid)
        quit(1)
    prune_boot_logs("web")
    webbrowser.open(f"http://127.0.0.1:{WEB_PORT}")
    _say(f"web app started -> http://127.0.0.1:{WEB_PORT} "
         f"(opened in the default browser; logs: logs\\web-{stamp}.*.log)")


def stop_all(skip_web, quit):
    _say("Stopping assistant:")
    print()
    targets = [(label, port) for label, port in
               [(f"llama-server {role}", port) for role, port in SERVERS]
               + ([("web app", WEB_PORT)] if not skip_web else [])]
    for label, port in targets:
        stop_by_port(port, label,
                     ("llama-server.exe",) if port != WEB_PORT
                     else ("python.exe", "pythonw.exe"))
    sweep_orphans()
    for _, port in targets:
        if not wait_port_closed(port):
            _warn(f"  :{port} still open after stopping")
    print()
    quit(0, 2.0)


# --- preflight ---------------------------------------------------------------

def preflight(quit):
    if not ENGINE.is_file():
        _warn(f"llama-server.exe not found: {ENGINE} "
              f"(run: pwsh deploy/fetch_llamacpp.ps1)")
        quit(1)
    if not GGUF.is_file() or not BGE_GGUF.is_file():
        _warn(f"model GGUFs missing under {ROOT / 'models'} "
              f"(run: pwsh deploy/fetch_bge.ps1)")
        quit(1)
    if not (ROOT / ".env").is_file():
        _warn(".env missing — copy .env.example to .env")
    try:
        probe = ROOT / ".launcher-write-test"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        _warn(f"folder is not writable ({e}); extract the app to a "
              f"user-writable path outside Program Files")
        quit(1)
    s = str(ROOT).lower()
    if "program files" in s or s.startswith("c:\\windows"):
        _warn("the app folder is under Program Files/Windows — model caches "
              "and session data may fail; prefer a short path like C:\\medassistant")


# --- main --------------------------------------------------------------------

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    bare = not argv
    p = argparse.ArgumentParser(
        prog="medassistant.launcher",
        description="Toggle launcher for the llama.cpp backend + web app")
    p.add_argument("--foreground", action="store_true",
                   help="run the web app in this console")
    p.add_argument("--skip-web", action="store_true",
                   help="start/stop the llama-servers only")
    p.add_argument("--canary", action="store_true",
                   help="force the startup canary")
    p.add_argument("--skip-canary", action="store_true",
                   help="never offer the canary (CLI full-start)")
    p.add_argument("--force-cpu", action="store_true",
                   help="-ngl 0 regardless of detected GPUs")
    p.add_argument("--force-gpu", action="store_true",
                   help="system-default layer offload regardless")
    args = p.parse_args(argv)

    quit = make_quit(bare)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    preflight(quit)

    # toggle: anything managed up -> stop everything managed
    up = [port for port in MANAGED_PORTS if port_open(port)]
    if up:
        if args.foreground:
            _warn("--foreground applies only to starting; stopping instead.")
        stop_all(args.skip_web, quit)

    _say("Starting assistant:")
    print()
    backend = decide_backend(args.force_cpu, args.force_gpu)
    sweep_orphans()
    start_servers(backend, stamp, quit)

    # startup canary (owner decision 2026-10-04): BARE runs get the 4 s
    # offer; --canary forces it; any argument (CLI) skips it. It runs before
    # the web app and warms the models; on failure the stack still comes up
    # and the launcher exits 1 at the very end.
    run_canary = args.canary or (bare and not args.skip_canary
                                 and request_canary(4))
    canary_failed = False
    if run_canary:
        _say("running canary (health checks + one E2E question, ~40 s)...")
        print()
        code = subprocess.call([sys.executable, "-m", "medassistant", "canary"],
                               cwd=str(ROOT))
        if code != 0:
            canary_failed = True
            _warn("CANARY FAILED — the stack still comes up; check the "
                  "output above, /admin and logs/llama-*.log")

    if args.skip_web:
        _say("skipping web app (--skip-web)")
        quit(1 if canary_failed else 0, 10.0)
    start_web(stamp, quit, foreground=args.foreground)
    quit(1 if canary_failed else 0, 10.0)


if __name__ == "__main__":
    main()
