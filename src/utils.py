"""Small shared helpers: timestamped progress printing and subprocess wrappers."""

import subprocess
import sys
import time


_START = time.time()


def log(stage: str, message: str) -> None:
    """Print a progress line with an elapsed-time stamp, flushed immediately
    so it shows up live in the terminal during long-running steps."""
    elapsed = time.time() - _START
    print(f"[{elapsed:7.1f}s] [{stage}] {message}", flush=True)


def run(cmd: list, desc: str = "") -> subprocess.CompletedProcess:
    """Run a subprocess command, streaming stderr on failure for debuggability."""
    if desc:
        log("cmd", desc)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {result.stderr[-2000:]}")
    return result


def format_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"
