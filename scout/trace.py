"""Bridge to `agent-trace-outcomes` (Arya's TS CLI) so scout runs leave a record.

Dogfooding: each scout run records what it checked and what it learned into
`.agent-trace/outcomes/`, and can read back lessons from prior runs before it starts.
The gateway is Python and the tool is a Node CLI, so we shell out. Everything here is
best-effort — if the CLI isn't installed, tracing silently no-ops and the scout still
runs. Resolution order: `atrace-outcomes` on PATH → `npx --yes agent-trace-outcomes`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

# Records land under the caller's working directory (.agent-trace/outcomes/), so a
# `python -m ag23_llm scout` run writes them into whatever project you run it from.
# Override with AG23_LLM_TRACE_DIR if you want a fixed location.
def _trace_cwd() -> str:
    return os.environ.get("AG23_LLM_TRACE_DIR") or os.getcwd()


def _base_cmd() -> Optional[list[str]]:
    if shutil.which("atrace-outcomes"):
        return ["atrace-outcomes"]
    if shutil.which("npx"):
        return ["npx", "--yes", "agent-trace-outcomes"]
    return None


def available() -> bool:
    return _base_cmd() is not None


def _run(args: list[str]) -> Optional[str]:
    base = _base_cmd()
    if base is None:
        return None
    try:
        proc = subprocess.run(
            base + args,
            cwd=_trace_cwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception:
        return None
    return (proc.stdout or "") + (proc.stderr or "")


def record(intent: str, checks: list[tuple[str, str, str]], *,
           lesson: Optional[str] = None, applies_to: Optional[list[str]] = None) -> bool:
    """Record an outcome. checks: [(name, kind, status), ...] e.g. ("scout","test","pass").

    Returns True if the CLI ran. No-ops (False) when the CLI is absent."""
    if not available():
        return False
    args = ["record", "--intent", intent]
    for name, kind, status in checks:
        args += ["--check", f"{name}:{kind}:{status}"]
    if lesson:
        args += ["--lesson", lesson]
    for path in (applies_to or []):
        args += ["--applies-to", path]
    return _run(args) is not None


def query_lessons(path: str) -> Optional[str]:
    """What's been tried/learned around a path, per prior outcome records."""
    return _run(["log", path])
