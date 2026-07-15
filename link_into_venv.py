#!/usr/bin/env python3
"""Link this repo into a Python venv so `import ag23_llm` works — without packaging it.

ag23-llm is a repo, not a pip package, so there's nothing to `pip install`. This writes a
`.pth` file (a one-line path file Python reads at interpreter startup) into a venv's
site-packages, pointing at this repo. That's the standard "just a repo on the path"
mechanism — reproducible and tracked here, instead of a hand-placed file in some venv.

The repo path is computed from THIS file's location, so it's always correct for wherever
the repo actually lives; re-run after moving the repo to fix the link.

Usage:
    # install into a specific venv (locates its python for you):
    python link_into_venv.py --venv /path/to/venv
    # ...or run it WITH the target venv's python to install into that venv:
    /path/to/venv/Scripts/python link_into_venv.py      # Windows
    /path/to/venv/bin/python   link_into_venv.py        # macOS/Linux
    # remove the link:
    python link_into_venv.py --venv /path/to/venv --uninstall
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PTH_NAME = "ag23-llm.pth"


def _venv_python(venv: Path) -> Path:
    for cand in (venv / "Scripts" / "python.exe", venv / "bin" / "python", venv / "bin" / "python3"):
        if cand.exists():
            return cand
    raise SystemExit(f"no python interpreter found under {venv}")


def _install_into_current(uninstall: bool) -> int:
    """Write/remove the .pth in the site-packages of the interpreter running this."""
    pth = Path(sysconfig.get_path("purelib")) / PTH_NAME
    if uninstall:
        if pth.exists():
            pth.unlink()
            print(f"removed {pth}")
        else:
            print(f"nothing to remove at {pth}")
        return 0
    pth.write_text(str(REPO_ROOT) + "\n", encoding="utf-8")
    print(f"linked ag23-llm -> {pth}\n  points at: {REPO_ROOT}")
    check = subprocess.run(
        [sys.executable, "-c", "import ag23_llm; print(ag23_llm.__version__)"],
        capture_output=True, text=True,
    )
    print("  import check:", (check.stdout or check.stderr).strip() or "FAILED")
    return 0 if check.returncode == 0 else 1


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Link this ag23-llm repo into a venv (writes a .pth).")
    ap.add_argument("--venv", help="target venv dir (defaults to the interpreter running this script)")
    ap.add_argument("--uninstall", action="store_true", help="remove the .pth instead of writing it")
    args = ap.parse_args(argv)

    if args.venv:
        # Re-run this script with the target venv's python so the .pth lands in ITS site-packages.
        cmd = [str(_venv_python(Path(args.venv).resolve())), str(Path(__file__).resolve())]
        if args.uninstall:
            cmd.append("--uninstall")
        raise SystemExit(subprocess.run(cmd).returncode)

    raise SystemExit(_install_into_current(args.uninstall))


if __name__ == "__main__":
    main()
