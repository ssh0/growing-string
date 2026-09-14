#!/usr/bin/env python3
"""Run the repository's scoped Python formatter or lint check.

The repository has an intentionally unformatted Python 2-era legacy baseline.  This
runner checks only changed Python files in the maintained quality scope so adopting
Ruff does not create a broad formatting-only change.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QUALITY_ROOTS = ("continuum_filament_model/", "scripts/")
EXCLUDED_PREFIXES = ("continuum_filament_model/notebooks/",)


def _git(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _available_revision(revision: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _base_revision() -> str | None:
    configured = os.environ.get("PYTHON_QUALITY_BASE")
    if configured:
        return configured

    github_base_ref = os.environ.get("GITHUB_BASE_REF")
    candidates = []
    if github_base_ref:
        candidates.append(f"origin/{github_base_ref}")
    candidates.extend(("origin/master", "master"))
    return next((candidate for candidate in candidates if _available_revision(candidate)), None)


def _changed_paths() -> list[str]:
    base = _base_revision()
    if base:
        paths = _git("diff", "--name-only", "--diff-filter=ACMR", base)
    else:
        paths = _git("diff", "--name-only", "--diff-filter=ACMR")
        paths += _git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    paths += _git("ls-files", "--others", "--exclude-standard")

    selected = {
        path
        for path in paths
        if path.endswith(".py")
        and path.startswith(QUALITY_ROOTS)
        and not path.startswith(EXCLUDED_PREFIXES)
    }
    return sorted(selected)


def _run(check: str, paths: list[str]) -> int:
    if not paths:
        print("No changed Python files in the maintained quality scope; nothing to check.")
        return 0

    print(f"Checking {len(paths)} Python file(s):")
    print("\n".join(f"  {path}" for path in paths))
    command = ["ruff", "format", "--check", *paths]
    if check == "lint":
        command = ["ruff", "check", *paths]
    return subprocess.run(command, cwd=REPO_ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=("format", "lint"))
    args = parser.parse_args()
    try:
        paths = _changed_paths()
    except subprocess.CalledProcessError as exc:
        print(f"Unable to determine changed files with git: {exc}", file=sys.stderr)
        return 2
    return _run(args.check, paths)


if __name__ == "__main__":
    raise SystemExit(main())
