#!/usr/bin/env python3
"""Print the state of this checkout's development environment.

The report names the platform, Python, Xcode and Metal toolchain, the Git
state, the build output and the free disk. It reads no environment variable
that holds a secret and runs no model. Agents and bug reporters can paste it
into an issue.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIN_MACOS = (26, 4)
MIN_PYTHON = (3, 12)
MAX_PYTHON = (3, 15)


def output(*command: str) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return (completed.stdout or completed.stderr).strip() or None


def macos_version() -> tuple[int, ...] | None:
    version = platform.mac_ver()[0]
    if not version:
        return None
    parts = version.split(".")[:2]
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return None


def python_supported() -> bool:
    return MIN_PYTHON <= sys.version_info[:2] < MAX_PYTHON


def git_state() -> dict[str, str | bool | None]:
    branch = output("git", "rev-parse", "--abbrev-ref", "HEAD")
    commit = output("git", "rev-parse", "--short", "HEAD")
    status = output("git", "status", "--porcelain")
    return {
        "branch": branch,
        "commit": commit,
        "dirty": bool(status),
    }


def collect() -> dict[str, object]:
    version = macos_version()
    supported_macos = version is not None and version >= MIN_MACOS
    xcode = output("xcodebuild", "-version")
    metal = output("xcrun", "-sdk", "macosx", "-f", "metal")
    disk = shutil.disk_usage(ROOT)
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0] or None,
            "macos_supported": supported_macos if version else None,
        },
        "python": {
            "version": platform.python_version(),
            "supported": python_supported(),
            "executable": sys.executable,
            "repository_venv": Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
            "venv_exists": (ROOT / ".venv" / "pyvenv.cfg").is_file(),
        },
        "xcode": xcode.splitlines()[0] if xcode else None,
        "metal": metal,
        "git": git_state(),
        "build": {
            "directory": (ROOT / "build").is_dir(),
            "splash": (ROOT / "build" / "splash").is_file(),
            "metallib": (ROOT / "build" / "splash.metallib").is_file(),
        },
        "models": {
            "install_models": (ROOT / "install" / "models").is_dir(),
            "hub_cache": os.environ.get("HF_HUB_CACHE")
            or str(Path.home() / ".cache" / "huggingface"),
        },
        "disk": {
            "root": str(ROOT),
            "free_gib": round(disk.free / (1 << 30), 1),
        },
    }


def next_step(report: dict[str, object]) -> str:
    platform_info = report["platform"]
    python_info = report["python"]
    build = report["build"]
    if not isinstance(platform_info, dict) or not isinstance(python_info, dict):
        return "make help"
    if not isinstance(build, dict):
        return "make help"
    if platform_info.get("system") != "Darwin":
        return (
            "native checks need macOS; run the Python checks from "
            "docs/contributing/verification.md"
        )
    if platform_info.get("macos_supported") is False:
        return "this macOS is below the supported floor; see docs/README.md"
    if not python_info.get("supported"):
        return "install Python 3.12-3.14, then run make install-environment"
    if not python_info.get("venv_exists"):
        return "make install-environment"
    if not build.get("splash"):
        return "make"
    return "make check-fast"


def text(report: dict[str, object]) -> str:
    platform_info = report["platform"]
    python_info = report["python"]
    git_info = report["git"]
    build = report["build"]
    models = report["models"]
    disk = report["disk"]
    if not all(
        isinstance(item, dict)
        for item in (platform_info, python_info, git_info, build, models, disk)
    ):
        return "could not read the environment"
    lines = ["Splash development environment"]
    lines.append(
        f"  platform: {platform_info['system']} {platform_info['release']} "
        f"({platform_info['machine']})"
    )
    if platform_info["system"] == "Darwin":
        supported = "yes" if platform_info["macos_supported"] else "no"
        lines.append(
            f"  macOS: {platform_info['macos'] or 'unknown'} (supported: {supported})"
        )
    lines.append(
        f"  python: {python_info['version']} "
        f"({'repository .venv' if python_info['repository_venv'] else 'not the repository .venv'})"
    )
    lines.append(f"  xcode: {report['xcode'] or 'not found'}")
    lines.append(f"  metal: {report['metal'] or 'not found'}")
    lines.append(
        f"  git: {git_info['branch'] or 'unknown'} at "
        f"{git_info['commit'] or 'unknown'}"
        f"{' (dirty)' if git_info['dirty'] else ' (clean)'}"
    )
    lines.append(
        f"  build: {'splash' if build['splash'] else 'not built'}"
        f"{', metallib' if build['metallib'] else ''}"
    )
    lines.append(
        f"  models: {'installed' if models['install_models'] else 'not installed'}; "
        f"Hub cache {models['hub_cache']}"
    )
    lines.append(f"  disk: {disk['free_gib']} GiB free at {disk['root']}")
    lines.append(f"  next: {next_step(report)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON")
    options = parser.parse_args(argv)
    report = collect()
    if options.json:
        print(json.dumps(report, indent=2))
    else:
        print(text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
