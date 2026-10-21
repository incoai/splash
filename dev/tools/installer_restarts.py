#!/usr/bin/env python3
"""Restart one installation's installer as a release checks it
(DEVELOPMENT.md, Release check).

`prepare` runs with the Hub, then three restarts must each start the same
installation within RESTART_SECONDS without writing to a Hub cache:
HF_HUB_OFFLINE=1, a Hub that refuses connections, and an empty HF_HUB_CACHE.
`verify --full` then hashes every source file.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from install import hub, models  # noqa: E402

# The installer's 5 s Hub request (hub.HUB_TIMEOUT) and its own start.
RESTART_SECONDS = 10
UNREACHABLE_HUB = "http://127.0.0.1:9"
# Each run's Hub, whatever the caller's HF_HUB_OFFLINE says.
ONLINE = {"HF_HUB_OFFLINE": "0"}


class RestartFailure(RuntimeError):
    pass


def run_installer(arguments, environment, timeout=None):
    """install/models.py's exit status and output with arguments, in this
    environment with environment's variables added."""
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "install/models.py"), *arguments],
            env=os.environ | environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"did not exit within {timeout} s"
    return result.returncode, result.stdout


def hub_files(cache: Path, repositories):
    """Every downloaded file and snapshot link of the repositories in a Hub
    cache, with its modification time; the installer's references to them
    are not downloads."""
    return {
        path: path.lstat().st_mtime_ns
        for repository in repositories
        for part in ("blobs", "snapshots")
        for path in (cache / hub.folder_name(repository) / part).glob("**/*")
    }


def installer(run, arguments, command, environment):
    """Run the installer command; fail unless it exits 0."""
    status, output = run([*arguments, *command], environment)
    print(output, end="", flush=True)
    if status != 0:
        raise RestartFailure(f"{' '.join(command)} failed")


def restart(run, arguments, name, environment, expected):
    started = time.monotonic()
    status, output = run([*arguments, "prepare"], environment, RESTART_SECONDS)
    elapsed = time.monotonic() - started
    if status != 0 or elapsed > RESTART_SECONDS:
        raise RestartFailure(f"{name} restart failed after {elapsed:.1f} s:\n{output}")
    for line in expected:
        if line not in output:
            raise RestartFailure(f"{name} restart did not print {line!r}:\n{output}")
    print(f"{name} restart: PASS ({elapsed:.1f} s)", flush=True)


def check(arguments, run=run_installer, hub_cache=None):
    """Run the installer with arguments, the --model and source options of
    one installation, as a release checks it."""
    options = models.parse_args([*arguments, "link"])
    selection = models.Selection.of(
        options.models,
        options.model,
        revision=options.revision,
        language_only=options.language_only,
        draft_model=options.draft_model,
    )
    installer(run, arguments, ["prepare"], ONLINE)
    if hub_cache is None:
        from huggingface_hub import constants

        hub_cache = Path(constants.HF_HUB_CACHE)
    installed = selection.link.resolve()
    sources = models.read_json(installed / "model.json")["sources"]
    target = sources["target"]
    # A local draft directory is no Hub repository.
    repositories = [s["repo"] for s in sources.values() if s["revision"]]
    started = f"Splash model {selection.model} is already installed in {selection.link}"
    # A commit revision starts without asking the Hub.
    fallback = (
        ()
        if models.is_hex_digest(selection.revision, 40)
        else (
            "Could not reach the Hub (",
            f"); using the installed {selection.repo_id}@{target['revision'][:12]}.",
        )
    )
    downloads = hub_files(hub_cache, repositories)
    with tempfile.TemporaryDirectory() as moved:
        for name, environment, expected in (
            ("offline", {"HF_HUB_OFFLINE": "1"}, (started,)),
            (
                "unreachable-Hub",
                ONLINE | {"HF_ENDPOINT": UNREACHABLE_HUB},
                (*fallback, started),
            ),
            ("moved-cache", ONLINE | {"HF_HUB_CACHE": moved}, (started,)),
        ):
            restart(run, arguments, name, environment, expected)
            if selection.link.resolve() != installed:
                raise RestartFailure(f"the {name} restart relinked {selection.link}")
            if hub_files(hub_cache, repositories) != downloads or any(
                Path(moved).iterdir()
            ):
                raise RestartFailure(f"the {name} restart wrote to a Hub cache")
    installer(run, arguments, ["verify", "--full"], {})


def main(argv=None):
    try:
        check(sys.argv[1:] if argv is None else argv)
    except (RestartFailure, models.ModelError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("installer restarts: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
