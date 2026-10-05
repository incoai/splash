#!/usr/bin/env python3

"""The official model catalog: a bundled seed plus a refreshable cache.

Shell completion must never block and must work offline, so it only ever reads
files. The bundled catalog is versioned with the source; the cache is
refreshed in the background by `splash serve` and lives
under the per-user data directory.

Readers (install/completions/models) take the union of the two. A cache that
is missing, stale, empty or corrupt can therefore only ever fail to *add*
entries — it can never remove a model the package already knew about, and it
can never make completion worse than a fresh install.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

if __name__ == "__main__" and not __package__:
    # Run as a script by spawn_refresh: import siblings as the install
    # package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "install"

from . import paths
from .models import ModelError, validate_repo_id

# The collection is the source of truth for which packages are official.
COLLECTION = "incoai/splash-6aac69afeba907af0511ec14"
HUB_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

CACHE = (
    paths.DATA if paths.PACKAGED else paths.RUNTIME
) / "catalog/official-models.txt"

# A model joins the roster a handful of times a year, so a day-old catalog is
# not meaningfully stale. This bound exists to stop a long-running install
# drifting for months, not to track the collection closely.
MAX_AGE_SECONDS = 24 * 60 * 60
# Bounded so a hung network cannot keep a background process alive.
TIMEOUT_SECONDS = 10


def is_stale(now: float | None = None) -> bool:
    try:
        age = (time.time() if now is None else now) - CACHE.stat().st_mtime
    except OSError:
        return True
    return age >= MAX_AGE_SECONDS


def _fetch(timeout: float) -> list[str]:
    # The Hub's JSON API is used directly rather than through huggingface_hub:
    # it gives a hard timeout, and keeps a background refresh independent of
    # the download stack.
    request = urllib.request.Request(
        f"{HUB_ENDPOINT}/api/collections/{COLLECTION}",
        headers={"Accept": "application/json"},
    )
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("invalid model collection response")
    identifiers = set()
    for item in payload["items"]:
        if not isinstance(item, dict):
            continue
        # The REST payload uses type/id; huggingface_hub exposes the same
        # fields as item_type/item_id. Accept either so this keeps working
        # whichever shape the endpoint returns.
        if (item.get("type") or item.get("item_type")) != "model":
            continue
        try:
            identifiers.add(
                validate_repo_id(item.get("id") or item.get("item_id") or "")
            )
        except ModelError:
            continue
    return sorted(identifiers)


def refresh(timeout: float = TIMEOUT_SECONDS, destination: Path | None = None) -> bool:
    """Refresh the catalog. True on success, including unchanged content.

    Never raises for an unreachable Hub. The previous cache is left untouched
    on any failure, including an empty or unparseable response, so a bad
    network can only ever leave completion where it already was.
    """
    try:
        identifiers = _fetch(timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False
    if not identifiers:
        # An empty collection is far more likely to be an API change or an
        # auth failure than a real roster of nothing.
        return False

    destination = CACHE if destination is None else Path(destination)
    body = "\n".join(identifiers) + "\n"
    try:
        try:
            current = destination.read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            current = None
        if current == body:
            os.utime(destination, None)  # Mark as checked so we back off.
            return True
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as pending:
            pending_path = Path(pending.name)
            try:
                pending.write(body)
                pending.flush()
                os.fsync(pending.fileno())
                pending_path.chmod(0o644)
                os.replace(pending_path, destination)
            finally:
                pending_path.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def _offline() -> bool:
    """HF_HUB_OFFLINE, as huggingface_hub reads it: the Hub must not be asked."""
    value = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
    return (value or "").upper() in {"1", "ON", "YES", "TRUE"}


def spawn_refresh() -> None:
    """Refresh the cache in a detached child, if it looks stale.

    `splash serve` replaces itself with the server via execve, so this cannot
    be a thread. It is deliberately fire-and-forget: the caller never learns
    the outcome, and a failure is indistinguishable from not having run.
    """
    if not is_stale() or _offline():
        return
    try:
        subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--refresh"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        pass


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="refresh the cache")
    args = parser.parse_args(argv)

    if args.refresh and is_stale() and not refresh():
        print("could not refresh the model catalog", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
