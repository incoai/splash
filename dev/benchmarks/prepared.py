"""The prepared weights of two builds, compared by their bytes.

A weight cache entry is <cache>/<key>/{weights, sha256, source}. Its key
hashes the build's preparation identity (build/engine/
WeightPreparationIdentity.hpp), so builds of different identities never share
an entry, and two such builds sharing one cache supersede each other's entries
at every start (DEVELOPMENT.md, Weight preparation). A comparison therefore
gives the baseline its own cache when the identities differ, while the
candidate keeps the cache its other release steps use, and compares bytes,
not keys: the candidate's cache must hold the bytes of every entry the
baseline prepared from the installation, for the same component and source
data when both entries record them.
"""

from __future__ import annotations

import re
from pathlib import Path

IDENTITY_HEADER = Path("engine/WeightPreparationIdentity.hpp")
PROVENANCE = "splash-prepared-weight-v1"
DIGEST = re.compile(r"[0-9a-f]{64}")


def preparation_identity(build: Path) -> bytes | None:
    """The preparation identity header of a build directory (a checkout's
    build/), or None for a build without one, which prepares nothing."""
    try:
        return (Path(build) / IDENTITY_HEADER).read_bytes()
    except FileNotFoundError:
        return None


def baseline_environment(directory: Path) -> dict:
    """The variable that gives a baseline of another preparation identity
    its own cache, directory/baseline-weights, created now."""
    cache = (Path(directory) / "baseline-weights").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    return {"SPLASH_WEIGHT_CACHE": str(cache)}


def cache_root(environment) -> Path:
    """The weight cache of a process started with environment, as
    PreparedWeights.cpp chooses it."""
    if path := environment.get("SPLASH_WEIGHT_CACHE"):
        return Path(path)
    if not (home := environment.get("HOME")):
        raise ValueError("cannot locate the prepared weight cache")
    return Path(home) / "Library/Caches/Splash/weights"


def entries(root: Path) -> list[dict]:
    """The complete entries of the cache at root: key and sha256 of the
    prepared bytes, plus component, inputs and source when the entry
    records its provenance."""
    records = []
    try:
        directories = sorted(Path(root).iterdir())
    except FileNotFoundError:
        return records
    for directory in directories:
        if not DIGEST.fullmatch(directory.name):
            continue
        try:
            digest = (directory / "sha256").read_text()
        except (FileNotFoundError, NotADirectoryError):
            continue
        if not DIGEST.fullmatch(digest):
            continue
        record = {"key": directory.name, "sha256": digest}
        try:
            lines = (directory / "source").read_text().splitlines()
        except FileNotFoundError:
            lines = []
        if len(lines) == 4 and lines[0] == PROVENANCE:
            for line, field in zip(lines[1:], ("component", "inputs", "source")):
                name, _, value = line.partition(" ")
                if name == field:
                    record[field] = value
        records.append(record)
    return records


def compare(
    baseline_root: Path, candidate_root: Path, *, package: Path, required: bool
) -> dict:
    """Whether the candidate's cache holds the bytes of every entry that the
    baseline prepared from package, the model root both builds were given,
    into its own cache. Entries there of other sources (another model, an
    earlier revision) are not compared. required: the installation prepares
    weights, so a baseline that prepared none from package is a failure."""
    root = str(package)
    baseline = [
        record
        for record in entries(baseline_root)
        if "source" not in record
        or record["source"] == root
        or record["source"].startswith(root + "/")
    ]
    candidate = entries(candidate_root)
    joined = {
        (record["component"], record["inputs"]): record["sha256"]
        for record in candidate
        if "component" in record and "inputs" in record
    }
    digests = {record["sha256"] for record in candidate}
    rows, failures = [], []
    for record in baseline:
        prepared = joined.get((record.get("component"), record.get("inputs")))
        match = (
            record["sha256"] == prepared
            if prepared is not None
            else record["sha256"] in digests
        )
        rows.append({**record, "candidate_sha256": prepared, "match": match})
        if not match:
            name = record.get("component", record["key"])
            failures.append(
                f"{name}: the candidate prepared {prepared}, the baseline "
                f"{record['sha256']}"
                if prepared is not None
                else f"{name}: the candidate's cache lacks the baseline's prepared "
                f"bytes {record['sha256']}"
            )
    if required and not baseline:
        failures.append(
            f"the baseline prepared no weights from {root} in {baseline_root}"
        )
    return {
        "baseline_cache": str(baseline_root),
        "candidate_cache": str(candidate_root),
        "entries": rows,
        "failures": failures,
        "pass": not failures,
    }
