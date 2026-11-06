"""The weight images two builds load from one model, compared by their bytes.

A build's engine-tests/weight-digests loads a model as its engine does and
prints the component, size and SHA-256 of every image it holds
(dev/tools/weight_digests.mm). Both builds must load the same images with the
same bytes. Given --expert-memory SIZE or --ple-memory SIZE, it streams the
routed experts or the PLE table from their files, as the engine does where
memory cannot hold them, and their bytes are not among the images.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

WEIGHT_DIGESTS = Path("engine-tests/weight-digests")


def loads_in_memory(build: Path) -> bool:
    """Whether a build directory (a checkout's build/) has weight-digests."""
    return (Path(build) / WEIGHT_DIGESTS).is_file()


def digests(build: Path, model_root: Path, options=()) -> dict:
    """The SHA-256 of each image, by component, that the build directory's
    engine loads from model_root, its weight-digests given options."""
    tool = Path(build) / WEIGHT_DIGESTS
    result = subprocess.run(
        [str(tool), str(Path(build) / "splash.metallib"), str(model_root), *options],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"{tool} exited {result.returncode}: {result.stderr.strip()}"
        )
    return {image["component"]: image["sha256"] for image in json.loads(result.stdout)}


def compare(baseline: dict, candidate: dict) -> dict:
    """Whether the builds loaded the same images, by component, with the
    same bytes."""
    rows = [
        {
            "component": component,
            "baseline_sha256": baseline.get(component),
            "candidate_sha256": candidate.get(component),
        }
        for component in sorted(baseline.keys() | candidate.keys())
    ]
    failures = [
        f"{row['component']}: the baseline loaded {row['baseline_sha256'] or 'nothing'}, "
        f"the candidate {row['candidate_sha256'] or 'nothing'}"
        for row in rows
        if row["baseline_sha256"] != row["candidate_sha256"]
    ]
    return {"images": rows, "failures": failures, "pass": not failures}
