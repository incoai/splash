#!/usr/bin/env python3
"""Restart one installation's installer as a release checks it, and record
the prepared weights it loads (DEVELOPMENT.md, Release check).

`prepare` runs with the Hub, then three restarts must each start the same
installation within RESTART_SECONDS without writing to a Hub cache:
HF_HUB_OFFLINE=1, a Hub that refuses connections, and an empty HF_HUB_CACHE.
`verify --full` then hashes every source file. A legacy package is only
verified. The output names, by component and SHA-256, the prepared-weight
cache entries the installation loads, which a load of it must have written.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.benchmarks import prepared  # noqa: E402
from install import families, gguf, hub, models  # noqa: E402

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
    one installation, as a release checks it; its selection."""
    options = models.parse_args([*arguments, "link"])
    selection = models.Selection.of(
        options.models,
        options.model,
        revision=options.revision,
        language_only=options.language_only,
        draft_model=options.draft_model,
    )
    if models.installation_kind(selection.link) != models.PACKAGE:
        installer(run, arguments, ["prepare"], ONLINE)
        if hub_cache is None:
            from huggingface_hub import constants

            hub_cache = Path(constants.HF_HUB_CACHE)
        installed = selection.link.resolve()
        sources = models.read_json(installed / "model.json")["sources"]
        target = sources["target"]
        # A local draft directory is no Hub repository.
        repositories = [s["repo"] for s in sources.values() if s["revision"]]
        started = (
            f"Splash model {selection.model} is already installed in {selection.link}"
        )
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
                    raise RestartFailure(
                        f"the {name} restart relinked {selection.link}"
                    )
                if hub_files(hub_cache, repositories) != downloads or any(
                    Path(moved).iterdir()
                ):
                    raise RestartFailure(f"the {name} restart wrote to a Hub cache")
    installer(run, arguments, ["verify", "--full"], {})
    return selection


def data_digest(path: Path) -> str:
    """The SHA-256 of a source file's tensor data, after its header, as a
    prepared entry's inputs name it (WeightSource::digest). The name of path
    gives the format, so path is an assembly's link, not the Hub cache blob
    it resolves to: a blob is named by its hash."""
    if path.suffix == ".gguf":
        header = gguf.Metadata(path, tensors=True)
        alignment = header.values.get("general.alignment", 32)
        offset = -(-header.consumed // alignment) * alignment
    else:
        with path.open("rb") as file:
            offset = 8 + int.from_bytes(file.read(8), "little")
    with path.open("rb") as file:
        file.seek(offset)
        return hashlib.file_digest(file, "sha256").hexdigest()


def loaded_entries(link: Path, cache: Path):
    """The (component, SHA-256) of each cache entry the installation at link
    loads. An entry's key hashes the preparation identity and plan too, but
    its source file names its component and the digest of the sorted digests
    of the source files it reads; one entry holds each component and inputs
    (PreparedWeights.cpp evictSuperseded)."""
    if models.installation_kind(link) == models.PACKAGE:
        return []  # A package's files are mapped as they are.
    record = models.read_json(link / "model.json")
    family = families.named(record["family"])
    layers = dict(family.signature)["num_hidden_layers"]
    components = {
        "target/embedding.bin",
        "target/head.bin",
        *(f"target/layer-{index}.bin" for index in range(layers)),
        "draft/model.bin",
        *(f"draft/layer-{index}.bin" for index in range(family.draft.layers)),
    }
    if record["vision_format"] != "none":
        components.add("vision/model.bin")
    # Each file is hashed once, through one of its links: the vision tower
    # of an MLX model links the target's shards again.
    sources = {}
    for name in record["files"]:
        if name.startswith(("target/", "draft/", "vision/")) and name.endswith(
            (".safetensors", ".gguf")
        ):
            sources.setdefault((link / name).resolve(), link / name)
    digests = sorted({data_digest(path) for path in sources.values()})
    inputs = {
        hashlib.sha256("".join(subset).encode()).hexdigest()
        for size in range(1, len(digests) + 1)
        for subset in itertools.combinations(digests, size)
    }
    found = {}
    for entry in prepared.entries(cache):
        if entry.get("component") in components and entry.get("inputs") in inputs:
            found.setdefault(entry["component"], []).append(entry)
    for component in sorted(components):
        entries = found.get(component, [])
        if not entries:
            raise RestartFailure(
                f"{cache} holds no prepared {component} of {link}; loading it "
                "prepares it (make test-real)"
            )
        if len(entries) > 1:
            raise RestartFailure(
                f"{cache} holds {len(entries)} prepared {component} of {link}: "
                "give each preparation identity its own SPLASH_WEIGHT_CACHE"
            )
    return [
        {"component": name, "sha256": found[name][0]["sha256"]}
        for name in sorted(components)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments, installer_arguments = parser.parse_known_args(argv)
    try:
        selection = check(installer_arguments)
        record = {
            "model": selection.model,
            "prepared": loaded_entries(selection.link, prepared.cache_root(os.environ)),
        }
    except (RestartFailure, models.ModelError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"installer restarts: PASS ({arguments.output})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
