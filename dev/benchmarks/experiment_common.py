"""Small provenance helpers for explicitly selected same-binary experiments.

No builds, downloads, clock overrides, or concurrent benchmark launches. Generated
results belong outside the checkout. Artifact hashes record the build being run;
they do not prove compilation correspondence to HEAD (retain the build log too).
"""

import datetime
import fcntl
import hashlib
import json
import math
import os
import signal
import statistics
import struct
import subprocess
import time
from pathlib import Path

ORDER = ("off", "on", "on", "off")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path, algorithm="sha256"):
    path = Path(path)
    result = hashlib.sha1() if algorithm == "git_blob_sha1" else hashlib.sha256()
    if algorithm == "git_blob_sha1":
        result.update(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            result.update(block)
    return result.hexdigest()


def artifact(path):
    path = Path(path)
    resolved = path.resolve(strict=True)
    return {
        "path": str(path.absolute()),
        "resolved": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest(resolved),
    }


def token_hash(tokens):
    value = 1469598103934665603  # Matches the native benchmark's FNV offset.
    for byte in struct.pack("<" + "I" * len(tokens), *tokens):
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def metric(values):
    require(
        len(values) == 4
        and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values),
        "invalid ABBA metric",
    )
    a, b = statistics.mean((values[0], values[3])), statistics.mean(values[1:3])
    return {
        "abba": values,
        "baseline": a,
        "candidate": b,
        "reduction_pct": 100 * (1 - b / a),
        "a_endpoint_symmetric_drift_pct": 100 * abs(values[3] - values[0]) / a,
        "b_symmetric_spread_pct": 100 * abs(values[2] - values[1]) / b,
    }


def decision(values):
    require(len(values) == 3, "exactly three predeclared groups required")
    a, b = (
        statistics.mean(v["baseline"] for v in values),
        statistics.mean(v["candidate"] for v in values),
    )
    drift = max(v["a_endpoint_symmetric_drift_pct"] for v in values)
    spread = max(v["b_symmetric_spread_pct"] for v in values)
    reduction, floor = 100 * (1 - b / a), max(2.0, 2 * drift)
    stable = drift <= 5 and spread <= 5
    positive = all(v["reduction_pct"] > 0 for v in values)
    qualified = stable and positive and reduction > floor
    return {
        "groups": 3,
        "baseline_ms": a,
        "candidate_ms": b,
        "reduction_pct": reduction,
        "max_a_endpoint_symmetric_drift_pct": drift,
        "max_b_symmetric_spread_pct": spread,
        "predeclared_floor_pct": floor,
        "all_groups_improve": positive,
        "stable": stable,
        "qualified": qualified,
        "status": "inconclusive_drift"
        if not stable
        else "qualified"
        if qualified
        else "no_qualified_gain",
    }


def pin_build(worktree, extra_files=()):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=worktree).decode().strip()

    require(
        not git("status", "--porcelain=v1", "--untracked-files=all"),
        "commit the benchmark source before running it",
    )
    files = (
        "build/engine-tests/backend-benchmark",
        "build/splash.metallib",
        "build/engine/model/Runtime.o",
        "build/engine/libsplash.a",
    )
    pins = {name: artifact(worktree / name) for name in files}
    optional = worktree / "build/host-step.metallib"
    pins["build/host-step.metallib"] = artifact(optional) if optional.exists() else None
    return {
        "head": git("rev-parse", "HEAD"),
        "worktree": str(worktree),
        "artifacts": pins,
        "extra_files": [artifact(p) for p in extra_files],
    }


def model_identity(model, reference, full=False):
    """Portable source/content pins: installation paths may differ between Macs."""
    expected = json.loads(Path(reference).read_text())
    resolver_path = model / "model.json"
    resolver = json.loads(resolver_path.read_text())
    for key in ("family", "target_format", "sources"):
        require(resolver.get(key) == expected[key], "model reference differs: " + key)
    require(
        set(resolver["files"]) == set(expected["files"]), "model file coverage differs"
    )
    rows, checked, memo = {}, {}, {}
    for name, value in sorted(resolver["files"].items()):
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "unsafe model filename",
        )
        recorded = expected["files"][name]
        require(
            value["bytes"] == recorded["bytes"]
            and value["digest"] == recorded["recorded_source_digest"],
            "model entry differs: " + name,
        )
        resolved = (model / relative).resolve(strict=True)
        require(
            resolved == Path(value["path"]).resolve(strict=True),
            "resolver path differs: " + name,
        )
        st = resolved.stat()
        require(st.st_size == value["bytes"], "model size differs: " + name)
        row = {
            "resolved": str(resolved),
            "bytes": st.st_size,
            "device": st.st_dev,
            "inode": st.st_ino,
            "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns,
            "recorded_digest": value["digest"],
        }
        if st.st_size <= 32 << 20:
            row["metadata_sha256"] = digest(resolved)
            wanted = recorded.get("metadata_content_sha256")
            require(
                not wanted or wanted == row["metadata_sha256"],
                "metadata differs: " + name,
            )
        rows[name] = row
        if full:
            wanted = value["digest"]
            algorithm = (
                "sha256"
                if len(wanted) == 64
                else "git_blob_sha1"
                if len(wanted) == 40
                else None
            )
            require(algorithm, "unknown model digest algorithm")
            key = (str(resolved), algorithm)
            if key not in memo:
                memo[key] = digest(resolved, algorithm)
            require(memo[key] == wanted, "model content differs: " + name)
            checked[name] = {
                "algorithm": algorithm,
                "expected": wanted,
                "actual": memo[key],
                "match": True,
            }
    return {
        "root": str(model),
        "resolver": artifact(resolver_path),
        "sources": resolver["sources"],
        "family": resolver["family"],
        "target_format": resolver["target_format"],
        "loaded_model_layout_sha256": expected["loaded_model_layout_sha256"],
        "entries": rows,
    }, checked


def clean_environment():
    env = {
        key: os.environ[key]
        for key in ("HOME", "USER", "LOGNAME", "TMPDIR")
        if key in os.environ
    }
    env.update(PATH="/usr/bin:/bin:/usr/sbin:/sbin", LANG="C", LC_ALL="C")
    return env


def run_cohort(args, command, analyze, extra_files=()):
    """Run one native process; retain failure evidence and verify all pins again."""
    existed = args.out.exists()
    try:
        return _run_cohort(args, command, analyze, extra_files)
    except BaseException as error:
        # Incomplete runs never acquire a successful summary. Keep a small
        # failure record even if hashing, locking, or JSON validation failed.
        if not existed and args.out.is_dir():
            write_json(
                args.out / "failure.json",
                {
                    "complete": False,
                    "failure": type(error).__name__ + ": " + str(error),
                },
            )
        raise


def _run_cohort(args, command, analyze, extra_files):
    out, worktree, model = (
        args.out.resolve(),
        args.worktree.resolve(),
        args.model_root.resolve(),
    )
    require(
        out != worktree and worktree not in out.parents,
        "results must be outside the checkout",
    )
    out.mkdir(parents=True, exist_ok=False)
    with args.machine_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build = pin_build(worktree, extra_files)
        require(
            not args.expected_revision or build["head"] == args.expected_revision,
            "HEAD differs",
        )
        model_pin, checked = model_identity(model, args.model_pins, full=True)
        expected = {"build": build, "model": model_pin}
        write_json(out / "model-content-before.json", checked)
        write_json(out / "pin-before.json", expected)
        manifest = {
            "schema": 1,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "command": command,
            "pin": expected,
            "model_pins": json.loads(args.model_pins.read_text()),
            "environment": "Whitelist HOME USER LOGNAME TMPDIR; fixed system PATH, LANG=C, LC_ALL=C; no inherited runtime overrides.",
            "source_binary_caveat": "Retain build receipts: hashes do not prove compilation correspondence.",
        }
        write_json(out / "manifest.json", manifest)
        start = time.monotonic()
        failure, code, interrupted = None, None, None
        with (
            (out / "cohort.json").open("wb") as stdout,
            (out / "stderr.txt").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=worktree,
                env=clean_environment(),
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait()
                failure = "native process timed out"
            except BaseException as error:
                interrupted = error
                failure = "interrupted: " + type(error).__name__
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait()
        write_json(
            out / "receipt.json",
            {
                "command": command,
                "exit_code": code,
                "elapsed_seconds": time.monotonic() - start,
                "failure": failure,
                "stdout": artifact(out / "cohort.json"),
                "stderr": artifact(out / "stderr.txt"),
            },
        )
        after_model, after_checked = model_identity(model, args.model_pins, full=True)
        after = {"build": pin_build(worktree, extra_files), "model": after_model}
        write_json(out / "model-content-after.json", after_checked)
        write_json(out / "pin-after.json", after)
        require(
            after == expected and after_checked == checked,
            "source/build/model pins changed",
        )
        if interrupted is not None:
            raise interrupted
        require(code == 0 and not failure, failure or "native process failed")
        result = analyze(json.loads((out / "cohort.json").read_text()))
        write_json(out / "summary.json", result)
        return result


def audit_integrity(directory, expected_command):
    """Check saved artifacts without referring to the original installation paths."""

    def load(name):
        return json.loads((directory / name).read_text())

    manifest, receipt = load("manifest.json"), load("receipt.json")
    require(
        receipt["command"] == manifest["command"] == expected_command,
        "saved command differs",
    )
    require(
        receipt["exit_code"] == 0 and receipt["failure"] is None,
        "native process failed",
    )
    require(
        load("pin-before.json") == load("pin-after.json") == manifest["pin"],
        "saved pins differ",
    )
    before, after = load("model-content-before.json"), load("model-content-after.json")
    require(
        before == after and set(before) == set(manifest["model_pins"]["files"]),
        "model check coverage differs",
    )
    require(
        all(
            v["match"] is True
            and v["actual"]
            == v["expected"]
            == manifest["model_pins"]["files"][name]["recorded_source_digest"]
            for name, v in after.items()
        ),
        "model content verification failed",
    )
    for key, filename in (("stdout", "cohort.json"), ("stderr", "stderr.txt")):
        require(
            digest(directory / filename) == receipt[key]["sha256"],
            "saved artifact differs: " + key,
        )
    return manifest
