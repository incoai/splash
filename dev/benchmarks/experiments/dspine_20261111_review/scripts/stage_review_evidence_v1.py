#!/usr/bin/env python3
"""Plan, stage, or archive final-review evidence. The default only lists metadata.

Run stage/archive only after the controlled campaign finishes. No source tree,
artifact directory, model directory, or HTTP cache is an allowed input root.
"""

import argparse
import codecs
import gzip
import hashlib
import io
import json
import os
import re
import stat
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("report", "evidence", "logits", "hardware")
ROOTS = ("scripts", "notes", "results", "logs", "cohorts", "patches", "http", "width-probes")
SUFFIXES = {
    "scripts": {".py", ".mm", ".h", ".hpp"}, "notes": {".json", ".md", ".txt"},
    "results": {".json", ".jsonl", ".stdout", ".txt", ".csv"},
    "logs": {".stderr", ".stdout", ".txt", ".jsonl", ".log"},
    "cohorts": {".txt", ".json"}, "patches": {".patch"},
}
REPORT_SCRIPTS = {
    "reproduce_review_v1.py", "study_driver_v2.mm", "stage_review_evidence_v1.py",
    "abba.py", "analyze_abba_v2.py", "review_native_abba_v1.py", "review_native_abba_v2.py", "analyze_mixed_serving_v2.py",
    "analyze_prefill_validation_v1.py", "review_source_witness_v1.py", "width_probe_compare_v1.py",
    "width_probe_driver_v1.mm", "width_probe_build_v1.py", "candidate_sampling_v1.py",
    "sampling_recovery_v1.py", "prefill_iq4xs_validation_probe_v1.mm",
    "build_prefill_iq4xs_validation_probe_v1.py", "analyze_agent_mechanisms_v1.py",
    "analyze_historical_mechanisms_v1.py",
}
REPORT_PATCHES = {"qkv-narrow-v1.patch", "combined-v2.patch"}
REPORT_RESULTS = {
    "review-source-witness-v1.json", "agent12-mechanisms-v1.json", "historical31-mechanisms-v1.json",
    "main-native-build-v1.validation.json", "main-runtime-bundle-witness-v1.json",
    "main-source-byte-witness-v1.json", "selected-campaign-v1.run.json",
    "build-combined-v2.stdout", "combined-final-default-shader-v1.receipt.json",
}
PINNED_SOURCE = {
    "patches/qkv-narrow-v1.patch": "6a830fd355f3373d006ef4debf7081ff7305654fce9538ee8ae91b4438959325",
    "patches/combined-v2.patch": "f27dc90a8d0104b7350f03c58a3ad72ff35c3aa7f5c09f29657cf14680e7e60d",
    "scripts/study_driver_v2.mm": "2b0b3898db9bee8941e169d8852c8f6659f256f7d00e9de9f01a2a054783879d",
}
MANIFEST = "manifest.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(value):
    require(isinstance(value, str) and "\\" not in value, "path must be a POSIX relative string")
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.parts and all(part not in {".", ".."} for part in path.parts)
            and str(path) == value, "unsafe or noncanonical relative path: " + value)
    require(path.parts[0] in ROOTS, "input outside allowed roots: " + value)
    return path


def private(path):
    return any(part in {"__pycache__", ".ruff_cache", ".git", "cache", "models", "artifacts", "build"}
               or part.endswith((".gputrace", ".trace")) for part in path.parts) or (
        path.parts[0] == "notes" and
        (path.name.startswith("CONTINUATION") or path.name == "FINAL_REVIEW_DRAFT.md"))


def allowed(value, profile):
    path = relative(value)
    require(profile in PROFILES, "unknown profile")
    if private(path):
        return False
    top = path.parts[0]
    if profile == "hardware":
        return value == "results/hardware.jsonl"
    if profile == "logits":
        return top == "width-probes" and len(path.parts) == 3 and path.name == "logits.f32"
    if value == "results/hardware.jsonl":
        return False
    if top == "http":
        return profile == "evidence" and len(path.parts) == 3 and path.name in {"result.json", "server.log"}
    if top == "width-probes":
        return len(path.parts) == 3 and path.name in {"rows.jsonl", "result.json"}
    if path.suffix not in SUFFIXES[top]:
        return False
    if profile == "evidence":
        return True
    if top == "scripts":
        return ((len(path.parts) == 2 and path.name in REPORT_SCRIPTS) or
                value == "scripts/snapshots/review_native_abba_v1.pre_serial_fix.py")
    if top == "patches":
        return len(path.parts) == 2 and path.name in REPORT_PATCHES
    if top in {"notes", "cohorts"}:
        return path.suffix in {".json", ".txt"}
    if top == "results":
        return path.name.endswith((".analysis.json", ".comparison.json", ".review.json")) or (
            path.name in REPORT_RESULTS)
    return False


def walk_regular(directory, root):
    """Never follow a link or enter a denied subtree, including HTTP caches."""
    require(not directory.is_symlink(), "symlink directory refused: " + str(directory))
    if not directory.exists():
        return
    for entry in sorted(directory.iterdir()):
        path = relative(entry.relative_to(root).as_posix())
        if private(path):
            continue
        require(not entry.is_symlink(), "symlink refused: " + str(path))
        mode = entry.lstat().st_mode
        if stat.S_ISDIR(mode):
            if path.parts[0] in {"http", "width-probes"}:
                if len(path.parts) == 2:
                    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", path.name), "unsafe round name")
                    for name in (["result.json", "server.log"] if path.parts[0] == "http"
                                 else ["result.json", "rows.jsonl", "logits.f32"]):
                        child = entry / name
                        require(not child.is_symlink(), "symlink refused: " + str(child))
                        if child.exists():
                            require(stat.S_ISREG(child.lstat().st_mode), "nonregular file refused: " + str(child))
                            yield child
            else:
                yield from walk_regular(entry, root)
        elif stat.S_ISREG(mode):
            yield entry
        else:
            raise ValueError("nonregular file refused: " + str(path))


def plan(root, profile):
    require(not root.is_symlink(), "symlink evidence root refused")
    root = root.resolve()
    files = {}
    for name in ROOTS:
        for path in walk_regular(root / name, root):
            name = path.relative_to(root).as_posix()
            if allowed(name, profile):
                files[name] = {"bytes": path.lstat().st_size}
    return {"profile": profile, "kind": "metadata-only selection; no payload hashes or file reads",
            "file_count": len(files), "total_bytes": sum(row["bytes"] for row in files.values()),
            "files": dict(sorted(files.items()))}


def open_regular(root, value, allow_manifest=False):
    """Open every path component beneath root with O_NOFOLLOW, including parents."""
    path = PurePosixPath(MANIFEST) if allow_manifest and value == MANIFEST else relative(value)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[:-1]:
            next_descriptor = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(path.parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    stream = os.fdopen(file_descriptor, "rb")
    require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "nonregular input refused: " + value)
    return stream


def stage(root, profile, output):
    selection = plan(root, profile)
    if profile == "report":
        required = set(PINNED_SOURCE) | {
            "scripts/reproduce_review_v1.py", "notes/selected-source-v1.json", "notes/selected-campaign-v1.json",
            "results/review-source-witness-v1.json", "results/m48-clean-b3-v2.comparison.json",
            "results/m48-clean-b3-v2.review.json", "results/prefill-iq4xs-validation-v1.analysis.json",
            "results/qkv-historical31-v1.analysis.json", "results/width-base8-vs-qkv16-v1.analysis.json",
            "results/agent12-mechanisms-v1.json", "scripts/analyze_agent_mechanisms_v1.py",
            "results/historical31-mechanisms-v1.json", "scripts/analyze_historical_mechanisms_v1.py",
            "results/main-native-build-v1.validation.json", "results/main-runtime-bundle-witness-v1.json",
            "results/main-source-byte-witness-v1.json", "scripts/review_native_abba_v1.py",
        }
        require(required <= set(selection["files"]),
                "report is missing required source/selection/calibration/validation records: " +
                ", ".join(sorted(required - set(selection["files"]))))
    root, output = root.resolve(), output.absolute()
    try:
        inside = output.relative_to(root)
    except ValueError:
        inside = None
    require(inside is None or not inside.parts or inside.parts[0] not in ROOTS,
            "stage cannot be inside an input root")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "dspine-review-evidence-v1", "profile": profile, "files": {}}
    for name, row in selection["files"].items():
        require(allowed(name, profile), "file is outside the profile: " + name)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest, count = hashlib.sha256(), 0
        decoder = None if profile == "logits" else codecs.getincrementaldecoder("utf-8")()
        with open_regular(root, name) as source, target.open("xb") as destination:
            before = os.fstat(source.fileno())
            require(before.st_size == row["bytes"], "file changed after selection: " + name)
            for block in iter(lambda: source.read(1024 * 1024), b""):
                if decoder is not None:
                    require(b"\0" not in block, "binary content refused: " + name)
                    decoder.decode(block)
                digest.update(block)
                count += len(block)
                destination.write(block)
            if decoder is not None:
                decoder.decode(b"", final=True)
            after = os.fstat(source.fileno())
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) and count == row["bytes"],
                    "input changed during staging: " + name)
        target.chmod(0o644)
        if name in PINNED_SOURCE:
            require(digest.hexdigest() == PINNED_SOURCE[name], "pinned source changed: " + name)
        manifest["files"][name] = {"bytes": count, "sha256": digest.hexdigest()}
    with (output / MANIFEST).open("x") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return {"stage": str(output), "profile": profile, "file_count": len(manifest["files"]),
            "total_bytes": sum(row["bytes"] for row in manifest["files"].values())}


def archive(staged, output):
    require(not staged.is_symlink(), "symlink stage refused")
    manifest_path = staged / MANIFEST
    require(not manifest_path.is_symlink(), "symlink manifest refused")
    with open_regular(staged, MANIFEST, allow_manifest=True) as stream:
        manifest_bytes = stream.read()
    manifest = json.loads(manifest_bytes)
    require(manifest["format"] == "dspine-review-evidence-v1" and manifest["profile"] in PROFILES,
            "unsupported manifest")
    names = sorted(manifest["files"])
    require(all(allowed(name, manifest["profile"]) for name in names), "manifest contains an unallowed path")
    actual = []
    for directory, directories, files in os.walk(staged, followlinks=False):
        for name in directories + files:
            require(not (Path(directory) / name).is_symlink(), "symlink in staged evidence")
        for name in files:
            require(stat.S_ISREG((Path(directory) / name).lstat().st_mode), "nonregular staged file refused")
        actual += [(Path(directory) / name).relative_to(staged).as_posix() for name in files]
    require(set(actual) == set(names) | {MANIFEST}, "staged files differ from manifest")
    require(not output.absolute().is_relative_to(staged.absolute()), "archive cannot be inside its staged input")
    with output.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name in sorted(names + [MANIFEST]):
                stream = io.BytesIO(manifest_bytes) if name == MANIFEST else open_regular(staged, name)
                with stream:
                    before = None if name == MANIFEST else os.fstat(stream.fileno())
                    size = len(manifest_bytes) if before is None else before.st_size
                    if name != MANIFEST:
                        digest = hashlib.sha256()
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                        require(size == manifest["files"][name]["bytes"] and
                                digest.hexdigest() == manifest["files"][name]["sha256"],
                                "staged content changed: " + name)
                        stream.seek(0)
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = size, 0o644, 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    tar.addfile(info, stream)
                    if before is not None:
                        after = os.fstat(stream.fileno())
                        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                                "staged input changed during compression: " + name)
    return {"archive": str(output), "bytes": output.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["plan", "stage", "archive"], default="plan")
    parser.add_argument("--root", type=Path, default=ROOT, help="evidence root; staged root for archive")
    parser.add_argument("--profile", choices=PROFILES, default="evidence")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action != "plan" and args.output is None:
        parser.error("stage/archive require a new --output path")
    if args.action == "plan":
        require(args.output is None, "plan lists only; redirect stdout to a new plan file if needed")
        result = plan(args.root, args.profile)
    elif args.action == "stage":
        result = stage(args.root, args.profile, args.output)
    else:
        result = archive(args.root, args.output)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
