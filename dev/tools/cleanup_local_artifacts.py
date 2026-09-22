"""Apply a reviewed local artifact manifest; never clean source or model stores."""

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
PROTECTED = (
    "build/flash-next",
    "build/flash-ple-ssd-integrated-v12",
    "build/flash-operand-optout-cpu",
    "build/references",
    "build/engine",
    "build/engine-tests",
    "build/release/metal41",
    "build/release/flash-next",
)
SOURCE_SUFFIXES = {
    ".cpp",
    ".cc",
    ".c",
    ".mm",
    ".m",
    ".h",
    ".hpp",
    ".metal",
    ".py",
    ".mk",
    ".sh",
    ".toml",
    ".swift",
    ".rs",
    ".go",
}
ALLOWED_TYPES = {
    "compiled_experiment",
    "experimental_extracted_tensor",
    "raw_instruments_trace_directory",
    "raw_trace_xml_export",
    "large_dispatch_trace_jsonl",
}


def checked_path(raw):
    rel = PurePosixPath(raw)
    if (
        rel.is_absolute()
        or ".." in rel.parts
        or not rel.parts
        or rel.parts[0] != "build"
    ):
        raise ValueError(f"outside build: {raw}")
    if any(raw == prefix or raw.startswith(prefix + "/") for prefix in PROTECTED):
        raise ValueError(f"protected artifact: {raw}")
    if raw.startswith("build/release/qwen27b") or (
        len(rel.parts) > 1
        and rel.parts[1].startswith(("int8", "metal41", "instrumentation"))
    ):
        raise ValueError(f"unrelated prior work: {raw}")
    target = ROOT.joinpath(*rel.parts)
    for part in (target, *target.parents):
        if part == ROOT:
            break
        if part.is_symlink():
            raise ValueError(f"symlink path: {raw}")
    return target


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def validate(entry, active_paths):
    target = checked_path(entry["path"])
    if entry["type"] not in ALLOWED_TYPES or str(target) in active_paths:
        raise ValueError(f"invalid type/active executable: {entry['path']}")
    if entry["type"] == "raw_instruments_trace_directory":
        if not target.is_dir() or target.suffix != ".trace":
            raise ValueError(f"invalid trace directory: {target}")
        actual = set()
        for parent, dirs, names in os.walk(target, followlinks=False):
            for name in dirs + names:
                if (Path(parent) / name).is_symlink():
                    raise ValueError(f"symlink inside trace: {target}")
            actual.update(
                (Path(parent) / n).relative_to(target).as_posix() for n in names
            )
        expected = {m["path"] for m in entry["members"]}
        if actual != expected:
            raise ValueError(f"trace inventory changed: {target}")
        for m in entry["members"]:
            rel = PurePosixPath(m["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("invalid trace member path")
            child = target.joinpath(*rel.parts)
            if (
                child.stat().st_size != m["logical_bytes"]
                or digest(child) != m["sha256"]
            ):
                raise ValueError(f"trace member changed: {child}")
    else:
        info = target.stat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size != entry["logical_bytes"]
            or info.st_mtime_ns != entry["mtime_ns"]
        ):
            raise ValueError(f"artifact changed: {target}")
        if (
            target.suffix in SOURCE_SUFFIXES
            or target.name == "Makefile"
            or target.suffix in {".json", ".md", ".txt", ".sha256", ".config"}
        ):
            raise ValueError(f"source/evidence candidate: {target}")
        if entry.get("sha256") and digest(target) != entry["sha256"]:
            raise ValueError(f"artifact hash changed: {target}")
    for report in entry.get("retained_companion_reports", []):
        path = ROOT / report["path"]
        if not path.is_file() or digest(path) != report["sha256"]:
            raise ValueError(f"companion report changed/missing: {path}")
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads(args.manifest.read_text())
    if (
        data["schema"] != "splash-local-artifact-cleanup-plan-v13"
        or Path(data["workspace"]) != ROOT
    ):
        raise ValueError("wrong workspace/manifest schema")
    proc = subprocess.run(
        ["lsof", "-nP", "-F", "n", "-d", "txt", "-c", "splash-flash"],
        capture_output=True,
        text=True,
    )
    active_paths = {
        line[1:] for line in proc.stdout.splitlines() if line.startswith("n")
    }
    entries = data["candidates"]
    paths = [validate(e, active_paths) for e in entries]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate cleanup target")
    before_kib = int(
        subprocess.check_output(["du", "-sk", str(ROOT / "build")], text=True).split()[
            0
        ]
    )
    removed = []
    if args.apply:
        for entry, path in zip(entries, paths):
            # Revalidate immediately before each mutation as well as the full preflight.
            validate(entry, active_paths)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(entry["path"])
        for parent in sorted(
            {p.parent for p in paths}, key=lambda x: len(x.parts), reverse=True
        ):
            while parent != ROOT / "build" and parent != ROOT:
                raw = parent.relative_to(ROOT).as_posix()
                if any(raw == x or x.startswith(raw + "/") for x in PROTECTED):
                    break
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
    after_kib = int(
        subprocess.check_output(["du", "-sk", str(ROOT / "build")], text=True).split()[
            0
        ]
    )
    result = dict(
        schema="splash-local-artifact-cleanup-result-v13",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        applied=args.apply,
        candidates_validated=len(paths),
        removed_count=len(removed),
        planned_removed_allocated_bytes=sum(e["allocated_bytes"] for e in entries)
        if args.apply
        else 0,
        build_allocated_bytes_before=before_kib * 1024,
        build_allocated_bytes_after=after_kib * 1024,
        observed_build_allocated_bytes_reduction=(before_kib - after_kib) * 1024,
        manifest_sha256=digest(args.manifest),
        manifest=str(args.manifest),
        source_models_reports_preserved=True,
        symlinks_followed=False,
        raw_historical_traces_removed=args.apply,
        exact_trace_recovery_promised=False,
        removed=removed,
    )
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "removed"}))


if __name__ == "__main__":
    main()
