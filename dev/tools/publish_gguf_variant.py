#!/usr/bin/env python3
"""Publish one converted GGUF variant into a multi-variant Splash package repository.

Uploads <target>/ as variants/<NAME>/target/ (parallel classic LFS transfers unless
--xet), verifies every file on the Hub by size and SHA-256, then adds or replaces the
variant in manifest.json and validates the result with the launcher's rules.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from install import models  # noqa: E402

LINEAR_NEEDLES = (
    ".attn_qkv",
    ".attn_gate",
    ".ssm_out",
    ".attn_q",
    ".attn_k",
    ".attn_v",
    ".attn_output",
    ".ffn_gate",
    ".ffn_up",
    ".ffn_down",
)


def role(name: str) -> str | None:
    """Role of a converter types.json key (tensor name without .weight)."""
    if name == "token_embd":
        return "token_embd"
    if name == "output":
        return "output"
    if ".ssm_alpha" in name or ".ssm_beta" in name:
        return "alpha_beta"
    if any(needle in name for needle in LINEAR_NEEDLES):
        return "linear"
    return None


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_types(target: Path) -> dict:
    """Per-role tensor type counts from the converter's types.json."""
    types = json.loads((target / "types.json").read_text())
    counts: dict = collections.defaultdict(collections.Counter)
    for name, kind in types.items():
        if (kind_role := role(name)) is not None:
            counts[kind_role][kind] += 1
    return {kind_role: dict(counter) for kind_role, counter in counts.items()}


def artifact_set(records) -> str:
    lines = "".join(
        f"{r['sha256']}  {r['path']}\n"
        for r in sorted(records, key=lambda r: r["path"])
    )
    return hashlib.sha256(lines.encode()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="owner/repo of the package")
    parser.add_argument(
        "--variant", required=True, help="variant name, e.g. UD-Q4_K_XL"
    )
    parser.add_argument("--target", required=True, type=Path, help="converted target/")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--source-sha256", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--xet", action="store_true", help="keep xet transfers enabled")
    args = parser.parse_args(argv)
    if not args.xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, hf_hub_download, upload_large_folder

    api = HfApi()
    variant = args.variant
    if not models.VARIANT.fullmatch(variant):
        raise SystemExit(f"invalid variant name: {variant}")
    prefix = f"variants/{variant}/target/"
    files = sorted(p for p in args.target.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"no files in {args.target}")
    records = [
        {"path": prefix + p.name, "size": p.stat().st_size, "sha256": sha256(p)}
        for p in files
    ]
    print(f"{len(records)} files, {sum(r['size'] for r in records) / 1e9:.2f} GB")

    def published():
        info = api.repo_info(args.repo, files_metadata=True)
        return {s.rfilename: s for s in info.siblings}

    def mismatched(hub):
        bad = []
        for record in records:
            item = hub.get(record["path"])
            lfs = getattr(item, "lfs", None) if item is not None else None
            if item is None or item.size != record["size"]:
                bad.append(record["path"])
            elif lfs is not None and lfs.sha256.lower() != record["sha256"]:
                bad.append(record["path"])
        return bad

    missing = mismatched(published())
    if missing:
        with tempfile.TemporaryDirectory() as stage:
            root = Path(stage) / prefix
            root.mkdir(parents=True)
            for p in files:
                if prefix + p.name in missing:
                    (root / p.name).symlink_to(p.resolve())
            print(f"uploading {len(missing)} files with {args.workers} workers")
            started = time.time()
            upload_large_folder(
                repo_id=args.repo,
                repo_type="model",
                folder_path=stage,
                num_workers=args.workers,
                print_report=True,
                print_report_every=60,
            )
            print(f"upload took {time.time() - started:.0f}s")
        still = mismatched(published())
        if still:
            raise SystemExit("Hub files do not match after upload: " + ", ".join(still))
    else:
        print("all files already on the Hub")

    manifest_path = Path(
        hf_hub_download(args.repo, "manifest.json", force_download=True)
    )
    manifest = json.loads(manifest_path.read_text())
    source = {"repo_id": args.source_repo, "file": args.source_file}
    if args.source_revision:
        source["revision"] = args.source_revision
    if args.source_sha256:
        source["sha256"] = args.source_sha256
    manifest.setdefault("variants", {})[variant] = {
        "artifacts": records,
        "source": source,
        "tensor_types": tensor_types(args.target),
        "bytes": sum(r["size"] for r in records),
    }
    everything = list(manifest["artifacts"])
    for entry in manifest["variants"].values():
        everything += entry["artifacts"]
    manifest["artifact_set_sha256"] = artifact_set(everything)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        validated = models.validate_package_manifest(path)
        assert models.select_variant(validated, variant) == variant
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo="manifest.json",
            repo_id=args.repo,
            commit_message=f"manifest: add variant {variant}",
        )
    print(
        f"published {args.repo}::{variant}; variants now: {sorted(manifest['variants'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
