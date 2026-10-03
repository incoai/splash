#!/usr/bin/env python3
"""Upload a built package to the private Hugging Face repo that testers install from."""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--repo", required=True, help="e.g. owner/splash-releases")
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="upload without moving the `latest` pointer",
    )
    args = parser.parse_args(argv)
    archive = ROOT / "dist" / f"splash-{args.version}-arm64-macos26.tar.gz"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    installer = ROOT / "dev/tools/install.sh"
    for path in (archive, checksum, installer):
        if not path.is_file():
            sys.exit(
                f"missing {path}; run `make package RELEASE_VERSION={args.version}` first"
            )

    api = HfApi()
    api.create_repo(args.repo, private=True, exist_ok=True)
    uploads = [
        (archive, archive.name),
        (checksum, checksum.name),
        (installer, "install.sh"),
    ]
    for path, name in uploads:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=args.repo)
        print(f"uploaded {name}")
    if not args.no_latest:
        api.upload_file(
            path_or_fileobj=(args.version + "\n").encode(),
            path_in_repo="latest",
            repo_id=args.repo,
        )
        print(f"latest -> {args.version}")
    print(
        "testers set SPLASH_TOKEN to the supplied read token, then run:\n"
        "export SPLASH_TOKEN\n"
        "curl -qfsSL --config - "
        f"https://huggingface.co/{args.repo}/resolve/main/install.sh <<EOF"
        f" | SPLASH_REPO={args.repo} sh\n"
        'header = "Authorization: Bearer $SPLASH_TOKEN"\n'
        "EOF"
    )


if __name__ == "__main__":
    main()
