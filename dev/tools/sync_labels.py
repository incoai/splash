#!/usr/bin/env python3
"""Create or update the repository's GitHub labels from .github/labels.json.

The file is the source of truth. `gh label create --force` updates an existing
label in place, so rerunning this after an edit is safe. The script never
deletes labels that are absent from the file.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / ".github" / "labels.json"
COLOR = re.compile(r"^[0-9a-fA-F]{6}$")


def load_labels(path: Path = LABELS) -> list[dict[str, str]]:
    """Read and validate the labels file."""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of labels")
    labels: list[dict[str, str]] = []
    names: set[str] = set()
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: every label must be an object")
        try:
            name = entry["name"]
            color = entry["color"]
            description = entry["description"]
        except KeyError as missing:
            raise ValueError(f"{path}: label is missing {missing.args[0]!r}") from None
        if not all(isinstance(value, str) for value in (name, color, description)):
            raise ValueError(f"{path}: name, color and description must be strings")
        if not name or not color or not description:
            raise ValueError(f"{path}: name, color and description must not be empty")
        if not COLOR.fullmatch(color):
            raise ValueError(f"{path}: {name!r} has an invalid color {color!r}")
        if name in names:
            raise ValueError(f"{path}: duplicate label {name!r}")
        names.add(name)
        labels.append(
            {"name": name, "color": color.lower(), "description": description}
        )
    if not labels:
        raise ValueError(f"{path}: the labels list is empty")
    return labels


def command(label: dict[str, str], repo: str | None) -> list[str]:
    result = [
        "gh",
        "label",
        "create",
        label["name"],
        "--color",
        label["color"],
        "--description",
        label["description"],
        "--force",
    ]
    if repo:
        result += ["--repo", repo]
    return result


def sync(labels: list[dict[str, str]], repo: str | None, dry_run: bool) -> None:
    for label in labels:
        args = command(label, repo)
        print(" ".join(args))
        if not dry_run:
            subprocess.run(args, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="OWNER/REPO; defaults to the current repository")
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands and do nothing"
    )
    options = parser.parse_args(argv)
    try:
        labels = load_labels()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        sync(labels, options.repo, options.dry_run)
    except FileNotFoundError:
        print("error: gh is not installed; see https://cli.github.com", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(
            f"error: gh label create failed with status {error.returncode}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
