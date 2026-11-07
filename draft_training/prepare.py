"""Normalize user JSONL and create reproducible, isolated conversation splits."""

import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

from .config import TrainingError, file_digest, write_json
from .data import conversation_identities, disjoint, split_manifest, validate_record


def read_source(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            location = f"{path}:{line}"
            try:
                record = json.loads(raw)
            except ValueError as error:
                raise TrainingError(f"{location}: invalid JSON") from error
            if isinstance(record, dict) and "messages" not in record:
                if "prompt" not in record:
                    raise TrainingError(
                        f"{location}: expected messages or a text prompt"
                    )
                if "response" in record and "completion" in record:
                    raise TrainingError(
                        f"{location}: provide response or completion, not both"
                    )
                messages = [{"role": "user", "content": record["prompt"]}]
                for key in ("response", "completion"):
                    if key in record:
                        messages.append({"role": "assistant", "content": record[key]})
                record = {**record, "messages": messages}
            records.append(validate_record(record, location))
    if not records:
        raise TrainingError(f"empty dataset: {path}")
    return records


def conversation_groups(records):
    # Transitive grouping keeps both shared source IDs and overlapping prompt
    # prefixes together, including records that connect two existing groups.
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for i, record in enumerate(records):
        for identity in sorted(conversation_identities(record)):
            first = seen.setdefault(identity, i)
            parents[root(i)] = root(first)
    groups = {}
    for i, record in enumerate(records):
        groups.setdefault(root(i), []).append(record)
    return list(groups.values())


def prepare_data(data, output, validation_fraction=0.1, test_fraction=0.1, seed=42):
    output = Path(output)
    if output.exists():
        raise TrainingError(f"prepared-data output already exists: {output}")
    if type(seed) is not int or seed < 0:
        raise TrainingError("seed must be a nonnegative integer")
    if (
        any(
            not math.isfinite(value) or not 0 < value < 1
            for value in (validation_fraction, test_fraction)
        )
        or validation_fraction + test_fraction >= 1
    ):
        raise TrainingError("split fractions must be positive and sum to less than one")
    groups = conversation_groups(read_source(data))
    if len(groups) < 3:
        raise TrainingError(
            "need at least three independent source/prompt groups for train, validation and test"
        )
    random.Random(seed).shuffle(groups)
    validation_count = max(1, int(len(groups) * validation_fraction))
    test_count = max(1, int(len(groups) * test_fraction))
    if validation_count + test_count >= len(groups):
        raise TrainingError("split fractions leave no training groups; reduce them")
    splits = {
        "validation": groups[:validation_count],
        "test": groups[validation_count : validation_count + test_count],
        "train": groups[validation_count + test_count :],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".draft-data-", dir=output.parent))
    try:
        manifests = {}
        for name, selected in splits.items():
            path = stage / f"{name}.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for group in selected:
                    for record in group:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifests[name] = split_manifest(path)
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            disjoint(manifests[left], manifests[right])
        report = {
            "input_sha256": file_digest(data),
            "seed": seed,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "fraction_scope": "independent groups; conversation counts may differ",
            "groups": {name: len(selected) for name, selected in splits.items()},
            "splits": manifests,
        }
        write_json(stage / "preparation.json", report)
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return report
