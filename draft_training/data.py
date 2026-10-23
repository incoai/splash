"""Text-only examples and conversation-level split isolation."""

import json
import re
import unicodedata
from pathlib import Path

from .config import TrainingError, digest, file_digest


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def conversations(path):
    records = []
    with Path(path).open() as stream:
        for line, raw in enumerate(stream, 1):
            try:
                record = json.loads(raw)
            except ValueError as error:
                raise TrainingError(f"{path}:{line}: invalid JSON") from error
            messages = record.get("messages") if isinstance(record, dict) else None
            if not isinstance(messages, list) or not messages:
                raise TrainingError(f"{path}:{line}: expected nonempty messages")
            for message in messages:
                if (
                    not isinstance(message, dict)
                    or message.get("role") not in ("system", "user", "assistant")
                    or not isinstance(message.get("content"), str)
                    or not message["content"].strip()
                ):
                    raise TrainingError(
                        f"{path}:{line}: only nonempty text system/user/assistant messages are supported"
                    )
            if not any(m["role"] == "user" for m in messages):
                raise TrainingError(f"{path}:{line}: conversation needs a user message")
            records.append(
                {
                    "messages": [
                        {"role": m["role"], "content": m["content"]} for m in messages
                    ],
                    "source_id": record.get("source_id"),
                }
            )
    if not records:
        raise TrainingError(f"empty dataset: {path}")
    return records


def split_manifest(path):
    records = conversations(path)
    identities = set()
    prompts = []
    for record in records:
        if record["source_id"] is not None:
            identities.add(digest(["source", record["source_id"]]))
        for i, message in enumerate(record["messages"]):
            if message["role"] == "user":
                prefix = [
                    {"role": m["role"], "content": normalized(m["content"])}
                    for m in record["messages"][: i + 1]
                ]
                identities.add(digest(["prompt", prefix]))
        prompts.append(digest(record["messages"]))
    return {
        "sha256": file_digest(path),
        "identities": sorted(identities),
        "conversations": len(records),
        "records": prompts,
    }


def disjoint(left, right):
    if set(left["identities"]) & set(right["identities"]):
        raise TrainingError(
            "dataset splits overlap in source conversation or normalized user prompt"
        )


def examples(records, supervision):
    for record in records:
        messages = record["messages"]
        for i, message in enumerate(messages):
            if message["role"] != "user":
                continue
            answer = None
            if supervision == "supplied-answers":
                if i + 1 >= len(messages) or messages[i + 1]["role"] != "assistant":
                    continue
                answer = messages[i + 1]["content"]
            yield messages[: i + 1], answer


def anchors(sample, block_size):
    """Only full blocks entirely inside an assistant continuation are scored."""
    mask = sample["assistant_mask"].tolist()
    for anchor in range(len(mask) - block_size + 1):
        if anchor >= 1 and all(mask[anchor + 1 : anchor + block_size]):
            yield anchor
