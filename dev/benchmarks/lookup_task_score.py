#!/usr/bin/env python3
"""Offline exact full-edit diagnostics; never filters the timing cohort.

Supports only the pinned Qwen ByteLevel decoder. A returned source must be the
whole response, either unfenced or inside exactly one correctly labelled fence.
Only CRLF->LF and outer newline differences are ignored; interior whitespace,
extra prose, missing lines and unrelated edits fail. EOS tokens are retained in
raw counts and removed only from displayed decoded text, like skip_special_tokens.
"""

import hashlib
import json
import re
from pathlib import Path

TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
LANGUAGES = {
    "lookup_edit_python": ("python", "py"),
    "lookup_edit_markdown": ("markdown", "md"),
    "lookup_edit_config": ("json",),
    "fresh_edit_sql": ("sql",),
    "fresh_edit_toml": ("toml",),
    "fresh_edit_js": ("javascript", "js"),
    "fresh_edit_markdown": ("markdown", "md"),
    "fresh_edit_csv": ("csv",),
    "fresh_edit_clause": ("text", "txt"),
}


def decoder(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != TOKENIZER_SHA256:
        raise RuntimeError("task decoder tokenizer bytes differ from fixed fixture")
    document = json.loads(data)
    if document["decoder"] != {
        "type": "ByteLevel",
        "add_prefix_space": True,
        "trim_offsets": True,
        "use_regex": True,
    }:
        raise RuntimeError("unsupported task decoder")
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = visible.copy()
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            chars.append(256 + extra)
            extra += 1
    inverse = {chr(character): byte for byte, character in zip(visible, chars)}
    vocabulary = {index: token for token, index in document["model"]["vocab"].items()}
    vocabulary.update(
        {token["id"]: token["content"] for token in document["added_tokens"]}
    )
    special = {token["id"] for token in document["added_tokens"] if token["special"]}

    def decode(tokens):
        return bytes(
            inverse[character]
            for token in tokens
            if token not in special and token in vocabulary
            for character in vocabulary[token]
        ).decode("utf-8", errors="replace")

    decode.unknown_ids = lambda tokens: sorted(set(tokens) - set(vocabulary))
    return decode


def assess_response(name, response, expected):
    text = response.replace("\r\n", "\n")
    expected = expected.replace("\r\n", "\n").strip("\n")
    languages = "|".join(re.escape(x) for x in LANGUAGES[name])
    fence = re.fullmatch(
        r"[ \t\n]*```(?:" + languages + r")?[ \t]*\n([\s\S]*?)\n```[ \t\n]*", text
    )
    body, extraction = (
        (fence.group(1), "single_full_response_fence")
        if fence
        else (text, "unfenced_full_response")
    )
    body = body.strip("\n")
    return {
        "exact_complete_edit": body == expected,
        "extraction": extraction,
        "response_text": response,
        "extracted_source": body,
        "response_utf8_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "expected_source_utf8_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        "extracted_source_utf8_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "expected_characters": len(expected),
        "extracted_characters": len(body),
    }


def score_document(document, fixtures, tokenizer):
    decode = decoder(tokenizer)
    fixtures = json.loads(Path(fixtures).read_text())
    expected = {
        row["name"]: row["expected_edited_source"]
        for row in fixtures["fixtures"]
        if row["name"] in LANGUAGES
    }
    rows = [("warmup", row) for row in document["lookup_thermal_warmup_measurements"]]
    rows += [("measurement", row) for row in document["measurements"]]
    references, outcomes = {}, []
    for phase, row in rows:
        name = row["scenario"]
        if name not in expected:
            continue
        response = decode(row["output_tokens"])
        if name not in references:
            references[name] = assess_response(name, response, expected[name])
            references[name]["unmapped_token_ids"] = decode.unknown_ids(
                row["output_tokens"]
            )
            if references[name]["unmapped_token_ids"]:
                references[name]["exact_complete_edit"] = False
        elif response != references[name]["response_text"]:
            raise RuntimeError("task diagnostic received unequal response arrays/text")
        outcomes.append(
            {
                "phase": phase,
                "scenario": name,
                "sample": row["sample"],
                "group": row.get("lookup_abba_group"),
                "slot": row.get("lookup_abba_slot"),
                "variant": row.get("lookup_abba_variant", "off"),
                "actual_output_tokens": len(row["output_tokens"]),
                "requested_output_tokens": row["requested_output_tokens"],
                "ended_with_stop_token": row["output_tokens"][-1] in (248044, 248046),
                "exact_complete_edit": references[name]["exact_complete_edit"],
            }
        )
    return {
        "schema": 1,
        "policy": __doc__,
        "included_in_performance_decision": False,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "fixture_tasks": references,
        "all_edit_requests": outcomes,
        "successful_unique_tasks": sum(
            x["exact_complete_edit"] for x in references.values()
        ),
        "unique_tasks": len(references),
        "successful_measured_requests": sum(
            x["phase"] == "measurement" and x["exact_complete_edit"] for x in outcomes
        ),
        "measured_edit_requests": sum(x["phase"] == "measurement" for x in outcomes),
    }
