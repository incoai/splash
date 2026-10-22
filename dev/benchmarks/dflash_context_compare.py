"""Compare native fresh/restored captures only at matching contexts."""

import argparse
import json
import math
import struct
from pathlib import Path

from .draft_replay import read_capture, summarize


def bf16(values):
    if (
        not isinstance(values, list)
        or not values
        or any(type(x) is not int or not 0 <= x <= 65535 for x in values)
    ):
        raise ValueError("invalid BF16 hidden capture")
    result = [
        struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in values
    ]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("nonfinite BF16 hidden capture")
    return result


def compare(left, right):
    lm, lr = left
    rm, rr = right
    # Include censoring/greedy-consistency and complete-file validation from
    # the capture reader rather than trusting arbitrary scalar summaries.
    left_summary, right_summary = summarize(lm, lr), summarize(rm, rr)
    for key in (
        "model_fingerprint",
        "materialize_prefix",
        "verify_rows",
        "budget_learning",
        "prompt_sha256",
        "prompt_tokens",
        "kv_format",
        "state_format",
        "presence_penalty",
        "width",
        "positions",
        "candidates",
    ):
        if key not in lm or key not in rm or lm[key] != rm[key]:
            raise ValueError(f"incomparable capture {key}")

    def key(row):
        digest = row.get("context_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(
                "capture requires a context SHA256 including its pending anchor"
            )
        return row["lane"], row["position"], digest

    lookup = {key(row): row for row in rr}
    if len(lookup) != len(rr):
        raise ValueError("duplicate matched-context row")
    matches = []
    for row in lr:
        other = lookup.get(key(row))
        if other is None:
            continue
        item = dict(
            lane=row["lane"],
            position=row["position"],
            accepted_left=min(row["accepted"], row["retained"] - 1),
            accepted_right=min(other["accepted"], other["retained"] - 1),
            accepted_scratch_left=row["accepted"],
            accepted_scratch_right=other["accepted"],
            verify_rows_left=row["verify_rows"],
            verify_rows_right=other["verify_rows"],
            retained_left=row["retained"],
            retained_right=other["retained"],
            confidence_max_abs=max(
                abs(a - b)
                for a, b in zip(row["confidence"], other["confidence"], strict=True)
            ),
            candidate_slot_changes=sum(
                a != b
                for a, b in zip(row["candidates"], other["candidates"], strict=True)
            ),
            proposed_token_changes=sum(
                a != b for a, b in zip(row["proposed"], other["proposed"], strict=True)
            ),
            target_token_prefix_equal=row["target_tokens"][
                : min(row["retained"], other["retained"])
            ]
            == other["target_tokens"][: min(row["retained"], other["retained"])],
        )
        if ("draft_hidden_bf16" in row) != ("draft_hidden_bf16" in other):
            raise ValueError(
                "both captures must include hidden rows for drift comparison"
            )
        if "draft_hidden_bf16" in row:
            a, b = bf16(row["draft_hidden_bf16"]), bf16(other["draft_hidden_bf16"])
            if len(a) != len(b) or len(a) % (lm["positions"] + 1):
                raise ValueError("hidden rows have incompatible geometry")
            differences = [x - y for x, y in zip(a, b, strict=True)]
            item["hidden_max_abs"] = max(map(abs, differences))
            item["hidden_rms"] = math.sqrt(sum(x * x for x in differences) / len(a))
            norm = sum(x * x for x in a)
            item["hidden_relative_l2"] = (
                math.sqrt(sum(x * x for x in differences) / norm) if norm else None
            )
        matches.append(item)
    if not matches:
        raise ValueError(
            "no identical contexts; divergent trajectories cannot measure restore drift"
        )
    return dict(
        scope="matched-context native capture comparison; no global-memory or speedup claim",
        model_fingerprint=lm["model_fingerprint"],
        prompt_sha256=lm["prompt_sha256"],
        restore_prefix_left=lm.get("restore_prefix", 0),
        restore_prefix_right=rm.get("restore_prefix", 0),
        matched_contexts=len(matches),
        unmatched_left=len(lr) - len(matches),
        unmatched_right=len(rr) - len(matches),
        comparisons=matches,
        left_acceptance=left_summary,
        right_acceptance=right_summary,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fresh", type=Path)
    parser.add_argument("restored", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compare(read_capture(args.fresh), read_capture(args.restored))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
