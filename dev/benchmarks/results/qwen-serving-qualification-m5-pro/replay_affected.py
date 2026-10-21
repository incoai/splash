"""Replay the three affected historical prompts using a matching legacy checkout.

Run from that checkout's root. --source-report accepts the portable expanded
GGUF report: prompts are reconstructed and checked against its hashes before
any model request. All remaining options belong to serving_regression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from dev.benchmarks import serving_regression as serving  # noqa: E402


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-report", type=Path, required=True)
    own, remainder = parser.parse_known_args()
    if (
        "--residency" not in remainder
        or remainder[remainder.index("--residency") + 1] != "sequential"
    ):
        raise ValueError("exact-case replay requires --residency sequential")
    original = json.loads(own.source_report.read_text())
    selected = [
        case for case in original["cases"] if not case["comparison"]["output_equal"]
    ]
    if len(selected) != 3:
        raise ValueError("expected the frozen expanded GGUF report's three cases")

    def cases(args, tokenizer):
        for case in selected:
            hashes = case["prompt_sha256"]
            if args.batch != len(hashes):
                raise ValueError("batch width differs from the source report")
            prompts = [
                serving.prepared_prompt(
                    tokenizer,
                    serving.WORKLOADS[case["workload"]],
                    case["context"],
                    [case["workload"], case["sample"], lane],
                )
                for lane in range(args.batch)
            ]
            if [hashlib.sha256(p.encode()).hexdigest() for p in prompts] != hashes:
                raise ValueError(
                    "reconstructed prompts differ; check tokenizer/tool revision"
                )
            yield {
                **{key: case[key] for key in ("context", "workload", "sample")},
                "prompts": prompts,
                "rounds": [],
            }

    serving.case_specs = cases
    sys.argv = [sys.argv[0], *remainder]
    serving.main()


if __name__ == "__main__":
    main()
