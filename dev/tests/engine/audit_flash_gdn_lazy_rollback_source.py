#!/usr/bin/env python3
"""Source-only promotion audit; no Metal, model or compiler execution."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def method(text, signature):
    start = text.index(signature)
    begin = text.index("{", start)
    depth = 0
    for offset in range(begin, len(text)):
        if text[offset] == "{":
            depth += 1
        elif text[offset] == "}":
            depth -= 1
            if depth == 0:
                return text[start : offset + 1]
    raise AssertionError("unterminated method")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", default="build/flash-gdn-lazy-rollback-cpu-v1/source-audit.json"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    original = root / "dev/benchmarks/flash_gdn_lazy_rollback.metal"
    promoted = root / "runtime/metal/kernels/shared/flash_gdn_lazy_rollback.metal"
    implementation = root / "runtime/flash/FlashGDNLazyRollback.cpp"
    forward = root / "runtime/flash/FlashForward.cpp"
    normalized = original.read_text()
    for old, new in (
        (
            '#include "flash_gdn_lazy_rollback.h"',
            '#include "metal/abi/FlashGDNLazyRollback.h"',
        ),
        ("GdnLazyVerifyParams", "FlashGDNLazyVerifyParams"),
        ("GdnLazyReplayParams", "FlashGDNLazyReplayParams"),
        ("GdnLazyCopyParams", "FlashGDNLazyCopyParams"),
        ("private_gdn_lazy_", "flash_gdn_lazy_"),
    ):
        normalized = normalized.replace(old, new)
    assert normalized == promoted.read_text(), (
        "promoted shader differs beyond qualified renames"
    )
    commit = method(implementation.read_text(), "void FlashGDNLazyRollback::commit(")
    assert not re.search(
        r"\b(?:weights?|dense|projections?|project|matmul)\b", commit, re.I
    ), "commit introduced model/weight/dense work"
    pipelines = re.findall(r'graph\.add\("([^"]+)"', commit)
    assert pipelines == [
        "flash_gdn_lazy_replay_sg16",
        "flash_gdn_lazy_restore_convolution",
    ]
    assert "b.mixed, b.decay, b.beta, initialRecurrent()" in commit
    assert "initialConvolution(), b.qkv" in commit
    assert "retained[lane] ? retained[lane] : impl_->rows" in commit
    planner = method(
        forward.read_text(), "uint64_t FlashForward::verificationWorkspaceBytes("
    )
    assert (
        "uint64_t{36} * FlashGDNLazyRollback::plannedBytes(maximumVerifyRows, 1)"
        in planner
    )
    assert "roundAllocation(2 * sizeof(int64_t))" in planner
    assert "roundAllocation(uint64_t{9} * kHyper * 2)" in planner
    assert "roundAllocation(sizeof(uint32_t))" in planner
    assert "if (maximumVerifyRows <= 1) return 0;" in planner
    report = {
        "pass": True,
        "shader_matches_qualified_private_after_renames": True,
        "state_remains_fp32_and_recurrence_order_unchanged": True,
        "commit_reads_saved_prework_only": True,
        "commit_has_no_weight_dense_projection_work": True,
        "terminal_zero_skips_restore": True,
        "forward_planner_preserves_all_ple_allowances": True,
        "gpu_commands": 0,
        "metal_backend_constructions": 0,
        "commit_pipelines": pipelines,
        "files": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (
                original,
                promoted,
                implementation,
                forward,
                Path(__file__).resolve(),
            )
        ],
    }
    destination = root / args.report
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "files"}
            | {"report": str(destination)}
        )
    )


if __name__ == "__main__":
    main()
