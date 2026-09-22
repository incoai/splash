"""Build an observed hot-expert plan from offline Flash expert-ID captures.

No model, inference, GPU or network libraries are used. Counts describe routed
expert assignments in the supplied captures; hit rates are coverage estimates,
not measured cache speedups. The native cache loader must validate coefficient
geometry and its actual allocations before applying a plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "splash-flash-hot-expert-plan-v1"
SOURCE_IDENTITY = "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e"
LAYERS = 48
EXPERTS = 512
TOP_K = 10
MAX_ROWS = 2048
LIMITS = (32, 64, 128, 256)
ALIGNMENT = 16384
MAX_LINE_BYTES = 16 * 1024 * 1024
COEFFICIENT_BYTES_PER_EXPERT = 3 * 640 * 2560 * 2
QUANTIZATION_POLICY = "F32 q*SF+bias once-roundedBF16"


class PlanError(ValueError):
    pass


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PlanError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise PlanError(f"non-finite JSON value: {value}")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)


def _validate_record(record, source_identity):
    if not isinstance(record, dict) or set(record) != {
        "source_identity",
        "phase",
        "rows",
        "layer_ids",
    }:
        raise PlanError(
            "capture record must contain source_identity, phase, rows and layer_ids"
        )
    if record["source_identity"] != source_identity:
        raise PlanError("capture source identity does not match the requested source")
    if record["phase"] not in ("prefill", "decode"):
        raise PlanError("capture phase must be prefill or decode")
    rows = record["rows"]
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise PlanError("capture rows must be an integer in [1,2048]")
    layers = record["layer_ids"]
    if not isinstance(layers, list) or len(layers) != LAYERS:
        raise PlanError("capture must contain exactly 48 layers")
    for index, ids in enumerate(layers):
        if not isinstance(ids, list) or len(ids) != rows * TOP_K:
            raise PlanError(f"layer {index} must contain exactly rows*10 expert IDs")
        if any(type(expert) is not int or not 0 <= expert < EXPERTS for expert in ids):
            raise PlanError(
                f"layer {index} IDs must be integers in [0,511]; booleans are invalid"
            )


def _snapshot(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _round_allocation(size):
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _memory_estimate(selected_counts, limit):
    per_layer = [count * COEFFICIENT_BYTES_PER_EXPERT for count in selected_counts]
    coefficients = sum(per_layer)
    maps = LAYERS * _round_allocation(EXPERTS * 4)
    diagnostics = LAYERS * _round_allocation(4)
    return {
        "scope": "estimated additional cache allocations; original weights and runtime workspace excluded",
        "assumed_geometry": {
            "layers": LAYERS,
            "experts_per_layer": EXPERTS,
            "top_k": TOP_K,
            "hidden_size": 2560,
            "intermediate_size": 640,
            "projections_per_expert": 3,
            "coefficient_dtype": "BF16",
            "allocation_alignment_bytes": ALIGNMENT,
        },
        "coefficients_bytes_per_expert": COEFFICIENT_BYTES_PER_EXPERT,
        "coefficients_bytes_per_layer": per_layer,
        "coefficients_bytes": coefficients,
        "coefficients_gib": coefficients / 1024**3,
        "requested_limit_coefficients_bytes_per_layer": limit
        * COEFFICIENT_BYTES_PER_EXPERT,
        "requested_limit_coefficients_bytes": LAYERS
        * limit
        * COEFFICIENT_BYTES_PER_EXPERT,
        "id_map_allocated_bytes": maps,
        "diagnostics_allocated_bytes": diagnostics,
        "total_estimated_allocated_bytes": coefficients + maps + diagnostics,
        "allocation_assumption": "separate per-layer int32[512] ID maps and uint32 diagnostics, each rounded to 16KiB",
        "native_geometry_and_allocation_validation_required": True,
    }


def create_plan(paths, *, source_identity=SOURCE_IDENTITY, limit=128, phase="all"):
    if (
        not isinstance(source_identity, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_identity) is None
    ):
        raise PlanError("source identity must be a lowercase SHA256")
    if type(limit) is not int or limit not in LIMITS:
        raise PlanError("requested limit must be 32,64,128 or 256")
    if phase not in ("all", "prefill", "decode"):
        raise PlanError("phase filter must be all, prefill or decode")
    paths = list(paths)
    if not paths:
        raise PlanError("at least one capture stream is required")
    counts = [[0] * EXPERTS for _ in range(LAYERS)]
    captures = []
    scope = {
        "phase_filter": phase,
        "validated_records": 0,
        "included_records": 0,
        "excluded_records": 0,
        "included_rows": 0,
        "excluded_rows": 0,
        "rows_by_phase": {"prefill": 0, "decode": 0},
        "count_unit": "routed expert assignment: 10 selections per row per target layer",
        "selection_policy": "positive counts only; count descending then ID ascending; selected maps sorted by ID",
        "capture_validation": "all records validated before phase exclusion; source files unchanged during reading",
    }
    for filename in paths:
        path = Path(filename).resolve()
        try:
            before = _snapshot(path)
            digest = hashlib.sha256()
            capture = {
                "path": str(path),
                "bytes": 0,
                "records": 0,
                "included_records": 0,
                "excluded_records": 0,
            }
            with path.open("rb") as stream:
                line_number = 0
                while raw := stream.readline(MAX_LINE_BYTES + 1):
                    line_number += 1
                    if len(raw) > MAX_LINE_BYTES:
                        raise PlanError(
                            f"{path}:{line_number}: capture line exceeds 16MiB"
                        )
                    digest.update(raw)
                    capture["bytes"] += len(raw)
                    try:
                        record = _json(raw)
                        _validate_record(record, source_identity)
                    except (
                        ValueError,
                        TypeError,
                        UnicodeError,
                        RecursionError,
                    ) as error:
                        raise PlanError(f"{path}:{line_number}: {error}") from error
                    capture["records"] += 1
                    scope["validated_records"] += 1
                    included = phase == "all" or record["phase"] == phase
                    key = "included_records" if included else "excluded_records"
                    capture[key] += 1
                    scope[key] += 1
                    scope["included_rows" if included else "excluded_rows"] += record[
                        "rows"
                    ]
                    if not included:
                        continue
                    scope["rows_by_phase"][record["phase"]] += record["rows"]
                    for layer, ids in enumerate(record["layer_ids"]):
                        for expert in ids:
                            counts[layer][expert] += 1
            if before != _snapshot(path) or capture["bytes"] != before[2]:
                raise PlanError(f"capture changed while reading: {path}")
        except OSError as error:
            raise PlanError(f"could not read capture {path}: {error}") from error
        capture["sha256"] = digest.hexdigest()
        captures.append(capture)
    selected_experts, layer_stats = [], []
    for index, layer in enumerate(counts):
        observed = sum(layer)
        if not observed:
            raise PlanError(
                f"layer {index} has zero observed assignments after phase filtering; no IDs invented"
            )
        hottest = sorted(
            (expert for expert in range(EXPERTS) if layer[expert]),
            key=lambda expert: (-layer[expert], expert),
        )[:limit]
        selected = sorted(hottest)
        selected_count = sum(layer[expert] for expert in selected)
        selected_experts.append(selected)
        layer_stats.append(
            {
                "layer": index,
                "observed_assignments": observed,
                "selected_assignments": selected_count,
                "hit_rate": selected_count / observed,
                "observed_experts": sum(count > 0 for count in layer),
                "selected_count": len(selected),
            }
        )
    observed = sum(layer["observed_assignments"] for layer in layer_stats)
    selected = sum(layer["selected_assignments"] for layer in layer_stats)
    aggregate = {
        "observed_assignments": observed,
        "selected_assignments": selected,
        "hit_rate": selected / observed,
        "selected_count": sum(map(len, selected_experts)),
    }
    if not all(math.isfinite(layer["hit_rate"]) for layer in layer_stats):
        raise PlanError("hit rates are non-finite")
    return {
        "schema": SCHEMA,
        "source_identity": source_identity,
        "requested_limit": limit,
        "quantization_policy": QUANTIZATION_POLICY,
        "selected_experts": selected_experts,
        "captures": captures,
        "scope": scope,
        "layer_stats": layer_stats,
        "aggregate": aggregate,
        "memory_estimate": _memory_estimate(list(map(len, selected_experts)), limit),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "captures",
        type=Path,
        nargs="+",
        help="completed offline expert-ID JSONL streams",
    )
    parser.add_argument("--source-identity", default=SOURCE_IDENTITY)
    parser.add_argument("--limit", type=int, choices=LIMITS, default=128)
    parser.add_argument("--phase", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = create_plan(
            args.captures,
            source_identity=args.source_identity,
            limit=args.limit,
            phase=args.phase,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
    except (PlanError, OSError) as error:
        print(f"Expert cache plan failed: {error}")
        return 1
    print(
        json.dumps(
            {
                "plan": str(args.output),
                "source_identity": plan["source_identity"],
                "limit": plan["requested_limit"],
                "phase": plan["scope"]["phase_filter"],
                "selected_experts": plan["aggregate"]["selected_count"],
                "observed_hit_rate": plan["aggregate"]["hit_rate"],
                "estimated_coefficients_gib": plan["memory_estimate"][
                    "coefficients_gib"
                ],
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
