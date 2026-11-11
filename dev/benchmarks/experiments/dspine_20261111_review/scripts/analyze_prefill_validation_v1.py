#!/usr/bin/env python3
"""Classify all six registered prefill validation probes, retaining expected aborts."""

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MODES = {"api": ("1", "0"), "shader": ("0", "1"), "both": ("1", "1")}
ORACLE_OK = "minimal IQ4_XS prefill: one plain dispatch, independent fp64 and buffer guards ok"
INPUTS = {}


def read(path):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if str(path) in INPUTS and INPUTS[str(path)] != digest:
        raise ValueError(f"input changed during analysis: {path}")
    INPUTS[str(path)] = digest
    return data


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def evaluate(root, protocol_path):
    INPUTS.clear()
    protocol_bytes = read(protocol_path)
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes)
    expected_fields = {f"{variant}_{mode}" for variant in ["baseline", "combined"] for mode in MODES}
    require({key for key, value in protocol.items() if isinstance(value, dict)} == expected_fields,
            "registered probe set differs")
    jobs, problems = {}, []
    for variant in ["baseline", "combined"]:
        for mode, (api, shader) in MODES.items():
            field = variant + "_" + mode
            job = protocol[field]
            ident = job["id"]
            row = {"job": ident, "variant": variant, "mode": mode, "valid": False}
            jobs[field] = row
            try:
                command = job["command"]
                require(command[:3] == ["/usr/bin/env", "MTL_DEBUG_LAYER=" + api, "MTL_SHADER_VALIDATION=" + shader]
                        and len(command) == 5, "registered flags or command differ")
                require(ident == f"prefill-iq4xs-{variant}-{mode}-v1", "job identity differs")
                expected_code = -6 if mode == "both" else 0
                require(job["expected_exit"] == ("-6 API assertion" if mode == "both" else "0"),
                        "predeclared expected exit differs")
                receipt = json.loads(read(root / "results" / (ident + ".receipt.json")))
                require(receipt["id"] == ident and receipt["state"] == "complete" and receipt["job"] == job,
                        "receipt identity, state or exact job differs")
                require(receipt["protocol_sha256"] == protocol_sha, "receipt names another protocol")
                require(set(job["artifacts"]) == set(job["expected_sha256"]), "artifact nomination differs")
                require(receipt["artifact_sha256"] == receipt["artifact_sha256_after"] == job["expected_sha256"],
                        "executed artifact identity changed or differs from nomination")
                require(receipt["exit_code"] == expected_code, "exit differs from predeclared expectation")
                stdout = read(root / "results" / (ident + ".stdout")).decode()
                stderr = read(root / "logs" / (ident + ".stderr")).decode()
                resources = [json.loads(line) for line in stderr.splitlines() if line.startswith("{")]
                resources = [value for value in resources if value.get("kind") == "pipeline_resources"]
                require(len(resources) == 1, "missing or duplicate pipeline resource record")
                resource = resources[0]
                require(resource["kernel"] == "gguf_prefill_iq4xs_a" and resource["device"] == "Apple M5 Pro",
                        "kernel or device differs")
                require(resource["MTL_DEBUG_LAYER"] == api and resource["MTL_SHADER_VALIDATION"] == shader,
                        "observed flags differ")
                require(("Metal API Validation Enabled" in stderr) == (api == "1") and
                        ("Metal GPU Validation Enabled" in stderr) == (shader == "1"),
                        "validation layer enablement log differs")
                source = resource["source_stage_bytes"] + resource["source_codebook_bytes"]
                require(resource["source_stage_bytes"] == 16384 and resource["source_codebook_bytes"] == 1024,
                        "source memory accounting differs")
                reported, maximum = resource["pipeline_static_threadgroup_bytes"], resource["device_max_threadgroup_bytes"]
                require(maximum == 32768 and reported == source * (2 if shader == "1" else 1),
                        "reported pipeline memory differs from the observed doubling pattern")
                assertions = re.findall(r"total used threadgroupMemoryLength\((\d+)\) must be <= (\d+)", stderr)
                if mode == "both":
                    require(assertions == [(str(reported), str(maximum))] and reported > maximum,
                            "expected assertion does not name the reported pipeline memory and device limit")
                    require(ORACLE_OK not in stdout, "aborted dispatch unexpectedly completed its oracle")
                else:
                    require(not assertions and ORACLE_OK in stdout and "FAIL" not in stdout,
                            "passing oracle is missing or a resource assertion occurred")
                row.update(valid=True, exit_code=receipt["exit_code"], expected_abort=mode == "both",
                           oracle_passed=ORACLE_OK in stdout, resources=resource,
                           assertion={"used": int(assertions[0][0]), "limit": int(assertions[0][1])} if assertions else None,
                           artifact_identity_before_after_matches=True,
                           artifact_sha256=receipt["artifact_sha256"], elapsed_seconds=receipt["elapsed_seconds"])
            except (OSError, KeyError, ValueError, TypeError) as error:
                row["problem"] = str(error)
                problems.append(ident + ": " + str(error))
    pairs = {}
    for mode in MODES:
        baseline, candidate = jobs["baseline_" + mode], jobs["combined_" + mode]
        same = baseline["valid"] and candidate["valid"] and all(
            baseline[key] == candidate[key] for key in ["exit_code", "expected_abort", "oracle_passed", "resources", "assertion"])
        pairs[mode] = {"same_behavior": same}
    supported = not problems and all(pair["same_behavior"] for pair in pairs.values())
    require(read(protocol_path) == protocol_bytes, "protocol changed during analysis")
    return {
        "classification_supported": supported,
        "classification": "pre-existing API/shader-validation resource-accounting interaction" if supported else "incomplete or inconsistent evidence",
        "observations": [
            "API-only: both libraries report 17,408 static threadgroup bytes and pass the fp64/buffer oracle.",
            "Shader-only: both report 34,816 bytes and pass the oracle with the API layer disabled.",
            "Both layers: both report 34,816 bytes against a 32,768-byte device limit and abort with the same API assertion; these failures remain in the evidence.",
        ] if supported else [],
        "inference": "Shader validation doubles reported static threadgroup memory for this unchanged prefill kernel; the API layer then rejects that instrumented footprint. The final candidate reproduces the baseline behavior, so this failure is not evidence of an M48/QKV regression." if supported else None,
        "jobs": jobs, "paired_modes": pairs, "problems": problems, "inputs_sha256": dict(INPUTS),
        "scope": "Exactly the six registered diagnostic jobs; no timings or throughput conclusions. Artifact provenance is checked against immutable registered before/after execution receipts; this saved-evidence analysis does not reopen the binaries. This is not a Metal-driver root-cause proof or a claim about every kernel/device.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = args.protocol or args.evidence / "notes/prefill-iq4xs-validation-v1.protocol.json"
    result = evaluate(args.evidence.resolve(), protocol.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"classification_supported": result["classification_supported"], "classification": result["classification"]}))


if __name__ == "__main__":
    main()
