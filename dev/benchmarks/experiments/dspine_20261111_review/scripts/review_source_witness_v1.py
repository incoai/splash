"""Witness frozen source and bundle consistency without a build or checkout mutation."""

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "combined"
BASE = "37bdd4f8710319bfad43709de2725df89ef776d3"
OUTPUT = ROOT / "results/review-source-witness-v1.json"
PATCHES = {
    "qkv-narrow-v1.patch": "6a830fd355f3373d006ef4debf7081ff7305654fce9538ee8ae91b4438959325",
    "combined-v2.patch": "f27dc90a8d0104b7350f03c58a3ad72ff35c3aa7f5c09f29657cf14680e7e60d",
}
FROZEN = {
    "notes/m48-clean-b3-v2.protocol.json": "9d35ca0282927e9ae5c269f36b2e22ccd54155ec666fa246a13ad8e04c67627c",
    "notes/m48-clean-b3-v2.artifact-barrier.json": "861cf11625550aef6bced699cbc84a17f2ec17e720d851f26712ea208f6c8500",
    "notes/combined-final-regression-v1.protocol.json": "c3f61dc79008848b320bbd810c9c6a271f605943fdbb95df4af775cdcd18ae59",
}
FILES = {
    "dev/native.mk",
    "dev/tests/engine/gguf_projection_test.mm",
    "dev/tests/engine/linear_plan_test.mm",
    "runtime/metal/kernels/shared/gguf_linear.metal",
    "runtime/ops/Linear.cpp",
    "runtime/ops/Linear.hpp",
    "runtime/ops/LinearGguf.cpp",
}
SEEN = {}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observed(path, expected=None):
    path = Path(path)
    digest = sha(path)
    assert expected is None or digest == expected, f"Frozen file changed: {path}"
    assert path not in SEEN or SEEN[path] == digest, f"Input changed: {path}"
    SEEN[path] = digest
    return digest


def read(path):
    observed(path)
    return json.loads(path.read_text())


def git(*arguments, env=None):
    result = subprocess.run(
        ["git", "--no-optional-locks", *arguments],
        cwd=TREE,
        env=env,
        capture_output=True,
        check=True,
    )
    return result.stdout


def compact(text):
    return re.sub(r"\s+", " ", text)


def guard_witnesses():
    paths = {name: TREE / name for name in FILES}
    text = {name: path.read_text() for name, path in paths.items()}
    for path in paths.values():
        observed(path)
    linear = compact(text["runtime/ops/Linear.cpp"])
    dispatch = compact(text["runtime/ops/LinearGguf.cpp"])
    metal = compact(text["runtime/metal/kernels/shared/gguf_linear.metal"])
    checks = {
        "exact_device": (
            'device.appleGpuFamily == 10 && device.gpuCoreCount == 20 && device.deviceName == "Apple M5 Pro"',
            linear,
        ),
        "qkv_layout": (
            "constexpr std::array<uint32_t, 3> widths{12288, 1024, 1024}, offsets{0, 12288, 13312};",
            linear,
        ),
        "qkv_planes": (
            "s.isFloat() || s.formatId != GGUF_FMT_AF4G64 || s.inputSize != 5120",
            linear,
        ),
        "qkv_one_lane": (
            "geometry_.rows() == 16 && w.rows == 16 && w.epilogue == LinearEpilogue::None && !gate",
            linear,
        ),
        "qkv_split": (
            "config.splits == 1 && p.destination == FloatOutput::BFloat16 && measuredAttentionLayout(p)) config.splits = 2;",
            linear,
        ),
        "m48_planes": (
            "af4(projection) && !segments.front().isFloat() && !projection.rotation && !projection.planeInputs()",
            linear,
        ),
        "m48_live_rows": (
            "geometry_.rows() == 16 && w.rows == 48 && w.matrix.outputSize == 17408 && w.matrix.inputSize == 5120",
            linear,
        ),
        "m48_both_tensors": (
            "gate->destination == FloatOutput::BFloat16 && wholeAf4(p) && wholeAf4(*gate)) config.computeRows = 48;",
            linear,
        ),
        "m48_plan_domain": (
            "destination_ != FloatOutput::BFloat16 || storageRows() != 64",
            dispatch,
        ),
        "partial_allocation_storage_stride": (
            "uint64_t{config_.splits} * storageRows() * columns * sizeof(float)",
            dispatch,
        ),
        "compact_dispatch_compute_rows": (
            "const uint32_t rows = plan.computeRows(), splits = config.splits;",
            dispatch,
        ),
        "m48_gate_kernel_partial_stride64": (
            "gguf_decode<FmtAF4G64, 48, EpNone, bfloat, 64>;",
            metal,
        ),
        "m48_up_kernel_partial_stride64": (
            "gguf_decode<FmtAF4G64, 48, EpUpWithGate, bfloat, 64>;",
            metal,
        ),
        "old_kernel_default_partial_stride": (
            "class Out, ushort PartialRows = Rows>",
            metal,
        ),
    }
    for name, (snippet, source) in checks.items():
        assert snippet in source, f"Source guard witness failed: {name}"
    add = dispatch[
        dispatch.index("void Linear::addGguf(") : dispatch.index(
            "void Linear::addGgufStaged("
        )
    ]
    compact_check = add.index("if (plan.configuration().computeRows)")
    assert (
        compact_check
        < add.index("addGgufFloatSegments(")
        < add.index('graph.add("gguf_rotate"')
    )
    assert (
        "!parts.front().isFloat() && parts.front().formatId == GGUF_FMT_AF4G64" in add
    )
    assert (
        "if (!gate || !takesCompact(p) || !takesCompact(*gate)) throw std::invalid_argument"
        in add
    )
    return {
        "checked_predicates": list(checks),
        "explicit_compact_domain_checked_before_float_rotation_or_dispatch": True,
        "interpretation": "Static source witnesses, supported by exact patch hashes; these checks are not execution or performance tests.",
    }


def main():
    assert not OUTPUT.exists(), "Refusing to overwrite a completed witness"
    observed(Path(__file__).resolve())
    for name, digest in FROZEN.items():
        observed(ROOT / name, digest)
    for name, digest in PATCHES.items():
        observed(ROOT / "patches" / name, digest)
    assert git("rev-parse", "HEAD").decode().strip() == BASE
    before_diff = git("diff", "--no-ext-diff", "--no-color", "--binary", BASE)
    assert before_diff == (ROOT / "patches/combined-v2.patch").read_bytes()
    assert set(git("diff", "--name-only", BASE).decode().splitlines()) == FILES
    assert not git("diff", "--cached", "--name-only"), (
        "Combined index has staged changes"
    )
    git("diff", "--check", BASE)
    index = Path(git("rev-parse", "--git-path", "index").decode().strip())
    if not index.is_absolute():
        index = TREE / index
    index_before = sha(index)
    status_before = git("status", "--porcelain=v1", "--untracked-files=no")
    apply_checks = []
    with tempfile.TemporaryDirectory(prefix="dspine-source-witness-") as directory:
        for name in PATCHES:
            alternate_index = Path(directory) / (name + ".index")
            environment = {**os.environ, "GIT_INDEX_FILE": str(alternate_index)}
            git("read-tree", BASE, env=environment)
            git(
                "apply",
                "--cached",
                "--check",
                "--whitespace=error",
                str(ROOT / "patches" / name),
                env=environment,
            )
            apply_checks.append(
                {"patch": name, "base": BASE, "temporary_index_check_passed": True}
            )
    protocol = read(ROOT / "notes/m48-clean-b3-v2.protocol.json")
    barrier = read(ROOT / "notes/m48-clean-b3-v2.artifact-barrier.json")
    for path, digest in barrier["sha256"].items():
        observed(Path(path), digest)
    assert (
        protocol["artifact_nomination"]["control"]["source_label"]
        == "qkv-narrow-v1-6a830fd3"
    )
    assert (
        protocol["artifact_nomination"]["candidate"]["source_label"]
        == "combined-v2-f27dc90a"
    )
    manifests = {}
    for arm in ["control", "candidate"]:
        nominated = protocol["artifact_nomination"][arm]
        build_path = Path(nominated["build_manifest"])
        observed(build_path, nominated["build_manifest_sha256"])
        build = read(build_path)
        assert build == nominated["build_manifest_data"]
        assert build["binary_sha256"] == observed(Path(nominated["driver"]))
        for path, digest in build["inputs"].items():
            observed(Path(path), digest)
        for path, digest in nominated["bundle_sha256"].items():
            observed(Path(path), digest)
        manifests[arm] = {
            "build_manifest": str(build_path),
            "sha256": sha(build_path),
            "verified_inputs": len(build["inputs"]),
        }
    regression = read(ROOT / "notes/combined-final-regression-v1.protocol.json")
    for job in regression.values():
        if not isinstance(job, dict) or "id" not in job:
            continue
        for path, digest in job["expected_sha256"].items():
            observed(Path(path), digest)
    bundle = ROOT / "artifacts/study-combined-v3"
    origins = {
        "splash": "build/splash",
        "splash.metallib": "build/splash.metallib",
        "host-step.metallib": "build/host-step.metallib",
        "linear-plan": "build/engine-tests/linear-plan",
        "gguf-projection": "build/engine-tests/gguf-projection",
        "production-and-test.metallib": "build/engine-tests/production-and-test.metallib",
    }
    for artifact, origin in origins.items():
        assert observed(bundle / artifact) == observed(TREE / origin), (
            f"Bundle copy differs: {artifact}"
        )
    guards = guard_witnesses()
    documentation = Path("/Users/linson/Documents/ai/splash/docs/performance.md")
    observed(documentation)
    docs = documentation.read_text()
    historical_phrases = [
        "The next numerical probe compares target logits at the first divergent token.",
        "Its statistical screen is incomplete.",
        "The fresh agent and separate M48 whole-model protocols remain unrun.",
    ]
    assert all(phrase in compact(docs) for phrase in historical_phrases)
    proposed_append = """### November 11 final-review supplement

The preceding section preserves the thirteen-hour study's cutoff. A separately frozen numerical follow-up restores the measured full-attention QKV reduction at one sixteen-row lane on the 20-core M5 Pro. Its 31 historical greedy arrays match eight-row execution; on the selected earliest-divergence prompt, all 18 captured full-vocabulary target-logit rows match bitwise. This is finite numerical evidence, not a throughput qualification.

The new M48 candidate computes 48 live gate/up rows while retaining 64-row storage and split-partial strides. Eight rows remain the default. Add the final frozen source and artifact identities and all controlled M48, agent, mixed-serving and candidate-sampling verdicts here when complete, retaining failed or inconclusive endpoints.

A separate recovery completes the original sampling seed ranges using retained outputs and only the missing control tail: 280 reserved Holm-corrected tests, zero rejections. The original cutoff report remains incomplete. This recovered historical screen does not test the new QKV/M48 candidate and non-rejection does not establish distribution equality.
"""
    assert git("diff", "--no-ext-diff", "--no-color", "--binary", BASE) == before_diff
    assert git("status", "--porcelain=v1", "--untracked-files=no") == status_before
    assert sha(index) == index_before, "Real index changed"
    for path, digest in SEEN.items():
        assert sha(path) == digest, f"Input changed during witness: {path}"
    result = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "pass",
        "base_commit": BASE,
        "patch_sha256": PATCHES,
        "apply_checks": apply_checks,
        "combined_live_diff_exact_final_patch": True,
        "tracked_changed_files": sorted(FILES),
        "real_index_unchanged": True,
        "tracked_worktree_unchanged": True,
        "untracked_and_ignored_build_outputs_excluded_from_diff": True,
        "bundle_manifests": manifests,
        "bundle_artifact_copies_match_current_build_outputs": origins,
        "source_guards": guards,
        "documentation_review": {
            "path": str(documentation),
            "historical_cutoff_statements_to_preserve": historical_phrases,
            "recommendation": "Append a dated supplement; do not overwrite the original cutoff, failed timing gates or source-specific sampling verdicts.",
            "proposed_dated_append": proposed_append,
            "main_document_edited": False,
        },
        "source_sha256": {str(path): digest for path, digest in sorted(SEEN.items())},
        "scope": "CPU-only current consistency witness. Temporary Git indexes checked both patches against the pinned base without checkout mutation. Frozen nomination/build inputs and artifact copies were rehashed; no compiler, native executable, GPU job, benchmark or independent reproducible-build claim is involved. Timed results remain governed by their own frozen protocols and receipts.",
    }
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "witness": str(OUTPUT),
                "status": result["status"],
                "patches_checked": len(apply_checks),
                "files": len(FILES),
                "hashed_inputs": len(SEEN),
            }
        )
    )


if __name__ == "__main__":
    main()
