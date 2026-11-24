"""Measure packed draft preparation in one native Engine on cold greedy requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "dev/benchmarks"))

PROBE = """#pragma once
#include <atomic>
#include <cstdint>
namespace splash::ops::muse_prepare_probe {
inline std::atomic<bool> enabled{false};
inline std::atomic<uint64_t> packed_calls{0};
inline std::atomic<uint64_t> prepare_calls{0};
inline bool baseline(uint32_t sample) {
  return (sample%4==0 || sample%4==3)==((sample/4)%2==0);
}
inline void select(uint32_t sample) { enabled.store(!baseline(sample)); }
}
"""
COHORT = """    if (selected.shortPrompts) {
      std::ifstream input(std::getenv("MUSE_PREPARE_COHORT"));
      if (!input) throw std::runtime_error("cannot read cohort");
      std::string line;
      while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string name;
        uint32_t cap = 0, token;
        row >> name >> cap;
        std::vector<uint32_t> tokens;
        while (row >> token) tokens.push_back(token);
        if (name.empty() || tokens.empty() || cap != 96)
          throw std::runtime_error("invalid cohort row");
        for (uint32_t sample = 0; sample < samples; ++sample) {
          evictAllCache(resources->cache());
          Measurement result = runRequest(
              engine, driver, *executor, events, progress.get(), requestId++,
              "cohort_" + name, sample, tokens, cap);
          if (result.cacheStatus != "miss")
            throw std::runtime_error("cohort request was not cold");
          measurements.push_back(std::move(result));
        }
      }
    }
"""


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"benchmark seam changed: {old[:80]}")
    return text.replace(old, new)


def prepare(out):
    (out / "PrepareProbe.hpp").write_text(PROBE)
    operator = (
        '#include "PrepareProbe.hpp"\n'
        + (ROOT / "runtime/ops/DraftAttention.cpp").read_text()
    )
    start = operator.index("void DraftAttention::addPrepare(")
    end = operator.index("void DraftAttention::addDecode(", start)
    part = operator[start:end]
    part = replace_once(
        part,
        '  graph.add("draft_attention_qkv",',
        "  const bool packed = muse_prepare_probe::enabled.load();\n"
        "  muse_prepare_probe::prepare_calls.fetch_add(1);\n"
        "  if(packed)muse_prepare_probe::packed_calls.fetch_add(1);\n"
        '  graph.add(packed?"muse_prepare_pack4":"draft_attention_qkv",',
    )
    part = replace_once(
        part,
        "{groups, lanes, 1});",
        "{packed?groups/4:groups, lanes, 1}, {packed?512u:256u,1,1});",
    )
    (out / "PrepareProbe.cpp").write_text(operator[:start] + part + operator[end:])
    bench = (
        '#include "PrepareProbe.hpp"\n#include <fstream>\n#include <sstream>\n'
        + (ROOT / "dev/benchmarks/backend_benchmark.mm").read_text()
    )
    start = bench.index("Measurement runRequest(engine::Engine &engine")
    end = bench.index("\n}\n", start) + 3
    function = bench[start:end]
    brace = function.index(" {") + 2
    function = (
        function[:brace]
        + "\n  ops::muse_prepare_probe::select(sample);\n  const auto beforePacked = ops::muse_prepare_probe::packed_calls.load();\n  const auto beforeCalls = ops::muse_prepare_probe::prepare_calls.load();\n"
        + function[brace:]
    )
    trace = """  std::fprintf(stderr, "PREPARE_REQUEST sample=%u enabled=%d packed_calls=%llu prepare_calls=%llu\\n", sample,
    ops::muse_prepare_probe::enabled.load(),
    static_cast<unsigned long long>(ops::muse_prepare_probe::packed_calls.load()-beforePacked),
    static_cast<unsigned long long>(ops::muse_prepare_probe::prepare_calls.load()-beforeCalls));
"""
    function = replace_once(function, "  return result;", trace + "  return result;")
    bench = bench[:start] + function + bench[end:]
    start = bench.index("    if (selected.shortPrompts) {")
    end = bench.index("\n    if (selected.transcripts)", start)
    (out / "benchmark.mm").write_text(bench[:start] + COHORT + bench[end:])


def command(args):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def baseline(sample):
    return (sample % 4 in (0, 3)) == ((sample // 4) % 2 == 0)


def main():
    import abba

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shader-validation", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build/experiments/draft-preparation-model",
    )
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    samples = 4 if args.shader_validation else 40
    command(
        [
            "make",
            "-j8",
            "build/engine/libsplash.a",
            "build/splash.metallib",
            "build/engine/BuildIdentity.hpp",
        ]
    )
    prepare(out)
    command(
        [
            "xcrun",
            "metal",
            "-std=metal4.0",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Iruntime",
            "-mmacosx-version-min=26.4",
            "-c",
            HERE / "kernels.metal",
            "-o",
            out / "experiment.air",
        ]
    )
    airs = sorted((ROOT / "build/metal").glob("*/*.air"))
    airs = [path for path in airs if path != ROOT / "build/metal/decode/draft.air"]
    command(
        [
            "xcrun",
            "metallib",
            *airs,
            out / "experiment.air",
            "-o",
            out / "experiment.metallib",
        ]
    )

    flags = [
        "-std=c++20",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-mmacosx-version-min=26.4",
        "-Iruntime",
        "-Idev/benchmarks",
        "-I" + str(out),
    ]
    command(
        [
            "xcrun",
            "clang++",
            *flags,
            "-c",
            out / "PrepareProbe.cpp",
            "-o",
            out / "PrepareProbe.o",
        ]
    )
    command(
        [
            "xcrun",
            "clang++",
            *flags,
            "-fobjc-arc",
            "-include",
            ROOT / "build/engine/BuildIdentity.hpp",
            out / "benchmark.mm",
            out / "PrepareProbe.o",
            ROOT / "build/engine/libsplash.a",
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            "-framework",
            "IOKit",
            "-framework",
            "IOSurface",
            "-o",
            out / "benchmark",
        ]
    )
    env = os.environ.copy()
    env["MUSE_PREPARE_COHORT"] = str(HERE / "model-cohort.txt")
    env.pop("MTL_SHADER_VALIDATION", None)
    if args.shader_validation:
        env["MTL_SHADER_VALIDATION"] = "1"
    with (
        (out / "raw.json").open("w") as output,
        (out / "stderr.log").open("w") as error,
    ):
        subprocess.run(
            [
                str(out / "benchmark"),
                str(out / "experiment.metallib"),
                str(args.model.resolve()),
                "--samples",
                str(samples),
                "--scenario",
                "short",
                "--ane-ffn-share",
                "0",
                "--progress",
                str(out / "progress.jsonl"),
            ],
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=error,
            check=True,
            timeout=2400,
        )
    raw = json.loads((out / "raw.json").read_text())
    groups = {}
    for row in raw["measurements"]:
        groups.setdefault(row["scenario"], {})[row["sample"]] = row
    if len(groups) != 6:
        raise RuntimeError("expected all six frozen prompts")
    traces = re.findall(
        r"PREPARE_REQUEST sample=(\d+) enabled=(\d+) packed_calls=(\d+) prepare_calls=(\d+)",
        (out / "stderr.log").read_text(),
    )
    if len(traces) != 6 * samples or any(
        bool(int(e)) == baseline(int(s)) for s, e, _, _ in traces
    ):
        raise RuntimeError("missing or incorrect request gate observations")
    if any(
        int(c) == 0 or int(p) != (int(c) if e == "1" else 0) for _, e, p, c in traces
    ):
        raise RuntimeError(
            "candidate preparation was inactive or the chain arm used it"
        )
    active = sum(int(p) for _, _, p, _ in traces)
    results = []
    for name, rows in groups.items():
        keys = (
            "output_tokens",
            "cache_status",
            "matched_tokens",
            "target_prefill_rows",
            "draft_context_rows",
            "draft_state_restore_skipped",
        )
        if (
            len(rows) != samples
            or len({json.dumps([r[k] for k in keys]) for r in rows.values()}) != 1
        ):
            raise RuntimeError(f"full output/cache mismatch: {name}")
        if rows[0]["cache_status"] != "miss":
            raise RuntimeError("request was not cold")
        result = {"scenario": name, "full_output_and_cache_match": True}
        if not args.shader_validation:
            for metric in ("wall_ms", "ttft_ms", "decode_gpu_ms"):
                blocks = []
                for first in range(8, 40, 4):
                    a = [
                        rows[s][metric] for s in range(first, first + 4) if baseline(s)
                    ]
                    b = [
                        rows[s][metric]
                        for s in range(first, first + 4)
                        if not baseline(s)
                    ]
                    blocks.append(abba.compare(a[0], b[0], b[1], a[1]))
                comparison = abba.compare_samples(zip(*(r["rounds"] for r in blocks)))
                comparison["blocks"] = blocks
                result[metric] = comparison
            count = len(rows[0]["output_tokens"])
            result["baseline_request_tokens_per_second"] = (
                1000 * count / result["wall_ms"]["baseline"]
            )
            result["candidate_request_tokens_per_second"] = (
                1000 * count / result["wall_ms"]["candidate"]
            )
            result["tps_change"] = (
                result["wall_ms"]["baseline"] / result["wall_ms"]["candidate"] - 1
            )
        results.append(result)
    files = [
        HERE / "model_probe.py",
        HERE / "model-cohort.txt",
        out / "PrepareProbe.cpp",
        out / "benchmark.mm",
        out / "PrepareProbe.hpp",
        out / "benchmark",
        out / "raw.json",
        ROOT / "build/engine/libsplash.a",
        out / "experiment.metallib",
        HERE / "kernels.metal",
    ]
    summary = {
        "scope": "One-Engine cold greedy diagnostic on this PR base: production four-dispatch attention in both arms; only preparation kernel changes. Not an HTTP serving benchmark or additive measurement with PR756.",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_diff": subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=ROOT, text=True
        ),
        "identity": raw["identity"],
        "shader_validation": args.shader_validation,
        "packed_calls": active,
        "prompts": results,
        "sha256": {
            (
                str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
            ): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }
    if not args.shader_validation:
        summary["geomean_request_tps_change"] = (
            statistics.geometric_mean(1 + r["tps_change"] for r in results) - 1
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(out), "packed_calls": active, "prompts": len(results)}
        )
    )


if __name__ == "__main__":
    main()
