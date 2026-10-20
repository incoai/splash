# Local validation record

Machine: Apple M5 Pro, 20 GPU cores, 48 GiB, macOS 26.6.0. Metal compiler
32023.921. Base commit: `be3b083c5f413673d44e44881042e875e674e333`.
This records local results; it does not represent a multi-device release gate.
Runtime/policy implementation: `c5bde8f`; measurement instrumentation:
`1894964`. Subsequent commits publish documentation and evidence. The runtime
source is unchanged from the measured candidate.

## Source and native checks

| Check | Result and scope |
| --- | --- |
| `make -j8 all` | PASS; normal CLI rebuilt and `./splash --help` succeeds |
| `make -j8 BUILD=build/deepinfer-candidate test-engine-cpu` | PASS |
| `.venv/bin/python dev/tools/check_architecture.py` | PASS |
| `git diff --check` and `git diff --cached --check` | PASS |
| `make -j8 check` | PASS, exit 0; complete CPU/Metal suites, all native test/tool builds, full Python discovery, compileall, Ruff lint and format |
| `make check-source` after packaging the evidence | PASS, exit 0 |

The full check builds all native tests/tools, runs the CPU and Metal suites,
runs Python discovery, compiles Python sources and checks Ruff lint/format.
Its Metal suite uses `MTL_SHADER_VALIDATION=1`.
The final full gate includes the numerical MoE suite after the in-place
refinement, not only its fast live-prefix qualification.

## Focused shader-validated checks

All completed with `MTL_SHADER_VALIDATION=1`, using the candidate Metal library.

| Native executable | Result | Coverage relevant to this change |
| --- | --- | --- |
| `live-verify-rows` | PASS | Chosen-path confidence, invalid inputs, one-to-eight-row limits, acceptance/continuation, stops, limits and sampled/mixed fallback |
| `gguf-moe` | PASS | Final implementation: existing FP64 checks and bitwise live-prefix comparisons across expert formats and B1–B4, repeated in the full `make check` gate |
| `gguf-moe --live-only` | PASS | Final in-place refinement, every supported expert format, staged/register paths, B1–B4 and all cutoff lengths; inactive outputs preserve the residual |
| `target-sampling` | PASS | Existing greedy/sampled acceptance and distribution checks |
| `draft-selector` | PASS | Existing selector numerical/selection contracts |
| `gdn-decode` | PASS | Existing pending-tape/state contracts; no GDN arithmetic was modified |
| `metal-backend` | PASS | Existing backend checks and new trace execution, interval and clock-order checks |

For example, after building the corresponding targets:

```sh
MTL_SHADER_VALIDATION=1 build/deepinfer-candidate/engine-tests/live-verify-rows \
  build/deepinfer-candidate/splash.metallib
MTL_SHADER_VALIDATION=1 build/deepinfer-candidate/engine-tests/gguf-moe \
  build/deepinfer-candidate/splash.metallib --live-only
```

## Real-model oracle matrix

All runs use INT8 KV and `--ane-ffn-share 0`. Both state formats completed the
full oracle, including restore/preemption, same-graph lane equivalence,
permutations, sampling, constraints, thinking budgets and applicable image
checks. Cross-width bitwise identity is not an existing oracle requirement.

| Model | Threshold | FP32 state | BF16 state |
| --- | ---: | --- | --- |
| `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_XXS` | 0.1 | PASS | PASS |
| `mlx-community/Qwen3.8-27B-4bit` | 0.1 | PASS | PASS |
| MoE model, final in-place refinement | 0.2 | PASS | PASS |

The threshold-0.1 MoE runs preceded the final in-place refinement. The final
refinement was qualified with all-format bitwise prefixes and threshold-0.2
real-model oracles; dense execution is unaffected by that refinement.

Pinned installed sources:

| Role | Repository | Revision |
| --- | --- | --- |
| MoE target | `unsloth/Qwen3.6-35B-A3B-GGUF` | `a483e9e6cbd595906af30beda3187c2663a1118c` |
| MoE draft | `incoai/Qwen3.6-35B-A3B-DFlash2` | `51ef7b6923ad6c14cb1bb41c37a9041446496ab6` |
| Dense target | `mlx-community/Qwen3.8-27B-4bit` | `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` |
| Dense draft | `incoai/Qwen3.8-27B-DFlash2` | `015e795645c74b1a0eeef3b570031fb62e769bc5` |

Reproduce each model/state pair with:

```sh
MTL_SHADER_VALIDATION=1 build/deepinfer-candidate/engine-tests/model-runtime-oracle \
  build/deepinfer-candidate/splash.metallib /path/to/installed-model \
  --kv-format int8 --state-format fp32 --ane-ffn-share 0 \
  --live-verify-threshold 0.1
```

Repeat with `--state-format bf16`; use threshold 0.2 for the final MoE matrix.
The safetensors/GGUF target models above are representative installed models,
not all combinations of model family and storage format.

## Python checks

The full Python discovery in `make check` completed **1,270 tests, with 27
skipped**, in 513.888 seconds. Compileall, Ruff lint and Ruff format checks
also pass (158 files checked).

Earlier focused runs include 38 passing tests:

```sh
.venv/bin/python -m unittest \
  dev.tests.tools.test_backend_regression dev.tests.tools.test_smoke_real
```

105 tests pass:

```sh
.venv/bin/python -m unittest \
  dev.tests.server.test_sampling dev.tests.server.test_thinking_token_budget \
  dev.tests.server.test_structured_tools dev.tests.server.test_tool_calls \
  dev.tests.server.test_partial_tool_output
```

## Measurement verification

Every exported native comparison was recomputed from `native-cycles.csv` using
`abba.compare`, reproducing baseline/candidate time, spread, regression bound
and verdict. Every HTTP comparison was recomputed from its raw metric deltas
using `abba.compare_samples`, reproducing the exact stored results. Generated
message hashes and sampled/strict-schema probe hashes match across each series'
four rounds. The checked-in HTTP workload snapshot preserves the tested
requests and measurement logic; formatting and the required model-root option
do not change the prompts or metric calculation.

## Limits

The full cross-device release gate, all model/storage combinations, BF16 KV
qualification and contended p50/p95 latency are not established here. ANE
prefill acceleration is not a claimed benefit. Enabled short-prompt HTTP
comparisons are inconclusive and do not qualify a default threshold. The
original feature is disabled unless explicitly enabled.

`make check` does not include the separate `make test-sanitizers` target; that
target was not run locally for this publication. GitHub's native CI job runs
the sanitizer suite. Its status is reported by the PR checks, independently of
these completed local results.
