# Same-forward penalty oracle: M5 Pro reproduction

Existing report: [#437](https://github.com/incoai/splash/issues/437).
Base: `1482f6bbc2abf12375332725302019594150b6cb`. Machine: M5 Pro,
20 GPU cores, 48 GiB, macOS 26.6.0; Metal 32023.921, clang 21.0.0,
SDK 27.0, deployment target 26.4. Only this machine was used for these runs.

## Reproduction and result

The unchanged current-main oracle fails on Qwen3.6-35B-A3B GGUF UD-IQ3_XXS,
BF16 KV, FP32 recurrent state, GPU-only FFN and shader validation enabled:

```sh
MTL_SHADER_VALIDATION=1 build/engine-tests/model-runtime-oracle \
  build/splash.metallib "$MODEL_ROOT" \
  --kv-format bf16 --state-format fp32 --ane-ffn-share 0 --max-context 32768
```

The target and draft revisions, installed file records and source hashes are
in [model.json](model.json) and [summary.json](summary.json). The original
[failed log](upstream-moe-gguf-bf16-fp32.log) records repetition 1.3, presence
1.5, frequency 0: cold-score loss 0.205502 exceeds allowance 0.1.

The initial correction `17966d05` passes the complete same invocation. Its
[full log](same-forward-moe-gguf-bf16-fp32.log) retains exactly the same
cold-score disagreement as a diagnostic. Twelve penalty/preemption runs check
248 initial/retained target rows, every vocabulary logit and each selected
winner; maximum normalized arithmetic error is `5.90884e-08`.
This supports a false assertion in this case, without claiming that the
underlying prefill/verify numerical difference has been located or repaired.

The later reference-only edge fix `c20ea1e1` preserves prompt-only logits when
repetition is inactive, including positive infinity. Its updated CPU unit
and ASAN/UBSAN checks pass. Four representative final-patch model cells are
queued separately; the initial run is not relabelled as a run of later source.

## What the assertion now checks

The instrumented runtime adds an explicitly installed observer before target
penalties/selection. Its host step snapshots that exact forward's raw fp32
logits into test-owned memory. The oracle independently rebuilds prompt
membership and output counts from its input/transcript, adds the pending
anchor and accepted proposal prefix per retained row, computes the policy in
fp64, and compares every logit with an eight-float-epsilon arithmetic allowance.
It separately requires the selected greedy token to attain the actual
post-policy maximum. It covers initial selection and no-preemption,
prompt-preemption and generation-replay paths for all four penalty fixtures.
It does not infer an exact decode reference by recomputing through prefill.

This is a numerical correctness test, not a timing run. The observer adds a
host step and can perturb command scheduling. The old cold-score diagnostic
still reproduces in that instrumented run. No tolerance was enlarged to make
that old comparison pass. Unretained suffix rows, sampled distributions,
model quality and the cause of cross-graph drift are outside this check;
existing target-sampling and runtime-oracle tests retain their coverage.

## Production isolation and tests

All runtime fields, methods and calls are inside
`SPLASH_RUNTIME_INSTRUMENTATION`; no observer is installed by default.
[production-parity.json](production-parity.json) records identical preprocessed
production `Runtime.cpp` for base and patch (4,279,706 bytes, SHA256
`931abfa5be9f190a387fe30460a4e498f5ac73dedd18b000e78f00a4d63e8ec6`),
after removing blank lines and preprocessor line markers only. Production
runtime behavior, shader code and defaults are unchanged.

- Full [CPU suite](cpu-suite.log): pass; the final updated helper unit also passes.
- [ASAN/UBSAN helper](penalty-oracle-asan.log): pass; [compile command](penalty-oracle-asan.json).
- Shader-validated [target-sampling](target-sampling.log): pass.
- Initial real-model fix run: pass, separately source-scoped above.
- Final representative real-model coverage: queued, not a passing result yet.

The CPU fixture uses hand-calculated positive/negative logits and policies;
it rejects wrong prompt repetition, stale output counts, a missing accepted
prefix even when the winner stays unchanged, wrong winners, nonfinite
corruption and errors beyond fp32 rounding. It accepts a one-ULP difference
and checks positive-infinity prompt/repetition behavior. The unit runs in
`make test-engine-cpu`. The runtime oracle explicitly depends on its helper
header so editing that reference rebuilds the test executable.

Paths in exported logs use placeholders; source and exported hashes are kept
separately. Full original local logs remain available. No model weights or
generated serving answers are included.
