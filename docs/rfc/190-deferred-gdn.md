# RFC 190: current deferred GDN replay

Related issue: [#190](https://github.com/incoai/splash/issues/190).
Rechecked against main `1482f6bbc2abf12375332725302019594150b6cb`.

## Current implementation

Main already folds a decode step's retained GDN rows into the lane's next
step. Commit [48a94aaa](https://github.com/incoai/splash/commit/48a94aaae6deda31bfbaec08e8a896c2d670e37a)
replaced the earlier standalone commit path. The original September RFC
scaffolding predates this implementation; it is not a second runtime feature
or an environment-controlled optimization.

- [`GdnTapeLane`](../../runtime/ops/GDN.hpp) identifies the previous step's
  pending tape, current step's tape and retained pending row count.
- [`GDN::addDecode`](../../runtime/ops/GDN.cpp) passes those tapes and counts
  to the fused decode kernels, with bounds, alignment and overlap checks.
- The [decode shader](../../runtime/metal/kernels/decode/gdn.metal) folds
  the pending rows into its incoming recurrent state before processing the
  current rows; convolution history also reads the pending input tape.
- [`QwenStateStorage::swapDecodeParity`](../../runtime/model/QwenState.cpp)
  records the retained rows for the next step. Snapshot/copy/materialization
  paths must resolve pending state before publishing a committed snapshot.

The original placeholder tracked `proposed - accepted`, which counts discarded
suffix rows. The runtime needs the retained prefix, including its anchor.
Zero accepted draft proposals therefore does not imply zero retained target
rows, and a full-width retained step still leaves rows pending. The stale
placeholder and its string/Python-mirror tests are removed from this proposal.
`SPLASH_DEFERRED_GDN_REPLAY` is not a supported product option.

## Contracts for further work

Any subsequent replay optimization must consume every previously retained row,
preserve convolution carry and tape ownership, and materialize a complete
state for capture, restore, preemption and cache publication. Previous pending
length and the next step's selected verify length are independent. Cover FP32
and BF16 state, supported packed layouts, lane permutations, zero accepted
draft proposals and partial/full retention.

Changing the floating-point graph requires an explicit numerical contract;
bitwise equality between different graphs is not assumed. Measure a complete
fused cycle and serving behavior, not the removed standalone pass in isolation.

## Review and validation scope

This PR now documents main's implemented mechanism. It changes no runtime,
shader, build registration or product default. Existing
[GDN decode tests](../../dev/tests/engine/gdn_decode_metal_test.mm) and
[model runtime oracle](../../dev/tests/engine/model_runtime_oracle_test.mm)
are the relevant executable checks. Rechecking this documentation is not a
new hardware, throughput or quality result. Issue #190 remains available for
maintainer discussion of any additional optimization beyond the current code.
