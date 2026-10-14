# PQ2 M8 target-forward research checkpoint

This checkpoint makes the tested HALF gather candidate, the ordinary forward
comparison harness, and its complete timing samples reviewable. It does not
change the shipping runtime or enable any candidate by default. The new
pair-spread experiment is excluded and will be considered in a later PR.

## What was measured

M5 Pro, eight target verification rows (M8), prefix length 128. Prefill is
untimed setup; draft, sampling, acceptance and scheduler are excluded. These
are target-forward results, not end-to-end serving or token-throughput results.

| Comparison | Original | Candidate | Median paired latency reduction | Winning blocks | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| ORIGINAL / HALF gate+up (V006) | 62.3678 ms | 60.9527 ms | 2.266% | 8/8 | Below frozen 3% threshold |
| ORIGINAL / HALF gate+up + fixed S2 down mapping (V027) | 62.3863 ms | 61.0545 ms | 2.133% | 8/8 | Below frozen 3% threshold |

Each screen uses all 32 timed forwards in eight alternating ABBA/BAAB blocks.
Both routes pass the frozen A/A checks. Median A/A must lie in [0.99, 1.01]
and every block in (0.97, 1.03); promotion additionally requires latency
reduction >=3% and at least seven winning blocks. Neither candidate is promoted.
Absolute columns are medians of block means; the paired gain uses the median
of block ratios, rather than a ratio of those absolute medians.

The standalone gate+up study observed 7.152% / 6.926% lower latency at
M8/K5120/N17408, split1. That local result is not a full-forward gain.
V007's legacy RTN4 result was 69.4062 ms; RTN4/HALF = 1.13869 is a
*different-session descriptive comparison*, not a paired bit-width experiment.
Full UD and context1024 remain unmeasured.

V027 is a combined package. It does not establish an isolated down benefit;
subtracting its 2.133% from the earlier 2.266% would mix sessions and controls.

## Source and mechanism

`source/r233-half-gather.metal` is the exact tested HALF pair construction
source. It changes how packed PQ2 scalar codes become the HALF operand consumed
by the existing MPP operation. PQ2 is scalar integer-coded linear quantization,
not vector-codebook lookup. The current results support a useful local supply
optimization; they do not establish decoding as the dominant full-forward cost.

`source/v024` preserves the exact private combined candidate, host and runtime
snapshots, build identity, source inverses and graph closure. Its down mapping
keeps split2 and the arithmetic but changes logical threadgroup indexing.
Logical adjacency does not prove physical issue order, cache benefit, DRAM
traffic reduction, occupancy or register behavior.

These are source snapshots from the tested base, not patches rebased into the
current production engine. Historical absolute paths and private entry names
are intentional and bound by `PROVENANCE.json`.

## CPU-only review

Run from this directory:

```sh
python3 -B validate_checkpoint.py
```

This checks every copied file hash, runs the archived analyzer's 32 known-answer
cases in a temporary directory, and recomputes every V027 aggregate from all
32 archived samples. It does not initialize Metal or load a model. Validation
is not a new GPU correctness or performance qualification.

The original analyzer and KAT are byte-exact `.py.txt` snapshots in
`source/v025`; the validator restores their original filenames only in its
temporary directory. They are historical source, not shipping Python modules.
All V006,
V007 and V027 timing samples and independent reviews are in `evidence`.
Historical `PENDING_ROOT_REVIEW` fields in analyzer output remain unchanged;
subsequent accepted root reviews supply the final status.

## Correctness and reproduction limits

V026 and the V027 qualification prefix compare all 37 endpoint buffers with
archived ORIGINAL V004 bytes. Candidate and repeat match; this is a differential
identity check, not an independent full-model numerical oracle. Durable raw
arrays cover qualification forwards; main timed forwards check endpoints in
the host. Those raw arrays are external and are not published here.

Original gate/up strict-oracle failures (the strict3/1 cohort), private gate
scratch write coverage per dispatch, other shapes/devices/toolchains,
context1024 and full UD validation remain unresolved. These results do not
waive those requirements or support production integration.

`EXTERNAL-DEPENDENCIES.json` records external weights, prepared caches,
Apple SDK/AIR and original raw arrays. No weights, binaries, metallibs, model
cache or runtime authorization are included. The snapshot cannot launch a
clean-clone GPU benchmark without new input admission and finite authorization.
Per-dispatch/grouped profiling failed stability checks; its descriptive times
are not published here as ordinary-forward exclusive shares.
