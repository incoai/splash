# GGUF K-quant kernels: harnesses, measurements, design notes

Development material for the `gguf-kquant` target format (llama.cpp K-quant, i-quant and
Q8_0 tensors served by `runtime/metal/kernels/shared/kquant.metal`). Everything here was run
on an M5 Pro (16-core GPU, 48 GB) against Unsloth's `Qwen3.8-27B-UD-Q4_K_M.gguf`; paths inside
the scripts point at that machine's `~/dev/q4k-m5` checkout and need adjusting.

Harnesses (build: `clang++ -std=c++20 -fobjc-arc -O2 -framework Foundation -framework Metal
<file>.mm -o <name>`; they compile Metal sources at runtime, so no Xcode Metal toolchain is
needed):

- `harness_kq.mm` + `kq.metal` + `iq3s_grid.inc`: kernel development harness for the eight
  formats (older buffer ABI: input0, w0, meta, output, params, w1). Modes `validate` (fp64
  reference), `quick`/`full` shape sweeps, `serial` (dependent dispatches, mimics in-situ
  bandwidth), `real <variant> <raw> <ref.f32> <N> <K>` (exported GGUF tensors).
- `harness_prod.mm`: validates the production `kquant.metal` kernels (same ABI as the engine)
  against fp64 and times the fused multi-segment, gate+up and split-K paths.
- `harness_native.mm` + `kqn.metal`: the "no repack" experiment, native GGUF block layout
  kernels vs. the repacked layout and vs. splash's Q4 kernels (`results-native-vs-repacked.md`).
- `compile_check.mm`, `inline_metal.py`: runtime compilation helpers (inline `#include`s, then
  `newLibraryWithSource:`).
- `bench_e2e.py`, `run_e2e.sh`, `accept_probe.sh`, `e2e_summary.py`: end-to-end ABAB benchmark
  against `server/server.py` (single stream, 4 concurrent, prefill) with a one-minute gap
  between runs; `entries.py` and `inspect_pkg.py` build the per-tensor result tables from
  harness logs and a converted package.

Results:

- `results-kernels-m5.md`: per-format kernel timings vs. splash's native Q4 kernels,
  bits/weight and theoretical vs. measured speedup.
- `results-per-tensor-m5.md`: one row per (layer, projection, format, rows) cell.
- `results-e2e-m5.md`, `results-batch1-analysis.md`: end-to-end speedups, the batch-1 decode
  attribution (profiler cycles, dispatch counts) and the kernel fusion work.
- `results-native-vs-repacked.md`: why the repacked layout is kept (native layout is 25-40 %
  slower at 8 rows).

Design note: `gguf-in-memory-loading.md` describes loading a GGUF directly, repacking into the
`MDKQ0001` layout in memory at load time instead of shipping a converted package.
