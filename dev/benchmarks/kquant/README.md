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
  `full` covers every gate/up format pair at 8/16/24/32 rows; `time-gu` measures
  fused and separate FFN projections. An optional input scale and seed follow
  the mode; a non-unit scale uses sparse activations to test BF16 input range.
- `harness_native.mm` + `kqn.metal`: the "no repack" experiment, native GGUF block layout
  kernels vs. the repacked layout and vs. splash's Q4 kernels (`results-native-vs-repacked.md`).
- `compile_check.mm`, `inline_metal.py`: runtime compilation helpers (inline `#include`s, then
  `newLibraryWithSource:`).
- `harness_embed.mm`: the three native-row embedding gathers against a CPU reference.
- `harness_image.mm`: builds layer/head/embedding images straight from a GGUF with the
  load-time `kq_repack` / `kq_copy` kernels and compares them byte for byte with converter
  output (`convert_gguf_to_splash.py`). Checked on an M4 with UD-Q4_K_M: layers 0, 1, 3 and 11
  (all eight tensor types), head (Q6_K) and embedding identical; 0.05-0.2 s per layer image
  from the page cache, 0.9 s for the 1 GB head.
- `bench_e2e.py`, `run_e2e.sh`, `accept_probe.sh`, `e2e_summary.py`: end-to-end ABAB benchmark
  against `server/server.py` (single stream, 4 concurrent, prefill) with a one-minute gap
  between runs; `entries.py` and `inspect_pkg.py` build the per-tensor result tables from
  harness logs and a converted package.

Results:

Correctness gates are built by `make build/engine-tests/kquant-projection` and
run by `make test-engine-metal`. The test-only `kquant-dequant.metallib` exposes
the production dequantizers: `kquant-projection <that-library> dequant` requires
exact agreement with FP16 rounding of the FP32 GGUF values, before any GEMM.
The projection checks retain a native-GGUF error check and separately check
the FP16-staged oracle and fused/separate equivalence. The absolute GEMM error
budget excludes the final BF16 rounding cell.

The precision contract follows llama.cpp's
[`dequantize.h`](https://github.com/ggml-org/llama.cpp/blob/7ab4ee7baad2d920464cbacfad4f4b07cf111fd2/ggml/src/ggml-metal/kernels/dequantize.h)
and [`mul_mm.metal`](https://github.com/ggml-org/llama.cpp/blob/7ab4ee7baad2d920464cbacfad4f4b07cf111fd2/ggml/src/ggml-metal/kernels/mul_mm.metal):
FP32 computed group coefficients, one rounding into a half weight tile, FP32
accumulation. Stored half scales without a group multiplier (IQ4_NL/Q8_0) can
use a single half multiply with the same rounding. Activations remain BF16.
To independently verify the CPU reference, build unmodified upstream
`ggml-base` and set `SPLASH_GGML_ORACLE` to its dylib when running the harness.
Every generated native tensor is then checked bit-for-bit against upstream's
`dequantize_row_*` output. This is a development-only optional dependency.

Historical performance results below predate the precision and Apple9 fixes:

- `results-kernels-m5.md`: per-format kernel timings vs. splash's native Q4 kernels,
  bits/weight and theoretical vs. measured speedup.
- `results-per-tensor-m5.md`: one row per (layer, projection, format, rows) cell.
- `results-e2e-m5.md`, `results-batch1-analysis.md`: end-to-end speedups, the batch-1 decode
  attribution (profiler cycles, dispatch counts) and the kernel fusion work.
- `results-native-vs-repacked.md`: why the repacked layout is kept (native layout is 25-40 %
  slower at 8 rows).

Design note: `gguf-in-memory-loading.md` is the plan the GGUF loader (`runtime/model/Gguf*.cpp`)
implements: load the GGUF directly and repack into the `MDKQ0001` layout in memory.
