# Splash Omni performance notes

Profile of the engine on an M5 Pro (16-core GPU, 48 GB), 4-bit checkpoints,
and the plan that follows from it. Measured with
`dev/benchmarks/omni_step_profile.mm` (pieces of a TTS step, each encoded
alone) and `dev/benchmarks/omni_pipeline_profile.mm` (every dispatch of a
real run, kernel by kernel, through the instrumented backend; a profiled
dispatch runs as its own command, so small kernels carry a few µs of launch
cost they do not pay inside a command).

## Where the time goes

### TTS step, one frame of 80 ms audio (per step command, 892 dispatches)

| Piece | GPU time | Share | Weights streamed |
| --- | ---: | ---: | ---: |
| Talker decode forward, 28 layers + head (198 dispatches) | 4.8 ms | 37% | 790 MB |
| 15 code predictor sub-steps (5 layers + head + sampling + input, 44 dispatches each) | 7.5 ms | 57% | 660 MB (15 × 44 MB) |
| Code-0 sampling, prompt input | 0.1 ms | 1% | |
| Dispatch boundaries inside the command (892 of them) | ~0.6 ms | 5% | |

The step costs the same with 1 or 8 lanes (rows): the Q4 decode tiles stream
each weight once for up to 32 rows, so batching is free until the codec.

The talker's projections run at 165–225 GB/s against roughly 250–300 GB/s of
memory bandwidth (the whole forward at 185 GB/s); the floor for its 790 MB
is about 3.2 ms. The predictor's layers are far from the floor: a sub-step
forward of 5 layers streams 44 MB in 0.42 ms (105 GB/s), about 10 µs per
dispatch, because the N=4096/K=1024-sized projections give the 16 cores
too little work per dispatch and each layer pays 8–9 dispatches. The
per-sub-step sampling kernel (top-k 50 over 2048 codes) costs 58 µs, 0.9 ms
per frame over the 16 samplings.

The lower bound of a frame at the memory floor is about 1.45 GB / 250 GB/s
= 5.8 ms; the step is at 13 ms.

### Codec (speech tokenizer decoder), per frame of audio

| Piece | GPU time per frame | Note |
| --- | ---: | --- |
| `omni_codec_conv` (12.5 dispatches per frame) | 1.6 ms | fp32 implicit GEMM on simdgroup_float8x8, 64×32 tiles |
| transformer, ConvNeXt, SnakeBeta, history | 0.1 ms | |
| per-command fixed cost | ~4 ms per chunk | |

The decoder's convolutions are 2.4 GMAC per frame (the 96-channel stage at
1920 samples alone 0.5 GMAC); 1.6 ms is 3 TFLOPS, about a third of the
GPU's fp32 rate. Chunks of different lanes are decoded as separate dispatch
sequences, so the small-length stages (4 to 32 rows at 1536 and 768
channels) fill a fraction of a 64-row tile.

At 8 lanes the codec's 8 × 1.6 ms per frame equals the step itself, which
is why the server's throughput at 8 streams (19.5 audio s/s) is far below
the step's capacity (8 × 80 ms / 13 ms = 49 audio s/s).

### ASR

Per generated token the text decoder is the talker's shape (28 layers, 4.8
ms) plus the 151936 × 2048 head (175 MB, 0.63 ms) and a 130 µs argmax. At
8 lanes a run of eight 14 s files spends 59% in decode, 37% in prefill
(220-row chunks per request, Q4 prefill tiles at about 18 TFLOPS,
`omni_gemm_bf16` of the audio tower included) and 3% in the mel/conv
front end. The paged decode attention over 220–300 tokens costs 45 µs per
layer at 8 lanes (1.3 ms per step), the fused ≤8-page kernel 16–22 µs.

## What Splash does that applies here

Splash (the LLM engine this derives from) gets its decode speed from
fusing producers with their consumers rather than from whole-layer
kernels: the norm that writes the next projection's operand table and
sums, the attention prepare that does q/k norm + RoPE + staging in one
dispatch, fused residual / gate-up-SiLU / fp32-logit epilogues, the GDN
mixer as one dispatch per (head, lane) passing phases through threadgroup
memory, and one command per decode cycle. Its weight format is the
256-row storage tile of codes, scales and biases that lets a simdgroup load
a contiguous 8 KiB quant group; the activation side is the Table64 X^T
layout with per-group sums so the nibbles enter the MMA as exact bf16.
Splash's kernels are general over N, K and lane counts (plans pick tiles
and splits per shape and GPU); here the shapes are fixed, so the kernels
can be written for exactly these sizes.

## Plan

The sizes the models use, and nothing else:

| Projection | N × K | Rows per dispatch |
| --- | --- | --- |
| Talker / ASR qkv | 4096 × 2048 | ≤ 32 (lanes of 8) |
| Talker / ASR o (+ residual) | 2048 × 2048 | |
| Talker / ASR gate/up (+ SiLU) | 2 × 6144 × 2048 | |
| Talker / ASR down (+ residual) | 2048 × 6144 | |
| Talker head (fp32) | 3072 × 2048 | |
| ASR head (fp32, argmax) | 151936 × 2048 | |
| Predictor qkv | 4096 × 1024 | ≤ 8 (one lane) |
| Predictor o (+ residual) | 1024 × 2048 | |
| Predictor gate/up (+ SiLU) | 2 × 3072 × 1024 | |
| Predictor down (+ residual) | 1024 × 3072 | |
| Predictor head (fp32) | 2048 × 1024 | |
| Text projection (bias, two layers) | 2048 × 2048 | prefill only |

1. **Code predictor sub-step** (7.5 → ~3 ms per frame). Dedicated decode
   kernels for the four predictor projections with the RMSNorm folded into
   the input load (the input is ≤ 8 rows of 1024, every threadgroup
   normalizes it in registers), the K split chosen so each dispatch covers
   the 16 cores several times, and the head + top-k sampling + next input
   embedding as one dispatch (the last-arriving threadgroup of the head
   samples and gathers). Attention over the predictor's one KV page stays
   fused. From 44 to about 22 dispatches per sub-step, and each projection
   near the bandwidth floor.
2. **Codec convolutions** (1.6 → ~0.5 ms per frame). Half-precision
   operands with fp32 accumulation in the implicit GEMM (weights repacked
   to fp16 at load; validated against the fp32 decode), the chunks of all
   due lanes batched into one GEMM per layer (M = lanes × length), and
   tile shapes per stage: 1536/768 channels at 4–32 rows, 384 at 160, 192
   at 640, 96 at 1920 (K = 672).
3. **Talker and ASR decoder** (4.8 → ~3.8 ms per step). The same norm
   folding into the qkv and gate/up kernels (two dispatches fewer per
   layer, no norm output round trip), kernels instantiated for the four
   shapes with splits tuned for 16 cores at ≤ 32 rows, the fp32 head with
   the argmax fused for ASR.
4. **Sampling** (0.9 → ~0.1 ms per frame): one threadgroup per row over
   a 2048/3072 vocabulary, top-k 50 by a two-pass threshold in threadgroup
   memory.
5. **ASR decode attention** at 9+ pages: a split count and tile for 8
   lanes × 8 KV heads over 7–10 pages instead of the general split kernel.

Expected per-frame TTS step after 1, 3 and 4: about 8 ms (13 now); with 2,
8 lanes of codec at about 4 ms per frame instead of 13.

## Results so far

- **Sampling**: the top-k by repeated argmax (58 µs per call) is now a
  radix select over the float order keys with ties to the lowest id, then
  a rank-by-counting order of the kept ids: 21 µs, 0.6 ms per frame.
- **Codec convolutions**: `omni_codec_conv_half` runs the implicit GEMM on
  the MPP matmul over half operands (the A tile gathered from the fp32
  rows and rounded, the weights repacked to fp16 at load) with fp32
  accumulation: 1.07 → 0.51 ms per frame of audio for whole decodes, 1.17
  → 0.58 on 25-frame chunks. Against mlx-audio the rms error of the
  waveform goes from 0.99e-3 to 1.07e-3 (the fp32 reference kernel stays:
  `OMNI_CODEC_FP32=1`).
- **Decode projections, what did not work** (dev/benchmarks/omni_dq4_bench.mm
  times every variant at the models' shapes): folding the RMSNorm into the
  projection by staging the normalized input in threadgroup memory costs
  more than the norm dispatch it saves; streaming the weights two groups
  ahead through a threadgroup ring (`kernels/omni/decode_q4.metal`, opt-in
  with `OMNI_FUSED_DECODE=1`) matches ops::Linear's plans but does not beat
  them; K splits beyond 4-8 lose to the split reduce; an eight-row scalar
  GEMV is compute-bound at 40-50 GB/s. Each MPP quant-group step costs
  about 1-1.5 µs per threadgroup whatever the operand's memory, so a 1-2 MB
  projection has a floor near 10 µs: the predictor is bound by its chain
  of ~44 dependent dispatches per sub-step, not by bandwidth.
- Measured on this GPU: streaming reads peak at 300 GB/s from 128+
  threadgroups; a dispatch boundary inside a command costs about 1.8 µs; a
  grid-wide spin barrier across resident threadgroups costs 1.7 µs at 16
  groups, 3 µs at 64 and 5.8 µs at 128, and deadlocks past about 160
  groups of 256 threads (the limit of co-residency), which a bounded spin
  must guard.
- **Predictor sub-step 0** ran its two rows per lane on the prefill
  kernels (about 1.4 ms per frame); the fused decode attention now takes
  two rows per sequence (`omni_attention_decode_fused_r2`), so the sub-step
  runs on the decode path like the other fourteen. The step went from 13.1
  to 11.9 ms at 1 and at 8 lanes.
- **ASR**: the argmax over the 151936-wide rows is a 1024-thread group with
  vector loads (130 → 18 µs per step); the fused decode attention covers
  up to 16 pages, so contexts to 512 tokens avoid the split and reduce
  pair.
- **Server, 4-bit, after this round** (`dev/benchmarks/audio/RESULTS.md`):
  TTS time to first audio 71 → 61 ms and RTF 0.19 → 0.17 at one stream;
  throughput 15.4 → 19.5 audio s/s at 4 streams and 19.5 → 31 at 8. ASR
  unchanged within noise (44 / 105 / 133 audio s/s at 1 / 4 / 8 files).

## Neural Engine prefill split

The 4-bit ASR decoder's prefill FFN runs a quarter of its intermediate
channels on the Neural Engine, beside the GPU (DEVELOPMENT.md, "Neural
Engine prefill"). Measured on the M5 Pro with
`dev/benchmarks/omni_ane_ffn_bench.mm`: a command of 28 FFN layers of the
ASR shape (random Q4 weights), GPU time per layer, median of 5, split
against the GPU alone.

**The ANE's weight format** (one FFN function of 3072 channels, the ANE
alone, evaluations chained): fp16 weights read from surfaces 8.2 ms at
2048 rows (9.4 TFLOPS); int8 weights and activations 4.1 ms (18.8 TFLOPS);
at 2048 channels int8 weights with fp16 activations 4.3 ms against W8A8's
2.8 ms. W8A8 it is; with the weights resident (rotated and quantized once,
280 MiB at a quarter of the channels) the prefill command spends no GPU
time restaging them, which Splash's larger models need.

**Share and least chunk.** The GPU alone takes 0.45 ms per FFN layer at
128 rows, 0.88 at 256, 1.71 at 512, 3.34 at 1024 and 6.55 at 2048. The
split's gain, by the ANE's units of 512 of the 6144 channels:

| Rows | 2/12 | 3/12 | 4/12 | 5/12 | 6/12 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | +5% | −25% | −55% | −74% | −97% |
| 160 | +7% | +16% | −28% | | |
| 192 | +12% | +16% | −6% | | |
| 256 | +12% | +16% | +7% | −4% | −20% |
| 384 | +9% | +18% | +14% | +4% | −17% |
| 512 | +10% | +19% | +27% | +19% | +4% |
| 1024 | +11% | +19% | +28% | +26% | +11% |
| 2048 | +12% | +20% | +29% | +33% | +18% |

The ANE's evaluations carry about 0.3 ms of fixed cost (0.31 ms at 128
rows and 1024 channels) and slow the GPU's part beside them a little;
7/12 and 8/12 lose at every size. At 3/12 the GPU's part takes longer than
the ANE's at every chunk (5.2 against 2.6 ms at 2048 rows), so chunks pad
up to the next function's rows for free. 4/12 gains more from 448 rows,
but its int8 error is too large: on the real weights the parity prompt's
prefix logits reach a cosine of 0.99887 to mlx-audio's (0.99921 on the GPU
alone, 0.99937 at 3/12), below the 0.999 the parity requires. Hadamard
blocks of 512 or 2048 over the inputs, or the intermediate rows left in
fp16, did not lower the error: int8's resolution bounds it, not outliers.
So the engine runs 3/12 from 160 rows, one program of 13 functions.

**End to end** (`omni-asr-run`, 8 lanes, median of three alternating
rounds; `OMNI_DISABLE_ANE=1` for the GPU alone):

| | GPU alone | Split | |
| --- | ---: | ---: | ---: |
| 8 LibriSpeech files, prefill GPU time | 267.4 ms | 249.0 ms | −6.9% |
| 8 LibriSpeech files, wall | 973.1 ms | 966.7 ms | −0.7% |
| long300_a (3950-token prompt), prefill GPU time | 763.5 ms | 692.9 ms | −9.2% |
| long300_a, wall | 9.00 s | 8.94 s | −0.7% |

Per chunk: 166 rows 31.1 → 28.3 ms, 276 rows 44.4 → 41.9, 315 rows 49.1 →
43.8, 323 rows 53.6 → 47.9, 2048 rows 335.1 → 298.4, 1902 rows 428.5 →
394.2; the four chunks under 160 rows run on the GPU alone. The FFN is
about 60% of a 300-row chunk's GPU time (norms, attention, the QKV and
output projections and the head the rest), so a 16-19% faster FFN is
7-11% of the chunk. The transcripts match but for one token of 382 (a
comma read as a full stop) in the 8 files, and all 1001 of the long
file's. The first start compiles the program in about 2 s; later starts
set the split up in 0.2 s and check it against the GPU alone (2.1% of
the ANE's part on the ASR weights).

The ASR engine prefills one request per command, so a batch of short files
mostly runs chunks under 400 rows, where the split gains least. Prefilling
the requests a tick admits as one ragged chunk would put them in the
range where both the GPU's tiles and the split do best.

## Neural Engine codec

At 8 streams the codec's commands took 26% of the GPU's time (445 of
1714 ms on the test text, `OMNI_TTS_DEBUG=1`), serially after the steps.
Its convolution stack (initial convolution through output convolution,
~95% of its FLOPs, 2.3 GMAC per frame) now runs on the Neural Engine
beside the steps (`ops::omni::CodecAneStack`, DEVELOPMENT.md "Neural
Engine codec"); the GPU keeps the front, about 0.1 ms per frame, encoded
at the head of the next step command.

What the program's shape came from (`omni_ane_stack_probe`):

| Variant (25 frames unless noted) | Result |
| --- | --- |
| Transposed convs as MIL `conv_transpose` (k = 2s, crop s) | correct alone, wrong inside the stack; bias wrong at 1536 -> 768 |
| Phase-stacked conv + `transpose` to interleave | correct; 4-16 frames compile, 18+ do not |
| Phase-stacked conv + interleaving `concat` | correct, same speed as the transpose |
| Whole tensors (to 32768 columns) | 18+ frames refused by the compiler (17 frames, 32640 samples, compile) |
| Late blocks in pieces of <= 32768 / 8192 / 4096 / 2048 samples | 21.2 / 18.6 / 16.6 / 20.0 ms |

Each piece reads the histories the piece before it left, so a function
is exact against chunked evaluation. The functions (two lanes' surfaces 3.6
MB, the program compiled in 14-15 s once):

| Frames | ANE evaluation | GPU codec command (front included) |
| ---: | ---: | ---: |
| 4 | 3.9 ms | 4.4 ms |
| 8 | 6.4 ms | |
| 16 | 11.3 ms | |
| 25 | 16.7 ms (0.67 ms per frame) | 14.2 ms (0.57 ms per frame) |

The ANE is slower per frame than the GPU's half-precision kernels, but it
is idle otherwise: at 8 streams it needs about 5.4 ms per 12 ms step.
Accuracy: streamed through the engine's schedule the ANE stack differs
from the whole GPU decode by rms 5.9e-4 (peak 0.89) and from mlx-audio by
1.3e-3, against 1.07e-3 for the GPU's half convolutions; the test text's
audio differs from the GPU codec's by rms 2.1e-4, same length.

`omni-tts-run` on the test text (89 frames per request, the median of
three runs; per-step GPU times from `OMNI_TTS_DEBUG=1`):

| Codec | 1 lane: RTF | first audio | audio s/s | 8 lanes: audio s/s | first audio | GPU time | step / step with fronts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU | 0.167 | 71.1 ms | 6.00 | 31.7 | 104.6 ms | 1714 ms | 11.85 ms + 12 codec commands (445 ms) |
| ANE, first chunk too | 0.161 | 74.9 ms | 6.19 | 40.8 | 92-117 ms | 1305 ms | 12.00 / 13.91 ms |
| ANE, first chunk on the GPU (default) | 0.162 | 71.5 ms | 6.18 | 40.0 | 106 ms | 1330 ms | 11.96 / 13.73 ms |

The first chunk on the ANE costs one stream 3-4 ms of first audio (the
front waits for the next step command, then the evaluation), and at 8
streams the eight 4-frame evaluations queue on the ANE (first audio
92-117 ms instead of 106 for all); on the GPU it keeps one stream's first
audio and costs 2% of throughput, so it is the default
(`OMNI_CODEC_ANE_FIRST=ane` sends it to the ANE). A step carrying the
due lanes' fronts takes up to 2 ms more.

Server (`run_benchmark.py tts --engine omni`, 60 requests, 4-bit):

| Codec | c | TTFA p50 / mean / p90 | RTF median | audio s/s | WER en | CER zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU | 1 | 61.6 / 61.7 / 62.3 ms | 0.166 | 6.0 | 0.45% | 0.18% |
| ANE | 1 | 61.5 / 61.6 / 62.1 ms | 0.160 | 6.3 | 0.36% | 0.42% |
| GPU | 8 | 94.5 / 97.5 / 127.2 ms | 0.237 | 30.8 | 0.36% | 0.30% |
| ANE | 8 | 83.9 / 89.2 / 125.0 ms | 0.189 | 37.0 | 0.45% | 0.36% |

Eight streams gain 20% throughput and 11 ms of median first audio; one
stream keeps its first audio and gains 4% RTF; WER and CER stay within
the runs' noise (0.36-0.45%, 0.18-0.42%). The codec is on the ANE by
default; `--disable-ane` or `OMNI_CODEC_ANE=0` keep it on the GPU.

## Where it stands, and what is left

Per frame of TTS at one stream the step is now 11.9 ms: the talker 4.8
ms (memory-bound at 185 GB/s against a 300 GB/s ceiling, 198 dispatches),
the predictor 15 × 0.46 ms (dispatch-chain-bound, 44 dispatches a
sub-step), the codec 0.5 ms per frame on top. The levers that remain, by
expected gain:

1. The predictor's chain: fusing qkv with attention per KV head (the last
   partition of the projection's K split does the head's attention), and
   the head with the sampling and the next embedding gather, would take a
   sub-step from 44 to about 33 dispatches (~0.1 ms, 1.5 ms per frame).
2. A persistent megakernel for the predictor is not worth it on this GPU:
   a grid barrier costs 3.5 µs at 64 threadgroups (device atomics
   serialize), more than a dispatch boundary, and the per-phase latency
   is the MPP step itself; only cross-phase weight prefetch could pay,
   and the threadgroup-memory operand path it needs measured slower.
3. The talker's projections at 185 GB/s: wider tiles and splits were tried
   (dev/benchmarks/omni_dq4_bench.mm) without a consistent gain; the
   remaining ~1.4 ms would need a different weight layout (e.g. scales and
   biases interleaved with the codes so a group is one contiguous 4.5 KB
   read) rather than tile tuning.
4. The codec at 8 streams now runs on the Neural Engine (above); what is
   left on the GPU is the fronts (up to 2 ms in a step that carries eight).
   Batching the due lanes' fronts into one pass, and the ANE's chunks of
   all due lanes into one evaluation of a batched function, would cut both.

## Round 3: ASR against sglang-omni, and GPU/ANE parallelism

- **Long-form ASR** was the one place sglang-omni led (49.6 against 35
  audio s/s on 60 s and 300 s files, one request at a time). Its MLX decode
  runs a 300 s file as one stream at about 5.6 ms per token at 4k context,
  ours at 7.7 ms. The server now cuts audio longer than `--segment-seconds`
  (default 30) at the quietest 100 ms near each boundary into at most one
  segment per lane and decodes the segments concurrently: 115-123 audio
  s/s with WER unchanged (1.40-1.51%). Short files: 46 / 109 / 140 audio
  s/s at 1 / 4 / 8 concurrent (sglang-omni 44.5 / 57.6 / 63.5).
- **Decode attention** reads each page as 32 contiguous 512-byte loads
  instead of 144 strided ones, eight pages per split: 85 µs per layer at
  4.4k tokens (18 MB of KV, 215 GB/s), long-form decode 8.6 → 7.7 s per
  1001 tokens.
- **Microbenchmarks must stream from DRAM**: the first projection
  benchmark reused one weight copy, which sat in the system cache and
  favoured the wrong kernels. `omni_dq4_bench` now cycles copies larger
  than 256 MB. From DRAM, the one-row GEMV with a per-shape K split beats
  the tiles on most shapes, but in the decoder only the predictor gains
  (2%), and its summation order flips near-tie greedy codes against the
  reference, so it stays opt-in (`OMNI_GEMV_DECODE=1`).
- **What did not work**:
  - Fusing a predictor layer into two dispatches (norm + QKV slice +
    attention + output-projection share per KV head; norm + gate/up chunk
    + down share per channel chunk) with scalar matrix-vector code: 4-6x
    slower. A dispatch boundary costs ~1.8 µs, while concentrating a
    projection into 8-24 threadgroups loses the MPP tiles' spread over the
    cores.
  - A second Metal queue for the codec: two concurrent 4-lane processes
    together reach 23 audio s/s against 31.7 for one 8-lane process; the
    GPU does not overlap these workloads.
  - Wider codec convolution tiles: the 64 x 64 x 32 tile stays best or
    within noise on every stage (`omni_codec_bench`, 10-14.5 TFLOPS); the
    codec is compute-bound on the GPU.
- **The Neural Engine** (runtime/ane, from Splash): a 6144 x 2048 fp16
  matmul over 1024 rows runs at ~18 TFLOPS pipelined
  (`omni_ane_probe`), and every codec decoder stage's residual unit,
  SnakeBeta included, compiles for it and runs at 4.4-8.3 TFLOPS
  (`omni_ane_codec_probe`), about 1.4x slower than the GPU but beside it.
  Two offloads follow from this: the ASR prefill FFN split (Splash's
  design) and the TTS codec's convolution stack.

## Round 4: the first Codex review

Codex reviewed the architecture and the operators (static read, no
benchmarks) and ranked nine suggestions. Taken so far:

- **Projected predictor tables** (its first item): sub-steps 1..14 fed
  the predictor `small_to_mtp(table[step - 1][code])`, a gather and a
  projection of a fixed function of the code. TtsModel projects all 14
  tables at load (56 MiB) through the same decode dispatch a step of up to
  eight rows runs, so greedy speech is bit-identical at 1 and 8 lanes on
  both checkpoints; decode is about 1% faster (Codex estimated 3-6%).
  `OMNI_PREDICTOR_TABLES=0` restores the old path.
- **Split grids sized to the active splits**: attention split dispatches
  launch the most splits any sequence takes instead of the maximum.
- **Batched ASR admission** and the **one-frame first chunk** (its second
  and fifth items) landed just before the review; the **Neural Engine
  codec** (its sixth) is merged above.
- **Host time per TTS step**, measured (`OMNI_TTS_DEBUG=1`): 0.15 ms to
  build the 863-dispatch graph and about 0.45 ms of encoding and
  completion around 11.6-11.8 ms of GPU time, about 5% of the step. The
  step and its codec work stay serial; overlapping the next step's
  encoding needs the step inputs read on the GPU and double-buffered
  tables.

Where the ASR decode step goes (`omni-pipeline-profile`, one 4-bit row,
~100 tokens of context, 5.28 ms per step in a real command): Q4
projections at 160-195 GB/s per dispatch against the 275 GB/s the
175 MB head reaches, fused attention ~18 µs per layer even at four pages
(isolated dispatch, so inflated by launch latency), RMS norms ~0.26 ms per
step. The weight floor (968 MB) is about 3.3 ms.

What did not work:

- **Segmenting short files** to fill idle lanes (`--segment-seconds` 4 or
  6 instead of 30) on the 30-file subset: one request at a time goes from
  45.8 to 68.8 audio s/s, but WER rises from 2.63% to 3.09-3.20% and 8
  concurrent files fall from 144 to 127 audio s/s (more prompts, more
  prefill). The default stays 30 s.

### bf16 decode, the harness, and the second and third reviews

- **bf16 GEMV by rows** (`omni-dense-bench`, weights from DRAM): the GEMV
  kept eight rows of sums whatever the step's rows and reread each row's
  input per output column. Compiled for at most R rows and C columns per
  simdgroup, with each lane's summation order unchanged (bit-identical),
  one row streams 250-290 GB/s on every model shape (qkv 94 -> 61 µs,
  predictor o 53 -> 16 µs); five or more rows take the 32x64 MPP tile,
  twice the GEMV at eight rows. Over HTTP, bf16 ASR 18.6 / 42.5 / 48.5 ->
  22.3 / 52.7 / 70.2 audio s/s at 1 / 4 / 8 files (64 long-form), bf16 TTS
  9.8 -> 16.0 audio s/s at 8 streams.
- **TTFA in the harness**: the TTS client read with `read(4096)`, which
  waits for 4096 bytes; the one-frame first chunk is 3840 bytes, so first
  audio was timed at the second chunk. With `read1`, one 4-bit stream's
  first audio is 23 ms (it read as 51).
- **Second review** (the Q4 decode projections): four-simdgroup pipelined
  split tiles tuned per shape, deeper register prefetch, a one-dispatch
  split gate/up, RMSNorm folded into the QKV and gate/up prologues with
  the same rounding, repacking only if loads stall. Being worked through.
- **Third review** (speculative ASR decoding with Qwen3-ASR-0.6B as the
  draft): teacher-forced on the 30-file subset, the 0.6B's argmax matches
  the 1.7B's greedy token at 95.1% of positions (5.7 tokens per verify
  round at k = 6). Codex's estimate: 1.1-1.3x end to end for 4-bit at one
  request (the draft's 28 layers are latency-bound, ~2.5-3.3 ms a step),
  1.6-2.1x for bf16, a loss at 8 concurrent files unless gated off.
- **Audio tower tiles**: at 151 rows the tower's bf16 projections run at
  9-13 TFLOPS, compute-bound; the best tile differs by shape by ~25 µs
  per layer, ~0.6 ms per 12 s file, not worth a policy.
- **Host time per TTS step**: 0.15 ms of graph building and ~0.25 ms in
  the submit call (encoding 863 dispatches). Splitting the step into a
  talker command and a predictor command built while the talker runs
  would hide ~0.3 ms (2.5%); not done.

## Round 5: one lane's decode projections (Codex items 1, 2 and 5)

Every decode step of up to eight streams runs its projections as one lane
of eight rows, so the one-lane plans set most of the step. Before this
round Apple10 ran them on the unpaired Split128 tile (eight simdgroups,
K split by apple10Splits), and gate/up on the unpipelined N256 tile (24
threadgroups for the talker) or, where split, as a gate pass plus an
up-with-SiLU pass. `LinearTile::LaneSplit128` replaces all three for one
lane: the paired N128 tile (two quant groups' matmuls in flight, eight
simdgroups) over 1 to 8 K partitions, the last-arriver reduction extended
to the gate/up pair (both projections accumulate in each threadgroup and
publish their partials; the SiLU product is taken once both are complete),
one partition storing directly. The split count is apple10Splits' rule
(fill at most four threadgroups per core) with at least two 256-input
blocks per partition.

Per dispatch, projection alone (`omni-dq4-bench --sweep --rows N`:
weights cycled from DRAM, three interleaved rounds, best of each; one lane
with N live rows, M5 Pro). Before is the old plan's configuration, timed
in the same run; GB/s counts codes, scales and biases:

| shape | before, 1 row | after, 1 row | before, 8 rows | after, 8 rows |
|---|---|---|---|---|
| talker / ASR qkv 4096 x 2048 | 24.3 µs, 194 GB/s | 23.2 µs, 203 | 24.4 µs, 193 | 23.2 µs, 203 |
| talker / ASR o 2048 x 2048 +res | 15.1, 157 | 14.4, 164 | 15.0, 157 | 14.3, 165 |
| talker / ASR gate/up 2 x 6144 x 2048 | 68.5, 207 | 55.7, 254 | 68.3, 207 | 55.8, 254 |
| talker / ASR down 2048 x 6144 +res | 35.2, 201 | 34.0, 208 | 35.1, 202 | 33.6, 211 |
| talker codec head 3072 x 2048 f32 | 20.3, 175 | 18.8, 188 | 20.2, 175 | 18.8, 188 |
| predictor qkv 4096 x 1024 | 14.1, 168 | 13.5, 175 | 14.1, 167 | 13.5, 175 |
| predictor o 1024 x 2048 +res | 11.2, 105 | 10.5, 112 | 11.2, 105 | 10.5, 112 |
| predictor gate/up 2 x 3072 x 1024 | 24.2, 146 | 18.3, 193 | 24.3, 146 | 18.3, 193 |
| predictor down 1024 x 3072 +res | 16.4, 108 | 13.8, 129 | 16.2, 109 | 13.7, 129 |
| predictor heads 2048 x 1024 f32 | 10.0, 118 | 9.7, 121 | 9.9, 119 | 9.7, 121 |
| small_to_mtp 1024 x 2048 +bias | 11.2, 106 | 10.5, 112 | 11.2, 105 | 10.5, 113 |

Before: Split128 at s2 (qkv, heads 3072), s4 (o, down, predictor heads)
or s8 (the predictor's 1024-wide projections), the N256 gate/up tile for
the talker and two Split128 passes at s2 for the predictor's gate/up.
After: LaneSplit128 at s2 (qkv, heads, predictor gate/up), s4 (o, down)
or s1 (talker gate/up). The kernel works on all eight rows of the lane
whatever their count, so one to eight live rows cost the same.

Real steps (interleaved with the previous build, two runs each, the median
step of each run): ASR one file 5.28 → 4.92 ms per step (RTFx 36 → 38),
eight files 5.95 → 5.58 ms (RTFx 128 → 133); TTS one lane 11.59 → 10.55
ms GPU per frame (RTF 0.160 → 0.148); eight lanes 12.1 → 11.1 ms GPU per
step, 39.0 → 43.1 audio s/s. ASR tokens are unchanged. At equal split
counts the new kernels are bit-identical to the tiles they replace (greedy
TTS is identical with the old counts); the predictor's 1024-wide
projections now split 4 ways instead of 8 (and its heads 2 instead of 4),
which moves greedy TTS codes from frame 10 on. Real-weight parity passes
on both models. Keeping the old split counts would keep greedy
bit-identical at about 0.2 ms per TTS frame.

What did not pay (same benchmark):

- **Four simdgroups (128 threads) or N64 tiles**, paired or not, at every
  split: never faster than the eight-simdgroup N128 paired tile beyond
  0.1 µs, mostly 3-21% slower (N64/four: qkv 25.0 vs 23.2 µs, o 14.8 vs
  14.3, gate/up 57.5 vs 55.8, predictor down 14.3 vs 13.7; N128/four:
  gate/up 63.2, predictor gate/up 22.2 vs 18.3). N256 of eight simdgroups
  ties on the talker's qkv and loses elsewhere.
- **Scale/bias prefetch**: loading each quant group's scales and biases
  into registers before its matmul is issued (instead of in the epilogue)
  made the best tile 2-6% slower on every shape (gate/up 58.5 vs 55.9 µs,
  predictor down 14.5 vs 13.7), and changes FMA contraction.
- **Four quant groups in flight** instead of two: within noise. One A/B
  showed 0.3-0.7 µs gains on the predictor's short partitions and 6-7%
  losses on the talker's qkv and down; a second run (on a busier GPU)
  reversed the predictor gains.
- **Publishing and reducing only the live rows** (an upper bound: row 0
  only) was not faster on any shape: 5-10% slower in two runs, including
  the unsplit gate/up whose path it does not touch, so the build's code
  generation moved too; either way there is no gain to take. A tile's
  partials are 4 KiB.
- **Code lookahead through a threadgroup ring** (the opt-in
  `OMNI_FUSED_DECODE=1` kernels already stage codes, scales and biases two
  groups ahead): 20-30% slower than the paired tile on the residual
  projections (talker down 43 vs 33.6 µs, o 18.8 vs 14.3).

## Round 6: the decode norms folded into the projections, rotations once per forward

Each decoder layer ran norm → qkv, attention, o (+residual), norm →
gate/up, down (+residual): two of its six dispatches were RMS norms. A
step of one or two live rows now folds both into the one-lane projection
that reads them (`LinearPlan::foldsNorm`, kernels
`decode_linear_q4_n128_lane_split{,_gate_up}_norm_r{1,2,4,8}`): every
threadgroup of the LaneSplit128 tile reduces the live rows' squares in
the order `rms_inverse` uses (logical thread t of 256 adds columns t + 256 i
in order, `simd_sum` per simdgroup, the eight simdgroups' sums in order),
then stages each 256-input block of its K partition as
bf16((x · inverse) · weight) in a double-buffered threadgroup operand and
takes the block's Q4 input sums from those rounded values. The projection
therefore sees the bits the norm dispatch wrote, and its outputs are
bit-identical to norm + projection. The other rows of the lane are zeros,
so only the live rows cost a reduction. The final norm stays a dispatch:
the TTS predictor reads the talker's normalized final rows
(`arena.selected`); folding it into the codec head changed the speech.
`OMNI_FOLD_NORM=0` restores the dispatches (same bits).

Per dispatch, the norm and the projection against the folded projection
(`omni-dq4-bench --rows N --plan-only`: weights cycled from DRAM, three
interleaved rounds, best of each; µs, M5 Pro; projection alone in
brackets):

| shape | 1 row | 2 rows | 4 rows | 8 rows |
|---|---|---|---|---|
| talker / ASR qkv 4096 x 2048 (23.3) | 25.3 → **22.8** | 25.2 → **23.6** | 25.2 → 25.7 | 25.2 → 29.9 |
| talker / ASR gate/up 2 x 6144 x 2048 (55.9) | 57.9 → **56.0** | 57.9 → **56.7** | 57.8 → 58.9 | 57.8 → 64.1 |
| predictor qkv 4096 x 1024 (13.4) | 15.1 → **13.8** | 15.1 → **14.5** | 15.1 → 15.8 | 15.1 → 18.7 |
| predictor gate/up 2 x 3072 x 1024 (18.3) | 20.2 → **18.7** | 20.1 → **19.0** | 20.1 → 20.1 | 20.0 → 22.6 |

Each threadgroup reads every live row whole (32 KiB at eight rows of
2048, 64 threadgroups for the talker's qkv), so the fold pays up to two
rows, ties at four and loses at eight, and the policy folds one or two.
At one row the folded tile beats the projection alone on the talker's
qkv: the padded threadgroup operand is no slower than the device rows.

**Rotations.** The fused decode attention computed `precise::cos` and
`sin` of each row's 64 angles in three simdgroups of each of the eight
KV-head threadgroups, in every layer. A forward's first layer now stores
them in the arena's rotation table (`Rotations::Store`, KV head 0) and
the later layers load them with the row (`Rotations::Load`). With no
trigonometry at all the predictor's forward took 6 µs less; the table
gets all of it (`omni-step-profile`: predictor forward 0.371 → 0.365 ms,
sub-step 0.415 → 0.410 ms; the talker's forward about 40 µs less).

Real steps, the previous build (base), the fold, and the fold with the
rotation table, alternating, four rounds, medians (`omni-asr-run`
`OMNI_ASR_DEBUG=1`, one and eight files; `omni-tts-run --seed 1`
`OMNI_TTS_DEBUG=1`):

| | base | fold | fold + rotations |
|---|---:|---:|---:|
| ASR one file, GPU per step | 4.98 ms | 4.86 ms | 4.80 ms |
| ASR one file, RTFx | 39.8 | 40.8 | 41.3 |
| ASR eight files, GPU per step | 5.77 ms | 5.76 ms | 5.72 ms |
| ASR eight files, RTFx | 132.1 | 132.3 | 133.1 |
| TTS one lane, dispatches per step | 788 | 582 | 582 |
| TTS one lane, GPU per step | 10.55 ms | 10.32 ms | 10.15 ms |
| TTS one lane, RTF / audio s/s | 0.148 / 6.76 | 0.145 / 6.92 | 0.143 / 7.02 |
| TTS eight lanes, GPU per step | 11.10 ms | 11.11 ms | 11.10 ms |
| TTS eight lanes, audio s/s | 42.97 | 43.04 | 43.0 |

(One eight-lane run of the last build, taken while the machine was busy,
11.8 ms and 38.7 audio s/s, is left out.) Two TTS streams fold too
(the predictor's first sub-step, two rows per stream, does not): 10.57 →
10.42 ms per step, 13.22 → 13.41 audio s/s. Eight streams do not fold and
are unchanged.

Bit identity: greedy TTS at one, two and eight lanes gives the previous
build's wav byte for byte, and ASR's tokens of one, two and eight files
are unchanged, with and without `OMNI_FOLD_NORM=0`. Real-weight parity
passes on both models.

What did not pay, or had to change (same benchmark):

- **The staged operand at a 512-element row stride**: every row of the
  matmul's operand load fell in the same threadgroup banks; the folded
  talker gate/up took 72.0 µs and the predictor's 22.6 against 58.5 and
  20.3 for norm + projection. A 528-element stride (520 measures the
  same) fixes it: 57.9 and 19.2.
- **The device rows as the operand** (an upper bound, wrong values): 24.7
  and 58.7 µs, no faster than the padded threadgroup operand.
- **Prefetching the first two quant groups' codes** before the reduction:
  no change (24.6 against 24.7 µs on the talker's qkv, 14.5 against 14.2
  on the predictor's).
- **Row loops left to the compiler**: the reduction's loads issued row by
  row, 34.9 µs on the talker's qkv at eight rows (30.8 unrolled). Staging
  with a simdgroup per row instead of per (row, quant group) was 2-4 µs
  faster on the gate/up pair and the head at eight rows (not on qkv) and
  1.4-2.5 µs slower at one; the fold runs at one or two rows, so the
  kernels keep the second.
- **Moving the sines and cosines** in the attention kernel (before the
  norm's reduction) changed the backend's fused multiply-add contraction
  of the rotation and the greedy speech; the rotation is now written as
  the two `fma` forms the compiler had chosen (found by trying the nine
  combinations of none, `fma(a, c, -b·s)` and `fma(-b, s, a·c)` per
  output), which every mode shares.

## Round 7: speculative ASR with a Qwen3-ASR-0.6B draft

One request at a time was the weakest ASR benchmark: an 11.6 s file spends
81% of its time in decode steps, and a step costs about the same for one
row as for eight (bandwidth bound). `serve --draft-model
mlx-community/Qwen3-ASR-0.6B-4bit` runs the 0.6B beside the 1.7B as a
greedy draft (`AsrEngine::speculate`): per round, one command has the
draft take the tokens it has not seen (its catch-up), propose k tokens per
lane in k - 1 more decode steps (each step's embedding reads the token the
previous step's argmax wrote, strided in place: `Sampling::addArgmax`'s
output stride and the Q4 embedding's token stride), then the target
verifies every lane's pending token and proposals in one forward and
takes every row's argmax. A lane emits the proposals that match the
target's tokens and the target's token after them (its correction, or a
bonus token when all match). Rollback is the committed counts only: KV
rows past them are written again later. The transcripts are the target's
greedy transcripts up to the numerics of verification (below).

**The draft.** Qwen3-ASR-0.6B shares the 1.7B's attention shape (16 query
/ 8 KV heads of 128) and the tower's 480-channel front end; its decoder is
1024 wide (intermediate 3072, the code predictor's shapes) and its tower
896 wide, 18 layers of 14 heads, FFN 3584, output 1024. The tower's
dimensions now come from config.json (`ops::omni::AudioEncoderDims`), the
dense encoder attention takes a runtime head count (`OmniDenseParams::heads`,
tested at 16 and 14), and the Q4 embedding has a hidden-1024 entry. Alone
(`omni-asr-run` on the 0.6B, 30-file subset, one file at a time): WER
2.51% (mlx-audio's 0.6B: 2.63%, the 1.7B's: 2.63%), decode step 2.86 ms
GPU before round 5's tiles and 2.44 ms after (`omni-asr-verify-bench`).

**Verification cost.** The existing arenas did not fit: decode attention
takes one or two rows per sequence, and a prefill arena runs the Q4
prefill tiles, which cost a flat 11.4 ms from 1 to 32 rows on the 1.7B
(7.0 ms on the 0.6B). A verify arena (`Qwen3Arena(..., verify)`) runs the
decode projections (one lane of 8 rows per 8 rows) with the ragged
prefill attention, up to 32 rows. `omni-asr-verify-bench`, history 200
tokens, GPU ms, medians, after round 6 (the bf16 rows before round 5):

| rows in the forward | 1 | 2-8 | 9-16 | 17-24 | 25-32 |
|---|---:|---:|---:|---:|---:|
| 4-bit 1.7B, one lane: decode step 4.82 | 5.18 | 5.23-5.29 | 6.23 (9) | | |
| 4-bit 1.7B, four lanes: decode step 5.02 | | 5.35-5.36 | 6.35-6.37 | 9.85-9.86 | 10.39-10.43 |
| bf16 1.7B, one lane: decode step 12.83 | 13.38 | 15.0 (2), 15.8 (3), 16.7 (4), 19.5-20.5 (5-8) | 19.9 (9) | | |
| bf16 1.7B, four lanes: decode step 15.90 | | 15.96 (4), 19.6 (8) | 19.7-19.8 | 19.9-20.0 | 20.1 |
| 4-bit 0.6B draft, one lane: decode step 2.44 | 2.78 | 2.82-2.85 | | | |

So a 4-bit round verifies up to 16 rows (k shrinks with the lanes: k =
min(5, 16 / lanes - 1)) and a bf16 round up to 32 (k = 9). The prefill
attention costs about 0.35 ms more than the fused decode attention at one
row; a draft catch-up of one row per lane runs as a decode step, of more
in the verify arena.

**k at one lane** (engine, `omni-asr-run --sequential`, 30 files, wall s,
before rounds 5-6; plain decoding 4-bit 7.60 s, bf16 16.81 s):

| k | 2 | 3 | 4 | 5 | 6 | 7 | 9 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4-bit wall | 6.60 | 6.41 | 6.43 | **6.32** | 6.31 | 6.34 | | |
| 4-bit tokens per round | 2.91 | 3.80 | 4.65 | 5.54 | 6.38 | 6.99 | | |
| bf16 wall | | 10.25 | | 9.92 | | 9.32 | **9.16** | 9.10 |
| bf16 tokens per round | | 3.77 | | 5.47 | | 6.92 | 8.18 | 9.39 |

The draft agrees with the target more than teacher-forced mlx-audio
predicted (0.951 per token): k = 4 gives 4.65 tokens per round against
4.33. Defaults: k = 5 (4-bit), 9 (bf16).

**Lanes** (engine, `omni-asr-run --concurrency N`, every request drafted,
before the policy below; wall s for the subset):

| | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| 4-bit plain | 7.60 | 4.56 | 3.10 | 2.34 |
| 4-bit speculative | 6.32 (1.20x) | 4.43 (1.03x) | 3.36 (0.92x; k=3), 3.52 (k=1) | 3.01 (0.78x) |
| bf16 plain | 16.81 | 9.83 | 6.76 | 5.03 |
| bf16 speculative | 9.16 (1.83x) | 6.02 (1.63x) | 4.48 (1.51x) | 3.79 (1.33x) |

At four 4-bit lanes the rounds themselves still save decode time (15.7 ms
for 15.4 tokens against 5.41 ms per four tokens), but every request pays
the draft's tower and prefill, about 25 ms for 11 s of audio (4 ms + 0.12
ms per prompt row), which the 1 ms a token saved does not repay. bf16
saves about 7.5 ms a token.

**Policy.** Lanes speculate while at most `--speculative-lanes` are active
(4-bit 1, bf16 8). The draft takes a lane at the first tick that would
speculate (its tower, then a prefill of the prompt and the tokens the lane
emitted so far), and only when the tokens its audio likely has left (3 per
second) times the milliseconds saved per token repay the tower and
prefill, and while the active lanes averaged over half a second stay
within the limit. A lane whose draft falls more than 4 tokens behind
(plain steps while too many lanes are active) or runs out of KV pages is
not taken again. Lanes the draft does not follow ride in a round's verify
forward with one row (a plain step's token); a 4-bit target speculates
only while every active lane has a draft, since a rider's one token per
19 ms round costs more than plain steps (5.6 tokens per 19 ms against 4
per 10 ms for two lanes), while a bf16 round with a rider still wins (9
per 43 ms against 2 per 13 ms).

What did not work, on the way to the policy (HTTP, 4-bit, c=2, subset):
drafting every request admitted while at most two lanes were active gave
58.8 audio s/s against 77.4 plain: a request admitted while three were
active (a long file's segments) decoded one token per 17 ms round beside
a speculating lane (a 31.6 s file took 2.82 s instead of 0.72). Riders
brought it to 63.9; a 4-bit target refusing riders, 73.9; drafting at
admission still paid for drafts that more requests made useless right
after (74.4; long-form -5%); lazy adoption took lanes alone for a tick
between two requests (14 drafts for 19 rounds); the load average fixed it.

**Results over HTTP** (subset of 30, `run_benchmark.py asr`, M5 Pro,
after round 6; audio s/s, p50 latency; long-form at c=1):

| | c=1 | c=2 | c=4 | c=8 | long-form |
|---|---:|---:|---:|---:|---:|
| 4-bit plain | 49.7, 203 ms | 78.8, 224 ms | 115.3, 296 ms | 148.3, 498 ms | 137.3 |
| 4-bit `--draft-model` | **56.3, 185 ms** | 78.1, 225 ms | 115.0, 297 ms | 146.9, 504 ms | 134.8 |
| bf16 plain | 22.2, 448 ms | 34.1, 509 ms | 48.8, 752 ms | 63.6, 1106 ms | 61.1 |
| bf16 `--draft-model` | **39.1, 246 ms** | **52.8, 356 ms** | **72.1, 581 ms** | **86.2, 877 ms** | **85.5** |

WER is unchanged everywhere (subset 2.63% 4-bit, 2.74% bf16; long-form
1.51% / 1.24%). Speculation pays plainly for bf16 (1.4-1.8x); at 4-bit it
is worth 13% for one request at a time and nothing above, so it stays
off by default (`--draft-model` is opt-in) and, when on, speculates alone.
sglang-omni's c=1 is 44.5 (4-bit) and 20.4 (bf16).

**Token parity** (engine, one file at a time, after round 6): bf16 30/30
files token-identical to plain decoding; 4-bit 29/30, the other differing
at token 61 of 69 by a comma ("flesh and blood, and" → "flesh and blood
and"), a near-tie that the verify forward's different attention kernel and
row count flip. `omni_decoder_test` checks the verify arena's logits
against the fp64 reference (within 0.009 of a 2.0 scale, as decode steps).

**Memory.** The draft adds 1.30 GiB to the server (both formats): its
weights (0.6B Q4 decoder and bf16 tower, about 0.85 GiB) and the engine's
draft arenas, tower workspace (120 s of audio at most) and the target's
verify arena (+430 MiB in `omni-asr-run`). The draft's KV pages come from
the shared pool: a speculating lane holds about twice the pages.

**Tests.** `omni_asr_speculation_test` (in `make test-engine`) runs small
decisive models (a vocabulary permutation over random layers, so
verification's numerics never flip a token) through plain and speculative
engines in both formats: the target as its own draft (every proposal
accepted), a draft that disagrees on a quarter of the tokens (rounds that
accept 0..k proposals), page boundaries, a stop token inside a round,
token limits under a round, lanes that stop speculating and catch up,
riders, drafts dropped and taken late, and cancellations; every request
must emit exactly the plain decoding's tokens.

Levers left: a fused attention for verify rows (0.35 ms of a 5.3 ms
verify at one lane) and a two-row fused catch-up for the draft (0.35 ms
of 2.8); adapting k to the observed acceptance; a cheaper draft tower and
prefill (the 0.6B's prefill costs 7 ms at any row count up to 32, its
tower 0.65 ms a layer), which decides whether a 4-bit target gains at two
lanes.

## Round 7b: the TTS code predictor

A 4-bit TTS step at one stream was 10.16 ms of GPU: the talker's forward
(~4.1 ms) and the code predictor's 15 sub-steps (~0.38 ms each: a 5-layer
forward, its head, a sampling). A second Codex review ranked five cuts to
the predictor; three landed.

**One-page attention** (`omni_attention_decode_page{,_r2}`). The
predictor's history never leaves its first page (at most 17 tokens), yet
the fused decode attention gave that page to one of eight simdgroups,
which scored all 32 tokens for each query vector in turn, then merged
eight simdgroups' states, seven of them empty. A decode step whose
sequences fit one page now runs a kernel of four simdgroups per row: the
prepare is the fused kernel's (now one shared function), then each query
simdgroup runs its own vector over only the loads holding a token it sees,
scoring the loads of earlier steps while the row's key is stored, and
writes its output with no merge. The arithmetic per vector is the paged
kernel's, so outputs are bit-identical (`omni_kernels_test` compares the
two exactly over a predictor-shaped sequence on stale pages). Chained
dispatches (`omni-step-profile`): one row at 8 tokens of history 6.8 →
4.0 µs, sub-step 0's two rows 14.8 → 2.5 µs (one simdgroup ran four query
vectors before). `OMNI_ONE_PAGE_ATTENTION=0` keeps the paged kernel.

**First-layer QKV tables.** Sub-steps 1..14 feed the predictor a projected
table row of the sampled code (round 4), so layer 0's input norm and QKV
projection are a fixed function of the code too. TtsModel runs every
projected row through them at load (`Qwen3Decoder::addFirstQkv`, the
dispatches a decode step of eight rows runs, or four rows for the bf16
GEMV at up to four lanes) into 14 tables of [2048][4096] bf16 (224 MiB;
load +0.07 s 4-bit, +0.3 s bf16), and a sub-step whose row count computes
each row with the same arithmetic (`sameFirstQkv`: one to eight rows on
the Q4 one-lane plans, the folded and unfolded norms alike; the dense
GEMV's one to four rows or the tile's five to 64) gathers its rows' QKV
beside their hidden and starts at layer 0's attention
(`ForwardStep::firstQkvReady`). Other row counts, the GEMV and the
pipelined Q4 paths compute it as before. `OMNI_PREDICTOR_QKV_TABLES=0`
skips the tables.

**The head folds the final norm** at one or two rows: the forward leaves
the selected rows raw (`ForwardStep::rawSelected`) and the head's one-lane
plan normalizes them with the norm dispatch's bits (15 fewer dispatches a
step). The talker keeps its materialized normalized rows.

Real steps, current main and the three steps, alternating, three rounds,
medians (`omni-tts-run --seed 1`, `OMNI_TTS_DEBUG=1`; GPU ms per step and
audio s/s; eight lanes are eight identical streams):

| | main | + one-page attention | + QKV tables | + head fold |
|---|---:|---:|---:|---:|
| 4-bit one lane | 10.16, 7.03 (RTF 0.142) | 9.69, 7.36 | 9.54, 7.47 | **9.47, 7.48 (RTF 0.134)** |
| 4-bit two lanes | 10.40, 13.45 | 9.85, 14.12 | 9.72, 14.25 | **9.71, 14.34** |
| 4-bit eight lanes | 11.04, 43.05 | 10.53, 44.98 | 10.33, 45.84 | 10.36, **45.64** |
| bf16 one lane | 21.44, 3.54 | 20.95, 3.62 | 20.51, 3.70 | **20.51, 3.69** |
| bf16 eight lanes | 32.77, 16.47 | 32.25, 16.70 | 31.80, 16.94 | **31.68, 16.87** |

Greedy TTS at one, two and eight lanes and seeded TTS (`--seed 1`) at
4-bit give the previous build's wav byte for byte after every step, on
both checkpoints for greedy; real-weight parity passes on both models.

What did not pay:

- **The sampler.** Top-k 50 sampling costs ~16 µs more than greedy argmax
  per sub-step (`omni-step-profile`: forward + head 0.348 ms, + greedy
  0.352, + top-k 0.369). Early returns put ~7 µs in the four radix passes,
  ~2-4 µs each in collecting, ranking and the draw. Reworked with the
  same selections and draws (bit-identical seeded speech): one barrier per
  radix pass (every simdgroup scans the histogram itself), the keys in
  registers with contiguous ids per thread, the kept ids placed by a
  prefix sum instead of atomics (ties exact without a second pass), the
  ranking reading kept values directly, the draw's running sums once with
  a parallel search. Chained samplings went 15.9 → 13.8 µs, but in a
  sub-step (forward + head + top-k) and in real steps nothing moved
  (9.52 → 9.55 ms at one lane, 10.31 → 10.26 at eight). Aggregating a
  simdgroup's atomics per bin did not change the sub-step either.
- **Gathering the next sub-step's inputs in the sampler** (its threads
  copy the drawn code's hidden and QKV rows after a barrier): 28 fewer
  dispatches a step, no faster (9.47 → 9.52 ms at one lane, 10.22 → 10.26
  at eight).
- **Loading the page's keys up front** in the one-page kernel: all 16
  loads unconditionally 5.4 µs, the visible loads into a register array
  before scoring 5.0, values prefetched too 4.4, against 4.1 for loading
  in the scoring loop (4.0 with the earlier steps' loads scored during the
  prepare). Without the device-memory barrier the kernel took the same.

Not done, with the reasons:

- **Sub-step 0's last layer for the selected row only.** The one-lane
  projections cost the same for one to eight live rows, so dropping row 0
  pays only when it saves a lane: five to eight streams, whose 16 rows run
  two lanes (`omni-step-profile`: the forward at 16 rows 0.466 ms, at 8
  rows 0.347; the last layer's O and FFN are at most ~20 µs of that, 0.2%
  of a step).
- **Batching the due lanes' codec fronts.** A step carrying eight fronts
  takes 12-14 ms instead of ~10.3 (about 100 dispatches a front), but with
  eight synchronized streams that is one step in 25 (~0.07 ms a step on
  average, less with unsynchronized streams), and every front kernel
  (lookup, causal convolutions and ConvNeXt with per-stream histories,
  the windowed transformer with its own positions) would need per-stream
  operands.

## Round 8: speculative ASR refinements

Four items from a Codex review of round 7, in order, each measured and
kept only where it paid. Engine figures are `omni-asr-run` on the 30-file
subset, one file at a time unless noted, wall seconds; HTTP figures are
`run_benchmark.py asr` (audio s/s), the round-7 binary (main at e33d597)
and this round's alternated on a quiet machine (load average below 2.5).

**1. A fused attention for verification and the draft's catch-up.** A
verify arena ran its attention through the prefill kernels (prepare,
split, reduce), about 0.67 ms over 28 layers at a 200-token history
whatever the rows. `attention_verify.metal` (`Attention::addVerifyFused`)
runs one threadgroup per (KV head, row): it prepares its row's two query
heads and the key and value of every row of its sequence up to its own
(the rows before it store the same bits into the same slots), then runs
the fused decode kernel's page loop and merge for its two query vectors.
A verified row's attention is thus a decode step's arithmetic: the
transcripts are now token-identical to plain decoding on 30 of 30 files
in both formats (4-bit was 29 of 30, the comma of round 7). The page loop
and its helpers moved into `kernels/common/omni_paged_attention.h`
(`attention.metal` keeps identical copies for now, to stay out of the
predictor's work; folding them in is a mechanical follow-up).

Every layer's attention alone (`omni-asr-verify-bench`, 4-bit 1.7B, GPU
ms over 28 layers, history 200; the bench now prints these columns):

| rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill kernels, one lane | 0.64 | 0.65 | 0.65 | 0.65 | 0.66 | 0.67 | 0.67 | 0.68 | 0.70 | 0.75 |
| fused verify, one lane | 0.30 | 0.29 | 0.30 | 0.31 | 0.33 | 0.34 | 0.42 | 0.43 | 0.52 | 0.59 |
| fused decode (one or two rows a lane) | 0.30 | 0.59 | | | | | | | | |
| prefill kernels, two lanes | | 0.74 | | 0.74 | | 0.75 | | 0.78 | | 0.78 |
| fused verify, two lanes | | 0.38 | | 0.39 | | 0.40 | | 0.44 | | 0.64 |

A row's threadgroup takes the history in rounds of 8 pages, one per
simdgroup, about 7 µs a round at one lane, while the prefill kernels
spread the history over more threadgroups: at 300 tokens (10 pages) six
rows cost 0.56 ms against 0.57, at 500 0.65 against 0.79, at 600 (19
pages) 0.85 against 0.74. So the kernel takes forwards whose rows times
their page rounds stay within 12, up to 16 pages; the prefill kernels
take the rest. The two-row fused decode kernel (0.59 ms) costs twice the
per-row kernel, so the draft's two-row catch-ups take the verify kernel
too. A whole 4-bit verify forward of 2-8 rows went from 5.2-5.3 ms to
4.86-4.93 (a decode step: 4.81).

Engine: 4-bit 6.32 → 6.22 s; bf16 9.01 → 9.01 (its 10-row verification
is 19.5 ms of projections; the attention saved is in the noise).

Did not help: a threadgroup per pair of rows (four query vectors in the
page loop) cost 0.66 ms at 2-8 rows, no better than the prefill kernels,
since the loop's latency doubles with its vectors and one lane leaves the
GPU idle; merging the simdgroups' states in a tree over a quarter of the
threadgroup memory changed nothing; issuing a page's 16 key loads into
registers before scoring took 0.42 ms against 0.30 (register pressure).

**2. A shortlisted draft head: not adopted.** The 0.6B draft's head reads
all 151936 rows (87 MB of Q4). `omni-asr-verify-bench --head-rows N` runs
it over a view of the first N rows: a draft step costs 2.43 ms with the
full head, 2.24 at 65536 rows, 2.16 at 32768, 2.13 at 16384, 2.11 at
8192, so a round of five draft forwards would save 1.0-1.5 ms of 17.4.
The shortlist was derived from the tokenizer alone: every one of Qwen's
151387 merges has id 256 + its merge rank (checked over vocab.json and
merges.txt), so low ids are the earliest, most frequent merges, and the 62
added tokens (151643-151704, among them `<|im_end|>` and `<asr_text>`)
would be appended with an id map. But the earliest merges are not what
transcripts are made of: of the 1.7B's tokens on the English subset,
12.5% lie outside the first 16384 ids, 7.7% outside 32768 and 2.6%
outside 65536; Chinese merges came late (14 CJK tokens below id 16384,
159 below 32768, 26071 in all, single characters at a median id of
119471), so 85% of the TTS benchmark's Chinese prompt tokens lie outside
16384 and 67% outside 65536. Every target token the draft cannot propose
ends a round there. Engine runs with the head over the leading N rows
(no id map, so without the added tokens, two rejections a file):

| head rows | 151936 | 98304 | 65536 | 32768 | 16384 |
|---|---:|---:|---:|---:|---:|
| tokens per round | 5.60 | 4.98 | 4.72 | 4.15 | 3.80 |
| wall s | 6.26 | 6.46 | 6.59 | 7.21 | 7.27 |

Acceptance falls faster than the head's cost; even 98304 rows lose, and
any size would collapse on Chinese. Ranking rows by the draft head's row
norm (a model-derived prior) was worse still (93% of English tokens
outside the top 16384). The full head stays.

A non-English check for the record: 30 Chinese files (synthetic Mandarin
renditions of the TTS benchmark's zh prompts already on disk, resampled
to 16 kHz) decode at 5.51 tokens per round with
the full head, 8.43 → 6.91 s against plain decoding, token-identical on
30 of 30 files.

**3. Log-mel features once per request.** Admission computed the features
for the target's tower and the draft's adoption computed them again. A
lane now keeps them from admission while the draft may still take it
(audio within the draft's tower) and drops them once the draft took it,
stopped following it or the request ended. The 21 adoptions' wall time
beyond their GPU time went from 36 to 17 ms in all (the features take
0.15-1.9 ms each, less than the 2.7 ms estimated); the run 6.26 → 6.20 s.

**4. Choosing k and adoption from measured costs.** What a token saves,
against which a lane's draft tower and prefill are weighed, was a fixed
1 ms for a 4-bit target, measured before the fused verify attention. The
engine now estimates it where lanes do not ride (4-bit): a plain token's
GPU time less a round's (a draft forward per proposal, the target's
verification of the lanes' rows, from the bench: 4-bit 4.85 ms for up to
8 rows, 6.3 to 16) over the tokens it emits, from each depth's acceptance
given the proposals before it held, counted over the lane rounds with a
decay (0.995 a round, a 0.9 prior worth 4 rounds). A lone lane's estimate
settles near 1.7 ms, so shorter files take the draft: 21 → 27 drafts,
6.23 → 6.08 s. The adoption cost was refitted to the draft's measured
tower and prefill (7 ms + 0.115 ms a prompt row; 16 ms at 86 rows, 56 at
427). With a draft that accepts too little the estimate falls and stops
taking lanes (`omni_asr_speculation_test`: one lane at a time, a draft
disagreeing on a quarter of the tokens takes fewer lanes than an exact
one). bf16 lanes ride other lanes' rounds at one token a round, which the
estimate does not price: it said 1.5 ms at eight lanes, took fewer lanes
and lost 5% (3.86 → 4.06 s at `--concurrency 8`), so bf16 keeps its fixed
7.5 ms. The load hysteresis and the 4-bit no-rider rule are unchanged.

k per round (engine wall s; tokens per round):

| k | 4 | 5 | 6 | 7 | 9 | 11 | 13 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4-bit, one file at a time | 6.27 (4.70) | 6.20-6.22 (5.60) | 6.18 (6.49) | 6.24 (7.12) | | | | |
| bf16, one file at a time | | | | 9.29 (6.88) | 9.08-9.12 (8.16) | 8.98-9.02 (9.38) | 8.88-9.05 (10.60) | 9.13-9.18 (11.15) |
| bf16, two at a time | | | | 6.10 | 6.06 | | 5.93 | |

4-bit stays at k = 5 (4 to 7 within 1%); bf16 now proposes 11 (1-2%
over 9). Choosing k each round from the same estimate did not help: it
settled on k = 4 (6.24-6.28 s against 6.13 for a fixed 5), since the
depths it stopped proposing kept their prior and it never learned that
deep proposals mostly hold (80% of 4-bit rounds accept all five).

Two 4-bit lanes still do not pay: speculating at two lanes (`--concurrency
2 --speculative-lanes 2`) ran 4.49-4.53 s at k = 3 (8 verify rows, one
tile lane) and 4.46-4.55 at k = 5, against 4.54-4.63 plain. A two-lane
round emits 7.5 tokens in 13.7 ms (1.8 ms a token against 2.5 plain), but
only 13 of 30 requests are long enough to repay the draft's 20-30 ms tower
and prefill, which is about what the rounds save. `--speculative-lanes`
stays 1 for 4-bit.

**Results over HTTP** (subset of 30, audio s/s; p50 latency at c=1 4-bit
181 ms):

| | c=1 | c=2 | c=4 | c=8 | long-form |
|---|---:|---:|---:|---:|---:|
| 4-bit plain | 49.7 | 78.6 | 114.7 | 147.9 | 136.6 |
| 4-bit `--draft-model`, round 7 | 56.0, 56.4 | 77.6, 78.3 | | | |
| 4-bit `--draft-model`, round 8 | **57.0, 57.1** | 78.1, 78.1 | 115.2 | 148.1 | 136.9 |
| bf16 plain | 22.4 | | | | |
| bf16 `--draft-model`, round 7 | 38.6, 39.2 | 52.2, 53.2 | | 85.8, 88.2 | |
| bf16 `--draft-model`, round 8 | **39.8, 40.1** | 53.2, 53.6 | 73.6 | 86.6, 87.0 | 85.8 |

(4-bit plain c=4, c=8 and long-form: the coordinator's clean main run.)
WER is unchanged (2.63% 4-bit, 2.74% bf16; long-form 1.51% / 1.24%). The
round gains 1.5-2% for one 4-bit request at a time and 1-2.5% for bf16 at
one and two; above that speculation behaves as before.

Levers left: the draft's forwards are 70% of a 4-bit round (five of 2.45
ms against a 4.9 ms verification) and latency-bound (a 0.6B step reads
about 250 MB, 0.8 ms at 300 GB/s); the page loop's ~7 µs a page round,
shared by the decode kernels, bounds both the decode and the verify
attention; and the draft's tower and prefill (20-30 ms a request) decide
whether two 4-bit lanes gain.

## Round 9: the sixth review's chunk schedule

Codex's sixth review ranked smaller TTS codec chunks first (eight 25-frame
Neural Engine evaluations at a stream's end against eight 8-frame ones).
Over HTTP, 4-bit, two alternating rounds: 8 streams give 44.3 / 44.3
audio s/s at the default 25 frames, 42.2 / 43.5 at 8 and 44.1 / 42.7 at 16;
one stream is unchanged (RTF 0.133). The default stays. The same review's
decode weight layout and batched ASR admission are being prototyped.


## Round 9a: decode weight layout

Codex's fifth item: the one-lane decode tile (LaneSplit128) reads three
planes per quant group, the codes of its 128 columns (4 KiB, 8 KiB from
the next group's, the other half of the 256-row storage tile between) and
their scales and biases (256 bytes each, in planes of their own). Decode
records put each 128-column tile's quant group in one 4.5 KiB record,
[N / 128][K / 64] {codes [128][32 bytes], scales and biases}, so a K
partition of a tile is one contiguous stream. `q4_mpp_tile_sums` now reads
its weights through a source (`Q4Planes`, the storage planes, or
`Q4Records`, kernels/common/q4_mpp_tiles.h) with the same matmul and
epilogue, and the `*_records` (scale/bias pairs) and `*_records_sb` (128
scales, then 128 biases) instances of the lane split tile, its residual,
gate/up (two record streams, or one alternating gate and up records:
`_paired`) and folded-norm (one and two rows) kernels give the planes'
bits (`q4_sgmatrix_metal_test`, against fp64 and the planes;
`omni-dq4-bench --records`).

Per dispatch, projection alone at the plan's split (`omni-dq4-bench
--records --rows 1`: weights cycled from DRAM, best of three interleaved
rounds, median of three runs; µs, M5 Pro):

| shape | planes | records | records sb |
|---|---:|---:|---:|
| talker / ASR qkv 4096 x 2048 (s2) | 23.3 | **22.6** | 22.9 |
| the same, norm folded | 23.1 | 22.6 | **22.5** |
| talker / ASR o 2048 x 2048 +res (s4) | 14.5 | 14.1 | **14.0** |
| talker / ASR gate/up 2 x 6144 x 2048 (s1) | 56.1 | 55.9 | **55.6** |
| the same, records paired | | 56.0 | 55.6 |
| the same, norm folded | 56.1 | 56.4 | **55.7** |
| talker / ASR down 2048 x 6144 +res (s4) | 33.7 | **32.8** | 33.2 |
| predictor qkv 4096 x 1024 (s2), norm folded | 13.9 | 13.8 | **13.7** |
| predictor o 1024 x 2048 +res (s4) | 10.6 | 10.5 | **10.4** |
| predictor gate/up 2 x 3072 x 1024 (s2), norm folded | 18.8 | 18.9 | **18.7** |
| predictor down 1024 x 3072 +res (s4) | 13.8 | 13.7 | **13.6** |

Eight rows give the same figures unfolded (talker qkv 23.4 / 22.7 / 23.0,
down 33.8 / 32.7 / 33.4, gate/up 56.1 / 56.2 / 55.5), and at every other
split the records track the planes within a microsecond, so the best split
does not move. The records save 0.4-1.0 µs on the talker's qkv, o and
down (2-3%), 0.1-0.5 µs on gate/up, the widest stream, and 0.1-0.2 µs on
the predictor: about 2.4 µs of a talker layer's 127 µs. The tile is bound
by how many groups it keeps in flight (64 threadgroups of two pipelined
groups for qkv), not by the number of streams those groups come from.

Real steps, with every projection the one-lane plans run on LaneSplit128
reading records (a prototype that kept a second copy of the weights:
793 MB for a 28-layer decoder, ASR's or the talker's, and 44 MB for the
predictor),
against the same build reading the planes and main's build, alternating,
six rounds, medians (ranges); ASR one file at a time over eight files
(`omni-asr-run --sequential`), TTS `--seed 1`:

| | main | planes | records |
|---|---:|---:|---:|
| ASR, GPU per step | 5.03 ms (5.01-5.05) | 5.02 (5.01-5.05) | 4.98 (4.97-5.00) |
| ASR, RTFx | 45.7 | 45.8 | 45.8 |
| TTS one lane, GPU per step | 9.49 ms | 9.53 | 9.46 |
| TTS one lane, RTF | 0.134 | 0.134 | 0.134 |
| TTS eight lanes, GPU per step | 10.07 ms | 10.13 | 10.06 |
| TTS eight lanes, audio s/s | 45.5 | 45.5 | 45.6 |

So the layout pays 0.04-0.07 ms of GPU per step (0.8%), as the per-shape
figures predict, and nothing measurable in RTFx or throughput. ASR tokens
and TTS speech (seed 1 and greedy, one and eight lanes) are identical
across the three. Adopting it would mean either the second copy of every
decode projection (about 0.8 GB per model) or records read by every other
tile of those projections: the prefill tiles, the N256 gate/up and
Split128 tiles of several lanes (verification), the Neural Engine FFN
split and Apple9's register tile. Neither is worth 0.8%, so the
planes stay the only layout. The prototype (the record kernels, the
`Q4Planes`/`Q4Records` sources, `omni-dq4-bench --records` and its
tests) was not merged; it is kept as the local tag prototype/decode-records (f9ad8de).

## Round 9b: batched ASR admission

Two items of a Codex review: batch the admission work still done request
by request (the audio towers, and the draft's adoption of lanes), and run
the fused attention of a few rows by query head.

**Where admission stood.** `admitBatch` already prefilled the requests
queued together as one ragged chunk, but encoded their towers one command
each. Engine, 4-bit, files submitted at once (`omni-asr-run`
`OMNI_ASR_DEBUG=1`; "alone" is each request admitted by itself, GPU ms):

| files (prompt rows) | towers alone | towers packed | prefills alone | ragged prefill | wall before → after |
|---|---:|---:|---:|---:|---:|
| 2 (149) | 18.3 | 13.0 | 30.6 | 26.1 | 166 → 161 ms |
| 4 (591) | 59.6 | 49.3 | 100.7 | 75.1 | 490 → 479 ms |
| 8 (1563) | 146.3 | 122.0 | 245.7 | 187 | 820 → 795 ms |

A tower costs about 4.5 ms plus 0.083 ms a token (7.6 ms at 42 tokens,
27 at 300, 68 at 793): the fixed part is about 180 latency-bound
dispatches of a few rows, which packing pays once. Packed, a tower runs at the long
file's rate (0.085 ms a token, about 11 TFLOPS: compute-bound).

**Packed towers.** `AudioEncoder::addEncode` takes several clips: each
runs its own convolution front end and positions into its rows of the
workspace, the norms and projections run once over all rows, and each
clip's dense attention is a dispatch over views starting at its first row,
so its key blocks of 32 start where they do in its own encode (a single
dispatch over the packed rows would block the keys by absolute row and
change the online softmax's grouping). Every clip's rows are the bits of
its own encode: the 64 x 128 and 32 x 64 GEMM tiles agree row by row
(`omni_audio_encoder_test` packs random and Qwen3-ASR towers, clips of 9
to 195 tokens, and compares bits); clips of four tokens or fewer (0.3 s)
run the GEMV and are encoded alone. `admitBatch` packs its requests'
towers, and a draft taking several lanes in one tick packs their towers
and prefills their prompts as one ragged chunk (two lanes of 163 rows:
23.9 ms against about 34 one at a time). Tokens are unchanged (4-bit
2/4/8 files, bf16 8 files, the subset).

**Over HTTP it barely shows.** Admission is material (towers and
prefills were about 40% of the GPU time of the c=2/4/8/long-form runs:
6.2 of 15.1 s), but requests arrive one at a time: of about 100
admissions in those runs 14 found a second request queued (8 pairs, 4
triples, one each of 4 and 7), and the rest have nothing to pack with.
Admission is not delayed to make batches. Long-form segments, which arrive
together, gain about 1% (within noise).

**Two speculating 4-bit lanes, recomputed.** With every lane drafted
(the adoption gate switched off for the measurement), engine, 30 files two
at a time: rounds of two lanes cost 1.79 ms a token at k = 3 and 1.78 at
k = 5 against 2.53 for plain two-row steps, so they save 0.74 ms a token;
the draft's tower and prefill cost 6.5 ms + 0.114 ms a prompt row (25.5
ms a request on the subset, 714 ms for 28 adoptions), and a lane emits
about 41 tokens after adoption (about 1160 in 308 lane rounds). That nets about
+5 ms a request: 4.35 s against 4.50 plain (k = 3; k = 5 4.45). The
shipped gate never drafts at two lanes (its estimate, 0.43 ms a token at
the 0.9 acceptance prior, never sees a two-lane round to learn from).
Over HTTP the same switch loses: c=2, 77.2-77.6 audio s/s (k = 3) and
76.7-77.2 (k = 5) against 80.1-81.2 plain and 78.7-81.4 with the default
one speculating lane; staggered arrivals put each adoption (and a lone
lane's 17 ms rounds) in front of the other lane, and p50 latency rises
from 220 to 255 ms. Packing adoptions does not change this: at c=2 the
lanes are taken one at a time except at the start. `--speculative-lanes`
stays 1 for 4-bit and the gate is unchanged.

**Fused attention by query head.** The fused decode attention ran a
threadgroup per (KV head, sequence) whose simdgroups each scored the KV
head's two query vectors per page; at one lane that is eight threadgroups
on an idle GPU. A decode step of one row per sequence, or a verify
forward, of at most four rows now runs a threadgroup per query head
(`omni_attention_decode_fused_head`, `omni_attention_verify_fused_head`),
one vector per page loop; both threadgroups of a KV head store its key and
values (the same bits into the same slots). A vector's page loop and merge
are unchanged, so the outputs are bit-identical (`omni_kernels_test`
compares them with the KV-head kernels exactly; ASR tokens, greedy TTS at
1/2/4 lanes on both checkpoints and seeded TTS are unchanged). Every
layer's attention alone, history 200 (`omni-asr-verify-bench`, GPU ms over
28 layers):

| rows | decode, 1 lane | decode, 2 lanes | decode, 4 lanes | verify 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| per KV head | 0.30 | 0.34 | 0.47 | 0.30 | 0.30 | 0.31 | 0.31 | 0.33 | 0.34 |
| per query head | **0.17** | **0.26** | 0.47 | **0.17** | **0.19** | **0.23** | **0.28** | 0.38 | 0.47 |

(history 400: one lane 0.53 → 0.30, two 0.61 → 0.43.) Beyond four rows
the KV-head kernels stay. Whole forwards (three alternated rounds): a
4-bit 1.7B decode step 4.82 → 4.72 ms, a 0.6B draft step 2.42 → 2.31, a
1/2/3-row verify 4.81 → 4.71, 4.87 → 4.78, 4.89 → 4.82 (a 6-row one-lane
verification is unchanged). Engine, 30 files, 4-bit speculative one at a
time: rounds 17.1-17.4 → 16.8-17.0 ms, 6.03-6.11 → 5.92-5.93 s; TTS
within noise (the predictor keeps the one-page kernel).
`OMNI_QUERY_HEAD_ATTENTION=0` keeps a threadgroup per KV head.

**Results over HTTP** (subset of 30, audio s/s, main at e2e42df and this
round alternated, two rounds each; WER unchanged: 2.63% 4-bit, 2.74%
bf16, long-form 1.51%):

| | c=1 | c=2 | c=4 | c=8 | long-form |
|---|---:|---:|---:|---:|---:|
| 4-bit plain, main | 49.9, 50.3 | 79.8, 79.0 | 116.7, 115.2 | 148.3, 149.2 | 133.2, 137.6 |
| 4-bit plain, round 9b | **51.2, 51.5** | **81.2, 81.5** | **117.5, 117.9** | 149.4, 151.7 | 136.4, 138.5 |
| 4-bit `--draft-model`, main | 57.3, 57.7 | 76.7, 79.0 | | | |
| 4-bit `--draft-model`, round 9b | **59.1, 58.7** | 80.5, 79.8 | | | |
| bf16 `--draft-model`, main | 40.1, 40.2 | | | | |
| bf16 `--draft-model`, round 9b | 41.3, 40.2 | | | | |

Plain 4-bit gains 2.5-3% at one and two requests (the query-head
attention; at two lanes a step's attention drops 0.08 ms) and 1-2% at
four and eight, well short of the 3-10% the review expected from batched
admission: the batches it would pack rarely form over HTTP.

What did not help, or was not done:

- **A joint adoption gate** (every active lane's likely tokens against
  the drafts taken, with a two-lane k of 3 to keep 8 verify rows): it
  still took no lanes at c=2 on the subset (its saving estimate stays at
  the prior), and drafting every lane loses over HTTP anyway.
- **The tower and prefill of a single admission in one command**: about
  0.3-1 ms of submission a request (0.5% at c=2), and the prefill's
  Neural Engine split retries its chunk on the GPU alone, which a shared
  command would complicate.

Levers left: admission is 40% of the GPU at c=2-8 and mostly single;
running the active lanes' decode rows inside an admission's prefill chunk
(a 166-row prefill takes about 28 ms, a separate two-lane step 5) would
save most of a step per admission, about 5% at c=8, but those rows would
take the prefill tiles' arithmetic and could flip near-tie tokens.

## Round 10: host time per request

A Codex review asked where a 4-bit transcription's time goes over HTTP
at one request at a time (51.4 audio s/s, about 220 ms a request), and
whether the host time between the engine's GPU commands could go.

**Tracing.** `SPLASH_AUDIO_TRACE=1` makes the server print each
transcription's instants (arrival, upload read, decoded, frame written,
first token and Done read, terminal delivered, tokens collected, answer
sent); `OMNI_ASR_TRACE=1` makes the engine keep each request's events
(frame read, admission, mel, and per command its graph build, submission,
commit, GPU start and end, and the wake after it) and write them when the
request ends. `CommandTiming` now carries the GPU's start and end as host
instants. All of them are mach absolute time, as is the server's
`time.monotonic()`, so `dev/tools/omni_request_trace.py` joins the two
sides per request. A request at one at a time, 200 LibriSpeech utterances
(mean milliseconds; the engine with steps one at a time):

| phase | ms |
|---|---:|
| upload read, WAV decode and split, frame written | 0.76 |
| engine reads the frame and admits | 0.18 |
| mel (CPU) | 0.70 |
| GPU: tower 15.5, prefill 26.4, 36.4 steps of 4.77 | 216.1 |
| host between commands: graph build 2.15, encode and commit 3.27, commit to GPU start 4.67, GPU end to wake 3.42, emit and tick 0.37 | 13.9 |
| Done written to answer sent (Python reads Done, delivers, detokenizes, answers) | 0.31 |
| total, arrival to answer | 232.0 |

The GPU idles 0.314 ms between two decode steps (median 0.310), 1.56 ms
between the tower and the prefill (median 0.43; the Neural Engine split's
setup), 0.34 between the prefill and the first step, and 3.4 ms between
one request's last step and the next request's tower, which is the
client's turnaround and outside the request. Python, the pipe and the
features take about 2 ms of the 232; the server's per-token events
are handled off the critical path (the engine does not wait for them), and
Done to the answer is 0.3 ms, so coalescing token events or trimming the
HTTP side cannot pay. What is left is the decode steps' host time: about
0.3 ms a token.

**Decode steps queued behind the one in flight.** `step()` submitted a
step, waited, read its argmax and emitted before encoding the next. Now,
while the same lanes keep decoding and no request waits for admission, it
submits step k+1 before waiting for step k: step k+1 embeds its tokens
from the buffer step k's argmax writes (the speculation path's GPU
feedback; two buffers alternate), at positions one on, with its KV row
reserved. `MetalBackend::submitCommandBehind` accepts one command queued
behind the one in flight (at most two; tickets consumed in order; the
watchdog follows the queued one once the first ticket is consumed;
`submitCommandAsync` still refuses a second command). A row of a lane that
ended meanwhile is dropped; the engine completes the queued step before KV
pages change hands (admission, after the next request's mel; the
speculative branch; an idle tick). `OMNI_ASR_PIPELINE=0` keeps one command
in flight.

A step queued behind the one that emits the stop token runs to no use: at
one request at a time the whole step (4.8 ms), and the next request's
admission waited for it (its mel phase grew from 0.7 to 2.8 ms: about 2.1
ms of the 4.8 exposed past the client's turnaround). So a step is not
queued when every lane's last token was followed by a stop token more than
a fifth of the time so far (counted per token as the engine emits; on
English LibriSpeech every transcript ends with a period, which precedes the
stop token about 0.7 of the time); skipping the queue costs one gap. Wasted
steps went from one per request to 1 in 30 files (engine) and about 4 in
200 (HTTP).

Traced again (same run conditions), the GPU idles 0.005 ms between steps,
the host time between commands falls from 13.9 to 3.0 ms a request and a
request from 232.0 to 220.3 ms. The engine alone, the 30-file subset one
at a time (`omni-asr-run --sequential`, wall s, alternating): main 7.25,
7.10; steps one at a time 7.07, 7.22; queued 6.76, 6.84 (and 7.11, 7.14
against 6.82, 6.85 in a second set): 4% faster. Lanes in lock step: up to 4
at a time 3.08 → 3.06 s, up to 8 2.38 → 2.35 s. Tokens are identical on
30 of 30 files one at a time and at up to 8 at a time; at up to 4 one file
lost a comma, from which requests were admitted together (a ragged prefill
of 160 rows or more takes the Neural Engine split): with
`OMNI_DISABLE_ANE=1` both sides are identical. Every HTTP transcript below
is identical between main and the branch.

**Over HTTP it does not show by default.** Main (ad841b6, whose runtime
51da065 keeps, built in a scratch directory) and the branch alternated,
three rounds, audio s/s,
c=1 on 200 utterances, the rest on the 30-file subset:

| | c=1 (200) | c=4 | c=8 | long-form | `--draft-model` c=1 (200) |
|---|---:|---:|---:|---:|---:|
| main | 50.4, 51.7, 51.9 | 119.9, 120.9, 121.4 | 152.3, 154.9, 155.0 | 141.3, 145.3, 144.8 | 56.8, 55.7, 57.5 |
| round 10 | 50.3, 50.7, 50.7 | 121.8, 120.8, 120.8 | 155.1, 154.8, 152.4 | 144.1, 145.3, 143.8 | 57.9, 57.7, 57.8 |

Six more c=1 pairs: main 50.8-51.7 (mean 51.31), the branch 50.4-51.8
(51.24); the sum of each file's median latency differs by -0.1%. WER is
unchanged (2.17% on the 200, 2.63% on the 30, long-form 1.51%).

The reason is that the host time between steps is not the same in every
server process. Fitting each run's latencies to tokens (counted by the
tokenizer over the transcripts) and audio seconds gives the cost of a
decode token: 4.82-4.93 ms for main started as the benchmark starts it,
5.12-5.21 ms for the same main binary started with one more environment
variable (`OMNI_ASR_PIPELINE=0`, which main does not read: 49.3 and 49.5
audio s/s, four runs), for the branch with steps one at a time, and under
the tracing variables; the step's GPU time is 4.75-4.77 in both. Another
variable (`SPLASH_BENCH_TAG=r10`) left main fast (51.2-51.6). Each
launch's state held over every run of it, so the engine's host path
between steps costs about 0.1 ms in some processes and about 0.4 in
others, for reasons not found (thread placement or code and stack layout
are candidates; the traces, taken in the slow state, show the 0.3 ms split
across the launch, the wake, the encoding and the build). Queued steps cost
4.81-4.94 ms a token in every launch tried (51.2-51.8 audio s/s, 14 runs,
three environments), so they gain where the process is slow (49.3 → 51.5,
4%) and nothing where it is already fast: a step queued behind another
also runs about 0.07 ms longer on the GPU than one started from idle
(4.82 against 4.75 in one traced pair), which is about what the fast
state's gap was. The speculative path's plain steps (lanes the draft does
not take) gain about 2% (57.8 against 56.7 mean).

What did not help, or was not done:

- **Chaining several steps in one command** (the review's form): the
  host still builds and encodes every forward of the chain while the GPU
  waits, so it saves only the launch and the wake (about 0.23 ms of 0.31)
  per chained step, and every step past the stop token is wasted (1.5
  forwards at four).
- **Coalescing per-token events, connection reuse, fewer copies**: the
  whole Python and pipe path is about 2 ms of 232 and off the decode loop.
- **Overlapping a request's mel with the GPU at c=1**: in a closed loop the
  next request arrives after the previous answer, when the GPU has nothing
  left to run (0.7 ms; it overlaps a wasted queued step at most).
- **Queuing the prefill behind the tower and the first step behind the
  prefill**: 0.4 and 0.3 ms a request (0.3%), and the prefill's Neural
  Engine split commits with event steps, which a queued command does not
  take; not done.

## Round 11: long contexts

A Codex review (the eighth) looked at long single files, where one 4-bit
request at a time is only level with sglang-omni: a 35 s file spends 80% of
its time in 94 decode steps of 4.79-4.97 ms. Past 16 pages (512 tokens) a
decode step left the fused attention for prepare, split and reduce, three
dispatches a layer instead of one (144 → 200 a step). The review asked for
an efficient one-row attention past 16 pages, chosen per history range by
measurement, and for the head fused with a partial argmax.

**What runs now.** A lone decode row past 16 pages runs
`omni_attention_decode_long`: the split kernel's threadgroups, by query head
(16 per split), each preparing its query vector as the fused kernel does,
the split holding the row's own token also storing its key and values; each
split stores its partial and the last split of a head to finish reduces
them in split order as the reduce kernel does (`split_reduce.h`). Up to 24
pages it takes the split kernel's splits (eight pages each, one page per
simdgroup) and gives their bits, queries and keys included
(`omni_kernels_test` compares it with prepare, split and reduce exactly over
three and two splits, and checks the counters return to zero); past 24 pages
a lone row takes two splits of longer page loops. Steps of several rows run
the fused kernels (by query head up to four rows, by KV head beyond) at any
history. `OMNI_LONG_ATTENTION=0` restores the old paths.

**Per history.** `omni-asr-verify-bench --decode-histories` times decode
steps alone, eight forwards and their argmax chained in one command so no
step starts from an idle GPU (GPU ms a forward; the old paths with
`OMNI_LONG_ATTENTION=0` and the new ones in alternating processes, three
each, the median of each process's three medians of 15 commands):

| tokens | 1 lane, old → new | 2 lanes | 4 lanes | 8 lanes |
|---|---|---|---|---|
| 470 | 4.886 → 4.893 | 5.095 → 5.095 | 5.486 → 5.508 | 6.365 → 6.376 |
| 500 | 4.894 → 4.910 | | | |
| 511 | 4.899 → 4.907 | | | |
| 512 | 5.062 → **4.941** | | | |
| 530 | 5.061 → **4.945** | 5.256 → **5.192** | 5.980 → **5.689** | 7.001 → **6.837** |
| 560 | 5.065 → **4.958** | | | |
| 700 | 5.097 → **5.025** | 5.399 → **5.288** | 6.248 → **5.870** | 7.473 → **7.136** |
| 850 | 5.145 → **5.083** | | | |
| 1000 | 5.203 → **5.151** | 5.724 → **5.550** | 6.776 → **6.426** | 8.487 → **8.151** |
| 1500 | 5.431 → **5.345** | | | |
| 2000 | 5.716 → **5.565** | 6.754 → **6.384** | 8.461 → **8.044** | 11.779 → **11.356** |
| 3000 | 6.313 → **5.996** | | | |
| 4000 | 6.791 → **6.416** | | | |

Up to 511 tokens the kernels are the same (the differences are noise). The
step past 512 tokens now costs 0.05 ms more than at 470, not 0.18.

**How the paths were chosen.** At one lane, 512 / 560 / 700 / 1000 tokens,
the candidates in one process, alternating (same benchmark):

| path | 512 | 560 | 700 | 1000 |
|---|---:|---:|---:|---:|
| prepare, split by KV head, reduce (the old path) | 5.056 | 5.080 | 5.099 | 5.204 |
| the same with the split by query head (bit-identical) | 4.980 | 5.009 | 5.061 | 5.183 |
| fused by query head, allowed to 24 / 32 pages | 4.963 / 4.960 | 4.978 / 4.978 | 5.012 / 5.010 | (old) / 5.141 |
| long kernel by KV head | 5.009 | 5.029 | 5.049 | 5.161 |
| long kernel by query head | **4.919** | **4.948** | 5.023 | 5.157 |

and the long kernel's splits (one lane): three splits of one page per
simdgroup at 17-18 pages (4.916) beat two (4.962) and one (5.010); past 24
pages two splits of longer loops beat one round of many splits (850 tokens
5.048 against 5.075 in four; 1000 5.108 against 5.141; 1500 5.31 against
5.54 in six; 2500 5.74 against 5.96; 4000 6.39 against 6.82 in sixteen;
6000 7.21 against 7.82; 8000 8.05 against 9.08), and against the fused
kernel over every page (1500 5.36, 4000 6.52, 6000 7.41, 8000 8.31). At
several lanes the fused kernel wins at every history to 4000 tokens (two
lanes at 4000: fused 8.03, long kernel in two splits 8.06, old 8.52; eight
lanes 17.65, 18.44, 18.35), including steps where one lane is long and the others short
(least of four alternating processes, old / fused / long kernel for every
lane):

| lanes at tokens | old | fused | long kernel |
|---|---:|---:|---:|
| 1 at 530, 1 at 200 | 5.153 | **5.085** | 5.197 |
| 1 at 530, 3 at 150 | 5.678 | **5.422** | 5.619 |
| 2 at 560, 6 at 300 | 6.700 | **6.458** | 6.667 |
| 1 at 530, 7 at 200 | 6.376 | **6.326** | 6.423 |

**Arithmetic.** The fused kernel groups a row's pages differently from the
splits (simdgroup g takes pages g, g + 8, ...), so beyond 16 pages steps of
several rows change the softmax grouping; a lone row up to 24 pages keeps
the old bits and past 24 pages its two splits differ from the old split
count. Tokens (`omni-asr-run`, main and the branch alternating):

- 200 utterances one at a time: identical, three runs each (only 103 of
  7940 steps a run are past 511 tokens). The 58 files over 15 s, four
  runs each: identical; the 35 s file's decode 461-463 → 451-454 ms, the
  three files that pass 512 tokens 1.479-1.487 → 1.461-1.469 s, the 58
  files 23.39-23.68 → 23.34-23.67 s of wall.
- Four and eight at a time: one file each differs from main, a comma at
  four, and "especial" → "a special" at eight, which is the reference's
  reading. With the branch, both runs give the one-at-a-time tokens on all
  200 files; main's did not (its four- and eight-way runs differ from its
  own one-at-a-time run on those two files). Wall unchanged (19.82, 19.90
  → 19.82, 19.81 s at four; 15.11, 15.11 → 15.14, 15.11 at eight): few
  steps there have a lane past 16 pages.
- The long-form files whole, one at a time (no server segmenting; 747 s of
  audio, contexts to about 4600 tokens): identical tokens, wall 17.84,
  17.82 → 17.12, 17.12 s, steps 6.60 → 6.28 ms on average.
- Greedy TTS (`omni-tts-run --greedy`): the usual text gives main's wav at
  one and four streams. Four copies of it (768 frames, past 24 pages) give
  main's wav at one stream, but not at four: there the fused kernel's
  grouping changes the speech, and the branch's speech past 512 tokens
  now depends on how many streams decode together, which main's did not.

Real-weight parity passes on both models; `make test-engine`,
`make test-python` and `make architecture-check` pass.

**Over HTTP it does not show.** c=1 on the 200 utterances and the
long-form set, main and the branch alternating, eight rounds each; rounds
fell into the two launch states round 10 found (about 51 and about 45 audio
s/s, on both sides, whole runs at a time), so the table keeps the fast-state
runs (audio s/s; a bin's figure is its audio over its summed latency):

| | all | < 8 s | 8-15 s | > 15 s | long-form |
|---|---|---|---|---|---|
| main | 51.79, 51.03, 51.50, 51.20 | 43.4-43.8 | 49.7-50.3 | 55.75, 55.33, 55.41, 55.13 | 145.1, 144.3, 146.3, 144.2 |
| round 11 | 51.19, 51.02, 50.99, 51.47, 51.52 | 43.4-43.7 | 49.8-50.1 | 55.70, 55.29, 55.10, 55.45, 55.43 | 144.7, 145.3, 138.6, 146.3, 145.9 |

Transcripts are identical in every run, WER 2.17% and 1.45% on both sides.
At c=1 only the files over about 33 s pass 512 tokens, for their last
dozens of steps: about 0.1-0.2 ms a step, 9 ms of the 35 s file's 565, so
the > 15 s bin moves by about 0.2%, under the noise. Long-form audio is
cut into segments of at most 30 s (about 390 prompt tokens and 100
generated), so its lanes stay within 512 tokens too. The gain is where
contexts are long: whole long files on one lane (4%), several long lanes
(4-6% a step at 530-2000 tokens on four or eight lanes) and long TTS.

**The head with a partial argmax** (the review's second item) was not
built: the whole argmax is 0.006-0.014 ms of a step (chained steps with and
without it, 1 and 8 lanes, four alternating processes), so fusing it into
the head can gain at most 0.3%. An argmax over 16, 32 or 64 threadgroups a
row with a last-arriver reduce (the same token: the maximum with the lowest
index does not depend on grouping) measured the same as the one-threadgroup
kernel, within 0.01 ms; it was not kept.

What did not pay, or was not done:

- **Query-head parallelism inside the split path** (prepare, a split by
  query head, reduce; bit-identical): 4.98 against 5.06 ms at 512 tokens,
  but the long kernel's one dispatch is 4.92.
- **The fused kernel past 16 pages at one lane**: 4.96 at 512 tokens and
  6.52 at 4000, behind the long kernel (4.92, 6.39), and it changes the
  arithmetic of single long files.
- **The long kernel by KV head, or at several lanes**: 5.01 at 512 tokens
  at one lane; at 530 tokens the query-head long kernel takes 5.33, 5.90
  and 7.02 ms at two, four and eight lanes against the fused 5.18, 5.67
  and 6.82 (the old path 5.26, 6.00, 7.02).
- **The long kernel below 17 pages**: 4.93 against the fused 4.90 at 400
  tokens; the fused kernel stays.
- **Split argmax**: above.

## Round 12: MLX-style one-row QMV

Over HTTP, one 4-bit ASR request at a time, a fit of each file's latency
to its audio seconds and output tokens gave 5.11 ms a token against 4.43
(sglang-omni) and 4.47 (mlx-audio), both on MLX: our tower and prefill were
faster, so MLX won token-heavy long files. MLX's batch-1 affine Q4
`quantized_matmul` (group 64, weights cycled from DRAM, one call at a time
of a chain evaluated together) took 21.0 µs for the QKV projection, 13.0
for O, 27.4 for gate (a pair ~54.8), 27.9 for down and 616 for the head,
where one lane's LaneSplit128 plans took 22.9, 14.5, 56.0 (gate/up in one
dispatch with its folded norm), 33.6 and 638 (the paired N256 tile after
the norm). Codex's ninth review traced the difference to geometry:
`qmv_fast` runs 64-thread threadgroups of two simdgroups, four outputs a
simdgroup, all 32 lanes on every output (16 inputs and 8 code bytes a lane
per 512-input step), one `simd_sum` an output and nothing else, where
LaneSplit128 computes eight padded rows with MPP operations per quant
group, barriers every 256 inputs and a last-arriver split reduction.

**What runs now.** `LinearTile::Qmv` (`kernels/decode/linear_q4_qmv.metal`,
following `qmv_fast`; MIT notice in `THIRD_PARTY_NOTICES`) projects a
lane's single live row: each simdgroup owns a run of adjacent outputs, lane
l takes inputs 16 l .. 16 l + 15 of every 512-input step, masks its four
16-bit code words in place against activations prescaled by 1, 1/16, 1/256
and 1/4096, and adds `scale * dot + bias * sum(x)` in fp32 per chunk; one
`simd_sum` per output ends it. It reads the existing affine planes (output
n's quant group g at parameter ((n / 256) * (K / 64) + g) * 256 + n % 256,
codes at 32 bytes each, lane l the 8 bytes at 8 (l % 4)): no repack and no
second copy of the weights. The fusions stay: the folded RMS norm (the
threadgroup's 256 threads reduce the row as `rms_inverse` does, so the
folded projection equals the norm dispatch and the projection bit for bit,
then each lane normalizes its chunk in registers to the norm's bf16), the
residual after the bf16 projection, gate/up in one dispatch with bf16 gate
and up before the SiLU product, unrounded fp32 logits.

`LinearWorkload::liveRows` carries a step's live rows (the decoder and the
TTS model now pass them); Apple10 runs a lane of one live row on the Qmv
tile, eight simdgroups a threadgroup, four outputs a simdgroup and two for
gate/up, folding the norm up to 16384 outputs (every layer and predictor
head; the ASR head's 152064 run the norm dispatch). Two or more live rows
keep the tiles. A one-row TTS predictor sub-step takes the first-layer QKV
tables too, though they hold the eight-row tile's arithmetic (one lane's
step 7.33 → 7.18 ms).

**Per projection** (`omni-dq4-bench --qmv`: one live row, weights cycled
from DRAM through at least 256 MB of copies, chains of 16 dispatches in
one command, best of three interleaved rounds; MLX's figures from
`mlx_qmv.py`, whose calls are not serialized by dependences):

| projection | LaneSplit128 / paired N256 | Qmv | MLX |
|---|---:|---:|---:|
| ASR/talker QKV 4096x2048, folded norm | 22.9 | **20.4** (18.4 without the norm) | 21.0 (no norm) |
| O 2048x2048 + residual | 14.5 | **10.2** | 13.0 |
| gate/up 2x6144x2048, folded norm, SiLU | 56.0 | **53.6** (50.4 without the norm) | 54.8 (the two calls) |
| down 2048x6144 + residual | 33.6 | **26.5** | 27.9 |
| ASR head 152064x2048 fp32, after the norm | 638 | **586** | 616 |
| talker head 3072x2048 fp32, folded | 19.0 | **15.5** | |
| predictor QKV 4096x1024, folded | 13.8 | **10.6** | |
| predictor O 1024x2048 + residual | 10.5 | **5.8** | |
| predictor gate/up 2x3072x1024, folded | 18.6 | **14.7** | |
| predictor down 1024x3072 + residual | 13.7 | **7.8** | |
| predictor head 2048x1024 fp32, folded | 10.3 | **6.4** | |

Every shape beats LaneSplit128 (4-45%) and, where measured, MLX's call,
norm folded in. The ASR head streams at 299 GB/s of the ~307 peak.

**Whole steps.** `omni-asr-run` on the 35 s file (`OMNI_ASR_DEBUG=1`,
main and the branch alternating, three each): the one-row step 4.83, 4.82,
4.83 → **4.14, 4.13, 4.13 ms** median (decode 450.5 → 384.5 ms), the same
94 tokens. The 200 utterances one at a time (`--sequential`): 47.41, 47.76
→ 41.69, 41.55 s of wall; with the 0.6B draft (`--draft`) 41.11, 41.16 →
36.56, 36.58 s. `omni-tts-run --seed 1`: one lane 9.37 → 7.14-7.22 ms GPU a
step (RTF 0.132 → 0.103); two and eight lanes, whose steps hold two rows or
more, unchanged (9.56 → 9.60 and 10.34 → 10.35 ms, within noise).

**Choices within the tile**, whole ASR steps on the long file (three
alternating processes each): two outputs a simdgroup for gate/up 4.10-4.11
against four 4.13-4.15; four for the residual projections 4.10 against two
4.15-4.17 (although two measured 0.4 µs faster on O alone); four for QKV
and the head 4.10-4.11 against two 4.15-4.17; folded norms 4.10-4.12
against the norm dispatches 4.09-4.12 (a tie; folding keeps 56 fewer
dispatches to encode a step).

**Arithmetic.** The sums associate differently from the tiles' (per lane
in K order, then the lanes), before the same single bf16 rounding.
`q4-sgmatrix` checks every epilogue against fp64 at four fixtures
(bf16 inputs past half's range, cancellation, zero weights), K of whole and
partial 512-input steps, fp32 logits, that two and four outputs a
simdgroup give the same bits, that nothing past the live row is read or
written, the folded norm against the norm dispatch bit for bit, and the
policy and its refusals. Real-weight parity passes on both models (ASR
decode logits cosine 0.999917 → 0.999922; TTS unchanged to 1e-5). Tokens:
194 of the 200 utterances identical one at a time (six differ by a comma
or "Mr."/"Mister"), the long file identical, the speculative path's 200
identical (its verify steps keep the tiles); WER 2.17% on both sides.
Greedy TTS is byte-identical at two and eight lanes; at one lane it
diverges at frame 6 (one code of sub-step 14), after which main runs to
the 192-frame cap and the branch stops at 151.

**Over HTTP** (`run_benchmark.py`, main 2f7b60e built in a scratch
directory, each server started from its own tree, alternating; audio s/s;
a bin's figure is its audio over its summed latency; the fit is each run's
latency against audio seconds and tokens counted by the tokenizer):

| ASR 4-bit | c=1 (200) | < 8 s | 8-15 s | > 15 s | ms a token | c=8 (200) | long-form |
|---|---|---|---|---|---|---|---|
| main | 51.4, 51.6, 51.4 | 43.4-43.7 | 50.0-50.1 | 55.3-55.6 | 4.83-4.87 | 156.5, 157.7, 157.5 | 146.0, 146.0, 147.0 |
| round 12 | **57.2, 58.0, 58.4** | **48.1-49.1** | **55.8-57.0** | **61.6-63.0** | **4.07-4.18** | 152.9, 156.3, 157.4 | 139.5, 147.2, 147.4 |
| sglang-omni (earlier run) | 46.1 | 33.1 | 45.7 | 52.5 | 4.43 | 66 | 49.6 |
| mlx-audio (earlier run) | 42.7 | 29.5 | 42.1 | 49.7 | 4.47 | | |

One request at a time is 12-13% faster in every duration bin and the
token cost is now below both MLX engines', so the lead holds on long files
too. c=8 and long-form, whose steps carry several rows (the long-form
server cuts a file into segments decoded together), are unchanged
(139.5 is one slow launch). Five of 200 transcripts differ at c=1, as
above; WER 2.17% / 1.45% on both sides.

| TTS 4-bit | c=1 audio s/s | RTF median | first audio p50 | c=8 audio s/s |
|---|---|---|---|---|
| main | 7.79, 7.05, 7.77, 7.77 | 0.129-0.143 | 21.3-21.8 ms | 37.3, 43.4, 44.6, 43.7 |
| round 12 | 7.93, **9.91, 9.93, 9.92** | 0.102 (0.127 in the first) | 19.7-19.8 ms (24.8) | 37.9, 44.1, 44.4, 43.5 |

The first round of each side ran slow (both c=8 at 37); in the other three
one stream is 28% faster. English WER 0.27-0.63% and Chinese CER
0.30-0.48% against main's 0.09-0.72% and 0.24-0.60%: in range.

What did not pay:

- **MLX's own threadgroup of two simdgroups with the norm folded**: every
  simdgroup then reduces the row itself (bit-exact by holding each of the
  eight `rms_inverse` simdgroups' chains in its lanes): QKV 28.1 µs,
  gate/up 67.3, against 20.4 and 53.6 with eight simdgroups sharing one
  reduction. Without the norm the width barely matters (QKV 18.7 at two,
  18.5 at four, 18.4 at eight simdgroups).
- **Eight inputs a lane** (eight lanes a quant group, 256-input steps):
  within 0.3 µs or up to 1.7 µs slower (down), except gate/up, where it
  ties two outputs a simdgroup; dropped to keep one arithmetic.
- **32-bit code words** (eight masks, activations prescaled to 2^-28):
  the same as 16-bit words (QKV 18.5 against 18.5, down 26.7 against 26.6).
- **Eight outputs a simdgroup**: gate/up 61-84 µs (spills).
- **Loading a step's codes ahead**: the first step's codes before the norm
  reduction, or each next step's before the current arithmetic: QKV folded
  21.8-22.5 against 20.4, down 27.6 against 27.0. Two steps a loop
  iteration: within 0.3 µs either way over repeated runs.
- **Two to eight live rows on QMV** (codes loaded once a step and reused by
  every row, each row's arithmetic the one-row kernel's): at two rows it
  wins the residual projections (talker O 14.6 → 13.7, predictor O 10.5 →
  8.5, predictor down 13.7 → 11.3) but loses QKV (23.8 → 28.3 folded) and
  gate/up (56.8 → 74.9); at four and eight rows it loses 2-5x, its scalar
  fp32 dot products growing with the rows while the tiles' matrix
  operations do not. Not used; the residual projections at two rows are
  the one place it would pay (about 1-2 µs each).
- **A row-major copy of the weights**: not needed; reading the tile
  storage already beats MLX's row-major reads.
- **A one-row path for the TTS QKV tables built with QMV**: would match one
  row only and lose the tables for two to eight rows; the one-row step takes
  the eight-row tables instead (above).

## Round 13: lossless compressed bf16 weights

Plain bf16 decoding is bound by memory bandwidth in every engine: a latency
fit over the 200 files gave 12.46 ms a token for Splash Omni, 12.72 for
sglang-omni and 12.80 for mlx-audio, each token streaming about 3.4 GB of
bf16 weights (the tied 151936x2048 head included), and one bf16 file at a
time led only 22.3 against 20.8 audio s/s. Codex's tenth review proposed
reading fewer bytes without changing a bit: in blocks of 64 weights along K
the exponents of almost every block fit a 16-value window, so each weight's
exponent can be a 4-bit code against a per-block base.

**The format** (`metal/abi/Omni.h`, packed by `ops::omni::packDenseRows`).
One buffer per weight holds four planes: a byte of sign and mantissa per
weight; 4-bit exponent codes, 32 bytes a block; a 32-bit header per block
(the base exponent, or 255 for an escape block with its record index in
bits 31:8); and 64-byte escape records holding a block's exponents as
bytes. A block is an escape when its exponents span more than 16 values or
one is 255 (NaN and infinity keep their payloads); the base is the least
exponent, zeros included, so zeros and subnormals beside small exponents
need no escape. That is 100 bytes a block against 128, plus 64 per escape:
recounted with these rules on both checkpoints, 0.30-0.34% of the decoder
blocks and 0.30-0.43% of the heads' are escapes, and the copies come to
0.7827-0.7834 of the bf16 bytes. The review's zero mode (a flag making
code 0 mean exponent 0) is not there: no decoder weight of either checkpoint
has a zero exponent (only 0.23% of the blocks of the TTS embedding tables
and heads hold one beside larger exponents), so such blocks are escapes.

Each lane's group of eight weights is stored in the order 0 2 1 3 4 6 5 7,
and its 32-bit code word holds weights 0 2 4 6 in the low half's nibbles
and 1 3 5 7 in the high half's (escape bytes in the sign/mantissa order):
then one mask of a sign/mantissa word yields two weights' sign and
mantissa in the two 16-bit halves, and one rotation and mask of the code
word (or of an escape word) their exponents at bits 14:7, so each pair of
bf16 weights is rebuilt in one 32-bit word with about ten integer
operations and unpacked as the bf16 kernel unpacks its words. The code
word sits at a fixed place (no dependent load); only an escape block's
lanes load its record through the header. The GEMV (`omni_gemv` in
`gemm_bf16.metal`) takes the weights through a column type, `OmniDenseColumn`
or `OmniPackedColumn`, and is otherwise unchanged: the same lane ownership
(`k = lane * 8; k += 256`), the same eight-product partial sums, the same
`simd_sum` and epilogues, so the packed kernels write the bf16 kernels'
output bits.

**Per projection** (`omni-dense-bench --real` on the Qwen3-ASR-1.7B-bf16
weights, successive layers as copies past 512 MB, the bf16 policy against
the packed kernel interleaved over five rounds, median µs; every packed
weight checked to unpack to its bits and every packed output to equal the
bf16 kernel's bytes):

| rows | QKV 4096x2048 | O 2048x2048 +res | gate/up 2x6144x2048 | down 2048x6144 +res | head 151936x2048 fp32 |
|---|---:|---:|---:|---:|---:|
| 1 | 59.0 → **48.3** | 30.9 → **26.9** | 170.4 → **140.0** | 89.6 → **74.0** | 2087 → **1629** |
| 2 | 63.1 → **56.1** | 34.7 → **30.6** | 175.7 → **144.6** | 100.0 → **82.8** | 2117 → **1841** |
| 3 | 70.0 → 77.2 | 40.9 → 41.6 | 181.2 → **158.2** | 109.4 → 113.0 | 2210 → 2688 |
| 4 | 81.8 → 92.7 | 46.6 → 49.6 | 187.4 → **179.4** | 130.8 → 135.6 | 2680 → 3262 |

At one row the packed kernel streams 244-299 GB/s of packed bytes (the
head 299 of the ~307 peak), 310-382 GB/s in bf16 terms. From three rows the
rows' sums plus the unpacking outweigh the bytes saved, except for gate/up,
whose two weights share each input load. The TTS shapes agree (the talker's
are the ASR decoder's; one row: predictor QKV 30.5 → 26.0, O 16.6 → 15.5,
gate/up 44.0 → 38.2, down 24.5 → 22.9, head 16.6 → 14.5, codec head 44.9 →
37.6; predictor gate/up at three and four rows 48.0 → 45.4 and 52.0 →
50.4). So `ops/Omni.cpp` sends one and two rows, and gate/up pairs up to
four, to the packed kernels (`omni_gemv_packed_c1_r{1,2,4}`), one column a
simdgroup; everything else keeps the bf16 kernels.

**Integration.** `DenseWeights::packed` is an optional second copy;
`packDenseWeights` (`omni/OmniLoader.hpp`) packs a bf16 model's decoder
projections (after the QKV fusion), heads, the TTS text projection and
small-to-MTP projection and the 15 predictor heads, over the load threads
in ranges of 64 rows (each range's escapes counted first). The bf16 weights
stay for prefill, the tiles of five rows or more and the embedding gather
(the ASR head stays the embedding table's view; logits stay fp32). Memory:
**+2.51 GiB for ASR** (server total 17.34 → 19.85 GiB) and **+2.24 GiB for
TTS** (13.67 → 15.91 GiB); load 0.7 → 1.0 s for ASR, unchanged 1.7 s for
TTS. The server packs when its budget leaves 7 GiB beyond the weights and
the copies (on 48 GB, 36 GiB of budget, so always here); `OMNI_PACKED_BF16=0`
turns it off. Like the fused bf16 QKV, the copies live outside the weight
images and stay through an idle release. Quantized checkpoints are
untouched.

**Whole steps**, main and the branch alternating, three each:

| | main | packed |
|---|---:|---:|
| ASR bf16, one-row step on the 35 s file (`OMNI_ASR_DEBUG=1`) | 12.47-12.57 ms | **10.29-10.33 ms** |
| its decode (94 tokens) | 1171-1177 ms | **968-971 ms** |
| ASR bf16, 200 files one at a time (`omni-asr-run --sequential`) | 111.6, 112.0 s | **94.2, 94.9 s** |
| the same with the 0.6B draft (`--draft`) | 52.8, 53.0 s | 52.8, 53.0 s |
| ASR bf16, 200 files at `--concurrency 8` | 33.0 s | 32.8 s |
| TTS bf16 one lane, `--seed 1`, GPU ms a step | 20.33-20.37 | **17.62-17.65** (RTF 0.268 → 0.235) |
| two lanes | 22.01-22.08 | **19.59-19.72** |
| four lanes | 26.05-26.08 | 25.68-25.87 |
| eight lanes | 30.80-30.99 | 30.71-31.16 |

The one-row ASR step lands at 10.3 ms, inside the review's 10-11 ms
hypothesis and above its 9.8-10.1 ms ideal by about the unpacking cost. The
speculative path is unchanged: its verify forwards carry 12 rows, which run
the MPP tiles. Eight TTS lanes run eight-row steps, also on the tiles.

**Bit identity** (bf16 has no parity reference, so the check is against
main): the 200 files' tokens one at a time, with the draft and at
concurrency 8, and the 35 s file's, are identical to main's; seeded TTS at
one, two, four and eight lanes and greedy TTS at one, two and eight lanes
write byte-identical WAVs. `OMNI_PACKED_BF16=0` gives main's tokens and
step times (12.46-12.49 ms). `omni-kernels` checks the round trip on
adversarial tensors (zeros of both signs, subnormals alone and beside small
exponents, zeros beside normal values, spans of exactly 16 and 17
exponents, exponents to 254, infinities and NaN payloads), that row ranges
pack the same bytes, the refusals (a buffer short of an escape record,
escape indices past 24 bits), one-hot reads of single weights through the
packed kernels, and DenseLinear with and without the packed copy bit for
bit over every epilogue, bf16 and fp32 outputs, one to five rows, with and
without infinities and NaNs; a broken rotation fails it. 4-bit real-weight
parity passes unchanged.

**Over HTTP** (`run_benchmark.py`, main 3541236 built in a scratch
directory, each server started from its own tree, alternating, three
rounds; audio s/s; the fit is each file's latency against audio seconds
and tokens):

| ASR bf16 | c=1 (200) | ms a token (fit) | c=8 (200) | long-form | speculative c=1 |
|---|---|---|---|---|---|
| main | 22.0, 22.0, 21.9 | 12.72-12.79 | 72.1, 70.4, 70.4 | 66.0, 65.3, 65.4 | 44.3, 43.9 |
| round 13 | **25.4, 25.8, 25.7** | **10.48-10.58** | 70.7, 71.1, 70.7 | **67.4, 67.5, 67.4** | 43.9, 44.1 |

| TTS bf16 | c=1 audio s/s | RTF median | first audio p50 | c=8 audio s/s (six rounds) |
|---|---|---|---|---|
| main | 3.68, 3.67, 3.67 | 0.274 | 32.6-32.8 ms | 16.8, 16.5, 16.5, 16.3, 16.1, 16.4 |
| round 13 | **4.20, 4.21, 4.22** | **0.238-0.240** | 31.6-31.7 ms | 14.8, 16.3, 16.2, 16.6, 17.0, 16.3 |

One bf16 file at a time is 16-17% faster (22.0 → 25.6 audio s/s, against
sglang-omni's 20.8 and mlx-audio's 20.0), and a token now costs 10.5 ms
against the MLX engines' 12.7-12.8. Long-form gains 3% (its segments decode
together, mostly in steps of several rows); c=8, speculative and eight TTS
streams are unchanged within noise (the TTS c=8 rounds sample, so their
audio differs run to run; 14.8 is one slow round). One TTS stream is 14%
faster. WER is 1.98% (200) and 1.24% (long-form) on both sides, and every
c=1, speculative and long-form transcript matches main's; at c=8 one of 200
("Caffr's" / "Caffar's") differs in two of the branch's three runs: which
rows share a step depends on timing, and a lane's row is summed by the GEMV
up to four rows and by the MPP tiles from five (round 12's caveat), so a
faster step can regroup a file. Offline at `--concurrency 8` all 200 match.

What did not pay, or was not needed:

- **Two or four columns a simdgroup** for packed weights: slower at every
  row count (one row: QKV 53.0 and 83.0 µs against 48.3 for one column,
  gate/up 146.5 and 207.9 against 140.0, the head 1640 and 2359 against
  1629; two rows: QKV 61.2 against 56.1), unlike the bf16 kernels, whose
  one-row plain projections take two. The unpacking per column leaves no
  room to share input loads.
- **Packed weights from three rows** (except gate/up): 2-22% slower (table
  above).
- **The zero mode**: no decoder weight needs it (above).
- **A three-bit exponent subset or entropy coding** (the review's later
  options): not tried; at one row the head already streams packed bytes at
  299 GB/s, so what is left is the 0.78 ratio itself.
- **A single copy** (decompressing into threadgroup memory before the MPP
  tiles, so prefill and the five-row-and-up tiles read the packed weights
  and the bf16 copy could go): not tried. It would return the 2.2-2.5 GiB
  only if the gathers, which read bf16 rows of the tied embedding, also
  learned the format.

## Round 14: multi-row bf16 projections

Eight bf16 files at once were our smallest lead (72.0 against sglang-omni's
61.1 audio s/s): an eight-row ASR step took 19.2 ms of GPU where streaming
its ~3.4 GB of weights takes ~11 ms. Every step of five to sixteen rows ran
the 32x64 MPP tile (`omni_gemm_bf16_m32n64`), which computes 32 rows
whatever the step's rows, with eight simdgroups, a barrier per 64-wide
chunk of K and only N/64 threadgroups (32 for O and down, 64 for QKV), at
150-210 GB/s. The same tile served the bf16 TTS talker and code predictor
at five to eight lanes and the speculative verify forwards (12 rows).
Codex's eleventh review ranked three options: smaller MPP tiles with K
split across threadgroups; 8x8 simdgroup MMAs on registers; and packed
weights decompressed into threadgroup memory before the MMAs.

**What runs now.** `omni_mma_*` (`gemm_bf16.metal`) runs 8x8
`simdgroup_multiply_accumulate` on plain registers through
`common/sgmatrix.h` (its lane mapping and float2 accumulators). Each
simdgroup computes W[8 columns][K] x X[8 rows][K]^T, four simdgroups a
threadgroup (32 columns), so every weight is read once a step and
multiplies all of the step's rows at once. In each 64-wide chunk of K, the
lane of (fm, fn) loads 16 consecutive weights of column fm from
(fn / 2) * 16, two `uint4` loads, and MMA j of the chunk takes their pair j
as its k = fn, fn + 1, so the weight operand needs no shuffle; the
activation operand holds rows fn and fn + 1 at the same input
((fm / 2) * 16 + 2j + (fm & 1)), one `insert_bits` of the two rows' words.
Rows past the step's read as zeros and are not stored. Nine to sixteen
rows take a second row fragment and two column fragments a simdgroup
(`_r16`, 64 columns a threadgroup), so each activation pair serves two
weight fragments. Up to eight rows the tile reads the packed copy of round
13 (`omni_mma_packed_r8`): a lane's two groups of eight are 16
sign/mantissa bytes, two code words and the chunk's block header, rebuilt
to the bf16 words two chunks a loop iteration; a simdgroup whose eight
columns hold no escape block in a chunk (`simd_any`; 0.3% of blocks are
escapes) skips the escape selects. Gate/up is a compile-time variant
(`_gu`), and the epilogue is the dense one: gate and up keep separate fp32
sums until the SiLU product, a residual adds to the fp32 projection, the
final store rounds once, logits stay fp32.

K splits in two when a projection would run fewer than 64 threadgroups or
K reaches 4096 (the decoders' down projections, the predictor's O and
down): each split stores its live rows' fp32 partials and the last to
arrive adds them in split order (`common/split_reduce.h`), from scratch
the decoder arenas own (`Qwen3Arena::denseSplitPartials` and
`denseSplitCounters`, sized by `ops::omni::denseSplitScratch`, passed
through `GemmBuffers`); without it a projection runs unsplit. The policy
(`ops/Omni.cpp denseTile`): one row, and gate/up up to three rows, the
GEMV (packed when there is a copy); two to eight rows the MMA tile, packed
when there is a copy; nine to sixteen the bf16 MMA tile; more the MPP
tiles. Splits take whole pairs of chunks whether or not the weights are
packed, so the packed copy still projects the bf16 bits. The audio tower
keeps the GEMV and MPP tiles (`GemmOptions::stepTiles`): a clip's rows
must encode to the same bits alone or batched, and the MPP tiles compute
a row alike from five rows up.

**Per projection** (`omni-dense-bench --real --policy` on the ASR and TTS
bf16 weights, layers cycled from DRAM past 512 MB of packed bytes, the
choices interleaved over five rounds, median µs; round 13's choice / MMA
over the bf16 weights / MMA over the packed copy; GB/s in bf16 bytes):

| rows | QKV 4096x2048 | O 2048x2048 +res | gate/up 2x6144x2048 | down 2048x6144 +res | head 151936x2048 fp32 |
|---|---:|---:|---:|---:|---:|
| 2 | 55.8 / 59.9 / **51.5** | 30.7 / 31.5 / **27.7** | **144.8** / 175.3 / 144.9 (GEMV) | 82.2 / 92.4 / **78.9** | 1858 / 2114 / **1690** |
| 4 | 82.6 / 60.2 / **52.0** | 46.6 / 31.5 / **28.1** | 179.0 / 180.7 / **157.5** | 131.6 / 93.2 / **80.1** | 2661 / 2125 / **1705** |
| 8 | 85.3 / 60.4 / **53.3** | 55.3 / 31.9 / **28.6** | 253.7 / 181.3 / **157.4** | 165.9 / 93.4 / **81.3** | 2827 / 2130 / **1746** |
| 16 | 84.9 / **71.1** | 54.9 / **40.5** | 253.6 / **194.1** | 166.4 / **108.1** | 2872 / **2293** |

At eight rows the packed tile streams 293-356 GB/s in bf16 terms (QKV 315,
gate/up 320, down 310, the head 356), against 152-220 for the MPP tile.
The TTS code predictor at eight rows: QKV 4096x1024 42.9 / 31.6 / 28.7, O
1024x2048 39.3 / 18.2 / 16.9, gate/up 2x3072x1024 64.0 / 49.0 / 46.4, down
1024x3072 58.8 / 26.0 / 23.7, head 2048x1024 27.3 / 17.0 / 16.1; the
talker's shapes are the ASR decoder's (talker head 3072x2048 66.5 → 43.8).
At sixteen rows the tile is bound by the MMAs, not the bytes: about 4.1
billion 8x8x8 MMAs a second (gate/up), so packed weights, whose words cost
ALU to rebuild, lose there (QKV 85.9, O 48.8, gate/up 215.8, down 130.4,
head 2621 µs).

**Whole steps**, main and the branch alternating:

| | main | round 14 |
|---|---:|---:|
| ASR bf16, eight files (`OMNI_ASR_DEBUG=1`), median eight-row step | 19.39-19.46 ms | **12.29-12.34 ms** |
| the same, five-row steps | 19.25-19.42 ms | **12.08-12.12 ms** |
| ASR bf16, 200 files at `--concurrency 8` | 32.4, 32.9 s | **25.5, 25.4 s** |
| ASR bf16, 200 files one at a time (`--sequential`) | 94.2 s | 94.2 s |
| the same with the 0.6B draft (`--draft`) | 52.4 s | **48.7 s** |
| TTS bf16 `--seed 1`, one lane, GPU ms a step | 17.62-17.63 | 17.58-17.61 |
| two lanes | 19.61-19.73 | **18.40-18.45** |
| four lanes | 25.68-25.81 | **19.67-19.81** |
| eight lanes | 31.43-31.48 | **20.41-20.54** (17.3 → 26.0 audio s/s) |
| TTS 4-bit, eight lanes | 10.33-10.37 | 10.31-10.37 (byte-identical WAVs) |

The eight-row step lands at 12.3 ms, near the review's 13-15 ms estimate
for plain weights at 260-290 GB/s and above its 10.5-12 ms for packed ones
by the head and the step's other work; a five-row step costs what an
eight-row one does, since the bytes do.

**Over HTTP** (`run_benchmark.py`, main d754304 built in a scratch
directory, each server started from its own tree, alternating; audio s/s):

| ASR bf16 | c=8 (200) | c=4 (200) | c=1 (200) | ms a token (fit, c=1) | long-form |
|---|---|---|---|---|---|
| main | 73.4, 70.8, 71.2, 72.7, 70.0 | 54.4, 54.2, 53.9 | 26.0, 26.0, 25.8 | 10.46-10.54 | 67.9, 67.5, 67.5, 68.5, 66.0 |
| round 14 | **90.0, 90.0, 90.0, 91.8, 90.7** | **62.4, 62.4, 63.0** | 25.9, 26.0, 26.0 | 10.48-10.51 | **81.6, 81.0, 81.6, 82.3, 81.6** |

| ASR bf16, speculative | c=1 (200) | ms a token (fit) | c=8 (200) |
|---|---|---|---|
| main | 44.1, 44.2 | 4.31 | 89.2, 89.7 |
| round 14 | **47.6, 47.7** | **3.82-3.86** | 88.7, 89.6 |

| TTS bf16 | c=1 audio s/s | RTF median | first audio p50 | c=4 audio s/s | c=8 audio s/s |
|---|---|---|---|---|---|
| main | 4.21, 4.22, 4.21 | 0.238-0.239 | 31.5-31.7 ms | 10.68, 10.66, 10.84 | 16.73, 16.70, 16.49 |
| round 14 | **4.31, 4.32, 4.32** | **0.233** | **24.7-24.8 ms** | **13.76, 13.94, 13.78** | **23.86, 24.23, 23.76** |

Eight bf16 files at once run 25% faster (72 → 91 audio s/s, against
sglang-omni's 61.1), four 16%, long-form 21% (its segments decode together);
one file at a time is unchanged (its steps hold one row). Speculative
decoding at one request gains 8% from its 12-row verify forwards; at eight
requests its verify forwards hold up to 96 rows, on the MPP tiles, and are
unchanged, now level with plain decoding. Eight TTS streams run 44% faster,
four 29%. One stream gains 2%: in the server, whose predictor QKV tables
are computed eight rows at a time, its sub-steps now take them (below;
`omni-tts-run --lanes 1` computes them four at a time on the GEMV, which
matched one row already, hence its unchanged step), and its first audio
comes 7 ms sooner, probably because the talker's prefill of a short
prompt, 16 rows or fewer, now takes the MMA tile too. The 4-bit models are untouched (ASR c=8 156.2
against 156.3, all 200 transcripts identical; TTS c=8 43.6, 43.8, 45.8
against 43.1, 43.6, 44.1, rounds whose sampled audio differs, and the
offline eight-lane steps above).

**Quality.** WER is 1.98% (200 files, every concurrency and the
speculative path) and 1.24% (long-form) on both sides. One file at a time
every transcript matches main's (the step holds one row, on the GEMV), and
offline at `--concurrency 8` all 200 tokens match. Over HTTP at c=8 three
of 200 differ in punctuation or case ("Caffr's" / "Caffar's", which main
also flips between its own runs; "Your Majesty, having"; "Screamed"), at
c=4 one or two, speculative c=1 one ("Your Majesty,"), long-form one word
of one file ("e'en." / "eatin."): the MMA tiles sum K in another order than
the MPP tile, and the GEMV's single row against the tile's rows was already
round 12's caveat. TTS English WER 0.27-0.63% and Chinese CER 0.18-0.42%
against main's 0.18-0.73% and 0.24-0.48% in the same runs: in range.

**Arithmetic.** A tile's sum of a row adds the chunks in K order, each
chunk's eight MMAs in order, so it does not depend on the step's rows, on
the row or column fragment a row lands in, on the chunks an iteration
loads or on which copy holds the weights. `omni-kernels` checks every
epilogue (bias, GELU, SiLU, residual, gate/up, fp32 logits) at rows 1-9,
12, 15-17, 20, 64 and 100 against fp64 and, from two to sixteen rows,
against the 32x64 MPP tile to a bf16 step; K of 3 chunks (no split, the
bf16 tile under a packed copy), 4 and 8 (split for few threadgroups) and
4096 (split); with and without split scratch, twice in one command and
every counter back at zero; the packed copy against the bf16 weights bit
for bit at rows 1-5, 8, 9 and 16 with and without splits (adversarial
weights, infinities and NaNs); and that every row count
`DenseLinear::sameRowArithmetic` pairs projects the same bits, across row
and column fragments and with the packed copy. So the TTS predictor's
first-layer QKV tables, computed eight (or four) rows at a time, now match
every step of two to eight lanes; a one-lane step on the GEMV takes them
too (`Qwen3Decoder::firstQkvOneRow`), as the affine Qmv tile already did,
its QKV differing only by the association of the fp32 sums.

**Codex's options, measured** (`omni-dense-bench --real`, eight ASR rows
unless stated):

- **Smaller MPP tiles with split K** (option 1): M of 8, 16 or 32 rows, N
  of 32 or 64 columns, four or eight simdgroups, K split 1, 2, 4 or 8 ways
  with the same split reduction: the tile shape changed nothing unsplit
  (QKV 80.9-84.5 µs against 82.7 for the 32x64 tile, gate/up 234.8-247.7
  against 241.4), and splits helped only the projections with few
  threadgroups (O 51.1 → 36.7, down 162.5 → 122.7 at best, 229 and 205 GB/s);
  QKV and gate/up stayed at 79-81 and 230-235 µs. Not used.
- **Register MMA** (option 2): what runs. With gate/up as a run-time flag
  its second weight's registers spilled the rest: 180.4 against 241 µs
  for eight rows of gate/up (QKV 68.9 → 59.5) once it was compile-time.
  Two or four column fragments a simdgroup at eight rows (QKV 62.2 and
  60.7 against 59.5, down 103.4 and 118.9 against 90.5), two or eight
  simdgroups a threadgroup (within 1 µs), prefetching the next chunk
  (QKV 87.9 against 78.4 µs before the compile-time gate), two or four
  chunks an iteration over bf16 weights (QKV 62.2 and 63.4 against 59.4,
  gate/up 188.2 and 230.5 against 180.3: spills), and more than two splits
  were slower or no better. Loads alone (the MMAs replaced by a XOR of the
  words) run QKV 58.7-59.2 and gate/up 172.5-174.4 µs over bf16 weights,
  48.7 and 148.0 over packed ones: the bf16 tile runs at its loads' rate,
  the packed one within 6-9%.
- **Packed weights staged in threadgroup memory** (option 3): not built.
  Its "direct register reconstruction" variant is what runs, and it
  reaches 293-356 GB/s in bf16 terms at eight rows; staging would add
  barriers for no rebuilt word that is not already in registers.

What did not pay:

- **The escape fast path on the packed GEMV**: one row's QKV 48.6 against
  48.5 µs without it, down 75.3 against 73.4, the head unchanged; dropped
  there.
- **Packed weights from nine rows**: 10-20% slower than bf16 (above).
- **The MMA tile for gate/up at two and three rows**: the packed GEMV's
  144.8 and 158.1 µs beat the packed tile's 157-165; the bf16 GEMV then
  serves those rows too, so both copies keep one arithmetic.
- **Packed MMA one chunk an iteration**: QKV 56.3, O 29.6, down 93.8 µs
  against 51.6, 27.5 and 81.7 at two. Rebuilding both chunks' words before
  their MMAs, instead of loading both chunks' raw planes and rebuilding
  each just before use, spilled gate/up (258 µs).

Gates: `make test-engine`, `make test-python` (470 tests), `make
architecture-check`, and 4-bit real-weight parity for ASR and TTS
(unchanged: decode logits cosine 0.999924, TTS talker 0.999978, predictor
0.999774) pass.

## Round 15: pipelines at load

Metal compiles a pipeline the first time a command names its kernel, so a
request after a start paid for every kernel its steps touched. How much
depends on Metal's shader cache, which keeps compiled pipelines on disk
across processes, keyed by each function's bitcode:

- **Cold cache** (the first start after the library changes: an install,
  an update, a rebuild of the kernels, or an OS update): a kernel compiles
  in 2-130 ms (the Q4 decode tiles 60-130), all 275 one at a time in
  12.9 s (`omni_dq4_*` 5.1 s, `decode_linear_q4_*` 4.8, the rest 3.0). The
  first request waited for the 27-48 its steps touch: a 4-bit TTS request
  got its first audio after 838 ms instead of 20.
- **Warm cache** (every later start): all 275 load in 6-8 ms, 25 µs each.
  The first request still ran 5-20 ms slower than the second, but
  compiling every pipeline ahead (an experiment) removed none of that: it
  is first use elsewhere (below).

**What runs now.** `loadEngine` (`OmniServer.cpp`) calls
`MetalBackend::compileLibraryPipelines` right after it creates the backend,
which lists the library's functions and compiles their pipelines on a
`dispatch_apply` of synchronous compiles from a background queue, while the
weights load; `waitForLibraryPipelines` after the engine is built moves
them into the backend's pipeline cache. The server logs `Pipelines
compiled: 275 in 1.55 s, 0.14 s after the weights.` Metal's compiler
service runs about as many requests at once as the Mac has cores: the 275
cold pipelines take 1.5 s, not 12.9. Its asynchronous API
(`newComputePipelineStateWithFunction:completionHandler:`) is as fast but
crashes under shader validation (`make test-engine`; it over-releases an
object in the autorelease pool), so the threads are ours. A command
submitted meanwhile compiles what it needs itself, as before.

The engine compiles the whole library rather than the pipelines its plans
select. Building every plan's graphs (decode steps of 1 to `lanes` rows,
verify rows, prefill chunks, the tower, the predictor's sub-steps, the
codec's chunks) without submitting them would have to touch the arenas'
table slots, their logit rows, the KV pages and the TTS step's buffers
and undo it, and a plan left out would compile lazily without anyone
noticing. The whole library costs 6-8 ms on a warm start, and its 275
pipelines take 5.8 MB of GPU memory. On a cold start it compiles kernels
the model never runs (a short workload touches 27-48; the 64 `omni_dq4_*`
run only with `OMNI_FUSED_DECODE=1`). Leaving out the other format's
kernels and `omni_dq4_*` would cut the cold compile from 12.9 s of
compiler time to about 2.4 (bf16) or 7.4 (4-bit), shortening a start that
happens once an update; not done, since a name filter that drifts from the
plans fails silently too.

Beside it, `AsrEngine` computes a log-mel spectrogram of 0.1 s of silence
at load: the spectrogram's DFT and filter tables and Accelerate's first
GEMM took 2.9 ms of the first request's admission (3.4 against 0.5 ms).
The Neural Engine programs (the ASR prefill FFN split, the TTS codec stack)
were already compiled and loaded at startup.

**Measured** over HTTP (a probe script: start the server, poll
`/ready` every 20 ms, then send one request, or eight at once, and the same
again warm; ASR one 5.8 s LibriSpeech file, or the first eight; TTS the
first English prompts, streamed PCM). Warm cache: main d8aa487 and round
15 alternating, three rounds each; cold cache: every function's bitcode
made new for each start by compiling the kernels with line tables from a
fresh source path. Time to `/ready` (the `Weights loaded` line), first
request (ASR latency, TTS first audio; at c=8 the slowest of the eight),
ms:

| | cache | `/ready` (weights) | first request c=1 | first burst c=8 | warm request c=1 / c=8 |
|---|---|---|---|---|---|
| ASR 4-bit main | warm | 1.65-1.78 s (0.7) | 120, 131, 131 | 819, 816, 835 | 112 / 768 |
| round 15 | warm | 1.65-1.75 s (0.7) | **117, 122, 123** | 817, 823, 820 | 113 / 772 |
| main | cold | 1.88-2.01 s (0.9) | 556 | 1973 | 112 / 771 |
| round 15 | cold | 2.69-2.71 s (1.4) | **121** | **799** | 112 / 770 |
| ASR bf16 main | warm | 1.68-1.75 s (0.9) | 246, 244, 244 | 1448, 1453, 1447 | 237 / 1420 |
| round 15 | warm | 1.67-1.79 s (0.9) | 242, 249, 241 | 1451, 1470, 1456 | 237 / 1427 |
| main | cold | 1.99-2.03 s (1.0) | 685 | 2109 | 236 / 1422 |
| round 15 | cold | 2.70-2.71 s (1.7) | **244** | **1454** | 236 / 1423 |
| TTS 4-bit main | warm | 2.01-2.29 s (1.2) | 33, 30, 32 | 81, 79, 85 | 20 / 70 |
| round 15 | warm | 1.99-2.25 s (1.2) | **22, 27, 26** | 80, 81, 79 | 20 / 70 |
| main | cold | 2.69 s (1.5) | 838 | 1323 | 20 / 70 |
| round 15 | cold | 2.81-2.87 s (1.9) | **23** | **78** | 20 / 70 |
| TTS bf16 main | warm | 2.69-2.73 s (1.8) | 35, 34, 37 | 127, 122, 127 | 24 / 101 |
| round 15 | warm | 2.69-2.71 s (1.8) | **30, 30, 31** | **117, 120, 123** | 25 / 100 |
| main | cold | 2.78-2.85 s (1.8) | 406 | 626 | 25 / 101 |
| round 15 | cold | 3.75-3.80 s (2.4) | **34** | **121** | 24 / 101 |
| ASR 4-bit + 0.6B draft main | warm | 1.66-1.73 s (0.8) | 126, 124, 130 | 816, 825, 817 | 106 / 771 |
| round 15 | warm | 1.66-1.68 s (0.8) | **120, 115, 120** | 821, 805, 817 | 105 / 772 |
| main | cold | 2.26-2.30 s (1.1) | 1394 | 2120 | 105 / 770 |
| round 15 | cold | 2.70-2.71 s (1.6) | **119** | **799** | 105 / 776 |

On a warm cache startup is unchanged (the compile takes 0.00-0.01 s) and
the first request gains about 6 ms at one request (pipeline lookups and
the spectrogram's tables; ASR bf16 nothing measurable), nothing
measurable at eight. On a cold cache the
first request now runs as a warm one (TTS 4-bit first audio 838 → 23 ms,
eight at once 1323 → 78; speculative ASR 1394 → 119 ms) and startup grows
by 0.1-1.0 s (TTS 4-bit 0.15, speculative ASR 0.4, ASR 0.7, TTS bf16 0.95):
the compiler service's processes take the cores the weights' loading and
the load-time GPU work (the ANE split's check, the TTS tables) would use,
so the weights take 0.4-0.7 s longer (probably; the compile always ends
within 0.2 s of them).
Below 3 s, so it is not configurable.

**What the warm first request still pays** (`OMNI_ASR_TRACE=1`, ASR 4-bit
with the draft, first against second request, engine time 113.9 against
103.2 ms): the first speculative round's host build and commit (2.2 ms),
its GPU starting 1.3 ms after the commit instead of 0.1, the draft's first
adoption (1.1 ms) and a few tenths of a millisecond in each other first
command; the GPU time of the commands matches. TTS's first audio comes
2-7 ms after a warm request's. None of it is compilation.

Steady state is unchanged (`run_benchmark.py`, c=1, main and round 15
alternating, one run each): ASR 4-bit 200 files 57.27 → 57.36 audio s/s
(WER 2.17% both), ASR bf16 25.80 → 25.89 (1.98% both), TTS 4-bit RTF
0.103 → 0.103 with first audio 20.0 ms both, TTS bf16 4.30 → 4.32 audio
s/s, first audio 24.6 → 24.8 ms.

Gates: `make test-engine` (with a new backend test: the library's
pipelines compile ahead, a submission meanwhile compiles its own, later
ones compile nothing), `make test-python` (470 tests), `make
architecture-check`, and 4-bit real-weight parity for ASR and TTS
(decode logits cosine 0.999924, TTS talker 0.999978, predictor 0.999774)
pass.

## Round 16: Q4 steps of several rows, and speculation repriced

Codex's twelfth review ranked first the 4-bit projections of steps of two
to sixteen rows (TTS at two to eight streams, ASR at two to eight files,
speculative verification and the draft's catch-up), which run one lane's
MPP tiles at 112-254 GB/s, and the register 8x8 MMA tile
(`linear_q4_sgmatrix.metal`) that Apple10 never runs; second, the
speculation defaults, which predate rounds 12-14.

**Every multi-row option, per projection** (`omni-dq4-bench --multirow
[--register]`: weights cycled from DRAM, chains of 16 dispatches, best of
three interleaved rounds, µs; the norm shapes include their norm as the
decoder runs it, folded or the norm dispatch, the register tile's norm
writing its Table64 input; its residual shapes read a table prepared
beforehand, the best case of a producer that writes it; the register tile
at its best K split; MLX's `quantized_matmul` from `mlx_qmm.py`, M rows,
calls not serialized, without norms):

| projection | 2 rows: plan / register / MLX | 4 rows | 8 rows | 16 rows |
|---|---|---|---|---|
| talker/ASR QKV 4096x2048 | 23.7 / 34.7 / 21.6 | 25.1 / 34.7 / 26.8 | 25.2 / 34.7 / 48.6 | 29.0 / 57.5 / 94.5 |
| O 2048x2048 +res | 14.4 / 18.2 / 12.9 | 14.4 / 18.2 / 16.7 | 14.5 / 18.1 / 27.4 | 17.7 / 29.8 / 49.5 |
| gate/up 2x6144x2048 | 56.8 / 81.8 / 52.3 | 57.8 / 81.9 / 69.0 | 57.8 / 81.8 / 136.2 | 74.1 / 149.6 / 299.3 |
| down 2048x6144 +res | 33.7 / 45.5 / 28.7 | 33.7 / 45.5 / 37.7 | 33.6 / 45.5 / 72.1 | 39.8 / 79.2 / 155.4 |
| talker head 3072x2048 f32 | 19.4 / 28.4 / 17.1 | 20.7 / 28.4 / 21.9 | 20.9 / 28.4 / 38.6 | 24.5 / 45.7 / 71.5 |
| predictor QKV 4096x1024 | 14.4 / 19.2 / 13.1 | 15.1 / 19.2 / 16.9 | 15.1 / 19.2 / 27.8 | 17.6 / 31.3 / 51.2 |
| predictor O 1024x2048 +res | 10.5 / 11.3 / 8.9 | 10.5 / 11.4 / 10.6 | 10.5 / 11.4 / 16.4 | 15.2 / 17.4 / 28.0 |
| predictor gate/up 2x3072x1024 | 18.9 / 25.3 / 17.2 | 20.1 / 25.3 / 22.4 | 20.1 / 25.3 / 38.7 | 29.2 / 42.5 / 73.2 |
| predictor down 1024x3072 +res | 13.7 / 15.1 / 11.0 | 13.8 / 15.1 / 13.7 | 13.7 / 15.1 / 22.4 | 20.7 / 24.2 / 39.0 |
| predictor head 2048x1024 f32 | 10.5 / 13.0 / 8.9 | 11.3 / 13.0 / 10.8 | 11.4 / 13.0 / 16.4 | 13.6 / 18.8 / 27.8 |
| small_to_mtp 1024x2048 +bias | 10.5 / 11.4 / 9.1 | 10.6 / 11.4 / 10.6 | 10.5 / 11.4 / 16.4 | 15.1 / 17.5 / 27.9 |
| ASR head 152064x2048 f32 | 637 / 879 / 635 | 638 / 879 / 843 | 639 / 879 / 1685 | 773 / 1715 / 1218 |

Twelve rows cost what sixteen do in every column but MLX's. The register
tile loses to the MPP plans at every shape and row count, by 8-120%: its
8x8 MMAs run on the FP32 pipe, about 4.1 billion a second (round 14), so
an eight-row projection is bound by its MMAs rather than its bytes (talker
O: 33.5 M multiply-adds in 18.2 µs), and two or four live rows cost what
eight do. Sixteen rows take twice the MMAs: shared-weight row fragments
(one weight load for two eight-row fragments) would remove only the
second read of the weights, while a sixteen-row talker O needs about 32 µs
of MMAs at that rate against the plan's 17.7, so they were not built. The
Table64 preparation adds 1.3-1.5 µs to a residual projection whose
producer cannot write the table (attention, the gate/up product). MLX's
batched path (`qmv_wide`: up to five rows a threadgroup, each lane
dequantizing a whole quant group once for its rows) wins at two rows on
every shape, ties at four on the small predictor shapes and loses beyond.

**What runs now.** Two live rows share the one-row Qmv tile's
threadgroups (`LinearConfig::sharedRows`, `decode_linear_q4_qmv*_r2`):
each 512-input step loads both rows' chunks, then masks each output's codes
into floats once for both rows' dot products, each row's terms in the
one-row kernel's order, so every row is the one-row kernel's bits. Apple10
runs a one-lane step of two live rows on it, after the norm's dispatch,
every epilogue (plain into bf16 or fp32, residual, gate/up); three or more
rows keep the tiles. Per projection, the same bench, norm included:

| two rows | LaneSplit128 plan | Qmv, two shared rows |
|---|---:|---:|
| talker/ASR QKV, norm | 23.7 | **21.5** |
| O + residual | 14.4 | **11.4** |
| gate/up, norm | 56.7 | **54.6** |
| down + residual | 33.6 | **28.2** |
| talker head, norm | 19.6 | **17.2** |
| ASR head, norm | 637 | **598** |
| predictor/draft QKV, norm | 14.4 | **12.4** |
| O + residual | 10.5 | **6.5** |
| gate/up, norm | 19.0 | **16.8** |
| down + residual | 13.7 | **8.6** |
| head, norm | 10.5 | **8.0** |
| small_to_mtp + bias | 10.5 | **6.5** |

**Whole steps** (main and the branch alternating, two runs each, median
GPU ms per step): TTS 4-bit `--seed 1 --lanes 2 --repeat 2` 9.59, 9.65 →
**8.14, 8.08** (greedy: 192 frames in 1904 → 1628 ms of decode); four and
eight lanes unchanged (9.74 → 9.69-9.76 and 10.27-10.31 → 10.32-10.35,
their steps of four to eight rows); one lane 7.18 → 7.11. ASR 4-bit eight
files: two-row steps 4.99 → **4.49** ms, the others unchanged; two files
4.75 → **4.26**; every token identical. Verification
(`omni-asr-verify-bench`, history 200): a two-row 4-bit forward 4.77 →
4.29 ms, the 0.6B draft's two-row forward (a catch-up after a fully
accepted round, or two lanes' step) 2.34 → 1.91.

**Arithmetic.** A row's sum is the one-row Qmv kernel's: `q4-sgmatrix`
checks two shared rows against two one-row dispatches bit for bit at
every epilogue and fixture, fp32 logits, K of whole and partial 512-input
steps, that nothing past the two rows is read or written, and the policy
and its refusals. Against the LaneSplit128 tile, as with round 12's one
row, the sums associate differently before the same bf16 rounding (greedy
TTS at two lanes diverges at frame 6, where one lane did in round 12). The
TTS predictor's first-layer QKV tables keep the eight-row tile's
arithmetic, which three to eight rows share; a two-row step takes them as
a one-row step does (`Qwen3Decoder::firstQkvRowwise`), its QKV differing
from them only by that association.

What did not pay:

- **A threadgroup per live row** (the one-row kernel with the rows the
  grid's fastest dimension, so a column block's rows read its codes back
  to back): each row streams the weights again, and two rows cost nearly
  twice one (ASR head 586 → 1168 µs, talker QKV 32.8 against the tile's
  23.7), winning only the predictor's residual projections and heads.
- **Three or four shared rows**: a third row's chunk spills (talker QKV
  71.5 µs at three rows); in passes of two rows, which read the codes again
  from the cache, three and four rows cost about twice two (talker O 21.2
  and 24.5 against the tile's 14.4, gate/up 118 and 146 against 58).
- **The norm folded into two shared rows**: every threadgroup reduces both
  rows, which costs more than the norm dispatch (talker QKV 25.2 against
  21.5, gate/up 64.7 against 54.6); those forms were dropped.
- **A sixteen-row LaneSplit128 tile** for 9-16 rows (Split128 m16 and the
  N256 gate/up tile run them, 15-28% over eight rows): not built. No
  default 4-bit path runs 9-16 rows (eight lanes at most, one speculating
  lane of six verify rows), and two speculating 4-bit lanes lose before
  verification's cost matters (below).

**Speculation's costs now** (`omni-asr-verify-bench`, history 200, GPU
ms): the 4-bit target 3.95 at one row, 4.3 at two, 4.9-5.0 to eight,
6.1-6.3 to sixteen, 9.8 to 24, 10.4 to 32; bf16 10.2, 10.5-10.7 at two,
11.5-12.4 to eight (the packed MMA tile), 14.8-15.7 to sixteen (the bf16
MMA tile), 19.4-20.1 to 32 (MPP); the 0.6B draft 1.65-1.8 at one row,
1.9-2.0 at two, 2.5-2.6 at four, 3.0 at eight. The draft's tower and
prefill for a lane cost 5.8 ms + 0.116 ms a prompt row (a fit over 183
adoptions; the engine's 7 + 0.115 stands). The engine's 4-bit table had a
one-row step at 4.85 ms and a draft forward at 2.45, so it put what a lone
lane saves at 1.8 ms a token where rounds now save about 1.4.

**k** (engine, `omni-asr-run --sequential --draft`, the 200 files, wall s;
tokens per round):

| k | 4 | 5 | 6 | 7 | 9 | 11 |
|---|---:|---:|---:|---:|---:|---:|
| 4-bit | 37.07 (4.64) | 36.48-36.78 (5.43) | 36.33-36.58 (6.21) | 36.39-36.74 (6.91) | | |
| bf16 | | 49.73 (5.48) | 48.52 (6.25) | **47.45-47.55** (6.97) | 49.68 (8.26) | 48.92-49.55 (9.51) |

A bf16 round of 7 verifies 8 rows on one packed MMA tile (11.8 ms) where
11 needed 12 rows on the bf16 one (15.1): bf16 now proposes 7. 4-bit stays
at 5 (4 to 7 within 1%). Plain decoding took 41.69 s (4-bit) and 94.49
(bf16).

**Lanes.** Engine runs at `--concurrency 2 / 4 / 8` (wall s): bf16 plain
56.39 / 36.32 / 25.32, speculative 35.85 / 28.62 / 25.27 (k 7); with fewer
speculating lanes than requests, the load average keeps every lane plain.
Two speculating 4-bit lanes: 29.08 s (k 5) and 28.81 (k 3) against 27.43
plain at `--concurrency 2`; a two-lane round saves about 0.7 ms a token,
and 51 adoptions cost 1.77 s of GPU for 145 rounds. Over HTTP eight bf16
files lost (89.6 and 89.5 audio s/s at k 11, 88.8 and 89.1 at k 7, against
92.2 and 92.4 plain): requests arrive one at a time, so each draft's tower
and prefill runs alone, about 25 ms of every lane's time, against about 0.5
ms a token that eight-lane rounds save. Limiting bf16 to four speculating
lanes recovered eight files (91.9, 92.6) but lost four (73.6, 74.1 against
76.7, 76.8) and long-form (86.6 against 90.6, 90.9).

**What runs now.**

- bf16 proposes 7 tokens a round (`asrSpeculationDefaults`).
- The cost tables (`AsrEngine::targetMilliseconds`, `draftMilliseconds`)
  hold today's forwards for both formats.
- Where lanes ride (bf16), lanes speculate only while their rounds pay for
  their drafts (`AsrEngine::speculationPays`): what a token saves in rounds
  of the active lanes (`savedPerToken`, every lane speculating) times the
  tokens they likely have left, against the towers and prefills of the
  lanes the draft would still take. Lanes it follows cost nothing more, so
  rounds that have started go on; eight short requests mostly decode
  plainly, while eight long-form segments and up to four short requests
  speculate. Which lanes the draft takes is unchanged (a bf16 lane still
  weighs a fixed 7.5 ms a token: about a lone lane's saving, and less than
  riding costs a lane beside others).
- A tick that considered speculation completed the step in flight before
  asking the draft for lanes, so a lane the draft refused (too few tokens
  left, a dropped draft) lost round 10's queued steps. It now completes
  the step only when the draft takes a lane or a round runs: speculative
  4-bit at two requests went from 80.5-80.7 to 85.3-85.4 audio s/s
  (85.0-85.3 plain), and the 4-bit deficits at eight files and long-form
  (README 154.5 against 155.9, 143 against 146) are gone.
- 4-bit keeps one speculating lane: eight (`--speculative-lanes 8`) lost
  6% at two requests (80.0 against 85.2) and 5% on long-form (139.7
  against 147.2) over HTTP, the per-lane estimate taking long segments
  whose two-lane rounds do not repay their drafts.

**Over HTTP** (`run_benchmark.py`, main 45c251c built in a scratch
directory, each server started from its own tree, alternating, two rounds;
audio s/s):

| | main | round 16 |
|---|---|---|
| ASR 4-bit c=1 (200) | 58.1, 58.0 | 57.6, 58.0 |
| ASR 4-bit c=8 | 155.8, 156.1 | 156.8, 156.8 |
| ASR 4-bit long-form | 142.2, 145.3 | 146.6, 147.0 |
| speculative 4-bit c=1 | 63.8, 63.6 | **64.6, 64.4** |
| speculative 4-bit c=2 | 80.7, 80.5 | **85.4, 85.3** |
| speculative 4-bit c=4 | 119.0, 118.3 | 119.3, 119.4 |
| speculative 4-bit c=8 | 156.2, 155.5 | 157.0, 156.1 |
| speculative 4-bit long-form | 145.5, 145.2 | 147.3, 147.2 |
| speculative bf16 c=1 | 47.6, 47.3 | **49.6, 49.8** |
| speculative bf16 c=2 | 57.2, 55.8 | **61.7, 61.9** |
| speculative bf16 c=4 | 75.8, 74.1 | 76.3, 77.3 |
| speculative bf16 c=8 | 90.2, 83.8 | **91.8, 94.3** |
| speculative bf16 long-form | 90.3, 90.1 | 90.7, 91.6 |
| plain bf16 c=1 / 2 / 4 / 8 / long-form | | 26.0, 26.4 / 41.8, 42.3 / 64.7, 65.3 / 91.9, 93.1 / 82.5, 84.0 |
| TTS 4-bit c=1: audio s/s; RTF; first audio p50 | 9.87, 9.82; 0.103; 19.9, 20.0 ms | 9.89, 9.89; 0.102, 0.103; 19.8, 20.0 ms |
| TTS 4-bit c=4 | 26.2, 26.2 | 26.2, 26.5 |
| TTS 4-bit c=8 | 45.0, 45.0 | 44.6, 44.6 |

Speculation now runs at least as fast as plain decoding at every
concurrency and on long-form in both formats: 4-bit 1.11x at one file and
within 0.5% from two up and on long-form; bf16 1.9x at one file, 1.5x at
two, 1.2x at four, 1.0x at eight, 1.1x on long-form. The TTS figures at
one, four and eight streams do not move: their steps hold one or four to
eight rows (the harness's four and eight streams reach two rows only at
their tails).

**Quality.** WER 2.17% / 1.45% (4-bit, 200 files / long-form) and 1.98% /
1.24% (bf16) on both sides, plain and speculative (bf16 speculative at
c=4: 2.00% on both; at c=8 2.00% on main, 1.98% on the branch). Plain
4-bit transcripts are identical to main's one file at a time (one-row
steps) and on long-form; at eight files 199 of 200 ("Mister"/"Mr.", a
two-row tail step). Speculative against plain decoding of the same build,
one file at a time: 4-bit 197 of 200, bf16 198 of 200 (commas, a capital,
"Mr."); main's speculative bf16 gave 199. TTS English WER 0.36-0.63% and
Chinese CER 0.30-0.48% against main's 0.27-0.91% and 0.30-0.42% in the
same runs (Qwen3-ASR judge): in range.

Gates: `make test-engine` (with the shared-row cases and policy in
`q4-sgmatrix`; the speculation test's fourth request asks for 12 tokens
instead of 7, so the exact draft still takes a lane that the disagreeing
one refuses at the new costs), `make test-python` (470 tests), `make
architecture-check`, and 4-bit real-weight parity for ASR and TTS
(unchanged: decode logits cosine 0.999924, TTS talker 0.999978, predictor
0.999774) pass.

## Round 17: speculation by default

Round 16 left speculation at least as fast as plain decoding at every
concurrency and on long-form in both formats, so serving Qwen3-ASR-1.7B
now loads `mlx-community/Qwen3-ASR-0.6B-4bit` as its draft unless
`--no-draft-model` turns it off; `--draft-model PATH_OR_ID` still picks
another (any transcription model takes one). The decoding is unchanged:
the defaults of rounds 8 and 16 decide where speculation engages (4-bit:
one active lane; bf16: while rounds repay their drafts).

- **Which models.** The server reads the checkpoint's `config.json`
  (`serve_options.default_draft_model`): a qwen3_asr model with a
  2048-wide decoder, 4-bit or bf16, gets the draft; speech models and the
  0.6B itself get none. `--draft-tokens` and `--speculative-lanes` no
  longer need `--draft-model`; they need a draft (the default one or an
  explicit one) and refuse `--no-draft-model`.
- **Fetching.** `splash serve`'s installer step downloads the draft into
  the Hub cache beside the 1.7B (`upstream.fetch_draft`, through
  `huggingface_hub.snapshot_download`, about 0.7 GB); the server looks for
  it in the cache and downloads it itself when missing (a server started by
  hand). Offline or unreachable, the server warns and decodes plainly; an
  explicit `--draft-model` that cannot be had still ends the startup. The
  download happens before the engine starts, so the engine's startup
  timeout is unaffected (59 s for the download into an empty cache, then
  ready as usual).
- **Memory.** The engine loads the draft only where the budget fits it:
  the weights with the draft's, about 3 GiB of arenas and workspace, the
  Neural Engine split's 0.5 GiB (4-bit), the draft's 0.5 GiB of workspace
  and the 1 GiB KV pool the engine falls back to. Otherwise it drops the
  draft and logs `Speculative decoding: off; the memory budget of 6.00 GiB
  cannot fit the draft (7.40 GiB needed).` (4-bit, `--max-memory 6G`; 3.96
  GiB allocated at ready, as without a draft). With the automatic budget the
  draft adds 1.30 GiB (4-bit 15.07 → 16.37 GiB, bf16 19.85 → 21.15).

**Startup without any draft flag** (4-bit):

```
Loading · mlx-community/Qwen3-ASR-1.7B-4bit · qwen3-asr
Draft · mlx-community/Qwen3-ASR-0.6B-4bit · speculative transcription by default; --no-draft-model turns it off
Speculative decoding: draft .../models--mlx-community--Qwen3-ASR-0.6B-4bit/snapshots/313d8501..., 5 tokens per round while at most 1 lanes are active.
Weights loaded in 1.0 s; 16.37 GiB allocated; 8 lanes.
Ready · mlx-community/Qwen3-ASR-1.7B-4bit · qwen3-asr · 8 lanes · http://127.0.0.1:8791
```

bf16 reports 7 tokens per round while at most 8 lanes are active. With
`HF_HUB_OFFLINE=1` and an empty `HF_HUB_CACHE`: `Downloading · draft
mlx-community/Qwen3-ASR-0.6B-4bit ...`, then `Warning · cannot download
draft mlx-community/Qwen3-ASR-0.6B-4bit (Cannot find an appropriate cached
snapshot folder ...); serving without speculative decoding`, and Ready.

**Over HTTP** (`run_benchmark.py asr --concurrency 1 --limit 200 --set
subset`, one pass each, audio s/s and WER):

| | default | `--no-draft-model` | round 16 (speculative, plain) |
|---|---:|---:|---|
| 4-bit c=1 | 64.6, 2.17% | 57.2, 2.17% | 64.6, 64.4; 57.6, 58.0 |
| bf16 c=1 | 49.2, 1.98% | 25.8, 1.98% | 49.6, 49.8; 26.0, 26.4 |

The default reproduces round 16's speculative figures and
`--no-draft-model` its plain ones, so the README's figures stand, with the
speculative ones now the headline Splash Omni columns and the plain ones
"without draft (`--no-draft-model`)".

Gates: `make test-python` (481 tests: the default for 4-bit and bf16
1.7B, none for speech or the 0.6B, `--no-draft-model`, an explicit draft,
a missing draft downloaded, a failed download falling back, the
installer's fetch and the launcher's `--no-draft-model`), `make
check-python`, `make architecture-check` and `make test-engine` pass.

## Round 18: the dense MPP tiles over the whole K, and the TTS step in two commands

Two items of Codex's thirteenth review: the audio tower's bf16 GEMMs
(its second item), which ran at 9-13 TFLOPS at 151 rows, and the host
time of a TTS step (its fifth).

**Where the tower's time went.** One 11.6 s file (151 tokens, 4-bit,
`OMNI_ASR_DEBUG=1 omni-asr-run`): the tower takes 16.0-16.5 GPU ms. By
kernel (`omni-pipeline-profile`, each dispatch alone): the GEMMs 12.2 ms
(the 64x128 tile 8.1, its fp32-row form 4.2), im2col 1.7, conv1 1.2,
the windowed attention 1.1, the layer norms 0.5. Of the GEMMs' 145 GFLOP
a file, conv2 and conv3 over the im2col rows (12 chunks: 9600 and 2496
rows of K 4352) are 54 and the 24 layers 91.

**The sweep** (`omni-dense-bench --tower [--conv | --decoder]`, a new
mode: every tower shape with its epilogue — bias, GELU, the fp32 residual
added in place — at 151, 166, 300 and 500 rows and the convolutions'
row counts, weights cycled from DRAM, chains of 16 dispatches, median of
three rounds, useful and padded TFLOPS, each checked against fp64 and
against the policy's bytes; the variants are
`dev/benchmarks/omni_dense_variants.metal`, linked into a bench library by
`make omni-dense-variants`). At 151 rows, the seven per-token shapes
summed as a file runs them (conv_out, proj1 and proj2 once, QKV, out,
fc1 and fc2 24 times), ms:

| tile (eight simdgroups unless noted) | K a call 64, barrier after each | barrier every 4 | no barrier | K 128 / 256, no barrier | whole K |
|---|---:|---:|---:|---:|---:|
| 64x128, round 17's kernel (a second accumulator for gate/up) | 8.93 | | | | |
| 64x128 | 8.02 | 8.05 | 6.87 | 6.77 / 6.67 | 6.70 |
| 64x64, four simdgroups | 7.04 | 7.05 | 6.66 | 6.49 / 6.40 | 6.40 |
| 32x64 | 7.10 | | | 6.71 (256) | 6.70 |
| 32x128, four simdgroups | | | | 6.26 (256) | 6.24 |
| best of every variant, per shape | | | | | 5.8 |

- The barrier after each call costs 5-15% and a barrier every fourth
  call as much as one every call; without it, larger K per call gains a
  little more, and one call over the whole K (MPP walks K itself) is
  best or tied on every tile.
- The production kernel's second cooperative accumulator, zeroed and
  carried for gate/up on every projection, cost another 10% (8.93 against
  8.02 for the same tile without it).
- K of 32 a call loses 4-12%; 128-row tiles lose up to 500 rows (they win
  only near 6400); four simdgroups beat eight from about 64 threadgroups.
- Pretransposed weights ([K][N], N contiguous): 0-6% faster on some shapes
  (32x128 at K 256 6.26 → 6.03 ms), against a second 0.6 GB layout of the
  tower; not taken.
- fp16 operands (fp32 sums): within 1% of bf16 on every tile (64x64 over
  the whole K 6.42 against 6.40); no speed to pay for half activations, so
  their range was not pursued.
- Every variant writes the same bytes as the policy (bf16 and fp32 rows,
  every tile, K split and barrier tried): MPP sums an output in the same
  order whatever the tile.

**What runs now.** `omni_gemm_tile` makes one matmul call over the whole K
with no barrier, gate/up compile-time (`_gu` kernels), in six tiles
(`omni_gemm_bf16_m{32,64}n{64,128}s{4,8}`: 32x64/8, 32x128/4, 64x64/4,
64x64/8, 64x128/4, 64x128/8), and `ops::omni::mppTile` picks: 32x64 of
eight simdgroups to 32 rows or while that makes at most 64 threadgroups;
64x128 of four for K of 2048 and more from 48 of those; 64x64 of eight up
to 48 of those; else 64x64 of four. Over every tower shape from 24 to 500
rows the policy is within 1.6% of the best tile per shape (57.7 against
56.8 ms summed), over the bf16 decoder's prefill shapes from 24 to 2000
rows within 3% (637 against 619). Per shape (µs, round 17's kernels in
brackets): at 151 rows QKV 65.8 (93.6), out 22.7 (36.1), fc1 87.2 (111.0),
fc2 81.3 (119.1), conv_out 143.8 (215.0); conv2 over 6400 rows 1176
(1404, 22.6 TFLOPS); the bf16 decoder at 166 rows QKV 166.2, O 79.7,
gate/up 593.1, down 220.1. `omni_kernels_test` checks every tile against
the others byte for byte and against fp64 (bias and GELU, fp32 residual,
gate/up, 17-130 rows), and the policy's wide tiles through `DenseLinear`.

**Engine** (main 09cb53e and the branch alternating, GPU ms): the 11.6 s
file's tower 16.0-16.5 → 13.1-13.6 at 4-bit and bf16 (by kernel the GEMMs
12.2 → 9.5); the bf16 prefill of its 166 rows 40.9-41.8 → 33.4-33.7. The
200 utterances (`omni-asr-run`, wall s): 4-bit one at a time 41.57 → 41.14,
eight at a time 15.24 → 14.62; bf16 94.31 → 91.39 and 25.23 → 22.79. The
tokens are identical to main's in all four.

**Over HTTP** (one file at a time, the 200, main and the branch
alternating, two rounds, audio s/s and p50 latency):

| | main | round 18 |
|---|---|---|
| 4-bit (default, speculative) | 63.9, 64.4; 154 ms | **65.9, 65.9**; 149 ms |
| 4-bit `--no-draft-model` | 57.9, 57.8; 171 ms | **58.4, 58.5**; 169 ms |
| bf16 (default, speculative) | 49.2, 49.3; 202 ms | **52.6, 52.9**; 191 ms |
| bf16 `--no-draft-model` | 25.8, 26.0; 377 ms | **26.8, 26.8**; 366 ms |

The speculative paths gain more: the 0.6B draft's own tower and, at bf16,
the target's prefill run on these tiles too. Every transcript is
identical to main's (200 of 200 in all eight runs); WER 2.17% (4-bit) and
1.98% (bf16).

What is left of the tower: im2col and conv1 take 2.9 of its 13 ms (each
dispatch alone); conv2 and conv3 read the im2col rows the im2col kernel
writes (105 MB a 12 s file at about 60 GB/s), so an implicit-GEMM
convolution that loads its A tile from conv1's output would save most of
the 1.7 ms. Not done in this round.

**The TTS step's host time** (`OMNI_TTS_DEBUG=1 omni-tts-run`, the test
text, 4-bit, seed 1): a step built the talker forward and the 15
predictor sub-steps (577 dispatches at one lane, 779 at eight) into one
command and waited. Medians a step, ms:

| | build | GPU | submit to wake | host exposed | step wall |
|---|---:|---:|---:|---:|---:|
| 1 lane, one command | 0.10 | 7.10-7.14 | 7.44-7.49 | 0.42-0.43 | 7.54-7.60 |
| 1 lane, two commands | 0.03 | 7.12-7.13 | 7.35-7.36 | 0.24-0.25 | 7.38-7.39 |
| 8 lanes, one command | 0.13 | 10.17-10.34 | 10.60-10.74 | 0.52 | 10.74-10.87 |
| 8 lanes, two commands | 0.04 | 10.25-10.35 | 10.47-10.57 | 0.26-0.27 | 10.50-10.61 |
| bf16, 1 lane, one / two | 0.09 / 0.02 | 17.6 | 18.06-18.10 / 17.88-17.93 | 0.54 / 0.30 | 18.15-18.19 / 17.90-17.95 |

So about 0.4-0.5 ms a step was host time the GPU waited for (round 4
measured 0.15 + 0.25). Now the talker command (the codec fronts, the step
input, the talker forward, code 0's sampling and the gather of the
predictor's input) is submitted first, and the predictor's sub-steps are
built and encoded while it runs and queued behind it
(`MetalBackend::submitCommandBehind`, which now takes a completion, so the
wait for the Neural Engine's chunks still wakes at the second command's
end). The predictor's commands read only what the talker command writes
on the GPU; their host tables and logit rows belong to the predictor's
arenas, which no command in flight reads, and the step's uniforms and
code-0 history are written before the first submission.
`OMNI_TTS_SPLIT_STEP=0` keeps one command (it measured as main). What
stays exposed is the talker command's own encoding, its launch and the
wake after the second command. Speech is bit-identical to main's: seeded
4-bit at one and eight lanes, greedy 4-bit and bf16 at one and four, seeded
bf16 at one and eight (wav md5). Engine RTF at one lane 0.1024-0.1034 (main and
`OMNI_TTS_SPLIT_STEP=0`) → 0.0999-0.1005, eight lanes 46.3-46.7 → 47.4-47.5
audio s/s; bf16 one lane
0.2334-0.2343 → 0.2302-0.2310.

Over HTTP (`run_benchmark.py tts --response-format pcm`, 60 requests, main
and the branch alternating, two rounds):

| | main | round 18 |
|---|---|---|
| 4-bit c=1: audio s/s; RTF; first audio p50 | 9.97, 9.91; 0.101, 0.102; 19.9, 19.8 ms | **10.11, 10.16; 0.100, 0.099**; 19.9, 20.0 ms |
| 4-bit c=8 | 44.5, 43.4 | **46.0, 45.6** |
| bf16 c=1 | 4.32, 4.33; 0.233, 0.233; 24.8, 24.8 ms | **4.37, 4.37; 0.230, 0.230**; 24.8, 24.8 ms |
| bf16 c=8 | 23.6, 24.0 | 24.3, 23.9 |

One stream gains 1-2% (first audio moves less than its noise: a step
saves 0.2 ms), 4-bit at eight streams 3%, bf16 at
eight within noise. English WER 0.27-0.72% and Chinese CER 0.18-0.54%
on the branch against 0.27-0.54% and 0.18-0.30% on main in the same runs
(Qwen3-ASR judge; the requests sample, so each run's speech differs): in
range.

**The README's tables**, rerun in one pass of the branch (each Splash Omni
row once; audio s/s, with the draft / `--no-draft-model`): 4-bit ASR one
file at a time 65.1 / 58.7 (README 64.4 / 58.0), eight files 161.2 / 161.1
(156.1 / 156.8), long-form 151 / 151 (147 / 147); bf16 52.7 / 26.8 (49.8 /
26.4), 103.7 / 102.9 (94.3 / 93.1), 101 / 89 (92 / 84); TTS 4-bit first
audio 20 ms, RTF 0.099 (0.103), four and eight streams 27.2 and 46.0 (26.2,
44.1); bf16 25 ms, 0.229 (0.233), 14.1 and 24.2 (13.9, 23.7). bf16 at eight
files and long-form gains most (10%): their prefills of 300-400 rows and
ragged chunks run on the new tiles. The 35 s file's tower takes 33.9 ms
(40.2 on main). WER 2.17% / 1.45% (4-bit, 200 files / long-form) and 1.98% /
1.24% (bf16) as before; TTS English WER 0.27-0.91% and Chinese CER
0.30-0.42% across the six runs (round 16's main: 0.27-0.91% and 0.30-0.42%).

Gates: `make test-engine` (with the MPP tile agreement test in
`omni-kernels` and the queued command's completion in `metal-backend`),
`make test-python` (481 tests), `make architecture-check`, and 4-bit
real-weight parity for ASR and TTS (unchanged: decode logits cosine
0.999924, TTS talker 0.999978, predictor 0.999774) pass.

## Round 19: cleanup

The rounds above left the paths they replaced, and some they tried and
dropped, in the tree behind environment switches or as kernels only a
benchmark named. With the performance work paused, they are gone; every
default path, and the HTTP API, is unchanged. What went, and why:

| Removed | Was | Why |
|---|---|---|
| `ops::omni::DecodeQ4`, `omni/decode_q4.metal`, `OMNI_FUSED_DECODE` | Q4 decode projections streaming weights two groups ahead through a threadgroup ring, norms folded producer to consumer (64 pipelines) | matched the plans without beating them ("Results so far"), 20-30% behind the paired tile on residual projections (round 5) |
| `omni/gemv_q4.metal`, `OMNI_GEMV_DECODE` | a scalar one-row Q4 GEMV (4 pipelines) | 2% on the predictor only, near-tie codes flipped (round 3); superseded by Qmv (round 12) |
| the LaneSplit128 tile's folded norms (`Q4NormInput`, `decode_linear_q4_n128_lane_split{,_gate_up}_norm_r{1,2,4,8}`) | round 6's norm folded into one lane's tile at one or two live rows (12 pipelines) | Qmv takes one live row (round 12, folding the norm itself) and two (round 16, after the norm's dispatch); the tile has run only three to eight rows since, where it never folded |
| `omni_gemv_bf16`, `omni_gemv_bf16_c4_r*`, `_c2_r{2,4,8}`, `_c1_r8` | bf16 GEMV instances timed while round 4 chose (18 pipelines) | `denseTile` names only `c1_r{1,2,4}` and `c2_r1` |
| `omni_codec_conv_half_m*` | eight wider half-precision codec tiles | never ahead beyond noise (round 3) |
| `OMNI_CODEC_FP32` | the fp32 codec convolutions instead of the half ones | the reference the half kernel was checked against ("Results so far"); the fp32 kernel stays for convolutions of other channel counts and as the tests' exact path |
| `OMNI_ONE_PAGE_ATTENTION`, `OMNI_QUERY_HEAD_ATTENTION`, `OMNI_LONG_ATTENTION` | A/B switches of rounds 7b, 9b and 11 | the defaults' kernels are the policy; prepare, split and reduce stay as the reference the long kernel is checked against |
| `OMNI_FOLD_NORM`, `OMNI_PREDICTOR_TABLES`, `OMNI_PREDICTOR_QKV_TABLES`, `OMNI_ASR_PIPELINE`, `OMNI_TTS_SPLIT_STEP` | bit-identical A/B switches of rounds 6, 4, 7b, 10 and 18 | no user needs the old path; the predictor's per-step projection and its buffer went with the tables' switch |
| `OMNI_CODEC_ANE_FIRST` | a request's first codec chunk on the Neural Engine | 3-4 ms later first audio at one stream (Neural Engine codec, above) |

Benchmark modes that timed only removed kernels went with them (the
pipelined variants of `omni-dq4-bench`, the extra GEMV columns of
`omni-dense-bench` and its round-13 baseline, whose 32x64 tile round 18
had renamed away, the tile sweep of `omni-codec-bench`, the decode split
path of `omni-step-profile`), as did the decoder test's norm fold of the
pipelined tiles and q4-sgmatrix's lane-split folded norms. Kept on
purpose: the GPU-family branches of `ops::Linear` (Apple9's register tile,
the N128/N256/Paired tiles of several lanes), the fused KV-head kernels and
the decode split path as the reference kernels of `omni_kernels_test`, the
bf16 GEMV and MMA tiles a model without its packed copy runs, the fp32
codec kernel, and the switches an operator may need:
`--disable-ane` / `OMNI_DISABLE_ANE=1`, `OMNI_CODEC_ANE=0`,
`OMNI_PACKED_BF16=0` (2.2-2.5 GiB less memory), `--no-draft-model`, the
memory limits and the debug and trace output (`OMNI_ASR_DEBUG`,
`OMNI_TTS_DEBUG`, `OMNI_ASR_TRACE`, `SPLASH_AUDIO_TRACE`).

The library the server compiles at load (round 15) went from 299
functions to 193, and from 5.51 to 2.60 MB; the engine's sources
(runtime/) lost about 1400 lines net. The decode arenas no longer
allocate the pipelined projections' partials, counters and norm partials.

**Bit identity** against main (fbd695d, built in a scratch directory):
the 200 utterances' tokens one at a time (`omni-asr-run --sequential`),
4-bit and bf16, plain and with the 0.6B draft, are identical on all 200;
greedy and seeded (`--seed 1`) TTS at one and eight lanes, 4-bit and
bf16, write byte-identical WAVs with every request's frame count equal.
`make test-engine`, `make test-python` (520 tests), `make check-python`,
`make architecture-check` and 4-bit real-weight parity for ASR and TTS
(decode logits cosine 0.999924, TTS talker 0.999978, predictor 0.999774)
pass.

**Over HTTP** (`run_benchmark.py`, the 4-bit models, main and the
branch alternating, audio s/s; ASR with the default draft on the 200,
TTS 60 requests streaming PCM):

| | main | round 19 |
|---|---|---|
| ASR c=1: audio s/s; p50 latency | 66.3, 67.0; 147 ms | 66.6, 65.8; 148 ms |
| ASR c=8 | 163.7, 163.2, 163.8, 163.6 | 162.3, 162.0, 164.4, 163.5 |
| TTS c=1: audio s/s; RTF; first audio p50 | 10.33, 10.42; 0.097; 19.7-19.8 ms | 10.36, 10.33; 0.097-0.098; 19.6-19.8 ms |
| TTS c=8 | 46.2, 47.3, 45.7, 46.2 | 46.0, 45.7, 45.9, 45.7 |

All within the runs' noise (the TTS requests sample, so each run's speech
and length differ). The engine alone, three alternating rounds, has the
same eight-way figures: the 200 files at `--concurrency 8` in 14.40-14.46
against 14.37-14.50 s with identical tokens, and 24 seeded TTS requests on
eight lanes in 3.765-3.772 against 3.763-3.769 s at 10.02-10.03 against
10.01-10.02 GPU ms a step, with identical WAVs. A start compiles 193
pipelines instead of 299 (0.00-0.01 s on a warm shader cache, 0.17 s the
first time the rebuilt library's changed functions met it), and the ASR
server's allocation drops from 16.37 to 16.36 GiB.

## VoiceDesign and CustomVoice (no optimization)

The Qwen3-TTS VoiceDesign and CustomVoice checkpoints reuse everything of
the Base model's (talker, predictor, codec, Neural Engine codec, packed
bf16 copies, every tile and lane); only the prompt differs (`ttsPrompt`,
docs/omni-design.md, TTS pipeline): the instruction's user turn leads as
text rows, a CustomVoice speaker is one more codec row. No kernel or
schedule was changed for them, and none was tuned.

**Correctness.** Real-weight parity of mlx-audio's own prompt
(`omni_parity_reference.py tts-prompt`, `omni-model-parity tts-prompt`),
talker logits: VoiceDesign-4bit 0.999980 (instruction), 0.999959
(instruction and English); CustomVoice-4bit 0.999919 (speaker), 0.999889
(speaker, instruction, English), 0.999966 (dialect speaker `dylan`,
automatic language to the Beijing dialect id); VoiceDesign-bf16 0.999987,
CustomVoice-bf16 0.999941; Base-4bit with English 0.999987; argmax equal
everywhere, predictor logits 0.99935-0.99997. One VoiceDesign-4bit
instruction ("A calm, clear male voice with a neutral tone...") has an
exact bf16 tie in mlx-audio's own top two talker logits (404 and 1342 both
19.125), which the 4-bit noise flips; the test uses another. Greedy
frames against mlx-audio (`omni_tts_greedy_reference.py`): frame 0's talker
code matches in every case, and the greedy chains part within 2-4 frames,
as Base-4bit's already do on main (its frame 0 shares 12 of 16 codes):
4-bit Base and VoiceDesign share 9-12 of frame 0's codes, CustomVoice all
16. Base speech is byte-identical to main's: `omni-tts-run`, greedy and
seeded, one and eight lanes, 4-bit and bf16 (8 WAVs), and over HTTP one
request at a time, greedy, seeded, English and Mandarin, WAV and streamed
PCM, 4-bit and bf16 (16 answers).

**Speed** (`run_benchmark.py tts`, HTTP, 60 standard prompts; README,
Performance): VoiceDesign-4bit runs as Base-4bit (first audio 20 ms, RTF
0.103, 26.3 and 45.1 audio s/s at four and eight streams; Base: 20 ms,
0.099, 27.2, 46.0), with 22 more prompt rows a request. CustomVoice-4bit
with `vivian` is 0.112 and 40.9: the speaker's speech is about 1.5x as long
for the same text, so the decode attends over longer histories. mlx-audio
is 0.212-0.215 one stream at a time. The first bf16 pass ran on battery
power and came out throttled (a 4-bit Base check after it measured 0.113
and 38.0 instead of 0.099 and 46.0); rerun on AC power on 2026-10-21, bf16
VoiceDesign is 0.234 with 13.4 and 22.6 audio s/s at four and eight
streams, CustomVoice 0.239, 13.1 and 24.6 (mlx-audio 0.374 and 0.377).
The Base spot check in the same session, this branch against the previous
main alternating twice, is level: 4-bit 0.097-0.098 and 46.3-46.7 against
0.097-0.098 and 46.0-46.1 at eight streams; bf16 0.226-0.227 and 24.3-25.1
against 0.226-0.227 and 23.8-24.9.

## Round 20: Codex's fourteenth review

Three items of the fourteenth review, each measured against main (5b1fceb,
built in a scratch directory) alternating, on AC power. Only the second
paid, and not in the form the review proposed.

**1. Compile-time gate/up in the bf16 GEMV: not kept.** `omni_gemv` reads
the gate/up flag at run time, sets up both weight columns and carries and
reduces both sums whatever the projection. Split into compile-time
instances (`_gu` for gate/up), `omni-dense-bench --real` (median of seven
interleaved rounds, µs, the policy's kernel, compile-time / run-time):

| one row | QKV | O +res | gate/up | down +res | head |
|---|---:|---:|---:|---:|---:|
| ASR, packed | 48.2 / 48.4 | 26.5 / 27.0 | 138.7 / 139.9 | 73.1 / 73.5 | 1622 / 1622 |
| talker, packed | 48.2 / 48.3 | 26.4 / 27.1 | 139.2 / 141.3 | 73.2 / 73.7 | 37.9 / 37.9 |
| predictor, packed | 25.9 / 25.9 | 15.7 / 15.5 | 38.3 / 38.4 | 22.3 / 23.0 | 14.3 / 14.4 |
| ASR, bf16 (two columns) | 58.9 / 59.0 | 30.8 / 30.9 | 169.8 / 170.7 | 89.2 / 89.2 | 2058 / 2059 |

Two rows gain more on the plain packed kernels (QKV 52.8 / 55.5, O 28.6 /
30.6, down 79.0 / 82.2, the ASR head 1702 / 1840 µs), but `denseTile` sends
two plain rows to the MMA tile; only gate/up runs the GEMV there, and it
does not gain (144.8 / 144.7; the predictor's 41.4 / 40.6). Whole steps
(`OMNI_TTS_DEBUG=1 omni-tts-run --lanes 1 --seed 1`, three rounds, median
GPU ms a step): bf16 Base 17.16-17.18 → 17.11-17.17, VoiceDesign
17.17-17.19 → 17.10-17.11, CustomVoice 17.21 → 17.10-17.13; the bf16 ASR
one-row step on the 35 s file 10.23-10.24 → 10.19-10.20. That is 0.04-0.1
ms (0.3-0.6%), against the review's 0.2-0.5 ms a frame.

And gate/up is not bit-identical. The plain instances write main's bytes
on every shape, but the compile-time gate/up instance differs in one of
6144 bf16 outputs of the ASR layer 0 (one ulp), and with fp32 outputs in
2827 of 6144 by a few ulps: the AIR of the two is the same loop, but with
the run-time branch gone from the inner loop the GPU compiler associates
the fp32 sums differently under fast math. Explicit `fma` chains or
`#pragma METAL fp contract(off)` do not reproduce main's arithmetic either
(they differ from it even on the plain projections, the head in 76886 of
151936 outputs). So TTS speech would change (the seeded WAVs differed) for
a gain under 1%; plain-only specialization would keep the bits for about
half of that. Neither was kept.

**2. The audio tower's front end.** The review proposed an implicit GEMM
for conv2 and conv3 to remove the 105 MB of im2col rows (round 18: about
1.7 ms a 11.6 s file). Most of that cost was the im2col kernel's form, not
the materialization: one thread per bf16 element, 2-byte loads and stores
at about 60 GB/s. `omni_audio_im2col` now copies eight channels of a tap a
thread as one 16-byte load and store (480 channels, the input strides and
the 4352 columns are whole multiples of eight), and `omni_audio_conv1`,
also one thread per output element before, computes eight channels of a
pixel a thread (the nine mel taps loaded once for the eight, their 72
weights contiguous, one 16-byte store). Each output's arithmetic is
unchanged. `OMNI_ASR_DEBUG=1 omni-asr-run`, 4-bit, three rounds each, tower
GPU ms:

| | main | im2col | im2col and conv1 |
|---|---:|---:|---:|
| 11.6 s file (151 tokens) | 13.1-13.3 | 12.0 | **11.5** |
| 35 s file (454 tokens) | 33.8 | | **28.9** |

With the vectorized im2col dispatched twice the tower took 12.4 ms, so the
im2col now costs about 0.4 ms a 11.6 s file: that is all an implicit GEMM
could still save, and staging gathered A tiles in threadgroup memory needs
a barrier per K step, which round 18 measured at 5-15% of these GEMMs
(conv2 and conv3 are about 2.3 ms of the file), so it was not built. MPP's
`convolution2d` op (MacOSX SDK) takes NHWC activations and HWIO weights
but has no padding, so it would need padded copies and a second weight
layout; not tried.

Bit identity: `test-omni-audio-encoder` prints the same errors as main's to
every digit (random towers, the four mlx-audio parity clips) and the packed
towers stay bit-identical to the lone ones; the 200 utterances one at a
time (`omni-asr-run --sequential`), 4-bit and bf16, plain and with the 0.6B
draft, give main's tokens on all 200 (wall 40.46 → 40.23 s, 35.23 → 34.63,
90.42 → 90.08, 44.50 → 43.62). The draft's tower has the same front end
and gains too.

Over HTTP (`run_benchmark.py asr`, c=1, main and the branch alternating,
two rounds; audio s/s, p50 latency):

| | main | round 20 |
|---|---|---|
| 4-bit, the 200 (default, speculative) | 66.2, 67.0; 147 ms | **67.2, 68.3; 144-145 ms** |
| 4-bit `--no-draft-model` | 58.7, 59.2; 165-166 ms | **59.8, 60.1**; 165 ms |
| 4-bit long-form, speculative | 152.7, 153.1 | 152.9, 156.3 |
| 4-bit long-form, plain | 150.9, 152.5 | **154.3, 156.2** |
| bf16, the 200, speculative | 53.4, 53.2; 187-189 ms | 53.8, 54.2; 186-187 ms |
| bf16 `--no-draft-model` | 27.2, 27.3; 361-362 ms | 27.2, 27.4; 360-361 ms |
| bf16 long-form, speculative | 102.9, 102.5 | **104.9, 105.3** |
| bf16 long-form, plain | 90.8, 90.7 | **92.0, 91.8** |

1-2.5% where the tower is a visible share of a request (4-bit, long-form),
within noise for plain bf16, whose decode dominates. Every transcript is
main's (200 of 200 and 4 of 4 in all 16 runs); WER 2.17% / 1.45% (4-bit)
and 1.98% / 1.24% (bf16). TTS does not run the tower.

**3. The Q4 prefill tile: not kept.** Two quant groups' MMAs issued before
either group's affine correction (the corrections still added in group
order, so the tokens were main's) made the four-simdgroup N128 tile Apple10
prefills with slower: the 4-bit ASR prefill of 166 rows 27.4-27.5 → 31.3-31.5
GPU ms, of 469 rows 62.8 → 69.6-69.9 (with `OMNI_DISABLE_ANE=1` 30.4 →
34.7 and 70.3 → 79.3); a rolling form, the next group's MMA before the
current group's correction, 69.9-70.2. The second partial cooperative tensor
costs more than the overlap gains. The tile policy at these rows (an
environment probe, removed): the N256 tile of eight simdgroups 27.2-31.3 and
62.7-62.9, the N128 tile of eight 29.4-29.6 and 68.0-68.3, against the
policy's 27.4 and 62.6-62.8. Nothing to change, so the Neural Engine split's
crossover (160 rows) stands as measured.

**The README's speech recognition tables**, rerun in one pass of the
branch (each Splash Omni row once, on AC power; audio s/s, with the draft /
`--no-draft-model`): 4-bit one file at a time 68.5 / 60.1 (README 65.1 /
58.7), eight files 167.3 / 166.2 (161.2 / 161.1), long-form 156 / 153 (151 /
151); bf16 54.1 / 27.4 (52.7 / 26.8), 107.6 / 106.3 (103.7 / 102.9), 106 /
92 (101 / 89). These include what rounds 19 and the VoiceDesign work left
since round 18's pass (main measured 66-67 at 4-bit one file at a time
above); the alternating runs are this round's own gain. The result files
are `results/quality/asr_omni-*`. The TTS tables are unchanged.

Gates: `make -j12`, `make omni-tools`, `make test-engine`, `make
test-python` (525 tests), `make check-python`, `make architecture-check`,
and real-weight parity, ASR 4-bit and TTS Base 4-bit and the eight
`tts-prompt` references (Base, VoiceDesign and CustomVoice, 4-bit and bf16)
pass; ASR and TTS Base equal main's to every digit (ASR decode logits
cosine 0.999965, TTS talker 0.999974, predictor 0.999859).

## Long single requests: the fade is the model's

One request of 248 English words (four held-out long prompts joined),
CustomVoice-4bit, speaker `ryan`, about two minutes of audio, grows quieter
toward its end: the speech-active loudness of 5 s windows falls a few dB
over the request, by up to 20 dB in some seeds. Three seeds of each engine
first suggested ours fell faster than mlx-audio's. It does not.

**Numerics at long histories.** `omni-model-parity tts-force`
(DEVELOPMENT.md) teacher-forces one of our 1618-frame generations through
the engine's own step and compares the logits of every frame with
mlx-audio's, run as its generate loop runs (one decode step a frame over a
KV cache) in bf16 and with fp32 activations and KV. Against the fp32
reference the engine's code 0 logits have a mean KL of 1.4-3.0e-4 in every
200-frame bin from 0 to 1617 (top-1 agreement 0.995-1.0), the predictor's
6-10e-4; mlx-audio's own bf16 run is further from the fp32 reference
(3.4-7.1e-4, top-1 0.97-1.0; predictor 1.1-1.8e-3), and its error grows
with the history as slowly as ours. The bf16 checkpoint gives the same
(1.3-3.0e-4 against fp32). The RoPE (fp32 angles, precise sin and cos), the
bf16 KV pages, the long-attention kernel past 16 pages and the trailing
text and `tts_pad` rows all agree with the reference at every position.
The sampler is calibrated: the sampled codes' probability integral
transform under the intended policy is uniform over the 1618 code 0 draws
and 1530 predictor draws, and their mean log-probability matches the
policy's expectation (-0.864 against -0.850 for code 0).

**The penalty window.** mlx-audio penalizes the last 64 code 0 tokens
(`repetition_context_size=64`); the official implementation (qwen_tts,
transformers' `RepetitionPenaltyLogitsProcessor` over `input_ids`, which
start empty when generate takes `inputs_embeds`) penalizes the whole
generated history, as the engine does. Neither window changes the fade.

**Loudness** (`slope.py` in the scratchpad: least-squares slope of the
envelope, and first three windows minus last three; mean ± standard error
over seeds):

| | seeds | slope (dB/min) | drop (dB) | duration (s) |
| --- | ---: | ---: | ---: | ---: |
| Splash Omni, whole history (default) | 16 | -3.65 ± 0.57 | 7.1 ± 1.3 | 122.3 |
| Splash Omni, last 64 | 8 | -3.84 ± 0.65 | 9.2 ± 1.8 | 125.8 |
| mlx-audio, last 64 (its default) | 16 | -4.25 ± 0.84 | 7.5 ± 1.7 | 119.6 |
| mlx-audio, whole history | 8 | -2.54 ± 0.95 | 5.9 ± 1.8 | 121.6 |

Both engines fade alike, with the same spread (mlx-audio's worst seed of
16 drops 24 dB to -49 dBFS, ours 23 dB), so the engine is unchanged. The
text is fed one token a frame and runs out after about 280 of 1600 frames,
and the rest of the request reads it back over an ever longer history,
which is a likely reason; shorter requests (the setup page's paragraph
mode, or mlx-audio's split at newlines) should keep the level, which was
not measured here.

**Chunks that continue the first** (measured after reference continuation,
`reference_speech_id`). The setup page now speaks a long text outside
paragraph mode in chunks of about 25 s cut at sentence ends
(`speechChunks` in gui/static/logic.js: five chunks for this text), the
first alone and the others as its continuations, sent once its answer
starts. Same text, CustomVoice-4bit `ryan`, eight seeds, the chunks joined
with 0.15 s between them:

| | seeds | slope (dB/min) | drop (dB) | duration (s) |
| --- | ---: | ---: | ---: | ---: |
| one request | 8 | -2.91 ± 0.43 | 6.5 ± 1.2 | 129.0 |
| five chunks, continuations of the first | 8 | -0.77 ± 0.16 | 2.5 ± 0.4 | 123.2 |

Qwen3-ASR transcribes 14 of the first 20 chunks (four seeds) exactly and
the other six with one or two word errors (2-5% WER); none repeats or
drops a sentence.

## CustomVoice 4-bit: the whole text in the prompt

On macOS 27.2 (2026-11-09) CustomVoice-4bit's standard-prompt error rates
rose to 2.4-375% (word, English) and 6.7-14.7% (character, Mandarin) at 1,
4 and 8 streams: one prompt looped on "ah" to the frame cap, others dropped
words or came out garbled. Base-4bit (0.45% / 0.30%) and CustomVoice-bf16
were unaffected, the Neural Engine was not the cause (`--disable-ane`: 4.8% /
9.1%), and neither the engine nor the checkpoint had changed since October,
when the same run gave 0.82% / 2.96%, already behind bf16 (0.48% CER).

mlx-audio 0.6 builds the CustomVoice prompt as the official implementation
always does for these checkpoints: the whole text in the prompt (every text
token and `tts_eos` over codec `pad`, `tts_pad` over codec `bos`), each frame
reading `tts_pad`. The engine had kept mlx-audio 0.5's layout, one text token
per frame, under which mlx-audio 0.5.8 itself scored 1.45% / 3.08% at 4-bit
and 4.89% WER at bf16 in October. With the whole text in the prompt the
engine's prompt matches mlx-audio 0.6.5's (tts-prompt parity: cosine
0.99993, same argmax); a second cause then showed: CustomVoice-4bit's
`vivian` takes up to ten frames per text token on long passages, and the
default cap of eight cut them short. The cap is now 16 frames per token.

| CustomVoice-4bit, WER / CER | 1 stream | 4 streams | 8 streams |
| --- | ---: | ---: | ---: |
| One text token per frame, cap 8 | 375% / 6.71% | 2.45% / 7.07% | 2.45% / 14.74% |
| Whole text, cap 8 | 1.81% / 5.86% | 1.00% / 2.54% | 1.09% / 3.99% |
| Whole text, cap 16 | 1.27% / 1.15% | 0.82% / 1.09% | 0.72% / 0.42% |
| mlx-audio 0.6.5 (whole text) | 0.91% / 0.42% | 0.72% / 2.30% | 0.91% / 2.18% |

Speed is unchanged (10.2 / 27.3 / 45.0 audio s/s). Base and VoiceDesign keep
their layout.

## Round 21: a DFlash draft

sglang-omni's native MLX server (commit bcb9aae6, 2026-11-09) speculates
with DFlash drafts: a few Qwen3 layers that read the target's hidden states
and propose a block of tokens in one forward. Two exist for Qwen3-ASR-1.7B:
`wilsonzheng0327/Qwen3-ASR-1.7B-dflash` (Speculators format, block 8, a
causal block, reads target layers 1, 13 and 24, its own head over 48,000
tokens with a `d2t` map) and `jburtoft/Qwen3-ASR-1.7B-DFlash` (z-lab format,
block 16, a bidirectional block, reads layers 1, 7, 13, 19 and 25, the
target's head). Both are Apache-2.0, 0.55-0.7 GB in bf16.

**Acceptance first** (`dflash_accept.py` in the scratchpad: the target's
greedy transcript replayed, no timing). Tokens committed per verify round
(accepted proposals plus the target's token), 200 LibriSpeech files:

| Draft, proposals | 4-bit target | bf16 target | 4-bit, Chinese* | bf16, Chinese* |
| --- | ---: | ---: | ---: | ---: |
| jburtoft, 7 | 5.42 | 5.87 | 2.83 | 3.33 |
| jburtoft, 15 | 6.01 | 7.04 | 2.86 | 3.37 |
| wilsonzheng, 7 | 6.20 | 6.32 | 5.98 | 5.94 |
| Qwen3-ASR-0.6B, 5 | 5.21 | 5.29 | 5.46 | 5.55 |
| Qwen3-ASR-0.6B, 7 | 6.54 | 6.66 | 6.93 | 7.06 |

\* 30 generated Mandarin clips (mlx-audio CustomVoice). jburtoft was trained
on little Chinese; it accepts about as much as the 0.6B in English only.

**Engine.** The draft runs on `Qwen3Decoder` (five dense bf16 layers, the
target's head or its own) with three additions: a capture of chosen layers'
outputs in any target forward, a keys-and-values-only pass for context rows,
and visibility tables for a bidirectional block (each block row a one-row
sequence that sees the whole block: the prefill attention kernels as they
are). `omni_argmax_mapped` maps the reduced vocabulary. Design: omni-design.md,
"Speculative decoding with a DFlash draft".

Round costs, M5 Pro, one lane (OMNI_ASR_DEBUG): 7.9 ms of GPU for 7
proposals (draft forward, its head, 8 verified rows on the 4-bit target,
context of 8 rows), against the 0.6B's 13.9 ms for 5. The estimate in
`dflashMilliseconds` (2.2 ms forward + 0.8 head + 0.5 context beside the
target's forward) was within 3%.

**Results** (HTTP, 200 files and the long-form set, audio s/s):

| | without draft | 0.6B draft | jburtoft | wilsonzheng |
| --- | ---: | ---: | ---: | ---: |
| 4-bit, one file | 59.6 | 67.3 | 106 | **119.4** |
| 4-bit, long-form | 157 | 152 | 175 | **201** |
| 4-bit, 8 files | 168.9 | 160.0 | 171 | **184.3** |
| bf16, one file | 27.1 | 53.5 | 69.9 | **79.5** |
| bf16, long-form | 90 | 103 | 124.5 | **150** |
| bf16, 8 files | 103.5 | 104.9 | 130.3 | **145.0** |

Word error rates are unchanged (2.17% and 1.98%). With the wilsonzheng
draft, transcripts equal decoding without it on 194 (4-bit) and 198 (bf16)
of 200 files, a comma or a spelling apart elsewhere, as with the 0.6B (197
and 198). The plain path is bit-identical to before the change (200 of 200
transcripts). Memory: 1.23 GiB with the draft against 1.29 GiB with the
0.6B. The wilsonzheng draft became the default (`DEFAULT_DRAFT_MODEL`).

Open: the context of plain steps (a lane the draft follows appends a row
per step, about 0.4 ms at bf16 weights) could use Q4 copies of fc and the
key-value projections; the mask token's row comes from the target's Q4
table at 4-bit, which may cost the z-lab draft some acceptance (6.01 at
4-bit against 7.04 at bf16).

## Launch re-run (2026-11-10)

Every launch benchmark, rerun on `main` at 524f915 (the DFlash default and
the CustomVoice fix), one engine at a time on AC power; README and
performance.md carry the figures, the result files are
`results/quality/*_2026-11.json`. Splash Omni against round 21's runs: ASR
4-bit 115.6 / 182.0 / 199 audio s/s (one file, 8 files, long-form; 119.4 /
184.3 / 201), bf16 79.9 / 145.9 / 151 (79.5 / 145.0 / 150); TTS Base 4-bit
20 ms first audio, 10.1 / 27.5 / 46.9 audio s/s at 1 / 4 / 8 streams
(October: 20 ms, 10.1 / 27.2 / 46.0), bf16 24 ms, 4.4 / 14.1 / 24.4 (25 ms,
4.4 / 14.1 / 24.2). Packed bf16 weights: TTS 4.44 against 3.88 audio s/s
unpacked (`OMNI_PACKED_BF16=0`, +14%), ASR without a draft 27.2 against 23.0
(+18%).

sglang-omni's native TTS server (`qwen3_tts_server`, commit bcb9aae6) joins
the text-to-speech baselines. It loads only Qwen3-TTS-12Hz-0.6B-CustomVoice
(the 1.7B checkpoints fail at load) and serves one request at a time: 4-bit
93 ms first audio, 4.8 / 5.0 / 5.0 audio s/s at 1 / 4 / 8 streams, bf16
131 ms, 3.3 / 3.3 / 3.3, speaker `vivian`. One of its 4-bit eight-stream
answers looped (English WER 372% in that run); its standard and held-out
runs score as the others do.
