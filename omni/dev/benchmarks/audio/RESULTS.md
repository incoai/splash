# Audio benchmark results

Host: `{"chip": "Apple M5 Pro", "machine": "arm64", "macos": "26.6.2", "mem_bytes": "51539607552", "python": "3.12.13"}`

## TTS

TTFA = time to first audio chunk ('-' = not streamed: continuous batching or an engine without a streaming API; latency is then the full generation); RTF = generation time / audio seconds; throughput = total audio seconds / wall seconds at the given concurrency; WER/CER = transcript of the generated speech (Qwen3-ASR via mlx-audio) vs the input text, normalized.

| Engine | Model | Conc | Mode | N | TTFA p50 / p95 (ms) | Latency p50 / p95 (s) | Audio (s) | RTF median | Throughput (audio s / wall s) | WER en | CER zh | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 1 | none | 60 | 91 / 123 | 2.05 / 8.93 | 777.8 | 0.243 | 3.70 | 0.45% | 0.36% | 0 |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | static | 60 | 197 / 759 | 10.74 / 35.38 | 779.4 | 1.129 | 3.16 | 0.45% | 0.36% | 0 |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | continuous | 60 | - / - | 5.82 / 23.56 | 794.2 | 0.665 | 5.38 | 0.27% | 0.48% | 0 |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 1 | none | 60 | 128 / 144 | 3.15 / 10.37 | 720.4 | 0.383 | 2.61 | 0.54% | 0.30% | 0 |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | static | 60 | 179 / 198 | 8.59 / 28.37 | 699.2 | 1.028 | 3.55 | 0.27% | 0.24% | 0 |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | continuous | 60 | - / - | 6.65 / 25.73 | 713.8 | 0.798 | 4.60 | 0.18% | 0.54% | 0 |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 1 | server | 60 | 21 / 22 | 1.11 / 4.00 | 777.5 | 0.131 | 7.72 | 0.36% | 0.30% | 0 |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | server | 60 | 30 / 51 | 1.23 / 4.28 | 793.8 | 0.149 | 26.39 | 0.54% | 0.24% | 0 |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 8 | server | 60 | 32 / 77 | 1.55 / 4.73 | 790.2 | 0.165 | 43.99 | 0.54% | 0.42% | 0 |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 1 | server | 60 | 33 / 33 | 2.24 / 7.43 | 707.6 | 0.276 | 3.65 | 0.36% | 0.36% | 0 |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | server | 60 | 56 / 91 | 3.07 / 9.95 | 728.7 | 0.369 | 10.47 | 0.72% | 0.36% | 0 |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 8 | server | 60 | 65 / 118 | 3.59 / 12.62 | 740.5 | 0.450 | 16.41 | 0.45% | 0.24% | 0 |

### TTS by prompt class

| Engine | Model | Conc | Class | N | TTFA p50 (ms) | Latency p50 (s) | Audio p50 (s) | RTF median | WER/CER |
|---|---|---|---|---|---|---|---|---|---|
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 1 | en/long | 10 | 92 | 6.20 | 26.16 | 0.241 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 1 | en/medium | 10 | 85 | 1.88 | 8.12 | 0.232 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 1 | en/short | 10 | 81 | 0.65 | 2.76 | 0.238 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 1 | zh/long | 10 | 116 | 8.56 | 29.28 | 0.299 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 1 | zh/medium | 10 | 95 | 2.41 | 9.40 | 0.241 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 1 | zh/short | 10 | 94 | 0.75 | 2.80 | 0.272 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | en/long | 10 | 196 | 33.73 | 27.28 | 1.326 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | en/medium | 10 | 194 | 8.98 | 8.24 | 1.046 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | en/short | 10 | 186 | 2.33 | 2.48 | 1.021 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | zh/long | 10 | 154 | 26.60 | 27.40 | 0.967 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | zh/medium | 10 | 241 | 11.18 | 8.88 | 1.236 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | zh/short | 10 | 229 | 3.40 | 3.00 | 1.157 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | en/long | 10 | - | 20.93 | 25.16 | 0.793 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | en/medium | 10 | - | 5.05 | 7.92 | 0.683 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | en/short | 10 | - | 1.59 | 2.64 | 0.594 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | zh/long | 10 | - | 20.35 | 30.24 | 0.674 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | zh/medium | 10 | - | 6.37 | 10.04 | 0.617 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | zh/short | 10 | - | 1.84 | 2.80 | 0.657 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 1 | en/long | 10 | 128 | 9.82 | 26.20 | 0.375 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 1 | en/medium | 10 | 129 | 3.13 | 8.04 | 0.380 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 1 | en/short | 10 | 131 | 1.00 | 2.44 | 0.408 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 1 | zh/long | 10 | 131 | 9.78 | 25.52 | 0.384 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 1 | zh/medium | 10 | 129 | 3.15 | 8.32 | 0.378 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 1 | zh/short | 10 | 125 | 0.96 | 2.48 | 0.388 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | en/long | 10 | 173 | 25.46 | 23.96 | 1.047 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | en/medium | 10 | 182 | 8.04 | 8.28 | 1.039 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | en/short | 10 | 173 | 2.18 | 2.56 | 0.844 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | zh/long | 10 | 183 | 26.95 | 25.12 | 1.059 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | zh/medium | 10 | 170 | 8.46 | 8.28 | 1.003 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | zh/short | 10 | 182 | 2.65 | 2.48 | 0.993 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | en/long | 10 | - | 22.96 | 24.20 | 0.879 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | en/medium | 10 | - | 7.13 | 8.20 | 0.820 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | en/short | 10 | - | 2.08 | 2.60 | 0.781 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | zh/long | 10 | - | 18.51 | 23.88 | 0.775 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | zh/medium | 10 | - | 6.20 | 8.08 | 0.784 | 0.00% |
| mlx-audio | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | zh/short | 10 | - | 1.97 | 2.32 | 0.794 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 1 | en/long | 10 | 21 | 3.39 | 26.56 | 0.127 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 1 | en/medium | 10 | 21 | 1.09 | 8.40 | 0.130 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 1 | en/short | 10 | 21 | 0.36 | 2.64 | 0.138 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 1 | zh/long | 10 | 22 | 3.67 | 28.72 | 0.128 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 1 | zh/medium | 10 | 21 | 1.16 | 8.88 | 0.131 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 1 | zh/short | 10 | 21 | 0.38 | 2.76 | 0.138 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | en/long | 10 | 30 | 3.92 | 28.04 | 0.140 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | en/medium | 10 | 30 | 1.17 | 7.80 | 0.149 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | en/short | 10 | 32 | 0.50 | 2.80 | 0.177 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | zh/long | 10 | 30 | 3.89 | 27.68 | 0.140 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | zh/medium | 10 | 31 | 1.37 | 9.44 | 0.147 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 4 | zh/short | 10 | 30 | 0.43 | 2.68 | 0.158 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 8 | en/long | 10 | 32 | 4.32 | 27.88 | 0.155 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 8 | en/medium | 10 | 32 | 1.53 | 9.28 | 0.164 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 8 | en/short | 10 | 77 | 0.54 | 2.56 | 0.211 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 8 | zh/long | 10 | 34 | 4.17 | 27.40 | 0.153 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 8 | zh/medium | 10 | 32 | 1.55 | 9.44 | 0.164 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-4bit | 8 | zh/short | 10 | 33 | 0.59 | 3.12 | 0.191 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 1 | en/long | 10 | 33 | 6.80 | 25.00 | 0.272 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 1 | en/medium | 10 | 33 | 2.24 | 8.12 | 0.276 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 1 | en/short | 10 | 33 | 0.68 | 2.32 | 0.291 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 1 | zh/long | 10 | 33 | 6.93 | 25.44 | 0.272 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 1 | zh/medium | 10 | 32 | 2.23 | 8.08 | 0.275 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 1 | zh/short | 10 | 32 | 0.70 | 2.44 | 0.288 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | en/long | 10 | 56 | 8.85 | 24.96 | 0.356 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | en/medium | 10 | 58 | 2.94 | 8.00 | 0.369 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | en/short | 10 | 58 | 1.03 | 2.48 | 0.405 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | zh/long | 10 | 57 | 9.16 | 26.44 | 0.355 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | zh/medium | 10 | 56 | 3.20 | 8.68 | 0.367 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 4 | zh/short | 10 | 56 | 1.01 | 2.48 | 0.399 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 8 | en/long | 10 | 65 | 10.96 | 24.88 | 0.431 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 8 | en/medium | 10 | 67 | 3.59 | 8.00 | 0.448 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 8 | en/short | 10 | 117 | 1.35 | 2.68 | 0.505 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 8 | zh/long | 10 | 64 | 10.96 | 26.68 | 0.426 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 8 | zh/medium | 10 | 65 | 3.71 | 8.32 | 0.448 | 0.00% |
| omni | Qwen3-TTS-12Hz-1.7B-Base-bf16 | 8 | zh/short | 10 | 64 | 1.29 | 2.60 | 0.498 | 0.00% |

## ASR

RTFx = audio seconds / wall seconds; WER on LibriSpeech test-clean references with a Whisper-style normalizer (lowercase, punctuation stripped, numbers as-is), computed with jiwer.

| Engine | Model | Set | Conc | N | Latency p50 / p95 (ms) | Audio (s) | RTFx (per file, median) | Throughput (audio s / wall s) | WER | Errors |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx-audio | Qwen3-ASR-1.7B-4bit | subset | 1 | 30 | 250 / 609 | 340.0 | 36.7 | 38.4 | 2.63% | 0 |
| mlx-audio | Qwen3-ASR-1.7B-4bit | longform | 1 | 4 | 4704 / 8431 | 746.8 | 33.9 | 37.5 | 1.51% | 0 |
| mlx-audio | Qwen3-ASR-1.7B-bf16 | subset | 1 | 30 | 509 / 1266 | 340.0 | 18.3 | 19.3 | 2.74% | 0 |
| mlx-audio | Qwen3-ASR-1.7B-bf16 | longform | 1 | 4 | 9066 / 16813 | 746.8 | 17.5 | 19.2 | 1.24% | 0 |
| omni | Qwen3-ASR-1.7B-4bit | subset | 1 | 30 | 197 / 464 | 340.0 | 49.0 | 51.4 | 2.63% | 0 |
| omni | Qwen3-ASR-1.7B-4bit | longform | 1 | 4 | 1310 / 1909 | 746.8 | 119.0 | 139.3 | 1.51% | 0 |
| omni | Qwen3-ASR-1.7B-4bit | subset | 4 | 30 | 304 / 790 | 340.0 | 29.2 | 117.7 | 2.63% | 0 |
| omni | Qwen3-ASR-1.7B-4bit | subset | 8 | 30 | 498 / 1007 | 340.0 | 17.8 | 151.0 | 2.63% | 0 |
| omni | Qwen3-ASR-1.7B-bf16 | subset | 1 | 30 | 449 / 1066 | 340.0 | 21.4 | 22.3 | 2.74% | 0 |
| omni | Qwen3-ASR-1.7B-bf16 | longform | 1 | 4 | 2826 / 4186 | 746.8 | 53.3 | 63.6 | 1.24% | 0 |
| omni | Qwen3-ASR-1.7B-bf16 | subset | 4 | 30 | 672 / 1730 | 340.0 | 13.1 | 52.6 | 2.74% | 0 |
| omni | Qwen3-ASR-1.7B-bf16 | subset | 8 | 30 | 1006 / 2203 | 340.0 | 8.6 | 69.7 | 2.74% | 0 |
| sglang-omni | Qwen3-ASR-1.7B-4bit | subset | 1 | 30 | 224 / 564 | 340.0 | 40.7 | 44.5 | 2.63% | 0 |
| sglang-omni | Qwen3-ASR-1.7B-4bit | longform | 1 | 4 | 3587 / 6270 | 746.8 | 44.5 | 49.6 | 1.72% | 0 |
| sglang-omni | Qwen3-ASR-1.7B-4bit | subset | 4 | 30 | 639 / 1690 | 340.0 | 14.2 | 57.6 | 2.63% | 0 |
| sglang-omni | Qwen3-ASR-1.7B-4bit | subset | 8 | 30 | 1094 / 2270 | 305.1 | 7.5 | 63.5 | 2.13% | 1 |
| sglang-omni | Qwen3-ASR-1.7B-bf16 | subset | 1 | 30 | 480 / 1258 | 340.0 | 19.4 | 20.4 | 2.74% | 0 |
| sglang-omni | Qwen3-ASR-1.7B-bf16 | longform | 1 | 4 | 8551 / 15312 | 746.8 | 18.5 | 20.6 | 1.24% | 0 |
| sglang-omni | Qwen3-ASR-1.7B-bf16 | subset | 4 | 30 | 835 / 2205 | 340.0 | 10.5 | 42.6 | 2.74% | 0 |
| sglang-omni | Qwen3-ASR-1.7B-bf16 | subset | 8 | 30 | 1129 / 2612 | 305.1 | 6.2 | 53.6 | 2.26% | 1 |
| whisper.cpp | whisper-large-v3 f16 (ggml, Metal) | subset | 1 | 30 | 550 / 1221 | 340.0 | 16.5 | 17.9 | 1.94% | 0 |
| whisper.cpp | whisper-large-v3 f16 (ggml, Metal) | longform | 1 | 4 | 16575 / 30686 | 746.8 | 11.2 | 10.8 | 7.48% | 0 |
| whisper.cpp | whisper-large-v3 q5_0 (ggml, Metal) | subset | 1 | 30 | 457 / 934 | 340.0 | 20.2 | 21.7 | 2.17% | 0 |
| whisper.cpp | whisper-large-v3 q5_0 (ggml, Metal) | longform | 1 | 4 | 7627 / 14205 | 746.8 | 20.8 | 22.5 | 5.33% | 0 |
| whisper.cpp | whisper-large-v3-turbo f16 (ggml, Metal) | subset | 1 | 30 | 277 / 496 | 340.0 | 34.2 | 37.5 | 2.63% | 0 |
| whisper.cpp | whisper-large-v3-turbo f16 (ggml, Metal) | longform | 1 | 4 | 3263 / 5744 | 746.8 | 51.8 | 55.1 | 2.64% | 0 |
| whisper.cpp | whisper-large-v3-turbo q5_0 (ggml, Metal) | subset | 1 | 30 | 269 / 466 | 340.0 | 35.0 | 38.8 | 2.74% | 0 |
| whisper.cpp | whisper-large-v3-turbo q5_0 (ggml, Metal) | longform | 1 | 4 | 3102 / 5263 | 746.8 | 57.2 | 59.3 | 1.67% | 0 |

## Engine and model notes

- **whisper.cpp / whisper-large-v3 f16 (ggml, Metal)** (asr): runtime: whisper.cpp v1.9.5 whisper-server, ggml Metal backend; quantization: f16; decoding: server defaults: greedy (best_of 2), temperature 0 with fallback, timestamps on, flash attention on; note: whisper-server serializes requests (global mutex): concurrency 1 only.
- **whisper.cpp / whisper-large-v3 q5_0 (ggml, Metal)** (asr): runtime: whisper.cpp v1.9.5 whisper-server, ggml Metal backend; quantization: q5_0; decoding: server defaults: greedy (best_of 2), temperature 0 with fallback, timestamps on, flash attention on; note: whisper-server serializes requests (global mutex): concurrency 1 only.
- **whisper.cpp / whisper-large-v3-turbo f16 (ggml, Metal)** (asr): runtime: whisper.cpp v1.9.5 whisper-server, ggml Metal backend; quantization: f16; decoding: server defaults: greedy (best_of 2), temperature 0 with fallback, timestamps on, flash attention on; note: whisper-server serializes requests (global mutex): concurrency 1 only.
- **whisper.cpp / whisper-large-v3-turbo q5_0 (ggml, Metal)** (asr): runtime: whisper.cpp v1.9.5 whisper-server, ggml Metal backend; quantization: q5_0; decoding: server defaults: greedy (best_of 2), temperature 0 with fallback, timestamps on, flash attention on; note: whisper-server serializes requests (global mutex): concurrency 1 only.

## Sources

- `results/asr_mlx-audio_4bit_c1.json`: asr mlx-audio mlx-community/Qwen3-ASR-1.7B-4bit c=1 (2026-10-07T18:18:53)
- `results/asr_mlx-audio_4bit_longform.json`: asr mlx-audio mlx-community/Qwen3-ASR-1.7B-4bit c=1 (2026-10-07T18:19:16)
- `results/asr_mlx-audio_bf16_c1.json`: asr mlx-audio mlx-community/Qwen3-ASR-1.7B-bf16 c=1 (2026-10-07T18:31:18)
- `results/asr_mlx-audio_bf16_longform.json`: asr mlx-audio mlx-community/Qwen3-ASR-1.7B-bf16 c=1 (2026-10-07T18:32:02)
- `results/asr_omni_4bit_c1.json`: asr omni mlx-community/Qwen3-ASR-1.7B-4bit c=1 (2026-10-08T12:48:47)
- `results/asr_omni_4bit_longform.json`: asr omni mlx-community/Qwen3-ASR-1.7B-4bit c=1 (2026-10-08T12:48:59)
- `results/asr_omni_4bit_c4.json`: asr omni mlx-community/Qwen3-ASR-1.7B-4bit c=4 (2026-10-08T12:48:50)
- `results/asr_omni_4bit_c8.json`: asr omni mlx-community/Qwen3-ASR-1.7B-4bit c=8 (2026-10-08T12:48:52)
- `results/asr_omni_bf16_c1.json`: asr omni mlx-community/Qwen3-ASR-1.7B-bf16 c=1 (2026-10-08T10:44:05)
- `results/asr_omni_bf16_longform.json`: asr omni mlx-community/Qwen3-ASR-1.7B-bf16 c=1 (2026-10-08T10:44:31)
- `results/asr_omni_bf16_c4.json`: asr omni mlx-community/Qwen3-ASR-1.7B-bf16 c=4 (2026-10-08T10:44:12)
- `results/asr_omni_bf16_c8.json`: asr omni mlx-community/Qwen3-ASR-1.7B-bf16 c=8 (2026-10-08T10:44:17)
- `results/asr_sglang-omni_4bit_c1.json`: asr sglang-omni mlx-community/Qwen3-ASR-1.7B-4bit c=1 (2026-10-07T18:32:23)
- `results/asr_sglang-omni_4bit_longform.json`: asr sglang-omni mlx-community/Qwen3-ASR-1.7B-4bit c=1 (2026-10-07T18:33:23)
- `results/asr_sglang-omni_4bit_c4.json`: asr sglang-omni mlx-community/Qwen3-ASR-1.7B-4bit c=4 (2026-10-07T18:32:40)
- `results/asr_sglang-omni_4bit_c8.json`: asr sglang-omni mlx-community/Qwen3-ASR-1.7B-4bit c=8 (2026-10-07T18:32:56)
- `results/asr_sglang-omni_bf16_c1.json`: asr sglang-omni mlx-community/Qwen3-ASR-1.7B-bf16 c=1 (2026-10-07T18:33:52)
- `results/asr_sglang-omni_bf16_longform.json`: asr sglang-omni mlx-community/Qwen3-ASR-1.7B-bf16 c=1 (2026-10-07T18:35:17)
- `results/asr_sglang-omni_bf16_c4.json`: asr sglang-omni mlx-community/Qwen3-ASR-1.7B-bf16 c=4 (2026-10-07T18:34:11)
- `results/asr_sglang-omni_bf16_c8.json`: asr sglang-omni mlx-community/Qwen3-ASR-1.7B-bf16 c=8 (2026-10-07T18:34:27)
- `results/asr_whisper.cpp_large-v3_c1.json`: asr whisper.cpp large-v3 c=1 (2026-10-08T05:08:34)
- `results/asr_whisper.cpp_large-v3_longform.json`: asr whisper.cpp large-v3 c=1 (2026-10-08T05:09:50)
- `results/asr_whisper.cpp_large-v3-q5_0_c1.json`: asr whisper.cpp large-v3-q5_0 c=1 (2026-10-08T05:07:35)
- `results/asr_whisper.cpp_large-v3-q5_0_longform.json`: asr whisper.cpp large-v3-q5_0 c=1 (2026-10-08T05:08:13)
- `results/asr_whisper.cpp_large-v3-turbo_c1.json`: asr whisper.cpp large-v3-turbo c=1 (2026-10-08T05:10:25)
- `results/asr_whisper.cpp_large-v3-turbo_longform.json`: asr whisper.cpp large-v3-turbo c=1 (2026-10-08T05:10:40)
- `results/asr_whisper.cpp_large-v3-turbo-q5_0_c1.json`: asr whisper.cpp large-v3-turbo-q5_0 c=1 (2026-10-08T05:10:00)
- `results/asr_whisper.cpp_large-v3-turbo-q5_0_longform.json`: asr whisper.cpp large-v3-turbo-q5_0 c=1 (2026-10-08T05:10:15)
- `results/tts_mlx-audio_4bit_c1.json`: tts mlx-audio mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit c=1 (2026-10-07T18:10:48)
- `results/tts_mlx-audio_4bit_c4static.json`: tts mlx-audio mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit c=4 (2026-10-07T18:15:29)
- `results/tts_mlx-audio_4bit_c4cont.json`: tts mlx-audio mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit c=4 (2026-10-07T18:18:21)
- `results/tts_mlx-audio_bf16_c1.json`: tts mlx-audio mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 c=1 (2026-10-07T18:23:55)
- `results/tts_mlx-audio_bf16_c4static.json`: tts mlx-audio mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 c=4 (2026-10-07T18:27:37)
- `results/tts_mlx-audio_bf16_c4cont.json`: tts mlx-audio mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 c=4 (2026-10-07T18:30:38)
- `results/tts_omni_4bit_c1.json`: tts omni mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit c=1 (2026-10-08T12:46:42)
- `results/tts_omni_4bit_c4.json`: tts omni mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit c=4 (2026-10-08T12:47:35)
- `results/tts_omni_4bit_c8.json`: tts omni mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit c=8 (2026-10-08T12:48:15)
- `results/tts_omni_bf16_c1.json`: tts omni mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 c=1 (2026-10-08T10:40:46)
- `results/tts_omni_bf16_c4.json`: tts omni mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 c=4 (2026-10-08T10:42:17)
- `results/tts_omni_bf16_c8.json`: tts omni mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 c=8 (2026-10-08T10:43:24)
