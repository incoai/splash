# Performance

[Back to Splash Omni](../README.md#performance)

## Launch benchmarks

Measured in November 2026 on an M5 Pro (16-core GPU, 48 GB) over HTTP, one
engine at a time on an otherwise idle machine. Each ratio compares with the
next-fastest engine measured: mlx-audio and sglang-omni for text to speech
(sglang-omni on the smaller 0.6B CustomVoice model, the only one its server
loads); sglang-omni, mlx-audio and whisper.cpp for speech recognition.

| Metric | 4-bit | bf16 |
| --- | ---: | ---: |
| Text to speech · time to first audio | 20 ms (4.6×) | 24 ms (5.2×) |
| Text to speech · speed, one stream | 10.0× real time (2.0×) | 4.4× real time (1.3×) |
| Text to speech · audio s/s, 8 concurrent streams | 46.9 (8.4×) | 24.4 (4.6×) |
| Speech to text · audio s/s, one file at a time | 116 (2.5×) | 79.9 (1.8×) |
| Speech to text · audio s/s, 8 concurrent files | 182 (3.9×) | 146 (3.3×) |
| Speech to text · audio s/s, long recordings | 199 (3.2×) | 151 (3.2×) |

Splash Omni led on every measure, and its lead grew with load: 8.4× the
text-to-speech throughput of mlx-audio at eight streams. Its transcripts and
generated speech are as accurate as the other engines'
([Accuracy](#accuracy)). The [launch post](https://inco.ai/blog/splash-omni/)
summarizes the comparison.

## Measurement details

### Method

Every engine serves the same checkpoints (`mlx-community/Qwen3-TTS-12Hz-1.7B-*`
and `mlx-community/Qwen3-ASR-1.7B-*`, 4-bit and bf16) and is measured by the
same harness, `dev/benchmarks/audio/`, over HTTP or in process for
mlx-audio.

- **Text to speech:** 60 prompts, English and Mandarin, short to long
  (`tts_prompts.json`), at the checkpoints' default sampling (temperature
  0.9, top-k 50). Speed is the real-time factor's inverse: audio seconds per
  second of generation. Concurrent figures are total audio seconds per
  second over all streams.
- **Speech recognition:** 200 LibriSpeech test-clean utterances of 2 to 35
  seconds, sampled evenly across lengths, and four long-form files of 61 to
  318 seconds built from consecutive utterances of one speaker.
- **Baselines:** mlx-audio 0.6.5 and sglang-omni for both tasks; for
  speech recognition also whisper.cpp v1.9.5 with Whisper large-v3-turbo
  (q5_0), a different model of similar accuracy. sglang-omni (commit
  bcb9aae6) runs its native C++ MLX servers. For text to speech,
  `qwen3_tts_server` loads only `Qwen3-TTS-12Hz-0.6B-CustomVoice` (the 1.7B
  checkpoints fail at load), so it runs that smaller model with the speaker
  `vivian`, and serves one request at a time. For speech recognition,
  `qwen3_asr_server` runs with its DFlash draft
  (`wilsonzheng0327/Qwen3-ASR-1.7B-dflash`, `--draft-model-path`): its
  fastest setting at every load, ahead of plain decoding and of
  `--max-batch 8` at 8 concurrent files. vllm-omni runs on CUDA only, so it
  has no figures here.
- **Splash Omni** runs its defaults: speculative transcription with the
  DFlash draft `wilsonzheng0327/Qwen3-ASR-1.7B-dflash`, and the Neural
  Engine for the speech codec and the 4-bit transcription prefill. "Without
  draft" is the same server with `--no-draft-model`.

Every engine was measured on 2026-11-10 against `main` at 524f915, on macOS
27.2 and AC power. The VoiceDesign and CustomVoice speed rows were measured
on 2026-10-21. Results and transcripts are in
`dev/benchmarks/audio/results/quality/` (`*_2026-11.json`); the development
rounds behind them are in [omni-performance.md](omni-performance.md).

### Text to speech

| 4-bit | Splash Omni | mlx-audio | sglang-omni (0.6B CustomVoice) |
| --- | ---: | ---: | ---: |
| Time to first audio, one stream | 20 ms | 95 ms | 93 ms |
| Real-time factor, one stream (lower is better) | 0.100 | 0.209 | 0.204 |
| Audio s/s, 4 concurrent streams | 27.5 | 5.4 | 5.0 |
| Audio s/s, 8 concurrent streams | 46.9 | 5.6 | 5.0 |

| bf16 | Splash Omni | mlx-audio | sglang-omni (0.6B CustomVoice) |
| --- | ---: | ---: | ---: |
| Time to first audio, one stream | 24 ms | 127 ms | 131 ms |
| Real-time factor, one stream (lower is better) | 0.227 | 0.371 | 0.305 |
| Audio s/s, 4 concurrent streams | 14.1 | 4.6 | 3.3 |
| Audio s/s, 8 concurrent streams | 24.4 | 5.3 | 3.3 |

Splash Omni and mlx-audio run Qwen3-TTS-12Hz-1.7B-Base.
mlx-audio's concurrent figures use its continuous batching; sglang-omni
serves one request at a time, so its total stays flat.

The VoiceDesign and CustomVoice models run on the same engine, with the
instruction or speaker in the prompt; nothing was tuned for them.
VoiceDesign used the instruction "A clear, neutral adult voice speaking at a
moderate pace.", and CustomVoice the speaker `vivian` without an
instruction. mlx-audio was measured one stream at a time, since its batched
paths do not take VoiceDesign.

| Text to speech | VoiceDesign: Splash Omni | VoiceDesign: mlx-audio | CustomVoice: Splash Omni | CustomVoice: mlx-audio |
| --- | ---: | ---: | ---: | ---: |
| 4-bit · time to first audio | 20 ms | 94 ms | 21 ms | 83 ms |
| 4-bit · real-time factor | 0.103 | 0.215 | 0.112 | 0.212 |
| 4-bit · audio s/s, 4 / 8 streams | 26.3 / 45.1 | | 23.6 / 40.9 | |
| bf16 · time to first audio | 30 ms | 128 ms | 27 ms | 128 ms |
| bf16 · real-time factor | 0.234 | 0.374 | 0.239 | 0.377 |
| bf16 · audio s/s, 4 / 8 streams | 13.4 / 22.6 | | 13.1 / 24.6 | |

The speaker `vivian` speaks slowly and pauses, making about 1.5 times as
much audio for the same text in both engines, so its long requests decode
over longer histories.

### Speech recognition

| Audio s/s, 4-bit | Splash Omni | Without draft | sglang-omni | mlx-audio | whisper.cpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| One file at a time | 115.6 | 58.7 | 45.8 | 43.1 | 41.6 |
| 8 concurrent files | 182.0 | 169.1 | 47.1 | | |
| Long-form, one file at a time | 199 | 157 | 47.5 | 42.9 | 62.3 |

| Audio s/s, bf16 | Splash Omni | Without draft | sglang-omni | mlx-audio |
| --- | ---: | ---: | ---: | ---: |
| One file at a time | 79.9 | 27.2 | 43.8 | 20.3 |
| 8 concurrent files | 145.9 | 105.0 | 44.8 | |
| Long-form, one file at a time | 151 | 92 | 47.1 | 20.6 |

mlx-audio transcribes one request at a time, and whisper.cpp's server
serializes requests, so neither has a concurrent figure. The draft helps
most where decoding is bound by memory bandwidth: one file at a time runs
2.0 times as fast with it at 4-bit and 2.9 times at bf16. Without its draft,
sglang-omni transcribes 35.2 / 43.5 (`--max-batch 8`) / 34.5 audio s/s at
4-bit and 18.1 / 39.8 / 18.7 at bf16.

### Accuracy

Speech recognition: word error rate against the LibriSpeech references,
after Whisper's normalizer.

| Word error rate | 4-bit, 200 files | 4-bit, long-form | bf16, 200 files | bf16, long-form |
| --- | ---: | ---: | ---: | ---: |
| Splash Omni | 2.17% | 1.51% | 1.98% | 1.24% |
| sglang-omni | 2.08% | 1.51% | 2.00% | 1.24% |
| mlx-audio | 2.12% | 1.51% | 1.98% | 1.24% |
| whisper.cpp (large-v3-turbo q5_0) | 2.32% | 1.67% | | |

Without the draft, Splash Omni's transcripts are identical to mlx-audio's on
194 of 200 files at 4-bit and 196 of 200 at bf16, after normalization. The
draft leaves them as they are: identical to decoding without it on 194 of 200
files at 4-bit and 198 of 200 at bf16, with the same word error rates. The
rest differ by a comma, a capital letter or "Mr." against "Mister": verifying
a block of rows rounds a near-tie differently from one row.

Text to speech: two independent judges, Qwen3-ASR-1.7B (through mlx-audio)
and Whisper large-v3-turbo (through whisper.cpp), transcribe the generated
speech, scored against the input text: word error rate for English,
character error rate for Mandarin. The 24 held-out prompts
(`tts_prompts_heldout.json`) were written after all tuning. Sampling is
random in both engines, so differences of a few tenths of a percent are
noise.

| Error rate, en / zh | Qwen3-ASR judge | Whisper judge | Held-out, Qwen3-ASR judge | Held-out, Whisper judge |
| --- | ---: | ---: | ---: | ---: |
| Splash Omni, 4-bit | 0.27% / 0.36% | 0.36% / 1.57% | 0.28% / 0.00% | 0.28% / 0.88% |
| mlx-audio, 4-bit | 0.54% / 0.30% | 0.91% / 1.57% | 0.28% / 0.00% | 0.28% / 0.88% |
| sglang-omni (0.6B CustomVoice), 4-bit | 0.45% / 0.66% | 0.54% / 2.42% | 0.28% / 0.00% | 0.56% / 1.06% |
| Splash Omni, bf16 | 0.36% / 0.48% | 0.36% / 1.45% | 0.28% / 0.00% | 0.00% / 0.71% |
| mlx-audio, bf16 | 0.63% / 0.24% | 0.54% / 1.51% | 0.28% / 0.18% | 0.28% / 0.53% |
| sglang-omni (0.6B CustomVoice), bf16 | 0.36% / 0.30% | 0.63% / 1.81% | 0.56% / 0.00% | 0.28% / 0.71% |

Splash Omni and mlx-audio make the same amount of speech for the same text:
775 and 760 seconds over the standard prompts at 4-bit, 716 and 721 at bf16
(sglang-omni's `vivian` speaks more slowly: 953 and 887). For VoiceDesign
(October, Qwen3-ASR judge, en / zh), Splash Omni scores 0.72% / 0.60% at
4-bit and 0.27% / 0.48% at bf16, against mlx-audio's 0.27% / 0.36% and
0.72% / 0.30%. CustomVoice-4bit, since its prompt carries the whole text
([omni-performance.md](omni-performance.md#customvoice-4-bit-the-whole-text-in-the-prompt)),
scores 1.27% / 1.15% at one stream, 0.82% / 1.09% at four and 0.72% / 0.42%
at eight, against mlx-audio 0.6.5's 0.91% / 0.42%, 0.72% / 2.30% and
0.91% / 2.18%.

### No caching

Nothing in the server or the engine caches results, KV or audio across
requests, apart from tokenizing the prompt's fixed text once. A freshly
started server transcribes the 200 utterances as fast on its first pass as
on a second (48.8 and 49.3 audio s/s at 4-bit without a draft, on an earlier
build). The 170 utterances never used during development ran at 48.1 audio
s/s against 50.9 for the 30 that were: their speech is denser (2.62 words a
second against 2.57), and the other engines show part of the same gap. Text
to speech on the held-out prompts runs at the same real-time factor as on the
standard prompts. The runs are in
`dev/benchmarks/audio/results/quality/caching-check/`.

### Long requests

A single request of two minutes grows quieter as it goes, by about 3.7 dB a
minute in Splash Omni and 4.3 in mlx-audio over 16 seeds; the fade is the
model's. Sent as chunks of about 25 seconds, each continuing the first, the
same text falls by 0.8 dB a minute, against 2.9 for one request in the same
run ([measurements](omni-performance.md#long-single-requests-the-fade-is-the-models)).

## Run benchmarks locally

The harness drives every engine with the same data and metrics. It needs a
Python 3.12 environment with `dev/benchmarks/audio/requirements-bench.txt`,
plus mlx-audio to measure mlx-audio and the Qwen3-ASR judge of text to
speech.

```bash
cd dev/benchmarks/audio
python datasets.py prepare --data-dir ~/bench-data   # LibriSpeech subset and long-form files

# in another terminal: splash-omni serve --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit
python run_benchmark.py tts --engine omni --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit \
    --out results/tts_omni_4bit_c1.json
python run_benchmark.py tts --engine omni --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit \
    --concurrency 8 --no-wer --out results/tts_omni_4bit_c8.json

# with splash-omni serve --model mlx-community/Qwen3-ASR-1.7B-4bit
python run_benchmark.py asr --engine omni --model mlx-community/Qwen3-ASR-1.7B-4bit \
    --data-dir ~/bench-data --set subset --out results/asr_omni_4bit_c1.json
python run_benchmark.py asr --engine omni --model mlx-community/Qwen3-ASR-1.7B-4bit \
    --data-dir ~/bench-data --set longform --out results/asr_omni_4bit_longform.json

python report.py results/*.json -o RESULTS.md
```

`--engine mlx-audio` runs the same benchmark in process, and
`--engine sglang-omni` and `--engine whisper.cpp` start those servers from
their checkouts (see `run_benchmark.py --help` and
[the harness's README](../dev/benchmarks/audio/README.md)).
`judge_whisper.py RESULT.json` adds the second, Whisper judge to a text-to-speech
result. Run one engine at a time on an otherwise idle Mac, on AC power: a
second process on the GPU, or battery power, changes the numbers.

The engine's own tests and real-weight parity checks are in
[DEVELOPMENT.md](../DEVELOPMENT.md#validate).
