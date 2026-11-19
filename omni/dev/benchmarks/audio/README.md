# Audio benchmarks

`run_benchmark.py` drives every engine over the same data (`datasets.py`; set
`OMNI_BENCH_DATA` or pass `--data-dir`) and the same metrics (`metrics.py`);
`report.py results/*.json -o RESULTS.md` renders the tables. Engines: `omni`,
`mlx-audio`, `sglang-omni` (see the docstring of `run_benchmark.py`), and
whisper.cpp below, installed outside the repository.

## whisper.cpp (ASR, ggml on Metal)

```sh
git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp && git checkout v1.9.5
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON && cmake --build build -j --config Release
for m in large-v3-q5_0 large-v3 large-v3-turbo-q5_0 large-v3-turbo; do ./models/download-ggml-model.sh $m; done
export WHISPER_CPP_DIR=$PWD   # optional: WHISPER_CPP_MODELS=<dir with ggml-*.bin>

python run_benchmark.py asr --engine whisper.cpp --model large-v3-q5_0 --set subset --limit 30 \
    --out results/asr_whisper.cpp_large-v3-q5_0_c1.json
python run_benchmark.py asr --engine whisper.cpp --model large-v3-q5_0 --set longform \
    --out results/asr_whisper.cpp_large-v3-q5_0_longform.json
```

The harness launches `whisper-server` once per run (the model loads once), on
`--inference-path /v1/audio/transcriptions`, and sends each file as one request with
`language=en`. Decoding is the server default (greedy, temperature fallback,
timestamps, flash attention). `whisper-server` holds a global mutex around
inference, so requests are serialized and only `--concurrency 1` is accepted.
`--model` without a quantization suffix is the official f16 file.

## Comparability caveats

- Whisper pads every window to 30 s, so short files pay a full-window encoder pass,
  and its sequential long-form decoding can repeat a sentence at a window seam
  (large-v3 inserted 60+ words on `long300_b`, which dominates its long-form WER).
