# Splash Omni

**Splash's speech engine for Apple silicon, built around the model.**

Splash Omni serves Qwen3-TTS (text to speech) and Qwen3-ASR (speech to text)
on one Mac behind OpenAI-compatible audio endpoints. It applies the approach
of [Splash](../README.md), the LLM engine of this repository, to speech:
Metal kernels sized for the two models, speech that starts streaming within
about 20 ms, continuous batching of concurrent requests, and speculative
decoding for transcription. A built-in page covers speech generation,
transcription and a spoken conversation with a local language model.

## Quick start

Splash Omni needs an Apple M3 or newer on macOS 26.4 or later, Xcode 26 or
newer and Python 3.12–3.14. It lives in this folder of the Splash checkout
and runs from source: the first start sets up its Python environment and
builds the engine.

```bash
git clone https://github.com/incoai/splash.git
cd splash/omni
./splash-omni ui
```

Its build, environment and downloaded models stay in `omni/`, apart from
Splash's own. [Building from source](DEVELOPMENT.md#build-and-run)

## Launch Splash Omni

### In the browser

```bash
./splash-omni ui
```

This is the easiest way to use Splash Omni. The page opens in your browser
at <http://127.0.0.1:7860> (or the next free port). Pick what you want to
do and press *Start*: the page downloads the model the first time, starts
it, and then lets you:

- generate speech from text and hear it as it streams, in a voice you
  describe or one of nine preset speakers;
- transcribe a file, a recording or the microphone live;
- talk with an assistant in *Voice chat*, whose language model is served by
  [Splash](../README.md), started from this checkout.

Every run shows its time to first audio and its speed. Press Ctrl+C in the
terminal to stop the page and the servers it started.
[Using the page](gui/README.md)

### From the command line

For your own applications, start a server for each model, each in its own
terminal. The first start downloads the model; later starts reuse it.

```bash
./splash-omni serve --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit
./splash-omni serve --model mlx-community/Qwen3-ASR-1.7B-4bit --port 8001
```

## Use the API

With the two servers above running, text to speech returns a 24 kHz WAV,
and `"stream": true` sends it as it is generated:

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
       "input": "Hello, this is a test of the new speech engine.",
       "stream": true}' \
  -o hello.wav
```

Speech to text accepts any common audio or video file:

```bash
curl http://127.0.0.1:8001/v1/audio/transcriptions \
  -F file=@hello.wav -F model=mlx-community/Qwen3-ASR-1.7B-4bit
```

The VoiceDesign and CustomVoice models choose their voice from
`instructions` and `voice`. Any speech request can continue earlier speech
(`reference_speech_id`) to keep the same voice across a long text.
[Voice control](docs/api.md#voice-control) ·
[Continuing earlier speech](docs/api.md#continuing-earlier-speech) ·
[API details](docs/api.md)

## Models

| Model | Task |
| --- | --- |
| Qwen3-TTS-12Hz-1.7B-Base | Text to speech in a voice the model picks |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | Text to speech in a voice described in words |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | Text to speech by nine preset speakers, with an optional style |
| Qwen3-ASR-1.7B | Speech to text |

Each model is served from its MLX checkpoint in 4-bit or bf16, such as
`mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit` and
`mlx-community/Qwen3-ASR-1.7B-bf16`. The 4-bit checkpoints are smaller and
faster (text to speech runs about twice as fast); bf16 keeps every weight at
full precision. Both reach nearly the same accuracy. Qwen3-ASR-1.7B decodes speculatively with
the DFlash draft `wilsonzheng0327/Qwen3-ASR-1.7B-dflash`, downloaded
automatically (0.7 GB). Transcripts stay the 1.7B model's own. Splash Omni supports only
these eight checkpoints and refuses others before downloading them.

## Settings

Memory and lanes are sized automatically. To change them, add these to
`splash-omni serve`:

| Option | Purpose |
| --- | --- |
| `--port 8001` | Listen on another port (default: 8000). |
| `--lanes 4` | Requests decoded together (default: 8). |
| `--max-memory 8G` | Cap Metal memory to leave room for other applications. |
| `--no-draft-model` | Transcribe without the draft (1.2 GiB less memory). |
| `--disable-ane` | Run on the GPU alone, without the Neural Engine. |
| `--offline` | Never contact the Hugging Face Hub. |

The server listens on localhost without authentication by default. See
`./splash-omni serve --help` for access control and the other options.
[Server options](DEVELOPMENT.md#server-options)

## Performance

Measured on an M5 Pro (16-core GPU, 48 GB) over HTTP in November 2026, one
engine at a time. Text to speech uses 60 English and Mandarin prompts; speech
recognition uses 200 LibriSpeech utterances and four long recordings of 1 to
5 minutes. Ratios compare Splash Omni with the next-fastest engine. Every
engine serves the same checkpoint except sglang-omni for text to speech: its
native MLX server loads only Qwen3-TTS 0.6B CustomVoice, so it runs that
smaller model (speaker `vivian`). For speech recognition it runs its DFlash
draft, its fastest setting at every load. mlx-audio is 0.6.5; whisper.cpp
runs Whisper large-v3-turbo (q5_0) in place of Qwen3-ASR.

**4-bit**

| Metric | Splash Omni | mlx-audio | sglang-omni | whisper.cpp |
| --- | ---: | ---: | ---: | ---: |
| Text to speech · time to first audio | **20 ms** (4.6×) | 95 ms | 93 ms (0.6B) | |
| Text to speech · speed, one stream | **10.0× real time** (2.0×) | 4.8× | 4.9× (0.6B) | |
| Text to speech · audio s/s, 8 concurrent streams | **46.9** (8.4×) | 5.6 | 5.0 (0.6B) | |
| Speech to text · audio s/s, one file at a time | **116** (2.5×) | 43.1 | 45.8 | 41.6 |
| Speech to text · audio s/s, 8 concurrent files | **182** (3.9×) | — | 47.1 | — |
| Speech to text · audio s/s, long recordings | **199** (3.2×) | 42.9 | 47.5 | 62.3 |
| Speech to text · word error rate | 2.17% | 2.12% | 2.08% | 2.32% |

**bf16**

| Metric | Splash Omni | mlx-audio | sglang-omni |
| --- | ---: | ---: | ---: |
| Text to speech · time to first audio | **24 ms** (5.2×) | 127 ms | 131 ms (0.6B) |
| Text to speech · speed, one stream | **4.4× real time** (1.3×) | 2.7× | 3.3× (0.6B) |
| Text to speech · audio s/s, 8 concurrent streams | **24.4** (4.6×) | 5.3 | 3.3 (0.6B) |
| Speech to text · audio s/s, one file at a time | **79.9** (1.8×) | 20.3 | 43.8 |
| Speech to text · audio s/s, 8 concurrent files | **146** (3.3×) | — | 44.8 |
| Speech to text · audio s/s, long recordings | **151** (3.2×) | 20.6 | 47.1 |
| Speech to text · word error rate | 1.98% | 1.98% | 2.00% |

mlx-audio transcribes one request at a time and whisper.cpp's server
serializes requests, so neither has a concurrent transcription figure (—).
sglang-omni's speech server also takes one request at a time, so its total
stays flat however many streams wait; mlx-audio's concurrent speech uses its
continuous batching. The speed costs no accuracy: generated speech
transcribes back to its text as well as the other engines' does.

[Launch post](https://inco.ai/blog/splash-omni/) ·
[Measurement details](docs/performance.md#measurement-details) ·
[Run benchmarks locally](docs/performance.md#run-benchmarks-locally)

## Design

- **The foundation of [Splash](https://github.com/incoai/splash):** Metal
  command graphs, continuous batching over a paged KV cache, a memory budget
  set at startup, and the OpenAI-compatible HTTP server.
- **Speculative speech recognition:** a DFlash draft reads Qwen3-ASR-1.7B's
  own hidden states and proposes up to seven tokens in one pass, which the
  1.7B model verifies in one pass, so transcripts stay its own.
- **Lossless bf16 weight compression:** each exponent is stored as a 4-bit
  offset from a shared base ([GAAC](https://arxiv.org/abs/2610.38210)
  encoding), so the weights take about four fifths of their bytes and a bf16
  decode step takes up to 18% less time.
- **Long texts through concurrency, in one voice:** a request can continue
  earlier speech (`reference_speech_id`), so the page speaks a long text in
  parts that run together, each keeping the first part's voice.
- **Shape-specialized, automatically generated kernels** for the models'
  exact shapes, with work on the Neural Engine beside the GPU.

[How Splash Omni works](docs/omni-design.md)

## More

- [Documentation](docs/README.md): the API, the page, performance and design.
- [Development](DEVELOPMENT.md): building from source, the code's layout,
  internals, tests and benchmarks.
- [Issues and feedback](https://github.com/incoai/splash/issues)
- [Apache-2.0](../LICENSE), as Splash. See Splash Omni's
  [third-party notices](THIRD_PARTY_NOTICES). Model weights keep their own
  licenses.
