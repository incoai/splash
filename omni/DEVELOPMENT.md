# Development

This document covers building, configuring and validating Splash Omni, and
how its parts fit together. The design rationale is in
[docs/omni-design.md](docs/omni-design.md), the HTTP API in
[docs/api.md](docs/api.md), the page's use in [gui/README.md](gui/README.md)
and the measurements in [docs/performance.md](docs/performance.md). Splash
Omni lives in the `omni/` folder of the Splash repository and builds on its
own: its engine, server and installer started as a copy of Splash's and are
documented here; Splash's are in [the repository's documentation](../docs/README.md).

In short: `install/` (launcher and installer) prepares a checkpoint and
starts `server/audio_server.py`, which speaks HTTP and drives the engine,
`build/splash serve-omni`, over a frame protocol on stdin and stdout. The
engine (`runtime/omni/`, with Splash's Metal backend under `runtime/`) runs
one model. `gui/` is a page that starts servers and calls their API like any
other client; voice chat adds a Splash LLM server beside two speech servers.

Building from source needs Apple silicon with macOS 26.4+, Xcode 26 or
newer, Python 3.12–3.14, and a Metal 4 compiler with `uint4b_format`
tensor support (the affine Q4 tiles need it).

## Build and run

```sh
git clone https://github.com/incoai/splash.git
cd splash/omni
make -j8
./splash-omni serve --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit
```

`--model` names one of the eleven supported checkpoints:
`mlx-community/Qwen3-TTS-12Hz-1.7B-{Base,VoiceDesign,CustomVoice}-{4bit,bf16}`,
`mlx-community/Qwen3-ASR-1.7B-{4bit,bf16}`,
`mlx-community/Qwen3-TTS-12Hz-0.6B-{Base,CustomVoice}-4bit` and
`mlx-community/Qwen3-ASR-0.6B-4bit`; any other repository is
refused before its weights download, with a message naming these (the
installer checks the checkpoint's `config.json` with the engine's
`omni-check`). The first serve sets up
`.venv`, builds the engine, downloads the checkpoint into the Hub cache and
assembles it under `install/models/` (config.json at the root, every
checkpoint file including `speech_tokenizer/` linked under `target/`, the
tokenizer files under `tokenizer/`). For Qwen3-ASR-1.7B it also downloads
the speculative draft, `wilsonzheng0327/Qwen3-ASR-1.7B-dflash`, into the Hub
cache (`upstream.fetch_draft`; skipped with `--no-draft-model` or another
`--draft-model`).

A start asks the Hub only for what the machine lacks (`upstream.prepare`).
An installation that verifies (`assembly.verify`: its links still name the
files, sizes and times it records) starts as it is. Without one, or when
it is damaged, a whole snapshot in the Hub cache is assembled without a
request (`hub.cached_commit`: the commit a `--revision` names, the one the
installation recorded or pinned, or the one `refs/<revision>`, by default
`refs/main`, names; whole means `config.json`, a safetensors file, every
indexed shard, and no dangling link). Only a model in neither place is
listed with one Hub request and downloaded, that model alone; the draft is
fetched only when the cache does not hold it whole where the server finds
it (`upstream.draft_present`). A model once here is never checked for a
newer commit: there is no update path; a newer commit is installed only by
naming it with `--revision`. `--offline` (`HF_HUB_OFFLINE=1`) forbids even
the download, so a missing model is an error. The launcher reads the assembly's
`model.json` and starts `server.audio_server`, the one serving process
(`server/http_server.py` holds the HTTP server, admission and security it
builds on). Ctrl+C stops serving.

To run the server by hand against a checkpoint directory:

```sh
.venv/bin/python -m server.audio_server MODEL_DIR --tokenizer MODEL_DIR \
    --model mlx-community/Qwen3-ASR-1.7B-4bit --binary build/splash --port 8000
```

The server picks the draft itself (`_draft_directory` in
`server/audio_server.py`): a Qwen3-ASR-1.7B checkpoint (its `config.json`
describes a qwen3_asr model with a 2048-wide decoder, 4-bit or bf16) gets
`serve_options.DEFAULT_DRAFT_MODEL` from the Hub cache, downloaded with
`huggingface_hub.snapshot_download` when missing (logging `Downloading ·
draft ...`); a failed download (offline, unreachable) logs a warning and
the server decodes plainly. Speech models and other transcription models
get no draft unless `--draft-model` names one, whose failed download ends
the startup. The engine then loads the draft only if the memory budget
fits it beside the weights, about 4 GiB of workspace and the smallest KV
pool (`Speculative decoding: off; the memory budget ... cannot fit the
draft` otherwise).

The engine process alone speaks the wire protocol on stdin/stdout:
`build/splash serve-omni MODEL_DIR [--lanes N] [--max-memory BYTES]
[--idle-release SECONDS|off] [--disable-ane] [--draft-model DIR] [--draft-tokens K]
[--speculative-lanes N]`; `dev/tools/omni_client.py` drives it
without the server. `build/splash omni-check MODEL_DIR` prints the model's
family and quantization, which the installer uses before it downloads
weights.

### Engine restarts

An engine that fails is restarted at once. If it fails again within 60 s of
starting, the next restart waits 5 s; after a third such failure (failed
restarts count) the server stops restarting it: requests in service get
500 `engine_failed`, and the server exits with status 69 (`EX_UNAVAILABLE`),
so that what supervises it (launchd's `KeepAlive`, a service manager)
decides whether to start it again; set a restart interval there for a cause
the next start would meet again. The message names the crash trace when
`SPLASH_CRASH_TRACE=1` recorded one
([crash traces](#failures-and-crash-traces)). Meanwhile requests get 503
`engine_recovering`, whose message names the last failure. An engine whose
loop leaves a status request unanswered for 30 seconds while requests are
pending is failed and restarted by the same rules (`server/backend.py`, `server/runtime.py`).

### Clients and the console

A client that vanishes without closing its connection (a sleeping laptop, a
dropped network) is found by TCP keepalive, 60 s after its last data, and
its request is cancelled as for a disconnect; a non-streaming request
otherwise reads nothing from its client until it ends. The server's console
lines are written by one thread of their own (`server/diagnostics.py`), so a
console whose reader stops draining, such as a stalled log pipe, delays no
request: past 1,024 waiting lines they are dropped, the next line says how
many, and `/status` counts them under `console.dropped_lines`.

### Shell completion

Source `install/completions/splash-omni.bash` for Bash,
`install/completions/_splash-omni` for Zsh after `compinit`, or
`install/completions/splash-omni.fish` for fish. Completion suggests the
commands (`serve`, `ui`), their options and the eleven supported checkpoints
(`install/completions/suggested-models.txt`), without network access.

### The page (`splash-omni ui`)

What the page does for its user is in [gui/README.md](gui/README.md); this
section is how it is built.

`splash-omni ui [--port P] [--no-browser]` (`install/launcher.py` hands it to
`gui/app.py`) serves `gui/static/` (two HTML pages, their CSS and vanilla
JavaScript, no build step) on `127.0.0.1` and a small JSON API beside them.
Like the launcher it needs the standard library alone, so it runs before
`.venv` exists. The backend does setup and lifecycle only: it starts, watches
and stops servers, and never runs a model or calls the engine. Every feature
the page offers (paragraph mode, long-text chunks, speed, live
transcription, voice chat) is page logic over the unchanged HTTP API. `make
architecture-check` holds it to that (`gui/` imports no `server` module but
`serve_options` and `origins`, and nothing under `server/` imports `gui/`).

The JSON API:

- `GET /api/catalog`: the eleven checkpoints and each transcription model's
  default draft, with whether each
  is installed (`install/models/`) or whole in the Hub cache, each
  checkpoint's speech model type (`variant`: `base`, `voice_design`,
  `custom_voice`) and display label, and the model types the page offers
  (`gui/session.py`, `VARIANTS`);
  `GET /api/port?port=N`: the launcher's own port check.
- `POST /api/start` (the setup as JSON) and `POST /api/stop`;
  `GET /api/state?since=SEQ`: phase (`idle`, `preparing`, `loading`,
  `ready`, `stopping`, `failed`), the step and detail parsed from the
  output, each download's bytes, the output lines after `SEQ`, and the
  command run.
- `GET /api/voice/catalog`, `POST /api/voice/start`, `POST /api/voice/stop`
  and `GET /api/voice/state?asr=SEQ&tts=SEQ&llm=SEQ`: the same for voice
  chat's three servers (below), each role's state as `/api/state` gives it.
- One feature's servers at a time: `/api/start` answers 409 while voice
  chat's servers run, and `/api/voice/start` while a text to speech or
  transcription server runs, so no page sends requests to servers it did
  not start. The pages follow the same rule: voice chat is the third task
  of the start page (`/`), which starts its servers and then opens `/voice`;
  `/` opens `/voice` while voice chat runs, and `/voice` returns to `/`
  when its servers do not run. The run view shows only the tabs its
  server's task serves (`runTabs` in `logic.js`).
- `/proxy/<path>`: forwards a request unchanged (method, path, query,
  body, headers but Host, Origin, Referer and Cookie) to the base URL in
  its `X-Splash-Upstream` header and streams the answer back, so the page
  can call servers that do not allow its origin (CORS) and read their
  `X-Splash-*` headers. The page can also call directly; the servers the
  page starts admit the page's origins (`--allowed-origin`).

The API refuses a Host other than loopback (DNS rebinding) and an Origin
other than the page's own. `gui/session.py` builds the start, deciding
what is here with the installer's own functions (`upstream.present`,
`upstream.draft_present`), so the page and `splash-omni serve` cannot disagree:
a checkpoint that is present starts with `splash-omni serve --offline`, which
the installer starts from without a Hub request (and the flag keeps the
server from asking either); a missing draft beside it is fetched alone first
(`serve_options.download_model`, the helper `upstream.fetch_draft` and the
server use); a missing checkpoint is left to `splash-omni serve`, which
downloads it and its draft. The process runs in a session of its own,
and a stop sends SIGINT to the group, as Ctrl+C would. `gui/progress.py`
reads download progress from the Hub cache: the checkpoint's file sizes
from `hub.Repository.resolve` (only for what is missing), against the
blobs and `.incomplete` files in its cache folder.

`gui/voice.py` runs voice chat's servers as a `VoiceStack` of three
`ServerSession`s: Qwen3-ASR and Qwen3-TTS, each a `splash-omni serve` with
few lanes (2 and 4) and the `--max-memory` its model takes there
(`speech_memory_gib`: 5G for 4-bit speech, 7G for bf16; 5G and 7G for
transcription, 8G and 10G with the DFlash draft, so the draft fits), and a
language model served by
Splash: `splash serve --model M --language-only --idle-release off`, with
the model's DFlash2 draft directory as `--draft-model`, `--offline` when
both are in the Hub cache, and a `--max-memory` that leaves the speech
servers room (`default_llm_memory`: the GPU's share of the Mac's memory,
less the two speech budgets and a 1.5 GiB margin; 21G on a 48 GB Mac with
the default 4-bit models and the draft, 24G without it). `splash`
is found as `$SPLASH_BIN`, else as the launcher of the Splash checkout that
holds Splash Omni (`../splash`), else on the PATH; a path given on the page
may also name the runtime of the Splash desktop app
(`Splash.app/Contents/Resources/splash`, whose Python runs its launcher). It
runs without this project's `PYTHONPATH`. A start whose later server fails
stops the ones it started; ports default to 8031-8033, apart from the setup
page's server.

The page's logic is split across three scripts.

- `gui/static/logic.js` holds what needs no DOM, so it is tested alone:
  the voice controls a speech model takes, read from its `/v1/models`
  entry (`voiceControls`: `tts_model_type`, `instructions`, `voices`,
  `references`); the speech request bodies (`speechBody`); paragraph
  mode's availability (`paragraphModeAllowed`: any model whose server
  continues speech) and the cutting of long text into chunks of about
  25 s at sentence ends (`speechChunks`); `TimeStretcher`, the *Speed*
  control's pitch-preserving WSOLA (40 ms frames, a ±10 ms search, Hann
  overlap-add; bypassed at 1×); `SentenceSplitter`, which cuts a streamed
  answer into sentences to speak (abbreviations kept, an early first piece,
  CJK weighted by character); `TimelinePlayer`, the one track a run
  plays: streamed audio is placed on one timeline in order and played with
  Web Audio from any position, so playing, pausing and seeking during
  generation and after it never start a second track; and the live
  transcription's streaming resampler to 16 kHz, energy-based `Segmenter`
  and WAV encoder.
- `gui/static/app.js` is the main page: setup (a task, then its few
  choices: a speech model type, or the assistant's model for voice chat;
  precision, speculation, port, lanes, memory and the other serve options
  under *Advanced settings*; a port nobody chose moves to the next free
  one), the player's progress bar,
  decoding streamed WAV or PCM as it arrives, and `SpeechRun`, which sends a
  run's requests: in paragraph mode or for long text, the first part alone
  and the others, once its `X-Splash-Speech-Id` arrives, as its
  continuations (retried without the reference on a 4xx), each through its
  own `TimeStretcher`; and `LiveSession`, live transcription (microphone
  through a ScriptProcessor, or a file fed at its own pace; each utterance,
  and about once a second the utterance so far, is an ordinary
  `/v1/audio/transcriptions` request with a 16 kHz mono WAV). Stopping the
  server stops the audio playing.
- `gui/static/voice.js` is voice chat: the microphone through the
  `Segmenter` (half duplex unless barge-in is on), each utterance
  transcribed, the answer streamed from Splash's chat completions
  (`reasoning_effort: "none"`), cut by the `SentenceSplitter` and each
  sentence spoken as soon as it is complete onto one `TimelinePlayer`.
  Every sentence continues a voice sample (`prepareVoice`: up to three
  takes of a fixed sample, generated but not played, keeping the first at
  a normal pace, else the fastest), so voice and pace hold across
  sentences and turns; the sample is prepared again when the voice,
  style, language or conversation changes, or when the server has
  forgotten it.

Tests: `dev/tests/gui/test_gui.py` (setup validation, the model types and
labels, the download plan, the command against the launcher's parser, the
session's phases with a stand-in server, download progress, the proxy, the
page's API), `dev/tests/gui/test_voice.py` (the voice stack's commands,
refused setups, Splash discovery,
cached models, partial-start cleanup, the routes and taken ports) and
`dev/tests/gui/test_logic.py`, which runs `logic.js` in JavaScriptCore
(`jsc`, the engine Safari uses; skipped where it is missing): the controls
per model type, the request bodies and paragraph mode, the chunks, the
sentence splitter, the time stretcher, the single-track player against a
recording fake AudioContext, and the resampler, segmenter and WAV encoder.

## Server options

`splash-omni serve` and `server.audio_server` share these options
(`server/serve_options.py`):

| Option | Purpose |
| --- | --- |
| `--revision`, `--offline` | Install a named commit; never contact the Hub (a model already here starts without it either way). |
| `--port`, `--host` | Listen address (default `127.0.0.1:8000`). |
| `--lanes N` | Concurrent requests decoded in lock step (default 8; TTS at most 16, ASR at most 32). |
| `--max-memory 28G` | Cap Metal memory; sizes the KV page pool after the weights. A model that does not fit, at the smallest pool (1 GiB), is refused, with what it takes. |
| `--idle-release 10m` | Accepted for compatibility; the audio engine keeps its weights loaded. |
| `--disable-ane` | The GPU alone. Transcription: by default a 4-bit model's prefill chunks of 160 tokens or more split their FFN with the Neural Engine; speech: by default the codec's convolution stack runs on the Neural Engine (`OMNI_DISABLE_ANE=1` also turns both off, `OMNI_CODEC_ANE=0` the codec's). See [Neural Engine prefill](#neural-engine-prefill) and [Neural Engine codec](#neural-engine-codec). |
| `--draft-model PATH_OR_ID` | Transcription: speculative decoding with this draft model (a directory, or a repository ID, downloaded into the Hugging Face cache if missing). Default: the DFlash draft `wilsonzheng0327/Qwen3-ASR-1.7B-dflash` for Qwen3-ASR-1.7B (4-bit or bf16), none for other models. A DFlash draft (z-lab or Speculators checkpoint format) reads the served model's hidden states and proposes a block of tokens in one forward; a Qwen3-ASR-0.6B checkpoint drafts one token per forward. The served model verifies the proposals in one forward; transcripts stay its greedy ones (a near-tie may flip, as between batch sizes). With the default draft: 4-bit 2.0x audio s/s one request at a time, 1.3x long-form, 1.1x at 8; bf16 2.9x, 1.7x and 1.4x. About 1.2 GiB more memory and 0.7 GB of disk. See [DFlash draft](docs/omni-performance.md#round-21-a-dflash-draft) and, for the 0.6B, [round 7](docs/omni-performance.md#round-7-speculative-asr-with-a-qwen3-asr-06b-draft). |
| `--no-draft-model` | Transcription: decode plainly, without the default draft (no download, about 1.2 GiB less memory). |
| `--draft-tokens K` | Tokens the draft proposes per lane and round, at most: 1-15 (a DFlash draft: its block size less one, 7 for the default draft, each round choosing the count that saves the most; a 0.6B draft: 5 for a 4-bit model, 7 for bf16); fewer when several lanes speculate. |
| `--speculative-lanes N` | Speculate while at most N requests decode, 1-8 (a DFlash draft: 8, where its rounds save time over plain steps; a 0.6B draft: 1 for a 4-bit model, 8 for bf16, where the rounds of the requests decoding pay for their drafts). |
| `--segment-seconds S` | Transcription: audio longer than S seconds (default 30) is cut at quiet points into about one segment per lane, decoded concurrently; 0 sends up to 20 minutes as one request. |
| `--queue-size`, `--request-timeout`, `--max-request-size` | HTTP admission, as in Splash. |
| `--api-key`, `--allowed-host`, `--allowed-origin` | Authentication and origin checks, as in Splash. |
| `--served-model-name` | An extra name the model answers to. |

The options of the Splash LLM engine (`--max-context`, `--kv-format`, the
cache options, `--decode-share`, `--max-image-pixels`, the reasoning and web
UI options) no longer exist; `splash-omni serve` refuses them.

### Network and authentication

The default listener is `127.0.0.1:8000`; `--port` or `SPLASH_PORT` selects
another port (`--port` wins). To accept LAN connections:

```sh
./splash-omni serve --model mlx-community/Qwen3-ASR-1.7B-4bit --host 0.0.0.0 --api-key YOUR_KEY
```

Clients connect to the server's LAN IP. `--host` selects the IPv4 bind
address; `--allowed-host NAME` accepts an additional HTTP Host name, such as
the Mac's `.local` name or a proxy hostname. It does not change the listener
or allowlist client IPs. A request that names the server any other way gets
403 with the `--allowed-host` flag that would accept it; the check keeps out
web pages that rebind a DNS name to this address.

`--api-key KEY` (or `SPLASH_API_KEY` in the server's environment) requires
`Authorization: Bearer KEY` or `x-api-key: KEY` on every request, `/status`
and `/metrics` included; only `/health`, `/ready` and browsers' `OPTIONS`
preflights stay public. Authentication is off by default.

A web page or an application's webview calls the API from another origin,
which its browser names in `Origin`. `--allowed-origin ORIGIN` admits one,
such as `http://localhost:3000` or `tauri://localhost`; repeat it for more,
or pass `'*'` for every origin. The server then answers the preflight and
names the origin in `Access-Control-Allow-Origin`, so the page can read the
`X-Splash-*` headers too. A page of any other origin gets 403, which its
browser hides from the page, so the server prints the refused origin and the
flag that would accept it, once per origin. Origins match exactly: a pattern
such as `http://*.example.com` is refused at startup. With `'*'`, set
`--api-key` too; the server warns at startup without one. The servers
`splash-omni ui` starts admit the page's own origins.

Separate ports allow separate servers. Each sizes its memory from the whole
Mac, as if it ran alone, so give each a `--max-memory` that leaves room for
the others, as voice chat does.

## Model storage

A start downloads into the Hugging Face cache: about 2.2 GB for a 4-bit
Qwen3-TTS checkpoint and 4.2 GB for bf16; 1.5 GB for 4-bit Qwen3-ASR-1.7B
and 3.8 GB for bf16, plus 0.7 GB for the DFlash draft. Splash Omni
keeps no other copy of the weights: an installation links into the cache.
It pins the commits it installs in the cache (`refs/splash`), so pruning the
cache keeps an installed model's files. How a start decides what to
download, and `--revision` and `--offline`, are under
[Build and run](#build-and-run).

To download to another disk, set the cache location before serving:

```sh
HF_HUB_CACHE=/Volumes/Models/huggingface ./splash-omni serve --model mlx-community/Qwen3-ASR-1.7B-4bit
```

`HF_HUB_CACHE` selects the download cache; `HF_HOME` relocates the Hugging
Face home, including its default `hub` cache. Existing downloads are not
moved.

### Files Splash Omni writes

| Path | Holds |
| --- | --- |
| `install/models`, `build/runtime` | The installed models' links into the Hugging Face cache, and the locks of running servers. |
| Hugging Face cache (`HF_HUB_CACHE`, by default `~/.cache/huggingface/hub`) | Model and draft downloads, with the pins under `refs/splash`. |
| `~/Library/Logs/Splash Omni/crash` | [Crash traces](#failures-and-crash-traces), with `SPLASH_CRASH_TRACE=1` only. |

These paths (`install/paths.py`) are under `omni/`, apart from Splash's, so
both run from one checkout.

## API

The endpoints, their fields and errors are in [docs/api.md](docs/api.md).
How the server handles them:

- **Speech.** `AudioFrontend` (`server/audio_frontend.py`) validates a
  request against the model's `ModelProfile` (its `tts_model_type`,
  speakers, languages and the checkpoint's sampling defaults), tokenizes
  the text and the instruction turn
  `<|im_start|>user\n{instructions}<|im_end|>\n` (at most 512 tokens) and
  submits one `SpeechRequest`. Every answer gets a speech id
  (`X-Splash-Speech-Id`); its text tokens and frames (the engine's `Codes`
  event) go into `SpeechReferences`, a 64 MiB LRU. A request naming a
  `reference_speech_id` takes that speech's text and frames, all but its
  last 2 frames, as its reference, waiting for one still running. Each
  answer's PCM passes through `audio_io.SpeechEdges` on its way out: a
  continuation that starts on its reference's last syllable dying away
  (louder than -40 dB) is cut at the first 20 ms window below -50 dB within
  0.3 s and faded in over 5 ms, and every answer's last 30 ms fade out,
  held back until the answer ends. A streamed answer is chunked transfer
  encoding of 24 kHz int16 mono PCM, after a WAV header for `wav`.
- **Transcription.** `audio_io.decode_upload` reads a PCM WAV with the
  standard library and anything else with PyAV (the first audio track,
  resampled to 16 kHz, channels averaged), up to two hours. Audio longer
  than `--segment-seconds` (30) is cut at the quietest 100 ms within 5 s
  before each boundary, into about one segment per lane, decoded
  concurrently and joined; `--segment-seconds 0` sends up to 20 minutes as
  one request. Each request carries the chat prefix and suffix tokens (with
  a forced `language X<asr_text>`) and the samples, and the server parses
  the language and text from the tokens it streams back.

Errors use the OpenAI shape; 400 for bad input, 404 for an unknown
reference, 415 for an unreadable upload, 503 while the engine is
recovering, 500 once it has stopped restarting.

## Internals

### Code layout

- `runtime/omni/`: the engine. `Qwen3Decoder` runs the decoder both models
  share over `KvPages`; `TtsModel` and `TtsEngine` the talker, code
  predictor and codec loop; `AsrEngine` the mel front end
  (`MelSpectrogram`), audio tower, prefill and greedy decode; `OmniLoader`
  and the two checkpoint loaders turn the MLX safetensors into weight
  images; `OmniConfig.mm` checks `config.json`; `OmniProtocol` and
  `OmniServer` the wire protocol and the process loop.
- `runtime/ops/Omni*.{hpp,cpp}` and `runtime/metal/kernels/omni/*.metal`:
  the operators and kernels (dense bf16 GEMM and GEMV, paged decoder
  attention, encoder attention, norms, gathers, sampling, the audio tower's
  convolutions, the codec decoder).
- Reused from Splash: `runtime/metal/` (the Metal backend, command graphs,
  residency), `runtime/ops/Linear` with the affine Q4 tiles,
  `runtime/ops/Normalization`, `runtime/ops/Embedding`, and the weight
  image stack under `runtime/model/`.
- `server/audio_*.py`: the speech and transcription server over
  `server/http_server.py`, `backend.py`, `runtime.py` and `protocol.py`,
  Splash's HTTP scaffolding, engine recovery, subprocess client and frame
  codec; `server/audio_io.py` decodes uploads and smooths the ends of
  speech; `install/`: the launcher and installer; `gui/`: the page
  `splash-omni ui` serves, and voice chat's server stack (`gui/voice.py`).
- `dev/`: tests, benchmarks and tools.

### Decoder and KV cache

Every decode and prefill row of the talker, the code predictor and the ASR
text decoder goes through `Qwen3Decoder::addForward`, which encodes a
whole forward into one command graph: per layer the RMSNorm, the fused QKV
projection (affine Q4 through `ops::Linear`, or bf16 through
`ops::omni::DenseLinear`), attention, the output projection with its
residual, the post-attention norm and the SwiGLU FFN; then the final norm
and the head over the selected rows into fp32 logits. Decode rows run the
Q4 tiles in lanes of 8 rows, so up to 32 requests share one weight read.

KV lives in 32-token pages of 128 KiB (keys and values of one layer, every
KV head, bf16), allocated per sequence as it grows (`KvPagePool`,
`LanePageTables`, `LaneKv`). Prefill attention is an MPP flash-attention
tile over pages; a decode row over at most 8 pages runs
`omni_attention_decode_fused` (q/k norm, RoPE, KV store, attention and
output in one dispatch), longer histories the split kernels.

### TTS step

A tick admits queued requests with one ragged talker prefill (their prompts
assembled on the GPU from the projected text rows and the codec prefix,
`ttsPrompt`: a VoiceDesign or CustomVoice request's instruction turn leads
as text rows, a CustomVoice speaker is one more codec row), then
runs one step for every active lane: the talker decode and code 0 sampling
(suppression of ids 2048..3071 except EOS, repetition penalty over the
lane's code 0 history, EOS masked for the first two frames, temperature,
top-k, top-p) in one GPU command, and the 15 code predictor sub-steps (each
a forward of the 5-layer predictor over its one-page KV, its head and a
sampling) in a second, encoded while the first runs and queued behind it.
A continuation's reference text leads its prompt and the reference's frames
are replayed as prompt rows in the same prefill; its EOS stays masked for
min(40, max(2, 2 × text tokens)) frames, per row, and its codec stream
starts warm from the reference's last frames
([Reference continuation](docs/omni-design.md#reference-continuation)).
Frames due for audio are decoded through the lane's
`CodecStream`, which carries the speech tokenizer decoder's convolution
histories and KV across chunks, so chunks of any size give the same samples
as a whole decode.

### Neural Engine codec

The speech tokenizer decoder's convolution stack (the initial convolution,
the four SnakeBeta / transposed-convolution / residual-unit blocks, the
output SnakeBeta and convolution: about 95% of its FLOPs) runs on the
Neural Engine beside the decode steps (`runtime/ops/OmniCodecAne.cpp`).
It is one MIL program of fp16 constants with a function per chunk size of
the schedule (4, 8, 16 and 25 frames); every causal convolution's history
is an input surface and its update an output, in two halves a lane's
chunks alternate between, so a stream decodes as the GPU's does, up to
fp16. The transposed convolutions run as their two taps of stacked phases
followed by an interleaving `concat`, and the blocks whose output passes
4096 columns run per piece of their input, each piece reading the
histories the piece before it left: the ANE compiler refuses tensors past
32768 columns, and pieces of about 4096 samples are the fastest.

The GPU keeps the decoder's front (RVQ lookup through the ConvNeXt
stages, `CodecDecoder::addFrontStream`): a due chunk's front and the copy
of its rows into the lane's input surface go at the head of the next step
command, which then signals the shared event the chunk's evaluation waits
for. The GPU goes on with the step; the engine waits for the command and
for the ANE's reports together and emits each chunk's audio as it
reports. A lane has one evaluation in flight at most, so its chunks stay
in order. A request's first chunk (4 frames) decodes on the GPU, where
its audio arrives 3-4 ms sooner, and its histories then move to the
lane's surfaces; a chunk size no function holds also runs on the GPU,
and a stream's last chunk is padded up to the next function. An
evaluation that fails, cannot be queued or does not report within 5 s
retires the Neural Engine for the engine's life: the GPU decodes that
chunk's stack from the lane's input and histories
(`CodecDecoder::addStackStream`), and every later chunk.
`--disable-ane`, `OMNI_DISABLE_ANE=1` or `OMNI_CODEC_ANE=0` keep the codec
on the GPU; the engine logs where it runs. The first start compiles the
program (about 15 s); the ANE service keeps the compilation, so later
starts load it in under a second.

### ASR request

The server sends the chat prefix and suffix tokens and the 16 kHz samples.
The engine computes Whisper log-mel features on the CPU (vDSP), runs the
24-layer audio tower (im2col + bf16 GEMM convolutions, fp32 residual,
104-token attention windows), splices its 13 embeddings per second between
the prefix and suffix embeddings, prefills in chunks of up to 2048 rows and
decodes greedily with the other lanes until `<|im_end|>`.

### Neural Engine prefill

The ASR decoder's prefill chunks of 160 rows or more split each layer's
SwiGLU FFN by intermediate channel between the GPU and the Neural Engine
(`runtime/ops/AneFfn.cpp`, ported from Splash). The GPU runs the leading
4608 of the 6144 channels on its prefill kernels: gate and up over views
of their leading rows, down over a view of the leading inputs of its rows
(`Projection::leadingInputs`, the `prefill_linear_q4_*_leading_inputs`
instances). The Neural Engine runs the last 1536 as one W8A8 program
(`runtime/ane/Program.mm`): the GPU rotates each chunk's normalized rows
with a Hadamard transform in blocks of 128 and packs them as channel-major
int8 under per-token scales (`kernels/prefill/ane_ffn.metal`); the ANE
multiplies them by per-row int8 weights, applies the SiLU and the up
product, rotates and quantizes the intermediate rows per token, and
multiplies them by down's int8 rows; the GPU adds that partial down
projection to its own (`ane_ffn_join`). A chunk runs the smallest of the
program's functions (160 to 2048 rows) that holds it. Shared events order
each layer's evaluation between the GPU's packing and the join inside the
one prefill command (`metal::EventStep`, `ane::Handoff`).

The ANE's int8 weights are rotated and quantized from the Q4 planes once,
at startup, into surfaces of every layer's own (280 MiB for the 28
layers), where Splash requantizes them one layer ahead into two staging
sets: the Omni decoder is small enough to keep them, and the prefill
command then spends no GPU time on them. Startup also checks every
function against the GPU alone on two layers (`AneFfn::verify`, about 2%
of what the ANE adds); a check that fails, a Mac without the private ANE
interface (`ane::unavailable`), `--disable-ane` or `OMNI_DISABLE_ANE=1`
leave the GPU alone, as do the bf16 checkpoints and the TTS models. Any
failure of the ANE's work at run time (an evaluation that fails, never
reports within 2 s, or writes values that are not finite) retires the
split: that chunk runs again on the GPU alone, as every later chunk does,
and a warning names why.

The share and the least chunk are fixed, measured with
`dev/benchmarks/omni_ane_ffn_bench.mm` (`build/engine-tests/omni-ane-ffn-bench
build/splash.metallib [--units 2,3,4] [--rows 160,256,2048]`), which times
a command of 28 FFN layers on the GPU alone and split at each share and
chunk size; the numbers are in
[docs/omni-performance.md](docs/omni-performance.md#neural-engine-prefill-split).
A larger share runs faster from 448 rows but its int8 error takes the
real-weight parity below its bound.

### Protocol

Version 11 of Splash's frame protocol; the layouts are in
[docs/omni-design.md](docs/omni-design.md#protocol-version-11) and
`server/audio_protocol.py`. `dev/tests/engine/omni_protocol_test.cpp`
(`make test-engine-cpu`) decodes the speech request the server's golden
test writes, and checks the model types' configurations and prompts.

### Failures and crash traces

Request logs omit bodies; full crash traces require `SPLASH_CRASH_TRACE=1`
and can contain private data (texts and audio). With it, a failed engine's
last frames, at most 512 and 16 MiB, are written to
`~/Library/Logs/Splash Omni/crash/`, which keeps the newest four traces within
64 MiB. A frame over 16 MiB, such as a long upload's samples, is kept only as
a marker with its size and SHA-256, and a trace missing engine input that
way cannot be replayed. `python -m server.crash_trace <trace>`, run from the
checkout root, replays a trace against the engine. `SPLASH_AUDIO_TRACE=1`
and `OMNI_ASR_TRACE=1` print request timings instead ([Validate](#validate)).

## Validate

```sh
make test-engine-cpu      # CPU tests of the kept Splash components
make test-engine-metal    # kernel tests against fp64, the decoder, codec and audio tower
make test-python          # the server, installer and setup page
make check-python         # ruff and the architecture check
```

The Omni tests alone: `make test-omni-kernels test-omni-decoder
test-omni-codec test-omni-audio-encoder test-ane-ffn`. The Neural Engine
split's test (`dev/tests/engine/ane_ffn_test.mm`) runs its kernel checks
under Metal validation and the split and its fault checks without it,
since the validation layer wraps the shared event in one the ANE cannot
share; likewise `omni-codec --ane` runs the Neural Engine codec's tests
(chunks streamed through the stack against the whole GPU decode and
mlx-audio's waveform, every GPU / ANE handoff, the functions' times)
without it. With a checkpoint in the Hub
cache, `make test-omni-parity-real OMNI_KIND=asr|tts|tts-prompt
OMNI_MODEL=DIR OMNI_REFERENCE=FILE` compares real-weight logits with
mlx-audio's (`dev/tools/omni_parity_reference.py` writes the reference); for
ASR it prefills again with the Neural Engine split (unless
`OMNI_DISABLE_ANE=1`). `tts-prompt` takes the prompt mlx-audio itself builds
for a text with `--instruct`, `--speaker` or `--language` (the VoiceDesign
and CustomVoice checkpoints, or Base with a language) and checks the
engine's assembly of it (`ttsPrompt`) through the talker and predictor:

```sh
# in a venv with mlx-audio (dev/benchmarks/audio/requirements-bench.txt)
python dev/tools/omni_parity_reference.py tts-prompt \
    MODEL_DIR /tmp/cv.bin --speaker vivian --instruct "Very happy." --language english
make test-omni-parity-real OMNI_KIND=tts-prompt OMNI_MODEL=MODEL_DIR OMNI_REFERENCE=/tmp/cv.bin
```

`tts-force` checks long generations: it feeds a generation's frames
(`--codes`, u32 [N][16], e.g. the frames an `OMNI_TTS_DEBUG=1` server logs)
back through the engine's own step at one lane, as mlx-audio's generate
loop feeds its own (the trailing text rows, then `tts_pad`), and compares
the talker's code 0 logits of every frame, and the predictor's of all 15
groups every 16 frames, with mlx-audio's, by history length (mean KL,
top-1 agreement); `--float32` makes the reference run with fp32
activations and KV, above bf16 noise. An optional fifth argument of
`omni-model-parity` saves the engine's logits in the reference's format:

```sh
python dev/tools/omni_parity_reference.py tts-force MODEL_DIR /tmp/force.bin \
    --codes codes.npy --text "$(cat long.txt)" --speaker ryan --language english --float32
make test-omni-parity-real OMNI_KIND=tts-force OMNI_MODEL=MODEL_DIR OMNI_REFERENCE=/tmp/force.bin
```

`dev/tools/omni_tts_greedy_reference.py` prints mlx-audio's greedy frames
for the same options (and the instruction's tokens and the speaker's id for
`omni-tts-run --instruction T,T,... --speaker ID`).
The codec and audio tower tests also compare against mlx-audio outputs
when the checkpoints are present.

A reference continuation (docs/omni-design.md) is checked on real weights
by `omni-tts-run METALLIB MODEL_DIR OUT.wav A... --continue B,B,...
--reference-frames K --greedy`: it speaks A then B as one request, then B
continuing A from that request's first K frames, and prints how many of
the continuation's frames equal the whole request's, the same count for
the whole request rerun in a batch of three (the engine's own numerics
horizon, which the first is read against), and the SNR of the codec
stream's start after different contexts against a whole decode.

Tools: `dev/tools/omni_tts_run.mm` and `omni_asr_run.mm` run the engines
without the server and print timings (`OMNI_TTS_DEBUG=1` /
`OMNI_ASR_DEBUG=1` print per-command GPU times, and which prefill chunks
the Neural Engine split);
`dev/benchmarks/omni_step_profile.mm` times the pieces of a TTS step,
`dev/benchmarks/omni_ane_ffn_bench.mm` the prefill FFN split and
`dev/benchmarks/omni_ane_stack_probe.mm` (`build/engine-tests/omni-ane-stack-probe
build/splash.metallib [--frames 4,8,16,25]`) the Neural Engine codec's
functions. `OMNI_TTS_DEBUG=1` also prints each codec evaluation's time
from queue to report.

The engine reads these environment variables, and no others:
`OMNI_DISABLE_ANE=1` (the GPU alone, as `--disable-ane`),
`OMNI_CODEC_ANE=0` (the TTS codec on the GPU), `OMNI_PACKED_BF16=0` (no
packed copy of a bf16 model's decode weights: 2.2-2.5 GiB less memory,
slower one- and two-row steps), `OMNI_TTS_DEBUG=1` and `OMNI_ASR_DEBUG=1`
(timings on stderr) and `OMNI_ASR_TRACE=1` (each request's engine events,
which `dev/tools/omni_request_trace.py` joins with the server's
`SPLASH_AUDIO_TRACE=1` instants). The A/B switches the performance rounds
added were removed once their changes landed
([round 19](docs/omni-performance.md#round-19-cleanup)).

## Benchmarks

`dev/benchmarks/audio/` holds the harness: `datasets.py prepare` fetches
LibriSpeech test-clean and builds the ASR subset and long-form files,
`run_benchmark.py tts|asr --engine {omni,mlx-audio,sglang-omni} ...`
measures time to first audio, latency, real-time factor, throughput at a
concurrency and WER/CER, and `report.py` renders the result JSONs into
`RESULTS.md`. See `run_benchmark.py --help`, the harness's
[README](dev/benchmarks/audio/README.md), and
[running the benchmarks locally](docs/performance.md#run-benchmarks-locally).
