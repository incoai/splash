# API

[Back to Splash Omni](../README.md#use-the-api)

A Splash Omni server serves one model. A text-to-speech model answers
`POST /v1/audio/speech`; Qwen3-ASR answers `POST /v1/audio/transcriptions`.
Both follow OpenAI's audio API, so OpenAI clients work by pointing their base
URL at the server. Every server also answers `GET /health`, `/ready`,
`/status`, `/metrics` (Prometheus), `/v1/models` and `/v1/audio/voices`.

## Text to speech

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
       "input": "Hello, this is a test of the new speech engine.",
       "language": "english", "stream": true}' \
  -o hello.wav
```

| Field | Meaning |
| --- | --- |
| `model`, `input` | The served model and the text to speak (at most 4096 tokens). |
| `response_format` | `wav` (default) or `pcm`: 24 kHz mono int16 samples without a header. |
| `stream` | `true` sends the audio as chunked transfer encoding while it is generated, after a WAV header for `wav`. |
| `language` | `english`, `chinese`, `german`, `italian`, `portuguese`, `spanish`, `japanese`, `korean`, `french` or `russian`; absent means automatic. |
| `instructions` | The voice's description (VoiceDesign, required) or a style or emotion (CustomVoice, optional). A Base model refuses it. See [Voice control](#voice-control). |
| `voice` | A CustomVoice model's preset speaker (required). Other models ignore it. |
| `reference_speech_id` | Speak as the continuation of earlier speech. See [Continuing earlier speech](#continuing-earlier-speech). |
| `temperature`, `top_k`, `top_p`, `repetition_penalty` | Sampling. Defaults come from the checkpoint: 0.9, 50, 1.0 and 1.05. Temperature 0 is greedy. |
| `seed` | A fixed seed makes a request repeatable. |
| `max_frames` | A cap on the speech's length, at 12.5 frames per second. The default is 16 frames per text token and at least 100, so a sample that misses its end still stops. |
| `chunk_frames` | Frames per streamed chunk. By default chunks grow from 1 frame to 25, so the first audio arrives early. |

The response headers describe the result: `X-Splash-Frames`,
`X-Splash-Audio-Seconds`, `X-Splash-Finish-Reason`,
`X-Splash-Time-To-First-Audio-Ms` and `X-Splash-Speech-Id`.

### Voice control

The VoiceDesign and CustomVoice models take OpenAI's `instructions` field, a
plain-text description of the voice or of how to speak. Send the same
instructions (VoiceDesign), or the same speaker and instructions
(CustomVoice), with every request to keep the same timbre.

**VoiceDesign** requires `instructions`, which describe the voice:

```bash
./splash-omni serve --model mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
       "input": "Hello, this is a test of the new speech engine.",
       "instructions": "A calm, warm female voice speaking slowly and clearly."}' \
  -o designed.wav
```

**CustomVoice** requires `voice`, one of its preset speakers: `aiden`,
`dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`, `uncle_fu` or
`vivian`, in any case. It takes optional `instructions` for emotion or style.
`dylan` and `eric` speak the Beijing and Sichuan dialects when the language is
automatic or Chinese.

```bash
./splash-omni serve --model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
       "input": "Hello, this is a test of the new speech engine.",
       "voice": "vivian", "language": "english",
       "instructions": "Very happy and excited."}' \
  -o vivian.wav
```

The **Base** model has no voice control: each request picks its own voice.
Continuing earlier speech keeps one voice across requests.

### Continuing earlier speech

Separate requests each pick their own way of speaking, so the same speaker
can read two paragraphs at different pitch and loudness. To keep a long text
in one voice, send it in parts and let each part continue the first.

Every speech answer carries an `X-Splash-Speech-Id` header. A later request
that names it as `reference_speech_id` is spoken as the continuation of that
speech, and its answer holds only the new text's audio. Send the same `voice`
and `instructions` with it. The reference may still be generating: the
request waits for it, so later parts can be sent as soon as the first part's
header arrives.

```bash
curl -sD first.h http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
       "input": "The first paragraph.", "voice": "ryan"}' -o p0.wav
id=$(awk 'tolower($1) == "x-splash-speech-id:" {print $2}' first.h | tr -d '\r')
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
       "input": "The second paragraph.", "voice": "ryan",
       "reference_speech_id": "'"$id"'"}' -o p1.wav
```

A continuation keeps its reference's voice, loudness and pace. A reference
spoken slowly, as a very short line often is, makes slow continuations; a
sentence or two of ordinary length makes a good one. A reference holds at
most 40 seconds of speech and 480 text tokens. The server keeps recent speech
in a bounded memory (64 MiB, least recently used first out), and answers an
unknown or expired id with 404 `reference_not_found`.

The model's speech grows quieter over a single request of several minutes.
Long texts keep their level when sent as parts of about 25 seconds, each
continuing the first ([measurements](omni-performance.md#long-single-requests-the-fade-is-the-models)).

## Speech to text

```bash
curl http://127.0.0.1:8001/v1/audio/transcriptions \
  -F file=@hello.wav -F model=mlx-community/Qwen3-ASR-1.7B-4bit \
  -F language=en -F response_format=json
```

| Field | Meaning |
| --- | --- |
| `file` | Any audio file FFmpeg reads, or a video's first audio track: WAV, MP3, M4A, AAC, FLAC, Ogg/Opus, AIFF, MP4, MOV, MKV, WebM and more, up to two hours and 128 MiB. The server decodes it in process with PyAV; no `ffmpeg` is needed on the PATH. |
| `model` | The served model. |
| `language` | An ISO code or name, such as `en` or `Chinese`. Absent means automatic. |
| `response_format` | `json` (default), `text` or `verbose_json`, which adds the language and duration. |

Audio longer than 30 seconds is cut at quiet points into segments transcribed
concurrently (`--segment-seconds`). Decoding is greedy. A file that holds no
audio is refused with 400, and a format PyAV cannot read with 415.

## Models and voices

`GET /v1/models` describes the served model. A speech model reports its
`tts_model_type` (`base`, `voice_design` or `custom_voice`), whether it takes
`instructions` (`unsupported`, `required` or `optional`), a CustomVoice
model's `voices`, and `"references": true` when it can continue earlier
speech. `GET /v1/audio/voices` lists the preset speakers.

## Errors

Errors use OpenAI's shape: 400 for invalid input, 404 for an unknown
reference, 415 for an unreadable upload, 503 while the engine restarts, and
500 once it has stopped restarting.
