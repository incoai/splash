# The Splash Omni page

[Back to Splash Omni](../README.md#in-the-browser)

```bash
./splash-omni ui
```

opens the page in your browser at <http://127.0.0.1:7860>, or the next free
port (`--port P` picks one, `--no-browser` only prints the address). It starts
and stops the servers for you, and every request it sends is an ordinary call
to their API. Ctrl+C in the terminal stops the page and every server it
started.

## Setup

The start page asks one question: what do you want to do?

- **Text to speech.** Then pick the kind of voice: *Preset speakers* (nine
  voices, with an optional mood or style; CustomVoice), *Describe a voice*
  (a voice made from a description; VoiceDesign) or *Natural* (the model
  picks the voice; Base).
- **Transcription.** Files, recordings and live captions, with Qwen3-ASR.
- **Voice chat.** Pick the assistant's language model (below).

The page says whether the model is already on this Mac or how much the
first start downloads, and *Start* does the rest: it downloads only the
chosen checkpoint (a checkpoint already installed or in the Hugging Face
cache is used as it is), starts the server and follows it until it is
ready. Everything else waits under *Advanced settings*: the precision
(4-bit or bf16), speculative decoding for transcription (on by default,
with the 0.7 GB DFlash draft), the port, lanes, memory limit,
Neural Engine and offline start, the exact `splash-omni serve` command, and
*Use a server that is already running*. When nobody chose the port and it
is taken, the page uses the next free one.

*Stop* in the top bar stops the server and returns to the start page.

## One task at a time

The page runs one task's servers at a time, so it never sends requests to
a server it did not start. A text to speech server's page has no
transcription tabs, and a transcription server's page has *Files* and
*Live* (they share its API) but no text to speech. While voice chat runs,
the start page opens it instead; *End voice chat* stops its three servers
and returns. The backend refuses to start one task's servers while
another's run, even when asked directly.

## Text to speech

- **Voice.** With preset speakers, pick the speaker and optionally a
  *Style* (a mood or emotion); with a described voice, a *Description*
  says how it sounds (timbre, rhythm, style). Examples fill either in one
  click, and both are sent as the request's `instructions`. A natural
  (Base) voice takes neither. The controls follow the server's
  `/v1/models`, so they also adapt to another server.
- **Playback.** The audio plays as it streams in, through one player: its bar
  grows with the audio received, and play, pause and seek work during
  generation and after it. The WAV can be downloaded.
- **Speed** (1.5× by default) plays and saves the audio faster at the same
  pitch. The model speaks at about 140 words a minute, and speech sped up
  1.5× or 2× transcribes as accurately as the original.
- **Paragraph mode** (the switch under the text, off by default; a hint
  beside it says so) sends text split at blank lines as one request per
  paragraph, generated in parallel, with the same voice settings and one
  seed. The first paragraph is sent alone; as soon as its answer starts, the
  others are sent at once as its continuations, so they keep its voice, pace
  and loudness, for the Base model too. The page plays the answers in order, the
  first as soon as it arrives.
- **Long text** outside paragraph mode is spoken the same way, in chunks of
  about 25 seconds cut at sentence ends, each continuing the first. A single
  request of several minutes grows quieter as it goes, which is the model's
  own behavior; short continuations keep the level. A switch turns this off.
- **Advanced:** long-text chunks and the request parameters: temperature,
  top-k, top-p, repetition penalty, seed, frame limits, format and
  streaming. Language and speed sit beside *Speak*.

After a run, the page shows the time to first audio, the total time, the
audio's length and its speed (audio seconds per second, and the real-time
factor).

## Transcription

- **Files.** Drop any common audio or video file, or record from the
  microphone. Choose a language or let the model detect it; the response
  format is under *Advanced*.
- **Live** listens to the microphone continuously, or plays a
  file in real time. A voice activity detector cuts the speech at pauses (by
  default 0.7 s ends a line and 15 s cuts a long utterance; both adjustable
  under *Advanced*, with the source). Each utterance is sent as an ordinary transcription
  request, and the utterance so far about once a second for a provisional
  line. The transcript shows each line's length and latency, with Copy and
  Clear.

## Voice chat

*Voice chat* on the start page holds a spoken conversation through three
local servers:

1. Qwen3-ASR transcribes what you say, cut at pauses.
2. A language model served by [Splash](../../README.md)
   writes the answer. The default is Qwen3.6-35B-A3B, and Qwen3.8-27B is
   offered too.
3. Qwen3-TTS speaks the answer sentence by sentence while the rest is still
   being written.

The page runs Splash from the checkout that holds Splash Omni (`../splash`),
or the `splash` named by `$SPLASH_BIN` or under Advanced settings; on the
first start, Splash builds itself as `./splash` does.

*Start voice chat* starts all three servers, downloading what is missing,
and caps the language model's memory so the speech servers fit beside it.
We recommend 48 GB of memory for the default models. *Advanced settings*
holds the speech models, the ports, the language model's memory, the
`splash` path and speculative transcription.

During the chat:

- Every sentence continues a short voice sample prepared silently when the
  chat starts, so the voice and its pace stay the same across sentences and
  turns (*Keep one voice*, under *Advanced*). Pick the language, the speaker,
  a style and the *Speed* (1.5× by default); the assistant's instructions
  are under *Advanced*.
- The microphone is ignored while the assistant answers, unless *Interrupt
  by speaking* is on (use headphones): talking over the answer stops it.
- You can also type a message instead of speaking.
- Every turn shows the transcription time, the language model's first
  token, the first audio, and the time from the end of your turn to the
  first sound.

## Other servers

The page sends its requests through its own transparent proxy, so any
OpenAI-compatible server works and its timing headers are readable. The
*API* panel at the bottom of the run view takes another base URL, model name and API key, so the
same page and requests can measure another framework's server (mlx-audio,
sglang-omni) beside Splash Omni. Options that server does not support have
no effect.

## For developers

The page is plain HTML, CSS and JavaScript in `static/`, with no build step,
served by `app.py` with the Python standard library alone. The design,
routes and tests are in [DEVELOPMENT.md](../DEVELOPMENT.md#the-page-splash-omni-ui).
