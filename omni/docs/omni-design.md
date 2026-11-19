# Splash Omni design

Splash Omni turns the Splash LLM engine into a TTS/ASR engine for Apple
silicon. It serves exactly eight models:

| Model | Kind | Weights |
| --- | --- | --- |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit` | TTS | talker + code predictor affine Q4 g64; embeddings, speech tokenizer, speaker encoder bf16/f32 |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | TTS | all bf16 |
| `mlx-community/Qwen3-ASR-1.7B-4bit` | ASR | text model affine Q4 g64; audio tower bf16 |
| `mlx-community/Qwen3-ASR-1.7B-bf16` | ASR | all bf16 |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit`, `-CustomVoice-4bit` | TTS | as Base-4bit, without the speaker encoder |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`, `-CustomVoice-bf16` | TTS | all bf16, without the speaker encoder |

The system is designed around these models: every kernel instance, buffer
layout and scheduler rule is chosen for their shapes.

## What is kept from Splash

- `runtime/metal/`: `MetalBackend`, `CommandGraph`, `MetalBuffer`, device
  capabilities, residency, the command watchdog.
- `runtime/ops/Linear`: the affine Q4 prefill (32-row MPP tiles) and decode
  (8..32-row tiles, persistent and split-K) kernels, with the residual and
  gate/up epilogues. A decode "lane" of 8 rows becomes 8 request rows.
- `runtime/ops/Normalization` (RMSNorm, fused Q4 sums), `Embedding`
  (`embedding_q4_h2048`), `RowCopy`.
- `runtime/model/`: `SafetensorsCheckpoint`, `WeightSource`, `WeightImages`,
  `WeightStore`, `AffinePreparation` (MLX Q4 to 256-row storage tiles).
- `runtime/engine/`: `Protocol` framing, `FdTransport`, `NativeRuntime`'s
  loop skeleton, status JSON.
- `server/`: the HTTP server, admission, security, errors, metrics,
  latency, the engine subprocess client (`runtime.py`), crash traces.
- `install/`: launcher, Hub access, assembly, model selection.
- `runtime/ane/` and `runtime/ops/AneFfn`: the Neural Engine client and
  the prefill FFN split, for the ASR decoder's affine Q4 prefill only
  (below), with the ANE's int8 weights resident instead of staged per layer,
  and a fixed share instead of the startup calibration.

Dropped: GDN, MoE, Splash's DFlash kernels, GGUF, vision, the Neural
Engine's calibration and idle release, KV disk tiers and prefix cache, tool
calls, judgments, chat UI.

## Shared decoder: Qwen3 1.7B

Both models decode with the same architecture: 28 layers, hidden 2048,
16 query heads over 8 KV heads of 128 dims, per-head q/k RMSNorm before
RoPE, RoPE theta 1e6 over all 128 dims (the models' interleaved M-RoPE with
equal T/H/W positions is identical to 1-D rotate-half RoPE), SwiGLU 6144,
RMSNorm eps 1e-6. The TTS code predictor is the same block at hidden 1024,
intermediate 3072, 5 layers.

`Qwen3Decoder` (runtime/omni) builds the prefill and decode graphs for a
`Qwen3Dims` and a weight set whose projections are either affine Q4
(`ops::Linear`) or bf16 (`ops::DenseLinear`, new). Activations are bf16,
accumulation fp32; the residual stream is bf16 as in Splash.

### KV cache

`KvPagePool`: pages of 32 tokens, one page per (sequence, layer, 32-token
block). A page holds K `[8 heads][32 tokens][128]` bf16 (token-major) and V
`[8 heads][128 dims][32 tokens]` bf16 (dimension-major, as the MPP PV tile
reads it). A sequence owns a page table `[layer][block] -> page index` in a
GPU buffer. INT8 KV is not implemented; the contexts are short (TTS a few
thousand frames at most, ASR at most ~2000 tokens per request).

The code predictor's KV is a single page per (lane, layer): its context is
at most 16 positions and is reset every talker step. It runs through the same
attention kernels, so the "MTP" (code predictor) decode path is the talker's
decode path at a smaller shape, batched over the same lanes, in the same
command graph as the talker step.

### Attention kernels (runtime/metal/kernels/omni)

- `omni_qkv_prepare`: per row, q/k RMSNorm per head, rotate-half RoPE at the
  row's position, writes grouped queries (KV-head-major) and stores K/V into
  the row's page slot.
- `omni_attention_prefill`: causal flash attention over pages, a tile of
  8 query rows x 2 query heads per KV head (M=16, N=32 tokens per page,
  D=128) on MPP matmuls, split over pages with a fixed-order reduce. Rows of
  one dispatch may belong to several sequences (ragged prefill): each row
  carries its sequence index, and the sequence descriptor its page table and
  committed length.
- `omni_attention_decode`: one query row per lane, simdgroup dot-product
  kernel over pages with online softmax, split across pages, then reduce.
  A decode step normally runs it fused with the prepare in one dispatch
  (`omni_attention_decode_fused*`, every page in one threadgroup); a lone
  row past 16 pages runs `omni_attention_decode_long`, the split kernel's
  threadgroups by query head with the prepare and, in the last split to
  finish, the reduce.
- Encoder attention (ASR audio tower, 16 heads x 64, bidirectional within
  104-token windows; speech tokenizer transformer, 16 heads x 64, causal,
  window 72): `omni_attention_dense`, non-paged, over a contiguous bf16 K/V
  slab per sequence, with a window descriptor.

### Dense bf16 projections

`ops::DenseLinear` runs `omni_gemm_bf16_m{32,64}n{64,128}s{4,8}` (MPP tiles,
the vision GEMM generalized: optional bias, epilogues none / GELU-erf / SiLU /
residual add / gate-up SiLU, bf16 or fp32 output) for prefill and steps of
more than sixteen rows, `omni_gemv_bf16_c{1,2}_r{1,2,4}` (one or two outputs
a simdgroup, 32 lanes over K) for one row (a gate/up pair up to three) and
the register MMA tiles `omni_mma_*` for two to sixteen rows; with the packed
copy of the weights (docs/omni-performance.md, round 13) the
`omni_gemv_packed_*` and `omni_mma_packed_*` instances read it instead. K
must be a multiple of 64 and N of 64.

## TTS pipeline

1. Prompt assembly (host builds per-position descriptors, `ttsPrompt`;
   GPU assembles embeddings): role tokens through `text_projection` (fc1,
   SiLU, fc2), codec prefix `[nothink, think_bos, think_eos]` or
   `[think, think_bos, lang, think_eos]`, `[pad, bos]`, then the first text
   token; trailing text rows precomputed per request. The checkpoint's
   `tts_model_type` decides the rest, as mlx-audio's
   `_prepare_generation_inputs` builds it: VoiceDesign leads with the
   instruction's user turn, `<|im_start|>user\n{instruct}<|im_end|>\n`
   tokenized whole, as text rows alone (no codec row); CustomVoice puts its
   preset speaker's codec row (`talker_config.spk_id`) between the codec
   prefix and `[pad, bos]`, under one more `tts_pad`, and takes the
   instruction turn too. A dialect speaker (`spk_is_dialect`) speaks with
   its dialect's language id when the language is automatic or Chinese
   (the server picks it). CustomVoice also reads its whole text in the
   prompt, as the official implementation always does for these
   checkpoints (non-streaming mode) and mlx-audio 0.6 builds it: every text
   token and `tts_eos` over codec `pad`, then `tts_pad` over codec `bos`,
   and each frame then reads `tts_pad`. Fed one text token per frame (the
   other models' layout, mlx-audio 0.5's), its 4-bit checkpoint dropped and
   garbled words and looped (4-bit word / character error rates 2.4-375% /
   6.7-14.7% on the 60 standard prompts, 0.7-1.3% / 0.4-1.1% with the
   whole text). Everything after the prompt is the Base model's.
2. Talker prefill, then per step: talker decode (B rows) -> code0 sampling
   (suppress 2048..3071 except EOS 2150, repetition penalty over past code0,
   temperature, top-k 50, top-p) -> code predictor: 15 sub-steps over
   [hidden, cemb(c0)] then codec_embedding[i](c_i), each sub-step a 5-layer
   decode over a 16-slot KV, sampled with the same policy without penalty
   -> frame of 16 codes -> next talker input = cemb(c0) + sum of the 15
   predictor embeddings + trailing text row (or tts_pad embedding).
3. Speech tokenizer decoder, every `chunk` frames with 25 frames of left
   context (re-decoded and trimmed, as the reference does): RVQ lookup,
   pre_conv, 8-layer transformer (hidden 512, window 72), two ConvNeXt
   upsample stages, the SnakeBeta/convtranspose decoder stack to 24 kHz.
   fp32 activations; the convolutions run an implicit GEMM over half
   operands with fp32 sums, and the convolution stack on the Neural Engine
   by default (DEVELOPMENT.md, "Neural Engine codec").
4. Audio chunks stream to the server as they are decoded.

Voice cloning (reference audio through the speech tokenizer encoder and the
ECAPA speaker encoder) is a later phase; the first phase serves the Base
model's non-ICL prompt, as `mlx-audio` does without a reference.

### Reference continuation

Independent requests each sample their own way of speaking: four
paragraphs of one CustomVoice speaker came out with median F0s 20-70%
apart and loudness 4-7 dB apart (mlx-audio alike). A request may instead
continue earlier speech, its *reference*: that speech's text tokens and
frames (16 codes each), all three model types, with the request's
instruction and preset speaker as usual.

- Prompt: the request's prompt for the text reference ++ request (the
  reference's text leads), then the reference's R frames replayed as the
  steps that sampled them would have fed them: frame j's row is trailing
  text row j (t_{j+1}, then tts_eos, then tts_pad) plus the sum of its 16
  codebook embeddings (`omni_prompt_assemble`'s frame rows). The rows are
  part of the request's one prefill, not R decode steps. The new frames'
  trailing rows go on from row R.
- The repetition penalty's code-0 history starts with the reference's.
- EOS is masked for the first min(40, max(2, 2 × request text tokens)) new
  frames. Having heard a whole utterance, the talker otherwise tends to
  end at once; two frames per token is half of normal speech's four, so a
  short text may still end early and a long one cannot end before it
  begins. The mask is per row (bit r of `OmniSampleParams::mask_eos`), so a
  continuation leaves its batch's other requests alone; the two-frame
  minimum of every request is per row too.
- Only new frames become audio, count toward max_frames and come back in
  `Codes`. Without a max_frames a continuation may run to twice its
  reference's frames per text token if that is more than the usual 16
  (a slow speaker's Mandarin takes about six a token, CustomVoice 4-bit's
  `vivian` up to ten). The codec stream starts as if it had decoded the reference: the
  codec's front (RVQ, pre-conv, transformer, upsampling) takes the last
  100 frames but 8, and those 8 decode whole, audio dropped, before the
  first new chunk. Measured against a whole-sequence decode, the new
  frames' first second is 37-41 dB SNR (25 frames of full context: 13-18
  dB; the conv stack alone converges within 4 frames), at about a fifth of
  a full decode's cost per context frame.
- Limits: 500 reference frames (40 s) and 480 reference text tokens (the
  text rows of an instruction, a reference and a text fit one 2048-row
  projection); the KV tables, the history and the frame buffers are sized
  for them. A request beyond them is refused (`limit_exceeded`).

The official ICL layout (all text first, then codec_bos and the reference
codes) ends at once on CustomVoice and is not used. The server keeps every
finished request's text tokens and frames (64 MiB, LRU) under its
`X-Splash-Speech-Id`; `reference_speech_id` fills a request's reference
from them, dropping the last 2 frames, and waits for a reference that is
still running. The prototype dropped 6 (0.48 s, its closing pause), but
the engine's speech ends soon after its last syllable: 6 frames cut into
it and continuations began by finishing it (a stray syllable at the start
of 6 of 9 Chinese paragraphs); keeping every frame garbled 1 of 9; 2
frames left all 18 Chinese and English paragraphs clean.

Measured over HTTP (4-bit checkpoints, the four held-out medium
paragraphs, paragraph 0 alone then 1-3 at once continuing it, against all
four at once independently; three seeds each; F0 spread is the highest
paragraph's median F0 over the lowest's, loudness range the spread of
their speech levels), mean over the seeds, independent → continued:
CustomVoice (ryan, vivian) F0 51% → 21% and loudness 4.7 → 2.2 dB in
English, 30% → 10% and 5.3 → 2.3 dB in Mandarin; VoiceDesign 40% → 13%,
3.7 → 2.6 dB and 16% → 8%, 4.8 → 1.8 dB; Base 46% → 10%, 9.0 → 1.4 dB and
48% → 18%, 12.1 → 2.2 dB. Qwen3-ASR transcribed every English
continuation exactly and none repeated paragraph 0; 2 of 27 Mandarin
continuations failed as independent requests also do (one ran long and
was cut at its frame limit, one looped; an independent paragraph was cut
as well).

## ASR pipeline

1. Server decodes the upload to 16 kHz mono f32 and sends samples.
2. Engine: Whisper log-mel (CPU, vDSP): 400-point periodic Hann, hop 160,
   128 Slaney mels to 8 kHz, log10, clamp to max-8, (x+4)/4.
3. Audio tower on GPU, per 100-frame chunk: three 3x3 stride-2 conv2d (480
   channels, GELU) -> conv_out 7680->1024 -> fixed sinusoid positions 0..12
   per chunk -> 24 pre-LayerNorm layers (biased QKV/out, fc1 4096 GELU,
   fc2), attention block-diagonal over 104-token windows -> ln_post, proj1
   GELU, proj2 -> 2048-dim audio embeddings, 13 per second.
4. Prompt `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
   [audio embeddings]<|audio_end|><|im_end|>\n<|im_start|>assistant\n`
   (+ `language X<asr_text>` when forced), prefill, greedy decode to
   `<|im_end|>`/`<|endoftext|>`, tokens streamed to the server, which parses
   `language X<asr_text>`.

### Speculative decoding with a DFlash draft

The default draft (`wilsonzheng0327/Qwen3-ASR-1.7B-dflash`, Speculators
format; the z-lab format of `jburtoft/Qwen3-ASR-1.7B-DFlash` loads too) is
five Qwen3 layers of the target's width. It reads the outputs of target
layers 1, 13 and 24 (the z-lab draft: 1, 7, 13, 19, 25), which every target
forward of a lane the draft follows copies into a capture buffer
(`ForwardStep::capture`, a strided row gather per layer). In the same
command, `fc` and `hidden_norm` project each row's captured outputs and each
draft layer stores their keys and values (k_proj, k_norm, RoPE at the row's
position, v_proj; no input norm) in the draft's own pages
(`Qwen3Decoder::addKeysValues`): its context. The draft follows a lane from
its prompt, so it never prefills or catches up.

A round is one command. (1) Each lane's block `[pending token, mask x 7]`,
embedded with the target's table at the lane's next positions, runs through
the draft's layers: the block's keys and values are stored past the context,
and each row attends to the context and to the block causally (the z-lab
draft bidirectionally: every row is a one-row sequence of the attention's
visibility tables, `ForwardStep::visibility`). The draft's final norm and
its head over 48,000 tokens give each row's argmax, mapped to the target's
token by `d2t` (`omni_argmax_mapped`); the z-lab draft uses the target's
head. (2) The target verifies `[pending, d1..dk]` of every lane, keeping
the layer outputs. (3) The verified rows become context at their positions.
The host accepts the leading proposals that match the target's greedy
tokens and the target's token after them; rows past them are written again
by the next block or verification, so rollback is the committed counts. Each
round picks the proposals per lane (1 to 7, and as few as 32 verify rows
allow) that save the most GPU time per token against plain steps, from the
acceptance the rounds observe (`AsrEngine::dflashRound`).

### Neural Engine prefill (4-bit ASR)

A prefill chunk of 160 to 2048 rows splits each layer's SwiGLU FFN by
intermediate channel: the GPU runs channels 0..4607 on its Q4 prefill
tiles (views of gate's and up's leading rows and of the leading inputs of
down's rows), the Neural Engine channels 4608..6143 as one W8A8 program, a
function per chunk size (160, 192, 256, 320, 384, 448, 512, 640, 768,
1024, 1280, 1536, 2048 rows; a chunk pads up to the next). Inputs and
weights are Hadamard-rotated (blocks of 128 over the hidden inputs, 512
over the intermediate channels) before their per-token and per-row int8
scales. The GPU quantizes every layer's ANE weights from the Q4 planes
once at startup into IOSurfaces (280 MiB), rotates and packs each chunk's
normalized rows per layer, and adds the ANE's partial down projection
(fp16, with its per-token scales applied in fp32) to its own. A shared
event per layer orders the ANE's evaluation after the packing and the join
after the evaluation, all in the chunk's one command; the ANE runs while
the GPU computes its own channels. A failed or late evaluation, or a value
that is not finite, retires the split and the chunk runs again on the GPU
alone. The share (a quarter) and the least chunk (160 rows) were measured
on an M5 Pro (docs/omni-performance.md); a third of the channels is faster
from 448 rows but its int8 error took the real-weight parity's logits below
a cosine of 0.999.

## Engine loop

`OmniEngine` owns a request queue and lanes (default 8 decode lanes). Each
tick builds one command graph: prefill chunks for admitted requests (ragged,
up to 2048 rows) and one decode step for every active lane, then the TTS
codec decode for lanes whose chunk is due. At most two commands are in
flight: a TTS step's predictor sub-steps are built while its talker
forward runs and queued behind it, and an ASR decode step can queue behind
the one before (docs/omni-performance.md rounds 10 and 18). Outputs (audio
chunks, tokens, done) go to the transport as events.

## Protocol (version 11)

The framing is Splash's (`runtime/engine/Protocol.hpp`, `server/protocol.py`):
a 16-byte header `"SPLH"`, u16 version (11), u16 frame type, u64 payload
length; all integers little-endian. Reused frames keep their layouts:
`Cancel` 0x0002 (u64 request id), `StatusRequest` 0x0004 (u64
correlation id), `Ready` 0x0100 (u32 max concurrent requests, u32 max
context tokens, u8 kind: 1 TTS, 2 ASR), `Start` 0x0101 (u64 id, u32 lane,
u32 matched prompt tokens = 0), `Tokens` 0x0102 (u64 id, u32 sequence
offset, u32 count, u32 tokens), `Done` 0x0104 (u64 id, u8 reason 0 stop /
1 length / 2 cancelled, u32 prompt tokens, u32 completion count (ASR:
tokens; TTS: frames), u64 prefill micros, u64 decode micros, u64 wall
micros, u32 logit count = 0), `Error` 0x0105 and `StatusJson` 0x0107 as
before. `PromptProgress` and `MaskRequest`/`MaskResponse` are gone.

New client frames:

- `SpeechRequest` 0x0011, fixed part `struct.Struct("<QIIfIfffIfQIIIIIIII")`
  (84 bytes): u64 request_id; u32 flags (bit 0: a `Codes` event before
  `Done`; the other bits reserved, 0); u32 language_id
  (a codec language id such as 2050 for English, 0 = automatic);
  f32 temperature (0 = greedy); u32 top_k; f32 top_p; f32 repetition_penalty;
  f32 predictor_temperature; u32 predictor_top_k; f32 predictor_top_p;
  u64 seed; u32 max_frames (12.5 per second; 0 = model default 8192);
  u32 chunk_frames (frames per audio chunk emitted while streaming, 0 =
  default 25); u32 text_count; u32 ref_text_count; u32 ref_frame_count;
  u32 speaker_present; u32 instruction_count (at most 512); u32
  preset_speaker (a CustomVoice speaker's codec id, 0 = none). Then u32
  text tokens [text_count] (the plain text, tokenized without special
  tokens; the engine adds the role tokens and the codec prefix), u32
  instruction tokens [instruction_count] (VoiceDesign, CustomVoice: the
  whole user turn `<|im_start|>user\n...<|im_end|>\n`, which leads the
  prompt), u32 reference text tokens [ref_text_count], u32 reference codes
  [ref_frame_count * 16], and f32 speaker embedding [2048] when
  speaker_present. A reference's text and codes (both or neither, at most
  480 tokens and 500 frames) make the request a continuation ("Reference
  continuation" below); a speaker embedding is refused. The engine
  refuses an instruction to a Base model, a VoiceDesign request without
  one, a CustomVoice request without one of its speakers, and a preset
  speaker to the other types. Version 10 added instruction_count,
  preset_speaker and the instruction tokens to version 9's request;
  version 11 the flags' bit 0, the `Codes` event and the references.
- `TranscribeRequest` 0x0012, fixed part `struct.Struct("<QIIIII")`
  (28 bytes): u64 request_id; u32 flags (reserved); u32 max_tokens (0 =
  default 4096); u32 prefix_count; u32 suffix_count; u32 sample_count.
  Then u32 prefix tokens (the chat prefix up to and including
  `<|audio_start|>`), u32 suffix tokens (from `<|audio_end|>` to the end of
  the assistant prefix, plus any forced `language X<asr_text>`), and f32
  samples [sample_count] at 16 kHz mono. The engine inserts the audio
  embeddings between prefix and suffix; audio longer than 20 minutes is
  refused (the server splits long audio).

New engine frames:

- `Audio` 0x0110: u64 request_id, u32 sample_offset, u32 count, then
  int16 PCM samples [count] at 24 kHz mono. Chunks arrive in order; `Done`
  follows the last chunk.
- `Codes` 0x0111: u64 request_id, u32 frame_count, then u32 codes
  [frame_count * 16], the request's generated frames (a continuation's new
  frames, not its reference's), code 0 first in each. Sent once, after the
  last `Audio` and before `Done` (whose completion count it equals), to a
  request whose flags' bit 0 asks for it, unless it was cancelled.

Limits: frames at most 256 MiB; samples at most 20 minutes (19,200,000);
text at most 4096 tokens.

## Server

`python -m server.audio_server`: `POST /v1/audio/speech` (OpenAI shape;
`response_format` wav or pcm; streamed chunked WAV; every answer names its
`X-Splash-Speech-Id`, and `reference_speech_id` continues earlier speech,
"Reference continuation" above), `POST
/v1/audio/transcriptions` (multipart; json, text or verbose_json; any
container PyAV reads), `GET /health`, `/ready`, `/status`, `/metrics`,
`/v1/models`. A continuation's first frames decode the end of its
reference's last syllable before a pause; the server cuts that onset at the
first quiet 20 ms window (when the start is louder than -40 dB) and fades
the last 30 ms of every answer, which may stop on a sound still ringing
(`audio_io.SpeechEdges`).

## Validation

- Kernel tests against fp64 references (`dev/tests/engine/omni_*`).
- Parity tests against `mlx-audio` dumps: talker and code predictor logits
  for a fixed prompt (greedy), speech tokenizer waveform for fixed codes,
  ASR transcript and audio-tower embeddings for fixed audio.
- Benchmarks against `mlx-audio` and `sglang-omni` (`dev/benchmarks/audio`).
  `vllm-omni` is not a baseline: its released engine runs on CUDA only and
  has no Apple silicon path, so it cannot be measured on the same machine.
