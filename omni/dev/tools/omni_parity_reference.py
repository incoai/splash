"""Reference logits from mlx-audio for the Omni decoder parity test.

Usage (in a venv with mlx-audio):
  python dev/tools/omni_parity_reference.py asr MODEL_DIR OUT.bin
  python dev/tools/omni_parity_reference.py tts MODEL_DIR OUT.bin

asr: runs the Qwen3-ASR text decoder on a fixed token sequence (no audio)
and writes, as little-endian float32: the token count L, the vocabulary V,
then the logits of the last position [V], then the logits of position L//2
[V] (prefix of the same sequence).

tts: runs the Qwen3-TTS talker on a fixed prompt built from text tokens
through text_projection plus codec embeddings, exactly as
dev/tests/engine/omni_model_parity_test.mm assembles it, and writes L, V
(3072), the last-position talker logits [V], then the code predictor's
logits for code group 1 given code 0 = 100 [2048].

  python dev/tools/omni_parity_reference.py tts-prompt MODEL_DIR OUT.bin \
      [--text TEXT] [--instruct TEXT] [--speaker NAME] [--language NAME]

tts-prompt: the prompt mlx-audio itself builds (_prepare_generation_inputs,
as generate_voice_design and generate_custom_voice call it) for a text with
an instruction, a preset speaker and a language, through the talker and the
code predictor as tts does. It writes u32 L, V, the codec language id (0 =
automatic), the speaker's codec id (0 = none), the text and instruction
token counts, the text tokens (the plain text's, as the reference slices
them from its chat template) and the instruction's (its whole user turn),
then the talker logits [V] and the predictor's [2][2048].

  python dev/tools/omni_parity_reference.py tts-force MODEL_DIR OUT.bin \
      --codes CODES.npy [--text TEXT] [--instruct TEXT] [--speaker NAME] \
      [--language NAME] [--frames N] [--predictor-every E] [--float32]

tts-force: teacher forcing over a long generation. The frames of CODES.npy
(u32 [N][16], a generation's codes, e.g. one OMNI_TTS_DEBUG=1 log's) are fed
back as mlx-audio's generate loop feeds its own (each frame's codec rows
plus the next trailing text row, then tts_pad), one decode step at a time
through a KV cache, and the talker's code 0 logits of every frame are
written, with the code predictor's logits of all 15 groups (given the
frame's own codes) every E frames. --float32 runs the reference with fp32
activations and KV (the quantized weights as they are), a reference above
bf16 noise at every position. It writes u32 L, V, language id, speaker, text
count, instruction count, N, E, the text and instruction tokens, the codes
[N][16], the talker logits [N][V] and the predictor's [ceil(N/E)][15][2048]
(f32).
"""

import argparse
import struct
import sys

import mlx.core as mx
import numpy as np

TOKENS = [
    151644,
    8948,
    198,
    2610,
    525,
    264,
    10950,
    17847,
    13,
    151645,
    198,
    151644,
    872,
    198,
    3838,
    374,
    279,
    6722,
    315,
    9625,
    30,
    151645,
    198,
    151644,
    77091,
    198,
    785,
    6722,
    315,
    9625,
    374,
    12095,
    13,
    1084,
    374,
    264,
    6662,
    3283,
    448,
    1657,
    1251,
    13,
]


def write(path, header, arrays):
    with open(path, "wb") as f:
        f.write(struct.pack("<" + "I" * len(header), *header))
        for a in arrays:
            f.write(np.ascontiguousarray(np.array(a, dtype=np.float32)).tobytes())


def asr(model_dir, out):
    from mlx_audio.stt.utils import load_model

    model = load_model(model_dir)
    inner = (
        model.model
        if hasattr(model, "model") and hasattr(model.model, "model")
        else model
    )
    ids = mx.array([TOKENS])
    logits = inner(ids)
    logits = np.array(logits.astype(mx.float32))[0]
    half = len(TOKENS) // 2
    prefix = np.array(inner(mx.array([TOKENS[:half]])).astype(mx.float32))[0]
    write(out, [len(TOKENS), logits.shape[-1]], [logits[-1], prefix[-1]])
    print("asr: wrote", out, "L", len(TOKENS), "V", logits.shape[-1])


def predictor_logits(talker, hidden):
    """The code predictor's logits for code group 1 given code 0 = 100, and
    for group 2 given group 1's argmax."""
    cp = talker.code_predictor
    code_input = mx.concatenate(
        [hidden[:, -1:, :], talker.model.codec_embedding(mx.array([[100]]))], axis=1
    )
    x = cp.small_to_mtp_projection(code_input)
    from mlx_audio.lm.models.cache import KVCache

    code_cache = [KVCache() for _ in cp.model.layers]
    h = cp.model(inputs_embeds=x, cache=code_cache)
    logits0 = np.array(cp.lm_head[0](h[:, -1]).astype(mx.float32))[0]
    c1 = int(np.argmax(logits0))
    x1 = cp.small_to_mtp_projection(cp.codec_embedding[0](mx.array([[c1]])))
    h1 = cp.model(inputs_embeds=x1, cache=code_cache)
    logits1 = np.array(cp.lm_head[1](h1[:, -1]).astype(mx.float32))[0]
    return logits0, logits1


def tts_prompt(model_dir, out, argv):
    from mlx_audio.tts.utils import load_model

    parser = argparse.ArgumentParser(prog="omni_parity_reference.py tts-prompt")
    parser.add_argument("--text", default="Hello, world.")
    parser.add_argument("--instruct")
    parser.add_argument("--speaker")
    parser.add_argument("--language", default="auto")
    args = parser.parse_args(argv)
    model = load_model(model_dir)
    talker, tc, tokenizer = model.talker, model.config.talker_config, model.tokenizer
    embeds, _, _ = model._prepare_generation_inputs(
        args.text,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
    )
    # The tokens the reference embeds (qwen3_tts.py, _prepare_generation_inputs).
    chat = f"<|im_start|>assistant\n{args.text}<|im_end|>\n<|im_start|>assistant\n"
    text = tokenizer.encode(chat)[3:-5]
    instruction = (
        tokenizer.encode(f"<|im_start|>user\n{args.instruct}<|im_end|>\n")
        if args.instruct
        else []
    )
    speaker = tc.spk_id[args.speaker.lower()] if args.speaker else 0
    language = 0
    if args.language.lower() != "auto":
        language = tc.codec_language_id[args.language.lower()]
    dialect = (tc.spk_is_dialect or {}).get((args.speaker or "").lower())
    if args.language.lower() in ("chinese", "auto") and dialect:
        language = tc.codec_language_id[dialect]
    hidden = talker.model(inputs_embeds=embeds)
    logits = np.array(talker.codec_head(hidden).astype(mx.float32))[0]
    head0, head1 = predictor_logits(talker, hidden)
    length = int(embeds.shape[1])
    with open(out, "wb") as f:
        f.write(
            struct.pack(
                "<6I",
                length,
                logits.shape[-1],
                language,
                speaker,
                len(text),
                len(instruction),
            )
        )
        f.write(struct.pack(f"<{len(text) + len(instruction)}I", *text, *instruction))
        for a in (logits[-1], head0, head1):
            f.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    print(
        "tts-prompt: wrote",
        out,
        "L",
        length,
        "text",
        text,
        "instruction",
        instruction,
        "speaker",
        speaker,
        "language",
        language,
    )


def tts_force(model_dir, out, argv):
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_audio.lm.models.cache import KVCache
    from mlx_audio.tts.utils import load_model

    parser = argparse.ArgumentParser(prog="omni_parity_reference.py tts-force")
    parser.add_argument("--codes", required=True)
    parser.add_argument("--text", default="Hello, world.")
    parser.add_argument("--instruct")
    parser.add_argument("--speaker")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--predictor-every", type=int, default=16)
    parser.add_argument("--float32", action="store_true")
    args = parser.parse_args(argv)
    model = load_model(model_dir)
    talker, tc, tokenizer = model.talker, model.config.talker_config, model.tokenizer
    if args.float32:
        params = [
            (k, v.astype(mx.float32) if v.dtype in (mx.bfloat16, mx.float16) else v)
            for k, v in tree_flatten(talker.parameters())
        ]
        talker.update(tree_unflatten(params))
    codes = np.load(args.codes).astype(np.uint32)
    if args.frames:
        codes = codes[: args.frames]
    frames, every = int(codes.shape[0]), args.predictor_every
    embeds, trailing, pad_embed = model._prepare_generation_inputs(
        args.text, language=args.language, speaker=args.speaker, instruct=args.instruct
    )
    dtype = mx.float32 if args.float32 else embeds.dtype
    embeds, trailing, pad_embed = (
        embeds.astype(dtype),
        trailing.astype(dtype),
        pad_embed.astype(dtype),
    )
    chat = f"<|im_start|>assistant\n{args.text}<|im_end|>\n<|im_start|>assistant\n"
    text = tokenizer.encode(chat)[3:-5]
    instruction = (
        tokenizer.encode(f"<|im_start|>user\n{args.instruct}<|im_end|>\n")
        if args.instruct
        else []
    )
    speaker = tc.spk_id[args.speaker.lower()] if args.speaker else 0
    language = 0
    if args.language.lower() != "auto":
        language = tc.codec_language_id[args.language.lower()]
    dialect = (tc.spk_is_dialect or {}).get((args.speaker or "").lower())
    if args.language.lower() in ("chinese", "auto") and dialect:
        language = tc.codec_language_id[dialect]
    cp = talker.code_predictor

    def frame_embed(frame):
        rows = talker.model.codec_embedding(mx.array([[int(frame[0])]]))
        for g in range(15):
            rows = rows + cp.codec_embedding[g](mx.array([[int(frame[g + 1])]]))
        return rows.astype(dtype)

    # The talker, a decode step per frame as mlx-audio's generate runs it.
    cache = [KVCache() for _ in talker.model.layers]
    talker_logits = np.zeros((frames, tc.vocab_size), np.float32)
    hiddens = []
    x = embeds
    for step in range(frames):
        hidden = talker.model(inputs_embeds=x, cache=cache)[:, -1:, :]
        logits = talker.codec_head(hidden)
        mx.eval(logits, hidden)
        talker_logits[step] = np.array(logits.astype(mx.float32))[0, 0]
        if step % every == 0:
            hiddens.append(hidden)
        text_row = (
            trailing[:, step : step + 1, :] if step < trailing.shape[1] else pad_embed
        )
        x = text_row + frame_embed(codes[step])
    # The predictor over each sampled frame's 16 rows at once (its causal
    # prefill equals its sub-steps): [hidden, codec_embedding(c0),
    # codec_embedding[g](c_{g+1}) for g < 14], lm_head[g] at row g + 1.
    picked = list(range(0, frames, every))
    predictor_logits = np.zeros((len(picked), 15, 2048), np.float32)
    for n, step in enumerate(picked):
        frame = codes[step]
        rows = [
            hiddens[n],
            talker.model.codec_embedding(mx.array([[int(frame[0])]])).astype(dtype),
        ]
        rows += [
            cp.codec_embedding[g](mx.array([[int(frame[g + 1])]])).astype(dtype)
            for g in range(14)
        ]
        x = mx.concatenate(rows, axis=1)
        if cp.small_to_mtp_projection is not None:
            x = cp.small_to_mtp_projection(x)
        h = cp.model(x)
        heads = mx.concatenate([cp.lm_head[g](h[:, g + 1]) for g in range(15)])
        predictor_logits[n] = np.array(heads.astype(mx.float32))
    with open(out, "wb") as f:
        f.write(
            struct.pack(
                "<8I",
                int(embeds.shape[1]),
                tc.vocab_size,
                language,
                speaker,
                len(text),
                len(instruction),
                frames,
                every,
            )
        )
        f.write(struct.pack(f"<{len(text) + len(instruction)}I", *text, *instruction))
        f.write(np.ascontiguousarray(codes, dtype="<u4").tobytes())
        f.write(talker_logits.tobytes())
        f.write(predictor_logits.tobytes())
    print(
        "tts-force: wrote",
        out,
        "L",
        int(embeds.shape[1]),
        "frames",
        frames,
        "predictor frames",
        len(picked),
        "float32" if args.float32 else str(dtype),
    )


def tts(model_dir, out):
    from mlx_audio.tts.utils import load_model

    model = load_model(model_dir)
    talker = model.talker
    cfg = model.config
    tc = cfg.talker_config
    # The Base model's non-ICL prompt for "Hello, world." with automatic
    # language and no speaker: role tokens through the text projection; then
    # tts_pad x3 and tts_bos over the codec prefix [nothink, think_bos,
    # think_eos, pad]; then the first text token over codec bos.
    text = [
        151644,
        77091,
        198,
        9707,
        11,
        1879,
        13,
    ]  # "<|im_start|>assistant\nHello, world."
    pad, bos = cfg.tts_pad_token_id, cfg.tts_bos_token_id

    def proj(ids):
        return talker.text_projection(talker.model.text_embedding(mx.array([ids])))

    def cemb(ids):
        return talker.model.codec_embedding(mx.array([ids]))

    role = proj(text[:3])
    prefix = proj([pad, pad, pad, bos]) + cemb(
        [
            tc.codec_nothink_id,
            tc.codec_think_bos_id,
            tc.codec_think_eos_id,
            tc.codec_pad_id,
        ]
    )
    first = proj([text[3]]) + cemb([tc.codec_bos_id])
    embeds = mx.concatenate([role, prefix, first], axis=1)
    hidden = talker.model(inputs_embeds=embeds)
    logits = np.array(talker.codec_head(hidden).astype(mx.float32))[0]
    # Code predictor: [hidden_last, codec_embedding(100)] through
    # small_to_mtp_projection and the predictor; lm_head[0] at the last row,
    # then codec_embedding[0](argmax) through the cached predictor, lm_head[1].
    head0, head1 = predictor_logits(talker, hidden)
    write(
        out,
        [int(embeds.shape[1]), logits.shape[-1]],
        [logits[-1], head0, head1],
    )
    print(
        "tts: wrote",
        out,
        "L",
        embeds.shape[1],
        "V",
        logits.shape[-1],
        "predictor V",
        head0.shape[-1],
    )


if __name__ == "__main__":
    kind, model_dir, out = sys.argv[1:4]
    if kind == "tts-prompt":
        tts_prompt(model_dir, out, sys.argv[4:])
    elif kind == "tts-force":
        tts_force(model_dir, out, sys.argv[4:])
    else:
        {"asr": asr, "tts": tts}[kind](model_dir, out)
