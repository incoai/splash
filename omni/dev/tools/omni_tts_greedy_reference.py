"""Greedy code frames from mlx-audio's Qwen3-TTS for a text, to compare with
the engine's greedy generation (OMNI_TTS_DEBUG=1 omni-tts-run --greedy).
  python dev/tools/omni_tts_greedy_reference.py MODEL_DIR "text" [frames]
      [--instruct TEXT] [--speaker NAME] [--language NAME]
The prompt is mlx-audio's own (_prepare_generation_inputs): an instruction
for a VoiceDesign or CustomVoice model, a CustomVoice model's preset
speaker, a language; it prints the instruction turn's and the speaker's
tokens for omni-tts-run's --instruction and --speaker.
"""

import argparse

import mlx.core as mx
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.tts.utils import load_model

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("text")
parser.add_argument("frames", nargs="?", type=int, default=12)
parser.add_argument("--instruct")
parser.add_argument("--speaker")
parser.add_argument("--language", default="auto")
args = parser.parse_args()
model = load_model(args.model_dir)
talker, cfg = model.talker, model.config
tc = cfg.talker_config
if args.instruct:
    turn = f"<|im_start|>user\n{args.instruct}<|im_end|>\n"
    print("instruction", ",".join(map(str, model.tokenizer.encode(turn))))
if args.speaker:
    print("speaker", tc.spk_id[args.speaker.lower()])
embeds, trailing, pad_embed = model._prepare_generation_inputs(
    args.text, language=args.language, speaker=args.speaker, instruct=args.instruct
)


def cemb(t):
    return talker.model.codec_embedding(mx.array([t]))


cache = [KVCache() for _ in range(len(talker.model.layers))]
cp = talker.code_predictor
code_cache = [KVCache() for _ in range(len(cp.model.layers))]
suppress = [i for i in range(2048, 3072) if i != tc.codec_eos_token_id]
generated = []
x = embeds
for step in range(args.frames):
    hidden = talker.model(inputs_embeds=x, cache=cache)
    logits = talker.codec_head(hidden[:, -1:, :]).astype(mx.float32)
    logits[:, :, suppress] = -1e30
    if step < 2:
        logits[:, :, tc.codec_eos_token_id] = -1e30
    for t in generated:
        v = logits[0, 0, t]
        logits[0, 0, t] = mx.where(v < 0, v * 1.05, v / 1.05)
    c0 = int(mx.argmax(logits[0, 0]))
    if c0 == tc.codec_eos_token_id:
        print("EOS at frame", step)
        break
    for c in code_cache:
        c.keys = None
        c.values = None
        c.offset = 0
    codes = [c0]
    for code_idx in range(15):
        if code_idx == 0:
            code_input = mx.concatenate([hidden[:, -1:, :], cemb([c0])], axis=1)
        else:
            code_input = cp.codec_embedding[code_idx - 1](mx.array([[codes[-1]]]))
        code_logits, code_cache, _ = cp(
            code_input, cache=code_cache, generation_step=code_idx
        )
        codes.append(int(mx.argmax(code_logits[0, -1].astype(mx.float32))))
    print("frame", step, ":", " ".join(map(str, codes)))
    generated.append(c0)
    text_embed = (
        trailing[:, step : step + 1, :] if step < trailing.shape[1] else pad_embed
    )
    codec_embed = cemb([c0])
    for i, c in enumerate(codes[1:]):
        codec_embed = codec_embed + cp.codec_embedding[i](mx.array([[c]]))
    x = text_embed + codec_embed
