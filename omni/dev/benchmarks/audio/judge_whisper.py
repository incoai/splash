#!/usr/bin/env python3
"""Second judge for TTS quality: Whisper large-v3-turbo (whisper.cpp, f16 on
Metal) transcribes the wavs a TTS result saved, independently of the Qwen3-ASR
judge run_benchmark.py uses.

    python judge_whisper.py RESULT.json [RESULT.json ...]

Each result gets a RESULT.whisper.json beside it with per-prompt transcripts
and corpus WER (en) / CER (zh). The language is forced per prompt, Chinese
output is converted to simplified characters (zhconv), audio is resampled to
16 kHz here, and Whisper's known Chinese hallucinations on trailing silence
(e.g. "优优独播剧场——YoYo Television Series Exclusive") are removed from
every engine's transcripts alike before scoring. Needs whisper.cpp built as
in README.md ($WHISPER_CPP_DIR, $WHISPER_CPP_MODELS with
ggml-large-v3-turbo.bin), soundfile, scipy and zhconv.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import metrics
import numpy as np
import soundfile
import zhconv
from clients.whisper_cpp import WhisperCppServer
from scipy.signal import resample_poly

HALLUCINATIONS = re.compile(
    r"(优优独播剧场.*|YoYo Television Series Exclusive|请不吝点赞.*|字幕由.*|明镜与点点栏目)"
)


def to_16k(path: str, out: Path) -> Path:
    audio, rate = soundfile.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:
        g = np.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // g, rate // g).astype(np.float32)
    soundfile.write(out, audio, 16000, subtype="PCM_16")
    return out


def main(paths: list[str]) -> int:
    server = WhisperCppServer(model="large-v3-turbo").start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav16 = Path(tmp) / "judge16.wav"
            for path in paths:
                data = json.loads(Path(path).read_text())
                items = data.get("items") or data.get("results")
                pairs: dict[str, list] = {"en": [], "zh": []}
                out = []
                for item in items:
                    lang = item["lang"]
                    r = server.client.transcribe(
                        "whisper", to_16k(item["audio_path"], wav16), language=lang
                    )
                    text = (r.text or "").strip()
                    if lang == "zh":
                        text = zhconv.convert(text, "zh-cn")
                    text = HALLUCINATIONS.sub("", text)
                    pairs[lang].append((item["text"], text))
                    out.append(
                        {
                            "id": item["id"],
                            "lang": lang,
                            "whisper": text,
                            "error_rate": metrics.error_rate(item["text"], text, lang),
                            "audio_s": item.get("audio_s"),
                        }
                    )
                summary = {
                    lang: metrics.corpus_error_rate(p, lang)
                    for lang, p in pairs.items()
                    if p
                }
                Path(path).with_suffix(".whisper.json").write_text(
                    json.dumps(
                        {"summary": summary, "items": out}, ensure_ascii=False, indent=1
                    )
                )
                print(path, summary, flush=True)
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
