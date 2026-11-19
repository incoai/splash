"""A raw version-10 protocol client for the Omni engine process (no HTTP):

  python dev/tools/omni_client.py tts MODEL_DIR OUT.wav "text" [--greedy] [--repeat N] [--language english]
  python dev/tools/omni_client.py asr MODEL_DIR FILE.wav [FILE.wav ...] [--language English]

Spawns `build/splash serve-omni MODEL_DIR`, waits for Ready, submits the
requests, and prints timings. Needs the model's tokenizer (transformers).
"""

import json
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = struct.Struct("<4sHHQ")
SPEECH = struct.Struct("<QIIfIfffIfQIIIIIIII")
TRANSCRIBE = struct.Struct("<QIIIII")
READY, START, TOKENS, DONE, ERROR, STATUS_JSON, AUDIO = (
    0x0100,
    0x0101,
    0x0102,
    0x0104,
    0x0105,
    0x0107,
    0x0110,
)


def frame(kind, payload):
    return HEADER.pack(b"SPLH", 10, kind, len(payload)) + payload


def read_exact(stream, count):
    data = bytearray()
    while len(data) < count:
        chunk = stream.read(count - len(data))
        if not chunk:
            raise EOFError("engine closed its output")
        data.extend(chunk)
    return bytes(data)


def read_frame(stream):
    magic, version, kind, length = HEADER.unpack(read_exact(stream, HEADER.size))
    assert magic == b"SPLH" and version == 10, (magic, version)
    payload = read_exact(stream, length) if length else b""
    return kind, payload


def start_engine(model_dir, lanes=None):
    cmd = [str(ROOT / "build/splash"), "serve-omni", model_dir]
    if lanes:
        cmd += ["--lanes", str(lanes)]
    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=0
    )
    kind, payload = read_frame(process.stdout)
    assert kind == READY, kind
    lanes, context, model_kind = struct.unpack("<IIB", payload)
    print(
        f"ready: lanes {lanes}, context {context}, kind {model_kind}", file=sys.stderr
    )
    return process


def tokenizer_for(model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


def run_tts(model_dir, out, text, greedy, repeat, language):
    tokenizer = tokenizer_for(model_dir)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    config = json.load(open(Path(model_dir) / "config.json"))
    language_id = (
        config["talker_config"]["codec_language_id"].get(language, 0) if language else 0
    )
    process = start_engine(model_dir)
    started = time.perf_counter()
    for r in range(repeat):
        temperature = 0.0 if greedy else 0.9
        head = SPEECH.pack(
            r + 1,
            0,
            language_id,
            temperature,
            50,
            1.0,
            1.05,
            temperature,
            50,
            1.0,
            1 + r,
            0,
            0,
            len(tokens),
            0,
            0,
            0,
            0,
            0,
        )
        process.stdin.write(
            frame(0x0011, head + struct.pack(f"<{len(tokens)}I", *tokens))
        )
    audio = {}
    first = {}
    done = 0
    while done < repeat:
        kind, payload = read_frame(process.stdout)
        if kind == AUDIO:
            rid, offset, count = struct.unpack_from("<QII", payload)
            if rid not in first:
                first[rid] = time.perf_counter() - started
            audio.setdefault(rid, bytearray()).extend(payload[16 : 16 + count * 2])
        elif kind == DONE:
            rid, reason, prompt, frames, prefill, decode, wall, _ = struct.unpack(
                "<QBIIQQQI", payload
            )
            print(
                f"request {rid}: reason {reason}, frames {frames}, prefill {prefill / 1e3:.1f} ms, decode {decode / 1e3:.1f} ms, wall {wall / 1e3:.1f} ms",
                file=sys.stderr,
            )
            done += 1
        elif kind == ERROR:
            failure, retryable, rid, code_len, msg_len = struct.unpack_from(
                "<BBQII", payload
            )
            code = payload[18 : 18 + code_len].decode()
            message = payload[18 + code_len : 18 + code_len + msg_len].decode()
            print(f"error {code}: {message}", file=sys.stderr)
            done += 1
    elapsed = time.perf_counter() - started
    seconds = sum(len(a) for a in audio.values()) / 2 / 24000
    print(
        f"wall {elapsed:.2f} s, audio {seconds:.2f} s, RTF {elapsed / max(seconds, 1e-9):.3f}, "
        f"first audio {[round(v * 1000) for v in first.values()]} ms",
        file=sys.stderr,
    )
    if 1 in audio:
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(bytes(audio[1]))
    process.stdin.close()
    process.wait(timeout=30)


def run_asr(model_dir, files, language):
    tokenizer = tokenizer_for(model_dir)
    prefix = tokenizer.encode(
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>",
        add_special_tokens=False,
    )
    suffix_text = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n" + (
        f"language {language}<asr_text>" if language else ""
    )
    suffix = tokenizer.encode(suffix_text, add_special_tokens=False)
    process = start_engine(model_dir)
    started = time.perf_counter()
    total_seconds = 0.0
    for i, path in enumerate(files):
        with wave.open(path, "rb") as w:
            assert (
                w.getframerate() == 16000
                and w.getnchannels() == 1
                and w.getsampwidth() == 2
            ), path
            pcm = w.readframes(w.getnframes())
        samples = [s / 32768.0 for s in struct.unpack(f"<{len(pcm) // 2}h", pcm)]
        total_seconds += len(samples) / 16000
        head = TRANSCRIBE.pack(i + 1, 0, 0, len(prefix), len(suffix), len(samples))
        body = (
            struct.pack(f"<{len(prefix)}I", *prefix)
            + struct.pack(f"<{len(suffix)}I", *suffix)
            + struct.pack(f"<{len(samples)}f", *samples)
        )
        process.stdin.write(frame(0x0012, head + body))
    tokens = {}
    done = 0
    while done < len(files):
        kind, payload = read_frame(process.stdout)
        if kind == TOKENS:
            rid, offset, count = struct.unpack_from("<QII", payload)
            tokens.setdefault(rid, []).extend(
                struct.unpack_from(f"<{count}I", payload, 16)
            )
        elif kind == DONE:
            rid, reason, prompt, generated, prefill, decode, wall, _ = struct.unpack(
                "<QBIIQQQI", payload
            )
            print(
                f"request {rid}: reason {reason}, prompt {prompt}, generated {generated}, prefill {prefill / 1e3:.1f} ms, decode {decode / 1e3:.1f} ms, wall {wall / 1e3:.1f} ms",
                file=sys.stderr,
            )
            done += 1
        elif kind == ERROR:
            print("error", payload[18:].decode(errors="replace"), file=sys.stderr)
            done += 1
    elapsed = time.perf_counter() - started
    for rid in sorted(tokens):
        print(rid, repr(tokenizer.decode(tokens[rid], skip_special_tokens=True)))
    print(
        f"wall {elapsed:.2f} s for {total_seconds:.1f} s of audio: RTFx {total_seconds / elapsed:.1f}",
        file=sys.stderr,
    )
    process.stdin.close()
    process.wait(timeout=30)


if __name__ == "__main__":
    args = sys.argv[1:]
    kind = args.pop(0)
    greedy = "--greedy" in args
    repeat = int(args[args.index("--repeat") + 1]) if "--repeat" in args else 1
    language = args[args.index("--language") + 1] if "--language" in args else None
    positional = [
        a
        for i, a in enumerate(args)
        if not a.startswith("--")
        and (i == 0 or args[i - 1] not in ("--repeat", "--language"))
    ]
    if kind == "tts":
        run_tts(positional[0], positional[1], positional[2], greedy, repeat, language)
    else:
        run_asr(positional[0], positional[1:], language)
