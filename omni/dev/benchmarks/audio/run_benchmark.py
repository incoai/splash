#!/usr/bin/env python3
"""Audio benchmark driver for Splash Omni and its baselines.

    python run_benchmark.py tts --engine {omni,mlx-audio,sglang-omni} --model ID \
        [--concurrency N] [--limit N] [--out results.json] [--no-wer] [--asr-model ID] \
        [--voice NAME] [--instructions TEXT]
    python run_benchmark.py asr --engine ... --model ID [--set subset|longform|all] \
        [--concurrency N] [--limit N] [--out results.json] [--data-dir DIR]

Engines:
  omni         HTTP, OpenAI-compatible (``--base-url``, default http://127.0.0.1:8000).
  mlx-audio    in-process mlx-audio (needs the mlx-audio venv). TTS concurrency > 1 uses
               mlx-audio batching (``--batch-mode static|continuous``); ASR concurrency > 1
               is refused (no cross-request batching in mlx-audio).
  sglang-omni  launches ``sgl-omni serve`` from ``--sglang-checkout`` / ``$SGLANG_OMNI_DIR``
               (ASR on MLX; TTS on Torch MPS), or with ``--sglang-native`` /
               ``$SGLANG_OMNI_NATIVE`` the native C++ MLX servers (qwen3_asr_server,
               qwen3_tts_server: CustomVoice), waits for /health, benchmarks over HTTP,
               stops the server. ``--no-launch`` benchmarks an already running server;
               ``--server-arg`` passes server flags (--max-batch, --draft-model-path).
  whisper.cpp  ASR only: launches ``whisper-server`` (Metal) from ``--whisper-cpp-dir`` /
               ``$WHISPER_CPP_DIR`` with ``--model`` = ggml model name (``large-v3-q5_0``) or
               path; the model loads once. The server serializes requests, so only
               concurrency 1 is accepted. See clients/whisper_cpp.py.

Qwen3-TTS VoiceDesign and CustomVoice: ``--instructions`` is sent as OpenAI's
``instructions`` (mlx-audio: ``instruct``), the same for every prompt, and ``--voice``
names a CustomVoice preset speaker (mlx-audio: ``voice``; ``default`` sends none to it).

TTS quality: generated audio is transcribed with mlx-audio's Qwen3-ASR (``--asr-model``)
and scored against the input text (WER for en, CER for zh). Results go to ``--out`` as
JSON; ``report.py`` renders Markdown.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import datasets  # noqa: E402
import metrics  # noqa: E402
from clients.openai_audio import OpenAIAudioClient  # noqa: E402

ENGINES = ("omni", "mlx-audio", "sglang-omni", "whisper.cpp")
DEFAULT_ASR_SCORER = "mlx-community/Qwen3-ASR-1.7B-4bit"
LANG_NAMES = {"en": "English", "zh": "Chinese"}
LANG_CODES = {"english": "en", "chinese": "zh"}


def host_info() -> dict:
    info = {
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
    }
    for key, name in (
        ("machdep.cpu.brand_string", "chip"),
        ("hw.memsize", "mem_bytes"),
    ):
        try:
            info[name] = subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return info


def ms(x) -> str:
    return "-" if x is None else f"{x * 1000:.0f}ms"


def log(msg: str) -> None:
    print(
        f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True
    )


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    tmp.replace(path)
    log(f"wrote {path}")


def run_threaded(items: list, fn, concurrency: int) -> tuple[list, float]:
    """Closed-loop: ``concurrency`` workers, each item once. Returns (results, wall_s)."""
    wall0 = time.perf_counter()

    def wrapped(it):
        t = time.perf_counter() - wall0
        r = fn(it)
        r.extra["start_offset_s"] = t
        return r

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(wrapped, items))
    return results, time.perf_counter() - wall0


def maybe_launch_sglang(args, kind: str, concurrency: int):
    if args.engine != "sglang-omni" or args.no_launch:
        return None
    from clients.sglang_omni import SglangNativeServer, SglangOmniServer

    log_path = Path(args.out).with_suffix(".server.log")
    native = {"native_dir": args.sglang_native} if args.sglang_native else None
    server = (SglangNativeServer if native else SglangOmniServer)(
        **(native or {"checkout": args.sglang_checkout}),
        model_path=args.model,
        model_name=args.model,
        kind=kind,
        port=args.port,
        max_running_requests=max(1, concurrency),
        log_path=log_path,
        extra_args=args.server_arg or [],
        config=args.server_config,
    )
    log(f"starting sgl-omni ({kind}) on port {args.port}, log {log_path}")
    server.start(ready_timeout=args.ready_timeout)
    log(f"sgl-omni ready in {server.startup_s:.1f}s on port {server.port}")
    args.base_url = f"http://127.0.0.1:{server.port}"
    return server


# --------------------------------------------------------------------------- TTS
def score_tts(items: list[dict], asr_model: str) -> dict:
    from clients.mlx_audio_inproc import MlxAudioASR

    log(f"scoring TTS output with {asr_model}")
    asr = MlxAudioASR(asr_model).load()
    first = next((it for it in items if it.get("audio_path")), None)
    if first:
        asr.warmup(first["audio_path"])
    pairs = {"en": [], "zh": []}
    for it in items:
        if not it.get("audio_path") or it.get("error"):
            continue
        r = asr.transcribe(it["audio_path"], language=LANG_NAMES.get(it["lang"]))
        it["transcript"] = metrics.strip_asr_tags(r.text)
        it["transcript_latency_s"] = r.latency_s
        if r.error:
            it["score_error"] = r.error
            continue
        it["error_rate"] = metrics.error_rate(it["text"], it["transcript"], it["lang"])
        pairs.setdefault(it["lang"], []).append((it["text"], it["transcript"]))
    asr.unload()
    return {
        "wer_en": metrics.corpus_error_rate(pairs["en"], "en") if pairs["en"] else None,
        "cer_zh": metrics.corpus_error_rate(pairs["zh"], "zh") if pairs["zh"] else None,
        "scorer": asr_model,
    }


def run_tts(args) -> int:
    prompts = datasets.load_tts_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    out = Path(args.out)
    audio_dir = (
        Path(args.audio_dir) if args.audio_dir else out.parent / (out.stem + "_audio")
    )
    audio_dir.mkdir(parents=True, exist_ok=True)
    extra = json.loads(args.extra_json) if getattr(args, "extra_json", None) else None
    if args.instructions is not None:
        extra = {**(extra or {}), "instructions": args.instructions}
    config = {
        "streaming_interval": args.streaming_interval,
        "lang_code": args.lang_code,
        "voice": args.voice,
        "instructions": args.instructions,
        "response_format": args.response_format,
        "warmup": args.warmup,
        "prompts": str(args.prompts or datasets.PROMPTS_PATH),
        "extra": extra,
    }
    extra_summary: dict = {}
    server = None
    log(
        f"tts: engine={args.engine} model={args.model} concurrency={args.concurrency} prompts={len(prompts)}"
    )
    model_info = None
    if args.engine == "whisper.cpp":
        log("whisper.cpp is an ASR engine; it has no TTS")
        return 2
    if args.engine == "mlx-audio":
        from clients.mlx_audio_inproc import MlxAudioTTS

        tts = MlxAudioTTS(
            args.model,
            streaming_interval=args.streaming_interval,
            max_tokens=args.max_tokens,
            voice=None if args.voice == "default" else args.voice,
            instruct=args.instructions,
        ).load()
        extra_summary["model_load_s"] = tts.load_s
        for _ in range(args.warmup):
            w = tts.warmup()
            log(
                f"warmup: ttfa={ms(w.ttfa_s)} total={w.total_s:.2f}s audio={w.audio_s:.2f}s err={w.error}"
            )
        if args.concurrency == 1:
            config["batch_mode"] = "none"
            wall0 = time.perf_counter()
            results = []
            for i, p in enumerate(prompts):
                t = time.perf_counter() - wall0
                r = tts.synth(
                    p["text"],
                    lang_code=args.lang_code,
                    save_path=audio_dir / f"{p['id']}.wav",
                )
                r.extra["start_offset_s"] = t
                results.append(r)
                log(
                    f"{i + 1}/{len(prompts)} {p['id']}: ttfa={ms(r.ttfa_s)} total={r.total_s:.2f}s "
                    f"audio={r.audio_s:.2f}s rtf={r.rtf and round(r.rtf, 3)} err={r.error}"
                )
            wall_s = time.perf_counter() - wall0
        else:
            config["batch_mode"] = args.batch_mode
            fn = (
                tts.synth_batch if args.batch_mode == "static" else tts.synth_continuous
            )
            if args.warmup:
                warm = [
                    {
                        "id": f"warm{i}",
                        "text": f"Warm up sentence number {i} for the batched path.",
                    }
                    for i in range(args.concurrency)
                ]
                _, w_wall = fn(warm, args.concurrency, lang_code=args.lang_code)
                log(
                    f"batch warmup ({args.batch_mode}, {args.concurrency}): {w_wall:.2f}s"
                )
            results, wall_s = fn(
                prompts, args.concurrency, lang_code=args.lang_code, save_dir=audio_dir
            )
            for p, r in zip(prompts, results):
                log(
                    f"{p['id']}: ttfa={ms(r.ttfa_s)} total={r.total_s:.2f}s audio={r.audio_s:.2f}s err={r.error}"
                )
        tts.unload()
    else:
        server = maybe_launch_sglang(args, "tts", args.concurrency)
        client = OpenAIAudioClient(args.base_url, timeout=args.timeout)
        if not client.healthy():
            log(f"server at {args.base_url} is not healthy")
            return 1
        config["batch_mode"] = "server"
        config["base_url"] = args.base_url
        for _ in range(args.warmup):
            w = client.speech(
                args.model,
                prompts[0]["text"],
                voice=args.voice,
                response_format=args.response_format,
                stream=True,
                extra=extra,
            )
            log(
                f"warmup: ttfa={ms(w.ttfa_s)} total={w.total_s:.2f}s audio={w.audio_s:.2f}s err={w.error}"
            )

        def one(p):
            return client.speech(
                args.model,
                p["text"],
                voice=args.voice,
                response_format=args.response_format,
                stream=True,
                save_path=audio_dir / f"{p['id']}.wav",
                extra=extra,
            )

        results, wall_s = run_threaded(prompts, one, args.concurrency)
        for p, r in zip(prompts, results):
            log(
                f"{p['id']}: ttfa={ms(r.ttfa_s)} total={r.total_s:.2f}s audio={r.audio_s:.2f}s err={r.error}"
            )
        if server:
            extra_summary["server_startup_s"] = server.startup_s
            server.stop()

    items = []
    for p, r in zip(prompts, results):
        d = r.to_dict()
        d.update(
            {
                "id": p["id"],
                "lang": p["lang"],
                "length": p["length"],
                "units": p["units"],
                "text": p["text"],
            }
        )
        items.append(d)
    ok = [i for i in items if not i["error"]]
    audio_total = sum(i["audio_s"] for i in ok)
    summary = {
        "n": len(items),
        "errors": len(items) - len(ok),
        "wall_s": wall_s,
        "ttfa_ms": metrics.summarize([i["ttfa_s"] for i in ok], 1000.0),
        "latency_s": metrics.summarize([i["total_s"] for i in ok]),
        "audio_s_total": audio_total,
        "audio_s": metrics.summarize([i["audio_s"] for i in ok]),
        "rtf": metrics.summarize([i["rtf"] for i in ok]),
        "rtf_median": (metrics.summarize([i["rtf"] for i in ok]) or {}).get("p50"),
        "rtf_aggregate": (sum(i["total_s"] for i in ok) / audio_total)
        if audio_total
        else None,
        "throughput_audio_s_per_s": audio_total / wall_s if wall_s else None,
        "peak_memory_gb": max((i["extra"].get("peak_memory_gb") or 0) for i in items)
        or None,
        **extra_summary,
    }
    result = {
        "task": "tts",
        "engine": args.engine,
        "model": args.model,
        **({"model_info": model_info} if model_info else {}),
        "concurrency": args.concurrency,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "host": host_info(),
        "config": config,
        "summary": summary,
        "items": items,
    }
    write_result(out, result)
    if not args.no_wer and ok:
        summary.update(score_tts(items, args.asr_model))
        write_result(out, result)
    s = summary
    log(
        f"done: n={s['n']} err={s['errors']} ttfa_p50={(s['ttfa_ms'] or {}).get('p50')} rtf_median={s['rtf_median']} "
        f"throughput={s['throughput_audio_s_per_s']} wer_en={s.get('wer_en')} cer_zh={s.get('cer_zh')}"
    )
    return 0


# --------------------------------------------------------------------------- ASR
def run_asr(args) -> int:
    sets = ["subset", "longform"] if args.set == "all" else [args.set]
    files = []
    for name in sets:
        part = datasets.load_asr_set(args.data_dir, name)
        files.extend(part[: args.limit] if args.limit else part)
    language = None if args.language.lower() in ("auto", "none", "") else args.language
    out = Path(args.out)
    config = {
        "set": args.set,
        "language": language,
        "data_dir": str(args.data_dir or datasets.default_data_dir()),
        "warmup": args.warmup,
    }
    extra_summary: dict = {}
    log(
        f"asr: engine={args.engine} model={args.model} concurrency={args.concurrency} files={len(files)} sets={sets}"
    )
    warm_file = min(files, key=lambda f: f["duration"])["wav"]
    model_info = None
    if args.engine == "whisper.cpp":
        if args.concurrency != 1:
            log(
                "whisper-server serializes requests (one global mutex around whisper_full); "
                "use --concurrency 1 for this engine."
            )
            return 2
        from clients import whisper_cpp

        server = whisper_cpp.WhisperCppServer(
            args.whisper_cpp_dir,
            model=args.model,
            models_dir=args.whisper_models_dir,
            port=args.port,
            threads=args.whisper_threads,
            extra_args=args.server_arg or [],
            log_path=Path(args.out).with_suffix(".server.log"),
        )
        log(f"starting whisper-server with {server.model_path}")
        server.start(ready_timeout=args.ready_timeout)
        log(f"whisper-server ready in {server.startup_s:.1f}s on port {server.port}")
        client = server.client
        wlang = LANG_CODES.get((language or "").lower(), language)
        config.update(
            {
                "base_url": f"http://127.0.0.1:{server.port}",
                "server_cmd": server.command(),
                "whisper_language": wlang,
            }
        )
        model_info = whisper_cpp.model_info(args.model, server)
        try:
            for _ in range(args.warmup):
                w = client.transcribe(args.model, warm_file, language=wlang)
                log(f"warmup: {w.latency_s:.2f}s err={w.error} text={w.text[:60]!r}")
            results, wall_s = run_threaded(
                files,
                lambda f: client.transcribe(args.model, f["wav"], language=wlang),
                1,
            )
        finally:
            extra_summary["server_startup_s"] = server.startup_s
            server.stop()
        for f, r in zip(files, results):
            log(
                f"{f['id']}: {r.latency_s:.2f}s for {f['duration']:.1f}s audio err={r.error}"
            )
    elif args.engine == "mlx-audio":
        if args.concurrency != 1:
            log(
                "mlx-audio Qwen3-ASR has no cross-request batching (a list of inputs is transcribed sequentially); "
                "use --concurrency 1 for this engine."
            )
            return 2
        from clients.mlx_audio_inproc import MlxAudioASR

        asr = MlxAudioASR(args.model).load()
        extra_summary["model_load_s"] = asr.load_s
        for _ in range(args.warmup):
            w = asr.transcribe(warm_file, language=language)
            log(f"warmup: {w.latency_s:.2f}s err={w.error}")
        wall0 = time.perf_counter()
        results = []
        for i, f in enumerate(files):
            t = time.perf_counter() - wall0
            r = asr.transcribe(f["wav"], language=language)
            r.extra["start_offset_s"] = t
            results.append(r)
            log(
                f"{i + 1}/{len(files)} {f['id']}: {r.latency_s:.2f}s for {f['duration']:.1f}s audio err={r.error}"
            )
        wall_s = time.perf_counter() - wall0
        asr.unload()
    else:
        server = maybe_launch_sglang(args, "asr", args.concurrency)
        client = OpenAIAudioClient(args.base_url, timeout=args.timeout)
        if not client.healthy():
            log(f"server at {args.base_url} is not healthy")
            return 1
        config["base_url"] = args.base_url
        for _ in range(args.warmup):
            w = client.transcribe(
                args.model,
                warm_file,
                language=language,
                response_format=args.response_format,
            )
            log(f"warmup: {w.latency_s:.2f}s err={w.error} text={w.text[:60]!r}")

        def one(f):
            return client.transcribe(
                args.model,
                f["wav"],
                language=language,
                response_format=args.response_format,
            )

        results, wall_s = run_threaded(files, one, args.concurrency)
        for f, r in zip(files, results):
            log(
                f"{f['id']}: {r.latency_s:.2f}s for {f['duration']:.1f}s audio err={r.error}"
            )
        if server:
            extra_summary["server_startup_s"] = server.startup_s
            server.stop()

    items = []
    for f, r in zip(files, results):
        d = r.to_dict()
        d["transcript"] = metrics.strip_asr_tags(d.pop("text"))
        d.update(
            {
                "id": f["id"],
                "set": f["set"],
                "duration": f["duration"],
                "reference": f["text"],
            }
        )
        d["rtfx"] = (
            f["duration"] / r.latency_s if (r.latency_s and not r.error) else None
        )
        d["wer"] = (
            metrics.error_rate(f["text"], d["transcript"], "en")
            if not r.error
            else None
        )
        items.append(d)
    ok = [i for i in items if not i["error"]]
    audio_total = sum(i["duration"] for i in ok)
    summary = {
        "n": len(items),
        "errors": len(items) - len(ok),
        "wall_s": wall_s,
        "latency_ms": metrics.summarize([i["latency_s"] for i in ok], 1000.0),
        "audio_s_total": audio_total,
        "rtfx": metrics.summarize([i["rtfx"] for i in ok]),
        "rtfx_median": (metrics.summarize([i["rtfx"] for i in ok]) or {}).get("p50"),
        "rtfx_aggregate": audio_total / sum(i["latency_s"] for i in ok) if ok else None,
        "throughput_audio_s_per_s": audio_total / wall_s if wall_s else None,
        "wer": metrics.corpus_error_rate(
            [(i["reference"], i["transcript"]) for i in ok], "en"
        )
        if ok
        else None,
        "peak_memory_gb": max((i["extra"].get("peak_memory_gb") or 0) for i in items)
        or None,
        **extra_summary,
    }
    result = {
        "task": "asr",
        "engine": args.engine,
        "model": args.model,
        **({"model_info": model_info} if model_info else {}),
        "concurrency": args.concurrency,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "host": host_info(),
        "config": config,
        "summary": summary,
        "items": items,
    }
    write_result(out, result)
    log(
        f"done: n={summary['n']} err={summary['errors']} latency_p50={(summary['latency_ms'] or {}).get('p50')}ms "
        f"rtfx_median={summary['rtfx_median']} throughput={summary['throughput_audio_s_per_s']} wer={summary['wer']}"
    )
    return 0


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="task", required=True)

    def common(p):
        p.add_argument("--engine", choices=ENGINES, required=True)
        p.add_argument("--model", required=True, help="model id as the engine names it")
        p.add_argument("--concurrency", type=int, default=1)
        p.add_argument(
            "--limit",
            type=int,
            default=0,
            help="first N prompts / files per set (0 = all)",
        )
        p.add_argument("--out", default="results.json")
        p.add_argument(
            "--warmup", type=int, default=1, help="unrecorded warm-up requests"
        )
        p.add_argument(
            "--base-url", default="http://127.0.0.1:8000", help="HTTP engines"
        )
        p.add_argument("--timeout", type=float, default=900.0)
        p.add_argument(
            "--port",
            type=int,
            default=8765,
            help="port for a launched sgl-omni server (a free one is picked if busy)",
        )
        p.add_argument("--sglang-checkout", default=os.environ.get("SGLANG_OMNI_DIR"))
        p.add_argument(
            "--sglang-native",
            default=os.environ.get("SGLANG_OMNI_NATIVE"),
            help="sglang-omni native runtime install dir (bin/qwen3_{asr,tts}_server); "
            "launches it instead of the Python sgl-omni",
        )
        p.add_argument(
            "--no-launch",
            action="store_true",
            help="sglang-omni: use the server at --base-url",
        )
        p.add_argument(
            "--server-arg",
            action="append",
            help="extra argument for sgl-omni serve (repeatable)",
        )
        p.add_argument("--server-config", default=None, help="sgl-omni --config file")
        p.add_argument("--ready-timeout", type=float, default=900.0)

    t = sub.add_parser("tts", help="text-to-speech benchmark")
    common(t)
    t.add_argument(
        "--prompts",
        default=None,
        help=f"prompt JSON (default {datasets.PROMPTS_PATH.name})",
    )
    t.add_argument(
        "--voice", default="default", help="voice; a CustomVoice preset speaker"
    )
    t.add_argument(
        "--instructions",
        default=None,
        help="Qwen3-TTS VoiceDesign/CustomVoice instruction, the same for every prompt",
    )
    t.add_argument(
        "--extra-json",
        default=None,
        help="JSON object merged into each /v1/audio/speech body (e.g. ref_audio/ref_text for sglang-omni Base)",
    )
    t.add_argument(
        "--response-format", default="pcm", choices=["pcm", "wav"], help="HTTP engines"
    )
    t.add_argument(
        "--streaming-interval",
        type=float,
        default=0.32,
        help="mlx-audio chunk interval (s)",
    )
    t.add_argument("--max-tokens", type=int, default=4096)
    t.add_argument(
        "--lang-code",
        default="auto",
        help="mlx-audio lang_code (auto, english, chinese)",
    )
    t.add_argument(
        "--batch-mode",
        choices=["static", "continuous"],
        default="static",
        help="mlx-audio concurrency > 1",
    )
    t.add_argument(
        "--audio-dir",
        default=None,
        help="where generated wavs go (default next to --out)",
    )
    t.add_argument(
        "--no-wer", action="store_true", help="skip transcribing the output for WER/CER"
    )
    t.add_argument(
        "--asr-model",
        default=DEFAULT_ASR_SCORER,
        help="mlx-audio ASR model used for scoring",
    )

    a = sub.add_parser("asr", help="speech-to-text benchmark")
    common(a)
    a.add_argument(
        "--data-dir",
        default=None,
        help=f"prepared data dir (default $OMNI_BENCH_DATA or {datasets.default_data_dir()})",
    )
    a.add_argument("--set", choices=["subset", "longform", "all"], default="subset")
    a.add_argument(
        "--language", default="English", help="language hint; 'auto' sends none"
    )
    a.add_argument(
        "--response-format", default="json", choices=["json", "text", "verbose_json"]
    )
    a.add_argument(
        "--whisper-cpp-dir",
        default=os.environ.get("WHISPER_CPP_DIR"),
        help="whisper.cpp checkout with build/bin/whisper-server",
    )
    a.add_argument(
        "--whisper-models-dir",
        default=None,
        help="dir with ggml-<model>.bin (default $WHISPER_CPP_MODELS or <checkout>/models)",
    )
    a.add_argument(
        "--whisper-threads", type=int, default=None, help="whisper-server -t"
    )
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run_tts(args) if args.task == "tts" else run_asr(args)


if __name__ == "__main__":
    sys.exit(main())
