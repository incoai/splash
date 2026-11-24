"""Turn benchmark result JSONs into Markdown tables.

python report.py results/*.json -o RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _f(x, fmt="{:.2f}", none="-"):
    return none if x is None else fmt.format(x)


def _ms(stats, key="p50"):
    return _f(stats[key] if stats else None, "{:.0f}")


def _pct(x):
    return _f(x * 100 if x is not None else None, "{:.2f}%")


def model_name(r: dict) -> str:
    """Model column: the engine's own label (model, quantization, runtime) when the
    result records one, else the model id's last path component."""
    return (r.get("model_info") or {}).get("label") or r["model"].split("/")[-1]


def notes_rows(results: list[dict]) -> list[str]:
    """One line per (engine, model) that records model_info: runtime, quantization,
    and the caveats that make it not directly comparable."""
    lines, seen = [], set()
    for r in results:
        info = r.get("model_info")
        key = (r["task"], r["engine"], model_name(r))
        if not info or key in seen:
            continue
        seen.add(key)
        parts = [
            f"{k}: {info[k]}"
            for k in ("runtime", "quantization", "decoding", "note")
            if info.get(k)
        ]
        lines.append(
            f"- **{r['engine']} / {model_name(r)}** ({r['task']}): " + "; ".join(parts)
        )
    return lines


def tts_rows(results: list[dict]) -> list[str]:
    head = (
        "| Engine | Model | Conc | Mode | N | TTFA p50 / p95 (ms) | Latency p50 / p95 (s) | Audio (s) | "
        "RTF median | Throughput (audio s / wall s) | WER en | CER zh | Errors |"
    )
    sep = "|" + "---|" * 13
    lines = [head, sep]
    for r in results:
        s = r["summary"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} / {} | {} / {} | {} | {} | {} | {} | {} | {} |".format(
                r["engine"],
                model_name(r),
                r["concurrency"],
                r["config"].get("batch_mode", "-"),
                s["n"],
                _ms(s.get("ttfa_ms")),
                _ms(s.get("ttfa_ms"), "p95"),
                _f(s["latency_s"]["p50"] if s.get("latency_s") else None),
                _f(s["latency_s"]["p95"] if s.get("latency_s") else None),
                _f(s.get("audio_s_total"), "{:.1f}"),
                _f(s.get("rtf_median"), "{:.3f}"),
                _f(s.get("throughput_audio_s_per_s"), "{:.2f}"),
                _pct(s.get("wer_en")),
                _pct(s.get("cer_zh")),
                s.get("errors", 0),
            )
        )
    return lines


def tts_breakdown_rows(results: list[dict]) -> list[str]:
    import numpy as np

    lines = [
        "| Engine | Model | Conc | Class | N | TTFA p50 (ms) | Latency p50 (s) | Audio p50 (s) | RTF median | WER/CER |",
        "|" + "---|" * 10,
    ]
    for r in results:
        groups = defaultdict(list)
        for it in r["items"]:
            groups[f"{it['lang']}/{it['length']}"].append(it)
        for cls in sorted(groups):
            its = [i for i in groups[cls] if not i.get("error")]
            if not its:
                continue

            def med(key):
                v = [i[key] for i in its if i.get(key) is not None]
                return float(np.median(v)) if v else None

            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    r["engine"],
                    model_name(r),
                    r["concurrency"],
                    cls,
                    len(its),
                    _f(
                        med("ttfa_s") * 1000 if med("ttfa_s") is not None else None,
                        "{:.0f}",
                    ),
                    _f(med("total_s")),
                    _f(med("audio_s")),
                    _f(med("rtf"), "{:.3f}"),
                    _pct(med("error_rate")),
                )
            )
    return lines


def asr_rows(results: list[dict]) -> list[str]:
    head = (
        "| Engine | Model | Set | Conc | N | Latency p50 / p95 (ms) | Audio (s) | RTFx (per file, median) | "
        "Throughput (audio s / wall s) | WER | Errors |"
    )
    lines = [head, "|" + "---|" * 11]
    for r in results:
        s = r["summary"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} / {} | {} | {} | {} | {} | {} |".format(
                r["engine"],
                model_name(r),
                r["config"].get("set", "-"),
                r["concurrency"],
                s["n"],
                _ms(s.get("latency_ms")),
                _ms(s.get("latency_ms"), "p95"),
                _f(s.get("audio_s_total"), "{:.1f}"),
                _f(s.get("rtfx_median"), "{:.1f}"),
                _f(s.get("throughput_audio_s_per_s"), "{:.1f}"),
                _pct(s.get("wer")),
                s.get("errors", 0),
            )
        )
    return lines


def render(paths: list[Path]) -> str:
    results = []
    for p in paths:
        try:
            r = json.loads(Path(p).read_text())
        except (OSError, ValueError) as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        r["_path"] = str(p)
        results.append(r)
    results.sort(
        key=lambda r: (
            r["task"],
            r["engine"],
            r["model"],
            r["concurrency"],
            r.get("timestamp", ""),
        )
    )
    tts = [r for r in results if r["task"] == "tts"]
    asr = [r for r in results if r["task"] == "asr"]
    out = ["# Audio benchmark results", ""]
    hosts = {json.dumps(r.get("host", {}), sort_keys=True) for r in results}
    for h in sorted(hosts):
        out.append(f"Host: `{h}`")
    out.append("")
    if tts:
        out += [
            "## TTS",
            "",
            "TTFA = time to first audio chunk ('-' = not streamed: continuous batching or an engine "
            "without a streaming API; latency is then the full generation); RTF = generation time / audio seconds; "
            "throughput = total audio seconds / wall seconds at the given concurrency; WER/CER = transcript of the "
            "generated speech (Qwen3-ASR via mlx-audio) vs the input text, normalized.",
            "",
        ]
        out += (
            tts_rows(tts)
            + ["", "### TTS by prompt class", ""]
            + tts_breakdown_rows(tts)
            + [""]
        )
    if asr:
        out += [
            "## ASR",
            "",
            "RTFx = audio seconds / wall seconds; WER on LibriSpeech test-clean references with a "
            "Whisper-style normalizer (lowercase, punctuation stripped, numbers as-is), computed with jiwer.",
            "",
        ]
        out += asr_rows(asr) + [""]
    notes = notes_rows(results)
    if notes:
        out += ["## Engine and model notes", ""] + notes + [""]
    out += (
        ["## Sources", ""]
        + [
            f"- `{r['_path']}`: {r['task']} {r['engine']} {r['model']} c={r['concurrency']} ({r.get('timestamp', '')})"
            for r in results
        ]
        + [""]
    )
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="write Markdown here (default: stdout)",
    )
    args = ap.parse_args(argv)
    md = render(args.results)
    if args.out:
        args.out.write_text(md)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
