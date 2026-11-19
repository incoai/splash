"""Metrics for the audio benchmarks: text normalization, WER/CER, latency stats."""

from __future__ import annotations

import re
import unicodedata

import numpy as np

try:
    import jiwer
except ImportError:  # pragma: no cover
    jiwer = None

_WS = re.compile(r"\s+")


def normalize_en(text: str) -> str:
    """Whisper-style basic English normalizer: lowercase, strip punctuation
    and symbols (apostrophes inside words are kept), numbers left as-is."""
    text = text.lower().replace("-", " ").replace("—", " ").replace("–", " ")
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch == "'" or ch == "’":
            out.append("'")
        elif cat[0] in "MSP":
            out.append(" ")
        else:
            out.append(ch)
    text = "".join(out)
    # Strip apostrophes that are not inside a word (e.g. quotes).
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    return _WS.sub(" ", text).strip()


def normalize_zh(text: str) -> str:
    """Chinese normalizer for CER: drop whitespace, punctuation and symbols;
    lowercase any embedded Latin text."""
    text = text.lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in "MSPZC":
            continue
        out.append(ch)
    return "".join(out)


def normalize(text: str, lang: str) -> str:
    return normalize_zh(text) if lang.startswith("zh") else normalize_en(text)


def strip_asr_tags(text: str) -> str:
    """Qwen3-ASR raw output can be ``language English<asr_text>...``."""
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    return text.strip()


def _edits(ref: str, hyp: str, chars: bool):
    if jiwer is None:
        raise RuntimeError("jiwer is required: pip install jiwer")
    if not ref:
        return (0, len(hyp)) if chars else (0, len(hyp.split()))
    if chars:
        o = jiwer.process_characters(ref, hyp)
    else:
        o = jiwer.process_words(ref, hyp)
    errors = o.substitutions + o.deletions + o.insertions
    n = o.hits + o.substitutions + o.deletions
    return errors, n


def error_rate(ref: str, hyp: str, lang: str) -> float:
    """WER for English (word level) or CER for Chinese (character level) on
    normalized text. Returns errors / reference length."""
    chars = lang.startswith("zh")
    r, h = normalize(ref, lang), normalize(hyp, lang)
    e, n = _edits(r, h, chars)
    return e / n if n else (0.0 if e == 0 else 1.0)


def corpus_error_rate(pairs: list[tuple[str, str]], lang: str) -> float | None:
    """Corpus-level WER (en) or CER (zh): total edits / total reference length."""
    chars = lang.startswith("zh")
    e_tot = n_tot = 0
    for ref, hyp in pairs:
        e, n = _edits(normalize(ref, lang), normalize(hyp, lang), chars)
        e_tot += e
        n_tot += n
    return e_tot / n_tot if n_tot else None


def summarize(values, scale: float = 1.0) -> dict | None:
    vals = [v * scale for v in values if v is not None]
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }
