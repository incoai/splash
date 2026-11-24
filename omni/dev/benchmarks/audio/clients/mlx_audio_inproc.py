"""In-process mlx-audio baseline: Qwen3-TTS generate (streaming, for TTFA) and
Qwen3-ASR transcribe, measured the same way as the HTTP clients.

Concurrency for TTS uses mlx-audio's own batching:
  * ``static``: ``Model.batch_generate(stream=True)`` on fixed batches of N
    prompts; per-sequence TTFA is measurable. Finish times of early sequences
    are reported when their last chunk is flushed, which happens when the
    whole batch finishes.
  * ``continuous``: the ``Qwen3TTSBatchSession`` continuous-batching session
    (closed loop, N in flight, non-streaming: no TTFA, latency = submit ->
    done).
mlx-audio's Qwen3-ASR has no cross-request batching (a list of inputs is
processed sequentially), so ASR concurrency > 1 is refused.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .openai_audio import ASRResult, TTSResult

TTS_SAMPLE_RATE = 24000


def _free():
    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


def _to_numpy(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _save(path, audio: np.ndarray, sr: int) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return str(path)


class MlxAudioTTS:
    def __init__(
        self,
        model_id: str,
        streaming_interval: float = 0.32,
        max_tokens: int = 4096,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        voice: str | None = None,
        instruct: str | None = None,
    ):
        self.model_id = model_id
        # A CustomVoice preset speaker and a VoiceDesign/CustomVoice
        # instruction, passed to generate() when set.
        self.voice_options = {
            key: value
            for key, value in (("voice", voice), ("instruct", instruct))
            if value is not None
        }
        self.streaming_interval = streaming_interval
        self.max_tokens = max_tokens
        self.sampling = dict(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        self.model = None
        self.load_s = None

    def load(self):
        from mlx_audio.tts.utils import load_model

        t = time.perf_counter()
        self.model = load_model(self.model_id)
        self.load_s = time.perf_counter() - t
        return self

    def unload(self):
        self.model = None
        _free()

    def warmup(
        self, text: str = "This is a short warm up sentence for the benchmark."
    ) -> TTSResult:
        return self.synth(text)

    def synth(self, text: str, lang_code: str = "auto", save_path=None) -> TTSResult:
        """Single request, streaming; TTFA = first chunk evaluated."""
        import mlx.core as mx

        res = TTSResult(sample_rate=TTS_SAMPLE_RATE)
        pieces = []
        t0 = time.perf_counter()
        try:
            for r in self.model.generate(
                text=text,
                lang_code=lang_code,
                stream=True,
                streaming_interval=self.streaming_interval,
                max_tokens=self.max_tokens,
                split_pattern="",
                **self.voice_options,
                **self.sampling,
            ):
                mx.eval(r.audio)
                now = time.perf_counter()
                if res.ttfa_s is None and r.audio.shape[-1] > 0:
                    res.ttfa_s = now - t0
                res.num_chunks += 1
                pieces.append(_to_numpy(r.audio))
            res.total_s = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001 - record and continue the run
            res.total_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {e}"
            return res
        audio = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
        res.audio_s = len(audio) / TTS_SAMPLE_RATE
        res.num_bytes = len(audio) * 2
        res.extra["peak_memory_gb"] = mx.get_peak_memory() / 1e9
        if save_path is not None:
            res.audio_path = _save(save_path, audio, TTS_SAMPLE_RATE)
        return res

    def synth_batch(
        self, items: list[dict], batch_size: int, lang_code: str = "auto", save_dir=None
    ) -> tuple[list[TTSResult], float]:
        """Static batches of ``batch_size`` through ``batch_generate(stream=True)``.
        items: dicts with ``id`` and ``text``. Returns results (same order) and wall seconds."""
        import mlx.core as mx

        results = [TTSResult(sample_rate=TTS_SAMPLE_RATE) for _ in items]
        wall0 = time.perf_counter()
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            pieces = [[] for _ in batch]
            t0 = time.perf_counter()
            try:
                for r in self.model.batch_generate(
                    texts=[b["text"] for b in batch],
                    lang_code=lang_code,
                    stream=True,
                    streaming_interval=self.streaming_interval,
                    max_tokens=self.max_tokens,
                    **self.sampling,
                ):
                    mx.eval(r.audio)
                    now = time.perf_counter()
                    k = r.sequence_idx
                    res = results[start + k]
                    if res.ttfa_s is None and r.audio.shape[-1] > 0:
                        res.ttfa_s = now - t0
                    res.num_chunks += 1
                    res.total_s = now - t0
                    pieces[k].append(_to_numpy(r.audio))
            except Exception as e:  # noqa: BLE001
                for k in range(len(batch)):
                    results[start + k].error = f"{type(e).__name__}: {e}"
                    results[start + k].total_s = time.perf_counter() - t0
                continue
            t_end = time.perf_counter()
            for k, b in enumerate(batch):
                res = results[start + k]
                if res.total_s is None:
                    res.total_s = t_end - t0
                audio = (
                    np.concatenate(pieces[k]) if pieces[k] else np.zeros(0, np.float32)
                )
                res.audio_s = len(audio) / TTS_SAMPLE_RATE
                res.num_bytes = len(audio) * 2
                res.extra["batch_wall_s"] = t_end - t0
                res.extra["batch_size"] = len(batch)
                if save_dir is not None:
                    res.audio_path = _save(
                        Path(save_dir) / f"{b['id']}.wav", audio, TTS_SAMPLE_RATE
                    )
            res.extra["peak_memory_gb"] = mx.get_peak_memory() / 1e9
        return results, time.perf_counter() - wall0

    def synth_continuous(
        self,
        items: list[dict],
        max_batch_size: int,
        lang_code: str = "auto",
        save_dir=None,
    ) -> tuple[list[TTSResult], float]:
        """Closed-loop continuous batching with ``max_batch_size`` in flight."""
        import mlx.core as mx
        from mlx_audio.tts.continuous import TTSBatchItem, TTSBatchOptions

        opts = TTSBatchOptions(
            max_batch_size=max_batch_size,
            max_tokens=self.max_tokens,
            lang_code=lang_code,
            stream=False,
            **self.sampling,
        )
        session = self.model.create_tts_batch_session(opts)
        results = [TTSResult(sample_rate=TTS_SAMPLE_RATE) for _ in items]
        submitted: dict[int, float] = {}
        pending = list(enumerate(items))
        wall0 = time.perf_counter()
        while pending or not session.idle:
            while pending and session.available_slots > len(session._pending):
                i, it = pending.pop(0)
                submitted[i] = time.perf_counter()
                session.add([TTSBatchItem(sequence_id=i, text=it["text"])])
            for ev in session.step():
                if not ev.done:
                    continue
                now = time.perf_counter()
                res = results[ev.sequence_id]
                res.total_s = now - submitted[ev.sequence_id]
                if ev.error is not None:
                    res.error = f"{type(ev.error).__name__}: {ev.error}"
                    continue
                mx.eval(ev.audio)
                audio = _to_numpy(ev.audio)
                res.audio_s = len(audio) / TTS_SAMPLE_RATE
                res.num_bytes = len(audio) * 2
                res.num_chunks = 1
                res.extra["tokens"] = ev.token_count
                if save_dir is not None:
                    res.audio_path = _save(
                        Path(save_dir) / f"{items[ev.sequence_id]['id']}.wav",
                        audio,
                        TTS_SAMPLE_RATE,
                    )
        for res in results:
            res.extra["peak_memory_gb"] = mx.get_peak_memory() / 1e9
        return results, time.perf_counter() - wall0


class MlxAudioASR:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.load_s = None

    def load(self):
        from mlx_audio.stt.utils import load_model

        t = time.perf_counter()
        self.model = load_model(self.model_id)
        self.load_s = time.perf_counter() - t
        return self

    def unload(self):
        self.model = None
        _free()

    def warmup(self, wav_path) -> ASRResult:
        return self.transcribe(wav_path)

    def transcribe(self, wav_path, language: str | None = None) -> ASRResult:
        import mlx.core as mx

        res = ASRResult()
        t0 = time.perf_counter()
        try:
            out = self.model.generate(str(wav_path), language=language)
            res.latency_s = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            res.latency_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {e}"
            return res
        res.text = (out.text or "").strip()
        lang = out.language
        res.language = (
            lang[0]
            if isinstance(lang, list) and lang
            else (lang if isinstance(lang, str) else None)
        )
        res.extra.update(
            {
                "prompt_tokens": out.prompt_tokens,
                "generation_tokens": out.generation_tokens,
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
            }
        )
        return res
