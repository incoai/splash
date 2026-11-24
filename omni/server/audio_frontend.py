"""Audio requests as the API takes them: what a served model offers, each
request's validation, its tokens, and its answer's shape."""

from __future__ import annotations

import array
import json
import math
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path

import numpy as np

from . import audio_io
from . import audio_protocol as audio_wire
from .audio_backend import AudioJob
from .errors import APIError
from .latency import LatencyMetrics
from .lru import LRUCache

TTS_FAMILY, ASR_FAMILY = "qwen3-tts", "qwen3-asr"
# The model types config.json names, by family.
MODEL_TYPES = {"qwen3_tts": TTS_FAMILY, "qwen3_asr": ASR_FAMILY}
# The ISO 639 codes the API takes for a language, by its English name as
# the models name it (Qwen3-ASR's support_languages, Qwen3-TTS's
# codec_language_id).
ISO_LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "tl": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}
# The ASR prompt around the audio embeddings (docs/omni-design.md, ASR
# pipeline): an empty system prompt, the user turn holding the audio, and
# the assistant prefix, which a forced language extends.
ASR_PREFIX = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>"
ASR_SUFFIX = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
ASR_LANGUAGE = "language {name}<asr_text>"
# What the recognizer writes before a transcript, when it was not forced.
_TRANSCRIPT = re.compile(r"\s*language\s+(?P<language>[^<\n]*?)\s*<asr_text>", re.S)
# The speech checkpoints' tts_model_type (config.json): Base speaks the text
# alone, VoiceDesign in a voice its instruction describes, CustomVoice in one
# of its preset speakers' voices, styled by an optional instruction.
BASE, VOICE_DESIGN, CUSTOM_VOICE = "base", "voice_design", "custom_voice"
TTS_MODEL_TYPES = (BASE, VOICE_DESIGN, CUSTOM_VOICE)
# The instruction's user turn, which leads the prompt (mlx-audio's
# Qwen3-TTS, _prepare_generation_inputs); it is tokenized whole.
INSTRUCTION_TURN = "<|im_start|>user\n{instructions}<|im_end|>\n"
SPEECH_FORMATS = ("wav", "pcm")
TRANSCRIPTION_FORMATS = ("json", "text", "verbose_json")
# The talker's and the code predictor's sampling when a request names none,
# as the model's generation_config.json sets it, else these (Qwen3-TTS's).
TTS_DEFAULTS = {
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "predictor_temperature": 0.9,
    "predictor_top_k": 50,
    "predictor_top_p": 1.0,
}
MAX_FRAMES_LIMIT = 0xFFFFFFFF
# A reference drops its last 2 frames (0.16 s), the end of its closing
# pause (docs/omni-design.md, "Reference continuation"). The engine's speech
# ends soon after its last syllable: 6 frames cut into it, and continuations
# began by finishing it (6 of 9 Chinese paragraphs); none dropped garbled
# one of 9.
REFERENCE_TRIM_FRAMES = 2
# Finished speech kept for references: its text tokens and frames, 4 bytes a
# value (a minute of speech is about 48 KiB).
REFERENCE_BUDGET_BYTES = 64 << 20
REFERENCE_CAPACITY = 4096


def _read_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


@dataclass(frozen=True)
class ModelProfile:
    """What an installed model's configuration says the server serves."""

    family: str
    # TTS: codec language id by lower-case name; ASR: the names the model
    # recognizes, capitalized as it writes them.
    languages: dict[str, int] | tuple[str, ...]
    # The talker's and predictor's default sampling (TTS_DEFAULTS' keys).
    generation: dict[str, float | int] = field(default_factory=dict)
    # Speech: the checkpoint's tts_model_type (TTS_MODEL_TYPES).
    tts_model_type: str = BASE
    # CustomVoice: the preset speakers' codec ids by lower-case name, and the
    # codec language id of the dialect a speaker speaks, which an automatic
    # or Chinese request takes (talker_config.spk_id, spk_is_dialect).
    speakers: dict[str, int] = field(default_factory=dict)
    dialects: dict[str, int] = field(default_factory=dict)

    @property
    def kind(self) -> audio_wire.ModelKind:
        return (
            audio_wire.ModelKind.TTS
            if self.family == TTS_FAMILY
            else audio_wire.ModelKind.ASR
        )

    @property
    def speech(self) -> bool:
        return self.family == TTS_FAMILY

    @classmethod
    def load(cls, model_root) -> ModelProfile:
        """The profile of the model at model_root, from its config.json and,
        for speech, its generation_config.json; ValueError names a
        configuration the server does not serve."""
        root = Path(model_root)
        config = _read_json(root / "config.json")
        family = MODEL_TYPES.get(config.get("model_type"))
        if family is None:
            raise ValueError(
                f"{root / 'config.json'} names model_type "
                f"{config.get('model_type')!r}; the audio server serves "
                + " or ".join(sorted(MODEL_TYPES))
            )
        if family == TTS_FAMILY:
            talker = config.get("talker_config")
            ids = talker.get("codec_language_id") if isinstance(talker, dict) else None
            if not isinstance(ids, dict) or not all(
                isinstance(name, str) and type(value) is int and value >= 0
                for name, value in ids.items()
            ):
                raise ValueError(
                    "config.json has no talker_config.codec_language_id table"
                )
            ids = {name.lower(): value for name, value in ids.items()}
            # A dialect is a speaker's, not a language a request names.
            languages = {
                name: value for name, value in ids.items() if "dialect" not in name
            }
            generation = dict(TTS_DEFAULTS)
            generation_path = root / "generation_config.json"
            if generation_path.is_file():
                generation |= _generation_defaults(_read_json(generation_path))
            model_type = config.get("tts_model_type", BASE)
            if model_type not in TTS_MODEL_TYPES:
                raise ValueError(
                    f"config.json names tts_model_type {model_type!r}; the "
                    "audio server serves " + ", ".join(TTS_MODEL_TYPES)
                )
            speakers, dialects = {}, {}
            if model_type == CUSTOM_VOICE:
                speakers, dialects = _speakers(talker, ids)
            return cls(family, languages, generation, model_type, speakers, dialects)
        names = config.get("support_languages")
        if not isinstance(names, list) or not all(
            isinstance(name, str) and name for name in names
        ):
            raise ValueError("config.json has no support_languages list")
        return cls(family, tuple(dict.fromkeys(name.capitalize() for name in names)))

    def language_id(self, value, speaker=None) -> int:
        """The codec language id a speech request's `language` names; 0,
        automatic, for None. A CustomVoice speaker who speaks a dialect
        speaks it when the language is automatic or Chinese, as mlx-audio
        has it."""
        name = None
        if value is not None:
            name = resolve_language(value, tuple(self.languages)).lower()
        if speaker in self.dialects and name in (None, "chinese"):
            return self.dialects[speaker]
        return 0 if name is None else self.languages[name]

    @property
    def instructions(self) -> str:
        """Whether a speech request's `instructions` are required, optional
        or unsupported: what the model type's prompt takes."""
        return {
            BASE: "unsupported",
            VOICE_DESIGN: "required",
            CUSTOM_VOICE: "optional",
        }[self.tts_model_type]


def _speakers(talker, ids) -> tuple[dict[str, int], dict[str, int]]:
    """A CustomVoice config's preset speakers and their dialects' language
    ids; ValueError for a table the model cannot have."""
    table = talker.get("spk_id")
    if (
        not isinstance(table, dict)
        or not table
        or not all(
            isinstance(name, str) and type(value) is int and value > 0
            for name, value in table.items()
        )
    ):
        raise ValueError("config.json has no talker_config.spk_id speakers")
    speakers = {name.lower(): value for name, value in table.items()}
    dialects = {}
    marks = talker.get("spk_is_dialect") or {}
    for name, dialect in marks.items():
        if isinstance(dialect, str) and dialect and name.lower() in speakers:
            if dialect.lower() not in ids:
                raise ValueError(
                    f"speaker {name}'s dialect {dialect} has no codec language id"
                )
            dialects[name.lower()] = ids[dialect.lower()]
    return speakers, dialects


def _generation_defaults(config) -> dict:
    """The sampling defaults a generation_config.json sets, as Qwen3-TTS
    writes them (the code predictor is its "subtalker")."""
    sources = {
        "temperature": "temperature",
        "top_k": "top_k",
        "top_p": "top_p",
        "repetition_penalty": "repetition_penalty",
        "predictor_temperature": "subtalker_temperature",
        "predictor_top_k": "subtalker_top_k",
        "predictor_top_p": "subtalker_top_p",
    }
    defaults = {}
    for name, key in sources.items():
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if name.endswith("top_k"):
            if type(value) is int and value >= 0:
                defaults[name] = value
        elif math.isfinite(value) and value >= 0:
            defaults[name] = float(value)
    if config.get("do_sample") is False:
        defaults["temperature"] = 0.0
    if config.get("subtalker_dosample") is False:
        defaults["predictor_temperature"] = 0.0
    return defaults


def resolve_language(value, supported) -> str:
    """The name among `supported` (capitalized, as the models write them)
    that `value`, a name in any case or an ISO 639 code, names."""
    if not isinstance(value, str) or not value.strip():
        raise APIError(400, "language must be a non-empty string")
    text = value.strip()
    name = ISO_LANGUAGES.get(text.lower(), text)
    by_lower = {item.lower(): item for item in supported}
    if name.lower() not in by_lower:
        raise APIError(
            400,
            f"language {text!r} is not supported; use one of "
            + ", ".join(sorted(supported))
            + " or its ISO 639 code",
        )
    return by_lower[name.lower()]


@dataclass(frozen=True)
class MultipartForm:
    """A multipart/form-data body: its text fields and its files, by name."""

    fields: dict[str, str]
    files: dict[str, tuple[str | None, bytes]]


def parse_multipart(body: bytes, boundary: str) -> MultipartForm:
    """The fields and files of a multipart/form-data body with `boundary`;
    APIError 400 for a body that is not one."""
    if not boundary or len(boundary) > 70 or not boundary.isprintable():
        raise APIError(400, "invalid multipart boundary")
    delimiter = b"--" + boundary.encode("ascii", "strict")
    fields, files = {}, {}
    position = body.find(delimiter)
    if position < 0:
        raise APIError(400, "multipart body has no boundary")
    position += len(delimiter)
    while True:
        if body[position : position + 2] == b"--":
            break
        if body[position : position + 2] != b"\r\n":
            raise APIError(400, "malformed multipart body")
        position += 2
        header_end = body.find(b"\r\n\r\n", position)
        if header_end < 0:
            raise APIError(400, "malformed multipart part")
        headers = _part_headers(body[position:header_end])
        content_start = header_end + 4
        next_delimiter = body.find(b"\r\n" + delimiter, content_start)
        if next_delimiter < 0:
            raise APIError(400, "multipart body ended inside a part")
        content = body[content_start:next_delimiter]
        position = next_delimiter + 2 + len(delimiter)
        name = headers.get_param("name", header="content-disposition")
        if headers.get_content_disposition() != "form-data" or not isinstance(
            name, str
        ):
            raise APIError(400, "every multipart part needs a form-data name")
        if name in fields or name in files:
            raise APIError(400, f"multipart field {name!r} is repeated")
        filename = headers.get_filename()
        if filename is not None or "content-type" in headers:
            files[name] = (filename, content)
        else:
            try:
                fields[name] = content.decode("utf-8")
            except UnicodeDecodeError:
                raise APIError(400, f"multipart field {name!r} is not UTF-8") from None
    return MultipartForm(fields, files)


def _part_headers(raw: bytes) -> EmailMessage:
    headers = EmailMessage()
    try:
        for line in raw.decode("utf-8").split("\r\n"):
            name, separator, value = line.partition(":")
            if not separator or not name.strip():
                raise ValueError(line)
            headers[name.strip()] = value.strip()
    except (UnicodeDecodeError, ValueError):
        raise APIError(400, "malformed multipart part headers") from None
    return headers


def _bool_field(value, label):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "1", "yes"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "0", "no", ""):
        return False
    raise APIError(400, f"{label} must be a boolean")


def _number(value, label, *, minimum=None, maximum=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise APIError(400, f"{label} must be a number")
    if integer and (type(value) is not int):
        raise APIError(400, f"{label} must be an integer")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        # An integer too large for a float, which no field takes.
        finite = False
    if not finite:
        raise APIError(400, f"{label} must be finite")
    if minimum is not None and value < minimum:
        raise APIError(400, f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise APIError(400, f"{label} must be at most {maximum}")
    return value


@dataclass
class SpeechPlan:
    """A speech request, validated: its job, how it is answered, and the
    speech id (X-Splash-Speech-Id) a later request names it by."""

    job: AudioJob
    response_format: str
    stream: bool
    text_tokens: int
    speech_id: str = ""


class SpeechReferences:
    """The text tokens and frames of recent speech by speech id, within a
    byte budget, least recently used first out: what a request's
    reference_speech_id names. A speech id is pending from its request's
    submission to its end; a request naming a pending one waits for it."""

    def __init__(
        self, budget_bytes=REFERENCE_BUDGET_BYTES, capacity=REFERENCE_CAPACITY
    ):
        self._lock = threading.Lock()
        self._finished = LRUCache(budget_bytes, capacity)
        self._pending: dict[str, threading.Event] = {}

    def begin(self, speech_id):
        with self._lock:
            self._pending[speech_id] = threading.Event()

    def finish(self, speech_id, text_tokens, codes):
        """The end of speech_id's request: its text tokens and frames, or
        None for codes when it failed or was cancelled."""
        with self._lock:
            event = self._pending.pop(speech_id, None)
            if codes:
                text = array.array("I", text_tokens)
                frames = array.array("I", codes)
                self._finished.put(
                    speech_id, (text, frames), text.itemsize * (len(text) + len(frames))
                )
        if event is not None:
            event.set()

    def get(self, speech_id, deadline) -> tuple[array.array, array.array]:
        """The text tokens and frames speech_id's request spoke, waiting
        until `deadline` for one still running; APIError 404 for an id
        unknown, expired or whose request failed."""
        with self._lock:
            entry = self._finished.get(speech_id)
            pending = self._pending.get(speech_id)
        if entry is None and pending is not None:
            timeout = None
            if math.isfinite(deadline):
                timeout = max(0.0, deadline - time.monotonic())
            if not pending.wait(timeout):
                raise APIError(
                    504,
                    f"reference_speech_id {speech_id!r} was still being spoken "
                    "when this request timed out",
                    "request_timeout",
                )
            with self._lock:
                entry = self._finished.get(speech_id)
            if entry is None:
                raise APIError(
                    404,
                    f"reference_speech_id {speech_id!r} names speech that did not "
                    "finish (it failed or was cancelled)",
                    "reference_not_found",
                )
        if entry is None:
            raise APIError(
                404,
                f"reference_speech_id {speech_id!r} names no recent speech: it is "
                "unknown, or expired from the server's memory of recent speech",
                "reference_not_found",
            )
        return entry


@dataclass
class TranscriptionPlan:
    """A transcription, validated and decoded: its samples, the spans each
    request transcribes, its prompt tokens and how it is answered."""

    samples: np.ndarray
    segments: list[audio_io.Segment]
    prefix_tokens: tuple[int, ...]
    suffix_tokens: tuple[int, ...]
    language: str | None
    response_format: str
    deadline: float

    @property
    def duration(self) -> float:
        return len(self.samples) / audio_io.ASR_SAMPLE_RATE


@dataclass
class Transcript:
    """One request's transcript, parsed."""

    text: str
    language: str | None
    tokens: int


class AudioFrontend:
    """Validates audio requests and shapes their answers."""

    def __init__(
        self,
        tokenizer,
        backend,
        model,
        profile: ModelProfile,
        request_timeout,
        max_concurrent,
        *,
        served_model_names=(),
        announce_served_name=False,
        segment_seconds=audio_io.PARALLEL_MINIMUM_SECONDS,
    ):
        self.tokenizer = tokenizer
        self.backend = backend
        self.model = model
        self.profile = profile
        self.request_timeout = request_timeout
        self.max_concurrent = max_concurrent
        # The least length of the segments long audio is decoded in
        # concurrently (audio_io.parallel_segment_seconds); 0 decodes each
        # 20-minute span as one request.
        self.segment_seconds = segment_seconds
        self.latencies = LatencyMetrics()
        served = tuple(served_model_names)
        self.response_model = served[0] if announce_served_name and served else model
        self.model_names = tuple(dict.fromkeys((self.response_model, model, *served)))
        self._request_ids = iter(range(1, 1 << 62))
        self._language_suffixes: dict[str, tuple[int, ...]] = {}
        self._lock = threading.Lock()
        self.references = SpeechReferences()
        if not profile.speech:
            self.asr_prefix = self.tokenize(ASR_PREFIX)
            self.asr_suffix = self.tokenize(ASR_SUFFIX)

    def tokenize(self, text) -> tuple[int, ...]:
        with self.latencies.measure("tokenization"):
            return tuple(
                int(token)
                for token in self.tokenizer(text, add_special_tokens=False)["input_ids"]
            )

    def accepts_model(self, name) -> bool:
        return isinstance(name, str) and name in self.model_names

    def _require_model(self, body):
        name = body.get("model", self.model)
        if not self.accepts_model(name):
            raise APIError(404, f"model {name} not found", "model_not_found")

    def request_deadline(self, started_at) -> float:
        return started_at + self.request_timeout

    def _next_request_id(self) -> int:
        with self._lock:
            return next(self._request_ids)

    @property
    def languages(self) -> tuple[str, ...]:
        if self.profile.speech:
            return tuple(sorted(name.capitalize() for name in self.profile.languages))
        return tuple(self.profile.languages)

    def prepare_speech(self, body, *, deadline) -> SpeechPlan:
        """The job of a /v1/audio/speech request, validated."""
        if not self.profile.speech:
            raise APIError(
                404, f"model {self.model} transcribes audio; it does not speak"
            )
        self._require_model(body)
        text = body.get("input")
        if not isinstance(text, str) or not text.strip():
            raise APIError(400, "input must be a non-empty string")
        voice = body.get("voice", "default")
        if voice is not None and not isinstance(voice, str):
            raise APIError(400, "voice must be a string")
        speaker, instruction = self._voice(voice, body.get("instructions"))
        response_format = body.get("response_format", "wav")
        if response_format not in SPEECH_FORMATS:
            raise APIError(
                400, "response_format must be " + " or ".join(SPEECH_FORMATS)
            )
        stream = body.get("stream", False)
        if stream is None:
            stream = False
        if not isinstance(stream, bool):
            raise APIError(400, "stream must be a boolean")
        language_id = self.profile.language_id(
            body.get("language"), voice.lower() if speaker else None
        )
        generation = self.profile.generation
        temperature = _number(
            body.get("temperature", generation["temperature"]),
            "temperature",
            minimum=0.0,
        )
        top_k = _number(
            body.get("top_k", generation["top_k"]), "top_k", minimum=0, integer=True
        )
        top_p = _number(body.get("top_p", generation["top_p"]), "top_p", maximum=1.0)
        if top_p <= 0.0:
            raise APIError(400, "top_p must be in (0, 1]")
        penalty = _number(
            body.get("repetition_penalty", generation["repetition_penalty"]),
            "repetition_penalty",
        )
        if penalty <= 0.0:
            raise APIError(400, "repetition_penalty must be positive")
        seed = body.get("seed")
        if seed is None:
            seed = secrets.randbits(64)
        elif isinstance(seed, bool) or type(seed) is not int or not 0 <= seed < 2**64:
            raise APIError(400, "seed must be an unsigned 64-bit integer")
        max_frames = _number(
            body.get("max_frames", 0),
            "max_frames",
            minimum=0,
            maximum=MAX_FRAMES_LIMIT,
            integer=True,
        )
        # Frames per audio chunk the engine decodes and streams at once (12.5
        # per second); 0 leaves the engine's default, 25 frames (2 s). A
        # smaller chunk lowers the time to the first chunk and costs more
        # codec work: each chunk is decoded with 25 frames of left context.
        chunk_frames = _number(
            body.get("chunk_frames", 0),
            "chunk_frames",
            minimum=0,
            maximum=MAX_FRAMES_LIMIT,
            integer=True,
        )
        tokens = self.tokenize(text)
        if not tokens:
            raise APIError(400, "input holds no text tokens")
        if len(tokens) > audio_wire.MAX_TEXT_TOKENS:
            raise APIError(
                400,
                f"input is {len(tokens)} tokens; the limit is "
                f"{audio_wire.MAX_TEXT_TOKENS}",
                "context_length_exceeded",
            )
        # The code predictor samples as the talker does, without a penalty
        # (docs/omni-design.md, TTS pipeline): with what the request names,
        # else with its own defaults.
        predictor = {
            name: body[name] if name in body else generation["predictor_" + name]
            for name in ("temperature", "top_k", "top_p")
        }
        # Last, as it may wait for the speech it names to finish.
        reference_text, reference_codes = self._reference(
            body.get("reference_speech_id"), deadline
        )
        frame = audio_wire.SpeechRequest(
            request_id=0,
            text_tokens=tokens,
            language_id=language_id,
            temperature=float(temperature),
            top_k=top_k,
            top_p=float(top_p),
            repetition_penalty=float(penalty),
            predictor_temperature=float(predictor["temperature"]),
            predictor_top_k=predictor["top_k"],
            predictor_top_p=float(predictor["top_p"]),
            seed=seed,
            max_frames=max_frames,
            chunk_frames=chunk_frames,
            reference_text_tokens=reference_text,
            reference_codes=reference_codes,
            # Every answer's frames are kept, for later requests' references.
            flags=audio_wire.SPEECH_FLAG_CODES,
            instruction_tokens=instruction,
            preset_speaker=speaker,
        )
        job = AudioJob(self._next_request_id(), frame, deadline)
        return SpeechPlan(
            job, response_format, stream, len(tokens), "speech-" + secrets.token_hex(12)
        )

    def _reference(
        self, speech_id, deadline
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The reference text tokens and codes reference_speech_id gives:
        the text that speech spoke and its frames but the last
        REFERENCE_TRIM_FRAMES (at least one is kept); none without one."""
        if speech_id is None:
            return (), ()
        if not isinstance(speech_id, str) or not speech_id:
            raise APIError(
                400,
                "reference_speech_id must be the X-Splash-Speech-Id of an earlier "
                "speech response",
            )
        text, codes = self.references.get(speech_id, deadline)
        frames = len(codes) // audio_wire.CODES_PER_FRAME
        kept = max(1, frames - REFERENCE_TRIM_FRAMES)
        return tuple(text), tuple(codes[: kept * audio_wire.CODES_PER_FRAME])

    def begin_speech(self, plan: SpeechPlan):
        """Marks the plan's speech id pending, before it is submitted."""
        self.references.begin(plan.speech_id)

    def finish_speech(self, plan: SpeechPlan, result):
        """Keeps what the plan's request spoke for later references: its
        text tokens and frames, from its result (None when it failed). A
        cancelled request's speech is incomplete and is not kept."""
        codes = None
        if result is not None and result.reason != "cancelled":
            codes = result.codes
        self.references.finish(plan.speech_id, plan.job.frame.text_tokens, codes)

    def _voice(self, voice, instructions) -> tuple[int, tuple[int, ...]]:
        """The preset speaker's codec id (0 for none) and the instruction
        turn's tokens a speech request's `voice` and `instructions` give, as
        the model type takes them: a Base model takes no instructions, a
        VoiceDesign model requires them (its voice is the one they describe;
        `voice` is not read), a CustomVoice model requires one of its preset
        speakers and takes instructions or none."""
        if instructions is not None and not isinstance(instructions, str):
            raise APIError(400, "instructions must be a string")
        model_type = self.profile.tts_model_type
        if model_type == BASE:
            if instructions:
                raise APIError(
                    400,
                    f"model {self.model} is a Qwen3-TTS Base model, which takes "
                    "no instructions; serve a VoiceDesign or CustomVoice "
                    "checkpoint to control the voice with them",
                )
            return 0, ()
        if model_type == VOICE_DESIGN and not (instructions and instructions.strip()):
            raise APIError(
                400,
                f"model {self.model} is a Qwen3-TTS VoiceDesign model: "
                "instructions must describe the voice, such as "
                "'A calm male voice with a low pitch'",
            )
        speaker = 0
        if model_type == CUSTOM_VOICE:
            speaker = self.profile.speakers.get((voice or "").lower(), 0)
            if not speaker:
                raise APIError(
                    400,
                    f"voice must be one of model {self.model}'s speakers: "
                    + ", ".join(sorted(self.profile.speakers)),
                )
        if not instructions:
            return speaker, ()
        tokens = self.tokenize(INSTRUCTION_TURN.format(instructions=instructions))
        if len(tokens) > audio_wire.MAX_INSTRUCTION_TOKENS:
            raise APIError(
                400,
                f"instructions are {len(tokens)} tokens with their turn's; the "
                f"limit is {audio_wire.MAX_INSTRUCTION_TOKENS}",
                "context_length_exceeded",
            )
        return speaker, tokens

    @property
    def voices(self) -> list[dict]:
        """The voices a speech request's `voice` names: a CustomVoice
        model's preset speakers, else the one default voice."""
        if self.profile.speakers:
            return [{"id": name} for name in sorted(self.profile.speakers)]
        return [{"id": "default"}]

    def prepare_transcription(
        self, form: MultipartForm, *, deadline, reserve
    ) -> TranscriptionPlan:
        """A /v1/audio/transcriptions request, validated and its upload
        decoded; `reserve(bytes)` charges the decoded samples to the body
        budget."""
        if self.profile.speech:
            raise APIError(
                404, f"model {self.model} speaks; it does not transcribe audio"
            )
        fields = form.fields
        self._require_model(fields)
        if "file" not in form.files:
            raise APIError(400, "a multipart file field is required")
        if _bool_field(fields.get("stream"), "stream"):
            raise APIError(400, "streaming transcriptions are not supported")
        response_format = fields.get("response_format", "json")
        if response_format not in TRANSCRIPTION_FORMATS:
            raise APIError(
                400, "response_format must be " + ", ".join(TRANSCRIPTION_FORMATS)
            )
        if "temperature" in fields:
            try:
                temperature = float(fields["temperature"])
            except ValueError:
                temperature = math.nan
            # Decoding is greedy; the field is accepted for compatibility.
            _number(temperature, "temperature", minimum=0.0)
        language = None
        if fields.get("language"):
            language = resolve_language(fields["language"], self.profile.languages)
        filename, data = form.files["file"]
        with self.latencies.measure("preparation"):
            samples = audio_io.decode_upload(data, filename)
        reserve(samples.nbytes)
        duration = len(samples) / audio_io.ASR_SAMPLE_RATE
        segments = audio_io.split_segments(
            samples,
            request_seconds=audio_io.parallel_segment_seconds(
                duration, self.max_concurrent, self.segment_seconds
            ),
        )
        suffix = self.asr_suffix
        if language is not None:
            suffix = self._language_suffix(language)
        return TranscriptionPlan(
            samples,
            segments,
            self.asr_prefix,
            suffix,
            language,
            response_format,
            deadline,
        )

    def _language_suffix(self, language: str) -> tuple[int, ...]:
        with self._lock:
            cached = self._language_suffixes.get(language)
        if cached is None:
            cached = self.tokenize(ASR_SUFFIX + ASR_LANGUAGE.format(name=language))
            with self._lock:
                self._language_suffixes[language] = cached
        return cached

    def transcription_job(
        self, plan: TranscriptionPlan, segment: audio_io.Segment
    ) -> AudioJob:
        frame = audio_wire.TranscribeRequest(
            request_id=0,
            prefix_tokens=plan.prefix_tokens,
            suffix_tokens=plan.suffix_tokens,
            samples=audio_io.float32_bytes(plan.samples[segment.start : segment.end]),
        )
        return AudioJob(self._next_request_id(), frame, plan.deadline)

    def transcript(self, tokens, forced_language) -> Transcript:
        """The transcript the recognizer's tokens spell, its `language X
        <asr_text>` prefix parsed off unless the language was forced."""
        text = self.tokenizer.decode(
            list(tokens), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        language = forced_language
        if forced_language is None:
            if match := _TRANSCRIPT.match(text):
                language = match.group("language").strip() or None
                text = text[match.end() :]
        return Transcript(text.strip(), language, len(tokens))

    def transcription_response(self, plan: TranscriptionPlan, transcripts):
        """The answer to a transcription, in its response_format: the body
        and its media type."""
        text = " ".join(item.text for item in transcripts if item.text)
        if plan.response_format == "text":
            return text, "text/plain; charset=utf-8"
        if plan.response_format == "json":
            return {"text": text}, "application/json"
        languages = [item.language for item in transcripts if item.language]
        segments = []
        for index, (segment, item) in enumerate(zip(plan.segments, transcripts)):
            start, end = segment.seconds()
            segments.append(
                {"id": index, "start": start, "end": end, "text": item.text}
            )
        return {
            "task": "transcribe",
            "language": languages[0] if languages else None,
            "duration": plan.duration,
            "text": text,
            "segments": segments,
        }, "application/json"

    def status(self):
        status = self.backend.status()
        status["frontend"] = {
            "model": self.model,
            "family": self.profile.family,
            "languages": list(self.languages),
            "max_concurrent": self.max_concurrent,
            "request_timeout": (
                None if math.isinf(self.request_timeout) else self.request_timeout
            ),
            "served_model_names": list(self.model_names),
        }
        status["latency"] = self.latencies.snapshot()
        status["time"] = time.time()
        return status

    def _speech_controls(self) -> dict:
        """What a speech model's listing tells a client about its voice: the
        model type, whether it takes instructions and its preset voices."""
        controls = {
            "tts_model_type": self.profile.tts_model_type,
            "instructions": self.profile.instructions,
            # A request may continue earlier speech (reference_speech_id).
            "references": True,
        }
        if self.profile.speakers:
            controls["voices"] = sorted(self.profile.speakers)
        return controls

    def models(self):
        return [
            {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "splash",
                "family": self.profile.family,
                "task": "text-to-speech" if self.profile.speech else "transcription",
                "languages": list(self.languages),
                **(self._speech_controls() if self.profile.speech else {}),
                **(
                    {"root": self.response_model} if name != self.response_model else {}
                ),
            }
            for name in self.model_names
        ]
