"""The voice chat demo's servers: speech recognition and speech, each a
`splash-omni serve` (session.Selection), and a language model served by
Splash (`splash serve`), all three started, watched and stopped as the setup
page's one server is (session.ServerSession).

The page then runs the conversation itself, through the three servers' APIs
(transcription, chat completions, speech): this module starts processes and
never runs a model.

Standard library only, with the installer and server modules that are too.
"""

from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from install import paths
from server import serve_options

from . import session

ASR, TTS, LLM = "asr", "tts", "llm"
ROLES = (ASR, TTS, LLM)
# Ports apart from the setup page's server (serve_options.DEFAULT_PORT), so
# that both may run.
DEFAULT_PORTS = {ASR: 8031, TTS: 8032, LLM: 8033}
# A conversation sends one utterance at a time and a few sentences at once,
# so the speech servers run few lanes in a small Metal budget: by default
# each sizes its KV page pool to whatever budget the weights leave, which
# would take the memory the language model needs.
DEFAULT_LANES = {ASR: 2, TTS: 4}
# Each speech server's --max-memory in GiB, by task, size and precision:
# what its engine takes with those lanes and its smallest KV pool, which it
# refuses to exceed (speech 2.6 GiB at 1.7B 4-bit, 5.1 at bf16, 1.9 at 0.6B;
# transcription 2.9, 5.2 and 1.8), and for transcription with its DFlash
# draft the budget the engine fits the draft in (4.1, 6.7 and 2.8; M5 Pro,
# the engine's own figures), each rounded up.
SPEECH_MEMORY_GIB = {
    (TTS, "1.7B", "4bit"): 3,
    (TTS, "1.7B", "bf16"): 6,
    (TTS, "0.6B", "4bit"): 2,
    (ASR, "1.7B", "4bit"): 3,
    (ASR, "1.7B", "bf16"): 6,
    (ASR, "0.6B", "4bit"): 2,
}
DRAFT_MEMORY_GIB = {("1.7B", "4bit"): 5, ("1.7B", "bf16"): 7, ("0.6B", "4bit"): 3}
DEFAULT_ASR = "mlx-community/Qwen3-ASR-1.7B-4bit"
DEFAULT_TTS = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"
# The models Splash serves (its README's table), the GGUF variant first.
LLM_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M",
    "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
    "mlx-community/Qwen3.6-35B-A3B-4bit",
    "mlx-community/Qwen3.8-27B-4bit",
)
DEFAULT_LLM = LLM_MODELS[0]
# The DFlash2 draft Splash pairs with each, by the target's base name.
DRAFTS = {
    "Qwen3.6-35B-A3B": "incoai/Qwen3.6-35B-A3B-DFlash2",
    "Qwen3.8-27B": "incoai/Qwen3.8-27B-DFlash2",
}
# What macOS lets the GPU wire of the Mac's memory, all processes together
# (about three quarters above 36 GiB, two thirds below), and the margin kept
# free of the three budgets, in GiB.
GPU_SHARE = (0.75, 0.67)
GPU_MARGIN_GIB = 1.5
MINIMUM_LLM_GIB = 12


# The runtime folder the Splash desktop app ships (Contents/Resources/splash),
# which runs as the `splash` command does: the page accepts one as the splash
# command it is given.
def app_runtime(path):
    """Whether path is a Splash runtime folder, as Splash.app holds one: its
    Python and launcher, which run as the `splash` command does."""
    root = Path(path)
    python = root / "python/bin/python3"
    return (root / "install/launcher.py").is_file() and os.access(python, os.X_OK)


def usable_splash(path):
    """Whether path runs Splash: an executable `splash`, or a runtime folder."""
    if not path:
        return False
    if os.path.isdir(path):
        return app_runtime(path)
    return os.path.isfile(path) and os.access(path, os.X_OK)


def find_splash():
    """The `splash` command: $SPLASH_BIN, else the launcher of the Splash
    checkout that holds Splash Omni (../splash), else `splash` on PATH; None
    when none is found."""
    candidates = [
        os.environ.get("SPLASH_BIN"),
        str(paths.ROOT.parent / "splash"),
        shutil.which("splash"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _hub_cache():
    if hub := os.environ.get("HF_HUB_CACHE"):
        return Path(hub)
    home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    return Path(home) / "hub"


def _snapshots(repo):
    return _hub_cache() / ("models--" + repo.replace("/", "--")) / "snapshots"


def draft_directory(model):
    """The cached snapshot of `model`'s DFlash2 draft, or None. Splash takes
    it as --draft-model: a draft fetched without a branch record (refs/main)
    cannot be resolved offline by name."""
    name = model.partition(":")[0].split("/")[-1]
    draft = next((repo for base, repo in DRAFTS.items() if name.startswith(base)), None)
    if draft is None:
        return None
    try:
        snapshots = sorted(
            _snapshots(draft).iterdir(), key=lambda path: path.stat().st_mtime
        )
    except OSError:
        return None
    for snapshot in reversed(snapshots):
        names = {entry.name for entry in snapshot.iterdir()}
        if "config.json" in names and any(
            name.endswith(".safetensors") for name in names
        ):
            return str(snapshot)
    return None


def llm_cached(model):
    """Whether the Hugging Face cache holds `model` (OWNER/REPO[:VARIANT]):
    a GGUF variant's file, or an MLX snapshot's config. Splash pairs and
    fetches the draft itself."""
    repo, _, variant = model.partition(":")
    snapshots = _snapshots(repo)
    try:
        for snapshot in snapshots.iterdir():
            names = [entry.name for entry in snapshot.iterdir()]
            if variant:
                if any(name.endswith(".gguf") and variant in name for name in names):
                    return True
            elif "config.json" in names:
                return True
    except OSError:
        return False
    return False


def total_memory_gib():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (OSError, ValueError):
        return None


def speech_memory_gib(task, model, speculative=False):
    """A speech server's --max-memory in GiB (SPEECH_MEMORY_GIB)."""
    precision = "bf16" if model.endswith("-bf16") else "4bit"
    size = "0.6B" if "-0.6B-" in model else "1.7B"
    if task == ASR and speculative:
        return DRAFT_MEMORY_GIB[(size, precision)]
    return SPEECH_MEMORY_GIB[(task, size, precision)]


def gpu_budget_gib():
    """What the three servers may take together: the GPU's share of the
    Mac's memory less the margin, or None when the memory is unknown."""
    total = total_memory_gib()
    if total is None:
        return None
    share = GPU_SHARE[0] if total >= 36 else GPU_SHARE[1]
    return total * share - GPU_MARGIN_GIB


def default_llm_memory(asr_model=None, tts_model=None, speculative=True):
    """Splash's --max-memory beside the two speech servers: what
    gpu_budget_gib leaves after their budgets, or None (Splash's own budget)
    when that is unknown or too little. The page computes the same from the
    catalog's memory (catalog)."""
    budget = gpu_budget_gib()
    if budget is None:
        return None
    speech = speech_memory_gib(ASR, asr_model or DEFAULT_ASR, speculative)
    speech += speech_memory_gib(TTS, tts_model or DEFAULT_TTS)
    budget = int(budget - speech)
    return f"{budget}G" if budget >= MINIMUM_LLM_GIB else None


@dataclass(frozen=True)
class LlmSelection:
    """The language model server: `splash serve` with the voice chat's
    options. It has the interface session.ServerSession starts a selection
    with."""

    model: str
    command_path: str
    port: int = DEFAULT_PORTS[LLM]
    max_memory: str | None = None
    # The cached draft's snapshot (draft_directory()), or None for Splash's
    # own choice.
    draft: str | None = None
    task: str = field(default=LLM)

    def downloads(self):
        # Splash downloads the model and its draft itself.
        return ()

    def serve_arguments(self, *, offline=False):
        argv = ["serve", "--model", self.model, "--port", str(self.port)]
        # Text only, kept loaded for the conversation.
        argv += ["--language-only", "--idle-release", "off"]
        if self.draft is not None:
            argv += ["--draft-model", self.draft]
        if offline:
            argv.append("--offline")
        if self.max_memory is not None:
            argv += ["--max-memory", self.max_memory]
        return argv

    def launcher(self):
        """The program that runs `splash`: the command itself, or a Splash
        runtime folder's Python and launcher."""
        if os.path.isdir(self.command_path):
            root = Path(self.command_path)
            return [
                str(root / "python/bin/python3"),
                "-u",
                str(root / "install/launcher.py"),
            ]
        return [self.command_path]

    def command(self, allowed_origins=(), python=None, *, offline=None):
        argv = [*self.launcher(), *self.serve_arguments(offline=bool(offline))]
        for origin in allowed_origins:
            argv += ["--allowed-origin", origin]
        return argv

    def shown_command(self, *, offline=None):
        return shlex.join(["splash", *self.serve_arguments(offline=bool(offline))])

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def process_environment(self):
        """Splash's environment and working directory: its own checkout's,
        without this one's PYTHONPATH (both have `server` packages)."""
        environment = dict(os.environ, PYTHONUNBUFFERED="1")
        environment.pop("PYTHONPATH", None)
        if os.path.isdir(self.command_path):
            # A signed app's runtime: nothing may be written into it.
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            return environment, str(Path(self.command_path).resolve())
        return environment, str(Path(self.command_path).resolve().parent)


def _llm_plan(selection):
    """Splash starts offline when the cache holds the model and its draft,
    so it asks the Hub nothing; otherwise it downloads what is missing."""
    return session.Plan(
        (), (), llm_cached(selection.model) and selection.draft is not None
    )


def catalog():
    """What the voice chat's setup offers: the speech checkpoints by task,
    the language models with what the cache holds, and the splash command."""
    return {
        "splash": find_splash(),
        "llm_models": [
            {"id": model, "cached": llm_cached(model)} for model in LLM_MODELS
        ],
        "defaults": {
            ASR: DEFAULT_ASR,
            TTS: DEFAULT_TTS,
            LLM: DEFAULT_LLM,
            "ports": DEFAULT_PORTS,
            "llm_max_memory": default_llm_memory(),
        },
        # What the page recomputes the language model's default from as
        # the speech models and the draft change (default_llm_memory).
        "memory": {
            "budget_gib": gpu_budget_gib(),
            "minimum_llm_gib": MINIMUM_LLM_GIB,
            "speech_gib": {
                checkpoint.model: {
                    "plain": speech_memory_gib(checkpoint.task, checkpoint.model),
                    "draft": speech_memory_gib(
                        checkpoint.task, checkpoint.model, speculative=True
                    ),
                }
                for checkpoint in session.CHECKPOINTS
            },
        },
    }


def _port(body, role):
    port = body.get(f"{role}_port", DEFAULT_PORTS[role])
    if type(port) is not int or not 1 <= port <= 65535:
        raise session.SelectionError(f"{role}_port must be a number from 1 to 65535")
    return port


def _speech_memory(task, model, speculative=False):
    """speech_memory_gib as --max-memory spells it; for a model that is not
    a string, None, which the selection refuses by its own rule."""
    if not isinstance(model, str):
        return None
    return f"{speech_memory_gib(task, model, speculative is True)}G"


def selections(body):
    """The three servers a start request names; SelectionError for a setup
    that cannot start."""
    if not isinstance(body, dict):
        raise session.SelectionError("the setup must be a JSON object")
    asr_model = body.get("asr_model", DEFAULT_ASR)
    tts_model = body.get("tts_model", DEFAULT_TTS)
    speculative = body.get("speculative", True)
    asr = session.Selection.from_request(
        {
            "task": session.ASR,
            "model": asr_model,
            "port": _port(body, ASR),
            "speculative": speculative,
            "lanes": body.get("asr_lanes", DEFAULT_LANES[ASR]),
            "max_memory": body.get(
                "speech_max_memory",
                _speech_memory(ASR, asr_model, speculative),
            ),
        }
    )
    tts = session.Selection.from_request(
        {
            "task": session.TTS,
            "model": tts_model,
            "port": _port(body, TTS),
            "lanes": body.get("tts_lanes", DEFAULT_LANES[TTS]),
            "max_memory": body.get("speech_max_memory", _speech_memory(TTS, tts_model)),
        }
    )
    model = body.get("llm_model", DEFAULT_LLM)
    if not isinstance(model, str) or not model.strip():
        raise session.SelectionError("llm_model must name a model Splash serves")
    command = body.get("splash") or find_splash()
    if not usable_splash(command):
        raise session.SelectionError(
            "the splash command was not found: run Splash Omni from a Splash "
            "checkout, set SPLASH_BIN, or give its path"
        )
    max_memory = body.get(
        "llm_max_memory",
        default_llm_memory(asr.model, tts.model, asr.speculative),
    )
    if isinstance(max_memory, str):
        max_memory = max_memory.strip() or None
    if max_memory is not None:
        try:
            if serve_options.parse_max_memory(max_memory) is None:
                max_memory = None
        except Exception:
            raise session.SelectionError(
                "llm_max_memory must be auto or a size such as 34G"
            ) from None
    model = model.strip()
    llm = LlmSelection(
        model, command, _port(body, LLM), max_memory, draft_directory(model)
    )
    ports = [asr.port, tts.port, llm.port]
    if len(set(ports)) != len(ports):
        raise session.SelectionError("the three servers need three different ports")
    return {ASR: asr, TTS: tts, LLM: llm}


class VoiceStack:
    """The voice chat's three servers, each a session.ServerSession."""

    def __init__(self, *, allowed_origins=(), session_factory=session.ServerSession):
        self.sessions = {
            ASR: session_factory(allowed_origins=allowed_origins),
            TTS: session_factory(allowed_origins=allowed_origins),
            LLM: session_factory(allowed_origins=allowed_origins, planner=_llm_plan),
        }

    @property
    def running(self):
        return any(
            entry.phase in (*session.RUNNING, session.STOPPING)
            for entry in self.sessions.values()
        )

    def start(self, chosen):
        """Start the servers `chosen` (selections()) names; RuntimeError
        while any of them runs. The speech servers start first: Splash
        sizes its memory with them loaded."""
        if self.running:
            raise RuntimeError("the voice chat servers are running; stop them first")
        started = []
        try:
            for role in ROLES:
                self.sessions[role].start(chosen[role])
                started.append(role)
        except BaseException:
            for role in started:
                self.sessions[role].stop()
            raise

    def stop(self):
        for entry in self.sessions.values():
            entry.stop()

    def shutdown(self):
        for entry in self.sessions.values():
            entry.shutdown()

    def snapshot(self, since=None):
        since = since or {}
        return {
            role: entry.snapshot(since.get(role, 0))
            for role, entry in self.sessions.items()
        }
