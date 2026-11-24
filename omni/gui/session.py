"""What the setup page selects, and the `splash-omni serve` process it starts.

The page never serves a model itself: a start runs the launcher's own
`splash-omni serve` with the options chosen, which builds what a source checkout
lacks, downloads the selected checkpoint (and, for transcription with
speculative decoding, its draft) through the installer, and execs the
server. This module watches that process: its output, its phase and, once
it loads, the server's /ready.

The page never asks the Hub about a model it already has (plan()): it
decides what is here with the installer's own functions (upstream.present,
upstream.draft_present), which `splash-omni serve` starts from without a Hub
request. When everything is here it serves with --offline, so neither the
installer nor the server can ask; only a missing checkpoint is downloaded
(by `splash-omni serve`), and a missing draft beside a present checkpoint is
fetched alone first, with the server's own download helper.

Standard library only, with the installer and server modules that are too:
`splash-omni ui` runs before .venv exists.
"""

from __future__ import annotations

import codecs
import collections
import http.client
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

from install import families, models, paths, upstream
from server import serve_options

from . import progress

TTS, ASR = "tts", "asr"
TASKS = (TTS, ASR)
# The most lanes each family's engine takes (`splash-omni serve --help`).
MAX_LANES = {TTS: 16, ASR: 32}


@dataclass(frozen=True)
class Variant:
    """A Qwen3-TTS model type, as the setup page offers it: its config's
    tts_model_type (what the server's /v1/models reports), its name and
    what it does with a voice."""

    id: str
    name: str
    summary: str


BASE, VOICE_DESIGN, CUSTOM_VOICE = "base", "voice_design", "custom_voice"
VARIANTS = (
    Variant(
        BASE,
        "Base",
        "Speaks the text in a voice it picks itself: no system prompt and no "
        "speaker. Paragraph mode keeps the first paragraph's voice: the "
        "others continue it.",
    ),
    Variant(
        VOICE_DESIGN,
        "VoiceDesign",
        "Describe the voice in words (timbre, rhythm, style) in a system "
        "prompt, which every request needs.",
    ),
    Variant(
        CUSTOM_VOICE,
        "CustomVoice",
        "Nine preset speakers, with an optional system prompt for style or emotion.",
    ),
)
VARIANT = {variant.id: variant for variant in VARIANTS}


@dataclass(frozen=True)
class Checkpoint:
    """One of the supported checkpoints, as the setup page offers it."""

    model: str
    task: str
    precision: str
    # The snapshot's size, as the page tells what a download costs.
    gigabytes: float
    # Speech: the model type (VARIANTS); None for transcription.
    variant: str | None = None
    # The model's size as its name gives it.
    size: str = "1.7B"

    @property
    def label(self):
        """The name the page shows: model, size, type and precision, so that
        no two checkpoints read alike."""
        if self.task == ASR:
            return f"Qwen3-ASR {self.size} · {self.precision}"
        return f"Qwen3-TTS {self.size} {VARIANT[self.variant].name} · {self.precision}"

    @property
    def draft(self):
        """The DFlash draft a transcription model decodes with by default
        (serve_options.DEFAULT_DRAFTS), None for speech."""
        if self.task != ASR:
            return None
        return (
            serve_options.SMALL_DRAFT_MODEL
            if self.size == "0.6B"
            else serve_options.DEFAULT_DRAFT_MODEL
        )


# families.SUPPORTED_MODELS, by task, model type and precision.
CHECKPOINTS = (
    Checkpoint("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit", TTS, "4-bit", 2.2, BASE),
    Checkpoint("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16", TTS, "bf16", 4.2, BASE),
    Checkpoint("mlx-community/Qwen3-ASR-1.7B-4bit", ASR, "4-bit", 1.5),
    Checkpoint("mlx-community/Qwen3-ASR-1.7B-bf16", ASR, "bf16", 3.8),
    Checkpoint(
        "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
        TTS,
        "4-bit",
        2.2,
        VOICE_DESIGN,
    ),
    Checkpoint(
        "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        TTS,
        "bf16",
        4.2,
        VOICE_DESIGN,
    ),
    Checkpoint(
        "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
        TTS,
        "4-bit",
        2.2,
        CUSTOM_VOICE,
    ),
    Checkpoint(
        "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        TTS,
        "bf16",
        4.2,
        CUSTOM_VOICE,
    ),
    Checkpoint(
        "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit", TTS, "4-bit", 1.6, BASE, "0.6B"
    ),
    Checkpoint(
        "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
        TTS,
        "4-bit",
        1.6,
        CUSTOM_VOICE,
        "0.6B",
    ),
    Checkpoint("mlx-community/Qwen3-ASR-0.6B-4bit", ASR, "4-bit", 0.7, size="0.6B"),
)
CHECKPOINT = {checkpoint.model: checkpoint for checkpoint in CHECKPOINTS}
# Each default draft's size on disk (Checkpoint.draft).
DRAFT_GIGABYTES = upstream.DRAFT_GIGABYTES

assert {c.model for c in CHECKPOINTS} == set(families.SUPPORTED_MODELS)


class SelectionError(ValueError):
    """A setup the page cannot start, with the reason it shows."""


def _flag(body, name, default):
    value = body.get(name, default)
    if not isinstance(value, bool):
        raise SelectionError(f"{name} must be true or false")
    return value


@dataclass(frozen=True)
class Selection:
    """A setup to serve: the task, the checkpoint and the serve options the
    page exposes. Everything else is `splash-omni serve`'s default."""

    task: str
    model: str
    # Transcription: decode with the default draft (serve's default), or
    # plainly (--no-draft-model).
    speculative: bool = True
    port: int = serve_options.DEFAULT_PORT
    lanes: int | None = None
    # --max-memory as typed, such as 28G; None for auto.
    max_memory: str | None = None
    disable_ane: bool = False
    offline: bool = False

    @classmethod
    def from_request(cls, body):
        """The selection a setup request names; SelectionError for one the
        launcher would refuse."""
        if not isinstance(body, dict):
            raise SelectionError("the setup must be a JSON object")
        task = body.get("task")
        if task not in TASKS:
            raise SelectionError("task must be tts or asr")
        model = body.get("model")
        checkpoint = CHECKPOINT.get(model)
        if checkpoint is None:
            raise SelectionError(
                "model must be one of " + families.supported_models_text()
            )
        if checkpoint.task != task:
            raise SelectionError(f"{model} does not serve {task}")
        port = body.get("port", serve_options.DEFAULT_PORT)
        if type(port) is not int or not 1 <= port <= 65535:
            raise SelectionError("port must be a number from 1 to 65535")
        lanes = body.get("lanes")
        if lanes is not None:
            if type(lanes) is not int or not 1 <= lanes <= MAX_LANES[task]:
                raise SelectionError(
                    f"lanes must be a number from 1 to {MAX_LANES[task]} "
                    "(or empty for the engine's default)"
                )
        max_memory = body.get("max_memory")
        if isinstance(max_memory, str):
            max_memory = max_memory.strip() or None
        if max_memory is not None:
            if not isinstance(max_memory, str):
                raise SelectionError("max_memory must be text such as 28G")
            try:
                if serve_options.parse_max_memory(max_memory) is None:
                    max_memory = None
            except Exception:
                raise SelectionError(
                    "max_memory must be auto or a size such as 28G"
                ) from None
        return cls(
            task=task,
            model=model,
            speculative=_flag(body, "speculative", True),
            port=port,
            lanes=lanes,
            max_memory=max_memory,
            disable_ane=_flag(body, "disable_ane", False),
            offline=_flag(body, "offline", False),
        )

    @property
    def draft(self):
        """The draft the server decodes with, or None: the default draft of
        its Qwen3-ASR size unless speculation is off. Speech models have
        none."""
        if self.task == ASR and self.speculative:
            return CHECKPOINT[self.model].draft
        return None

    def downloads(self):
        """The repositories this start fetches into the Hub cache when they
        are missing: the selected checkpoint, and its draft when it has one.
        No other checkpoint is downloaded."""
        return (self.model,) + ((self.draft,) if self.draft else ())

    def serve_arguments(self, allowed_origins=(), *, offline=None):
        """`splash-omni serve`'s command line for this selection, with --offline
        when `offline` (default: the selection's own choice). The origins
        are the setup page's own, admitted so that it may also call the API
        directly from the browser."""
        argv = ["serve", "--model", self.model, "--port", str(self.port)]
        if self.offline if offline is None else offline:
            argv.append("--offline")
        if self.lanes is not None:
            argv += ["--lanes", str(self.lanes)]
        if self.max_memory is not None:
            argv += ["--max-memory", self.max_memory]
        if self.disable_ane:
            argv.append("--disable-ane")
        if self.task == ASR and not self.speculative:
            argv.append("--no-draft-model")
        for origin in allowed_origins:
            argv += ["--allowed-origin", origin]
        return argv

    def command(self, allowed_origins=(), python=None, *, offline=None):
        """The process that serves this selection: the launcher, run as the
        `splash-omni` command runs it."""
        return [
            python or sys.executable,
            "-u",
            str(paths.ROOT / "install/launcher.py"),
            *self.serve_arguments(allowed_origins, offline=offline),
        ]

    def shown_command(self, *, offline=None):
        """The command a terminal would run for this selection."""
        return shlex.join(["splash-omni", *self.serve_arguments(offline=offline)])

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"


def model_present(model):
    """Whether `splash-omni serve --model MODEL` starts without the Hub: the
    installer's answer (upstream.present: installed and verified, or whole
    in the Hugging Face cache)."""
    return upstream.present(models.Selection.of(paths.MODELS, model))


def draft_present(draft):
    """Whether the cache holds the draft whole, where the server finds it
    (upstream.draft_present, the installer's answer)."""
    return upstream.draft_present(draft)


# Fetches one repository into the Hub cache with the helper the installer
# (upstream.fetch_draft) and the server use, run by .venv's Python.
FETCH = """
import sys
from server import serve_options
repo = sys.argv[1]
print(f"Downloading {repo}, the speculative draft (about 0.7 GB).", flush=True)
try:
    serve_options.download_model(repo)
except Exception as error:
    print(f"error: cannot download {repo}: {error}", flush=True)
    raise SystemExit(1)
print(f"Downloaded {repo}.", flush=True)
"""


@dataclass(frozen=True)
class Plan:
    """How a start gets what it serves without asking the Hub about what
    the machine already has."""

    # The repositories missing locally, which this start downloads.
    missing: tuple[str, ...]
    # The repositories the page fetches itself before `splash-omni serve`.
    fetch: tuple[str, ...]
    # Whether `splash-omni serve` runs with --offline: everything is local.
    offline: bool


def plan(selection, *, model_present=model_present, draft_present=draft_present):
    """The plan of a start of `selection`. SelectionError for an offline
    start of something that is not here."""
    missing = tuple(
        repo
        for repo in selection.downloads()
        if not (model_present(repo) if repo == selection.model else draft_present(repo))
    )
    if selection.offline:
        if missing:
            raise SelectionError(
                f"{', '.join(missing)} is neither installed nor in the Hugging "
                "Face cache; turn Offline off to download it"
            )
        return Plan((), (), True)
    if selection.model in missing:
        # serve resolves and downloads the checkpoint, and the draft with it.
        return Plan(missing, (), False)
    if missing and not paths.PYTHON.is_file():
        # No environment to fetch the draft with yet: serve sets it up.
        return Plan(missing, (), False)
    return Plan(missing, missing, True)


# The session's phases. A start runs PREPARING (build, device check and
# download, the launcher's work) and LOADING (the server loading the model)
# to READY; a stop goes through STOPPING to IDLE; a process that ends on its
# own leaves FAILED.
IDLE = "idle"
PREPARING = "preparing"
LOADING = "loading"
READY = "ready"
STOPPING = "stopping"
FAILED = "failed"
RUNNING = (PREPARING, LOADING, READY)

# The server's status lines start with the time (server/diagnostics.py).
_STATUS = re.compile(r"^\d\d:\d\d:\d\d (?P<status>\w+) · (?P<detail>.*)$")
# What the launcher and installer print while they prepare the model.
_PREPARE_STEPS = (
    ("Fetching ", "download"),
    ("Downloading ", "download"),
    ("Installing ", "install"),
    ("Splash model ", "install"),
    ("Reinstalling ", "install"),
    ("Using ", "install"),
)
LOG_LINES = 2000
# Seconds a stop waits for the server to release the engine before it is
# killed.
STOP_GRACE = 30


def classify(line):
    """What a line of `splash-omni serve`'s output says: ("loading" | "ready" |
    "error" | "download" | "install" | "build", detail), or None."""
    if match := _STATUS.match(line):
        status, detail = match.group("status"), match.group("detail")
        if status == "Loading":
            return "loading", detail
        if status == "Ready":
            return "ready", detail
        if status == "Error":
            return "error", detail
        if status == "Downloading":
            return "download", detail
        return None
    if line.startswith("error: "):
        return "error", line.removeprefix("error: ")
    for prefix, step in _PREPARE_STEPS:
        if line.startswith(prefix):
            return step, line
    if line.startswith(("make", "xcrun", "Another Splash build")) or " clang++ " in (
        f" {line} "
    ):
        return "build", "Building Splash (first run only)"
    return None


class OutputLog:
    """The last LOG_LINES lines of a process's output, numbered. Text a
    carriage return ends (a progress bar redrawing itself) is the live line,
    which the next such text replaces, as a terminal shows it; a newline
    commits it."""

    def __init__(self, limit=LOG_LINES):
        self.lines = collections.deque(maxlen=limit)
        self.sequence = 0
        self.live = ""
        self._partial = ""

    def feed(self, text):
        """Add output; the lines it commits, in order."""
        committed = []
        for piece in re.split(r"(\r\n|\r|\n)", self._partial + text):
            if piece == "\r":
                if self._partial:
                    self.live, self._partial = self._partial, ""
            elif piece in ("\r\n", "\n"):
                line = self._partial or self.live
                self._partial = self.live = ""
                self.sequence += 1
                self.lines.append((self.sequence, line))
                committed.append(line)
            else:
                self._partial = piece
        return committed

    @property
    def current(self):
        """The line being written: the text since the last line ended, else
        the live line."""
        return self._partial or self.live

    def since(self, sequence):
        return [
            {"seq": number, "text": line}
            for number, line in self.lines
            if number > sequence
        ]


def _ready(base_url, timeout=2.0):
    """Whether the server at base_url answers /ready with 200. Asked
    directly, never through a proxy the environment names."""
    parts = urlsplit(base_url)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    try:
        connection.request("GET", "/ready")
        return connection.getresponse().status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


class ServerSession:
    """The one `splash-omni serve` process the page starts, stops and watches."""

    def __init__(
        self,
        *,
        allowed_origins=(),
        popen=subprocess.Popen,
        ready=_ready,
        monitor=progress.DownloadMonitor,
        poll_seconds=0.5,
        planner=None,
    ):
        self.allowed_origins = tuple(allowed_origins)
        self._popen = popen
        self._ready = ready
        self._monitor_type = monitor
        self._poll = poll_seconds
        if planner is not None:
            self._plan = planner
        self._lock = threading.RLock()
        self.phase = IDLE
        self.selection = None
        self.process = None
        self.log = OutputLog()
        self.step = None
        self.detail = ""
        self.error = None
        self.started_at = None
        self.ready_at = None
        self.exit_status = None
        self.monitor = None
        self.plan = None
        self._stop_requested = False
        # The commands still to run after the current one ends well, and
        # the thread reading the current one's output.
        self._pending = []
        self._reader = None

    # The page's actions.

    def start(self, selection):
        """Start serving `selection`; RuntimeError while a server runs."""
        with self._lock:
            if self.phase in (*RUNNING, STOPPING):
                raise RuntimeError(
                    "a server is already running; stop it before starting another"
                )
            # Raises SelectionError for an offline start of a missing model.
            chosen = self._plan(selection)
            serve = selection.command(self.allowed_origins, offline=chosen.offline)
            fetches = [
                [str(paths.PYTHON), "-u", "-c", FETCH, repo] for repo in chosen.fetch
            ]
            self.plan = chosen
            self.selection = selection
            self.log = OutputLog()
            self.step, self.detail = "check", "Checking the installation"
            self.error = None
            self.exit_status = None
            self.ready_at = None
            self._stop_requested = False
            self.started_at = time.time()
            # Only what is missing is listed on the Hub.
            self.monitor = self._monitor_type(
                selection.downloads(), offline=not chosen.missing, listed=chosen.missing
            )
            self.monitor.start()
            self._pending = [*fetches[1:], serve] if fetches else []
            self._launch(fetches[0] if fetches else serve)
            self.phase = PREPARING

    def _launch(self, command):
        """Run one of the start's commands, with its output and exit
        watched. A session of its own, so that a stop reaches every process
        it starts (make, the installer, the server and its engine) as
        Ctrl+C in its terminal would."""
        environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=str(paths.ROOT))
        cwd = str(paths.ROOT)
        # Another program's server (the voice chat's Splash) runs from its
        # own directory, without this checkout on its path.
        if hasattr(self.selection, "process_environment"):
            environment, cwd = self.selection.process_environment()
        # The page names the port; the launcher's default is not read.
        environment.pop("SPLASH_PORT", None)
        process = self._popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.process = process
        self._reader = threading.Thread(target=self._read, args=(process,), daemon=True)
        self._reader.start()
        threading.Thread(target=self._watch, args=(process,), daemon=True).start()

    def _plan(self, selection):
        return plan(selection)

    def stop(self):
        """Stop the server as Ctrl+C would; returns at once, and the session
        reaches IDLE once the process has exited."""
        with self._lock:
            if self.phase not in RUNNING:
                if self.phase == FAILED:
                    self.phase = IDLE
                return
            self.phase = STOPPING
            self._stop_requested = True
            self.detail = "Stopping the server"
            process = self.process
        self._signal(process, signal.SIGINT)
        threading.Thread(target=self._reap, args=(process,), daemon=True).start()

    def shutdown(self, timeout=STOP_GRACE):
        """Stop the server and wait for it to exit (the page is closing)."""
        process = self.process
        self.stop()
        if process is not None:
            try:
                process.wait(timeout)
            except subprocess.TimeoutExpired:
                self._signal(process, signal.SIGKILL)
                process.wait()

    def snapshot(self, since=0):
        """The session as the page shows it, with the log lines after
        `since`."""
        with self._lock:
            selection = self.selection
            return {
                "phase": self.phase,
                "step": self.step,
                "detail": self.detail,
                "error": self.error,
                "exit_status": self.exit_status,
                "selection": None if selection is None else asdict(selection),
                "command": None
                if selection is None
                else selection.shown_command(offline=self.plan.offline),
                "fetch": [] if self.plan is None else list(self.plan.fetch),
                "base_url": None if selection is None else selection.base_url,
                "downloads": [] if self.monitor is None else self.monitor.snapshot(),
                "started_at": self.started_at,
                "ready_at": self.ready_at,
                "log": self.log.since(since),
                "log_seq": self.log.sequence,
                "live": self.log.current,
            }

    # The watchers.

    def _signal(self, process, number):
        try:
            os.killpg(process.pid, number)
        except (ProcessLookupError, PermissionError):
            try:
                process.send_signal(number)
            except (ProcessLookupError, OSError):
                pass

    def _reap(self, process):
        try:
            process.wait(STOP_GRACE)
        except subprocess.TimeoutExpired:
            self._signal(process, signal.SIGKILL)

    def _read(self, process):
        # A read may end inside a character (the server's "·").
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        with process.stdout as stream:
            while data := stream.read1(65536):
                text = decoder.decode(data)
                with self._lock:
                    if process is not self.process:
                        return
                    for line in self.log.feed(text):
                        self._observe(line)

    def _observe(self, line):
        """Follow the process's phase in one line of its output."""
        found = classify(line.strip())
        if found is None:
            return
        kind, detail = found
        if kind == "error":
            # A served request's error is the request's; only a start's
            # error is the session's.
            if self.phase in (PREPARING, LOADING):
                self.error = detail
            return
        if kind == "loading" and self.phase == PREPARING:
            self.phase = LOADING
            self.step, self.detail = "load", f"Loading {detail}"
        elif kind == "ready":
            self.step, self.detail = "load", f"Ready · {detail}"
        elif kind == "download" and self.phase in (PREPARING, LOADING):
            self.step, self.detail = "download", detail
        elif self.phase == PREPARING:
            self.step, self.detail = kind, detail

    def _watch(self, process):
        while True:
            status = process.poll()
            if status is not None:
                break
            with self._lock:
                phase, selection = self.phase, self.selection
            if phase == LOADING and self._ready(selection.base_url):
                with self._lock:
                    if self.process is process and self.phase == LOADING:
                        self.phase = READY
                        self.step = "ready"
                        self.detail = (
                            f"Serving {selection.model} at {selection.base_url}"
                        )
                        self.ready_at = time.time()
            time.sleep(self._poll)
        reader = self._reader
        if reader is not None and reader.is_alive():
            # Every line of a step before the next one starts.
            reader.join(2)
        with self._lock:
            if self.process is not process:
                return
            if status == 0 and self._pending and not self._stop_requested:
                self._launch(self._pending.pop(0))
                return
            self._pending = []
            self.exit_status = status
            if self.monitor is not None:
                self.monitor.stop()
            if self._stop_requested:
                self.phase, self.step, self.detail = IDLE, None, "Server stopped"
                return
            if self.error is None or self.phase == READY:
                self.error = f"splash-omni serve exited with status {status}"
            self.phase = FAILED
            self.detail = self.error
