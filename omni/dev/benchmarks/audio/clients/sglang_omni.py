"""Launch and drive an sglang-omni server on Apple silicon.

ASR follows docs/cookbook/qwen3_asr.md#apple-silicon-mlx in the sglang-omni
checkout: ``SGLANG_USE_MLX=1``, ``DYLD_LIBRARY_PATH`` pointing at Homebrew's
``ffmpeg@7``, an MLX-converted checkpoint and
``--asr.engine.max_running_requests N``.  TTS (docs/cookbook/qwen3_tts.md) has
no MLX path; ``kind="tts"`` launches the Torch MPS path with ``SGLANG_USE_MLX``
unset and a PyTorch checkpoint, and needs ``qwen-tts`` installed with
``--no-deps`` in the checkout's venv.

The checkout is ``--sglang-checkout`` or ``$SGLANG_OMNI_DIR``; the venv is
``<checkout>/.venv-apple``.  Requests go through ``OpenAIAudioClient``.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from .openai_audio import OpenAIAudioClient


def ffmpeg7_lib() -> str | None:
    try:
        prefix = subprocess.run(
            ["brew", "--prefix", "ffmpeg@7"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    lib = Path(prefix) / "lib"
    return str(lib) if lib.is_dir() else None


class SglangOmniServer:
    def __init__(
        self,
        checkout: str | Path | None = None,
        *,
        model_path: str,
        model_name: str | None = None,
        kind: str = "asr",
        port: int = 8000,
        host: str = "127.0.0.1",
        max_running_requests: int = 1,
        use_mlx: bool | None = None,
        extra_args: list[str] | None = None,
        log_path: str | Path | None = None,
        config: str | None = None,
    ):
        checkout = Path(checkout or os.environ.get("SGLANG_OMNI_DIR", "")).expanduser()
        if not checkout.is_dir():
            raise FileNotFoundError(
                "sglang-omni checkout not found; pass --sglang-checkout or set SGLANG_OMNI_DIR"
            )
        self.checkout = checkout
        self.exe = checkout / ".venv-apple" / "bin" / "sgl-omni"
        if not self.exe.exists():
            raise FileNotFoundError(
                f"{self.exe} missing; run ./install.sh in the checkout first"
            )
        self.kind = kind
        self.model_path = model_path
        self.model_name = model_name or model_path
        self.port, self.host = port, host
        self.max_running_requests = max_running_requests
        self.use_mlx = (kind == "asr") if use_mlx is None else use_mlx
        self.extra_args = list(extra_args or [])
        self.config = config
        self.log_path = Path(log_path) if log_path else None
        self.proc: subprocess.Popen | None = None
        self.client = OpenAIAudioClient(f"http://{host}:{port}")
        self.startup_s: float | None = None

    def command(self) -> list[str]:
        cmd = [
            str(self.exe),
            "serve",
            "--model-path",
            self.model_path,
            "--model-name",
            self.model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.config:
            cmd += ["--config", self.config]
        if self.kind == "asr":
            cmd += ["--asr.engine.max_running_requests", str(self.max_running_requests)]
        return cmd + self.extra_args

    def env(self) -> dict:
        env = dict(os.environ)
        env.pop("SGLANG_USE_MLX", None)
        if self.use_mlx:
            env["SGLANG_USE_MLX"] = "1"
        lib = ffmpeg7_lib()
        if lib:
            env["DYLD_LIBRARY_PATH"] = lib + (
                ":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else ""
            )
        env["PYTHONUNBUFFERED"] = "1"
        return env

    @staticmethod
    def port_free(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return True
            except OSError:
                return False

    @staticmethod
    def free_port(host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            return s.getsockname()[1]

    def start(self, ready_timeout: float = 900.0) -> "SglangOmniServer":
        # sgl-omni silently moves to another port when the requested one is busy,
        # so pick a free one up front and poll that.
        if not self.port_free(self.host, self.port):
            new_port = self.free_port(self.host)
            print(f"port {self.port} is busy; using {new_port}", file=sys.stderr)
            self.port = new_port
            self.client = OpenAIAudioClient(f"http://{self.host}:{self.port}")
        log = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        cmd = self.command()
        print("launching:", " ".join(cmd), file=sys.stderr)
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.checkout,
            env=self.env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            self.wait_ready(ready_timeout)
        except Exception:
            self.stop()
            raise
        self.startup_s = time.perf_counter() - t0
        return self

    def wait_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"sgl-omni exited with code {self.proc.returncode}; see {self.log_path}"
                )
            if self.client.healthy():
                return
            time.sleep(1.0)
        raise TimeoutError(f"sgl-omni not ready after {timeout}s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            self.proc = None
            return
        pgid = os.getpgid(self.proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            self.proc.wait(timeout=10)
        self.proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def local_snapshot(model: str) -> str:
    """A model directory as given, or a Hub id's cached snapshot (no download)."""
    if Path(model).expanduser().is_dir():
        return str(Path(model).expanduser())
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(model, local_files_only=True)
    except LocalEntryNotFoundError:
        # Downloaded at a pinned revision only: no refs/main. Take the
        # newest cached snapshot.
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == model and repo.revisions:
                return str(
                    max(repo.revisions, key=lambda r: r.last_modified).snapshot_path
                )
        raise


class SglangNativeServer(SglangOmniServer):
    """sglang-omni's native C++ MLX servers (``sglang_omni_mlx/native``).

    Since October 2026 sglang-omni serves Qwen3-ASR on Apple silicon from
    ``qwen3_asr_server`` (``--max-batch N`` batches concurrent requests,
    ``--draft-model-path DIR`` decodes with a DFlash draft) and Qwen3-TTS
    CustomVoice from ``qwen3_tts_server`` (one request at a time, the rest
    queued; ``--max-queued-requests`` is raised to the concurrency so that
    no request gets 503). ``native_dir`` is the install directory
    ``scripts/build_runtime.sh`` wrote (``--sglang-native`` or
    ``$SGLANG_OMNI_NATIVE``); models resolve to their Hub cache snapshots.
    """

    def __init__(self, native_dir: str | Path | None = None, **kwargs):
        native = Path(
            native_dir or os.environ.get("SGLANG_OMNI_NATIVE", "")
        ).expanduser()
        kind = kwargs.get("kind", "asr")
        exe = native / "bin" / f"qwen3_{kind}_server"
        if not exe.exists():
            raise FileNotFoundError(
                f"{exe} missing; build sglang_omni_mlx/native with scripts/build_runtime.sh "
                "and pass --sglang-native or set SGLANG_OMNI_NATIVE"
            )
        # SglangOmniServer.__init__ looks for the Python checkout's venv;
        # set its fields for the native binary instead (start, wait_ready
        # and stop are shared).
        self.checkout = native
        self.exe = exe
        self.kind = kind
        self.model_name = kwargs.get("model_name") or kwargs["model_path"]
        self.model_path = local_snapshot(kwargs["model_path"])
        self.port, self.host = kwargs.get("port", 8000), kwargs.get("host", "127.0.0.1")
        self.max_running_requests = kwargs.get("max_running_requests", 1)
        self.use_mlx = False
        self.extra_args = list(kwargs.get("extra_args") or [])
        self.config = None
        log_path = kwargs.get("log_path")
        self.log_path = Path(log_path) if log_path else None
        self.proc = None
        self.client = OpenAIAudioClient(f"http://{self.host}:{self.port}")
        self.startup_s = None

    def command(self) -> list[str]:
        cmd = [
            str(self.exe),
            "--model-path",
            self.model_path,
            "--model-name",
            self.model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.kind == "tts":
            cmd += ["--max-queued-requests", str(max(2, self.max_running_requests))]
        return cmd + self.extra_args

    def env(self) -> dict:
        return dict(os.environ)
