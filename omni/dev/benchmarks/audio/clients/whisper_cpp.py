"""Launch and drive whisper.cpp's ``whisper-server`` (ggml, Metal backend).

Install (outside the repository)::

    git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp
    git checkout v1.9.5
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
    cmake --build build -j --config Release          # -> build/bin/whisper-server
    ./models/download-ggml-model.sh large-v3-q5_0    # also large-v3, large-v3-turbo[-q5_0]

The checkout is ``--whisper-cpp-dir`` / ``$WHISPER_CPP_DIR``; ``--model`` is a ggml
file path or a name such as ``large-v3-q5_0`` resolved as ``ggml-<name>.bin`` in
``--whisper-models-dir`` / ``$WHISPER_CPP_MODELS`` (default ``<checkout>/models``).

The server loads the model once at startup and is started with
``--inference-path /v1/audio/transcriptions``, so ``OpenAIAudioClient.transcribe``
drives it unchanged (multipart: file, response_format=json, language=en).
Decoding uses the server's defaults: greedy (best_of 2, beam search off),
temperature 0 with whisper's temperature fallback, timestamps on, flash attention
on. whisper-server holds one global mutex around ``whisper_full`` for every
request, so concurrent requests are serialized; only concurrency 1 is meaningful.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from .openai_audio import OpenAIAudioClient
from .sglang_omni import SglangOmniServer

INFERENCE_PATH = "/v1/audio/transcriptions"

# ggml type names as whisper.cpp's quantize tool and model files spell them.
_QUANT = re.compile(r"-(q[0-9]_[0-9k]|q[0-9]_k(?:_[sm])?|q8_0|f16|f32)$", re.IGNORECASE)


def model_quantization(name: str) -> str:
    """``large-v3-q5_0`` -> ``q5_0``; un-suffixed official ggml files are f16."""
    m = _QUANT.search(Path(name).stem.removeprefix("ggml-"))
    return m.group(1).lower() if m else "f16"


def model_base(name: str) -> str:
    stem = Path(name).stem.removeprefix("ggml-")
    return _QUANT.sub("", stem)


def model_label(name: str) -> str:
    return f"whisper-{model_base(name)} {model_quantization(name)} (ggml, Metal)"


def model_info(name: str, server: "WhisperCppServer") -> dict:
    """What goes into the result JSON's ``model_info`` (report.py shows ``label``)."""
    return {
        "label": model_label(name),
        "runtime": f"whisper.cpp {server.version()} whisper-server, ggml Metal backend",
        "model_file": server.model_path.name,
        "model_bytes": server.model_path.stat().st_size,
        "quantization": model_quantization(name),
        "decoding": "server defaults: greedy (best_of 2), temperature 0 with fallback, "
        "timestamps on, flash attention on",
        "note": "whisper-server serializes requests (global mutex): concurrency 1 only.",
    }


class WhisperCppServer:
    def __init__(
        self,
        checkout: str | Path | None = None,
        *,
        model: str,
        models_dir: str | Path | None = None,
        port: int = 8766,
        host: str = "127.0.0.1",
        threads: int | None = None,
        extra_args: list[str] | None = None,
        log_path: str | Path | None = None,
    ):
        checkout = Path(checkout or os.environ.get("WHISPER_CPP_DIR", "")).expanduser()
        self.exe = checkout / "build" / "bin" / "whisper-server"
        if not self.exe.exists():
            raise FileNotFoundError(
                f"{self.exe} missing; build whisper.cpp (see clients/whisper_cpp.py) and pass "
                "--whisper-cpp-dir or set WHISPER_CPP_DIR"
            )
        self.checkout = checkout
        self.model_path = self.resolve_model(model, models_dir, checkout)
        self.port, self.host = port, host
        self.threads = threads
        self.extra_args = list(extra_args or [])
        self.log_path = Path(log_path) if log_path else None
        self.proc: subprocess.Popen | None = None
        self.client = OpenAIAudioClient(f"http://{host}:{port}")
        self.startup_s: float | None = None

    @staticmethod
    def resolve_model(
        model: str, models_dir: str | Path | None, checkout: Path
    ) -> Path:
        p = Path(model).expanduser()
        if p.is_file():
            return p.resolve()
        d = Path(
            models_dir
            or os.environ.get("WHISPER_CPP_MODELS", "")
            or (checkout / "models")
        ).expanduser()
        p = d / f"ggml-{model}.bin"
        if not p.is_file():
            raise FileNotFoundError(f"whisper.cpp model {p} not found")
        return p.resolve()

    def version(self) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(self.checkout), "describe", "--tags", "--always"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def command(self) -> list[str]:
        cmd = [
            str(self.exe),
            "-m",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--inference-path",
            INFERENCE_PATH,
        ]
        if self.threads:
            cmd += ["-t", str(self.threads)]
        return cmd + self.extra_args

    def start(self, ready_timeout: float = 600.0) -> "WhisperCppServer":
        if not SglangOmniServer.port_free(self.host, self.port):
            new_port = SglangOmniServer.free_port(self.host)
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
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.time() + ready_timeout
            while not self.client.healthy():
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"whisper-server exited with code {self.proc.returncode}; see {self.log_path}"
                    )
                if time.time() > deadline:
                    raise TimeoutError(
                        f"whisper-server not ready after {ready_timeout}s"
                    )
                time.sleep(0.5)
        except Exception:
            self.stop()
            raise
        self.startup_s = time.perf_counter() - t0
        return self

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
