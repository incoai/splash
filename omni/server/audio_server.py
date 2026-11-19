"""The audio server: HTTP routes, speech and transcription responses, and
the serving process's startup.

`python -m server.audio_server MODEL_ROOT --tokenizer DIR --model ID
--binary PATH --port P [options]` serves POST /v1/audio/speech (OpenAI's
shape; a WAV or PCM answer, complete or streamed as it is decoded), POST
/v1/audio/transcriptions (multipart; json, text or verbose_json), and GET
/health, /ready, /status, /metrics, /v1/models and /v1/audio/voices, over
the HTTP server, admission, security, errors, metrics and latency of
http_server.py.
"""

import argparse
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path

from transformers import AutoTokenizer

from . import audio_io, audio_runtime, json_codec, serve_options
from .audio_backend import AudioBackend, print_audio_request
from .audio_frontend import AudioFrontend, ModelProfile, parse_multipart
from .diagnostics import CONSOLE, print_status
from .errors import APIError
from .http_server import (
    NATIVE_START_TIMEOUT,
    FrontendHandler,
    FrontendServer,
    PostRoute,
    interrupt,
    raise_descriptor_limit,
)
from .latency import RequestLatency
from .metrics import prometheus_metrics
from .origins import ANY_ORIGIN

ROOT = Path(__file__).parents[1]
AUDIO_ROUTES = {
    "/v1/audio/speech": PostRoute("_post_speech"),
    "/v1/audio/transcriptions": PostRoute("_post_transcriptions"),
}
# SPLASH_AUDIO_TRACE=1 prints each transcription's instants on stderr
# (dev/tools/omni_request_trace.py joins them with the engine's).
TRACE = os.environ.get("SPLASH_AUDIO_TRACE") == "1"
# The status of a server that stopped relaunching its engine, apart from a
# stop's 0 and a failed start's 1: what supervises the server, as launchd's
# KeepAlive does, decides whether to start it again. 69 is sysexits'
# EX_UNAVAILABLE.
ENGINE_STOPPED_STATUS = os.EX_UNAVAILABLE
# How long an exiting server waits for its last console lines to be written.
CONSOLE_FLUSH_SECONDS = 2.0
_CONTENT_TYPES = {"wav": "audio/wav", "pcm": "audio/pcm"}


def _audio_headers(result, pcm_bytes, first_audio_ms):
    headers = [
        ("X-Splash-Frames", str(result.completion_count)),
        (
            "X-Splash-Audio-Seconds",
            f"{pcm_bytes / 2 / audio_io.TTS_SAMPLE_RATE:.3f}",
        ),
        ("X-Splash-Finish-Reason", result.reason),
    ]
    if first_audio_ms is not None:
        headers.append(("X-Splash-Time-To-First-Audio-Ms", f"{first_audio_ms:.1f}"))
    return headers


class AudioHandler(FrontendHandler):
    post_routes = AUDIO_ROUTES
    # A page continuing speech reads its id.
    exposed_headers = FrontendHandler.exposed_headers + ", X-Splash-Speech-Id"

    def do_GET(self):
        path = self.route
        if path == "/health":
            self._json(200, {"status": "ok"})
        elif path == "/ready":
            ready = self.app.backend.is_ready()
            self._json(
                200 if ready else 503, {"status": "ready" if ready else "unavailable"}
            )
        elif path == "/status":
            self._json(200, self.server.status())
        elif path == "/metrics":
            self._send(
                200,
                prometheus_metrics(self.server.status()).encode(),
                "text/plain; version=0.0.4; charset=utf-8",
            )
        elif path == "/v1/models":
            self._json(200, {"object": "list", "data": self.app.models()})
        elif path.startswith("/v1/models/"):
            name = path.removeprefix("/v1/models/")
            model = next(
                (item for item in self.app.models() if item["id"] == name), None
            )
            if model is None:
                self._safe_error(APIError(404, "model not found", "model_not_found"))
            else:
                self._json(200, model)
        elif path == "/v1/audio/voices":
            self._json(200, self.app.voices)
        else:
            self._safe_error(APIError(404, "not found", "not_found"))

    def do_DELETE(self):
        self._safe_error(APIError(404, "not found", "not_found"))

    def _require_multipart_content_type(self):
        content_types = self.headers.get_all("Content-Type", [])
        if (
            len(content_types) != 1
            or self.headers.get_content_type().lower() != "multipart/form-data"
        ):
            raise APIError(415, "Content-Type must be multipart/form-data")

    def _read_multipart_body(self, deadline):
        """The request's body, a multipart form, read by `deadline`."""
        payload = self._read_body(deadline, self._require_multipart_content_type)
        boundary = self.headers.get_param("boundary")
        if not isinstance(boundary, str):
            raise APIError(400, "multipart boundary is missing")
        return parse_multipart(payload, boundary)

    def _post_speech(self, started_at):
        with self.app.latencies.measure("upload"):
            body = self._read_json_body(started_at + self.app.request_timeout)
        plan = self.app.prepare_speech(
            body, deadline=self.app.request_deadline(started_at)
        )
        job = plan.job
        job.latency = RequestLatency(self.app.latencies, started_at)
        # The speech id is pending until the request ends; then its text and
        # frames are kept for later requests' reference_speech_id, unless it
        # failed.
        self.app.begin_speech(plan)
        result = None
        try:
            self._submit(job)
            if plan.stream:
                result = self._speech_stream(plan, started_at)
            else:
                result = self._speech_complete(plan, started_at)
        finally:
            self.app.finish_speech(plan, result)

    def _speech_complete(self, plan, started_at):
        job = plan.job
        chunks = []
        first_audio_ms = None
        edges = audio_io.SpeechEdges(bool(job.frame.reference_codes))
        while True:
            kind, value = self._next_event(job)
            if kind == "audio":
                if first_audio_ms is None:
                    first_audio_ms = (time.monotonic() - started_at) * 1000.0
                chunks.append(edges.push(value))
            elif kind == "done":
                result = value
                chunks.append(edges.finish())
                break
        pcm = b"".join(chunks)
        data = audio_io.wav_bytes(pcm) if plan.response_format == "wav" else pcm
        self._send(
            200,
            data,
            _CONTENT_TYPES[plan.response_format],
            [("X-Splash-Speech-Id", plan.speech_id)]
            + _audio_headers(result, len(pcm), first_audio_ms),
        )
        return result

    def _speech_stream(self, plan, started_at):
        """The audio as it is decoded: a WAV of unknown length, or raw PCM,
        in chunked transfer encoding (HTTP/1.0 clients read to the close).
        The response starts with the first chunk, so a request that fails
        before any audio gets its error status; one that fails after is
        cut short, without the final chunk. The request's result, or None
        when it did not finish."""
        job = plan.job
        wav = plan.response_format == "wav"
        chunked = self.request_version != "HTTP/1.0"
        edges = audio_io.SpeechEdges(bool(job.frame.reference_codes))
        started = False
        result = None

        def write(data):
            if chunked:
                self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
            else:
                self.wfile.write(data)
            self.wfile.flush()

        def begin():
            nonlocal started
            started = True
            self.send_response(200)
            self.send_header("Content-Type", _CONTENT_TYPES[plan.response_format])
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Splash-Speech-Id", plan.speech_id)
            self.send_header(
                "X-Splash-Time-To-First-Audio-Ms",
                f"{(time.monotonic() - started_at) * 1000.0:.1f}",
            )
            self._response_started = True
            self.end_headers()
            if wav:
                write(audio_io.wav_header(None))

        def run():
            nonlocal result
            while True:
                kind, value = self._next_event(job)
                if kind == "audio":
                    if not started:
                        begin()
                    if data := edges.push(value):
                        write(data)
                elif kind == "done":
                    if not started:
                        begin()
                    if data := edges.finish():
                        write(data)
                    if chunked:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                    result = value
                    return

        # An error after the audio began cannot be told inside it: the
        # stream ends without its final chunk, which the client sees as cut
        # short.
        self._guarded_stream(job, run, lambda error: None)
        return result

    def _post_transcriptions(self, started_at):
        self._segment_jobs = []
        marks = {"arrival": started_at} if TRACE else None
        with self.app.latencies.measure("upload"):
            form = self._read_multipart_body(started_at + self.app.request_timeout)
        if marks is not None:
            marks["uploaded"] = time.monotonic()
        plan = self.app.prepare_transcription(
            form,
            deadline=self.app.request_deadline(started_at),
            reserve=self._body_reservation.grow,
        )
        del form
        if marks is not None:
            marks["prepared"] = time.monotonic()
        # Every segment is submitted before any is collected, so they decode
        # concurrently on the engine's lanes; their transcripts join in order.
        jobs = []
        for segment in plan.segments:
            job = self.app.transcription_job(plan, segment)
            job.latency = RequestLatency(self.app.latencies, started_at)
            if marks is not None:
                job.marks = {}
            self._submit(job)
            jobs.append(job)
            self._segment_jobs.append(job)
        transcripts = []
        for job in jobs:
            tokens = self._collect_tokens(job)
            if job.marks is not None:
                job.marks["collected"] = time.monotonic()
            transcripts.append(self.app.transcript(tokens, plan.language))
        body, content_type = self.app.transcription_response(plan, transcripts)
        data = (
            body.encode("utf-8") if isinstance(body, str) else json_codec.encode(body)
        )
        self._send(
            200,
            data,
            content_type,
            [
                ("X-Splash-Audio-Seconds", f"{plan.duration:.3f}"),
                ("X-Splash-Segments", str(len(plan.segments))),
                ("X-Splash-Output-Tokens", str(sum(t.tokens for t in transcripts))),
            ],
        )
        if marks is not None:
            marks["sent"] = time.monotonic()
            _print_trace(marks, jobs)

    def _cancel_submitted(self):
        # A transcription's segments are all in flight: a failure cancels
        # every one of them.
        jobs = getattr(self, "_segment_jobs", [])
        for job in jobs:
            if job is not self._submitted:
                self.app.backend.cancel(job)
        self._segment_jobs = []
        super()._cancel_submitted()

    def _collect_tokens(self, job):
        tokens = []
        while True:
            kind, value = self._next_event(job)
            if kind == "tokens":
                tokens.extend(value)
            elif kind == "done":
                return tokens


def _print_trace(marks, jobs):
    """One line per transcription: its instants, then each segment's."""
    parts = [f"{name}={value:.6f}" for name, value in marks.items()]
    for index, job in enumerate(jobs):
        for name, value in (job.marks or {}).items():
            text = str(value) if name == "engine_id" else f"{value:.6f}"
            parts.append(f"s{index}.{name}={text}")
    sys.stderr.write("audio trace " + " ".join(parts) + "\n")


class AudioServer(FrontendServer):
    handler_type = AudioHandler


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_root",
        metavar="MODEL_DIRECTORY",
        help="installed model directory holding config.json and target/",
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--model",
        type=serve_options.parse_model_id,
        required=True,
        metavar="OWNER/REPO",
    )
    parser.add_argument("--port", type=int, default=serve_options.DEFAULT_PORT)
    parser.add_argument("--binary", default=str(ROOT / "build" / "splash"))
    serve_options.add_serve_arguments(parser)
    args = parser.parse_args(argv)
    serve_options.check_serve_arguments(parser, args)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    return args


def _draft_directory(args, profile):
    """The directory of the draft serve-omni decodes with, or None. By
    default a Qwen3-ASR checkpoint takes its DFlash draft
    (serve_options.default_draft_model) and speech models none; --draft-model names another and --no-draft-model
    none. A repository ID missing from the Hub cache is downloaded into it:
    the default draft's failure warns and serves without speculation, an
    explicit draft's ends the startup. The engine still serves without a
    draft its memory budget cannot fit."""
    if args.no_draft_model:
        return None
    explicit = args.draft_model is not None
    if explicit and profile.kind.name != "ASR":
        raise audio_runtime.EngineUnhealthy("--draft-model takes a transcription model")
    name = (
        args.draft_model
        if explicit
        else serve_options.default_draft_model(args.model_root)
    )
    if name is None:
        for flag in ("--draft-tokens", "--speculative-lanes"):
            if getattr(args, flag[2:].replace("-", "_")) is not None:
                raise audio_runtime.EngineUnhealthy(
                    f"{flag} needs a draft model: Qwen3-ASR has one by "
                    "default, unless --no-draft-model"
                )
        return None
    if os.path.isdir(name):
        directory = name
    elif (directory := serve_options.hub_snapshot(name)) is None:
        print_status(
            f"Downloading · draft {name} for speculative transcription"
            + ("" if explicit else " (about 0.7 GB); --no-draft-model skips it")
        )
        try:
            directory = serve_options.download_model(name)
        except Exception as error:
            # Offline, unreachable or refused: whatever the Hub client raises.
            reason = " ".join(str(error).split()) or type(error).__name__
            if explicit:
                raise audio_runtime.EngineUnhealthy(
                    f"cannot download draft model {name}: {reason}"
                ) from None
            print_status(
                f"Warning · cannot download draft {name} ({reason}); serving "
                "without speculative decoding (--no-draft-model silences this)",
                error=True,
            )
            return None
    print_status(
        f"Draft · {name} · speculative transcription"
        + ("" if explicit else " by default; --no-draft-model turns it off")
    )
    return directory


def _native_command(args, draft=None):
    """The engine's command line: the shared options it takes, and the
    draft directory _draft_directory chose."""
    return audio_runtime.engine_command(
        args.binary,
        args.model_root,
        max_memory=args.max_memory,
        lanes=args.lanes,
        idle_release=(
            None
            if args.idle_release is None
            else serve_options.idle_release_text(args.idle_release)
        ),
        disable_ane=args.disable_ane,
        draft_model=draft,
        draft_tokens=None if draft is None else args.draft_tokens,
        speculative_lanes=None if draft is None else args.speculative_lanes,
    )


def main():
    args = parse_args()
    server = None
    runtime = None
    backend = None
    # Set once the backend stops relaunching its engine for good.
    engine_stopped = threading.Event()
    # A server started in the background from a non-interactive shell inherits
    # SIGINT as ignored and Python then leaves it alone; install both stop
    # signals explicitly so scripts and supervisors can interrupt it.
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        # The launcher blocks both across its exec: one sent while this module
        # imported arrives here and ends the startup cleanly.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
        raise_descriptor_limit()
        # Bind before loading the tokenizer or model so duplicates fail early.
        # Activate only after the engine is ready, keeping a partially started
        # service from receiving requests.
        server = AudioServer(
            (args.host, args.port),
            None,
            bind_and_activate=False,
            request_capacity=args.queue_size,
            allowed_hosts=args.allowed_host,
            api_key=args.api_key,
            max_request_bytes=args.max_request_size,
            allowed_origins=args.allowed_origin,
        )
        server.server_bind()
        if ANY_ORIGIN in args.allowed_origin and args.api_key is None:
            print_status(
                "Warning · --allowed-origin '*' without --api-key lets every web "
                "page open in a browser that reaches this server use it",
                error=True,
            )
        try:
            profile = ModelProfile.load(args.model_root)
        except ValueError as error:
            raise audio_runtime.EngineUnhealthy(str(error)) from None
        print_status(f"Loading · {args.model} · {profile.family}")
        draft = _draft_directory(args, profile)
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, local_files_only=True, trust_remote_code=False
        )
        runtime = audio_runtime.AudioRuntime(
            _native_command(args, draft),
            startup_timeout=NATIVE_START_TIMEOUT,
            pending_limit=args.queue_size,
        )
        backend = AudioBackend(runtime, request_logger=print_audio_request)

        def stop_serving():
            # From the thread that saw the last failure, which may be one
            # serving a request: shutdown waits for serve_forever to return.
            engine_stopped.set()
            threading.Thread(
                target=server.shutdown, name="splash-engine-stopped", daemon=True
            ).start()

        backend.on_stopped = stop_serving
        if not runtime.wait_ready():
            raise audio_runtime.EngineUnhealthy("native runtime did not become ready")
        readiness = runtime.readiness
        if readiness is None or readiness.kind != profile.kind:
            raise audio_runtime.EngineUnhealthy(
                "native runtime serves "
                + (
                    "speech"
                    if readiness and readiness.kind.name == "TTS"
                    else "transcription"
                )
                + f"; the model's configuration describes {profile.family}"
            )
        app = AudioFrontend(
            tokenizer,
            backend,
            args.model,
            profile,
            # No deadline unless given; a request still ends when its client
            # disconnects.
            math.inf if args.request_timeout is None else args.request_timeout,
            readiness.max_concurrent_requests,
            served_model_names=args.served_model_name,
            announce_served_name=args.announce_served_name,
            segment_seconds=args.segment_seconds,
        )
        server.app = app
        server.server_activate()
        address = f"http://{args.host}:{server.server_port}"
        print_status(
            f"Ready · {args.model} · {profile.family} · "
            f"{readiness.max_concurrent_requests} lanes · {address}"
        )
        server.serve_forever()
        if engine_stopped.is_set():
            raise SystemExit(ENGINE_STOPPED_STATUS)
    except audio_runtime.EngineRuntimeError as error:
        print_status(f"Error · {error}", error=True)
        raise SystemExit(1) from None
    except OSError as error:
        print_status(f"Error · unable to start HTTP server: {error}", error=True)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        pass
    finally:
        # main owns this process. Keep stop signals idempotent through child
        # cleanup and interpreter teardown, except that a second Ctrl+C during
        # cleanup stops the engine without waiting for its graceful exit.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(
            signal.SIGINT,
            signal.SIG_IGN if runtime is None else lambda *_: runtime.kill(),
        )
        try:
            if backend is not None:
                print_status("Stopping · releasing engine resources")
                backend.close()
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if server is not None:
                server.server_close()
            # The console's writer is a daemon thread: give it the last lines.
            CONSOLE.flush(CONSOLE_FLUSH_SECONDS)


if __name__ == "__main__":
    main()
