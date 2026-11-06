"""One serving benchmark run: a tree's own server, one client scenario, and
the records that make the run checkable.

Run from a checkout holding these tools, with the tree to measure:

    python3 -m dev.benchmarks.serving.run --tree TREE --machine m6 \\
        --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0 --revision ed59f92... \\
        --draft-model DRAFT --scenario churn --order 2,0,1 \\
        --minutes 15 --label m6-fn-churn-cand-A1 --out OUT

It installs the model with the tree's own installer, offline, or takes
--model-root as installed; starts the tree's own server (the wire differs
between trees, so a build never runs under another tree's server) on a port
nothing answers on; checks that /status names the server it started (pid) and
the tree's own sources (build id); runs the scenario; stops the server with
SIGINT, or SIGTERM and SIGKILL when it outlasts that, and kills the processes
it started, its engine, that outlast it; and writes beside the
client's records:

- LABEL.out: what ran, the checks, and DONE as its last line once all of it
  completed;
- LABEL.meta.json: tree, binary, build id, model, scenario, spawn-to-ready
  seconds and whether the binary had started on this machine before (a first
  start compiles Metal), host memory and pressure at start, ps snapshots at
  the start, middle and end of the scenario while the server serves, the pids
  this run started, each server's stop (the signals it took, its exit code
  and the processes it started that it left running), on the M5 Pro the
  idle and sleep-assertion samples (LABEL.idle.jsonl) that show whether the
  machine was in use, and for a run that did not complete why (error) and
  whose failure that was (failure: server or request for the server's own,
  client for a client's own, else the refusal's kind);
- LABEL.server.log (one per server: LABEL-nN for each concurrency of ttft).

No output goes under ~/.cache/huggingface or an install/models directory,
whatever symlinks lead there.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from dev.benchmarks.serving import clients

# The test Macs, and any other (a release gate runner).
MACHINES = ("studio", "m5", "m6", "mini", "other")
SCENARIOS = ("churn", "ttft", "serve1", "warm")
READY_TIMEOUT_SECONDS = 900
# A server that ignores SIGINT this long gets SIGTERM, then SIGKILL (a server
# under memory pressure once ignored SIGINT), and a process it started, its
# engine, still running then gets SIGKILL: G8 fails either.
STOP_GRACE_SECONDS = 60
# The M5 Pro is also used interactively: its idle time and sleep assertions are
# sampled this often during a run.
IDLE_SAMPLE_SECONDS = 10
# On a shared machine that does not hold our lock (the mini-M4), a timed
# run waits until the GPU is as idle as it was recorded with nothing running:
# five samples 2 s apart, for up to 10 minutes.
GPU_IDLE_SAMPLES, GPU_IDLE_PERIOD, GPU_IDLE_WAIT_SECONDS = 5, 2.0, 600
# The processes of the top of `ps -r` a snapshot keeps.
PS_LINES = 15
PRINT_CONSTANTS = "splash-benchmark-print-build-id-constants:\n\t@printf '%s\\n' $(BUILD_ID_CONSTANT_ARGS)\n"


class RunError(RuntimeError):
    """Why a run cannot complete, and whose doing that is: `kind`, recorded
    in meta.json as `failure`, is one of SERVER_FAILURES when the server
    failed the run, `client` when a client failed on its own
    (clients.ClientError), else this harness refused to measure it (a port
    that answers, a build or a server that is not the tree's, an
    installation or an output it cannot make)."""

    def __init__(self, message: str, kind: str):
        super().__init__(message)
        self.kind = kind


# The server's own failures, which fail G8: it
# exited, was not ready in time or, once ready, ended its scenario otherwise
# (a request or /status it did not answer in time, a wait for idle it
# outlasted), or it failed a request.
SERVER_FAILURES = ("server", "request")


def protected_roots() -> list[Path]:
    return [Path(os.path.realpath(os.path.expanduser("~/.cache/huggingface")))]


def refuse_protected(path: Path) -> Path:
    """The real path of an output directory, refused under the Hugging Face
    cache or any install/models directory, whatever symlinks lead there
    (output written through such links once corrupted model blobs)."""
    real = Path(os.path.realpath(path))
    for root in protected_roots():
        if real == root or root in real.parents:
            raise RunError(f"refusing output {path}: it resolves into {root}", "output")
    parts = real.parts
    if any(parts[i : i + 2] == ("install", "models") for i in range(len(parts) - 1)):
        raise RunError(
            f"refusing output {path}: it resolves into an install/models directory ({real})",
            "output",
        )
    return real


def port_answers(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str], **kwargs) -> str:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=60, **kwargs
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return f"<{command[0]}: {error}>"


def tree_build_id(tree: Path) -> str:
    """The build id of the tree's sources, as its own build_identity.py
    derives it with the constants its own Makefile passes, verified against
    the build's generated header and stamp and embedded in build/splash."""
    with tempfile.NamedTemporaryFile("w", suffix=".mk") as extra:
        extra.write(PRINT_CONSTANTS)
        extra.flush()
        listed = subprocess.run(
            [
                "make",
                "-s",
                "--no-print-directory",
                "-C",
                str(tree),
                "-f",
                "Makefile",
                "-f",
                extra.name,
                "splash-benchmark-print-build-id-constants",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    if listed.returncode:
        raise RunError(
            f"{tree}: cannot read its build identity constants: {listed.stderr[-300:]}",
            "build",
        )
    checked = subprocess.run(
        [
            sys.executable,
            str(tree / "dev/tools/build_identity.py"),
            "check",
            "--root",
            str(tree),
            "--header",
            str(tree / "build/engine/BuildIdentity.hpp"),
            "--stamp",
            str(tree / "build/engine/build-identity.json"),
            *listed.stdout.splitlines(),
            "--binary",
            str(tree / "build/splash"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if checked.returncode:
        raise RunError(
            f"{tree}: its build does not match its sources: {checked.stderr[-300:]}",
            "build",
        )
    return json.loads(checked.stdout)["build_id"]


def host_memory() -> dict:
    """What macOS reports of its memory now: the pressure level
    (kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical),
    the free, inactive and speculative pages in bytes, and memory_pressure's
    free percentage."""
    page = int(command_output(["sysctl", "-n", "hw.pagesize"]).strip() or 16384)
    pages = {}
    for line in command_output(["vm_stat"]).splitlines():
        name, _, value = line.partition(":")
        if value.strip().rstrip(".").isdigit():
            pages[name.strip()] = int(value.strip().rstrip("."))
    level = command_output(
        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]
    ).strip()
    available = sum(
        pages.get(f"Pages {kind}", 0) for kind in ("free", "inactive", "speculative")
    )
    return {
        "available_bytes": available * page,
        "pressure_level": int(level) if level.isdigit() else None,
        "memory_pressure": command_output(["memory_pressure"])
        .strip()
        .splitlines()[-1:],
    }


def ps_snapshot(ours: set[int]) -> dict:
    """The top of `ps -r` by CPU, each row marked whether this run started
    the process."""
    rows = []
    for line in command_output(["ps", "-Ao", "pid,pcpu,pmem,comm", "-r"]).splitlines()[
        1 : PS_LINES + 1
    ]:
        fields = line.split(None, 3)
        if len(fields) == 4 and fields[0].isdigit():
            pid = int(fields[0])
            rows.append(
                [pid, float(fields[1]), float(fields[2]), fields[3], pid in ours]
            )
    return {
        "at": time.time(),
        "rows": rows,
        "therm": command_output(["pmset", "-g", "therm"]).strip(),
    }


def gpu_utilization() -> int:
    """The GPU's Device Utilization % now (IOAccelerator statistics)."""
    for line in command_output(["ioreg", "-r", "-c", "AGXAccelerator"]).splitlines():
        if '"Device Utilization %"=' in line:
            return int(
                line.split('"Device Utilization %"=')[1].split(",")[0].split("}")[0]
            )
    return 0


def wait_gpu_idle(idle_percent: int) -> dict:
    """Waits until five samples over 10 s are at most the machine's recorded
    idle level, up to GPU_IDLE_WAIT_SECONDS; says whether they were."""
    started = time.time()
    while True:
        highest = 0
        for _ in range(GPU_IDLE_SAMPLES):
            highest = max(highest, gpu_utilization())
            time.sleep(GPU_IDLE_PERIOD)
        waited = round(time.time() - started, 1)
        if highest <= idle_percent or waited >= GPU_IDLE_WAIT_SECONDS:
            return {
                "ok": highest <= idle_percent,
                "max_percent": highest,
                "waited_s": waited,
            }
        time.sleep(10)


def idle_sample() -> dict:
    """The M5 Pro's seconds since its owner last touched it (HIDIdleTime) and
    the processes holding a display or system sleep assertion."""
    idle = None
    for line in command_output(["ioreg", "-c", "IOHIDSystem"]).splitlines():
        if "HIDIdleTime" in line:
            idle = int(line.split()[-1]) / 1e9
            break
    holders = []
    for line in command_output(["pmset", "-g", "assertions"]).splitlines():
        if "pid " in line and (
            "PreventUserIdleDisplaySleep" in line
            or "PreventUserIdleSystemSleep" in line
        ):
            holders.append(line.strip())
    return {"at": time.time(), "hid_idle_s": idle, "holders": holders}


class Server:
    """A tree's own server, started on `port`, stopped by stop()."""

    def __init__(
        self,
        tree: Path,
        root: str,
        model: str,
        port: int,
        server_args: list[str],
        log: Path,
    ):
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        }
        environment.update(
            HF_HUB_OFFLINE="1",
            HF_TOKEN_PATH="/nonexistent",
            PYTHONUNBUFFERED="1",
            TRANSFORMERS_VERBOSITY="error",
        )
        command = [
            str(tree / ".venv/bin/python"),
            "-m",
            "server.server",
            root,
            "--model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--binary",
            "build/splash",
            "--api-key",
            clients.API_KEY,
            "--no-webui",
            *server_args,
        ]
        self.port = port
        self.log = log.open("w")
        self.spawned = time.time()
        self.process = subprocess.Popen(
            command,
            cwd=tree,
            env=environment,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )

    def wait_ready(self, timeout: float = READY_TIMEOUT_SECONDS) -> float:
        """Seconds from spawn until /ready answered."""
        deadline = self.spawned + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RunError(
                    f"the server exited {self.process.returncode} before it was ready",
                    "server",
                )
            try:
                clients.get_json(self.port, "/ready", timeout=5)
                return round(time.time() - self.spawned, 2)
            except Exception:
                time.sleep(1)
        raise RunError(f"the server was not ready within {timeout} s", "server")

    def children(self) -> set[int]:
        """The processes the server started: its engine, and any document
        worker."""
        listed = command_output(["pgrep", "-P", str(self.process.pid)])
        return {int(pid) for pid in listed.split() if pid.isdigit()}

    def engine_pids(self) -> set[int]:
        return {self.process.pid, *self.children()}

    def stop(self) -> dict:
        """Sends SIGINT, then SIGTERM and SIGKILL to a server that outlasts
        each, until it exits, and SIGKILL to the processes it started that
        still run once SIGINT's grace is over. Returns the signals sent, none
        to a server that had exited, its exit code, None if it outlasted
        SIGKILL, and the pids of those it started that it left running."""
        children = self.children()
        window = time.monotonic() + STOP_GRACE_SECONDS
        signals = []
        for sent, grace in (
            (signal.SIGINT, STOP_GRACE_SECONDS),
            (signal.SIGTERM, 30),
            (signal.SIGKILL, 30),
        ):
            if self.process.poll() is not None:
                break
            self.process.send_signal(sent)
            signals.append(sent.name)
            try:
                self.process.wait(grace)
            except subprocess.TimeoutExpired:
                pass
        # Its processes have until SIGINT's grace is over, the same window,
        # to exit and, once orphaned, be reaped.
        while True:
            children = {pid for pid in children if running(pid)}
            if not children or time.monotonic() >= window:
                break
            time.sleep(0.1)
        for pid in children:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        self.log.close()
        return {
            "signals": signals,
            "exit": self.process.poll(),
            "left_running": sorted(children),
        }


def running(pid: int) -> bool:
    """Whether process pid exists (a zombie not yet reaped among them)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def install(args, out: Path) -> str:
    """The model's installation root: --model-root, or the tree's own
    installer's offline prepare and link."""
    if args.model_root:
        return str(args.model_root)
    options = ["--model", args.model]
    if args.revision:
        options += ["--revision", args.revision]
    if args.draft_model:
        options += ["--draft-model", args.draft_model]
    if args.language_only:
        options.append("--language-only")
    python = str(args.tree / ".venv/bin/python")
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_TOKEN_PATH": "/nonexistent"}
    with (out / f"{args.label}.install.log").open("w") as log:
        for step in ("prepare", "link"):
            done = subprocess.run(
                [python, "install/models.py", *options, step],
                cwd=args.tree,
                env=environment,
                stdout=subprocess.PIPE if step == "link" else log,
                stderr=log,
                text=True,
            )
            if done.returncode:
                raise RunError(
                    f"install/models.py {step} exited {done.returncode}", "install"
                )
    return done.stdout.strip().splitlines()[-1]


def first_start(starts: Path, binary_sha256: str) -> bool:
    """Whether this binary had not started on this machine before; marks it
    started."""
    starts.mkdir(parents=True, exist_ok=True)
    marker = starts / binary_sha256
    if marker.exists():
        return False
    marker.touch()
    return True


def check_identity(port: int, server: Server, build_id: str) -> dict:
    status = clients.get_json(port, "/status")
    pid = clients.dig(status, "instance.pid")
    served = clients.dig(status, "identity.cache.build_id")
    if pid != server.process.pid:
        raise RunError(
            f"/status names pid {pid}, not the server this run started ({server.process.pid})",
            "identity",
        )
    if served != build_id:
        raise RunError(
            f"/status names build {served}, not the tree's {build_id}", "identity"
        )
    return {
        "pid": pid,
        "build_id": served,
        "thermal_state": status.get("thermal_state"),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--tree",
        required=True,
        type=Path,
        help="the checkout whose server and build run",
    )
    parser.add_argument(
        "--tree-name",
        help="its name among the compared trees, recorded",
    )
    parser.add_argument(
        "--commit", help="the tree's commit, recorded (synced trees have no .git)"
    )
    parser.add_argument("--machine", required=True, choices=MACHINES)
    parser.add_argument(
        "--model", required=True, help="the model as splash serve --model takes it"
    )
    parser.add_argument("--revision")
    parser.add_argument("--draft-model")
    parser.add_argument(
        "--language-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--model-root", type=Path, help="an installed model root; skips the installer"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument(
        "--order", default="2,0,1", help="churn: the first requests' arrival order"
    )
    parser.add_argument(
        "--minutes", type=float, default=15.0, help="churn: the load window"
    )
    parser.add_argument(
        "--request-seconds",
        type=float,
        default=clients.REQUEST_SECONDS,
        help="how long a request may take from its send (plan: twice the machine's longest recorded)",
    )
    parser.add_argument(
        "--n", default="1,2,4", help="ttft: concurrency levels, a fresh server each"
    )
    parser.add_argument("--rounds", type=int, default=3, help="ttft: rounds per level")
    parser.add_argument(
        "--samples", type=int, default=3, help="serve1: requests per size"
    )
    parser.add_argument(
        "--sizes",
        default=",".join(map(str, clients.SERVE1_SIZES)),
        help="serve1: prompt sizes in tokens; warm: the cached depths",
    )
    parser.add_argument(
        "--trial", type=int, default=1, help="warm: the trial, which picks the texts"
    )
    parser.add_argument(
        "--gpu-idle-percent",
        type=int,
        help="wait until the GPU is this idle before the run (the machine's recorded idle level)",
    )
    parser.add_argument(
        "--server-args", default="", help="the server's own options, one shell string"
    )
    parser.add_argument(
        "--env", action="append", default=[], help="KEY=VALUE: CHURN_BASE, CHURN_STEP"
    )
    parser.add_argument("--port", type=int, default=8594)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cell", help="the comparison this run belongs to, recorded")
    parser.add_argument(
        "--repeat",
        type=int,
        help="the run's repeat in its cell, tree and order, recorded",
    )
    parser.add_argument("--out", required=True, type=Path, help="the results directory")
    parser.add_argument(
        "--starts", type=Path, help="where first starts are marked (OUT/starts)"
    )
    args = parser.parse_args(argv)
    args.order_list = [int(x) for x in args.order.split(",")]
    if sorted(args.order_list) != list(range(clients.CHURN_WORKERS)):
        parser.error(f"--order must order the workers 0-{clients.CHURN_WORKERS - 1}")
    args.environment = {}
    for item in args.env:
        key, separator, value = item.partition("=")
        if not separator or key not in ("CHURN_BASE", "CHURN_STEP"):
            parser.error(f"--env takes CHURN_BASE=N or CHURN_STEP=N, not {item}")
        args.environment[key] = int(value)
    args.levels = [int(x) for x in args.n.split(",")]
    args.size_list = [int(x) for x in args.sizes.split(",")]
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    out = refuse_protected(args.out)
    starts = refuse_protected(args.starts or out / "starts")
    out.mkdir(parents=True, exist_ok=True)
    label_out = out / f"{args.label}.out"
    if any(out.glob(f"{args.label}.*")):
        raise RunError(f"label {args.label} already used in {out}", "label")
    tree = args.tree.resolve()
    args.tree = tree
    lines = label_out.open("w")

    def note(text):
        lines.write(text + "\n")
        lines.flush()

    meta = {
        "label": args.label,
        "cell": args.cell,
        "machine": args.machine,
        "tree": str(tree),
        "tree_name": args.tree_name,
        "commit": args.commit,
        "model": args.model,
        "revision": args.revision,
        "draft": args.draft_model,
        "server_args": args.server_args,
        "scenario": args.scenario,
        "order": args.order_list if args.scenario == "churn" else None,
        "repeat": args.repeat,
        "env": args.environment,
        "client_python": sys.version.split()[0],
        "tools_root": str(Path(__file__).resolve().parents[3]),
        "started": time.time(),
    }
    ours = {os.getpid()}
    snapshots, idle = [], []
    stop_idle = threading.Event()
    try:
        if port_answers(args.port):
            raise RunError(
                f"port {args.port} already answers: a stale server would serve this run",
                "port",
            )
        binary = tree / "build/splash"
        meta["binary_sha256"] = sha256(binary)
        meta["binary_mtime"] = binary.stat().st_mtime
        meta["build_id"] = tree_build_id(tree)
        root = install(args, out)
        meta["model_root"] = root
        meta["host_at_start"] = host_memory()
        meta["first_start"] = first_start(starts, meta["binary_sha256"])
        if args.gpu_idle_percent is not None:
            meta["gpu_idle_check"] = wait_gpu_idle(args.gpu_idle_percent)
        note(
            f"root {root} tree {tree} binary {meta['binary_sha256'][:12]} build {meta['build_id']}"
        )
        if args.machine == "m5":

            def sample_idle():
                while not stop_idle.is_set():
                    idle.append(idle_sample())
                    stop_idle.wait(IDLE_SAMPLE_SECONDS)

            threading.Thread(target=sample_idle, daemon=True).start()
        server_args = shlex.split(args.server_args)
        parts = meta["parts"] = []
        levels = args.levels if args.scenario == "ttft" else [None]
        for index, level in enumerate(levels):
            part = args.label if level is None else f"{args.label}-n{level}"
            server = Server(
                tree,
                root,
                args.model,
                args.port,
                server_args,
                out / f"{part}.server.log",
            )
            ours |= {server.process.pid}
            record = {"label": part}
            parts.append(record)
            try:
                ready = server.wait_ready()
                ours |= server.engine_pids()
                checks = check_identity(args.port, server, meta["build_id"])
                note(
                    f"{part} ready=yes spawn_to_ready_s={ready} pid={checks['pid']} build_id=ok"
                )
                record.update(spawn_to_ready_s=ready, **checks)
                # The snapshots are taken while the server serves, so that
                # its processes are among this run's.
                if index == 0:
                    snapshots.append(ps_snapshot(ours))
                if args.scenario == "churn":
                    middle = threading.Timer(
                        args.minutes * 30, lambda: snapshots.append(ps_snapshot(ours))
                    )
                    middle.daemon = True
                    middle.start()
                    clients.run_churn(
                        args.port,
                        str(out),
                        part,
                        args.minutes,
                        args.order_list,
                        base=args.environment.get(
                            "CHURN_BASE", clients.CHURN_DEFAULT_BASE
                        ),
                        step=args.environment.get(
                            "CHURN_STEP", clients.CHURN_DEFAULT_STEP
                        ),
                        timeout=args.request_seconds,
                    )
                    middle.cancel()
                elif args.scenario == "ttft":
                    summary = clients.run_ttft(
                        args.port,
                        str(out),
                        part,
                        level,
                        rounds=args.rounds,
                        timeout=args.request_seconds,
                    )
                    note(f"{part} summary {json.dumps(summary)}")
                    if 0 < index < len(levels) - 1 and index == len(levels) // 2:
                        snapshots.append(ps_snapshot(ours))
                elif args.scenario == "serve1":
                    clients.run_serve1(
                        args.port,
                        str(out),
                        part,
                        samples=args.samples,
                        sizes=args.size_list,
                        timeout=args.request_seconds,
                    )
                else:
                    clients.run_warm(
                        args.port,
                        str(out),
                        part,
                        args.size_list,
                        args.trial,
                        timeout=args.request_seconds,
                    )
                # A server gone by the scenario's end failed the run, also
                # where its clients carried on (a churn client stops at the
                # first request the server no longer answers).
                if server.process.poll() is not None:
                    raise RunError(
                        f"the server exited {server.process.returncode} during the scenario",
                        "server",
                    )
                if index == len(levels) - 1:
                    snapshots.append(ps_snapshot(ours))
            except RunError:
                raise
            except clients.RequestFailed as error:
                raise RunError(str(error), "request") from error
            except clients.ClientError as error:
                raise RunError(str(error), "client") from error
            except Exception as error:
                # Past Ready, whatever else ends the scenario is the server's
                # failure: it exited, or, alive on 127.0.0.1, it did not answer
                # a request or /status in time (a request it kept streaming
                # past --request-seconds among them) or did not go idle.
                code = server.process.poll()
                raise RunError(
                    f"{type(error).__name__}: {error}"
                    if code is None
                    else f"the server exited {code} during the scenario: {error}",
                    "server",
                ) from error
            finally:
                record["stop"] = stop = server.stop()
                note(
                    f"{part} stop: signals {', '.join(stop['signals']) or 'none'}, "
                    f"exit {stop['exit']}, left running "
                    f"{', '.join(map(str, stop['left_running'])) or 'none'}"
                )
        meta["first_start_spawn_to_ready_s"] = (
            parts[0]["spawn_to_ready_s"] if parts else None
        )
        note(f"done {args.label}")
        meta["ended"] = time.time()
        meta["complete"] = True
        return 0
    except RunError as error:
        note(f"FAIL {error}")
        meta["error"] = str(error)
        meta["failure"] = error.kind
        meta["complete"] = False
        return 1
    finally:
        stop_idle.set()
        meta["ps"] = snapshots
        meta["pids"] = sorted(ours)
        meta["host_at_end"] = host_memory()
        if idle:
            with (out / f"{args.label}.idle.jsonl").open("w") as samples:
                for sample in idle:
                    samples.write(json.dumps(sample) + "\n")
        (out / f"{args.label}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        if meta.get("complete"):
            note("DONE")
        lines.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunError as error:
        print(f"run: {error}", file=sys.stderr)
        raise SystemExit(1) from None
