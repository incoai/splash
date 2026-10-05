#!/usr/bin/env python3
"""Exercise installed coding clients using the production launcher adapter.

No model aliases, replacement prompts, tool filters, synthetic assistant turns,
or client upgrades. Reports retain failures and each client's native history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import secrets
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.tests import smoke_real  # noqa: E402
from dev.tools import build_identity  # noqa: E402
from install import clients, launcher  # noqa: E402
from server import serve_options  # noqa: E402

CLIENTS = tuple(clients.INSTALL_URLS)
# The server this harness starts or finds, on the default port.
BASE_URL = launcher._base_url(serve_options.DEFAULT_PORT)
# The project's tests: the prompts give their command with python3, and the
# harness reruns them with its own Python.
TEST_ARGUMENTS = ("-m", "unittest", "-v")
TEST_COMMAND = " ".join(("python3", *TEST_ARGUMENTS))
# A successful command running the unittest module: the prompt names python3,
# but an agent may run the tests with its own interpreter or its full path.
RAN_TESTS = re.compile(r"\bpython[\d.]*\s+-m\s+unittest\b")
# The test_clients.py classes that test a client as installed, with no model,
# each taking its executable from SPLASH_<NAME>_BINARY.
INSTALLED_CLIENT_TESTS = {
    "opencode": "InstalledOpenCodeTests",
    "codex": "InstalledCodexTests",
    "pi": "InstalledPiTests",
}


class AgentFailure(RuntimeError):
    pass


def current_build_id():
    # The make gate verifies these against the current production sources.
    # Also verify the on-disk binary here, so a stale/missing stamp cannot
    # authorize reuse of a server from another native build.
    directory = ROOT / "build/engine"
    stamp = directory / "build-identity.json"
    try:
        identity = json.loads(stamp.read_text())["build_id"]
        build_identity.verify_generated(
            argparse.Namespace(
                header=directory / "BuildIdentity.hpp",
                stamp=stamp,
                binary=[ROOT / "build/splash"],
            ),
            identity,
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        build_identity.BuildIdentityError,
    ) as error:
        raise AgentFailure(
            "current native build identity is unavailable or stale; "
            "run make verify-build-identity"
        ) from error
    return identity


def validate_server_configuration(initial, model, context, identity):
    if (
        initial["instance"]["model"] != model
        or initial["maximum_context_tokens"] != context
    ):
        raise AgentFailure(
            "running server model/context differs from the test configuration"
        )
    if initial.get("identity", {}).get("cache", {}).get("build_id") != identity:
        raise AgentFailure(
            "running server native build differs from the verified build"
        )


def run_installed_client_tests(versions):
    """Runs test_clients.py's tests of the clients versions located that it
    has tests for, each client at the path located."""
    environment, tests = dict(os.environ), []
    for name, test in INSTALLED_CLIENT_TESTS.items():
        if name in versions:
            environment[f"SPLASH_{name.upper()}_BINARY"] = versions[name]["path"]
            tests.append(f"dev.tests.install.test_clients.{test}")
    if not tests:
        return
    command = [sys.executable, "-m", "unittest", *tests]
    if subprocess.run(command, cwd=ROOT, env=environment).returncode:
        raise AgentFailure("the installed clients failed test_clients.py's tests")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def events(text):
    result = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def executed_commands(name, parsed, messages=(), turns=()):
    commands = []
    calls = {}
    if name == "hermes":
        for message in messages:
            for call in json.loads(message.get("tool_calls") or "[]"):
                function = call.get("function", {})
                if function.get("name") == "terminal":
                    calls[call["id"]] = json.loads(function["arguments"]).get(
                        "command", ""
                    )
            if message.get("tool_name") == "terminal":
                try:
                    result = json.loads(message.get("content") or "{}")
                except ValueError:
                    continue
                if (
                    result.get("exit_code") == 0
                    and message.get("tool_call_id") in calls
                ):
                    commands.append(calls[message["tool_call_id"]])
        return commands
    if name == "opencode":
        # From its session record: the turns the phase added.
        return [command for turn in turns for command in turn["commands"]]
    for event in parsed:
        if name == "pi" and event.get("toolName") == "bash":
            if event.get("type") == "tool_execution_start":
                calls[event["toolCallId"]] = event.get("args", {}).get("command", "")
            if (
                event.get("type") == "tool_execution_end"
                and event.get("isError") is False
                and event.get("toolCallId") in calls
            ):
                commands.append(calls[event["toolCallId"]])
        if name == "codex" and event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "command_execution" and item.get("exit_code") == 0:
                commands.append(item.get("command", ""))
        if name == "claude":
            # System events such as permission_denied carry a string message.
            message = event.get("message")
            content = message.get("content", []) if isinstance(message, dict) else []
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("name") == "Bash":
                    calls[block["id"]] = block.get("input", {}).get("command", "")
                if (
                    block.get("type") == "tool_result"
                    and not block.get("is_error")
                    and block.get("tool_use_id") in calls
                ):
                    commands.append(calls[block["tool_use_id"]])
    return commands


def pi_completed(parsed):
    """Pi's agent ended on a stopped assistant message with text."""
    messages = [
        event["message"]
        for event in parsed
        if event.get("type") == "message_end"
        and event.get("message", {}).get("role") == "assistant"
    ]
    return bool(
        messages
        and messages[-1].get("stopReason") == "stop"
        and any(
            block.get("text", "").strip() for block in messages[-1].get("content", [])
        )
        and any(event.get("type") == "agent_end" for event in parsed)
    )


def opencode_session(history, version):
    """OpenCode's session export, the same record for both major versions:
    the turns, each the assistant steps that answer a user message, as how
    the last step finished, their text and the shell commands that
    succeeded; and the automatic compactions, as the messages holding them."""
    turns, compactions = [], []
    for message in history["messages"]:
        if version >= 2:
            # 2.x keeps a message's role as its type and an assistant's parts
            # in content, names its bash tool shell, and records a compaction
            # as an entry of its own; an idle entry ends each turn.
            role, finish = message["type"], message.get("finish")
            parts = message.get("content") or []
            shells = [
                p for p in parts if p.get("type") == "tool" and p.get("name") == "shell"
            ]
            if (
                role == "compaction"
                and message.get("reason") == "auto"
                and message.get("status") == "completed"
            ):
                compactions.append({"message_id": message["id"]})
        else:
            # 1.x keeps a message's role and finish in its info, and marks the
            # user message that asks for a compaction with an automatic
            # compaction part.
            info, parts = message["info"], message.get("parts") or []
            role, finish = info["role"], info.get("finish")
            shells = [
                p for p in parts if p.get("type") == "tool" and p.get("tool") == "bash"
            ]
            if any(p.get("type") == "compaction" and p.get("auto") for p in parts):
                compactions.append({"message_id": info["id"]})
        if role == "user":
            turns.append({"finish": None, "text": "", "commands": []})
        elif role == "assistant" and turns:
            turn = turns[-1]
            turn["finish"] = finish
            turn["text"] += "".join(
                p.get("text") or "" for p in parts if p.get("type") == "text"
            )
            for part in shells:
                state = part.get("state") or {}
                if (
                    state.get("status") == "completed"
                    and (state.get("metadata") or {}).get("exit", 0) == 0
                ):
                    turn["commands"].append(
                        (state.get("input") or {}).get("command", "")
                    )
    return {"turns": turns, "compactions": compactions}


def opencode_completed(turns):
    """OpenCode answered a phase, given the turns it added: there is one, and
    the last ended on a stopped assistant step and has text."""
    return bool(turns and turns[-1]["finish"] == "stop" and turns[-1]["text"].strip())


# The engine's own critical verdict drops every evictable cache entry and
# answers new requests 503 until host availability recovers; a request that is
# already running continues. A caller that tolerates it receives the fresh
# snapshot with ready=false and bounds how long the window may last.
ENGINE_CRITICAL_GRACE_SECONDS = 60


def engine_critical(value):
    transport = (value or {}).get("transport", {})
    return (
        transport.get("ready") is True
        and not transport.get("status_stale")
        and (value or {}).get("metal", {}).get("healthy") is True
        and (value or {}).get("memory_pressure") == "critical"
    )


def status(*, wait_for_fresh=True, tolerate_critical=False):
    deadline = time.monotonic() + 10
    announced = False
    while True:
        value = launcher._request_json("/status", timeout=10)
        if value and value.get("ready"):
            smoke_real.validate_status(value)
            return value
        if tolerate_critical and engine_critical(value):
            return value
        transport = (value or {}).get("transport", {})
        # Busy GPU work may outlast the HTTP status snapshot's 50 ms refresh.
        # Wait for a fresh ready snapshot; never treat stale telemetry as pass.
        # A stale snapshot keeps the last pressure verdict; under tolerance it is
        # handled as stale, not as a dead server.
        if (
            not transport.get("status_stale")
            or (
                not tolerate_critical
                and (value or {}).get("memory_pressure") == "critical"
            )
            or (value or {}).get("metal", {}).get("healthy") is not True
        ):
            raise AgentFailure(f"server is not ready: {transport}")
        if not wait_for_fresh:
            return None
        if time.monotonic() >= deadline:
            raise AgentFailure(f"fresh status timed out: {transport}")
        if not announced:
            print(f"Waiting for fresh native status: {transport}", flush=True)
            announced = True
        time.sleep(0.25)


def idle_status():
    # Up to 60 s: a phase boundary may fall inside the engine's critical
    # window, which clears once macOS has registered the memory it shed.
    for _ in range(240):
        value = status(tolerate_critical=True)
        if not value.get("ready"):
            time.sleep(0.25)
            continue
        if (
            not any(
                value["scheduler"][k]
                for k in (
                    "queued",
                    "waiting_resources",
                    "prefilling",
                    "decoding",
                    "waiting_mask",
                )
            )
            and value["state"]["active_lanes"] == 0
            and value["state"]["pinned"] == 0
            and value["state"]["in_use"] == 0
            and value["kv"]["pages_active"] == 0
        ):
            return value
        time.sleep(0.25)
    raise AgentFailure("server did not return to idle")


def prefix_reuse(before, after):
    """What a phase's requests reused of the cache, from the idle status
    before and after it."""

    def delta(section, key):
        return after[section][key] - before[section][key]

    return {
        "reused_tokens": delta("cache", "reused_tokens"),
        "prefill_input_tokens": delta("metrics", "prefill_input_tokens"),
        "lost_state_misses": delta("cache", "lost_state_misses"),
        "in_use_evictions": delta("state", "in_use_evictions"),
        "completed": delta("requests", "completed"),
    }


def pressure_stop_level():
    """macOS pressure level at which a client workflow stops: 1 normal, 2
    warning, 4 critical. The default stops at any warning; set
    SPLASH_TEST_PRESSURE_STOP=4 to observe how the engine sheds cache under
    warning pressure and stop only when the OS reports critical."""
    return int(os.environ.get("SPLASH_TEST_PRESSURE_STOP", "2"))


def memory_sample():
    raw = subprocess.check_output(["vm_stat"], text=True)
    page_size = int(raw.split("page size of ")[1].split(" bytes")[0])
    counts = {
        key: int(value.strip().rstrip("."))
        for key, value in (line.split(":", 1) for line in raw.splitlines()[1:])
    }
    return {
        "pressure": int(
            subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            )
        ),
        "compressed_bytes": counts["Pages occupied by compressor"] * page_size,
        "swapout_bytes": counts["Swapouts"] * page_size,
        "top_rss_mib": top_resident_processes(),
    }


def top_resident_processes(limit=6):
    """Largest resident processes, so a pressure episode can be attributed."""
    try:
        raw = subprocess.check_output(
            ["ps", "-axo", "rss=,comm=", "-m"], text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in raw.splitlines()[:limit]:
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            rows.append([parts[1].rsplit("/", 1)[-1][:40], int(parts[0]) // 1024])
    return rows


def stop_process(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(8)


def fixture(workspace):
    workspace.mkdir(parents=True)
    (workspace / "ledger.py").write_text("def summarize(records):\n    return {}\n")
    (workspace / "README.md").write_text(
        "# Receipt ledger\n\n"
        "Implement summarize(records) in ledger.py. Each record has amount_cents "
        "(an integer or null) and a category string. Ignore null amounts. Return "
        "exactly a dictionary with count (number of included records) and "
        "total_cents (their sum). Negative amounts are refunds and must count. "
        "Empty input returns zero for both fields. Do not mutate the input.\n\n"
        "Use Python's standard library. Add unittest coverage in test_ledger.py. "
        f"The project test command is `{TEST_COMMAND}`.\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)


def validate_artifact(workspace, stage):
    rng = random.Random(104729)
    cases = [[], [{"amount_cents": None, "category": "food"}]]
    for _ in range(24):
        cases.append(
            [
                {
                    "amount_cents": rng.choice([None, -501, -1, 0, 7, 1234]),
                    "category": rng.choice(["food", "travel"]),
                    "void": rng.choice([False, False, True]),
                }
                for _ in range(rng.randrange(1, 30))
            ]
        )
    oracle = """import copy, json, sys
from ledger import summarize
stage, cases = json.load(sys.stdin)
for rows in cases:
    for category in ([None, 'food', 'travel', 'missing'] if stage == 2 else [None]):
        original = copy.deepcopy(rows)
        included = [r for r in rows if r['amount_cents'] is not None
                    and (stage == 0 or not r.get('void', False))
                    and (category is None or r['category'] == category)]
        actual = summarize(rows, category=category) if stage == 2 else summarize(rows)
        expected = {'count': len(included), 'total_cents': sum(r['amount_cents'] for r in included)}
        assert actual == expected, (rows, category, actual, expected)
        assert all(type(v) is int for v in actual.values()), actual
        assert rows == original, 'input mutated'
print('independent oracle passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", oracle],
        cwd=workspace,
        input=json.dumps([stage, cases]),
        text=True,
        capture_output=True,
        timeout=15,
    )
    if result.returncode:
        raise AgentFailure(
            "independent artifact oracle failed: " + result.stderr[-2000:]
        )
    if not (workspace / "test_ledger.py").is_file():
        raise AgentFailure("client did not create the requested unit tests")
    # Rerun the client's tests as its test command does, with the harness's
    # Python like the oracle above: the python3 first on PATH can be a shim
    # that refuses to run, as Xcode's does until its license is accepted.
    result = subprocess.run(
        [sys.executable, *TEST_ARGUMENTS],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode or "Ran 0 tests" in result.stderr:
        raise AgentFailure(
            "client's tests failed or discovered no tests: " + result.stderr[-2000:]
        )
    return {
        "stage": stage,
        "oracle": "pass",
        "unit_tests": result.stderr[-2000:],
        "source_sha256": hashlib.sha256(
            (workspace / "ledger.py").read_bytes()
        ).hexdigest(),
    }


class ClientRun:
    def __init__(
        self,
        name,
        path,
        folder,
        model,
        context,
        timeout,
        input_modalities,
        version=None,
    ):
        self.name, self.path, self.folder = name, path, folder
        self.model, self.context, self.timeout = model, context, timeout
        # What the served model accepts, as /v1/models reports it.
        self.input_modalities = input_modalities
        # The client's major version, on which only OpenCode's launch depends,
        # as for splash.
        self.version = version
        self.workspace = (folder / "project").resolve()
        self.session = None
        self.phases = []
        self.codex_home = (folder / "codex-home").resolve()
        self.pi_home = (folder / "pi-agent").resolve()
        self.opencode_data = (folder / "opencode-data").resolve()
        # A profile of the developer's own Hermes root: Hermes takes any other
        # home for a root of its own, where it would install its tools and to
        # which it would point the developer's hermes command. finish_hermes
        # moves it into this folder.
        self.hermes_profile = f"splash-test-{secrets.token_hex(4)}"
        self.hermes_home = clients.hermes_profile_home(os.environ, self.hermes_profile)
        self.hermes_profiles_existed = self.hermes_home.parent.is_dir()
        # OpenCode's record of the session, as the last phase exported it.
        self.opencode = {"turns": [], "compactions": []}
        folder.mkdir(parents=True)
        fixture(self.workspace)

    def argv(self):
        if self.name == "claude":
            # Normal edit authorization and one explicit project test command;
            # --allowedTools grants permission, unlike --tools it does not filter
            # the registered tool inventory. Production still defaults to default.
            arguments = [
                "--print",
                "--output-format",
                "stream-json",
                "--verbose",
                "--permission-mode",
                "acceptEdits",
                "--allowedTools",
                f"Bash({TEST_COMMAND})",
            ]
            if self.session:
                arguments += ["--resume", self.session]
        elif self.name == "opencode":
            arguments = ["run", "--format", "json"]
            if self.session:
                arguments += ["--session", self.session]
        elif self.name == "codex":
            # Test overrides leave the shipped launcher profile unchanged.
            arguments = [
                "exec",
                "--sandbox",
                "workspace-write",
                *os.environ.get("SPLASH_TEST_CODEX_ARGS", "").split(),
            ]
            if self.session:
                arguments += ["resume", self.session]
            arguments += ["--json", "-"]
        elif self.name == "pi":
            arguments = ["--print", "--mode", "json"]
            if self.session:
                arguments += ["--session", self.session]
        else:
            arguments = ["chat", "--oneshot", "--query-file", "-"]
            if self.session:
                arguments += ["--resume", self.session]
        return self.command(arguments)

    def command(self, arguments):
        """The client's command, as `splash NAME -- ARGUMENTS` runs it, with
        the client's own state in this run's folder, leaving the developer's
        untouched: Pi's agent directory (providers, sessions, settings and
        extensions), Codex's home, and OpenCode's data directory, whose
        session database OpenCode 2 migrates to a schema OpenCode 1 cannot
        open. Hermes runs in this run's own profile."""
        environment = dict(os.environ)
        match self.name:
            case "pi":
                environment["PI_CODING_AGENT_DIR"] = str(self.pi_home)
            case "codex":
                self.codex_home.mkdir(exist_ok=True)
                environment["CODEX_HOME"] = str(self.codex_home)
            case "opencode":
                environment["XDG_DATA_HOME"] = str(self.opencode_data)
        argv, env = clients.command(
            self.name,
            self.path,
            BASE_URL,
            self.model,
            self.context,
            environment,
            input_modalities=self.input_modalities,
            client_args=arguments,
            client_version=self.version,
            hermes_profile=self.hermes_profile,
        )
        # subprocess(cwd=...) does not update inherited PWD. Keep both views
        # consistent, just as a user shell entering the project would.
        env["PWD"] = str(self.workspace)
        return argv, env

    def hermes_messages(self):
        path = self.hermes_home / "state.db"
        # Hermes creates it with its first session; the phase reports its absence.
        if not path.exists():
            return []
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            if self.session is None:
                row = db.execute(
                    "SELECT id FROM sessions WHERE cwd=? ORDER BY started_at DESC LIMIT 1",
                    (str(self.workspace),),
                ).fetchone()
                if row:
                    self.session = row["id"]
            return [
                dict(r)
                for r in db.execute(
                    "SELECT id,role,content,tool_name,tool_calls,tool_call_id,active,_compressed_summary "
                    "FROM messages WHERE session_id=? ORDER BY id",
                    (self.session,),
                )
            ]

    def phase(self, label, prompt, cancel=False, may_compact=False):
        before = idle_status()
        previous_message = 0
        if self.name == "hermes" and self.session:
            previous_message = max((m["id"] for m in self.hermes_messages()), default=0)
        argv, env = self.argv()
        log = self.folder / f"{label}.log"
        started = time.monotonic()
        samples, interrupted, reason = [], False, None
        with log.open("x") as output:
            process = subprocess.Popen(
                argv,
                cwd=self.workspace,
                env=env,
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                process.stdin.write(prompt + "\n")
                process.stdin.close()
                critical_since = None
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    sample = {"elapsed": elapsed, **memory_sample()}
                    samples.append(sample)
                    # Readiness is required before/after a phase. During work,
                    # unavailable fresh telemetry is not a failed user request;
                    # the client deadline, the OS pressure guard and the bounded
                    # engine-critical window still apply.
                    current = status(wait_for_fresh=False, tolerate_critical=True)
                    sample["status_stale"] = current is None
                    sample["engine_critical"] = current is not None and not current.get(
                        "ready"
                    )
                    if current is not None:
                        # Enough engine telemetry per second to explain a stall
                        # or a readiness change after the fact.
                        governor = current.get("memory_governor", {})
                        scheduler = current.get("scheduler", {})
                        sample["engine"] = {
                            "ready": current.get("ready"),
                            "pressure": current.get("memory_pressure"),
                            "host_available_gib": round(
                                governor.get("host_available_bytes", 0) / 2**30, 2
                            ),
                            "growth_allowed": governor.get("growth_allowed"),
                            "prefill_rows": scheduler.get("prefill_rows"),
                            "decode_batches": scheduler.get("decode_batches"),
                            "kv_pages_cache": current.get("kv", {}).get("pages_cache"),
                            "state_entries": current.get("state", {}).get("entries"),
                        }
                    if sample["engine_critical"]:
                        critical_since = (
                            elapsed if critical_since is None else critical_since
                        )
                        if elapsed - critical_since > ENGINE_CRITICAL_GRACE_SECONDS:
                            reason = (
                                f"engine memory pressure critical for "
                                f"{elapsed - critical_since:.0f} s"
                            )
                            break
                    else:
                        critical_since = None
                    if sample["pressure"] >= pressure_stop_level():
                        reason = (
                            f"OS memory pressure level {sample['pressure']} reached "
                            f"SPLASH_TEST_PRESSURE_STOP={pressure_stop_level()}; "
                            "stopped for desktop safety"
                        )
                        break
                    if (
                        cancel
                        and current is not None
                        and current["requests"]["submitted"]
                        > before["requests"]["submitted"]
                        and any(
                            current["scheduler"][k]
                            for k in ("prefilling", "decoding", "waiting_mask")
                        )
                    ):
                        interrupted = True
                        break
                    if elapsed > self.timeout:
                        reason = "bounded client timeout"
                        break
                    time.sleep(1)
            except KeyboardInterrupt:
                reason = "test interrupted"
            except Exception as error:
                reason = f"monitor failed: {error}"
            finally:
                stop_process(process)
        text = log.read_text(errors="replace")
        parsed = events(text)
        for e in parsed:
            if self.name == "pi" and e.get("type") == "session":
                self.session = e.get("id", self.session)
            self.session = e.get(
                "session_id", e.get("sessionID", e.get("thread_id", self.session))
            )
        messages = (
            [m for m in self.hermes_messages() if m["id"] > previous_message]
            if self.name == "hermes"
            else []
        )
        # The turns this phase added to OpenCode's record of the session.
        turns = []
        if self.name == "opencode" and self.session:
            previous = self.opencode
            try:
                self.opencode = opencode_session(self.opencode_history(), self.version)
            except Exception as error:
                reason = reason or f"session export failed: {error}"
            turns = self.opencode["turns"][len(previous["turns"]) :]
        try:
            after = idle_status()
        except Exception as error:
            after = None
            reason = reason or f"final status failed: {error}"
        row = {
            "phase": label,
            "exit_code": process.returncode,
            "wall_seconds": time.monotonic() - started,
            "interrupted": interrupted,
            "error": reason,
            "session": self.session,
            "command": argv,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_chars": len(prompt),
            "before": before,
            "after": after,
            "memory_samples": samples,
            "log": str(log),
            "executed_commands": executed_commands(self.name, parsed, messages, turns),
        }
        if after is not None:
            row["reuse"] = prefix_reuse(before, after)
        self.phases.append(row)
        atomic_json(self.folder / f"{label}.json", row)
        atomic_json(self.folder / "session.json", {"session": self.session})
        if reason == "test interrupted":
            raise KeyboardInterrupt
        if reason:
            raise AgentFailure(reason)
        if not self.session and self.name == "hermes":
            raise AgentFailure(
                f"Hermes exited {process.returncode} without a session record in "
                f"{self.hermes_home / 'state.db'}; see {log}"
            )
        if not self.session:
            raise AgentFailure("client did not expose a real session id")
        if after["requests"]["failed"] != before["requests"]["failed"]:
            raise AgentFailure("native request failed")
        reuse = row["reuse"]
        # Running work takes the replay points unfinished requests hold after
        # everything else; at a tight --max-memory that is legitimate, so it
        # is recorded, not gated.
        if reuse["in_use_evictions"]:
            print(
                f"{self.name}/{label}: warning: {reuse['in_use_evictions']} replay "
                "points of unfinished requests were evicted",
                flush=True,
            )
        if cancel:
            if (
                not interrupted
                or after["requests"]["cancelled"] <= before["requests"]["cancelled"]
            ):
                raise AgentFailure("did not observe an in-flight request cancellation")
        else:
            if process.returncode:
                raise AgentFailure(f"client exited {process.returncode}; see {log}")
            # Clients may cancel their own auxiliary title/summary requests
            # at exit. Count those in before/after telemetry, but judge the
            # user turn by its native completion event and artifact checks.
            if after["requests"]["completed"] <= before["requests"]["completed"]:
                raise AgentFailure("no successful real model request")
            if self.name == "hermes":
                if (
                    not messages
                    or messages[-1]["role"] != "assistant"
                    or not messages[-1]["content"]
                ):
                    raise AgentFailure(
                        "Hermes exited without a completed assistant response"
                    )
            elif self.name == "claude":
                if not any(
                    e.get("type") == "result"
                    and e.get("subtype") == "success"
                    and not e.get("is_error")
                    for e in parsed
                ):
                    raise AgentFailure("Claude did not report successful completion")
            elif self.name == "codex":
                if not any(e.get("type") == "turn.completed" for e in parsed):
                    raise AgentFailure("Codex did not complete its turn")
            elif self.name == "pi":
                if not pi_completed(parsed):
                    raise AgentFailure(
                        "Pi did not finish its user turn with assistant text"
                    )
            else:
                if any(e.get("type") == "error" for e in parsed):
                    raise AgentFailure("OpenCode reported a request error")
                # Its session records how the turn ended; OpenCode 2 prints no
                # final step_finish to say so.
                if not opencode_completed(turns):
                    raise AgentFailure(
                        "OpenCode did not finish its user turn with assistant text"
                    )
            # A phase that may compact is judged once its caller knows
            # whether it did.
            if not may_compact:
                self.require_reuse()
        print(
            f"{self.name}/{label}: completed ({row['wall_seconds']:.1f}s)", flush=True
        )
        return parsed

    def require_reuse(self):
        """Fails the last phase when two or more of its requests completed and
        none reused a cached prompt token. Every request after a phase's
        first resends the conversation, so a working replay point always
        reuses some of it."""
        reuse = self.phases[-1]["reuse"]
        if reuse["completed"] >= 2 and not reuse["reused_tokens"]:
            raise AgentFailure(
                "phase reused no cached prompt tokens across "
                f"{reuse['completed']} requests"
            )

    def compaction(self):
        if self.name == "hermes":
            return [
                {"message_id": m["id"]}
                for m in self.hermes_messages()
                if m["_compressed_summary"] and m["active"]
            ]
        if self.name == "codex":
            home = self.codex_home
            found = list((home / "sessions").rglob(f"*{self.session}*.jsonl"))
            return [
                {"timestamp": e.get("timestamp"), "file": str(p)}
                for p in found
                for e in events(p.read_text())
                if e.get("type") == "compacted"
            ]
        if self.name == "pi":
            found = (self.pi_home / "sessions").rglob(f"*_{self.session}.jsonl")
            return [
                entry
                for path in found
                for entry in events(path.read_text())
                if entry.get("type") == "compaction"
            ]
        if self.name == "opencode":
            return self.opencode["compactions"]
        return [
            e
            for p in self.folder.glob("*.log")
            for e in events(p.read_text())
            if e.get("type") == "system"
            and e.get("subtype") == "compact_boundary"
            and e.get("compact_metadata", {}).get("trigger") == "auto"
        ]

    def opencode_history(self):
        """OpenCode's own record of the session, which OpenCode 2 exports
        with its session command."""
        export = ["session", "export"] if self.version >= 2 else ["export"]
        argv, env = self.command([*export, self.session])
        # A regular file avoids losing buffered pipe output when the CLI
        # exits immediately after printing a large session export.
        with tempfile.TemporaryFile(mode="w+") as output:
            result = subprocess.run(
                argv,
                env=env,
                cwd=self.workspace,
                stdout=output,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
            )
            if result.returncode:
                raise AgentFailure(
                    f"exit status {result.returncode}: {result.stderr[-2000:]}"
                )
            output.seek(0)
            history = json.load(output)
        atomic_json(self.folder / "history.json", history)
        return history

    def finish_hermes(self):
        """Move this run's Hermes profile, with its native history, into the
        run's folder, leaving the developer's Hermes root as it was."""
        if self.hermes_home.exists():
            shutil.move(self.hermes_home, self.folder / "hermes-profile")
        if not self.hermes_profiles_existed:
            try:
                self.hermes_home.parent.rmdir()
            except OSError:
                pass

    def check_artifact(self, stage):
        if not any(
            RAN_TESTS.search(command)
            for command in self.phases[-1]["executed_commands"]
        ):
            raise AgentFailure(
                "client did not successfully execute the project test command"
            )
        return validate_artifact(self.workspace, stage)

    def run(self, scenario, reference):
        self.phase(
            "implement",
            "Read README.md, implement the ledger, add the requested unit tests, "
            f"and run {TEST_COMMAND}. Report the result briefly.",
        )
        checks = [self.check_artifact(0)]
        self.phase(
            "resume-edit",
            "Extend the ledger to exclude records whose void flag is true. "
            "Keep all earlier behavior and update the documentation and tests. "
            f"Run {TEST_COMMAND}.",
        )
        checks.append(self.check_artifact(1))
        if scenario == "smoke":
            return {"artifact_checks": checks}
        return self.finish(reference, checks)

    def finish(self, reference, checks):
        compact = self.compaction()
        first_wave = 1 + max(
            (int(p.stem.split("-")[-1]) for p in self.folder.glob("reference-*.log")),
            default=0,
        )
        for wave in range(first_wave, first_wave + 20):
            if compact:
                break
            self.phase(
                f"reference-{wave:02}",
                reference.replace("BATCH_ID", str(wave)),
                may_compact=True,
            )
            compact = self.compaction()
            # The wave that compacts resends no conversation: its summary
            # request and the request after it share only the system prompt
            # and tools.
            if not compact:
                self.require_reuse()
        if not compact:
            raise AgentFailure("no genuine automatic compaction observed")
        self.phase(
            "post-compact-edit",
            "Now add an optional category=None argument to summarize. "
            "When a category is provided, include only that category. Preserve every earlier "
            f"rule, update tests and run {TEST_COMMAND}.",
        )
        checks.append(self.check_artifact(2))
        self.phase(
            "cancel",
            "Give a detailed code review of the ledger and discuss edge cases. "
            "Do not modify files.",
            cancel=True,
        )
        self.phase(
            "cancel-recovery",
            f"Read the current ledger and run {TEST_COMMAND}. "
            "Do not change files; report whether the existing tests pass.",
        )
        checks.append(self.check_artifact(2))
        return {"artifact_checks": checks, "automatic_compaction": compact}


def reference_fixture(tokenizer_path, context):
    # Size data with the installed model tokenizer, without loading weights.
    # The client's native history, not this size estimate, proves compaction.
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    prefix = (
        "Here is archived receipt data for the ongoing ledger project, batch BATCH_ID. "
        "Keep it as historical reference; do not change the implementation yet. "
        "Acknowledge briefly.\n"
    )
    rows = [
        json.dumps(
            {
                "receipt": f"ARCHIVE-{i:05}",
                "amount_cents": i * 73 - 4000,
                "category": "food" if i % 2 else "travel",
                "note": "historical receipt",
            }
        )
        for i in range(1024)
    ]
    low, high, chosen = 1, len(rows), prefix
    while low <= high:
        middle = (low + high) // 2
        prompt = prefix + "\n".join(rows[:middle])
        if len(tokenizer.encode(prompt).ids) <= min(8192, context // 10):
            chosen, low = prompt, middle + 1
        else:
            high = middle - 1
    return chosen


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", default=",".join(CLIENTS))
    parser.add_argument(
        "--model",
        type=serve_options.parse_model_id,
        required=True,
    )
    # The installation's source options, which splash serve is given, and
    # its selection link (install/models.py link), which they name by default.
    parser.add_argument("--revision")
    parser.add_argument(
        "--draft-model", type=launcher.model_artifacts.parse_draft_model
    )
    parser.add_argument("--language-only", action="store_true")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--max-context", default="100K")
    # Complete runs include several long-context turns and can take minutes.
    parser.add_argument("--client-timeout", type=float, default=900)
    parser.add_argument("--scenario", choices=("smoke", "complete"), default="complete")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--http-smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/release/agent-real.json"
    )
    args = parser.parse_args(argv)
    if args.model_root is None:
        args.model_root = launcher.model_artifacts.Selection.of(
            launcher.model_artifacts.MODELS,
            args.model,
            revision=args.revision,
            language_only=args.language_only,
            draft_model=args.draft_model,
        ).link
    return args


def main(argv=None):
    args = parse_args(argv)
    selected = args.clients.split(",")
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(n not in CLIENTS for n in selected)
    ):
        raise AgentFailure("--clients must contain unique known client names")
    versions = {}
    for name in selected:
        path = clients.find_executable(name)
        result = subprocess.run(
            [path, "--version"], text=True, capture_output=True, timeout=20
        )
        if result.returncode:
            raise AgentFailure(f"cannot determine {name} version")
        versions[name] = {
            "path": path,
            "version": (result.stdout or result.stderr).strip(),
        }
        # OpenCode's launch depends on its major version, read as splash reads
        # it. The launcher starts a version it cannot read as OpenCode 1,
        # which would run OpenCode 2 through its background service.
        if name == "opencode":
            major = clients.major_version(result.stdout)
            if major is None:
                raise AgentFailure(
                    f"cannot determine opencode's major version: {result.stdout!r}"
                )
            versions[name]["major_version"] = major
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        run_installed_client_tests(versions)
        atomic_json(
            args.output,
            {"schema_version": 2, "result": "preflight_only", "clients": versions},
        )
        print(json.dumps(versions, indent=2))
        return 0
    directory = Path(tempfile.mkdtemp(prefix="agent-real-", dir=args.output.parent))
    document = {
        "schema_version": 2,
        "result": "running",
        "scenario": args.scenario,
        "artifacts": str(directory),
        "clients": {},
    }
    process = None
    try:
        identity = current_build_id()
        if launcher._request_json("/status") is None:
            command = [
                str(ROOT / "splash"),
                "serve",
                "--max-context",
                args.max_context,
                "--model",
                args.model,
                *(("--revision", args.revision) if args.revision else ()),
                *(("--draft-model", args.draft_model) if args.draft_model else ()),
                *(("--language-only",) if args.language_only else ()),
            ]
            with (directory / "server.log").open("x") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            deadline = time.monotonic() + 900
            while launcher._request_json("/status", timeout=10) is None:
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise AgentFailure(
                        f"serve did not become ready; see {directory / 'server.log'}"
                    )
                time.sleep(0.5)
        initial = idle_status()
        # Clients get the first entry, the name responses report, as with
        # splash <client>; /status names the loaded model.
        served = launcher._request_json("/v1/models")["data"][0]
        model = served["id"]
        context = initial["maximum_context_tokens"]
        validate_server_configuration(
            initial,
            args.model,
            serve_options.parse_max_context(args.max_context),
            identity,
        )
        document.update(model=model, context=context, identity=initial["identity"])
        document["client_adapter_sha256"] = hashlib.sha256(
            (ROOT / "install/clients.py").read_bytes()
        ).hexdigest()
        document["validation_script_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        port = serve_options.DEFAULT_PORT
        if args.http_smoke:
            smoke_real.run(port, model)
        reference = (
            reference_fixture(args.model_root / "tokenizer", context)
            if args.scenario == "complete"
            else ""
        )
        for name in selected:
            runner = ClientRun(
                name,
                versions[name]["path"],
                directory / name,
                model,
                context,
                args.client_timeout,
                served["input_modalities"],
                version=versions[name].get("major_version"),
            )
            entry = {**versions[name], "result": "running", "phases": runner.phases}
            document["clients"][name] = entry
            try:
                entry.update(runner.run(args.scenario, reference), result="pass")
            except Exception as error:
                entry.update(result="fail", error=str(error))
                print(f"{name}: FAIL: {error}", flush=True)
            finally:
                runner.finish_hermes()
            atomic_json(args.output, document)
            remaining = selected[selected.index(name) + 1 :]
            if remaining and memory_sample()["pressure"] >= pressure_stop_level():
                raise AgentFailure(
                    "stopping remaining clients: OS memory pressure reached "
                    f"SPLASH_TEST_PRESSURE_STOP={pressure_stop_level()}"
                )
        document["final_status"] = idle_status()
        document["result"] = (
            "pass"
            if all(v["result"] == "pass" for v in document["clients"].values())
            else "fail"
        )
        return 0 if document["result"] == "pass" else 1
    except KeyboardInterrupt:
        document.update(result="interrupted")
        for entry in document["clients"].values():
            if entry["result"] == "running":
                entry.update(result="interrupted")
        return 130
    except Exception as error:
        document.update(result="fail", error=str(error))
        raise
    finally:
        atomic_json(args.output, document)
        if process is not None:
            stop_process(process)


if __name__ == "__main__":
    # A shell starts a background job with SIGINT ignored, and the clients
    # would inherit that: stop_process could not interrupt them, and this
    # run could not be interrupted either. Handled here, the signal is back
    # to its default in every program the run starts.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
