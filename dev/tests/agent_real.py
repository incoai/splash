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

CLIENTS = tuple(clients.INSTALL_URLS)
TEST_COMMAND = "python3 -m unittest -v"


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


def validate_server_configuration(initial, model, expected_model, context, identity):
    if model != expected_model or initial["maximum_context_tokens"] != context:
        raise AgentFailure(
            "running server model/context differs from the test configuration"
        )
    if initial.get("identity", {}).get("cache", {}).get("build_id") != identity:
        raise AgentFailure(
            "running server native build differs from the verified build"
        )


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


def executed_commands(name, parsed, messages=()):
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
        if name == "opencode" and event.get("type") == "tool_use":
            part = event.get("part", {})
            state = part.get("state", {})
            if (
                part.get("tool") == "bash"
                and state.get("status") == "completed"
                and state.get("metadata", {}).get("exit", 0) == 0
            ):
                commands.append(state.get("input", {}).get("command", ""))
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
    # window, which clears once shed memory is released at the paced rate.
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
            and value["state"]["active_cells"] == 0
            and value["state"]["pinned"] == 0
            and value["kv"]["pages_active"] == 0
        ):
            return value
        time.sleep(0.25)
    raise AgentFailure("server did not return to idle")


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
    result = subprocess.run(
        TEST_COMMAND.split(),
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
    def __init__(self, name, path, folder, model, context, timeout):
        self.name, self.path, self.folder = name, path, folder
        self.model, self.context, self.timeout = model, context, timeout
        self.workspace = (folder / "project").resolve()
        self.session = None
        self.phases = []
        self.codex_home = (folder / "codex-home").resolve()
        folder.mkdir(parents=True)
        fixture(self.workspace)

    def argv(self):
        argv, env = clients.command(
            self.name,
            self.path,
            launcher.BASE_URL,
            self.model,
            self.context,
            launcher.RUNTIME_DIR,
        )
        # subprocess(cwd=...) does not update inherited PWD. Keep both views
        # consistent, just as a user shell entering the project would.
        env["PWD"] = str(self.workspace)
        if self.name == "claude":
            # Normal edit authorization and one explicit project test command;
            # --allowedTools grants permission, unlike --tools it does not filter
            # the registered tool inventory. Production still defaults to default.
            argv += [
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
                argv += ["--resume", self.session]
        elif self.name == "opencode":
            argv += ["run", "--format", "json"]
            if self.session:
                argv += ["--session", self.session]
        elif self.name == "codex":
            self.codex_home.mkdir(exist_ok=True)
            env["CODEX_HOME"] = str(self.codex_home)
            argv += ["exec", "--sandbox", "workspace-write"]
            # Test overrides leave the shipped launcher profile unchanged.
            argv += os.environ.get("SPLASH_TEST_CODEX_ARGS", "").split()
            if self.session:
                argv += ["resume", self.session]
            argv += ["--json", "-"]
        elif self.name == "pi":
            argv += [
                "--print",
                "--mode",
                "json",
                "--session-dir",
                str((self.folder / "pi-sessions").resolve()),
            ]
            if self.session:
                argv += ["--session", self.session]
        else:
            argv += ["--oneshot", "--query-file", "-"]
            if self.session:
                argv += ["--resume", self.session]
        return argv, env

    def hermes_messages(self):
        path = launcher.RUNTIME_DIR / "hermes/state.db"
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

    def phase(self, label, prompt, cancel=False):
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
                            "kv_blocks": current.get("kv", {}).get("blocks"),
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
                            f"OS memory pressure level {sample['pressure']}; "
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
            "executed_commands": executed_commands(self.name, parsed, messages),
        }
        self.phases.append(row)
        atomic_json(self.folder / f"{label}.json", row)
        atomic_json(self.folder / "session.json", {"session": self.session})
        if reason == "test interrupted":
            raise KeyboardInterrupt
        if reason:
            raise AgentFailure(reason)
        if not self.session:
            raise AgentFailure("client did not expose a real session id")
        if after["requests"]["failed"] != before["requests"]["failed"]:
            raise AgentFailure("native request failed")
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
                if not any(
                    e.get("type") == "step_finish"
                    and e.get("part", {}).get("reason") == "stop"
                    for e in parsed
                ):
                    raise AgentFailure("OpenCode did not finish its user turn")
                if not any(
                    e.get("type") == "text"
                    and e.get("part", {}).get("text", "").strip()
                    for e in parsed
                ):
                    raise AgentFailure("OpenCode produced no assistant text")
        print(
            f"{self.name}/{label}: completed ({row['wall_seconds']:.1f}s)", flush=True
        )
        return parsed

    def compaction(self):
        if self.name == "pi":
            return [
                entry
                for path in (self.folder / "pi-sessions").glob(
                    f"*_{self.session}.jsonl"
                )
                for entry in events(path.read_text())
                if entry.get("type") == "compaction"
            ]
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
        if self.name == "opencode":
            argv, env = clients.command(
                self.name,
                self.path,
                launcher.BASE_URL,
                self.model,
                self.context,
                launcher.RUNTIME_DIR,
            )
            env["PWD"] = str(self.workspace)
            # A regular file avoids losing buffered pipe output when the CLI
            # exits immediately after printing a large session export.
            with tempfile.TemporaryFile(mode="w+") as output:
                result = subprocess.run(
                    argv + ["export", self.session],
                    env=env,
                    cwd=self.workspace,
                    stdout=output,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=20,
                )
                if result.returncode:
                    raise AgentFailure("OpenCode session export failed")
                output.seek(0)
                history = json.load(output)
            atomic_json(self.folder / "history.json", history)
            return [
                part
                for message in history["messages"]
                for part in message["parts"]
                if part.get("type") == "compaction" and part.get("auto")
            ]
        return [
            e
            for p in self.folder.glob("*.log")
            for e in events(p.read_text())
            if e.get("type") == "system"
            and e.get("subtype") == "compact_boundary"
            and e.get("compact_metadata", {}).get("trigger") == "auto"
        ]

    def check_artifact(self, stage):
        if not any(
            "python3 -m unittest" in command
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
            self.phase(f"reference-{wave:02}", reference.replace("BATCH_ID", str(wave)))
            compact = self.compaction()
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
        type=launcher.model_artifacts.parse_repo_id,
        required=True,
    )
    parser.add_argument("--max-context", default="100K")
    # Complete runs include several long-context turns and can take minutes.
    parser.add_argument("--client-timeout", type=float, default=900)
    parser.add_argument("--scenario", choices=("smoke", "complete"), default="complete")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--http-smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/release/agent-real.json"
    )
    return parser.parse_args(argv)


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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
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
            while launcher._running_status() is None:
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise AgentFailure(
                        f"serve did not become ready; see {directory / 'server.log'}"
                    )
                time.sleep(0.5)
        initial = idle_status()
        model = launcher._request_json("/v1/models")["data"][0]["id"]
        context = initial["maximum_context_tokens"]
        validate_server_configuration(
            initial,
            model,
            args.model,
            launcher._parse_max_context(args.max_context),
            identity,
        )
        document.update(model=model, context=context, identity=initial["identity"])
        document["client_adapter_sha256"] = hashlib.sha256(
            (ROOT / "install/clients.py").read_bytes()
        ).hexdigest()
        document["validation_script_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        port = int(launcher.BASE_URL.rsplit(":", 1)[1])
        if args.http_smoke:
            smoke_real.run(port, model)
        installed = launcher.model_artifacts.installed_root(
            launcher.model_artifacts.MODELS, args.model
        )
        reference = (
            reference_fixture(installed / "tokenizer", context)
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
            )
            entry = {**versions[name], "result": "running", "phases": runner.phases}
            document["clients"][name] = entry
            try:
                entry.update(runner.run(args.scenario, reference), result="pass")
            except Exception as error:
                entry.update(result="fail", error=str(error))
                print(f"{name}: FAIL: {error}", flush=True)
            atomic_json(args.output, document)
            remaining = selected[selected.index(name) + 1 :]
            if remaining and memory_sample()["pressure"] >= pressure_stop_level():
                raise AgentFailure("stopping remaining clients for OS memory pressure")
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
    raise SystemExit(main())
