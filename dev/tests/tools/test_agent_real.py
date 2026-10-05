import contextlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests import agent_real as agent
from server import serve_options

MODEL_IDS = (
    "incoai/Qwen3.8-27B-Splash",
    "incoai/Qwen3.6-35B-A3B-Splash",
    "community/custom-splash",
)


def idle(
    submitted=0,
    completed=0,
    cancelled=0,
    failed=0,
    reused=0,
    prefill=0,
    in_use_evictions=0,
):
    """An idle status with these request, cache, state and metric counters."""
    return {
        "requests": {
            "submitted": submitted,
            "completed": completed,
            "cancelled": cancelled,
            "failed": failed,
        },
        "cache": {"reused_tokens": reused, "lost_state_misses": 0},
        "state": {"in_use_evictions": in_use_evictions},
        "metrics": {"prefill_input_tokens": prefill},
    }


class AgentRunnerTests(unittest.TestCase):
    def test_pi_counts_only_successful_bash_results(self):
        for is_error in (False, True):
            with self.subTest(is_error=is_error):
                parsed = [
                    {
                        "type": "tool_execution_start",
                        "toolName": "bash",
                        "toolCallId": "t1",
                        "args": {"command": agent.TEST_COMMAND},
                    },
                    {
                        "type": "tool_execution_end",
                        "toolName": "bash",
                        "toolCallId": "unknown",
                        "isError": False,
                    },
                    {
                        "type": "tool_execution_end",
                        "toolName": "bash",
                        "toolCallId": "t1",
                        "isError": is_error,
                    },
                ]
                self.assertEqual(
                    agent.executed_commands("pi", parsed),
                    [] if is_error else [agent.TEST_COMMAND],
                )

    def test_pi_completion_requires_final_successful_assistant_text(self):
        def message(reason, text):
            return {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": reason,
                    "content": [{"type": "text", "text": text}],
                },
            }

        success = message("stop", "Done")
        ended = {"type": "agent_end"}
        for parsed, expected in (
            ([success, ended], True),
            ([message("toolUse", ""), success, ended], True),
            ([success], False),
            ([ended], False),
            ([message("stop", " "), ended], False),
            ([success, message("error", "failed"), ended], False),
            ([message("aborted", "partial"), ended], False),
            ([message("length", "partial"), ended], False),
        ):
            with self.subTest(parsed=parsed):
                self.assertEqual(agent.pi_completed(parsed), expected)

    def test_pi_compaction_reads_the_selected_saved_session(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = agent.ClientRun(
                "pi", "/bin/pi", Path(directory) / "run", "test-model", 102400, 10, []
            )
            runner.session = "selected-id"
            # Pi saves sessions per working directory under its agent directory.
            sessions = runner.pi_home / "sessions/--project--"
            sessions.mkdir(parents=True)
            entry = {"type": "compaction", "summary": "Earlier work"}
            for name in ("time_selected-id.jsonl", "time_other-id.jsonl"):
                (sessions / name).write_text(
                    json.dumps({"type": "message"}) + "\n" + json.dumps(entry) + "\n"
                )
            self.assertEqual(runner.compaction(), [entry])

    def test_hermes_session_lookup_uses_canonical_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            physical = root / "physical"
            physical.mkdir()
            alias = root / "alias"
            alias.symlink_to(physical, target_is_directory=True)
            runner = agent.ClientRun(
                "hermes", "hermes", alias / "run", "test-model", 102400, 10, ["text"]
            )
            workspace = physical / "run/project"
            runner.hermes_home = root / "profile"
            runner.hermes_home.mkdir()
            with (
                contextlib.closing(
                    sqlite3.connect(runner.hermes_home / "state.db")
                ) as db,
                db,
            ):
                db.executescript(
                    "CREATE TABLE sessions (id TEXT, cwd TEXT, started_at INTEGER);"
                    "CREATE TABLE messages (id INTEGER, session_id TEXT, role TEXT, "
                    "content TEXT, tool_name TEXT, tool_calls TEXT, tool_call_id TEXT, "
                    "active INTEGER, _compressed_summary INTEGER);"
                )
                db.executemany(
                    "INSERT INTO sessions VALUES (?, ?, ?)",
                    [("matching", str(workspace), 1), ("other", str(root), 2)],
                )
                db.execute(
                    "INSERT INTO messages VALUES (1, 'matching', 'assistant', "
                    "'Task complete', NULL, NULL, NULL, 1, 0)"
                )
            with mock.patch.object(
                agent.clients, "command", return_value=(["hermes"], {})
            ):
                messages = runner.hermes_messages()
                _, environment = runner.argv()
            self.assertEqual(runner.workspace, workspace)
            self.assertEqual(environment["PWD"], str(workspace))
            self.assertEqual(runner.session, "matching")
            self.assertEqual(
                [message["content"] for message in messages], ["Task complete"]
            )

    def test_server_configuration_rejects_wrong_model_context_or_build(self):
        identity = "src-" + "a" * 64
        context = 102400
        for selected in MODEL_IDS:
            with self.subTest(model=selected):
                initial = {
                    "instance": {"model": selected},
                    "maximum_context_tokens": context,
                    "identity": {"cache": {"build_id": identity}},
                }
                agent.validate_server_configuration(
                    initial, selected, context, identity
                )
                other = next(value for value in MODEL_IDS if value != selected)
                for model, expected_context, expected_identity in (
                    (other, context, identity),
                    (selected, context + 1, identity),
                    (selected, context, "src-" + "b" * 64),
                ):
                    with self.assertRaises(agent.AgentFailure):
                        agent.validate_server_configuration(
                            initial, model, expected_context, expected_identity
                        )
                with self.assertRaisesRegex(agent.AgentFailure, "native build"):
                    agent.validate_server_configuration(
                        {
                            "instance": {"model": selected},
                            "maximum_context_tokens": context,
                        },
                        selected,
                        context,
                        identity,
                    )

    def test_expected_build_requires_matching_stamp_header_and_native_binary(self):
        identity = "src-" + "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "build/engine"
            generated.mkdir(parents=True)
            (generated / "BuildIdentity.hpp").write_bytes(
                agent.build_identity.header_bytes(identity)
            )
            (generated / "build-identity.json").write_bytes(
                agent.build_identity.stamp_bytes(identity)
            )
            binary = root / "build/splash"
            binary.write_bytes(identity.encode())
            with mock.patch.object(agent, "ROOT", root):
                self.assertEqual(agent.current_build_id(), identity)
                binary.write_bytes(("src-" + "b" * 64).encode())
                with self.assertRaisesRegex(agent.AgentFailure, "identity"):
                    agent.current_build_id()
                binary.unlink()
                with self.assertRaisesRegex(agent.AgentFailure, "identity"):
                    agent.current_build_id()

    def test_mismatched_existing_server_is_rejected_without_stopping_it(self):
        model = "incoai/Qwen3.8-27B-Splash"
        identity = "src-" + "a" * 64
        initial = {
            "instance": {"model": model},
            "maximum_context_tokens": serve_options.parse_max_context("100K"),
            "identity": {"cache": {"build_id": "src-" + "b" * 64}},
        }
        # An announced alias leads /v1/models while the loaded model matches,
        # so only the build is rejected.
        served = {"data": [{"id": "local"}, {"id": model}]}
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(
                    agent.clients, "find_executable", return_value="client"
                ),
                mock.patch.object(
                    agent.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0, "version", ""),
                ),
                mock.patch.object(agent, "current_build_id", return_value=identity),
                mock.patch.object(
                    agent.launcher,
                    "_request_json",
                    side_effect=[initial, served],
                ),
                mock.patch.object(agent, "idle_status", return_value=initial),
                mock.patch.object(agent.subprocess, "Popen") as start,
                mock.patch.object(agent, "stop_process") as stop,
            ):
                with self.assertRaisesRegex(agent.AgentFailure, "native build"):
                    agent.main(
                        [
                            "--model",
                            model,
                            "--clients",
                            "codex",
                            "--output",
                            str(Path(directory) / "report.json"),
                        ]
                    )
                start.assert_not_called()
                stop.assert_not_called()

    def test_model_is_required(self):
        for arguments in ([], ["--preflight-only"]):
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    agent.parse_args(arguments)

    def test_real_harnesses_accept_exactly_the_ids_splash_serve_accepts(self):
        parsers = {
            "splash serve": lambda model: agent.launcher.parse_args(
                ["serve", "--model", model]
            ),
            "agent_real": lambda model: agent.parse_args(["--model", model]),
            "smoke_real": lambda model: agent.smoke_real.parse_args(["--model", model]),
        }
        for model, served in (
            *((model, True) for model in MODEL_IDS),
            ("mlx-community/Qwen3.8-27B-4bit", True),
            ("unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M", True),
            ("unsloth/Qwen3.6-35B-A3B-GGUF:", False),
            ("unsloth/Qwen3.6-35B-A3B-GGUF:UD/Q4_K_M", False),
            ("qwen3.8-27b", False),
        ):
            for name, parse in parsers.items():
                with (
                    self.subTest(parser=name, model=model),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    if served:
                        self.assertEqual(parse(model).model, model)
                    else:
                        with self.assertRaises(SystemExit):
                            parse(model)

    def test_complete_phase_handles_auxiliary_cancellation_and_retains_failures(self):
        for mode in (
            "complete",
            "incomplete",
            "exit_error",
            "monitor_error",
            "interrupt",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                runner = agent.ClientRun.__new__(agent.ClientRun)
                runner.name, runner.session, runner.version = "opencode", "s", 2
                runner.opencode = {"turns": [], "compactions": []}
                runner.folder = runner.workspace = Path(directory)
                runner.timeout, runner.phases = 10, []
                process = mock.Mock(returncode=1 if mode == "exit_error" else 0)
                process.poll.return_value = (
                    None if mode in ("monitor_error", "interrupt") else 0
                )
                before = idle()
                after = idle(submitted=2, completed=1, cancelled=1)

                # OpenCode's record of the turn: stopped, unless incomplete.
                record = {
                    "messages": [
                        {"type": "user"},
                        {
                            "type": "assistant",
                            "finish": None if mode == "incomplete" else "stop",
                            "content": [{"type": "text", "text": "Done"}],
                        },
                    ]
                }
                with (
                    mock.patch.object(runner, "argv", return_value=(["client"], {})),
                    mock.patch.object(runner, "opencode_history", return_value=record),
                    mock.patch.object(
                        agent, "idle_status", side_effect=[before, after]
                    ),
                    mock.patch.object(agent.subprocess, "Popen", return_value=process),
                    mock.patch.object(agent, "stop_process") as stop,
                    mock.patch.object(
                        agent, "memory_sample", return_value={"pressure": 1}
                    ),
                    mock.patch.object(
                        agent,
                        "status",
                        side_effect=KeyboardInterrupt
                        if mode == "interrupt"
                        else RuntimeError("lost status"),
                    ),
                ):
                    if mode == "complete":
                        runner.phase("test", "task")
                    else:
                        with self.assertRaises(
                            KeyboardInterrupt
                            if mode == "interrupt"
                            else agent.AgentFailure
                        ):
                            runner.phase("test", "task")
                    stop.assert_called_once_with(process)
                row = json.loads((runner.folder / "test.json").read_text())
                self.assertEqual(row["after"]["requests"]["cancelled"], 1)
                self.assertEqual(len(runner.phases), 1)
                if mode == "interrupt":
                    self.assertEqual(row["error"], "test interrupted")

    def test_pi_phase_resumes_its_session_and_requires_a_finished_turn(self):
        header = {"type": "session", "id": "pi-session"}
        answer = {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "stop",
                "content": [{"type": "text", "text": "Done"}],
            },
        }
        for finished in (True, False):
            with (
                self.subTest(finished=finished),
                tempfile.TemporaryDirectory() as directory,
            ):
                runner = agent.ClientRun.__new__(agent.ClientRun)
                runner.name, runner.session = "pi", None
                runner.folder = runner.workspace = Path(directory)
                runner.timeout, runner.phases = 10, []
                process = mock.Mock(returncode=0)
                process.poll.return_value = 0

                def launch(*args, **kwargs):
                    ended = [{"type": "agent_end"}] if finished else []
                    for event in [header, answer, *ended]:
                        kwargs["stdout"].write(json.dumps(event) + "\n")
                    return process

                with (
                    mock.patch.object(runner, "argv", return_value=(["pi"], {})),
                    mock.patch.object(
                        agent,
                        "idle_status",
                        side_effect=[idle(), idle(submitted=1, completed=1)],
                    ),
                    mock.patch.object(agent.subprocess, "Popen", side_effect=launch),
                    mock.patch.object(agent, "stop_process"),
                    mock.patch.object(
                        agent, "memory_sample", return_value={"pressure": 1}
                    ),
                ):
                    if finished:
                        runner.phase("test", "task")
                    else:
                        with self.assertRaisesRegex(
                            agent.AgentFailure, "Pi did not finish"
                        ):
                            runner.phase("test", "task")
                self.assertEqual(runner.session, "pi-session")

    def test_phase_requires_prefix_reuse_across_requests(self):
        # Requests after a phase's first resend the conversation: with no
        # prompt token reused, its replay points stopped working. Evicting a
        # replay point an unfinished request holds is recorded, not gated.
        for completed, reused, evictions, error in (
            (3, 0, 0, "phase reused no cached prompt tokens across 3 requests"),
            (3, 64, 0, None),
            (1, 0, 0, None),
            (3, 64, 2, None),
        ):
            with (
                self.subTest(completed=completed, reused=reused, evictions=evictions),
                tempfile.TemporaryDirectory() as directory,
            ):
                runner = agent.ClientRun.__new__(agent.ClientRun)
                runner.name, runner.session = "codex", None
                runner.folder = runner.workspace = Path(directory)
                runner.timeout, runner.phases = 10, []
                process = mock.Mock(returncode=0)
                process.poll.return_value = 0

                def launch(*args, **kwargs):
                    for event in (
                        {"type": "thread.started", "thread_id": "thread"},
                        {"type": "turn.completed"},
                    ):
                        kwargs["stdout"].write(json.dumps(event) + "\n")
                    return process

                after = idle(
                    submitted=completed,
                    completed=completed,
                    reused=reused,
                    prefill=500,
                    in_use_evictions=evictions,
                )
                output = io.StringIO()
                with (
                    mock.patch.object(runner, "argv", return_value=(["codex"], {})),
                    mock.patch.object(
                        agent, "idle_status", side_effect=[idle(prefill=100), after]
                    ),
                    mock.patch.object(agent.subprocess, "Popen", side_effect=launch),
                    mock.patch.object(agent, "stop_process"),
                    mock.patch.object(
                        agent, "memory_sample", return_value={"pressure": 1}
                    ),
                    contextlib.redirect_stdout(output),
                ):
                    if error is None:
                        runner.phase("test", "task")
                    else:
                        with self.assertRaisesRegex(agent.AgentFailure, error):
                            runner.phase("test", "task")
                self.assertEqual(
                    runner.phases[0]["reuse"],
                    {
                        "reused_tokens": reused,
                        "prefill_input_tokens": 400,
                        "lost_state_misses": 0,
                        "in_use_evictions": evictions,
                        "completed": completed,
                    },
                )
                self.assertEqual(
                    "codex/test: warning: 2 replay points" in output.getvalue(),
                    bool(evictions),
                )

    def test_only_the_wave_that_compacts_may_reuse_no_cached_prompt(self):
        # Its summary request and the request after it share only the system
        # prompt and tools; any other reference wave resends the conversation.
        for compactions, error in (
            ([[], []], "phase reused no cached prompt tokens across 2 requests"),
            ([[], [{"auto": True}]], None),
        ):
            with (
                self.subTest(compacts=bool(compactions[-1])),
                tempfile.TemporaryDirectory() as directory,
            ):
                runner = agent.ClientRun.__new__(agent.ClientRun)
                runner.folder, runner.phases = Path(directory), []

                def phase(label, prompt, cancel=False, may_compact=False):
                    runner.phases.append(
                        {"phase": label, "reuse": {"completed": 2, "reused_tokens": 0}}
                    )

                with (
                    mock.patch.object(runner, "compaction", side_effect=compactions),
                    mock.patch.object(runner, "phase", side_effect=phase) as waves,
                    mock.patch.object(
                        runner, "check_artifact", return_value={"oracle": "pass"}
                    ),
                ):
                    if error is None:
                        runner.finish("BATCH_ID", [])
                    else:
                        with self.assertRaisesRegex(agent.AgentFailure, error):
                            runner.finish("BATCH_ID", [])
                self.assertEqual(waves.call_args_list[0].kwargs, {"may_compact": True})

    def test_hermes_runs_in_its_own_profile_of_the_developers_root(self):
        # A root of its own would be the bug the launcher avoids: Hermes would
        # install its tools there and point the developer's hermes command at
        # them. The run's profile moves into the run's folder afterwards.
        for existed in (False, True):
            with (
                self.subTest(profiles_existed=existed),
                tempfile.TemporaryDirectory() as directory,
                mock.patch.dict(os.environ, {"HOME": directory}),
            ):
                os.environ.pop("HERMES_HOME", None)
                profiles = Path(directory, ".hermes/profiles")
                if existed:
                    (profiles / "splash").mkdir(parents=True)
                runner = agent.ClientRun(
                    "hermes",
                    "hermes",
                    Path(directory, "run"),
                    "m",
                    102400,
                    10,
                    ["text"],
                )
                self.assertRegex(runner.hermes_profile, r"^splash-test-[0-9a-f]{8}$")
                self.assertEqual(runner.hermes_home, profiles / runner.hermes_profile)
                runner.hermes_home.mkdir(parents=True)
                (runner.hermes_home / "state.db").write_bytes(b"sessions")
                runner.finish_hermes()
                self.assertEqual(
                    (runner.folder / "hermes-profile/state.db").read_bytes(),
                    b"sessions",
                )
                self.assertEqual(
                    sorted(p.name for p in Path(directory, ".hermes").iterdir()),
                    ["profiles"] if existed else [],
                )
                if existed:
                    self.assertEqual([p.name for p in profiles.iterdir()], ["splash"])

    def test_hermes_phase_without_a_session_record_reports_the_exit(self):
        # Hermes creates state.db with its first session; its absence is not
        # a sqlite error.
        for code in (1, 0):
            with (
                self.subTest(exit_code=code),
                tempfile.TemporaryDirectory() as directory,
            ):
                runner = agent.ClientRun.__new__(agent.ClientRun)
                runner.name, runner.session = "hermes", None
                runner.folder = runner.workspace = Path(directory)
                runner.hermes_home = Path(directory, "profile")
                runner.timeout, runner.phases = 10, []
                process = mock.Mock(returncode=code)
                process.poll.return_value = code
                with (
                    mock.patch.object(runner, "argv", return_value=(["hermes"], {})),
                    mock.patch.object(
                        agent, "idle_status", side_effect=[idle(), idle()]
                    ),
                    mock.patch.object(agent.subprocess, "Popen", return_value=process),
                    mock.patch.object(agent, "stop_process"),
                    mock.patch.object(
                        agent, "memory_sample", return_value={"pressure": 1}
                    ),
                    self.assertRaisesRegex(
                        agent.AgentFailure,
                        f"^Hermes exited {code} without a session record in "
                        f"{runner.hermes_home / 'state.db'}; see ",
                    ),
                ):
                    runner.phase("test", "task")
                self.assertEqual(runner.phases[0]["executed_commands"], [])

    def test_continuation_does_not_overwrite_interrupted_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = agent.ClientRun.__new__(agent.ClientRun)
            runner.folder = Path(directory)
            old = runner.folder / "reference-01.log"
            old.write_text("interrupted evidence")
            with (
                mock.patch.object(
                    runner, "compaction", side_effect=[[], [{"auto": True}]]
                ),
                mock.patch.object(runner, "phase") as phase,
                mock.patch.object(
                    runner, "check_artifact", return_value={"oracle": "pass"}
                ),
            ):
                runner.finish("BATCH_ID", [])
            self.assertEqual(phase.call_args_list[0].args, ("reference-02", "2"))
            self.assertEqual(old.read_text(), "interrupted evidence")

    def test_events_ignore_non_json_and_non_object_lines(self):
        self.assertEqual(
            agent.events('startup\n[]\n{"type":"turn.completed"}\n'),
            [{"type": "turn.completed"}],
        )

    def test_status_waits_for_fresh_snapshot_not_stale_readiness(self):
        stale = {
            "ready": False,
            "metal": {"healthy": True},
            "transport": {"status_stale": True, "error": "refresh pending"},
        }
        fresh = {"ready": True}
        with (
            mock.patch.object(
                agent.launcher, "_request_json", side_effect=[stale, fresh]
            ),
            mock.patch.object(agent.time, "sleep"),
            mock.patch.object(agent.smoke_real, "validate_status") as validate,
        ):
            self.assertIs(agent.status(), fresh)
            validate.assert_called_once_with(fresh)
        with mock.patch.object(
            agent.launcher, "_request_json", return_value={"ready": False}
        ):
            with self.assertRaisesRegex(agent.AgentFailure, "not ready"):
                agent.status()

    def test_active_monitor_skips_stale_samples_without_claiming_readiness(self):
        stale = {
            "ready": False,
            "metal": {"healthy": True},
            "transport": {"status_stale": True, "status_age_ms": 25000},
        }
        with (
            mock.patch.object(agent.launcher, "_request_json", return_value=stale),
            mock.patch.object(agent.smoke_real, "validate_status") as validate,
            mock.patch.object(agent.time, "sleep") as sleep,
        ):
            for _ in range(100):
                self.assertIsNone(agent.status(wait_for_fresh=False))
            validate.assert_not_called()
            sleep.assert_not_called()
            stale["metal"]["healthy"] = False
            with self.assertRaises(agent.AgentFailure):
                agent.status(wait_for_fresh=False)

    def test_claude_system_events_with_string_messages_are_skipped(self):
        # Claude Code emits permission_denied system events whose "message"
        # is a plain string; they must not break command extraction.
        self.assertEqual(
            agent.executed_commands(
                "claude",
                [
                    {
                        "type": "system",
                        "subtype": "permission_denied",
                        "message": "This command requires approval",
                    },
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "t1",
                                    "name": "Bash",
                                    "input": {"command": agent.TEST_COMMAND},
                                }
                            ]
                        },
                    },
                    {
                        "type": "user",
                        "message": {
                            "content": [{"type": "tool_result", "tool_use_id": "t1"}]
                        },
                    },
                ],
            ),
            [agent.TEST_COMMAND],
        )

    def test_opencode_shell_commands_count_in_both_major_versions(self):
        # A user turn with one tool call, as OpenCode 1.x and 2.0.18 export
        # it: 2.x names its bash tool shell and keeps an assistant's parts in
        # content.
        def v1(state):
            return [
                {"info": {"role": "user"}, "parts": []},
                {
                    "info": {"role": "assistant"},
                    "parts": [{"type": "tool", "tool": "bash", "state": state}],
                },
            ]

        def v2(state):
            return [
                {"type": "user"},
                {
                    "type": "assistant",
                    "content": [{"type": "tool", "name": "shell", "state": state}],
                },
            ]

        ran = {"status": "completed", "input": {"command": agent.TEST_COMMAND}}
        for version, turn in ((1, v1), (2, v2)):
            for state, expected in (
                ({**ran, "metadata": {"exit": 0}}, [agent.TEST_COMMAND]),
                ({**ran, "metadata": {"exit": 1}}, []),
                ({**ran, "status": "error", "metadata": {"exit": 0}}, []),
                # A tool part may hold null for its state or its metadata.
                ({**ran, "metadata": None}, [agent.TEST_COMMAND]),
                (None, []),
            ):
                with self.subTest(version=version, state=state):
                    session = agent.opencode_session({"messages": turn(state)}, version)
                    self.assertEqual(
                        agent.executed_commands("opencode", [], turns=session["turns"]),
                        expected,
                    )

    def test_only_real_successful_commands_count_as_client_execution(self):
        self.assertEqual(
            agent.executed_commands(
                "codex",
                [
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": "I ran python3 -m unittest",
                        },
                    }
                ],
            ),
            [],
        )
        for exit_code, expected in ((1, []), (0, [agent.TEST_COMMAND])):
            with self.subTest(exit_code=exit_code):
                self.assertEqual(
                    agent.executed_commands(
                        "codex",
                        [
                            {
                                "type": "item.completed",
                                "item": {
                                    "type": "command_execution",
                                    "exit_code": exit_code,
                                    "command": agent.TEST_COMMAND,
                                },
                            }
                        ],
                    ),
                    expected,
                )
        self.assertEqual(
            agent.executed_commands(
                "claude",
                [
                    {
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "t1",
                                    "name": "Bash",
                                    "input": {"command": agent.TEST_COMMAND},
                                }
                            ]
                        }
                    },
                    {
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "t1",
                                    "is_error": True,
                                }
                            ]
                        }
                    },
                ],
            ),
            [],
        )

    def test_each_client_reuses_production_adapter_and_persists_sessions(self):
        for name in agent.CLIENTS:
            for session in (None, "test-session"):
                with (
                    self.subTest(name=name, session=session),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    runner = agent.ClientRun.__new__(agent.ClientRun)
                    runner.name, runner.path = name, f"/test/{name}"
                    runner.model, runner.context = "Actual-model", 102400
                    runner.input_modalities = ["text"]
                    runner.workspace = Path("/test/project")
                    runner.codex_home = Path(directory) / "codex-home"
                    runner.pi_home = Path(directory) / "pi-agent"
                    runner.opencode_data = Path(directory) / "opencode-data"
                    runner.hermes_profile = "splash-test-0123abcd"
                    runner.session = session
                    runner.version = 2 if name == "opencode" else None

                    def launch(*_, client_args, **__):
                        return [runner.path, *client_args], {}

                    with mock.patch.object(
                        agent.clients, "command", side_effect=launch
                    ) as adapter:
                        argv, env = runner.argv()
                    # Pi, Codex and OpenCode keep their state in the run, and
                    # Hermes in the run's profile.
                    state = {
                        "pi": {"PI_CODING_AGENT_DIR": str(runner.pi_home)},
                        "codex": {"CODEX_HOME": str(runner.codex_home)},
                        "opencode": {"XDG_DATA_HOME": str(runner.opencode_data)},
                    }.get(name, {})
                    self.assertEqual(env["PWD"], "/test/project")
                    if name == "codex":
                        self.assertTrue(runner.codex_home.is_dir())
                    adapter.assert_called_once_with(
                        name,
                        runner.path,
                        agent.BASE_URL,
                        "Actual-model",
                        102400,
                        dict(agent.os.environ, **state),
                        input_modalities=["text"],
                        client_args=argv[1:],
                        client_version=runner.version,
                        hermes_profile=runner.hermes_profile,
                    )
                    for forbidden in (
                        "--ephemeral",
                        "--no-session-persistence",
                        "--tools",
                        "--disallowedTools",
                        "--dangerously-skip-permissions",
                        "--dangerously-bypass-approvals-and-sandbox",
                        "--ignore-user-config",
                        "--pure",
                    ):
                        self.assertNotIn(forbidden, argv)
                    self.assertEqual(session in argv, session is not None)
                    if name == "claude":
                        self.assertEqual(
                            argv[argv.index("--permission-mode") + 1], "acceptEdits"
                        )
                        self.assertEqual(
                            argv[argv.index("--allowedTools") + 1],
                            f"Bash({agent.TEST_COMMAND})",
                        )
                    if name == "pi":
                        expected = ["--print", "--mode", "json"]
                        if session:
                            expected += ["--session", session]
                        self.assertEqual(argv[1:], expected)
                    if name == "hermes":
                        expected = ["chat", "--oneshot", "--query-file", "-"]
                        if session:
                            expected += ["--resume", session]
                        self.assertEqual(argv[1:], expected)

    def test_opencode_runs_as_splash_launches_it(self):
        # OpenCode 2 reaches the inline configuration only through a private
        # server, whose flag follows the subcommand; a background service
        # would outlive the run. Version 1 rejects the flag.
        for version, run in (
            (2, ["run", "--format", "json", "--standalone"]),
            (1, ["run", "--format", "json"]),
        ):
            with self.subTest(version=version):
                runner = agent.ClientRun.__new__(agent.ClientRun)
                runner.name, runner.path = "opencode", "/test/opencode"
                runner.model, runner.context = "Actual-model", 102400
                runner.input_modalities = ["text"]
                runner.workspace = Path("/test/project")
                runner.opencode_data = Path("/test/run/opencode-data")
                runner.hermes_profile = None
                runner.session, runner.version = None, version
                self.assertEqual(runner.argv()[0][1:], run)

    def test_opencode_turn_completion_comes_from_its_session_record(self):
        # The shapes OpenCode 1.x and 2.0.18 export; 2.x prints no final
        # step_finish, so its record is what says the turn ended.
        def v1(role, finish=None, *texts):
            parts = [{"type": "text", "text": text} for text in texts]
            return {"info": {"role": role, "finish": finish}, "parts": parts}

        def v2(role, finish=None, *texts):
            content = [{"type": "text", "text": text} for text in texts]
            return {"type": role, "finish": finish, "content": content}

        for version, message in ((1, v1), (2, v2)):
            with self.subTest(version=version):
                # OpenCode 2 ends a turn with an idle marker after its messages.
                idle = (
                    [{"type": "idle", "outcome": "succeeded"}] if version == 2 else []
                )
                for messages, completed in (
                    (
                        [message("user"), message("assistant", "stop", "done"), *idle],
                        True,
                    ),
                    # Text from an earlier step of the turn counts.
                    (
                        [
                            message("user"),
                            message("assistant", "tool-calls", "writing"),
                            message("assistant", "stop"),
                        ],
                        True,
                    ),
                    ([message("user"), message("assistant", "tool-calls", "x")], False),
                    ([message("user"), message("assistant", None, "partial")], False),
                    # A previous turn's text does not answer this one.
                    (
                        [
                            message("user"),
                            message("assistant", "stop", "done"),
                            message("user"),
                            message("assistant", "stop", " "),
                        ],
                        False,
                    ),
                ):
                    session = agent.opencode_session({"messages": messages}, version)
                    self.assertEqual(
                        agent.opencode_completed(session["turns"]), completed
                    )
        # A phase that added no turn did not answer its prompt.
        self.assertFalse(agent.opencode_completed([]))

    def test_opencode_automatic_compactions_in_both_major_versions(self):
        # OpenCode 1.x: a compaction part in the message that asks for one.
        def v1(auto):
            parts = [{"type": "compaction", "auto": auto}]
            return {"info": {"role": "user", "id": f"auto {auto}"}, "parts": parts}

        session = agent.opencode_session({"messages": [v1(False), v1(True)]}, 1)
        self.assertEqual(session["compactions"], [{"message_id": "auto True"}])

        # OpenCode 2.0.18: a message of its own, with its reason and status.
        def v2(reason, status):
            return {
                "type": "compaction",
                "id": f"{reason} {status}",
                "reason": reason,
                "status": status,
                "summary": "Earlier work",
                "recent": "[User]: the turns the summary replaced",
            }

        messages = [
            v2("manual", "completed"),
            v2("auto", "running"),
            v2("auto", "failed"),
            v2("auto", "completed"),
        ]
        session = agent.opencode_session({"messages": messages}, 2)
        self.assertEqual(session["compactions"], [{"message_id": "auto completed"}])

    def test_opencode_session_export_follows_its_version(self):
        for version, export in ((2, ["session", "export"]), (1, ["export"])):
            with self.subTest(version=version):
                runner = agent.ClientRun.__new__(agent.ClientRun)
                runner.name, runner.path = "opencode", "/test/opencode"
                runner.model, runner.context = "Actual-model", 102400
                runner.input_modalities = ["text"]
                runner.workspace = Path("/test/project")
                runner.folder = Path("/test/run")
                runner.opencode_data = runner.folder / "opencode-data"
                runner.hermes_profile = None
                runner.session, runner.version = "ses_1", version
                history = {"messages": []}

                def exported(argv, *, stdout, **_):
                    stdout.write(json.dumps(history))
                    return subprocess.CompletedProcess(argv, 0)

                with (
                    mock.patch.object(
                        agent.subprocess, "run", side_effect=exported
                    ) as run,
                    mock.patch.object(agent, "atomic_json"),
                ):
                    self.assertEqual(runner.opencode_history(), history)
                argv = run.call_args.args[0]
                self.assertEqual(argv[1 : 1 + len(export) + 1], [*export, "ses_1"])
                self.assertEqual("--standalone" in argv, version == 2)
                # The export reads the run's own session database.
                self.assertEqual(
                    run.call_args.kwargs["env"]["XDG_DATA_HOME"],
                    "/test/run/opencode-data",
                )
                # A failed export says why.
                failed = subprocess.CompletedProcess(argv, 1, stderr="Error: gone\n")
                with (
                    mock.patch.object(agent.subprocess, "run", return_value=failed),
                    self.assertRaisesRegex(agent.AgentFailure, "Error: gone"),
                ):
                    runner.opencode_history()

    def test_opencode_phase_is_judged_on_the_turn_it_added(self):
        def turn(*entries):
            return [{"type": "user"}, *entries, {"type": "idle"}]

        def answer(text):
            content = [{"type": "text", "text": text}]
            return {"type": "assistant", "finish": "stop", "content": content}

        compaction = {
            "type": "compaction",
            "id": "msg_compaction",
            "reason": "auto",
            "status": "completed",
        }
        earlier = turn(answer("Done"))
        for history, error in (
            ({"messages": [*earlier, *turn(compaction, answer("Again"))]}, None),
            # The prompt never reached OpenCode: the session's last turn is
            # the previous phase's.
            ({"messages": earlier}, "did not finish its user turn"),
            (
                agent.AgentFailure("exit status 1: Error: gone"),
                "session export failed: exit status 1: Error: gone",
            ),
        ):
            with (
                self.subTest(error=error),
                tempfile.TemporaryDirectory() as directory,
            ):
                runner = agent.ClientRun.__new__(agent.ClientRun)
                runner.name, runner.session, runner.version = "opencode", "s", 2
                runner.opencode = agent.opencode_session({"messages": earlier}, 2)
                runner.folder = runner.workspace = Path(directory)
                runner.timeout, runner.phases = 10, []
                process = mock.Mock(returncode=0)
                process.poll.return_value = 0
                with (
                    mock.patch.object(runner, "argv", return_value=(["client"], {})),
                    mock.patch.object(
                        runner, "opencode_history", side_effect=[history]
                    ) as export,
                    mock.patch.object(
                        agent,
                        "idle_status",
                        side_effect=[idle(), idle(submitted=1, completed=1)],
                    ),
                    mock.patch.object(agent.subprocess, "Popen", return_value=process),
                    mock.patch.object(agent, "stop_process"),
                    mock.patch.object(
                        agent, "memory_sample", return_value={"pressure": 1}
                    ),
                ):
                    if error is None:
                        runner.phase("test", "task")
                    else:
                        with self.assertRaisesRegex(agent.AgentFailure, error):
                            runner.phase("test", "task")
                # One export per phase, which the compaction check reuses.
                export.assert_called_once_with()
                self.assertEqual(
                    runner.compaction(),
                    [{"message_id": "msg_compaction"}] if error is None else [],
                )
                self.assertEqual(len(runner.phases), 1)

    def test_artifact_oracle_rejects_stub_and_wrong_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            agent.fixture(root)
            with self.assertRaisesRegex(agent.AgentFailure, "oracle failed"):
                agent.validate_artifact(root, 0)

            (root / "ledger.py").write_text(
                "def summarize(records, category=None):\n"
                "    rows = [r for r in records if r['amount_cents'] is not None "
                "and not r.get('void', False) "
                "and (category is None or r['category'] == category)]\n"
                "    return {'count': len(rows), 'total_cents': sum(r['amount_cents'] for r in rows)}\n"
            )
            (root / "test_ledger.py").write_text(
                "import unittest\nfrom ledger import summarize\n"
                "class LedgerTests(unittest.TestCase):\n"
                "    def test_empty(self):\n"
                "        self.assertEqual(summarize([]), {'count': 0, 'total_cents': 0})\n"
            )
            self.assertEqual(agent.validate_artifact(root, 2)["oracle"], "pass")
            with self.assertRaisesRegex(agent.AgentFailure, "oracle failed"):
                agent.validate_artifact(root, 0)

    def test_artifact_tests_run_with_the_harness_python(self):
        # A python3 that fails first in PATH, as the Xcode shim does on a Mac
        # whose Xcode license is not accepted.
        with tempfile.TemporaryDirectory() as directory:
            commands = Path(directory) / "commands"
            commands.mkdir()
            (commands / "python3").write_text("#!/bin/sh\necho shim >&2\nexit 69\n")
            (commands / "python3").chmod(0o755)
            root = Path(directory) / "project"
            agent.fixture(root)
            (root / "ledger.py").write_text(
                "def summarize(records):\n"
                "    rows = [r for r in records if r['amount_cents'] is not None]\n"
                "    return {'count': len(rows), "
                "'total_cents': sum(r['amount_cents'] for r in rows)}\n"
            )
            (root / "test_ledger.py").write_text(
                "import unittest\nfrom ledger import summarize\n"
                "class LedgerTests(unittest.TestCase):\n"
                "    def test_empty(self):\n"
                "        self.assertEqual(summarize([]), {'count': 0, 'total_cents': 0})\n"
            )
            path = f"{commands}{os.pathsep}{os.environ['PATH']}"
            with mock.patch.dict(os.environ, {"PATH": path}):
                self.assertEqual(agent.validate_artifact(root, 0)["oracle"], "pass")

    def test_prior_tool_execution_cannot_validate_the_current_phase(self):
        runner = agent.ClientRun.__new__(agent.ClientRun)
        runner.phases = [
            {"executed_commands": [agent.TEST_COMMAND]},
            {"executed_commands": []},
        ]
        with mock.patch.object(agent, "validate_artifact") as validate:
            with self.assertRaisesRegex(agent.AgentFailure, "execute"):
                runner.check_artifact(2)
            validate.assert_not_called()

    def test_any_python_running_unittest_executes_the_project_tests(self):
        runner = agent.ClientRun.__new__(agent.ClientRun)
        runner.workspace = Path("project")
        # As Claude Code, Hermes and Codex ran them on the M3 and M6.
        for command in (
            agent.TEST_COMMAND,
            "python -m unittest -v",
            "/bin/zsh -lc '/opt/homebrew/bin/python3.14 -m unittest -v 2>&1'",
            "cd project && .venv/bin/python -m unittest",
        ):
            with (
                self.subTest(command=command),
                mock.patch.object(agent, "validate_artifact", return_value="ok"),
            ):
                runner.phases = [{"executed_commands": [command]}]
                self.assertEqual(runner.check_artifact(2), "ok")
        for command in ("python3 -m pytest", "pytest -q", "mypython -m unittest"):
            with self.subTest(command=command):
                runner.phases = [{"executed_commands": [command]}]
                with self.assertRaisesRegex(agent.AgentFailure, "execute"):
                    runner.check_artifact(2)

    def test_missing_client_fails_before_server_start(self):
        with (
            mock.patch.object(
                agent.clients,
                "find_executable",
                side_effect=agent.clients.ClientError("missing"),
            ),
            mock.patch.object(agent.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(agent.clients.ClientError, "missing"):
                agent.main(
                    [
                        "--model",
                        "incoai/Qwen3.8-27B-Splash",
                        "--clients",
                        "codex",
                    ]
                )
            run.assert_not_called()

    def test_preflight_does_not_claim_real_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with (
                mock.patch.object(
                    agent.clients, "find_executable", return_value="/test/claude"
                ),
                mock.patch.object(
                    agent.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0, "version", ""),
                ) as run,
            ):
                self.assertEqual(
                    agent.main(
                        [
                            "--clients",
                            "claude",
                            "--model",
                            "incoai/Qwen3.8-27B-Splash",
                            "--preflight-only",
                            "--output",
                            str(report),
                        ]
                    ),
                    0,
                )
            self.assertEqual(json.loads(report.read_text())["result"], "preflight_only")
            run.assert_called_once_with(
                ["/test/claude", "--version"],
                text=True,
                capture_output=True,
                timeout=20,
            )

    def test_preflight_runs_the_installed_client_tests_of_the_clients_found(self):
        tests = [
            "dev.tests.install.test_clients.InstalledCodexTests",
            "dev.tests.install.test_clients.InstalledPiTests",
        ]
        for status in (0, 1):
            calls = []

            def run(argv, **options):
                calls.append((argv, options))
                tests_run = argv[1:3] == ["-m", "unittest"]
                return subprocess.CompletedProcess(
                    argv, status if tests_run else 0, "version", ""
                )

            with (
                self.subTest(status=status),
                tempfile.TemporaryDirectory() as directory,
                mock.patch.object(
                    agent.clients, "find_executable", side_effect=lambda n: f"/test/{n}"
                ),
                mock.patch.object(agent.subprocess, "run", side_effect=run),
                mock.patch.dict(os.environ, clear=True),
            ):
                report = Path(directory) / "report.json"
                arguments = [
                    "--clients",
                    "claude,codex,pi",
                    "--model",
                    "incoai/Qwen3.8-27B-Splash",
                    "--preflight-only",
                    "--output",
                    str(report),
                ]
                if status:
                    with self.assertRaisesRegex(agent.AgentFailure, "installed"):
                        agent.main(arguments)
                    self.assertFalse(report.exists())
                else:
                    self.assertEqual(agent.main(arguments), 0)
            argv, options = calls[-1]
            self.assertEqual(argv, [sys.executable, "-m", "unittest", *tests])
            self.assertEqual(options["cwd"], agent.ROOT)
            self.assertEqual(
                options["env"],
                {"SPLASH_CODEX_BINARY": "/test/codex", "SPLASH_PI_BINARY": "/test/pi"},
            )

    def test_preflight_requires_opencode_major_version(self):
        # Unread, the launch would be OpenCode 1's: OpenCode 2 would run
        # through its background service.
        for printed, major in (
            ("opencode v2.0.18\n", 2),
            ("1.18.32\n", 1),
            ("unknown\n", None),
            ("", None),
        ):
            with (
                self.subTest(printed=printed),
                tempfile.TemporaryDirectory() as directory,
                mock.patch.object(
                    agent.clients, "find_executable", return_value="/test/opencode"
                ),
                mock.patch.object(
                    agent.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0, printed, ""),
                ),
            ):
                report = Path(directory) / "report.json"
                arguments = [
                    "--clients",
                    "opencode",
                    "--model",
                    "incoai/Qwen3.8-27B-Splash",
                    "--preflight-only",
                    "--output",
                    str(report),
                ]
                if major is None:
                    with self.assertRaisesRegex(agent.AgentFailure, "major version"):
                        agent.main(arguments)
                    self.assertFalse(report.exists())
                    continue
                self.assertEqual(agent.main(arguments), 0)
                entries = json.loads(report.read_text())["clients"]
                self.assertEqual(entries["opencode"]["major_version"], major)

    def test_opencode_runs_with_the_major_version_preflight_read(self):
        model = "incoai/Qwen3.8-27B-Splash"
        identity = "src-" + "a" * 64
        initial = {
            "instance": {"model": model},
            "maximum_context_tokens": serve_options.parse_max_context("100K"),
            "identity": {"cache": {"build_id": identity}},
        }
        served = {"data": [{"id": model, "input_modalities": ["text"]}]}
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                agent.clients, "find_executable", return_value="/test/opencode"
            ),
            mock.patch.object(
                agent.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "2.0.18\n", ""),
            ),
            mock.patch.object(agent, "current_build_id", return_value=identity),
            mock.patch.object(
                agent.launcher, "_request_json", side_effect=[initial, served]
            ),
            mock.patch.object(agent, "idle_status", return_value=initial),
            mock.patch.object(agent, "ClientRun") as client,
        ):
            client.return_value.phases = []
            client.return_value.run.return_value = {}
            report = Path(directory) / "report.json"
            self.assertEqual(
                agent.main(
                    [
                        "--model",
                        model,
                        "--clients",
                        "opencode",
                        "--scenario",
                        "smoke",
                        "--output",
                        str(report),
                    ]
                ),
                0,
            )
            self.assertEqual(json.loads(report.read_text())["result"], "pass")
        self.assertEqual(client.call_args.kwargs["version"], 2)

    def test_client_list_rejects_unknown_and_duplicates(self):
        for value in ("codex,codex", "unknown", ""):
            with self.subTest(value=value), self.assertRaises(agent.AgentFailure):
                agent.main(
                    [
                        "--model",
                        "incoai/Qwen3.8-27B-Splash",
                        "--clients",
                        value,
                    ]
                )


if __name__ == "__main__":
    unittest.main()
