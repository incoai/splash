import contextlib
import io
import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests import agent_real as agent

MODEL_IDS = (
    "incoai/Qwen3.8-27B-Splash",
    "incoai/Qwen3.6-35B-A3B-Splash",
    "community/custom-splash",
)


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

    def test_pi_compaction_reads_saved_session_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = agent.ClientRun(
                "pi", "/bin/pi", Path(directory) / "run", "test-model", 102400, 10
            )
            runner.session = "selected-id"
            sessions = runner.folder / "pi-sessions"
            sessions.mkdir()
            entry = {"type": "compaction", "summary": "Earlier work"}
            (sessions / "time_selected-id.jsonl").write_text(json.dumps(entry) + "\n")
            (sessions / "time_other-id.jsonl").write_text(json.dumps(entry) + "\n")
            self.assertEqual(runner.compaction(), [entry])

    def test_hermes_session_lookup_uses_canonical_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            physical = root / "physical"
            physical.mkdir()
            alias = root / "alias"
            alias.symlink_to(physical, target_is_directory=True)
            runner = agent.ClientRun(
                "hermes", "hermes", alias / "run", "test-model", 102400, 10
            )
            workspace = physical / "run/project"
            runtime = root / "runtime"
            (runtime / "hermes").mkdir(parents=True)
            with (
                contextlib.closing(sqlite3.connect(runtime / "hermes/state.db")) as db,
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
            with (
                mock.patch.object(agent.launcher, "RUNTIME_DIR", runtime),
                mock.patch.object(
                    agent.clients, "command", return_value=(["hermes"], {})
                ),
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
        initial = {
            "maximum_context_tokens": context,
            "identity": {"cache": {"build_id": identity}},
        }
        for selected in MODEL_IDS:
            with self.subTest(model=selected):
                agent.validate_server_configuration(
                    initial, selected, selected, context, identity
                )
                other = next(value for value in MODEL_IDS if value != selected)
                for model, expected_context, expected_identity in (
                    (other, context, identity),
                    (selected, context + 1, identity),
                    (selected, context, "src-" + "b" * 64),
                ):
                    with self.assertRaises(agent.AgentFailure):
                        agent.validate_server_configuration(
                            initial,
                            model,
                            selected,
                            expected_context,
                            expected_identity,
                        )
                with self.assertRaisesRegex(agent.AgentFailure, "native build"):
                    agent.validate_server_configuration(
                        {"maximum_context_tokens": context},
                        selected,
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
            "maximum_context_tokens": agent.launcher._parse_max_context("100K"),
            "identity": {"cache": {"build_id": "src-" + "b" * 64}},
        }
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
                    side_effect=[initial, {"data": [{"id": model}]}],
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

    def test_real_make_gates_select_model_and_verify_build(self):
        makefile = (agent.ROOT / "dev/Makefile").read_text()
        for target in ("test-agent-real", "test-release-real", "test-http-real"):
            prerequisites = next(
                line for line in makefile.splitlines() if line.startswith(target + ":")
            )
            self.assertIn("verify-build-identity", prerequisites)
        self.assertIn('dev/tests/agent_real.py --model "$(MODEL)"', makefile)
        self.assertIn('--model "$(MODEL)" --http-smoke', makefile)
        self.assertIn('--model "$(MODEL)" $(HTTP_SMOKE_ARGS)', makefile)

    def test_model_is_required_and_only_canonical_ids_are_accepted(self):
        for arguments in ([], ["--preflight-only"], ["--model", "qwen3.8-27b"]):
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    agent.parse_args(arguments)
        for model in MODEL_IDS:
            self.assertEqual(agent.parse_args(["--model", model]).model, model)

    def test_complete_phase_handles_auxiliary_cancellation_and_retains_failures(self):
        finished = [
            {"type": "text", "sessionID": "s", "part": {"text": "Done"}},
            {"type": "step_finish", "part": {"reason": "stop"}},
        ]
        for mode in (
            "complete",
            "incomplete",
            "exit_error",
            "monitor_error",
            "interrupt",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                runner = agent.ClientRun.__new__(agent.ClientRun)
                runner.name, runner.session = "opencode", "s"
                runner.folder = runner.workspace = Path(directory)
                runner.timeout, runner.phases = 10, []
                process = mock.Mock(returncode=1 if mode == "exit_error" else 0)
                process.poll.return_value = (
                    None if mode in ("monitor_error", "interrupt") else 0
                )
                before = {
                    "requests": {
                        "submitted": 0,
                        "completed": 0,
                        "cancelled": 0,
                        "failed": 0,
                    }
                }
                after = {
                    "requests": {
                        "submitted": 2,
                        "completed": 1,
                        "cancelled": 1,
                        "failed": 0,
                    }
                }

                def launch(*args, **kwargs):
                    for event in finished[:1] if mode == "incomplete" else finished:
                        kwargs["stdout"].write(json.dumps(event) + "\n")
                    return process

                with (
                    mock.patch.object(runner, "argv", return_value=(["client"], {})),
                    mock.patch.object(
                        agent, "idle_status", side_effect=[before, after]
                    ),
                    mock.patch.object(agent.subprocess, "Popen", side_effect=launch),
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
                    runner.workspace = Path("/test/project")
                    runner.codex_home = Path(directory) / "codex-home"
                    runner.folder = Path(directory)
                    runner.session = session
                    with mock.patch.object(
                        agent.clients, "command", return_value=([runner.path], {})
                    ) as adapter:
                        argv, env = runner.argv()
                    self.assertEqual(env["PWD"], "/test/project")
                    if name == "codex":
                        self.assertEqual(env["CODEX_HOME"], str(runner.codex_home))
                        self.assertTrue(runner.codex_home.is_dir())
                    adapter.assert_called_once_with(
                        name,
                        runner.path,
                        agent.launcher.BASE_URL,
                        "Actual-model",
                        102400,
                        agent.launcher.RUNTIME_DIR,
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
                        self.assertIn("--print", argv)
                        self.assertEqual(argv[argv.index("--mode") + 1], "json")
                        self.assertEqual(
                            argv[argv.index("--session-dir") + 1],
                            str((Path(directory) / "pi-sessions").resolve()),
                        )
                        if session:
                            self.assertEqual(argv[argv.index("--session") + 1], session)

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
