import contextlib
import dataclasses
import io
import json
import math
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dev.tests.installer_fixtures import DENSE, MODEL, family_config
from dev.tests.server_fixtures import (
    NO_THINKING_END,
    engine_ready,
    main_args,
    unprepared_hold,
)
from install import families
from server import diagnostics
from server import protocol as native_wire
from server import server as api
from server.origins import ANY_ORIGIN

# The goldens the app reads the server's records by.
GOLDEN = Path(__file__).parent


class ServerMainTests(unittest.TestCase):
    def test_sigterm_uses_the_normal_main_cleanup_path(self):
        args = main_args()
        runtime = mock.Mock()
        runtime.capabilities = engine_ready(262144)
        backend = mock.Mock()
        server = mock.Mock(server_port=8000)
        tokenizer = object()
        handlers = {}
        order = []
        closing_handlers = []

        def install(signum, handler):
            if handler is api._interrupt:
                handlers[signum] = handler
                return signal.SIG_DFL
            handlers[signum] = handler
            self.assertIn(signum, (signal.SIGTERM, signal.SIGINT))
            if handler is not signal.SIG_IGN:
                self.assertEqual(signum, signal.SIGINT)

        def close_backend():
            closing_handlers.append(dict(handlers))
            # The server stops accepting before the engine's stop, which can
            # take seconds.
            self.assertTrue(server.socket.close.called)

        backend.close.side_effect = close_backend

        def serve():
            self.assertEqual(set(handlers), {signal.SIGTERM, signal.SIGINT})
            handlers[signal.SIGTERM](signal.SIGTERM, None)

        def bind(*_, **__):
            return server

        server.server_bind.side_effect = lambda: order.append("bind")

        runtime.wait_ready.side_effect = lambda: order.append("runtime") or True
        server.serve_forever.side_effect = serve
        with (
            mock.patch.object(api, "parse_args", return_value=args),
            unprepared_hold(),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(api, "check_capabilities"),
            mock.patch.object(api, "ChatTemplates", **NO_THINKING_END),
            mock.patch.object(
                api.engine_runtime, "MultiplexedRuntime", return_value=runtime
            ) as runtime_type,
            mock.patch.object(
                api, "NativeBackend", return_value=backend
            ) as backend_type,
            mock.patch.object(api, "ConstraintFactory", return_value=object()),
            mock.patch.object(api, "Frontend", return_value=mock.Mock()) as app_type,
            mock.patch.object(api, "FrontendServer", side_effect=bind),
            mock.patch.object(api.signal, "signal", side_effect=install),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            api.main()
        self.assertEqual(
            handlers, {signal.SIGTERM: signal.SIG_IGN, signal.SIGINT: signal.SIG_IGN}
        )
        # While the engine exits gracefully, a second Ctrl+C stops it now.
        (closing,) = closing_handlers
        self.assertIs(closing[signal.SIGTERM], signal.SIG_IGN)
        runtime.kill.assert_not_called()
        closing[signal.SIGINT](signal.SIGINT, None)
        runtime.kill.assert_called_once_with()
        command = ["splash", "serve-native", "model", "auto", "auto"]
        runtime_type.assert_called_once_with(
            command, relaunch_command=mock.ANY, pending_limit=1
        )
        # Every relaunch asks for the context the first engine served.
        relaunch_command = runtime_type.call_args.kwargs["relaunch_command"]
        self.assertEqual(
            relaunch_command(engine_ready(61433)),
            [*command, "--relaunch-context", "61433"],
        )
        backend_type.assert_called_once_with(
            runtime,
            tokenizer,
            request_logger=diagnostics.print_request,
            thinking_end=None,
        )
        # No --request-timeout, no deadline.
        self.assertEqual(app_type.call_args.args[3], math.inf)
        runtime.wait_ready.assert_called_once_with()
        server.serve_forever.assert_called_once()
        server.server_close.assert_called_once()
        backend.close.assert_called_once()
        server.server_bind.assert_called_once_with()
        server.server_activate.assert_called_once_with()
        self.assertEqual(order, ["bind", "runtime"])

    def test_main_takes_vision_from_the_ready_event(self):
        args = main_args()
        for vision in (True, False):
            with self.subTest(vision=vision):
                runtime = mock.Mock()
                runtime.capabilities = engine_ready(131072, vision=vision)
                with (
                    mock.patch.object(api, "parse_args", return_value=args),
                    unprepared_hold(),
                    mock.patch.object(api, "load_thinking_key", return_value=None),
                    mock.patch.object(
                        api.AutoTokenizer, "from_pretrained", return_value=object()
                    ),
                    mock.patch.object(api, "validate_tokenizer"),
                    mock.patch.object(api, "check_capabilities"),
                    mock.patch.object(
                        api, "ChatTemplates", **NO_THINKING_END
                    ) as templates_type,
                    mock.patch.object(
                        api.engine_runtime,
                        "MultiplexedRuntime",
                        return_value=runtime,
                    ),
                    mock.patch.object(api, "NativeBackend"),
                    mock.patch.object(api, "ConstraintFactory"),
                    mock.patch.object(api, "Frontend") as app_type,
                    mock.patch.object(
                        api,
                        "FrontendServer",
                        return_value=mock.Mock(server_port=8000),
                    ),
                    mock.patch.object(api.signal, "signal"),
                    mock.patch.object(api, "print_status") as status,
                ):
                    api.main()
                self.assertIs(
                    app_type.call_args.kwargs["chat_templates"],
                    templates_type.return_value,
                )
                describe = templates_type.return_value.describe
                describe.assert_called_once_with()
                self.assertIn(
                    mock.call(f"Chat template · {describe.return_value}"),
                    status.call_args_list,
                )
                self.assertIn(
                    mock.call(
                        "Ready · test-model · context 128K"
                        + ("" if vision else " · text only")
                        + " · http://127.0.0.1:8000"
                    ),
                    status.call_args_list,
                )

    def main_starting(self, args, runtime, tokenizer=None):
        """main() over a runtime stand-in, its prints captured: its patches."""
        return (
            mock.patch.object(api, "parse_args", return_value=args),
            unprepared_hold(),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer, "from_pretrained", return_value=tokenizer or object()
            ),
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(api, "ChatTemplates", **NO_THINKING_END),
            mock.patch.object(
                api.engine_runtime, "MultiplexedRuntime", return_value=runtime
            ),
            mock.patch.object(api, "NativeBackend"),
            mock.patch.object(api, "ConstraintFactory"),
            mock.patch.object(api, "Frontend"),
            mock.patch.object(
                api, "FrontendServer", return_value=mock.Mock(server_port=8000)
            ),
            mock.patch.object(api.signal, "signal"),
            mock.patch.object(api.os, "getpid", return_value=4242),
        )

    def test_main_serves_the_context_the_engine_announces(self):
        """The engine is authoritative for its context: it refuses a
        --max-context it cannot serve, and the server serves what it
        announces, past the native window too."""
        for max_context, window in ((1010000, 1010000), (None, 262144)):
            with self.subTest(max_context=max_context):
                runtime = mock.Mock()
                runtime.capabilities = engine_ready(window)
                with contextlib.ExitStack() as stack:
                    for patch in self.main_starting(
                        main_args(max_context=max_context), runtime
                    ):
                        stack.enter_context(patch)
                    stack.enter_context(mock.patch.object(api, "check_capabilities"))
                    status = stack.enter_context(mock.patch.object(api, "print_status"))
                    api.main()
                context = "1,010,000" if window == 1010000 else "256K"
                status.assert_any_call(
                    f"Ready · test-model · context {context} · text only"
                    " · http://127.0.0.1:8000"
                )

    def test_main_checks_the_tokenizer_against_the_engines_ready(self):
        """Once the engine is Ready, a tokenizer whose stop texts are not its
        stop tokens fails the start as the model's, naming both."""
        record = Path(self.enterContext(tempfile.TemporaryDirectory())) / "s.json"
        runtime = mock.Mock()
        runtime.capabilities = engine_ready(131072)
        tokenizer = mock.Mock(eos_token_id=248046)
        tokenizer.get_vocab.return_value = {"<|im_end|>": 248046, "x": 1}
        tokenizer.encode.side_effect = lambda text, **_: {
            "<|endoftext|>": [248045],
            "<|im_end|>": [248046],
        }[text]
        with contextlib.ExitStack() as stack:
            for patch in self.main_starting(
                main_args(serve_record=record), runtime, tokenizer
            ):
                stack.enter_context(patch)
            status = stack.enter_context(mock.patch.object(api, "print_status"))
            with self.assertRaises(SystemExit):
                api.main()
        message = (
            "the tokenizer encodes the stop texts ('<|endoftext|>', "
            "'<|im_end|>') as tokens (248045, 248046), but the native model "
            "stops at (248044, 248046)"
        )
        status.assert_any_call(f"Error · {message}", error=True)
        self.assertEqual(
            json.loads(record.read_text())["failure"],
            {"kind": "model", "message": message},
        )

    def test_every_failed_start_records_its_kind(self):
        """A tokenizer the loader cannot read fails the start as the model's;
        an error the server does not expect, as internal, its traceback kept.
        Either way the record names the failure, so the app reports it
        instead of a crash."""
        record = Path(self.enterContext(tempfile.TemporaryDirectory())) / "s.json"
        for fault, ends, written in (
            (
                mock.patch.object(
                    api.AutoTokenizer,
                    "from_pretrained",
                    side_effect=KeyError("added_tokens"),
                ),
                SystemExit,
                {
                    "kind": "model",
                    "message": "cannot load the tokenizer of test-model: "
                    "KeyError: 'added_tokens'",
                },
            ),
            (
                mock.patch.object(
                    api, "check_capabilities", side_effect=AttributeError("vision")
                ),
                AttributeError,
                {"kind": "internal", "message": "AttributeError: vision"},
            ),
        ):
            with self.subTest(kind=written["kind"]):
                runtime = mock.Mock()
                with contextlib.ExitStack() as stack:
                    for patch in self.main_starting(
                        main_args(serve_record=record), runtime
                    ):
                        stack.enter_context(patch)
                    stack.enter_context(fault)
                    stack.enter_context(mock.patch.object(api, "print_status"))
                    with self.assertRaises(ends):
                        api.main()
                self.assertEqual(
                    json.loads(record.read_text()),
                    {"pid": 4242, "phase": "failed", "failure": written},
                )

    def test_a_start_writes_its_serve_record(self):
        """The record says the server is loading while the engine starts,
        warming once the engine says it warms up, then, for a start the
        engine refuses, the engine's class and numbers, as the app reads them
        (serve_record_golden.json); for one the server refuses, its own
        class."""
        record = Path(self.enterContext(tempfile.TemporaryDirectory())) / "s.json"
        golden = json.loads((GOLDEN / "serve_record_golden.json").read_text())
        failure = golden["failure"]
        event = native_wire.StartFailedEvent(
            native_wire.StartFailureKind.MEMORY_BUDGET,
            native_wire.ShortfallBound.MAX_MEMORY,
            failure["required_bytes"],
            failure["budget_bytes"],
            failure["plan_context_tokens"],
            failure["message"].encode(),
        )
        for error, written in (
            (api.engine_runtime.EngineStartFailed(event), golden),
            (
                api.engine_runtime.EngineUnhealthy("native protocol reached EOF"),
                {
                    "pid": 4242,
                    "phase": "failed",
                    "failure": {
                        "kind": "internal",
                        "message": "native protocol reached EOF",
                    },
                },
            ),
            (
                api.ModelError("the model has no chat template"),
                {
                    "pid": 4242,
                    "phase": "failed",
                    "failure": {
                        "kind": "model",
                        "message": "the model has no chat template",
                    },
                },
            ),
        ):
            with self.subTest(error=type(error).__name__):
                runtime = mock.Mock()
                phases = []

                def refuse(runtime=runtime, phases=phases, error=error):
                    phases.append(json.loads(record.read_text()))
                    runtime.on_warming()
                    phases.append(json.loads(record.read_text()))
                    raise error

                runtime.wait_ready.side_effect = refuse
                with contextlib.ExitStack() as stack:
                    for patch in self.main_starting(
                        main_args(serve_record=record), runtime
                    ):
                        stack.enter_context(patch)
                    stack.enter_context(mock.patch.object(api, "print_status"))
                    with self.assertRaises(SystemExit):
                        api.main()
                self.assertEqual(
                    phases,
                    [
                        {"pid": 4242, "phase": "loading"},
                        {"pid": 4242, "phase": "warming"},
                    ],
                )
                self.assertEqual(json.loads(record.read_text()), written)
                self.assertEqual(list(record.parent.iterdir()), [record])

    def test_the_engines_refusals_before_it_plans_reach_the_serve_record(self):
        """The engine (build/splash, which make test-python builds) refuses a
        MAX_CONTEXT past the model's, a root without the model's directories
        and a model it cannot serve before it plans a start: the serve record
        keeps each refusal's class and the words the engine reported."""
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        record = directory / "s.json"

        def model_root(name, **text_config):
            root = directory / name
            (root / "target").mkdir(parents=True)
            (root / "draft").mkdir()
            (root / "model.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "model": MODEL,
                        "target_format": "mlx-affine",
                        "vision_format": "none",
                    }
                )
            )
            config = family_config(DENSE)
            config["text_config"] |= text_config
            (root / "config.json").write_text(json.dumps(config))
            (root / "draft/config.json").write_text(
                json.dumps(family_config(DENSE, "draft/config.json"))
            )
            return root

        bare = directory / "bare"
        bare.mkdir()
        for root, max_context, kind, names in (
            (model_root("served"), 1_010_001, "configuration", "1010000"),
            (bare, None, "model", "target/"),
            (model_root("unserved", model_type="llama"), None, "model", "llama"),
        ):
            with self.subTest(root=root.name):
                args = main_args(
                    model_root=str(root),
                    max_context=max_context,
                    binary=str(api.ROOT / "build/splash"),
                    serve_record=record,
                )
                stderr = directory / f"{root.name}.stderr"

                # The engine under a HOME of its own, its report kept apart.
                def spawn(command=api._native_command(args, root), stderr=stderr):
                    with stderr.open("wb") as report:
                        return subprocess.Popen(
                            command,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=report,
                            bufsize=0,
                            env=os.environ | {"HOME": str(directory)},
                        )

                runtime = api.engine_runtime.MultiplexedRuntime(process_factory=spawn)
                self.addCleanup(runtime.close)
                with contextlib.ExitStack() as stack:
                    for patch in self.main_starting(args, runtime):
                        stack.enter_context(patch)
                    stack.enter_context(mock.patch.object(api, "print_status"))
                    with self.assertRaises(SystemExit):
                        api.main()
                failure = json.loads(record.read_text())["failure"]
                self.assertEqual(failure["kind"], kind)
                self.assertIn(names, failure["message"])
                self.assertIn(
                    f"error: {failure['message']}", stderr.read_text().splitlines()
                )

    def test_a_port_the_server_cannot_bind_is_recorded_as_the_port(self):
        record = Path(self.enterContext(tempfile.TemporaryDirectory())) / "s.json"
        with contextlib.ExitStack() as stack:
            for patch in self.main_starting(
                main_args(serve_record=record), mock.Mock()
            ):
                stack.enter_context(patch)
            stack.enter_context(mock.patch.object(api, "print_status"))
            api.FrontendServer.return_value.server_bind.side_effect = OSError(
                48, "Address already in use"
            )
            with self.assertRaises(SystemExit):
                api.main()
        self.assertEqual(
            json.loads(record.read_text())["failure"],
            {
                "kind": "port",
                "message": "unable to start HTTP server: [Errno 48] Address already in use",
            },
        )

    def test_a_server_says_ready_in_its_record_once_it_answers(self):
        """The record says ready once the server listens, before it serves."""
        record = Path(self.enterContext(tempfile.TemporaryDirectory())) / "s.json"
        runtime = mock.Mock()
        runtime.capabilities = engine_ready(131072)
        phases = []
        with contextlib.ExitStack() as stack:
            for patch in self.main_starting(main_args(serve_record=record), runtime):
                stack.enter_context(patch)
            stack.enter_context(mock.patch.object(api, "check_capabilities"))
            stack.enter_context(mock.patch.object(api, "print_status"))
            server = api.FrontendServer.return_value
            server.serve_forever.side_effect = lambda: phases.append(
                (server.server_activate.called, json.loads(record.read_text()))
            )
            api.main()
        self.assertEqual(phases, [(True, {"pid": 4242, "phase": "ready"})])

    def test_a_failed_start_whose_record_cannot_be_written_ends_as_it_would(self):
        """With the record's directory gone, a start the server refuses still
        exits with status 1 and an error it does not expect still raises
        itself: the record cannot replace the failure."""
        record = Path(self.enterContext(tempfile.TemporaryDirectory())) / "gone/s.json"
        for fault, ends in (
            (
                mock.patch.object(
                    api.AutoTokenizer,
                    "from_pretrained",
                    side_effect=KeyError("added_tokens"),
                ),
                SystemExit,
            ),
            (
                mock.patch.object(
                    api, "check_capabilities", side_effect=AttributeError("vision")
                ),
                AttributeError,
            ),
        ):
            with self.subTest(ends=ends.__name__):
                with contextlib.ExitStack() as stack:
                    for patch in self.main_starting(
                        main_args(serve_record=record), mock.Mock()
                    ):
                        stack.enter_context(patch)
                    stack.enter_context(fault)
                    stack.enter_context(mock.patch.object(api, "print_status"))
                    log = stack.enter_context(mock.patch("sys.stderr", io.StringIO()))
                    with self.assertRaises(ends) as ended:
                        api.main()
                if ends is SystemExit:
                    self.assertEqual(ended.exception.code, 1)
                self.assertIn(
                    f"Warning · cannot write the serve record {record}: "
                    "No such file or directory",
                    log.getvalue(),
                )

    def test_a_server_whose_ready_record_cannot_be_written_ends_before_serving(self):
        """The app takes a server for ready from its record alone: one that
        cannot say ready in it, its directory gone once it listens, has not
        started and exits with status 1 before it serves, its one error
        naming the record and the system's error, so that the app sees it
        stop instead of waiting on it."""
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (directory / "run").mkdir()
        record = directory / "run/s.json"
        runtime = mock.Mock()
        runtime.capabilities = engine_ready(131072)
        with contextlib.ExitStack() as stack:
            for patch in self.main_starting(main_args(serve_record=record), runtime):
                stack.enter_context(patch)
            stack.enter_context(mock.patch.object(api, "check_capabilities"))
            output = stack.enter_context(mock.patch("sys.stdout", io.StringIO()))
            log = stack.enter_context(mock.patch("sys.stderr", io.StringIO()))
            server = api.FrontendServer.return_value
            server.server_activate.side_effect = lambda: (directory / "run").rename(
                directory / "moved"
            )
            with self.assertRaises(SystemExit) as ended:
                api.main()
        self.assertEqual(ended.exception.code, 1)
        server.serve_forever.assert_not_called()
        (line,) = log.getvalue().splitlines()
        self.assertTrue(
            line.endswith(
                f" Error · cannot write the serve record {record}: "
                "No such file or directory"
            ),
            line,
        )
        self.assertNotIn("Ready · ", output.getvalue())
        self.assertEqual(
            json.loads((directory / "moved/s.json").read_text()),
            {"pid": 4242, "phase": "loading"},
        )

    def test_main_wires_allowed_origins_and_warns_without_a_key(self):
        warning = mock.call(
            "Warning · --allowed-origin '*' without --api-key lets every web "
            "page open in a browser that reaches this server use it",
            error=True,
        )
        for origins, key, warned in (
            ([("tauri", "localhost", None)], None, False),
            ([ANY_ORIGIN], None, True),
            ([ANY_ORIGIN], "key", False),
        ):
            with self.subTest(origins=origins, key=key):
                runtime = mock.Mock()
                runtime.capabilities = engine_ready(131072)
                with (
                    mock.patch.object(
                        api,
                        "parse_args",
                        return_value=main_args(allowed_origin=origins, api_key=key),
                    ),
                    unprepared_hold(),
                    mock.patch.object(api, "load_thinking_key", return_value=None),
                    mock.patch.object(
                        api.AutoTokenizer, "from_pretrained", return_value=object()
                    ),
                    mock.patch.object(api, "validate_tokenizer"),
                    mock.patch.object(api, "check_capabilities"),
                    mock.patch.object(api, "ChatTemplates", **NO_THINKING_END),
                    mock.patch.object(
                        api.engine_runtime,
                        "MultiplexedRuntime",
                        return_value=runtime,
                    ),
                    mock.patch.object(api, "NativeBackend"),
                    mock.patch.object(api, "ConstraintFactory"),
                    mock.patch.object(api, "Frontend"),
                    mock.patch.object(
                        api,
                        "FrontendServer",
                        return_value=mock.Mock(server_port=8000),
                    ) as server_type,
                    mock.patch.object(api.signal, "signal"),
                    mock.patch.object(api, "print_status") as status,
                ):
                    api.main()
                self.assertEqual(
                    server_type.call_args.kwargs["allowed_origins"], origins
                )
                self.assertIs(warning in status.call_args_list, warned)

    def test_main_cleans_up_when_native_startup_fails_after_reserved_bind(self):
        args = main_args(max_context=128, max_memory=32 * 1024**3)
        runtime = mock.Mock()
        runtime.wait_ready.side_effect = api.engine_runtime.EngineUnhealthy("late")
        backend = mock.Mock()
        server = mock.Mock()
        with (
            mock.patch.object(api, "parse_args", return_value=args),
            unprepared_hold(),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer, "from_pretrained", return_value=object()
            ),
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(api, "check_capabilities"),
            mock.patch.object(api, "ChatTemplates", **NO_THINKING_END),
            mock.patch.object(
                api.engine_runtime, "MultiplexedRuntime", return_value=runtime
            ),
            mock.patch.object(api, "NativeBackend", return_value=backend),
            mock.patch.object(api, "FrontendServer", return_value=server),
            mock.patch.object(api.signal, "signal"),
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            api.main()
        server.server_bind.assert_called_once_with()
        server.server_activate.assert_not_called()
        server.server_close.assert_called_once_with()
        backend.close.assert_called_once()
        self.assertIn("Error · late", stderr.getvalue())

    def test_main_rejects_an_unservable_chat_template_before_starting_native(self):
        server = mock.Mock()
        with (
            mock.patch.object(api, "parse_args", return_value=main_args()),
            unprepared_hold(),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer,
                "from_pretrained",
                return_value=SimpleNamespace(chat_template=None),
            ),
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(api, "check_capabilities"),
            mock.patch.object(api.engine_runtime, "MultiplexedRuntime") as runtime,
            mock.patch.object(api, "FrontendServer", return_value=server),
            mock.patch.object(api.signal, "signal"),
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            api.main()
        runtime.assert_not_called()
        server.server_activate.assert_not_called()
        server.server_close.assert_called_once_with()
        self.assertIn(
            "Error · the tokenizer defines no chat template", stderr.getvalue()
        )

    def test_main_rejects_port_conflict_before_loading_or_starting_native(self):
        args = main_args(port=8000)
        server = mock.Mock()
        server.server_bind.side_effect = OSError(48, "Address already in use")
        with (
            mock.patch.object(api, "parse_args", return_value=args),
            mock.patch.object(api, "FrontendServer", return_value=server),
            mock.patch.object(api.AutoTokenizer, "from_pretrained") as tokenizer,
            mock.patch.object(api.engine_runtime, "MultiplexedRuntime") as runtime,
            mock.patch.object(api.signal, "signal"),
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            api.main()
        tokenizer.assert_not_called()
        runtime.assert_not_called()
        server.server_activate.assert_not_called()
        server.server_close.assert_called_once_with()
        self.assertIn("Address already in use", stderr.getvalue())

    def test_main_holds_and_serves_the_assembly_its_model_directory_names(self):
        models = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        served = models / ".resolved/assembly"
        served.mkdir(parents=True)
        (served / "model.json").write_text("{}")
        link = models / "owner/model"
        link.parent.mkdir()
        link.symlink_to(served, target_is_directory=True)
        runtime = mock.Mock()
        runtime.capabilities = engine_ready(131072)
        held = []

        def start(command, **options):
            # Installations remove no assembly a server holds.
            held.append(api.assembly.is_held(served))
            return runtime

        with (
            mock.patch.object(
                api, "parse_args", return_value=main_args(model_root=str(link))
            ),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer, "from_pretrained", return_value=object()
            ) as tokenizer,
            # The record names the family the model was installed as, whose
            # sampling requests get unless they name their own.
            mock.patch.object(
                api.assembly,
                "family",
                return_value=dataclasses.replace(
                    families.named("Qwen3.8-27B"),
                    sampling=families.SamplingDefaults(0.7, 0.8, 40),
                ),
            ),
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(api, "check_capabilities"),
            mock.patch.object(api, "ChatTemplates", **NO_THINKING_END),
            mock.patch.object(
                api.engine_runtime, "MultiplexedRuntime", side_effect=start
            ) as runtime_type,
            mock.patch.object(api, "NativeBackend"),
            mock.patch.object(api, "ConstraintFactory"),
            mock.patch.object(api, "Frontend") as frontend,
            mock.patch.object(
                api, "FrontendServer", return_value=mock.Mock(server_port=8000)
            ),
            mock.patch.object(api.signal, "signal"),
            mock.patch.object(api, "print_status"),
        ):
            api.main()
        self.assertEqual(held, [True])
        # The tokenizer and every engine, a relaunched one too, are the
        # assembly's the link named, wherever the link points later.
        tokenizer.assert_called_once_with(
            served / "tokenizer", local_files_only=True, trust_remote_code=False
        )
        command = runtime_type.call_args.args[0]
        relaunch = runtime_type.call_args.kwargs["relaunch_command"]
        self.assertEqual(command[2], str(served))
        self.assertEqual(relaunch(runtime.capabilities)[2], str(served))
        self.assertEqual(
            frontend.call_args.kwargs["sampling_defaults"],
            {"temperature": 0.7, "top_p": 0.8, "top_k": 40},
        )

    def test_main_refuses_a_directory_that_names_no_prepared_model(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        models = root / "models"
        (models / ".resolved").mkdir(parents=True)
        (models / ".resolved/file").touch()
        (models / "owner").mkdir()
        (models / "owner/model").symlink_to(models / ".resolved/collected")
        gone = "names no model Splash prepared: prepare the model again"
        for directory, refusal in (
            # A directory Splash did not prepare.
            (root, "is not a model directory Splash prepared"),
            # One a later preparation collected, and a selection link to it.
            (models / ".resolved/collected", gone),
            (models / "owner/model", gone),
            # A file in its place, and one in a models directory since removed.
            (models / ".resolved/file", gone),
            (root / "removed/.resolved/collected", gone),
        ):
            with self.subTest(directory=str(directory.relative_to(root))):
                self.assertIn(f"Error · {directory} {refusal}", self.refusal(directory))

    def test_main_refuses_a_model_directory_it_cannot_hold(self):
        # Holding takes the installation lock of the models directory, which
        # the server must be able to write. A lock it cannot open is no
        # failure to start the HTTP server, whose socket is already bound.
        models = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        served = models / ".resolved/assembly"
        served.mkdir(parents=True)
        (served / "model.json").write_text("{}")
        lock = models / ".install.lock"
        refusal = (
            f"Error · cannot hold {served} in {models}: "
            f"[Errno 13] Permission denied: '{lock}'"
        )
        # A read-only lock.
        lock.touch(mode=0o444)
        self.assertIn(refusal, self.refusal(served))
        # None, in a read-only models directory.
        lock.unlink()
        models.chmod(0o555)
        self.addCleanup(models.chmod, 0o755)
        self.assertIn(refusal, self.refusal(served))

    def refusal(self, directory):
        """What main reports on stderr when it refuses to serve directory,
        before it loads a tokenizer or starts an engine."""
        server = mock.Mock()
        with (
            mock.patch.object(
                api, "parse_args", return_value=main_args(model_root=str(directory))
            ),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(api, "FrontendServer", return_value=server),
            mock.patch.object(api.AutoTokenizer, "from_pretrained") as tokenizer,
            mock.patch.object(api.engine_runtime, "MultiplexedRuntime") as runtime,
            mock.patch.object(api.signal, "signal"),
            mock.patch("sys.stdout", new_callable=io.StringIO),
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            api.main()
        tokenizer.assert_not_called()
        runtime.assert_not_called()
        server.server_close.assert_called_once_with()
        return stderr.getvalue()


if __name__ == "__main__":
    unittest.main()
