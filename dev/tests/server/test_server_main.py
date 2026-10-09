import io
import math
import signal
import unittest
from types import SimpleNamespace
from unittest import mock

from dev.tests.server_fixtures import main_args
from server import diagnostics
from server import protocol as native_wire
from server import server as api
from server.origins import ANY_ORIGIN


class ServerMainTests(unittest.TestCase):
    def test_sigterm_uses_the_normal_main_cleanup_path(self):
        args = main_args()
        runtime = mock.Mock()
        runtime.readiness = native_wire.ReadyEvent(
            max_concurrent_requests=4, max_context_tokens=262144, vision=False
        )
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

        backend.close.side_effect = lambda: closing_handlers.append(dict(handlers))

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
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(api, "ChatTemplates"),
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
        runtime_type.assert_called_once_with(
            [
                "splash",
                "serve-native",
                "model",
                "auto",
                "auto",
            ],
            startup_timeout=api.NATIVE_START_TIMEOUT,
            pending_limit=1,
        )
        backend_type.assert_called_once_with(
            runtime, tokenizer, request_logger=diagnostics.print_request
        )
        self.assertEqual(app_type.call_args.args[3], 262144)
        # No --request-timeout, no deadline.
        self.assertEqual(app_type.call_args.args[4], math.inf)
        self.assertEqual(app_type.call_args.args[5], 4)
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
                runtime.readiness = native_wire.ReadyEvent(4, 131072, vision)
                with (
                    mock.patch.object(api, "parse_args", return_value=args),
                    mock.patch.object(api, "load_thinking_key", return_value=None),
                    mock.patch.object(
                        api.AutoTokenizer, "from_pretrained", return_value=object()
                    ),
                    mock.patch.object(api, "validate_tokenizer"),
                    mock.patch.object(api, "ChatTemplates") as templates_type,
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
                self.assertIs(app_type.call_args.kwargs["vision"], vision)
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
                        + ("" if vision else " · language only")
                        + " · http://127.0.0.1:8000"
                    ),
                    status.call_args_list,
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
                runtime.readiness = native_wire.ReadyEvent(
                    max_concurrent_requests=4, max_context_tokens=131072, vision=False
                )
                with (
                    mock.patch.object(
                        api,
                        "parse_args",
                        return_value=main_args(allowed_origin=origins, api_key=key),
                    ),
                    mock.patch.object(api, "load_thinking_key", return_value=None),
                    mock.patch.object(
                        api.AutoTokenizer, "from_pretrained", return_value=object()
                    ),
                    mock.patch.object(api, "validate_tokenizer"),
                    mock.patch.object(api, "ChatTemplates"),
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
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer, "from_pretrained", return_value=object()
            ),
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(api, "ChatTemplates"),
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
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer,
                "from_pretrained",
                return_value=SimpleNamespace(chat_template=None),
            ),
            mock.patch.object(api, "validate_tokenizer"),
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


if __name__ == "__main__":
    unittest.main()
