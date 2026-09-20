import io
import json
import os
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cryptography.fernet import Fernet

from dev.tests.test_server import FakeRuntime, Harness, Plan
from server import server as api
from server.thinking import ThinkingCodec, ThinkingKeyError, load_thinking_key


class ThinkingKeyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.home = Path(self.directory.name)
        self.path = self.home / "state/thinking.key"

    def test_default_path_is_shared_across_instances(self):
        with mock.patch("server.thinking.Path.home", return_value=self.home):
            key = load_thinking_key()
            codec = ThinkingCodec(key)
            signature = codec.encode("private reasoning")
            replacement = ThinkingCodec(load_thinking_key())
        self.assertEqual(replacement.decode(signature), "private reasoning")
        path = self.home / "Library/Application Support/Splash/thinking.key"
        self.assertEqual(path.read_bytes(), key)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(path.stat().st_uid, os.getuid())

    def test_concurrent_first_start_publishes_one_complete_key(self):
        barrier = threading.Barrier(12)

        def start(_):
            barrier.wait(timeout=10)
            return load_thinking_key(self.path)

        with ThreadPoolExecutor(max_workers=12) as pool:
            keys = list(pool.map(start, range(12)))
        self.assertEqual(len(set(keys)), 1)
        self.assertEqual(self.path.read_bytes(), keys[0])
        self.assertEqual(list(self.path.parent.glob(".thinking-*")), [])
        self.assertEqual(
            ThinkingCodec(keys[-1]).decode(ThinkingCodec(keys[0]).encode("one key")),
            "one key",
        )

    def test_bad_existing_key_is_never_replaced(self):
        self.path.parent.mkdir()
        for contents in (b"", b"invalid", b"!" * 44, b"A" * 45):
            with self.subTest(length=len(contents)):
                self.path.write_bytes(contents)
                self.path.chmod(0o600)
                with self.assertRaisesRegex(ThinkingKeyError, "Invalid thinking key"):
                    load_thinking_key(self.path)
                self.assertEqual(self.path.read_bytes(), contents)

    def test_unsafe_permissions_and_symlinks_are_rejected(self):
        key = load_thinking_key(self.path)
        self.path.chmod(0o644)
        with self.assertRaisesRegex(ThinkingKeyError, "0400 or 0600"):
            load_thinking_key(self.path)
        self.assertEqual(self.path.read_bytes(), key)
        self.path.chmod(0o600)
        alias = self.path.with_name("alias.key")
        alias.symlink_to(self.path)
        with self.assertRaisesRegex(ThinkingKeyError, "Cannot use thinking key"):
            load_thinking_key(alias)
        self.assertTrue(alias.is_symlink())
        fifo = self.path.with_name("fifo.key")
        os.mkfifo(fifo, 0o600)
        with self.assertRaisesRegex(ThinkingKeyError, "regular file"):
            load_thinking_key(fifo)

    def test_private_read_only_key_is_reused_without_chmod(self):
        key = load_thinking_key(self.path)
        self.path.chmod(0o400)
        self.assertEqual(load_thinking_key(self.path), key)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o400)

    def test_failed_publication_does_not_leave_a_key_or_temporary(self):
        with (
            mock.patch(
                "server.thinking.os.link", side_effect=PermissionError(13, "denied")
            ),
            self.assertRaisesRegex(ThinkingKeyError, "Cannot use thinking key"),
        ):
            load_thinking_key(self.path)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_explicit_codec_key_does_not_get_replaced(self):
        with self.assertRaises(ValueError):
            ThinkingCodec(b"")
        key = Fernet.generate_key()
        self.assertEqual(
            ThinkingCodec(key).decode(ThinkingCodec(key).encode("explicit")),
            "explicit",
        )

    def test_hidden_history_survives_new_frontend_and_count_tokens(self):
        first = Harness(
            FakeRuntime(Plan([[1], [2], [3]])),
            thinking_codec=ThinkingCodec(load_thinking_key(self.path)),
        )
        self.addCleanup(first.close)
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
            "thinking": {"type": "adaptive", "display": "omitted"},
        }
        status, _, payload = first.request("POST", "/v1/messages", body)
        self.assertEqual(status, 200, payload)
        response = json.loads(payload)
        signature = response["content"][0]["signature"]
        first.close()
        second = Harness(
            FakeRuntime(Plan([[3]])),
            thinking_codec=ThinkingCodec(load_thinking_key(self.path)),
        )
        self.addCleanup(second.close)
        self.assertIsNot(first.app, second.app)
        self.assertEqual(second.app.thinking_codec.decode(signature), "because ")
        body["messages"] += [
            {"role": "assistant", "content": response["content"]},
            {"role": "user", "content": "continue"},
        ]
        for path in ("/v1/messages/count_tokens", "/v1/messages"):
            status, _, payload = second.request("POST", path, body)
            self.assertEqual(status, 200, payload)
            self.assertEqual(
                second.tokenizer.templates[-1][0][1]["reasoning_content"],
                "because ",
            )

    def test_main_reports_key_error_before_model_loading(self):
        server = mock.Mock()
        args = SimpleNamespace(
            host="127.0.0.1",
            port=0,
            queue_size=1,
            allowed_host=[],
            api_key=None,
            no_webui=False,
            max_request_size=api.DEFAULT_MAX_REQUEST_BYTES,
        )
        with (
            mock.patch.object(api, "parse_args", return_value=args),
            mock.patch.object(api.signal, "signal"),
            mock.patch.object(api, "FrontendServer", return_value=server),
            mock.patch.object(
                api,
                "load_thinking_key",
                side_effect=ThinkingKeyError("invalid key file"),
            ),
            mock.patch.object(api.AutoTokenizer, "from_pretrained") as tokenizer,
            mock.patch.object(api.engine_runtime, "MultiplexedRuntime") as runtime,
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            api.main()
        self.assertIn("invalid key file", stderr.getvalue())
        tokenizer.assert_not_called()
        runtime.assert_not_called()
        server.server_activate.assert_not_called()
        server.server_close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
