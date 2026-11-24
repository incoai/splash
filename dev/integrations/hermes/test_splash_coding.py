import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

path = Path(__file__).parent / "splash-coding" / "__init__.py"
spec = importlib.util.spec_from_file_location("splash_coding", path)
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)


class CodingPolicyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.cwd = self.root
        self.rev = "first"
        self.config = {
            "endpoint": "http://127.0.0.1:11502/v1",
            "route_model": "coder",
            "workspace": str(self.root),
            "cache_group": "approved",
            "shared_prefix": True,
            "bounded_reads": True,
            "system_prefix": "Reviewed common instructions",
        }
        self.policy = plugin.CodingPolicy(
            self.config, lambda: self.cwd, lambda _: self.rev
        )
        self.request = {
            "messages": [
                {"role": "system", "content": "Original"},
                {"role": "user", "content": "Task"},
            ],
            "tools": [{"name": "read_file"}],
            "extra_body": {"priority": "background"},
        }

    def prepare(self, session_id="a", request=None, **overrides):
        opts = dict(
            session_id=session_id,
            model="coder",
            base_url=self.config["endpoint"],
            api_mode="chat_completions",
        )
        return self.policy.request(
            self.request if request is None else request, **{**opts, **overrides}
        )

    def test_disabled_policy_neither_changes_requests_nor_bounds_tools(self):
        self.policy = plugin.CodingPolicy(
            {**self.config, "shared_prefix": False, "bounded_reads": False},
            lambda: self.cwd,
            lambda _: self.rev,
        )
        self.assertIsNone(self.prepare())
        self.assertIsNone(
            self.policy.before_tool("read_file", {"path": "large.py"}, session_id="a")
        )

    def test_same_workspace_and_revision_share_without_mutating_request(self):
        a = self.prepare()["request"]
        b = self.prepare("b")["request"]
        self.assertEqual(a["extra_body"]["cache_salt"], b["extra_body"]["cache_salt"])
        self.assertEqual(
            a["messages"][0]["content"], "Reviewed common instructions\n\nOriginal"
        )
        self.assertIs(a["messages"][1], self.request["messages"][1])
        self.assertEqual(self.request["messages"][0]["content"], "Original")
        self.assertEqual(a["extra_body"]["priority"], "background")

    def test_new_revision_and_new_tool_order_have_separate_scopes(self):
        a = self.prepare()["request"]["extra_body"]["cache_salt"]
        self.rev = "second"
        self.assertEqual(a, self.prepare()["request"]["extra_body"]["cache_salt"])
        self.assertNotEqual(a, self.prepare("b")["request"]["extra_body"]["cache_salt"])
        tools = [{"name": "write"}, {"name": "read"}]
        c = self.prepare(request={**self.request, "tools": tools})["request"][
            "extra_body"
        ]["cache_salt"]
        d = self.prepare(request={**self.request, "tools": tools[::-1]})["request"][
            "extra_body"
        ]["cache_salt"]
        self.assertNotEqual(c, d)

    def test_other_route_workspace_and_existing_salt_are_preserved(self):
        self.assertIsNone(self.prepare(base_url="http://splash.example:11502/v1"))
        self.assertIsNone(self.prepare(model="other"))
        self.assertIsNone(
            self.prepare(
                request={**self.request, "extra_body": {"cache_salt": "caller"}}
            )
        )
        self.cwd = self.root.parent
        self.assertIsNone(self.prepare())
        for url in (
            "http://localhost.evil:11502/v1",
            "http://u:p@localhost:11502/v1",
            "http://127.0.0.1:11502/v1?q=1",
            "http://[::1]/v1",
        ):
            self.assertIsNone(plugin.local_endpoint(url))
        self.assertEqual(
            plugin.local_endpoint("http://[::1]:11502/v1/"), "http://[::1]:11502/v1"
        )

    def test_bounded_read_can_expand_or_retry_without_another_veto(self):
        path = self.root / "large.py"
        path.write_text("print('example')\n" * 1000)

        def read(**kw):
            return self.policy.before_tool(
                "read_file", {"path": path.name, **kw}, session_id="a"
            )

        self.assertIsNone(read())
        self.prepare()
        self.assertIsNone(read(offset=50, limit=100))
        self.assertEqual(read()["action"], "block")
        self.assertIsNone(read())
        self.assertIsNone(read(offset=1, limit=950))
        self.policy.end("a")
        self.assertIsNone(read())

    def test_symlink_outside_workspace_and_unknown_revision(self):
        self.prepare()
        with TemporaryDirectory() as outside:
            path = Path(outside) / "large.py"
            path.write_text("x" * 12001)
            (self.root / "link.py").symlink_to(path)
            self.assertIsNone(
                self.policy.before_tool(
                    "read_file", {"path": "link.py"}, session_id="a"
                )
            )
        self.rev = None
        self.assertIsNone(self.prepare("b"))


if __name__ == "__main__":
    unittest.main()
