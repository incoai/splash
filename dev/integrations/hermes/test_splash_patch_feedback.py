import importlib.util
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location(
    "splash_patch_feedback",
    Path(__file__).with_name("splash-patch-feedback") / "__init__.py",
)
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)


class PatchFeedbackTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "main.py"
        self.path.write_text("value = 1\n")
        self.cwd = self.root
        self.check = Mock(return_value={"passed": True})
        self.policy = plugin.PatchFeedback(
            "http://127.0.0.1:11502/v1",
            "coder",
            str(self.root),
            lambda: self.cwd,
            self.check,
        )
        self.receipt = json.dumps({"files_modified": [str(self.path)], "success": True})

    def authorize(self, **overrides):
        options = dict(
            session_id="fixture",
            model="coder",
            base_url="http://127.0.0.1:11502/v1",
            api_mode="chat_completions",
        )
        self.policy.request(
            {"tools": [{"function": {"name": "run_visible_tests"}}]},
            **{**options, **overrides},
        )

    def invoke(self, result=None, session_id="fixture", tool_name="patch"):
        return self.policy.after_tool(
            tool_name,
            self.receipt if result is None else result,
            session_id=session_id,
            task_id="task",
        )

    def test_attaches_current_check_after_successful_patch(self):
        self.authorize()
        result = json.loads(self.invoke())
        self.assertEqual(result["files_modified"], [str(self.path)])
        self.assertEqual(result["visible_tests"], {"passed": True})
        self.check.assert_called_once_with("fixture", "task")

    def test_requires_current_session_route_workspace_and_successful_patch(self):
        self.assertIsNone(self.invoke())
        self.authorize(base_url="https://other.example/v1")
        self.assertIsNone(self.invoke())
        self.authorize()
        self.assertIsNone(self.invoke(session_id="other"))
        self.assertIsNone(self.invoke(tool_name="run_visible_tests"))
        self.assertIsNone(
            self.invoke(
                result=json.dumps(
                    {"files_modified": [str(self.path)], "error": "stale match"}
                )
            )
        )
        self.assertIsNone(self.invoke(result='{"files_modified":["/outside.py"]}'))
        self.cwd = self.root.parent
        self.assertIsNone(self.invoke())
        self.cwd = self.root
        self.policy.end("fixture")
        self.assertIsNone(self.invoke())
        self.check.assert_not_called()

    def test_timeout_keeps_the_patch_receipt(self):
        self.authorize()
        self.check.side_effect = subprocess.TimeoutExpired("visible-check", 1)
        result = json.loads(self.invoke())
        self.assertTrue(result["success"])
        self.assertIn("Run it separately", result["visible_tests"]["error"])

    def test_missing_checker_leaves_result_to_hermes(self):
        self.authorize()
        self.check.return_value = None
        self.assertIsNone(self.invoke())
