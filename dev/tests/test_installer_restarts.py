import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from dev.tests.installer_fixtures import DENSE, MODEL, fake_hub, mlx_target
from dev.tools import installer_restarts as restarts
from install import hub, models


def in_process(fake):
    """The installer as run_installer runs it, in this process, where fake
    stands in for the Hub the environment names."""

    def run(arguments, environment, timeout=None):
        with contextlib.ExitStack() as stack:
            offline = environment.get("HF_HUB_OFFLINE") == "1"
            stack.enter_context(
                mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", offline)
            )
            if "HF_HUB_CACHE" in environment:
                stack.enter_context(
                    mock.patch(
                        "huggingface_hub.constants.HF_HUB_CACHE",
                        environment["HF_HUB_CACHE"],
                    )
                )
            fake.failure = (
                httpx.ConnectError("[Errno 61] Connection refused")
                if environment.get("HF_ENDPOINT") == restarts.UNREACHABLE_HUB
                else None
            )
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(output))
            status = models.main(arguments)
        return status, output.getvalue()

    return run


class InstallerRestartsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "hub"
        models_root = str(self.root / "models")
        self.arguments = ["--models", models_root, "--model", MODEL, "--language-only"]

    def check(self, run, arguments):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            restarts.check(arguments, run=run, hub_cache=self.cache)
        return output.getvalue()

    def test_every_restart_starts_the_installation_without_a_download(self):
        for revision in (None, "a" * 40):
            with self.subTest(revision=revision):
                fake = fake_hub(self, self.cache)
                arguments = self.arguments + (
                    ["--revision", revision] if revision else []
                )
                output = self.check(in_process(fake), arguments)
                for name in ("offline", "unreachable-Hub", "moved-cache"):
                    self.assertIn(f"{name} restart: PASS", output)

    def test_a_restart_that_installs_another_commit_fails(self):
        fake = fake_hub(self, self.cache)
        run = in_process(fake)

        def run_then_move(arguments, environment, timeout=None):
            result = run(arguments, environment, timeout)
            fake.publish(MODEL, "b" * 40, lambda p: mlx_target(p, DENSE))
            return result

        with self.assertRaisesRegex(restarts.RestartFailure, "moved-cache restart"):
            self.check(run_then_move, self.arguments)

    def test_a_restart_that_writes_to_a_hub_cache_fails(self):
        def blob(environment):
            if environment.get("HF_HUB_OFFLINE") == "1":
                return self.cache / hub.folder_name(MODEL) / "blobs" / "new"
            return None

        def moved(environment):
            if "HF_HUB_CACHE" in environment:
                return Path(environment["HF_HUB_CACHE"]) / "new"
            return None

        for name, where in (("offline", blob), ("moved-cache", moved)):
            with self.subTest(restart=name):
                self.setUp()
                run = in_process(fake_hub(self, self.cache))

                def writing(arguments, environment, timeout=None):
                    result = run(arguments, environment, timeout)
                    if path := where(environment):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("")
                    return result

                with self.assertRaisesRegex(
                    restarts.RestartFailure, f"the {name} restart wrote to a Hub cache"
                ):
                    self.check(writing, self.arguments)

    def test_a_restart_without_the_documented_line_fails(self):
        fake = fake_hub(self, self.cache)
        run = in_process(fake)

        def run_silently(arguments, environment, timeout=None):
            status, output = run(arguments, environment, timeout)
            return status, output.replace("Could not reach the Hub", "")

        with self.assertRaisesRegex(
            restarts.RestartFailure, "unreachable-Hub restart did not print"
        ):
            self.check(run_silently, self.arguments)


if __name__ == "__main__":
    unittest.main()
