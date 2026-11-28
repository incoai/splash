import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import paths


class PathsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name) / "home"
        self.addCleanup(importlib.reload, paths)

    def _reload_paths(self, *, packaged, splash_data_dir):
        env = {**os.environ, "HOME": str(self.home)}
        if splash_data_dir is None:
            env.pop("SPLASH_DATA_DIR", None)
        else:
            env["SPLASH_DATA_DIR"] = splash_data_dir
        orig_is_file = Path.is_file

        def is_file(candidate):
            if candidate == paths.ROOT / "release.json":
                return packaged
            return orig_is_file(candidate)

        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(Path, "is_file", autospec=True, side_effect=is_file),
        ):
            return importlib.reload(paths)

    def test_defaults_without_splash_data_dir(self):
        for env_value in (None, ""):
            with self.subTest(packaged=False, env_value=env_value):
                mod = self._reload_paths(packaged=False, splash_data_dir=env_value)
                self.assertEqual(mod.DATA, mod.ROOT)
                self.assertEqual(mod.MODELS, mod.ROOT / "install/models")
                self.assertEqual(mod.RUNTIME, mod.ROOT / "build/runtime")

            with self.subTest(packaged=True, env_value=env_value):
                mod = self._reload_paths(packaged=True, splash_data_dir=env_value)
                expected = self.home / "Library/Application Support/Splash"
                self.assertEqual(mod.DATA, expected)
                self.assertEqual(mod.MODELS, expected / "models")
                self.assertEqual(mod.RUNTIME, expected / "runtime")

    def test_splash_data_dir_overrides_paths_in_source_and_packaged_modes(self):
        for packaged in (False, True):
            with self.subTest(packaged=packaged):
                mod = self._reload_paths(
                    packaged=packaged, splash_data_dir="~/isolated-splash"
                )
                expected = self.home / "isolated-splash"
                self.assertEqual(mod.DATA, expected)
                self.assertEqual(mod.MODELS, expected / "models")
                self.assertEqual(mod.RUNTIME, expected / "runtime")
