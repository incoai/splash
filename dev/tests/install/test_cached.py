import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.installer_fixtures import (
    DENSE,
    MODEL,
    PROCESSOR,
    cached_snapshot,
    fake_hub,
    mlx_target,
    selection,
)
from install import cached, upstream


def vision_target(root):
    """An MLX target whose checkpoint holds its vision tower, as the app
    serves it."""
    shards = {
        "vision_tower.blocks.0.attn.qkv.weight": "model-00001-of-00002.safetensors",
        "language_model.lm_head.weight": "model-00002-of-00002.safetensors",
    }
    mlx_target(root, DENSE)
    (root / "model.safetensors").unlink()
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": shards})
    )
    for name in set(shards.values()):
        (root / name).write_text(name)
    (root / "preprocessor_config.json").write_text(json.dumps(PROCESSOR))


def readme(root):
    root.mkdir(parents=True)
    (root / "README.md").write_text("Notes")


class CachedTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "hub"

    def install_cached(self):
        with mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            return list(cached.install_cached(self.root / "app"))

    def test_installs_what_the_cache_holds_whole_without_the_hub(self):
        """A model another Splash downloaded installs here from the cache, at
        the commit it recorded for the default branch; a repository of no
        model, one whose default branch no download recorded, or one only a
        --revision installed, installs nothing; an installed model is left as
        it is."""
        fake = fake_hub(self, self.cache)
        fake.publish(MODEL, "d" * 40, vision_target)
        fake.publish("owner/branch-only", "e" * 40, vision_target, branch="dev")
        with contextlib.redirect_stdout(io.StringIO()):
            upstream.prepare(selection(self.root / "checkout", language_only=False))
            upstream.prepare(
                selection(
                    self.root / "checkout",
                    "owner/branch-only",
                    revision="dev",
                    language_only=False,
                )
            )
        cached_snapshot(self.cache, "someone/notes", "b" * 40, readme)
        (self.cache / "models--someone--notes/refs").mkdir()
        (self.cache / "models--someone--notes/refs/main").write_text("b" * 40)
        cached_snapshot(self.cache, "someone/unrecorded", "c" * 40, readme)
        fake.requests.clear(), fake.downloads.clear()
        self.assertEqual(self.install_cached(), [MODEL])
        self.assertEqual(self.install_cached(), [])
        self.assertEqual((fake.requests, fake.downloads), ([], []))

    def test_candidates_follow_the_installer_s_names(self):
        """One ID for each GGUF target the cached files hold whole, which
        selects it from them and from the Hub's listing of all the
        repository's files alike: a suggested one, or else its own."""
        suggested = {"owner/m-GGUF": ["owner/m-GGUF:UD-Q4_K_M", "owner/m-GGUF:IQ1_S"]}
        hub_files = {
            f"m-{variant}.gguf"
            for variant in ("Q4_K_M", "Q8_0", "UD-Q4_K_M", "UD-Q8_0", "UD-IQ1_S")
        } | {"mmproj-BF16.gguf", "m.gguf"}
        for cached_files, expected in (
            # Variants named as on the Hub, not past the words only the
            # cached files share.
            (
                {"m-UD-Q4_K_M.gguf", "m-UD-Q8_0.gguf", "mmproj-BF16.gguf"},
                ["owner/m-GGUF:UD-Q4_K_M", "owner/m-GGUF:UD-Q8_0"],
            ),
            ({"m-Q4_K_M.gguf"}, ["owner/m-GGUF:Q4_K_M"]),
            # A root GGUF beside a split GGUF's folder; an incomplete split
            # holds no model whole.
            (
                {
                    "m-Q4_K_M.gguf",
                    "BF16/m-BF16-00001-of-00002.gguf",
                    "BF16/m-BF16-00002-of-00002.gguf",
                    "Q2_0/m-Q2_0-00001-of-00002.gguf",
                    "README.md",
                },
                ["owner/m-GGUF:BF16", "owner/m-GGUF:Q4_K_M"],
            ),
            # No ID selects the GGUF named as the model beside another.
            ({"m.gguf", "m-Q8_0.gguf"}, ["owner/m-GGUF:Q8_0"]),
        ):
            with self.subTest(cached=sorted(cached_files)):
                ids = cached.candidates("owner/m-GGUF", cached_files, {})
                self.assertEqual(ids, expected)
                for model in ids:
                    variant = model.partition(":")[2] or None
                    self.assertEqual(
                        upstream.select_gguf(cached_files, variant)[0],
                        upstream.select_gguf(
                            hub_files | {n for n in cached_files if "/" in n},
                            variant,
                        )[0],
                    )
        # A split GGUF's parts at the root beside another variant's
        # (upstream.root_split_gguf), as ISTA-DASLab's DeepSeek-V4.1 GGUFs
        # keep theirs, are one target; an incomplete split holds none.
        root_split = {
            "m-Q8_0.gguf",
            "m-IQ1_M-00001-of-00002.gguf",
            "m-IQ1_M-00002-of-00002.gguf",
            "m-IQ2_M-00001-of-00002.gguf",
        }
        self.assertEqual(
            cached.candidates("owner/m-GGUF", root_split, {}),
            ["owner/m-GGUF:IQ1_M", "owner/m-GGUF:Q8_0"],
        )
        self.assertEqual(
            upstream.select_gguf(root_split, "IQ1_M")[0],
            ["m-IQ1_M-00001-of-00002.gguf", "m-IQ1_M-00002-of-00002.gguf"],
        )
        # A suggested repository's cached files take its suggested IDs, and
        # the others their own.
        self.assertEqual(
            cached.candidates(
                "owner/m-GGUF", {"m-UD-IQ1_S.gguf", "m-Q8_0.gguf"}, suggested
            ),
            ["owner/m-GGUF:Q8_0", "owner/m-GGUF:IQ1_S"],
        )
        # A file named for another model than the repository's is named past
        # its first word.
        self.assertEqual(
            cached.candidates("owner/x-GGUF", {"m-Q4_K_M.gguf", "m-Q8_0.gguf"}, {}),
            ["owner/x-GGUF:Q4_K_M", "owner/x-GGUF:Q8_0"],
        )
        self.assertEqual(cached.candidates("owner/m", {"m.gguf"}, {}), ["owner/m"])
        self.assertEqual(
            cached.candidates("owner/m-4bit", {"config.json", "model.safetensors"}, {}),
            ["owner/m-4bit"],
        )


if __name__ == "__main__":
    unittest.main()
