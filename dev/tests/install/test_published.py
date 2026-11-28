"""The installer's knowledge of each family against what its publishers
release, read from the Hub cache: the sampling defaults (families.py)
against their generation_config.json, and the tokenizer the installer
builds from a GGUF against the publisher's tokenizer.json. A test whose
files the cache lacks is skipped, by the repository it lacks."""

import json
import random
import unittest
from pathlib import Path

from tokenizers import Tokenizer

from install import families, gguf, hub

REPO = Path(__file__).resolve().parents[3]
# Repositories whose generation_config.json is the family's publisher's, as
# the publisher or a conversion that keeps its file releases it.
GENERATION_CONFIGS = {
    "Qwen3.8-27B": ("Qwen/Qwen3.8-27B", "nvidia/Qwen3.8-27B-NVFP4"),
    "Qwen3.6-35B-A3B": ("Qwen/Qwen3.6-35B-A3B", "mlx-community/Qwen3.6-35B-A3B-4bit"),
    "Qwen3.8-Flash-Next": (
        "Qwen/Qwen3.8-Flash-Next",
        "nvidia/Qwen3.8-Flash-Next-NVFP4",
    ),
    "Flint": ("mlx-community/flint-mixed-3-4bit",),
    "Inkling-2-Mini": (
        "thinkingmachines/Inkling-2-Mini",
        "mlx-community/Inkling-2-Mini-4bit",
    ),
}
# A GGUF repository of each family, and a repository holding the
# publisher's tokenizer.json.
TOKENIZERS = (
    ("unsloth/Qwen3.8-27B-GGUF", "Qwen/Qwen3.8-27B"),
    ("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen/Qwen3.6-35B-A3B"),
    ("unsloth/Qwen3.8-Flash-Next-GGUF", "Qwen/Qwen3.8-Flash-Next"),
    ("unsloth/flint-GGUF", "mlx-community/flint-mixed-3-4bit"),
    ("unsloth/Inkling-2-Mini-GGUF", "mlx-community/Inkling-2-Mini-4bit"),
)


def cached_file(repo_id, *patterns):
    """A file of repo_id's snapshots in the Hub cache that matches one of
    patterns, not a GGUF projector (mmproj), from the newest snapshot that
    holds one; None when the cache holds none."""
    for commit in hub.cached_snapshots(repo_id):
        for pattern in patterns:
            for path in sorted(hub.snapshot(repo_id, commit).glob(pattern)):
                if path.is_file() and not path.name.startswith("mmproj"):
                    return path
    return None


def corpus():
    """Text to encode: paragraphs of this repository's docs and code,
    hand-written pieces of scripts, spacing, numbers and control-token
    spellings, and seeded random strings."""
    pieces = []
    for pattern in ("README.md", "docs/*.md", "server/*.py", "install/*.py"):
        for path in sorted(REPO.glob(pattern))[:20]:
            text = path.read_text(errors="replace")
            pieces += [piece for piece in text.split("\n\n") if piece.strip()]
    pieces += [
        "你好，世界！这是一个分词器一致性测试。",
        "こんにちは世界。トークナイザーのテストです。",
        "안녕하세요, 토크나이저 테스트입니다.",
        "Привет, мир! Проверка токенизатора.",
        "مرحبا بالعالم، اختبار المُرمِّز.",
        "Ça coûte 12 345,67 € — déjà vu? naïve façade.",
        "emoji: 😀👍🏽👨‍👩‍👧‍👦🇨🇳 ✈️ ½ ① ㊙️",
        "a\u0301 e\u0308 ﬁ ｆｕｌｌｗｉｄｔｈ １２３",
        "    indented\n\tTabbed\t\tcolumns   and      runs of spaces",
        "0 00 000 1234567890 3.14159 1e-9 0xDEADBEEF 2026-11-25T02:30:00Z",
        "https://example.com/a/b?c=d&e=f#g C:\\path\\to\\file.txt",
        "<|im_start|>user\nhello<|im_end|> <think>x</think> <tool_call>{}</tool_call>",
        "don't can't I'll we've O'Neil 'quoted' \"double\"",
        "\r\nwindows\r\nline endings\r\n",
        "x" * 300,
        " leading space",
        "trailing space ",
    ]
    rng = random.Random(7)
    for _ in range(2000):
        pieces.append(
            "".join(
                chr(
                    rng.choice(
                        (
                            rng.randint(32, 126),
                            rng.randint(0x4E00, 0x4E80),
                            rng.randint(0x400, 0x44F),
                            0x20,
                            0x0A,
                            rng.randint(0x1F600, 0x1F64F),
                        )
                    )
                )
                for _ in range(rng.randint(5, 80))
            )
        )
    return pieces


class PublishedTests(unittest.TestCase):
    def test_sampling_defaults_are_the_publishers(self):
        # Each value the publisher's file sets; one it leaves out is the
        # GGUF release's to state (Inkling 2's sets temperature alone).
        compared = 0
        for name, repositories in GENERATION_CONFIGS.items():
            sampling = families.named(name).sampling
            for repo_id in repositories:
                path = cached_file(repo_id, "generation_config.json")
                with self.subTest(family=name, repository=repo_id):
                    if path is None:
                        self.skipTest(f"the Hub cache holds no {repo_id}")
                    published = json.loads(path.read_text())
                    for field in ("temperature", "top_p", "top_k"):
                        if field in published:
                            self.assertEqual(
                                getattr(sampling, field), published[field], field
                            )
                    compared += 1
        if not compared:
            self.skipTest("the Hub cache holds no publisher's generation_config.json")

    def test_a_gguf_tokenizer_encodes_as_the_publishers(self):
        pieces = corpus()
        compared = 0
        for gguf_repo, publisher in TOKENIZERS:
            with self.subTest(gguf=gguf_repo, publisher=publisher):
                target = cached_file(gguf_repo, "*.gguf", "*/*-00001-of-*.gguf")
                published = cached_file(publisher, "tokenizer.json")
                if target is None or published is None:
                    self.skipTest(f"the Hub cache holds no {gguf_repo} or {publisher}")
                files = gguf.tokenizer_files(gguf.Metadata(target))
                built = Tokenizer.from_str(files["tokenizer/tokenizer.json"].decode())
                theirs = Tokenizer.from_file(str(published))
                differing = [
                    piece
                    for piece in pieces
                    if built.encode(piece, add_special_tokens=False).ids
                    != theirs.encode(piece, add_special_tokens=False).ids
                ]
                self.assertEqual(differing[:3], [], f"{len(differing)} differ")
                compared += 1
        if not compared:
            self.skipTest(
                "the Hub cache holds no GGUF beside its publisher's tokenizer"
            )


if __name__ == "__main__":
    unittest.main()
