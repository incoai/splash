"""What the installer tests share: model and draft repositories, GGUF
headers, selections, Hub errors, and FakeHub, the stand-in for the Hugging
Face Hub that the upstream and GGUF tests install from. The installer checks
a model's configuration with the engine, build/splash, which make
test-python builds."""

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
from huggingface_hub.errors import (
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.hf_api import RepoSibling
from tokenizers import pre_tokenizers

from dev.tests import fixture_files
from install import families, hub, models

DENSE = families.named("Qwen3.8-27B")
MOE = families.named("Qwen3.6-35B-A3B")
FLASH_NEXT = families.named("Qwen3.8-Flash-Next")
FLINT = families.named("Flint")
# The families whose targets the engine loads from a GGUF only.
GGUF_ONLY = (FLASH_NEXT, FLINT)
MODEL = "mlx-community/Qwen3.8-27B-4bit"
# An MLX config.json (of a GGUF_ONLY family, the one the installer derives
# from its GGUF), a DFlash2 draft config and, of Qwen3.6-35B-A3B and the
# GGUF_ONLY families, a GGUF header's architecture metadata built like the
# family's (gguf-metadata.json) and a real GGUF's tensor table
# (gguf-tensors.json), in the family's lower-case name. The native
# configuration test reads the configs too.
CONFIGS = Path(__file__).resolve().parent / "fixtures" / "model-configs"
# The commit the main branch of every family's draft repository names.
DRAFT_COMMIT = "d" * 40
# The image preprocessing Splash implements (server/images.py).
PROCESSOR = {
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5] * 3,
    "image_std": [0.5] * 3,
}


def family_config(family, name="config.json"):
    """The family's fixture configuration at name: its MLX config.json,
    draft/config.json, its DFlash2 draft's, gguf-metadata.json or
    gguf-tensors.json."""
    return json.loads((CONFIGS / family.name.lower() / name).read_text())


def mlx_target(root, family, *, changes=None):
    """An MLX repository of family's target: its config.json, with changes
    to its text_config, tokenizer files and weights."""
    root.mkdir(parents=True, exist_ok=True)
    target = family_config(family)
    target["text_config"] |= changes or {}
    (root / "config.json").write_text(json.dumps(target))
    for name in ("tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        (root / name).write_text("{}")
    return root


def draft_dir(root, family, **changes):
    """A DFlash2 release of family's draft: its config.json, with changes
    (dotted keys), and weights."""
    root.mkdir(parents=True, exist_ok=True)
    draft = family_config(family, "draft/config.json")
    for key, value in changes.items():
        *objects, name = key.split(".")
        node = draft
        for part in objects:
            node = node[part]
        node[name] = value
    (root / "config.json").write_text(json.dumps(draft))
    (root / "model.safetensors").write_bytes(b"draft")
    return root


# The GGUF specification's token types (llama_token_type), stated
# independently of the reader under test.
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6
# GGML tensor types (ggml_type).
GGML = {
    "F32": 0,
    "F16": 1,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q2_K": 10,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "IQ2_XXS": 16,
    "IQ3_XXS": 18,
    "IQ1_S": 19,
    "IQ4_NL": 20,
    "IQ3_S": 21,
    "IQ2_S": 22,
    "IQ4_XS": 23,
    "IQ1_M": 29,
    "BF16": 30,
    "TQ2_0": 35,
    "MXFP4": 39,
    "NVFP4": 40,
    "Q2_0": 42,
    "PQ2_0": 142,
}


def write_gguf(path, values, tensors=()):
    """A GGUF header of the values and the tensors' (name, GGML type), each
    tensor 256 values without data: the metadata reader reads no more."""
    table = [(name, [256], kind, b"") for name, kind in tensors]
    return fixture_files.write_gguf(path, values, table)


def tensor_table(family=MOE):
    """The tensors of a real GGUF of family, in its order: name, extents
    (innermost first) and GGML type. Qwen3.6-35B-A3B's are
    unsloth/Qwen3.6-35B-A3B-GGUF's Qwen3.6-35B-A3B-UD-Q4_K_M.gguf (revision
    a483e9e6cbd595906af30beda3187c2663a1118c), Qwen3.8-Flash-Next's the three
    parts of unsloth/Qwen3.8-Flash-Next-GGUF's UD-IQ3_XXS (revision
    766911a6b7369840a91dbcd95f9f997acaab6cd6), whose headers alone were read."""
    return [
        (name, extents, GGML[kind])
        for name, (kind, extents) in family_config(family, "gguf-tensors.json").items()
    ]


def write_target(path, values, tensors):
    """A GGUF header of the values and the tensors' (name, extents, GGML
    type) without data, as tensor_table lists them."""
    table = [(name, extents, kind, b"") for name, extents, kind in tensors]
    return fixture_files.write_gguf(path, values, table)


def fixture(*, native=False, family=MOE):
    """A GGUF's metadata of family's (Qwen3.6-35B-A3B's by default): its
    architecture's and a tokenizer's, native=True of Qwen's vocabulary
    size."""
    tokens = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokens += ["ab", "<|endoftext|>", "<|im_end|>", "<think>", "<|im_start|>"]
    types = [NORMAL] * 257 + [CONTROL, CONTROL, USER_DEFINED, CONTROL]
    if native:
        padding = 248320 - len(tokens)
        tokens += [f"[unused{i}]" for i in range(padding)]
        types += [UNUSED] * padding
    return family_config(family, "gguf-metadata.json") | {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.eos_token_id": 258,
        "tokenizer.ggml.bos_token_id": 257,
        "tokenizer.ggml.padding_token_id": 257,
        "tokenizer.chat_template": "{% for message in messages %}{{ message.content }}<|im_end|>{% endfor %}",
    }


def vision_fixture():
    return {
        "general.architecture": "clip",
        "clip.projector_type": "qwen3vl_merger",
        "clip.use_gelu": True,
        "clip.vision.block_count": 27,
        "clip.vision.embedding_length": 1152,
        "clip.vision.attention.head_count": 16,
        "clip.vision.feed_forward_length": 4304,
        "clip.vision.projection_dim": 2048,
        "clip.vision.patch_size": 16,
        "clip.vision.spatial_merge_size": 2,
        "clip.vision.image_size": 768,
        "clip.vision.is_deepstack_layers": [False] * 27,
        "clip.vision.image_mean": [0.5] * 3,
        "clip.vision.image_std": [0.5] * 3,
    }


def selection(root, model=MODEL, **options):
    """model's selection under root/models, text only unless options say."""
    return models.Selection.of(
        root / "models",
        model,
        revision=options.get("revision"),
        language_only=options.get("language_only", True),
        draft_model=options.get("draft_model"),
    )


def http_error(status, kind=HfHubHTTPError, missing="Revision"):
    """A Hub refusal of status, as huggingface_hub raises it: kind, by the
    Hub's error code (RepositoryNotFoundError for a repository it does not
    have, RevisionNotFoundError for a revision its repository lacks)."""
    request = httpx.Request("GET", "https://huggingface.co/api/models/owner/model")
    return kind(
        f"{status} Client Error.\n\n{missing} Not Found for url: {request.url}.",
        response=httpx.Response(status, request=request),
    )


def missing_repository():
    """The Hub's answer, to a signed-in request, for a repository it does not
    have."""
    return http_error(404, RepositoryNotFoundError, "Repository")


def missing_revision():
    """The Hub's answer for a revision a repository lacks."""
    return http_error(404, RevisionNotFoundError)


def listing(directory):
    return sorted(
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    )


class FakeHub:
    """The Hub as huggingface_hub presents it to the installer: repositories
    whose branches name commits, and downloads that fill the test's Hub cache
    with snapshots of only the files requested. It patches huggingface_hub for
    the test's duration."""

    def __init__(self, test, cache):
        self.cache = cache
        self.remote = cache.parent / "remote"
        self.branches = {}
        # What the next Hub request raises: resolution, or downloads.
        self.failure = self.download_failure = None
        self.requests, self.downloads, self.range_reads = [], [], []
        for name, replacement in (
            ("huggingface_hub.constants.HF_HUB_CACHE", str(cache)),
            ("huggingface_hub.constants.HF_HUB_OFFLINE", False),
            ("huggingface_hub.get_token", lambda: None),
            ("huggingface_hub.HfApi", lambda **options: self),
            ("huggingface_hub.snapshot_download", self.snapshot_download),
            ("huggingface_hub.hf_hub_download", self.hf_hub_download),
            ("huggingface_hub.HfFileSystem", lambda: self),
            ("huggingface_hub.try_to_load_from_cache", self.try_to_load_from_cache),
        ):
            patch = mock.patch(name, replacement)
            patch.start()
            test.addCleanup(patch.stop)

    def publish(self, repo_id, commit, build, branch="main"):
        build(self.remote / repo_id / commit)
        self.branches[repo_id, branch] = commit

    def snapshot(self, repo_id, commit):
        return self.cache / hub.folder_name(repo_id) / "snapshots" / commit

    def model_info(self, repo_id, *, revision=None, files_metadata, timeout):
        self.requests.append((repo_id, revision))
        assert files_metadata and timeout == hub.HUB_TIMEOUT
        if self.failure:
            raise self.failure
        commit = revision
        if not models.is_hex_digest(revision, 40):
            commit = self.branches.get((repo_id, revision or "main"))
        if not (self.remote / repo_id).is_dir():
            raise missing_repository()
        root = self.remote / repo_id / str(commit)
        if not root.is_dir():
            raise missing_revision()
        return SimpleNamespace(
            sha=commit,
            siblings=[
                RepoSibling(
                    rfilename=name,
                    size=(root / name).stat().st_size,
                    blob_id=hashlib.sha1((root / name).read_bytes()).hexdigest(),
                )
                for name in listing(root)
            ],
        )

    def fetch(self, repo_id, name, revision):
        if self.download_failure:
            raise self.download_failure
        path = self.snapshot(repo_id, revision) / name
        # As huggingface_hub does, a cached file is returned as it is.
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.remote / repo_id / revision / name, path)
            self.downloads.append(f"{repo_id}/{name}")
        return path

    def snapshot_download(
        self, repo_id, *, revision, allow_patterns, max_workers, tqdm_class=None
    ):
        for name in allow_patterns:
            self.fetch(repo_id, name, revision)
        return str(self.snapshot(repo_id, revision))

    def hf_hub_download(self, repo_id, filename, *, revision):
        return str(self.fetch(repo_id, filename, revision))

    def try_to_load_from_cache(self, repo_id, filename, *, revision):
        path = self.snapshot(repo_id, revision) / filename
        return str(path) if path.is_file() else None

    def open(self, path, mode, *, revision, block_size):
        owner, name, filename = path.split("/", 2)
        self.range_reads.append(path)
        return (self.remote / owner / name / revision / filename).open(mode)


def fake_hub(test, cache, *, target=DENSE, commit="a" * 40):
    """A FakeHub publishing MODEL at commit on main and the DFlash2 draft
    released for each family at DRAFT_COMMIT on main."""
    fake = FakeHub(test, cache)
    fake.publish(MODEL, commit, lambda p: mlx_target(p, target))
    for family in families.FAMILIES:
        fake.publish(family.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, family))
    return fake


def cached_snapshot(cache, repo_id, commit, build):
    """repo_id at commit as a download leaves it in the Hub cache: a snapshot
    of only the files downloaded."""
    return build(cache / hub.folder_name(repo_id) / "snapshots" / commit)


def pins(cache):
    return sorted(ref.name for ref in cache.glob("*/refs/splash/*/*"))
