"""What the installer tests share: model and draft repositories, selections,
Hub errors, and FakeHub, the stand-in for the Hugging Face Hub that the
upstream and GGUF tests install from. The legacy package tests
(test_models.py) mock huggingface_hub's functions directly: the frozen legacy
installer calls them with other arguments (token, repo_type,
force_download, and model_info without a timeout)."""

import hashlib
import json
import shutil
from types import SimpleNamespace
from unittest import mock

import httpx
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import RepoSibling

from install import families, hub, models

DENSE = families.named("Qwen3.8-27B")
MOE = families.named("Qwen3.6-35B-A3B")
MODEL = "mlx-community/Qwen3.8-27B-4bit"
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


def text_config(family, **changes):
    return dict(family.signature) | changes


def mlx_target(root, family, *, changes=None):
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "text_config": text_config(family, **(changes or {})),
        "quantization": {"bits": 4, "group_size": 64},
    }
    (root / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        (root / name).write_text("{}")
    return root


def draft_dir(root, family, **changes):
    """A DFlash2 release of family's draft: its configuration, stating the
    draft signature with changes (dotted keys), and weights."""
    root.mkdir(parents=True, exist_ok=True)
    config = {}
    for key, value in (dict(family.draft.signature) | changes).items():
        *objects, name = key.split(".")
        node = config
        for part in objects:
            node = node.setdefault(part, {})
        node[name] = list(value) if isinstance(value, tuple) else value
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.safetensors").write_bytes(b"draft")
    return root


def selection(root, model=MODEL, **options):
    """model's selection under root/models, text only unless options say."""
    return models.Selection.of(
        root / "models",
        model,
        revision=options.get("revision"),
        language_only=options.get("language_only", True),
        draft_model=options.get("draft_model"),
    )


def http_error(status):
    request = httpx.Request("GET", "https://huggingface.co/api/models/owner/model")
    return HfHubHTTPError(
        f"{status} Client Error.\n\nRevision Not Found for url: {request.url}.",
        response=httpx.Response(status, request=request),
    )


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
        root = self.remote / repo_id / str(commit)
        if not root.is_dir():
            raise http_error(404)
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

    def snapshot_download(self, repo_id, *, revision, allow_patterns, max_workers):
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
    """A FakeHub publishing MODEL at commit on main and every family's
    draft repository at DRAFT_COMMIT on main."""
    fake = FakeHub(test, cache)
    fake.publish(MODEL, commit, lambda p: mlx_target(p, target))
    for family in families.FAMILIES:
        fake.publish(family.draft.repo, DRAFT_COMMIT, lambda p: draft_dir(p, family))
    return fake


def cached_snapshot(cache, repo_id, commit, build):
    """repo_id at commit as a download leaves it in the Hub cache: a snapshot
    of only the files downloaded."""
    return build(cache / hub.folder_name(repo_id) / "snapshots" / commit)


def pins(cache):
    return sorted(ref.name for ref in cache.glob("*/refs/splash/*/*"))
