"""What the installer tests share: the audio repositories as the Hub
publishes them, selections, Hub errors, and FakeHub, the stand-in for the
Hugging Face Hub that the upstream tests install from. The installer checks
a model's configuration with the engine, build/splash; the tests stub
models.run_engine."""

import contextlib
import hashlib
import json
import os
import shutil
from types import SimpleNamespace
from unittest import mock

import httpx
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import RepoSibling

from install import families, hub, models

TTS = families.named("qwen3-tts")
ASR = families.named("qwen3-asr")
TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
# The model the upstream tests install, at the commit its main names.
MODEL = TTS_MODEL
COMMIT = "a" * 40
# The files audio_repo writes beside config.json, every one downloaded with
# the checkpoint (the whole snapshot goes under target/).
REPO_FILES = (
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "preprocessor_config.json",
    "model.safetensors.index.json",
    "README.md",
)


def tts_config(quantized=True):
    config = {
        "model_type": "qwen3_tts",
        "architectures": ["Qwen3TTSForConditionalGeneration"],
        "talker_config": {"codec_language_id": {"english": 2050, "chinese": 2055}},
    }
    if quantized:
        config["quantization_config"] = {"bits": 4, "group_size": 64, "mode": "affine"}
    return config


def asr_config():
    return {
        "model_type": "qwen3_asr",
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "support_languages": ["Chinese", "English"],
    }


def audio_repo(root, config=None, *, speech_tokenizer=False):
    """An MLX audio repository as the Hub publishes it: vocab.json and
    merges.txt rather than tokenizer.json, and for speech the speech
    tokenizer's subdirectory. The TTS configuration unless given."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(tts_config() if config is None else config)
    )
    for name in REPO_FILES:
        (root / name).write_text("{}")
    (root / "model.safetensors").write_bytes(b"weights")
    if speech_tokenizer:
        (root / "speech_tokenizer").mkdir(exist_ok=True)
        (root / "speech_tokenizer/config.json").write_text("{}")
        (root / "speech_tokenizer/model.safetensors").write_bytes(b"codec")
    return root


def omni_check(family="qwen3-tts"):
    """An engine whose omni-check names family, recording its command lines:
    (calls, the stand-in for models.run_engine)."""
    calls = []

    def run(arguments, check):
        calls.append(arguments)
        if arguments[0] != "omni-check":
            raise AssertionError(f"unexpected engine check {arguments}")
        return json.dumps({"family": family})

    return calls, run


def engine_patch(family="qwen3-tts"):
    """A patch of models.run_engine with an engine whose omni-check names
    family."""
    return mock.patch.object(models, "run_engine", side_effect=omni_check(family)[1])


def selection(root, model=MODEL, *, revision=None):
    """model's selection under root/models."""
    return models.Selection.of(root / "models", model, revision=revision)


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
        the test's duration, and points HF_HUB_CACHE at the test's cache, as
    huggingface_hub reads it."""

    def __init__(self, test, cache):
        self.cache = cache
        self.remote = cache.parent / "remote"
        self.branches = {}
        # What the next Hub request raises: resolution, or downloads.
        self.failure = self.download_failure = None
        self.requests, self.downloads, self.range_reads = [], [], []
        environment = mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(cache)})
        environment.start()
        test.addCleanup(environment.stop)
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

    def calls(self):
        """Every Hub request made so far: listings, downloads and range
        reads."""
        return self.requests + self.downloads + self.range_reads

    def clear(self):
        self.requests.clear(), self.downloads.clear(), self.range_reads.clear()

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


def fake_hub(test, cache, *, commit=COMMIT, config=None):
    """A FakeHub publishing MODEL at commit on main, an audio repository of
    the TTS configuration unless config says otherwise."""
    fake = FakeHub(test, cache)
    fake.publish(MODEL, commit, lambda p: audio_repo(p, config))
    return fake


def cached_snapshot(cache, repo_id, commit, build):
    """repo_id at commit as a download leaves it in the Hub cache: a snapshot
    of only the files downloaded."""
    return build(cache / hub.folder_name(repo_id) / "snapshots" / commit)


def cached_model(cache, repo_id, commit, build, *, branch="main"):
    """repo_id as `hf download` leaves it in the Hub cache: a whole snapshot
    of commit, which refs/<branch> names."""
    path = cached_snapshot(cache, repo_id, commit, build)
    refs = cache / hub.folder_name(repo_id) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / branch).write_text(commit)
    return path


@contextlib.contextmanager
def hub_cache(path):
    """The Hub cache moved to path, as huggingface_hub and the installer read
    it."""
    with (
        mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(path)}),
        mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(path)),
    ):
        yield


@contextlib.contextmanager
def no_hub():
    """The Hub client's entry points replaced by mocks that fail the test if
    called; each is asserted uncalled on the way out."""
    names = ("HfApi", "snapshot_download", "hf_hub_download", "HfFileSystem")
    with contextlib.ExitStack() as stack:
        clients = [
            stack.enter_context(
                mock.patch(
                    f"huggingface_hub.{name}",
                    side_effect=AssertionError(f"the Hub was asked ({name})"),
                )
            )
            for name in names
        ]
        yield clients
        for client in clients:
            client.assert_not_called()


def pins(cache):
    return sorted(ref.name for ref in cache.glob("*/refs/splash/*/*"))
