"""The Hugging Face Hub and its cache, as the installers read them.

A source is one repository at one commit (Repository): read from that
commit's snapshot in the Hub cache, listed by the Hub when the cache lacks
it, or a local directory. What the cache holds is decided here, without the
Hub (cached_commit, whole_snapshot), for the installer and the setup page
alike. An installation pins every snapshot it links with a ref of its own,
refs/splash/<installation>/<commit>, so neither pruning the cache nor
another installation can remove its files.

This module needs the standard library alone until it asks the Hub:
huggingface_hub is imported where it is used, since the launcher and the
setup page import the installer modules before the environment that
provides it exists, and the tests replace its functions.
"""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from server import serve_options

from . import models

# Seconds the Hub may take to list a repository a start downloads.
HUB_TIMEOUT = 10


def token():
    """The token Hub requests carry: HF_TOKEN, else the saved `hf auth login`."""
    from huggingface_hub import get_token

    return os.environ.get("HF_TOKEN") or get_token()


def reason(error, used_token=None) -> str:
    """The Hub's reason for error on one line, with the token redacted (the
    one the request used, or the configured one) and, for denied access, how
    to authenticate."""
    from huggingface_hub.errors import HfHubHTTPError

    message = " ".join(str(error).split()) or type(error).__name__
    if secret := used_token or token():
        message = message.replace(secret, "[redacted]")
    if (
        isinstance(error, HfHubHTTPError)
        and error.response is not None
        and error.response.status_code in (401, 403)
    ):
        message += (
            "; set HF_TOKEN or run 'hf auth login' with access to this repository"
        )
    return message


@contextmanager
def as_model_errors(context: str):
    """Report a Hub, network or cache failure in the block as a ModelError,
    "context: reason", which callers print without a traceback."""
    import httpx

    try:
        yield
    except (OSError, httpx.HTTPError) as error:
        raise models.ModelError(f"{context}: {reason(error)}") from error


def folder_name(repo_id: str) -> str:
    """The folder of a model repository in any Hub cache."""
    return "models--" + repo_id.replace("/", "--")


def cache() -> Path:
    """The Hub cache, by the variables huggingface_hub reads it from:
    HF_HUB_CACHE, else HUGGINGFACE_HUB_CACHE, else HF_HOME's hub, else
    XDG_CACHE_HOME's huggingface/hub, else ~/.cache/huggingface/hub."""
    home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join("~", ".cache"),
        "huggingface",
    )
    path = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.join(home, "hub")
    )
    return Path(os.path.expanduser(os.path.expandvars(path)))


def folder(repo_id: str) -> Path:
    """Where the Hub cache keeps repo_id."""
    return cache() / folder_name(repo_id)


def snapshot(repo_id: str, commit: str) -> Path:
    return folder(repo_id) / "snapshots" / commit


def snapshot_commit(path: Path, repo_id: str) -> str:
    """The commit of path, which must be a snapshot folder of repo_id in a Hub
    cache, wherever that cache is."""
    if (
        path.parent.name != "snapshots"
        or path.parent.parent.name != folder_name(repo_id)
        or not models.is_hex_digest(path.name, 40)
    ):
        raise models.ModelError(
            "installed package is not a snapshot of the requested Hub repository"
        )
    return path.name


def snapshot_of(path) -> Path | None:
    """The Hub snapshot folder a cached file's path lies in; None for a file
    outside any Hub cache."""
    for parent in Path(path).absolute().parents:
        if parent.parent.name == "snapshots":
            return parent
    return None


def pin_owner(installation: Path) -> str:
    """The refs/splash folder holding an installation's pins. Each
    installation owns its references; Hub branch updates and other
    installations must not unpin this installation's current weights."""
    owner = installation.parent.resolve() / installation.name
    return hashlib.sha256(os.fsencode(owner)).hexdigest()


def pinned(path: Path, installation: Path) -> Path:
    """installation's pin of the snapshot at path, the reference pin writes:
    refs/splash/<installation>/<commit> in its repository's folder."""
    return path.parent.parent / "refs" / "splash" / pin_owner(installation) / path.name


def pin(path: Path, repo_id: str, installation: Path) -> Path:
    """Pin the snapshot at path for installation (pinned), so pruning the Hub
    cache cannot remove files the installation links."""
    commit = snapshot_commit(path, repo_id)
    ref = pinned(path, installation)
    try:
        existing = ref.read_text()
    except FileNotFoundError:
        pass
    else:
        if existing != commit:
            raise models.ModelError(f"invalid installed snapshot reference: {ref}")
        return ref
    ref.parent.mkdir(parents=True, exist_ok=True)
    # A pin's content is its own name, so writers never disagree: publish it by
    # rename, which file systems without hard links (exFAT) support too.
    with tempfile.NamedTemporaryFile(dir=ref.parent, delete=False) as temporary:
        pending = Path(temporary.name)
        try:
            temporary.write(commit.encode())
            temporary.flush()
            os.fsync(temporary.fileno())
            os.replace(pending, ref)
        finally:
            pending.unlink(missing_ok=True)
    return ref


def retire_other_pins(pins):
    """Remove the installation's pins beside pins: those of a commit it no
    longer links, which reinstalling from another commit leaves (a pin of a
    snapshot no longer cached is one `hf cache scan` reports). Call it only
    after publishing and verifying the installation these pin; other
    installations own other folders."""
    try:
        for folder in {ref.parent for ref in pins}:
            for previous in folder.iterdir() if folder.is_dir() else ():
                if previous not in pins and models.is_hex_digest(previous.name, 40):
                    previous.unlink()
    except OSError as error:
        # Keeping an old pin uses cache space but cannot invalidate the
        # verified installation or its successfully retained current pin.
        models.warn(f"could not retire old Hub cache references: {error}")


def repair_pins(installation: Path, snapshots):
    """Pin a verified installation's (snapshot, repository ID) pairs again,
    restoring a pin that was lost or never written, and retire its older
    pins. Only local files are written; a read-only cache leaves the verified
    model usable, with a warning."""
    try:
        pins = [pin(path, repo_id, installation) for path, repo_id in snapshots]
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
            raise
        models.warn(
            "the verified model can be used, but its Hub cache reference could "
            f"not be retained; protect this snapshot from external cache pruning: {error}"
        )
        return
    retire_other_pins(pins)


def is_whole(path: Path) -> bool:
    """Whether the snapshot folder at path holds a whole checkpoint:
    config.json and a safetensors file among its files, every file resolving
    to its blob, and every shard its safetensors index names."""
    try:
        files = [file for file in path.rglob("*") if not file.is_dir()]
    except OSError:
        return False
    if not (path / "config.json").is_file() or not any(
        file.suffix == ".safetensors" for file in files
    ):
        return False
    if not all(file.exists() for file in files):
        return False
    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = models.read_json(index).get("weight_map", {})
        except models.ModelError:
            return False
        if not isinstance(shards, dict) or not all(
            models.is_safe_path(name) and (path / name).is_file()
            for name in set(shards.values())
        ):
            return False
    return True


def cached_commit(repo_id: str, revision=None, installation=None) -> str | None:
    """The commit of a whole snapshot of repo_id in the Hub cache that serves
    revision without the Hub, else None. The candidates, most specific
    first: revision itself when it is a commit; else the commit installation
    (a selection link) records, then the ones it pinned (refs/splash), found
    however the installation was damaged; then the commit the cache records
    for the branch or tag (refs/<revision>, refs/main by default), which
    `hf download` writes and Splash, downloading by commit, does not. A
    cached commit is taken as it is: the Hub is never asked whether the
    branch has moved since."""
    if models.is_hex_digest(revision, 40):
        commits = [revision.lower()]
    else:
        commits = []
        if installation is not None:
            commits += _installation_commits(repo_id, Path(installation))
        try:
            ref = folder(repo_id) / "refs" / (revision or "main")
            commits.append(ref.read_text().strip())
        except (OSError, UnicodeDecodeError):
            pass
    for commit in dict.fromkeys(c for c in commits if models.is_hex_digest(c, 40)):
        if is_whole(snapshot(repo_id, commit)):
            return commit
    return None


def _installation_commits(repo_id, installation):
    """The commits of repo_id an installation records (its model.json, read
    as it is) and pins, newest pin first."""
    commits = []
    try:
        sources = models.read_json(installation / "model.json").get("sources")
    except models.ModelError:
        sources = None
    if isinstance(sources, dict):
        commits += [
            source.get("revision")
            for source in sources.values()
            if isinstance(source, dict) and source.get("repo") == repo_id
        ]
    pins = folder(repo_id) / "refs" / "splash" / pin_owner(installation)
    try:
        refs = sorted(
            pins.iterdir(), key=lambda ref: ref.stat().st_mtime_ns, reverse=True
        )
    except OSError:
        refs = []
    return commits + [ref.name for ref in refs]


def whole_snapshot(repo_id: str) -> Path | None:
    """The snapshot of repo_id the server finds in the Hub cache
    (serve_options.hub_snapshot: the one refs/main names, else the newest),
    if it is whole; else None. A draft is present when it is."""
    try:
        found = serve_options.hub_snapshot(repo_id)
    except ValueError:
        return None
    if found is None or not is_whole(Path(found)):
        return None
    return Path(found)


class Repository:
    """One source at one commit: a Hub repository at a commit, read from its
    snapshot in the Hub cache (directory set) or listed by the Hub
    (directory None), or a local directory (revision None)."""

    def __init__(self, name, revision, files, directory=None, *, sizes=None):
        self.name, self.revision, self.files = name, revision, files
        self.directory = directory
        # Hub filename -> (bytes, blob ID), to report what a download fetches.
        self.sizes = sizes or {}

    @classmethod
    def local_directory(cls, path):
        path = Path(path)
        if not path.is_dir():
            raise models.ModelError(f"local directory not found: {path}")
        return cls(str(path), None, _listing(path), path)

    @classmethod
    def cached(cls, name, commit):
        """name at commit from its snapshot in the Hub cache, without the Hub;
        only the files downloaded before are available."""
        path = snapshot(name, commit)
        if not path.is_dir():
            raise models.ModelError(f"the Hub cache has no snapshot {commit} of {name}")
        return cls(name, commit, _listing(path), path)

    @classmethod
    def resolve(cls, name, revision=None):
        """name at the commit revision names now, listed by the Hub with one
        request: what a start downloads when neither an installation nor the
        Hub cache holds it (cached_commit). Only an absolute path is a local
        directory; a target is always a Hub ID, whatever the working
        directory holds. A ModelError says why the Hub did not answer;
        HF_HUB_OFFLINE forbids the request."""
        import httpx
        from huggingface_hub import HfApi, constants

        if Path(name).is_absolute():
            return cls.local_directory(name)
        models.validate_repo_id(name)
        wanted = f"{name}@{revision}" if revision else name
        if constants.HF_HUB_OFFLINE:
            raise models.ModelError(
                f"{wanted} is neither installed nor in the Hugging Face cache, "
                "and HF_HUB_OFFLINE is set (splash-omni serve --offline); start "
                "without it to download the model"
            )
        try:
            info = HfApi().model_info(
                name, revision=revision, files_metadata=True, timeout=HUB_TIMEOUT
            )
        except (OSError, httpx.HTTPError) as error:
            raise models.ModelError(
                f"cannot download {wanted}, which is neither installed nor in "
                f"the Hugging Face cache: {reason(error)}"
            ) from error
        if not models.is_hex_digest(info.sha, 40):
            raise models.ModelError(f"the Hub did not resolve {wanted} to a commit")
        return cls(
            name,
            info.sha,
            {item.rfilename for item in info.siblings},
            sizes={
                item.rfilename: (
                    item.size,
                    item.lfs.sha256 if item.lfs else item.blob_id,
                )
                for item in info.siblings
                if item.size is not None
            },
        )

    def _require(self, name):
        if name not in self.files:
            raise models.ModelError(f"missing {name} in {self.name}")
        if not models.is_safe_path(name):
            raise models.ModelError(f"unsupported file name in {self.name}: {name}")

    def file(self, name):
        self._require(name)
        if self.directory is not None:
            return self.directory / name
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(self.name, name, revision=self.revision))

    def open(self, name):
        """A binary stream of one file read on demand: reading a safetensors
        header costs a few range requests, not a download."""
        self._require(name)
        if self.directory is not None:
            return (self.directory / name).open("rb")
        from huggingface_hub import HfFileSystem, try_to_load_from_cache

        cached = try_to_load_from_cache(self.name, name, revision=self.revision)
        if isinstance(cached, str):
            return open(cached, "rb")
        return HfFileSystem().open(
            f"{self.name}/{name}", "rb", revision=self.revision, block_size=8 << 20
        )

    def download(self, names):
        """Each of names -> its local path, downloading what the cache lacks."""
        for name in names:
            self._require(name)
        if self.directory is not None:
            return {name: self.directory / name for name in names}
        from huggingface_hub import snapshot_download

        blobs = folder(self.name) / "blobs"
        fetch = [
            size
            for size, blob in (self.sizes[n] for n in names if n in self.sizes)
            if not (blob and (blobs / blob).exists())
        ]
        if fetch:
            print(
                f"Fetching {len(fetch)} file(s), {sum(fetch) / 1e9:.2f} GB, "
                f"from {self.name}@{self.revision[:12]}; cached files are reused.",
                flush=True,
            )
        path = Path(
            snapshot_download(
                self.name,
                revision=self.revision,
                allow_patterns=sorted(names),
                max_workers=4,
            )
        )
        return {name: path / name for name in names}

    def identity(self):
        return {"repo": self.name, "revision": self.revision}


def _listing(directory):
    return {
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    }
