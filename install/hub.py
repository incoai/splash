"""The Hugging Face Hub and its cache, as the installers read them.

A source is one repository at one commit (Repository): listed by the Hub,
read from that commit's snapshot in the Hub cache, or a local draft
directory. An installation pins every snapshot it links with a ref of its
own, refs/splash/<installation>/<commit>, so neither pruning the cache nor
another installation can remove its files.

huggingface_hub is imported where it is used: the launcher imports the
installer modules before the environment that provides it exists, and the
tests replace its functions.
"""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from . import models

# Seconds the Hub may take to resolve a revision before the installed
# assembly starts without it.
HUB_TIMEOUT = 5


def token():
    """The token Hub requests carry: HF_TOKEN, else the saved `hf auth login`."""
    from huggingface_hub import get_token

    return os.environ.get("HF_TOKEN") or get_token()


def reason(error) -> str:
    """The Hub's reason for error on one line, with the configured token
    redacted and, for denied access, how to authenticate."""
    from huggingface_hub.errors import HfHubHTTPError

    message = " ".join(str(error).split()) or type(error).__name__
    if secret := token():
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


def folder(repo_id: str) -> Path:
    """Where the Hub cache keeps repo_id."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE) / folder_name(repo_id)


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
        raise models.ModelError(f"{path} is not a snapshot of {repo_id} in a Hub cache")
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


def retire_other_pins(pins, replaced=()):
    """Remove the installation's pins beside pins, and beside replaced, its
    pins before these, which may be in repositories pins do not name. Call
    it only after publishing and verifying the installation these pin; other
    installations own other folders."""
    try:
        for folder in {ref.parent for ref in (*pins, *replaced)}:
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


class Repository:
    """One source at one commit: a Hub repository at the commit its revision
    resolved to, listed by the Hub (directory None) or read from that
    commit's cached snapshot, or a local draft directory (revision None).
    The commit of a verified installation that starts without the Hub is
    not listed at all: that installation's links name its files."""

    def __init__(
        self,
        name,
        revision,
        files,
        directory=None,
        *,
        sizes=None,
        unreachable_reason=None,
    ):
        self.name, self.revision, self.files = name, revision, files
        self.directory = directory
        # Hub filename -> (bytes, blob ID), to report what a download fetches.
        self.sizes = sizes or {}
        # Why the Hub did not resolve this source, when its cached snapshot
        # stands in; None when no stand-in was needed.
        self.unreachable_reason = unreachable_reason

    @classmethod
    def local_directory(cls, path):
        path = Path(path)
        if not path.is_dir():
            raise models.ModelError(f"local draft directory not found: {path}")
        return cls(str(path), None, _listing(path), path)

    @classmethod
    def cached(cls, name, commit, *, unreachable_reason=None):
        """name at commit from its snapshot in the Hub cache, without the Hub;
        only the files downloaded before are available."""
        path = snapshot(name, commit)
        if not path.is_dir():
            raise models.ModelError(f"the Hub cache has no snapshot {commit} of {name}")
        return cls(
            name, commit, _listing(path), path, unreachable_reason=unreachable_reason
        )

    @classmethod
    def recorded(cls, source):
        """A source as an assembly record names it, without the Hub."""
        if source["revision"] is None:
            return cls.local_directory(source["repo"])
        return cls.cached(source["repo"], source["revision"])

    @classmethod
    def resolve(
        cls, name, revision=None, *, installation=None, installed=None, unreachable=None
    ):
        """name at the commit revision names now. This decides whether the
        Hub is asked.

        Only an absolute path is a local directory: parse_draft_model makes a
        --draft-model directory absolute, and a target is always a Hub ID,
        whatever the working directory holds. installed is the commit a
        verified installation of this selection records; its links name its
        files, wherever HF_HUB_CACHE points now. A commit revision never
        moves and HF_HUB_OFFLINE forbids requests, so with installed either
        returns that commit, unlisted, without a request or a look at the
        Hub cache. Otherwise one request resolves revision. When the Hub
        cannot answer, installed stands in the same way, with the reason in
        unreachable_reason; without it, the cached snapshot of a commit this
        selection already names does (_cached_commits). unreachable, why the
        Hub did not answer for another repository this start, stands for its
        answer without a request. A different revision is never
        substituted."""
        import httpx
        from huggingface_hub import HfApi, constants

        if Path(name).is_absolute():
            return cls.local_directory(name)
        models.validate_repo_id(name)
        if installed and (
            constants.HF_HUB_OFFLINE or models.is_hex_digest(revision, 40)
        ):
            return cls(name, installed, frozenset())
        if constants.HF_HUB_OFFLINE:
            why = "HF_HUB_OFFLINE is set"
        elif unreachable:
            why = unreachable
        else:
            try:
                info = HfApi().model_info(
                    name, revision=revision, files_metadata=True, timeout=HUB_TIMEOUT
                )
            except (OSError, httpx.HTTPError) as error:
                why = reason(error)
            else:
                if not models.is_hex_digest(info.sha, 40):
                    raise models.ModelError(
                        f"the Hub did not resolve {name} to a commit"
                    )
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
        if installed:
            return cls(name, installed, frozenset(), unreachable_reason=why)
        commits = _cached_commits(name, revision, installation)
        for commit in commits:
            if snapshot(name, commit).is_dir():
                return cls.cached(name, commit, unreachable_reason=why)
        cache = (
            "the Hub cache has no snapshot of " + ", ".join(commits)
            if commits
            else "neither this installation nor the Hub cache records a commit "
            f"for {revision or 'the default branch'}"
        )
        raise models.ModelError(f"cannot resolve {name}: {why}; {cache}")

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
        """A binary stream of one file read on demand: reading a GGUF header
        costs a few range requests, not a download."""
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


def _cached_commits(name, revision, installation):
    """The commits of name whose cached snapshots may stand in for revision
    without the Hub, most specific first: revision itself when it is a
    commit; the commit installation recorded, then the ones it pinned
    (refs/splash); then the commit the cache recorded for the branch or tag.
    Splash downloads by commit, which never records a branch. The record is
    read as it is: this is how a damaged installation is rebuilt."""
    if models.is_hex_digest(revision, 40):
        return [revision.lower()]
    commits = []
    if installation is not None:
        try:
            sources = models.read_json(installation / "model.json").get("sources")
        except models.ModelError:
            sources = None
        if isinstance(sources, dict):
            commits += [
                source.get("revision")
                for source in sources.values()
                if isinstance(source, dict) and source.get("repo") == name
            ]
        pins = folder(name) / "refs" / "splash" / pin_owner(installation)
        if pins.is_dir():
            commits += [
                ref.name
                for ref in sorted(
                    pins.iterdir(), key=lambda ref: ref.stat().st_mtime_ns, reverse=True
                )
            ]
    try:
        commits.append(
            (folder(name) / "refs" / (revision or "main")).read_text().strip()
        )
    except OSError:
        pass
    return list(dict.fromkeys(c for c in commits if models.is_hex_digest(c, 40)))
