"""The Hugging Face Hub and its cache, as the installers read them.

A source is one repository at one commit (Repository): listed by the Hub,
read from that commit's snapshot in the Hub cache, or local files. An
installation pins every snapshot it links with a ref of its
own, refs/splash/<installation>/<commit>, so neither pruning the cache nor
another installation can remove its files. The commit the Hub names for a
default branch is recorded as refs/main, as huggingface_hub records a branch
it downloads, for installs without the Hub.

huggingface_hub is imported where it is used: the launcher imports the
installer modules before the environment that provides it exists, and the
tests replace its functions.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

from . import models

# Seconds the Hub may take to resolve a revision before the installed
# assembly starts without it.
HUB_TIMEOUT = 5
# Seconds between two measurements of a download's progress.
PROGRESS_INTERVAL = 0.25
# Seconds a download's caller waits on it at a time, the longest Ctrl+C
# takes to reach that caller (_interruptible).
INTERRUPT_INTERVAL = 0.1

# Whether Repository.download reports its progress as JSON lines on stdout,
# for a front end (report_json_progress).
json_progress = False


def report_json_progress():
    """Have every download report its progress as JSON lines on stdout
    (_download_progress), as `install/models.py ... prepare --json-progress`
    asks, and huggingface_hub draw no progress bars: their carriage returns
    would run into those lines and the installer's own."""
    global json_progress
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    json_progress = True


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


def failure_kind(error) -> str | None:
    """What the app words a Hub or network failure by
    (models.ModelError.kind): no such repository, no connection or access
    denied; None for any other, which the app shows by its message, such as
    a revision or file the repository lacks. A gated repository is access
    denied, though huggingface_hub refuses it as a RepositoryNotFoundError
    (GatedRepoError)."""
    import httpx
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    status = (
        error.response.status_code
        if isinstance(error, HfHubHTTPError) and error.response is not None
        else None
    )
    if isinstance(error, GatedRepoError):
        return "auth"
    if isinstance(error, RepositoryNotFoundError):
        return "not_found"
    if status in (401, 403):
        return "auth"
    if isinstance(error, (httpx.ConnectError, httpx.TimeoutException)):
        return "offline"
    return None


@contextmanager
def as_model_errors(context: str):
    """Report a Hub, network or cache failure in the block as a ModelError,
    "context: reason", which callers print without a traceback, classed for
    the app (failure_kind)."""
    import httpx

    try:
        yield
    except (OSError, httpx.HTTPError) as error:
        raise models.ModelError(
            f"{context}: {reason(error)}", kind=failure_kind(error)
        ) from error


def folder_name(repo_id: str) -> str:
    """The folder of a model repository in any Hub cache."""
    return "models--" + repo_id.replace("/", "--")


def folder(repo_id: str) -> Path:
    """Where the Hub cache keeps repo_id."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE) / folder_name(repo_id)


def cached_repositories() -> list[str]:
    """The IDs of the model repositories the Hub cache holds."""
    from huggingface_hub import constants

    return sorted(
        path.name.removeprefix("models--").replace("--", "/")
        for path in Path(constants.HF_HUB_CACHE).glob("models--*")
    )


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
    # A pin's content is its own name, so writers never disagree.
    _publish(ref, commit)
    return ref


def retire_other_pins(pins, replaced=()):
    """Remove the installation's pins beside pins, and beside replaced, its
    pins before these, which may be in repositories pins do not name. Call
    it only after publishing and verifying the installation these pin; other
    installations own other folders. replaced may name a cache by another
    path than pins do, so pins are told apart as files (_file)."""
    try:
        kept = {_file(ref) for ref in pins}
        for folder in {ref.parent for ref in (*pins, *replaced)}:
            for previous in folder.iterdir() if folder.is_dir() else ():
                if (
                    models.is_hex_digest(previous.name, 40)
                    and _file(previous) not in kept
                ):
                    previous.unlink()
    except OSError as error:
        # Keeping an old pin uses cache space but cannot invalidate the
        # verified installation or its successfully retained current pin.
        models.warn(f"could not retire old Hub cache references: {error}")


def _file(path):
    """The entry at path, however the path spells it (through a symbolic link
    to a folder, or in another case on a case-insensitive volume): its device
    and inode. A symbolic link at path is itself, not the file it names, so
    one that names nothing is no pin rather than an error."""
    status = path.lstat()
    return status.st_dev, status.st_ino


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
    resolved to, listed by the Hub (paths None) or read from that commit's
    cached snapshot, or local files (revision None): a draft directory, or
    the files a model is prepared from (local.py). The commit of a verified
    installation that starts without the Hub is not listed at all: that
    installation's links name its files."""

    def __init__(
        self,
        name,
        revision,
        files,
        paths=None,
        *,
        sizes=None,
        unreachable_reason=None,
        unreachable_kind=None,
        default_branch=False,
    ):
        self.name, self.revision, self.files = name, revision, files
        # Whether revision is the commit the Hub named for the default
        # branch, which a download records (_record_default_branch).
        self.default_branch = default_branch
        # Each file's local path, for a cached snapshot or local files.
        self.paths = paths
        # Hub filename -> (bytes, blob ID), to report what a download fetches.
        self.sizes = sizes or {}
        # Why the Hub did not resolve this source, when its cached snapshot
        # stands in, and that failure's class (failure_kind); None when no
        # stand-in was needed.
        self.unreachable_reason = unreachable_reason
        self.unreachable_kind = unreachable_kind

    @classmethod
    def local_directory(cls, path):
        path = Path(path)
        if not path.is_dir():
            raise models.ModelError(f"local draft directory not found: {path}")
        return cls.local_files(str(path), _paths(path))

    @classmethod
    def local_files(cls, name, paths):
        """name, the local files paths maps each file name to, wherever each
        lies."""
        return cls(name, None, set(paths), dict(paths))

    @classmethod
    def cached(
        cls, name, commit, path=None, *, unreachable_reason=None, unreachable_kind=None
    ):
        """name at commit from its snapshot folder in a Hub cache, path, by
        default HF_HUB_CACHE's, without the Hub; only the files downloaded
        before are available."""
        path = path or snapshot(name, commit)
        if not path.is_dir():
            raise models.ModelError(f"the Hub cache has no snapshot {commit} of {name}")
        paths = _paths(path)
        return cls(
            name,
            commit,
            set(paths),
            paths,
            unreachable_reason=unreachable_reason,
            unreachable_kind=unreachable_kind,
        )

    @classmethod
    def resolve(
        cls,
        name,
        revision=None,
        *,
        installation=None,
        installed=None,
        unreachable=None,
        unreachable_kind=None,
    ):
        """name at the commit revision names now; this decides whether the
        Hub is asked.

        Only an absolute path is a local directory (parse_draft_model makes
        --draft-model absolute; a target is always a Hub ID). installed is the
        commit a verified installation records, whose links name its files
        wherever HF_HUB_CACHE points now; with it, a commit revision (which
        never moves) or HF_HUB_OFFLINE returns that commit, unlisted, without a
        request or a look at the Hub cache. Otherwise one request resolves
        revision, and the default branch's commit is recorded once the cache
        holds it (_record_default_branch). When the Hub cannot answer,
        installed stands in the same way, with unreachable_reason and its class
        (unreachable_kind, failure_kind); without it, a cached snapshot does
        (_cached_commits); without either, the ModelError carries that class
        for the app. unreachable and unreachable_kind, from another repository
        this start, stand for the Hub's answer without a request. A different
        revision is never substituted."""
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
            # Not a network failure: the reason, which the app shows as it
            # is, names the setting.
            why, kind = "HF_HUB_OFFLINE is set", None
        elif unreachable:
            why, kind = unreachable, unreachable_kind
        else:
            try:
                info = HfApi().model_info(
                    name, revision=revision, files_metadata=True, timeout=HUB_TIMEOUT
                )
            except (OSError, httpx.HTTPError) as error:
                why, kind = reason(error), failure_kind(error)
            else:
                if not models.is_hex_digest(info.sha, 40):
                    raise models.ModelError(
                        f"the Hub did not resolve {name} to a commit"
                    )
                if revision is None:
                    _record_default_branch(name, info.sha)
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
                    default_branch=revision is None,
                )
        if installed:
            return cls(
                name,
                installed,
                frozenset(),
                unreachable_reason=why,
                unreachable_kind=kind,
            )
        commits = _cached_commits(name, revision, installation)
        for commit in commits:
            if snapshot(name, commit).is_dir():
                return cls.cached(
                    name, commit, unreachable_reason=why, unreachable_kind=kind
                )
        branch = revision or "the default branch"
        if commits:
            cache = "the Hub cache has no snapshot of " + ", ".join(commits)
        elif installation is not None:
            cache = (
                "neither this installation nor the Hub cache records a commit "
                f"for {branch}"
            )
        else:
            cache = f"the Hub cache records no commit for {branch}"
        raise models.ModelError(f"cannot resolve {name}: {why}; {cache}", kind=kind)

    def _require(self, name):
        if name not in self.files:
            raise models.ModelError(f"missing {name} in {self.name}")
        if not models.is_safe_path(name):
            raise models.ModelError(f"unsupported file name in {self.name}: {name}")

    def file(self, name):
        self._require(name)
        if self.paths is not None:
            return self.paths[name]
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(self.name, name, revision=self.revision))

    def open(self, name):
        """A binary stream of one file read on demand: reading a GGUF header
        costs a few range requests, not a download."""
        self._require(name)
        if self.paths is not None:
            return self.paths[name].open("rb")
        from huggingface_hub import HfFileSystem, try_to_load_from_cache

        cached = try_to_load_from_cache(self.name, name, revision=self.revision)
        if isinstance(cached, str):
            return open(cached, "rb")
        return HfFileSystem().open(
            f"{self.name}/{name}", "rb", revision=self.revision, block_size=8 << 20
        )

    def download(self, names):
        """Each of names -> its local path, downloading what the cache lacks.
        Before a download, the partial files killed ones left in the
        repository's blobs folder are removed. Ctrl+C stops the wait for it
        at once (_interruptible)."""
        for name in names:
            self._require(name)
        if self.paths is not None:
            return {name: self.paths[name] for name in names}
        from huggingface_hub import snapshot_download

        blobs = folder(self.name) / "blobs"
        # (bytes, blob ID) of each file whose blob the cache lacks.
        fetch = [
            (size, blob)
            for size, blob in (self.sizes[n] for n in names if n in self.sizes)
            if not (blob and (blobs / blob).exists())
        ]
        total = sum(size for size, _ in fetch)
        if fetch:
            _remove_abandoned_downloads(blobs)
            _require_room(self.name, blobs, total)
            print(
                f"Fetching {len(fetch)} file(s), {total / 1e9:.2f} GB, "
                f"from {self.name}@{self.revision[:12]}; cached files are reused.",
                flush=True,
            )
        with _download_progress(self.name, blobs, fetch) as counted:
            path = Path(
                _interruptible(
                    snapshot_download,
                    self.name,
                    revision=self.revision,
                    allow_patterns=sorted(names),
                    max_workers=4,
                    tqdm_class=counted,
                )
            )
        if self.default_branch:
            _record_default_branch(self.name, self.revision)
        return {name: path / name for name in names}

    def identity(self):
        return {"repo": self.name, "revision": self.revision}


def _paths(directory):
    """Each file under directory, by its name relative to it, -> its path."""
    return {
        p.relative_to(directory).as_posix(): p
        for p in directory.rglob("*")
        if p.is_file()
    }


def _cached_commits(name, revision, installation):
    """The commits of name whose cached snapshots may stand in for revision
    without the Hub, most specific first: revision itself when it is a
    commit; the commit installation recorded, then the ones it pinned
    (refs/splash); then the cache's record of the branch or tag (refs/main,
    _record_default_branch). Other installations' pins never stand in: a pin
    does not say which revision it was installed for. The record is read as
    it is, so a damaged installation can be rebuilt."""
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


def _record_default_branch(name, commit):
    """Record commit as name's default branch in the Hub cache, refs/main, as
    huggingface_hub does when it downloads the branch (Splash downloads by
    commit, which records no branch), so a new selection of it installs without
    the Hub (_cached_commits). It is recorded only once the cache holds the
    commit's snapshot, so no record names what the cache lacks, and written
    only when it changes, so a read-only cache is left alone."""
    if not snapshot(name, commit).is_dir():
        return
    ref = folder(name) / "refs" / "main"
    try:
        if not ref.is_file() or ref.read_text().strip() != commit:
            _publish(ref, commit)
    except OSError:
        # A cache this process cannot write keeps no record; the resolve
        # stands, and only a new selection of the branch needs the Hub then.
        pass


def _publish(path, content):
    """Write content to path by rename, which file systems without hard links
    (exFAT) support too, so a reader sees all of it or none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        pending = Path(temporary.name)
        try:
            # Readable by every account, as huggingface_hub writes its refs:
            # a cache may be shared.
            os.fchmod(temporary.fileno(), 0o644)
            temporary.write(content.encode())
            temporary.flush()
            os.fsync(temporary.fileno())
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)


def _require_room(repo_id, blobs, needed):
    """Refuse a download of needed bytes into the blobs folder that its volume
    has no room for, before it fills the disk: the Hub cache can be on
    another volume (HF_HOME, HF_HUB_CACHE) than the home folder."""
    existing = next(p for p in (blobs, *blobs.parents) if p.exists())
    available = _available_bytes(existing)
    if needed > available:
        raise models.ModelError(
            f"not enough disk space for {repo_id}: the download needs "
            f"{needed / 1e9:.2f} GB and {available / 1e9:.2f} GB is available "
            f"in {blobs.parent.parent}"
        )


def _available_bytes(path):
    """The bytes the volume of path takes for a download the user asked for,
    as Finder counts them: with the purgeable space macOS frees as it is
    needed, which statvfs leaves out (6.6 GB of 53 GB on one Mac). That count
    reads 0 off volumes Finder does not show (mounted nobrowse), so the larger
    of it and statvfs's free bytes decides."""
    import ctypes
    import shutil

    cf = ctypes.CDLL(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )
    cf.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
    cf.CFURLCreateFromFileSystemRepresentation.argtypes = (
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_long,
        ctypes.c_bool,
    )
    cf.CFURLCopyResourcePropertyForKey.restype = ctypes.c_bool
    cf.CFURLCopyResourcePropertyForKey.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    )
    cf.CFNumberGetValue.restype = ctypes.c_bool
    cf.CFNumberGetValue.argtypes = (
        ctypes.c_void_p,
        ctypes.c_long,
        ctypes.POINTER(ctypes.c_int64),
    )
    cf.CFRelease.argtypes = (ctypes.c_void_p,)
    key = ctypes.c_void_p.in_dll(
        cf, "kCFURLVolumeAvailableCapacityForImportantUsageKey"
    )
    important = ctypes.c_int64()
    name = os.fsencode(path)
    url = cf.CFURLCreateFromFileSystemRepresentation(None, name, len(name), True)
    if url:
        value = ctypes.c_void_p()
        if (
            cf.CFURLCopyResourcePropertyForKey(url, key, ctypes.byref(value), None)
            and value
        ):
            sint64 = 4  # kCFNumberSInt64Type
            cf.CFNumberGetValue(value, sint64, ctypes.byref(important))
            cf.CFRelease(value)
        cf.CFRelease(url)
    return max(important.value, shutil.disk_usage(path).free)


def _remove_abandoned_downloads(blobs):
    """Remove the partial files in the blobs folder that no live download
    writes. huggingface_hub writes a blob to <blob>.<random>.incomplete, a
    name no later download reuses, and holds <Hub cache>/.locks/<repository
    folder>/<blob>.lock until it renames or removes that file, so a killed
    download leaves it behind. A blob's partial files are removed only while
    that lock is taken here without waiting, as huggingface_hub takes it, so
    a live download of any process keeps its own and none starts
    meanwhile."""
    from filelock import FileLock, Timeout

    locks = blobs.parent.parent / ".locks" / blobs.parent.name
    partial = {}
    for path in blobs.glob("*.*.incomplete"):
        partial.setdefault(path.name.split(".", 1)[0], []).append(path)
    for blob, paths in partial.items():
        try:
            # Created group-writable when missing, as huggingface_hub does.
            with FileLock(locks / f"{blob}.lock", mode=0o664, blocking=False):
                for path in paths:
                    path.unlink(missing_ok=True)
        except Timeout:
            pass  # a live download holds it


def _interruptible(call, *args, **kwargs):
    """call(*args, **kwargs) on a thread of its own: its result returned
    here, or its exception raised. This thread joins it every
    INTERRUPT_INTERVAL, so Ctrl+C raises KeyboardInterrupt here at once;
    interrupted in its own thread, snapshot_download would first let each
    pool thread finish its file, tens of seconds. The interrupted call goes
    on, and install/models.py ends the installer without the exit-time join
    of its pool (concurrent.futures joins every pool thread at exit, daemon
    or not)."""
    outcome = {}

    def run():
        try:
            outcome["result"] = call(*args, **kwargs)
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    while thread.is_alive():
        thread.join(INTERRUPT_INTERVAL)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


@contextmanager
def _download_progress(repo_id, blobs, fetch):
    """With json_progress, report the download in the block of fetch,
    (bytes, blob ID) pairs in the blobs folder, as JSON lines on stdout,
    and yield the tqdm class the block's snapshot_download counts with (None
    without json_progress): done 0 first, then, measured every
    PROGRESS_INTERVAL, whenever it grew, the bytes received, and done == total
    once the block succeeds. Received is what the network has brought or what
    the folder holds of fetch, whichever is more, and at most the total: under
    Xet a file is written in order from chunks that arrive out of order and
    wait in memory, so the folder alone can stay at 0 for minutes, then jump."""
    if not (json_progress and fetch):
        yield None
        return
    total = sum(size for size, _ in fetch)
    counted = _network_counter()
    stop = threading.Event()

    def measure():
        done = 0
        while not stop.wait(PROGRESS_INTERVAL):
            held = _blob_bytes(blobs)
            written = sum(min(size, held.get(blob, 0)) for size, blob in fetch)
            now = min(max(written, counted.received), total)
            # A partial file renamed to its blob between the listing and its
            # size goes uncounted once, so only growth is reported.
            if now > done:
                done = now
                _report_progress(repo_id, done, total)

    _report_progress(repo_id, 0, total)
    thread = threading.Thread(target=measure)
    thread.start()
    try:
        yield counted
    finally:
        stop.set()
        thread.join()
    _report_progress(repo_id, total, total)


# The progress bar snapshot_download counts the bytes from the network on, as
# huggingface_hub names it.
NETWORK_BAR = "huggingface_hub.snapshot_download.transfer"


def _network_counter():
    """A tqdm class for snapshot_download that counts in `received` the bytes
    its network bar is told have arrived; json_progress has its bars draw
    nothing."""
    from huggingface_hub.utils import tqdm

    lock = threading.Lock()

    class Counted(tqdm):
        received = 0

        def __init__(self, *args, **kwargs):
            self.network = kwargs.get("name") == NETWORK_BAR
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            if self.network and n:
                with lock:
                    Counted.received += n
            return super().update(n)

    return Counted


def _blob_bytes(blobs):
    """Each blob in the blobs folder -> the most bytes a file of it holds,
    <blob> itself or a partial <blob>.<random>.incomplete; none before the
    folder exists."""
    try:
        entries = list(os.scandir(blobs))
    except FileNotFoundError:
        return {}
    held = {}
    for entry in entries:
        try:
            size = entry.stat().st_size
        except FileNotFoundError:
            continue  # renamed to its blob or removed since the listing
        blob = entry.name.partition(".")[0]
        held[blob] = max(held.get(blob, 0), size)
    return held


def _report_progress(repo_id, done, total):
    # One write per line, newline included, then a flush, so a front end
    # reading stdout, or the log it shares with stderr, sees whole lines.
    event = {"event": "download", "repo": repo_id, "done": done, "total": total}
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()
