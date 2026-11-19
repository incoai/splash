"""How far a start's downloads have come, read from the Hugging Face cache.

The installer downloads (install/hub.py's Repository.download for the
checkpoint, upstream.fetch_draft for the draft); this module only looks. It
lists a repository's files and sizes as the installer does
(hub.Repository.resolve), and counts the bytes the cache holds of them: the
complete blobs, and the .incomplete files a download writes beside them.

Standard library only, with the installer modules that are too: without
huggingface_hub (before .venv exists) or the Hub (offline), sizes are
unknown and a repository is reported by its snapshot alone.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from install import hub

INCOMPLETE = ".incomplete"


def cache_folder(repo):
    """Where the Hub cache keeps repo (hub.folder)."""
    return hub.folder(repo)


def listing(repo):
    """repo's files at the commit its default branch names now, as {name:
    (bytes, blob ID)}; None when the Hub does not list them (offline,
    unreachable, or no huggingface_hub). Asked only of a repository the
    start downloads."""
    source = hub.Repository.resolve(repo)
    return dict(source.sizes) or None


def snapshot_complete(repo):
    """Whether the cache holds a whole snapshot of repo where the server
    finds one (hub.whole_snapshot)."""
    return hub.whole_snapshot(repo) is not None


def _blob_entries(folder):
    try:
        return list(os.scandir(Path(folder) / "blobs"))
    except OSError:
        return []


def present_bytes(folder, files):
    """The bytes of `files` ({name: (bytes, blob ID)}) the cache folder holds:
    each complete blob's size, else what its .incomplete files hold."""
    entries = _blob_entries(folder)
    partial = {}
    for entry in entries:
        if entry.name.endswith(INCOMPLETE):
            blob = entry.name.split(".", 1)[0]
            try:
                partial[blob] = partial.get(blob, 0) + entry.stat().st_size
            except OSError:
                pass
    names = {entry.name for entry in entries}
    present = 0
    for size, blob in files.values():
        if not blob:
            continue
        if blob in names and (Path(folder) / "blobs" / blob).exists():
            present += size
        else:
            present += min(partial.get(blob, 0), size)
    return present


def partial_bytes(folder):
    """What the .incomplete files in the cache folder hold."""
    total = 0
    for entry in _blob_entries(folder):
        if entry.name.endswith(INCOMPLETE):
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


class DownloadMonitor:
    """The downloads of one start: for each repository, its size and the
    bytes the cache holds, and whether the cache held it all before."""

    def __init__(
        self,
        repos,
        *,
        offline=False,
        listed=None,
        resolve=listing,
        folder=cache_folder,
        complete=snapshot_complete,
    ):
        self.repos = tuple(repos)
        self.offline = offline
        # The repositories the Hub is asked about (default: all): the ones
        # missing locally. A repository the machine has is never listed.
        self.listed = self.repos if listed is None else tuple(listed)
        self._resolve = resolve
        self._folder = folder
        self._complete = complete
        self._lock = threading.Lock()
        # repo -> {name: (bytes, blob)}, or None when unknown; absent while
        # it is listed.
        self._files = {}
        # repo -> the bytes the cache held when it was listed.
        self._initial = {}
        # repo -> whether the cache held a whole snapshot at the start.
        self._cached = {repo: self._complete(repo) for repo in self.repos}

    def start(self):
        threading.Thread(target=self.list, daemon=True).start()

    def list(self):
        """List each repository's files (one Hub request each)."""
        for repo in self.repos:
            files = None
            if not self.offline and repo in self.listed:
                try:
                    files = self._resolve(repo)
                except Exception:
                    # Offline, unreachable, refused, or no huggingface_hub:
                    # the snapshot alone tells.
                    files = None
            initial = (
                present_bytes(self._folder(repo), files) if files is not None else None
            )
            with self._lock:
                self._files[repo] = files
                self._initial[repo] = initial

    def stop(self):
        pass

    def snapshot(self):
        """Each repository as the page shows it: {repo, total, present,
        state}, state one of listing, cached, pending, downloading,
        complete or unknown (bytes are None when unknown)."""
        result = []
        for repo in self.repos:
            with self._lock:
                listed = repo in self._files
                files = self._files.get(repo)
                initial = self._initial.get(repo)
            folder = self._folder(repo)
            total = present = None
            if files is not None:
                total = sum(size for size, _ in files.values())
                present = present_bytes(folder, files)
                if initial is not None and initial >= total:
                    state = "cached"
                elif present >= total:
                    state = "complete"
                elif present > (initial or 0):
                    state = "downloading"
                else:
                    state = "pending"
            else:
                partial = partial_bytes(folder)
                if partial:
                    state, present = "downloading", partial
                elif self._cached[repo] or repo not in self.listed:
                    # A repository not listed is one the start found here:
                    # installed, or whole in the cache.
                    state = "cached"
                elif self._complete(repo):
                    state = "complete"
                else:
                    state = "listing" if not listed else "unknown"
            result.append(
                {"repo": repo, "total": total, "present": present, "state": state}
            )
        return result
