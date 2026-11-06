import _thread
import contextlib
import errno
import fcntl
import hashlib
import inspect
import io
import json
import logging
import os
import shutil
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from install import hub

REPO = "owner/model"
COMMIT = "c" * 40
# The repository's files: config.json is in the Hub cache before each test.
FILES = {
    "config.json": b"{}",
    "model-1.safetensors": b"1" * 300,
    "model-2.safetensors": b"2" * 200,
}
WEIGHTS = {"model-1.safetensors", "model-2.safetensors"}
# The bytes huggingface_hub appends to a partial file at a time, here.
CHUNK = 100


def blob(name):
    return hashlib.sha256(FILES[name]).hexdigest()


class Stdout(io.StringIO):
    """stdout recording each write, and each flush as None."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def write(self, text):
        self.calls.append(text)
        return super().write(text)

    def flush(self):
        self.calls.append(None)
        super().flush()


class DownloadTest(unittest.TestCase):
    """Repository.download into the Hub cache as huggingface_hub fills it:
    each blob written to a partial file of its own, then renamed. The
    partial files a killed download left, and the progress it reports."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name) / "hub"
        repository = self.cache / hub.folder_name(REPO)
        self.blobs = repository / "blobs"
        self.blobs.mkdir(parents=True)
        (self.blobs / blob("config.json")).write_bytes(FILES["config.json"])
        self.locks = self.cache / ".locks" / repository.name
        self.snapshot = repository / "snapshots" / COMMIT
        self.stdout = Stdout()
        self.downloads = []
        # Called with the bytes downloaded so far after each chunk.
        self.after_chunk = lambda done: None
        # Whether files come as Xet brings them: chunks counted on the network
        # bar as they arrive, the file written whole once all have; and the
        # bytes each chunk costs the network beyond its own.
        self.xet = False
        self.overhead = 0
        # As report_json_progress has them, the bars draw nothing.
        from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

        disable_progress_bars()
        self.addCleanup(enable_progress_bars)
        for name, replacement in (
            ("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache)),
            ("huggingface_hub.snapshot_download", self.snapshot_download),
            ("install.hub.PROGRESS_INTERVAL", 0.001),
        ):
            patch = mock.patch(name, replacement)
            patch.start()
            self.addCleanup(patch.stop)

    def snapshot_download(
        self, repo_id, *, revision, allow_patterns, max_workers, tqdm_class=None
    ):
        """Each file's blob, appended to <blob>.<random>.incomplete a chunk at
        a time, then renamed, and linked from the snapshot; or, with xet, its
        chunks counted on the network bar, then the file written whole."""
        from huggingface_hub.utils import tqdm
        from huggingface_hub.utils.tqdm import _create_progress_bar

        self.downloads.append(sorted(allow_patterns))
        # As huggingface_hub makes it, with the caller's class.
        network = _create_progress_bar(
            cls=tqdm_class or tqdm,
            log_level=logging.WARNING,
            name=hub.NETWORK_BAR,
            total=0,
            unit="B",
        )
        done = 0
        for name in allow_patterns:
            target = self.blobs / blob(name)
            if not target.exists():
                partial = self.blobs / f"{blob(name)}.0123abcd.incomplete"
                with partial.open("wb") as file:
                    for start in range(0, len(FILES[name]), CHUNK):
                        if self.xet:
                            network.update(CHUNK + self.overhead)
                        else:
                            file.write(FILES[name][start : start + CHUNK])
                            file.flush()
                        done += CHUNK
                        self.after_chunk(done)
                    if self.xet:
                        file.write(FILES[name])
                partial.rename(target)
            self.snapshot.mkdir(parents=True, exist_ok=True)
            (self.snapshot / name).symlink_to(target)
        return str(self.snapshot)

    @staticmethod
    def repository(listed=None):
        """REPO at COMMIT as the Hub lists it, listed giving some files'
        bytes in place of their contents'."""
        listed = listed or {}
        return hub.Repository(
            REPO,
            COMMIT,
            set(FILES),
            sizes={
                name: (listed.get(name, len(content)), blob(name))
                for name, content in FILES.items()
            },
        )

    def download(self, names=WEIGHTS, *, json_progress=True, listed=None):
        with (
            mock.patch.object(hub, "json_progress", json_progress),
            contextlib.redirect_stdout(self.stdout),
        ):
            return self.repository(listed).download(names)

    def events(self):
        return [
            json.loads(line)
            for line in self.stdout.getvalue().splitlines()
            if line.startswith("{")
        ]

    def wait_for_report(self, done):
        deadline = time.monotonic() + 10
        while done not in [event["done"] for event in self.events()]:
            self.assertLess(time.monotonic(), deadline, f"{done} was not reported")
            time.sleep(0.001)

    def test_a_download_reports_the_bytes_of_the_blobs_it_fetches(self):
        self.after_chunk = self.wait_for_report
        paths = self.download(WEIGHTS | {"config.json"})
        self.assertEqual(paths["config.json"], self.snapshot / "config.json")
        events = self.events()
        # Only the blobs the cache lacked count: 300 + 200 bytes.
        self.assertEqual(
            events[0], {"event": "download", "repo": REPO, "done": 0, "total": 500}
        )
        self.assertEqual({event["total"] for event in events}, {500})
        dones = [event["done"] for event in events]
        self.assertEqual(dones, sorted(dones))
        self.assertEqual(set(dones), {0, 100, 200, 300, 400, 500})
        self.assertEqual(dones[-1], 500)
        # Each event is one write of its whole line, flushed at once.
        for index, call in enumerate(self.stdout.calls):
            if call and call.startswith("{"):
                self.assertTrue(call.endswith("}\n"), call)
                self.assertIsNone(self.stdout.calls[index + 1])
        # Beside the line people read, unchanged.
        self.assertIn(
            f"Fetching 2 file(s), 0.00 GB, from {REPO}@{'c' * 12}; "
            "cached files are reused.\n",
            self.stdout.getvalue(),
        )

    def test_a_xet_download_reports_the_bytes_the_network_brought(self):
        # The folder holds nothing of a file until all its chunks have come.
        self.xet = True
        self.after_chunk = self.wait_for_report
        self.download()
        dones = [event["done"] for event in self.events()]
        self.assertEqual(dones, sorted(dones))
        self.assertEqual(set(dones), {0, 100, 200, 300, 400, 500})

    def test_network_bytes_beyond_the_files_count_no_more_than_the_total(self):
        self.xet, self.overhead = True, 60
        self.after_chunk = lambda done: done < 500 or self.wait_for_report(500)
        self.download()
        dones = [event["done"] for event in self.events()]
        self.assertEqual(dones, sorted(dones))
        self.assertEqual(max(dones), 500)
        self.assertEqual(dones[-2:], [500, 500])

    def test_the_pinned_huggingface_hub_names_its_network_bar_so(self):
        from huggingface_hub import _snapshot_download

        self.assertIn(
            f'name="{hub.NETWORK_BAR}"', inspect.getsource(_snapshot_download)
        )

    def test_a_successful_download_reports_its_completion(self):
        # Measured too seldom to report, the download still reports its end.
        with mock.patch.object(hub, "PROGRESS_INTERVAL", 3600):
            self.download()
        self.assertEqual([event["done"] for event in self.events()], [0, 500])

    def test_progress_is_reported_only_when_asked(self):
        self.download(json_progress=False)
        self.assertEqual(self.events(), [])
        self.assertIn("Fetching 2 file(s)", self.stdout.getvalue())

    def test_a_download_of_cached_blobs_reports_nothing(self):
        self.download({"config.json"})
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertEqual(self.downloads, [["config.json"]])

    def test_a_download_the_cache_volume_has_no_room_for_is_refused(self):
        """Measured on the Hub cache's own volume, before anything is
        fetched, so the disk is never filled; a model's first download, whose
        folder is not there yet, measures the folder it goes in."""
        shutil.rmtree(self.blobs.parent)
        with (
            mock.patch.object(
                hub, "_available_bytes", wraps=hub._available_bytes
            ) as measured,
            self.assertRaises(hub.models.ModelError),
        ):
            self.download(listed={"model-1.safetensors": 1 << 60})
        volume = measured.call_args.args[0]
        self.assertIn(self.cache, [volume, *volume.parents])
        self.assertEqual(self.downloads, [])
        self.assertEqual(self.events(), [])
        self.blobs.mkdir(parents=True)
        with (
            mock.patch.object(hub, "_available_bytes", lambda path: 499),
            self.assertRaisesRegex(
                hub.models.ModelError, "needs 0.00 GB and 0.00 GB is available"
            ),
        ):
            self.download()
        self.assertEqual(self.downloads, [])
        with mock.patch.object(hub, "_available_bytes", lambda path: 500):
            self.download()
        self.assertEqual(self.downloads, [sorted(WEIGHTS)])

    def test_a_failed_download_reports_no_completion(self):
        def fail(done):
            self.wait_for_report(done)
            raise OSError("connection reset")

        self.after_chunk = fail
        threads = set(threading.enumerate())
        with self.assertRaisesRegex(OSError, "connection reset"):
            self.download()
        self.assertEqual([event["done"] for event in self.events()], [0, 100])
        # The measuring thread is stopped.
        self.assertEqual(set(threading.enumerate()), threads)

    def test_the_download_thread_returns_or_raises_to_the_caller(self):
        downloading = []
        error = OSError("connection reset")

        def fail(done):
            downloading.append(threading.current_thread())
            raise error

        self.after_chunk = fail
        with self.assertRaises(OSError) as raised:
            self.download()
        self.assertIs(raised.exception, error)
        self.after_chunk = lambda done: downloading.append(threading.current_thread())
        self.assertEqual(
            self.download(), {name: self.snapshot / name for name in WEIGHTS}
        )
        self.assertNotIn(threading.current_thread(), downloading)

    def test_ctrl_c_stops_the_wait_for_a_download_at_once(self):
        # Interrupted, huggingface_hub would first wait for each thread of its
        # pool to finish its file; this download goes on until released.
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        downloading, interrupted = [], []

        def snapshot_download(repo_id, **options):
            downloading.append(threading.current_thread())
            started.set()
            release.wait(60)
            return str(self.snapshot)

        def interrupt():
            started.wait()
            interrupted.append(time.monotonic())
            _thread.interrupt_main()

        # A test run as a background job inherits SIGINT ignored, which
        # interrupt_main would not raise.
        self.addCleanup(signal.signal, signal.SIGINT, signal.getsignal(signal.SIGINT))
        signal.signal(signal.SIGINT, signal.default_int_handler)
        threads = set(threading.enumerate())
        interrupter = threading.Thread(target=interrupt)
        interrupter.start()
        with (
            mock.patch("huggingface_hub.snapshot_download", snapshot_download),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.download()
        self.assertLess(time.monotonic() - interrupted[0], 2)
        interrupter.join()
        # Only the download goes on: the measuring thread is stopped, and the
        # download reports no completion.
        self.assertEqual(set(threading.enumerate()) - threads, set(downloading))
        self.assertEqual([event["done"] for event in self.events()], [0])
        release.set()
        downloading[0].join()

    def test_partial_files_a_killed_download_left_are_removed(self):
        killed = self.blobs / f"{blob('model-1.safetensors')}.deadbeef.incomplete"
        killed.write_bytes(b"1" * 150)
        # Of a blob this download does not fetch, as a cancelled download of
        # another file of the repository leaves.
        other = self.blobs / f"{'e' * 64}.deadbeef.incomplete"
        other.write_bytes(b"e")
        self.after_chunk = lambda done: self.assertFalse(killed.exists())
        self.download()
        self.assertFalse(killed.exists())
        self.assertFalse(other.exists())

    def test_a_partial_file_whose_download_holds_its_lock_is_kept(self):
        # Other processes downloading a blob this download fetches and one it
        # does not, as huggingface_hub locks them.
        self.locks.mkdir(parents=True)
        live, locks = [], []
        for held in (blob("model-1.safetensors"), "e" * 64):
            live.append(self.blobs / f"{held}.feedface.incomplete")
            live[-1].write_bytes(b"1" * 150)
            locks.append(os.open(self.locks / f"{held}.lock", os.O_CREAT))
            self.addCleanup(os.close, locks[-1])
            fcntl.flock(locks[-1], fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.download()
        self.assertEqual([path.exists() for path in live], [True, True])
        # Once those downloads end, a later one removes their partial files.
        for lock in locks:
            fcntl.flock(lock, fcntl.LOCK_UN)
        for name in WEIGHTS:
            (self.blobs / blob(name)).unlink()
            (self.snapshot / name).unlink()
        self.download()
        self.assertEqual([path.exists() for path in live], [False, False])


class PinTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "hub"
        self.installation = self.root / "models" / "model"

    def snapshot(self, cache, commit):
        return cache / hub.folder_name(REPO) / "snapshots" / commit

    def test_retiring_keeps_the_new_pin_however_its_cache_is_spelled(self):
        """The assembly a new one replaces may name the same Hub cache by
        another path, here a symbolic link to it."""
        alias = self.root / "alias"
        alias.symlink_to(self.cache)
        old = hub.pin(self.snapshot(self.cache, "b" * 40), REPO, self.installation)
        new = hub.pin(self.snapshot(self.cache, COMMIT), REPO, self.installation)
        hub.retire_other_pins(
            [new], [hub.pinned(self.snapshot(alias, "b" * 40), self.installation)]
        )
        self.assertEqual([old.exists(), new.exists()], [False, True])

    def test_retiring_removes_a_pin_name_that_links_nothing(self):
        """A symbolic link named like a pin is not the new pin, even when the
        file it named is gone."""
        new = hub.pin(self.snapshot(self.cache, COMMIT), REPO, self.installation)
        dangling = new.parent / ("b" * 40)
        dangling.symlink_to(self.root / "removed")
        hub.retire_other_pins([new])
        self.assertEqual([os.path.lexists(dangling), new.exists()], [False, True])


class FailureKindTest(unittest.TestCase):
    def test_hub_failures_carry_their_kind(self):
        """A Hub or network failure says which, as the app words it. A gated
        repository exists, though huggingface_hub refuses it as a
        RepositoryNotFoundError, and a revision or file the Hub lacks is not
        a missing repository: the message names it."""
        import httpx
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            RemoteEntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )

        request = httpx.Request("GET", f"https://huggingface.co/api/models/{REPO}")

        def refused(status, kind=HfHubHTTPError):
            return kind("refused", response=httpx.Response(status, request=request))

        for error, kind in (
            (refused(401, RepositoryNotFoundError), "not_found"),
            (refused(404, RepositoryNotFoundError), "not_found"),
            (refused(401, GatedRepoError), "auth"),
            (refused(403, GatedRepoError), "auth"),
            (refused(404, RevisionNotFoundError), None),
            (refused(404, RemoteEntryNotFoundError), None),
            (refused(404), None),
            (refused(401), "auth"),
            (refused(403), "auth"),
            (httpx.ConnectError("no route", request=request), "offline"),
            (httpx.ReadTimeout("slow", request=request), "offline"),
            (OSError(errno.EACCES, "Permission denied"), None),
            (refused(500), None),
        ):
            with self.subTest(error=error):
                with self.assertRaises(hub.models.ModelError) as caught:
                    with hub.as_model_errors(f"cannot install {REPO}"):
                        raise error
                self.assertEqual(caught.exception.kind, kind)
                self.assertTrue(str(caught.exception).startswith("cannot install"))


if __name__ == "__main__":
    unittest.main()
