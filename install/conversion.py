"""Weights an installation converts from upstream checkpoints as it reads
them: a recipe's units (h3.py, MiniMax-H3's), run into a derived entry's
staging folder (assembly.converting).

A unit names the tensors it reads, by source file (Source), and the entry's
files it writes. The driver reads them by HTTP range requests of the
safetensors files on the Hub (hub.Repository.open: the header once, then
each tensor's bytes, adjacent tensors in one range), so no source file lands
on disk; a file the Hub cache holds is read there. While one unit runs, the
next one's tensors are read. Each file a unit writes is fsynced and
published by rename, so an interrupted conversion resumes at the first unit
whose files are missing. Scratch a unit writes for a later one lies under
names starting with "."; once the entry's files are written, FILES lists
them with their sizes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import models

# An entry's list of its files, by path, with their sizes.
FILES = "files.json"
# numpy's types of safetensors' dtypes (BF16 as its bits); numpy is imported
# where it is used, as huggingface_hub is (hub.py).
DTYPES = {
    "BF16": "<u2",
    "F16": "<f2",
    "F32": "<f4",
    "I8": "i1",
    "U8": "u1",
    "I32": "<i4",
}
# One range request reads at most CHUNK bytes; WORKERS run at once, and each
# is tried READ_ATTEMPTS times.
CHUNK = 64 << 20
WORKERS = 8
READ_ATTEMPTS = 4


def key(files, recipe, variant, spec=None):
    """The derived entry of a recipe's variant converted from files, each
    file it reads (REPO/NAME) -> its content's ID: what the conversion reads,
    not the commits it is read at, so a commit that changes no such file
    names the same entry; and the variant's spec, if it has one."""
    identity = {"files": files, "recipe": recipe, "variant": variant}
    if spec is not None:
        identity["spec"] = spec
    return hashlib.sha256(models.json_bytes(identity)).hexdigest()


class Source:
    """One safetensors file of a repository (hub.Repository), by its tensors'
    entries in its header, which is read once."""

    def __init__(self, repo, name):
        self.repo, self.name = repo, name
        with repo.open(name, readahead=False) as stream:
            length = struct.unpack("<Q", _exactly(stream, 8))[0]
            header = json.loads(_exactly(stream, length))
        # The header's metadata, which a trimmed copy keeps (video.py).
        self.metadata = header.pop("__metadata__", None)
        self.tensors, self.base = header, 8 + length

    def read_into(self, start, view):
        """The file's bytes at start into view, retried."""
        for attempt in range(READ_ATTEMPTS):
            try:
                with self.repo.open(self.name, readahead=False) as stream:
                    stream.seek(self.base + start)
                    _fill(stream, view)
                return
            except Exception as error:  # the Hub's and the network's errors
                if attempt + 1 == READ_ATTEMPTS:
                    raise models.ModelError(
                        f"cannot read {self.repo.name}/{self.name}: {error}"
                    ) from error
                time.sleep(2**attempt)


def _exactly(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise OSError(f"short read: {len(data)} of {size} bytes")
    return data


def _fill(stream, view):
    """view filled from stream, read into it in place."""
    done = 0
    while done < len(view):
        count = stream.readinto(view[done:])
        if not count:
            raise OSError(f"short read: {done} of {len(view)} bytes")
        done += count


@dataclass(frozen=True)
class Unit:
    """One step of a recipe: it writes outputs, the entry paths run(tensors)
    returns (bytes, buffers written in order, or a file to move there), from
    the tensors reads() names,
    (Source, tensor names) pairs, which run gets by name as (dtype, array).
    It is done once its outputs are written, or the output consumed_by of a
    later unit that reads them; discard is scratch removed after it."""

    name: str
    outputs: tuple[str, ...]
    run: Callable[[dict], dict]
    reads: Callable[[], list] | None = None
    consumed_by: str | None = None
    discard: tuple[str, ...] = ()

    def done(self, stage: Path):
        if self.consumed_by and (stage / self.consumed_by).is_file():
            return True
        return all((stage / name).is_file() for name in self.outputs)


def fetch(reads, pool):
    """The tensors of reads, (Source, names) pairs, by name -> (dtype,
    array): each run of adjacent tensors one buffer, read in chunks on
    pool."""
    import numpy as np

    spans = []
    for source, names in reads:
        for (start, end), name in sorted(
            (source.tensors[n]["data_offsets"], n) for n in names
        ):
            if spans and spans[-1][0] is source and spans[-1][2] == start:
                spans[-1][2] = end
                spans[-1][3].append((name, start))
            else:
                spans.append([source, start, end, [(name, start)]])
    jobs, tensors = [], {}
    for source, start, end, members in spans:
        buffer = bytearray(end - start)
        view = memoryview(buffer)
        for at in range(start, end, CHUNK):
            part = view[at - start : min(at + CHUNK, end) - start]
            jobs.append(pool.submit(source.read_into, at, part))
        for name, offset in members:
            entry = source.tensors[name]
            if name in tensors:
                raise models.ModelError(f"tensor {name} is read twice")
            array = np.frombuffer(
                buffer,
                DTYPES[entry["dtype"]],
                count=int(np.prod(entry["shape"])),
                offset=offset - start,
            )
            tensors[name] = (entry["dtype"], array.reshape(entry["shape"]))
    for job in jobs:
        job.result()
    return tensors


def _publish(path: Path, data):
    """Write one file of the staging folder durably, by rename: data is
    bytes, buffers written in order (none joined in memory), or a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, Path):
        os.replace(data, path)
    else:
        partial = path.with_name(path.name + ".partial")
        if isinstance(data, (bytes, bytearray, memoryview)):
            data = (data,)
        with partial.open("wb") as stream:
            for part in data:
                stream.write(part)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def convert(stage: Path, units, label):
    """Run, in order, the units the staging folder stage has not done, and
    list its files (FILES). label names the conversion in its progress."""
    pending = [unit for unit in units if not unit.done(stage)]
    done = len(units) - len(pending)
    with ThreadPoolExecutor(WORKERS) as pool, ThreadPoolExecutor(1) as ahead:

        def read(unit):
            return fetch(unit.reads(), pool) if unit.reads else {}

        upcoming = ahead.submit(read, pending[0]) if pending else None
        try:
            for number, unit in enumerate(pending, done + 1):
                tensors = upcoming.result()
                if number - done < len(pending):
                    upcoming = ahead.submit(read, pending[number - done])
                print(
                    f"Converting {label}: {unit.name} ({number} of {len(units)})",
                    flush=True,
                )
                # Each file is released once written, the tensors it holds
                # with it.
                written = unit.run(tensors)
                del tensors
                for name in list(written):
                    _publish(stage / name, written.pop(name))
                for name in unit.discard:
                    shutil.rmtree(stage / name, ignore_errors=True)
                os.utime(stage)  # its progress, which garbage collection reads
        except BaseException:
            # An interruption waits for the reads under way, not for the
            # next unit's.
            pool.shutdown(cancel_futures=True)
            raise
    files = {
        path.relative_to(stage).as_posix(): path.stat().st_size
        for path in sorted(stage.rglob("*"))
        if path.is_file()
        and not any(p.startswith(".") for p in path.relative_to(stage).parts)
        and path.name != FILES
    }
    _publish(stage / FILES, models.json_bytes(files))
