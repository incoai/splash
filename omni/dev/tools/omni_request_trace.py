#!/usr/bin/env python3
"""Where a transcription's time goes, from a server log traced on both sides.

    SPLASH_AUDIO_TRACE=1 OMNI_ASR_TRACE=1 python -m server.audio_server ... 2> server.log
    python dev/tools/omni_request_trace.py server.log [--skip N]

The server prints one `audio trace` line per transcription (time.monotonic()
instants: arrival, upload read, preparation, the frame written, the first
token and Done read, the terminal delivered, the tokens collected, the
answer sent); the engine prints `asr trace` lines per request (the frame
read, admission, mel, and per command its graph build, submission, commit,
GPU start and end, and the wake after it). Both are on mach absolute time,
which Metal's GPU instants share. For one request at a time (concurrency 1)
every command between a request's arrival and its answer is its own; the
table averages each phase over the requests, in milliseconds.
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict

_FIELD = re.compile(r"(\S+?)=(\S+)")


def _fields(line: str) -> dict[str, str]:
    return dict(_FIELD.findall(line))


def parse(path: str):
    requests, commands, engine = [], [], defaultdict(dict)
    emitted = []
    with open(path, errors="replace") as handle:
        for line in handle:
            if line.startswith("audio trace "):
                requests.append(
                    {
                        k: float(v)
                        for k, v in _fields(line[12:]).items()
                        if "engine_id" not in k
                    }
                    | {
                        k: int(v)
                        for k, v in _fields(line[12:]).items()
                        if "engine_id" in k
                    }
                )
            elif line.startswith("asr trace "):
                kind = line.split()[2]
                fields = _fields(line)
                if "gpu0" in fields:
                    command = {
                        k: float(v)
                        for k, v in fields.items()
                        if k not in ("id", "rows")
                    }
                    command["kind"] = kind
                    commands.append(command)
                elif kind == "emitted":
                    emitted.append(float(fields["t"]))
                elif "id" in fields:
                    record = engine[int(fields["id"])]
                    for key, value in fields.items():
                        if key not in ("id", "tokens", "frames", "samples"):
                            record[f"{kind}.{key}"] = float(value)
    commands.sort(key=lambda c: c["gpu0"])
    emitted.sort()
    return requests, commands, engine, emitted


def breakdown(request, commands, engine, emitted):
    """A request's phases, in milliseconds."""
    start, end = request["arrival"], request["sent"]
    own = [c for c in commands if start <= c["submit"] <= end]
    e = engine.get(int(request.get("s0.engine_id", -1)), {})
    ms = lambda a, b: 1000.0 * (b - a)  # noqa: E731
    phases = {
        "http: upload read": ms(start, request["uploaded"]),
        "http: decode and split": ms(request["uploaded"], request["prepared"]),
        "frame built and written": ms(request["prepared"], request["s0.written"]),
    }
    read = e.get("received.read", request["s0.written"])
    admit = e.get("admit.start", read)
    phases["pipe to engine read"] = ms(request["s0.submit"], read) - ms(
        request["s0.submit"], request["s0.written"]
    )
    phases["engine wake and queue"] = ms(read, admit)
    phases["mel (CPU)"] = ms(admit, e.get("admit.mel", admit))
    # The GPU's idle time before each command: for one committed after the
    # command before it ended, the host's wake after that one, its work
    # between commands, the graph build, the encoding and the launch; for
    # one queued behind it (committed before it ended), the GPU's switch.
    gpu = build = encode = launch = wake = other = queued = 0.0
    previous, before = e.get("admit.mel", admit), None
    for command in own:
        if before is not None and command["commit"] < before["gpu1"]:
            queued += ms(before["gpu1"], command["gpu0"])
        else:
            if before is not None:
                wake += ms(before["gpu1"], before["woke"])
            other += ms(previous, command["build"])
            build += ms(command["build"], command["submit"])
            encode += ms(command["submit"], command["commit"])
            launch += ms(command["commit"], command["gpu0"])
        gpu += ms(command["gpu0"], command["gpu1"])
        previous, before = command["woke"], command
    phases["GPU busy"] = gpu
    phases["graph build (host)"] = build
    phases["encode and commit (host)"] = encode
    phases["commit to GPU start"] = launch
    phases["GPU end to wake"] = wake
    phases["emit, tick, between commands"] = other
    phases["GPU switch to a queued command"] = queued
    last = own[-1] if own else None
    phases["last GPU end to wake"] = ms(last["gpu1"], last["woke"]) if last else 0.0
    done = e.get("done.written", previous)
    phases["last wake to Done written"] = ms(previous, done)
    phases["Done to server read"] = ms(done, request.get("s0.done_read", done))
    phases["terminal delivered"] = ms(
        request.get("s0.done_read", done), request.get("s0.finalized", done)
    )
    phases["handler woken"] = ms(
        request.get("s0.finalized", done), request.get("s0.collected", done)
    )
    phases["detokenize and answer sent"] = ms(request.get("s0.collected", done), end)
    phases["total (arrival to answer)"] = ms(start, end)
    counts = defaultdict(int)
    for command in own:
        counts[command["kind"]] += 1
    return phases, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument(
        "--skip", type=int, default=0, help="leading requests to skip (warm-up)"
    )
    args = parser.parse_args()
    requests, commands, engine, emitted = parse(args.log)
    requests = requests[args.skip :]
    rows = [breakdown(r, commands, engine, emitted) for r in requests]
    if not rows:
        raise SystemExit("no traced requests")
    print(f"{len(rows)} requests; mean milliseconds per request")
    for name in rows[0][0]:
        values = [phases[name] for phases, _ in rows]
        print(
            f"  {name:34s} {statistics.mean(values):8.2f}   (median {statistics.median(values):.2f})"
        )
    kinds = defaultdict(int)
    for _, counts in rows:
        for kind, count in counts.items():
            kinds[kind] += count
    print(
        "  commands per request: "
        + ", ".join(f"{k} {v / len(rows):.1f}" for k, v in kinds.items())
    )
    # The gaps between consecutive commands: GPU idle from one end to the next start.
    gaps = defaultdict(list)
    for a, b in zip(commands, commands[1:]):
        if b["gpu0"] - a["gpu1"] < 0.05:
            gaps[(a["kind"], b["kind"])].append(1000.0 * (b["gpu0"] - a["gpu1"]))
    print("GPU idle between consecutive commands (ms): mean, median, count")
    for (a, b), values in sorted(gaps.items(), key=lambda item: -len(item[1])):
        print(
            f"  {a:>15s} -> {b:15s} {statistics.mean(values):6.3f} {statistics.median(values):6.3f} {len(values):6d}"
        )


if __name__ == "__main__":
    main()
