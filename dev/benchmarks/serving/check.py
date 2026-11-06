"""Checks of serving benchmark runs: one run on its own, or two builds
compared on four runs in ABBA order.

    python3 -m dev.benchmarks.serving.check run RUN_DIR LABEL \\
        [--minutes N] [--streamed] [--build BUILD_ID]
    python3 -m dev.benchmarks.serving.check compare A1 B1 B2 A2

`run` prints a FAIL line per failure, an INVALID line per reason the run
measured nothing (repeat it) and a NOTE line per readout, and exits 1 on any
FAIL or INVALID. `compare` takes each run as RUN_DIR/LABEL, baseline build A
first and last, checks each on its own and that its machine was quiet
(not_quiet), then compares their per-request metrics by
abba.compare_samples, and exits 1 unless every metric passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dev.benchmarks import abba
from dev.benchmarks.serving import clients
from dev.benchmarks.serving.run import IDLE_SAMPLE_SECONDS, SERVER_FAILURES

# Content chunks closer than this are one token event's: the server sends
# each token of an event as its own chunk, back to back, and a lane's next
# event comes a decode command later, the shortest measured being 17.3 ms.
# Half of that is the split farthest from both.
TOKEN_EVENT_MS = 17.3 / 2
# G4: a decode that stops this long is a freeze, not a slow step: the
# freezes 1.5.0 showed under memory pressure lasted 110 s and more.
GAP_MAX_S = 10.0
# G4: the p99 client gap, so that repeated multi-second stalls fail even when
# none reaches GAP_MAX_S.
GAP_P99_S = 5.0
# Gaps past this are listed one by one: five times the decode gap target.
GAP_REPORT_S = 2.5
# The decode gap target G: gaps past it are attributed to a cause.
GAP_TARGET_S = 0.5
# G8: supervision's GPU wait limit; a loop tick that long is a hang.
MAX_TICK_MS = 5000
# G8: a server stops on SIGINT, as on Ctrl+C, exits 0 and leaves none of the
# processes it started, its engine, running past SIGINT's grace
# (run.Server.stop).
CLEAN_STOP = {"signals": ["SIGINT"], "exit": 0, "left_running": []}
# G9: the engine's resource wait timeout; a claim waiting that long starves.
CLAIM_WAIT_MS = 30_000
# A regrown store unit taken back within this many seconds of its regrow
# counts as taken back, and a second take back within it of the first is an
# oscillation: a store that changes this fast pays reads for nothing.
STORE_HORIZON_S = 60
# A churn request's first token waits at most for the prefill of the work
# the scheduler may put ahead of it: the requests in flight when it was
# sent, later arrivals for OVERTAKE_ROWS and its own prompt, each by its
# tokens the cache did not hold. The scheduler lets at most three prefill
# commands in a row serve later arrivals and not a waiting request (the
# batch width, 4, less one), each of at most 8,192 rows (a group of four
# 2,048-row chunks, the most a model's command holds).
OVERTAKE_ROWS = 3 * 4 * 2048
# A first token later than this many times that work's prefill time at the
# run's own rate starved: of 247 churn runs on four Macs none waited more
# than 1.93 times it, and a 226 s wait put into the 11-04 pressure run is
# 2.35 times it or more.
FIRST_TOKEN_MARGIN = 2
THERMAL_HOT = ("serious", "critical")
# A process the run did not start, among the top five by CPU at this share or
# more, competes with the server and voids a timed comparison (not_quiet):
# twice the busiest steady system process on the Studio (WindowServer,
# 11.9 %). A Mac whose display is in use exceeds it (the M5 Pro's
# WindowServer at 36.6 % and more): compare fails its runs, and their own
# check notes it.
FOREIGN_CPU_PERCENT = 24


def read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open() as lines:
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut by a killed writer
    return records


def value(document, path: str, default=None):
    found = clients.dig(document, path)
    return default if found is None else found


def run_files(directory: Path, label: str, kind: str) -> list[Path]:
    """A run's LABEL.KIND.jsonl, or a ttft run's, one per concurrency."""
    single = directory / f"{label}.{kind}.jsonl"
    return (
        [single]
        if single.exists()
        else sorted(directory.glob(f"{label}-n*.{kind}.jsonl"))
    )


def store_size(status: dict):
    """The streamed experts' size: their account, or on trees whose status
    lists no accounts, their store's chunks."""
    experts = next(
        (
            a
            for a in value(status, "memory_governor.accounts", [])
            if a.get("name") == "experts"
        ),
        None,
    )
    return (
        experts["current_bytes"]
        if experts
        else value(status, "streamed_experts.chunks")
    )


class Run:
    """One run.py run's records: its meta, whether it ended with DONE, its
    build, its client events and its /status samples."""

    def __init__(self, directory: Path, label: str):
        self.directory, self.label = Path(directory), label
        meta = self.directory / f"{label}.meta.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        out = self.directory / f"{label}.out"
        text = out.read_text() if out.exists() else ""
        self.complete = text.rstrip().endswith("DONE")
        # A run killed before it wrote its meta named its build in its .out.
        self.build = self.meta.get("build_id") or next(
            (
                line.rsplit(" build ", 1)[1]
                for line in text.splitlines()
                if line.startswith("root ") and " build " in line
            ),
            None,
        )
        self.events = [
            e
            for path in run_files(self.directory, label, "events")
            for e in read_jsonl(path)
        ]
        self.done = [e for e in self.events if e.get("kind") == "done"]
        # Requests the client got no whole answer to (clients.run_churn).
        self.unanswered = [e for e in self.events if e.get("kind") == "unanswered"]
        # HTTP errors and failed TTFT rounds; a prompt past the context is
        # the client's.
        self.errors = [
            failure
            for e in self.events
            for failure in (
                [f"{e.get('code')} {e.get('body', '')}"]
                if e.get("kind") == "http_error"
                else e.get("failures", [])
                if e.get("kind") == "round_failed"
                else []
            )
            if "context_length_exceeded" not in failure
        ]
        self.end = next((e for e in self.events if e.get("kind") == "end"), None)
        self.samples = [
            s
            for path in run_files(self.directory, label, "status")
            for s in read_jsonl(path)
            if "status" in s
        ]
        self.idle = read_jsonl(self.directory / f"{label}.idle.jsonl")

    @property
    def last(self) -> dict:
        return self.samples[-1]["status"] if self.samples else {}

    def gaps(self) -> list[tuple[float, float]]:
        """Every client gap between token events, as (seconds, seconds into
        the run at its end): those of each answered request, and of each
        unanswered one up to when the client stopped waiting for it."""
        found = []
        for record in [*self.done, *self.unanswered]:
            chunks = record.get("chunks_ms") or []
            if record["kind"] == "unanswered":
                chunks = [*chunks, (record["t"] - record["start"]) * 1000]
            found += [
                ((later - earlier) / 1000, record["start"] + later / 1000)
                for earlier, later in zip(chunks, chunks[1:])
                if later - earlier >= TOKEN_EVENT_MS
            ]
        return found


def app_assertion(holder: str, ours: set[int]) -> bool:
    """Whether a `pmset -g assertions` line ("pid 351(powerd): ...") is held
    by a process the run did not start other than powerd, which holds one
    itself whenever the display is on."""
    _, found, rest = holder.partition("pid ")
    digits, _, rest = rest.partition("(")
    pid = int(digits) if found and digits.strip().isdigit() else None
    return pid not in ours and rest.partition(")")[0] != "powerd"


def invalid(run: Run, build: str | None, streamed: bool) -> list[str]:
    """Why a run measured nothing and is to be repeated: it did not end with
    DONE for a reason that is not the server's (G8 judges those), it served
    another build, or the experts did not stream from the start when they
    were to."""
    reasons = []
    if not run.complete and run.meta.get("failure") not in SERVER_FAILURES:
        reasons.append(
            "the run did not end with DONE"
            + (f": {run.meta['error']}" if run.meta.get("error") else "")
        )
    if build and run.build != build:
        reasons.append(f"build {run.build}, not {build}")
    if streamed and not value(
        run.samples[0]["status"] if run.samples else {}, "streamed_experts.cap_bytes"
    ):
        reasons.append("the routed experts did not stream from the start")
    return reasons


def not_quiet(run: Run) -> list[str]:
    """Why the machine was not quiet during a run: it ran hot, the GPU was
    busy at its start, the machine was in use or another process held a
    sleep assertion, or a foreign process competed for the CPU. Only a timed
    comparison needs a quiet machine: `compare` fails a run on these, as
    they void the times it compares. A run's own check judges freezes and
    correctness (G4, G8, G9), which they do not excuse, and notes them."""
    reasons = []
    thermal = {s["status"].get("thermal_state") for s in run.samples} & set(THERMAL_HOT)
    if thermal:
        reasons.append(f"thermal state {sorted(thermal)}")
    busy = run.meta.get("gpu_idle_check")
    if busy and not busy["ok"]:
        reasons.append(f"the GPU was busy at the start ({busy['max_percent']} %)")
    # A touch within one sampling period of a sample: the machine was used.
    touched = [
        s["hid_idle_s"]
        for s in run.idle
        if s.get("hid_idle_s") is not None and s["hid_idle_s"] < IDLE_SAMPLE_SECONDS
    ]
    if touched:
        reasons.append(f"the machine was in use (HIDIdleTime {touched[0]:.1f} s)")
    ours = set(run.meta.get("pids") or [])
    held = [h for s in run.idle for h in s.get("holders", []) if app_assertion(h, ours)]
    if held:
        reasons.append(f"another process held a sleep assertion: {held[0]}")
    for snapshot in run.meta.get("ps") or []:
        foreign = [
            row
            for row in snapshot["rows"][:5]
            if not row[4] and row[1] >= FOREIGN_CPU_PERCENT
        ]
        if foreign:
            reasons.append(
                "a process this run did not start in the top five by CPU: "
                f"{foreign[0][3]} at {foreign[0][1]:g} %"
            )
            break
    return reasons


def g8_failures(run: Run) -> list[str]:
    """G8, the server's own failures: an engine restart, a request it
    failed or left unanswered, a Metal failure, a loop tick past the GPU
    wait limit, an exit, a start not ready in time or a failed request that
    ended the run, or a stop other than CLEAN_STOP."""
    failures = []
    restarts = [
        r
        for r in (value(s["status"], "transport.restarts") for s in run.samples)
        if r is not None
    ]
    if len(set(restarts)) > 1:
        failures.append(
            f"G8: the engine restarted (transport.restarts {restarts[0]} -> {max(restarts)})"
        )
    # A ttft run's servers each start afresh: the largest count is the run's.
    failed = max(
        (value(s["status"], "requests.failed", 0) for s in run.samples), default=0
    )
    if failed or run.errors:
        failures.append(
            f"G8: {failed} failed requests in /status, {len(run.errors)} at the client"
        )
    if run.unanswered:
        failures.append(
            f"G8: {len(run.unanswered)} requests unanswered at the client: "
            + "; ".join(f"{e['tag']} ({e['error']})" for e in run.unanswered)
        )
    if any(
        value(s["status"], "metrics.metal_failures", 0)
        or not value(s["status"], "metal.healthy", True)
        for s in run.samples
    ):
        failures.append("G8: a Metal failure")
    ticks = [value(s["status"], "loop.max_tick_ms", 0) for s in run.samples]
    if ticks and max(ticks) >= MAX_TICK_MS:
        failures.append(f"G8: loop.max_tick_ms {max(ticks):.0f}")
    if run.meta.get("failure") in SERVER_FAILURES:
        failures.append(f"G8: {run.meta.get('error')}")
    for part in run.meta.get("parts", []):
        stop = part.get("stop") or {}
        if stop != CLEAN_STOP:
            failures.append(
                f"G8: the server of {part['label']} did not stop on SIGINT with "
                "exit 0 and its engine gone: signals "
                f"{', '.join(stop.get('signals') or []) or 'none'}, "
                f"exit {stop.get('exit')}, left running {stop.get('left_running')}"
            )
    return failures


def g9_failures(run: Run) -> list[str]:
    """G9, memory bookkeeping: the accounts sum to the charged bytes in
    every sample, no claim waits CLAIM_WAIT_MS, nothing is owed at the end."""
    failures = []
    for sample in run.samples:
        governor = sample["status"].get("memory_governor") or {}
        accounts = governor.get("accounts")
        if accounts and sum(a["current_bytes"] for a in accounts) != governor.get(
            "charged_bytes"
        ):
            failures.append(
                f"G9: the accounts do not sum to charged_bytes at t={sample['t']}"
            )
            break
    oldest = max(
        (
            value(s["status"], "memory_governor.claims.oldest_ms", 0)
            for s in run.samples
        ),
        default=0,
    )
    if oldest >= CLAIM_WAIT_MS:
        failures.append(f"G9: a claim waited {oldest:.0f} ms")
    owed = value(run.last, "memory_governor.owed.bytes", 0)
    if owed:
        failures.append(f"G9: {owed} bytes owed at the end")
    return failures


def quantile(ordered: list[float], q: float) -> float:
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0.0


def g4(run: Run) -> tuple[list[str], list[str]]:
    """G4 on the client gaps: the largest at most GAP_MAX_S and the p99 at
    most GAP_P99_S; each gap past GAP_REPORT_S is noted."""
    gaps = run.gaps()
    lengths = sorted(length for length, _ in gaps)
    largest, p99 = (lengths[-1] if lengths else 0.0), quantile(lengths, 0.99)
    failures, notes = [], []
    if largest > GAP_MAX_S:
        failures.append(f"G4: a client gap of {largest:.2f} s, past {GAP_MAX_S:g} s")
    if p99 > GAP_P99_S:
        failures.append(f"G4: client gap p99 {p99:.2f} s, past {GAP_P99_S:g} s")
    notes.append(f"client gaps: {len(lengths)}, max {largest:.2f} s, p99 {p99:.3f} s")
    notes += [
        f"gap of {length:.2f} s ending at {end:.1f} s"
        for length, end in sorted(gaps, key=lambda gap: gap[1])
        if length > GAP_REPORT_S
    ]
    return failures, notes


def rose(path: str):
    """A gap cause shown by a counter that rose across the samples."""
    return lambda around: value(around[-1], path, 0) > value(around[0], path, 0)


# The causes of a decode gap, in the order gap_causes tries them on the
# /status samples around it. Nearly every window commits a prefill command,
# so a prefill unit comes last.
GAP_CAUSES = (
    (
        "claim wait",
        lambda around: any(
            value(s, "memory_governor.claims.waiting", 0) > 0 for s in around
        ),
    ),
    ("store resize", lambda around: len({store_size(s) for s in around}) > 1),
    ("compaction", rose("kv.extent_compactions")),
    ("replan", rose("scheduler.replans")),
    ("history wait", rose("kv_spill.history_stream.wait_ms.total")),
    ("prefill unit", rose("scheduler.prefill_batches")),
)


def gap_causes(run: Run) -> dict[str, int]:
    """Every client gap past GAP_TARGET_S, counted by the first cause in
    GAP_CAUSES that the samples from 2 s before it to 2 s after it show,
    else as unattributed."""
    causes: dict[str, int] = {}
    for length, end in run.gaps():
        if length <= GAP_TARGET_S:
            continue
        around = [
            s["status"] for s in run.samples if end - length - 2 <= s["t"] <= end + 2
        ]
        cause = "unattributed"
        if len(around) >= 2:
            cause = next((name for name, shows in GAP_CAUSES if shows(around)), cause)
        causes[cause] = causes.get(cause, 0) + 1
    return causes


def store_churn(run: Run) -> dict:
    """The experts' grows and shrinks, the shrinks within STORE_HORIZON_S of
    the regrow before (a take back uses its regrow up), and of those the
    ones within STORE_HORIZON_S of the take back before: oscillations."""
    series = [(s["t"], store_size(s["status"])) for s in run.samples]
    series = [(t, size) for t, size in series if size is not None]
    grows = shrinks = taken_back = oscillations = 0
    last_grow = last_take_back = None
    for (_, before), (t, after) in zip(series, series[1:]):
        if after > before:
            grows += 1
            last_grow = t
        elif after < before:
            shrinks += 1
            if last_grow is not None and t - last_grow <= STORE_HORIZON_S:
                taken_back += 1
                if last_take_back is not None and t - last_take_back <= STORE_HORIZON_S:
                    oscillations += 1
                last_take_back = t
                last_grow = None
    return {
        "grows": grows,
        "shrinks": shrinks,
        "taken_back": taken_back,
        "oscillations": oscillations,
    }


def first_token_waits(run: Run) -> list[tuple[dict, float]]:
    """Each answered request with the prefill time of the work ahead of it
    (OVERTAKE_ROWS) at the run's own rate: the tokens its answered requests
    prefilled (prompt less cached) over the time from the first one's send
    to the last one's end."""
    served = [r for r in run.done if r.get("prompt") and r.get("ttft") is not None]
    if not served:
        return []

    def prefilled(record):
        return record["prompt"] - (record.get("cached") or 0)

    def client(record):
        return record["tag"].split("-")[0]

    span = max(r["t"] for r in served) - min(r["start"] for r in served)
    rate = sum(prefilled(r) for r in served) / span
    waits = []
    for record in served:
        # A churn client sends its next request once its last is answered,
        # so its own requests are never in flight beside it, whatever the
        # rounding of a send (0.1 s) against an answer (0.01 s) says.
        in_flight = sum(
            prefilled(other)
            for other in served
            if client(other) != client(record)
            and other["start"] <= record["start"] < other["t"]
        )
        work = in_flight + OVERTAKE_ROWS + prefilled(record)
        waits.append((record, work / rate))
    return waits


def check_run(
    run: Run,
    minutes: float | None = None,
    streamed: bool = False,
    build: str | None = None,
) -> dict:
    """One run on its own: `invalid` reasons, `failures` and `notes`, among
    them why its machine was not quiet. Every run is judged on G8; a churn
    run also on its load window, every client served, G4, G9, the store
    line and its first tokens' waits (FIRST_TOKEN_MARGIN)."""
    result = {
        "invalid": invalid(run, build, streamed),
        "failures": g8_failures(run),
        "notes": [f"not quiet: {reason}" for reason in not_quiet(run)],
    }
    if run.meta.get("scenario", "churn") != "churn":
        return result
    failures, notes = result["failures"], result["notes"]
    if minutes is not None and (run.end is None or run.end["t"] < minutes * 60):
        failures.append(f"the run ended before its {minutes:g} min load window")
    workers = {record["tag"].split("-")[0] for record in run.done}
    missing = [f"w{w}" for w in range(clients.CHURN_WORKERS) if f"w{w}" not in workers]
    if missing:
        failures.append(f"workers that completed no request: {missing}")
    gap_failures, gap_notes = g4(run)
    failures += gap_failures + g9_failures(run)
    notes += gap_notes
    causes = gap_causes(run)
    if causes:
        notes.append(f"gaps past {GAP_TARGET_S:g} s by cause: {causes}")
    store = store_churn(run)
    if store["oscillations"]:
        failures.append(
            f"store: {store['oscillations']} take backs within {STORE_HORIZON_S} s "
            f"of an earlier one (oscillation)"
        )
    notes.append(f"store: {store}")
    waits = first_token_waits(run)
    if waits:
        record, seconds = max(waits, key=lambda wait: wait[0]["ttft"] / wait[1])
        notes.append(
            f"first token: the longest wait {record['ttft'] / seconds:.2f} x the "
            f"prefill time of the work ahead of it ({record['tag']}, "
            f"{record['ttft']:g} s, {seconds:.0f} s)"
        )
    failures += [
        f"first token: {record['tag']} waited {record['ttft']:g} s, past "
        f"{FIRST_TOKEN_MARGIN} x {seconds:.0f} s, the prefill time of the work "
        "ahead of it at the run's rate"
        for record, seconds in waits
        if record["ttft"] > FIRST_TOKEN_MARGIN * seconds
    ]
    return result


def metrics(run: Run) -> dict[str, list[float]]:
    """The per-request samples of each lower-is-better metric of a run:
    churn's TTFT, completion latency and decode milliseconds per token,
    ttft's TTFT per concurrency, serve1's TTFT and decode per size, warm's
    prompt time per depth."""
    found: dict[str, list[float]] = {}
    scenario = run.meta.get("scenario", "churn")

    def add(name, sample):
        if sample:
            found.setdefault(name, []).append(sample)

    for record in run.done:
        add("churn_ttft_s", record.get("ttft"))
        add("churn_latency_s", record["t"] - record["start"])
        if record.get("decode_tps"):
            add("churn_decode_ms_per_token", 1000 / record["decode_tps"])
    for record in run.events:
        if record.get("kind") == "request" and scenario == "ttft":
            add(f"ttft_n{record['n']}_ttft_ms", record.get("ttft_ms"))
        elif record.get("kind") == "request" and scenario == "serve1":
            add(f"serve1_{record['size']}_ttft_ms", record.get("ttft_ms"))
            add(
                f"serve1_{record['size']}_decode_ms_per_token",
                record.get("decode_ms_per_token"),
            )
        elif record.get("kind") == "warm" and not record.get("discarded"):
            add(f"warm_{record['depth']}_prompt_ms", record.get("prompt_ms"))
    return found


def compare(runs: list[Run]) -> tuple[list[str], list[str]]:
    """Two builds on four runs in ABBA order: each run must pass its own
    check on a quiet machine, the two A runs and the two B runs must each
    be one build, the builds two, and every metric the four share must pass
    abba.compare. Returns the problems and one line per metric."""
    problems, lines = [], []
    for run in runs:
        checked = check_run(run)
        problems += [
            f"{run.label}: {reason}"
            for reason in checked["invalid"] + checked["failures"] + not_quiet(run)
        ]
    a1, b1, b2, a2 = (run.build for run in runs)
    if a1 != a2 or b1 != b2:
        problems.append(f"the rounds are not A B B A of two builds: {[a1, b1, b2, a2]}")
    elif a1 == b1:
        problems.append(f"both builds are {a1}")
    rounds = [metrics(run) for run in runs]
    shared = sorted(set.intersection(*(set(r) for r in rounds)))
    if not shared:
        problems.append("the four runs share no metric")
    for name in shared:
        result = abba.compare_samples([r[name] for r in rounds])
        lines.append(f"{name}: {abba.describe(result)}")
        if not result["pass"]:
            problems.append(f"{name}: {result['verdict']}")
    return problems, lines


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run", help="check one run on its own")
    one.add_argument("directory", type=Path)
    one.add_argument("label")
    one.add_argument("--minutes", type=float, help="the run's load window")
    one.add_argument(
        "--streamed",
        action="store_true",
        help="the model must have started with its routed experts streamed",
    )
    one.add_argument("--build", help="the build id the run must have served")
    two = commands.add_parser("compare", help="compare two builds on ABBA runs")
    two.add_argument(
        "runs", type=Path, nargs=4, metavar="RUN", help="RUN_DIR/LABEL, as A1 B1 B2 A2"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        result = check_run(
            Run(args.directory, args.label), args.minutes, args.streamed, args.build
        )
        for note in result["notes"]:
            print(f"NOTE {note}")
        for reason in result["invalid"]:
            print(f"INVALID {reason}")
        for failure in result["failures"]:
            print(f"FAIL {failure}")
        bad = result["invalid"] or result["failures"]
        print("check: " + ("FAIL" if bad else "PASS"))
        return 1 if bad else 0
    problems, lines = compare([Run(path.parent, path.name) for path in args.runs])
    for line in lines:
        print(line)
    for problem in problems:
        print(f"FAIL {problem}")
    print("compare: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
