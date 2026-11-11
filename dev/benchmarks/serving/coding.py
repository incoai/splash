"""Complete, concurrent coding conversations against an already running server.

Workloads select whole files or explicit inclusive line ranges, never a token
prefix. Validators are trusted local argv lists in the workload, invoked with
the saved answer as their last argument. Model output is never a command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dev.benchmarks import abba
from dev.benchmarks.serving import clients


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def prepare(repo: Path, workload: dict) -> tuple[str, list[dict]]:
    """Read the exact supplied source ranges and record their provenance."""
    sections = []
    sources = []
    repo = repo.resolve()
    for source in workload["sources"]:
        path = (repo / source["path"]).resolve()
        if not path.is_relative_to(repo):
            raise ValueError(f"source outside repository: {path}")
        text = path.read_bytes().decode("utf-8")
        lines = text.splitlines(keepends=True)
        start = source.get("start", 1)
        end = source.get("end", len(lines))
        if not 1 <= start <= end <= len(lines):
            raise ValueError(f"invalid source range: {path}:{start}-{end}")
        excerpt = "".join(lines[start - 1 : end])
        sections.append(f"--- {source['path']}:{start}-{end} ---\n{excerpt}")
        sources.append(
            {
                "path": source["path"],
                "start": start,
                "end": end,
                "file_sha256": digest(text),
                "excerpt_sha256": digest(excerpt),
            }
        )
    if not sections or not workload["cases"]:
        raise ValueError("a workload needs sources and cases")
    names = [case["id"] for case in workload["cases"]]
    if len(names) != len(set(names)):
        raise ValueError("case ids must be unique")
    for case in workload["cases"]:
        if not case["turns"] or not all(case["turns"]):
            raise ValueError("a case needs nonempty turns")
    return "\n\n".join(sections), sources


def validate(command: list[str], answer: Path, repo: Path, timeout: float) -> dict:
    """Run a workload author's validator, preserving failure and timeout."""
    try:
        with subprocess.Popen(
            [*command, str(answer.resolve())],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except BaseException:
                # A validator can start a compiler or a test binary. End its
                # process group on timeout or interruption before returning.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                raise
        return {
            "pass": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"pass": False, "error": str(error)}


def generate(
    port: int, payload: dict, record: dict, answer_path: Path, timeout: float
) -> dict:
    body = json.dumps(payload)
    record = {**record, "request_sha256": digest(body), "pass": False}
    try:
        stream = clients.Stream(port, body, timeout=timeout)
        answer = "".join(stream.text)
        record.update(
            output=answer,
            finish_reason=stream.finish_reason,
            complete=(
                stream.ok
                and stream.finish_reason == "stop"
                and bool(answer.strip())
                and stream.usage.get("completion_tokens", 0) > 0
            ),
            usage=stream.usage,
            metrics=stream.metrics,
            wall_s=stream.ended - stream.sent,
            ttft_s=stream.first - stream.sent if stream.first else None,
            gaps=clients.gap_summary(stream.chunks, stream.sent),
            error=stream.error_body,
        )
        answer_path.write_text(answer)
    except Exception as error:
        record["error"] = str(error)
    return record


def health_errors(before: dict, after: dict) -> list[str]:
    errors = []
    for path in ("metal.healthy", "memory_audit.valid", "transport.ready"):
        if clients.dig(after, path) is False:
            errors.append(path)
    for path in (
        "transport.restarts",
        "memory_governor.count_checks.overcounted",
        "memory_governor.count_checks.short_releases",
    ):
        if (clients.dig(after, path) or 0) > (clients.dig(before, path) or 0):
            errors.append(path)
    return errors


def run(args) -> dict:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    workload = json.loads(args.workload.read_text())
    source, sources = prepare(args.repo, workload)
    (out / "source.txt").write_text(source)
    (out / "workload.json").write_text(json.dumps(workload, indent=2))
    meta = {
        "build": args.build,
        "model": args.model,
        "users": args.users,
        "rounds": args.rounds,
        "max_completion_tokens": args.tokens,
        "sources": sources,
        "workload_sha256": digest(json.dumps(workload, sort_keys=True)),
        "source_sha256": digest(source),
        "status_before": clients.get_json(args.port, "/status"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    assigned = [
        workload["cases"][user % len(workload["cases"])] for user in range(args.users)
    ]
    expected = sum(len(case["turns"]) for case in assigned)
    records = []
    with (out / "results.jsonl").open("w") as output:
        for round_id in range(1, args.rounds + 1):
            started = time.monotonic()
            generation_wall = 0.0
            rows = []
            histories = [
                [
                    {
                        "role": "system",
                        "content": workload.get(
                            "system", "Assist with the supplied repository code."
                        ),
                    },
                    {"role": "user", "content": source},
                ]
                for _ in assigned
            ]
            active = set(range(args.users))
            with ThreadPoolExecutor(max_workers=args.users) as workers:
                for turn in range(max(len(case["turns"]) for case in assigned)):
                    pending = []
                    for user, case in enumerate(assigned):
                        if user not in active or turn >= len(case["turns"]):
                            continue
                        histories[user].append(
                            {
                                "role": "user",
                                "content": f"{case['turns'][turn]}\nConversation {user + 1}.",
                            }
                        )
                        payload = {
                            "temperature": 0,
                            "reasoning_effort": "none",
                            **workload.get("request_options", {}),
                            "model": args.model,
                            "messages": histories[user],
                            "max_completion_tokens": args.tokens,
                            "stream": True,
                            "stream_options": {"include_usage": True},
                            "n": 1,
                        }
                        record = {
                            "case": case["id"],
                            "user": user + 1,
                            "round": round_id,
                            "turn": turn + 1,
                        }
                        path = out / f"r{round_id}-u{user + 1}-t{turn + 1}.txt"
                        pending.append((user, path, payload, record))
                    wave_started = time.monotonic()
                    futures = [
                        workers.submit(
                            generate, args.port, payload, record, path, args.timeout
                        )
                        for _, path, payload, record in pending
                    ]
                    wave = [future.result() for future in futures]
                    generation_wall += time.monotonic() - wave_started
                    # No inference is active while trusted validators compile and run.
                    for (user, path, _, _), row in zip(pending, wave):
                        rows.append(row)
                        if not row.get("complete"):
                            active.discard(user)
                            continue
                        histories[user].append(
                            {"role": "assistant", "content": row["output"]}
                        )
                        command = (
                            assigned[user].get("validators", {}).get(str(turn + 1))
                        )
                        validation = (
                            validate(command, path, args.repo, args.validator_timeout)
                            if command
                            else {"pass": False, "error": "no validator supplied"}
                        )
                        row["validation"] = validation
                        row["pass"] = validation["pass"]
                        feedback = json.dumps(validation, ensure_ascii=False)
                        histories[user].append(
                            {
                                "role": "user",
                                "content": "Local validator result for your previous answer (first 8000 characters):\n"
                                + feedback[:8000],
                            }
                        )
            record = {
                "round": round_id,
                "wall_s": time.monotonic() - started,
                "generation_wall_s": generation_wall,
                "expected": expected,
                "requests": rows,
            }
            try:
                record["status_after"] = clients.get_json(args.port, "/status")
                record["health_errors"] = health_errors(
                    meta["status_before"], record["status_after"]
                )
                _, observed_sources = prepare(args.repo, workload)
                if observed_sources != sources:
                    record["health_errors"].append("source changed during the run")
            except Exception as error:
                record["health_errors"] = [str(error)]
            records.append(record)
            output.write(json.dumps(record) + "\n")
            output.flush()
            print(
                f"round {round_id}: {sum(row['pass'] for row in rows)}/{expected} validated",
                flush=True,
            )
    summary = {
        "pass": all(
            len(r["requests"]) == expected
            and not r["health_errors"]
            and all(q["pass"] for q in r["requests"])
            for r in records
        ),
        "rounds": len(records),
        "completed": sum(
            q.get("complete", False) for r in records for q in r["requests"]
        ),
        "validated": sum(q["pass"] for r in records for q in r["requests"]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def compare(paths: list[Path]) -> dict:
    """Require complete matching ABBA cohorts before comparing elapsed time."""
    metadata = [json.loads((path / "meta.json").read_text()) for path in paths]
    runs = [
        [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
        for path in paths
    ]
    keys = (
        "model",
        "users",
        "rounds",
        "max_completion_tokens",
        "workload_sha256",
        "source_sha256",
    )
    if any(any(meta[key] != metadata[0][key] for key in keys) for meta in metadata):
        raise ValueError("ABBA workloads or source differ")
    builds = [meta["build"] for meta in metadata]
    if builds[0] != builds[3] or builds[1] != builds[2] or builds[0] == builds[1]:
        raise ValueError("build labels must be in ABBA order")
    cohorts = []
    for meta, rounds in zip(metadata, runs):
        if len(rounds) != meta["rounds"] or [r["round"] for r in rounds] != list(
            range(1, meta["rounds"] + 1)
        ):
            raise ValueError("missing or duplicate rounds")
        cohort = [
            (r["round"], q["user"], q["case"], q["turn"], q["request_sha256"])
            for r in rounds
            for q in r["requests"]
        ]
        if any(
            len(r["requests"]) != r["expected"]
            or r.get("health_errors")
            or not all(
                q["pass"]
                and q.get("complete")
                and q.get("finish_reason") == "stop"
                and q.get("validation", {}).get("pass")
                for q in r["requests"]
            )
            for r in rounds
        ):
            raise ValueError("incomplete or unvalidated coding responses")
        identities = [q[:4] for q in cohort]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate requests")
        cohorts.append(cohort)
    # Later requests contain each build's real answer. Their hashes may differ.
    identities = [[q[:4] for q in cohort] for cohort in cohorts]
    if any(cohort != identities[0] for cohort in identities):
        raise ValueError("request cohorts differ")
    seeds = [[q for q in cohort if q[3] == 1] for cohort in cohorts]
    if any(seed != seeds[0] for seed in seeds):
        raise ValueError("initial requests differ")
    result = abba.compare_samples(
        [[r["generation_wall_s"] for r in rounds] for rounds in runs]
    )
    result["gain"] = result["pass"] and -result["regression"] > result["bound"]
    result["median_output_tokens"] = [
        statistics.median(
            q["usage"]["completion_tokens"] for r in rounds for q in r["requests"]
        )
        for rounds in runs
    ]
    return result


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("run")
    collect.add_argument("--port", type=positive_int, required=True)
    collect.add_argument("--model", required=True)
    collect.add_argument("--repo", type=Path, required=True)
    collect.add_argument("--workload", type=Path, required=True)
    collect.add_argument("--out", type=Path, required=True)
    collect.add_argument("--build", required=True)
    collect.add_argument("--users", type=positive_int, default=4)
    collect.add_argument("--rounds", type=positive_int, default=3)
    collect.add_argument("--tokens", type=positive_int, default=4096)
    collect.add_argument("--timeout", type=positive_int, default=600)
    collect.add_argument("--validator-timeout", type=positive_int, default=120)
    comparison = commands.add_parser("compare")
    comparison.add_argument("runs", type=Path, nargs=4)
    args = parser.parse_args()
    result = run(args) if args.command == "run" else compare(args.runs)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
