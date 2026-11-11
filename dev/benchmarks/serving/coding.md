Run complete coding conversations against an isolated, already running Splash
server with the same model and workload for each arm.

```sh
python3 -m dev.benchmarks.serving.coding run --port 11519 \
  --model MODEL --repo /path/to/repo --workload workload.json \
  --out results/a1 --build baseline --users 4 --rounds 3 --tokens 4096
python3 -m dev.benchmarks.serving.coding compare \
  results/a1 results/b1 results/b2 results/a2
```

Each arm starts on a fresh server with its own cache. Keep the model weights,
source excerpts, workload, output budget, and environment fixed. Use the
candidate's build label in b1 and b2. Record source revisions and runtime
options in the build label or the accompanying experiment record. The harness
records the server's status before the run and after each round.

A workload selects source files, or inclusive line ranges, and the tasks and
trusted validator commands for each turn. Paths are relative to `--repo`.

```json
{
  "system": "Write regression tests using the supplied code.",
  "request_options": {"reasoning_effort": "none", "temperature": 0},
  "sources": [{"path": "runtime/engine/Scheduler.cpp"}],
  "cases": [{
    "id": "scheduler",
    "turns": ["Write a complete regression test.", "Add a priority case."],
    "validators": {
      "1": ["python3", "/path/to/validate_scheduler.py"],
      "2": ["python3", "/path/to/validate_scheduler.py"]
    }
  }]
}
```

The validator receives the saved response's absolute path as its last argument
and runs from `--repo`. It should extract the generated test, compile it against
the supplied source, run it, and return nonzero when an assertion or compilation
fails. The workload author chooses the checks. A validator that checks only
syntax cannot establish task correctness. Validate assertions with deliberate
faults when practical. Commands come from the local workload, never an answer.

The optional `request_options` object sets sampling and reasoning fields. The
harness fixes the model, messages, single completion, streaming mode, usage recording and output
budget for every request.

Cases are assigned round-robin to users. Every user keeps its real assistant
answers and local validator feedback in the next turn. Source context and the system prefix stay fixed
across rounds. Answers and request hashes are saved with source hashes and
exact ranges. Each turn runs as a concurrent wave. After every active user
finishes that turn, validators run before the next wave starts. The first 8000
characters of each validator result enter the next turn, with full results
saved. `generation_wall_s` sums the generation waves and excludes validation.
`wall_s` includes validation and measures the complete iteration.

Every turn needs a nonempty response ending with `stop`, positive usage, and a
passing validator to qualify. A token limit, transport error, timeout, missing
turn or missing validator fails the run. Partial answers remain saved for
inspection. Output directories must be new, so retries preserve prior runs.
Validator timeouts and interruptions stop their process group, including
compilers or test programs they started. Status snapshots that fail, unhealthy
Metal or transport, engine restarts and increased reclaim error counters also
fail qualification.

Comparison requires matching workloads and initial requests in ABBA order,
all rounds and validated turns. It applies `abba.compare_samples` to round
generation times. `gain` is true only when the improvement exceeds that rule's
noise bound. Different complete answers can pass the same validators, and
their output-token medians are reported. Inspect validator coverage, TTFT,
stream gaps, cache reuse and server health alongside the elapsed-time verdict.
