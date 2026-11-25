# Verification

Use this page to choose the minimum checks for a change. The full reference is
`docs/validate.md`; this page is the decision table.

Run commands from the repository root. On a supported Mac, `make check-fast`
runs the source and Python checks without a native build. In a container or a
non-macOS environment, install the Python requirements from
`install/requirements.txt` and `dev/requirements.txt` with Python 3.12-3.14 and
run the same Python commands; native, Metal, app and real-model checks cannot
run there.

## Change matrix

| Change | Minimum before a pull request | Before review | Evidence to include |
| --- | --- | --- | --- |
| Documentation only | `make check-fast` | Render or preview the changed page | Commands and a short preview note |
| `server/` or `install/` Python | `make check-fast`, `make check-python` | The focused test file and the HTTP smoke if behavior changed | Test names and count; model and revision for real HTTP |
| `app/` | `make check-app` | The affected app test | Test result and macOS version |
| `runtime/engine/` | `make check-fast`, `make check-native-cpu` | `make test-sanitizers` when threads, lifetime or memory change | Counters, sanitizer result, before/after behavior |
| `runtime/model/` or a loader | `make check-fast`, `make check-native-cpu` | `make test-real MODEL=... REVISION=...` | Model, pinned revision, Mac, oracle result |
| `runtime/ops/` or `runtime/metal/` | `make check-fast`, `make check-native-metal` | `make test-performance-real BASELINE=...` for a speed change | ABBA table, spread, complete workload |
| Wire protocol or status schema | Both checks above | The version bump and both sides' tests | Old and new versions, compatibility note |
| Release or packaging | `make check`, the package targets in `docs/package.md` | The release check in `docs/validate.md` | Release report and hardware matrix |

`make check` runs `check-source`, `check-native-cpu`, `check-native-metal`,
`check-native-build`, `check-python` and `check-app`. It needs no model weights
but needs a supported Mac. `make test-sanitizers` is long; run it when the
change touches threads, shared state, memory lifetime or the cache.

## Python tests

Run every Python test:

```sh
make check-python
```

Run one file:

```sh
make test-python PYTHON_TEST_PATTERN='test_foo.py'
```

Lint and format alone:

```sh
make check-fast
```

## Real-model checks

Install the model first, or in the same command:

```sh
make install test-real MODEL=mlx-community/Qwen3.8-27B-4bit REVISION=<commit>
make install test-http-real MODEL=mlx-community/Qwen3.8-27B-4bit REVISION=<commit>
```

Pin the revision. `docs/validate.md` names the model and revision used by the
release gate, the oracle arguments, and the smaller-Mac variants.

## Performance evidence

A performance claim needs:

- the same idle Mac, model, quantisation and workload for both builds;
- an ABBA comparison, not one run per build;
- the metric, baseline, candidate, and the run's spread;
- the complete workload, including cache state, concurrency and context;
- the command and the result file, kept outside the source tree.

Use `make test-performance-real MODEL=... BASELINE=...` and the benchmark
rules in `docs/validate.md`. A single timing, a warm cache against a cold one,
or a different prompt set is not evidence.

## What to paste into a pull request

```text
Commands:
Results:
Mac and macOS:
Model and revision:
Performance (if any):
Checks not run, and why:
```

If a check fails, include the full output. Do not summarize an error into a
sentence; the exact message usually names the cause.
