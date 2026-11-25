# Splash contributor and agent guide

This is the short contract for changing Splash. The reference documentation is
under `docs/`; this file says where to start and what must be true before a
change is reviewable.

Read first: `README.md`, `CONTRIBUTING.md`, `docs/README.md`, and
`docs/validate.md`.

## Where things live

| Path | Owns | Reference |
| --- | --- | --- |
| `server/` | HTTP APIs, chat templates, tool calls, streaming | `docs/internals.md` |
| `runtime/engine/` | scheduling, memory admission, cache | `docs/internals.md` |
| `runtime/model/` | architectures, weights, draft, vision | `docs/models.md` |
| `runtime/ops/`, `runtime/metal/` | operators and Metal kernels | `docs/internals.md` |
| `install/` | installer, Hub access, launchers | `docs/server.md` |
| `app/` | desktop app | `docs/package.md` |
| `dev/` | tests, benchmarks and build tools | `docs/validate.md` |
| `docs/` | user and developer documentation | `docs/README.md` |

## Commands

| Goal | Command |
| --- | --- |
| List documented targets | `make help` |
| Show this checkout's toolchain | `make doctor` |
| Source and Python checks, no native build | `make check-fast` |
| Everything model-free (Mac with Metal) | `make check` |
| Architecture and whitespace | `make check-source` |
| Python tests, lint and format | `make check-python` |
| One Python test file | `make test-python PYTHON_TEST_PATTERN='test_foo.py'` |
| Native CPU tests | `make check-native-cpu` |
| Metal tests | `make check-native-metal` |
| Every native test and tool builds | `make check-native-build` |
| Desktop app tests | `make check-app` |
| Sanitizers (long) | `make test-sanitizers` |
| Real-model oracle | `make test-real MODEL=... REVISION=...` |
| Release gate for one model | `make install release-check MODEL=... REVISION=... BASELINE=...` |

`make check` needs no model weights but needs a Mac Splash supports. The
real-model targets need both a supported Mac and an installed model. Use
`docs/contributing/verification.md` to choose the minimum checks for a change.

## Non-negotiables

- Five layers, each depending only on those below it. `make architecture-check`
  enforces the boundaries.
- The architecture checker's exception and line-budget lists only shrink.
  `runtime/engine` is capped at 31,209 production lines.
- Comments give the rule, its reason and one number. Cite nothing outside the
  repository.
- Engine policy reads what the model reports, never which model it is. Concrete
  models are assembled only in `Bootstrap`, `NativeArguments`,
  `RuntimeResources`, and `StreamedMemoryPlan`.
- Measure durations on `AwakeClock`; do not use `steady_clock` or its timed
  waits.
- Do not attach residency sets to a queue or command buffer; a command makes
  resident what its encoders use. Only page storage writes GPU addresses.
- When a native wire layout changes, bump `kProtocolVersion` and
  `PROTOCOL_VERSION` together in `runtime/engine/Protocol.hpp` and
  `server/protocol.py`. When a status field is removed or changes meaning, bump
  `kStatusSchemaVersion` and `STATUS_SCHEMA_VERSION` together.
- Keep reports, profiles, local paths, model files and experiment notes out of
  the tree.
- Never put a Hugging Face token in a workflow. CI installs public models only.
- Pin an upstream model to one revision when reporting a result.
- Do not edit generated outputs or force-add ignored directories such as
  `build/`, `install/models/`, or `.venv`.

## Verification and review

- Start with `docs/contributing/verification.md`.
- `make check-fast` is the minimum before opening a pull request.
- Run the checks for the area you changed before asking for review.
- If you cannot run a check, say exactly which check and why. Do not claim a
  run you did not make.
- State the Mac model, macOS version, model and pinned revision for any
  real-model result.
- Performance claims need an ABBA comparison on an idle Mac, with the metric,
  baseline, spread and complete workload named. A single run is not evidence.

## Pull request contract

State what changed, why, the layer, and the commands and results. Include the
hardware and model details above when they apply. Disclose AI tools in the pull
request. The human author must be able to explain every line and respond to
review.

Keep one logical change per pull request. Exploratory work stays in a draft.
If a change is larger than about 400 lines or ten files, explain why it cannot
be split. See `docs/contributing/ai-policy.md` for generated code, stacked
changes and accountability.

## Useful references

- `docs/server.md`: building, serving, options, model storage.
- `docs/api.md`: endpoints, streaming, tool calls, images and PDFs.
- `docs/models.md`: formats, drafts, vision, chat templates.
- `docs/internals.md`: code boundaries, kernels, memory, cache, protocol.
- `docs/validate.md`: every check, release gate and benchmark.
- `docs/contributing/README.md`: task-shaped contributor paths.
