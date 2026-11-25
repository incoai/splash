# Contributing

Issues are welcome here: bugs, model requests, and questions about running
Splash on your machine. Include the Splash version, the Mac and macOS version,
and the model you were serving.

Pull requests are reviewed and merged in this repository. Open them against
`main`; for a larger change, open an issue first so we can agree on the
approach before you build it. Describe in the pull request what changed and
how you tested it: `make check` needs no model weights, but it runs the Metal
tests, so it needs a Mac that Splash supports; the real-model checks a change
may need are in [Validate](docs/validate.md).

[The documentation](docs/README.md) covers building from source, configuration,
the API, the models Splash loads, its internals and the test suite.

## Start here

- [AGENTS.md](AGENTS.md) is the short working contract for contributors and
  agents: where code lives, the commands, and the rules a change must keep.
- [Contributing paths](docs/contributing/README.md) route you by the kind of
  change you want to make.
- [Verification](docs/contributing/verification.md) maps a change to the
  minimum checks and the evidence a review needs.
- [AI-assisted contributions](docs/contributing/ai-policy.md) explains
  disclosure, accountability and size.

Run `make doctor` to check your platform and toolchain, and `make help` to list
the documented targets. `make check-fast` runs the source and Python checks
without a native build. `make check` runs everything that needs no model
weights, but it runs the Metal tests, so it needs a Mac that Splash supports.

Use the issue forms for bugs, model requests, features and hardware reports.
They ask for the details that make a report reproducible.

## Evidence

Describe what changed and how you tested it. Include the commands and their
results, the Mac model and macOS version, and the model and pinned revision for
any real-model result. If you could not run a check, say which one and why.
Performance claims need an ABBA comparison on an idle Mac with the baseline,
metric, complete workload and spread named; a single run is not evidence.

Keep one logical change per pull request. Exploratory work stays in a draft.
For a change above about 400 lines or ten files, explain why it cannot be
split; above about 2,000 lines, link an issue or design discussion that agreed
on the approach.

## AI-assisted contributions

AI tools are welcome. Disclose the tool in the pull request, keep a human
accountable for every line, and provide first-hand evidence. An agent may draft
an issue, comment or pull request; a human sends, labels, closes and merges it.
See [the full policy](docs/contributing/ai-policy.md).

## Engineering rules

- Clean and general: one rule for every case, no special cases, flags or
  hacks; a change deletes the mechanism it supersedes.
- A fixed rule with measured feedback first; an estimator only after an A/B
  shows it beats a simple baseline.
- Tests assert behaviour, not implementation.
- Comments give the rule, its reason and one number, and cite nothing outside
  the repository.
- Five layers, each depending only on those below it. The frontend (`server/`,
  `splash`, `install/`, `app/`: HTTP APIs, chat templates, tool calls, the
  installer and the app) reaches the engine through the wire protocol alone.
  The engine (`runtime/engine/`: requests, scheduling, memory, cache) uses the
  model through its interface headers alone, except engine assembly
  (`Bootstrap`, `NativeArguments`, `RuntimeResources`, `StreamedMemoryPlan`),
  the one part that builds the concrete model. The model (`runtime/model/`:
  architectures, formats, weights and the runtime that encodes commands), ops
  (`runtime/ops/`: operators and their kernel selection), and kernels
  (`runtime/metal/`, `*.metal`: Metal shaders and their ABI) follow.
- `make architecture-check` enforces these; its exception lists only shrink.
