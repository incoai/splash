# Contributing to Splash

Start with `CONTRIBUTING.md` for the repository rules and `AGENTS.md` for the
short working contract. This directory holds task-shaped guides; the reference
documentation stays under `docs/`.

## Choose your path

| You want to... | Start here |
| --- | --- |
| Install, serve or troubleshoot Splash | `README.md`, `docs/server.md` |
| Report a bug | The bug report issue form |
| Test Splash on your Mac | `docs/contributing/verification.md`, the hardware validation issue form |
| Change the server or installer | `docs/internals.md`, `docs/contributing/verification.md` |
| Add a model family | `docs/contributing/adding-a-model.md` |
| Work on kernels or performance | `docs/internals.md`, `docs/performance.md`, `docs/contributing/verification.md` |
| Review a pull request | `docs/contributing/verification.md`, `docs/contributing/ai-policy.md` |
| Contribute with an AI agent | `AGENTS.md`, `docs/contributing/verification.md` |

## Before you start

1. Run `make doctor` and check that the platform, Python and toolchain are
   supported.
2. Run `make help` to see the documented make targets.
3. Read the section of `docs/validate.md` that covers the area you will change.
4. For a larger change, open an issue first and agree on the approach.

`make check-fast` runs the source and Python checks without a native build. It
is the minimum before opening a pull request. `make check` runs everything that
needs no model weights, but it needs a Mac that Splash supports and Metal
tests.

## Contribution ladder

Not every contribution needs a kernel. Useful paths, from smallest to largest:

- Reproduce a bug on your Mac and attach the version, hardware, model and
  command.
- Validate a release or a model on hardware the release gate does not cover.
- Improve documentation, examples, or translations.
- Fix a server, installer or app issue with a focused test.
- Add a model family with its draft, fixtures and release-check evidence.
- Optimize a kernel with the benchmark evidence described in
  `docs/validate.md`.

## Working with AI agents

AI tools are welcome. The human author remains accountable, must be able to
explain every line, and must provide first-hand evidence. See
`docs/contributing/ai-policy.md`.

## Getting unstuck

If a check fails and the message does not name the cause, include the exact
command, the full output, and the state from `make doctor` in your issue or
pull request. That is usually faster than describing the symptom.

## Maintainers

After changing `.github/labels.json`, run `make labels` once with an
authenticated `gh` to create or update the labels in the repository. The issue
forms refer to those label names.
