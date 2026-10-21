# Contributing

Issues are welcome here: bugs, model requests, and questions about running
Splash on your machine. Include the Splash version, the Mac and macOS version,
and the model you were serving.

Pull requests are reviewed and merged in this repository. Open them against
`main`; for a larger change, open an issue first so we can agree on the
approach before you build it. Describe in the pull request what changed and
how you tested it: `make check` needs no model weights, but it runs the Metal
tests, so it needs a Mac that Splash supports; the real-model checks a change
may need are in [DEVELOPMENT.md](DEVELOPMENT.md#validate).

[DEVELOPMENT.md](DEVELOPMENT.md) covers building from source, configuration,
the API, the models Splash loads, its internals and the test suite.
