# Splash documentation

These documents run from using Splash to working on it:

- [Server](server.md): building and running Splash, its server's options, where
  models are kept, API model aliases, reasoning and thinking settings, agent
  launchers, and upgrading.
- [API](api.md): the endpoints, request fields and limits, sampling, streaming,
  tool calls, structured output, images and PDFs, status and errors.
- [Models](models.md): the model formats Splash loads, drafts, vision, chat
  templates, local model files and Splash packages.
- [Internals](internals.md): the code's boundaries, installation, weight
  loading, kernels, sparse attention, Neural Engine prefill, host steps,
  residency, admission and eviction, sliding windows, the disk tier, the native
  protocol, and failures.
- [Validate](validate.md): the checks, the release check and local benchmarks.
- [Package](package.md): the release archive and bottle, the desktop app, its
  updates and releases, and private test releases.
- [Performance](performance.md): measurements and how they were made.

Building from source needs Apple Silicon with macOS 26.4+, Xcode 26 or newer,
Python 3.12–3.14, and a Metal 4 compiler with `uint4b_format` tensor support.
The macOS 26.2 SDK can compile the host code, but Xcode 26.2's default Metal
component cannot compile the kernels; select a newer Metal toolchain when
using that SDK. Packaged users need none of these development tools.
