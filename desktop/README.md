# Splash for macOS

Splash for macOS is a native menu bar app that runs one local model at a time through
`splash serve`. It requires Apple Silicon and macOS 26.4 or later. Opening the app may
install a compatible runtime, but it does not start the API or load a model.

A manual launch opens the manager on Models. Launch at login stays in the menu bar.

## Install

When a signed DMG is available on GitHub Releases, open it and drag Splash to
Applications. The app can download its pinned runtime on first launch; Homebrew is
not required for the release build. Download a model from All, then run it from
Installed or the menu bar.

## Build from source

Install Xcode 27, Ruby with the `xcodeproj` gem, and a compatible Splash CLI. From the
repository root, run:

```sh
ruby desktop/project.rb
make -C desktop test
make -C desktop build
open desktop/.build/DerivedData/Build/Products/Debug/Splash.app
```

`make -C desktop test` runs the Swift app tests and the Python model-tool and release
checks. The Xcode project, target, scheme, and app bundle are all named `Splash`.

With no bundled runtime manifest, the Debug build can use a compatible managed runtime
or a Homebrew installation at `/opt/homebrew/opt/splash/libexec`.

The app icon and menu bar mark come from `desktop/branding/splash-mark.svg`. After
editing the artwork, regenerate the assets before rebuilding:

```sh
swift desktop/scripts/generate_brand_assets.swift desktop
```

## Models and downloads

The manager's collapsible sidebar opens on Models. **All** and **Installed** are its
main tabs; **Downloads (n)** appears while downloads are active. All shows the curated
model sources, their installed counts, and Hugging Face links. Installed shows prepared
variants with full model IDs, format, quantization, local size, and Run, Stop, Chat,
and Hugging Face actions. Rows include publisher icons while retaining the
quantization repository in each model ID.

Choose any source, even one with a single variant, to open its quantization picker.
Installed variants show their local size without another download action. The picker
requests sizes from Hugging Face when it opens. Missing variants show approximate
sizes when available; an unknown size does not block a download. You can reopen the
picker for a fully installed source.
Ornith-1.5-35B-A3B MLX variants are marked **Text only** and use Splash's
`--language-only` option for downloads and serving.

Several variants can download at once. Each has its own progress and cancellation
control in Downloads; hovering over its percentage in the picker reveals a cancel
button. Progress samples cached bytes once a second, so reused cache files and
concurrent downloads from the same repository can skew the percentage.

## Service and chat

Run or stop an installed model from the manager or menu bar. Switching models waits for
active requests, stops the current service, and starts the selected model. The app
controls only the service process it started. It reports an occupied port without
stopping the other process. Closing the manager leaves the service running; quitting
waits for active requests before stopping it. **Force Stop** interrupts the owned
process when a graceful stop is waiting.

The API listens on `127.0.0.1:8000` by default. The menu can copy its current `/v1`
URL. **Browse models** opens the model list. The menu footer shows the app and runtime
versions, with Settings and Quit beside them. The menu bar remains available while the
manager is closed.

Chat opens a separate window titled `Splash Chat · <full model ID>` and shows startup
progress until the embedded Splash chat page is ready. Opening Chat from the manager
leaves the manager open; opening it from the menu closes the menu panel. Chat history
stays in `localStorage`. The app's WebView and each external browser have separate
histories. Each API port is a separate origin, so history from an old port reappears
when you switch back to it.

## Settings and runtime

Settings can launch Splash at login, change the server port, and release idle model
weights after 5 minutes, 10 minutes, or never. Port and idle-release changes take
effect on the next service start. You can also choose a Hugging Face download cache
directory or restore the default. New downloads use the selected directory immediately;
existing cache files stay where they are. The service uses the selected directory on
its next start.

The app does not delete installed models or shared Hugging Face cache files. A release
build bundles a manifest with a pinned runtime version, HTTPS archive URL, SHA-256
digest, and minimum compatible version. It can reuse a compatible external runtime;
otherwise, it downloads and verifies the archive under
`~/Library/Application Support/Splash/desktop/runtimes`. It never overwrites Homebrew.
Models and settings remain separate from runtime versions.

## Release candidate

The first public release is `v0.1.0-rc.1`. The Git tag, GitHub Release, DMG filename,
and app menu display `0.1.0-rc.1`. macOS requires a numeric
`CFBundleShortVersionString`, so the bundle uses `0.1.0` with an incrementing numeric
`CFBundleVersion`. This does not constitute a stable `0.1.0` release. The RC is a
manually installed DMG; in-app updates, custom model sources, and model removal are
not included.

The archive referenced by the runtime manifest must be available at its HTTPS URL
before building the DMG. `SPLASH_MINIMUM_RUNTIME_VERSION` is optional and defaults to
`SPLASH_RUNTIME_VERSION`.

On a Mac with Xcode, a Developer ID Application signing identity, and a configured
notarytool keychain profile, replace the example values and run:

```sh
export SPLASH_RUNTIME_ARCHIVE=/absolute/path/to/splash-runtime.tar.gz
export SPLASH_RUNTIME_VERSION=1.4.0
export SPLASH_RUNTIME_SHA256="$(shasum -a 256 "$SPLASH_RUNTIME_ARCHIVE" | awk '{print $1}')"
export SPLASH_RUNTIME_URL=https://example.org/splash-runtime.tar.gz
export SPLASH_APP_VERSION=0.1.0-rc.1
export SPLASH_APP_BUILD=1
export DEVELOPER_ID_APPLICATION='Developer ID Application: Your Name (TEAMID)'
export NOTARY_PROFILE=splash-notary
sh desktop/scripts/release.sh --validate
sh desktop/scripts/release.sh
```

The script validates the local archive and its HTTPS copy against the same SHA-256,
builds and verifies the app signature, notarizes the app, and staples its ticket. It
packages the app, an Applications shortcut, `LICENSE`, and `THIRD_PARTY_NOTICES` into
`desktop/.build/releases/Splash-0.1.0-rc.1.dmg`, then notarizes and staples the DMG.

Before uploading, install the DMG on a clean supported Mac and verify first launch,
runtime download, model serving, and quit behavior. Create the Git tag and GitHub
Release only after that acceptance check.
