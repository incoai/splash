# Package

Release archives contain no Hugging Face credentials.
Users accessing private models supply their own `HF_TOKEN` or Hugging Face login.

Release versions are three-part, `x.y.z`, with no `v` prefix: after `1.5.0`
comes `1.5.1` for a fix or `1.6.0` for a feature. Use the same version in all
three commands:

```sh
VERSION=1.6.0  # the release being built
make package RELEASE_VERSION=$VERSION
make package-bottle RELEASE_VERSION=$VERSION
make package-check RELEASE_VERSION=$VERSION
```

`make package` writes the runtime archive,
`dist/splash-<version>-arm64-macos26.tar.gz`, its checksum beside it
(`<archive>.sha256`) and the formula, `dist/splash.rb`; `make package-bottle`
adds the bottle to `dist/` and its block to the formula. These commands do not
publish. Build bottles on the oldest supported macOS. Bottle/check commands use
a temporary tap and remove their installation; they refuse to replace an
existing Splash installation. The install check requires a poured bottle and
runs the bundled launcher without a compiler or separate Python installation.

To publish, tag the verified release commit in `incoai/splash` with the version
(no `v` prefix), preserving existing history and tags, and make the release
with `make release-app` ([releases](#releases)). Its folder carries the runtime
archive, its `.sha256` file and the bottle beside the desktop app's disk images
and appcast, and `dist/splash.rb` as `Formula/splash.rb`, which the pull
request in `incoai/homebrew-tap` takes beside the app's cask; the formula's
URLs must point to the published release assets. After the tap update merges,
verify a fresh install and an upgrade from the previous release through the
public tap, including a real model request and preservation of user data. Run
`brew audit --strict --online incoai/tap/splash`.

Before publishing, verify that the models the README names and the draft
repositories of `families.FAMILIES` are publicly accessible. Check the installed
bottle on a supported Mac without developer tools; building from the source tree
is not an installation check.

The runtime package allowlists engine, Python, server and launcher files; tests,
benchmarks and developer documents are excluded. User model links survive
upgrades; downloads remain in the Hugging Face cache.

## Desktop app

`app/` is the Splash desktop app, a SwiftUI app for the Mac distributed as a
disk image:

```sh
make app RELEASE_VERSION=$VERSION   # dist/Splash.app, signed ad hoc
make dmg RELEASE_VERSION=$VERSION   # and dist/Splash-<version>.dmg
```

`dev/tools/package_app.py` builds the app with `swift build`, puts the runtime
`make package` ships, staged by the same function, under
`Contents/Resources/splash`, signs each of its Mach-O files and then the app,
and verifies them. Without `SIGN_IDENTITY` the signature is ad hoc, without the
hardened runtime, which cannot load the wheels' libraries under an ad hoc
signature: such a disk image is for local testing, opened with Open Anyway in
System Settings › Privacy & Security. `SIGN_IDENTITY`, a Developer ID
Application identity, adds the hardened runtime, which needs no entitlements;
only a release is notarized ([releases](#releases)).

The app adds no second way to serve. It runs the packaged runtime's own
commands, each in a process group of its own, as a terminal runs a foreground
job, with its output appended to a log in `~/Library/Logs/Splash`:
`install/models.py --model ID prepare --json-progress` downloads a model, and
`splash serve --model ID --port PORT --offline` serves an installed one, with
the launch options picked in the app that differ from the server's defaults,
ready once the port's lock names that process and `/health` answers; stopping
is Ctrl+C's SIGINT to the group. Switching models stops one server and starts
the next. The app starts nothing by itself: its window opens on the Server,
where models download, start and switch, beside the Chat. The toolbar's switch
opens the Chat, whose sidebar of conversations ⌘B hides, and the Server's
button at its end the Server: it shows the model that runs, a spinner while
one starts and a ring for a download's progress. Models, locks and
downloads are the ones a packaged `splash` uses, in `~/Library/Application
Support/Splash` and the Hugging Face cache. The app offers
`install/suggested-models.tsv`, per model the variant meant for the Mac's
memory with its download size and a few words about the model, and starts
coding agents with `splash AGENT --port PORT` in Terminal, in a folder the user
picks, from a script that holds the running server's API key and removes itself
as it starts. Its launch options are every option `splash serve` takes but the
ones of one model (`--revision`, `--draft-model`, `--mmproj`) and the ones it
sets itself, from one table (`LaunchOption.all`) the tests hold to `splash
serve --help`. It offers a model only those its family can use, and passes
no other: at launch it runs `install/families.py`, which reports for each
family, from the engine's `families` command and the options' own needs
(`ServeOption.needs`), the options none of its models can use and the context
they serve; a model's family is its assembly record's, and a model whose
family is not known yet gets every option. The API key reaches the server as
`SPLASH_API_KEY`, from the
Keychain, and nothing starts while the Keychain does not give it. The server
stops as SIGTERM stops it once the app has ended, by a crash too, since the app
passes its process ID as `SPLASH_APP_PID`; the app's next launch stops one still
running, as it does a download, by the process ID and start time it keeps. An
app before 1.6 kept the ID alone, and the server's port: the next launch stops
that server if the ID still holds the port's lock, and that download if the ID
still runs the download command above; it leaves any other process alone and
forgets the record. It serves model files on this Mac
the user adds with `splash serve --model local/NAME
--model-dir PATH`, NAME being the file's or folder's name made a repository
name, and at launch it runs `install/cached.py`. While a model runs, the Server
shows its context limit, from `/v1/models`, and its `/status`: requests,
answering and prompt speed, time to first token, the tokens generated, read and
taken from the cache, and the Metal memory it holds, with each request's line
of the server's log, folded. Its chat is its own, not the chat page: it streams
`/v1/chat/completions` and keeps each conversation as a JSON file in
`~/Library/Application Support/Splash/Chats`, so a conversation outlives a
switch of models. The window's drawer, ⌘J or the toolbar's terminal button,
holds terminals, each the user's login shell in the home folder, through
SwiftTerm, and the server's and the downloads' logs as they are written. Its
icon is `app/AppIcon.icon`, an Icon Composer document
`actool` compiles, which needs the full Xcode; its mark, three arcs of water
springing from a rippled surface, is the menu bar's too (`SplashMark`), and the
chat page's (`server/favicon.svg`). `swift test --package-path app` runs its
tests; with `SPLASH_RUNTIME` naming a packaged runtime and `SPLASH_APP_MODELS`
installed models, they also drive the app end to end and, with
`SPLASH_APP_SNAPSHOTS`, render its windows to that folder.

### Updates

The app updates itself with [Sparkle](https://sparkle-project.org), the
release `app/Package.swift` pins. It checks the appcast of the latest GitHub
release, `https://github.com/incoai/splash/releases/latest/download/appcast.xml`
(`APPCAST_URL` in `dev/tools/package_app.py`), a URL every app shipped keeps,
and opens a disk image it downloads only once it has checked who signed it:
the release key, an EdDSA key whose public half is `SPARKLE_PUBLIC_KEY` there,
or, to replace a lost release key, the app's Developer ID. An app built while
`SPARKLE_PUBLIC_KEY` is empty checks for nothing. Sparkle asks on the second
launch whether to check by itself, once a day, and installs an update by
quitting the app as its Quit command does, which stops what runs first.
`package_app.py` puts `Sparkle.framework` in `Contents/Frameworks` without its
XPC services, which only a sandboxed app needs, and signs its installer, its
progress app and the framework before the app.

The release key is made once, on the Mac that publishes releases, with the
tools `swift build` unpacks into `app/.build/artifacts/sparkle/Sparkle/bin`:

```sh
app/.build/artifacts/sparkle/Sparkle/bin/generate_keys --account ai.inco.splash
app/.build/artifacts/sparkle/Sparkle/bin/generate_keys --account ai.inco.splash \
    -x /Volumes/OFFLINE/splash-release-key
```

The first keeps the private key in the login keychain and prints the public
one, which goes into `SPARKLE_PUBLIC_KEY`, and into `RELEASE_KEY` in
`dev/tests/tools/test_package_app.py`, which holds it to that value, as every
app shipped keeps it; the second exports the private key for a backup kept
offline, such as on an encrypted drive, never in the checkout;
`generate_keys --account ai.inco.splash -f FILE` imports it on another Mac. A
lost key can be replaced with a new one: the apps shipped also take a disk
image the same team signed, if its app carries the new key's public half and
the new key signed it in the appcast (Sparkle's key rotation). With both the
key and that team's Developer ID lost, they update no more.

Each release carries `appcast.xml` beside its disk images, which
`make release-app` writes ([releases](#releases)) with `dev/tools/appcast.py`:
Sparkle's `generate_appcast` offers `Splash-<version>.dmg` from the release
tagged `<version>`, with `RELEASE_NOTES`, a Markdown file, and what it reads
from the app in the disk image: the build number Sparkle compares
(`CFBundleVersion`), the version it shows and `LSMinimumSystemVersion`. It
signs the disk image with the release key from the login keychain, or the
file `SPARKLE_KEY_FILE` names, and checks the signature with
`SPARKLE_PUBLIC_KEY`, and that the build number is above that of the release
before, the latest annotated `x.y.z` tag below the version, whose apps Sparkle
would otherwise never update. The build number is the commit count, so
`package_app.py` builds nothing from a shallow clone. The apps check whichever
release is the latest, and cannot while it carries no appcast. A release
candidate, `x.y.z-rc.N`, is a prerelease, never the latest.

### Releases

A release is made from its annotated tag, on the Mac that holds the Developer
ID and the release key, once `make package` and `make package-bottle`
([package](#package)) have left the command line's files of its version in
`dist/`, and published by hand. Its notes are `docs/release-notes/<version>.md`,
committed before the tag; [1.6.0's](release-notes/1.6.0.md) are the first kept
there, and earlier releases' are only on their GitHub releases:

```sh
git tag -a $VERSION -m $VERSION    # on the release commit
make release-app RELEASE_VERSION=$VERSION \
    RELEASE_NOTES=docs/release-notes/$VERSION.md \
    SIGN_IDENTITY="Developer ID Application: NAME (TEAMID)" \
    NOTARY_PROFILE=splash-notary
```

`make release-app` (`dev/tools/release_app.py`) first lists what keeps the
release from being whole, and stops on any of it: a change in the checkout,
untracked files included, which is why the notes are committed first; a HEAD
that is not the tag's commit, as the app shows the tag's `x.y.z`; an identity no
keychain holds; no notary credentials; an empty `SPARKLE_PUBLIC_KEY`; a
`dist/release/<version>` already made; and, unless it is a release candidate,
`x.y.z-rc.N`, which carries the app alone, a `dist/` without the version's
runtime archive, its `.sha256` file, its bottle and the formula naming them.
While a renewed certificate and the one it renews are both valid, their
identities share a name, and `SIGN_IDENTITY` is the SHA-1 that
`security find-identity -v -p codesigning` lists for the one to sign with.

It signs `dist/Splash.app` with the hardened runtime and a secure timestamp,
runs the notary service's checks on it (`syspolicy_check notary-submission`),
which warn of each Mach-O file in `Resources` and stop the release on anything
worse, notarizes it and staples its ticket, builds the disk image from it, then
signs, notarizes and staples the disk image too, keeping each submission's
log. An App Store Connect Team API key, `NOTARY_KEY` with `NOTARY_KEY_ID` and
`NOTARY_ISSUER`, can stand in for `NOTARY_PROFILE`. It checks the app in the
disk image as a downloaded copy (`codesign`, `spctl`, `stapler validate`,
`syspolicy_check distribution`, its versions and update keys) and prints the
team its signature names, stopping unless it is `SIGN_IDENTITY`'s. A run that
stops once it has begun keeps what it made, the notary logs among it, in a
folder it names.

It writes `dist/release/<version>` and uploads nothing: `Splash-<version>.dmg`
and the same file as `Splash.dmg`, which
`https://github.com/incoai/splash/releases/latest/download/Splash.dmg` names,
the command line's archive, `.sha256` file and bottle, `SHA256SUMS` of the disk
images and archives, `appcast.xml`, `notes.md`, the notary logs and, unless it
is a release candidate, `Formula/splash.rb` and `Casks/splash-app.rb` for
`incoai/homebrew-tap`. The cask's `zap` removes the app's chats, logs, caches
and settings, not the models, locks and runtime the command line keeps in
`~/Library/Application Support/Splash`, nor the Hugging Face cache. It prints
what publishing takes, from the folder, so that a copy of it can be published
from another Mac, one with `gh` signed in with write access to `incoai/splash`
and `incoai/homebrew-tap`, and Homebrew with `incoai/tap` tapped: the tag, a
draft release of the folder's files, published once its `Splash.dmg` opens on
a clean Mac, and the tap's pull request.

`DRY_RUN=1` rehearses a release without a tag, Developer ID, notary
credentials or release key. It lists what would stop the release, at its start
and its end, replaces `dist/Splash.app` with an ad hoc build, runs the notary
service's checks on it, prints the commands it leaves out, the signature of the
app's 70-odd files as one, and writes `dist/release/<version>-dry-run`,
replacing an earlier one, which a release never reads. That folder has no
notary logs, and no `appcast.xml` while `SPARKLE_PUBLIC_KEY` is empty.

The setup, once, on a personal Apple Developer account:

1. Create a Developer ID Application certificate, in Xcode › Settings ›
   Accounts › Manage Certificates or on developer.apple.com with the G2
   Sub-CA. `security find-identity -v -p codesigning` then lists it,
   `Developer ID Application: NAME (TEAMID)`, `TEAMID` being the team's
   10-character ID: that is `SIGN_IDENTITY`. Export it with its private key to
   a `.p12` kept offline. The first release asks for the login keychain's
   password to sign with that key: Always Allow, or it asks again for each of
   the 70-odd files a release signs.
2. Keep the notary service's credentials in the login keychain, with an
   app-specific password from account.apple.com:

   ```sh
   xcrun notarytool store-credentials splash-notary --apple-id EMAIL --team-id TEAMID
   ```

   `--key AuthKey_ID.p8 --key-id ID --issuer UUID` in place of `--apple-id`
   and `--team-id` stores an App Store Connect Team API key instead; an
   individual key, which has no issuer, is not taken.
3. Make the release key and set `SPARKLE_PUBLIC_KEY` ([updates](#updates)).
   Then dry-run a release on this Mac: with the key set, though not yet
   committed, a dry run also writes the appcast, signs it with the key from the
   login keychain and checks the signature, as a release does.

### Moving to the organization account

A release of the Inco AI organization's account differs from a personal one
only in who signs it. `SIGN_IDENTITY` becomes the organization's Developer ID
Application identity, as `security find-identity -v -p codesigning` lists it,
under the legal name the organization enrolled with. Its Account Holder
creates it from a certificate signing request made on the Mac that signs, not
as a cloud-managed certificate, whose private key stays with Apple, and
exports its `.p12`. The notary credentials are stored for the organization's
team under a profile of their own, such as `splash-notary-inco`, which keeps
the personal one. The bundle ID, `ai.inco.splash`, the release key, which
moves with its backup to the Mac that publishes, and `APPCAST_URL` must not
change: the apps shipped know the app by them.

The first organization release reaches the apps the personal account signed
through Sparkle, which takes a disk image the release key signed whatever team
signed it. Only key rotation needs the same team, so after the move the
release key is all that links the two: a new key would strand every app still
on a personal release. Sparkle installs that first update without its atomic
swap and Gatekeeper pre-scan, as its installer is still the personal team's,
so the first launch may show Verifying; later updates are the organization's
alone. Some of what macOS grants the app is tied to its team and is asked for
once more. Anyone who saved an API key is asked, at the first launch of the
organization's app, to let Splash read it: Always Allow ends it, Allow asks
again at each launch, and Deny keeps models from starting until Try Again or
until the Splash item is removed in Keychain Access. So is access to model
files in Desktop, Documents, Downloads or on other disks and, with the
firewall on and Listen on All Networks, to accept connections. The first
organization release's notes say so.

Rehearse the move before it reaches anyone. Make the first organization
release `x.y.z-rc.1`, a prerelease, which the apps don't check, and on a Mac
with the last personal release, an API key saved and a model added from
Downloads or another disk, point Sparkle at its appcast:

```sh
defaults write ai.inco.splash SUFeedURL \
    https://github.com/incoai/splash/releases/download/x.y.z-rc.1/appcast.xml
```

Check for Updates… then installs it and relaunches; each prompt comes once, and
the model serves. `defaults delete ai.inco.splash SUFeedURL` ends the
rehearsal, and `x.y.z` follows once it passes.

Before publishing, check that a release carries the intended team:
`make release-app` prints it, and `codesign -dv dist/Splash.app` shows it as
`TeamIdentifier`. Do not revoke the personal certificate: every app it signed
would stop opening; left to expire, its releases keep opening by their secure
timestamp. Keep its `.p12` offline until then, as a disk image it signs is the
only way to rotate the release key for apps still on a personal release, and
remove its identity from the keychain of the Mac that publishes, so that
`make release-app` refuses to sign with it.

## Private test releases

Testers can install a build before it is public, from a private Hugging Face
repository:

```sh
make package RELEASE_VERSION=$VERSION
make publish-test RELEASE_VERSION=$VERSION TEST_RELEASE_REPO=owner/splash-releases
```

`make publish-test` (`dev/tools/publish_test.py`) uploads the archive, its
checksum, the tester installer `dev/tools/install.sh` and `latest`, which
names this version, to `TEST_RELEASE_REPO` with the developer's own
`hf auth login`, creating the repository as a private one if it does not
exist; `publish_test.py --no-latest` uploads without moving `latest`. It
prints the command testers run with a read token for the repository in
`SPLASH_TOKEN`: curl reads the token's header from its standard input, so the
token is in no command line or URL. The installer requires an Apple Silicon
Mac on macOS 26.4 or newer. It downloads the version `latest` names, or
`SPLASH_VERSION`, checks its SHA-256, unpacks it under
`~/Library/Application Support/Splash/app`, points `current` at it, writes a
`splash` command into `/opt/homebrew/bin` or `/usr/local/bin` when writable,
else `~/.local/bin` (`SPLASH_BIN_DIR` chooses), and removes older versions. It
refuses to replace a `splash` command it did not write, and to switch versions
while a server runs. Running it again upgrades; installed models stay. To try
an archive before uploading it, serve `dist/` (`python3 -m http.server`) and
run the installer with `SPLASH_BASE_URL` naming that server, `SPLASH_VERSION`
and any `SPLASH_TOKEN`.
