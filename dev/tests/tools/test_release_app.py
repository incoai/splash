import base64
import io
import json
import os
import plistlib
import re
import shlex
import subprocess
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from dev.tools import appcast, package, package_app, release_app

IDENTITY = "Developer ID Application: Jane Doe (ABCDE12345)"
SHA1 = "0123456789ABCDEF0123456789ABCDEF01234567"
KEY = base64.b64encode(bytes(range(32))).decode()
PROFILE = ["--keychain-profile", "splash-notary"]
API_KEY = ["--key", "/keys/AuthKey_KEYID.p8", "--key-id", "KEYID"]
API_KEY += ["--issuer", "69a6de70-03db-47e3-e053-5b8c7c11a4d1"]
# The zip notarytool takes the app in, wherever its temporary folder is.
ZIP = re.compile(r".*/\.notarize-[^/]+/Splash\.zip")
# The command line's files of release 1.5.0, as make package and make
# package-bottle name them.
ARCHIVE = "splash-1.5.0-arm64-macos26.tar.gz"
BOTTLE = "splash-1.5.0.arm64_tahoe.bottle.tar.gz"


def recorded(commands, *paths):
    """commands with each of paths, and the zip of the app, by name."""

    def name(argument):
        for path in paths:
            argument = argument.replace(str(path), path.name)
        return "Splash.zip" if ZIP.fullmatch(argument) else argument

    return [[name(argument) for argument in command] for command in commands]


def identities(*listed):
    """security find-identity -v -p codesigning's output for listed, pairs of
    a SHA-1 and a name."""
    lines = [f'  {n}) {sha1} "{name}"' for n, (sha1, name) in enumerate(listed, 1)]
    return "\n".join([*lines, f"     {len(listed)} valid identities found", ""])


class ReleaseAppTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.notes = self.root / "notes.md"
        self.notes.write_text("Notes\n")
        self.key = self.root / "AuthKey_KEYID.p8"
        self.key.write_text("throwaway\n")

    def parse(self, *arguments):
        return release_app.parse_args(
            ["--version", "1.5.0", "--macos-min", "26.4"]
            + ["--notes", str(self.notes), *arguments]
        )

    def test_invalid_arguments_stop_before_the_build(self):
        # A release is a tag, x.y.z or x.y.z-rc.N, signed with a Developer ID
        # whose team it names, by its name or its SHA-1, and notarized with
        # one kind of credentials; each refusal names the make variable.
        api_key = ["--notary-key", str(self.key), "--notary-key-id", "KEYID"]
        api_key += ["--notary-issuer", "ISSUER"]
        for arguments, refusal in (
            (["--version", "1.4.1-65-g847bc4bc"], "RELEASE_VERSION .* not 1.4.1-65"),
            (["--version", "1.5"], "RELEASE_VERSION"),
            (["--version", "v1.5.0"], "RELEASE_VERSION"),
            (["--version", "1.5.0-beta.1"], "RELEASE_VERSION"),
            (["--notes", "missing.md"], "RELEASE_NOTES .* names no file: missing.md"),
            (
                ["--sign", "Apple Development: Jane Doe (ABCDE12345)"],
                "SIGN_IDENTITY .* not 'Apple Development",
            ),
            (["--sign", "Developer ID Application: Jane Doe"], "SIGN_IDENTITY"),
            (
                ["--sign", "Developer ID Application: NAME (TEAMID)"],
                "10-character ID",
            ),
            (["--sign", SHA1[:-1]], "SIGN_IDENTITY"),
            (["--notary-profile", "splash-notary", *api_key], "not both"),
            (api_key[:4], "NOTARY_KEY, NOTARY_KEY_ID and NOTARY_ISSUER"),
            (["--notary-key", "missing.p8", *api_key[2:]], "no such key file"),
            (["--sparkle-key-file", "missing"], "no such key file"),
        ):
            stderr = io.StringIO()
            with (
                self.subTest(arguments=arguments),
                mock.patch.object(
                    package_app,
                    "build_app",
                    side_effect=AssertionError("built despite invalid arguments"),
                ),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit),
            ):
                release_app.main(
                    ["--version", "1.5.0", "--macos-min", "26.4"]
                    + ["--notes", str(self.notes), *arguments, "--dry-run"]
                )
            self.assertRegex(stderr.getvalue(), refusal)
        # A dry run needs neither the identity nor the credentials; given,
        # they are what it prints. A SHA-1 is taken in either case.
        self.assertTrue(self.parse("--dry-run").dry_run)
        release = ["--sign", IDENTITY, "--notary-profile", "splash-notary"]
        self.assertEqual(release_app.notary_options(self.parse(*release)), PROFILE)
        self.assertEqual(
            release_app.notary_options(self.parse("--sign", IDENTITY, *api_key)),
            ["--key", str(self.key), "--key-id", "KEYID", "--issuer", "ISSUER"],
        )
        self.assertIsNone(release_app.notary_options(self.parse("--sign", IDENTITY)))
        self.assertEqual(self.parse("--sign", SHA1.lower()).sign, SHA1)
        self.assertEqual(
            self.parse("--version", "1.5.0-rc.1", *release).version, "1.5.0-rc.1"
        )

    def test_a_release_is_built_from_a_clean_checkout_of_its_annotated_tag(self):
        checkout = self.root / "checkout"
        checkout.mkdir()
        environment = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }

        def git(*arguments):
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com"]
                + list(arguments),
                cwd=checkout,
                env=environment,
                check=True,
                capture_output=True,
            )

        git("init", "-q")
        (checkout / "README.md").write_text("Splash\n")
        git("add", "README.md")
        git("commit", "-q", "-m", "First")
        git("tag", "-a", "1.5.0", "-m", "1.5.0")
        git("tag", "1.5.1")
        with mock.patch.object(release_app, "ROOT", checkout):
            self.assertEqual(release_app.checkout_problems("1.5.0"), [])
            for version in ("1.5.1", "1.5.2"):
                self.assertEqual(
                    release_app.checkout_problems(version),
                    [f"there is no annotated tag {version}"],
                )
            # Release notes kept in the checkout are a change too, by name.
            (checkout / "README.md").write_text("The Splash desktop app\n")
            (checkout / "notes.md").write_text("Notes\n")
            self.assertEqual(
                release_app.checkout_problems("1.5.0"),
                ["the checkout has changes: README.md, notes.md"],
            )
            git("add", "README.md", "notes.md")
            git("commit", "-q", "-m", "Second")
            self.assertEqual(
                release_app.checkout_problems("1.5.0"),
                ["HEAD is not the commit tag 1.5.0 names"],
            )

    def test_the_release_before_is_the_latest_annotated_x_y_z_below(self):
        checkout = self.root / "checkout"
        checkout.mkdir()
        environment = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }

        def git(*arguments):
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com"]
                + list(arguments),
                cwd=checkout,
                env=environment,
                check=True,
                capture_output=True,
            )

        git("init", "-q")
        for tags in (["1.4.0"], ["1.10.0-rc.1"], ["1.9.0"], [], ["1.10.0"]):
            git("commit", "-q", "--allow-empty", "-m", "Commit")
            for tag in tags:
                git("tag", "-a", tag, "-m", tag)
        # A tag that names no release: lightweight, or not x.y.z.
        git("tag", "1.9.5")
        git("tag", "-a", "v1.9.6", "-m", "v1.9.6")
        with (
            mock.patch.object(release_app, "ROOT", checkout),
            mock.patch.object(package_app, "ROOT", checkout),
        ):
            for version, previous in (
                ("1.4.0", None),
                ("1.9.0", ("1.4.0", "1")),
                ("1.10.0-rc.1", ("1.9.0", "3")),
                ("1.10.0", ("1.9.0", "3")),
                ("1.10.1", ("1.10.0", "5")),
            ):
                with self.subTest(version=version):
                    self.assertEqual(release_app.previous_release(version), previous)

    def signer(self, identity, *listed):
        """signer for identity, the keychains holding listed."""
        found = subprocess.CompletedProcess([], 0, identities(*listed), "")
        with mock.patch.object(release_app.subprocess, "run", return_value=found):
            return release_app.signer(identity)

    def test_a_release_signs_with_an_identity_the_keychains_hold(self):
        renewed = "FEDCBA9876543210FEDCBA9876543210FEDCBA98"
        development = "Apple Development: Jane Doe (VWXYZ67890)"
        held = (SHA1, IDENTITY), (renewed[::-1], development)
        self.assertEqual(self.signer(IDENTITY, *held), (IDENTITY, "ABCDE12345", None))
        # By its SHA-1, as when a renewed certificate and the one it renews,
        # both valid, have one name, which codesign would find ambiguous.
        self.assertEqual(self.signer(SHA1, *held), (IDENTITY, "ABCDE12345", None))
        name, team, problem = self.signer(IDENTITY, *held, (renewed, IDENTITY))
        self.assertEqual((name, team), (IDENTITY, "ABCDE12345"))
        self.assertRegex(problem, "^2 valid identities are named .* SHA-1")
        self.assertEqual(
            self.signer(SHA1, (renewed, IDENTITY), (SHA1, IDENTITY)),
            (IDENTITY, "ABCDE12345", None),
        )
        # One no keychain holds is printed as given, its team by its name.
        self.assertEqual(
            self.signer(IDENTITY),
            (IDENTITY, "ABCDE12345", f"no keychain holds the identity {IDENTITY}"),
        )
        self.assertEqual(
            self.signer(SHA1),
            (SHA1, "TEAMID", f"no keychain holds the identity {SHA1}"),
        )
        self.assertEqual(
            self.signer(renewed[::-1], *held),
            (
                development,
                "TEAMID",
                f"{renewed[::-1]} is {development}, not a Developer ID "
                "Application identity",
            ),
        )
        name, team, problem = release_app.signer(None)
        self.assertEqual((name, team), (release_app.UNSET_IDENTITY, "TEAMID"))
        self.assertIn("SIGN_IDENTITY", problem)

    def command_line(self, formula=True):
        """Leave the command line's files of release 1.5.0 in self.root/dist,
        the formula naming them unless formula is false; return them."""
        dist = self.root / "dist"
        dist.mkdir(exist_ok=True)
        files = [dist / ARCHIVE, dist / f"{ARCHIVE}.sha256", dist / BOTTLE]
        for path in files:
            path.write_text(f"{path.name}\n")
        digests = [package.digest(dist / ARCHIVE), package.digest(dist / BOTTLE)]
        (dist / "splash.rb").write_text(
            "".join(f'sha256 "{digest}"\n' for digest in digests[: 2 if formula else 1])
        )
        return files

    def test_a_release_carries_the_command_lines_files_of_its_version(self):
        with mock.patch.object(release_app, "ROOT", self.root):
            files, formula, problem = release_app.command_line("1.5.0", "26.4")
            self.assertEqual(
                problem,
                f"dist/ has no {ARCHIVE}, {ARCHIVE}.sha256, "
                "splash-1.5.0.*.bottle.tar.gz: make package and make package-bottle",
            )
            made = self.command_line()
            (self.root / "dist/splash-1.4.1.arm64_tahoe.bottle.tar.gz").touch()
            self.assertEqual(
                release_app.command_line("1.5.0", "26.4"),
                (made, self.root / "dist/splash.rb", None),
            )
            # A formula of other files, as an earlier release's.
            self.command_line(formula=False)
            self.assertRegex(
                release_app.command_line("1.5.0", "26.4")[2],
                f"^dist/splash.rb is not the formula of {re.escape(ARCHIVE)}",
            )

    def precheck(self, *findings, stdout=None):
        """precheck an app whose notary checks find findings, pairs of a level
        and an error; return what it printed."""
        app = self.root / "Splash.app"
        output = [
            {
                "SyspolicyCheckErrorFile": f"{app}/Contents/Resources/{n}",
                "SyspolicyCheckShortError": error,
                "SyspolicyCheckLongError": f"{error} in file {n}.",
                "SyspolicyCheckErrorLevel": level,
            }
            for n, (level, error) in enumerate(findings)
        ]
        checked = subprocess.CompletedProcess(
            [],
            70 if findings else 0,
            json.dumps({"output": output}) if stdout is None else stdout,
            "",
        )
        printed = io.StringIO()
        with (
            mock.patch.object(release_app.subprocess, "run", return_value=checked),
            redirect_stdout(printed),
        ):
            release_app.precheck(app)
        return printed.getvalue()

    def test_the_notary_services_checks_run_before_its_submission(self):
        # Its warnings are counted and accepted, as the notary service does.
        printed = self.precheck(
            ("Warning", "Incorrect Bundle Structure"),
            ("Warning", "Incorrect Bundle Structure"),
            ("Warning", "Adhoc Signed App"),
        )
        self.assertEqual(
            printed,
            "The notary service's checks of Splash.app (syspolicy_check):\n"
            "  Warning: Adhoc Signed App (1)\n"
            "  Warning: Incorrect Bundle Structure (2)\n",
        )
        self.assertIn("  nothing found\n", self.precheck(stdout=""))
        with self.assertRaisesRegex(
            RuntimeError,
            r"would reject Splash.app: .*/Contents/Resources/1: "
            r"No Hardened Runtime in file 1\.$",
        ):
            self.precheck(
                ("Warning", "Incorrect Bundle Structure"),
                ("Fatal", "No Hardened Runtime"),
            )

    def notarize(self, path, notary, status="Accepted"):
        """Notarize path with notary, the notary service answering status;
        return the commands run, paths by name, and what was printed."""
        log = self.root / "notary" / f"{path.name}.json"
        commands = self.commands = []

        def run(command, check, **options):
            commands.append(command)
            if command[:3] == ["xcrun", "notarytool", "submit"]:
                self.assertFalse(check)
                answer = {"id": "2efe2717", "status": status, "message": "Done"}
                return subprocess.CompletedProcess(command, 0, json.dumps(answer), "")
            self.assertTrue(check)
            if command[:3] == ["xcrun", "notarytool", "log"]:
                Path(command[-1]).write_text('{"issues": null}\n')
            return subprocess.CompletedProcess(command, 0)

        printed = io.StringIO()
        with (
            mock.patch.object(release_app.subprocess, "run", side_effect=run),
            redirect_stdout(printed),
        ):
            release_app.notarize(path, notary, log, release_app.Runner(False))
        self.assertEqual(log.read_text(), '{"issues": null}\n')
        return recorded(commands, path, log), printed.getvalue()

    def test_the_app_and_its_disk_image_are_notarized_and_stapled(self):
        # The app goes zipped, as notarytool takes it; with either kind of
        # credentials, each submission's log is kept.
        app = self.root / "Splash.app"
        app.mkdir()
        image = self.root / "Splash-1.5.0.dmg"
        image.write_bytes(b"disk image")
        for notary in (PROFILE, API_KEY):
            with self.subTest(notary=notary):
                commands, printed = self.notarize(app, notary)
                self.assertEqual(
                    commands,
                    [
                        ["ditto", "-c", "-k", "--keepParent", "Splash.app"]
                        + ["Splash.zip"],
                        ["xcrun", "notarytool", "submit", "Splash.zip", *notary]
                        + ["--wait", "--output-format", "json"],
                        ["xcrun", "notarytool", "log", "2efe2717", *notary]
                        + ["Splash.app.json"],
                        ["xcrun", "stapler", "staple", "Splash.app"],
                    ],
                )
                self.assertTrue(printed.startswith("Notarizing Splash.app:\n"))
                commands, _ = self.notarize(image, notary)
                self.assertEqual(
                    commands,
                    [
                        ["xcrun", "notarytool", "submit", "Splash-1.5.0.dmg", *notary]
                        + ["--wait", "--output-format", "json"],
                        ["xcrun", "notarytool", "log", "2efe2717", *notary]
                        + ["Splash-1.5.0.dmg.json"],
                        ["xcrun", "stapler", "staple", "Splash-1.5.0.dmg"],
                    ],
                )

    def test_a_rejected_submission_stops_the_release_with_its_log(self):
        image = self.root / "Splash-1.5.0.dmg"
        image.write_bytes(b"disk image")
        with self.assertRaisesRegex(
            RuntimeError, r'found Splash-1.5.0.dmg Invalid: Done\n\{"issues": null\}'
        ):
            self.notarize(image, PROFILE, status="Invalid")
        self.assertEqual(
            [command[:3] for command in self.commands],
            [["xcrun", "notarytool", "submit"], ["xcrun", "notarytool", "log"]],
        )

    def test_a_dry_run_prints_the_notarization_and_runs_none_of_it(self):
        # The log's temporary folder is printed as the one it becomes.
        staged = self.root / ".1.5.0-dry-run-x1y2"
        app = self.root / "Splash.app"
        app.mkdir()
        log = staged / "notary/Splash.app.json"
        printed = io.StringIO()
        run = release_app.Runner(True)
        run.shown[str(staged)] = "/release/1.5.0"
        with (
            mock.patch.object(
                release_app.subprocess,
                "run",
                side_effect=AssertionError("ran in a dry run"),
            ),
            redirect_stdout(printed),
        ):
            release_app.notarize(app, list(release_app.UNSET_NOTARY), log, run)
        lines = printed.getvalue().splitlines()
        self.assertEqual(lines[0], "Would notarize Splash.app:")
        self.assertEqual(
            recorded([shlex.split(line) for line in lines[1:]], app),
            [
                ["ditto", "-c", "-k", "--keepParent", "Splash.app", "Splash.zip"],
                ["xcrun", "notarytool", "submit", "Splash.zip"]
                + ["--keychain-profile", "NOTARY_PROFILE"]
                + ["--wait", "--output-format", "json"],
                ["xcrun", "notarytool", "log", "SUBMISSION_ID"]
                + ["--keychain-profile", "NOTARY_PROFILE"]
                + ["/release/1.5.0/notary/Splash.app.json"],
                ["xcrun", "stapler", "staple", "Splash.app"],
            ],
        )
        self.assertFalse(log.parent.exists())

    def verify(self, dry_run, info=None, team="ABCDE12345", assessed=None):
        """Verify a disk image whose app has info (the right one by default),
        is signed by team and, for spctl, has the assessment assessed; return
        the commands run, the image and its app by name, and what was
        printed."""
        volume = self.root / "volume"
        app = volume / "Splash.app"
        (app / "Contents").mkdir(parents=True, exist_ok=True)
        right = {
            "CFBundleIdentifier": "ai.inco.splash",
            "CFBundleShortVersionString": "1.5.0",
            "CFBundleVersion": "1700",
            "SUFeedURL": package_app.APPCAST_URL,
            "SUPublicEDKey": KEY,
        }
        with (app / "Contents/Info.plist").open("wb") as stream:
            plistlib.dump(info or right, stream)
        image = self.root / "Splash-1.5.0.dmg"
        commands = []

        def run(command, check=True, **options):
            commands.append(command)
            if command[:2] == ["codesign", "-dv"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    "",
                    f"Identifier=ai.inco.splash\nTeamIdentifier={team}\n",
                )
            if command[0] == "spctl":
                return subprocess.CompletedProcess(
                    command,
                    0,
                    "",
                    assessed
                    or f"{command[-1]}: accepted\nsource=Notarized Developer ID\n",
                )
            return subprocess.CompletedProcess(command, 0)

        @contextmanager
        def mounted(path):
            self.assertEqual(path, image)
            yield volume

        printed = io.StringIO()
        with (
            mock.patch.object(release_app, "mounted", mounted),
            mock.patch.object(release_app.subprocess, "run", side_effect=run),
            mock.patch.object(package_app, "build_number", return_value="1700"),
            mock.patch.object(package_app, "SPARKLE_PUBLIC_KEY", KEY),
            redirect_stdout(printed),
        ):
            release_app.verify(
                image, "1.5.0", "ABCDE12345", release_app.Runner(dry_run)
            )
        return recorded(commands, image, app), printed.getvalue()

    def test_a_release_is_checked_as_a_mac_that_downloads_it_would(self):
        commands, printed = self.verify(dry_run=False)
        self.assertEqual(
            commands,
            [
                ["codesign", "--verify", "--deep", "--strict", "Splash.app"],
                ["codesign", "-dv", "--verbose=2", "Splash.app"],
                ["spctl", "--assess", "--type", "exec", "-vv", "Splash.app"],
                ["spctl", "--assess", "--type", "open"]
                + ["--context", "context:primary-signature", "-vv"]
                + ["Splash-1.5.0.dmg"],
                ["xcrun", "stapler", "validate", "Splash.app"],
                ["xcrun", "stapler", "validate", "Splash-1.5.0.dmg"],
                ["syspolicy_check", "distribution", "Splash.app"],
            ],
        )
        self.assertIn("in Splash-1.5.0.dmg is signed by team ABCDE12345.", printed)

    def test_a_release_of_another_team_or_version_is_not_ready(self):
        # The team that signed it is SIGN_IDENTITY's, so that a personal
        # release and an organization's cannot be mistaken for each other.
        with self.assertRaisesRegex(RuntimeError, "by team FGHIJ67890, not ABCDE"):
            self.verify(dry_run=False, team="FGHIJ67890")
        for field, value in (
            ("CFBundleShortVersionString", "1.4.1"),
            ("CFBundleVersion", "1699"),
            ("CFBundleIdentifier", "ai.inco.splash.beta"),
            ("SUPublicEDKey", None),
        ):
            info = {
                "CFBundleIdentifier": "ai.inco.splash",
                "CFBundleShortVersionString": "1.5.0",
                "CFBundleVersion": "1700",
                "SUFeedURL": package_app.APPCAST_URL,
                "SUPublicEDKey": KEY,
                field: value,
            }
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(RuntimeError, "Splash.app has"),
            ):
                self.verify(
                    dry_run=False,
                    info={name: entry for name, entry in info.items() if entry},
                )
        # Accepted, but signed and not notarized.
        with self.assertRaisesRegex(RuntimeError, "takes no notarized Splash.app"):
            self.verify(dry_run=False, assessed="accepted\nsource=Developer ID\n")

    def test_a_dry_run_checks_what_an_ad_hoc_build_allows(self):
        # The folder the disk image is attached at for a moment is printed as
        # what it is.
        commands, printed = self.verify(dry_run=True, team="not set")
        self.assertEqual(
            commands,
            [
                ["codesign", "--verify", "--deep", "--strict", "Splash.app"],
                ["codesign", "-dv", "--verbose=2", "Splash.app"],
            ],
        )
        self.assertIn(
            "Splash.app 1.5.0 (1700) in Splash-1.5.0.dmg is signed ad hoc.", printed
        )
        self.assertIn(
            "  syspolicy_check distribution MOUNTED_IMAGE/Splash.app\n", printed
        )
        self.assertIn("  xcrun stapler validate MOUNTED_IMAGE/Splash.app\n", printed)

    def test_sha256sums_lists_each_file_as_shasum_does(self):
        image = self.root / "Splash-1.5.0.dmg"
        image.write_bytes(b"disk image")
        latest = self.root / "Splash.dmg"
        os.link(image, latest)
        shasum = subprocess.run(
            ["shasum", "-a", "256", image.name, latest.name],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(release_app.checksums(image, latest), shasum)

    def test_the_cask_installs_the_releases_disk_image(self):
        checksum = "a" * 64
        self.assertEqual(
            release_app.cask("1.5.0", checksum, "26.4"),
            f"""cask "splash-app" do
  version "1.5.0"
  sha256 "{checksum}"

  url "https://github.com/incoai/splash/releases/download/#{{version}}/Splash-#{{version}}.dmg"
  name "Splash"
  desc "Local inference engine for Apple silicon, built around the model"
  homepage "https://github.com/incoai/splash"

  livecheck do
    url :url
    strategy :github_latest
  end

  auto_updates true
  depends_on arch: :arm64
  depends_on macos: :tahoe

  app "Splash.app"

  uninstall quit: "ai.inco.splash"

  zap trash: [
    "~/Library/Application Support/Splash/Chats",
    "~/Library/Caches/ai.inco.splash",
    "~/Library/Caches/Splash",
    "~/Library/HTTPStorages/ai.inco.splash",
    "~/Library/Logs/Splash",
    "~/Library/Preferences/ai.inco.splash.plist",
    "~/Library/Saved Application State/ai.inco.splash.savedState",
    "~/Library/WebKit/ai.inco.splash",
  ]

  caveats "Splash needs an Apple M3 or newer and macOS 26.4 or later."
end
""",
        )

    def test_the_casks_zap_keeps_what_the_command_line_shares(self):
        # Its models, locks and runtime in Application Support/Splash, which
        # the app's chats share, and the Hugging Face cache; zap lists each
        # path in the order brew style keeps.
        zapped = re.findall(
            r'^    "(.+)",$', release_app.cask("1.5.0", "0", "26.4"), re.M
        )
        self.assertEqual(zapped, sorted(zapped, key=str.lower))
        self.assertNotIn("~/Library/Application Support/Splash", zapped)
        self.assertFalse(any("huggingface" in path for path in zapped))

    def test_the_casks_macos_is_the_minimum_macos_major_release(self):
        # Tahoe is macOS 26; a cask names no minor release.
        makefile = (package_app.ROOT / "Makefile").read_text()
        version = re.search(r"^MACOS_MIN_VERSION := (\S+)$", makefile, re.M)[1]
        self.assertEqual(version.split(".")[0], "26")
        self.assertIn(
            "depends_on macos: :tahoe\n", release_app.cask("1.5.0", "0", version)
        )

    def test_publishing_is_printed_never_run(self):
        # From the release folder, by the names of its files, so that another
        # Mac can publish a copy of it.
        folder = Path("/splash/dist/release/1.5.0")
        assets = ["SHA256SUMS", "Splash-1.5.0.dmg", "Splash.dmg", "appcast.xml"]
        commands = [
            line
            for line in release_app.publishing("1.5.0", folder, assets)
            if not line.startswith("#")
        ]
        self.assertEqual(
            commands,
            [
                "git push origin refs/tags/1.5.0",
                "cd /splash/dist/release/1.5.0",
                "gh release create 1.5.0 --repo incoai/splash --draft --verify-tag "
                "--title 1.5.0 --notes-file notes.md",
                "gh release upload 1.5.0 --repo incoai/splash "
                "SHA256SUMS Splash-1.5.0.dmg Splash.dmg appcast.xml",
                "gh release edit 1.5.0 --repo incoai/splash --draft=false --latest",
                'cp -R Formula Casks "$(brew --repository incoai/tap)"',
                'cd "$(brew --repository incoai/tap)" && git switch -c splash-1.5.0',
                "brew audit --strict --online incoai/tap/splash",
                "brew audit --cask --strict --online incoai/tap/splash-app",
                "git add Formula/splash.rb Casks/splash-app.rb && "
                "git commit -m 'splash 1.5.0'",
                "git push -u origin splash-1.5.0 && gh pr create --fill",
                "brew install --cask incoai/tap/splash-app",
            ],
        )
        # A release candidate is a prerelease, never the latest, and leaves
        # the tap alone.
        candidate = release_app.publishing("1.5.0-rc.1", folder, assets)
        self.assertIn(
            "gh release create 1.5.0-rc.1 --repo incoai/splash --draft --verify-tag "
            "--title 1.5.0-rc.1 --notes-file notes.md --prerelease",
            candidate,
        )
        self.assertIn(
            "gh release edit 1.5.0-rc.1 --repo incoai/splash --draft=false",
            candidate,
        )
        self.assertFalse(any("brew" in line for line in candidate))

    def release(
        self, version, *arguments, problems=(), keychain=IDENTITY, checked=None
    ):
        """Run main for version, everything but its own steps mocked, in a
        checkout at self.root whose keychains hold keychain, or nothing for
        None, the appcast's check raising checked; return the release folder,
        what was printed and the steps taken, in order."""
        steps = []
        (self.root / "dist").mkdir(exist_ok=True)

        def build_app(dist, version, macos_min, identity):
            steps.append(("build_app", version, identity))
            app = dist / "Splash.app"
            app.mkdir(exist_ok=True)
            return app

        def precheck(app):
            steps.append(("precheck", app.name))

        def build_image(app, image, identity):
            steps.append(("build_image", identity))
            image.write_bytes(b"disk image")

        def notarize(path, notary, log, run):
            steps.append(("notarize", path.name, tuple(notary), run.dry_run))
            if not run.dry_run:
                log.parent.mkdir(exist_ok=True)
                log.write_text("{}\n")
                if path.suffix == ".dmg":
                    path.write_bytes(b"stapled disk image")

        def verify(image, version, team, run):
            steps.append(("verify", image.read_bytes(), team, run.dry_run))

        def generate(image, notes, version, key_file):
            steps.append(("appcast", image.read_bytes(), key_file))
            return b"<rss/>\n"

        def check(feed, image, version, macos_min, build, previous):
            steps.append(
                ("check", feed, image.name, version, macos_min, build, previous)
            )
            if checked:
                raise checked

        def previous_release(version):
            steps.append(("previous_release", version))
            return ("1.4.1", "1600")

        def keychain_identity(identity):
            if not keychain:
                raise LookupError(f"no keychain holds the identity {identity}")
            return keychain

        with (
            mock.patch.object(release_app, "ROOT", self.root),
            mock.patch.object(
                release_app, "checkout_problems", return_value=list(problems)
            ),
            mock.patch.object(
                release_app, "keychain_identity", side_effect=keychain_identity
            ),
            mock.patch.object(package_app, "build_app", side_effect=build_app),
            mock.patch.object(release_app, "precheck", side_effect=precheck),
            mock.patch.object(package_app, "build_image", side_effect=build_image),
            mock.patch.object(release_app, "notarize", side_effect=notarize),
            mock.patch.object(release_app, "verify", side_effect=verify),
            mock.patch.object(appcast, "generate", side_effect=generate),
            mock.patch.object(appcast, "check", side_effect=check),
            mock.patch.object(package_app, "build_number", return_value="1700"),
            mock.patch.object(
                release_app, "previous_release", side_effect=previous_release
            ),
            redirect_stdout(io.StringIO()) as printed,
        ):
            release_app.main(
                ["--version", version, "--macos-min", "26.4"]
                + ["--notes", str(self.notes), *arguments]
            )
        return self.root / "dist/release" / version, printed.getvalue(), steps

    def files(self, folder):
        return sorted(
            str(path.relative_to(folder))
            for path in folder.rglob("*")
            if path.is_file()
        )

    def test_a_dry_run_writes_a_folder_of_its_own_and_lists_what_would_stop(self):
        # A release's folder, there or not, is left alone, and an earlier dry
        # run is replaced; each refusal is listed at its start and its end.
        release = self.root / "dist/release/1.5.0"
        release.mkdir(parents=True)
        earlier = self.root / "dist/release/1.5.0-dry-run"
        earlier.mkdir()
        (earlier / "Splash-1.5.0.dmg").write_text("earlier\n")
        problems = ["the checkout has changes: notes.md", "there is no tag"]
        with mock.patch.object(package_app, "SPARKLE_PUBLIC_KEY", ""):
            folder, printed, steps = self.release(
                "1.5.0", "--dry-run", problems=problems, keychain=None
            )
        self.assertEqual(list(release.iterdir()), [])
        unset = tuple(release_app.UNSET_NOTARY)
        self.assertEqual(
            steps,
            [
                ("build_app", "1.5.0", "-"),
                ("precheck", "Splash.app"),
                ("notarize", "Splash.app", unset, True),
                ("build_image", "-"),
                ("notarize", "Splash-1.5.0.dmg", unset, True),
                ("verify", b"disk image", "TEAMID", True),
            ],
        )
        self.assertEqual(
            self.files(earlier),
            [
                "Casks/splash-app.rb",
                "SHA256SUMS",
                "Splash-1.5.0.dmg",
                "Splash.dmg",
                "notes.md",
            ],
        )
        self.assertTrue((earlier / "Splash.dmg").samefile(earlier / "Splash-1.5.0.dmg"))
        self.assertEqual(
            (earlier / "SHA256SUMS").read_text(),
            release_app.checksums(earlier / "Splash-1.5.0.dmg", earlier / "Splash.dmg"),
        )
        self.assertEqual((earlier / "notes.md").read_text(), "Notes\n")
        self.assertIn(
            f'sha256 "{package.digest(earlier / "Splash.dmg")}"',
            (earlier / "Casks/splash-app.rb").read_text(),
        )
        self.assertEqual(
            sorted(path.name for path in release.parent.iterdir()),
            ["1.5.0", "1.5.0-dry-run"],
        )
        refusals = [
            "the checkout has changes: notes.md",
            "there is no tag",
            "SIGN_IDENTITY (--sign), a Developer ID Application identity, is not set",
            "there are no notary credentials: NOTARY_PROFILE, or NOTARY_KEY, "
            "NOTARY_KEY_ID and NOTARY_ISSUER",
            "SPARKLE_PUBLIC_KEY in dev/tools/package_app.py is empty, and apps "
            "built without it check for no update",
            f"{release} exists",
            f"dist/ has no {ARCHIVE}, {ARCHIVE}.sha256, "
            "splash-1.5.0.*.bottle.tar.gz: make package and make package-bottle",
        ]
        stops = [f"A release would stop here: {refusal}." for refusal in refusals]
        lines = printed.splitlines()
        self.assertEqual(
            lines[0],
            f"Dry run of release 1.5.0, signed ad hoc, into {earlier}; nothing is "
            "notarized, stapled or published.",
        )
        self.assertEqual(lines[1 : 1 + len(stops)], stops)
        ending = lines.index(
            "Signed ad hoc; a release would be signed by team TEAMID, "
            "Developer ID Application: NAME (TEAMID)."
        )
        self.assertEqual(lines[ending + 1 : ending + 1 + len(stops)], stops)
        for line in (
            "  codesign --force --sign 'Developer ID Application: NAME (TEAMID)' "
            f"--timestamp --options runtime {self.root}/dist/Splash.app",
            "Would sign Splash-1.5.0.dmg:",
            "  codesign --force --sign 'Developer ID Application: NAME (TEAMID)' "
            f"--timestamp --identifier ai.inco.splash.dmg {release}/Splash-1.5.0.dmg",
            "Skipped appcast.xml: SPARKLE_PUBLIC_KEY in dev/tools/package_app.py "
            "is empty.",
            "Publishing release 1.5.0, made without DRY_RUN, would take:",
            f"  cd {release}",
            "  gh release upload 1.5.0 --repo incoai/splash SHA256SUMS "
            "Splash-1.5.0.dmg Splash.dmg",
        ):
            self.assertIn(line, lines)
        self.assertFalse(any(".1.5.0-dry-run-" in line for line in lines))

    def test_a_release_staples_the_app_before_its_disk_image_is_built(self):
        # And the appcast signs the disk image as stapled, which is what the
        # checksums and the cask describe too. A release carries the command
        # line's files and puts its formula beside the cask; a release
        # candidate carries the app alone.
        arguments = ["--sign", IDENTITY, "--notary-profile", "splash-notary"]
        self.command_line()
        for version in ("1.5.0", "1.5.0-rc.1"):
            with (
                self.subTest(version=version),
                mock.patch.object(package_app, "SPARKLE_PUBLIC_KEY", KEY),
            ):
                folder, printed, steps = self.release(version, *arguments)
                image = f"Splash-{version}.dmg"
                self.assertEqual(
                    steps,
                    [
                        ("build_app", version, IDENTITY),
                        ("precheck", "Splash.app"),
                        ("notarize", "Splash.app", tuple(PROFILE), False),
                        ("build_image", IDENTITY),
                        ("notarize", image, tuple(PROFILE), False),
                        ("verify", b"stapled disk image", "ABCDE12345", False),
                        ("appcast", b"stapled disk image", None),
                        ("previous_release", version),
                        (
                            "check",
                            b"<rss/>\n",
                            image,
                            version,
                            "26.4",
                            "1700",
                            ("1.4.1", "1600"),
                        ),
                    ],
                )
                files = [image, "SHA256SUMS", "Splash.dmg", "appcast.xml", "notes.md"]
                files += ["notary/Splash.app.json", f"notary/{image}.json"]
                listed = [image, "Splash.dmg"]
                if "-rc." not in version:
                    files += [ARCHIVE, f"{ARCHIVE}.sha256", BOTTLE]
                    files += ["Casks/splash-app.rb", "Formula/splash.rb"]
                    listed += [ARCHIVE, BOTTLE]
                    self.assertTrue(
                        (folder / BOTTLE).samefile(self.root / "dist" / BOTTLE)
                    )
                    self.assertEqual(
                        (folder / "Formula/splash.rb").read_text(),
                        (self.root / "dist/splash.rb").read_text(),
                    )
                self.assertEqual(self.files(folder), sorted(files))
                self.assertEqual(
                    (folder / "SHA256SUMS").read_text(),
                    release_app.checksums(*sorted(folder / name for name in listed)),
                )
                self.assertIn(
                    f"Release {version} is signed by team ABCDE12345, {IDENTITY}.\n",
                    printed,
                )
                uploaded = sorted(
                    path for path in files if "/" not in path and path != "notes.md"
                )
                self.assertIn(
                    f"  gh release upload {version} --repo incoai/splash "
                    f"{' '.join(uploaded)}\n",
                    printed,
                )

    def test_a_release_names_the_identity_the_keychain_holds_for_its_sha1(self):
        arguments = ["--sign", SHA1, "--notary-profile", "splash-notary"]
        self.command_line()
        with mock.patch.object(package_app, "SPARKLE_PUBLIC_KEY", KEY):
            _, printed, steps = self.release("1.5.0", *arguments)
        self.assertEqual(steps[0], ("build_app", "1.5.0", SHA1))
        self.assertIn(("verify", b"stapled disk image", "ABCDE12345", False), steps)
        self.assertIn(f"is signed by team ABCDE12345, {IDENTITY}.\n", printed)

    def test_a_release_refuses_what_would_keep_it_from_being_whole(self):
        # Every reason at once, before anything is built.
        with (
            mock.patch.object(package_app, "SPARKLE_PUBLIC_KEY", ""),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as refused,
        ):
            self.release(
                "1.5.0",
                "--sign",
                IDENTITY,
                problems=["the checkout has changes: notes.md"],
                keychain=None,
            )
        self.assertEqual(
            refused.exception.code.splitlines(),
            [
                "error: the checkout has changes: notes.md",
                f"error: no keychain holds the identity {IDENTITY}",
                "error: there are no notary credentials: NOTARY_PROFILE, or "
                "NOTARY_KEY, NOTARY_KEY_ID and NOTARY_ISSUER",
                "error: SPARKLE_PUBLIC_KEY in dev/tools/package_app.py is empty, "
                "and apps built without it check for no update",
                f"error: dist/ has no {ARCHIVE}, {ARCHIVE}.sha256, "
                "splash-1.5.0.*.bottle.tar.gz: make package and make package-bottle",
            ],
        )
        self.assertFalse((self.root / "dist/release").exists())
        # One already made is not made again.
        (self.root / "dist/release/1.5.0").mkdir(parents=True)
        self.command_line()
        with (
            mock.patch.object(package_app, "SPARKLE_PUBLIC_KEY", KEY),
            self.assertRaisesRegex(SystemExit, "^error: .*/dist/release/1.5.0 exists$"),
        ):
            self.release("1.5.0", "--sign", IDENTITY, "--notary-profile", "splash")

    def test_a_release_that_stops_keeps_what_it_made(self):
        # Its notary logs among it, which a second run would not bring back.
        self.command_line()
        with (
            mock.patch.object(package_app, "SPARKLE_PUBLIC_KEY", KEY),
            redirect_stderr(io.StringIO()) as stderr,
            self.assertRaisesRegex(RuntimeError, "not signed"),
        ):
            self.release(
                "1.5.0",
                "--sign",
                IDENTITY,
                "--notary-profile",
                "splash-notary",
                checked=RuntimeError("not signed"),
            )
        (kept,) = (self.root / "dist/release").iterdir()
        self.assertTrue(kept.name.startswith(".1.5.0-"))
        self.assertIn("notary/Splash-1.5.0.dmg.json", self.files(kept))
        self.assertEqual(stderr.getvalue(), f"Stopped; what it made is in {kept}.\n")

    def test_the_release_script_takes_the_arguments_make_passes(self):
        variables = (
            "RELEASE_VERSION=1.5.0",
            f"RELEASE_NOTES={self.notes}",
            f"SIGN_IDENTITY={IDENTITY}",
        )
        api_key = (
            f"NOTARY_KEY={self.key}",
            "NOTARY_KEY_ID=KEYID",
            "NOTARY_ISSUER=ISSUER",
        )
        for extra, dry_run, notary in (
            (("NOTARY_PROFILE=splash-notary",), False, PROFILE),
            (api_key, False, ["--key", str(self.key), "--key-id", "KEYID"]),
            (("NOTARY_PROFILE=splash-notary", "DRY_RUN=1"), True, PROFILE),
            (("NOTARY_PROFILE=splash-notary", "DRY_RUN=0"), False, PROFILE),
        ):
            with self.subTest(extra=extra):
                result = subprocess.run(
                    ["make", "--no-print-directory", "-n", "release-app"]
                    + [*variables, *extra],
                    cwd=package_app.ROOT,
                    env={
                        key: value
                        for key, value in os.environ.items()
                        if key not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL")
                    },
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                # One line, which make echoes as it runs it.
                (line,) = [
                    line
                    for line in result.stdout.splitlines()
                    if "dev/tools/release_app.py" in line
                ]
                command = shlex.split(line)
                args = release_app.parse_args(
                    command[command.index("dev/tools/release_app.py") + 1 :]
                )
                self.assertEqual(
                    (args.version, args.macos_min, args.sign, args.notes),
                    ("1.5.0", "26.4", IDENTITY, self.notes),
                )
                self.assertEqual(args.dry_run, dry_run)
                self.assertEqual(release_app.notary_options(args)[:4], notary)


if __name__ == "__main__":
    unittest.main()
