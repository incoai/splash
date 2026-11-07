import base64
import io
import os
import plistlib
import re
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from dev.tools import package_app

THIN_ARM64 = bytes.fromhex("cffaedfe0c000001")
# The release key's public half, SPARKLE_PUBLIC_KEY, as generate_keys printed
# it once (docs/package.md, Updates).
RELEASE_KEY = "QAOqAR9iWlOPmkqq0XgbaAPjP8Y/X0lJwfd+W4a9kSc="
SPARKLE = "APP/Contents/Frameworks/Sparkle.framework"


class PackageAppTests(unittest.TestCase):
    def test_the_app_builds_for_the_minimum_macos_its_bundle_declares(self):
        # LSMinimumSystemVersion is MACOS_MIN_VERSION; the Swift package
        # cannot read the Makefile, so it repeats the version.
        makefile = (package_app.ROOT / "Makefile").read_text()
        version = re.search(r"^MACOS_MIN_VERSION := (\S+)$", makefile, re.M)[1]
        manifest = (package_app.ROOT / "app/Package.swift").read_text()
        self.assertIn(f'platforms: [.macOS("{version}")]', manifest)

    def test_bundle_short_version_is_the_release_versions_x_y_z(self):
        for version, short in (
            ("1.4.1", "1.4.1"),
            ("1.4.1-36-g6e9dfe3f", "1.4.1"),
            ("10.20.300", "10.20.300"),
            ("6e9dfe3f", None),
            ("1.4", None),
            ("v1.4.1", None),
            ("1.4.1.2", None),
            ("1.4.1rc1", None),
            ("1.4.1-", None),
            ("1.4.1-../x", None),
        ):
            with self.subTest(version=version):
                self.assertEqual(package_app.short_version(version), short)

    def test_the_build_number_counts_commits_and_needs_the_whole_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, clone = Path(temporary) / "source", Path(temporary) / "clone"
            source.mkdir()
            environment = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}

            def git(folder, *arguments):
                subprocess.run(
                    ["git", "-c", "user.name=Test", "-c", "user.email=t@example.com"]
                    + list(arguments),
                    cwd=folder,
                    env={**os.environ, **environment},
                    check=True,
                    capture_output=True,
                )

            git(source, "init", "-q")
            for n in range(3):
                git(source, "commit", "-q", "--allow-empty", "-m", f"Commit {n}")
            git(source, "tag", "-a", "1.5.0", "-m", "1.5.0", "HEAD~1")
            with mock.patch.object(package_app, "ROOT", source):
                self.assertEqual(package_app.build_number(), "3")
                self.assertEqual(package_app.build_number("1.5.0"), "2")
            # A shallow clone would count 1, below every release's build.
            git(temporary, "clone", "-q", "--depth", "1", source.as_uri(), "clone")
            with (
                mock.patch.object(package_app, "ROOT", clone),
                self.assertRaisesRegex(RuntimeError, "shallow clone.*--unshallow"),
            ):
                package_app.build_number()

    def test_info_plist_describes_the_bundle_assemble_lays_out(self):
        # Without a release key; the next test adds the keys that come with one.
        with mock.patch.object(package_app, "SPARKLE_PUBLIC_KEY", ""):
            info = package_app.info_plist("1.4.1-36-g6e9dfe3f", "1505", "26.4")
        info = plistlib.loads(plistlib.dumps(info))
        self.assertEqual(
            info,
            {
                "CFBundleIdentifier": "ai.inco.splash",
                "CFBundleName": "Splash",
                "CFBundleDisplayName": "Splash",
                "CFBundleExecutable": "Splash",
                "CFBundlePackageType": "APPL",
                "CFBundleShortVersionString": "1.4.1",
                "CFBundleVersion": "1505",
                "CFBundleIconFile": "AppIcon",
                "CFBundleIconName": "AppIcon",
                "LSMinimumSystemVersion": "26.4",
                "LSApplicationCategoryType": "public.app-category.developer-tools",
                "NSHighResolutionCapable": True,
                "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
                "NSHumanReadableCopyright": "Copyright © Inco AI",
                "SUFeedURL": "https://github.com/incoai/splash/releases/latest/"
                "download/appcast.xml",
            },
        )

    def test_the_resource_bundle_is_the_one_the_app_reads(self):
        # SwiftPM names it after the package and its target, and the app looks
        # for it by that name in Contents/Resources.
        source = (package_app.ROOT / "app/Sources/Splash/WebPage.swift").read_text()
        name, extension = package_app.RESOURCE_BUNDLE.rsplit(".", 1)
        self.assertIn(
            f'Bundle.main.url(forResource: "{name}", withExtension: "{extension}")',
            source,
        )

    def test_the_resource_bundle_goes_into_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            products = root / "products"
            page = (
                products / package_app.RESOURCE_BUNDLE / "Contents/Resources/Transcript"
            )
            page.mkdir(parents=True)
            (page / "transcript.html").write_text("<!doctype html>\n")
            (products / "Splash").write_bytes(THIN_ARM64)
            resources = root / "Splash.app/Contents/Resources"
            resources.mkdir(parents=True)
            package_app.copy_resources(products, resources)
            self.assertEqual(
                (
                    resources
                    / package_app.RESOURCE_BUNDLE
                    / "Contents/Resources/Transcript/transcript.html"
                ).read_text(),
                "<!doctype html>\n",
            )
            self.assertEqual(
                sorted(path.name for path in (root / "Splash.app").iterdir()),
                ["Contents"],
            )

    def test_the_updater_checks_only_with_the_release_keys_public_half(self):
        # Without SUPublicEDKey the app leaves Sparkle off (Updates.swift);
        # with it, Sparkle checks a download's signature before it opens it.
        with mock.patch.object(package_app, "SPARKLE_PUBLIC_KEY", ""):
            info = package_app.info_plist("1.5.0", "1600", "26.4")
        self.assertNotIn("SUPublicEDKey", info)
        self.assertNotIn("SUVerifyUpdateBeforeExtraction", info)
        key = base64.b64encode(bytes(range(32))).decode()
        with mock.patch.object(package_app, "SPARKLE_PUBLIC_KEY", key):
            info = package_app.info_plist("1.5.0", "1600", "26.4")
        self.assertEqual(info["SUPublicEDKey"], key)
        self.assertIs(info["SUVerifyUpdateBeforeExtraction"], True)
        # Sparkle asks on the second launch whether to check by itself.
        self.assertNotIn("SUEnableAutomaticChecks", info)

    def test_the_sparkle_public_key_is_the_release_keys(self):
        # Every app shipped holds it and takes only updates its private half
        # signed; Sparkle replaces it only from a disk image the app's own
        # team signed, so a new key strands every app of an earlier team
        # (docs/package.md, Moving to the organization account). RELEASE_KEY
        # pins it, to be changed on purpose only.
        self.assertEqual(package_app.SPARKLE_PUBLIC_KEY, RELEASE_KEY)
        # As generate_keys prints it: 32 bytes in base64. Sparkle refuses to
        # start with any other.
        if RELEASE_KEY:
            self.assertEqual(len(base64.b64decode(RELEASE_KEY, validate=True)), 32)

    def test_mach_o_files_are_found_by_content_not_by_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "python/bin").mkdir(parents=True)
            (root / "engine").mkdir()
            (root / "lib").mkdir()
            binaries = []
            for name, magic in (
                ("python/bin/python3.13", "cffaedfe"),
                ("engine/splash", "cefaedfe"),
                ("lib/big-endian.so", "feedfacf"),
                ("lib/big-endian-32.so", "feedface"),
                ("lib/universal.dylib", "cafebabe"),
                ("lib/universal-64.dylib", "cafebabf"),
            ):
                (root / name).write_bytes(bytes.fromhex(magic) + b"\0" * 28)
                binaries.append(root / name)
            (root / "python/bin/python3").symlink_to("python3.13")
            (root / "python/bin/hf").write_text("#!/usr/bin/env python3\n")
            (root / "lib/module.so.txt").write_text("not a library")
            (root / "lib/short").write_bytes(b"\xcf\xfa")
            (root / "lib/empty").write_bytes(b"")
            (root / "lib/folder.so").mkdir()
            self.assertEqual(package_app.mach_o_files(root), sorted(binaries))

    def test_sparkle_is_copied_with_its_links_and_without_its_xpc_services(self):
        with tempfile.TemporaryDirectory() as temporary:
            framework = Path(temporary) / "products/Sparkle.framework"
            version = framework / "Versions/B"
            for folder in ("Resources", "Updater.app/Contents/MacOS"):
                (version / folder).mkdir(parents=True)
            for folder in ("Installer.xpc", "Downloader.xpc"):
                (version / "XPCServices" / folder / "Contents").mkdir(parents=True)
            for name in ("Sparkle", "Autoupdate", "Updater.app/Contents/MacOS/Updater"):
                (version / name).write_bytes(THIN_ARM64)
            (framework / "Versions/Current").symlink_to("B")
            for name in ("Sparkle", "Autoupdate", "Updater.app", "Resources"):
                (framework / name).symlink_to(f"Versions/Current/{name}")
            (framework / "XPCServices").symlink_to("Versions/Current/XPCServices")
            frameworks = Path(temporary) / "Splash.app/Contents/Frameworks"
            frameworks.parent.mkdir(parents=True)

            package_app.add_sparkle(frameworks, framework)

            copy = frameworks / "Sparkle.framework"
            self.assertEqual(
                sorted(
                    str(path.relative_to(copy))
                    + (" -> " + str(path.readlink()) if path.is_symlink() else "")
                    for path in copy.rglob("*")
                ),
                [
                    "Autoupdate -> Versions/Current/Autoupdate",
                    "Resources -> Versions/Current/Resources",
                    "Sparkle -> Versions/Current/Sparkle",
                    "Updater.app -> Versions/Current/Updater.app",
                    "Versions",
                    "Versions/B",
                    "Versions/B/Autoupdate",
                    "Versions/B/Resources",
                    "Versions/B/Sparkle",
                    "Versions/B/Updater.app",
                    "Versions/B/Updater.app/Contents",
                    "Versions/B/Updater.app/Contents/MacOS",
                    "Versions/B/Updater.app/Contents/MacOS/Updater",
                    "Versions/Current -> B",
                ],
            )

    def compile_icon(self, stdout, written=(), returncode=0):
        """Run compile_icon with actool printing stdout, writing the files
        written and exiting with returncode."""
        with tempfile.TemporaryDirectory() as temporary:
            resources = Path(temporary)

            def actool(command, **options):
                self.assertEqual(options, {"stdout": subprocess.PIPE})
                for name in written:
                    (resources / name).touch()
                return subprocess.CompletedProcess(command, returncode, stdout)

            with mock.patch.object(package_app.subprocess, "run", side_effect=actool):
                package_app.compile_icon(resources, "26.4")

    def test_an_icon_actool_fails_to_compile_stops_the_build(self):
        compiled = {"com.apple.actool.compilation-results": {"output-files": []}}
        # As actool reports a missing .icon, exiting 0 or not.
        failed = {
            **compiled,
            "com.apple.actool.errors": [
                {"description": "Icon export exited with status 255, signal 0"},
                {"description": "Failed to create a catalog"},
            ],
        }
        for returncode in (0, 1):
            with (
                self.subTest(returncode=returncode),
                self.assertRaisesRegex(
                    RuntimeError,
                    "^actool: Icon export exited with status 255, signal 0; "
                    "Failed to create a catalog$",
                ),
            ):
                self.compile_icon(plistlib.dumps(failed), returncode=returncode)
        # Without errors, as when --app-icon does not name the .icon, it must
        # still have written both files.
        for written in ((), ("Assets.car",), ("AppIcon.icns",)):
            with (
                self.subTest(written=written),
                self.assertRaisesRegex(RuntimeError, "actool wrote no "),
            ):
                self.compile_icon(plistlib.dumps(compiled), written)
        self.compile_icon(plistlib.dumps(compiled), ("Assets.car", "AppIcon.icns"))
        # xcrun prints nothing on stdout when it finds no actool.
        with self.assertRaises(subprocess.CalledProcessError):
            self.compile_icon(b"", returncode=72)

    def signing_commands(self, identity):
        """The commands sign runs for a runtime of three Mach-O files, a link
        to one and a script, beside Sparkle.framework: runtime paths relative
        to the runtime, the app as APP."""
        with tempfile.TemporaryDirectory() as temporary:
            app = Path(temporary) / "Splash.app"
            sparkle = app / "Contents/Frameworks/Sparkle.framework/Versions/B"
            (sparkle / "Updater.app/Contents/MacOS").mkdir(parents=True)
            for name in ("Sparkle", "Autoupdate", "Updater.app/Contents/MacOS/Updater"):
                (sparkle / name).write_bytes(THIN_ARM64)
            runtime = app / "Contents/Resources/splash"
            for folder in ("python/bin", "python/lib", "engine", "install"):
                (runtime / folder).mkdir(parents=True)
            for name in ("python/bin/python3.13", "python/lib/x.so", "engine/splash"):
                (runtime / name).write_bytes(THIN_ARM64)
            (runtime / "python/bin/python3").symlink_to("python3.13")
            (runtime / "install/launcher.py").write_text("# launcher\n")
            with mock.patch.object(
                package_app.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run:
                self.assertEqual(package_app.sign(app, identity), 3)
            return [
                [
                    argument.removeprefix(f"{runtime}/").replace(str(app), "APP")
                    for argument in call.args[0]
                ]
                for call in run.call_args_list
            ]

    def test_ad_hoc_signing_has_no_hardened_runtime_or_timestamp(self):
        adhoc = ["codesign", "--force", "--sign", "-", "--timestamp=none"]
        self.assertEqual(
            self.signing_commands("-"),
            [
                [*adhoc, "--identifier", "ai.inco.splash.engine", "engine/splash"],
                [
                    *adhoc,
                    "--identifier",
                    "ai.inco.splash.python",
                    "python/bin/python3.13",
                ],
                [*adhoc, "python/lib/x.so"],
                [*adhoc, f"{SPARKLE}/Versions/B/Autoupdate"],
                [*adhoc, f"{SPARKLE}/Versions/B/Updater.app"],
                [*adhoc, SPARKLE],
                [*adhoc, "APP"],
                ["codesign", "--verify", "--strict", "engine/splash"],
                ["codesign", "--verify", "--strict", "python/bin/python3.13"],
                ["codesign", "--verify", "--strict", "python/lib/x.so"],
                ["codesign", "--verify", "--deep", "--strict", "--verbose=2", "APP"],
            ],
        )

    def test_identity_signing_has_hardened_runtime_and_timestamp(self):
        identity = "Developer ID Application: Test (TEAM)"
        signed = ["codesign", "--force", "--sign", identity, "--timestamp"]
        signed += ["--options", "runtime"]
        self.assertEqual(
            self.signing_commands(identity)[:7],
            [
                [*signed, "--identifier", "ai.inco.splash.engine", "engine/splash"],
                [
                    *signed,
                    "--identifier",
                    "ai.inco.splash.python",
                    "python/bin/python3.13",
                ],
                [*signed, "python/lib/x.so"],
                [*signed, f"{SPARKLE}/Versions/B/Autoupdate"],
                [*signed, f"{SPARKLE}/Versions/B/Updater.app"],
                [*signed, SPARKLE],
                [*signed, "APP"],
            ],
        )

    def image_commands(self, identity):
        """The commands build_image runs for identity, the app as APP and the
        disk image's temporary folder as FOLDER."""
        with tempfile.TemporaryDirectory() as temporary:
            app = Path(temporary) / "Splash.app"
            app.mkdir()
            image = Path(temporary) / "Splash-1.5.0.dmg"
            commands = []

            def run(command, check):
                if command[0] == "hdiutil":
                    Path(command[-1]).write_bytes(b"disk image")
                commands.append(command)
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(package_app.subprocess, "run", side_effect=run):
                package_app.build_image(app, image, identity)
            self.assertEqual(image.read_bytes(), b"disk image")
            folder = str(Path(commands[0][-1]).parents[1])
            return [
                [
                    argument.replace(str(app), "APP").replace(folder, "FOLDER")
                    for argument in command
                ]
                for command in commands
            ]

    def test_the_disk_image_is_lzfse_and_signed_with_its_own_identifier(self):
        built = [
            ["ditto", "APP", "FOLDER/Splash/Splash.app"],
            ["hdiutil", "create", "-volname", "Splash", "-srcfolder", "FOLDER/Splash"]
            + ["-format", "ULFO", "FOLDER/Splash-1.5.0.dmg"],
        ]
        self.assertEqual(self.image_commands("-"), built)
        identity = "Developer ID Application: Test (ABCDE12345)"
        self.assertEqual(
            self.image_commands(identity),
            [
                *built,
                ["codesign", "--force", "--sign", identity, "--timestamp"]
                + ["--identifier", "ai.inco.splash.dmg", "FOLDER/Splash-1.5.0.dmg"],
            ],
        )

    def test_invalid_arguments_stop_before_the_build(self):
        # The bundle's short version needs the release version's x.y.z.
        with (
            mock.patch.object(
                package_app,
                "build_products",
                side_effect=AssertionError("built despite invalid arguments"),
            ),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            package_app.main(["dmg", "--version", "6e9dfe3f", "--macos-min", "26.4"])


if __name__ == "__main__":
    unittest.main()
