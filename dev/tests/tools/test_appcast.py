import base64
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from dev.tools import appcast, package_app

RELEASES = "https://github.com/incoai/splash/releases"
APPCAST = """<?xml version="1.0" standalone="yes"?>
<rss xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle" version="2.0">
    <channel>
        <title>Splash</title>
{items}
    </channel>
</rss>
"""
# An item as generate_appcast writes it.
ITEM = """        <item>
            <title>1.5.0</title>
            <sparkle:version>{build}</sparkle:version>
            <sparkle:shortVersionString>{short}</sparkle:shortVersionString>
            <sparkle:minimumSystemVersion>{minimum}</sparkle:minimumSystemVersion>
            <sparkle:hardwareRequirements>arm64</sparkle:hardwareRequirements>
            <description sparkle:format="markdown"><![CDATA[Notes]]></description>
            <enclosure url="{url}" length="{length}" type="application/octet-stream"{signature}/>
        </item>"""


def public_half(key):
    return base64.b64encode(
        key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode()


class AppcastTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.image = Path(temporary.name) / "Splash-1.5.0.dmg"
        self.image.write_bytes(b"disk image" * 100)
        self.key = Ed25519PrivateKey.generate()
        patch = mock.patch.object(
            package_app, "SPARKLE_PUBLIC_KEY", public_half(self.key)
        )
        patch.start()
        self.addCleanup(patch.stop)

    def item(self, key=None, **fields):
        """An item that offers the image, signed with key (the release key by
        default; False for none), with fields in place of the right ones."""
        key = self.key if key is None else key
        signature = ""
        if key:
            signed = base64.b64encode(key.sign(self.image.read_bytes())).decode()
            signature = f' sparkle:edSignature="{signed}"'
        right = {
            "url": f"{RELEASES}/download/1.5.0/Splash-1.5.0.dmg",
            "length": self.image.stat().st_size,
            "build": "1600",
            "short": "1.5.0",
            "minimum": "26.4",
        }
        return ITEM.format(**{**right, **fields}, signature=signature)

    def check(self, *items, previous=("1.4.1", "1567")):
        appcast.check(
            APPCAST.format(items="\n".join(items)).encode(),
            self.image,
            "1.5.0",
            "26.4",
            "1600",
            previous,
        )

    def test_the_release_signed_with_the_release_key_is_published(self):
        self.check(self.item())

    def test_an_update_the_apps_shipped_would_refuse_is_not_published(self):
        # generate_appcast leaves the item unsigned when its key is not the
        # app's, and only warns.
        for key in (False, Ed25519PrivateKey.generate()):
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(RuntimeError, "not signed with the release key"),
            ):
                self.check(self.item(key))

    def test_an_appcast_must_offer_this_release_alone(self):
        for fields in (
            {"url": f"{RELEASES}/download/1.4.9/Splash-1.5.0.dmg"},
            {"length": 1},
            {"build": "1599"},
            {"short": "1.4.9"},
            {"minimum": "26.0"},
        ):
            with (
                self.subTest(fields=fields),
                self.assertRaisesRegex(RuntimeError, "the appcast offers {'url'"),
            ):
                self.check(self.item(**fields))
        for items in ((), (self.item(), self.item())):
            with (
                self.subTest(items=len(items)),
                self.assertRaisesRegex(RuntimeError, f"offers {len(items)} updates"),
            ):
                self.check(*items)

    def test_an_update_must_have_a_larger_build_than_the_release_before(self):
        # Sparkle compares CFBundleVersion alone: the apps of the release
        # before would never be offered a build no larger than theirs.
        self.check(self.item(), previous=None)
        for build in ("1600", "2492"):
            with (
                self.subTest(build=build),
                self.assertRaisesRegex(
                    RuntimeError,
                    f"offers build 1600, not above build {build} of release 1.4.1",
                ),
            ):
                self.check(self.item(), previous=("1.4.1", build))

    def generate_command(self, key_file):
        """The command generate runs, with the files of its folder."""
        notes = self.image.parent / "notes.md"
        notes.write_text("Notes\n")
        commands = []

        def generate_appcast(command, check):
            folder = Path(command[-1])
            commands.append(
                [argument.replace(str(folder), "FOLDER") for argument in command]
                + sorted(path.name for path in folder.iterdir())
            )
            self.assertEqual((folder / "Splash-1.5.0.md").read_text(), "Notes\n")
            self.assertTrue((folder / self.image.name).samefile(self.image))
            (folder / "appcast.xml").write_text("appcast")
            return subprocess.CompletedProcess(command, 0)

        with mock.patch.object(appcast.subprocess, "run", side_effect=generate_appcast):
            self.assertEqual(
                appcast.generate(self.image, notes, "1.5.0", key_file), b"appcast"
            )
        return commands[0]

    def test_generate_appcast_signs_with_the_keychains_key_or_a_key_file(self):
        tool = str(appcast.ROOT / "app/.build/artifacts/sparkle/Sparkle/bin")
        options = [
            "--download-url-prefix",
            f"{RELEASES}/download/1.5.0/",
            "--embed-release-notes",
            "--full-release-notes-url",
            RELEASES,
            "-o",
            "FOLDER/appcast.xml",
            "FOLDER",
            "Splash-1.5.0.dmg",
            "Splash-1.5.0.md",
        ]
        self.assertEqual(
            self.generate_command(None),
            [f"{tool}/generate_appcast", "--account", "ai.inco.splash", *options],
        )
        self.assertEqual(
            self.generate_command(Path("/keys/release")),
            [f"{tool}/generate_appcast", "--ed-key-file", "/keys/release", *options],
        )


if __name__ == "__main__":
    unittest.main()
