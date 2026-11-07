"""The appcast a release of the Splash desktop app carries (release_app.py): it
offers the release's disk image as the update, with the release notes, signed
with the release key."""

import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from dev.tools import package_app

ROOT = Path(__file__).resolve().parents[2]
# The release key's item in the login keychain, apart from any other key
# Sparkle's tools keep there.
ACCOUNT = package_app.BUNDLE_ID
# Sparkle's tools, which SwiftPM unpacks, as the app's build does, with the
# package app/Package.swift pins.
TOOLS = ROOT / "app/.build/artifacts/sparkle/Sparkle/bin"
SPARKLE = "{http://www.andymatuschak.org/xml-namespaces/sparkle}"


def generate(image, notes, version, key_file):
    """Run Sparkle's generate_appcast on image and notes, in a folder of their
    own, and return the appcast it writes: image as the download from the
    release, notes embedded, signed with the private key in key_file or, by
    default, the login keychain's. It reads the versions Sparkle compares, the
    bundle's CFBundleVersion and CFBundleShortVersionString, and the minimum
    macOS, LSMinimumSystemVersion, from the app in image."""
    with tempfile.TemporaryDirectory(prefix=".appcast-", dir=image.parent) as temporary:
        folder = Path(temporary)
        os.link(image, folder / image.name)
        shutil.copyfile(notes, folder / f"{image.stem}.md")
        key = ["--ed-key-file", str(key_file)] if key_file else ["--account", ACCOUNT]
        subprocess.run(
            [str(TOOLS / "generate_appcast"), *key]
            + ["--download-url-prefix", f"{package_app.RELEASES}/download/{version}/"]
            + ["--embed-release-notes"]
            + ["--full-release-notes-url", package_app.RELEASES]
            + ["-o", str(folder / "appcast.xml"), str(folder)],
            check=True,
        )
        return (folder / "appcast.xml").read_bytes()


def check(appcast, image, version, macos_min, build, previous):
    """Check that appcast offers image alone, as release version's download for
    macOS macos_min or later, with build, the CFBundleVersion of the app in
    image, as the version Sparkle compares, signed with the key whose public
    half every app shipped holds. generate_appcast only warns when the key it
    signs with is not the one the app in image holds, and then signs nothing.
    previous is the release before, its tag and build, or None: Sparkle offers
    the apps that have it only a larger build."""
    items = ElementTree.fromstring(appcast).findall("channel/item")
    if len(items) != 1:
        raise RuntimeError(f"the appcast offers {len(items)} updates, not 1")
    item = items[0]
    enclosure = item.find("enclosure")
    expected = {
        "url": f"{package_app.RELEASES}/download/{version}/{image.name}",
        "length": str(image.stat().st_size),
        "version": build,
        "shortVersionString": package_app.short_version(version),
        "minimumSystemVersion": macos_min,
    }
    found = {
        "url": enclosure.get("url"),
        "length": enclosure.get("length"),
        "version": item.findtext(f"{SPARKLE}version"),
        "shortVersionString": item.findtext(f"{SPARKLE}shortVersionString"),
        "minimumSystemVersion": item.findtext(f"{SPARKLE}minimumSystemVersion"),
    }
    if found != expected:
        raise RuntimeError(f"the appcast offers {found}, not {expected}")
    if previous and int(build) <= int(previous[1]):
        raise RuntimeError(
            f"the appcast offers build {build}, not above build {previous[1]} of "
            f"release {previous[0]}, whose apps Sparkle would not update"
        )
    key = base64.b64decode(package_app.SPARKLE_PUBLIC_KEY)
    signature = base64.b64decode(enclosure.get(f"{SPARKLE}edSignature", ""))
    try:
        Ed25519PublicKey.from_public_bytes(key).verify(signature, image.read_bytes())
    except InvalidSignature:
        raise RuntimeError(
            f"{image.name} is not signed with the release key whose public half "
            "is SPARKLE_PUBLIC_KEY in dev/tools/package_app.py"
        ) from None
