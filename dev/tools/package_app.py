#!/usr/bin/env python3
"""Build Splash.app around the packaged runtime, sign it, and wrap it in a DMG."""

import argparse
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.tools import package  # noqa: E402

BUNDLE_ID = "ai.inco.splash"
# Where Splash is released: each release carries the app's disk image and the
# appcast that offers it (docs/package.md, Updates).
RELEASES = f"https://github.com/{package.REPOSITORY}/releases"
# The appcast the app checks for updates, the latest release's. Every app
# shipped keeps this URL, so it never changes.
APPCAST_URL = f"{RELEASES}/latest/download/appcast.xml"
# The release key's public half, as Sparkle's generate_keys prints it
# (docs/package.md, Updates): the app takes an update whose disk image the
# private half signed or, to replace a lost key, one signed with the app's
# Developer ID. Without it, the app checks for no update.
SPARKLE_PUBLIC_KEY = "QAOqAR9iWlOPmkqq0XgbaAPjP8Y/X0lJwfd+W4a9kSc="
# The code Sparkle.framework holds besides its library, which installs an
# update: the installer and the app that shows its progress.
SPARKLE_HELPERS = ("Versions/B/Autoupdate", "Versions/B/Updater.app")
# The first four bytes of a Mach-O file: a 64- or 32-bit image in either byte
# order, or a universal one. The runtime's interpreter and engine have no
# extension, so its Mach-O files are found by content, not by name.
MACH_O_MAGICS = frozenset(
    bytes.fromhex(magic)
    for magic in (
        "cffaedfe",
        "cefaedfe",
        "feedfacf",
        "feedface",
        "cafebabe",
        "cafebabf",
    )
)
# The app's own resources, the Chat's transcript page among them: the bundle
# SwiftPM builds beside the executable, which the app reads from
# Contents/Resources (Bundle.resources, app/Sources/Splash/WebPage.swift).
RESOURCE_BUNDLE = "Splash_Splash.bundle"
# Signing identifiers of the runtime's two programs, by their paths in the
# runtime; codesign names every other file after itself.
IDENTIFIERS = {
    "python/bin/python3": f"{BUNDLE_ID}.python",
    "engine/splash": f"{BUNDLE_ID}.engine",
}
# The disk image's signing identifier; codesign would name it after its file,
# which carries the version.
IMAGE_IDENTIFIER = f"{BUNDLE_ID}.dmg"


def short_version(version):
    """CFBundleShortVersionString for a release version: its x.y.z, which
    `git describe` suffixes (1.4.1-36-g6e9dfe3f), or None for a version that
    does not start with one."""
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(-[A-Za-z0-9._-]+)?", version)
    return match and match[1]


def build_number(commit="HEAD"):
    """CFBundleVersion of a build of commit: the number of commits it has. It
    grows with every commit on main, so builds of one release version, as
    `git describe` names them, differ, and a later build has the larger number
    macOS and Sparkle prefer. A shallow clone counts only the commits it
    holds, a number below the releases', which Sparkle would never offer as
    an update, so it builds nothing."""

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    if git("rev-parse", "--is-shallow-repository") == "true":
        raise RuntimeError(
            "the checkout is a shallow clone, whose commit count, the app's build "
            "number (CFBundleVersion), is below the releases': git fetch --unshallow"
        )
    return git("rev-list", "--count", commit)


def info_plist(version, build, macos_min):
    """The app's Info.plist. Sparkle reads its keys: the appcast; once there is
    a release key, its public half and SUVerifyUpdateBeforeExtraction, for
    Sparkle to check a download's signature before it mounts the disk image,
    not after; and, left out, SUEnableAutomaticChecks, for Sparkle to ask on
    the second launch whether to check by itself."""
    sparkle = {"SUFeedURL": APPCAST_URL}
    if SPARKLE_PUBLIC_KEY:
        sparkle["SUPublicEDKey"] = SPARKLE_PUBLIC_KEY
        sparkle["SUVerifyUpdateBeforeExtraction"] = True
    return {
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleName": "Splash",
        "CFBundleDisplayName": "Splash",
        "CFBundleExecutable": "Splash",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": short_version(version),
        "CFBundleVersion": build,
        "CFBundleIconFile": "AppIcon",
        "CFBundleIconName": "AppIcon",
        "LSMinimumSystemVersion": macos_min,
        "LSApplicationCategoryType": "public.app-category.developer-tools",
        "NSHighResolutionCapable": True,
        # The app's chat, health and status requests go to its own server,
        # over plain HTTP on 127.0.0.1.
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
        "NSHumanReadableCopyright": "Copyright © Inco AI",
        **sparkle,
    }


def mach_o_files(root):
    """The Mach-O files under root, symbolic links aside."""
    found = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                if stream.read(4) in MACH_O_MAGICS:
                    found.append(path)
    return found


def build_products():
    """Build the app's executable, arm64 in release configuration, and return
    the folder that holds it and the Sparkle.framework it links."""
    command = [
        "swift",
        "build",
        "-c",
        "release",
        "--arch",
        "arm64",
        "--package-path",
        str(ROOT / "app"),
        "--product",
        "Splash",
    ]
    subprocess.run(command, check=True)
    products = subprocess.run(
        [*command, "--show-bin-path"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return Path(products)


def compile_icon(resources, macos_min):
    """Compile app/AppIcon.icon, an Icon Composer icon, into resources: the
    Assets.car macOS 26 draws it from, with its dark, tinted and clear
    appearances, and AppIcon.icns for older readers of CFBundleIconFile.
    actool prints its errors only on stdout, in a property list, and can
    exit 0 having written neither file, as when --app-icon does not name
    the .icon."""
    with tempfile.TemporaryDirectory() as temporary:
        compiled = subprocess.run(
            ["xcrun", "actool", str(ROOT / "app/AppIcon.icon")]
            + ["--compile", str(resources), "--platform", "macosx"]
            + ["--minimum-deployment-target", macos_min, "--app-icon", "AppIcon"]
            + ["--output-partial-info-plist", str(Path(temporary) / "icon.plist")],
            stdout=subprocess.PIPE,
        )
    results = plistlib.loads(compiled.stdout) if compiled.stdout else {}
    errors = results.get("com.apple.actool.errors", [])
    if errors:
        raise RuntimeError(
            "actool: " + "; ".join(error["description"] for error in errors)
        )
    compiled.check_returncode()
    for name in ("Assets.car", "AppIcon.icns"):
        if not (resources / name).is_file():
            raise RuntimeError(f"actool wrote no {name} into {resources}")


def copy_resources(products, resources):
    """Copy the app's resource bundle, which SwiftPM built into products, into
    resources: a bundle at the .app's root would leave it unsealed."""
    shutil.copytree(
        products / RESOURCE_BUNDLE, resources / RESOURCE_BUNDLE, symlinks=True
    )


def add_sparkle(frameworks, framework):
    """Copy Sparkle.framework into frameworks with ditto, which keeps the links
    a framework's layout is made of, without its XPC services: only a
    sandboxed app needs them, and Splash is not one."""
    frameworks.mkdir()
    copy = frameworks / framework.name
    subprocess.run(["ditto", str(framework), str(copy)], check=True)
    shutil.rmtree(copy / "Versions/B/XPCServices")
    (copy / "XPCServices").unlink()


def assemble(app, products, version, macos_min):
    """Lay out app: the executable, Sparkle.framework in Frameworks, where the
    executable's rpath finds it (app/Package.swift), Info.plist, the icon, the
    app's resource bundle and, in Resources/splash, the runtime build_runtime
    builds, whose layout install/paths.py resolves."""
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    shutil.copy2(products / "Splash", contents / "MacOS/Splash")
    add_sparkle(contents / "Frameworks", products / "Sparkle.framework")
    with (contents / "Info.plist").open("wb") as stream:
        plistlib.dump(info_plist(version, build_number(), macos_min), stream)
    runtime = contents / "Resources/splash"
    runtime.mkdir(parents=True)
    copy_resources(products, contents / "Resources")
    compile_icon(contents / "Resources", macos_min)
    package.build_runtime(runtime, version)


def codesign_command(path, identity, *options):
    """The command that signs path ad hoc ("-") or with identity and the secure
    timestamp notarization requires."""
    timestamp = "--timestamp=none" if identity == "-" else "--timestamp"
    return ["codesign", "--force", "--sign", identity, timestamp, *options, str(path)]


def codesign(path, identity, *options):
    subprocess.run(codesign_command(path, identity, *options), check=True)


def sign(app, identity):
    """Sign every Mach-O file of app's runtime, Sparkle's code from the inside
    out, then app, and verify them; return the number of runtime files signed.

    codesign --deep signs and checks only the code a bundle nests where code
    belongs, not the runtime's files in Resources, so each is signed on its
    own. The hardened runtime's library validation loads only libraries signed
    by the process's own team, and an ad-hoc signature names none, so an
    ad-hoc build signs without it."""
    hardened = () if identity == "-" else ("--options", "runtime")
    runtime = app / "Contents/Resources/splash"
    identifiers = {
        (runtime / path).resolve(): identifier
        for path, identifier in IDENTIFIERS.items()
    }
    binaries = mach_o_files(runtime)
    for binary in binaries:
        identifier = identifiers.get(binary.resolve())
        named = ("--identifier", identifier) if identifier else ()
        codesign(binary, identity, *hardened, *named)
    sparkle = app / "Contents/Frameworks/Sparkle.framework"
    for helper in SPARKLE_HELPERS:
        codesign(sparkle / helper, identity, *hardened)
    codesign(sparkle, identity, *hardened)
    codesign(app, identity, *hardened)
    for binary in binaries:
        subprocess.run(["codesign", "--verify", "--strict", str(binary)], check=True)
    subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)],
        check=True,
    )
    return len(binaries)


def check_signed(app):
    """Run the signed runtime as the app does, its libraries loading under
    their new signatures, then check that it left the bundle sealed: a
    bytecode file written into it would break the seal."""
    runtime = app / "Contents/Resources/splash"
    subprocess.run(
        [str(runtime / "python/bin/python3"), "-I", "-B", "-c", package.SMOKE_IMPORTS],
        cwd=runtime,
        check=True,
    )
    subprocess.run([str(runtime / "engine/splash"), "device-check"], check=True)
    if any(app.rglob("__pycache__")):
        raise RuntimeError(f"{app} holds bytecode")
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)


def build_image(app, image, identity):
    """Write image, a DMG of app beside a link to /Applications, compressed
    with LZFSE (ULFO), as Sparkle recommends, rather than LZMA (ULMO), which
    is slower to open, and signed with identity unless that is ad hoc."""
    with tempfile.TemporaryDirectory(
        prefix=".package-app-", dir=image.parent
    ) as temporary:
        folder = Path(temporary) / "Splash"
        folder.mkdir()
        subprocess.run(["ditto", str(app), str(folder / app.name)], check=True)
        (folder / "Applications").symlink_to("/Applications")
        temporary_image = Path(temporary) / image.name
        subprocess.run(
            ["hdiutil", "create", "-volname", "Splash", "-srcfolder", str(folder)]
            + ["-format", "ULFO", str(temporary_image)],
            check=True,
        )
        if identity != "-":
            codesign(temporary_image, identity, "--identifier", IMAGE_IDENTIFIER)
        temporary_image.replace(image)


def build_app(dist, version, macos_min, identity):
    """Build dist/Splash.app, signed with identity, and return it."""
    products = build_products()
    app = dist / "Splash.app"
    with tempfile.TemporaryDirectory(prefix=".package-app-", dir=dist) as temporary:
        bundle = Path(temporary) / app.name
        assemble(bundle, products, version, macos_min)
        signed = sign(bundle, identity)
        check_signed(bundle)
        if app.exists():
            shutil.rmtree(app)
        bundle.replace(app)
    print(f"Built {app}; signed it and the {signed} Mach-O files of its runtime.")
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=("app", "dmg"),
        help="app builds dist/Splash.app; dmg also wraps it in dist/Splash-VERSION.dmg",
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--macos-min", required=True, help="minimum macOS, e.g. 26.4")
    parser.add_argument(
        "--sign",
        default="-",
        metavar="IDENTITY",
        help="codesign identity, a Developer ID Application; ad hoc (-) by default",
    )
    args = parser.parse_args(argv)
    if not short_version(args.version):
        parser.error("release version must start with x.y.z, e.g. 1.4.1-36-g6e9dfe3f")
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    image = dist / f"Splash-{args.version}.dmg"
    if args.target == "dmg" and image.exists():
        parser.error(f"release already exists: {image}")
    app = build_app(dist, args.version, args.macos_min, args.sign)
    if args.target == "app":
        return
    build_image(app, image, args.sign)
    print(
        f"Built {image}. It is not notarized and is for local testing only: on "
        "another Mac, Gatekeeper blocks the Splash.app copied from it until it is "
        "allowed with Open Anyway in System Settings › Privacy & Security, or "
        "its quarantine is removed with "
        "xattr -dr com.apple.quarantine /Applications/Splash.app. make "
        "release-app makes a release (docs/package.md, Releases)."
    )


if __name__ == "__main__":
    main()
