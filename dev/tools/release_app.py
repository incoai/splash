#!/usr/bin/env python3
"""Make release VERSION of Splash in dist/release/VERSION, ready to publish by
hand: the app and its disk image signed with a Developer ID, each notarized and
stapled and checked as a Mac that downloads them would, the command line's
files make package leaves in dist/, their checksums, the appcast, and the
formula and cask for the tap. It publishes nothing; it prints what publishing
takes. A dry run signs ad hoc, prints, instead of running, what needs the
Developer ID or the notary service, and writes dist/release/VERSION-dry-run."""

import argparse
import collections
import json
import os
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.tools import appcast, package, package_app  # noqa: E402

# The tap incoai/homebrew-tap: the command line's formula, splash, and the
# app's cask, splash-app.
TAP = "incoai/tap"
FORMULA = "splash"
CASK = "splash-app"
# A Developer ID Application identity as codesign takes it, by its name, whose
# team, a person's or an organization's, is all that tells their releases
# apart, or by its certificate's SHA-1, which tells apart two of one name.
IDENTITY = re.compile(r"Developer ID Application: .+ \(([A-Z0-9]{10})\)")
SHA1 = re.compile(r"[0-9A-F]{40}")
# A release's version, which its tag is: x.y.z, or x.y.z-rc.N for a release
# candidate.
RELEASE_VERSION = re.compile(r"\d+\.\d+\.\d+(-rc\.\d+)?")
# What a dry run prints for an identity, notary credentials or a submission it
# does not have, and commands print for the folder a disk image is attached at.
UNSET_IDENTITY = "Developer ID Application: NAME (TEAMID)"
UNSET_TEAM = "TEAMID"
UNSET_NOTARY = ("--keychain-profile", "NOTARY_PROFILE")
UNSET_SUBMISSION = "SUBMISSION_ID"
MOUNTED = "MOUNTED_IMAGE"
# The source spctl names for code signed with a Developer ID and notarized.
NOTARIZED = "source=Notarized Developer ID"


class Runner:
    """Prints and runs the commands that need the Developer ID or the notary
    service; a dry run only prints them. A path in shown is printed as what
    it stands for: a temporary folder as the one it becomes."""

    def __init__(self, dry_run):
        self.dry_run = dry_run
        self.shown = {}

    def __call__(self, command, check=True, **options):
        print(f"  {shlex.join(map(self.show, command))}")
        if self.dry_run:
            return None
        return subprocess.run(command, check=check, **options)

    def show(self, argument):
        for path, shown in self.shown.items():
            argument = argument.replace(path, shown)
        return argument


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="the release's version and tag, x.y.z or x.y.z-rc.N (RELEASE_VERSION)",
    )
    parser.add_argument("--macos-min", required=True, help="minimum macOS, e.g. 26.4")
    parser.add_argument(
        "--notes",
        required=True,
        type=Path,
        help="the release notes, in Markdown (RELEASE_NOTES)",
    )
    parser.add_argument(
        "--sign",
        metavar="IDENTITY",
        help="the Developer ID Application identity, "
        "'Developer ID Application: NAME (TEAMID)' or its SHA-1 (SIGN_IDENTITY)",
    )
    parser.add_argument(
        "--notary-profile",
        metavar="NAME",
        help="notarytool's credentials, as notarytool store-credentials keeps "
        "them (NOTARY_PROFILE)",
    )
    parser.add_argument(
        "--notary-key",
        type=Path,
        metavar="FILE",
        help="or an App Store Connect Team API key's private key, AuthKey_ID.p8 "
        "(NOTARY_KEY)",
    )
    parser.add_argument(
        "--notary-key-id", metavar="ID", help="the API key's ID (NOTARY_KEY_ID)"
    )
    parser.add_argument(
        "--notary-issuer", metavar="UUID", help="the API key's issuer (NOTARY_ISSUER)"
    )
    parser.add_argument(
        "--sparkle-key-file",
        type=Path,
        help="the release key's private half, as generate_keys -x exports it; "
        "the login keychain's by default (SPARKLE_KEY_FILE)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="sign ad hoc and print what needs the Developer ID or the notary "
        "service (DRY_RUN=1)",
    )
    args = parser.parse_args(argv)
    if not RELEASE_VERSION.fullmatch(args.version):
        parser.error(
            "RELEASE_VERSION (--version) must be x.y.z or x.y.z-rc.N, e.g. 1.5.0, "
            f"not {args.version}"
        )
    if not args.notes.is_file():
        parser.error(f"RELEASE_NOTES (--notes) names no file: {args.notes}")
    if args.sign and SHA1.fullmatch(args.sign.upper()):
        args.sign = args.sign.upper()
    elif args.sign and not IDENTITY.fullmatch(args.sign):
        parser.error(
            "SIGN_IDENTITY (--sign) must be 'Developer ID Application: NAME "
            "(TEAMID)', TEAMID being the team's 10-character ID, or that "
            f"certificate's SHA-1, not {args.sign!r}"
        )
    key = (args.notary_key, args.notary_key_id, args.notary_issuer)
    if args.notary_profile and any(key):
        parser.error(
            "notarize with NOTARY_PROFILE (--notary-profile) or an API key, not both"
        )
    if any(key) and not all(key):
        parser.error(
            "an API key takes NOTARY_KEY, NOTARY_KEY_ID and NOTARY_ISSUER "
            "(--notary-key, --notary-key-id and --notary-issuer)"
        )
    for path in (args.notary_key, args.sparkle_key_file):
        if path and not path.is_file():
            parser.error(f"no such key file: {path}")
    return args


def notary_options(args):
    """notarytool's options for the credentials args names: a keychain
    profile, or an App Store Connect API key; None for none."""
    if args.notary_profile:
        return ["--keychain-profile", args.notary_profile]
    if args.notary_key:
        key, key_id, issuer = args.notary_key, args.notary_key_id, args.notary_issuer
        return ["--key", str(key), "--key-id", key_id, "--issuer", issuer]
    return None


def checkout_problems(version):
    """Why the checkout is not release version's: a release is built from a
    clean checkout, untracked files included, of the commit its annotated tag
    names."""

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True, text=True
        ).stdout.rstrip()

    problems = []
    changed = [line[3:] for line in git("status", "--porcelain").splitlines()]
    if changed:
        problems.append(f"the checkout has changes: {', '.join(changed)}")
    tagged = git("rev-parse", f"refs/tags/{version}^{{commit}}")
    if git("cat-file", "-t", f"refs/tags/{version}") != "tag":
        problems.append(f"there is no annotated tag {version}")
    elif tagged != git("rev-parse", "HEAD"):
        problems.append(f"HEAD is not the commit tag {version} names")
    return problems


def previous_release(version):
    """The release before version, the latest x.y.z, release candidates aside,
    whose annotated tag is below version's x.y.z, and the build number of its
    app, its tagged commit's (package_app.build_number); None for none."""
    tags = subprocess.run(
        ["git", "for-each-ref", "--format=%(objecttype) %(refname:short)", "refs/tags"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    def number(release):
        return tuple(map(int, release.split(".")))

    below = number(version.split("-")[0])
    releases = [
        tag
        for kind, _, tag in (line.partition(" ") for line in tags)
        if kind == "tag" and re.fullmatch(r"\d+\.\d+\.\d+", tag) and number(tag) < below
    ]
    if not releases:
        return None
    tag = max(releases, key=number)
    return tag, package_app.build_number(tag)


def keychain_identity(identity):
    """The identity valid for signing code that the keychains hold for
    identity, its name or its certificate's SHA-1, as security find-identity
    lists it: its name. Raises LookupError when none does, or when two
    have its name, as while a renewed certificate and the one it renews are
    both valid, which codesign would refuse as ambiguous."""
    listed = subprocess.run(
        ["security", "find-identity", "-v", "-p", "codesigning"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    found = {
        sha1: name
        for sha1, name in re.findall(r'^\s*\d+\) ([0-9A-F]{40}) "(.*)"$', listed, re.M)
        if identity in (sha1, name)
    }
    if not found:
        raise LookupError(f"no keychain holds the identity {identity}")
    if len(found) > 1:
        raise LookupError(
            f"{len(found)} valid identities are named {identity}: set "
            "SIGN_IDENTITY to the SHA-1 of the one to sign with, which "
            "security find-identity -v -p codesigning lists"
        )
    return next(iter(found.values()))


def signer(identity):
    """The name and team of identity, SIGN_IDENTITY, and why a release cannot
    sign with it, or None when it can."""
    if not identity:
        return (
            UNSET_IDENTITY,
            UNSET_TEAM,
            "SIGN_IDENTITY (--sign), a Developer ID Application identity, is not set",
        )
    try:
        name = keychain_identity(identity)
        problem = None
    except LookupError as error:
        name, problem = identity, str(error)
    named = IDENTITY.fullmatch(name)
    if named:
        return name, named[1], problem
    return (
        name,
        UNSET_TEAM,
        problem or f"{identity} is {name}, not a Developer ID Application identity",
    )


def command_line(version, macos_min):
    """The command line's files of release version, as make package and make
    package-bottle leave them in dist/: the runtime archive, its checksum file
    and the bottles, which the release carries, and the formula, which goes to
    the tap; and why they are not ready, or None."""
    dist = ROOT / "dist"
    archive = dist / f"{package.archive_name(version, macos_min)}.tar.gz"
    bottles = sorted(dist.glob(f"splash-{version}.*.bottle.tar.gz"))
    files = [archive, archive.with_name(f"{archive.name}.sha256"), *bottles]
    formula = dist / "splash.rb"
    missing = [path.name for path in files if not path.is_file()]
    if not bottles:
        missing.append(f"splash-{version}.*.bottle.tar.gz")
    if missing:
        problem = (
            f"dist/ has no {', '.join(missing)}: make package and make package-bottle"
        )
        return files, formula, problem
    text = formula.read_text() if formula.is_file() else ""
    if not all(package.digest(path) in text for path in (archive, *bottles)):
        problem = (
            f"dist/splash.rb is not the formula of {archive.name} and its bottle: "
            "make package-bottle"
        )
        return files, formula, problem
    return files, formula, None


def precheck(app):
    """Run the notary service's checks on app before it is submitted, as
    syspolicy_check notary-submission does, print what they find, counted,
    and stop on what the service would reject. It accepts their warnings,
    among them one for each of the runtime's Mach-O files in Resources."""
    print(f"The notary service's checks of {app.name} (syspolicy_check):")
    checked = subprocess.run(
        ["syspolicy_check", "notary-submission", str(app), "--json"],
        capture_output=True,
        text=True,
    )
    if checked.stdout:
        findings = json.loads(checked.stdout)["output"]
    else:
        checked.check_returncode()
        findings = []
    counted = collections.Counter(
        (finding["SyspolicyCheckErrorLevel"], finding["SyspolicyCheckShortError"])
        for finding in findings
    )
    for (level, error), count in sorted(counted.items()):
        print(f"  {level}: {error} ({count})")
    if not findings:
        print("  nothing found")
    rejected = [
        f"{finding.get('SyspolicyCheckErrorFile', app)}: "
        f"{finding['SyspolicyCheckLongError']}"
        for finding in findings
        if finding["SyspolicyCheckErrorLevel"] != "Warning"
    ]
    if rejected:
        raise RuntimeError(
            f"the notary service would reject {app.name}: " + "; ".join(rejected)
        )


def notarize(path, notary, log, run):
    """Have the notary service check path, an app, which notarytool takes
    zipped, or a disk image; keep the submission's log in log, which Apple
    asks to be read even when it accepts, for its warnings; and staple the
    ticket to path, for Gatekeeper to find without the network."""
    print(f"{'Would notarize' if run.dry_run else 'Notarizing'} {path.name}:")
    with tempfile.TemporaryDirectory(prefix=".notarize-", dir=path.parent) as temporary:
        archive = path
        if path.suffix == ".app":
            archive = Path(temporary) / f"{path.stem}.zip"
            run(["ditto", "-c", "-k", "--keepParent", str(path), str(archive)])
        submitted = run(
            ["xcrun", "notarytool", "submit", str(archive), *notary]
            + ["--wait", "--output-format", "json"],
            check=False,
            capture_output=True,
            text=True,
        )
        submission = {"id": UNSET_SUBMISSION, "status": "Accepted"}
        if submitted:
            try:
                submission = json.loads(submitted.stdout)
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"notarytool submit {archive.name}: {submitted.stderr.strip()}"
                ) from None
            log.parent.mkdir(exist_ok=True)
        run(["xcrun", "notarytool", "log", submission["id"], *notary, str(log)])
        if submission["status"] != "Accepted":
            raise RuntimeError(
                f"the notary service found {path.name} {submission['status']}: "
                f"{submission.get('message')}\n{log.read_text()}"
            )
        run(["xcrun", "stapler", "staple", str(path)])


@contextmanager
def mounted(image):
    """The folder image is attached at, read-only and hidden from the Finder."""
    with tempfile.TemporaryDirectory(prefix="splash-release-") as volume:
        subprocess.run(
            ["hdiutil", "attach", "-quiet", "-readonly", "-nobrowse", "-noautoopen"]
            + ["-mountpoint", volume, str(image)],
            check=True,
        )
        try:
            yield Path(volume)
        finally:
            subprocess.run(["hdiutil", "detach", "-quiet", volume], check=True)


def signing_team(app):
    """The team app's signature names, as codesign shows it: "not set" for an
    ad hoc signature."""
    shown = subprocess.run(
        ["codesign", "-dv", "--verbose=2", str(app)],
        check=True,
        capture_output=True,
        text=True,
    ).stderr
    return re.search(r"^TeamIdentifier=(.*)$", shown, re.M)[1]


def verify(image, version, team, run):
    """Check image and the app in it as a Mac that downloads image gets them:
    the app's identity, versions and update keys, its signature and the team
    it names, then that Gatekeeper takes both as notarized from their stapled
    tickets. A dry run's app, signed ad hoc, is checked as far as an ad hoc
    signature allows; the rest is printed."""
    with mounted(image) as volume:
        run.shown[str(volume)] = MOUNTED
        app = volume / "Splash.app"
        with (app / "Contents/Info.plist").open("rb") as stream:
            info = plistlib.load(stream)
        expected = {
            "CFBundleIdentifier": package_app.BUNDLE_ID,
            "CFBundleShortVersionString": package_app.short_version(version),
            "CFBundleVersion": package_app.build_number(),
            "SUFeedURL": package_app.APPCAST_URL,
            "SUPublicEDKey": package_app.SPARKLE_PUBLIC_KEY or None,
        }
        found = {key: info.get(key) for key in expected}
        if found != expected:
            raise RuntimeError(f"{app.name} has {found}, not {expected}")
        subprocess.run(
            ["codesign", "--verify", "--deep", "--strict", str(app)], check=True
        )
        signed = signing_team(app)
        print(
            f"{app.name} {found['CFBundleShortVersionString']} "
            f"({found['CFBundleVersion']}) in {image.name} is signed "
            + ("ad hoc." if signed == "not set" else f"by team {signed}.")
        )
        if not run.dry_run and signed != team:
            raise RuntimeError(f"{app.name} is signed by team {signed}, not {team}")
        if run.dry_run:
            print(f"Would check that it is team {team}'s and that both are notarized:")
        for command in (
            ["spctl", "--assess", "--type", "exec", "-vv", str(app)],
            ["spctl", "--assess", "--type", "open"]
            + ["--context", "context:primary-signature", "-vv", str(image)],
        ):
            assessed = run(command, check=False, capture_output=True, text=True)
            if assessed:
                print(assessed.stderr.strip())
                if assessed.returncode or NOTARIZED not in assessed.stderr:
                    name = Path(command[-1]).name
                    raise RuntimeError(f"Gatekeeper takes no notarized {name}")
        for path in (app, image):
            run(["xcrun", "stapler", "validate", str(path)])
        run(["syspolicy_check", "distribution", str(app)])


def checksums(*paths):
    """SHA256SUMS of paths, as shasum -a 256 writes it."""
    return "".join(f"{package.digest(path)}  {path.name}\n" for path in paths)


def cask(version, checksum, macos_min):
    """The Homebrew cask of release version, whose disk image's SHA-256 is
    checksum. A cask names a major macOS, so macos: is macos_min's, 26,
    Tahoe, and the caveat names its minor one and the Apple M3 the app needs,
    which a cask cannot require. zap removes what the app alone keeps and the
    caches, the engine's among them; not what the command line shares in
    Application Support/Splash, its models, locks and runtime, nor the
    Hugging Face cache other tools share."""
    return f"""cask "{CASK}" do
  version "{version}"
  sha256 "{checksum}"

  url "{package_app.RELEASES}/download/#{{version}}/Splash-#{{version}}.dmg"
  name "Splash"
  desc "Local inference engine for Apple silicon, built around the model"
  homepage "https://github.com/{package.REPOSITORY}"

  livecheck do
    url :url
    strategy :github_latest
  end

  auto_updates true
  depends_on arch: :arm64
  depends_on macos: :tahoe

  app "Splash.app"

  uninstall quit: "{package_app.BUNDLE_ID}"

  zap trash: [
    "~/Library/Application Support/Splash/Chats",
    "~/Library/Caches/{package_app.BUNDLE_ID}",
    "~/Library/Caches/Splash",
    "~/Library/HTTPStorages/{package_app.BUNDLE_ID}",
    "~/Library/Logs/Splash",
    "~/Library/Preferences/{package_app.BUNDLE_ID}.plist",
    "~/Library/Saved Application State/{package_app.BUNDLE_ID}.savedState",
    "~/Library/WebKit/{package_app.BUNDLE_ID}",
  ]

  caveats "Splash needs an Apple M3 or newer and macOS {macos_min} or later."
end
"""


def make(args, folder, identity, team, carried, run):
    """Make release args.version in folder: build, sign, notarize and check
    the app and its disk image, then add the command line's files and
    formula, carried, unless that is None, the checksums, the notes, the
    appcast and, unless it is a release candidate, the cask."""
    app = package_app.build_app(
        ROOT / "dist", args.version, args.macos_min, "-" if run.dry_run else identity
    )
    if run.dry_run:
        print(
            "Would sign the runtime's Mach-O files, Sparkle's code and, last, "
            "the app, as package_app.py does, each as:"
        )
        run(package_app.codesign_command(app, identity, "--options", "runtime"))
    precheck(app)
    notary = notary_options(args) or list(UNSET_NOTARY)
    notarize(app, notary, folder / "notary" / f"{app.name}.json", run)
    image = folder / f"Splash-{args.version}.dmg"
    package_app.build_image(app, image, "-" if run.dry_run else identity)
    if run.dry_run:
        print(f"Would sign {image.name}:")
        run(
            package_app.codesign_command(
                image, identity, "--identifier", package_app.IMAGE_IDENTIFIER
            )
        )
    notarize(image, notary, folder / "notary" / f"{image.name}.json", run)
    verify(image, args.version, team, run)
    os.link(image, folder / "Splash.dmg")
    if carried:
        files, formula = carried
        for path in files:
            os.link(path, folder / path.name)
        (folder / "Formula").mkdir()
        shutil.copyfile(formula, folder / "Formula" / f"{FORMULA}.rb")
    # The disk images and the archives, not the files that describe them.
    (folder / "SHA256SUMS").write_text(
        checksums(
            *sorted(path for path in folder.iterdir() if path.suffix in (".dmg", ".gz"))
        )
    )
    shutil.copyfile(args.notes, folder / "notes.md")
    if package_app.SPARKLE_PUBLIC_KEY:
        feed = appcast.generate(image, args.notes, args.version, args.sparkle_key_file)
        appcast.check(
            feed,
            image,
            args.version,
            args.macos_min,
            package_app.build_number(),
            previous_release(args.version),
        )
        (folder / "appcast.xml").write_bytes(feed)
    else:
        print(
            "Skipped appcast.xml: SPARKLE_PUBLIC_KEY in dev/tools/package_app.py "
            "is empty."
        )
    if "-rc." not in args.version:
        (folder / "Casks").mkdir()
        (folder / "Casks" / f"{CASK}.rb").write_text(
            cask(args.version, package.digest(image), args.macos_min)
        )


def publishing(version, folder, assets):
    """What publishing release version from folder takes, which release_app.py
    prints and never runs: the tag, a draft release of assets, the folder's
    files, checked before it is published, and then, unless it is a release
    candidate, the tap's formula and cask. Paths are the folder's own, so that
    another Mac can publish a copy of it."""
    candidate = "-rc." in version
    repository = ["--repo", package.REPOSITORY]
    lines = [
        f"# With gh, signed in with write access to {package.REPOSITORY}. "
        "The tag, from the checkout:",
        shlex.join(["git", "push", "origin", f"refs/tags/{version}"]),
        "# A draft release of the folder's files, from the folder:",
        shlex.join(["cd", str(folder)]),
        shlex.join(
            ["gh", "release", "create", version, *repository, "--draft"]
            + ["--verify-tag", "--title", version, "--notes-file", "notes.md"]
            + (["--prerelease"] if candidate else [])
        ),
        shlex.join(["gh", "release", "upload", version, *repository, *assets]),
        "# Once the draft's Splash.dmg opens on a clean Mac, publish it:",
        shlex.join(
            ["gh", "release", "edit", version, *repository, "--draft=false"]
            + ([] if candidate else ["--latest"])
        ),
    ]
    if candidate:
        return lines + [
            "# A release candidate is a prerelease and changes nothing in the tap."
        ]
    branch = f"{FORMULA}-{version}"
    tap = f'"$(brew --repository {TAP})"'
    return lines + [
        f"# Then, with write access to incoai/homebrew-tap and Homebrew, {TAP} "
        "tapped, the formula and the cask, on a branch of the tap:",
        f"cp -R Formula Casks {tap}",
        f"cd {tap} && git switch -c {branch}",
        shlex.join(["brew", "audit", "--strict", "--online", f"{TAP}/{FORMULA}"]),
        shlex.join(
            ["brew", "audit", "--cask", "--strict", "--online", f"{TAP}/{CASK}"]
        ),
        shlex.join(["git", "add", f"Formula/{FORMULA}.rb", f"Casks/{CASK}.rb"])
        + " && "
        + shlex.join(["git", "commit", "-m", f"{FORMULA} {version}"]),
        shlex.join(["git", "push", "-u", "origin", branch]) + " && gh pr create --fill",
        "# Once it merges, on a Mac without a Splash.app, which brew does not replace:",
        shlex.join(["brew", "install", "--cask", f"{TAP}/{CASK}"]),
    ]


def main(argv=None):
    args = parse_args(argv)
    release = ROOT / "dist/release" / args.version
    folder = release.with_name(f"{args.version}-dry-run") if args.dry_run else release
    problems = checkout_problems(args.version)
    name, team, problem = signer(args.sign)
    problems += [problem] if problem else []
    if not notary_options(args):
        problems.append(
            "there are no notary credentials: NOTARY_PROFILE, or NOTARY_KEY, "
            "NOTARY_KEY_ID and NOTARY_ISSUER"
        )
    if not package_app.SPARKLE_PUBLIC_KEY:
        problems.append(
            "SPARKLE_PUBLIC_KEY in dev/tools/package_app.py is empty, and apps "
            "built without it check for no update"
        )
    if release.exists():
        problems.append(f"{release} exists")
    carried = None
    if "-rc." not in args.version:
        files, formula, problem = command_line(args.version, args.macos_min)
        problems += [problem] if problem else []
        carried = None if problem else (files, formula)
    if args.dry_run:
        print(
            f"Dry run of release {args.version}, signed ad hoc, into {folder}; "
            "nothing is notarized, stapled or published."
        )
        for problem in problems:
            print(f"A release would stop here: {problem}.")
        if folder.exists():
            shutil.rmtree(folder)
    elif problems:
        sys.exit("\n".join(f"error: {problem}" for problem in problems))
    run = Runner(args.dry_run)
    folder.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{folder.name}-", dir=folder.parent))
    run.shown[str(staged)] = str(release)
    try:
        make(args, staged, args.sign or UNSET_IDENTITY, team, carried, run)
    except BaseException:
        print(f"Stopped; what it made is in {staged}.", file=sys.stderr)
        raise
    staged.replace(folder)
    print(f"Wrote {folder}:")
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(folder)}")
    assets = sorted(
        path.name
        for path in folder.iterdir()
        if path.is_file() and path.name != "notes.md"
    )
    if args.dry_run:
        print(f"Signed ad hoc; a release would be signed by team {team}, {name}.")
        for problem in problems:
            print(f"A release would stop here: {problem}.")
        print(f"Publishing release {args.version}, made without DRY_RUN, would take:")
    else:
        print(f"Release {args.version} is signed by team {team}, {name}.")
        print("To publish it, which release_app.py never does:")
    for line in publishing(args.version, release, assets):
        print(f"  {line}")


if __name__ == "__main__":
    # Line by line, so that what it prints keeps its place among the output of
    # the tools it runs when it goes to a file.
    sys.stdout.reconfigure(line_buffering=True)
    main()
