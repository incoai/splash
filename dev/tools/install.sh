#!/bin/sh
# Splash private-test installer for Apple Silicon Macs.
#
#   export SPLASH_TOKEN=hf_...
#   curl -qfsSL --config - <install.sh URL> <<EOF | SPLASH_REPO=owner/repo sh
#   header = "Authorization: Bearer $SPLASH_TOKEN"
#   EOF
#
# Downloads the pinned release archive from the private Hugging Face repo,
# verifies its SHA-256, unpacks it under ~/Library/Application Support/Splash/app,
# points the `current` link at it and writes a `splash` command onto the PATH.
# Running it again upgrades in place; model weights and sessions are untouched.
#
#   SPLASH_TOKEN     read token from the invitation (required)
#   SPLASH_VERSION   install this version instead of the repo's `latest`
#   SPLASH_REPO      Hugging Face repo to install from (required)
#   SPLASH_BASE_URL  file server override, e.g. http://127.0.0.1:8123 for a local check
#   SPLASH_BIN_DIR   where to put the `splash` command (default: Homebrew bin or ~/.local/bin)
set -eu

REPO=${SPLASH_REPO:-}
if [ -z "${SPLASH_BASE_URL:-}" ] && [ -z "$REPO" ]; then
  echo "set SPLASH_REPO to the Hugging Face repo to install from" >&2
  exit 1
fi
BASE_URL=${SPLASH_BASE_URL:-https://huggingface.co/$REPO/resolve/main}
TOKEN=${SPLASH_TOKEN:-}
APP="$HOME/Library/Application Support/Splash/app"
MARKER="Splash/app/current"

fail() { echo "splash install: $*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || fail "Splash runs on Apple Silicon Macs only."
os=$(sw_vers -productVersion)
major=${os%%.*}
minor=${os#"$major"}
minor=${minor#.}
minor=${minor%%.*}
[ -n "$minor" ] || minor=0
[ "$major" -gt 26 ] 2>/dev/null || { [ "$major" -eq 26 ] && [ "$minor" -ge 4 ]; } \
    || fail "Splash requires macOS 26.4 or newer; this Mac runs $os."
command -v curl >/dev/null 2>&1 || fail "curl is required."
[ -n "$TOKEN" ] || fail "set SPLASH_TOKEN to the access token from your invitation."

dir=${SPLASH_BIN_DIR:-}
[ -n "$dir" ] || for candidate in /opt/homebrew/bin /usr/local/bin; do
    if [ -d "$candidate" ] && [ -w "$candidate" ]; then dir=$candidate; break; fi
done
[ -n "$dir" ] || dir="$HOME/.local/bin"
wrapper="$dir/splash"
if [ -e "$wrapper" ] && ! grep -q "$MARKER" "$wrapper" 2>/dev/null; then
    fail "$wrapper exists and was not created by this installer; remove it first."
fi

fetch() {
    curl -q -fsSL --retry 3 --config "$work/curl.conf" -o "$2" "$BASE_URL/$1" \
        || fail "could not download $BASE_URL/$1 (expired or wrong token?)"
}

work=$(mktemp -d "${TMPDIR:-/tmp}/splash-install.XXXXXX")
trap 'rm -rf "$work"' EXIT
case "$TOKEN" in *[!A-Za-z0-9_./~+=-]*) fail "invalid access token format.";; esac
(umask 077; printf 'header = "Authorization: Bearer %s"\n' "$TOKEN" > "$work/curl.conf")

version=${SPLASH_VERSION:-}
if [ -z "$version" ]; then
    fetch latest "$work/latest"
    version=$(tr -d '[:space:]' < "$work/latest")
    [ -n "$version" ] || fail "the release index is empty."
fi
name="splash-$version-arm64-macos26"

mkdir -p "$APP"
candidate="$APP/$name"
if [ -f "$candidate/release.json" ]; then
    echo "Splash $version is already downloaded."
else
    echo "Downloading Splash $version..."
    fetch "$name.tar.gz" "$work/$name.tar.gz"
    fetch "$name.tar.gz.sha256" "$work/$name.tar.gz.sha256"
    expected=$(cut -d' ' -f1 < "$work/$name.tar.gz.sha256")
    actual=$(shasum -a 256 "$work/$name.tar.gz" | cut -d' ' -f1)
    [ -n "$expected" ] && [ "$expected" = "$actual" ] || fail "checksum mismatch for $name.tar.gz."
    mkdir "$work/extract"
    tar -xzf "$work/$name.tar.gz" -C "$work/extract"
    [ -f "$work/extract/$name/release.json" ] || fail "unexpected archive layout."
    candidate="$work/extract/$name"
fi
# Validate before touching an installed version or waiting on its lifecycle lock.
if ! PYTHONDONTWRITEBYTECODE=1 "$candidate/python/bin/python3" -u \
        "$candidate/install/launcher.py" --help >/dev/null 2>&1; then
    fail "Splash $version fails 'splash --help'; nothing was changed."
fi
cat > "$work/apply.sh" <<'INSTALL'
set -eu
APP=$1
name=$2
candidate=$3
wrapper=$4
dir=${wrapper%/*}
fail() { echo "splash install: $*" >&2; exit 1; }
if [ "$candidate" != "$APP/$name" ] && [ ! -f "$APP/$name/release.json" ]; then
    rm -rf "$APP/$name"
    mv "$candidate" "$APP/$name"
fi
# Stage the wrapper before switching current; restore the old link if the
# final rename fails. Remove older versions only after both changes succeed.
mkdir -p "$dir"
cat > "$wrapper.tmp" <<WRAPPER || fail "could not write $wrapper; Splash $name was not installed."
#!/bin/sh
export PYTHONDONTWRITEBYTECODE=1
exec "$APP/current/python/bin/python3" -u "$APP/current/install/launcher.py" "\$@"
WRAPPER
chmod 0755 "$wrapper.tmp"
previous=$(readlink "$APP/current" 2>/dev/null || true)
ln -sfn "$APP/$name" "$APP/current"
if ! mv -f "$wrapper.tmp" "$wrapper"; then
    if [ -n "$previous" ]; then ln -sfn "$previous" "$APP/current"; else rm -f "$APP/current"; fi
    rm -f "$wrapper.tmp"
    fail "could not write $wrapper; Splash $name was not installed."
fi
for old in "$APP"/splash-*-arm64-macos26; do
    [ -d "$old" ] && [ "$old" != "$APP/$name" ] && rm -rf "$old"
done
exit 0
INSTALL
# Use the validated bundled Python, so the installer needs no system Python.
# The descriptor survives exec and protects every switch, replacement and prune.
PYTHONDONTWRITEBYTECODE=1 "$candidate/python/bin/python3" - \
    "$APP/../runtime/serve.lock" "$work/apply.sh" "$APP" "$name" "$candidate" "$wrapper" <<'PYTHON'
import fcntl
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a+") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("splash install: stop the running Splash server before upgrading; another installer may also hold the lock.")
    os.set_inheritable(lock.fileno(), True)
    os.execv("/bin/sh", ["/bin/sh", *sys.argv[2:]])
PYTHON

echo
echo "Splash $version installed: $wrapper"
echo "Private models require your own HF_TOKEN or 'hf auth login'."
case ":$PATH:" in
    *":$dir:"*) ;;
    *) echo "Add it to your PATH first:  export PATH=\"$dir:\$PATH\"" ;;
esac
echo "  splash serve --model incoai/Qwen3.6-35B-A3B-Splash"
echo "  splash claude|opencode|codex|hermes|pi   connect a coding agent to it"
if [ -f "$APP/current/install/completions/splash.bash" ] && [ -f "$APP/current/install/completions/_splash" ]; then
    echo "  Optional shell completion (Zsh needs compinit initialized):"
    echo '    Bash: source "$HOME/Library/Application Support/Splash/app/current/install/completions/splash.bash"'
    echo '    Zsh:  source "$HOME/Library/Application Support/Splash/app/current/install/completions/_splash"'
fi
echo "  Upgrade: run this installer again.  Uninstall: rm -rf \"$APP\" \"$wrapper\""
