#!/bin/sh
set -eu

invocation_dir=$PWD
cd "$(dirname "$0")/.."
archive=${SPLASH_RUNTIME_ARCHIVE:-}
case "$archive" in
    /*|'') ;;
    *) archive="$invocation_dir/$archive" ;;
esac
checksum=${SPLASH_RUNTIME_SHA256:-}
runtime_version=${SPLASH_RUNTIME_VERSION:-}

fail() { printf '%s\n' "release: $*" >&2; exit 1; }
[ -n "$archive" ] && [ -f "$archive" ] || fail "SPLASH_RUNTIME_ARCHIVE is missing"
[ -n "$runtime_version" ] || fail "SPLASH_RUNTIME_VERSION is missing"
[ -n "$checksum" ] || fail "SPLASH_RUNTIME_SHA256 is missing"
actual=$(shasum -a 256 "$archive" | awk '{print $1}')
[ "$actual" = "$checksum" ] || fail "runtime checksum mismatch"

expected_folder="splash-$runtime_version-arm64-macos26"
entries=$(tar -tzf "$archive") || fail "cannot read runtime archive"
[ -n "$entries" ] || fail "runtime archive is empty"
printf '%s\n' "$entries" | awk -v prefix="$expected_folder/" '
  $0 != prefix && index($0, prefix) != 1 { bad = 1 }
  { for (i = 1; i <= split($0, segments, "/"); i++) if (segments[i] == "..") bad = 1 }
  END { exit bad }
' || fail "unexpected runtime archive layout"
validation_dir=$(mktemp -d "${TMPDIR:-/tmp}/splash-release.XXXXXXXX") || fail "cannot create validation directory"
trap 'rm -rf "$validation_dir"' 0
runtime_dir="$validation_dir/$expected_folder"
tar -xzf "$archive" -C "$validation_dir" || fail "cannot extract runtime archive"
[ -x "$runtime_dir/python/bin/python3" ] || fail "runtime archive is missing executable python/bin/python3"
[ -f "$runtime_dir/install/launcher.py" ] || fail "runtime archive is missing install/launcher.py"
[ -f "$runtime_dir/release.json" ] || fail "runtime archive is missing release.json"
ruby -rjson -e 'abort "runtime version mismatch" unless JSON.parse(File.read(ARGV[0]))["version"] == ARGV[1]' \
    "$runtime_dir/release.json" "$runtime_version" || fail "runtime archive has an invalid release.json"
PYTHONPATH="$runtime_dir" "$runtime_dir/python/bin/python3" -u \
    "$runtime_dir/install/launcher.py" --help >/dev/null || fail "runtime launcher self-check failed"

if [ "${1:-}" = "--validate" ]; then
    printf '%s\n' "Release inputs valid."
    exit 0
fi

: "${SPLASH_RUNTIME_URL:?Set SPLASH_RUNTIME_URL to an HTTPS URL}"
: "${DEVELOPER_ID_APPLICATION:?Set DEVELOPER_ID_APPLICATION}"
: "${NOTARY_PROFILE:?Set NOTARY_PROFILE}"
: "${SPLASH_APP_VERSION:?Set SPLASH_APP_VERSION}"
: "${SPLASH_APP_BUILD:?Set SPLASH_APP_BUILD}"
case "$SPLASH_RUNTIME_URL" in https://*) ;; *) fail "runtime URL must use HTTPS" ;; esac
marketing_version=$(ruby -e '
  match = /\A([0-9]+\.[0-9]+\.[0-9]+)-rc\.[1-9][0-9]*\z/.match(ARGV.fetch(0))
  abort "expected an RC version such as 0.1.0-rc.1" unless match
  print match[1]
' "$SPLASH_APP_VERSION") || fail "invalid SPLASH_APP_VERSION"
case "$SPLASH_APP_BUILD" in ''|*[!0-9]*) fail "SPLASH_APP_BUILD must be a number" ;; esac
[ "$SPLASH_APP_BUILD" -gt 0 ] || fail "SPLASH_APP_BUILD must be positive"
remote_archive="$validation_dir/remote-runtime.tar.gz"
curl --fail --location --silent --show-error --output "$remote_archive" "$SPLASH_RUNTIME_URL" || \
    fail "cannot download runtime archive from SPLASH_RUNTIME_URL"
[ "$(shasum -a 256 "$remote_archive" | awk '{print $1}')" = "$checksum" ] || \
    fail "remote runtime checksum mismatch"

mkdir -p .build/releases
ruby -rjson -e '
  File.write(ARGV[0], JSON.pretty_generate({
    "version" => ARGV[1], "archiveURL" => ARGV[2], "sha256" => ARGV[3],
    "minimumCompatibleVersion" => ARGV[4]
  }) + "\n")
' .build/runtime-manifest.json "$runtime_version" "$SPLASH_RUNTIME_URL" "$checksum" "${SPLASH_MINIMUM_RUNTIME_VERSION:-$runtime_version}"
ruby project.rb
xcodebuild -project Splash.xcodeproj -scheme Splash -configuration Release \
    -destination 'platform=macOS' -derivedDataPath .build/DerivedData \
    MARKETING_VERSION="$marketing_version" CURRENT_PROJECT_VERSION="$SPLASH_APP_BUILD" \
    SPLASH_RELEASE_VERSION="$SPLASH_APP_VERSION" \
    CODE_SIGNING_ALLOWED=YES CODE_SIGN_STYLE=Manual \
    CODE_SIGN_IDENTITY="$DEVELOPER_ID_APPLICATION" clean build

app=.build/DerivedData/Build/Products/Release/Splash.app
[ -d "$app" ] || fail "Release app was not built"
info="$app/Contents/Info.plist"
[ "$(plutil -extract CFBundleShortVersionString raw -o - "$info")" = "$marketing_version" ] || \
    fail "Release app marketing version is incorrect"
[ "$(plutil -extract CFBundleVersion raw -o - "$info")" = "$SPLASH_APP_BUILD" ] || \
    fail "Release app build number is incorrect"
[ "$(plutil -extract SplashReleaseVersion raw -o - "$info")" = "$SPLASH_APP_VERSION" ] || \
    fail "Release app display version is incorrect"
codesign --verify --deep --strict --verbose=2 "$app" || fail "Release app signature is invalid"

notarize() {
    result=$(xcrun notarytool submit "$1" --keychain-profile "$NOTARY_PROFILE" \
        --wait --output-format json) || fail "notarization request failed for $1"
    printf '%s\n' "$result" | ruby -rjson -e '
      response = JSON.parse(STDIN.read)
      abort "notarization status: #{response.fetch("status", "unknown")} (#{response["id"]})" \
        unless response["status"] == "Accepted"
    ' || fail "notarization rejected for $1"
}

zip="$validation_dir/Splash.app.zip"
ditto -c -k --keepParent "$app" "$zip"
notarize "$zip"
xcrun stapler staple "$app"
xcrun stapler validate "$app"

package_dir="$validation_dir/package"
mkdir "$package_dir"
ditto "$app" "$package_dir/Splash.app"
cp ../LICENSE ../THIRD_PARTY_NOTICES "$package_dir/"
ln -s /Applications "$package_dir/Applications"
dmg=".build/releases/Splash-$SPLASH_APP_VERSION.dmg"
hdiutil create -volname "Splash $SPLASH_APP_VERSION" -srcfolder "$package_dir" \
    -format UDZO -ov "$dmg"
hdiutil verify "$dmg"
notarize "$dmg"
xcrun stapler staple "$dmg"
xcrun stapler validate "$dmg"

printf '%s\n' "Release DMG ready: desktop/$dmg"
shasum -a 256 "$dmg"
