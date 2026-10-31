#!/bin/sh
set -eu

desktop_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname -- "$desktop_root")
app_bundle="$repo_root/build/Splash.app"

swift build --package-path "$desktop_root" --configuration release
binary_dir=$(swift build --package-path "$desktop_root" --configuration release --show-bin-path)
mkdir -p "$app_bundle/Contents/MacOS"
mkdir -p "$app_bundle/Contents/Resources"
cp "$binary_dir/SplashDesktop" "$app_bundle/Contents/MacOS/SplashDesktop"
cp "$desktop_root/Info.plist" "$app_bundle/Contents/Info.plist"
swift "$desktop_root/Icon.swift" "$repo_root/build/Splash.iconset"
iconutil -c icns "$repo_root/build/Splash.iconset" -o "$app_bundle/Contents/Resources/Splash.icns"
codesign --force --sign - "$app_bundle"
printf 'Built %s\n' "$app_bundle"
