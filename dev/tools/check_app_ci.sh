#!/bin/sh
# CI app tests assert animation frames. Preserve the runner's original setting.
set -eu
motion_domain=com.apple.universalaccess
if motion_before=$(defaults read "$motion_domain" reduceMotion 2>/dev/null); then
    case "$motion_before" in 0|1) ;; *) echo 'Unexpected Reduce Motion value' >&2; exit 1;; esac
else
    motion_before=absent
fi
restore_motion() {
    app_status=$?
    trap - 0
    if [ "$motion_before" = absent ]; then
        if defaults read "$motion_domain" reduceMotion >/dev/null 2>&1; then
            defaults delete "$motion_domain" reduceMotion || exit 1
        fi
        if defaults read "$motion_domain" reduceMotion >/dev/null 2>&1; then
            echo 'Could not restore absent Reduce Motion preference' >&2
            exit 1
        fi
    else
        defaults write "$motion_domain" reduceMotion -bool "$motion_before" || exit 1
        restored=$(defaults read "$motion_domain" reduceMotion) || exit 1
        [ "$restored" = "$motion_before" ] || exit 1
    fi
    exit "$app_status"
}
trap restore_motion 0
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
defaults write "$motion_domain" reduceMotion -bool false
[ "$(defaults read "$motion_domain" reduceMotion)" = 0 ]
make check-app
