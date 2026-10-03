#!/bin/sh
# Check the actual CLI parser without opening a Metal device or model.
set -eu
binary=$1
work=$(mktemp -d "${TMPDIR:-/tmp}/splash-tuning-cli.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

reject() {
    expected=$1
    shift
    status=0
    "$binary" "$work/missing.metallib" "$work/missing-model" "$@" \
        >"$work/out" 2>"$work/err" || status=$?
    if [ "$status" -ne 1 ] || ! grep -Fq -- "$expected" "$work/err"; then
        echo "tune-kernels accepted or misclassified arguments: $*" >&2
        cat "$work/err" >&2
        exit 1
    fi
    test ! -s "$work/out"
}

# Reaching an unknown trailing option proves that preceding values parsed.
sentinel='unknown option or missing value: --sentinel'
reject "$sentinel" --sentinel
reject "$sentinel" --candidates --sentinel
for value in 12 13 32 63 64; do
    reject "$sentinel" --pairs "$value" --sentinel
done
for value in 0 11 65 12.5 1e100 18446744073709551616 12x 0x10 nan inf +12 '' ' 12' '12 '; do
    reject "--pairs requires an integer between 12 and 64" --pairs "$value"
done
reject 'unknown option or missing value:' --pairs -1
reject 'unknown option or missing value: --pairs' --pairs
reject 'unknown option or missing value: --pairs' --pairs --candidates
reject 'unknown option or missing value: --seconds' --seconds
reject 'unknown option or missing value: --seconds' --seconds --candidates
for value in 0 1x nan inf ''; do
    reject '--seconds requires a positive number' --seconds "$value"
done
reject "$sentinel" --seconds 0.25 --pairs 64 --candidates --sentinel
reject "$sentinel" --seconds 1e2 --pairs 12 --sentinel
echo 'tune-kernels CLI validation: PASS'
