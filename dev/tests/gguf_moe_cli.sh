#!/bin/sh
# Invalid numeric arguments must fail before opening a Metal device or metallib.
set -eu
binary=$1
work=$(mktemp -d "${TMPDIR:-/tmp}/splash-moe-cli.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

reject() {
    status=0
    "$binary" "$work/missing.metallib" "$@" >"$work/out" 2>"$work/err" || status=$?
    if [ "$status" -ne 2 ] || ! grep -Fq 'rounds must be a positive uint32 and the decode pool 8 to 256' "$work/err"; then
        echo "gguf-moe-benchmark accepted or misclassified arguments: $*" >&2
        cat "$work/err" >&2
        exit 1
    fi
    test ! -s "$work/out"
}

for value in 0 -1 +31 31x 31.0 0x1f '' ' 31' '31 ' 4294967296 4294967327 18446744073709551616; do
    reject "$value" q4k q5k 24
done
for value in 0 7 257 -1 +8 256junk 8.0 0x10 '' ' 8' '8 ' 4294967296 4294967320 18446744073709551616; do
    reject 31 q4k q5k "$value"
done
echo 'gguf-moe-benchmark CLI validation: PASS'
