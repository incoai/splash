#!/bin/sh
# Reject incompatible output/epilogue combinations before opening Metal.
set -eu
binary=$1
work=$(mktemp -d "${TMPDIR:-/tmp}/splash-projection-cli.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

reject() {
    expected=$1
    shift
    status=0
    "$binary" "$work/missing.metallib" "$@" >"$work/out" 2>"$work/err" || status=$?
    if [ "$status" -ne 2 ] || ! grep -Fqx "$expected" "$work/err"; then
        echo 'gguf-projection-benchmark did not reject incompatible options before opening Metal' >&2
        cat "$work/err" >&2
        exit 1
    fi
    test ! -s "$work/out"
}

incompatible='output=f32 takes one quantized plain decode projection; gateup=both takes quantized decode gateup'
reject "$incompatible" af4g64 256 512 residual 1 output=f32
reject "$incompatible" af4g64 256 512 none 1 prefill=64 output=f32
reject "$incompatible" af4g64+af4g64 128+128 512 none 1 output=f32
reject "$incompatible" af4g64 256 512 none 1 gateup=both
reject 'unknown or repeated benchmark option: output=f32' af4g64 256 512 none 1 output=bf16 output=f32
reject 'unknown or repeated benchmark option: output=fp16' af4g64 256 512 none 1 output=fp16
echo 'gguf-projection-benchmark CLI validation: PASS'
