#!/bin/sh
# backend-benchmark serves its --max-context as serve does (model::servedFor):
# one past the model's YaRN window fails before a device or metallib is opened.
set -eu
binary=$1
configs=$2/qwen3.8-27b
work=$(mktemp -d "${TMPDIR:-/tmp}/splash-benchmark-cli.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

# An MLX Qwen3.8-27B's record and configurations, all the benchmark reads of a
# model before it opens Metal.
model=$work/model
mkdir -p "$model/draft"
echo '{"version":1,"model":"mlx-community/Qwen3.8-27B-4bit","target_format":"mlx-affine","vision_format":"none"}' \
    >"$model/model.json"
cp "$configs/config.json" "$model/config.json"
cp "$configs/draft/config.json" "$model/draft/config.json"

status=0
"$binary" "$work/missing.metallib" "$model" --max-context 1010001 >"$work/out" 2>"$work/err" || status=$?
if [ "$status" -ne 1 ] ||
    ! grep -Fqx 'backend-benchmark: Qwen3.8-27B serves at most 1010000 tokens of context' "$work/err"; then
    echo 'backend-benchmark did not refuse a context past the YaRN window as serve does' >&2
    cat "$work/err" >&2
    exit 1
fi
test ! -s "$work/out"

# These options must fail before even reading a model configuration, as well
# as before opening the missing Metal library or acquiring a device.
rejects_junction_option() {
    expected_error=$1
    shift
    status=0
    "$binary" "$work/missing.metallib" "$work/missing-model" "$@" >"$work/out" 2>"$work/err" || status=$?
    if [ "$status" -ne 1 ] ||
        ! grep -Fqx "backend-benchmark: $expected_error" "$work/err" ||
        test -s "$work/out"; then
        echo "backend-benchmark did not reject the junction arguments before model loading: $*" >&2
        cat "$work/err" >&2
        exit 1
    fi
}

rejects_junction_option 'junction benchmark options take positive integers' --junction-shared-gain 0
rejects_junction_option 'junction benchmark options take positive integers' --junction-shared-gain text
rejects_junction_option 'junction benchmark options take positive integers' --junction-shared-gain 512x
rejects_junction_option 'junction benchmark options take positive integers' --junction-shared-gain 4294967296
rejects_junction_option 'junction benchmark option is outside its supported range' --junction-shared-gain 768
rejects_junction_option 'junction benchmark option is outside its supported range' --junction-shared-gain 2048
rejects_junction_option 'junction benchmark option is outside its supported range' --junction-reuse-count 17
rejects_junction_option 'junction benchmark option is outside its supported range' --junction-output-tokens 513
rejects_junction_option 'benchmark option requires a value' --junction-output-tokens

echo 'backend-benchmark CLI validation: PASS'
