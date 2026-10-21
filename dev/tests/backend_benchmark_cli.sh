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
echo 'backend-benchmark CLI validation: PASS'
