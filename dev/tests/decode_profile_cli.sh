#!/bin/sh
# decode-profile sends its chat and counting requests in the model's own
# tokens: built in for the Qwen families and Flint, by --chat-tokens for
# another family's model, whose vocabulary lacks Qwen's tokens. Without
# --chat-tokens such a model, a malformed --chat-tokens file or a prompt
# token outside the vocabulary is refused before a device or metallib is
# opened.
set -eu
binary=$1
configs=$2/inkling-2-mini
work=$(mktemp -d "${TMPDIR:-/tmp}/splash-decode-profile-cli.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

# An MLX Inkling-2-Mini's record and configurations, all the profile reads of a
# model before it opens Metal (its vocabulary: 201024).
model=$work/model
mkdir -p "$model/draft"
echo '{"version":1,"model":"thinkingmachines/Inkling-2-Mini","target_format":"mlx-affine","vision_format":"none"}' \
    >"$model/model.json"
cp "$configs/config.json" "$model/config.json"
cp "$configs/draft/config.json" "$model/draft/config.json"

refuses() {
    message=$1
    shift
    status=0
    "$binary" "$work/missing.metallib" "$model" "$@" >"$work/out" 2>"$work/err" || status=$?
    if [ "$status" -ne 1 ] || ! grep -Fqx "decode-profile: $message" "$work/err" || [ -s "$work/out" ]; then
        echo "decode-profile did not refuse $*: $message" >&2
        cat "$work/err" >&2
        exit 1
    fi
}
refuses "the built-in prompts are the Qwen families' and Flint's tokens, not thinkingmachines/Inkling-2-Mini's: \
--chat-tokens FILE gives its own, as \`python -m dev.benchmarks.chat_prompts decode-profile MODEL_ROOT FILE\` renders them"
refuses "the built-in prompts are the Qwen families' and Flint's tokens, not thinkingmachines/Inkling-2-Mini's: \
--chat-tokens FILE gives its own, as \`python -m dev.benchmarks.chat_prompts decode-profile MODEL_ROOT FILE\` renders them" \
    --prompt counting
refuses "cannot read --chat-tokens $work/missing" --chat-tokens "$work/missing"
printf '1 2\n3\n4\n' >"$work/chat"
refuses "--chat-tokens takes 4 lines of tokens: the chat request's user header, its sentence and its assistant \
header, and the counting request" --chat-tokens "$work/chat"
printf '1 2\n3\n\n5\n' >"$work/chat"
refuses "--chat-tokens takes 4 lines of tokens: the chat request's user header, its sentence and its assistant \
header, and the counting request" --chat-tokens "$work/chat"
printf '1 2\n3 x\n4\n5\n' >"$work/chat"
refuses '--chat-tokens reads lines of token ids' --chat-tokens "$work/chat"
printf '1 2\n3\n4\n5 201024\n' >"$work/chat"
refuses "--chat-tokens token 201024 lies outside the model's vocabulary of 201024 tokens" --chat-tokens "$work/chat"
echo '5 201024' >"$work/prompt"
refuses "--prompt token 201024 lies outside the model's vocabulary of 201024 tokens" --prompt "$work/prompt"

# Tokens inside the vocabulary pass: the profile goes on to open the metallib.
printf '1 2\n3\n4\n5 201023\n' >"$work/chat"
status=0
"$binary" "$work/missing.metallib" "$model" --chat-tokens "$work/chat" >"$work/out" 2>"$work/err" || status=$?
if [ "$status" -ne 1 ] || ! grep -Fq 'missing.metallib' "$work/err"; then
    echo 'decode-profile refused --chat-tokens inside the vocabulary' >&2
    cat "$work/err" >&2
    exit 1
fi
echo 'decode-profile CLI validation: PASS'
