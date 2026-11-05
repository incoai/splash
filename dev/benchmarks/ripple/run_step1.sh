#!/bin/sh
# Step 1 of the Ripple draft-kernel prototypes: build the variant metallib and
# run the sweep against the production kernels. The sweep's h10 case asks the
# device about the one-80-row-group design; its refusal (or acceptance) is the
# receipt for the two-40-row-group design, and lands in the run's log. The
# run's output lands in the workspace artifacts tree, whose path is printed.
#
# usage: run_step1.sh [sweep options...]
set -eu
cd "$(dirname "$0")/../../.."

make ripple-step1

art="${SPLASH_STEP1_ARTIFACTS:-$HOME/local-ai-project/artifacts/splash-ripple/step1-draft-kernels/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$art"
{
    echo "run: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host: $(sysctl -n hw.model) macOS $(sw_vers -productVersion)"
    echo "worktree: $(pwd)"
    echo "variant metallib: sha256 $(shasum -a 256 build/ripple/ripple_draft_variant.metallib | cut -d' ' -f1)"
    echo "production metallib: sha256 $(shasum -a 256 build/splash.metallib | cut -d' ' -f1)"
    echo "metal flags: $(make -s print-prod-metalflags)"
} > "$art/receipts.txt"

build/engine-tests/ripple-draft-sweep build/ripple/ripple_draft_variant.metallib \
    --baseline build/splash.metallib "$@" 2>&1 | tee "$art/sweep.log"
echo "artifacts: $art"
