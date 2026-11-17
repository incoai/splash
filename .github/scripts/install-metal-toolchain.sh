#!/bin/bash
# Reuse an installed compiler; transient Apple catalog failures must not
# remove a working toolchain or silently skip the subsequent native checks.
set -euo pipefail

compiler_ready() {
  xcrun -sdk macosx metal --version >/dev/null 2>&1
}

if compiler_ready; then
  echo "Metal compiler is already available."
  exit 0
fi

for attempt in 1 2 3; do
  if xcodebuild -downloadComponent MetalToolchain && compiler_ready; then
    echo "Metal compiler is available after installation."
    exit 0
  fi
  echo "Metal toolchain attempt ${attempt}/3 did not provide a compiler." >&2
  if [ "$attempt" -lt 3 ]; then
    sleep "$((attempt * 15))"
  fi
done

echo "Metal compiler is unavailable after three installation attempts." >&2
exit 1
