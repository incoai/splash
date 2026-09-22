"""Inline splash's #include "…" headers into one MSL source (runtime compiler has no include paths)."""

import os
import re
import sys

ROOT = os.path.expanduser("~/dev/splash/runtime")
seen = set()


def inline(path):
    out = []
    for line in open(path):
        m = re.match(r'\s*#include\s+"([^"]+)"', line)
        if m:
            inc = m.group(1)
            cand = [os.path.join(ROOT, inc), os.path.join(os.path.dirname(path), inc)]
            full = next((c for c in cand if os.path.exists(c)), None)
            if full is None:
                raise SystemExit(f"missing include {inc} from {path}")
            full = os.path.realpath(full)
            if full in seen:
                continue
            seen.add(full)
            out.append(f"// ---- begin {inc}\n")
            out.append(inline(full))
            out.append(f"// ---- end {inc}\n")
        elif line.strip() == "#pragma once":
            continue
        else:
            out.append(line)
    return "".join(out)


src = "".join(inline(os.path.join(ROOT, f)) for f in sys.argv[1:])
sys.stdout.write(src)
