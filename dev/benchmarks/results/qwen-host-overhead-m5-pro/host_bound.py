import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def exclusive(begin, end, blocks):
    intervals = sorted(
        (max(begin, a), min(end, b)) for a, b in blocks if a < end and b > begin
    )
    merged = []
    for a, b in intervals:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return end - begin - sum(b - a for a, b in merged)


assert exclusive(0, 10, [(1, 5), (3, 7)]) == 4
assert exclusive(0, 10, [(-1, 2), (8, 11)]) == 6
reports = []
for model in ["moe", "dense"]:
    path = ROOT / f"build/deepinfer-plan/quiet/{model}-timeline/server-0-trace.json"
    if not path.exists():
        continue
    doc = json.loads(path.read_text())
    events = [e for e in doc["traceEvents"] if e.get("ph") == "X"]
    blocks = [
        (e["ts"], e["ts"] + e["dur"])
        for e in events
        if e["name"] in ["backend.prepare", "metal.encode"]
    ]
    parents = [e for e in events if e["name"] == "runtime.decode_submit"]
    bywidth = []
    for width in sorted(set(e["args"]["width"] for e in parents)):
        rows = [e for e in parents if e["args"]["width"] == width]
        times = [exclusive(e["ts"], e["ts"] + e["dur"], blocks) / 1000 for e in rows]
        bywidth.append(
            {
                "width": width,
                "count": len(times),
                "median_submit_ms": statistics.median(e["dur"] / 1000 for e in rows),
                "median_remainder_ms": statistics.median(times),
                "maximum_remainder_ms": max(times),
            }
        )
    reports.append(
        {
            "model": model,
            "widths": bywidth,
            "method": "Per-parent duration minus the union of intersecting backend.prepare and metal.encode host intervals. Nested phase medians are not subtracted.",
        }
    )
    print(model, bywidth)
(ROOT / "build/deepinfer-plan/quiet/host-bound.json").write_text(
    json.dumps(reports, indent=2) + "\n"
)
