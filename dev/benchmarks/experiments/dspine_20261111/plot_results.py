"""Publication figures from saved component summaries, never timing inputs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
root = args.root
out = root / "figures"
out.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.hashsalt": "dspine-13h-20261111",
    }
)
blue, orange, gray = "#246BCE", "#D56B26", "#687483"


def read(name):
    return json.loads((root / "results" / name).read_text())


def save(fig, name):
    svg = out / (name + ".svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fig.savefig(out / (name + ".png"), dpi=180, metadata={"Software": "Matplotlib"})
    plt.close(fig)


wide = read("dspine-wide-gateup-2026-11-10.json")["controlled"]
assert wide["3"]["qualified"] and not wide["4"]["qualified"]
groups = wide["3"]["groups"]
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(3)
base = [g["primary_decode_abba"]["baseline"] for g in groups]
candidate = [g["primary_decode_abba"]["candidate"] for g in groups]
ax.bar(
    x - 0.18, base, width=0.36, label="Original sixteen-row gate/up policy", color=gray
)
ax.bar(x + 0.18, candidate, width=0.36, label="Existing separate passes", color=blue)
for i, g in enumerate(groups):
    ax.text(
        i + 0.18,
        candidate[i] + 0.3,
        "+%.1f%% TPS" % (100 * g["primary_throughput_change"]),
        ha="center",
        color=blue,
        fontsize=10,
        weight="bold",
    )
ax.set_xticks(x, ["ABBA group 1", "ABBA group 2", "ABBA group 3"])
ax.set_ylabel("Decode time per output token, ms\n(lower is better)")
ax.set_ylim(0, 19)
ax.set_title(
    "Qualified B3 gain from fixing the M64 gate/up policy", loc="left", weight="bold"
)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.grid(axis="y", alpha=0.15)
ax.set_axisbelow(True)
fig.text(
    0.12,
    0.06,
    "Three fixed physical lanes × sixteen rows; shared-prompt native clones.\n"
    "Historical 3-prompt component study, not a DFlash2 comparison. B4 is unqualified after AC loss.",
    fontsize=9,
    color=gray,
)
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.23, top=0.88)
save(fig, "wide-gateup-b3")

attention = read("dspine-attention-scratch-2026-11-10.json")
short = read("dspine-short-context-memory-2026-11-10.json")[
    "actual_four_lane_state_memory"
]
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
values = [
    attention["physical_workspace_saving_bytes"][str(w)] / 1048576 for w in [8, 16]
]
axes[0].barh(["Eight rows", "Sixteen rows"], values, color=blue, height=0.55)
for i, v in enumerate(values):
    axes[0].text(v + 2, i, "%.2f" % v, va="center")
axes[0].set_xlim(0, 118)
axes[0].set_xlabel("Scratch bytes saved, MiB")
axes[0].set_title("Explicit 32K served context", loc="left", weight="bold", fontsize=11)
values = [v / 1048576 for v in short["state_allocated_bytes"]]
axes[1].bar(
    ["8K draft layout", "2K draft layout"], values, color=[gray, blue], width=0.55
)
axes[1].set_ylabel("Measured state allocation, MiB")
axes[1].set_ylim(0, max(values) * 1.16)
axes[1].set_title(
    "Four lanes, 2K served context", loc="left", weight="bold", fontsize=11
)
for i, v in enumerate(values):
    axes[1].text(i, v + 30, "%.1f" % v, ha="center")
fig.suptitle("Memory reductions have separate scopes", x=0.08, ha="left", weight="bold")
fig.text(
    0.08,
    0.035,
    "2K draft layout frees 564.375 MiB of pooled/cached state; the elastic budget is unchanged.\n"
    "No throughput claim from either memory measurement.",
    fontsize=9,
    color=gray,
)
fig.subplots_adjust(left=0.12, right=0.98, wspace=0.5, bottom=0.27, top=0.79)
save(fig, "memory-reductions")

rejected = read("operator-rejections-v1.json")
n16 = read("n16-operator-v2.json")
labels = [
    "Prefill N32",
    "Prefill blocked 4×4 N64",
    "Prefill blocked 4×4 N32",
    "Paired N16, 48 live rows",
    "Paired N16, 64 live rows",
]
values = [
    v["geometric_throughput_change"] * 100 for v in rejected["prefill_candidates"]
]
values += [v["throughput_change"] * 100 for v in n16["primary_cases"]]
fig, ax = plt.subplots(figsize=(8.5, 4.3))
ax.barh(labels, values, color=orange, height=0.58)
ax.axvline(0, color=gray, linewidth=0.8)
ax.set_xlim(-18, 1)
ax.invert_yaxis()
for i, v in enumerate(values):
    ax.text(v - 0.3, i, "%.2f%%" % v, ha="right", va="center")
ax.set_xlabel("Diagnostic operator throughput change versus the existing path")
ax.set_title(
    "Numerically correct prototypes that missed nomination", loc="left", weight="bold"
)
fig.text(
    0.34,
    0.05,
    "Balanced operator screens; not controlled whole-model rates.\n"
    "Prefill values aggregate all 27 registered shapes/epilogues.",
    fontsize=9,
    color=gray,
)
fig.subplots_adjust(left=0.34, right=0.98, bottom=0.24, top=0.86)
save(fig, "operator-screens")


m48 = read("m48-operator-v3.json")["primary"]
fig, ax = plt.subplots(figsize=(8, 4.2))
values = [m48["control_ms"], m48["candidate_ms"]]
ax.bar(
    ["Existing M64 separate pass", "Research M48 separate pass"],
    values,
    color=[gray, blue],
    width=0.52,
)
for i, value in enumerate(values):
    ax.text(i, value + 0.025, "%.4f ms" % value, ha="center")
ax.set_ylim(0, 1)
ax.set_ylabel("Operator time, ms (lower is better)")
ax.set_title("48 useful rows: +34.78% operator throughput", loc="left", weight="bold")
ax.grid(axis="y", alpha=0.15)
ax.set_axisbelow(True)
fig.text(
    0.12,
    0.04,
    "AF4G64, N=17,408, K=5,120; split1 / spread enabled. All four settings retained.\n"
    "Balanced operator screen, not whole-model throughput. Allocation capacity stays 64 rows.",
    fontsize=9,
    color=gray,
)
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.24, top=0.87)
save(fig, "m48-operator")
