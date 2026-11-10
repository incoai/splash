#!/usr/bin/env python3
"""Render the qualified serial result from the saved numerical evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Evidence root containing results/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads((args.root / "results/block16-vs-m16-final.json").read_text())
    assert record["qualified"] and len(record["groups"]) == 3
    first = record["groups"][0]
    primary = [c for c in first["cases"] if not c["key"].startswith("count_secondary:")]
    cycles = [sum(c["cycles"][i] for c in primary) for i in (0, 1)]
    changes = [g["primary_throughput_change"] * 100 for g in record["groups"]]
    assert cycles == [1303, 1062]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.hashsalt": "dspine-20261110",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"wspace": 0.4})
    fig.patch.set_facecolor("#ffffff")
    colors = ["#617187", "#007f7b"]
    axes[0].bar(["8 rows + M16", "16 rows"], cycles, color=colors, width=0.58)
    axes[0].set_ylim(0, 1500)
    axes[0].set_ylabel("Speculative cycles, 30 primary cases")
    axes[0].set_title("Same complete greedy outputs", loc="left", fontsize=12)
    for i, value in enumerate(cycles):
        axes[0].text(i, value + 35, f"{value:,}", ha="center", fontweight="bold")
    axes[1].bar(["Group 1", "Group 2", "Group 3"], changes, color=colors[1], width=0.58)
    axes[1].set_ylim(0, 20)
    axes[1].set_ylabel("Decode throughput change (%)")
    axes[1].set_title("Three fixed ABBA groups", loc="left", fontsize=12)
    for i, value in enumerate(changes):
        axes[1].text(i, value + 0.5, f"+{value:.2f}%", ha="center", fontweight="bold")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color="#e5e9ec", linewidth=0.8)
    fig.suptitle(
        "Sixteen-row DSpine: serial research comparison",
        x=0.07,
        ha="left",
        y=0.98,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.89,
        "M5 Pro, 20 GPU cores / 48 GB; short prompts (33-78 tokens); INT8 KV / FP32 state",
        fontsize=10,
        color="#42546a",
    )
    fig.text(
        0.07,
        0.035,
        "Research prototype. Registered output/operating/noise gates passed. Some prose regresses.\n"
        "Serial native requests; no concurrent-performance or general DFlash2 ranking established.\n"
        "G1 changed AC/battery; G2/G3 AC at recorded boundaries. AC was not a registered gate.",
        fontsize=9,
        color="#42546a",
        linespacing=1.5,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.77, bottom=0.25)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output / "serial16.svg", metadata={"Date": None}, facecolor="white"
    )
    svg = args.output / "serial16.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fig.savefig(args.output / "serial16.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
