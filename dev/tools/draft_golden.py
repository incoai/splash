#!/usr/bin/env python3
"""Grades the installed model's DFlash2 draft against a reference's golden
vectors: for each golden block, teacher-logits runs the block's anchor after
the golden captured rows, which the draft reads in place of the target's own
(--captured-rows), and writes the draft's lattice of that cycle (--draft yes);
its candidates, unary scores, softcapped selector scores and greedy
proposals are graded by the golden set's rules.

    python -m dev.tools.draft_golden METALLIB MODEL_ROOT GOLDEN_DIR WORK_DIR [--set af4g64]

GOLDEN_DIR holds NAME.safetensors files and manifest.json in the layout of
the Inkling-2-Mini oracle's golden vectors (as their README states it): each
file's tokens, captures (bf16 [tokens, captured width]), block_anchor_position
and, per block, logits_topk_ids/values, logits_top24_ids/values,
selector_predecessor_ids, selector_capped, selector_margin and proposals,
prefixed by the set graded (af4g64., the draft as Splash quantizes its
projections, by default). The draft runs with bf16 activations and KV, as the
rules assume. Rules, per block:
- top-16 sets: an id in one list alone only within the 16th logit's
  tolerance (LOGIT_TOL) of the reference's 16th logit, and none past the
  head's tokens;
- unary scores of the ids in both: |d| <= LOGIT_TOL, row RMS <= LOGIT_ROW_RMS;
- softcapped scores along the reference's greedy chain, of the ids in both,
  where its predecessor is among the draft's candidates: |d| <= CAPPED_TOL;
- proposals: a position whose earlier proposals matched, whose top-16 sets
  are equal and whose reference margin is at least MARGIN must propose the
  reference's token; the others are counted.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

LOGIT_TOL = 1.5
LOGIT_ROW_RMS = 1.0
CAPPED_TOL = 1.0
MARGIN = 1.0
POSITIONS = 7
CANDIDATES = 16


def read_safetensors(path: Path) -> dict:
    """The tensors of a safetensors file as numpy arrays, BF16 widened to fp32."""
    data = path.read_bytes()
    size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + size])
    header.pop("__metadata__", None)
    kinds = {"F32": "<f4", "I32": "<i4", "BF16": "<u2", "I64": "<i8"}
    tensors = {}
    for name, entry in header.items():
        begin, end = entry["data_offsets"]
        values = np.frombuffer(
            data[8 + size + begin : 8 + size + end], kinds[entry["dtype"]]
        )
        if entry["dtype"] == "BF16":
            values = (values.astype(np.uint32) << 16).view(np.float32)
        tensors[name] = values.reshape(entry["shape"])
    return tensors


def softcap(values, cap):
    return (
        cap * np.tanh(np.asarray(values, np.float64) / cap)
        if cap
        else np.asarray(values, np.float64)
    )


def grade(lattice: dict, ref: dict, block: int, cap: float, tokens: int) -> dict:
    ids = np.array(lattice["candidates"]).reshape(POSITIONS, CANDIDATES)
    unary = np.array(lattice["unary"], np.float64).reshape(POSITIONS, CANDIDATES)
    edges = np.array(lattice["edges"], np.float64).reshape(
        POSITIONS, CANDIDATES, CANDIDATES
    )
    ref_ids, ref_unary = ref["logits_topk_ids"][block], ref["logits_topk_values"][block]
    top24_ids, top24 = ref["logits_top24_ids"][block], ref["logits_top24_values"][block]
    failures = []
    if (ids >= tokens).any():
        failures.append("a candidate past the head's tokens")
    worst_logit = worst_rms = worst_capped = 0.0
    equal_sets = []
    for p in range(POSITIONS):
        ours, theirs = set(ids[p].tolist()), set(ref_ids[p].tolist())
        sixteenth = ref_unary[p][-1]
        for token in ours ^ theirs:
            where = np.where(top24_ids[p] == token)[0]
            if not where.size or abs(top24[p][where[0]] - sixteenth) > LOGIT_TOL:
                failures.append(
                    f"position {p}: token {token} in one top-16 alone, past the 16th logit's tolerance"
                )
        equal_sets.append(ours == theirs)
        shared = [
            (i, int(np.where(ref_ids[p] == token)[0][0]))
            for i, token in enumerate(ids[p])
            if token in theirs
        ]
        if shared:
            d = np.array([unary[p][i] - ref_unary[p][j] for i, j in shared])
            worst_logit = max(worst_logit, float(np.abs(d).max()))
            worst_rms = max(worst_rms, float(np.sqrt((d**2).mean())))
    # Softcapped scores along the reference's chain: the anchor, then each
    # position's reference predecessor where the draft holds it.
    predecessors, capped = (
        ref["selector_predecessor_ids"][block],
        ref["selector_capped"][block],
    )
    for p in range(POSITIONS):
        row = 0
        if p:
            held = np.where(ids[p - 1] == predecessors[p])[0]
            if not held.size:
                continue
            row = int(held[0])
        ours = softcap(unary[p] + edges[p][row], cap)
        for i, token in enumerate(ids[p]):
            where = np.where(ref_ids[p] == token)[0]
            if where.size:
                worst_capped = max(
                    worst_capped, float(abs(ours[i] - capped[p][where[0]]))
                )
    if worst_logit > LOGIT_TOL or worst_rms > LOGIT_ROW_RMS:
        failures.append(
            f"unary scores off by {worst_logit:.3f} (row RMS {worst_rms:.3f})"
        )
    if worst_capped > CAPPED_TOL:
        failures.append(f"softcapped scores off by {worst_capped:.3f}")
    # The draft's greedy chain against the reference's proposals.
    proposals, margins = ref["proposals"][block], ref["selector_margin"][block]
    graded = equal = skipped = 0
    row, matched = 0, True
    for p in range(POSITIONS):
        slot = int(np.argmax(softcap(unary[p] + edges[p][row], cap)))
        token = int(ids[p][slot])
        if matched and equal_sets[p] and margins[p] >= MARGIN:
            graded += 1
            equal += token == proposals[p]
            if token != proposals[p]:
                failures.append(
                    f"position {p}: proposed {token}, the reference {proposals[p]}"
                )
        else:
            skipped += 1
        matched = matched and token == proposals[p]
        row = slot
    return {
        "failures": failures,
        "unary_max": worst_logit,
        "unary_rms": worst_rms,
        "capped_max": worst_capped,
        "proposals": (graded, equal, skipped),
        "equal_sets": sum(equal_sets),
    }


def target_tokens(model_root: Path, flash: dict) -> int:
    """The tokens a draft's candidates may name: its config's valid_vocab_size
    where the config states it, else its target's, the target text config's
    unpadded_vocab_size or, where every head row is a token, its vocab_size, as
    the engine takes them (ModelDescriptor.mm, readDraftLayout)."""
    if "valid_vocab_size" in flash:
        return int(flash["valid_vocab_size"])
    config = json.loads((model_root / "config.json").read_text())
    text = config.get("text_config", config)
    return int(text.get("unpadded_vocab_size", text["vocab_size"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metallib")
    parser.add_argument("model_root", type=Path)
    parser.add_argument("golden", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument(
        "--set",
        default="af4g64",
        help="the golden set graded (af4g64, or '' for unprefixed)",
    )
    parser.add_argument("--teacher-logits", default="build/engine-tests/teacher-logits")
    args = parser.parse_args()
    flash = json.loads((args.model_root / "draft" / "config.json").read_text())[
        "dflash_config"
    ]
    cap = float(flash.get("final_logit_softcapping", 0.0))
    tokens = target_tokens(args.model_root, flash)
    manifest = json.loads((args.golden / "manifest.json").read_text())
    args.work.mkdir(parents=True, exist_ok=True)
    prefix = args.set + "." if args.set else ""
    failed = 0
    for name, entry in manifest["prompts"].items():
        golden = read_safetensors(args.golden / entry["file"])
        tokens = golden["tokens"].astype("<u4")
        tokens_file = args.work / f"{name}.u32"
        tokens.tofile(tokens_file)
        ref = {
            key[len(prefix) :]: value
            for key, value in golden.items()
            if key.startswith(prefix)
        }
        ref = {**golden, **ref} if prefix else golden
        for block, anchor in enumerate(golden["block_anchor_position"].tolist()):
            rows_file = args.work / f"{name}-{anchor}.rows.f32"
            golden["captures"][:anchor].astype("<f4").tofile(rows_file)
            out = args.work / name
            command = [
                args.teacher_logits,
                args.metallib,
                str(args.model_root),
                str(tokens_file),
                str(out),
                "--window",
                f"{anchor + 8}:8",
                "--draft",
                "yes",
                "--activations",
                "bf16",
                "--kv-format",
                "bf16",
                "--captured-rows",
                str(rows_file),
            ]
            run = subprocess.run(command, capture_output=True, text=True)
            if run.returncode:
                print(
                    f"FAIL {name} block {block} (anchor {anchor}): teacher-logits: {run.stderr.strip()}"
                )
                failed += 1
                continue
            lattice = json.loads(
                (args.work / f"{name}-{anchor + 8}.draft.json").read_text()
            )
            result = grade(lattice, ref, block, cap, tokens)
            graded, equal, skipped = result["proposals"]
            print(
                f"{'FAIL ' if result['failures'] else ''}{name} block {block} (anchor {anchor}): "
                f"top-16 sets equal at {result['equal_sets']}/{POSITIONS}, unary max |d| {result['unary_max']:.3f} "
                f"(row RMS {result['unary_rms']:.3f}), softcapped max |d| {result['capped_max']:.3f}, "
                f"proposals {equal}/{graded} graded equal ({skipped} not graded)"
                + "".join(f"\n    {failure}" for failure in result["failures"]),
                flush=True,
            )
            failed += bool(result["failures"])
    print("draft golden: " + (f"{failed} blocks failed" if failed else "passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
