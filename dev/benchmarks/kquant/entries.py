#!/usr/bin/env python3
"""Per-entry results: one row per (phase, rows, shape, GGUF type) that occurs in UD-Q4_K_M, with bits/element, theoretical
decode ratio (4.5 / bits), measured ratio (splash time / best K-quant kernel time) and efficiency (measured / theoretical)."""

import collections
import json
import sys

logs = sys.argv[1:]
BITS = {
    "q4k": (4.5, 4.5),
    "iq4xs": (4.25, 4.25),
    "iq4nl": (4.5, 4.5),
    "q5k": (5.5, 5.5),
    "q6k": (6.5625, 6.625),
    "q3k": (3.4375, 3.5),
    "q80": (8.5, 8.5),
    "iq3s": (3.4375, 4.0625),
}
GT = {
    "Q4_K": "q4k",
    "IQ4_XS": "iq4xs",
    "IQ4_NL": "iq4nl",
    "Q5_K": "q5k",
    "Q6_K": "q6k",
    "Q3_K": "q3k",
    "Q8_0": "q80",
    "IQ3_S": "iq3s",
}


def base(f):
    return {"iq4xsB": "iq4xs", "iq4xsT": "iq4xs", "iq4nlB": "iq4nl"}.get(f, f)


data = collections.defaultdict(list)
splash = {}
for path in logs:
    for line in open(path):
        if not line.startswith("| ") or line.startswith("| phase"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        phase, r, shape, fmt, kernel, groups, ms = (
            c[0],
            int(c[1]),
            c[2],
            c[3],
            c[4],
            c[5],
            float(c[6]),
        )
        if fmt == "splash":
            splash[(phase, shape, r)] = min(ms, splash.get((phase, shape, r), 9e9))
        else:
            data[(phase, shape, r, base(fmt))].append((ms, kernel))
# which (shape, type) cells exist in the file, and who uses them
cells = json.load(open("/tmp/qk/m5/cells.json"))
shape_of = {
    (17408, 5120): "gate/up 17408x5120",
    (5120, 17408): "down 5120x17408",
    (10240, 5120): "attn_qkv 10240x5120",
    (1024, 5120): "attn_kv 1024x5120",
    (5120, 6144): "out 5120x6144",
    (6144, 5120): "attn_gate 6144x5120",
    (12288, 5120): "attn_q 12288x5120",
    (248320, 5120): "lm_head 248320x5120",
    (48, 5120): "ab 256x5120",
}
users = collections.defaultdict(list)
for c in cells:
    if c["role"] == "token_embd.weight":
        continue
    users[(shape_of[(c["N"], c["K"])], GT[c["type"]])].append((c["role"], c["count"]))


def who(shape, fmt):
    u = users.get((shape, fmt), [])
    return ", ".join(f"{r} x{n}" for r, n in u)


order = [
    "gate/up 17408x5120",
    "down 5120x17408",
    "attn_qkv 10240x5120",
    "attn_gate 6144x5120",
    "ab 256x5120",
    "out 5120x6144",
    "attn_q 12288x5120",
    "attn_kv 1024x5120",
    "lm_head 248320x5120",
]
fmts = ["q4k", "iq4xs", "iq4nl", "q5k", "q6k", "q3k", "q80", "iq3s"]
missing = []
for phase in ("decode", "prefill"):
    print(f"\n### {phase}\n")
    if phase == "decode":
        print(
            "| shape | type | tensors in file | rows | bits/elem GGUF (streamed) | theoretical | measured | efficiency | best kernel |\n|---|---|---|---|---|---|---|---|---|"
        )
    else:
        print(
            "| shape | type | tensors in file | rows | bits/elem GGUF (streamed) | theoretical | measured | best kernel |\n|---|---|---|---|---|---|---|---|"
        )
    for shape in order:
        for fmt in fmts:
            if (shape, fmt) not in users:
                continue
            rows_list = sorted({k[2] for k in data if k[0] == phase and k[1] == shape})
            if not rows_list:
                missing.append((phase, shape, fmt))
                continue
            for r in rows_list:
                v = data.get((phase, shape, r, fmt))
                if not v:
                    missing.append((phase, shape, fmt, r))
                    continue
                ms, kern = min(v)
                s = splash[(phase, shape, r)]
                meas = s / ms
                gb, sb = BITS[fmt]
                theo = 4.5 / gb
                kern = kern.split(" ")[0] + (" +splitK" if "splits" in kern else "")
                if phase == "decode":
                    print(
                        f"| {shape} | {fmt} | {who(shape, fmt)} | {r} | {gb:.4g} ({sb:.4g}) | {theo:.2f} | {meas:.2f} | {meas / theo:.2f} | {kern} |"
                    )
                else:
                    print(
                        f"| {shape} | {fmt} | {who(shape, fmt)} | {r} | {gb:.4g} ({sb:.4g}) | 1.00 | {meas:.2f} | {kern} |"
                    )
if missing:
    print("\nmissing cells:", missing)
