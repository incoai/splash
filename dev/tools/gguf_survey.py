#!/usr/bin/env python3
"""Survey the tensor types of GGUF files on the Hub from their headers alone.

Fetches only each file's header with a Range request, parses the GGUF v3 layout and
prints per-role tensor type counts (linear, output, token_embd, alpha_beta, small,
nextn), so a variant can be checked against the converter's supported types before
downloading tens of gigabytes:

    dev/tools/gguf_survey.py unsloth/Qwen3.8-27B-GGUF out.json Qwen3.8-27B-UD-Q5_K_M.gguf ...
"""

import collections
import json
import struct
import sys
import urllib.request

TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
}


class Trunc(Exception):
    pass


def parse(buf):
    pos = 0

    def take(n):
        nonlocal pos
        if pos + n > len(buf):
            raise Trunc()
        pos += n
        return buf[pos - n : pos]

    def u32():
        return struct.unpack("<I", take(4))[0]

    def u64():
        return struct.unpack("<Q", take(8))[0]

    def string():
        n = u64()
        return take(n).decode("utf-8", "replace")

    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    FMT = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "<?",
        10: "<Q",
        11: "<q",
        12: "<d",
    }

    def value(t, skip=False):
        if t == 8:
            return string()
        if t == 9:
            et = u32()
            n = u64()
            if et == 8:
                out = []
                for _ in range(n):
                    s = string()
                    if not skip and len(out) < 4:
                        out.append(s)
                return out if not skip else f"<{n} strings>"
            raw = take(SZ[et] * n)
            return (
                list(
                    struct.unpack(
                        "<" + FMT[et][1] * min(n, 8), raw[: SZ[et] * min(n, 8)]
                    )
                )
                if n <= 8
                else f"<{n} x {FMT[et]}>"
            )
        return struct.unpack(FMT[t], take(SZ[t]))[0]

    assert take(4) == b"GGUF", "bad magic"
    u32()  # version
    n_tensors = u64()
    n_kv = u64()
    kv = {}
    for _ in range(n_kv):
        k = string()
        t = u32()
        kv[k] = value(t, skip=k.startswith("tokenizer."))
    tensors = []
    for _ in range(n_tensors):
        name = string()
        nd = u32()
        dims = [u64() for _ in range(nd)]
        t = u32()
        off = u64()
        tensors.append(
            {"name": name, "dims": dims, "type": TYPES.get(t, str(t)), "offset": off}
        )
    align = kv.get("general.alignment", 32)
    data_offset = (pos + align - 1) // align * align
    return kv, tensors, data_offset


def fetch(url, nbytes):
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


if __name__ == "__main__":
    repo, out = sys.argv[1], sys.argv[2]
    names = sys.argv[3:]
    results = {}
    for name in names:
        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
        for nbytes in (12 << 20, 40 << 20):
            try:
                buf = fetch(url, nbytes)
                kv, tensors, data_offset = parse(buf)
                break
            except Trunc:
                continue
        else:
            print(name, "header too large")
            continue
        roles = collections.defaultdict(collections.Counter)
        for t in tensors:
            n = t["name"]
            if n.startswith("blk.64."):
                role = "nextn"
            elif n == "token_embd.weight":
                role = "token_embd"
            elif n == "output.weight":
                role = "output"
            elif any(
                s in n
                for s in (
                    ".attn_qkv.",
                    ".attn_gate.",
                    ".ssm_out.",
                    ".attn_q.",
                    ".attn_k.",
                    ".attn_v.",
                    ".attn_output.",
                    ".ffn_gate.",
                    ".ffn_up.",
                    ".ffn_down.",
                )
            ):
                role = "linear"
            elif ".ssm_alpha." in n or ".ssm_beta." in n:
                role = "alpha_beta"
            else:
                role = "small"
            roles[role][t["type"]] += 1
        results[name] = {
            "file_type": kv.get("general.file_type"),
            "block_count": kv.get("qwen35.block_count"),
            "arch": kv.get("general.architecture"),
            "n_tensors": len(tensors),
            "data_offset": data_offset,
            "roles": {r: dict(c) for r, c in roles.items()},
        }
        print(name, json.dumps(results[name]["roles"]), flush=True)
    json.dump(results, open(out, "w"), indent=1)
