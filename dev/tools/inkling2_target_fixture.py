#!/usr/bin/env python3
"""Writes the expected values of the Inkling-2-Mini target's engine test
(dev/tests/engine/inkling2_target_test.mm) from the fp32 reference forward
(dev/tools/inkling2_reference.py) of the test's own fixture: every layer
application's output rows at the test's sampled rows of its prompt, the
logits of the prompt's last row over the tokens, and the final norm's rows
(output_norm of the final block mix, the rows the head reads and the draft
captures at the id past the applications) at the sampled rows.

    build/engine-tests/inkling2-target build/splash.metallib --write-fixture DIR
    python -m dev.tools.inkling2_target_fixture DIR dev/tests/fixtures/inkling2-target

DIR holds what the test writes: target/inkling2-fixture.gguf and prompt.u32,
its ids. OUT_DIR receives expected.f32 (little-endian fp32: [applications]
[rows][hidden] outputs, then [tokens] logits, then [rows][hidden] final norm
rows) and expected.json, which names
that layout, the rows, the fixture's hash and these commands. Needs gguf-py
with NVFP4; the expected values are committed, so the test does not.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dev.tools.inkling2_reference import InklingModel, forward, rms_norm

# The rows the test compares (inkling2_target_test.mm kSampledRows): the
# first, either side of a page and of the window's and extent's edge, the last.
ROWS = (0, 1, 2, 3, 511, 512, 1022, 1023, 1024, 1099)
COMMANDS = (
    "build/engine-tests/inkling2-target build/splash.metallib --write-fixture DIR",
    "python -m dev.tools.inkling2_target_fixture DIR dev/tests/fixtures/inkling2-target",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "fixture", type=Path, help="the directory the test's --write-fixture wrote"
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    gguf_path = args.fixture / "target" / "inkling2-fixture.gguf"
    model = InklingModel(gguf_path)
    tokens = np.fromfile(args.fixture / "prompt.u32", dtype="<u4").astype(np.int64)
    applications = len(model.layout.schedule())
    result = forward(
        model, tokens, residual_applications=range(applications), logit_rows=[ROWS[-1]]
    )
    outputs = result.residuals[:, 1, list(ROWS), :].astype("<f4")
    logits = (
        np.asarray(result.logits).reshape(-1, result.logits.shape[-1])[-1].astype("<f4")
    )
    final_norm = rms_norm(
        result.final[list(ROWS)],
        model.tensor("output_norm.weight"),
        model.layout.norm_eps,
    ).astype("<f4")
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "expected.f32", "wb") as file:
        file.write(outputs.tobytes())
        file.write(logits.tobytes())
        file.write(final_norm.tobytes())
    description = {
        "fixture_sha256": hashlib.sha256(gguf_path.read_bytes()).hexdigest(),
        "reference": "dev/tools/inkling2_reference.py, fp32",
        "applications": applications,
        "rows": list(ROWS),
        "hidden": int(outputs.shape[-1]),
        "tokens": int(logits.shape[0]),
        "layout": "fp32 [applications][rows][hidden] outputs, then fp32 [tokens] logits of the last row,"
        " then fp32 [rows][hidden] final norm rows",
        "regenerate": list(COMMANDS),
    }
    (args.output / "expected.json").write_text(json.dumps(description, indent=2) + "\n")
    print(
        f"{args.output / 'expected.f32'}: "
        f"{outputs.nbytes + logits.nbytes + final_norm.nbytes} bytes"
    )


if __name__ == "__main__":
    main()
