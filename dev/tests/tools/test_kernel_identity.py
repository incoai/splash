import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tools import kernel_identity


def listing(path: str, functions: dict[str, str], offset: int = 0x1E8B1) -> str:
    """metal-objdump's disassembly of a metallib of these functions."""
    text = f"{path}:\nDisassembly of section MODULE_LIST:\n"
    for name, body in functions.items():
        text += f"\n{offset:#016x} -- {name}:\n{body}\n"
        offset += 0x1000
    return text


MODULE = 'source_filename = "{0}"\n\ndefine void @{0}() {{\n  ret void\n}}\n'


class KernelIdentityTests(unittest.TestCase):
    def run_main(self, baseline: dict, candidate: dict, *options: str):
        """main on two metallibs of these functions: its exit status and
        output."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            listings = {}
            for name, functions in (("baseline", baseline), ("candidate", candidate)):
                (root / name / "build").mkdir(parents=True)
                metallib = root / name / "build/splash.metallib"
                metallib.touch()
                # Another layout of the library moves every module.
                offset = 0x1E8B1 if name == "baseline" else 0x2F000
                listings[str(metallib)] = listing(str(metallib), functions, offset)

            def objdump(command, **_):
                return subprocess.CompletedProcess(command, 0, listings[command[-1]])

            output = io.StringIO()
            with (
                mock.patch.object(subprocess, "run", objdump),
                contextlib.redirect_stdout(output),
            ):
                status = kernel_identity.main(
                    [str(root / "baseline"), str(root / "candidate/build"), *options]
                )
        return status, output.getvalue().splitlines()

    def test_modules_are_compared_without_their_offsets(self):
        functions = {
            name: MODULE.format(name)
            for name in ("decode_sample", "verify_gdn", "norm_rms", "moe_combine")
        }
        # The candidate lists its modules in another order.
        status, lines = self.run_main(functions, dict(reversed(functions.items())))
        self.assertEqual(status, 0)
        self.assertEqual(
            lines[:-1],
            [f"identical: {name}" for name in sorted(functions)],
        )
        self.assertTrue(lines[-1].startswith("kernel identity: 4 of 4 identical"))

    def test_by_default_kernels_named_for_other_paths_are_left_out(self):
        baseline = {
            name: MODULE.format(name)
            for name in (
                "decode_linear",
                "gguf_decode_q4k_m8_a",
                "draft_conv",
                "prefill_linear",
                "gguf_prefill_q4k",
                "vision_patchify",
                "rope_build_tables",
            )
        }
        candidate = {
            **baseline,
            "gguf_decode_q4k_m8_a": MODULE.format("gguf_decode_q4k_m8_a") + "; x\n",
            "prefill_linear": "changed",
            "ane_ffn_join": MODULE.format("ane_ffn_join"),
        }
        del candidate["draft_conv"]
        status, lines = self.run_main(baseline, candidate)
        self.assertEqual(status, 1)
        self.assertEqual(
            lines[:-1],
            [
                "identical: decode_linear",
                "only in baseline: draft_conv",
                "different: gguf_decode_q4k_m8_a",
                "identical: rope_build_tables",
            ],
        )
        self.assertTrue(lines[-1].startswith("kernel identity: 2 of 4 identical"))
        # --match takes the names to compare instead.
        status, lines = self.run_main(
            baseline, candidate, "--match", "*prefill*", "--match", "ane_ffn_*"
        )
        self.assertEqual(status, 1)
        self.assertEqual(
            lines[:-1],
            [
                "only in candidate: ane_ffn_join",
                "identical: gguf_prefill_q4k",
                "different: prefill_linear",
            ],
        )

    def test_a_pattern_that_matches_nothing_is_an_error(self):
        functions = {"decode_sample": MODULE.format("decode_sample")}
        with (
            contextlib.redirect_stderr(io.StringIO()) as errors,
            self.assertRaises(SystemExit),
        ):
            self.run_main(functions, functions, "--match", "vision_*")
        self.assertIn("no function matches", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
