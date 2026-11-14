"""Exercise the real executable's dry run; no Metal library or model exists."""

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    binary = str(Path(sys.argv[1]).resolve())
    checks = 0
    with tempfile.TemporaryDirectory(prefix="splash-profile-cli-") as tmp:
        root = Path(tmp)
        fixture = root / "private.tokens"
        fixture.write_text("0 1 2\n")
        report, trace = root / "report.json", root / "dispatch.txt"

        def call(args):
            return subprocess.run(
                [binary, str(root / "NO-METALLIB"), str(root / "NO-MODEL"), *args],
                capture_output=True,
                text=True,
                timeout=10,
            )

        result = call(["--config-only"])
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["config_only"] and data["fusion_enabled"] is False
        assert data["http_ttft_s"] is None and data["queue_wait_s"] is None
        checks += 1
        valid = [
            "--config-only",
            "--require-flash-next",
            "--activations",
            "bf16",
            "--state-format",
            "bf16",
            "--kv-format",
            "int8",
            "--max-memory",
            "96G",
            "--max-context",
            "262144",
            "--input-tokens",
            str(fixture),
        ]
        result = call(valid + ["--report", str(report), "--dispatch-list", str(trace)])
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert (
            data["input_token_ids_sha256_le"]
            == hashlib.sha256(struct.pack("<3I", 0, 1, 2)).hexdigest()
        )
        assert (
            str(fixture) not in result.stdout
            and not report.exists()
            and not trace.exists()
        )
        assert (
            data["requested_memory_bytes"] == 96 * 2**30 and data["prompt_tokens"] == 3
        )
        checks += 1
        for args, error in [
            (["--config-only", "--require-flash-next"], "matched Flash-Next"),
            (["--config-only", "--max-memory", "0"], "positive"),
            (["--config-only", "--cycles", "4294967296"], "out of range"),
            (["--config-only", "--activations", "int8"], "auto or bf16"),
            (["--config-only", "--max-cache-disk", "512G"], "unknown option"),
            (["--config-only", "--images", "0x1"], "positive"),
            (["--config-only", "--input-tokens", str(root)], "bounded regular file"),
            (["--config-only", "--report", str(fixture)], "new file"),
            (["--config-only", "--dispatch-list", str(fixture)], "new file"),
        ]:
            result = call(args)
            assert result.returncode != 0 and error in result.stderr, (
                args,
                result.stderr,
            )
            assert "Metal backend" not in result.stderr
            checks += 1
        assert fixture.read_text() == "0 1 2\n"
        assert not report.exists() and not trace.exists()
    print(
        f"{checks} real-profiler CLI preflight cases passed; no model or Metal library opened."
    )


if __name__ == "__main__":
    main()
