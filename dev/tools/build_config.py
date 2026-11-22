"""Track the configuration of each successful compiler/linker output."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def record_path(output):
    return Path(str(output) + ".config")


def matches(output, config):
    try:
        return record_path(output).read_text(encoding="ascii") == config + "\n"
    except (OSError, UnicodeError):
        return False


def run_configured(output, config, command):
    record = record_path(output)
    # An interrupted or failed compiler must not leave a partial output with
    # an old configuration record that would make a retry look unnecessary.
    record.unlink(missing_ok=True)
    result = subprocess.run(command)
    if result.returncode:
        return result.returncode if result.returncode > 0 else 128 - result.returncode
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", dir=record.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(config + "\n")
        os.replace(temporary, record)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    # Every output with the configuration it must have been built with, one
    # Python start for all of them: every make invocation runs this as it
    # parses the Makefile, and a start for each of its eight groups made a
    # parse take 0.38-0.45 s against 0.19-0.26 s (M5 Max).
    stale = modes.add_parser("stale")
    stale.add_argument("pairs", nargs="*", metavar="CONFIG=OUTPUT")
    record = modes.add_parser("record")
    record.add_argument("--config", required=True)
    record.add_argument("--output", required=True)
    record.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "stale":
        pairs = [pair.partition("=") for pair in args.pairs]
        print(
            " ".join(
                output for config, _, output in pairs if not matches(output, config)
            )
        )
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("record requires a compiler or linker command after --")
    return run_configured(args.output, args.config, command)


if __name__ == "__main__":
    raise SystemExit(main())
