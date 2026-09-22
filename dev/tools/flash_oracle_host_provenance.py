"""Record immutable host object identities for an isolated Flash oracle build."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--abi-header", type=Path, required=True)
    parser.add_argument("objects", type=Path, nargs="+")
    args = parser.parse_args()
    header = args.abi_header.resolve()
    header_stat = header.stat()
    records = []
    for original in args.objects:
        path = original.resolve()
        stat = path.stat()
        if stat.st_mtime_ns < header_stat.st_mtime_ns:
            raise SystemExit(
                f"refusing stale host object older than current ABI header: {path}"
            )
        data = path.read_bytes()
        config_path = path.with_suffix(path.suffix + ".config")
        config = config_path.read_text().strip() if config_path.is_file() else None
        records.append(
            {
                "path": str(path),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "build_config_digest": config,
            }
        )
    result = {
        "schema": "flash-private-oracle-host-object-provenance-v1",
        "abi_header": {
            "path": str(header),
            "sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
            "mtime_ns": header_stat.st_mtime_ns,
        },
        "objects": records,
    }
    content = (
        "#pragma once\ninline constexpr const char *kFlashOracleHostObjectProvenance = "
        + json.dumps(json.dumps(result, sort_keys=True))
        + ";\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)


if __name__ == "__main__":
    main()
