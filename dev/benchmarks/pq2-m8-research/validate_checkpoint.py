"""CPU-only validation of byte-exact sources, original KAT, and frozen samples."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    if sys.flags.optimize:
        raise RuntimeError("The archived analyzer/KAT require assertions enabled")
    root = Path(__file__).resolve().parent
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    for record in provenance["copies"]:
        data = (root / record["path"]).read_bytes()
        if (
            len(data) != record["bytes"]
            or hashlib.sha256(data).hexdigest() != record["sha256"]
        ):
            raise ValueError("Source identity mismatch: " + record["path"])
    source = root / "source/v025"
    with tempfile.TemporaryDirectory(prefix="pq2-checkpoint-") as scratch:
        temporary = Path(scratch)
        for name in ("forward_analysis.py", "kat1.py"):
            shutil.copyfile(source / (name + ".txt"), temporary / name)
        shutil.copyfile(
            source / "FROZEN-SOURCE-PINS.json", temporary / "FROZEN-SOURCE-PINS.json"
        )
        subprocess.run([sys.executable, "-B", str(temporary / "kat1.py")], check=True)
        kat = json.loads((temporary / "KAT-RESULT.json").read_text())
        if kat != json.loads((source / "KAT-RESULT.json").read_text()):
            raise ValueError("Known-answer results differ from accepted archive")
        spec = importlib.util.spec_from_file_location(
            "archived_forward_analysis", temporary / "forward_analysis.py"
        )
        analyzer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analyzer)
        frozen = json.loads((root / "evidence/v027/SCREEN-ANALYSIS.json").read_text())
        actual = analyzer.evaluate(frozen["all_samples"], frozen["parse_errors"])
        if actual != frozen:
            raise ValueError(
                "V027 aggregate differs from frozen complete-sample report"
            )
    print(
        json.dumps(
            {
                "status": "PASS",
                "source_files": len(provenance["copies"]),
                "known_answer_cases": kat["count"],
                "timed_samples_recomputed": len(frozen["all_samples"]),
                "decision": actual["decision"],
                "no_GPU": True,
            }
        )
    )


if __name__ == "__main__":
    main()
