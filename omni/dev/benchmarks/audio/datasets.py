"""Benchmark datasets for the Splash Omni audio benchmarks.

ASR: a deterministic 200-utterance subset of LibriSpeech test-clean (mixed
lengths), converted to 16 kHz PCM16 wav with reference transcripts, plus a few
long-form samples (~60 s and ~300 s) built by concatenating consecutive
utterances of one speaker/chapter.

TTS: 60 prompts (en/zh x short/medium/long) in ``tts_prompts.json`` next to
this file.

Data lives outside the repo: ``--data-dir`` or ``$OMNI_BENCH_DATA`` (default
``~/.cache/splash-omni/bench-data``).

    python datasets.py prepare --data-dir /path/to/data
    python datasets.py info --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

LIBRISPEECH_URL = "http://www.openslr.org/resources/12/test-clean.tar.gz"
SUBSET_SIZE = 200
SEED = 20251007
SAMPLE_RATE = 16000
# Duration buckets (seconds) sampled evenly for the subset.
BUCKETS = [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, 1e9)]
# Long-form targets: (name, target seconds); each uses a distinct speaker/chapter.
LONGFORM = [
    ("long60_a", 60.0),
    ("long60_b", 60.0),
    ("long300_a", 300.0),
    ("long300_b", 300.0),
]
GAP_SECONDS = 0.3

HERE = Path(__file__).resolve().parent
PROMPTS_PATH = HERE / "tts_prompts.json"


def default_data_dir() -> Path:
    return Path(
        os.environ.get(
            "OMNI_BENCH_DATA", Path.home() / ".cache" / "splash-omni" / "bench-data"
        )
    )


def download_librispeech(data_dir: Path) -> Path:
    """Download and extract test-clean; returns the ``LibriSpeech/test-clean`` dir."""
    dl_dir = data_dir / "librispeech_dl"
    tar_path = dl_dir / "test-clean.tar.gz"
    root = data_dir / "librispeech" / "LibriSpeech" / "test-clean"
    if root.is_dir():
        return root
    dl_dir.mkdir(parents=True, exist_ok=True)
    if not tar_path.exists():
        print(f"downloading {LIBRISPEECH_URL} -> {tar_path}", file=sys.stderr)
        tmp = tar_path.with_suffix(".part")
        urllib.request.urlretrieve(LIBRISPEECH_URL, tmp)
        tmp.rename(tar_path)
    print(f"extracting {tar_path}", file=sys.stderr)
    (data_dir / "librispeech").mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(data_dir / "librispeech", filter="data")
    return root


def scan_librispeech(root: Path) -> list[dict]:
    """All utterances of a LibriSpeech split with transcript and duration."""
    utts = []
    for trans in sorted(root.glob("*/*/*.trans.txt")):
        chapter_dir = trans.parent
        for line in trans.read_text().splitlines():
            if not line.strip():
                continue
            utt_id, text = line.split(" ", 1)
            flac = chapter_dir / f"{utt_id}.flac"
            info = sf.info(str(flac))
            spk, chap, _ = utt_id.split("-")
            utts.append(
                {
                    "id": utt_id,
                    "speaker": spk,
                    "chapter": chap,
                    "flac": str(flac),
                    "text": text.strip(),
                    "duration": info.frames / info.samplerate,
                }
            )
    utts.sort(key=lambda u: u["id"])
    return utts


def select_subset(
    utts: list[dict], n: int = SUBSET_SIZE, seed: int = SEED
) -> list[dict]:
    """Deterministic mixed-length selection: equal share per duration bucket."""
    rng = random.Random(seed)
    per_bucket = [[u for u in utts if lo <= u["duration"] < hi] for lo, hi in BUCKETS]
    want = n // len(BUCKETS)
    chosen, leftovers = [], []
    for bucket in per_bucket:
        bucket = sorted(bucket, key=lambda u: u["id"])
        rng.shuffle(bucket)
        chosen.extend(bucket[:want])
        leftovers.extend(bucket[want:])
    if len(chosen) < n:
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: n - len(chosen)])
    rng.shuffle(chosen)
    return chosen[:n]


def _write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


def _read_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # LibriSpeech is 16 kHz already; fall back to linear interpolation otherwise.
        n = int(round(len(audio) * SAMPLE_RATE / sr))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
        ).astype("float32")
    return audio


def build_longform(
    utts: list[dict], used_ids: set[str], seed: int = SEED
) -> list[dict]:
    """Concatenate consecutive utterances of one chapter up to each target."""
    rng = random.Random(seed + 1)
    chapters: dict[tuple[str, str], list[dict]] = {}
    for u in utts:
        chapters.setdefault((u["speaker"], u["chapter"]), []).append(u)
    keys = sorted(chapters)
    rng.shuffle(keys)
    out = []
    for name, target in LONGFORM:
        for key in list(keys):
            seq = chapters[key]
            total = sum(u["duration"] + GAP_SECONDS for u in seq)
            if total < target:
                continue
            keys.remove(key)
            picked, acc = [], 0.0
            for u in seq:
                picked.append(u)
                acc += u["duration"] + GAP_SECONDS
                if acc >= target:
                    break
            out.append(
                {
                    "id": name,
                    "speaker": key[0],
                    "chapter": key[1],
                    "parts": picked,
                    "target": target,
                }
            )
            break
        else:
            raise RuntimeError(f"no chapter long enough for {name} ({target}s)")
    return out


def prepare_asr(data_dir: Path) -> None:
    root = download_librispeech(data_dir)
    utts = scan_librispeech(root)
    print(f"{len(utts)} utterances in test-clean", file=sys.stderr)
    subset = select_subset(utts)
    out_dir = data_dir / "asr"
    manifest = []
    for u in subset:
        wav = out_dir / "subset" / f"{u['id']}.wav"
        if not wav.exists():
            _write_wav(wav, _read_16k(u["flac"]))
        manifest.append(
            {
                "id": u["id"],
                "wav": str(wav),
                "text": u["text"],
                "duration": round(u["duration"], 3),
                "speaker": u["speaker"],
            }
        )
    (out_dir / "subset.json").write_text(json.dumps(manifest, indent=1))
    durs = [m["duration"] for m in manifest]
    print(
        f"subset: {len(manifest)} files, {sum(durs) / 60:.1f} min, {min(durs):.1f}-{max(durs):.1f} s",
        file=sys.stderr,
    )

    long_manifest = []
    gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype="float32")
    for item in build_longform(utts, {u["id"] for u in subset}):
        wav = out_dir / "longform" / f"{item['id']}.wav"
        if not wav.exists():
            pieces = []
            for u in item["parts"]:
                pieces.append(_read_16k(u["flac"]))
                pieces.append(gap)
            _write_wav(wav, np.concatenate(pieces[:-1]))
        info = sf.info(str(wav))
        long_manifest.append(
            {
                "id": item["id"],
                "wav": str(wav),
                "text": " ".join(u["text"] for u in item["parts"]),
                "duration": round(info.frames / info.samplerate, 3),
                "speaker": item["speaker"],
                "utterances": [u["id"] for u in item["parts"]],
            }
        )
        print(
            f"longform {item['id']}: {long_manifest[-1]['duration']:.1f} s from {len(item['parts'])} utterances",
            file=sys.stderr,
        )
    (out_dir / "longform.json").write_text(json.dumps(long_manifest, indent=1))


def load_asr_set(data_dir: Path | None = None, name: str = "subset") -> list[dict]:
    """``name`` is ``subset``, ``longform`` or ``all``."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    names = ["subset", "longform"] if name == "all" else [name]
    items = []
    for n in names:
        p = data_dir / "asr" / f"{n}.json"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing; run: python datasets.py prepare --data-dir {data_dir}"
            )
        for it in json.loads(p.read_text()):
            it["set"] = n
            items.append(it)
    return items


def load_tts_prompts(path: Path | None = None) -> list[dict]:
    return json.loads(Path(path or PROMPTS_PATH).read_text())["prompts"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["prepare", "info"])
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    args = ap.parse_args(argv)
    if args.command == "prepare":
        prepare_asr(args.data_dir)
    for name in ("subset", "longform"):
        try:
            items = load_asr_set(args.data_dir, name)
        except FileNotFoundError as e:
            print(e)
            continue
        total = sum(i["duration"] for i in items)
        print(f"asr/{name}: {len(items)} files, {total:.1f} s audio")
    prompts = load_tts_prompts()
    print(f"tts prompts: {len(prompts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
