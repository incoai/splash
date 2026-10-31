"""Write the video fixtures of dev/tests/server/test_videos.py (README.md).

    python dev/tests/fixtures/videos/generate.py

Needs FFmpeg with libx264 and libx265 (Homebrew's) and NumPy; the tests read
the files it writes and run neither.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 160, 96
# A frame's index is drawn as a 5x5 grid of black and white cells; three
# corners mark the grid's orientation (index_of in the tests reads it).
GRID = 5
MARKERS = {(0, 0): 255, (0, GRID - 1): 0, (GRID - 1, 0): 0}
BITS = [(r, c) for r in range(GRID) for c in range(GRID) if (r, c) not in MARKERS]


def index_frame(index, width=WIDTH, height=HEIGHT):
    frame = np.full((height, width, 3), 128, np.uint8)
    cell = min(width, height) // (GRID + 1)
    x0, y0 = (width - cell * GRID) // 2, (height - cell * GRID) // 2
    cells = dict(MARKERS)
    for bit, position in enumerate(BITS):
        cells[position] = 255 if (index >> bit) & 1 else 0
    for (row, column), value in cells.items():
        frame[
            y0 + row * cell : y0 + (row + 1) * cell,
            x0 + column * cell : x0 + (column + 1) * cell,
        ] = value
    # A band that changes colour, so that no two frames are alike.
    frame[: height // 8] = (index * 7 % 256, 255 - index * 3 % 256, index * 13 % 256)
    return frame


def pattern_frame(index, width, height):
    """Sharp-edged colour, for comparing resized frames exactly."""
    y, x = np.mgrid[0:height, 0:width]
    return np.stack(
        [
            (x * 37 + y * 11 + index * 53) % 256,
            (x * x + 3 * y + index * 17) % 256,
            ((x ^ y) * 9 + index * 29) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


def encode(name, frames, width, height, arguments, fps=30, filters=None):
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-",
    ]  # fmt: skip
    if filters:
        command += ["-vf", filters]
    process = subprocess.Popen(
        command + arguments + [str(HERE / name)], stdin=subprocess.PIPE
    )
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        sys.exit(f"ffmpeg failed on {name}")


def remux(source, name, before=(), after=()):
    """A stream copy of source, with input options before and output options
    after it."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *before, "-i", str(HERE / source)]
        + ["-c", "copy", *after, str(HERE / name)],
        check=True,
    )


def main():
    indexed = [index_frame(i) for i in range(60)]
    # 4:2:0, as cameras record: VideoToolbox decodes 4:4:4 H.264 in hardware only.
    x264 = [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryslow",
        "-crf",
        "18",
    ]
    # B-frames: a reordered stream whose edit list shifts it to start at 0.
    encode("index.mp4", indexed, WIDTH, HEIGHT, x264 + ["-g", "12", "-bf", "3"])
    encode(
        "index_hevc10.mov",
        indexed,
        WIDTH,
        HEIGHT,
        ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1"]
        + ["-x265-params", "log-level=error:keyint=24", "-crf", "20"],
    )
    encode(
        "index_prores.mov",
        indexed[:20],
        WIDTH,
        HEIGHT,
        ["-c:v", "prores_ks", "-profile:v", "0"],
    )
    # Variable frame rate: frames 30 to 44 dropped, the others keeping their
    # times, so half a second is missing.
    encode(
        "index_vfr.mp4",
        indexed,
        WIDTH,
        HEIGHT,
        x264 + ["-g", "12", "-fps_mode", "passthrough"],
        filters="select='not(between(n,30,44))'",
    )
    encode("index_single.mp4", indexed[:1], WIDTH, HEIGHT, x264)
    for degrees in (90, 180, 270):
        remux(
            "index.mp4", f"index_rot{degrees}.mp4", ["-display_rotation", str(degrees)]
        )
    # Moov first, then the file cut through its media data.
    remux("index.mp4", "faststart.mp4", after=["-movflags", "+faststart"])
    whole = (HERE / "faststart.mp4").read_bytes()
    (HERE / "faststart.mp4").unlink()
    (HERE / "index_truncated.mp4").write_bytes(whole[: len(whole) * 3 // 5])
    # Uncompressed BGRA: decoded exactly, so resizing compares byte for byte.
    width, height = 64, 48
    encode(
        "pattern_bgra.mov",
        [pattern_frame(i, width, height) for i in range(4)],
        width,
        height,
        ["-c:v", "rawvideo", "-pix_fmt", "bgra"],
        fps=10,
    )
    remux("pattern_bgra.mov", "pattern_bgra_rot90.mov", ["-display_rotation", "90"])
    # VP9, which decodes in hardware only, and AV1, on Macs with its decoder.
    encode(
        "index_vp9.mp4",
        indexed,
        WIDTH,
        HEIGHT,
        ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-b:v", "200k", "-g", "12"],
    )
    encode(
        "index_av1.mp4",
        indexed[:20],
        WIDTH,
        HEIGHT,
        ["-c:v", "libsvtav1", "-pix_fmt", "yuv420p", "-g", "12", "-svtav1-params"]
        + ["log-level=0"],
    )
    # HLG, as phones record HDR: tone mapped to SDR.
    encode(
        "index_hlg.mp4",
        indexed[:12],
        WIDTH,
        HEIGHT,
        ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", "-x265-params"]
        + [
            "log-level=error:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc"
        ],
    )
    # MPEG transport stream, which AVFoundation reads when told its type.
    encode("index.ts", indexed[:30], WIDTH, HEIGHT, x264 + ["-g", "12"])
    # Pixels twice as wide as tall: displayed at 320x96.
    encode("index_sar2.mp4", indexed[:4], WIDTH, HEIGHT, x264, filters="setsar=2")
    # A container AVFoundation does not read.
    encode("index.flv", indexed[:10], WIDTH, HEIGHT, x264)


if __name__ == "__main__":
    main()
