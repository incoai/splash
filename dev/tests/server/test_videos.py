import base64
import json
import math
import random
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from server import videos, workers
from server.errors import APIError

ROOT = Path(__file__).resolve().parents[3]
# The engine, whose video-frames command decodes; make test-python builds it.
BINARY = str(ROOT / "build" / "splash")
FIXTURES = ROOT / "dev" / "tests" / "fixtures" / "videos"
# The fixtures' frame rate, the index grid drawn in their frames and the
# pattern of the uncompressed ones (generate.py).
FPS = 30
GRID = 5
MARKERS = {(0, 0), (0, GRID - 1), (GRID - 1, 0)}
BITS = [(r, c) for r in range(GRID) for c in range(GRID) if (r, c) not in MARKERS]


def pattern_frame(index, width, height):
    y, x = np.mgrid[0:height, 0:width]
    return np.stack(
        [
            (x * 37 + y * 11 + index * 53) % 256,
            (x * x + 3 * y + index * 17) % 256,
            ((x ^ y) * 9 + index * 29) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


# Reference implementations, copied from transformers' Qwen3VLVideoProcessor
# (sample_frames, resize with cap_pixels_per_frame, smart_resize) and
# Qwen3VLProcessor._calculate_timestamps, so the server keeps the model's own
# preprocessing.


def reference_sample_frames(
    total_num_frames, video_fps, fps, num_frames, min_frames, max_frames
):
    if num_frames is None and fps is not None:
        num_frames = int(total_num_frames / video_fps * fps)
        num_frames = min(max(num_frames, min_frames), max_frames, total_num_frames)
    if num_frames is None:
        num_frames = min(max(total_num_frames, min_frames), max_frames)
    return np.linspace(0, total_num_frames - 1, num_frames).round().astype(int).tolist()


def reference_smart_resize(
    num_frames, height, width, temporal_factor, factor, min_pixels, max_pixels
):
    if num_frames < temporal_factor:
        raise ValueError(
            f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}"
        )
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height = int(height * scale)
        width = int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def reference_resize(
    num_frames, height, width, shortest_edge, longest_edge, max_video_tokens=768
):
    factor = 32
    frame_cap = max_video_tokens * factor * factor
    pixels_per_frame = max(
        min(frame_cap, longest_edge // num_frames), int(shortest_edge * 1.05)
    )
    max_pixels = pixels_per_frame * num_frames
    return reference_smart_resize(
        num_frames, height, width, 2, factor, shortest_edge, max_pixels
    )


def reference_timestamps(indices, video_fps, merge_size=2):
    indices = list(indices)
    if len(indices) % merge_size != 0:
        indices.extend(
            indices[-1] for _ in range(merge_size - len(indices) % merge_size)
        )
    timestamps = [idx / video_fps for idx in indices]
    return [
        (timestamps[i] + timestamps[i + merge_size - 1]) / 2
        for i in range(0, len(timestamps), merge_size)
    ]


def index_of(frame):
    """The index generate.py drew in an RGB frame, read in whichever quarter
    turn puts its corner markers right; -1 if none does."""
    gray = frame.astype(float).mean(axis=2)
    for turns in range(4):
        image = np.rot90(gray, turns)
        height, width = image.shape
        cell = min(width, height) / (GRID + 1)
        x0, y0 = (width - cell * GRID) / 2, (height - cell * GRID) / 2

        def value(row, column, image=image, cell=cell, x0=x0, y0=y0):
            top, left = int(y0 + (row + 0.25) * cell), int(x0 + (column + 0.25) * cell)
            side = max(1, int(cell / 2))
            return image[top : top + side, left : left + side].mean()

        if value(0, 0) > 170 and value(0, GRID - 1) < 85 and value(GRID - 1, 0) < 85:
            return sum(
                1 << bit for bit, cell_ in enumerate(BITS) if value(*cell_) > 128
            )
    return -1


def frames_of(output, width, height):
    return np.frombuffer(output, np.uint8).reshape(-1, height, width, 3)


def run_frames(*arguments, stdin=b""):
    return subprocess.run(
        [BINARY, "video-frames", *map(str, arguments)],
        input=stdin,
        capture_output=True,
        timeout=60,
        check=False,
    )


def probe(name):
    result = run_frames("probe", FIXTURES / name)
    if result.returncode:
        raise AssertionError(result.stderr.decode())
    return json.loads(result.stdout)


def decode(name, width, height, times=None, software=False):
    """Every displayed frame of a fixture, or those at times, as RGB, from the
    hardware decoding path or the software one."""
    probed = probe(name)
    times = probed["frames"] if times is None else times
    request = {
        "timescale": probed["timescale"],
        "frames": times,
        "ranges": videos.read_ranges(times, probed["keyframes"]),
        "keyframes": probed["keyframes"],
    }
    result = run_frames(
        "decode",
        FIXTURES / name,
        width,
        height,
        *(["--software"] if software else []),
        stdin=json.dumps(request).encode(),
    )
    if result.returncode:
        raise AssertionError(result.stderr.decode())
    return frames_of(result.stdout, width, height)


class SamplingTest(unittest.TestCase):
    def test_indices_match_the_reference_processor(self):
        rng = random.Random(7)
        cases = [
            (1, 30.0),
            (3, 30.0),
            (4, 24.0),
            (50, 25.0),
            (1800, 29.97),
            (240, 60.0),
        ]
        cases += [
            (rng.randrange(1, 5000), rng.choice([23.976, 25, 30, 50, 60]))
            for _ in range(300)
        ]
        for total, video_fps in cases:
            for fps in (0.5, 1.0, 2.0, 4.0):
                options = videos.VideoOptions(fps=fps)
                with self.subTest(total=total, video_fps=video_fps, fps=fps):
                    self.assertEqual(
                        videos.sample_indices(total, video_fps, options),
                        reference_sample_frames(total, video_fps, fps, None, 4, 768),
                    )
            options = videos.VideoOptions(num_frames=17, max_frames=768)
            self.assertEqual(
                videos.sample_indices(total, video_fps, options),
                reference_sample_frames(total, video_fps, None, 17, 4, 768),
            )

    def test_frame_size_matches_the_reference_processor(self):
        rng = random.Random(11)
        sizes = [
            (240, 320),
            (720, 1280),
            (1080, 1920),
            (2160, 3840),
            (1920, 1080),
            (17, 600),
            (600, 17),
        ]
        sizes += [
            (rng.randrange(20, 4000), rng.randrange(20, 4000)) for _ in range(200)
        ]
        for height, width in sizes:
            if max(height, width) / min(height, width) > 200:
                continue
            for frames in (2, 4, 7, 60, 120, 768):
                for longest in (
                    videos.PUBLISHED_CONFIG.max_pixels,
                    4 * 1024 * 1024,
                    videos.MAX_PIXELS,
                ):
                    options = videos.VideoOptions(max_pixels=longest)
                    with self.subTest(
                        height=height, width=width, frames=frames, longest=longest
                    ):
                        self.assertEqual(
                            videos.smart_resize(frames, height, width, options),
                            reference_resize(
                                frames,
                                height,
                                width,
                                videos.PUBLISHED_CONFIG.min_pixels,
                                longest,
                            ),
                        )

    def test_frames_stay_within_the_token_budgets(self):
        for frames in (2, 30, 120, 768):
            height, width = videos.smart_resize(
                frames, 2160, 3840, videos.VideoOptions()
            )
            per_frame_tokens = height * width // (32 * 32)
            self.assertLessEqual(per_frame_tokens, videos.MAX_TOKENS_PER_FRAME)
            self.assertLessEqual(
                frames * height * width, videos.PUBLISHED_CONFIG.max_pixels
            )

    def test_frames_stay_within_the_engine_patch_limit(self):
        # --max-image-pixels below the 768-token frame cap bounds every frame,
        # a request's shortest_edge included.
        for frame_pixels in (65_536, 300_000):
            for min_pixels in (4096, 2_000_000):
                options = videos.VideoOptions(
                    frame_pixels=frame_pixels, min_pixels=min_pixels, num_frames=4
                )
                for height, width in ((2160, 3840), (3840, 2160), (100, 4000)):
                    with self.subTest(frame_pixels=frame_pixels, min=min_pixels):
                        h, w = videos.smart_resize(4, height, width, options)
                        self.assertLessEqual(h * w, frame_pixels)
                        self.assertEqual((h % 32, w % 32), (0, 0))

    def test_timestamps_match_the_reference_processor(self):
        rng = random.Random(3)
        for count in range(1, 40):
            fps = rng.choice([24.0, 25.0, 29.97, 30.0])
            indices = sorted(rng.sample(range(5000), count))
            seconds = [i / fps for i in indices]
            expected = reference_timestamps(indices, fps)
            for got, want in zip(
                videos.patch_timestamps(seconds), expected, strict=True
            ):
                self.assertAlmostEqual(got, want, places=9)

    def test_placeholder_matches_the_processor_text(self):
        self.assertEqual(
            videos.placeholder([0.25, 1.75]),
            "<0.2 seconds><|vision_start|><|video_pad|><|vision_end|>"
            "<1.8 seconds><|vision_start|><|video_pad|><|vision_end|>",
        )

    def test_a_range_starts_where_a_keyframe_ends_the_previous_one(self):
        self.assertEqual(videos.read_ranges([5], [0]), [[0, 0]])
        # Keyframes at 0, 10 and 20: decoding on from 3 to 8 reads no
        # keyframe; 12 comes after the one at 10, 25 after the one at 20.
        self.assertEqual(
            videos.read_ranges([3, 8, 12, 15, 25], [0, 10, 20]),
            [[0, 1], [2, 3], [4, 4]],
        )
        # A frame that is itself a keyframe starts its own range.
        self.assertEqual(videos.read_ranges([5, 10], [0, 10]), [[0, 0], [1, 1]])


class ConfigTest(unittest.TestCase):
    PUBLISHED = {
        "size": {"longest_edge": 25165824, "shortest_edge": 4096},
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "processor_class": "Qwen3VLProcessor",
        "video_processor_type": "Qwen3VLVideoProcessor",
    }
    # The video_processor of processor_config.json, as newer transformers
    # releases save it (mlx-community/Qwen3.8-27B-4bit).
    LEGACY = {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "fps": 2.0,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "max_frames": 768,
        "max_pixels": 25165824,
        "merge_size": 2,
        "min_frames": 4,
        "min_pixels": 4096,
        "patch_size": 16,
        "rescale_factor": 0.00392156862745098,
        "temporal_patch_size": 2,
        "video_processor_type": "Qwen3VLVideoProcessor",
    }

    def test_published_configurations_give_the_published_sampling(self):
        self.assertEqual(videos.video_config(self.PUBLISHED), videos.PUBLISHED_CONFIG)
        self.assertEqual(videos.video_config(self.LEGACY), videos.PUBLISHED_CONFIG)

    def test_keys_left_out_take_the_processor_class_defaults(self):
        config = videos.video_config({"video_processor_type": "Qwen3VLVideoProcessor"})
        self.assertEqual(
            config,
            videos.VideoConfig(2.0, 4, 768, 128 * 32 * 32, 768 * 32 * 32),
        )
        config = videos.video_config(
            {**self.PUBLISHED, "fps": 1, "min_frames": 8, "max_frames": 256}
        )
        self.assertEqual(
            (config.fps, config.min_frames, config.max_frames), (1.0, 8, 256)
        )

    def test_unsupported_configurations_are_refused(self):
        for changes in (
            {"video_processor_type": "Qwen2VLVideoProcessor"},
            {"patch_size": 14},
            {"temporal_patch_size": 1},
            {"merge_size": 4},
            {"image_mean": [0.48, 0.46, 0.41]},
            {"do_sample_frames": False},
            {"fps": 0},
            {"fps": "2"},
            {"max_frames": 1024},
            {"min_frames": 10, "max_frames": 8},
            {"size": {"shortest_edge": 4096, "longest_edge": videos.MAX_PIXELS + 1}},
            {"size": [4096]},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaises(videos.VideoError),
            ):
                videos.video_config({**self.PUBLISHED, **changes})

    def test_an_installation_reads_the_configuration_it_links(self):
        with tempfile.TemporaryDirectory() as directory:
            vision = Path(directory)
            # No configuration: a GGUF installation's.
            self.assertEqual(videos.installed_config(vision), videos.PUBLISHED_CONFIG)
            (vision / "processor_config.json").write_text(
                json.dumps({"video_processor": {**self.LEGACY, "fps": 1}})
            )
            self.assertEqual(videos.installed_config(vision).fps, 1.0)
            (vision / "video_preprocessor_config.json").write_text(
                json.dumps({**self.PUBLISHED, "fps": 4})
            )
            self.assertEqual(videos.installed_config(vision).fps, 4.0)
            (vision / "video_preprocessor_config.json").write_text("{")
            with self.assertRaises(videos.VideoError):
                videos.installed_config(vision)


class OptionsTest(unittest.TestCase):
    def test_defaults_follow_the_installed_video_config(self):
        options = videos.video_options(None)
        self.assertEqual(
            (
                options.fps,
                options.min_frames,
                options.max_frames,
                options.min_pixels,
                options.max_pixels,
            ),
            (2.0, 4, 768, 4096, 25_165_824),
        )
        config = videos.VideoConfig(1.0, 8, 64, 8192, 4_000_000)
        defaults = videos.VideoOptions.from_config(config)
        self.assertEqual(videos.video_options(None, defaults), defaults)
        self.assertEqual(
            (defaults.fps, defaults.min_frames, defaults.max_frames), (1.0, 8, 64)
        )
        replaced = videos.VideoOptions.from_config(config, 1_000_000, 300_000)
        self.assertEqual(
            (replaced.max_pixels, replaced.frame_pixels), (1_000_000, 300_000)
        )

    def test_mm_processor_kwargs_override_sampling_and_budget(self):
        options = videos.video_options(
            {
                "fps": 1,
                "max_frames": 64,
                "size": {"longest_edge": 8_000_000},
                "do_sample_frames": True,
            }
        )
        self.assertEqual(
            (options.fps, options.max_frames, options.max_pixels), (1.0, 64, 8_000_000)
        )
        self.assertEqual(videos.video_options({"num_frames": 8}).num_frames, 8)

    def test_invalid_kwargs_are_rejected(self):
        for kwargs in (
            [],
            {"fps": 2, "num_frames": 8},
            {"fps": 0},
            {"fps": True},
            {"num_frames": 0},
            {"max_frames": 1000},
            {"min_frames": 10, "max_frames": 8},
            {"num_frames": 900},
            {"size": {"longest_edge": videos.MAX_PIXELS + 1}},
            {"size": {"shortest_edge": 10, "longest_edge": 5}},
            {"size": {"height": 10}},
            {"do_sample_frames": False},
            {"temporal_patch_size": 4},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(videos.VideoError):
                videos.video_options(kwargs)


class DataUrlTest(unittest.TestCase):
    def test_only_base64_video_data_urls_decode(self):
        self.assertEqual(
            videos.decode_data_url(
                "data:video/mp4;base64," + base64.b64encode(b"abc").decode()
            ),
            b"abc",
        )
        for url in (
            "data:image/png;base64,AAAA",
            "https://example.com/a.mp4",
            "data:video/mp4,abc",
            "data:video/mp4;base64,@@",
            5,
        ):
            with self.subTest(url=url), self.assertRaises(videos.VideoError):
                videos.decode_data_url(url)


class DecoderTest(unittest.TestCase):
    """The engine's video-frames command on the fixtures of generate.py, on
    the hardware decoding path and the software one it falls back to."""

    def assert_indices(self, frames, expected):
        self.assertEqual([index_of(frame) for frame in frames], list(expected))

    def test_every_frame_is_the_one_its_time_names(self):
        # B-frames reordered behind an edit list, 10-bit HEVC and ProRes.
        for name, count in (
            ("index.mp4", 60),
            ("index_hevc10.mov", 60),
            ("index_prores.mov", 20),
        ):
            probed = probe(name)
            self.assertEqual((probed["width"], probed["height"]), (160, 96))
            times = [t / probed["timescale"] for t in probed["frames"]]
            np.testing.assert_allclose(times, np.arange(count) / FPS, atol=1e-3)
            self.assertEqual(probed["keyframes"][0], probed["frames"][0])
            for software in (False, True):
                with self.subTest(name=name, software=software):
                    self.assert_indices(
                        decode(name, 160, 96, software=software), range(count)
                    )

    def test_both_paths_decode_the_same_pixels(self):
        for name in ("index.mp4", "index_hevc10.mov"):
            with self.subTest(name=name):
                np.testing.assert_array_equal(
                    decode(name, 160, 96), decode(name, 160, 96, software=True)
                )

    def test_sampled_frames_are_decoded_across_keyframes(self):
        probed = probe("index.mp4")
        picked = [0, 1, 13, 14, 30, 47, 59]
        times = [probed["frames"][i] for i in picked]
        for software in (False, True):
            with self.subTest(software=software):
                self.assert_indices(
                    decode("index.mp4", 128, 64, times, software=software), picked
                )

    def test_rotated_tracks_are_turned_upright(self):
        upright = decode("index.mp4", 160, 96)
        for degrees in (90, 180, 270):
            probed = probe(f"index_rot{degrees}.mp4")
            width, height = (96, 160) if degrees % 180 else (160, 96)
            self.assertEqual((probed["width"], probed["height"]), (width, height))
            for software in (False, True):
                with self.subTest(degrees=degrees, software=software):
                    turned = decode(
                        f"index_rot{degrees}.mp4", width, height, software=software
                    )
                    # The display matrix turns counterclockwise, as np.rot90.
                    np.testing.assert_array_equal(
                        turned, np.rot90(upright, degrees // 90, axes=(1, 2))
                    )

    def test_frame_times_follow_a_variable_rate(self):
        probed = probe("index_vfr.mp4")
        for software in (False, True):
            with self.subTest(software=software):
                frames = decode("index_vfr.mp4", 160, 96, software=software)
                indices = [index_of(frame) for frame in frames]
                self.assertEqual(indices, [*range(30), *range(45, 60)])
                expected = [i / FPS for i in indices]
                times = [t / probed["timescale"] for t in probed["frames"]]
                np.testing.assert_allclose(times, expected, atol=1e-3)

    def test_a_truncated_file_yields_the_frames_before_the_cut(self):
        probed = probe("index_truncated.mp4")
        self.assertLess(len(probed["frames"]), 60)
        # The decoder stops where the data it needs ends, at or before the
        # last frame the probe lists; the software one sooner.
        for software in (False, True):
            with self.subTest(software=software):
                decoded = decode("index_truncated.mp4", 160, 96, software=software)
                self.assertLess(0, len(decoded))
                self.assertLessEqual(len(decoded), len(probed["frames"]))
                self.assert_indices(decoded, range(len(decoded)))

    def test_frames_are_resized_as_pillow_resizes_them(self):
        # Uncompressed frames decode exactly, so the resize alone differs.
        for name, turns in (("pattern_bgra.mov", 0), ("pattern_bgra_rot90.mov", 1)):
            for width, height in ((64, 48), (40, 30), (96, 64), (32, 32)):
                if turns:
                    width, height = height, width
                with self.subTest(name=name, size=(width, height)):
                    decoded = decode(name, width, height)
                    self.assertEqual(len(decoded), 4)
                    for index, frame in enumerate(decoded):
                        source = Image.fromarray(
                            np.rot90(pattern_frame(index, 64, 48), turns).copy()
                        )
                        expected = source.resize(
                            (width, height), Image.Resampling.BICUBIC
                        )
                        np.testing.assert_array_equal(frame, np.asarray(expected))

    def test_a_transport_stream_is_read_by_its_bytes(self):
        # The file's name says nothing (the server's has no extension of its
        # own): its first bytes tell AVFoundation's reader.
        probed = probe("index.ts")
        self.assertEqual((probed["width"], probed["height"]), (160, 96))
        self.assert_indices(decode("index.ts", 160, 96), range(30))

    def test_pixels_keep_their_aspect_ratio(self):
        probed = probe("index_sar2.mp4")
        self.assertEqual((probed["width"], probed["height"]), (320, 96))
        # The stored frames, stretched to their displayed width.
        stored = decode("index.mp4", 160, 96)[:4]
        stretched = np.stack(
            [
                np.asarray(
                    Image.fromarray(frame).resize((320, 96), Image.Resampling.BICUBIC)
                )
                for frame in stored
            ]
        )
        frames = decode("index_sar2.mp4", 320, 96)
        self.assertLess(np.abs(frames.astype(int) - stretched).mean(), 2)

    def test_vp9_decodes_in_hardware_only(self):
        self.assert_indices(decode("index_vp9.mp4", 160, 96), range(60))

    def test_a_codec_without_a_decoder_is_named(self):
        request = json.dumps(
            {"timescale": 1000, "frames": [0], "ranges": [[0, 0]], "keyframes": [0]}
        ).encode()
        result = run_frames(
            "decode", FIXTURES / "index_vp9.mp4", 32, 32, "--software", stdin=request
        )
        self.assertEqual(result.returncode, videos.NO_SOFTWARE_DECODER)
        self.assertIn("no software decoder", result.stderr.decode())
        # AV1 decodes on the Macs with its hardware decoder (M3 and later).
        try:
            self.assert_indices(decode("index_av1.mp4", 160, 96), range(20))
        except AssertionError as error:
            self.assertIn("has no decoder on this Mac", str(error))

    def test_unreadable_files_name_the_reason(self):
        for name, message in (
            ("index.flv", "no video track"),
            ("generate.py", "no video track"),
        ):
            with self.subTest(name=name):
                result = run_frames("probe", FIXTURES / name)
                self.assertEqual(result.returncode, videos.UNREADABLE)
                self.assertIn(message, result.stderr.decode())

    def test_invalid_decode_requests_are_refused(self):
        probed = probe("index.mp4")
        frames = probed["frames"]
        for request in (
            {},
            {"timescale": 0, "frames": frames[:1], "ranges": [[0, 0]]},
            {"timescale": 15360, "frames": [frames[2], frames[1]], "ranges": [[0, 1]]},
            {"timescale": 15360, "frames": frames[:2], "ranges": [[0, 0]]},
            {"timescale": 15360, "frames": frames[:2], "ranges": [[1, 1], [0, 0]]},
            {"timescale": 15360, "frames": [1.5], "ranges": [[0, 0]]},
        ):
            with self.subTest(request=request):
                result = run_frames(
                    "decode",
                    FIXTURES / "index.mp4",
                    32,
                    32,
                    stdin=json.dumps(request).encode(),
                )
                self.assertEqual(result.returncode, videos.UNREADABLE)
                self.assertEqual(result.stdout, b"")
        self.assertEqual(
            run_frames("decode", FIXTURES / "index.mp4", 0, 32).returncode,
            videos.UNREADABLE,
        )


def no_deadline():
    return math.inf


class PrepareTest(unittest.TestCase):
    def prepare(self, name, options=videos.VideoOptions()):
        payload = (FIXTURES / name).read_bytes()
        return videos.prepare(payload, options, BINARY, no_deadline)[0]

    def frames(self, patch):
        frame = len(patch.pixels) // 2
        width, height = patch.grid_width * 16, patch.grid_height * 16
        return [
            frames_of(patch.pixels[half * frame : (half + 1) * frame], width, height)[0]
            for half in (0, 1)
        ]

    def test_sampled_frames_pair_into_temporal_patches_with_their_times(self):
        prepared = self.prepare("index.mp4")
        # 2 s at 30 fps: 2 fps gives 4 frames (the minimum), indices 0, 20,
        # 39 and 59.
        self.assertEqual(len(prepared.patches), 2)
        np.testing.assert_allclose(
            prepared.timestamps, reference_timestamps([0, 20, 39, 59], FPS)
        )
        height, width = videos.smart_resize(4, 96, 160, videos.VideoOptions())
        for patch, indices in zip(prepared.patches, ((0, 20), (39, 59)), strict=True):
            self.assertEqual(
                (patch.grid_height, patch.grid_width, patch.frames),
                (height // 16, width // 16, 2),
            )
            self.assertEqual(patch.tokens, (height // 32) * (width // 32))
            self.assertEqual([index_of(f) for f in self.frames(patch)], list(indices))
        self.assertNotEqual(
            (prepared.patches[0].digest_lo, prepared.patches[0].digest_hi),
            (prepared.patches[1].digest_lo, prepared.patches[1].digest_hi),
        )

    def test_an_odd_frame_count_repeats_the_last_frame(self):
        prepared = self.prepare("index.mp4", videos.VideoOptions(num_frames=3))
        self.assertEqual(len(prepared.patches), 2)
        last = prepared.patches[1].pixels
        self.assertEqual(last[: len(last) // 2], last[len(last) // 2 :])
        self.assertAlmostEqual(prepared.timestamps[1], 59 / FPS)

    def test_more_frames_than_the_video_holds_repeat_them(self):
        prepared = self.prepare("index_prores.mov", videos.VideoOptions(num_frames=30))
        indices = [index_of(f) for p in prepared.patches for f in self.frames(p)]
        self.assertEqual(indices, reference_sample_frames(20, FPS, None, 30, 4, 768))

    def test_a_single_frame_still_fills_one_temporal_patch(self):
        prepared = self.prepare("index_single.mp4")
        self.assertEqual(len(prepared.patches), 1)
        self.assertEqual(prepared.timestamps, (0.0,))

    def test_a_truncated_file_keeps_the_sampled_frames_before_the_cut(self):
        prepared = self.prepare("index_truncated.mp4")
        indices = [index_of(f) for p in prepared.patches for f in self.frames(p)]
        self.assertTrue(indices)
        self.assertLess(max(indices), 60)

    def test_unreadable_payloads_are_video_errors(self):
        for payload in (b"not a video", (FIXTURES / "index.flv").read_bytes()):
            with self.subTest(size=len(payload)), self.assertRaises(videos.VideoError):
                videos.prepare(payload, videos.VideoOptions(), BINARY, no_deadline)

    def test_a_stalled_hardware_decoder_is_replaced_by_software(self):
        run = workers.run
        commands = []

        def stall(command, *args, **kwargs):
            commands.append(command)
            if "decode" in command and "--software" not in command:
                return workers.Finished(videos.STALLED, b"", "stalled")
            return run(command, *args, **kwargs)

        with mock.patch.object(workers, "run", stall):
            prepared = self.prepare("index.mp4")
        self.assertEqual(
            [index_of(f) for p in prepared.patches for f in self.frames(p)],
            [0, 20, 39, 59],
        )
        self.assertEqual([c[-1] for c in commands[1:]], ["96", "--software"])

    def test_after_a_stall_videos_decode_in_software_first(self):
        run = workers.run
        commands = []

        def stall(command, *args, **kwargs):
            commands.append(command)
            if "decode" in command and "--software" not in command:
                return workers.Finished(videos.STALLED, b"", "stalled")
            return run(command, *args, **kwargs)

        cache = videos.VideoCache(BINARY)
        with mock.patch.object(workers, "run", stall):
            cache.prepare(
                (FIXTURES / "index.mp4").read_bytes(),
                videos.VideoOptions(),
                no_deadline,
            )
            commands.clear()
            cache.prepare(
                (FIXTURES / "index_hevc10.mov").read_bytes(),
                videos.VideoOptions(),
                no_deadline,
            )
        decodes = [c for c in commands if "decode" in c]
        self.assertEqual([c[-1] for c in decodes], ["--software"])

    def test_software_first_still_reads_a_codec_only_hardware_decodes(self):
        run = workers.run
        commands = []

        def record(command, *args, **kwargs):
            commands.append(command)
            return run(command, *args, **kwargs)

        cache = videos.VideoCache(BINARY)
        cache._software_until = time.monotonic() + 60
        with mock.patch.object(workers, "run", record):
            prepared = cache.prepare(
                (FIXTURES / "index_vp9.mp4").read_bytes(),
                videos.VideoOptions(),
                no_deadline,
            )
        self.assertEqual(
            [c[-1] for c in commands if "decode" in c], ["--software", "96"]
        )
        self.assertTrue(prepared.patches)

    def test_a_stall_on_a_codec_only_hardware_decodes_says_both(self):
        run = workers.run

        def stall(command, *args, **kwargs):
            if "decode" in command and "--software" not in command:
                return workers.Finished(videos.STALLED, b"", "stalled")
            return run(command, *args, **kwargs)

        with mock.patch.object(workers, "run", stall):
            with self.assertRaisesRegex(
                videos.VideoError, "no software decoder.*hardware decoder has stalled"
            ):
                self.prepare("index_vp9.mp4")

    def test_a_failing_decoder_is_a_request_error(self):
        start = subprocess.Popen

        def spawn(command, **kwargs):
            return start(["/usr/bin/false"], **kwargs)

        with mock.patch.object(workers.subprocess, "Popen", spawn):
            with self.assertRaises(APIError) as caught:
                self.prepare("index.mp4")
        self.assertEqual(caught.exception.status, 400)
        self.assertFalse(workers._workers)

    def test_cache_reuses_prepared_videos_per_options(self):
        payload = (FIXTURES / "index.mp4").read_bytes()
        cache = videos.VideoCache(BINARY)
        first = cache.prepare(payload, videos.VideoOptions(), no_deadline)
        self.assertIs(cache.prepare(payload, videos.VideoOptions(), no_deadline), first)
        self.assertIsNot(
            cache.prepare(payload, videos.VideoOptions(num_frames=6), no_deadline),
            first,
        )
        self.assertEqual(cache.stats()["entries"], 2)

    def test_a_cache_without_a_decoder_refuses_videos(self):
        with self.assertRaises(videos.VideoError):
            videos.VideoCache().prepare(b"video", videos.VideoOptions(), no_deadline)


if __name__ == "__main__":
    unittest.main()
