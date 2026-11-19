"""Uploads decoded, resampled and split; WAV files and streams written."""

import io
import struct
import unittest
from unittest import mock

import av
import numpy as np

from dev.tests.audio_fixtures import sine, wav_upload
from server import audio_io
from server.errors import APIError


def encoded(samples, rate, container, codec, *, channels=1, video=False):
    """samples (float, -1..1) encoded by PyAV into an in-memory file, with a
    short black video track ahead of the audio when video is set."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=container) as output:
        if video:
            picture = output.add_stream("mpeg4", rate=10)
            picture.width, picture.height, picture.pix_fmt = 64, 48, "yuv420p"
        stream = output.add_stream(codec, rate=rate)
        stream.layout = "mono" if channels == 1 else "stereo"
        if video:
            for index in range(5):
                frame = av.VideoFrame.from_ndarray(
                    np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24"
                )
                frame.pts = index
                output.mux(picture.encode(frame))
            output.mux(picture.encode(None))
        planar = np.tile(np.asarray(samples, dtype=np.float32), (channels, 1))
        frame = av.AudioFrame.from_ndarray(
            planar, format="fltp", layout=stream.layout.name
        )
        frame.rate = rate
        output.mux(stream.encode(frame))
        output.mux(stream.encode(None))
    return buffer.getvalue()


def tone_frequency(samples, rate):
    """The strongest frequency of samples, by FFT."""
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return np.fft.rfftfreq(len(samples), 1 / rate)[spectrum.argmax()]


class DecodeTests(unittest.TestCase):
    def test_pcm_wav_widths_and_channels_decode_to_16k_mono_float(self):
        source = sine(0.25, 16_000)
        for width in (1, 2, 3, 4):
            for channels in (1, 2):
                with self.subTest(width=width, channels=channels):
                    decoded = audio_io.decode_upload(
                        wav_upload(source, 16_000, channels, width)
                    )
                    self.assertEqual(decoded.dtype, np.float32)
                    self.assertEqual(len(decoded), len(source))
                    tolerance = 0.02 if width == 1 else 1e-3
                    self.assertLess(np.abs(decoded - source).max(), tolerance)

    def test_higher_rates_are_resampled_and_keep_the_tone(self):
        for rate in (48_000, 44_100, 32_000, 22_050):
            with self.subTest(rate=rate):
                source = sine(0.5, rate, frequency=1000.0)
                decoded = audio_io.decode_upload(wav_upload(source, rate))
                self.assertEqual(len(decoded), 8000)
                self.assertAlmostEqual(
                    tone_frequency(decoded, 16_000), 1000.0, delta=20
                )
                self.assertAlmostEqual(np.abs(decoded).max(), 0.5, delta=0.03)

    def test_a_tone_above_8_khz_is_filtered_out_before_decimation(self):
        source = sine(0.5, 48_000, frequency=11_000.0)
        decoded = audio_io.resample(source.astype(np.float32), 48_000)
        # Away from the edges, where the abrupt start of the tone rings.
        self.assertLess(np.abs(decoded[200:-200]).max(), 0.01)

    def test_lower_rates_are_interpolated_up(self):
        source = sine(0.5, 8_000, frequency=500.0)
        decoded = audio_io.decode_upload(wav_upload(source, 8_000))
        self.assertEqual(len(decoded), 8000)
        self.assertAlmostEqual(tone_frequency(decoded, 16_000), 500.0, delta=20)

    def test_empty_and_malformed_uploads_are_refused(self):
        with self.assertRaises(APIError) as refused:
            audio_io.decode_upload(b"")
        self.assertEqual(refused.exception.status, 400)
        for data in (b"RIFF\x00\x00\x00\x00WAVEjunk", b"\x00" * 64):
            with self.subTest(data=data[:4]), self.assertRaises(APIError) as refused:
                audio_io.decode_upload(data, "clip.mp3")
            self.assertIn(refused.exception.status, (400, 415))
        with self.assertRaises(APIError) as refused:
            audio_io.decode_upload(wav_upload([], 16_000))
        self.assertEqual(refused.exception.status, 400)

    def test_an_overlong_upload_is_refused(self):
        with mock.patch.object(audio_io, "MAX_UPLOAD_SECONDS", 1):
            with self.assertRaises(APIError) as refused:
                audio_io.decode_upload(wav_upload(sine(1.5), 16_000))
        self.assertEqual(refused.exception.status, 400)
        self.assertIn("longer than", refused.exception.message)

    def test_a_float_wav_decodes_through_pyav(self):
        # A float WAV, which the standard library refuses.
        source = sine(0.25, 16_000, frequency=440.0).astype(np.float32)
        data = (
            struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF",
                36 + 4 * len(source),
                b"WAVE",
                b"fmt ",
                16,
                3,
                1,
                16_000,
                64_000,
                4,
                32,
                b"data",
                4 * len(source),
            )
            + source.tobytes()
        )
        decoded = audio_io.decode_upload(data, "tone.wav")
        self.assertEqual(len(decoded), len(source))
        self.assertLess(np.abs(decoded - source).max(), 1e-3)

    def test_compressed_audio_and_video_files_keep_the_tone(self):
        cases = (
            ("mp4", "aac", 2, False),
            ("ipod", "aac", 1, False),
            ("mp4", "aac", 2, True),
            ("mov", "aac", 1, True),
            ("matroska", "aac", 2, True),
            ("ogg", "libopus", 2, False),
            ("flac", "flac", 1, False),
            ("aiff", "pcm_s16be", 2, False),
        )
        for container, codec, channels, video in cases:
            with self.subTest(container=container, codec=codec, video=video):
                rate = 48_000 if codec == "libopus" else 44_100
                source = sine(1.0, rate, frequency=1000.0)
                data = encoded(
                    source, rate, container, codec, channels=channels, video=video
                )
                decoded = audio_io.decode_upload(data, f"clip.{container}")
                self.assertEqual(decoded.dtype, np.float32)
                self.assertAlmostEqual(len(decoded) / 16_000, 1.0, delta=0.1)
                middle = decoded[4000:12000]
                self.assertAlmostEqual(tone_frequency(middle, 16_000), 1000.0, delta=20)
                self.assertAlmostEqual(np.abs(middle).max(), 0.5, delta=0.05)

    def test_a_video_without_sound_and_an_unknown_format_are_refused(self):
        buffer = io.BytesIO()
        with av.open(buffer, mode="w", format="mp4") as output:
            picture = output.add_stream("mpeg4", rate=10)
            picture.width, picture.height, picture.pix_fmt = 64, 48, "yuv420p"
            frame = av.VideoFrame.from_ndarray(
                np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24"
            )
            output.mux(picture.encode(frame))
            output.mux(picture.encode(None))
        with self.assertRaises(APIError) as refused:
            audio_io.decode_upload(buffer.getvalue(), "silent.mp4")
        self.assertEqual(refused.exception.status, 400)
        self.assertIn("no audio track", refused.exception.message)
        with self.assertRaises(APIError) as refused:
            audio_io.decode_upload(b"\x00" * 64, "noise.mp3")
        self.assertIn(refused.exception.status, (400, 415))

    def test_decoding_stops_past_the_upload_limit(self):
        source = sine(3.0, 44_100, frequency=440.0)
        with mock.patch.object(audio_io, "MAX_UPLOAD_SECONDS", 1):
            with self.assertRaises(APIError) as refused:
                audio_io.decode_upload(
                    encoded(source, 44_100, "mp4", "aac"), "long.m4a"
                )
        self.assertIn("longer than", refused.exception.message)


class SplitTests(unittest.TestCase):
    def test_short_audio_is_one_segment(self):
        samples = np.zeros(16_000, dtype=np.float32)
        self.assertEqual(
            audio_io.split_segments(samples), [audio_io.Segment(0, 16_000)]
        )
        self.assertEqual(audio_io.Segment(0, 16_000).seconds(), (0.0, 1.0))

    def test_long_audio_is_cut_at_the_quietest_window_before_each_boundary(self):
        rate = 1000
        # Loud throughout, except for 200 ms of silence 3 s before each
        # 10-second boundary, which the cut finds within its 5-second search.
        samples = np.ones(25 * rate, dtype=np.float32)
        samples[7_000:7_200] = 0.0
        samples[16_900:17_100] = 0.0
        segments = audio_io.split_segments(
            samples, rate, request_seconds=10, search_seconds=5, window_seconds=0.1
        )
        self.assertEqual(len(segments), 3)
        self.assertEqual(
            [s.start for s in segments], [0, segments[0].end, segments[1].end]
        )
        self.assertTrue(7_000 <= segments[0].end <= 7_200)
        self.assertTrue(16_900 <= segments[1].end <= 17_100)
        self.assertEqual(segments[-1].end, len(samples))
        for segment in segments:
            self.assertLessEqual(segment.end - segment.start, 10 * rate)

    def test_uniform_audio_is_cut_at_the_boundary_search_window(self):
        samples = np.ones(2_500, dtype=np.float32)
        segments = audio_io.split_segments(
            samples, 100, request_seconds=10, search_seconds=2, window_seconds=0.1
        )
        self.assertEqual(len(segments), 3)
        for previous, current in zip(segments, segments[1:]):
            self.assertEqual(previous.end, current.start)
            self.assertTrue(800 <= previous.end - previous.start <= 1000)


class ParallelSegmentTests(unittest.TestCase):
    def test_long_audio_spreads_over_the_lanes(self):
        # Five minutes over eight lanes: 37.5 s each plus the split search, so
        # that cuts made early still leave at most eight segments.
        self.assertEqual(audio_io.parallel_segment_seconds(300.0, 8), 42.5)
        samples = np.ones(300 * 16_000, dtype=np.float32)
        segments = audio_io.split_segments(
            samples, request_seconds=audio_io.parallel_segment_seconds(300.0, 8)
        )
        self.assertEqual(len(segments), 8)
        self.assertEqual(segments[0].start, 0)
        self.assertEqual(segments[-1].end, len(samples))
        for left, right in zip(segments, segments[1:]):
            self.assertEqual(left.end, right.start)

    def test_short_audio_and_disabled_segmenting_stay_whole(self):
        # Never shorter than the minimum, and 0 or one lane keeps the request maximum.
        self.assertEqual(audio_io.parallel_segment_seconds(40.0, 8), 30.0)
        self.assertEqual(
            audio_io.parallel_segment_seconds(300.0, 8, 0.0),
            audio_io.MAX_REQUEST_SECONDS,
        )
        self.assertEqual(
            audio_io.parallel_segment_seconds(300.0, 1), audio_io.MAX_REQUEST_SECONDS
        )
        self.assertEqual(
            audio_io.parallel_segment_seconds(10 * 3600.0, 8),
            audio_io.MAX_REQUEST_SECONDS,
        )


class WavTests(unittest.TestCase):
    def test_wav_header_and_file(self):
        pcm = struct.pack("<4h", 1, -1, 2, -2)
        data = audio_io.wav_bytes(pcm)
        self.assertEqual(len(data), 44 + 8)
        self.assertEqual(data[:4], b"RIFF")
        self.assertEqual(struct.unpack_from("<I", data, 4)[0], 36 + 8)
        self.assertEqual(data[8:12], b"WAVE")
        fmt = struct.unpack_from("<HHIIHH", data, 20)
        self.assertEqual(fmt, (1, 1, 24_000, 48_000, 2, 16))
        self.assertEqual(data[36:40], b"data")
        self.assertEqual(struct.unpack_from("<I", data, 40)[0], 8)
        self.assertEqual(data[44:], pcm)
        decoded = audio_io.decode_upload(audio_io.wav_bytes(struct.pack("<2h", 0, 0)))
        self.assertEqual(len(decoded), 2 * 16_000 // 24_000 or 1)

    def test_stream_header_has_unknown_lengths(self):
        header = audio_io.wav_header(None)
        self.assertEqual(len(header), audio_io.WAV_HEADER_BYTES)
        self.assertEqual(struct.unpack_from("<I", header, 4)[0], 0xFFFFFFFF)
        self.assertEqual(struct.unpack_from("<I", header, 40)[0], 0xFFFFFFFF)

    def test_float32_bytes_are_little_endian(self):
        self.assertEqual(
            audio_io.float32_bytes(np.array([0.5, -0.25])),
            struct.pack("<2f", 0.5, -0.25),
        )


if __name__ == "__main__":
    unittest.main()


class SpeechEdgesTests(unittest.TestCase):
    RATE = audio_io.TTS_SAMPLE_RATE

    def tone(self, seconds, amplitude, frequency=220.0):
        times = np.arange(round(seconds * self.RATE)) / self.RATE
        return amplitude * np.sin(2 * np.pi * frequency * times)

    def pcm(self, samples):
        return np.round(np.asarray(samples) * 32767).astype("<i2").tobytes()

    def run_edges(self, data, continuation, step):
        edges = audio_io.SpeechEdges(continuation)
        out = b"".join(
            edges.push(data[i : i + step]) for i in range(0, len(data), step)
        )
        out += edges.finish()
        return np.frombuffer(out, dtype="<i2")

    def test_an_answer_keeps_its_samples_and_fades_its_end(self):
        source = self.pcm(self.tone(1.0, 0.5))
        expected = np.frombuffer(source, dtype="<i2")
        for step in (1000, 4801, 2 * len(source)):
            with self.subTest(step=step):
                out = self.run_edges(source, False, step)
                self.assertEqual(len(out), len(expected))
                tail = round(audio_io.SpeechEdges.FADE_OUT_SECONDS * self.RATE)
                np.testing.assert_array_equal(out[:-tail], expected[:-tail])
                self.assertEqual(out[-1], 0)
                self.assertLess(
                    np.abs(out[-tail // 4 :]).max(), np.abs(expected).max() * 0.3
                )

    def test_a_continuation_drops_the_tail_it_starts_with(self):
        leaked = self.tone(0.08, 0.05) * np.linspace(1.0, 0.2, round(0.08 * self.RATE))
        pause = np.zeros(round(0.1 * self.RATE))
        speech = self.tone(0.5, 0.4)
        source = self.pcm(np.concatenate([leaked, pause, speech]))
        for step in (960, 9600, len(source)):
            with self.subTest(step=step):
                out = self.run_edges(source, True, step)
                # Cut inside the pause: the speech is kept whole.
                self.assertLess(
                    len(out), len(source) // 2 - round(0.08 * self.RATE) + 1
                )
                self.assertGreaterEqual(len(out), len(speech) + 1)
                self.assertLess(np.abs(out[:240]).max(), 50)

    def test_a_continuation_that_starts_quiet_or_without_a_pause_is_kept(self):
        quiet = self.pcm(np.concatenate([np.zeros(2400), self.tone(0.5, 0.4)]))
        out = self.run_edges(quiet, True, 4800)
        self.assertEqual(len(out), len(quiet) // 2)
        np.testing.assert_array_equal(out[:2400], 0)
        loud = self.pcm(self.tone(0.5, 0.4))
        out = self.run_edges(loud, True, 4800)
        # No quiet point to cut at: only a short fade in.
        self.assertEqual(len(out), len(loud) // 2)
        self.assertEqual(out[0], 0)

    def test_a_short_continuation_is_trimmed_at_its_end(self):
        leaked = self.tone(0.05, 0.05)
        source = self.pcm(
            np.concatenate([leaked, np.zeros(2400), self.tone(0.05, 0.3)])
        )
        out = self.run_edges(source, True, 100000)
        self.assertLess(len(out), len(source) // 2)
