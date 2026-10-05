import concurrent.futures
import json
import math
import random
import unittest
from unittest import mock

from server import json_codec


class JsonCodecTests(unittest.TestCase):
    def assert_roundtrip(self, value):
        text = json_codec.dumps(value)
        data = json_codec.encode(value)
        self.assertEqual(text.encode("utf-8"), data)
        self.assertEqual(json_codec.encoded_size(value), len(data))
        self.assertEqual(json.loads(data), json.loads(json.dumps(value)))
        self.assertFalse(any(char in text for char in "\x85\u2028\u2029"))
        return data

    def test_unicode_controls_and_nested_json(self):
        value = {
            "中文🌍": [
                'café e\u0301 "quoted" C:\\Users\\file.txt\n\r\t\x00',
                "\x85\u2028\u2029\ufeff\uffff\U0010ffff",
                "literal \\ud800 and \\u2028",
                {"integer": 2**60, "float": 1.25, "bool": True, "null": None},
            ]
        }
        data = self.assert_roundtrip(value)
        self.assertIn("中文🌍".encode(), data)
        self.assertNotIn(b"\\u4e2d", data)
        arguments = json_codec.dumps(value)
        outer = self.assert_roundtrip({"function": {"arguments": arguments}})
        self.assertEqual(json.loads(outer)["function"]["arguments"], arguments)

    def test_all_isolated_surrogates_are_escaped_in_values_and_keys(self):
        value = {chr(cp): chr(cp) for cp in range(0xD800, 0xE000)}
        data = self.assert_roundtrip(value)
        self.assertIn(b"\\ud800", data)
        self.assertIn(b"\\udfff", data)

    def test_fragment_encoding_matches_complete_strings(self):
        rng = random.Random(42)
        alphabet = 'a中🌍é\x00\n\r\t"\\\x85\u2028\u2029\ud800\udfff'
        for _ in range(1000):
            value = "".join(rng.choices(alphabet, k=rng.randrange(1, 80)))
            split = rng.randrange(len(value) + 1)
            with self.subTest(value=ascii(value), split=split):
                partial = (
                    json_codec.dumps(value[:split])[1:-1]
                    + json_codec.dumps(value[split:])[1:-1]
                )
                self.assertEqual(partial, json_codec.dumps(value)[1:-1])
                self.assert_roundtrip({"text": value})

    def test_invalid_values_fail_without_affecting_later_encoding(self):
        cycle = []
        cycle.append(cycle)
        for value in (
            math.nan,
            math.inf,
            -math.inf,
            object(),
            {object(): 1},
            cycle,
        ):
            for encode in (
                json_codec.dumps,
                json_codec.encode,
                json_codec.encoded_size,
            ):
                with self.subTest(kind=type(value).__name__, encoder=encode.__name__):
                    with self.assertRaises(json_codec.JSONEncodingError) as caught:
                        encode(value)
                    self.assertNotIsInstance(caught.exception, ValueError)
                    self.assert_roundtrip({"ok": "中文"})

    def test_byte_count_does_not_encode_the_whole_document(self):
        value = {"large": "中文🌍\ud800\u2028" * 10000, "nested": [1, 2, False]}
        expected = len(json_codec.encode(value))
        with mock.patch.object(
            json_codec._ENCODER, "encode", side_effect=AssertionError
        ):
            self.assertEqual(json_codec.encoded_size(value), expected)

    def test_encoder_calls_are_independent_across_threads(self):
        values = [{"index": i, "text": "中文🌍\ud800\u2028" * i} for i in range(128)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            data = list(pool.map(self.assert_roundtrip, values))
        self.assertEqual([json.loads(item) for item in data], values)

    def test_strict_loading_preserves_finite_numbers_and_numeric_strings(self):
        data = '{"values":[1.7976931348623157e308, 5e-324, -0.0, 12345678901234567890, "1e400"]}'
        self.assertEqual(json_codec.loads(data), json.loads(data))
        for number in ("1e400", "-1e400", "NaN", "Infinity", "-Infinity"):
            with self.subTest(number=number), self.assertRaises(ValueError):
                json_codec.loads('{"value":' + number + "}")
