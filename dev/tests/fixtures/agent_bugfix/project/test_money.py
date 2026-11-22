import unittest

from shop import money


class MoneyTests(unittest.TestCase):
    def test_to_cents(self):
        self.assertEqual(money.to_cents("12.34"), 1234)
        self.assertEqual(money.to_cents(" $7 "), 700)
        self.assertEqual(money.to_cents("-0.99"), -99)

    def test_to_cents_rejects(self):
        for text in ("", "abc", "1.234", "1.x"):
            with self.assertRaises(ValueError):
                money.to_cents(text)

    def test_format_cents(self):
        self.assertEqual(money.format_cents(1234), "$12.34")
        self.assertEqual(money.format_cents(-5), "-$0.05")
        self.assertEqual(money.format_cents(123456789), "$1,234,567.89")

    def test_split_evenly(self):
        self.assertEqual(money.split_evenly(1000, 3), [334, 333, 333])
        self.assertEqual(sum(money.split_evenly(7, 4)), 7)

    def test_percent_of(self):
        self.assertEqual(money.percent_of(1999, 10), 200)
        self.assertEqual(money.percent_of(-1999, 10), -200)


if __name__ == "__main__":
    unittest.main()
