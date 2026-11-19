import unittest

from shop.cart import Cart
from shop.catalog import Catalog

CATALOG = """sku,name,price
MUG,Coffee mug,8.5
PEN,Gel pen,1.25
BAG,Tote bag,12
LAMP,Desk lamp,64.9
"""


class CartTests(unittest.TestCase):
    def setUp(self):
        self.cart = Cart(Catalog.from_csv(CATALOG), tax_percent=8)

    def test_subtotal(self):
        self.cart.add("PEN", 4)
        self.cart.add("BAG")
        self.assertEqual(self.cart.subtotal(), 1700)

    def test_mug_line(self):
        self.cart.add("MUG", 2)
        self.assertEqual(self.cart.subtotal(), 1700)

    def test_discount_and_tax(self):
        self.cart.add("LAMP", 2)
        self.cart.apply_code("BIG20")
        self.assertEqual(self.cart.subtotal(), 12980)
        self.assertEqual(self.cart.discount(), 2596)
        self.assertEqual(self.cart.tax(), 831)
        self.assertEqual(self.cart.total(), 11215)

    def test_receipt(self):
        self.cart.add("MUG")
        self.cart.add("PEN", 2)
        self.assertEqual(
            self.cart.receipt(),
            [
                "1 x Coffee mug: $8.50",
                "2 x Gel pen: $2.50",
                "Subtotal: $11.00",
                "Tax: $0.88",
                "Total: $11.88",
            ],
        )

    def test_remove(self):
        self.cart.add("PEN", 3)
        self.cart.remove("PEN", 1)
        self.assertEqual(self.cart.lines, {"PEN": 2})
        self.cart.remove("PEN")
        self.assertEqual(self.cart.lines, {})


if __name__ == "__main__":
    unittest.main()
