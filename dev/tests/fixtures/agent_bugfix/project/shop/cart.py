"""A shopping cart: lines of priced items, discount codes and tax."""

from shop import money
from shop.catalog import Catalog

# Discount codes: a percentage off the subtotal, and the subtotal in cents
# from which each applies.
DISCOUNTS = {
    "WELCOME10": (10, 0),
    "BIG20": (20, 10_000),
}


class Cart:
    def __init__(self, catalog: Catalog, tax_percent=0):
        self.catalog = catalog
        self.tax_percent = tax_percent
        self.lines = {}
        self.code = None

    def add(self, sku, quantity=1):
        """Adds `quantity` of an item the catalog lists; adding an item again
        adds to its line."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self.catalog.item(sku)
        self.lines[sku] = self.lines.get(sku, 0) + quantity

    def remove(self, sku, quantity=None):
        """Removes `quantity` of an item, or its whole line."""
        if sku not in self.lines:
            raise KeyError(sku)
        if quantity is None or quantity >= self.lines[sku]:
            del self.lines[sku]
        else:
            self.lines[sku] -= quantity

    def apply_code(self, code):
        if code not in DISCOUNTS:
            raise ValueError(f"unknown discount code {code!r}")
        self.code = code

    def subtotal(self):
        """The sum of the lines' prices, in cents."""
        return sum(
            money.to_cents(self.catalog.item(sku).price) * quantity
            for sku, quantity in self.lines.items()
        )

    def discount(self):
        """The discount the cart's code gives, in cents (0 without one, or
        below its minimum subtotal)."""
        if self.code is None:
            return 0
        percent, minimum = DISCOUNTS[self.code]
        subtotal = self.subtotal()
        return money.percent_of(subtotal, percent) if subtotal >= minimum else 0

    def tax(self):
        """Tax on the discounted subtotal, in cents."""
        return money.percent_of(self.subtotal() - self.discount(), self.tax_percent)

    def total(self):
        return self.subtotal() - self.discount() + self.tax()

    def receipt(self):
        """The cart as receipt lines, then its totals."""
        lines = []
        for sku, quantity in sorted(self.lines.items()):
            item = self.catalog.item(sku)
            price = money.to_cents(item.price) * quantity
            lines.append(f"{quantity} x {item.name}: {money.format_cents(price)}")
        lines.append(f"Subtotal: {money.format_cents(self.subtotal())}")
        if self.discount():
            lines.append(
                f"Discount ({self.code}): -{money.format_cents(self.discount())}"
            )
        if self.tax_percent:
            lines.append(f"Tax: {money.format_cents(self.tax())}")
        lines.append(f"Total: {money.format_cents(self.total())}")
        return lines
