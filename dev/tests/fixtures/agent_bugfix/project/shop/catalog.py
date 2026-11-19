"""The items a shop sells, read from CSV text."""

import csv
import io
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    sku: str
    name: str
    # The price as the catalog writes it, a decimal amount such as "4.99".
    price: str


class Catalog:
    def __init__(self, items):
        self.items = {item.sku: item for item in items}

    @classmethod
    def from_csv(cls, text):
        """A catalog from CSV text with the header sku,name,price."""
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames != ["sku", "name", "price"]:
            raise ValueError(f"unexpected header {reader.fieldnames}")
        return cls(
            Item(row["sku"], row["name"], row["price"].strip()) for row in reader
        )

    def item(self, sku):
        try:
            return self.items[sku]
        except KeyError:
            raise KeyError(f"no item {sku!r} in the catalog") from None
