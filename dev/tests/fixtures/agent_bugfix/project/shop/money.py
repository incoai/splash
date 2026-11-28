"""Amounts of money as integer cents, and their text forms."""


def to_cents(text):
    """The cents of a decimal amount written as text: "12.34" is 1234,
    "12.5" is 1250, "7" is 700 and "-0.25" is -25. Surrounding spaces and a
    leading "$" are allowed; more than two decimals are an error."""
    value = text.strip()
    negative = value.startswith("-")
    if negative:
        value = value[1:]
    if value.startswith("$"):
        value = value[1:]
    whole, _, fraction = value.partition(".")
    if not whole.isdigit() and not (whole == "" and fraction):
        raise ValueError(f"not an amount: {text!r}")
    if len(fraction) > 2 or (fraction and not fraction.isdigit()):
        raise ValueError(f"not an amount: {text!r}")
    cents = int(whole or "0") * 100 + int(fraction or "0")
    return -cents if negative else cents


def format_cents(cents):
    """The text form of an amount of cents: 1234 is "$12.34", -5 is
    "-$0.05"."""
    sign = "-" if cents < 0 else ""
    whole, fraction = divmod(abs(cents), 100)
    return f"{sign}${whole:,}.{fraction:02d}"


def split_evenly(cents, parts):
    """Splits an amount into `parts` amounts that differ by at most a cent
    and add up to it, the larger ones first."""
    if parts <= 0:
        raise ValueError("parts must be positive")
    share, remainder = divmod(cents, parts)
    return [share + 1 if index < remainder else share for index in range(parts)]


def percent_of(cents, percent):
    """`percent` percent of an amount, rounded half away from zero to a
    cent."""
    scaled = cents * percent
    whole, rest = divmod(abs(scaled), 100)
    if rest * 2 >= 100:
        whole += 1
    return -whole if scaled < 0 else whole
