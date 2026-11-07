"""Canonical crypto pair symbols, and the mapping to each venue's own spelling.

A crypto instrument is a **pair**: a base asset quoted in a quote asset. The same
instrument is written several ways depending on who is asking — ``BTC/USD`` in a
report, ``BTCUSD`` on a broker's order ticket, ``BTCUSDT`` on a Binance stream,
``BTC-USD`` on Yahoo. Treating those as different symbols is how a system ends up
with two positions in the same coin, so this module defines one canonical form and
parses everything else into it.

Canonical form is ``BASE/QUOTE``, upper-cased, with a single slash: ``BTC/USD``.
:class:`CryptoPair` is the value; :func:`parse_pair` reads a user's spelling into
one, and :meth:`CryptoPair.venue_symbol` writes the venue's spelling back out.

The quote is kept as the venue spells it, so a pair parsed from ``BTCUSDT`` keeps
``USDT`` as its quote rather than being silently rewritten to ``USD`` — the two are
different instruments, and only the venue's own asset list can say which is
tradeable.
"""

from __future__ import annotations

from dataclasses import dataclass

# Quote assets a user may write, longest first so ``USDT``/``USDC`` match before
# the ``USD`` substring they contain. This is only for *parsing* a bare string; the
# authoritative list of a venue's assets comes from the venue (see ``instruments``).
KNOWN_QUOTES: tuple[str, ...] = (
    "USDT",
    "USDC",
    "BUSD",
    "TUSD",
    "FDUSD",
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "TRY",
    "BRL",
    "BTC",
    "ETH",
    "BNB",
)

_SEPARATORS = ("/", "-", "_", ":")


class SymbolError(ValueError):
    """A crypto symbol could not be parsed into a pair."""


@dataclass(frozen=True)
class CryptoPair:
    """A base/quote crypto instrument, canonical and comparable.

    Equality and hashing are on the canonical ``BASE/QUOTE`` string, so a pair
    built from ``BTCUSD`` and one built from ``BTC/USD`` are the same object for
    every dict key and set membership.
    """

    base: str
    quote: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "base", self.base.strip().upper())
        object.__setattr__(self, "quote", self.quote.strip().upper())
        if not self.base or not self.quote:
            raise SymbolError(f"a pair needs a base and a quote, got {self.base!r}/{self.quote!r}")
        if self.base == self.quote:
            raise SymbolError(f"a pair's base and quote must differ, got {self.canonical}")

    @property
    def canonical(self) -> str:
        """The canonical ``BASE/QUOTE`` spelling."""
        return f"{self.base}/{self.quote}"

    def venue_symbol(self, *, separator: str = "") -> str:
        """The venue's spelling: ``separator`` between base and quote.

        ``separator=""`` gives Binance's ``BTCUSDT``; ``separator="/"`` gives
        Alpaca's ``BTC/USD``; ``separator="-"`` gives Yahoo's ``BTC-USD``.
        """
        return f"{self.base}{separator}{self.quote}"

    @property
    def binance_symbol(self) -> str:
        return self.venue_symbol(separator="")

    @property
    def alpaca_symbol(self) -> str:
        return self.venue_symbol(separator="/")

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.canonical


def _split_quoted(text: str) -> tuple[str, str] | None:
    """Split ``BTC/USD``-style text on its first separator."""
    for separator in _SEPARATORS:
        if separator in text:
            base, _, quote = text.partition(separator)
            if base and quote:
                return base, quote
    return None


def _split_bare(text: str) -> tuple[str, str] | None:
    """Split ``BTCUSDT``-style text by matching a known quote suffix.

    The *longest* known quote wins, so ``BTCUSDT`` splits into ``BTC``/``USDT``
    rather than ``BTCU``/``SDT``. A base that is itself a known quote (``ETH`` in
    ``ETHBTC``) is fine: the base is whatever is left after the suffix.
    """
    for quote in sorted(KNOWN_QUOTES, key=len, reverse=True):
        if text.endswith(quote) and len(text) > len(quote):
            return text[: -len(quote)], quote
    return None


def parse_pair(symbol: str) -> CryptoPair:
    """Parse a user- or venue-spelled crypto symbol into a :class:`CryptoPair`.

    Accepts ``BTC/USD``, ``BTC-USD``, ``BTC_USD``, ``BTCUSD`` and ``BTCUSDT``.
    Raises :class:`SymbolError` when the text is not a recognisable pair, so a bad
    symbol is a loud error rather than a lookup that silently finds nothing.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise SymbolError(f"a crypto symbol must be a non-empty string, got {symbol!r}")
    text = symbol.strip().upper()
    split = _split_quoted(text)
    if split is None:
        split = _split_bare(text)
    if split is None:
        raise SymbolError(
            f"cannot parse {symbol!r} as a crypto pair; expected BASE/QUOTE or a known "
            f"quote suffix (one of {', '.join(KNOWN_QUOTES)})"
        )
    return CryptoPair(base=split[0], quote=split[1])


def try_parse_pair(symbol: str) -> CryptoPair | None:
    """:func:`parse_pair`, but ``None`` instead of raising. For detection."""
    try:
        return parse_pair(symbol)
    except SymbolError:
        return None


def is_crypto_symbol(symbol: str) -> bool:
    """Whether ``symbol`` looks like a crypto pair this module can canonicalise."""
    return try_parse_pair(symbol) is not None


def canonical_symbol(symbol: str) -> str:
    """The canonical ``BASE/QUOTE`` spelling of ``symbol``."""
    return parse_pair(symbol).canonical


__all__ = [
    "KNOWN_QUOTES",
    "CryptoPair",
    "SymbolError",
    "canonical_symbol",
    "is_crypto_symbol",
    "parse_pair",
    "try_parse_pair",
]
