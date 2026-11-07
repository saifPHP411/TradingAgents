"""The tradable universe: which pairs a run may touch, resolved against the venue.

A user asks for ``XRP/USD``; a venue may serve ``XRP/USDT`` and not ``XRP/USD``.
Silently trading a different instrument than the one asked for is exactly the kind
of substitution this project refuses, so this module makes the resolution
**explicit and reported**:

- a requested pair the venue lists and marks tradable is *kept*;
- a requested pair the venue does not list is *substituted* only when a
  substitution is allowed, and the substitution (``XRP/USD -> XRP/USDT``) is
  recorded, not hidden;
- otherwise it is *unsupported*, with the venue's own reason.

The default universe is the ten major pairs the project was asked to support,
written in canonical form. :func:`resolve_universe` turns it into the set a run
may actually touch, and reports what it could not use — so the dashboard can show
"XRP/USD is not listed; using XRP/USDT" rather than a pair that quietly produces
no data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .instruments import InstrumentRegistry
from .symbols import CryptoPair, parse_pair

DEFAULT_SYMBOLS: tuple[str, ...] = (
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "BNB/USD",
    "DOGE/USD",
    "AVAX/USD",
    "LINK/USD",
    "LTC/USD",
    "BCH/USD",
)
"""The ten major pairs the system starts with, canonical ``BASE/USD``.

The list is a *request*, not a promise: whether each pair is tradeable is decided
by the venue (see :func:`resolve_universe`), and a pair the venue does not serve is
reported rather than assumed.
"""

QUOTE_PREFERENCE: tuple[str, ...] = ("USD", "USDT", "USDC")
"""Quote assets to prefer, in order, when a requested pair is not listed.

The venue's deep market for most majors is USDT, so a ``/USD`` request resolves to
``/USDT`` where ``/USD`` is absent. Only used when substitution is allowed.
"""


@dataclass
class UniverseResolution:
    """The outcome of resolving a requested universe against a venue's registry."""

    requested: tuple[str, ...]
    tradable: tuple[str, ...] = ()
    substituted: dict[str, str] = field(default_factory=dict)
    """``requested -> resolved`` for every pair whose quote was substituted."""
    unsupported: dict[str, str] = field(default_factory=dict)
    """``requested -> reason`` for every pair the venue will not trade."""

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.tradable

    def to_dict(self) -> dict:
        return {
            "requested": list(self.requested),
            "tradable": list(self.tradable),
            "substituted": dict(self.substituted),
            "unsupported": dict(self.unsupported),
        }

    def describe(self) -> str:
        parts = [f"{len(self.tradable)} tradable"]
        if self.substituted:
            pairs = ", ".join(f"{k}->{v}" for k, v in self.substituted.items())
            parts.append(f"substituted: {pairs}")
        if self.unsupported:
            pairs = ", ".join(f"{k} ({v})" for k, v in self.unsupported.items())
            parts.append(f"unsupported: {pairs}")
        return "; ".join(parts)


def _preferred_pair(pair: CryptoPair, registry: InstrumentRegistry) -> CryptoPair | None:
    """A same-base pair in a preferred quote the venue trades, or ``None``."""
    for quote in QUOTE_PREFERENCE:
        candidate = CryptoPair(base=pair.base, quote=quote)
        rules = registry.get(candidate.canonical)
        if rules is not None and rules.tradable:
            return candidate
    return None


def resolve_universe(
    registry: InstrumentRegistry,
    requested: tuple[str, ...] | list[str] | None = None,
    *,
    allow_substitution: bool = True,
) -> UniverseResolution:
    """Resolve ``requested`` (default :data:`DEFAULT_SYMBOLS`) against ``registry``.

    A pair the venue lists and marks tradable is kept as-is. A pair the venue does
    not list, or lists as non-tradable, is either substituted for the venue's
    same-base pair in a preferred quote (when ``allow_substitution``) or recorded as
    unsupported with the venue's status. Nothing is dropped silently: every
    requested pair appears in exactly one of ``tradable``, ``substituted`` or
    ``unsupported``.
    """
    symbols = tuple(requested) if requested is not None else DEFAULT_SYMBOLS
    resolution = UniverseResolution(requested=tuple(parse_pair(s).canonical for s in symbols))
    tradable: list[str] = []
    for raw in symbols:
        pair = parse_pair(raw)
        canonical = pair.canonical
        rules = registry.get(canonical)
        if rules is not None and rules.tradable:
            tradable.append(canonical)
            continue
        reason = "not listed" if rules is None else rules.status
        if allow_substitution:
            preferred = _preferred_pair(pair, registry)
            if preferred is not None and preferred.canonical != canonical:
                tradable.append(preferred.canonical)
                resolution.substituted[canonical] = preferred.canonical
                continue
        resolution.unsupported[canonical] = reason
    resolution.tradable = tuple(sorted(set(tradable)))
    return resolution


__all__ = [
    "DEFAULT_SYMBOLS",
    "QUOTE_PREFERENCE",
    "UniverseResolution",
    "resolve_universe",
]
