"""Alpaca's **crypto** price and asset registry — the execution side of the crypto path.

The crypto paper path reads two different markets and must not confuse them:

- **Binance** answers "what is the crypto market doing right now" and drives the TJR
  analysis. Its pairs are USDT-quoted (``BTC/USDT``).
- **Alpaca** answers "what would this cost *me*, right now, in the pair I can
  actually trade" and drives order sizing. Its crypto pairs are USD-quoted
  (``BTC/USD``).

A Binance ``BTC/USDT`` price is therefore **not** an Alpaca ``BTC/USD`` price: the
quote assets differ, the venues differ, and the two can drift apart. Sizing an
Alpaca order from a Binance price would commit a notional computed from the wrong
instrument. This module exists so the execution side has its own, honest source:

- :meth:`AlpacaCryptoClient.latest_snapshot` reads Alpaca's crypto **snapshots**
  endpoint for a current price in the *execution* pair, with the timestamp of the
  field the price came from.
- :meth:`AlpacaCryptoClient.assets` reads the venue's own **assets** registry, so
  "is this pair tradable here?" is answered by the venue and not by a hard-coded
  list.

Both are read-only market-data calls on ``data.alpaca.markets``. Neither places an
order, and neither touches the live trading host. The transport is injectable, so
every branch is exercised offline.

Endpoint facts, taken from Alpaca's own reference documentation rather than
assumed:

- ``GET https://data.alpaca.markets/v1beta3/crypto/us/snapshots?symbols=BTC/USD``
  returns ``{"snapshots": {"BTC/USD": {...}}}``, where each entry carries a
  ``latestTrade`` (``p``, ``t``), a ``latestQuote`` (``bp``/``ap``, ``t``) and a
  ``minuteBar`` (``c``, ``t``).
- ``GET https://paper-api.alpaca.markets/v2/assets?asset_class=crypto`` returns the
  tradable crypto assets, each with ``symbol`` (``BTC/USD``), ``tradable``,
  ``fractionable``, ``min_order_size``, ``min_trade_increment`` and
  ``price_increment``.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from ..broker.alpaca_paper import assert_paper_endpoint
from .base import (
    ProviderAuthError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderUnsupportedError,
    Transport,
    fetch_json,
    _default_transport,
    _redact,
)

DATA_BASE_URL = "https://data.alpaca.markets"
"""The only market-data host this provider will talk to."""

LIVE_DATA_HOSTS = ("api.alpaca.markets", "live-api.alpaca.markets")
"""Hosts that serve the *live* trading API. Never a data host, and never allowed."""


def assert_data_host(base_url: str) -> str:
    """Return ``base_url`` if it is the approved market-data host, else raise.

    The market-data host is read-only, so a live *trading* host is not merely
    unnecessary here — it is a sign that the wrong endpoint was configured, and
    this provider refuses it rather than issuing the request. The check happens
    at construction, before any credential is read, so a misconfiguration cannot
    send a request anywhere.
    """
    host = base_url.strip().lower().split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    if host in LIVE_DATA_HOSTS:
        raise AlpacaCryptoHostRefused(
            f"{base_url!r} is an Alpaca LIVE trading host; this provider reads market "
            f"data only, from {DATA_BASE_URL}"
        )
    if host != DATA_BASE_URL.split("://", 1)[-1]:
        raise AlpacaCryptoHostRefused(
            f"{base_url!r} is not the approved Alpaca market-data host ({DATA_BASE_URL}); "
            "refusing to read market data from an unapproved host"
        )
    return base_url


REQUEST_TIMEOUT = 15.0

API_KEY_ENV = "ALPACA_PAPER_API_KEY"
API_SECRET_ENV = "ALPACA_PAPER_API_SECRET"

CRYPTO_LOCATION = "us"
"""Alpaca's US crypto location (``/v1beta3/crypto/{loc}/...``)."""

SNAPSHOTS_PATH = f"/v1beta3/crypto/{CRYPTO_LOCATION}/snapshots"


class AlpacaCryptoError(ProviderError):
    """Alpaca's crypto market data could not serve a request."""


class AlpacaCryptoHostRefused(AlpacaCryptoError):
    """A host other than the approved market-data host was requested."""


def _auth_headers(key: str, secret: str) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }


def _classify_status(status: int, body: str) -> None:
    """Give Alpaca's own refusal codes a specific, named error."""
    if status == 401:
        raise ProviderAuthError(f"Alpaca rejected the credential (HTTP 401): {body}")
    if status == 403:
        raise ProviderUnsupportedError(
            f"Alpaca refused the crypto request (HTTP 403); the account or plan may "
            f"not include crypto market data: {body}"
        )
    if status == 429:
        raise ProviderError(f"Alpaca throttled the request (HTTP 429): {body}")


class AlpacaCryptoClient:
    """Read-only access to Alpaca's crypto prices and crypto asset registry.

    The client never sends an order and never reads a live-trading credential
    variable: it uses the **paper** credential pair, which is also the credential
    the paper broker uses, and it only ever calls the market-data host.
    """

    provider_name = "alpaca_crypto"
    kind = "vendor_rest"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = DATA_BASE_URL,
        transport: Transport | None = None,
        timeout: float = REQUEST_TIMEOUT,
    ) -> None:
        self._base_url = assert_data_host(base_url).rstrip("/")
        self._transport = transport
        self._timeout = timeout
        key = (api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")) or ""
        secret = (
            api_secret if api_secret is not None else os.environ.get(API_SECRET_ENV, "")
        ) or ""
        self._key = key.strip()
        self._secret = secret.strip()
        self.last_error: str | None = None

    @property
    def configured(self) -> bool:
        """Whether a credential is available. Never reveals the credential."""
        return bool(self._key and self._secret)

    def describe(self) -> dict[str, Any]:
        return {
            "execution_data_provider": self.provider_name,
            "execution_data_host": self._base_url,
            "execution_data_requires_key": True,
            "execution_data_configured": self.configured,
            "execution_data_last_error": self.last_error,
        }

    # -- prices -----------------------------------------------------------

    def latest_snapshot(self, symbols: list[str] | tuple[str, ...]) -> dict[str, Any]:
        """The latest snapshot for each **execution** symbol, keyed by that symbol.

        ``symbols`` are Alpaca execution spellings (``BTC/USD``). The response is
        Alpaca's own ``{"snapshots": {...}}`` body, unparsed, so the caller chooses
        which field it trusts and can apply its own freshness rule.
        """
        if not symbols:
            return {}
        if not self.configured:
            raise ProviderNotConfiguredError(
                f"no Alpaca paper credentials: set {API_KEY_ENV} and {API_SECRET_ENV} "
                "so an execution-side price can be read"
            )
        payload = fetch_json(
            url=f"{self._base_url}{SNAPSHOTS_PATH}",
            params={"symbols": ",".join(symbols)},
            timeout=self._timeout,
            secrets=(self._key, self._secret),
            headers=_auth_headers(self._key, self._secret),
            transport=self._transport,
            classify_status=_classify_status,
        )
        snapshots = payload.get("snapshots")
        if not isinstance(snapshots, dict):
            raise AlpacaCryptoError("Alpaca's snapshot response carried no 'snapshots' object")
        self.last_error = None
        return snapshots

    # -- assets -----------------------------------------------------------

    def assets(self, *, base_url: str, transport: Transport | None = None) -> list[dict[str, Any]]:
        """The venue's crypto assets, read from the **paper** trading host.

        The asset registry lives on the trading API (not the market-data host), so
        this is a paper-only read against ``paper-api.alpaca.markets``. It issues
        no order.

        The host is validated here with the project's own ``assert_paper_endpoint``
        gate, not left to the caller: a live trading host must be refused by
        *this* method, so the refusal does not depend on how it was reached.
        """
        base_url = assert_paper_endpoint(base_url)
        if not self.configured:
            raise ProviderNotConfiguredError(
                f"no Alpaca paper credentials: set {API_KEY_ENV} and {API_SECRET_ENV} "
                "so the execution venue's asset registry can be read"
            )
        # Use the injected transport, or the client's transport, or the default transport
        # from providers.base. This ensures a default HTTP client is available when
        # no transport is explicitly provided (tests can still inject their own).
        send = transport or self._transport or _default_transport
        secrets = (self._key, self._secret)
        try:
            response = send(
                url=f"{base_url.rstrip('/')}/v2/assets",
                params={"asset_class": "crypto", "status": "active"},
                headers=_auth_headers(self._key, self._secret),
                timeout=self._timeout,
            )
        except ProviderError:
            raise
        except Exception as exc:  # a transport failure is reported, never hidden
            message = _redact(f"{type(exc).__name__}: {exc}", secrets)
            raise ProviderError(f"the request failed: {message}") from None
        if response.status_code != 200:
            body = _redact((response.text or "")[:200], secrets)
            _classify_status(response.status_code, body)
            raise AlpacaCryptoError(f"Alpaca answered HTTP {response.status_code}: {body}")
        import json

        try:
            payload = json.loads(response.text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise AlpacaCryptoError(
                f"Alpaca did not answer with JSON: {_redact(str(exc), secrets)}"
            ) from None
        if not isinstance(payload, list):
            raise AlpacaCryptoError("Alpaca's asset response was not a JSON array")
        self.last_error = None
        return [entry for entry in payload if isinstance(entry, dict)]


def snapshot_price_and_time(snapshot: Any) -> tuple[float | None, datetime | None, str | None]:
    """The newest usable price in an Alpaca crypto snapshot, with its own timestamp.

    Returns ``(price, observed_at, field)``. The candidates are the latest trade,
    the quote midpoint and the minute-bar close, ranked by the timestamp each one
    carries — the same "newest observation wins" rule the equity snapshot seam
    uses, because a quote can stay fresh while a symbol goes minutes without
    printing a trade.

    A candidate with no usable timestamp cannot outrank one that has one; it is
    returned only when *nothing* has a usable timestamp, so the caller can name
    precisely why the price is unusable. Every candidate must be finite and
    positive, so a NaN cannot pass a comparison and poison a later sizing step.
    """
    if not isinstance(snapshot, dict):
        return None, None, None

    candidates: list[tuple[datetime, int, float, str]] = []
    order = {"latestTrade": 0, "latestQuote": 1, "minuteBar": 2}

    trade = snapshot.get("latestTrade")
    if isinstance(trade, dict):
        price = _as_positive_float(trade.get("p"))
        stamp = _parse_time(trade.get("t"))
        if price is not None and stamp is not None:
            candidates.append((stamp, order["latestTrade"], price, "latestTrade"))

    quote = snapshot.get("latestQuote")
    if isinstance(quote, dict):
        bid = _as_positive_float(quote.get("bp"))
        ask = _as_positive_float(quote.get("ap"))
        if bid is not None and ask is not None:
            price = (bid + ask) / 2.0
        elif bid is not None:
            price = bid
        elif ask is not None:
            price = ask
        else:
            price = None
        stamp = _parse_time(quote.get("t"))
        if price is not None and stamp is not None:
            candidates.append((stamp, order["latestQuote"], price, "latestQuote"))

    bar = snapshot.get("minuteBar")
    if isinstance(bar, dict):
        price = _as_positive_float(bar.get("c"))
        stamp = _parse_time(bar.get("t"))
        if price is not None and stamp is not None:
            candidates.append((stamp, order["minuteBar"], price, "minuteBar"))

    if candidates:
        # Newest wins; an exact tie is settled by the fixed field preference.
        best = max(candidates, key=lambda c: (c[0], -c[1]))
        return best[2], best[0], best[3]

    # No candidate carried a usable timestamp: fall back to the first usable
    # price so the caller can report *why* the observation is unverifiable.
    for field in ("latestTrade", "latestQuote", "minuteBar"):
        entry = snapshot.get(field)
        if not isinstance(entry, dict):
            continue
        raw = entry.get("p") if field == "latestTrade" else entry.get("c")
        if field == "latestQuote":
            bid = _as_positive_float(entry.get("bp"))
            ask = _as_positive_float(entry.get("ap"))
            raw = (bid + ask) / 2.0 if (bid is not None and ask is not None) else (bid or ask)
        price = _as_positive_float(raw)
        if price is not None:
            return price, None, field
    return None, None, None


def _as_positive_float(value: Any) -> float | None:
    import math

    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = [
    "API_KEY_ENV",
    "API_SECRET_ENV",
    "CRYPTO_LOCATION",
    "DATA_BASE_URL",
    "REQUEST_TIMEOUT",
    "SNAPSHOTS_PATH",
    "AlpacaCryptoClient",
    "AlpacaCryptoError",
    "snapshot_price_and_time",
]
