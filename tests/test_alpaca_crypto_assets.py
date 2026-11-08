"""The Alpaca crypto **asset registry** read: its default transport, its injected
transport, its refusals, and the fact that a credential never reaches an error.

The production failure this pins down was a ``TypeError: 'NoneType' object is
not callable``. ``AlpacaCryptoClient.assets()`` chose its HTTP seam with
``transport or self._transport``, and in a real run *both* were ``None``: the
method had no transport of its own to fall back on, so it called ``None``. The
fix is to fall back on the project's own ``_default_transport`` — the same one
``providers/base.py`` uses, so the timeout, the headers and the params are the
ones every other provider read already uses.

The tests here cover the four things that must hold:

- ``assets()`` works with **no** transport passed at all;
- ``assets()`` still honours an **injected** transport;
- HTTP 401, HTTP 403 and a connection failure are each reported as themselves,
  not swallowed and not turned into a success;
- no Alpaca credential (key or secret) appears in any error message.

Every test runs offline: the default transport is exercised through the seam it
uses, not through a live socket.
"""

from __future__ import annotations

import json

import pytest

from tradingagents.paper_experiment.providers import alpaca_crypto as crypto_module
from tradingagents.paper_experiment.providers.alpaca_crypto import AlpacaCryptoClient
from tradingagents.paper_experiment.providers.base import (
    HttpResponse,
    ProviderAuthError,
    ProviderError,
    ProviderUnsupportedError,
)

PAPER_URL = "https://paper-api.alpaca.markets"

API_KEY = "AKIAPAPERKEY0123456789"
API_SECRET = "papersecret0123456789abcdefghijklmnop"

ASSET_ROWS = [
    {
        "id": "b61b74c0-1642-4a3f-b5f0-4f7ad4b7ad01",
        "class": "crypto",
        "exchange": "CRYPTO",
        "symbol": "BTC/USD",
        "status": "active",
        "tradable": True,
        "marginable": False,
        "fractionable": True,
        "min_order_size": "0.0001",
        "min_trade_increment": "0.00000001",
        "price_increment": "0.01",
        "attributes": ["fractionable"],
    },
    {
        "id": "b61b74c0-1642-4a3f-b5f0-4f7ad4b7ad02",
        "class": "crypto",
        "exchange": "CRYPTO",
        "symbol": "DOGE/USD",
        "status": "active",
        "tradable": True,
        "marginable": False,
        "fractionable": True,
        "min_order_size": "1",
        "min_trade_increment": "1",
        "price_increment": "0.0001",
        "attributes": ["fractionable"],
    },
]


def _client(**kwargs) -> AlpacaCryptoClient:
    return AlpacaCryptoClient(
        api_key=kwargs.pop("api_key", API_KEY),
        api_secret=kwargs.pop("api_secret", API_SECRET),
        **kwargs,
    )


@pytest.fixture
def default_transport(monkeypatch):
    """Replace the project's default HTTP transport and record what it was asked.

    The provider imports ``_default_transport`` by name, so the seam that is
    actually used is the one in the provider's own namespace.
    """
    sent: list[dict] = []

    def fake_default(*, url, params, headers, timeout):
        sent.append(
            {
                "url": url,
                "params": dict(params),
                "headers": dict(headers),
                "timeout": timeout,
            }
        )
        return HttpResponse(200, json.dumps(ASSET_ROWS))

    monkeypatch.setattr(crypto_module, "_default_transport", fake_default)
    return sent


# ---------------------------------------------------------------------------
# The default transport
# ---------------------------------------------------------------------------


class TestTheDefaultTransport:
    def test_assets_works_with_no_transport_passed(self, default_transport):
        """The production call: no transport at all, and no client's own either.

        This is the exact shape that raised ``'NoneType' object is not callable``
        in production. The method must reach its own default transport instead.
        """
        client = _client()
        assert client._transport is None, "this test is about the no-transport case"

        assets = client.assets(base_url=PAPER_URL)

        assert len(assets) == 2
        assert assets[0]["symbol"] == "BTC/USD"
        assert default_transport, "the default transport was never used"

    def test_the_default_transport_keeps_the_timeout_headers_and_params(
        self, default_transport
    ):
        """The fallback must read like every other provider read."""
        client = _client()
        client.assets(base_url=PAPER_URL)

        call = default_transport[0]
        assert call["url"] == f"{PAPER_URL}/v2/assets"
        assert call["params"] == {"asset_class": "crypto", "status": "active"}
        assert call["timeout"] == client._timeout
        # The credential is sent in the header; it is the *request* that carries it.
        assert "APCA-API-KEY-ID" in call["headers"]
        assert "APCA-API-SECRET-KEY" in call["headers"]

    def test_the_read_goes_to_the_paper_host(self, default_transport):
        """The asset registry is read from the paper trading host, never a live one."""
        _client().assets(base_url=PAPER_URL)

        assert [call["url"] for call in default_transport] == [
            f"{PAPER_URL}/v2/assets"
        ]
        assert "live" not in default_transport[0]["url"]


# ---------------------------------------------------------------------------
# The injected transport
# ---------------------------------------------------------------------------


class TestTheInjectedTransport:
    def test_an_injected_transport_is_used(self):
        calls: list[dict] = []

        def transport(*, url, params, headers, timeout):
            calls.append({"url": url, "params": dict(params), "timeout": timeout})
            return HttpResponse(200, json.dumps(ASSET_ROWS))

        client = _client()
        assets = client.assets(base_url=PAPER_URL, transport=transport)

        assert len(assets) == 2
        assert len(calls) == 1
        assert calls[0]["url"] == f"{PAPER_URL}/v2/assets"

    def test_a_client_level_transport_is_used(self):
        """A transport built into the client is used when none is passed."""
        calls: list[dict] = []

        def transport(*, url, params, headers, timeout):
            calls.append({"url": url})
            return HttpResponse(200, json.dumps(ASSET_ROWS))

        client = _client(transport=transport)
        assets = client.assets(base_url=PAPER_URL)

        assert len(assets) == 2
        assert len(calls) == 1

    def test_the_passed_transport_wins_over_the_client_transport(self):
        calls: list[str] = []

        def client_transport(*, url, params, headers, timeout):
            calls.append("client")
            return HttpResponse(200, json.dumps(ASSET_ROWS))

        def passed_transport(*, url, params, headers, timeout):
            calls.append("passed")
            return HttpResponse(200, json.dumps(ASSET_ROWS))

        client = _client(transport=client_transport)
        client.assets(base_url=PAPER_URL, transport=passed_transport)
        assert calls == ["passed"]

    def test_a_transport_that_fails_is_reported_not_hidden(self):
        def transport(**_):
            raise OSError("connection refused")

        client = _client()
        with pytest.raises(ProviderError, match="connection refused"):
            client.assets(base_url=PAPER_URL, transport=transport)


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


class TestTheRefusals:
    def test_http_401_is_reported_as_an_auth_failure(self):
        def transport(**_):
            return HttpResponse(401, '{"message":"access key verification failed"}')

        client = _client()
        with pytest.raises(ProviderAuthError, match="401"):
            client.assets(base_url=PAPER_URL, transport=transport)

    def test_http_403_is_reported_as_an_unsupported_request(self):
        def transport(**_):
            return HttpResponse(403, '{"message":"crypto is not enabled for this account"}')

        client = _client()
        with pytest.raises(ProviderUnsupportedError, match="403"):
            client.assets(base_url=PAPER_URL, transport=transport)

    def test_a_connection_failure_is_reported(self):
        def transport(**_):
            raise ConnectionError("connection refused")

        client = _client()
        with pytest.raises(ProviderError, match="connection refused"):
            client.assets(base_url=PAPER_URL, transport=transport)

    def test_a_non_json_body_is_reported(self):
        def transport(**_):
            return HttpResponse(200, "<html>maintenance</html>")

        client = _client()
        with pytest.raises(ProviderError, match="JSON"):
            client.assets(base_url=PAPER_URL, transport=transport)

    def test_a_body_that_is_not_an_array_is_reported(self):
        def transport(**_):
            return HttpResponse(200, '{"assets": []}')

        client = _client()
        with pytest.raises(ProviderError, match="array"):
            client.assets(base_url=PAPER_URL, transport=transport)

    def test_a_refusal_is_never_read_as_an_empty_registry(self):
        """A failed read must not look like "this venue trades nothing"."""

        def transport(**_):
            return HttpResponse(401, '{"message":"access key verification failed"}')

        client = _client()
        with pytest.raises(ProviderError):
            client.assets(base_url=PAPER_URL, transport=transport)
        assert client.last_error is not None

    def test_a_live_host_is_refused_before_any_request(self):
        """A live trading host is refused by the method itself."""
        calls: list[dict] = []

        def transport(**_):
            calls.append({"url": "called"})
            return HttpResponse(200, json.dumps(ASSET_ROWS))

        client = _client()
        with pytest.raises(Exception) as excinfo:
            client.assets(base_url="https://api.alpaca.markets", transport=transport)
        assert "paper" in str(excinfo.value).lower()
        assert calls == []


# ---------------------------------------------------------------------------
# A credential never reaches an error message
# ---------------------------------------------------------------------------


class TestNoCredentialLeaks:
    def test_the_key_is_absent_from_a_401_message(self):
        def transport(**_):
            return HttpResponse(401, f'{{"message":"bad key {API_KEY}"}}')

        client = _client()
        with pytest.raises(ProviderError) as excinfo:
            client.assets(base_url=PAPER_URL, transport=transport)
        assert API_KEY not in str(excinfo.value)

    def test_the_secret_is_absent_from_a_403_message(self):
        def transport(**_):
            return HttpResponse(403, f'{{"message":"bad secret {API_SECRET}"}}')

        client = _client()
        with pytest.raises(ProviderError) as excinfo:
            client.assets(base_url=PAPER_URL, transport=transport)
        assert API_SECRET not in str(excinfo.value)

    def test_the_key_is_absent_from_a_transport_failure(self):
        def transport(**_):
            raise OSError(f"failed with {API_KEY} and {API_SECRET}")

        client = _client()
        with pytest.raises(ProviderError) as excinfo:
            client.assets(base_url=PAPER_URL, transport=transport)
        message = str(excinfo.value)
        assert API_KEY not in message
        assert API_SECRET not in message

    def test_the_secret_is_absent_from_a_non_json_body(self):
        def transport(**_):
            return HttpResponse(200, f"<html>{API_SECRET}</html>")

        client = _client()
        with pytest.raises(ProviderError) as excinfo:
            client.assets(base_url=PAPER_URL, transport=transport)
        assert API_SECRET not in str(excinfo.value)

    def test_no_credential_reaches_a_non_200_body(self):
        """The generic refusal path redacts the body too."""
        def transport(**_):
            return HttpResponse(500, f"upstream saw {API_KEY}")

        client = _client()
        with pytest.raises(ProviderError) as excinfo:
            client.assets(base_url=PAPER_URL, transport=transport)
        assert API_KEY not in str(excinfo.value)


# ---------------------------------------------------------------------------
# The registry the read feeds
# ---------------------------------------------------------------------------


class TestTheRegistryItFeeds:
    def test_the_read_feeds_the_execution_registry(self, monkeypatch):
        """The registry is built from what the venue answered, not from a list."""
        from tradingagents.paper_experiment.crypto_venue import (
            read_alpaca_execution_registry,
        )

        monkeypatch.setenv("ALPACA_PAPER_API_KEY", API_KEY)
        monkeypatch.setenv("ALPACA_PAPER_API_SECRET", API_SECRET)

        def transport(**_):
            return HttpResponse(200, json.dumps(ASSET_ROWS))

        registry = read_alpaca_execution_registry(
            base_url=PAPER_URL, transport=transport
        )
        assert registry.get("BTC/USD") is not None
        assert registry.get("BTC/USD").tradable is True

    def test_a_symbol_the_venue_does_not_list_is_not_tradable(self, monkeypatch):
        """A coin is never accepted without the venue's own registry."""
        from tradingagents.paper_experiment.crypto_venue import (
            read_alpaca_execution_registry,
        )

        monkeypatch.setenv("ALPACA_PAPER_API_KEY", API_KEY)
        monkeypatch.setenv("ALPACA_PAPER_API_SECRET", API_SECRET)

        def transport(**_):
            return HttpResponse(200, json.dumps(ASSET_ROWS))

        registry = read_alpaca_execution_registry(
            base_url=PAPER_URL, transport=transport
        )
        assert registry.get("DOGE/USD") is not None
        assert registry.get("XRP/USD") is None


__all__ = ["ASSET_ROWS", "PAPER_URL"]
