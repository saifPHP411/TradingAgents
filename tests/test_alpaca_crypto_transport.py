"""The Alpaca crypto **asset registry** read: its transport, its failures, its secrets.

The registry answers "is this pair tradable on this venue, and in what
increment?". It is read from the **paper** trading host, so it is a paper-only
read that issues no order — but a read that cannot answer must **fail**, never
come back empty, because an empty registry looks like "this venue trades
nothing", which is a different and wrong claim.

The bug this module pins is a real one: ``assets()`` used to call
``transport or self._transport`` and nothing else. In a real run neither is set,
so the call became ``None(...)`` — a ``TypeError`` that surfaced as
``EXECUTION_REGISTRY_UNAVAILABLE`` and looked like a venue problem rather than
a null callable.

So the tests below are written to catch that premise, not to confirm the code:

- the **default** path is exercised for real, by patching the ``requests.get``
  that :func:`_default_transport` calls — not by passing a transport and calling
  it "the default";
- the injected paths are exercised separately, so an injected transport cannot
  silently stand in for the default one;
- every failure — 401, 403, a dead socket, a body that is not JSON, a body that
  is not an array — is checked for its **named** error and for the **absence of
  the credential**;
- the paper host is the only host the read will accept, and the check happens
  before any credential is read.

Every test is hermetic: no socket is opened and no order is placed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tradingagents.paper_experiment.broker.alpaca_paper import (
    LiveEndpointRefusedError,
)
from tradingagents.paper_experiment.providers.alpaca_crypto import (
    AlpacaCryptoClient,
    AlpacaCryptoError,
)
from tradingagents.paper_experiment.providers.base import (
    HttpResponse,
    ProviderAuthError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderUnsupportedError,
)

PAPER_URL = "https://paper-api.alpaca.markets"
API_KEY = "alpaca-test-key-abc123"
API_SECRET = "alpaca-test-secret-xyz789"


def _assets_payload() -> list[dict[str, Any]]:
    return [
        {
            "symbol": "BTC/USD",
            "class": "crypto",
            "status": "active",
            "tradable": True,
            "fractionable": True,
            "min_order_size": "0.0001",
            "min_trade_increment": "0.0001",
            "price_increment": "0.01",
        },
        {
            "symbol": "ETH/USD",
            "class": "crypto",
            "status": "active",
            "tradable": True,
            "fractionable": True,
            "min_order_size": "0.001",
            "min_trade_increment": "0.001",
            "price_increment": "0.01",
        },
    ]


class _RecordingTransport:
    """A transport that records what it was asked for and answers a fixed body."""

    def __init__(self, status_code: int = 200, body: str | None = None) -> None:
        self.status_code = status_code
        self.body = body if body is not None else json.dumps(_assets_payload())
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, url, params, headers, timeout):
        self.calls.append(
            {
                "url": url,
                "params": dict(params or {}),
                "headers": dict(headers or {}),
                "timeout": timeout,
            }
        )
        return HttpResponse(self.status_code, self.body)


class _BoomTransport:
    """A transport whose connection fails, the way a dead socket does."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def __call__(self, *, url, params, headers, timeout):
        raise self.exc


class _FakeRequestsResponse:
    """The part of ``requests.Response`` the default transport reads."""

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


@pytest.fixture
def patched_get(monkeypatch):
    """Patch the ``requests.get`` that the **default** transport calls.

    This is the real seam: :func:`_default_transport` calls ``requests.get``, so
    patching it exercises the default path end to end without opening a socket.
    A test that passes a transport cannot make this claim.
    """
    calls: list[dict[str, Any]] = []
    response = _FakeRequestsResponse(200, json.dumps(_assets_payload()))

    def fake_get(url, *, params=None, headers=None, timeout=None):
        calls.append(
            {
                "url": url,
                "params": dict(params or {}),
                "headers": dict(headers or {}),
                "timeout": timeout,
            }
        )
        return response

    monkeypatch.setattr("requests.get", fake_get)
    return calls


# ---------------------------------------------------------------------------
# 1. The default transport — the bug itself
# ---------------------------------------------------------------------------


class TestTheDefaultTransport:
    """``assets()`` must work with no transport at all.

    This is the exact shape of the reported failure: a real run passes no
    transport, so the method has to find one for itself.
    """

    def test_assets_works_with_no_transport_anywhere(self, patched_get):
        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)
        assert client._transport is None

        assets = client.assets(base_url=PAPER_URL)

        assert [entry["symbol"] for entry in assets] == ["BTC/USD", "ETH/USD"]
        assert patched_get, "the default transport was never called"

    def test_the_default_transport_asks_for_the_crypto_registry(self, patched_get):
        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)

        client.assets(base_url=PAPER_URL)

        call = patched_get[0]
        assert call["url"] == f"{PAPER_URL}/v2/assets"
        assert call["params"]["asset_class"] == "crypto"
        assert call["params"]["status"] == "active"

    def test_the_default_transport_carries_the_credential_headers(self, patched_get):
        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)

        client.assets(base_url=PAPER_URL)

        headers = patched_get[0]["headers"]
        assert headers["APCA-API-KEY-ID"] == API_KEY
        assert headers["APCA-API-SECRET-KEY"] == API_SECRET

    def test_the_default_transport_keeps_the_timeout(self, patched_get):
        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)

        client.assets(base_url=PAPER_URL)

        # An unbounded call is the one thing a provider must never make.
        assert patched_get[0]["timeout"] is not None
        assert patched_get[0]["timeout"] > 0

    def test_a_custom_timeout_reaches_the_default_transport(self, patched_get):
        client = AlpacaCryptoClient(
            api_key=API_KEY, api_secret=API_SECRET, timeout=3.5
        )

        client.assets(base_url=PAPER_URL)

        assert patched_get[0]["timeout"] == 3.5

    def test_the_default_transport_is_the_one_base_uses(self):
        """The fallback is the *same* transport ``providers.base`` uses.

        Pinned so the two cannot drift: a second, differently-configured default
        would mean the registry was read under different timeout or header rules
        than every other provider read.
        """
        from tradingagents.paper_experiment.providers import base

        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)
        # The client resolves to ``base._default_transport`` when nothing is
        # injected, exactly as ``fetch_json`` does.
        resolved = client._transport or base._default_transport
        assert resolved is base._default_transport

    def test_the_transport_is_never_none(self):
        """The regression guard: ``send`` is never ``None``.

        The reported failure was ``NoneType object is not callable`` — the
        method resolved its transport to ``None`` and then called it. This
        pins the *shape* of the defect, so the guard does not depend on which
        transport happens to be installed.
        """
        from tradingagents.paper_experiment.providers import base

        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)
        resolved = client._transport or base._default_transport
        assert resolved is not None
        assert callable(resolved)

    def test_a_none_transport_is_not_accepted_as_a_transport(self):
        """A caller passing ``transport=None`` means "no transport", not "call None"."""
        from tradingagents.paper_experiment.providers import base

        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)
        resolved = client._transport or base._default_transport
        assert resolved is not None


# ---------------------------------------------------------------------------
# 2. Injection still works
# ---------------------------------------------------------------------------


class TestTheInjectedTransport:
    """A caller may still inject its own transport, and it wins."""

    def test_a_transport_passed_to_the_method_is_used(self):
        transport = _RecordingTransport()
        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)

        assets = client.assets(base_url=PAPER_URL, transport=transport)

        assert len(assets) == 2
        assert len(transport.calls) == 1

    def test_a_transport_on_the_client_is_used(self):
        transport = _RecordingTransport()
        client = AlpacaCryptoClient(
            api_key=API_KEY, api_secret=API_SECRET, transport=transport
        )

        assets = client.assets(base_url=PAPER_URL)

        assert len(assets) == 2
        assert len(transport.calls) == 1

    def test_the_method_transport_wins_over_the_client_transport(self):
        client_transport = _RecordingTransport()
        method_transport = _RecordingTransport()
        client = AlpacaCryptoClient(
            api_key=API_KEY, api_secret=API_SECRET, transport=client_transport
        )

        client.assets(base_url=PAPER_URL, transport=method_transport)

        assert len(method_transport.calls) == 1
        assert client_transport.calls == []

    def test_the_default_transport_is_not_reached_when_one_is_injected(
        self, patched_get
    ):
        transport = _RecordingTransport()
        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)

        client.assets(base_url=PAPER_URL, transport=transport)

        assert patched_get == [], "the default transport was reached"

    def test_an_injected_transport_keeps_the_same_timeout_and_headers(self):
        transport = _RecordingTransport()
        client = AlpacaCryptoClient(
            api_key=API_KEY, api_secret=API_SECRET, timeout=7.0
        )

        client.assets(base_url=PAPER_URL, transport=transport)

        call = transport.calls[0]
        assert call["timeout"] == 7.0
        assert call["headers"]["APCA-API-KEY-ID"] == API_KEY
        assert call["params"]["asset_class"] == "crypto"


# ---------------------------------------------------------------------------
# 3. The failures are named, and they are failures
# ---------------------------------------------------------------------------


class TestTheFailures:
    """A failed read must raise a **named** error, never return an empty list."""

    def test_a_401_is_an_authentication_failure(self):
        transport = _RecordingTransport(401, '{"message": "access key verification failed"}')
        client = AlpacaCryptoClient(
            api_key=API_KEY, api_secret=API_SECRET, transport=transport
        )

        with pytest.raises(ProviderAuthError) as excinfo:
            client.assets(base_url=PAPER_URL)

        assert "401" in str(excinfo.value)

    def test_a_403_is_an_unsupported_request(self):
        transport = _RecordingTransport(403, '{"message": "crypto is not enabled"}')
        client = AlpacaCryptoClient(
            api_key=API_KEY, api_secret=API_SECRET, transport=transport
        )

        with pytest.raises(ProviderUnsupportedError) as excinfo:
            client.assets(base_url=PAPER_URL)

        assert "403" in str(excinfo.value)

    def test_a_429_is_a_provider_error(self):
        transport = _RecordingTransport(429, "rate limited")
        client = AlpacaCryptoClient(
            api_key=API_KEY, api_secret=API_SECRET, transport=transport
        )

        with pytest.raises(ProviderError):
            client.assets(base_url=PAPER_URL)

    def test_a_500_is_a_provider_error(self):
        transport = _RecordingTransport(500, "internal server error")
        client = AlpacaCryptoClient(
            api_key=API_KEY, api_secret=API_SECRET, transport=transport
        )

        with pytest.raises(ProviderError):
            client.assets(base_url=PAPER_URL)

    def test_a_connection_failure_is_a_provider_error(self):
        client = AlpacaCryptoClient(
            api_key=API_KEY,
            api_secret=API_SECRET,
            transport=_BoomTransport(ConnectionError("connection refused")),
        )

        with pytest.raises(ProviderError) as excinfo:
            client.assets(base_url=PAPER_URL)

        assert "connection" in str(excinfo.value).lower()

    def test_a_connection_failure_through_the_default_path_is_a_provider_error(
        self, monkeypatch
    ):
        def boom(url, *, params=None, headers=None, timeout=None):
            raise ConnectionError("connection refused")

        monkeypatch.setattr("requests.get", boom)
        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)

        with pytest.raises(ProviderError):
            client.assets(base_url=PAPER_URL)

    def test_a_body_that_is_not_json_is_a_provider_error(self):
        transport = _RecordingTransport(200, "<html>not json</html>")
        client = AlpacaCryptoClient(
            api_key=API_KEY, api_secret=API_SECRET, transport=transport
        )

        with pytest.raises(AlpacaCryptoError):
            client.assets(base_url=PAPER_URL)

    def test_a_body_that_is_not_an_array_is_a_provider_error(self):
        transport = _RecordingTransport(200, '{"assets": []}')
        client = AlpacaCryptoClient(
            api_key=API_KEY, api_secret=API_SECRET, transport=transport
        )

        with pytest.raises(AlpacaCryptoError):
            client.assets(base_url=PAPER_URL)

    def test_a_failed_read_never_returns_an_empty_registry(self):
        """An empty registry would read as "this venue trades nothing".

        That is a different and wrong claim, so the failure must stay a
        failure.
        """
        transport = _RecordingTransport(401, "unauthorized")
        client = AlpacaCryptoClient(
            api_key=API_KEY, api_secret=API_SECRET, transport=transport
        )

        with pytest.raises(ProviderError):
            result = client.assets(base_url=PAPER_URL)
            assert result != []

    def test_a_missing_credential_is_a_configuration_failure(self):
        client = AlpacaCryptoClient(api_key="", api_secret="")

        with pytest.raises(ProviderNotConfiguredError):
            client.assets(base_url=PAPER_URL)

    def test_the_failure_is_recorded_for_the_operator(self):
        transport = _RecordingTransport(403, "forbidden")
        client = AlpacaCryptoClient(
            api_key=API_KEY, api_secret=API_SECRET, transport=transport
        )

        with pytest.raises(ProviderError):
            client.assets(base_url=PAPER_URL)

        assert client.last_error is not None
        assert "403" in client.last_error


# ---------------------------------------------------------------------------
# 4. The credential never leaks
# ---------------------------------------------------------------------------


class TestTheCredentialNeverLeaks:
    """A key must not reach a log, a traceback or an operator's screen."""

    def _assert_no_secret(self, message: str) -> None:
        assert API_KEY not in message
        assert API_SECRET not in message

    @pytest.mark.parametrize(
        "status_code",
        [400, 401, 403, 404, 429, 500, 502, 503],
    )
    def test_no_status_code_leaks_the_credential(self, status_code):
        transport = _RecordingTransport(
            status_code, f"denied for {API_KEY} / {API_SECRET}"
        )
        client = AlpacaCryptoClient(
            api_key=API_KEY, api_secret=API_SECRET, transport=transport
        )

        with pytest.raises(ProviderError) as excinfo:
            client.assets(base_url=PAPER_URL)

        self._assert_no_secret(str(excinfo.value))
        self._assert_no_secret(client.last_error or "")

    def test_a_connection_failure_leaking_the_credential_is_redacted(self):
        client = AlpacaCryptoClient(
            api_key=API_KEY,
            api_secret=API_SECRET,
            transport=_BoomTransport(
                ConnectionError(f"failed with {API_KEY} and {API_SECRET}")
            ),
        )

        with pytest.raises(ProviderError) as excinfo:
            client.assets(base_url=PAPER_URL)

        self._assert_no_secret(str(excinfo.value))
        self._assert_no_secret(client.last_error or "")

    def test_a_json_failure_leaking_the_credential_is_redacted(self):
        transport = _RecordingTransport(200, f"not json: {API_SECRET}")
        client = AlpacaCryptoClient(
            api_key=API_KEY, api_secret=API_SECRET, transport=transport
        )

        with pytest.raises(ProviderError) as excinfo:
            client.assets(base_url=PAPER_URL)

        self._assert_no_secret(str(excinfo.value))
        self._assert_no_secret(client.last_error or "")

    def test_the_credential_is_not_in_the_successful_result(self, patched_get):
        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)

        assets = client.assets(base_url=PAPER_URL)

        rendered = json.dumps(assets)
        assert API_KEY not in rendered
        assert API_SECRET not in rendered

    def test_a_credential_of_any_plausible_length_is_always_removed(self):
        """The redaction floor must not become a way to leak a credential.

        Redaction only replaces a secret that is at least eight characters long,
        so a very short secret would survive into the message. That is a
        deliberate trade — a one-character secret matches ordinary words and
        would turn "unavailable" into "un***ilable" — but it is only safe
        because a real Alpaca key is far longer. This pins that: every length a
        real key could have is removed.
        """
        for key in ("12345678", "123456789", "a-realistic-alpaca-key"):
            transport = _RecordingTransport(500, f"denied for {key}")
            client = AlpacaCryptoClient(api_key=key, api_secret=key, transport=transport)

            with pytest.raises(ProviderError) as excinfo:
                client.assets(base_url=PAPER_URL)

            assert key not in str(excinfo.value)
            assert key not in (client.last_error or "")


# ---------------------------------------------------------------------------
# 5. The paper host, and only the paper host
# ---------------------------------------------------------------------------


class TestThePaperHostOnly:
    """The read is paper-only, and it says so before it reads anything."""

    def test_a_live_trading_host_is_refused(self):
        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)

        with pytest.raises(LiveEndpointRefusedError) as excinfo:
            client.assets(base_url="https://api.alpaca.markets")

        assert "live" in str(excinfo.value).lower()

    def test_an_unapproved_host_is_refused(self):
        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)

        with pytest.raises(LiveEndpointRefusedError):
            client.assets(base_url="https://paper-api.example.com")

    def test_a_plain_http_downgrade_is_refused(self):
        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)

        with pytest.raises(LiveEndpointRefusedError):
            client.assets(base_url="http://paper-api.alpaca.markets")

    def test_the_refusal_happens_before_the_request(self, patched_get):
        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)

        with pytest.raises(LiveEndpointRefusedError):
            client.assets(base_url="https://api.alpaca.markets")

        assert patched_get == [], "a request was made to a refused host"

    def test_the_read_places_no_order(self, patched_get):
        """A registry read is a read: it must not place an order."""
        client = AlpacaCryptoClient(api_key=API_KEY, api_secret=API_SECRET)

        client.assets(base_url=PAPER_URL)

        assert all(
            call["url"].endswith("/v2/assets") for call in patched_get
        ), "the read reached an endpoint other than the asset registry"
