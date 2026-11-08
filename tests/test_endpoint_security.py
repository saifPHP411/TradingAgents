"""The Alpaca endpoint gate must accept one URL and refuse everything else.

The broker is the only component that can place an order, so its endpoint
check is the last line before a request leaves the process. A hostname-only
check is not enough: it accepts ``http://paper-api.alpaca.markets`` (a
cleartext downgrade), ``https://user:pass@paper-api.alpaca.markets``
(credentials in the URL) and ``https://paper-api.alpaca.markets:8080`` (a
different service on the same name).

These tests pin the whole-URL rule: exactly one URL is approved, and every
other spelling of it is refused.
"""

from __future__ import annotations

import pytest

from tradingagents.paper_experiment.broker import (
    PAPER_BASE_URL,
    AlpacaPaperBroker,
    LiveEndpointRefusedError,
    assert_paper_endpoint,
    assert_paper_request_url,
)

APPROVED = "https://paper-api.alpaca.markets"
APPROVED_HOST = "paper-api.alpaca.markets"


class TestTheApprovedEndpoint:
    def test_the_exact_approved_endpoint_is_accepted(self):
        assert assert_paper_endpoint(APPROVED) == APPROVED

    def test_the_approved_endpoint_is_the_one_alpaca_publishes(self):
        assert PAPER_BASE_URL == APPROVED

    def test_the_explicit_default_https_port_is_accepted(self):
        """``:443`` is the same endpoint written out; it is not a new service."""
        assert assert_paper_endpoint(f"{APPROVED}:443") == f"{APPROVED}:443"

    def test_the_hostname_is_matched_case_insensitively(self):
        """DNS hostnames are case-insensitive, so this is the same host."""
        assert assert_paper_endpoint("https://PAPER-API.ALPACA.MARKETS")

    def test_a_trailing_slash_is_the_same_endpoint(self):
        assert assert_paper_endpoint(f"{APPROVED}/") == f"{APPROVED}/"


class TestHttpIsRefused:
    @pytest.mark.parametrize(
        "url",
        [
            "http://paper-api.alpaca.markets",
            "http://paper-api.alpaca.markets/",
            "http://paper-api.alpaca.markets:80",
            "http://paper-api.alpaca.markets/v2/account",
        ],
    )
    def test_plain_http_is_refused(self, url):
        """A cleartext downgrade would send the credentials in the open."""
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint(url)

    def test_the_refusal_names_the_scheme(self):
        with pytest.raises(LiveEndpointRefusedError, match="https"):
            assert_paper_endpoint("http://paper-api.alpaca.markets")


class TestLiveHostsAreRefused:
    @pytest.mark.parametrize(
        "url",
        [
            "https://api.alpaca.markets",
            "https://alpaca.markets",
            "https://broker-api.alpaca.markets",
            "https://API.ALPACA.MARKETS",
            "http://api.alpaca.markets",
            "https://api.alpaca.markets:443",
        ],
    )
    def test_a_live_endpoint_is_refused(self, url):
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint(url)

    def test_the_live_refusal_says_so(self):
        """The operator must see that the URL was live, not merely unapproved."""
        with pytest.raises(LiveEndpointRefusedError, match="LIVE"):
            assert_paper_endpoint("https://api.alpaca.markets")

    def test_the_live_host_is_refused_before_the_credentials_are_read(self, monkeypatch):
        """No credential is read and no request is attempted for a live host."""
        import tradingagents.paper_experiment.broker.alpaca_paper as module

        read_credentials = []

        class _Env(dict):
            def get(self, key, default=None):  # noqa: ANN001
                read_credentials.append(key)
                return super().get(key, default)

        monkeypatch.setattr(module.os, "environ", _Env())
        with pytest.raises(LiveEndpointRefusedError):
            AlpacaPaperBroker(api_key=None, api_secret=None,
                              base_url="https://api.alpaca.markets")
        assert read_credentials == [], (
            f"the endpoint check must run before any credential is read, "
            f"but {read_credentials} was read"
        )


class TestUnexpectedHostsAreRefused:
    @pytest.mark.parametrize(
        "url",
        [
            "https://evil.example.com",
            "https://paper-api.example.com",
            "https://alpaca.markets.evil.com",
            "https://notpaper-api.alpaca.markets",
        ],
    )
    def test_an_unapproved_host_is_refused(self, url):
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint(url)


class TestUnexpectedPortsAreRefused:
    @pytest.mark.parametrize(
        "url",
        [
            "https://paper-api.alpaca.markets:80",
            "https://paper-api.alpaca.markets:4433",
            "https://paper-api.alpaca.markets:8080",
            "https://paper-api.alpaca.markets:8443",
            "https://paper-api.alpaca.markets:9999",
        ],
    )
    def test_a_port_other_than_443_is_refused(self, url):
        """A different port is a different service on the same host name."""
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint(url)

    def test_a_malformed_port_is_refused(self):
        """A port outside the valid range is a malformed URL, not a usable one."""
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint("https://paper-api.alpaca.markets:99999")


class TestUserinfoIsRefused:
    @pytest.mark.parametrize(
        "url",
        [
            "https://user:pass@paper-api.alpaca.markets",
            "https://user@paper-api.alpaca.markets",
            "https://:pass@paper-api.alpaca.markets",
            "https://user:pass@paper-api.alpaca.markets:443",
        ],
    )
    def test_credentials_in_the_url_are_refused(self, url):
        """A URL with userinfo can carry a different host than it appears to."""
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint(url)

    def test_the_refusal_names_the_userinfo(self):
        with pytest.raises(LiveEndpointRefusedError, match="username or password"):
            assert_paper_endpoint("https://user:pass@paper-api.alpaca.markets")


class TestSchemeLessUrlsAreRefused:
    @pytest.mark.parametrize(
        "url",
        [
            "//paper-api.alpaca.markets",
            "paper-api.alpaca.markets",
            "paper-api.alpaca.markets:443",
            "paper-api.alpaca.markets/v2/account",
        ],
    )
    def test_a_scheme_less_url_is_refused(self, url):
        """Without a scheme the URL does not say it is HTTPS."""
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint(url)

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://paper-api.alpaca.markets",
            "file://paper-api.alpaca.markets",
            "wss://paper-api.alpaca.markets",
            "gopher://paper-api.alpaca.markets",
        ],
    )
    def test_an_unsupported_scheme_is_refused(self, url):
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint(url)


class TestHostnameSuffixAttacksAreRefused:
    @pytest.mark.parametrize(
        "url",
        [
            "https://paper-api.alpaca.markets.evil.com",
            "https://paper-api.alpaca.markets.example.org",
            "https://evil.paper-api.alpaca.markets",
            "https://x.paper-api.alpaca.markets",
            "https://evilpaper-api.alpaca.markets",
            "https://paper-api.alpaca.markets.",
            "https://paper-api.alpaca.markets%2F.evil.com",
        ],
    )
    def test_a_suffix_or_subdomain_trick_is_refused(self, url):
        """A host that merely *contains* the approved name is a different host."""
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint(url)

    def test_a_backslash_host_confusion_is_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint("https://paper-api.alpaca.markets\\@evil.com")

    def test_the_approved_host_is_not_matched_by_suffix(self):
        """The approved hostname must not be a suffix of a longer name."""
        assert APPROVED_HOST != "paper-api.alpaca.markets.evil.com"


class TestMalformedUrlsAreRefused:
    @pytest.mark.parametrize(
        "url",
        [
            "",
            "   ",
            "https://[::1",
            "https://paper-api.alpaca.markets:not-a-port",
            "://paper-api.alpaca.markets",
            "ht!tp://paper-api.alpaca.markets",
        ],
    )
    def test_a_malformed_url_is_refused(self, url):
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint(url)

    def test_a_non_string_is_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "url",
        [
            " https://paper-api.alpaca.markets",
            "https://paper-api.alpaca.markets ",
            "\thttps://paper-api.alpaca.markets",
            "https://paper-api.alpaca.markets\n",
        ],
    )
    def test_surrounding_whitespace_is_refused(self, url):
        """Silently trimming would let a differently-written string through."""
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint(url)


class TestTheRequestUrlGuard:
    """A request URL carries a path; the base endpoint URL must not.

    The base check and the request check are different questions, so they are
    different functions. This pins that the request guard still refuses every
    host trick.
    """

    def test_a_request_url_on_the_approved_host_is_accepted(self):
        url = "https://paper-api.alpaca.markets/v2/account"
        assert assert_paper_request_url(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "http://paper-api.alpaca.markets/v2/account",
            "https://api.alpaca.markets/v2/account",
            "https://evil.example.com/v2/account",
            "https://paper-api.alpaca.markets:8080/v2/account",
            "https://user:pass@paper-api.alpaca.markets/v2/account",
            "https://paper-api.alpaca.markets.evil.com/v2/account",
        ],
    )
    def test_a_request_url_off_the_approved_host_is_refused(self, url):
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_request_url(url)

    def test_the_base_check_still_refuses_a_path(self):
        """The base URL must be exactly the endpoint, with no path on it."""
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint("https://paper-api.alpaca.markets/v2/account")


class TestEveryBrokerRequestGoesThroughTheGate:
    """The gate must be on the request path, not only on the constructor."""

    def test_a_request_to_another_host_is_refused_before_it_is_sent(self):
        """A broker whose stored base URL was tampered with still cannot send."""
        sent: list[str] = []

        def _transport(*, method, url, headers, params, json_body, timeout):
            sent.append(url)
            raise AssertionError("no request may be sent")

        broker = AlpacaPaperBroker(
            api_key="k", api_secret="s", transport=_transport
        )
        # Tamper with the stored base URL the way a bad configuration would.
        broker.base_url = "https://api.alpaca.markets"
        with pytest.raises(LiveEndpointRefusedError):
            broker._request("GET", "/v2/account")
        assert sent == [], f"a request was sent: {sent}"

    def test_the_broker_stores_the_approved_endpoint(self):
        broker = AlpacaPaperBroker(api_key="k", api_secret="s")
        assert broker.base_url == APPROVED

