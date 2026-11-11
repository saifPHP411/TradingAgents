"""The Bybit Demo *selection* layer: which broker, and when it may start.

The adapter's own behaviour is covered by ``test_bybit_demo_broker.py`` and
``test_bybit_demo_audit.py``. This file covers the seam **around** it: the
settings that choose the broker, the CLI and environment plumbing, and the
rule that a Bybit selection without credentials stops the run instead of
quietly trading somewhere else.

The rule under test throughout is the one the operator depends on:

- Alpaca Paper is the default and stays reachable exactly as before;
- ``bybit_demo`` is an explicit, opt-in choice;
- a Bybit choice never falls back to Alpaca (or to anything else);
- the Bybit Demo endpoint is the only endpoint the adapter may reach.
"""

from __future__ import annotations

import pytest

from tradingagents.paper_experiment.broker import (
    EXECUTION_PROVIDER_ALPACA_PAPER,
    EXECUTION_PROVIDER_BYBIT_DEMO,
    EXECUTION_PROVIDER_CHOICES,
)
from tradingagents.paper_experiment.broker.bybit_demo import (
    API_KEY_ENV,
    API_SECRET_ENV,
    DEMO_BASE_URL,
    LiveEndpointRefusedError,
)
from tradingagents.paper_experiment.broker.trading import (
    PaperTradingConfig,
    WorkerError,
)
from tradingagents.paper_experiment.market_data_wiring import (
    EXECUTION_PROVIDER_ENV,
    MarketDataWiringError,
    describe_execution_provider,
    env_execution_provider,
    resolve_execution_provider,
)

# ---------------------------------------------------------------------------
# The choice itself
# ---------------------------------------------------------------------------


class TestExecutionProviderChoice:
    def test_the_two_choices_are_exactly_alpaca_and_bybit_demo(self):
        assert set(EXECUTION_PROVIDER_CHOICES) == {"alpaca_paper", "bybit_demo"}

    def test_alpaca_paper_is_the_default_constant(self):
        assert EXECUTION_PROVIDER_ALPACA_PAPER == "alpaca_paper"
        assert EXECUTION_PROVIDER_BYBIT_DEMO == "bybit_demo"

    def test_resolving_alpaca_paper_returns_it(self):
        assert resolve_execution_provider("alpaca_paper") == "alpaca_paper"

    def test_resolving_bybit_demo_returns_it(self):
        assert resolve_execution_provider("bybit_demo") == "bybit_demo"

    def test_an_unknown_choice_is_refused_rather_than_defaulted(self):
        with pytest.raises(MarketDataWiringError, match="unknown execution provider"):
            resolve_execution_provider("binance_live")

    def test_a_live_sounding_name_is_refused(self):
        # There is no live provider, and a name that sounds like one must not
        # be resolved to something that trades.
        for name in ("bybit_live", "bybit", "bybit_demo_live", "alpaca_live"):
            with pytest.raises(MarketDataWiringError):
                resolve_execution_provider(name)

    def test_none_is_refused_so_the_caller_must_choose(self):
        with pytest.raises(MarketDataWiringError, match="no execution provider"):
            resolve_execution_provider(None)


# ---------------------------------------------------------------------------
# Alpaca Paper stays the default
# ---------------------------------------------------------------------------


class TestAlpacaRemainsTheDefault:
    def test_the_default_config_is_alpaca_paper(self):
        config = PaperTradingConfig()
        assert config.execution_provider == EXECUTION_PROVIDER_ALPACA_PAPER

    def test_the_default_config_keeps_the_alpaca_paper_endpoint(self):
        config = PaperTradingConfig()
        assert config.base_url == "https://paper-api.alpaca.markets"

    def test_the_environment_default_is_alpaca_paper(self, monkeypatch):
        monkeypatch.delenv(EXECUTION_PROVIDER_ENV, raising=False)
        assert env_execution_provider() == EXECUTION_PROVIDER_ALPACA_PAPER

    def test_an_empty_environment_variable_is_alpaca_paper(self, monkeypatch):
        monkeypatch.setenv(EXECUTION_PROVIDER_ENV, "")
        assert env_execution_provider() == EXECUTION_PROVIDER_ALPACA_PAPER

    def test_the_environment_can_select_bybit_demo(self, monkeypatch):
        monkeypatch.setenv(EXECUTION_PROVIDER_ENV, "bybit_demo")
        assert env_execution_provider() == "bybit_demo"

    def test_a_bad_environment_value_is_refused_not_defaulted(self, monkeypatch):
        monkeypatch.setenv(EXECUTION_PROVIDER_ENV, "bybit_live")
        with pytest.raises(MarketDataWiringError):
            env_execution_provider()


# ---------------------------------------------------------------------------
# The CLI flag
# ---------------------------------------------------------------------------


class TestCliFlag:
    def _parser(self):
        from tradingagents.paper_experiment.__main__ import _build_parser

        return _build_parser()

    def test_the_flag_defaults_to_unset_so_the_env_can_choose(self):
        args = self._parser().parse_args(["serve"])
        assert args.paper_execution_provider is None

    def test_the_flag_selects_bybit_demo(self):
        args = self._parser().parse_args(["serve", "--paper-execution-provider", "bybit_demo"])
        assert args.paper_execution_provider == "bybit_demo"

    def test_the_flag_selects_alpaca_paper(self):
        args = self._parser().parse_args(["serve", "--paper-execution-provider", "alpaca_paper"])
        assert args.paper_execution_provider == "alpaca_paper"

    def test_an_unknown_flag_value_is_refused_by_the_parser(self):
        with pytest.raises(SystemExit):
            self._parser().parse_args(["serve", "--paper-execution-provider", "bybit_live"])

    def test_the_server_resolver_defaults_to_alpaca_paper(self, monkeypatch):
        from tradingagents.paper_experiment.dashboard.server import (
            _resolve_execution_provider,
        )

        monkeypatch.delenv(EXECUTION_PROVIDER_ENV, raising=False)
        args = self._parser().parse_args(["serve"])
        assert _resolve_execution_provider(args) == EXECUTION_PROVIDER_ALPACA_PAPER

    def test_the_server_resolver_honours_the_flag(self, monkeypatch):
        from tradingagents.paper_experiment.dashboard.server import (
            _resolve_execution_provider,
        )

        monkeypatch.delenv(EXECUTION_PROVIDER_ENV, raising=False)
        args = self._parser().parse_args(["serve", "--paper-execution-provider", "bybit_demo"])
        assert _resolve_execution_provider(args) == "bybit_demo"

    def test_the_server_resolver_honours_the_environment(self, monkeypatch):
        from tradingagents.paper_experiment.dashboard.server import (
            _resolve_execution_provider,
        )

        monkeypatch.setenv(EXECUTION_PROVIDER_ENV, "bybit_demo")
        args = self._parser().parse_args(["serve"])
        assert _resolve_execution_provider(args) == "bybit_demo"

    def test_the_flag_wins_over_the_environment(self, monkeypatch):
        from tradingagents.paper_experiment.dashboard.server import (
            _resolve_execution_provider,
        )

        monkeypatch.setenv(EXECUTION_PROVIDER_ENV, "bybit_demo")
        args = self._parser().parse_args(["serve", "--paper-execution-provider", "alpaca_paper"])
        assert _resolve_execution_provider(args) == "alpaca_paper"

    def test_the_main_resolver_defaults_to_alpaca_paper(self, monkeypatch):
        from tradingagents.paper_experiment.__main__ import _paper_execution_provider

        monkeypatch.delenv(EXECUTION_PROVIDER_ENV, raising=False)
        args = self._parser().parse_args(["serve"])
        assert _paper_execution_provider(args) == EXECUTION_PROVIDER_ALPACA_PAPER

    def test_the_main_resolver_refuses_a_bad_value(self, monkeypatch):
        from tradingagents.paper_experiment.__main__ import RunnerError, _paper_execution_provider

        monkeypatch.setenv(EXECUTION_PROVIDER_ENV, "bybit_live")
        args = self._parser().parse_args(["serve"])
        with pytest.raises(RunnerError):
            _paper_execution_provider(args)


# ---------------------------------------------------------------------------
# The Bybit choice selects the demo endpoint, and only that endpoint
# ---------------------------------------------------------------------------


class TestBybitSelectsTheDemoEndpoint:
    def test_selecting_bybit_demo_selects_the_demo_base_url(self):
        config = PaperTradingConfig(execution_provider="bybit_demo")
        assert config.base_url == DEMO_BASE_URL

    def test_the_demo_base_url_is_the_official_demo_host(self):
        assert DEMO_BASE_URL == "https://api-demo.bybit.com"

    def test_the_live_host_is_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            PaperTradingConfig(
                execution_provider="bybit_demo",
                base_url="https://api.bybit.com",
            )

    def test_the_live_host_is_refused_even_with_a_path(self):
        with pytest.raises(LiveEndpointRefusedError):
            PaperTradingConfig(
                execution_provider="bybit_demo",
                base_url="https://api.bybit.com/v5/order/create",
            )

    def test_an_http_downgrade_is_refused(self):
        with pytest.raises(LiveEndpointRefusedError):
            PaperTradingConfig(
                execution_provider="bybit_demo",
                base_url="http://api-demo.bybit.com",
            )

    def test_an_unknown_provider_is_refused(self):
        with pytest.raises(WorkerError, match="unknown execution_provider"):
            PaperTradingConfig(execution_provider="kraken_demo")


# ---------------------------------------------------------------------------
# Missing credentials stop the run; they never switch brokers
# ---------------------------------------------------------------------------


class TestMissingCredentialsStopTheRun:
    def test_the_credential_variable_names_are_the_documented_ones(self):
        assert API_KEY_ENV == "BYBIT_DEMO_API_KEY"
        assert API_SECRET_ENV == "BYBIT_DEMO_API_SECRET"

    def test_a_bybit_broker_without_a_key_refuses_to_start(self, monkeypatch):
        from tradingagents.paper_experiment.broker.bybit_demo import (
            BrokerNotConfiguredError,
            BybitDemoBroker,
        )

        monkeypatch.delenv(API_KEY_ENV, raising=False)
        monkeypatch.delenv(API_SECRET_ENV, raising=False)
        with pytest.raises(BrokerNotConfiguredError, match="BYBIT_DEMO_API_KEY"):
            BybitDemoBroker()

    def test_a_bybit_broker_without_a_secret_refuses_to_start(self, monkeypatch):
        from tradingagents.paper_experiment.broker.bybit_demo import (
            BrokerNotConfiguredError,
            BybitDemoBroker,
        )

        monkeypatch.setenv(API_KEY_ENV, "demo-key")
        monkeypatch.delenv(API_SECRET_ENV, raising=False)
        with pytest.raises(BrokerNotConfiguredError, match="BYBIT_DEMO_API_SECRET"):
            BybitDemoBroker()

    def test_a_bybit_broker_with_both_credentials_starts(self, monkeypatch):
        from tradingagents.paper_experiment.broker.bybit_demo import BybitDemoBroker

        monkeypatch.setenv(API_KEY_ENV, "demo-key")
        monkeypatch.setenv(API_SECRET_ENV, "demo-secret")
        broker = BybitDemoBroker()
        assert broker.base_url == DEMO_BASE_URL

    def test_missing_credentials_do_not_change_the_selected_provider(self, monkeypatch):
        """The selection is a *choice*, not a hint the broker may override.

        A Bybit selection whose credentials are missing must not become an
        Alpaca selection: that would trade at a venue the operator did not
        choose, and would do it silently.
        """
        from tradingagents.paper_experiment.broker.trading import PaperTradingConfig

        monkeypatch.delenv(API_KEY_ENV, raising=False)
        monkeypatch.delenv(API_SECRET_ENV, raising=False)
        config = PaperTradingConfig(execution_provider="bybit_demo")
        assert config.execution_provider == "bybit_demo"
        assert config.base_url == DEMO_BASE_URL

    def test_alpaca_paper_does_not_need_bybit_credentials(self, monkeypatch):
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        monkeypatch.delenv(API_SECRET_ENV, raising=False)
        config = PaperTradingConfig()
        assert config.execution_provider == EXECUTION_PROVIDER_ALPACA_PAPER


# ---------------------------------------------------------------------------
# The start log names the broker
# ---------------------------------------------------------------------------


class TestTheStartLogNamesTheBroker:
    def test_the_alpaca_description_names_alpaca(self):
        text = describe_execution_provider(EXECUTION_PROVIDER_ALPACA_PAPER)
        assert "alpaca" in text.lower()

    def test_the_bybit_description_names_bybit_and_says_demo(self):
        text = describe_execution_provider(EXECUTION_PROVIDER_BYBIT_DEMO)
        lowered = text.lower()
        assert "bybit" in lowered
        assert "demo" in lowered

    def test_the_bybit_description_names_the_credential_variables(self):
        text = describe_execution_provider(EXECUTION_PROVIDER_BYBIT_DEMO)
        assert "BYBIT_DEMO_API_KEY" in text
        assert "BYBIT_DEMO_API_SECRET" in text


# ---------------------------------------------------------------------------
# The per-pair minimum report
# ---------------------------------------------------------------------------


class TestTheInstrumentsReport:
    """The report states what the venue enforces for each pair, pair by pair.

    The point of the report is that a reader can see a *real* value next to an
    absent one, so the report never prints a made-up default for a pair the
    venue does not describe.
    """

    def test_the_command_is_offered(self):
        from tradingagents.paper_experiment.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["bybit-instruments", "--pairs", "BTC/USDT", "ETH/USDT"])
        assert args.command == "bybit-instruments"
        assert args.pairs == ["BTC/USDT", "ETH/USDT"]

    def test_the_default_pairs_are_the_ones_under_review(self):
        from tradingagents.paper_experiment.__main__ import BYBIT_REPORT_PAIRS

        assert BYBIT_REPORT_PAIRS == (
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "DOGE/USDT",
            "AVAX/USDT",
            "LINK/USDT",
            "LTC/USDT",
            "BCH/USDT",
        )

    def test_the_report_reads_the_public_market_endpoint(self, monkeypatch, capsys):
        """The market endpoint is public: no credential is read and none needed."""
        import json

        from tradingagents.paper_experiment.__main__ import _bybit_instruments

        payload = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "status": "Trading",
                        "baseCoin": "BTC",
                        "quoteCoin": "USDT",
                        "lotSizeFilter": {
                            "basePrecision": "0.000001",
                            "minOrderQty": "0.00001",
                            "maxOrderQty": "63.01155123",
                            "minOrderAmt": "1",
                        },
                        "priceFilter": {"tickSize": "0.01"},
                    }
                ]
            },
        }
        body = json.dumps(payload)

        from tradingagents.crypto.transport import HttpResponse

        class FakeTransport:
            def __init__(self):
                self.calls = []

            def __call__(self, *, url, params, headers, timeout):
                self.calls.append((url, params))
                return HttpResponse(200, body)

        transport = FakeTransport()
        monkeypatch.setattr(
            "tradingagents.crypto.transport._default_transport",
            lambda **kwargs: transport(**kwargs),
        )

        class Args:
            pairs = ["BTC/USDT", "PEPE/USDT"]

        assert _bybit_instruments(Args()) == 0
        out = capsys.readouterr().out
        # The venue's own values are reported, in the plain decimal form.
        assert "BTCUSDT" in out
        assert "0.000001" in out
        # A pair the venue does not describe is named, not given a default.
        assert "PEPEUSDT" in out
        assert "NOT DESCRIBED BY THE VENUE" in out
