"""Phase 26: the documented entrypoints actually wire the selected price seam.

The audit finding this phase fixes: the documented Railway ``serve`` entrypoint
did not pass a ``market_data_factory``, so the paper worker silently used the
historical ``SourceMarketData`` (a past daily-bar open) no matter what the
operator asked for. ``AlpacaSnapshotMarketData`` was reachable only from the
undocumented module form.

These tests drive the **real entrypoints** — the ``serve`` subcommand's argument
parser and its call into ``serve`` — not just the isolated classes, and pin:

- both entrypoints expose ``--paper-market-data`` / ``--paper-fractional-shares``
  and read their environment variables, with CLI taking precedence;
- ``_serve`` passes the selected factory, provider label and whole-shares setting
  into the dashboard, and ``alpaca_snapshot`` builds a real
  ``AlpacaSnapshotMarketData``;
- snapshot mode never falls back to ``SourceMarketData``;
- a missing credential fails the worker build rather than papering over it;
- a stale or failing snapshot never submits an order;
- fractional sizing is opt-in and still respects the $25 notional cap;
- historical ``bar_open`` mode is unchanged, and the paper-only / long-only
  restrictions are intact.

Every test is offline: the snapshot transport is mocked and the experiment uses a
local static price source. No order is ever sent anywhere.
"""

from __future__ import annotations

import json
import socket
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tradingagents.paper_experiment import ExperimentConfig, PaperExperiment
from tradingagents.paper_experiment.broker import (
    DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
    AlpacaSnapshotMarketData,
    BrokerAccount,
    MarketDataError,
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    PreflightError,
    SourceMarketData,
    StaticMarketData,
    TradingDecision,
)
from tradingagents.paper_experiment.dashboard.paper_control import PaperTradingController
from tradingagents.paper_experiment.inference import StubInference
from tradingagents.paper_experiment.market_data_wiring import (
    FRACTIONAL_SHARES_ENV,
    MARKET_DATA_ENV,
    SNAPSHOT_MAX_AGE_ENV,
    MarketDataWiringError,
    env_fractional,
    env_max_age,
    env_provider,
    market_data_factory_for,
    parse_bool,
    resolve_max_age,
    resolve_provider,
)
from tradingagents.paper_experiment.providers.alpaca_iex import (
    API_KEY_ENV,
    API_SECRET_ENV,
)
from tradingagents.paper_experiment.providers.base import HttpResponse
from tradingagents.paper_experiment.sources import StaticPriceSource
from tradingagents.risk import WHOLE_SHARES

SYMBOL = "SPY"
SPY_PRICE = 580.0
NOW = datetime(2026, 10, 30, 20, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _frame(periods: int = 12, *, start: float = 100.0, step: float = 1.0):
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    opens = [start + step * i for i in range(periods)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 2.0 for o in opens],
            "low": [o - 1.0 for o in opens],
            "close": [o + 1.0 for o in opens],
            "volume": [1e6] * periods,
        },
        index=index,
    )


def _experiment_root(tmp_path, *, periods: int = 12, processed: int = 1):
    root = tmp_path / "exp"
    config = ExperimentConfig.from_protocol(
        symbols=(SYMBOL,), decision_frequency_sessions=1, evaluation_lag_sessions=2
    )
    source = StaticPriceSource({SYMBOL: _frame(periods)}, name="static_test")
    experiment = PaperExperiment(
        source=source, config=config, root=root, inference=StubInference()
    )
    experiment.ensure_created()
    for session in list(source.sessions())[:processed]:
        experiment.process_session(session)
    return root


def _stamp(offset_seconds: float = 0.0) -> str:
    return (NOW + timedelta(seconds=offset_seconds)).isoformat().replace("+00:00", "Z")


def _snapshot_body(*, price: float = SPY_PRICE, stamp: str | None = None):
    return {SYMBOL: {"latestTrade": {"p": price, "t": stamp if stamp is not None else _stamp()}}}


def _transport(body, *, status_code: int = 200):
    def send(*, url, params, headers, timeout):
        text = body if isinstance(body, str) else json.dumps(body)
        return HttpResponse(status_code, text)

    return send


def _snapshot_market_data(body, *, status_code: int = 200, now=NOW, **kwargs):
    return AlpacaSnapshotMarketData(
        (SYMBOL,),
        api_key="test-key-not-real",
        api_secret="test-secret-not-real",
        transport=_transport(body, status_code=status_code),
        clock=lambda: now,
        **kwargs,
    )


def _set_credentials(monkeypatch, *, present: bool = True):
    if present:
        monkeypatch.setenv(API_KEY_ENV, "test-key-not-real")
        monkeypatch.setenv(API_SECRET_ENV, "test-secret-not-real")
    else:
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        monkeypatch.delenv(API_SECRET_ENV, raising=False)


class _FailIfCalledBroker:
    """A broker that fails the test if any order path is reached."""

    def submit_order(self, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("an order was submitted when it should not have been")

    def _reconcile(self, client_order_id):  # pragma: no cover
        raise AssertionError("no reconcile should happen")

    def positions(self, refresh=False):  # pragma: no cover
        return {}

    def connect(self):  # pragma: no cover
        raise AssertionError("connect should not be reached")


class _OneDecision:
    def __init__(self, *, notional: float = 25.0, quantity: float | None = None):
        self._done = False
        self._notional = notional
        self._quantity = quantity

    def next_decision(self):
        if self._done:
            return None
        self._done = True
        return TradingDecision(
            decision_id="d1",
            symbol=SYMBOL,
            side="buy",
            notional=None if self._quantity is not None else self._notional,
            quantity=self._quantity,
        )


# ---------------------------------------------------------------------------
# 1. Both entrypoints expose the options
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntrypointOptions:
    def test_serve_help_exposes_the_new_options(self):
        from tradingagents.paper_experiment.__main__ import _build_parser

        help_text = _build_parser().format_help()
        assert "--paper-market-data" in help_text
        assert "--paper-fractional-shares" in help_text
        assert "--paper-snapshot-max-age-seconds" in help_text
        assert "bar_open" in help_text and "alpaca_snapshot" in help_text

    def test_paper_trade_help_exposes_the_new_options(self):
        from tradingagents.paper_experiment.__main__ import _build_parser

        help_text = _build_parser().format_help()
        # The same parser backs every subcommand, so paper-trade sees them too.
        assert "--paper-market-data" in help_text

    def test_module_form_help_exposes_the_new_options(self):
        import io
        from contextlib import redirect_stdout

        from tradingagents.paper_experiment.dashboard import server

        captured = io.StringIO()
        with pytest.raises(SystemExit), redirect_stdout(captured):
            server.main(["--help"])
        help_text = captured.getvalue()
        assert "--paper-market-data" in help_text
        assert "--paper-fractional-shares" in help_text
        assert "--paper-snapshot-max-age-seconds" in help_text

    def test_serve_defaults_to_bar_open(self, tmp_path, monkeypatch):
        from tradingagents.paper_experiment import __main__ as cli

        monkeypatch.delenv(MARKET_DATA_ENV, raising=False)
        captured = _capture_serve(monkeypatch)
        assert cli.main(["serve", "--root", str(tmp_path / "exp")]) == 0
        assert captured["paper_market_data_provider"] == "bar_open"
        assert captured["market_data_factory"] is None

    def test_serve_reads_the_provider_env_var(self, tmp_path, monkeypatch):
        from tradingagents.paper_experiment import __main__ as cli

        _set_credentials(monkeypatch)
        monkeypatch.setenv(MARKET_DATA_ENV, "alpaca_snapshot")
        captured = _capture_serve(monkeypatch)
        assert cli.main(["serve", "--root", str(tmp_path / "exp")]) == 0
        assert captured["paper_market_data_provider"] == "alpaca_snapshot"
        assert callable(captured["market_data_factory"])

    def test_cli_flag_overrides_the_env_var(self, tmp_path, monkeypatch):
        from tradingagents.paper_experiment import __main__ as cli

        monkeypatch.setenv(MARKET_DATA_ENV, "alpaca_snapshot")
        captured = _capture_serve(monkeypatch)
        assert (
            cli.main(
                ["serve", "--root", str(tmp_path / "exp"), "--paper-market-data", "bar_open"]
            )
            == 0
        )
        assert captured["paper_market_data_provider"] == "bar_open"
        assert captured["market_data_factory"] is None


def _capture_serve(monkeypatch):
    """Monkeypatch the dashboard's ``serve`` and capture the kwargs ``_serve`` passes.

    ``_serve`` imports ``serve`` from the ``dashboard`` package at call time, so
    patching the attribute on that package intercepts the real call without a
    socket ever being bound.
    """
    captured: dict = {}

    def fake_serve(root_arg, **kwargs):
        captured.update(kwargs)
        captured["root"] = str(root_arg)

    import tradingagents.paper_experiment.dashboard as dashboard_pkg

    monkeypatch.setattr(dashboard_pkg, "serve", fake_serve)
    return captured


# ---------------------------------------------------------------------------
# 2. Environment parsing and precedence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnvironmentParsing:
    def test_provider_defaults_to_bar_open(self, monkeypatch):
        monkeypatch.delenv(MARKET_DATA_ENV, raising=False)
        assert env_provider() == "bar_open"

    def test_provider_reads_the_env_var(self, monkeypatch):
        monkeypatch.setenv(MARKET_DATA_ENV, "alpaca_snapshot")
        assert env_provider() == "alpaca_snapshot"

    def test_an_unknown_provider_is_refused_not_defaulted(self, monkeypatch):
        monkeypatch.setenv(MARKET_DATA_ENV, "live_alpaca")
        with pytest.raises(MarketDataWiringError):
            env_provider()

    def test_resolve_provider_rejects_unknown_names(self):
        with pytest.raises(MarketDataWiringError):
            resolve_provider("live")
        with pytest.raises(MarketDataWiringError):
            resolve_provider(None)

    @pytest.mark.parametrize("value,expected", [
        ("1", True), ("true", True), ("yes", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("off", False), ("", False),
    ])
    def test_boolean_parsing(self, value, expected):
        assert parse_bool(value, default=not expected) is expected

    def test_a_bad_boolean_is_refused(self):
        with pytest.raises(MarketDataWiringError):
            parse_bool("maybe", default=False)

    def test_fractional_defaults_to_the_parity_constant(self, monkeypatch):
        from tradingagents.risk import FRACTIONAL_SHARES

        monkeypatch.delenv(FRACTIONAL_SHARES_ENV, raising=False)
        assert env_fractional() is FRACTIONAL_SHARES

    def test_fractional_reads_the_env_var(self, monkeypatch):
        monkeypatch.setenv(FRACTIONAL_SHARES_ENV, "yes")
        assert env_fractional() is True

    def test_max_age_defaults_to_the_conservative_value(self, monkeypatch):
        monkeypatch.delenv(SNAPSHOT_MAX_AGE_ENV, raising=False)
        assert env_max_age() == DEFAULT_MAX_SNAPSHOT_AGE_SECONDS == 300.0

    def test_max_age_reads_the_env_var(self, monkeypatch):
        monkeypatch.setenv(SNAPSHOT_MAX_AGE_ENV, "45")
        assert env_max_age() == 45.0

    def test_max_age_can_be_disabled_explicitly(self):
        assert resolve_max_age("none") is None

    def test_a_bad_max_age_is_refused(self):
        with pytest.raises(MarketDataWiringError):
            resolve_max_age("soon")
        with pytest.raises(MarketDataWiringError):
            resolve_max_age("-1")


# ---------------------------------------------------------------------------
# 3. The factory builds the right seam
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFactorySelection:
    def test_bar_open_selects_no_factory(self):
        assert market_data_factory_for("bar_open") is None

    def test_alpaca_snapshot_builds_a_snapshot_source(self, monkeypatch):
        _set_credentials(monkeypatch)
        factory = market_data_factory_for("alpaca_snapshot", max_age_seconds=90.0)
        assert factory is not None
        market = factory((SYMBOL,))
        assert isinstance(market, AlpacaSnapshotMarketData)
        assert market.max_age_seconds == 90.0

    def test_the_factory_defaults_to_the_conservative_age(self, monkeypatch):
        _set_credentials(monkeypatch)
        market = market_data_factory_for("alpaca_snapshot")((SYMBOL,))
        assert market.max_age_seconds == DEFAULT_MAX_SNAPSHOT_AGE_SECONDS

    def test_an_unknown_choice_is_refused(self):
        with pytest.raises(MarketDataWiringError):
            market_data_factory_for("something_else")


# ---------------------------------------------------------------------------
# 4. _serve passes the factory into the controller (end-to-end wiring)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestServeWiring:
    def test_snapshot_mode_passes_a_factory_that_builds_snapshot_market_data(
        self, tmp_path, monkeypatch
    ):
        _set_credentials(monkeypatch)
        from tradingagents.paper_experiment import __main__ as cli

        captured = _capture_serve(monkeypatch)

        assert (
            cli.main(
                ["serve", "--root", str(tmp_path / "exp"),
                 "--paper-market-data", "alpaca_snapshot"]
            )
            == 0
        )
        factory = captured["market_data_factory"]
        assert callable(factory)
        assert isinstance(factory((SYMBOL,)), AlpacaSnapshotMarketData)

    def test_serve_passes_the_parity_sizing_by_default(self, tmp_path, monkeypatch):
        from tradingagents.paper_experiment import __main__ as cli
        from tradingagents.risk import WHOLE_SHARES

        captured = _capture_serve(monkeypatch)
        assert cli.main(["serve", "--root", str(tmp_path / "exp")]) == 0
        assert captured["paper_whole_shares"] is WHOLE_SHARES

    def test_serve_enables_fractional_only_when_asked(self, tmp_path, monkeypatch):
        from tradingagents.paper_experiment import __main__ as cli

        captured = _capture_serve(monkeypatch)
        assert (
            cli.main(["serve", "--root", str(tmp_path / "exp"), "--paper-fractional-shares"])
            == 0
        )
        assert captured["paper_whole_shares"] is False

    def test_fractional_env_var_is_honoured(self, tmp_path, monkeypatch):
        from tradingagents.paper_experiment import __main__ as cli

        monkeypatch.setenv(FRACTIONAL_SHARES_ENV, "true")
        captured = _capture_serve(monkeypatch)
        assert cli.main(["serve", "--root", str(tmp_path / "exp")]) == 0
        assert captured["paper_whole_shares"] is False

    def test_controller_receives_the_configured_provider(self, tmp_path, monkeypatch):
        """The controller's config names the provider, so status is unambiguous."""
        _set_credentials(monkeypatch)
        root = _experiment_root(tmp_path)
        controller = PaperTradingController(
            root,
            enabled=True,
            market_data_factory=market_data_factory_for("alpaca_snapshot"),
            market_data_provider="alpaca_snapshot",
            market_data_max_age_seconds=120.0,
        )
        config = controller._worker_config()  # noqa: SLF001 - the seam under test
        assert config.market_data_provider == "alpaca_snapshot"
        assert config.market_data_max_age_seconds == 120.0

    def test_module_form_main_passes_the_provider_into_serve(self, tmp_path, monkeypatch):
        """The module-form entrypoint wires the provider too, not only the subcommand."""
        from tradingagents.paper_experiment.dashboard import server

        captured: dict = {}

        def fake_serve(root_arg, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(server, "serve", fake_serve)
        monkeypatch.setenv(MARKET_DATA_ENV, "alpaca_snapshot")
        _set_credentials(monkeypatch)
        root = _experiment_root(tmp_path)
        assert server.main(["--root", str(root), "--host", "127.0.0.1"]) == 0
        assert captured["paper_market_data_provider"] == "alpaca_snapshot"
        assert callable(captured["market_data_factory"])
        assert captured["paper_whole_shares"] is WHOLE_SHARES

    def test_module_form_main_honours_fractional_env(self, tmp_path, monkeypatch):
        from tradingagents.paper_experiment.dashboard import server

        captured: dict = {}

        def fake_serve(root_arg, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(server, "serve", fake_serve)
        monkeypatch.setenv(FRACTIONAL_SHARES_ENV, "1")
        root = _experiment_root(tmp_path)
        assert server.main(["--root", str(root), "--host", "127.0.0.1"]) == 0
        assert captured["paper_whole_shares"] is False

    def test_an_invalid_provider_is_a_clean_error_not_a_fallback(
        self, tmp_path, monkeypatch, capsys
    ):
        from tradingagents.paper_experiment import __main__ as cli

        monkeypatch.setenv(MARKET_DATA_ENV, "live_alpaca")
        # A bad provider is a clean non-zero exit with a message; serve is never
        # reached with a default, so no price seam is silently substituted.
        assert cli.main(["serve", "--root", str(tmp_path / "exp")]) == 2
        assert "unknown market-data provider" in capsys.readouterr().err

    def test_paper_trade_passes_the_same_factory(self, tmp_path, monkeypatch):
        """The CLI paper-trade path selects the identical seam as serve."""
        from tradingagents.paper_experiment import __main__ as cli

        _set_credentials(monkeypatch)
        seen: dict = {}
        real_controller = PaperTradingController

        def spy_controller(root, **kwargs):
            seen.update(kwargs)
            return real_controller(root, **kwargs)

        import tradingagents.paper_experiment.dashboard.paper_control as pc

        monkeypatch.setattr(pc, "PaperTradingController", spy_controller)
        # --dry-run builds the worker (and thus the seam) but places no order.
        cli.main(
            ["paper-trade", "--root", str(_experiment_root(tmp_path)),
             "--paper-market-data", "alpaca_snapshot", "--dry-run"]
        )
        assert seen["market_data_provider"] == "alpaca_snapshot"
        assert callable(seen["market_data_factory"])


# ---------------------------------------------------------------------------
# 5. No fallback: snapshot mode never degrades to historical prices
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoSilentFallback:
    def test_snapshot_mode_does_not_build_source_market_data(self, tmp_path, monkeypatch):
        _set_credentials(monkeypatch)
        root = _experiment_root(tmp_path)
        controller = PaperTradingController(
            root,
            enabled=True,
            market_data_factory=market_data_factory_for("alpaca_snapshot"),
            market_data_provider="alpaca_snapshot",
        )
        experiment = PaperExperiment.load(root, inference=None)
        source = experiment.source
        sessions = source.sessions()
        market = controller._build_market_data(experiment, source, sessions)  # noqa: SLF001
        assert isinstance(market, AlpacaSnapshotMarketData)
        assert not isinstance(market, SourceMarketData)

    def test_a_missing_credential_fails_the_build_rather_than_falling_back(
        self, tmp_path, monkeypatch
    ):
        _set_credentials(monkeypatch, present=False)
        root = _experiment_root(tmp_path)
        controller = PaperTradingController(
            root,
            enabled=True,
            market_data_factory=market_data_factory_for("alpaca_snapshot"),
            market_data_provider="alpaca_snapshot",
        )
        with pytest.raises(MarketDataError):
            controller._ensure_worker()  # noqa: SLF001 - the seam under test

    def test_bar_open_still_builds_source_market_data(self, tmp_path, monkeypatch):
        """The historical default is preserved unchanged."""
        root = _experiment_root(tmp_path)
        controller = PaperTradingController(root, enabled=True)
        experiment = PaperExperiment.load(root, inference=None)
        source = experiment.source
        sessions = source.sessions()
        market = controller._build_market_data(experiment, source, sessions)  # noqa: SLF001
        assert isinstance(market, SourceMarketData)
        assert market.kind == "historical"
        assert market.price(SYMBOL) > 0


# ---------------------------------------------------------------------------
# 6. Snapshot failures never submit an order
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSnapshotFailuresNeverOrder:
    def _worker(self, tmp_path, market_data):
        return PaperTradingWorker(
            config=PaperTradingConfig(
                order_notional=25.0,
                limits=PaperRiskLimits(),
                state_path=tmp_path / "worker_state.json",
                market_data_provider="alpaca_snapshot",
            ),
            decision_source=_OneDecision(),
            market_data=market_data,
            broker_factory=lambda: _FailIfCalledBroker(),
        )

    def _run(self, tmp_path, market_data):
        worker = self._worker(tmp_path, market_data)
        worker.state.status = "RUNNING"
        worker._broker = _FailIfCalledBroker()  # noqa: SLF001
        return worker.run_once()

    def test_a_stale_snapshot_never_submits(self, tmp_path):
        md = _snapshot_market_data(_snapshot_body(stamp=_stamp(-400)))
        result = self._run(tmp_path, md)
        assert result.ran is False
        assert result.skipped_reason == "market data was unavailable"
        assert result.market_data_provider == "alpaca_snapshot"

    def test_a_missing_timestamp_never_submits(self, tmp_path):
        body = {SYMBOL: {"latestTrade": {"p": SPY_PRICE}}}
        result = self._run(tmp_path, _snapshot_market_data(body))
        assert result.ran is False

    def test_a_malformed_timestamp_never_submits(self, tmp_path):
        result = self._run(tmp_path, _snapshot_market_data(_snapshot_body(stamp="not-a-time")))
        assert result.ran is False

    def test_a_future_timestamp_never_submits(self, tmp_path):
        result = self._run(tmp_path, _snapshot_market_data(_snapshot_body(stamp=_stamp(600))))
        assert result.ran is False

    def test_a_source_error_never_submits(self, tmp_path):
        md = _snapshot_market_data("boom", status_code=500)
        result = self._run(tmp_path, md)
        assert result.ran is False

    def test_a_fresh_snapshot_does_size(self, tmp_path):
        """The positive control: a fresh snapshot is usable (sizing only, no order)."""
        md = _snapshot_market_data(_snapshot_body())
        assert md.price(SYMBOL) == pytest.approx(SPY_PRICE)


# ---------------------------------------------------------------------------
# 7d. Preflight accepts a fresh field even when a peer field is stale
# ---------------------------------------------------------------------------


class _PreflightBroker:
    """A broker that only answers the read-only preflight calls."""

    def connect(self):
        return BrokerAccount(
            account_id="paper-acct-1",
            status="ACTIVE",
            cash=1000.0,
            equity=1000.0,
            buying_power=1000.0,
            currency="USD",
            trading_blocked=False,
            account_blocked=False,
        )

    def positions(self, refresh=False):
        return {}

    def submit_order(self, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("preflight must not submit an order")


def _preflight_worker(tmp_path, market_data):
    return PaperTradingWorker(
        config=PaperTradingConfig(
            order_notional=25.0,
            limits=PaperRiskLimits(),
            state_path=tmp_path / "worker_state.json",
            market_data_provider="alpaca_snapshot",
        ),
        decision_source=_OneDecision(),
        market_data=market_data,
        broker_factory=lambda: _PreflightBroker(),
    )


@pytest.mark.unit
class TestPreflightAcceptsAFreshField:
    def test_a_fresh_quote_passes_preflight_despite_a_stale_trade(self, tmp_path):
        body = {
            SYMBOL: {
                "latestTrade": {"p": SPY_PRICE, "t": _stamp(-400)},
                "latestQuote": {"bp": 99.0, "ap": 101.0, "t": _stamp(-1)},
            }
        }
        worker = _preflight_worker(tmp_path, _snapshot_market_data(body))
        result = worker.preflight()
        assert {"check": "market_data", "ok": True, "detail": "alpaca_snapshot"} in result["checks"]

    def test_all_fields_stale_fails_preflight(self, tmp_path):
        body = {
            SYMBOL: {
                "latestTrade": {"p": SPY_PRICE, "t": _stamp(-400)},
                "latestQuote": {"bp": 99.0, "ap": 101.0, "t": _stamp(-500)},
                "minuteBar": {"c": 123.0, "t": _stamp(-600)},
            }
        }
        worker = _preflight_worker(tmp_path, _snapshot_market_data(body))
        with pytest.raises(PreflightError) as excinfo:
            worker.preflight()
        assert excinfo.value.check == "market_data"
        assert "stale" in excinfo.value.detail


# ---------------------------------------------------------------------------
# 7. Freshness semantics on the seam itself
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSnapshotFreshness:
    def test_a_fresh_snapshot_is_accepted(self):
        md = _snapshot_market_data(_snapshot_body(stamp=_stamp(-10)))
        assert md.price(SYMBOL) == pytest.approx(SPY_PRICE)
        assert md.last_status == "ok"
        assert md.last_observation.as_of is not None

    def test_a_stale_snapshot_is_refused(self):
        md = _snapshot_market_data(_snapshot_body(stamp=_stamp(-400)))
        with pytest.raises(MarketDataError, match="stale"):
            md.price(SYMBOL)
        assert md.last_status == "stale"

    def test_the_boundary_is_configurable(self):
        md = _snapshot_market_data(_snapshot_body(stamp=_stamp(-400)), max_age_seconds=600.0)
        assert md.price(SYMBOL) == pytest.approx(SPY_PRICE)

    def test_a_missing_timestamp_is_refused(self):
        md = _snapshot_market_data({SYMBOL: {"latestTrade": {"p": SPY_PRICE}}})
        with pytest.raises(MarketDataError, match="no timestamp"):
            md.price(SYMBOL)

    def test_a_malformed_timestamp_is_refused(self):
        md = _snapshot_market_data(_snapshot_body(stamp="yesterday"))
        with pytest.raises(MarketDataError, match="malformed"):
            md.price(SYMBOL)

    def test_a_future_timestamp_is_refused(self):
        md = _snapshot_market_data(_snapshot_body(stamp=_stamp(600)))
        with pytest.raises(MarketDataError, match="future"):
            md.price(SYMBOL)

    def test_a_small_future_skew_is_tolerated(self):
        md = _snapshot_market_data(_snapshot_body(stamp=_stamp(2)))
        assert md.price(SYMBOL) == pytest.approx(SPY_PRICE)

    def test_the_quote_timestamp_pairs_with_the_quote_price(self):
        """When the quote is the selected field, its own timestamp is judged."""
        body = {
            SYMBOL: {
                "latestTrade": None,
                "latestQuote": {"bp": 99.0, "ap": 101.0, "t": _stamp(-1)},
            }
        }
        md = _snapshot_market_data(body)
        assert md.price(SYMBOL) == pytest.approx(100.0)
        assert md.last_observation.field == "latestQuote"

    def test_a_stale_trade_no_longer_blocks_a_fresh_quote(self):
        """Replaces the old ``test_a_stale_preferred_field_is_refused_not_skipped``.

        That test asserted a stale ``latestTrade`` was refused outright, without
        ever considering a fresher ``latestQuote``. That rule is gone: the newest
        verifiable observation is now selected and *then* freshness is enforced.
        The security property it protected — a stale price is never used — is
        retained by
        ``TestNewestObservationSelection::test_all_fields_stale_is_still_refused``
        and ``::test_the_freshest_field_is_still_subject_to_the_freshness_limit``.
        """
        body = {
            SYMBOL: {
                "latestTrade": {"p": SPY_PRICE, "t": _stamp(-400)},
                "latestQuote": {"bp": 99.0, "ap": 101.0, "t": _stamp(-1)},
            }
        }
        md = _snapshot_market_data(body, max_age_seconds=300.0)
        assert md.price(SYMBOL) == pytest.approx(100.0)
        assert md.last_observation.field == "latestQuote"

    def test_available_reports_false_for_a_stale_snapshot(self):
        md = _snapshot_market_data(_snapshot_body(stamp=_stamp(-400)))
        assert md.available() is False
        assert md.last_status == "stale"


# ---------------------------------------------------------------------------
# 7b. Selection is by newest verifiable timestamp, not field order
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNewestObservationSelection:
    """The newest verifiable current observation wins; freshness is then enforced.

    A symbol can go minutes without printing on the IEX feed while its quote keeps
    updating, so a trade-first order would refuse a perfectly fresh market. These
    tests pin that a stale trade no longer masks a fresh quote or bar, that an
    unverifiable field cannot outrank a verifiable one, that ties stay
    deterministic, and that the freshness limit itself is unchanged.
    """

    def test_a_stale_trade_yields_to_a_fresh_quote(self):
        body = {
            SYMBOL: {
                "latestTrade": {"p": SPY_PRICE, "t": _stamp(-400)},
                "latestQuote": {"bp": 99.0, "ap": 101.0, "t": _stamp(-1)},
            }
        }
        md = _snapshot_market_data(body)
        assert md.price(SYMBOL) == pytest.approx(100.0)
        assert md.last_observation.field == "latestQuote"
        assert md.last_status == "ok"

    def test_a_stale_trade_yields_to_a_fresh_minute_bar(self):
        body = {
            SYMBOL: {
                "latestTrade": {"p": SPY_PRICE, "t": _stamp(-400)},
                "minuteBar": {"c": 123.0, "t": _stamp(-5)},
            }
        }
        md = _snapshot_market_data(body)
        assert md.price(SYMBOL) == pytest.approx(123.0)
        assert md.last_observation.field == "minuteBar"

    def test_the_newer_of_a_quote_and_a_bar_is_chosen(self):
        newer_bar = {
            SYMBOL: {
                "latestQuote": {"bp": 99.0, "ap": 101.0, "t": _stamp(-30)},
                "minuteBar": {"c": 123.0, "t": _stamp(-1)},
            }
        }
        md = _snapshot_market_data(newer_bar)
        assert md.price(SYMBOL) == pytest.approx(123.0)
        assert md.last_observation.field == "minuteBar"

        newer_quote = {
            SYMBOL: {
                "latestQuote": {"bp": 99.0, "ap": 101.0, "t": _stamp(-1)},
                "minuteBar": {"c": 123.0, "t": _stamp(-30)},
            }
        }
        md = _snapshot_market_data(newer_quote)
        assert md.price(SYMBOL) == pytest.approx(100.0)
        assert md.last_observation.field == "latestQuote"

    def test_all_fields_stale_is_still_refused(self):
        """The safety property: a genuinely stale market still yields no price."""
        body = {
            SYMBOL: {
                "latestTrade": {"p": SPY_PRICE, "t": _stamp(-400)},
                "latestQuote": {"bp": 99.0, "ap": 101.0, "t": _stamp(-500)},
                "minuteBar": {"c": 123.0, "t": _stamp(-600)},
            }
        }
        md = _snapshot_market_data(body)
        with pytest.raises(MarketDataError, match="stale"):
            md.price(SYMBOL)
        assert md.last_status == "stale"

    def test_a_fresh_trade_beats_a_stale_quote(self):
        body = {
            SYMBOL: {
                "latestTrade": {"p": SPY_PRICE, "t": _stamp(-1)},
                "latestQuote": {"bp": 1.0, "ap": 1.2, "t": _stamp(-9999)},
            }
        }
        md = _snapshot_market_data(body)
        assert md.price(SYMBOL) == pytest.approx(SPY_PRICE)
        assert md.last_observation.field == "latestTrade"

    def test_an_exact_tie_prefers_the_trade(self):
        body = {
            SYMBOL: {
                "latestTrade": {"p": 111.0, "t": _stamp(-10)},
                "latestQuote": {"bp": 200.0, "ap": 202.0, "t": _stamp(-10)},
            }
        }
        md = _snapshot_market_data(body)
        assert md.price(SYMBOL) == pytest.approx(111.0)
        assert md.last_observation.field == "latestTrade"

    @pytest.mark.parametrize("bad_stamp", ["not-a-time", None, "", "   "])
    def test_an_unverifiable_trade_cannot_outrank_a_fresh_quote(self, bad_stamp):
        body = {
            SYMBOL: {
                "latestTrade": {"p": SPY_PRICE, "t": bad_stamp},
                "latestQuote": {"bp": 99.0, "ap": 101.0, "t": _stamp(-1)},
            }
        }
        md = _snapshot_market_data(body)
        assert md.price(SYMBOL) == pytest.approx(100.0)
        assert md.last_observation.field == "latestQuote"

    def test_no_verifiable_candidate_is_refused(self):
        body = {
            SYMBOL: {
                "latestTrade": {"p": SPY_PRICE},
                "latestQuote": {"bp": 99.0, "ap": 101.0},
            }
        }
        md = _snapshot_market_data(body)
        with pytest.raises(MarketDataError, match="no timestamp"):
            md.price(SYMBOL)

    def test_a_daily_bar_is_never_a_candidate(self):
        body = {SYMBOL: {"dailyBar": {"o": 500.0, "c": 580.0, "t": _stamp(-1)}}}
        md = _snapshot_market_data(body)
        with pytest.raises(MarketDataError, match="no usable current price"):
            md.price(SYMBOL)

    def test_the_freshest_field_is_still_subject_to_the_freshness_limit(self):
        """Selection never weakens the check: the newest field must still be fresh."""
        body = {
            SYMBOL: {
                "latestTrade": {"p": SPY_PRICE, "t": _stamp(-400)},
                "latestQuote": {"bp": 99.0, "ap": 101.0, "t": _stamp(-350)},
            }
        }
        md = _snapshot_market_data(body)
        with pytest.raises(MarketDataError, match="stale"):
            md.price(SYMBOL)

    def test_a_future_dated_newest_field_is_refused_not_downgraded(self):
        """A future newest field is refused; it is not skipped for an older one."""
        body = {
            SYMBOL: {
                "latestTrade": {"p": SPY_PRICE, "t": _stamp(-400)},
                "latestQuote": {"bp": 99.0, "ap": 101.0, "t": _stamp(600)},
            }
        }
        md = _snapshot_market_data(body)
        with pytest.raises(MarketDataError, match="future"):
            md.price(SYMBOL)

    def test_a_malformed_quote_price_still_falls_through_to_a_fresh_bar(self):
        body = {
            SYMBOL: {
                "latestQuote": {"bp": 0, "ap": 101.0, "t": _stamp(-1)},
                "minuteBar": {"c": 123.0, "t": _stamp(-5)},
            }
        }
        md = _snapshot_market_data(body)
        assert md.price(SYMBOL) == pytest.approx(123.0)
        assert md.last_observation.field == "minuteBar"

    def test_selection_is_deterministic_across_repeated_reads(self):
        body = {
            SYMBOL: {
                "latestTrade": {"p": SPY_PRICE, "t": _stamp(-400)},
                "latestQuote": {"bp": 99.0, "ap": 101.0, "t": _stamp(-1)},
                "minuteBar": {"c": 123.0, "t": _stamp(-5)},
            }
        }
        reads = [_snapshot_market_data(body) for _ in range(3)]
        prices = [md.price(SYMBOL) for md in reads]
        fields = [md.last_observation.field for md in reads]
        assert prices == [100.0, 100.0, 100.0]
        assert fields == ["latestQuote", "latestQuote", "latestQuote"]


# ---------------------------------------------------------------------------
# 8. Fractional sizing is opt-in and respects the cap
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFractionalSizing:
    def _worker(self, tmp_path, *, whole_shares: bool):
        return PaperTradingWorker(
            config=PaperTradingConfig(
                order_notional=25.0,
                whole_shares=whole_shares,
                limits=PaperRiskLimits(),
                state_path=tmp_path / "worker_state.json",
            ),
            decision_source=_OneDecision(),
            market_data=StaticMarketData({SYMBOL: SPY_PRICE}),
        )

    def test_whole_shares_floors_to_zero_and_refuses(self, tmp_path):
        from tradingagents.paper_experiment.broker import WorkerError

        worker = self._worker(tmp_path, whole_shares=True)
        with pytest.raises(WorkerError, match="fractional"):
            worker._size(  # noqa: SLF001
                TradingDecision(decision_id="d1", symbol=SYMBOL, side="buy", notional=25.0),
                SPY_PRICE,
            )

    def test_fractional_sizing_fits_the_notional(self, tmp_path):
        worker = self._worker(tmp_path, whole_shares=False)
        quantity = worker._size(  # noqa: SLF001
            TradingDecision(decision_id="d1", symbol=SYMBOL, side="buy", notional=25.0),
            SPY_PRICE,
        )
        assert quantity > 0
        assert quantity * SPY_PRICE <= 25.0

    def test_the_25_dollar_cap_is_unchanged(self, tmp_path):
        """Fractional sizing never lets an order exceed the configured cap."""
        limits = PaperRiskLimits(max_order_notional=25.0)
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                order_notional=25.0, whole_shares=False, limits=limits,
                state_path=tmp_path / "worker_state.json",
            ),
            decision_source=_OneDecision(),
            market_data=StaticMarketData({SYMBOL: SPY_PRICE}),
        )
        quantity = worker._size(  # noqa: SLF001
            TradingDecision(decision_id="d1", symbol=SYMBOL, side="buy", notional=25.0),
            SPY_PRICE,
        )
        assert quantity * SPY_PRICE <= limits.max_order_notional


# ---------------------------------------------------------------------------
# 9. Safety boundaries preserved
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSafetyBoundariesPreserved:
    def test_allow_short_is_still_false_by_default(self):
        assert PaperRiskLimits().allow_short is False

    def test_the_paper_endpoint_is_the_only_one_accepted(self):
        from tradingagents.paper_experiment.broker import (
            PAPER_BASE_URL,
            LiveEndpointRefusedError,
            assert_paper_endpoint,
        )

        assert assert_paper_endpoint(PAPER_BASE_URL) == PAPER_BASE_URL
        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint("https://api.alpaca.markets")

    def test_the_snapshot_seam_refuses_a_live_data_host(self):
        from tradingagents.paper_experiment.providers.base import ProviderError

        with pytest.raises(ProviderError):
            AlpacaSnapshotMarketData(
                (SYMBOL,), api_key="k", api_secret="s",
                base_url="https://api.alpaca.markets",
            )

    def test_a_worker_config_refuses_an_unknown_provider(self):
        from tradingagents.paper_experiment.broker import WorkerError

        with pytest.raises(WorkerError):
            PaperTradingConfig(market_data_provider="live_alpaca")

    def test_a_worker_config_refuses_a_live_endpoint(self):
        from tradingagents.paper_experiment.broker import LiveEndpointRefusedError

        with pytest.raises(LiveEndpointRefusedError):
            PaperTradingConfig(base_url="https://api.alpaca.markets")


# ---------------------------------------------------------------------------
# 10. Status and cycle results name the provider
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProviderIsSurfaced:
    def test_the_status_view_names_the_configured_provider(self, tmp_path):
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                order_notional=25.0,
                limits=PaperRiskLimits(),
                state_path=tmp_path / "worker_state.json",
                market_data_provider="alpaca_snapshot",
            ),
            decision_source=_OneDecision(),
            market_data=StaticMarketData({SYMBOL: SPY_PRICE}),
        )
        status = worker.status()
        assert status["market_data_provider_configured"] == "alpaca_snapshot"
        assert status["risk_limits"]["whole_shares"] is WHOLE_SHARES

    def test_the_cycle_result_carries_the_reference_price(self, tmp_path):
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                order_notional=25.0,
                limits=PaperRiskLimits(),
                state_path=tmp_path / "worker_state.json",
            ),
            decision_source=_OneDecision(),
            market_data=StaticMarketData({SYMBOL: 10.0}),
        )
        worker.state.status = "RUNNING"
        worker._broker = _RecordingBroker()  # noqa: SLF001
        result = worker.run_once()
        assert result.ran is True
        assert result.reference_price == pytest.approx(10.0)
        assert result.market_data_provider == "static"
        assert result.to_dict()["reference_price"] == pytest.approx(10.0)

    def test_a_market_data_failure_still_names_the_provider(self, tmp_path):
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                order_notional=25.0,
                limits=PaperRiskLimits(),
                state_path=tmp_path / "worker_state.json",
                market_data_provider="alpaca_snapshot",
            ),
            decision_source=_OneDecision(),
            market_data=_snapshot_market_data(_snapshot_body(stamp=_stamp(-400))),
        )
        worker.state.status = "RUNNING"
        worker._broker = _FailIfCalledBroker()  # noqa: SLF001
        result = worker.run_once()
        assert result.ran is False
        assert result.market_data_provider == "alpaca_snapshot"
        assert result.market_data_status == "stale"

    def test_the_status_view_reports_kind_and_freshness_fields(self, tmp_path):
        md = _snapshot_market_data(_snapshot_body(stamp=_stamp(-5)))
        md.price(SYMBOL)
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                order_notional=25.0,
                limits=PaperRiskLimits(),
                state_path=tmp_path / "worker_state.json",
                market_data_provider="alpaca_snapshot",
                market_data_max_age_seconds=300.0,
            ),
            decision_source=_OneDecision(),
            market_data=md,
        )
        status = worker.status()
        assert status["market_data_provider"] == "alpaca_snapshot"
        assert status["market_data_kind"] == "current"
        assert status["market_data_status"] == "ok"
        assert status["last_snapshot_price"] == pytest.approx(SPY_PRICE)
        assert status["last_snapshot_age_seconds"] == pytest.approx(5.0)
        assert status["max_snapshot_age_seconds"] == 300.0
        # The status view is JSON-serializable (no internal objects leak).
        json.dumps(status)

    def test_the_status_view_without_a_worker_names_the_configured_provider(self, tmp_path):
        """A stopped controller still names the seam it would use, without building."""
        root = _experiment_root(tmp_path)
        controller = PaperTradingController(
            root,
            enabled=True,
            market_data_provider="alpaca_snapshot",
            market_data_max_age_seconds=120.0,
        )
        status = controller.status()
        assert status["market_data_provider_configured"] == "alpaca_snapshot"
        assert status["market_data_max_age_seconds"] == 120.0
        assert status["status"] == "STOPPED"
        json.dumps(status)

    def test_the_historical_seam_reports_a_historical_kind(self, tmp_path):
        """A bar-backed seam reports its *historical* kind, never `current`."""
        root = _experiment_root(tmp_path)
        controller = PaperTradingController(root, enabled=True)
        experiment = PaperExperiment.load(root, inference=None)
        source = experiment.source
        market = controller._build_market_data(  # noqa: SLF001
            experiment, source, source.sessions()
        )
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                order_notional=25.0,
                limits=PaperRiskLimits(),
                state_path=tmp_path / "worker_state.json",
            ),
            decision_source=_OneDecision(),
            market_data=market,
        )
        status = worker.status()
        assert status["market_data_kind"] == "historical"
        assert status["market_data_provider"] == "bar_open"

    def test_the_static_seam_reports_its_test_kind(self, tmp_path):
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                order_notional=25.0,
                limits=PaperRiskLimits(),
                state_path=tmp_path / "worker_state.json",
            ),
            decision_source=_OneDecision(),
            market_data=StaticMarketData({SYMBOL: 10.0}),
        )
        status = worker.status()
        assert status["market_data_kind"] == "test"
        assert status["market_data_provider"] == "static"


class _RecordingBroker:
    """A broker that records the submit and returns an accepted order."""

    def __init__(self):
        self.submitted: dict | None = None

    def submit_order(self, **kwargs):
        self.submitted = kwargs
        from tradingagents.paper_experiment.broker import BrokerOrder

        return BrokerOrder(
            client_order_id=kwargs["client_order_id"],
            broker_order_id="broker-1",
            symbol=kwargs["symbol"],
            side=kwargs["side"],
            quantity=kwargs["quantity"],
            filled_quantity=0.0,
            filled_avg_price=None,
            status="accepted",
        )

    def positions(self, refresh=False):  # pragma: no cover
        return {}


# ---------------------------------------------------------------------------
# 11. The dashboard route carries the market-data fields
# ---------------------------------------------------------------------------


_REAL_CONNECT, _REAL_LOOKUP = socket.socket.connect, socket.getaddrinfo


def _refuse(*args, **kwargs):
    raise OSError(f"test tried to reach the network: {args[1:] or kwargs}")


@pytest.fixture
def allow_loopback(monkeypatch):
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda sock, address: (
            _REAL_CONNECT if address[0] == "127.0.0.1" else _refuse)(sock, address),
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **k: (
            _REAL_LOOKUP if host == "127.0.0.1" else _refuse)(host, *a, **k),
    )


@pytest.mark.unit
class TestDashboardStatusRoute:
    def test_the_status_route_reports_the_configured_provider(self, tmp_path, allow_loopback):
        import threading
        import urllib.request
        from http.server import ThreadingHTTPServer

        from tradingagents.paper_experiment.dashboard.server import Dashboard, make_handler

        token = "phase26-token"
        root = _experiment_root(tmp_path)
        dashboard = Dashboard(
            root,
            token=token,
            allow_paper_trading=True,
            paper_market_data_provider="alpaca_snapshot",
            paper_market_data_max_age_seconds=120.0,
        )
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(dashboard))
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{httpd.server_address[1]}/api/paper/status"
            )
            request.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(request, timeout=10) as response:
                body = json.loads(response.read())
            assert body["market_data_provider_configured"] == "alpaca_snapshot"
            assert body["market_data_max_age_seconds"] == 120.0
            # The whole payload stays JSON-serializable and names no secret.
            assert "api_secret" not in json.dumps(body).lower()
        finally:
            dashboard.paper.shutdown()
            httpd.shutdown()
            httpd.server_close()
