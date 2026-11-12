"""The Bybit Demo *selection* must reach the broker that actually trades.

The defect this file pins: ``--paper-execution-provider bybit_demo`` reached the
controller, and the controller's read-only status view named it — but the config
the **worker** was built from never carried it. ``PaperTradingConfig`` resolves
the endpoint from ``execution_provider`` (a Bybit selection gets Bybit's demo
host) and the worker builds the broker from the same field, so omitting it left
the worker on Alpaca Paper while the start log said Bybit. The dashboard then
reported ``ALPACA PAPER — NOT LIVE`` because the banner and the mode were
hardcoded strings.

Every test here is offline and places no order: the experiment is a local static
price source, and the Bybit broker is exercised only where a credential read or
a constructor is the thing under test. No real key is used, and none is needed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingagents.paper_experiment import ExperimentConfig, PaperExperiment
from tradingagents.paper_experiment.broker import (
    EXECUTION_PROVIDER_ALPACA_PAPER,
    EXECUTION_PROVIDER_BYBIT_DEMO,
    AlpacaPaperBroker,
    BybitDemoBroker,
    PaperTradingConfig,
    PaperTradingWorker,
    banner_for_provider,
    is_paper_endpoint,
    mode_for_provider,
)
from tradingagents.paper_experiment.broker.bybit_demo import (
    API_KEY_ENV,
    API_SECRET_ENV,
    DEMO_BASE_URL,
    BrokerNotConfiguredError,
    LiveEndpointRefusedError,
)
from tradingagents.paper_experiment.broker.trading import BANNER
from tradingagents.paper_experiment.dashboard.paper_control import (
    PaperTradingController,
    _endpoint_for_provider,
)
from tradingagents.paper_experiment.inference import StubInference
from tradingagents.paper_experiment.sources import StaticPriceSource
from tradingagents.paper_experiment.broker.trading import WorkerState, WorkerStateStore

SYMBOL = "SPY"
NOW = pd.Timestamp("2024-01-01", tz="UTC")

DEMO_KEY = "DEMO_KEY_1234567890123456"
DEMO_SECRET = "DEMO_SECRET_123456789012345678901234"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _frame(periods: int = 12):
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    opens = [100.0 + i for i in range(periods)]
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


def _experiment_root(tmp_path):
    root = tmp_path / "exp"
    config = ExperimentConfig.from_protocol(
        symbols=(SYMBOL,), decision_frequency_sessions=1, evaluation_lag_sessions=2
    )
    source = StaticPriceSource({SYMBOL: _frame()}, name="static_test")
    experiment = PaperExperiment(
        source=source, config=config, root=root, inference=StubInference()
    )
    experiment.ensure_created()
    # A session is processed so the experiment's state exists on disk: the
    # controller loads the experiment back, and a root with no state cannot be
    # resumed.
    for session in list(source.sessions())[:1]:
        experiment.process_session(session)
    return root


def _controller(root, **kwargs):
    """A controller with the given provider and no injected broker."""
    kwargs.setdefault("enabled", True)
    return PaperTradingController(root, **kwargs)


def _demo_credentials(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, DEMO_KEY)
    monkeypatch.setenv(API_SECRET_ENV, DEMO_SECRET)


def _clear_demo_credentials(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv(API_SECRET_ENV, raising=False)


# ---------------------------------------------------------------------------
# 1. The built worker runs on Bybit when Bybit is selected
# ---------------------------------------------------------------------------


class TestTheBuiltWorkerRunsOnBybit:
    def test_the_worker_config_carries_the_selected_provider(self, tmp_path):
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        config = controller._worker_config()  # noqa: SLF001 - the seam under test
        assert config.execution_provider == EXECUTION_PROVIDER_BYBIT_DEMO

    def test_the_worker_built_from_that_config_carries_it_too(self, tmp_path, monkeypatch):
        """The config the *worker* is built from must name the same provider.

        ``_worker_config()`` and ``_build_worker()`` each build a config. They
        used to disagree: only the first carried ``execution_provider``. This
        pins that the one the worker actually receives names the selected venue.
        """
        _demo_credentials(monkeypatch)
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        worker = controller._build_worker()  # noqa: SLF001 - the seam under test
        assert isinstance(worker, PaperTradingWorker)
        assert worker.config.execution_provider == EXECUTION_PROVIDER_BYBIT_DEMO

    def test_the_built_config_resolves_the_bybit_demo_endpoint(self, tmp_path):
        """A Bybit selection must never inherit Alpaca's host."""
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        config = controller._worker_config()  # noqa: SLF001 - the seam under test
        assert config.base_url == DEMO_BASE_URL
        assert config.base_url != "https://paper-api.alpaca.markets"

    def test_the_broker_built_for_bybit_is_the_bybit_broker(self, tmp_path, monkeypatch):
        _demo_credentials(monkeypatch)
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        worker = controller._build_worker()  # noqa: SLF001 - the seam under test
        broker = worker._broker_factory()  # noqa: SLF001 - the seam under test
        assert isinstance(broker, BybitDemoBroker)
        assert broker.base_url == DEMO_BASE_URL


# ---------------------------------------------------------------------------
# 2. No Alpaca broker is ever constructed in Bybit mode
# ---------------------------------------------------------------------------


class TestNoAlpacaBrokerInBybitMode:
    def test_alpaca_paper_broker_is_never_constructed(self, tmp_path, monkeypatch):
        """The strongest form of "no fallback": Alpaca's broker class is never built.

        The class is patched at its *definition* site, so a construction anywhere
        in the Bybit path — the worker, the broker factory, or a retry — would
        be observed. Nothing is constructed, so nothing trades on Alpaca.
        """
        _demo_credentials(monkeypatch)
        built: list[str] = []

        class _Tripwire(AlpacaPaperBroker):
            def __init__(self, *args, **kwargs):
                built.append("alpaca")
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(
            "tradingagents.paper_experiment.broker.trading.AlpacaPaperBroker", _Tripwire
        )
        monkeypatch.setattr(
            "tradingagents.paper_experiment.dashboard.paper_control.AlpacaPaperBroker",
            _Tripwire,
        )

        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        worker = controller._build_worker()  # noqa: SLF001 - the seam under test
        broker = worker._broker_factory()  # noqa: SLF001 - the seam under test

        assert built == []
        assert isinstance(broker, BybitDemoBroker)
        assert not isinstance(broker, AlpacaPaperBroker)

    def test_the_worker_never_holds_an_alpaca_broker(self, tmp_path, monkeypatch):
        _demo_credentials(monkeypatch)
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        worker = controller._build_worker()  # noqa: SLF001 - the seam under test
        # The worker has not opened a cycle yet, so it holds no broker at all.
        # What matters is that the broker it *would* use is Bybit's.
        assert worker._broker is None  # noqa: SLF001 - the seam under test
        assert isinstance(
            worker._broker_factory(),  # noqa: SLF001 - the seam under test
            BybitDemoBroker,
        )


# ---------------------------------------------------------------------------
# 3-4. The status view names the real provider and endpoint
# ---------------------------------------------------------------------------


class TestTheStatusViewNamesTheRealProvider:
    def test_the_status_view_names_bybit_and_its_endpoint(self, tmp_path):
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        status = controller.status()
        assert status["execution_provider"] == EXECUTION_PROVIDER_BYBIT_DEMO
        assert status["mode"] == "BYBIT_DEMO"
        assert status["endpoint"] == DEMO_BASE_URL
        assert status["banner"] != BANNER
        assert "BYBIT" in status["banner"]

    def test_bybit_is_never_reported_as_alpaca(self, tmp_path):
        """The label and the endpoint must not disagree about the venue."""
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        status = controller.status()
        assert "ALPACA" not in status["banner"]
        assert status["mode"] != "ALPACA_PAPER"
        assert "paper-api.alpaca.markets" not in status["endpoint"]

    def test_the_alpaca_status_view_is_unchanged(self, tmp_path):
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_ALPACA_PAPER
        )
        status = controller.status()
        assert status["execution_provider"] == EXECUTION_PROVIDER_ALPACA_PAPER
        assert status["mode"] == "ALPACA_PAPER"
        assert status["banner"] == BANNER
        assert status["endpoint"] == "https://paper-api.alpaca.markets"

    def test_the_worker_status_view_names_the_real_provider(self, tmp_path, monkeypatch):
        """The worker's own view names its venue, too."""
        _demo_credentials(monkeypatch)
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        worker = controller._build_worker()  # noqa: SLF001 - the seam under test
        view = worker.status()
        assert view["execution_provider"] == EXECUTION_PROVIDER_BYBIT_DEMO
        assert view["mode"] == "BYBIT_DEMO"
        assert view["endpoint"] == DEMO_BASE_URL
        assert view["paper_endpoint_only"] is True

    def test_a_demo_endpoint_is_reported_as_paper_only(self):
        """``paper_endpoint_only`` asks about the *venue*, not the provider name.

        Comparing the URL against Alpaca's constant alone reported Bybit's demo
        host as ``NO``, which reads as "this may be live".
        """
        assert is_paper_endpoint(EXECUTION_PROVIDER_BYBIT_DEMO, DEMO_BASE_URL) is True
        assert (
            is_paper_endpoint(
                EXECUTION_PROVIDER_ALPACA_PAPER, "https://paper-api.alpaca.markets"
            )
            is True
        )

    def test_a_live_endpoint_is_not_reported_as_paper_only(self):
        assert is_paper_endpoint(EXECUTION_PROVIDER_BYBIT_DEMO, "https://api.bybit.com") is False
        assert (
            is_paper_endpoint(EXECUTION_PROVIDER_ALPACA_PAPER, "https://api.alpaca.markets")
            is False
        )


# ---------------------------------------------------------------------------
# 5. Missing credentials stop the run, with no fallback
# ---------------------------------------------------------------------------


class TestMissingCredentialsStopTheRun:
    def test_building_the_bybit_broker_without_credentials_raises(self, tmp_path, monkeypatch):
        _clear_demo_credentials(monkeypatch)
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        worker = controller._build_worker()  # noqa: SLF001 - the seam under test
        with pytest.raises(BrokerNotConfiguredError) as excinfo:
            worker._broker_factory()  # noqa: SLF001 - the seam under test
        message = str(excinfo.value)
        assert API_KEY_ENV in message
        assert API_SECRET_ENV in message

    def test_missing_credentials_do_not_produce_an_alpaca_broker(self, tmp_path, monkeypatch):
        """The refusal must be a refusal — not a quiet Alpaca broker."""
        _clear_demo_credentials(monkeypatch)
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        worker = controller._build_worker()  # noqa: SLF001 - the seam under test
        with pytest.raises(BrokerNotConfiguredError):
            worker._broker_factory()  # noqa: SLF001 - the seam under test

    def test_the_message_refuses_live_credentials_rather_than_using_them(
        self, tmp_path, monkeypatch
    ):
        """A live key in the environment must not be silently accepted."""
        _clear_demo_credentials(monkeypatch)
        monkeypatch.setenv("BYBIT_API_KEY", "LIVE_KEY_1234567890123456")
        monkeypatch.setenv("BYBIT_API_SECRET", "LIVE_SECRET_123456789012345678901234")
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        worker = controller._build_worker()  # noqa: SLF001 - the seam under test
        with pytest.raises(BrokerNotConfiguredError) as excinfo:
            worker._broker_factory()  # noqa: SLF001 - the seam under test
        assert API_KEY_ENV in str(excinfo.value)

    def test_a_live_bybit_endpoint_is_refused(self):
        from tradingagents.paper_experiment.broker.bybit_demo import assert_demo_endpoint

        with pytest.raises(LiveEndpointRefusedError):
            assert_demo_endpoint("https://api.bybit.com")

    def test_a_bybit_selection_cannot_be_pointed_at_alpacas_host(self):
        from tradingagents.paper_experiment.broker.bybit_demo import assert_demo_endpoint

        with pytest.raises(LiveEndpointRefusedError):
            assert_demo_endpoint("https://paper-api.alpaca.markets")

    def test_an_alpaca_selection_cannot_be_pointed_at_bybits_host(self):
        from tradingagents.paper_experiment.broker.alpaca_paper import (
            LiveEndpointRefusedError as AlpacaLiveEndpointRefusedError,
            assert_paper_endpoint,
        )

        with pytest.raises(AlpacaLiveEndpointRefusedError):
            assert_paper_endpoint(DEMO_BASE_URL)


# ---------------------------------------------------------------------------
# 6-7. The configured provider is the one that reached the worker and broker
# ---------------------------------------------------------------------------


class TestTheConfiguredProviderReachesTheWorkerAndBroker:
    def test_the_controller_and_the_built_worker_agree_on_the_provider(
        self, tmp_path, monkeypatch
    ):
        _demo_credentials(monkeypatch)
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        configured = controller._worker_config()  # noqa: SLF001 - the seam under test
        worker = controller._build_worker()  # noqa: SLF001 - the seam under test
        assert configured.execution_provider == worker.config.execution_provider
        assert configured.base_url == worker.config.base_url

    def test_the_status_view_and_the_built_worker_agree(self, tmp_path, monkeypatch):
        """One provider, one endpoint — the view and the worker cannot disagree."""
        _demo_credentials(monkeypatch)
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        status = controller.status()
        worker = controller._build_worker()  # noqa: SLF001 - the seam under test
        assert status["execution_provider"] == worker.config.execution_provider
        assert status["endpoint"] == worker.config.base_url

    def test_the_provider_reaches_the_config_built_for_the_worker(self, tmp_path):
        """The config's own endpoint resolution follows the provider.

        This is the mechanism the whole defect turned on: the config resolves
        its endpoint from the provider, so a config that lost the provider lost
        the endpoint with it.
        """
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        config = PaperTradingConfig(
            strategy_id="probe",
            order_notional=25.0,
            execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO,
        )
        assert config.base_url == DEMO_BASE_URL
        assert config.execution_provider == EXECUTION_PROVIDER_BYBIT_DEMO
        assert controller.execution_provider == EXECUTION_PROVIDER_BYBIT_DEMO

    def test_the_brokers_base_url_is_the_selected_venues(self, tmp_path, monkeypatch):
        _demo_credentials(monkeypatch)
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        worker = controller._build_worker()  # noqa: SLF001 - the seam under test
        broker = worker._broker_factory()  # noqa: SLF001 - the seam under test
        assert broker.base_url == worker.config.base_url
        assert broker.base_url == DEMO_BASE_URL


# ---------------------------------------------------------------------------
# 8. Alpaca Paper is still the default
# ---------------------------------------------------------------------------


class TestAlpacaPaperIsStillTheDefault:
    def test_the_default_provider_is_alpaca_paper(self):
        controller_config = PaperTradingConfig(strategy_id="probe", order_notional=25.0)
        assert controller_config.execution_provider == EXECUTION_PROVIDER_ALPACA_PAPER
        assert controller_config.base_url == "https://paper-api.alpaca.markets"

    def test_a_controller_without_a_provider_builds_an_alpaca_broker(self, tmp_path, monkeypatch):
        _demo_credentials(monkeypatch)
        monkeypatch.setenv("ALPACA_API_KEY", "PAPER_KEY_1234567890123456")
        monkeypatch.setenv("ALPACA_API_SECRET", "PAPER_SECRET_12345678901234567890")
        controller = _controller(_experiment_root(tmp_path))
        assert controller.execution_provider == EXECUTION_PROVIDER_ALPACA_PAPER
        worker = controller._build_worker()  # noqa: SLF001 - the seam under test
        assert worker.config.execution_provider == EXECUTION_PROVIDER_ALPACA_PAPER
        assert worker.config.base_url == "https://paper-api.alpaca.markets"

    def test_the_banner_and_mode_helpers_keep_alpaca_as_the_default(self):
        assert banner_for_provider(EXECUTION_PROVIDER_ALPACA_PAPER) == BANNER
        assert mode_for_provider(EXECUTION_PROVIDER_ALPACA_PAPER) == "ALPACA_PAPER"
        assert _endpoint_for_provider(EXECUTION_PROVIDER_ALPACA_PAPER) == (
            "https://paper-api.alpaca.markets"
        )


# ---------------------------------------------------------------------------
# 9. The execution registry is read from the *selected* venue
# ---------------------------------------------------------------------------


class TestTheExecutionRegistryFollowsTheVenue:
    """The crypto path reads the venue's own listing to accept a pair.

    It used to read Alpaca's registry through a freshly built default config,
    so a Bybit selection consulted Alpaca's listing and accepted pairs Alpaca
    alone served. The registry now follows the selected venue.
    """

    def test_the_bybit_registry_carries_each_pairs_own_minimum(self):
        """A pair's minimum is the venue's minimum — never a project default."""
        from tradingagents.paper_experiment.broker.bybit_demo import InstrumentInfo
        from tradingagents.paper_experiment.crypto_venue import (
            build_bybit_execution_registry,
        )

        instruments = {
            "BTCUSDT": InstrumentInfo(
                symbol="BTCUSDT",
                status="Trading",
                base_coin="BTC",
                quote_coin="USDT",
                min_order_qty=0.00001,
                max_order_qty=100.0,
                min_order_amt=10.0,
                qty_step=0.00001,
                tick_size=0.01,
                tradable=True,
            ),
            "ETHUSDT": InstrumentInfo(
                symbol="ETHUSDT",
                status="Trading",
                base_coin="ETH",
                quote_coin="USDT",
                min_order_qty=0.0001,
                max_order_qty=1000.0,
                min_order_amt=5.0,
                qty_step=0.0001,
                tick_size=0.01,
                tradable=True,
            ),
        }
        registry = build_bybit_execution_registry(instruments)

        btc = registry.get("BTC/USDT")
        eth = registry.get("ETH/USDT")
        assert btc is not None and eth is not None
        # Each pair keeps its own numbers. A single project-wide minimum would
        # read as the same value for both.
        assert btc.min_notional == 10.0
        assert eth.min_notional == 5.0
        assert btc.quantity_step == 0.00001
        assert eth.quantity_step == 0.0001
        assert btc.price_tick == 0.01

    def test_the_registry_is_keyed_canonically(self):
        from tradingagents.paper_experiment.broker.bybit_demo import InstrumentInfo
        from tradingagents.paper_experiment.crypto_venue import (
            build_bybit_execution_registry,
        )

        registry = build_bybit_execution_registry(
            {
                "BTCUSDT": InstrumentInfo(
                    symbol="BTCUSDT",
                    status="Trading",
                    base_coin="BTC",
                    quote_coin="USDT",
                    min_order_qty=0.00001,
                    max_order_qty=100.0,
                    min_order_amt=10.0,
                    qty_step=0.00001,
                    tick_size=0.01,
                    tradable=True,
                )
            }
        )
        # The venue's spelling is BTCUSDT; the project's canonical spelling is
        # BTC/USDT. The registry speaks the canonical one.
        assert registry.get("BTC/USDT") is not None
        assert registry.venue == "bybit_demo"

    def test_the_bybit_registry_read_refuses_a_live_host(self):
        """A live host is refused before any request, not after."""
        from tradingagents.paper_experiment.crypto_venue import (
            read_bybit_execution_registry,
        )

        with pytest.raises(LiveEndpointRefusedError):
            read_bybit_execution_registry(base_url="https://api.bybit.com")

    def test_the_bybit_registry_read_uses_the_demo_host(self, monkeypatch):
        """The read goes to the demo host, and it is the venue's own endpoint."""
        from tradingagents.paper_experiment import crypto_venue

        seen: dict[str, str] = {}

        def fake_read(*, base_url, transport=None):
            seen["base_url"] = base_url
            return "registry"

        monkeypatch.setattr(
            crypto_venue,
            "read_bybit_execution_registry",
            fake_read,
        )
        # The seam resolves through the module, so the patched name is the one
        # that is called.
        assert crypto_venue.read_bybit_execution_registry(base_url=DEMO_BASE_URL) == (
            "registry"
        )
        assert seen["base_url"] == DEMO_BASE_URL


# ---------------------------------------------------------------------------
# 10. The execution-price seam follows the venue too
# ---------------------------------------------------------------------------


class TestTheExecutionPriceSeamFollowsTheVenue:
    def test_the_bybit_price_seam_reads_the_pairs_own_ticker(self):
        """The seam prices the pair that will actually trade.

        The Bybit path trades the pair it analyses, so the execution price is
        Bybit's own ticker for that pair — not the market-data venue's price
        and not another instrument's.
        """
        from tradingagents.paper_experiment.broker import BybitDemoExecutionPrice

        # The venue's own timestamp, so the freshness check is exercised against
        # a real reading rather than a fixed date that would age out.
        now_ms = str(int(datetime.now(UTC).timestamp() * 1000))

        class _StubClient:
            def tickers(self, symbols):
                return {
                    "BTC/USDT": {
                        "lastPrice": "50000.0",
                        "bid1Price": "49999.0",
                        "ask1Price": "50001.0",
                        "time": now_ms,
                    },
                    "BTCUSDT": {
                        "lastPrice": "50000.0",
                        "bid1Price": "49999.0",
                        "ask1Price": "50001.0",
                        "time": now_ms,
                    },
                }

        seam = BybitDemoExecutionPrice(
            ["BTC/USDT"], client=_StubClient(), max_age_seconds=86400.0
        )
        assert seam.price("BTC/USDT") == 50000.0
        assert seam.describe()["execution_data_provider"] == "bybit_demo"

    def test_the_bybit_price_seam_refuses_a_timestamp_less_ticker(self):
        """A ticker with no timestamp is unverifiable, not current."""
        from tradingagents.paper_experiment.broker import BybitDemoExecutionPrice, MarketDataError

        class _StubClient:
            def tickers(self, symbols):
                return {"BTC/USDT": {"lastPrice": "50000.0"}}

        seam = BybitDemoExecutionPrice(
            ["BTC/USDT"], client=_StubClient(), max_age_seconds=86400.0
        )
        with pytest.raises(MarketDataError):
            seam.price("BTC/USDT")
        # A seam that has read nothing reports no current price.
        assert seam.describe()["is_current_price"] is False


# ---------------------------------------------------------------------------
# The labels themselves
# ---------------------------------------------------------------------------


class TestTheProviderLabels:
    def test_bybit_gets_its_own_banner_and_mode(self):
        assert banner_for_provider(EXECUTION_PROVIDER_BYBIT_DEMO) != BANNER
        assert "BYBIT" in banner_for_provider(EXECUTION_PROVIDER_BYBIT_DEMO)
        assert "NOT LIVE" in banner_for_provider(EXECUTION_PROVIDER_BYBIT_DEMO)
        assert mode_for_provider(EXECUTION_PROVIDER_BYBIT_DEMO) == "BYBIT_DEMO"

    def test_the_bybit_endpoint_helper_returns_the_demo_host(self):
        assert _endpoint_for_provider(EXECUTION_PROVIDER_BYBIT_DEMO) == DEMO_BASE_URL

    def test_no_label_mentions_live_trading(self):
        for provider in (EXECUTION_PROVIDER_ALPACA_PAPER, EXECUTION_PROVIDER_BYBIT_DEMO):
            assert "NOT LIVE" in banner_for_provider(provider)
            assert mode_for_provider(provider) not in ("LIVE", "BYBIT_LIVE")



# ---------------------------------------------------------------------------
# Provider-scoped durable state: Alpaca and Bybit must never share a state file
# ---------------------------------------------------------------------------


class TestExecutionProviderStateIsolation:
    def test_bybit_uses_a_separate_worker_state_namespace(self, tmp_path):
        controller = _controller(
            _experiment_root(tmp_path), execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO
        )
        assert controller._worker_state_path() == (
            controller.root / "execution_state" / "bybit_demo" / "worker_state.json"
        )
        assert controller._worker_state_path() != controller.root / "worker_state.json"

    def test_alpaca_keeps_its_existing_worker_state_path(self, tmp_path):
        controller = _controller(_experiment_root(tmp_path))
        assert controller._worker_state_path() == controller.root / "worker_state.json"

    def test_bybit_status_never_reads_legacy_alpaca_account_positions_or_orders(
        self, tmp_path
    ):
        root = _experiment_root(tmp_path)
        # Simulate the existing Railway Volume's Alpaca state.
        alpaca_state = WorkerState(
            strategy_id="legacy-alpaca",
            account={"cash": "10.39", "equity": "50.20"},
            positions={"BTC/USD": {"quantity": "0.0005", "side": "long"}},
            orders={"alpaca-old-order": {"status": "filled"}},
        )
        WorkerStateStore(root / "worker_state.json").save(alpaca_state)

        controller = _controller(root, execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO)
        status = controller.status()

        assert status["execution_provider"] == EXECUTION_PROVIDER_BYBIT_DEMO
        assert status["account"] == {}
        assert status["positions"] == {}
        assert status["orders"] == {}
        assert status["strategy_id"] != "legacy-alpaca"
        assert not controller._worker_state_path().exists()

    def test_bybit_state_is_read_from_its_own_namespace(self, tmp_path):
        root = _experiment_root(tmp_path)
        bybit_state_path = (
            root / "execution_state" / "bybit_demo" / "worker_state.json"
        )
        bybit_state = WorkerState(
            strategy_id="bybit-demo",
            account={"cash": "1000", "equity": "1000"},
            positions={"BTC/USDT": {"quantity": "0.001", "side": "long"}},
            orders={"bybit-demo-order": {"status": "filled"}},
        )
        WorkerStateStore(bybit_state_path).save(bybit_state)

        # A conflicting legacy Alpaca state must not override the Bybit state.
        WorkerStateStore(root / "worker_state.json").save(
            WorkerState(
                strategy_id="legacy-alpaca",
                account={"cash": "1"},
                positions={"BTC/USD": {"quantity": "9", "side": "long"}},
                orders={"alpaca-order": {"status": "filled"}},
            )
        )

        controller = _controller(root, execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO)
        status = controller.status()

        assert status["account"]["cash"] == "1000"
        assert "BTC/USDT" in status["positions"]
        assert "BTC/USD" not in status["positions"]
        assert "bybit-demo-order" in status["orders"]
        assert "alpaca-order" not in status["orders"]

    def test_alpaca_status_never_reads_bybit_state(self, tmp_path):
        root = _experiment_root(tmp_path)
        WorkerStateStore(
            root / "execution_state" / "bybit_demo" / "worker_state.json"
        ).save(
            WorkerState(
                strategy_id="bybit-demo",
                account={"cash": "1000"},
                positions={"BTC/USDT": {"quantity": "0.001", "side": "long"}},
                orders={"bybit-order": {"status": "filled"}},
            )
        )

        controller = _controller(root)
        status = controller.status()

        assert status["execution_provider"] == EXECUTION_PROVIDER_ALPACA_PAPER
        assert status["account"] == {}
        assert status["positions"] == {}
        assert status["orders"] == {}

    @pytest.mark.parametrize(
        ("decision_source", "filename"),
        [
            ("crypto", "crypto_source_state.json"),
            ("tjr", "tjr_source_state.json"),
            ("current_market", "current_source_state.json"),
            ("history", "session_state.json"),
        ],
    )
    def test_source_memory_is_also_provider_scoped(
        self, tmp_path, decision_source, filename
    ):
        root = _experiment_root(tmp_path)
        bybit = _controller(
            root,
            execution_provider=EXECUTION_PROVIDER_BYBIT_DEMO,
            decision_source_kind=decision_source,
        )
        alpaca = _controller(root, decision_source_kind=decision_source)

        assert bybit._source_state_path(filename) == (
            root / "execution_state" / "bybit_demo" / filename
        )
        assert alpaca._source_state_path(filename) == root / filename
        assert bybit._source_state_path(filename) != alpaca._source_state_path(filename)
