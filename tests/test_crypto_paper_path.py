"""The crypto paper path: the TJR rule chain over live crypto market data.

This is the crypto counterpart of ``tests/test_phase30_tjr_paper_source.py``. It
pins the properties that make the crypto path *the same* strategy on a *different*
market, without touching the network (the suite's autouse guard refuses it):

1. the strategy is the **unchanged** TJR rule chain — the crypto path builds the
   same :class:`TjrDecisionSource` and the same :class:`TjrStrategy`, so there is
   exactly one strategy in the project;
2. the decision data is **real Binance public market data** (no key), sliced to
   bars that had *closed* at the decision instant, freshness-checked and
   future-checked;
3. the session clock is 24/7: the gate reads *open* from a readable clock and
   *unknown* (never *open*) from an unreadable one;
4. the price seam is a **current** crypto price, validated for freshness, so a
   stale ticker sizes no order;
5. the risk caps are the project's own — a $25 order ceiling, 5 positions, 10
   orders a day, long-only, fractional — applied to the **shared** book;
6. the path is **paper-only**: the live host is refused and no order is placed
   without an approved decision;
7. the source writes only its own state file, never the experiment's.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tradingagents.paper_experiment.broker import (
    DECISION_SOURCE_CHOICES,
    DECISION_SOURCE_CRYPTO,
    DECISION_SOURCE_TJR,
    MARKET_DATA_CHOICES,
    NO_TRADE_MARKET_DATA_UNAVAILABLE,
    PROVIDER_BINANCE_SNAPSHOT,
    BinanceSnapshotMarketData,
    MarketDataError,
)
from tradingagents.paper_experiment.config import ExperimentConfig
from tradingagents.paper_experiment.crypto_intraday_data import (
    CryptoIntradayBars,
    CryptoIntradayDataError,
    StaticCryptoIntradayBars,
)
from tradingagents.paper_experiment.crypto_market_hours import (
    CryptoMarketHours,
    StaticCryptoMarketHours,
)
from tradingagents.paper_experiment.experiment import PaperExperiment
from tradingagents.paper_experiment.market_data_wiring import (
    describe_decision_source,
    describe_provider,
    market_data_factory_for,
    resolve_decision_source,
)
from tradingagents.paper_experiment.tjr_source import (
    TjrBookSnapshot,
    TjrDecisionSource,
)
from tradingagents.risk.paper_parity import (
    MAX_DAILY_ORDERS,
    MAX_OPEN_POSITIONS,
    MAX_ORDER_NOTIONAL,
    paper_parity_limits,
)
from tradingagents.tjr.sessions import utc_day_session

SYMBOL = "BTC/USDT"
OTHER = "ETH/USDT"

SESSION = "2026-01-01"
"""A crypto session is the UTC calendar day — the one boundary every venue agrees on."""

_AS_OF = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame(start: datetime, bars: int, *, minutes: int, price: float = 100.0) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [start + timedelta(minutes=minutes * i) for i in range(bars)], tz="UTC"
    )
    return pd.DataFrame(
        {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1.0},
        index=index,
    )


def _frames(symbol: str = SYMBOL) -> dict[str, dict[str, pd.DataFrame]]:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    return {
        symbol: {
            "1m": _frame(start, 720, minutes=1),
            "5m": _frame(start, 144, minutes=5),
        }
    }


def _data(*, at: datetime = _AS_OF, symbol: str = SYMBOL) -> StaticCryptoIntradayBars:
    """A static crypto intraday seam whose clock matches the decision instant."""
    return StaticCryptoIntradayBars(
        _frames(symbol), max_stale_minutes=None, clock=lambda: at
    )


def _clock(*, session: str = SESSION, error: str | None = None):
    return StaticCryptoMarketHours(session=session, error=error)


def _source(
    *,
    data=None,
    market_hours=None,
    book: TjrBookSnapshot | None = None,
    price: float | None = None,
    symbols=(SYMBOL,),
    state_path=None,
    is_consumed=None,
    clock=None,
    session: str = SESSION,
) -> TjrDecisionSource:
    instant = clock() if clock is not None else _AS_OF
    if data is None:
        data = _data(at=instant)
    return TjrDecisionSource(
        symbols=tuple(symbols),
        data=data,
        market_hours=market_hours if market_hours is not None else _clock(session=session),
        order_notional=25.0,
        book_provider=(lambda: book) if book is not None else None,
        price_provider=(lambda _s: price) if price is not None else None,
        state_path=state_path,
        is_consumed=is_consumed,
        clock=lambda: instant,
    )


class _FakeTicker:
    """A ticker shaped like the crypto layer's, for the price seam."""

    def __init__(self, *, last, observed_at, bid=None, ask=None):
        self.last = last
        self.bid = bid
        self.ask = ask
        self.observed_at = observed_at

    @property
    def mid(self):
        if self.bid is None or self.ask is None or self.ask < self.bid:
            return None
        return (self.bid + self.ask) / 2.0


class _FakeTickerClient:
    def __init__(self, tickers, *, error=None):
        self._tickers = tickers
        self._error = error

    def ticker_24h(self, symbols):
        if self._error is not None:
            from tradingagents.crypto.transport import CryptoDataError

            raise CryptoDataError(self._error)
        return {s: self._tickers[s] for s in symbols if s in self._tickers}

    def describe(self):
        return {"crypto_data_provider": "fake", "crypto_data_requires_key": False}


# ---------------------------------------------------------------------------
# 1. The crypto path is the SAME strategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCryptoReusesTheTjrChain:
    def test_the_crypto_source_is_the_same_tjr_source_class(self):
        from tradingagents.tjr.strategy import TjrStrategy

        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.0)
        assert isinstance(source, TjrDecisionSource)
        # The rule chain the crypto path runs is the unchanged TJR one.
        assert source.settings.__class__.__name__ == "TjrSettings"
        assert TjrStrategy.__module__ == "tradingagents.tjr.strategy"

    def test_the_crypto_source_is_opt_in(self):
        assert DECISION_SOURCE_CRYPTO == "crypto"
        assert DECISION_SOURCE_CRYPTO in DECISION_SOURCE_CHOICES
        # The default is unchanged: history is still first and the default.
        assert DECISION_SOURCE_CHOICES[0] == "history"

    def test_the_session_policy_is_the_utc_day(self):
        policy = utc_day_session()
        assert policy.label
        # A 24/7 session: every UTC instant belongs to a session, none is closed.
        instant = pd.Timestamp("2026-01-01 03:07", tz="UTC")
        assert policy.session_of(instant) == "2026-01-01"
        assert policy.contains(instant, "2026-01-01") is True

    def test_the_crypto_source_describes_itself_without_a_secret(self):
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.0)
        view = source.describe()
        assert view["decision_source_kind"] == "tjr"
        json.dumps(view)


# ---------------------------------------------------------------------------
# 2. The decision data is closed, fresh, real-shaped crypto bars
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCryptoDecisionData:
    def test_only_bars_closed_by_the_instant_are_served(self):
        as_of = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)
        client = _FakeClient(_frames())
        seam = CryptoIntradayBars(client=client, max_stale_minutes=None, clock=lambda: as_of)
        five, one = seam.series(SYMBOL, session=SESSION, as_of=as_of)
        assert (five.frame.index + pd.Timedelta(minutes=5) <= as_of).all()
        assert (one.frame.index + pd.Timedelta(minutes=1) <= as_of).all()

    def test_the_seam_reads_the_venue_and_needs_no_key(self):
        client = _FakeClient(_frames())
        seam = CryptoIntradayBars(client=client, max_stale_minutes=None)
        described = seam.describe()
        assert described["crypto_data_requires_key"] is False
        assert described["tjr_data_session_policy"] == utc_day_session().label

    def test_a_stale_feed_is_refused(self):
        # The decision instant is hours after the newest bar closed.
        as_of = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        frames = {
            SYMBOL: {
                "1m": _frame(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), 60, minutes=1),
                "5m": _frame(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), 12, minutes=5),
            }
        }
        client = _FakeClient(frames)
        seam = CryptoIntradayBars(client=client, max_stale_minutes=20.0, clock=lambda: as_of)
        with pytest.raises(CryptoIntradayDataError) as err:
            seam.series(SYMBOL, session=SESSION, as_of=as_of)
        assert err.value.kind == "stale"
        assert err.value.code == NO_TRADE_MARKET_DATA_UNAVAILABLE

    def test_a_transport_failure_is_a_data_failure_not_a_signal(self):
        as_of = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        client = _FakeClient({}, error="network down")
        seam = CryptoIntradayBars(client=client, max_stale_minutes=None, clock=lambda: as_of)
        with pytest.raises(CryptoIntradayDataError) as err:
            seam.series(SYMBOL, session=SESSION, as_of=as_of)
        assert err.value.kind == "transport"
        # The worker's closed vocabulary code, never a prose-only reason.
        assert err.value.code == NO_TRADE_MARKET_DATA_UNAVAILABLE


class _FakeClient:
    provider_name = "fake"
    kind = "vendor_rest"

    def __init__(self, frames, *, error=None):
        self._frames = frames
        self._error = error

    def describe(self):
        return {"crypto_data_provider": self.provider_name, "crypto_data_requires_key": False}

    def klines(self, symbol, interval, *, limit=1000, start=None, end=None):
        if self._error is not None:
            from tradingagents.crypto.transport import CryptoDataError

            raise CryptoDataError(self._error)
        from tradingagents.crypto.types import Candle

        frame = self._frames[symbol][interval]
        candles = []
        for stamp, row in frame.iterrows():
            step = timedelta(minutes=1 if interval == "1m" else 5)
            candles.append(
                Candle(
                    symbol=symbol,
                    interval=interval,
                    open_time=stamp.to_pydatetime(),
                    close_time=(stamp + step).to_pydatetime(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 1.0)),
                )
            )
        return candles


# ---------------------------------------------------------------------------
# 3. The 24/7 clock
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCryptoClock:
    def test_the_market_is_always_open(self):
        status = CryptoMarketHours(clock=lambda: _AS_OF).status()
        assert status.state == "open"
        assert status.is_open is True
        assert status.session == SESSION

    def test_an_unreadable_clock_is_unknown_never_open(self):
        def boom():
            raise RuntimeError("clock down")

        status = CryptoMarketHours(clock=boom).status()
        assert status.state == "unknown"
        assert status.is_open is None

    def test_the_session_is_the_utc_day_not_midnight_open(self):
        # 03:07 UTC is mid-session for a 24/7 market: the session exists, but
        # nothing about it is an "opening bell".
        instant = pd.Timestamp("2026-01-01 03:07", tz="UTC")
        policy = utc_day_session()
        assert policy.session_of(instant) == "2026-01-01"
        assert policy.contains(instant, "2026-01-01") is True
        assert policy.windows == ()


# ---------------------------------------------------------------------------
# 4. The current-price seam
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBinancePriceSeam:
    def test_a_fresh_ticker_prices_the_order(self):
        client = _FakeTickerClient(
            {SYMBOL: _FakeTicker(last=100.0, observed_at=_AS_OF)}
        )
        seam = BinanceSnapshotMarketData([SYMBOL], client=client, clock=lambda: _AS_OF)
        assert seam.price(SYMBOL) == 100.0
        assert seam.available() is True
        assert seam.last_status == "ok"

    def test_a_stale_ticker_sizes_no_order(self):
        stale = _AS_OF - timedelta(seconds=3600)
        client = _FakeTickerClient({SYMBOL: _FakeTicker(last=100.0, observed_at=stale)})
        seam = BinanceSnapshotMarketData([SYMBOL], client=client, clock=lambda: _AS_OF)
        with pytest.raises(MarketDataError) as err:
            seam.price(SYMBOL)
        assert "stale" in str(err.value)
        assert seam.last_status == "stale"

    def test_a_future_dated_ticker_sizes_no_order(self):
        future = _AS_OF + timedelta(seconds=600)
        client = _FakeTickerClient({SYMBOL: _FakeTicker(last=100.0, observed_at=future)})
        seam = BinanceSnapshotMarketData([SYMBOL], client=client, clock=lambda: _AS_OF)
        with pytest.raises(MarketDataError) as err:
            seam.price(SYMBOL)
        assert "future" in str(err.value)
        assert seam.last_status == "future"

    def test_a_missing_ticker_sizes_no_order(self):
        client = _FakeTickerClient({})
        seam = BinanceSnapshotMarketData([SYMBOL], client=client, clock=lambda: _AS_OF)
        with pytest.raises(MarketDataError) as err:
            seam.price(SYMBOL)
        assert "no ticker" in str(err.value)
        assert seam.last_status == "no_data"

    def test_a_nan_price_is_refused(self):
        client = _FakeTickerClient({SYMBOL: _FakeTicker(last=float("nan"), observed_at=_AS_OF)})
        seam = BinanceSnapshotMarketData([SYMBOL], client=client, clock=lambda: _AS_OF)
        with pytest.raises(MarketDataError):
            seam.price(SYMBOL)

    def test_the_quote_mid_is_used_when_no_last_trade(self):
        client = _FakeTickerClient(
            {
                SYMBOL: _FakeTicker(
                    last=None, observed_at=_AS_OF, bid=99.0, ask=101.0
                )
            }
        )
        seam = BinanceSnapshotMarketData([SYMBOL], client=client, clock=lambda: _AS_OF)
        assert seam.price(SYMBOL) == 100.0

    def test_the_seam_is_opt_in_and_reads_no_credential(self):
        assert PROVIDER_BINANCE_SNAPSHOT == "binance_snapshot"
        assert PROVIDER_BINANCE_SNAPSHOT in MARKET_DATA_CHOICES
        seam = BinanceSnapshotMarketData([SYMBOL])
        described = seam.describe()
        assert "no key" in described["market_data_detail"]
        assert "is_current_price" in described


# ---------------------------------------------------------------------------
# 5. The risk caps are the project's own
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCryptoRiskCaps:
    def test_the_caps_are_the_projects_own(self):
        limits = paper_parity_limits()
        assert limits.max_order_value == MAX_ORDER_NOTIONAL == 25.0
        assert limits.max_open_positions == MAX_OPEN_POSITIONS == 5
        assert limits.max_daily_orders == MAX_DAILY_ORDERS == 10
        assert limits.long_only is True
        assert limits.fractional_shares is True

    def test_the_crypto_source_sorts_against_the_shared_book(self):
        # A book with cash and holdings: the source sizes against THIS book, so a
        # second symbol cannot each assume a fresh 5000 USD.
        book = TjrBookSnapshot(cash=5000.0, positions={OTHER: 1.0}, daily_orders=3)
        source = _source(book=book, price=100.0)
        # The source carries the book's own figures; it does not invent cash.
        snapshot = source.book_provider()
        assert snapshot.cash == 5000.0
        assert snapshot.positions == {OTHER: 1.0}
        assert snapshot.daily_orders == 3

    def test_no_book_means_no_order(self):
        source = _source(book=None, price=100.0)
        # No book supplied: the RiskEngine is handed an empty book and refuses to
        # size, so a decision cannot become an order.
        decision = source.next_decision()
        assert decision is None or decision.quantity is None


# ---------------------------------------------------------------------------
# 6. The wiring: CLI choices, describe, factories
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCryptoWiring:
    def test_the_cli_resolves_the_crypto_source(self):
        assert resolve_decision_source(DECISION_SOURCE_CRYPTO) == DECISION_SOURCE_CRYPTO

    def test_the_crypto_source_is_described_without_a_secret(self):
        described = describe_decision_source(DECISION_SOURCE_CRYPTO)
        assert "TJR rule chain" in described
        assert "no API key" in described
        assert "24/7" in described

    def test_the_binance_provider_is_described(self):
        described = describe_provider(PROVIDER_BINANCE_SNAPSHOT)
        assert "CURRENT crypto price" in described
        assert "no API key" in described

    def test_the_binance_factory_builds_the_crypto_seam(self):
        factory = market_data_factory_for(PROVIDER_BINANCE_SNAPSHOT)
        assert factory is not None
        seam = factory((SYMBOL,))
        assert isinstance(seam, BinanceSnapshotMarketData)

    def test_the_historical_provider_still_builds_nothing(self):
        assert market_data_factory_for("bar_open") is None


# ---------------------------------------------------------------------------
# 7. Paper-only: no live host is reachable from the crypto path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCryptoIsPaperOnly:
    def test_the_live_host_is_refused(self):
        from tradingagents.paper_experiment.broker.alpaca_paper import (
            LiveEndpointRefusedError,
            assert_paper_endpoint,
        )

        with pytest.raises(LiveEndpointRefusedError):
            assert_paper_endpoint("https://api.alpaca.markets")

    def test_the_paper_host_is_accepted(self):
        from tradingagents.paper_experiment.broker.alpaca_paper import (
            PAPER_BASE_URL,
            assert_paper_endpoint,
        )

        assert assert_paper_endpoint(PAPER_BASE_URL) == PAPER_BASE_URL


# ---------------------------------------------------------------------------
# 8. The controller builds the crypto source offline
# ---------------------------------------------------------------------------


@pytest.mark.unit
def _experiment_root(tmp_path):
    """A persisted experiment with a frame-backed (restorable) price source."""
    from tradingagents.paper_experiment.inference import StubInference
    from tradingagents.paper_experiment.sources import StaticPriceSource

    root = tmp_path / "exp"
    index = pd.date_range(end="2026-10-29", periods=6, freq="B")
    frame = pd.DataFrame(
        {
            "open": [100.0 + i for i in range(6)],
            "high": [102.0 + i for i in range(6)],
            "low": [99.0 + i for i in range(6)],
            "close": [101.0 + i for i in range(6)],
            "volume": [1_000_000.0] * 6,
        },
        index=index,
    )
    config = ExperimentConfig.from_protocol(
        symbols=(SYMBOL,), decision_frequency_sessions=1, evaluation_lag_sessions=1
    )
    source = StaticPriceSource({SYMBOL: frame}, name="static_test")
    experiment = PaperExperiment(
        source=source,
        config=config,
        root=root,
        inference=StubInference(),
    )
    experiment.ensure_created()
    for session in list(source.sessions())[:1]:
        experiment.process_session(session)
    return root


class TestControllerCryptoWiring:

    def test_the_controller_builds_a_crypto_source_with_injected_seams(self, tmp_path):
        from tradingagents.paper_experiment.dashboard.paper_control import (
            PaperTradingController,
        )

        root = _experiment_root(tmp_path)
        controller = PaperTradingController(
            root,
            enabled=True,
            decision_source_kind=DECISION_SOURCE_CRYPTO,
            crypto_market_hours_gate=lambda: _clock(),
            crypto_data_factory=lambda: _data(),
        )
        experiment = PaperExperiment.load(root, inference=None)
        holder: dict = {}
        source = controller._build_crypto_source(experiment, holder)  # noqa: SLF001
        assert isinstance(source, TjrDecisionSource)
        assert source.symbols == (SYMBOL,)
        # Its own state file — never the experiment's, and never the equity TJR
        # source's file, so it is not a second writer to the experiment directory.
        assert source._store == root / "crypto_source_state.json"  # noqa: SLF001

    def test_the_controller_status_does_not_build_the_crypto_source(self, tmp_path):
        from tradingagents.paper_experiment.dashboard.paper_control import (
            PaperTradingController,
        )

        root = _experiment_root(tmp_path)
        controller = PaperTradingController(
            root,
            enabled=True,
            decision_source_kind=DECISION_SOURCE_CRYPTO,
        )
        status = controller.status()
        assert status["status"] == "STOPPED"
        assert status["decision_source_configured"] == DECISION_SOURCE_CRYPTO


class TestTheDashboardShowsTheChainsOwnEvaluation:
    """The dashboard's strategy section must show the chain's real answer.

    The view is wired to the decision source's persisted records, so it shows
    the stage the chain reached and the reason it stopped. It must never
    evaluate the chain a second time (a second evaluation fetches a second
    window and could disagree with the one the worker acted on), and it must
    never build the worker (a status poll that builds the worker freezes the
    live view).
    """

    def _controller(self, root, **kwargs):
        from tradingagents.paper_experiment.dashboard.paper_control import (
            PaperTradingController,
        )

        return PaperTradingController(
            root,
            enabled=True,
            decision_source_kind=DECISION_SOURCE_CRYPTO,
            **kwargs,
        )

    def test_it_reads_the_crypto_sources_own_state_file(self, tmp_path):
        root = _experiment_root(tmp_path)
        (root / "crypto_source_state.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "records": {
                        SYMBOL: {
                            "symbol": SYMBOL,
                            "side": "buy",
                            "stage": "ENTRY",
                            "reason_code": "TJR_ENTRY",
                            "detail": "1m BOS confirmed the entry",
                            "stop": 98.0,
                            "target": 104.0,
                            "reference_price": 100.0,
                            "evaluated_at": "2026-01-01T12:00:00+00:00",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        evaluations = self._controller(root).strategy_evaluations()
        row = evaluations[SYMBOL]
        assert row["stage"] == "ENTRY"
        assert row["direction"] == "long"
        assert row["entry"] == 100.0
        assert row["stop_loss"] == 98.0
        assert row["take_profit"] == 104.0
        # The ratio is derived, never invented: reward 4 over risk 2.
        assert row["risk_reward"] == 2.0
        assert row["actionable"] is True

    def test_a_no_trade_keeps_its_stage_and_has_no_levels(self, tmp_path):
        root = _experiment_root(tmp_path)
        (root / "crypto_source_state.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "records": {
                        SYMBOL: {
                            "symbol": SYMBOL,
                            "side": None,
                            "stage": "SWEEP",
                            "reason_code": "NO_5M_REVERSAL",
                            "detail": "the 5m close did not reverse the sweep",
                            "stop": None,
                            "target": None,
                            "reference_price": None,
                            "evaluated_at": "2026-01-01T12:00:00+00:00",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        row = self._controller(root).strategy_evaluations()[SYMBOL]
        assert row["stage"] == "SWEEP"
        assert row["reason_code"] == "NO_5M_REVERSAL"
        assert row["direction"] is None
        assert row["actionable"] is False
        # An undefined ratio is None, never zero.
        assert row["risk_reward"] is None

    def test_it_reports_nothing_when_the_source_has_not_evaluated(self, tmp_path):
        root = _experiment_root(tmp_path)
        assert self._controller(root).strategy_evaluations() == {}

    def test_it_survives_a_corrupt_state_file(self, tmp_path):
        root = _experiment_root(tmp_path)
        (root / "crypto_source_state.json").write_text("{not json", encoding="utf-8")
        assert self._controller(root).strategy_evaluations() == {}

    def test_it_reads_the_file_of_the_configured_source(self, tmp_path):
        """It must not show another source's evaluation."""
        from tradingagents.paper_experiment.dashboard.paper_control import (
            PaperTradingController,
        )

        root = _experiment_root(tmp_path)
        (root / "tjr_source_state.json").write_text(
            json.dumps({"records": {SYMBOL: {"stage": "TJR_ONLY", "side": None}}}),
            encoding="utf-8",
        )
        controller = PaperTradingController(
            root, enabled=True, decision_source_kind=DECISION_SOURCE_TJR
        )
        assert controller.strategy_evaluations()[SYMBOL]["stage"] == "TJR_ONLY"

        crypto = PaperTradingController(
            root, enabled=True, decision_source_kind=DECISION_SOURCE_CRYPTO
        )
        # The crypto controller reads the crypto file, which does not exist.
        assert crypto.strategy_evaluations() == {}

    def test_reading_it_does_not_build_the_worker(self, tmp_path):
        """A poll that builds the worker freezes the live view."""
        root = _experiment_root(tmp_path)
        controller = self._controller(root)
        controller.strategy_evaluations()
        assert controller._worker is None  # noqa: SLF001


class TestTheControllerRestoresTheExperimentSource:
    """The controller must load the experiment *with* its price source.

    The frozen dataset carries no persisted bars (it reloads from disk), so a
    load that omitted the source failed with "this experiment's state carries no
    restorable price bars" — and the paper status reported an error even though
    the experiment was perfectly resumable. The runner's shared helper rebuilds
    the source honestly for both cases.
    """

    def test_the_controller_loads_the_experiment_without_error(self, tmp_path):
        from tradingagents.paper_experiment.dashboard.paper_control import (
            PaperTradingController,
        )

        root = _experiment_root(tmp_path)
        controller = PaperTradingController(
            root,
            enabled=True,
            decision_source_kind=DECISION_SOURCE_CRYPTO,
        )
        experiment = controller._load_experiment()  # noqa: SLF001
        assert experiment.config.symbols

    def test_the_paper_status_is_stopped_not_an_error(self, tmp_path):
        from tradingagents.paper_experiment.dashboard.paper_control import (
            PaperTradingController,
        )

        root = _experiment_root(tmp_path)
        controller = PaperTradingController(
            root,
            enabled=True,
            decision_source_kind=DECISION_SOURCE_CRYPTO,
        )
        status = controller.status()
        assert status["status"] == "STOPPED"
        assert status["last_error"] is None

    def test_the_status_still_does_not_build_the_worker(self, tmp_path):
        """Restoring the source must not start a worker or place an order."""
        from tradingagents.paper_experiment.dashboard.paper_control import (
            PaperTradingController,
        )

        root = _experiment_root(tmp_path)
        controller = PaperTradingController(
            root,
            enabled=True,
            decision_source_kind=DECISION_SOURCE_CRYPTO,
        )
        status = controller.status()
        assert status["loop_running"] is False
        assert status["cycles_completed"] == 0
        assert status["orders"] == {}
        assert status["positions"] == {}
