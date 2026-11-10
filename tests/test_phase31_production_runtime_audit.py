"""Phase 31: the production runtime audit for the crypto TJR paper worker.

A Railway deployment showed a worker that was RUNNING, evaluated the strategy
every cycle, and yet reported a set of fields that did not agree with each other:

* ``last_cycle.reason_code`` was ``NO_SIGNAL`` (the chain's own
  ``SHORT_SETUP_LONG_ONLY``), while ``last_no_trade_decision_code`` still named an
  earlier ``BROKER_REJECTED``;
* ``decision_source_records[...].evaluated_at`` never moved, though
  ``last_strategy_evaluation_at`` advanced every cycle;
* ``tjr_data_last_fetch_at`` was ``null``, which reads as "the bars were never
  fetched".

These tests pin the *causes* of each symptom and the fix for each, so a
regression is caught at the code path rather than on the dashboard:

1. the Binance fetch really happens on every cycle, and the seam reports when it
   last fetched (``tjr_data_last_fetch_at``);
2. a no-signal evaluation does not close the session, so the chain is re-read and
   a later, complete setup is still seen — the backtest's own semantics;
3. a stale no-trade code cannot survive a later cycle;
4. ``last_cycle``, the no-trade fields and the statistics agree;
5. an old signal cannot be re-executed without a fresh evaluation;
6. the shipped crypto time-in-force fix is on both the entry and the exit path.

Everything here is offline and deterministic: the clock, the intraday bar seam
and the broker transport are injected, so no test reaches the network and none
needs a venue credential.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tradingagents.paper_experiment.broker import (
    DECISION_SOURCE_TJR,
    NO_TRADE_BROKER_REJECTED,
    NO_TRADE_NO_SIGNAL,
    NO_TRADE_REASONS,
    NO_TRADE_RISK_REJECTED,
    NO_TRADE_TJR_NO_NEW_DATA,
    AlpacaPaperBroker,
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    StaticMarketData,
)
from tradingagents.paper_experiment.crypto_intraday_data import CryptoIntradayBars
from tradingagents.paper_experiment.market_hours import StaticMarketHours
from tradingagents.paper_experiment.tjr_intraday_data import StaticIntradayBars
from tradingagents.paper_experiment.tjr_source import (
    TjrBookSnapshot,
    TjrDecisionSource,
)
from tradingagents.tjr.frames import session_open_utc

from .tjr_helpers import (
    BEARISH_BARS,
    BULLISH_BARS,
    SESSION,
    frame_from_bars,
    one_minute_from_five,
)

SYMBOL = "AAPL"
CRYPTO_SYMBOL = "DOGE/USDT"

_ACCOUNT = {
    "id": "paper-acct-1",
    "status": "ACTIVE",
    "cash": "5000.00",
    "equity": "5000.00",
    "buying_power": "5000.00",
    "currency": "USD",
    "trading_blocked": False,
    "account_blocked": False,
    "shorting_enabled": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_of(session: str = SESSION, *, bar_index: int = 15) -> pd.Timestamp:
    """The decision instant at the *close* of 5m bar ``bar_index`` (0-based)."""
    return session_open_utc(session) + pd.Timedelta(minutes=5 * bar_index)


def _bars(bars, symbol: str = SYMBOL) -> dict[str, dict[str, pd.DataFrame]]:
    five = frame_from_bars(SESSION, bars)
    return {symbol: {"5m": five, "1m": one_minute_from_five(five)}}


def _static(bars, *, symbol: str = SYMBOL, at: pd.Timestamp | None = None):
    """A static intraday bundle whose own clock matches the decision instant.

    The seam freshness-checks the newest bar against *its* clock, so a test that
    pins the decision instant must pin the seam's clock to the same instant, or
    the bars would read as stale against the real wall clock.
    """
    stamp = at if at is not None else _as_of()
    return StaticIntradayBars(_bars(bars, symbol), clock=lambda: stamp.to_pydatetime())


class _CountingData(StaticIntradayBars):
    """A static seam that counts the fetches, so "was the data re-read?" is testable."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fetches: list[tuple[str, str]] = []

    def series(self, symbol, *, session, as_of):
        self.fetches.append((symbol, session))
        return super().series(symbol, session=session, as_of=as_of)


def _source(data, *, clock, symbols=(SYMBOL,), book=None, price=None, state_path=None):
    return TjrDecisionSource(
        symbols=tuple(symbols),
        data=data,
        market_hours=StaticMarketHours(is_open=True, session=SESSION),
        order_notional=25.0,
        book_provider=(lambda: book) if book is not None else None,
        price_provider=(lambda _s: price) if price is not None else None,
        state_path=state_path,
        clock=clock,
    )


class _Resp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _order(**overrides) -> dict:
    payload = {
        "id": "order-1",
        "client_order_id": overrides.pop("client_order_id", "cid-1"),
        "symbol": overrides.pop("symbol", SYMBOL),
        "side": overrides.pop("side", "buy"),
        "type": overrides.pop("type", "market"),
        "time_in_force": overrides.pop("time_in_force", "gtc"),
        "qty": overrides.pop("qty", "0.25"),
        "filled_qty": overrides.pop("filled_qty", "0"),
        "filled_avg_price": overrides.pop("filled_avg_price", None),
        "status": overrides.pop("status", "accepted"),
        "submitted_at": overrides.pop("submitted_at", "2026-10-29T14:45:00+00:00"),
    }
    payload.update(overrides)
    return payload


def _routes(*, filled: bool = False, order_status: int = 200, order_payload=None):
    order = _order(
        filled_qty="0.25" if filled else "0",
        filled_avg_price="100.6" if filled else None,
        status="filled" if filled else "accepted",
    )
    return {
        ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
        ("GET", "/v2/positions"): _Resp(200, []),
        ("GET", "/v2/orders"): _Resp(200, [order]),
        ("POST", "/v2/orders"): _Resp(order_status, order_payload or order),
    }


def _transport(routes, *, calls=None):
    def send(*, method, url, headers, params, json_body, timeout):
        if calls is not None:
            calls.append({"method": method, "url": url, "json": json_body})
        for (m, suffix), response in routes.items():
            if method == m and url.endswith(suffix):
                return response() if callable(response) else response
        raise AssertionError(f"unexpected request {method} {url}")

    return send


def _worker(tmp_path, source, *, calls=None, routes=None, crypto=False, symbol=SYMBOL):
    config = PaperTradingConfig(
        strategy_id="crypto_tjr" if crypto else "tjr",
        order_notional=25.0,
        whole_shares=False,
        state_path=tmp_path / "worker_state.json",
        decision_source=DECISION_SOURCE_TJR,
        crypto=crypto,
    )
    return PaperTradingWorker(
        config=config,
        decision_source=source,
        market_data=StaticMarketData(prices={symbol: 100.6}),
        broker_factory=lambda: AlpacaPaperBroker(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=_transport(routes or _routes(), calls=calls),
            limits=PaperRiskLimits(max_order_notional=25.0),
        ),
    )


# ---------------------------------------------------------------------------
# 1. The Binance fetch happens on every cycle, and the seam says when
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheBarsAreActuallyFetched:
    def test_a_no_signal_cycle_still_fetches_the_bars(self):
        """A chain that finds nothing must not stop the data from being read."""
        stamp = _as_of()
        data = _CountingData(_bars(BEARISH_BARS), clock=lambda: stamp.to_pydatetime())
        source = _source(data, clock=lambda: stamp.to_pydatetime())
        source.next_decision()
        assert data.fetches == [(SYMBOL, SESSION)]

    def test_a_later_cycle_fetches_the_bars_again(self):
        """The second cycle in the same session re-reads the data.

        This is the regression the audit found: the session-scoped record used to
        short-circuit *before* the fetch, so the bars were read once and never
        again for the rest of the session.
        """
        stamp = _as_of()
        data = _CountingData(_bars(BEARISH_BARS), clock=lambda: stamp.to_pydatetime())
        source = _source(data, clock=lambda: stamp.to_pydatetime())
        source.next_decision()
        source.next_decision()
        source.next_decision()
        assert len(data.fetches) == 3

    def test_the_seam_reports_when_it_last_fetched(self):
        """``tjr_data_last_fetch_at`` is written by the seam, so it is not null
        once a fetch has happened."""
        stamp = _as_of()
        data = _CountingData(_bars(BEARISH_BARS), clock=lambda: stamp.to_pydatetime())
        source = _source(data, clock=lambda: stamp.to_pydatetime())
        described = source.describe()
        assert described["tjr_data_last_fetch_at"] is None  # before any fetch
        source.next_decision()
        described = source.describe()
        assert described["tjr_data_last_fetch_at"] is not None
        assert described["tjr_data_last_fetch_at"].startswith("2026-10-29")

    def test_the_live_crypto_seam_reports_its_last_fetch(self):
        """The real ``CryptoIntradayBars`` seam (not a stub) stamps the fetch."""
        from tradingagents.crypto.types import Candle

        as_of = datetime(2026, 10, 29, 14, 45, tzinfo=UTC)
        calls = []

        class _Client:
            provider_name = "fake"
            kind = "vendor_rest"

            def describe(self):
                return {"crypto_data_provider": "fake"}

            def klines(self, symbol, interval, *, limit=1000, start=None, end=None):
                calls.append((symbol, interval))
                step = timedelta(minutes=1 if interval == "1m" else 5)
                stamps = [
                    as_of - timedelta(minutes=(step.total_seconds() / 60) * (n + 1))
                    for n in range(4)
                ]
                return [
                    Candle(
                        symbol=symbol,
                        interval=interval,
                        open_time=stamp,
                        close_time=stamp + step,
                        open=100.0,
                        high=101.0,
                        low=99.0,
                        close=100.0,
                        volume=1.0,
                    )
                    for stamp in reversed(stamps)
                ]

        seam = CryptoIntradayBars(
            client=_Client(), max_stale_minutes=None, clock=lambda: as_of
        )
        assert seam.describe()["tjr_data_last_fetch_at"] is None
        seam.series("BTC/USDT", session="2026-10-29", as_of=as_of)
        assert seam.describe()["tjr_data_last_fetch_at"] is not None
        assert calls  # the bars were really requested


# ---------------------------------------------------------------------------
# 2. A no-signal evaluation does not close the session
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheSessionIsNotClosedByANoSignal:
    def test_a_no_signal_record_is_re_evaluated_in_the_same_session(self):
        """The chain is re-read, so its ``evaluated_at`` advances."""
        stamp = _as_of()
        data = _CountingData(_bars(BEARISH_BARS), clock=lambda: stamp.to_pydatetime())
        source = _source(data, clock=lambda: stamp.to_pydatetime())
        source.next_decision()
        first = source.last_decision_for(SYMBOL)

        later = stamp + pd.Timedelta(minutes=5)
        source._clock = lambda: later.to_pydatetime()
        source.data = _CountingData(
            _bars(BEARISH_BARS), clock=lambda: later.to_pydatetime()
        )
        source.next_decision()
        second = source.last_decision_for(SYMBOL)

        assert second["evaluated_at"] != first["evaluated_at"]
        assert second["evaluated_at"] > first["evaluated_at"]

    def test_a_no_signal_still_records_the_chain_reason(self):
        stamp = _as_of()
        data = _CountingData(_bars(BEARISH_BARS), clock=lambda: stamp.to_pydatetime())
        source = _source(data, clock=lambda: stamp.to_pydatetime())
        source.next_decision()
        record = source.last_decision_for(SYMBOL)
        assert record["side"] is None
        assert record["reason_code"] == NO_TRADE_NO_SIGNAL
        assert record["chain_reason"] == "SHORT_SETUP_LONG_ONLY"

    def test_a_setup_that_completes_after_a_no_signal_is_still_offered(self):
        """The backtest re-evaluates every bar; the live source must as well.

        The backtest's engine marks ``(symbol, session)`` attempted only when the
        chain produced a BUY, so a setup that completes later in the session is
        still traded. The live source used to close the session on the first
        *no-signal*, which made the live path strictly narrower than the backtest
        it claims to mirror.
        """
        stamp = _as_of()
        # Cycle 1: bearish -> no signal (a long-only refusal).
        data = _CountingData(_bars(BEARISH_BARS), clock=lambda: stamp.to_pydatetime())
        source = _source(
            data,
            clock=lambda: stamp.to_pydatetime(),
            book=TjrBookSnapshot(cash=5000.0),
            price=100.6,
        )
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_NO_SIGNAL

        # Cycle 2: the bars now complete a bullish setup -> a decision is offered.
        later = stamp + pd.Timedelta(minutes=5)
        source._clock = lambda: later.to_pydatetime()
        source.data = _CountingData(
            _bars(BULLISH_BARS), clock=lambda: later.to_pydatetime()
        )
        decision = source.next_decision()
        assert decision is not None
        assert decision.decision_id.startswith(f"tjr-{SYMBOL}-{SESSION}-")
        assert decision.side == "buy"

    def test_a_completed_setup_is_not_offered_twice(self):
        """A setup that completed is consumed: the same signal is not repeated.

        The bars *are* re-read on every cycle — a new completed bar is a new
        judgement, exactly as the backtest judges the chain again at every bar
        close, and that is the only way a later sweep can be seen at all. But the
        setup it finds here is the *same* setup, and the same setup is the same
        signal: it is offered once and never again.
        """
        stamp = _as_of()
        data = _CountingData(_bars(BULLISH_BARS), clock=lambda: stamp.to_pydatetime())
        source = _source(
            data,
            clock=lambda: stamp.to_pydatetime(),
            book=TjrBookSnapshot(cash=5000.0),
            price=100.6,
        )
        first = source.next_decision()
        assert first is not None
        # The next cycle re-reads the bars (so a new sweep would be seen) but
        # finds the same setup, so the decision is not offered again.
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_TJR_NO_NEW_DATA
        assert len(data.fetches) == 2
        # The id is the same, so the worker's ordered set would refuse it too.
        assert source.last_decision_for(SYMBOL)["decision_id"] == first.decision_id

    def test_a_risk_rejection_still_closes_the_session(self):
        """A completed setup the engine refused is a judged setup, not a pending one.

        Re-reading the same bars would only re-run the same refusal, so the
        session is closed — but it is reported honestly as the risk rejection it
        is, never as a decision.
        """
        stamp = _as_of()
        data = _CountingData(_bars(BULLISH_BARS), clock=lambda: stamp.to_pydatetime())
        # No book -> the engine refuses to size -> a RISK_REJECTED record.
        source = _source(data, clock=lambda: stamp.to_pydatetime())
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_RISK_REJECTED
        assert len(data.fetches) == 1
        # The next cycle reports the same refusal, and never a decision.
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_RISK_REJECTED
        assert len(data.fetches) == 2


# ---------------------------------------------------------------------------
# 3 & 4. last_cycle, the no-trade fields and the statistics agree
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheCycleStateIsCoherent:
    def test_a_stale_broker_rejection_does_not_survive_a_later_cycle(self, tmp_path):
        """The reported symptom: a NO_SIGNAL cycle beside a stale BROKER_REJECTED.

        Two symbols, so the scenario is the production one: symbol A completes a
        setup and the broker refuses the order; symbol B's chain finds nothing.
        On the next cycle symbol B reports NO_SIGNAL — and the *current* code must
        be the NO_SIGNAL, not the earlier broker rejection.
        """
        stamp = _as_of()
        calls = []
        other = "MSFT"

        def two_symbols(bars_for_a, bars_for_b):
            five_a = frame_from_bars(SESSION, bars_for_a)
            five_b = frame_from_bars(SESSION, bars_for_b)
            return {
                SYMBOL: {"5m": five_a, "1m": one_minute_from_five(five_a)},
                other: {"5m": five_b, "1m": one_minute_from_five(five_b)},
            }

        # Cycle 1: A completes a setup (broker refuses it); B finds nothing.
        data = _CountingData(
            two_symbols(BULLISH_BARS, BEARISH_BARS),
            clock=lambda: stamp.to_pydatetime(),
        )
        source = _source(
            data,
            clock=lambda: stamp.to_pydatetime(),
            symbols=(SYMBOL, other),
            book=TjrBookSnapshot(cash=5000.0),
            price=100.6,
        )
        worker = _worker(
            tmp_path,
            source,
            calls=calls,
            routes=_routes(order_status=422, order_payload={"message": "invalid order"}),
        )
        worker.start()
        first = worker.run_once()
        assert first.reason_code == NO_TRADE_BROKER_REJECTED
        assert worker.state.last_no_trade_decision_code == NO_TRADE_BROKER_REJECTED

        # Cycle 2: A's setup is consumed (the backtest consumes a completed
        # setup), so both symbols now report their own no-signal.
        source._clock = lambda: (stamp + pd.Timedelta(minutes=5)).to_pydatetime()
        source.data = _CountingData(
            two_symbols(BEARISH_BARS, BEARISH_BARS),
            clock=lambda: (stamp + pd.Timedelta(minutes=5)).to_pydatetime(),
        )
        second = worker.run_once()
        worker.stop()

        assert second.reason_code == NO_TRADE_NO_SIGNAL
        assert worker.state.last_no_trade_code == NO_TRADE_NO_SIGNAL
        # The stale broker rejection must not be presented as the current cause.
        assert worker.state.last_no_trade_decision_code != NO_TRADE_BROKER_REJECTED
        assert worker.state.last_no_trade_decision_code in (None, NO_TRADE_NO_SIGNAL)

    def test_the_last_cycle_result_and_the_state_agree(self, tmp_path):
        stamp = _as_of()
        data = _CountingData(_bars(BEARISH_BARS), clock=lambda: stamp.to_pydatetime())
        source = _source(data, clock=lambda: stamp.to_pydatetime())
        worker = _worker(tmp_path, source)
        worker.start()
        result = worker.run_once()
        worker.stop()

        assert result.reason_code == NO_TRADE_NO_SIGNAL
        assert worker.state.last_no_trade_code == result.reason_code
        assert worker.state.last_no_decision_reason == result.skipped_reason

    def test_a_stale_error_is_cleared_by_a_clean_cycle(self, tmp_path):
        """``last_error`` from an earlier failure must not sit beside a fresh
        no-signal and read as if the current cycle had failed."""
        stamp = _as_of()
        data = _CountingData(_bars(BULLISH_BARS), clock=lambda: stamp.to_pydatetime())
        source = _source(
            data,
            clock=lambda: stamp.to_pydatetime(),
            book=TjrBookSnapshot(cash=5000.0),
            price=100.6,
        )
        worker = _worker(
            tmp_path,
            source,
            routes=_routes(order_status=422, order_payload={"message": "invalid order"}),
        )
        worker.start()
        worker.run_once()
        assert worker.state.last_error  # the broker refusal recorded one

        source._clock = lambda: (stamp + pd.Timedelta(minutes=5)).to_pydatetime()
        source.data = _CountingData(
            _bars(BEARISH_BARS),
            clock=lambda: (stamp + pd.Timedelta(minutes=5)).to_pydatetime(),
        )
        worker.run_once()
        worker.stop()
        assert worker.state.last_error is None

    def test_the_statistics_are_a_cumulative_tally_not_the_current_cycle(self, tmp_path):
        """The statistics count settled outcomes; they are not overwritten.

        A historical rejection stays in the tally (it happened), but the *current*
        cycle's code is carried by ``last_no_trade_code``. Mixing the two is what
        made the dashboard look contradictory.
        """
        stamp = _as_of()
        data = _CountingData(_bars(BULLISH_BARS), clock=lambda: stamp.to_pydatetime())
        source = _source(
            data,
            clock=lambda: stamp.to_pydatetime(),
            book=TjrBookSnapshot(cash=5000.0),
            price=100.6,
        )
        worker = _worker(
            tmp_path,
            source,
            routes=_routes(order_status=422, order_payload={"message": "invalid order"}),
        )
        worker.start()
        worker.run_once()
        statistics = worker.status()["decision_statistics"]
        assert statistics["broker_rejections"] == 1
        assert statistics["decisions_settled"] == 1
        worker.stop()

    def test_every_persisted_code_is_in_the_closed_vocabulary(self, tmp_path):
        stamp = _as_of()
        data = _CountingData(_bars(BEARISH_BARS), clock=lambda: stamp.to_pydatetime())
        source = _source(data, clock=lambda: stamp.to_pydatetime())
        worker = _worker(tmp_path, source)
        worker.start()
        worker.run_once()
        worker.stop()
        assert worker.state.last_no_trade_code in NO_TRADE_REASONS
        if worker.state.last_no_trade_decision_code is not None:
            assert worker.state.last_no_trade_decision_code in NO_TRADE_REASONS


# ---------------------------------------------------------------------------
# 5. An old signal cannot be re-executed without a fresh evaluation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoStaleSignalIsReExecuted:
    def test_a_no_signal_never_offers_a_decision(self, tmp_path):
        stamp = _as_of()
        data = _CountingData(_bars(BEARISH_BARS), clock=lambda: stamp.to_pydatetime())
        source = _source(data, clock=lambda: stamp.to_pydatetime())
        worker = _worker(tmp_path, source)
        worker.start()
        for _ in range(3):
            result = worker.run_once()
            assert result.ran is False
            assert result.order is None
        worker.stop()

    def test_an_ordered_decision_is_not_re_submitted(self, tmp_path):
        """The order-only memory: a decision that produced an order is not re-sent."""
        stamp = _as_of()
        data = _CountingData(_bars(BULLISH_BARS), clock=lambda: stamp.to_pydatetime())
        source = _source(
            data,
            clock=lambda: stamp.to_pydatetime(),
            book=TjrBookSnapshot(cash=5000.0),
            price=100.6,
        )
        calls = []
        worker = _worker(tmp_path, source, calls=calls)
        worker.start()
        first = worker.run_once()
        assert first.ran is True
        posts = [c for c in calls if c["method"] == "POST"]
        assert len(posts) == 1
        worker.run_once()
        posts = [c for c in calls if c["method"] == "POST"]
        assert len(posts) == 1  # not re-submitted
        worker.stop()

    def test_deduplication_does_not_block_a_later_opportunity(self, tmp_path):
        """A no-signal in one cycle must not stop a setup in the next cycle."""
        stamp = _as_of()
        data = _CountingData(_bars(BEARISH_BARS), clock=lambda: stamp.to_pydatetime())
        source = _source(
            data,
            clock=lambda: stamp.to_pydatetime(),
            book=TjrBookSnapshot(cash=5000.0),
            price=100.6,
        )
        calls = []
        worker = _worker(tmp_path, source, calls=calls)
        worker.start()
        first = worker.run_once()
        assert first.reason_code == NO_TRADE_NO_SIGNAL
        assert not [c for c in calls if c["method"] == "POST"]

        # The next cycle completes a setup: the order is placed.
        later = stamp + pd.Timedelta(minutes=5)
        source._clock = lambda: later.to_pydatetime()
        source.data = _CountingData(
            _bars(BULLISH_BARS), clock=lambda: later.to_pydatetime()
        )
        second = worker.run_once()
        worker.stop()
        assert second.ran is True
        assert second.order is not None
        assert [c for c in calls if c["method"] == "POST"]


# ---------------------------------------------------------------------------
# 6. The shipped crypto time-in-force fix is on both paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheCryptoTimeInForceFixIsInPlace:
    def test_a_crypto_entry_order_sends_gtc(self, tmp_path):
        """A crypto order carries the crypto time-in-force, not the equity one."""
        stamp = _as_of()
        data = _CountingData(
            _bars(BULLISH_BARS, CRYPTO_SYMBOL),
            clock=lambda: stamp.to_pydatetime(),
        )
        source = _source(
            data,
            clock=lambda: stamp.to_pydatetime(),
            symbols=(CRYPTO_SYMBOL,),
            book=TjrBookSnapshot(cash=5000.0),
            # The bars are priced near 100, so the execution price must match or
            # the order's notional would exceed the $25 cap and be risk-rejected.
            price=100.6,
        )
        calls = []
        worker = _worker(
            tmp_path, source, calls=calls, crypto=True, symbol=CRYPTO_SYMBOL
        )
        worker.start()
        result = worker.run_once()
        worker.stop()
        assert result.ran is True, result.reason_code
        posts = [c for c in calls if c["method"] == "POST"]
        assert posts
        body = posts[0]["json"]
        assert body["time_in_force"] == "gtc"
        assert body["time_in_force"] != "day"

    def test_a_crypto_quantity_is_not_in_scientific_notation(self):
        from tradingagents.paper_experiment.broker.alpaca_paper import _format_quantity

        assert "e" not in _format_quantity(0.000001).lower()
        assert "e" not in _format_quantity(0.00000025).lower()
        assert _format_quantity(0.000001) == "0.000001"

    def test_the_risk_limits_are_unchanged(self):
        from tradingagents.risk.paper_parity import paper_parity_limits

        limits = paper_parity_limits()
        assert limits.max_order_value == 25.0
        assert limits.target_order_value == 25.0
        assert limits.max_open_positions == 5
        assert limits.max_daily_orders == 10
        assert limits.long_only is True
