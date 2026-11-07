"""Phase 30: the TJR rule chain as the paper worker's decision source.

Phase 1 validated the *strategy* over historical bars. Phase 2 makes it runnable
on Alpaca **paper**: a decision source that evaluates the deterministic TJR rule
chain over current *intraday* (1m/5m) Alpaca bars, sizes the entry through the
project's own ``RiskEngine`` under ``paper_parity_limits()``, and hands the worker
a ``TradingDecision`` the existing risk layer and broker already understand.

Everything here is offline and deterministic: the market clock, the intraday bar
fetch and the broker transport are injected, so no test reaches the network (the
suite's autouse guard refuses it) and none needs an Alpaca key. The tests drive
the real ``TjrDecisionSource``, the real ``PaperTradingWorker`` and the real risk
checks — not stubs of them.

The safety properties these tests pin, in the order they appear:

1. no model is consulted (the strategy is the rule chain);
2. the chain completes only on *closed* bars — no look-ahead, no in-progress bar;
3. one decision per symbol per session, deterministic id, restart-safe;
4. a completed setup is sized by the RiskEngine under the paper-parity caps, and
   the caps are applied to the *shared* book;
5. an incomplete chain / closed market / data failure is a named no-trade, never a
   fabricated signal, size, stop or target;
6. the source writes only its own state file, never the experiment's;
7. the path is paper-only: no live host is reachable and no order is placed
   without an approved decision.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tradingagents.paper_experiment import ExperimentConfig, PaperExperiment
from tradingagents.paper_experiment.broker import (
    DECISION_SOURCE_CHOICES,
    DECISION_SOURCE_TJR,
    NO_TRADE_MARKET_CLOSED,
    NO_TRADE_MARKET_DATA_UNAVAILABLE,
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
from tradingagents.paper_experiment.market_data_wiring import (
    DECISION_SOURCE_ENV,
    describe_decision_source,
    resolve_decision_source,
)
from tradingagents.paper_experiment.market_hours import (
    StaticMarketHours,
)
from tradingagents.paper_experiment.tjr_intraday_data import (
    AlpacaIntradayBars,
    StaticIntradayBars,
    TjrIntradayDataError,
)
from tradingagents.paper_experiment.tjr_source import (
    TjrBookSnapshot,
    TjrDecisionSource,
    TjrSourceError,
    ledger_from_snapshot,
)
from tradingagents.risk.paper_parity import (
    MAX_DAILY_ORDERS,
    MAX_OPEN_POSITIONS,
    MAX_ORDER_NOTIONAL,
    paper_parity_limits,
)
from tradingagents.tjr.frames import session_open_utc
from tradingagents.tjr.rules import TjrSettings, evaluate

from .tjr_helpers import (
    BEARISH_BARS,
    BULLISH_BARS,
    SESSION,
    frame_from_bars,
    one_minute_from_five,
)

SYMBOL = "AAPL"
OTHER = "MSFT"

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


def _bars(session: str = SESSION, bars=None) -> dict[str, dict[str, pd.DataFrame]]:
    """A ``StaticIntradayBars`` bundle for one symbol from 5m bars."""
    five = frame_from_bars(session, bars if bars is not None else BULLISH_BARS)
    one = one_minute_from_five(five)
    return {SYMBOL: {"5m": five, "1m": one}}


def _data(
    *,
    session: str = SESSION,
    bars=None,
    at: pd.Timestamp | None = None,
    clock=None,
    **kwargs,
):
    """A static intraday bundle whose own clock matches the decision instant.

    The seam freshness-checks the newest bar against *its* clock, so a test that
    pins the decision instant must pin the seam's clock to the same instant, or
    the bars would read as stale against the real wall clock. ``clock`` overrides
    the pinned instant when a test wants to drive staleness explicitly.
    """
    if clock is None:
        stamp = at if at is not None else _as_of(session)
        clock = lambda: stamp.to_pydatetime()  # noqa: E731
    return StaticIntradayBars(_bars(session, bars), clock=clock, **kwargs)


def _clock(*, is_open: bool = True, session: str = SESSION, error: str | None = None):
    return StaticMarketHours(is_open=is_open, session=session, error=error)


def _closed_series(symbol: str, session: str, as_of):
    """The (5m, 1m) series for one session's bullish bars, sliced to closed bars.

    A minimal in-memory seam for tests that need the *session* to vary: it serves
    each session's own bars rather than a fixed bundle.
    """
    from tradingagents.tjr.frames import IntradaySeries

    five = IntradaySeries(symbol, "5m", frame_from_bars(session, BULLISH_BARS))
    one = IntradaySeries(symbol, "1m", one_minute_from_five(frame_from_bars(session, BULLISH_BARS)))
    stamp = pd.Timestamp(as_of)
    five_closed = five.frame[five.frame.index <= stamp - pd.Timedelta(minutes=5)]
    one_closed = one.frame[one.frame.index <= stamp - pd.Timedelta(minutes=1)]
    return (
        IntradaySeries(symbol, "5m", five_closed),
        IntradaySeries(symbol, "1m", one_closed),
    )


def _as_of(session: str = SESSION, *, bar_index: int = 15) -> pd.Timestamp:
    """The decision instant at the *close* of 5m bar ``bar_index`` (0-based)."""
    return session_open_utc(session) + pd.Timedelta(minutes=5 * bar_index)


def _source(
    *,
    data=None,
    market_hours=None,
    is_open: bool = True,
    book: TjrBookSnapshot | None = None,
    price: float | None = None,
    symbols=(SYMBOL,),
    state_path=None,
    is_consumed=None,
    clock=None,
    session: str = SESSION,
):
    # The source's clock and the data seam's clock must agree, or the newest bar
    # would read as stale against the real wall clock.
    instant = pd.Timestamp(clock()) if clock is not None else _as_of(session)
    if data is None:
        data = _data(session=session, at=instant)
    return TjrDecisionSource(
        symbols=tuple(symbols),
        data=data,
        market_hours=market_hours if market_hours is not None else _clock(is_open=is_open),
        order_notional=25.0,
        book_provider=(lambda: book) if book is not None else None,
        price_provider=(lambda _s: price) if price is not None else None,
        state_path=state_path,
        is_consumed=is_consumed,
        clock=lambda: instant.to_pydatetime(),
    )


# ---------------------------------------------------------------------------
# 1. The strategy is the rule chain, not a model
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheChainIsTheStrategy:
    def test_the_source_runs_the_rule_chain_and_never_a_model(self):
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6)
        decision = source.next_decision()
        assert decision is not None
        assert decision.side == "buy"
        # The decision's stop/target are exactly the chain's, not invented.
        evaluation = evaluate(
            symbol=SYMBOL,
            session=SESSION,
            as_of=_as_of(),
            series_5m=_data().series(SYMBOL, session=SESSION, as_of=_as_of().to_pydatetime())[0],
            series_1m=_data().series(SYMBOL, session=SESSION, as_of=_as_of().to_pydatetime())[1],
            settings=TjrSettings(),
            allow_short=False,
        )
        assert evaluation.stage == "READY"
        assert decision.stop_loss == pytest.approx(evaluation.stop)
        assert decision.take_profit == pytest.approx(evaluation.target)
        assert decision.strategy_name == "tjr"

    def test_a_bearish_setup_is_never_a_fabricated_long(self):
        # TJR is long-only; a complete bearish setup must not become a BUY.
        data = _data(bars=BEARISH_BARS)
        source = _source(data=data, book=TjrBookSnapshot(cash=5000.0), price=100.0)
        decision = source.next_decision()
        assert decision is None
        assert source.skip_code == NO_TRADE_NO_SIGNAL
        record = source.last_decision_for(SYMBOL)
        assert record["side"] is None
        assert record["chain_reason"]


# ---------------------------------------------------------------------------
# 2. Closed bars only — no look-ahead, no in-progress bar
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClosedBarsOnly:
    def test_a_decision_before_the_chain_completes_is_a_no_trade(self):
        # At bar 10 the chain is still at NO_SWEEP; it must not fire early.
        source = _source(
            book=TjrBookSnapshot(cash=5000.0),
            price=100.0,
            clock=lambda: _as_of(bar_index=10).to_pydatetime(),
        )
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_NO_SIGNAL

    def test_the_decision_is_made_on_bars_that_have_closed(self):
        # The chain completes at bar 15's close; a decision taken at that instant
        # must be identical to one the closed-bar reader would produce, and must
        # not see bar 16.
        at_15 = _source(
            book=TjrBookSnapshot(cash=5000.0),
            price=100.6,
            clock=lambda: _as_of(bar_index=15).to_pydatetime(),
        ).next_decision()
        assert at_15 is not None
        evaluation = evaluate(
            symbol=SYMBOL, session=SESSION, as_of=_as_of(bar_index=15),
            series_5m=_data().series(SYMBOL, session=SESSION, as_of=_as_of(bar_index=15).to_pydatetime())[0],
            series_1m=_data().series(SYMBOL, session=SESSION, as_of=_as_of(bar_index=15).to_pydatetime())[1],
        )
        assert at_15.stop_loss == pytest.approx(evaluation.stop)

    def test_an_in_progress_bar_is_dropped_not_used(self):
        # At 10:44 ET the 10:40 5m bar (closing at 10:45) is still forming, so the
        # closed 5m series still ends at 10:35, where the chain has not completed.
        # The forming bar is exactly the one that would complete it at 10:45.
        mid = _as_of(bar_index=15) - pd.Timedelta(minutes=1)
        source = _source(
            data=_data(at=mid),
            book=TjrBookSnapshot(cash=5000.0),
            price=100.6,
            clock=lambda: mid.to_pydatetime(),
        )
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_NO_SIGNAL

    def test_the_seam_never_returns_a_bar_after_the_instant(self):
        data = _data()
        as_of = _as_of(bar_index=15).to_pydatetime()
        five, one = data.series(SYMBOL, session=SESSION, as_of=as_of)
        assert (five.frame.index <= pd.Timestamp(as_of)).all()
        assert (one.frame.index <= pd.Timestamp(as_of)).all()


# ---------------------------------------------------------------------------
# 3. One decision per symbol per session, deterministic, restart-safe
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCadenceAndIdempotency:
    def test_one_decision_per_symbol_per_session(self):
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6)
        first = source.next_decision()
        assert first is not None
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_TJR_NO_NEW_DATA

    def test_the_decision_id_is_deterministic(self):
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6)
        decision = source.next_decision()
        assert decision.decision_id == f"tjr-{SYMBOL}-{SESSION}"
        other = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6)
        assert other.next_decision().decision_id == decision.decision_id

    def test_a_new_session_yields_a_new_decision(self):
        # The clock rolls to the next session; the source serves that session's own
        # bars and yields a decision with the new session in its id.
        next_session = "2026-10-30"

        class _PerSessionData:
            """Serves each session's own bullish bundle, keyed by the session."""

            def series(self, symbol, *, session, as_of):
                return _closed_series(symbol, session, as_of)

            def describe(self):
                return {"tjr_data_provider": "per_session_test"}

        now = {"instant": _as_of().to_pydatetime()}
        clock = _clock(session=SESSION)
        source = TjrDecisionSource(
            symbols=(SYMBOL,),
            data=_PerSessionData(),
            market_hours=clock,
            order_notional=25.0,
            book_provider=lambda: TjrBookSnapshot(cash=5000.0),
            price_provider=lambda _s: 100.6,
            clock=lambda: now["instant"],
        )
        first = source.next_decision()
        assert first is not None
        assert first.decision_id == f"tjr-{SYMBOL}-{SESSION}"
        # The market rolls to the next session.
        clock._session = next_session
        now["instant"] = _as_of(next_session).to_pydatetime()
        second = source.next_decision()
        assert second is not None
        assert second.decision_id == f"tjr-{SYMBOL}-{next_session}"
        assert second.decision_id != first.decision_id

    def test_a_restart_within_a_session_replays_the_same_decision(self, tmp_path):
        state = tmp_path / "tjr_source_state.json"
        first = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6, state_path=state)
        decision = first.next_decision()
        assert decision is not None

        restarted = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6, state_path=state)
        replayed = restarted.next_decision()
        assert replayed is not None
        assert replayed.decision_id == decision.decision_id

    def test_a_consumed_decision_is_not_re_offered(self):
        offered = {f"tjr-{SYMBOL}-{SESSION}"}
        source = _source(
            book=TjrBookSnapshot(cash=5000.0),
            price=100.6,
            is_consumed=lambda did: did in offered,
        )
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_TJR_NO_NEW_DATA

    def test_the_chain_is_not_re_evaluated_once_consumed(self):
        calls = []

        class _CountingData(StaticIntradayBars):
            def series(self, symbol, *, session, as_of):
                calls.append(session)
                return super().series(symbol, session=session, as_of=as_of)

        source = _source(
            data=_CountingData(_bars()),
            book=TjrBookSnapshot(cash=5000.0),
            price=100.6,
            is_consumed=lambda did: True,
        )
        assert source.next_decision() is None
        assert len(calls) == 1  # evaluated once, then suppressed


# ---------------------------------------------------------------------------
# 4. The RiskEngine sizes the entry under the paper-parity caps
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRiskEngineGate:
    def test_a_completed_setup_is_sized_by_the_risk_engine(self):
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6)
        decision = source.next_decision()
        assert decision is not None
        # $25 / $100.6 ≈ 0.2485 shares; the engine's own sizing, not the source's.
        assert decision.quantity == pytest.approx(25.0 / 100.6, rel=1e-6)

    def test_the_source_requests_no_size_of_its_own(self):
        # The decision carries the engine's quantity; the source fabricates none.
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6)
        decision = source.next_decision()
        assert decision.notional is None
        assert decision.quantity is not None

    def test_an_empty_book_is_refused_by_the_engine(self):
        # No cash -> NO_EQUITY -> a named risk rejection, never an order.
        source = _source(book=TjrBookSnapshot(cash=0.0), price=100.6)
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_RISK_REJECTED

    def test_no_book_supplied_is_a_fail_safe_no_order(self):
        source = _source(price=100.6)  # book_provider is None
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_RISK_REJECTED

    def test_the_notional_cap_holds_even_on_a_large_book(self):
        source = _source(book=TjrBookSnapshot(cash=1_000_000.0), price=100.6)
        decision = source.next_decision()
        assert decision is not None
        # The $25 cap binds: 25 / 100.6, not a fraction of the large book.
        assert decision.quantity * 100.6 == pytest.approx(MAX_ORDER_NOTIONAL, rel=1e-6)

    def test_the_position_cap_is_read_from_the_shared_book(self):
        # Five positions already held: a sixth entry is refused by the shared cap.
        held = {f"S{i}": 1.0 for i in range(MAX_OPEN_POSITIONS)}
        source = _source(
            book=TjrBookSnapshot(cash=5000.0, positions=held),
            price=100.6,
        )
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_RISK_REJECTED

    def test_the_daily_order_cap_is_read_from_the_book(self):
        source = _source(
            book=TjrBookSnapshot(cash=5000.0, daily_orders=MAX_DAILY_ORDERS),
            price=100.6,
        )
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_RISK_REJECTED

    def test_the_engine_sizes_under_the_paper_parity_limits(self):
        limits = paper_parity_limits()
        assert limits.max_order_value == MAX_ORDER_NOTIONAL
        assert limits.max_open_positions == MAX_OPEN_POSITIONS
        assert limits.max_daily_orders == MAX_DAILY_ORDERS
        assert limits.long_only is True
        # The source defaults to the same single source of limits.
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6)
        assert source.risk_limits.max_order_value == MAX_ORDER_NOTIONAL

    def test_the_reference_price_is_the_execution_seam(self):
        # The engine must size at the price the order fills against, not the bar
        # close, so a mismatch between them cannot mis-size the order.
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=250.0)
        decision = source.next_decision()
        assert decision is not None
        assert decision.quantity == pytest.approx(25.0 / 250.0, rel=1e-6)


@pytest.mark.unit
class TestLedgerMirror:
    def test_the_ledger_mirrors_the_broker_book(self):
        snapshot = TjrBookSnapshot(cash=1234.0, positions={SYMBOL: 2.0})
        ledger = ledger_from_snapshot(snapshot, prices={SYMBOL: 100.0})
        assert ledger.cash == pytest.approx(1234.0)
        assert ledger.equity == pytest.approx(1234.0 + 2.0 * 100.0)
        assert set(ledger.open_positions()) == {SYMBOL}

    def test_a_short_position_in_the_book_is_ignored(self):
        # Long-only: a negative quantity is not mirrored as a holding.
        snapshot = TjrBookSnapshot(cash=1000.0, positions={SYMBOL: -3.0})
        ledger = ledger_from_snapshot(snapshot, prices={SYMBOL: 100.0})
        assert ledger.open_positions() == {}


# ---------------------------------------------------------------------------
# 5. Honest no-trades: closed market, data failure, incomplete chain
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHonestNoTrade:
    def test_a_closed_market_is_a_named_no_trade(self):
        source = _source(is_open=False, book=TjrBookSnapshot(cash=5000.0), price=100.6)
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_MARKET_CLOSED

    def test_an_unreadable_clock_is_market_data_unavailable(self):
        source = _source(
            market_hours=_clock(error="clock 500"),
            book=TjrBookSnapshot(cash=5000.0),
            price=100.6,
        )
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_MARKET_DATA_UNAVAILABLE

    def test_a_data_failure_is_market_data_unavailable(self):
        class _BrokenData:
            def series(self, symbol, *, session, as_of):
                raise TjrIntradayDataError("Alpaca 503", kind="transport")

            def describe(self):
                return {}

        source = _source(data=_BrokenData(), book=TjrBookSnapshot(cash=5000.0), price=100.6)
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_MARKET_DATA_UNAVAILABLE

    def test_a_generic_data_failure_is_contained(self):
        class _ExplodingData:
            def series(self, symbol, *, session, as_of):
                raise RuntimeError("boom")

            def describe(self):
                return {}

        source = _source(data=_ExplodingData(), book=TjrBookSnapshot(cash=5000.0), price=100.6)
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_MARKET_DATA_UNAVAILABLE

    def test_no_decision_ever_carries_a_fabricated_side(self):
        source = _source(book=TjrBookSnapshot(cash=0.0), price=100.6)
        assert source.next_decision() is None
        assert source.last_decision_for(SYMBOL)["side"] is None

    def test_every_reason_code_is_in_the_closed_vocabulary(self):
        sources = [
            _source(is_open=False, book=TjrBookSnapshot(cash=5000.0), price=100.6),
            _source(book=TjrBookSnapshot(cash=0.0), price=100.6),
            _source(
                book=TjrBookSnapshot(cash=5000.0),
                price=100.0,
                clock=lambda: _as_of(bar_index=10).to_pydatetime(),
            ),
        ]
        for source in sources:
            assert source.next_decision() is None
            assert source.skip_code in NO_TRADE_REASONS

    def test_an_empty_symbol_set_is_refused_at_construction(self):
        with pytest.raises(TjrSourceError):
            TjrDecisionSource(
                symbols=(),
                data=_data(),
                market_hours=_clock(),
                order_notional=25.0,
            )


# ---------------------------------------------------------------------------
# 6. The source writes only its own state file
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStateIsolation:
    def test_the_source_writes_only_its_own_state_file(self, tmp_path):
        root = tmp_path / "exp"
        root.mkdir()
        source = _source(
            book=TjrBookSnapshot(cash=5000.0),
            price=100.6,
            state_path=root / "tjr_source_state.json",
        )
        source.next_decision()
        names = sorted(p.name for p in root.iterdir())
        assert names == ["tjr_source_state.json"]
        assert "session_state.json" not in names
        assert "journal.json" not in names

    def test_the_state_file_is_versioned_and_carries_the_records(self, tmp_path):
        state = tmp_path / "tjr_source_state.json"
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6, state_path=state)
        source.next_decision()
        payload = json.loads(state.read_text())
        assert payload["schema_version"] == 1
        assert SYMBOL in payload["records"]
        assert payload["records"][SYMBOL]["session"] == SESSION

    def test_a_corrupt_state_file_is_ignored_not_fatal(self, tmp_path):
        state = tmp_path / "tjr_source_state.json"
        state.write_text("{ not json")
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6, state_path=state)
        assert source.next_decision() is not None


# ---------------------------------------------------------------------------
# 7. The full chain: source -> worker -> risk -> broker (offline)
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)


def _order(**over):
    payload = {
        "id": "broker-order-1",
        "client_order_id": "c1",
        "symbol": SYMBOL,
        "side": "buy",
        "qty": "0.25",
        "filled_qty": "0.0",
        "filled_avg_price": None,
        "status": "accepted",
        "type": "market",
        "time_in_force": "day",
        "submitted_at": "2026-10-29T13:30:00Z",
    }
    payload.update(over)
    return payload


def _transport(routes, *, calls=None):
    def send(*, method, url, headers, params, json_body, timeout):
        if calls is not None:
            calls.append({"method": method, "url": url, "json": json_body})
        for (m, suffix), response in routes.items():
            if method == m and url.endswith(suffix):
                return response() if callable(response) else response
        raise AssertionError(f"unexpected request {method} {url}")

    return send


def _routes(*, filled: bool = False):
    order = (
        _order(filled_qty="0.25", filled_avg_price="100.6", status="filled")
        if filled
        else _order()
    )
    return {
        ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
        ("GET", "/v2/positions"): _Resp(200, []),
        ("POST", "/v2/orders"): _Resp(200, order),
        ("GET", "/v2/orders:by_client_order_id"): _Resp(200, order),
    }


def _broker_factory(routes, *, limits=None):
    def build():
        return AlpacaPaperBroker(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=_transport(routes),
            limits=limits or PaperRiskLimits(),
        )

    return build


def _worker(tmp_path, source, *, market_price=100.6, limits=None, filled=False):
    config = PaperTradingConfig(
        strategy_id="tjr",
        order_notional=25.0,
        whole_shares=False,
        state_path=tmp_path / "worker_state.json",
        market_data_provider="bar_open",
        decision_source=DECISION_SOURCE_TJR,
        limits=limits or PaperRiskLimits(),
    )
    return PaperTradingWorker(
        config=config,
        decision_source=source,
        market_data=StaticMarketData(prices={SYMBOL: market_price}),
        broker_factory=_broker_factory(_routes(filled=filled), limits=limits),
    )


@pytest.mark.unit
class TestTheFullChain:
    def test_a_tjr_buy_reaches_the_broker_as_a_paper_order(self, tmp_path):
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6)
        worker = _worker(tmp_path, source)
        worker.start()
        result = worker.run_once()
        worker.stop()
        assert result.ran is True
        assert result.reason_code is None
        assert result.order is not None
        assert result.order.symbol == SYMBOL
        assert result.order.side == "buy"

    def test_the_worker_records_the_chain_stop_and_target(self, tmp_path):
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6)
        worker = _worker(tmp_path, source, filled=True)
        worker.start()
        worker.run_once()
        worker.stop()
        # The stop/target are recorded from the *filled* decision, never fabricated.
        assert SYMBOL in worker.state.stops
        assert worker.state.stops[SYMBOL]["stop_loss"] is not None
        assert worker.state.stops[SYMBOL]["take_profit"] is not None

    def test_an_incomplete_chain_places_no_order(self, tmp_path):
        source = _source(
            book=TjrBookSnapshot(cash=5000.0),
            price=100.0,
            clock=lambda: _as_of(bar_index=10).to_pydatetime(),
        )
        worker = _worker(tmp_path, source, market_price=100.0)
        worker.start()
        result = worker.run_once()
        worker.stop()
        assert result.ran is False
        assert result.reason_code == NO_TRADE_NO_SIGNAL
        assert worker.state.orders == {}

    def test_a_closed_market_places_no_order(self, tmp_path):
        source = _source(is_open=False, book=TjrBookSnapshot(cash=5000.0), price=100.6)
        worker = _worker(tmp_path, source)
        worker.start()
        result = worker.run_once()
        worker.stop()
        assert result.ran is False
        assert result.reason_code == NO_TRADE_MARKET_CLOSED
        assert worker.state.orders == {}

    def test_a_risk_rejection_prevents_order_submission(self, tmp_path):
        calls = []
        # A tiny broker-side notional cap refuses the $25 order before submission.
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6)
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                strategy_id="tjr",
                order_notional=25.0,
                whole_shares=False,
                state_path=tmp_path / "worker_state.json",
                decision_source=DECISION_SOURCE_TJR,
            ),
            decision_source=source,
            market_data=StaticMarketData(prices={SYMBOL: 100.6}),
            broker_factory=lambda: AlpacaPaperBroker(
                api_key="paper-key",
                api_secret="paper-secret",
                transport=_transport(_routes(), calls=calls),
                limits=PaperRiskLimits(max_order_notional=5.0),
            ),
        )
        worker.start()
        result = worker.run_once()
        worker.stop()
        assert result.ran is False
        assert result.reason_code == NO_TRADE_RISK_REJECTED
        assert not any(c["method"] == "POST" for c in calls)

    def test_a_duplicate_cannot_be_ordered_after_a_restart(self, tmp_path):
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6)
        worker = _worker(tmp_path, source)
        worker.start()
        first = worker.run_once()
        worker.stop()
        assert first.order is not None
        client_order_id = first.order.client_order_id

        restarted = PaperTradingWorker(
            config=worker.config,
            decision_source=source,
            market_data=worker.market_data,
            broker_factory=_broker_factory(_routes()),
        )
        restarted.start()
        second = restarted.run_once()
        restarted.stop()
        assert second.reason_code in (
            NO_TRADE_TJR_NO_NEW_DATA,
            "ALREADY_PROCESSED",
            "ORDER_EXISTS",
        )
        assert set(restarted.state.orders) == {client_order_id}


# ---------------------------------------------------------------------------
# 8. Configuration and wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWiring:
    def test_tjr_is_a_valid_decision_source_choice(self):
        assert DECISION_SOURCE_TJR in DECISION_SOURCE_CHOICES

    def test_resolve_decision_source_accepts_tjr(self):
        assert resolve_decision_source("tjr") == DECISION_SOURCE_TJR

    def test_resolve_decision_source_is_case_sensitive(self):
        # A choice is a stable token, matched exactly; a wrong case is a loud error,
        # never a silent fall-back to a different source.
        from tradingagents.paper_experiment.market_data_wiring import MarketDataWiringError

        with pytest.raises(MarketDataWiringError):
            resolve_decision_source("TJR")

    def test_a_bad_choice_raises_rather_than_falling_back(self):
        from tradingagents.paper_experiment.market_data_wiring import MarketDataWiringError

        with pytest.raises(MarketDataWiringError):
            resolve_decision_source("live")

    def test_describe_names_the_rule_chain_and_the_intraday_data(self):
        text = describe_decision_source("tjr")
        assert "deterministic" in text.lower()
        assert "intraday" in text.lower()
        assert "tjr" in text.lower()

    def test_the_decision_source_env_is_used(self, monkeypatch):
        from tradingagents.paper_experiment.market_data_wiring import env_decision_source

        monkeypatch.setenv(DECISION_SOURCE_ENV, "tjr")
        assert env_decision_source() == DECISION_SOURCE_TJR

    def test_the_worker_config_accepts_the_tjr_source(self):
        config = PaperTradingConfig(decision_source=DECISION_SOURCE_TJR)
        assert config.decision_source == DECISION_SOURCE_TJR


# ---------------------------------------------------------------------------
# 9. The intraday seam itself
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIntradaySeam:
    def test_the_static_seam_serves_closed_bars_only(self):
        data = _data()
        as_of = _as_of(bar_index=15).to_pydatetime()
        five, one = data.series(SYMBOL, session=SESSION, as_of=as_of)
        assert five.frame.index[-1] <= pd.Timestamp(as_of)
        assert one.frame.index[-1] <= pd.Timestamp(as_of)

    def test_a_stale_bar_is_refused(self):
        # The seam's clock is far past the newest bar's close: the data is stale.
        later = _as_of(bar_index=15) + pd.Timedelta(hours=6)
        data = _data(clock=lambda: later.to_pydatetime())
        with pytest.raises(TjrIntradayDataError) as exc:
            data.series(SYMBOL, session=SESSION, as_of=_as_of(bar_index=15).to_pydatetime())
        assert exc.value.kind == "stale"

    def test_a_future_dated_bar_is_refused(self):
        # The decision session is before the bars: future-dated data.
        data = _data()
        with pytest.raises(TjrIntradayDataError) as exc:
            data.series(SYMBOL, session="2026-10-28", as_of=_as_of().to_pydatetime())
        assert exc.value.kind == "future"

    def test_a_naive_instant_is_refused(self):
        data = _data()
        with pytest.raises(TjrIntradayDataError) as exc:
            data.series(SYMBOL, session=SESSION, as_of=pd.Timestamp("2026-10-29T14:45:00"))
        assert exc.value.kind == "malformed"

    def test_a_missing_symbol_is_refused(self):
        data = _data()
        with pytest.raises(TjrIntradayDataError) as exc:
            data.series("ZZZZ", session=SESSION, as_of=_as_of().to_pydatetime())
        assert exc.value.kind == "no_data"

    def test_the_alpaca_seam_refuses_a_non_data_host(self):
        from tradingagents.paper_experiment.providers.base import ProviderUnsupportedError

        with pytest.raises(ProviderUnsupportedError):
            AlpacaIntradayBars(
                api_key="k",
                api_secret="s",
                base_url="https://paper-api.alpaca.markets",
            )

    def test_the_alpaca_seam_refuses_a_live_host(self):
        from tradingagents.paper_experiment.providers.base import ProviderUnsupportedError

        with pytest.raises(ProviderUnsupportedError):
            AlpacaIntradayBars(
                api_key="k",
                api_secret="s",
                base_url="https://api.alpaca.markets",
            )

    def test_missing_credentials_are_reported_not_defaulted(self, monkeypatch):
        monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_PAPER_API_SECRET", raising=False)
        bars = AlpacaIntradayBars(api_key="", api_secret="")
        with pytest.raises(TjrIntradayDataError) as exc:
            bars.series(SYMBOL, session=SESSION, as_of=_as_of().to_pydatetime())
        assert exc.value.kind == "not_configured"

    def test_the_alpaca_seam_requests_1min_and_5min_from_the_data_host(self):
        calls = []

        def send(*, url, params, headers, timeout):
            calls.append({"url": url, "params": dict(params)})
            # Return a small but valid payload for each interval.
            five = frame_from_bars(SESSION, BULLISH_BARS[:16])
            frame = five if params["timeframe"] == "5Min" else one_minute_from_five(five)
            bars = [
                {
                    "t": stamp.isoformat().replace("+00:00", "Z"),
                    "o": float(row["open"]),
                    "h": float(row["high"]),
                    "l": float(row["low"]),
                    "c": float(row["close"]),
                    "v": float(row["volume"]),
                }
                for stamp, row in frame.iterrows()
            ]
            return _Resp(200, {"bars": {SYMBOL: bars}})

        bars = AlpacaIntradayBars(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=send,
            clock=lambda: _as_of(bar_index=15).to_pydatetime(),
        )
        bars.series(SYMBOL, session=SESSION, as_of=_as_of(bar_index=15).to_pydatetime())
        timeframes = {c["params"]["timeframe"] for c in calls}
        assert timeframes == {"1Min", "5Min"}
        assert all(c["url"].startswith("https://data.alpaca.markets") for c in calls)
        assert all(c["params"]["feed"] == "iex" for c in calls)

    def test_the_source_describes_itself_without_a_secret(self):
        source = _source(book=TjrBookSnapshot(cash=5000.0), price=100.6)
        view = source.describe()
        assert view["decision_source_kind"] == "tjr"
        assert view["tjr_data_provider"] == "static_intraday_bars"
        json.dumps(view)


# ---------------------------------------------------------------------------
# 10. The controller builds the TJR source offline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestControllerWiring:
    def _experiment_root(self, tmp_path):
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
        # Process one session so the persisted session state exists; the controller
        # reads it to name the configured limits without building a worker.
        for session in list(source.sessions())[:1]:
            experiment.process_session(session)
        return root

    def test_the_controller_builds_a_tjr_source_with_injected_seams(self, tmp_path):
        from tradingagents.paper_experiment.dashboard.paper_control import (
            PaperTradingController,
        )

        root = self._experiment_root(tmp_path)
        controller = PaperTradingController(
            root,
            enabled=True,
            decision_source_kind=DECISION_SOURCE_TJR,
            market_hours_gate=lambda: _clock(),
            tjr_data_factory=lambda: _data(),
        )
        experiment = PaperExperiment.load(root, inference=None)
        holder: dict = {}
        source = controller._build_tjr_source(experiment, holder)  # noqa: SLF001
        assert isinstance(source, TjrDecisionSource)
        assert source.symbols == (SYMBOL,)
        assert source.source_kind == "tjr"
        # It writes its own state file under the experiment root, not the
        # experiment's state.
        assert source._store == root / "tjr_source_state.json"  # noqa: SLF001

    def test_the_controller_status_does_not_build_the_tjr_source(self, tmp_path):
        from tradingagents.paper_experiment.dashboard.paper_control import (
            PaperTradingController,
        )

        root = self._experiment_root(tmp_path)
        controller = PaperTradingController(
            root,
            enabled=True,
            decision_source_kind=DECISION_SOURCE_TJR,
        )
        status = controller.status()
        assert status["status"] == "STOPPED"
        assert status["decision_source_configured"] == DECISION_SOURCE_TJR
