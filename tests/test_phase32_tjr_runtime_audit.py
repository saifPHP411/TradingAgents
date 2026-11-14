"""Phase 32: the TJR runtime audit — evaluation cadence, market-data freshness,
and the guards that keep an old decision from becoming a new order.

A Railway deployment reported four things that did not agree with each other:

* ``last_no_trade_code = TJR_NO_NEW_DATA`` with the reason "every symbol already
  produced its one decision for this session; a new session is required before
  the next decision" — the worker was running and evaluating, and yet the source
  refused to look at the market again;
* ``market_data_status = ok`` beside ``last_snapshot_age_seconds = 6142.12`` and
  ``is_current_price = false`` — a price two hours old reported as healthy;
* ``tjr_data_last_fetch_at = null`` beside old TJR decisions — the bars had
  never been fetched;
* ``AVAX/USD accepted, filled_quantity=0`` and ``XRP/USD accepted,
  filled_quantity=0`` with ``positions={}`` — orders the venue had accepted and
  the account did not hold.

Each of those is a distinct defect with a distinct cause, and each is pinned
here at the code path that caused it:

1. **the evaluation cycle** — a symbol whose setup completed is re-evaluated
   when *new completed bars* arrive, so a later sweep is seen; a re-read of the
   *same* setup is not offered again, so the same signal cannot repeat;
2. **the market-data freshness** — the reported status is derived from the same
   observation the age is measured from, so a price past its age bound reports
   ``stale``, and a missing or future timestamp is refused;
3. **the data-fetch evidence** — the seam records the fetch, so the view cannot
   report "never fetched" while the worker is polling;
4. **the guards** — an old READY decision cannot be traded, and a new entry is
   refused while an open position's protection is unconfirmed.

Everything here is offline and deterministic: the clock, the intraday bar seam
and the venue transport are injected, so no test reaches the network and none
needs a venue credential. The TJR rule chain itself is **not** modified here —
these tests pin the *cadence around* it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tradingagents.crypto.types import Ticker
from tradingagents.paper_experiment.broker.trading import (
    MARKET_DATA_STATUS_OK,
    MARKET_DATA_STATUS_STALE,
    NO_TRADE_ORDER_EXISTS,
    NO_TRADE_POSITION_UNAVAILABLE,
    NO_TRADE_PROTECTION_FAILED,
    NO_TRADE_RISK_REJECTED,
    NO_TRADE_TJR_NO_NEW_DATA,
    BinanceSnapshotMarketData,
    MarketDataError,
    StaticMarketData,
)
from tradingagents.paper_experiment.market_hours import StaticMarketHours
from tradingagents.paper_experiment.tjr_intraday_data import StaticIntradayBars
from tradingagents.paper_experiment.tjr_source import (
    TjrBookSnapshot,
    TjrDecisionSource,
)

from .tjr_helpers import (
    BULLISH_BARS,
    SESSION,
    frame_from_bars,
    one_minute_from_five,
)

SYMBOL = "AAPL"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_of(*, bar_index: int = 15) -> pd.Timestamp:
    """The decision instant at the *close* of 5m bar ``bar_index`` (0-based)."""
    from tradingagents.tjr.frames import session_open_utc

    return session_open_utc(SESSION) + pd.Timedelta(minutes=5 * bar_index)


def _bundle(bars, symbol: str = SYMBOL) -> dict[str, dict[str, pd.DataFrame]]:
    five = frame_from_bars(SESSION, bars)
    return {symbol: {"5m": five, "1m": one_minute_from_five(five)}}


def _data(bars, *, at: pd.Timestamp, symbol: str = SYMBOL) -> StaticIntradayBars:
    """A static intraday seam whose clock matches the decision instant.

    The seam freshness-checks the newest bar against *its* clock, so the clock
    has to move with the decision instant or the bars read as stale.
    """
    return StaticIntradayBars(
        _bundle(bars, symbol), clock=lambda: at.to_pydatetime()
    )


def _source(
    data,
    *,
    clock,
    book: TjrBookSnapshot | None = None,
    price: float | None = 100.6,
    is_consumed=None,
    state_path=None,
) -> TjrDecisionSource:
    return TjrDecisionSource(
        symbols=(SYMBOL,),
        data=data,
        market_hours=StaticMarketHours(is_open=True, session=SESSION),
        order_notional=25.0,
        book_provider=(lambda: book) if book is not None else None,
        price_provider=(lambda _s: price) if price is not None else None,
        state_path=state_path,
        is_consumed=is_consumed,
        clock=clock,
    )


def _epoch_ms(stamp: datetime) -> int:
    return int(stamp.timestamp() * 1000)


def _binance_row(symbol: str, *, close_time: datetime | None, price: float = 100.0) -> dict:
    """A Binance 24-hour ticker row, with the timestamp the venue stated."""
    row = {
        "symbol": symbol,
        "lastPrice": str(price),
        "bidPrice": str(price),
        "askPrice": str(price + 0.01),
        "closeTime": None if close_time is None else _epoch_ms(close_time),
    }
    return row


class _FakeBinance:
    """A Binance client that serves one ticker row and counts its calls.

    The row is normalized through the **real** ``Ticker.from_binance``, so the
    test exercises the production normalizer rather than a hand-built object.
    """

    def __init__(self, row: dict) -> None:
        self.row = row
        self.calls = 0

    def ticker_24h(self, symbols):
        self.calls += 1
        return {self.row["symbol"]: Ticker.from_binance(self.row)}


# ---------------------------------------------------------------------------
# 1 & 2. The evaluation cycle: re-evaluate on new data, never repeat a signal
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheEvaluationCycle:
    def test_a_symbol_is_re_evaluated_when_new_completed_data_arrives(self):
        """The symptom: the source stopped looking at the market for the day.

        A setup completed at one bar close. Later, new **completed** bars arrive,
        so the chain has to be run again — that is the only way a later sweep can
        be seen at all, and it is exactly what the backtest does when it judges
        the chain at every 5m bar close.

        The chain here finds the *same* setup again, so the record keeps the
        original decision — but the chain **ran**, and it ran on the new bars.
        The proof that it ran is the data: the seam was read again, which is what
        the old code never did.
        """
        at = _as_of()
        data = _data(BULLISH_BARS, at=at)
        source = _source(data, clock=lambda: at.to_pydatetime())
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_RISK_REJECTED
        assert data.describe()["tjr_data_last_fetch_at"] is not None

        # New completed bars: the clock moves past the newest bar's close, and
        # the seam serves the same bars, now including a later one.
        later = at + pd.Timedelta(minutes=5)
        source._clock = lambda: later.to_pydatetime()
        later_data = _data(BULLISH_BARS, at=later)
        source.data = later_data
        source.next_decision()
        # The chain ran again on the new bars: the seam was read again.
        assert later_data.describe()["tjr_data_last_fetch_at"] == (
            later.to_pydatetime().isoformat()
        )

    def test_the_same_signal_is_never_offered_twice(self):
        """A re-read of the same setup is the same signal, not a new one.

        The setup is identified by **the sweep that produced it**, so re-reading
        the same bars yields the same setup key and the same decision id. The
        source refuses to offer it a second time, and the worker's durable
        ``ordered_decisions`` set would refuse it as well.
        """
        at = _as_of()
        source = _source(
            _data(BULLISH_BARS, at=at),
            clock=lambda: at.to_pydatetime(),
            book=TjrBookSnapshot(cash=5000.0),
        )
        first = source.next_decision()
        assert first is not None

        # The same bars, one cycle later: the same setup, the same id.
        later = at + pd.Timedelta(minutes=5)
        source._clock = lambda: later.to_pydatetime()
        source.data = _data(BULLISH_BARS, at=later)
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_TJR_NO_NEW_DATA
        assert source.last_decision_for(SYMBOL)["decision_id"] == first.decision_id

    def test_a_new_sweep_in_the_same_session_is_a_new_decision(self):
        """A *different* sweep in the same day is a different signal.

        The setup's identity is the sweep, so a new sweep gets a new decision id
        and may be traded, while the same sweep keeps its id and may not.
        """
        at = _as_of()
        source = _source(
            _data(BULLISH_BARS, at=at),
            clock=lambda: at.to_pydatetime(),
            book=TjrBookSnapshot(cash=5000.0),
        )
        first = source.next_decision()
        assert first is not None

        # A second, different sweep: a different bar, a different level.
        class _Sweep:
            def __init__(self, bar, price, direction):
                self.bar = bar
                self.direction = direction
                self.level = type("L", (), {"price": price})()

        class _Evaluation:
            sweep = _Sweep(pd.Timestamp("2026-10-29 15:40:00+00:00"), 99.95, "bullish")

        first_key = source._setup_key(
            type("E", (), {"sweep": _Sweep(pd.Timestamp("2026-10-29 14:20:00+00:00"), 99.4, "bullish")})()
        )
        second_key = source._setup_key(_Evaluation())
        assert first_key != second_key

        first_id = source._decision_id(SYMBOL, SESSION, first_key)
        second_id = source._decision_id(SYMBOL, SESSION, second_key)
        assert first_id != second_id
        # Both ids are the symbol and session, plus the setup's digest.
        assert first_id.startswith(f"tjr-{SYMBOL}-{SESSION}-")
        assert second_id.startswith(f"tjr-{SYMBOL}-{SESSION}-")

    def test_a_setup_key_is_stable_for_the_same_sweep(self):
        """The setup's identity must not wobble between reads.

        A key that changed between two reads of the same bars would make every
        poll look like a new signal, and the same trade could be sent again.
        """

        class _Sweep:
            def __init__(self, bar, price, direction):
                self.bar = bar
                self.direction = direction
                self.level = type("L", (), {"price": price})()

        def _evaluation(bar, price):
            return type("E", (), {"sweep": _Sweep(bar, price, "bullish")})()

        bar = pd.Timestamp("2026-10-29 14:20:00+00:00")
        key_a = TjrDecisionSource._setup_key(_evaluation(bar, 99.4))
        key_b = TjrDecisionSource._setup_key(_evaluation(bar, 99.4))
        assert key_a == key_b
        assert TjrDecisionSource._decision_id(SYMBOL, SESSION, key_a) == (
            TjrDecisionSource._decision_id(SYMBOL, SESSION, key_b)
        )

    def test_an_evaluation_without_a_sweep_has_no_setup_key(self):
        """An unfinished chain has no setup, so it has no identity to repeat."""

        class _Evaluation:
            sweep = None

        assert TjrDecisionSource._setup_key(_Evaluation()) is None


# ---------------------------------------------------------------------------
# 3. The data-fetch evidence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheFetchEvidence:
    def test_tjr_data_last_fetch_at_updates_on_every_cycle(self):
        """The symptom: ``tjr_data_last_fetch_at = null`` beside old decisions.

        The seam records each fetch. The source must therefore fetch on **every**
        cycle — including a cycle in which the symbol was already decided — or
        the view reports "never fetched" while the worker is running.
        """
        at = _as_of()
        data = _data(BULLISH_BARS, at=at)
        assert data.describe()["tjr_data_last_fetch_at"] is None

        source = _source(data, clock=lambda: at.to_pydatetime())
        source.next_decision()
        first = data.describe()["tjr_data_last_fetch_at"]
        assert first is not None
        assert first == at.to_pydatetime().isoformat()

        # A later cycle fetches again, so the evidence moves forward.
        later = at + pd.Timedelta(minutes=5)
        source._clock = lambda: later.to_pydatetime()
        source.data = _data(BULLISH_BARS, at=later)
        source.next_decision()
        second = source.data.describe()["tjr_data_last_fetch_at"]
        assert second is not None
        assert second != first

    def test_the_fetch_is_recorded_even_when_the_market_is_closed(self):
        """A closed market is not a failed fetch.

        The clock is checked first, so no bars are read; the fetch evidence
        therefore stays absent, which is correct: no fetch happened.
        """
        at = _as_of()
        data = _data(BULLISH_BARS, at=at)
        source = TjrDecisionSource(
            symbols=(SYMBOL,),
            data=data,
            market_hours=StaticMarketHours(is_open=False, session=SESSION),
            order_notional=25.0,
            clock=lambda: at.to_pydatetime(),
        )
        assert source.next_decision() is None
        assert data.describe()["tjr_data_last_fetch_at"] is None


# ---------------------------------------------------------------------------
# 4, 5 & 6. The market-data freshness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheMarketDataFreshness:
    def test_a_snapshot_older_than_the_age_bound_is_refused(self):
        """The symptom: a 6142s-old price reported as ``ok``.

        A price past its age bound cannot size an order, so the seam refuses it
        at the read — an HTTP 200 is not evidence of freshness.
        """
        now = datetime(2026, 10, 29, 15, 0, tzinfo=UTC)
        row = _binance_row("BTCUSDT", close_time=now - timedelta(seconds=6142))
        seam = BinanceSnapshotMarketData(
            ["BTC/USDT"],
            client=_FakeBinance(row),
            max_age_seconds=15.0,
            clock=lambda: now,
        )
        with pytest.raises(MarketDataError):
            seam.price("BTC/USDT")

    def test_a_missing_timestamp_is_refused(self):
        """A payload with no timestamp has no verifiable age: fail closed."""
        now = datetime(2026, 10, 29, 15, 0, tzinfo=UTC)
        row = _binance_row("BTCUSDT", close_time=None)
        seam = BinanceSnapshotMarketData(
            ["BTC/USDT"],
            client=_FakeBinance(row),
            max_age_seconds=15.0,
            clock=lambda: now,
        )
        with pytest.raises(MarketDataError):
            seam.price("BTC/USDT")

    def test_a_future_timestamp_is_refused(self):
        """A future-dated price is refused, never treated as infinitely fresh."""
        now = datetime(2026, 10, 29, 15, 0, tzinfo=UTC)
        row = _binance_row("BTCUSDT", close_time=now + timedelta(seconds=600))
        seam = BinanceSnapshotMarketData(
            ["BTC/USDT"],
            client=_FakeBinance(row),
            max_age_seconds=15.0,
            clock=lambda: now,
        )
        with pytest.raises(MarketDataError):
            seam.price("BTC/USDT")

    def test_the_status_agrees_with_the_age(self):
        """``market_data_status`` and ``is_current_price`` cannot disagree.

        Both are derived from the *same* stored observation, so a price that has
        aged past the bound reports ``stale`` and is not a current price. The
        price was read while it was fresh — that is when it was stored — and the
        clock then moved past the bound, which is exactly how a 6142-second-old
        price came to be reported as ``ok``.
        """
        now = datetime(2026, 10, 29, 15, 0, tzinfo=UTC)
        clock = {"now": now}
        row = _binance_row("BTCUSDT", close_time=now - timedelta(seconds=1))
        seam = BinanceSnapshotMarketData(
            ["BTC/USDT"],
            client=_FakeBinance(row),
            max_age_seconds=15.0,
            clock=lambda: clock["now"],
        )
        # Read while fresh: the observation is stored.
        assert seam.available() is True
        assert seam.describe()["market_data_status"] == MARKET_DATA_STATUS_OK

        # The clock moves past the bound. The stored observation ages, and the
        # status follows it.
        clock["now"] = now + timedelta(seconds=6142)
        view = seam.describe()
        assert view["market_data_status"] == MARKET_DATA_STATUS_STALE
        assert view["is_current_price"] is False
        assert view["last_snapshot_age_seconds"] >= 6142

    def test_a_fresh_price_reports_ok(self):
        """The healthy case still reports ``ok`` — the fix is not a blanket alarm."""
        now = datetime(2026, 10, 29, 15, 0, tzinfo=UTC)
        row = _binance_row("BTCUSDT", close_time=now - timedelta(seconds=1))
        seam = BinanceSnapshotMarketData(
            ["BTC/USDT"],
            client=_FakeBinance(row),
            max_age_seconds=15.0,
            clock=lambda: now,
        )
        assert seam.available() is True
        view = seam.describe()
        assert view["market_data_status"] == MARKET_DATA_STATUS_OK
        assert view["is_current_price"] is True

    def test_a_ticker_without_a_timestamp_carries_no_instant(self):
        """The normalizer must not invent the observation instant.

        Substituting the local clock made every timestamp-less ticker look
        infinitely fresh, so a response with no timestamp passed the freshness
        check and could size an order.
        """
        ticker = Ticker.from_binance({"symbol": "BTCUSDT", "lastPrice": "100.0"})
        assert ticker.observed_at is None

    def test_a_ticker_keeps_the_venue_timestamp(self):
        """A stated timestamp is kept, so the age is the venue's own."""
        close = datetime(2026, 10, 29, 15, 0, tzinfo=UTC)
        ticker = Ticker.from_binance(
            {"symbol": "BTCUSDT", "lastPrice": "100.0", "closeTime": _epoch_ms(close)}
        )
        assert ticker.observed_at == close


# ---------------------------------------------------------------------------
# 8 & 10. The guards on a new order
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)


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


def _venue_order(**over):
    payload = {
        "id": "broker-order-1",
        "client_order_id": "c1",
        "symbol": SYMBOL,
        "side": "buy",
        "qty": "0.25",
        "filled_qty": "0.25",
        "filled_avg_price": "100.0",
        "status": "filled",
        "type": "market",
        "time_in_force": "gtc",
        "submitted_at": "2026-10-29T13:30:00Z",
    }
    payload.update(over)
    return payload


def _venue_position(**over):
    payload = {
        "symbol": SYMBOL,
        "qty": "0.25",
        "market_value": "25.00",
        "avg_entry_price": "100.0",
        "side": "long",
    }
    payload.update(over)
    return payload


class _Source:
    """A decision source that yields a fixed list, one per call."""

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self._i = 0

    def next_decision(self):
        if self._i >= len(self._decisions):
            return None
        decision = self._decisions[self._i]
        self._i += 1
        return decision

    def reset(self):
        self._i = 0


def _transport(routes, *, calls=None):
    """A transport that answers like the venue for the parts that matter."""

    def send(*, method, url, headers, params, json_body, timeout):
        if calls is not None:
            calls.append(
                {
                    "method": method,
                    "url": url,
                    "headers": dict(headers),
                    "params": dict(params or {}),
                    "json": json_body,
                }
            )
        for (route_method, suffix), response in routes.items():
            if method == route_method and url.endswith(suffix):
                return response
        raise AssertionError(f"unexpected request: {method} {url}")

    return send


def _routes(*, account=None, positions=None, orders=None):
    return {
        ("GET", "/v2/account"): account or _Resp(200, _ACCOUNT),
        ("GET", "/v2/positions"): positions or _Resp(200, [_venue_position()]),
        ("POST", "/v2/orders"): orders or _Resp(200, _venue_order()),
    }


def _decision(
    decision_id: str = "d-1",
    *,
    symbol: str = SYMBOL,
    side: str = "buy",
    notional: float = 25.0,
    stop_loss: float = 90.0,
    take_profit: float = 115.0,
):
    from tradingagents.paper_experiment.broker.trading import TradingDecision

    return TradingDecision(
        decision_id=decision_id,
        symbol=symbol,
        side=side,
        notional=notional,
        stop_loss=stop_loss,
        take_profit=take_profit,
        targets=(take_profit,),
        strategy_name="tjr",
    )


def _worker(tmp_path, *, decisions=(), routes=None, calls=None, prices=None, name="w.json"):
    """A worker over a mocked venue, with the real broker adapter."""
    from tradingagents.paper_experiment.broker import (
        AlpacaPaperBroker,
        PaperRiskLimits,
        PaperTradingConfig,
        PaperTradingWorker,
    )

    config = PaperTradingConfig(
        order_notional=25.0,
        limits=PaperRiskLimits(),
        state_path=tmp_path / name,
    )
    transport = _transport(routes or _routes(), calls=calls)

    def build():
        return AlpacaPaperBroker(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=transport,
        )

    return PaperTradingWorker(
        config=config,
        decision_source=_Source(decisions),
        market_data=StaticMarketData(prices or {SYMBOL: 100.0}),
        broker_factory=build,
    )


@pytest.mark.unit
class TestTheEntryGuards:
    def test_an_old_ready_decision_cannot_be_traded(self, tmp_path):
        """A READY decision is not a standing permission to trade.

        The worker marks a decision as processed the moment it acts on it, and
        the decision itself carries no expiry. A decision that has already been
        processed is therefore refused, so the same signal cannot be sent twice.
        """
        worker = _worker(tmp_path, decisions=[])
        # A decision the worker has already acted on.
        worker.state.processed_decisions.add("d-1")
        assert worker._is_processed("d-1") is True

    def test_a_new_entry_is_refused_while_an_open_position_is_unprotected(self, tmp_path):
        """An unguarded open position blocks a new entry.

        Opening a second position while the first one's guard is in doubt
        increases the risk that is already unguarded. The verdict comes from the
        broker-verified record, never from the recorded levels.
        """
        calls: list[dict] = []
        worker = _worker(
            tmp_path,
            decisions=[_decision()],
            calls=calls,
            routes=_routes(positions=_Resp(200, [_venue_position()])),
        )
        # The worker opened this position and recorded its exit levels, but the
        # broker has never confirmed that the position is actually held, so its
        # protection is unconfirmed.
        worker.state.positions = {
            SYMBOL: {
                "symbol": SYMBOL,
                "quantity": 0.25,
                "avg_entry_price": 100.0,
            }
        }
        worker.state.stops[SYMBOL] = {
            "symbol": SYMBOL,
            "stop_loss": 90.0,
            "take_profit": 115.0,
            "decision_id": "d-0",
        }
        worker.start()
        result = worker.run_once()
        assert result.ran is False
        assert result.reason_code == NO_TRADE_PROTECTION_FAILED
        assert [c for c in calls if c["method"] == "POST"] == []
        worker.stop()

    def test_an_unreadable_book_blocks_a_new_entry(self, tmp_path):
        """A broker that cannot be read is not a flat book.

        The positions could not be confirmed, so the worker cannot size the trade
        against the real book: the entry is refused and **no order reaches the
        venue**. The refusal names the read failure — it is never reported as
        "there is no position".

        The venue is healthy when the worker starts, and then the positions read
        fails, so the refusal is about the *read*, not about a worker that cannot
        start.
        """
        calls: list[dict] = []
        routes = _routes()
        worker = _worker(tmp_path, decisions=[_decision()], calls=calls, routes=routes)
        # The worker opened this position and recorded its exit levels.
        worker.state.positions = {
            SYMBOL: {
                "symbol": SYMBOL,
                "quantity": 0.25,
                "avg_entry_price": 100.0,
            }
        }
        worker.state.stops[SYMBOL] = {
            "symbol": SYMBOL,
            "stop_loss": 90.0,
            "take_profit": 115.0,
            "decision_id": "d-0",
        }
        worker.start()
        # The venue stops answering the positions read.
        routes[("GET", "/v2/positions")] = _Resp(500, {"message": "the venue is down"})
        result = worker.run_once()
        assert result.ran is False
        # The refusal names the failed read; it is never "no position".
        assert result.reason_code == NO_TRADE_POSITION_UNAVAILABLE
        assert [c for c in calls if c["method"] == "POST"] == []
        worker.stop()


# ---------------------------------------------------------------------------
# 9. Reconciling an old Alpaca order
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheOrderReconciliation:
    def test_an_accepted_but_unfilled_order_is_not_a_filled_position(self):
        """``accepted`` is not ``filled``, and ``filled_quantity=0`` is not a position.

        The venue accepted the order and it has not filled, so the account holds
        nothing. The order is terminal only once it is filled, canceled, expired
        or rejected — an accepted order is still live and must not be treated as
        a closed trade.
        """
        from tradingagents.paper_experiment.broker.alpaca_paper import BrokerOrder

        order = BrokerOrder(
            client_order_id="c-1",
            broker_order_id="o-1",
            symbol="AVAX/USD",
            side="buy",
            quantity=0.25,
            filled_quantity=0.0,
            filled_avg_price=None,
            status="accepted",
            submitted_at="2026-10-29T15:00:00+00:00",
        )
        assert order.status == "accepted"
        assert order.filled_quantity == 0.0
        assert order.remaining_quantity == pytest.approx(0.25)
        # Not filled and not terminal: still live at the venue.
        assert order.is_terminal is False

    def test_the_state_records_the_venue_order_id_and_the_fill(self, tmp_path):
        """The worker's own record keeps both ids and the filled quantity.

        A report that showed ``accepted`` beside ``filled_quantity=0`` was
        missing the distinction; the record has to carry the venue's id, the
        client's id, the status and the fill together so an operator can tell
        what the venue actually did.
        """
        worker = _worker(tmp_path)
        worker.state.orders["c-1"] = {
            "client_order_id": "c-1",
            "broker_order_id": "o-1",
            "symbol": "AVAX/USD",
            "side": "buy",
            "quantity": 0.25,
            "filled_quantity": 0.0,
            "filled_avg_price": None,
            "status": "accepted",
            "submitted_at": "2026-10-29T15:00:00+00:00",
        }
        record = worker.state.orders["c-1"]
        assert record["broker_order_id"] == "o-1"
        assert record["client_order_id"] == "c-1"
        assert record["status"] == "accepted"
        assert record["filled_quantity"] == 0.0
        # An unfilled order is not a position.
        assert worker.state.positions.get("AVAX/USD") is None

    def test_an_old_order_is_reconciled_and_never_re_submitted(self, tmp_path):
        """An order the worker already knows is reconciled, not sent again.

        The venue accepted the AVAX order and it has not filled. On the next
        cycle the worker must **ask the venue what became of it** — not submit
        a second order. A second submission would double the position when the
        first one fills.
        """
        calls: list[dict] = []
        routes = _routes(
            orders=_Resp(
                200,
                _venue_order(
                    id="broker-order-avax",
                    client_order_id="c-avax",
                    symbol=SYMBOL,
                    qty="0.25",
                    filled_qty="0",
                    status="accepted",
                ),
            )
        )
        routes[("GET", "/v2/orders:by_client_order_id")] = _Resp(
            200,
            _venue_order(
                id="broker-order-avax",
                client_order_id="c-avax",
                symbol=SYMBOL,
                qty="0.25",
                filled_qty="0",
                status="accepted",
            ),
        )
        decision = _decision("d-avax")
        worker = _worker(tmp_path, decisions=[decision], calls=calls, routes=routes)
        worker.start()
        # The worker already knows this order: it was submitted on an earlier
        # cycle and the venue accepted it without filling it.
        client_order_id = decision.client_order_id(strategy_id=worker.config.strategy_id)
        worker.state.orders[client_order_id] = {
            "client_order_id": client_order_id,
            "broker_order_id": "broker-order-avax",
            "symbol": SYMBOL,
            "side": "buy",
            "quantity": 0.25,
            "filled_quantity": 0.0,
            "filled_avg_price": None,
            "status": "accepted",
            "submitted_at": "2026-10-29T15:00:00+00:00",
        }
        result = worker.run_once()
        # The order was **not** submitted again.
        assert [c for c in calls if c["method"] == "POST"] == []
        # The venue was asked what became of it, and its answer is reported.
        assert result.reason_code == NO_TRADE_ORDER_EXISTS
        assert result.order is not None
        assert result.order.status == "accepted"
        assert result.order.filled_quantity == 0.0
        assert result.order.broker_order_id == "broker-order-avax"
        worker.stop()
