"""SELL is a *close* on the paper worker's long-only book, never a short.

The defect these tests pin: ``PaperTradingWorker._size`` sized a SELL from the
configured dollar notional, exactly like a BUY. A close of a position worth more
than the notional therefore sold a tiny fraction of it — or floored to zero — so
a strategy could not actually close what it held. The fix sizes a SELL from the
held quantity, clamped to it, and lets a close larger than
``max_order_notional`` through, because a close only *reduces* exposure.

The claims:

- a SELL with 5 held closes 5, not ``25 / price``;
- a SELL with nothing held is a named ``NO_POSITION`` no-trade, and sends no order;
- a SELL never exceeds the held quantity, whatever the decision asks for;
- a close worth more than the entry cap is allowed (the cap is unchanged, it just
  does not bound a reduction);
- BUY sizing is unchanged;
- ``allow_short=False`` still refuses a genuine short at the broker;
- the existing stop/target exit path is unchanged;
- a decision is not ordered twice, and the daily-order cap still applies.

Every test drives a **mocked HTTP transport** (the suite's autouse guard refuses
the network) and no test needs an Alpaca key.
"""

from __future__ import annotations

import json

import pytest

from tradingagents.paper_experiment.broker import (
    NO_TRADE_ALREADY_PROCESSED,
    NO_TRADE_NO_POSITION,
    NO_TRADE_POSITION_UNAVAILABLE,
    NO_TRADE_REASONS,
    NO_TRADE_RISK_REJECTED,
    AlpacaPaperBroker,
    ListDecisionSource,
    NoPositionError,
    PaperRiskLimits,
    PaperRiskRejected,
    PaperTradingConfig,
    PaperTradingWorker,
    PositionReadError,
    StaticMarketData,
    TradingDecision,
    WorkerError,
)

SYMBOL = "SPY"
PRICE = 100.0

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


def _order(**over):
    payload = {
        "id": "broker-order-1",
        "client_order_id": "c1",
        "symbol": SYMBOL,
        "side": "buy",
        "qty": "1",
        "filled_qty": "0",
        "filled_avg_price": None,
        "status": "accepted",
        "type": "market",
        "time_in_force": "day",
        "submitted_at": "2026-10-29T13:30:00Z",
    }
    payload.update(over)
    return payload


def _position(*, qty: float):
    return {
        "symbol": SYMBOL,
        "qty": str(qty),
        "side": "long",
        "avg_entry_price": str(PRICE),
        "market_value": str(qty * PRICE),
    }


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)


def _transport(routes, *, calls=None, fail_positions=None):
    """A transport over ``routes``. ``fail_positions`` may be a bool or a callable.

    A callable is used when the test needs preflight's position read to succeed and
    a *later* read to fail, so a genuine mid-cycle broker failure can be simulated.
    """

    def send(*, method, url, headers, params, json_body, timeout):
        if calls is not None:
            calls.append({"method": method, "url": url, "json": json_body})
        if method == "GET" and url.endswith("/v2/positions") and fail_positions is not None:
            should_fail = fail_positions() if callable(fail_positions) else fail_positions
            if should_fail:
                raise TimeoutError("simulated positions read failure")
        for (m, suffix), response in routes.items():
            if method == m and url.endswith(suffix):
                return response() if callable(response) else response
        raise AssertionError(f"unexpected request {method} {url}")

    return send


def _routes(*, positions=(), orders=None, by_client_order_id=None):
    return {
        ("GET", "/v2/account"): _Resp(200, _ACCOUNT),
        ("GET", "/v2/positions"): _Resp(200, list(positions)),
        ("POST", "/v2/orders"): orders or _Resp(200, _order()),
        ("GET", "/v2/orders:by_client_order_id"): by_client_order_id
        or _Resp(200, _order()),
    }


def _worker(
    tmp_path,
    *,
    decisions=(),
    positions=(),
    orders=None,
    calls=None,
    limits=None,
    broker_limits=None,
    fail_positions=False,
    order_notional=25.0,
    whole_shares=False,
):
    routes = _routes(positions=positions, orders=orders)
    broker_limits = broker_limits or limits or PaperRiskLimits()

    def factory():
        return AlpacaPaperBroker(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=_transport(routes, calls=calls, fail_positions=fail_positions),
            limits=broker_limits,
        )

    config = PaperTradingConfig(
        order_notional=order_notional,
        whole_shares=whole_shares,
        limits=limits or PaperRiskLimits(),
        state_path=tmp_path / "worker_state.json",
    )
    return PaperTradingWorker(
        config=config,
        decision_source=ListDecisionSource(list(decisions)),
        market_data=StaticMarketData({SYMBOL: PRICE}),
        broker_factory=factory,
    )


def _decision(side, *, decision_id="d1", quantity=None, notional=None):
    return TradingDecision(
        decision_id=decision_id,
        symbol=SYMBOL,
        side=side,
        notional=None if quantity is not None else (25.0 if notional is None else notional),
        quantity=quantity,
    )


def _posts(calls):
    return [c for c in calls if c["method"] == "POST"]


def _sell_order(qty: float):
    return _Resp(
        200,
        _order(
            side="sell",
            qty=str(qty),
            filled_qty=str(qty),
            filled_avg_price=str(PRICE),
            status="filled",
        ),
    )


# ---------------------------------------------------------------------------
# 1. A SELL closes the held long, not the notional
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSellClosesTheHeldLong:
    def test_holding_five_and_a_sell_closes_five_not_the_notional(self, tmp_path):
        calls = []
        worker = _worker(
            tmp_path,
            decisions=[_decision("sell")],  # notional 25, no explicit quantity
            positions=[_position(qty=5)],
            orders=_sell_order(5),
            calls=calls,
        )
        worker.start()
        result = worker.run_once()
        worker.stop()

        assert result.ran is True
        assert result.order is not None and result.order.side == "sell"
        posts = _posts(calls)
        assert len(posts) == 1
        # 25 / 100 would be 0.25; the close must be the full 5.
        assert float(posts[0]["json"]["qty"]) == pytest.approx(5.0)
        assert posts[0]["json"]["side"] == "sell"

    def test_the_sell_quantity_is_the_held_quantity_not_a_dollar_target(self, tmp_path):
        worker = _worker(tmp_path, positions=[_position(qty=5)], whole_shares=True)
        worker.start()
        try:
            assert worker._size(_decision("sell"), PRICE) == pytest.approx(5.0)  # noqa: SLF001
        finally:
            worker.stop()

    def test_a_sell_quantity_above_the_holding_is_clamped_to_it(self, tmp_path):
        calls = []
        worker = _worker(
            tmp_path,
            decisions=[_decision("sell", quantity=99)],
            positions=[_position(qty=5)],
            orders=_sell_order(5),
            calls=calls,
        )
        worker.start()
        result = worker.run_once()
        worker.stop()

        assert result.ran is True
        assert float(_posts(calls)[0]["json"]["qty"]) == pytest.approx(5.0)

    def test_an_explicit_partial_close_is_honoured(self, tmp_path):
        calls = []
        worker = _worker(
            tmp_path,
            decisions=[_decision("sell", quantity=2)],
            positions=[_position(qty=5)],
            orders=_sell_order(2),
            calls=calls,
        )
        worker.start()
        worker.run_once()
        worker.stop()
        assert float(_posts(calls)[0]["json"]["qty"]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 2. A SELL with no position is a named no-trade, never a short
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSellWithNoPosition:
    def test_a_sell_with_no_position_places_no_order_and_is_named(self, tmp_path):
        calls = []
        worker = _worker(
            tmp_path,
            decisions=[_decision("sell")],
            positions=[],  # flat
            calls=calls,
        )
        worker.start()
        result = worker.run_once()
        worker.stop()

        assert result.ran is False
        assert result.reason_code == NO_TRADE_NO_POSITION
        assert result.reason_code != NO_TRADE_RISK_REJECTED
        assert result.error is None
        assert _posts(calls) == []

    def test_a_sell_with_no_position_raises_a_named_no_position_error(self, tmp_path):
        worker = _worker(tmp_path, positions=[])
        worker.start()
        try:
            with pytest.raises(NoPositionError):
                worker._size(_decision("sell"), PRICE)  # noqa: SLF001
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# 3. A close larger than the entry cap is allowed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCloseBeyondTheEntryCap:
    def test_a_close_worth_more_than_the_cap_is_allowed(self, tmp_path):
        """5 shares at 100 is a 500 notional close under a 25 cap: it must go."""
        calls = []
        worker = _worker(
            tmp_path,
            decisions=[_decision("sell")],
            positions=[_position(qty=5)],
            orders=_sell_order(5),
            calls=calls,
            limits=PaperRiskLimits(max_order_notional=25.0),
        )
        worker.start()
        result = worker.run_once()
        worker.stop()

        assert result.ran is True
        assert len(_posts(calls)) == 1
        assert float(_posts(calls)[0]["json"]["qty"]) == pytest.approx(5.0)

    def test_the_broker_still_caps_a_large_buy(self):
        """The cap is unchanged: it still bounds an entry."""
        broker = AlpacaPaperBroker(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=_transport(_routes()),
            limits=PaperRiskLimits(max_order_notional=25.0),
        )
        broker.connect()
        with pytest.raises(PaperRiskRejected, match="exceeds the cap"):
            broker.submit_order(
                symbol=SYMBOL,
                quantity=5,
                side="buy",
                reference_price=PRICE,
                client_order_id="c1",
                allow_orders=True,
            )


# ---------------------------------------------------------------------------
# 4. allow_short=False still refuses a genuine short at the broker
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShortIsStillRefused:
    def test_the_broker_refuses_a_sell_beyond_the_holding(self):
        broker = AlpacaPaperBroker(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=_transport(_routes(positions=[_position(qty=5)])),
            limits=PaperRiskLimits(max_order_notional=25.0),
        )
        broker.connect()
        broker.positions(refresh=True)
        # The sell is not blocked by the entry cap (it is a reduction); it is
        # refused by the long-only guard, which is what this pins.
        with pytest.raises(PaperRiskRejected, match="short"):
            broker.submit_order(
                symbol=SYMBOL,
                quantity=10,
                side="sell",
                reference_price=PRICE,
                client_order_id="c1",
                allow_orders=True,
            )

    def test_allow_short_is_still_false(self):
        assert PaperRiskLimits().allow_short is False


# ---------------------------------------------------------------------------
# 5. BUY sizing is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuyIsUnchanged:
    def test_a_buy_is_still_sized_from_the_notional(self, tmp_path):
        worker = _worker(tmp_path, whole_shares=False)
        assert worker._size(_decision("buy", notional=25.0), PRICE) == pytest.approx(  # noqa: SLF001
            0.25
        )

    def test_a_buy_with_whole_shares_still_floors_and_refuses(self, tmp_path):
        worker = _worker(tmp_path, whole_shares=True)
        with pytest.raises(WorkerError, match="fractional"):
            worker._size(_decision("buy", notional=25.0), PRICE)  # noqa: SLF001


# ---------------------------------------------------------------------------
# 6. The stop/target exit path is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExitPathUnchanged:
    def test_a_triggered_stop_still_closes_the_held_long(self, tmp_path):
        calls = []
        worker = _worker(
            tmp_path,
            decisions=[],  # no new decision; the exit policy runs first
            positions=[_position(qty=1)],
            orders=_sell_order(1),
            calls=calls,
        )
        worker.start()
        # Seed a long the worker holds with a stop above the price, so it triggers.
        worker.state.stops[SYMBOL] = {"symbol": SYMBOL, "stop_loss": 200.0, "take_profit": None}
        worker.state.positions = {SYMBOL: {"symbol": SYMBOL, "quantity": 1.0}}
        worker._save()

        result = worker.run_once()
        worker.stop()

        assert result.ran is True
        assert result.order is not None and result.order.side == "sell"
        posts = _posts(calls)
        assert len(posts) == 1
        assert posts[0]["json"]["side"] == "sell"
        assert float(posts[0]["json"]["qty"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 7. Idempotency and the daily-order cap
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIdempotencyAndLimits:
    def test_the_same_decision_is_not_ordered_twice(self, tmp_path):
        calls = []
        source = ListDecisionSource([_decision("sell", decision_id="dup")])
        worker = _worker(
            tmp_path, positions=[_position(qty=5)], orders=_sell_order(5), calls=calls
        )
        worker.decision_source = source
        worker.start()
        first = worker.run_once()
        # Rewind the source so the same decision_id is offered again.
        source.reset()
        second = worker.run_once()
        worker.stop()

        assert first.ran is True
        assert second.ran is False
        assert second.reason_code == NO_TRADE_ALREADY_PROCESSED
        assert len(_posts(calls)) == 1

    def test_the_daily_order_cap_still_applies(self, tmp_path):
        calls = []
        limits = PaperRiskLimits(max_order_notional=25.0, max_daily_orders=1)
        worker = _worker(
            tmp_path,
            decisions=[_decision("sell", decision_id="s1"), _decision("sell", decision_id="s2")],
            positions=[_position(qty=5)],
            orders=_sell_order(5),
            calls=calls,
            limits=limits,
            broker_limits=limits,
        )
        worker.start()
        first = worker.run_once()
        second = worker.run_once()
        worker.stop()

        assert first.ran is True
        assert second.ran is False
        assert len(_posts(calls)) == 1


# ---------------------------------------------------------------------------
# 8. A broker read failure on a SELL is not swallowed as "no position"
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPositionReadFailureIsNotSwallowed:
    def test_a_positions_read_failure_is_reported_apart_from_no_position(self, tmp_path):
        calls = []
        reads = {"n": 0}

        def fail_after_preflight():
            # Preflight's read (the first) must succeed so the worker starts; a
            # later read failing is the mid-cycle broker failure under test.
            reads["n"] += 1
            return reads["n"] > 1

        worker = _worker(
            tmp_path,
            decisions=[_decision("sell")],
            calls=calls,
            fail_positions=fail_after_preflight,
        )
        worker.start()
        result = worker.run_once()
        worker.stop()

        assert result.ran is False
        assert result.reason_code == NO_TRADE_POSITION_UNAVAILABLE
        assert result.reason_code != NO_TRADE_NO_POSITION
        assert _posts(calls) == []

    def test_a_positions_read_failure_raises_a_named_error(self, tmp_path):
        reads = {"n": 0}

        def fail_after_preflight():
            reads["n"] += 1
            return reads["n"] > 1

        worker = _worker(tmp_path, fail_positions=fail_after_preflight)
        worker.start()
        try:
            with pytest.raises(PositionReadError):
                worker._size(_decision("sell"), PRICE)  # noqa: SLF001
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# 9. The new codes are in the closed vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVocabulary:
    def test_the_new_codes_are_in_the_closed_vocabulary(self):
        assert NO_TRADE_NO_POSITION in NO_TRADE_REASONS
        assert NO_TRADE_POSITION_UNAVAILABLE in NO_TRADE_REASONS
