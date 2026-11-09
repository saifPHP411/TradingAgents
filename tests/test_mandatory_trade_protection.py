"""The mandatory trade protection: every open trade carries a stop and a target.

The rule this module pins: **a trade that cannot be protected is not taken.**
Alpaca's crypto API has no bracket order and no OCO, so a stop and a target
cannot be left resting at the venue — the worker's exit monitor is the only
protection mechanism there is. That makes the gate that runs *before* the
order the load-bearing part: if it lets an unprotectable trade through, the
position exists with nothing watching it.

Each test drives a **mocked HTTP transport**; no test reaches the network and
no test needs a credential. The adapter's endpoint validation, risk checks and
idempotency all run for real.

The claims pinned here:

- a BUY with no stop is refused, and no order reaches the broker;
- a BUY with no target is refused;
- a BUY whose stop sits on the wrong side of the entry is refused;
- a BUY whose reward does not justify its risk is refused;
- a BUY whose levels are fine reaches the broker;
- the levels are recorded when the entry fills, and the protection is *pending*
  until the broker confirms the position — never "protected" on the strength of
  a record alone;
- a position the broker does not hold is reported unprotected, not protected;
- the recorded levels survive a restart;
- the risk limits are unchanged by any of this: $25 max order notional, five max
  open positions, ten max daily orders, long only;
- the strategy is unchanged: the gate reads the levels the strategy derived and
  never alters one.
"""

from __future__ import annotations

import contextlib
import json

import pytest

from tradingagents.paper_experiment.broker import (
    NO_TRADE_INVALID_RISK_REWARD,
    NO_TRADE_NO_VALID_STOP_LOSS,
    NO_TRADE_NO_VALID_TAKE_PROFIT,
    NO_TRADE_PROTECTION_UNAVAILABLE,
    NO_TRADE_REASONS,
    PROTECTION_STATUS_CLOSED,
    PROTECTION_STATUS_PARTIALLY_CLOSED,
    PROTECTION_STATUS_PENDING,
    PROTECTION_STATUS_PROTECTED,
    AlpacaPaperBroker,
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    PreflightError,
    StaticMarketData,
    TradingDecision,
)
from tradingagents.paper_experiment.protection import (
    INVALID_RISK_REWARD,
    NO_VALID_STOP_LOSS,
    NO_VALID_TAKE_PROFIT,
    PROTECTION_REASON_CODES,
    ProtectionPlan,
    ProtectionRefusal,
    validate_protection,
)

SYMBOL = "SPY"

_ACCOUNT = {
    "id": "paper-acct-1",
    "status": "ACTIVE",
    "cash": "50.00",
    "equity": "50.00",
    "buying_power": "50.00",
    "currency": "USD",
    "trading_blocked": False,
    "account_blocked": False,
    "shorting_enabled": False,
}


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
        "qty": "2",
        "filled_qty": "2",
        "filled_avg_price": "10.0",
        "status": "filled",
        "type": "market",
        "time_in_force": "day",
        "submitted_at": "2026-10-29T13:30:00Z",
    }
    payload.update(over)
    return payload


def _position(**over):
    payload = {
        "symbol": SYMBOL,
        "qty": "2",
        "market_value": "20.00",
        "avg_entry_price": "10.0",
        "side": "long",
    }
    payload.update(over)
    return payload


def _transport(routes, *, calls=None, held=None):
    """A transport that behaves like the venue for the parts that matter.

    An order's fill reports the quantity the order actually asked for, a later
    lookup by client id returns that same order, and a sell reduces the
    position the venue reports. A fixed fill quantity or a static position
    would make a *partial* exit look like a full close, which is the opposite
    of what the venue does.
    """
    placed: dict[str, dict] = {}
    position = {"qty": held}

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
        if method == "POST" and url.endswith("/v2/orders"):
            requested = str(json_body.get("qty", ""))
            payload = _order(
                qty=requested,
                filled_qty=requested,
                side=json_body.get("side", "buy"),
                client_order_id=json_body.get("client_order_id", "c1"),
            )
            placed[payload["client_order_id"]] = payload
            if payload["side"] == "sell" and position["qty"] is not None:
                position["qty"] = max(
                    0.0, float(position["qty"]) - float(requested or 0.0)
                )
            return _Resp(200, payload)
        if method == "GET" and url.endswith("/v2/orders:by_client_order_id"):
            client_id = (params or {}).get("client_order_id")
            if client_id in placed:
                return _Resp(200, placed[client_id])
        if (
            method == "GET"
            and url.endswith("/v2/positions")
            and position["qty"] is not None
        ):
            return _Resp(200, [_position(qty=str(position["qty"]))])
        for (route_method, suffix), response in routes.items():
            if method == route_method and url.endswith(suffix):
                return response
        raise AssertionError(f"unexpected request: {method} {url}")

    return send


def _sells(calls):
    """The sell orders among the recorded requests (the exits, not the entry)."""
    return [
        c
        for c in calls
        if c["method"] == "POST" and (c.get("json") or {}).get("side") == "sell"
    ]


def _routes(*, account=None, positions=None, orders=None, by_client_order_id=None):
    return {
        ("GET", "/v2/account"): account or _Resp(200, _ACCOUNT),
        ("GET", "/v2/positions"): positions or _Resp(200, [_position()]),
        ("POST", "/v2/orders"): orders or _Resp(200, _order()),
        ("GET", "/v2/orders:by_client_order_id"): by_client_order_id
        or _Resp(200, _order()),
    }


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


class _MutablePrice:
    """A market-data seam whose price can move between cycles."""

    def __init__(self, price):
        self.price_value = float(price)

    def price(self, symbol):
        return float(self.price_value)

    def available(self):
        return True

    def identity(self):
        return {
            "market_data_provider": "static",
            "market_data_kind": "test",
            "market_data_status": "ok",
        }


def _decision(
    decision_id="d1",
    *,
    symbol=SYMBOL,
    side="buy",
    notional=25.0,
    quantity=None,
    stop_loss=9.0,
    take_profit=12.0,
    targets=(),
):
    """A decision, with the levels left off by default so a test has to opt in."""
    return TradingDecision(
        decision_id=decision_id,
        symbol=symbol,
        side=side,
        notional=None if quantity is not None else notional,
        quantity=quantity,
        stop_loss=stop_loss,
        take_profit=take_profit,
        targets=tuple(targets),
    )


_WORKERS: list[PaperTradingWorker] = []


@pytest.fixture(autouse=True)
def _stop_workers():
    """Stop every worker started in a test, so no loop outlives the test.

    ``start()`` launches a non-terminating loop; without this teardown those
    loops would keep running (and keep writing state) for the rest of the
    session.
    """
    start = len(_WORKERS)
    yield
    for worker in _WORKERS[start:]:
        with contextlib.suppress(Exception):
            worker.stop()
    del _WORKERS[start:]


def _worker(
    tmp_path,
    *,
    decisions=(),
    prices=None,
    routes=None,
    calls=None,
    limits=None,
    order_notional=25.0,
    min_reward_risk=1.0,
    max_risk_per_trade=None,
    name="worker_state.json",
    market_data=None,
    held=None,
):
    config = PaperTradingConfig(
        order_notional=order_notional,
        min_reward_risk=min_reward_risk,
        max_risk_per_trade=max_risk_per_trade,
        limits=limits or PaperRiskLimits(),
        state_path=tmp_path / name,
    )
    # The transport is built **once** and shared, so the venue's own view of
    # the position survives from one cycle to the next — a partial exit has to
    # reduce it, or the next rung would be sized against a stale holding.
    transport = _transport(routes or _routes(), calls=calls, held=held)

    def build():
        return AlpacaPaperBroker(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=transport,
            limits=limits or PaperRiskLimits(),
        )

    worker = PaperTradingWorker(
        config=config,
        decision_source=_Source(decisions),
        market_data=market_data or StaticMarketData(prices or {SYMBOL: 10.0}),
        broker_factory=build,
    )
    _WORKERS.append(worker)
    return worker


# ---------------------------------------------------------------------------
# 1. The validator itself
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheValidator:
    def test_valid_levels_produce_a_plan(self):
        verdict = validate_protection(
            entry=10.0, stop_loss=9.0, targets=[12.0], min_reward_risk=1.0
        )
        assert isinstance(verdict, ProtectionPlan)
        assert verdict.stop_loss == 9.0
        assert verdict.take_profit == 12.0
        assert verdict.risk == pytest.approx(1.0)
        assert verdict.reward == pytest.approx(2.0)
        assert verdict.risk_reward == pytest.approx(2.0)

    def test_a_missing_stop_is_refused_with_its_own_reason(self):
        verdict = validate_protection(entry=10.0, stop_loss=None, targets=[12.0])
        assert isinstance(verdict, ProtectionRefusal)
        assert verdict.reason_code == NO_VALID_STOP_LOSS

    def test_a_missing_target_is_refused_with_its_own_reason(self):
        verdict = validate_protection(entry=10.0, stop_loss=9.0, targets=[])
        assert isinstance(verdict, ProtectionRefusal)
        assert verdict.reason_code == NO_VALID_TAKE_PROFIT

    def test_a_stop_on_the_wrong_side_is_refused(self):
        verdict = validate_protection(entry=10.0, stop_loss=11.0, targets=[12.0])
        assert isinstance(verdict, ProtectionRefusal)
        assert verdict.reason_code == NO_VALID_STOP_LOSS

    def test_a_target_on_the_wrong_side_is_refused(self):
        verdict = validate_protection(entry=10.0, stop_loss=9.0, targets=[9.5])
        assert isinstance(verdict, ProtectionRefusal)
        assert verdict.reason_code == NO_VALID_TAKE_PROFIT

    def test_a_reward_below_the_required_multiple_is_refused(self):
        verdict = validate_protection(
            entry=10.0, stop_loss=9.0, targets=[10.5], min_reward_risk=1.0
        )
        assert isinstance(verdict, ProtectionRefusal)
        assert verdict.reason_code == INVALID_RISK_REWARD

    def test_a_nan_level_is_refused_rather_than_accepted(self):
        # NaN passes every naive ``> 0`` guard, because every comparison against
        # it is false. It must not survive the validator.
        verdict = validate_protection(entry=10.0, stop_loss=float("nan"), targets=[12.0])
        assert isinstance(verdict, ProtectionRefusal)
        assert verdict.reason_code == NO_VALID_STOP_LOSS

    def test_an_infinite_level_is_refused(self):
        verdict = validate_protection(
            entry=10.0, stop_loss=9.0, targets=[float("inf")]
        )
        assert isinstance(verdict, ProtectionRefusal)
        assert verdict.reason_code == NO_VALID_TAKE_PROFIT

    def test_the_reason_vocabulary_is_closed(self):
        assert set(PROTECTION_REASON_CODES) == {
            NO_VALID_STOP_LOSS,
            NO_VALID_TAKE_PROFIT,
            INVALID_RISK_REWARD,
            "INVALID_TARGET_ORDER",
            "PROTECTION_UNAVAILABLE",
        }


# ---------------------------------------------------------------------------
# 2. The gate refuses an unprotectable BUY before it reaches the broker
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheGateRefusesBeforeTheBroker:
    def test_a_buy_without_a_stop_places_no_order(self, tmp_path):
        calls: list[dict] = []
        worker = _worker(
            tmp_path,
            decisions=[_decision(stop_loss=None, take_profit=12.0)],
            calls=calls,
        )
        worker.start()
        result = worker.run_once()

        assert result.ran is False
        assert result.reason_code == NO_TRADE_NO_VALID_STOP_LOSS
        assert result.reason_code in NO_TRADE_REASONS
        assert not any(c["method"] == "POST" for c in calls)

    def test_a_buy_without_a_target_places_no_order(self, tmp_path):
        calls: list[dict] = []
        worker = _worker(
            tmp_path,
            decisions=[_decision(stop_loss=9.0, take_profit=None)],
            calls=calls,
        )
        worker.start()
        result = worker.run_once()

        assert result.ran is False
        assert result.reason_code == NO_TRADE_NO_VALID_TAKE_PROFIT
        assert not any(c["method"] == "POST" for c in calls)

    def test_a_buy_with_an_inverted_stop_places_no_order(self, tmp_path):
        calls: list[dict] = []
        worker = _worker(
            tmp_path,
            decisions=[_decision(stop_loss=11.0, take_profit=12.0)],
            calls=calls,
        )
        worker.start()
        result = worker.run_once()

        assert result.ran is False
        assert result.reason_code == NO_TRADE_NO_VALID_STOP_LOSS
        assert not any(c["method"] == "POST" for c in calls)

    def test_a_buy_with_a_bad_reward_risk_places_no_order(self, tmp_path):
        # A 0.5-wide stop against a 0.6-wide target is 1.2R, which is fine at
        # the default 1.0R requirement — so the reward/risk refusal needs a
        # target that is genuinely too close: a 1.0-wide stop against a 0.5
        # target is 0.5R.
        calls: list[dict] = []
        worker = _worker(
            tmp_path,
            decisions=[_decision(stop_loss=9.0, take_profit=10.5)],
            calls=calls,
        )
        worker.start()
        result = worker.run_once()

        assert result.ran is False
        assert result.reason_code == NO_TRADE_INVALID_RISK_REWARD
        assert not any(c["method"] == "POST" for c in calls)

    def test_a_buy_with_no_price_seam_places_no_order(self, tmp_path):
        # The exit monitor needs a price to watch the levels against. Without
        # one the position could not be protected, so the worker refuses to
        # start at all — a stronger guarantee than refusing the trade, because
        # no cycle ever runs.
        calls: list[dict] = []
        worker = _worker(
            tmp_path,
            decisions=[_decision()],
            calls=calls,
            market_data=StaticMarketData({}),
        )
        with pytest.raises(PreflightError):
            worker.start()
        assert not any(c["method"] == "POST" for c in calls)

    def test_a_buy_is_refused_when_the_exit_monitor_cannot_be_armed(self, tmp_path):
        # The levels are fine, but the monitor cannot be armed: the broker is
        # gone, so there is nowhere to send the exit. The trade is refused
        # rather than opened and left unguarded.
        worker = _worker(tmp_path, decisions=[_decision()])
        worker.start()
        worker._broker = None

        refusal = worker._check_protection(worker.decision_source.next_decision(), 10.0)

        assert refusal is not None
        assert refusal[0] == NO_TRADE_PROTECTION_UNAVAILABLE

    def test_a_protectable_buy_reaches_the_broker(self, tmp_path):
        calls: list[dict] = []
        worker = _worker(tmp_path, decisions=[_decision()], calls=calls)
        worker.start()
        result = worker.run_once()

        assert result.ran is True
        assert any(c["method"] == "POST" and c["url"].endswith("/v2/orders") for c in calls)

    def test_a_refused_decision_is_not_retried_forever(self, tmp_path):
        calls: list[dict] = []
        worker = _worker(
            tmp_path,
            decisions=[_decision(stop_loss=None, take_profit=12.0)],
            calls=calls,
        )
        worker.start()
        worker.run_once()
        worker.run_once()

        assert not any(c["method"] == "POST" for c in calls)


# ---------------------------------------------------------------------------
# 3. The protection is bound to the position after the fill
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheProtectionIsBound:
    def test_the_levels_are_recorded_when_the_entry_fills(self, tmp_path):
        worker = _worker(tmp_path, decisions=[_decision()])
        worker.start()
        worker.run_once()

        record = worker.state.protections[SYMBOL]
        assert record["stop_loss"] == 9.0
        assert record["take_profit"] == 12.0
        assert record["entry_price"] == pytest.approx(10.0)
        assert record["status"] == PROTECTION_STATUS_PENDING

    def test_a_recorded_level_alone_is_not_a_protected_position(self, tmp_path):
        # The entry filled, but the broker has not confirmed the position yet.
        # Reporting "protected" on the strength of a record is the lie this
        # whole feature exists to prevent.
        worker = _worker(tmp_path, decisions=[_decision()])
        worker.start()
        worker.run_once()

        assert worker.state.protections[SYMBOL]["status"] == PROTECTION_STATUS_PENDING

    def test_a_position_the_broker_holds_is_protected(self, tmp_path):
        # The entry fills two units and the venue reports the same two units
        # held, so the position is exactly what the worker recorded.
        worker = _worker(
            tmp_path, decisions=[_decision(quantity=2.0)], held=2.0
        )
        worker.start()
        worker.run_once()
        worker._verify_all_protections()

        record = worker.state.protections[SYMBOL]
        assert record["status"] == PROTECTION_STATUS_PROTECTED
        assert worker.state.protection_status == PROTECTION_STATUS_PROTECTED

    def test_a_position_the_broker_does_not_hold_is_unprotected(self, tmp_path):
        routes = _routes(positions=_Resp(200, []))
        worker = _worker(tmp_path, decisions=[_decision()], routes=routes)
        worker.start()
        worker.run_once()
        worker._verify_all_protections()

        record = worker.state.protections[SYMBOL]
        assert record["status"] == PROTECTION_STATUS_CLOSED
        # The worker-level verdict is the worst of the positions' verdicts.
        assert worker.state.protection_status != PROTECTION_STATUS_PROTECTED

    def test_a_reduced_position_is_reported_as_partially_closed(self, tmp_path):
        routes = _routes(positions=_Resp(200, [_position(qty="1")]))
        worker = _worker(tmp_path, decisions=[_decision()], routes=routes)
        worker.start()
        worker.run_once()
        worker._verify_all_protections()

        assert (
            worker.state.protections[SYMBOL]["status"]
            == PROTECTION_STATUS_PARTIALLY_CLOSED
        )

    def test_the_levels_survive_a_restart(self, tmp_path):
        worker = _worker(tmp_path, decisions=[_decision()])
        worker.start()
        worker.run_once()
        worker.stop()

        reloaded = _worker(tmp_path, decisions=[], name="worker_state.json")
        assert reloaded.state.protections[SYMBOL]["stop_loss"] == 9.0
        assert reloaded.state.protections[SYMBOL]["take_profit"] == 12.0
        assert reloaded.state.stops[SYMBOL]["stop_loss"] == 9.0

    def test_the_worker_status_view_carries_the_protection(self, tmp_path):
        from tradingagents.paper_experiment.broker.trading import worker_status_view

        worker = _worker(tmp_path, decisions=[_decision()])
        worker.start()
        worker.run_once()
        worker._verify_all_protections()

        view = worker_status_view(state=worker.state, config=worker.config)
        assert view["protection_status"] == PROTECTION_STATUS_PROTECTED
        assert view["protections"][SYMBOL]["stop_loss"] == 9.0
        assert view["position_protection"][SYMBOL]["stop_loss"] == 9.0


# ---------------------------------------------------------------------------
# 4. The risk limits are untouched
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheRiskLimitsAreUnchanged:
    def test_the_limits_are_still_the_documented_ones(self):
        limits = PaperRiskLimits()
        assert limits.max_order_notional == 25.0
        assert limits.max_open_positions == 5
        assert limits.max_daily_orders == 10
        assert limits.allow_short is False

    def test_a_protectable_buy_still_respects_the_notional_cap(self, tmp_path):
        calls: list[dict] = []
        worker = _worker(
            tmp_path,
            decisions=[_decision(notional=25.0)],
            calls=calls,
            prices={SYMBOL: 10.0},
        )
        worker.start()
        worker.run_once()

        posted = [c for c in calls if c["method"] == "POST"]
        assert posted, "the order should have been placed"
        # 25 USD at 10.0 is 2.5 units; the broker rounds to a tradable quantity.
        assert posted[0]["json"]["qty"] is not None

    def test_an_oversized_notional_is_still_refused(self, tmp_path):
        calls: list[dict] = []
        worker = _worker(
            tmp_path,
            decisions=[_decision(notional=500.0)],
            calls=calls,
        )
        worker.start()
        result = worker.run_once()

        assert result.ran is False
        assert not any(c["method"] == "POST" for c in calls)

    def test_the_gate_does_not_relax_the_open_position_cap(self, tmp_path):
        limits = PaperRiskLimits(max_open_positions=1)
        calls: list[dict] = []
        worker = _worker(
            tmp_path,
            decisions=[_decision("d1"), _decision("d2", symbol="QQQ")],
            calls=calls,
            limits=limits,
        )
        worker.start()
        worker.run_once()
        worker.run_once()

        posted = [c for c in calls if c["method"] == "POST"]
        assert len(posted) <= 1

    def test_the_gate_does_not_relax_the_daily_order_cap(self, tmp_path):
        limits = PaperRiskLimits(max_daily_orders=1)
        calls: list[dict] = []
        worker = _worker(
            tmp_path,
            decisions=[_decision("d1"), _decision("d2", symbol="QQQ")],
            calls=calls,
            limits=limits,
        )
        worker.start()
        worker.run_once()
        worker.run_once()

        posted = [c for c in calls if c["method"] == "POST"]
        assert len(posted) <= 1

    def test_the_gate_does_not_enable_shorting(self, tmp_path):
        calls: list[dict] = []
        worker = _worker(
            tmp_path,
            decisions=[_decision(side="sell", stop_loss=None, take_profit=None)],
            calls=calls,
        )
        worker.start()
        worker.run_once()

        # A SELL is only ever a **close** on this long-only book. The broker
        # reports 2 units held, so the sell closes those 2 units — it is never
        # turned into a short, and it never exceeds what is held.
        posted = [c for c in calls if c["method"] == "POST"]
        assert posted
        assert float(posted[0]["json"]["qty"]) <= 2.0

    def test_a_sell_with_nothing_held_is_refused_not_a_short(self, tmp_path):
        calls: list[dict] = []
        routes = _routes(positions=_Resp(200, []))
        worker = _worker(
            tmp_path,
            decisions=[_decision(side="sell", stop_loss=None, take_profit=None)],
            calls=calls,
            routes=routes,
        )
        worker.start()
        result = worker.run_once()

        assert result.ran is False
        assert not any(c["method"] == "POST" for c in calls)

    def test_a_risk_budget_can_only_make_an_order_smaller(self, tmp_path):
        calls: list[dict] = []
        worker = _worker(
            tmp_path,
            decisions=[_decision(stop_loss=8.0, take_profit=14.0)],
            calls=calls,
            max_risk_per_trade=1.0,
        )
        worker.start()
        worker.run_once()

        posted = [c for c in calls if c["method"] == "POST"]
        assert posted, "the order should still have been placed"
        # 1.0 of risk against a 2.0 stop is 0.5 units, which is less than the
        # 2.5 the notional alone would buy.
        assert float(posted[0]["json"]["qty"]) <= 0.5


# ---------------------------------------------------------------------------
# 5. A multi-target ladder is managed in stages
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheTargetLadder:
    def _ladder_worker(self, tmp_path, *, calls=None, targets=(12.0, 14.0, 16.0)):
        """A worker whose entry filled at 10.0 and whose price can then move.

        The entry is sized at two units and the venue reports the same two
        units held, so the ladder's thirds are exact.
        """
        seam = _MutablePrice(10.0)
        worker = _worker(
            tmp_path,
            decisions=[
                _decision(
                    stop_loss=9.0,
                    take_profit=targets[0] if targets else None,
                    targets=targets,
                    quantity=2.0,
                )
            ],
            calls=calls,
            market_data=seam,
            held=2.0,
        )
        worker.start()
        worker.run_once()
        return worker, seam

    def test_a_three_target_ladder_is_recorded_whole(self, tmp_path):
        worker, _ = self._ladder_worker(tmp_path)

        progress = worker.state.target_progress[SYMBOL]
        assert progress["targets"] == [12.0, 14.0, 16.0]
        assert progress["taken"] == []

    def test_the_first_target_closes_a_third_and_leaves_the_position_open(
        self, tmp_path
    ):
        calls: list[dict] = []
        worker, seam = self._ladder_worker(tmp_path, calls=calls)
        # The price has reached the first target.
        seam.price_value = 12.0

        worker._check_exits()

        sells = _sells(calls)
        assert len(sells) == 1
        # Two units held; three targets, so the first closes a third.
        assert float(sells[0]["json"]["qty"]) == pytest.approx(2.0 / 3.0, abs=1e-6)
        # The position is still open: it was a stage, not the end.
        assert SYMBOL not in worker.state.exits

    def test_a_target_that_was_taken_is_not_sold_into_again(self, tmp_path):
        worker, seam = self._ladder_worker(tmp_path)
        seam.price_value = 12.0
        worker._check_exits()

        progress = worker.state.target_progress[SYMBOL]
        assert progress["taken"] == [12.0]

    def test_the_next_target_is_the_one_that_triggers_next(self, tmp_path):
        calls: list[dict] = []
        worker, seam = self._ladder_worker(tmp_path, calls=calls)
        seam.price_value = 12.0
        worker._check_exits()
        assert float(_sells(calls)[-1]["json"]["qty"]) == pytest.approx(
            2.0 / 3.0, abs=1e-6
        )

        # The price climbs to the second target. The first is spent, so this
        # closes the second slice — and only the second slice.
        seam.price_value = 14.0
        worker._check_exits()
        sells = _sells(calls)
        assert len(sells) == 2
        assert float(sells[-1]["json"]["qty"]) == pytest.approx(2.0 / 3.0, abs=1e-6)
        assert worker.state.target_progress[SYMBOL]["taken"] == [12.0, 14.0]

    def test_a_stop_closes_the_whole_position_not_a_slice(self, tmp_path):
        calls: list[dict] = []
        worker, seam = self._ladder_worker(tmp_path, calls=calls)
        # The price has fallen through the stop.
        seam.price_value = 8.5

        worker._check_exits()

        sells = _sells(calls)
        assert len(sells) == 1
        # A stop is the end of the trade, not a stage of it: it closes
        # everything that is held.
        assert float(sells[0]["json"]["qty"]) == pytest.approx(2.0)
        assert SYMBOL in worker.state.exits

    def test_a_single_target_closes_the_whole_position(self, tmp_path):
        calls: list[dict] = []
        worker, seam = self._ladder_worker(tmp_path, calls=calls, targets=(12.0,))
        seam.price_value = 12.0

        worker._check_exits()

        sells = _sells(calls)
        assert len(sells) == 1
        assert float(sells[0]["json"]["qty"]) == pytest.approx(2.0)
        assert SYMBOL in worker.state.exits

    def test_a_price_below_the_first_target_does_not_trigger(self, tmp_path):
        calls: list[dict] = []
        worker, seam = self._ladder_worker(tmp_path, calls=calls)
        # Below the first target and above the stop: nothing has been reached.
        seam.price_value = 10.5

        assert worker._check_exits() is None
        assert _sells(calls) == []

    def test_a_price_past_the_first_target_triggers_the_first_target(self, tmp_path):
        # A price *between* two rungs has still crossed the first one, so the
        # first target is what triggers — never the second.
        worker, seam = self._ladder_worker(tmp_path)
        seam.price_value = 13.0
        worker._check_exits()

        assert worker.state.target_progress[SYMBOL]["taken"] == [12.0]


# ---------------------------------------------------------------------------
# 6. The strategy is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheStrategyIsUnchanged:
    def test_the_gate_never_adjusts_a_level(self, tmp_path):
        """The levels the strategy derived reach the record verbatim.

        A gate that "helpfully" moved a stop to a rounder number would be
        trading the strategy's idea for its own. The stop and the target are
        recorded exactly as the decision carried them.
        """
        worker = _worker(
            tmp_path,
            decisions=[_decision(stop_loss=9.37, take_profit=12.61)],
        )
        worker.start()
        worker.run_once()

        record = worker.state.protections[SYMBOL]
        assert record["stop_loss"] == 9.37
        assert record["take_profit"] == 12.61

    def test_the_gate_refuses_rather_than_inventing_a_level(self, tmp_path):
        """A decision with no levels is refused — never given a default stop."""
        worker = _worker(
            tmp_path,
            decisions=[_decision(stop_loss=None, take_profit=None)],
        )
        worker.start()
        result = worker.run_once()

        assert result.ran is False
        assert worker.state.protections == {}
        assert worker.state.stops == {}
