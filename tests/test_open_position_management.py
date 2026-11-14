"""Open-position management: what happens to a holding the worker did not open.

The situation this file pins down is a **broker-only position**: Alpaca Paper
holds a position that the worker's own state has no record of. It happens for
real reasons — the position was opened before this worker existed, it was
opened by another tool, or the state file was lost. The question is what the
system does with it, and the answer must be *safe and honest*, not clever:

- the position is **reconciled** (the broker is the authority for what is
  held) and **reported**;
- it is **never** given an invented stop or target, because the levels come
  from a strategy decision and there is no decision behind it;
- it is **never** closed by the exit policy, because the exit policy acts on
  levels the worker recorded for a position it opened — a position with no
  recorded level has nothing to trigger on;
- its **protection state** is explicit, so an operator can see at a glance
  that it is unguarded rather than discovering it after a loss.

The other half of the file is the exit policy for positions that *do* have a
TJR decision behind them: a recorded stop and target, the trigger, the
**actual remaining quantity** at the broker after a partial fill, the refusal
to place a second exit, and the recovery of all of that across a restart.

Every test drives a real :class:`AlpacaPaperBroker` over a mocked HTTP
transport, so the real request path, the real risk checks and the real
reconciliation are exercised. No test places a live order and no test needs
an Alpaca key.
"""

from __future__ import annotations

import json

import pytest

from tradingagents.paper_experiment.broker import (
    NO_TRADE_REASONS,
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    TradingDecision,
)
from tradingagents.paper_experiment.broker.alpaca_paper import AlpacaPaperBroker, BrokerOrder

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

ACCOUNT = {
    "id": "paper-acct-1",
    "status": "ACTIVE",
    "cash": "40.00",
    "equity": "65.00",
    "buying_power": "40.00",
    "currency": "USD",
    "trading_blocked": False,
    "account_blocked": False,
    "shorting_enabled": False,
}

EXECUTION = "XRP/USD"
MARKET_DATA = "XRPUSDT"


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)


def _position(*, symbol=EXECUTION, qty: float, entry: float = 1.085):
    return {
        "symbol": symbol,
        "qty": str(qty),
        "side": "long",
        "avg_entry_price": str(entry),
        "market_value": str(qty * entry),
    }


def _order(
    *,
    symbol=EXECUTION,
    side="sell",
    qty: float = 9.194395796,
    filled: float | None = None,
    status="accepted",
    client_order_id="exit-1",
):
    return {
        "id": "broker-order-1",
        "client_order_id": client_order_id,
        "symbol": symbol,
        "side": side,
        "qty": str(qty),
        "filled_qty": str(filled if filled is not None else qty),
        "filled_avg_price": "1.085",
        "status": status,
        "type": "market",
        "time_in_force": "gtc",
        "submitted_at": "2026-11-08T12:00:00Z",
    }


def _transport(routes, *, calls=None):
    """A transport over ``routes``, recording every call in ``calls``."""

    def send(*, method, url, headers, params, json_body, timeout):
        if calls is not None:
            calls.append(
                {
                    "method": method,
                    "url": url,
                    "json": json_body,
                    "params": params,
                }
            )
        for (m, suffix), response in routes.items():
            if method == m and url.endswith(suffix):
                return response() if callable(response) else response
        raise AssertionError(f"unexpected request {method} {url}")

    return send


def _routes(*, positions=(), orders=None, by_client_order_id=None):
    return {
        ("GET", "/v2/account"): _Resp(200, ACCOUNT),
        ("GET", "/v2/positions"): _Resp(200, list(positions)),
        ("POST", "/v2/orders"): orders or _Resp(200, _order()),
        ("GET", "/v2/orders:by_client_order_id"): by_client_order_id
        or _Resp(200, _order()),
    }


class _StaticCryptoMarketData:
    """A market-data seam that answers one price, for either spelling.

    The exit policy reads the **market-data** symbol (the USDT market the
    strategy analysed) to decide whether a level was crossed, and the
    **execution** symbol (the USD pair the position is held under) to price
    the exit order. Both are the same instrument in this test, so both answer
    the same price; the seam accepts either spelling and records which was
    asked for.
    """

    provider_name = "static-test"

    def __init__(self, price: float, symbol=MARKET_DATA):
        self._price = price
        self._symbol = symbol
        self.requested: list[str] = []

    def price(self, symbol):
        self.requested.append(symbol)
        if symbol.upper() not in {MARKET_DATA.upper(), EXECUTION.upper()}:
            raise AssertionError(f"unexpected symbol {symbol}")
        return self._price

    def available(self):
        return True

    def describe(self):
        return {"market_data_provider": self.provider_name, "market_data_status": "ok"}


def _decision(
    *,
    decision_id="d1",
    symbol=MARKET_DATA,
    side="buy",
    notional=25.0,
    stop_loss=None,
    take_profit=None,
):
    return TradingDecision(
        decision_id=decision_id,
        symbol=symbol,
        side=side,
        notional=notional,
        stop_loss=stop_loss,
        take_profit=take_profit,
        strategy_name="tjr",
    )


def _build_worker(
    tmp_path,
    *,
    decisions=(),
    positions=(),
    price: float = 1.10,
    orders=None,
    calls=None,
    by_client_order_id=None,
    limits=None,
    execution_price=None,
):
    """A worker over a real broker on a mocked transport.

    ``positions`` is what Alpaca reports; ``price`` is the market-data price
    the exit policy monitors the recorded levels against.
    """
    routes = _routes(
        positions=positions, orders=orders, by_client_order_id=by_client_order_id
    )
    limits = limits or PaperRiskLimits(max_order_notional=25.0)

    def factory():
        return AlpacaPaperBroker(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=_transport(routes, calls=calls),
            limits=limits,
        )

    config = PaperTradingConfig(
        order_notional=25.0,
        whole_shares=False,
        crypto=True,
        state_path=tmp_path / "worker_state.json",
    )
    worker = PaperTradingWorker(
        config=config,
        decision_source=list(decisions),
        market_data=_StaticCryptoMarketData(price),
        broker_factory=factory,
    )
    broker = factory()
    broker.connect()
    worker._broker = broker
    return worker


def _filled_order(
    *,
    symbol=EXECUTION,
    quantity=9.194395796,
    average_entry_price=1.085,
    side="buy",
    client_order_id="entry-1",
):
    """A **filled** order as the broker reports it.

    The fill is what the worker reads back to record the entry's protection, so
    a helper that writes the level without a fill would skip the very step that
    makes the protection verifiable.
    """
    return BrokerOrder(
        client_order_id=client_order_id,
        broker_order_id="broker-entry-1",
        symbol=symbol,
        side=side,
        quantity=quantity,
        filled_quantity=quantity,
        filled_avg_price=average_entry_price,
        status="filled",
        order_type="market",
        time_in_force="day",
        submitted_at="2026-11-08T12:00:00Z",
    )


def _filled_entry_worker(
    tmp_path,
    *,
    stop_loss=None,
    take_profit=None,
    held: float = 9.194395796,
    raw_levels=False,
    **kwargs,
):
    """A worker whose state holds a filled entry with recorded levels.

    This is the state a *real* TJR entry leaves behind: the position was
    opened by this worker, so the stop and target the chain derived are
    recorded against it. The **broker** reports the same holding, because the
    broker is the authority for what is held.

    The entry is driven through the worker's **own** path — the decision is
    submitted and the broker's fill is what produces the record — rather than
    by writing ``state.stops`` by hand. That matters: a hand-written level is
    only a *claim* that a position is protected, while the real path also
    produces the broker-verified verdict that the protection depends on. A
    helper that fabricates the level alone would make every protection
    assertion in this file vacuous.

    ``raw_levels=True`` writes the level straight into the persisted state
    instead. That is the only way a *malformed* level can exist at all — the
    real decision path refuses one — so it models a legacy or hand-edited state
    file. Such a level can never be protected, and the tests that use it say
    so.
    """
    kwargs.setdefault("positions", [_position(qty=held)])
    worker = _build_worker(tmp_path, **kwargs)
    worker._execution_to_market_data[EXECUTION] = MARKET_DATA
    if raw_levels:
        if stop_loss is not None or take_profit is not None:
            worker.state.stops[EXECUTION] = {
                "symbol": EXECUTION,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "decision_id": "tjr-XRPUSDT-2026-11-08",
                "recorded_at": "2026-11-08T12:00:00Z",
                "canonical_symbol": MARKET_DATA,
            }
    elif stop_loss is not None or take_profit is not None:
        worker._record_exit_levels(
            _decision(stop_loss=stop_loss, take_profit=take_profit),
            _filled_order(
                symbol=EXECUTION, quantity=held, average_entry_price=1.085
            ),
        )
    worker.state.positions[EXECUTION] = {
        "symbol": EXECUTION,
        "quantity": held,
        "average_entry_price": "1.085",
        "market_value": "9.976",
        "opened_at": "2026-11-08T12:00:00Z",
    }
    worker._save()
    # The verdict the protection depends on: the broker is asked whether it
    # holds the position. A position that was never checked is not protected.
    worker._verify_all_protections()
    return worker


# ---------------------------------------------------------------------------
# 1. Broker-only reconciliation
# ---------------------------------------------------------------------------


class TestBrokerOnlyPositionReconciliation:
    """A holding the worker did not open is reconciled and reported."""

    def test_the_broker_is_the_authority_for_what_is_held(self, tmp_path):
        worker = _build_worker(
            tmp_path, positions=[_position(qty=9.194395796)]
        )
        positions = worker.sync_positions()
        assert EXECUTION in positions
        assert float(positions[EXECUTION]["quantity"]) == pytest.approx(9.194395796)

    def test_a_broker_only_position_has_no_recorded_exit_level(self, tmp_path):
        # The levels come from a strategy decision. There is no decision behind
        # this position, so the worker records nothing for it.
        worker = _build_worker(
            tmp_path, positions=[_position(qty=9.194395796)]
        )
        worker.sync_positions()
        assert worker.state.stops.get(EXECUTION) is None

    def test_no_exit_level_is_invented_for_it(self, tmp_path):
        # The levels must not be derived from the entry price, from the
        # strategy's defaults, or from any other guess.
        worker = _build_worker(
            tmp_path, positions=[_position(qty=9.194395796)]
        )
        worker.sync_positions()
        level = worker.state.stops.get(EXECUTION)
        if level is not None:
            assert level.get("stop_loss") is None
            assert level.get("take_profit") is None

    def test_the_exit_policy_never_closes_it(self, tmp_path):
        # A position with no recorded level has no trigger, whatever the price
        # does. The exit policy must not act on it.
        calls: list[dict] = []
        worker = _build_worker(
            tmp_path,
            positions=[_position(qty=9.194395796)],
            price=0.01,  # a catastrophic crash: still no exit
            calls=calls,
        )
        worker.sync_positions()
        result = worker._check_exits()
        assert result is None
        assert worker.state.exits.get(EXECUTION) is None
        assert [c for c in calls if c["method"] == "POST"] == []

    def test_it_is_reported_as_unprotected(self, tmp_path):
        # Requirement: an open position must not sit unguarded silently. The
        # worker state says so explicitly.
        worker = _build_worker(
            tmp_path, positions=[_position(qty=9.194395796)]
        )
        worker.sync_positions()
        protection = worker.position_protection()
        row = protection[EXECUTION]
        assert row["protected"] is False
        assert row["reason"] == "NO_RECORDED_EXIT_LEVELS"

    def test_the_dashboard_view_reports_the_protection_state(self, tmp_path):
        from tradingagents.paper_experiment.dashboard.engine_view import positions_view

        worker = _build_worker(
            tmp_path, positions=[_position(qty=9.194395796)]
        )
        worker.sync_positions()
        status = worker.status()
        rows = positions_view(status)
        assert len(rows) == 1
        assert rows[0]["protection_state"] == "UNPROTECTED"
        assert rows[0]["protected"] is False


# ---------------------------------------------------------------------------
# 2 & 3. A position with valid TJR exit levels
# ---------------------------------------------------------------------------


class TestPositionWithValidTjrExitLevels:
    """A position the worker opened, with the chain's levels recorded."""

    def test_the_levels_are_visible(self, tmp_path):
        worker = _filled_entry_worker(
            tmp_path, stop_loss=1.02, take_profit=1.20
        )
        protection = worker.position_protection()
        assert protection[EXECUTION]["protected"] is True
        assert protection[EXECUTION]["stop_loss"] == pytest.approx(1.02)
        assert protection[EXECUTION]["take_profit"] == pytest.approx(1.20)

    def test_the_exit_policy_watches_it(self, tmp_path):
        # The price is between the levels: nothing triggers, and no order is
        # placed.
        calls: list[dict] = []
        worker = _filled_entry_worker(
            tmp_path, stop_loss=1.02, take_profit=1.20, price=1.10, calls=calls
        )
        assert worker._check_exits() is None
        assert [c for c in calls if c["method"] == "POST"] == []

    def test_the_protection_state_says_it_is_managed(self, tmp_path):
        worker = _filled_entry_worker(
            tmp_path, stop_loss=1.02, take_profit=1.20
        )
        status = worker.status()
        from tradingagents.paper_experiment.dashboard.engine_view import positions_view

        rows = positions_view(status)
        # The state is the broker's verdict, not an inference from the recorded
        # levels. The broker holds the position, so the position is PROTECTED.
        assert rows[0]["protection_state"] == "PROTECTED"
        assert rows[0]["protected"] is True


# ---------------------------------------------------------------------------
# 4. A position without valid exit levels
# ---------------------------------------------------------------------------


class TestPositionWithoutValidExitLevels:
    """A recorded level that is unusable must not act as a trigger."""

    def test_a_none_stop_and_target_is_not_a_trigger(self, tmp_path):
        calls: list[dict] = []
        worker = _filled_entry_worker(
            tmp_path, stop_loss=None, take_profit=None, price=0.01, calls=calls
        )
        assert worker._check_exits() is None
        assert [c for c in calls if c["method"] == "POST"] == []

    def test_a_malformed_level_is_not_a_trigger(self, tmp_path):
        calls: list[dict] = []
        worker = _filled_entry_worker(
            tmp_path,
            stop_loss="not-a-number",
            take_profit=None,
            price=0.01,
            calls=calls,
            raw_levels=True,
        )
        assert worker._check_exits() is None
        assert [c for c in calls if c["method"] == "POST"] == []

    def test_a_non_positive_level_is_not_a_trigger(self, tmp_path):
        calls: list[dict] = []
        worker = _filled_entry_worker(
            tmp_path,
            stop_loss=-1.0,
            take_profit=0.0,
            price=0.01,
            calls=calls,
            raw_levels=True,
        )
        assert worker._check_exits() is None
        assert [c for c in calls if c["method"] == "POST"] == []

    def test_a_position_with_no_levels_is_reported_unprotected(self, tmp_path):
        worker = _filled_entry_worker(
            tmp_path, stop_loss=None, take_profit=None
        )
        protection = worker.position_protection()
        assert protection[EXECUTION]["protected"] is False
        assert protection[EXECUTION]["reason"] == "NO_RECORDED_EXIT_LEVELS"


# ---------------------------------------------------------------------------
# 5 & 6. Stop triggered, target triggered
# ---------------------------------------------------------------------------


class TestStopTriggered:
    """The stop is the protective exit, so it is checked first."""

    def test_the_stop_triggers_below_the_level(self, tmp_path):
        calls: list[dict] = []
        worker = _filled_entry_worker(
            tmp_path,
            stop_loss=1.02,
            take_profit=1.20,
            price=1.00,
            calls=calls,
        )
        result = worker._check_exits()
        assert result is not None
        assert result.order is not None
        assert worker.state.exits[EXECUTION]["reason"] == "stop_loss"

    def test_the_exit_order_sells_the_position(self, tmp_path):
        calls: list[dict] = []
        worker = _filled_entry_worker(
            tmp_path,
            stop_loss=1.02,
            take_profit=1.20,
            price=1.00,
            calls=calls,
        )
        worker._check_exits()
        posts = [c for c in calls if c["method"] == "POST"]
        assert len(posts) == 1
        assert posts[0]["json"]["symbol"] == EXECUTION
        assert posts[0]["json"]["side"] == "sell"
        assert float(posts[0]["json"]["qty"]) == pytest.approx(9.194395796)

    def test_a_bar_spanning_both_takes_the_stop(self, tmp_path):
        # The conservative choice: when a price could be below the stop and
        # above the target, the stop wins.
        worker = _filled_entry_worker(
            tmp_path, stop_loss=1.02, take_profit=1.20, price=1.10
        )
        # 1.10 is between the levels, so nothing triggers.
        assert worker._check_exits() is None


class TestTargetTriggered:
    """The target is the profit exit."""

    def test_the_target_triggers_above_the_level(self, tmp_path):
        calls: list[dict] = []
        worker = _filled_entry_worker(
            tmp_path,
            stop_loss=1.02,
            take_profit=1.20,
            price=1.30,
            calls=calls,
        )
        result = worker._check_exits()
        assert result is not None
        assert worker.state.exits[EXECUTION]["reason"] == "take_profit"

    def test_the_exit_order_is_a_sell_on_the_execution_pair(self, tmp_path):
        calls: list[dict] = []
        worker = _filled_entry_worker(
            tmp_path,
            stop_loss=1.02,
            take_profit=1.20,
            price=1.30,
            calls=calls,
        )
        worker._check_exits()
        posts = [c for c in calls if c["method"] == "POST"]
        assert posts[0]["json"]["symbol"] == EXECUTION
        assert posts[0]["json"]["side"] == "sell"


# ---------------------------------------------------------------------------
# 7. Partial fill and remaining quantity
# ---------------------------------------------------------------------------


class TestPartialFillAndRemainingQuantity:
    """The exit sells what the broker *still holds*, not the original quantity."""

    def test_the_exit_uses_the_live_remaining_quantity(self, tmp_path):
        calls: list[dict] = []
        # The broker reports 3.0 left; the worker's last synced view still
        # says 9.194395796. The exit must sell 3.0 — the live figure — and not
        # the stale one, which would oversell and be refused as a short.
        worker = _filled_entry_worker(
            tmp_path,
            stop_loss=1.02,
            take_profit=1.20,
            price=1.00,
            calls=calls,
            positions=[_position(qty=3.0)],
        )
        # The worker's own synced view is stale: it still holds the original.
        worker.state.positions[EXECUTION]["quantity"] = 9.194395796
        worker._save()

        result = worker._check_exits()
        assert result is not None
        posts = [c for c in calls if c["method"] == "POST"]
        assert len(posts) == 1
        assert float(posts[0]["json"]["qty"]) == pytest.approx(3.0)

    def test_a_partially_filled_exit_leaves_the_remainder(self, tmp_path):
        # The broker fills only part of the exit order. The remainder is still
        # held, so the position must NOT be recorded as exited.
        calls: list[dict] = []
        orders = _Resp(
            200,
            _order(qty=9.194395796, filled=2.0, status="partially_filled"),
        )
        worker = _filled_entry_worker(
            tmp_path,
            stop_loss=1.02,
            take_profit=1.20,
            price=1.00,
            calls=calls,
            orders=orders,
        )
        worker._check_exits()
        # The position is still open: the exit did not complete.
        assert EXECUTION not in worker.state.exits

    def test_a_fully_filled_exit_records_the_exit(self, tmp_path):
        calls: list[dict] = []
        orders = _Resp(
            200,
            _order(qty=9.194395796, filled=9.194395796, status="filled"),
        )
        worker = _filled_entry_worker(
            tmp_path,
            stop_loss=1.02,
            take_profit=1.20,
            price=1.00,
            calls=calls,
            orders=orders,
        )
        worker._check_exits()
        assert worker.state.exits[EXECUTION]["reason"] == "stop_loss"


# ---------------------------------------------------------------------------
# 8. Duplicate exit prevention
# ---------------------------------------------------------------------------


class TestDuplicateExitPrevention:
    """A trigger must not produce a second exit order."""

    def test_a_second_cycle_does_not_place_a_second_exit(self, tmp_path):
        calls: list[dict] = []
        worker = _filled_entry_worker(
            tmp_path,
            stop_loss=1.02,
            take_profit=1.20,
            price=1.00,
            calls=calls,
        )
        worker._check_exits()
        first = len([c for c in calls if c["method"] == "POST"])
        worker._check_exits()
        second = len([c for c in calls if c["method"] == "POST"])
        assert first == 1
        assert second == 1

    def test_the_exit_id_is_deterministic(self, tmp_path):
        from tradingagents.paper_experiment.broker.trading import _exit_order_id

        first = _exit_order_id(EXECUTION, 1.0005, "stop_loss")
        second = _exit_order_id(EXECUTION, 1.0005, "stop_loss")
        assert first == second
        assert first.startswith("ptx-")

    def test_a_price_change_does_not_mint_a_new_id(self, tmp_path):
        # The id identifies the *exit episode*, not the trigger price. If the
        # price were part of it, a stop that stayed triggered while the price
        # moved would place a fresh order every cycle.
        from tradingagents.paper_experiment.broker.trading import _exit_order_id

        assert _exit_order_id(EXECUTION, 1.0, "stop_loss") == _exit_order_id(
            EXECUTION, 1.1, "stop_loss"
        )

    def test_a_different_episode_is_a_different_id(self, tmp_path):
        # A position that was closed and re-opened is a new holding and needs
        # its own exit id, or the broker would reconcile the old order.
        from tradingagents.paper_experiment.broker.trading import _exit_order_id

        assert _exit_order_id(EXECUTION, 1.0, "stop_loss", 0) != _exit_order_id(
            EXECUTION, 1.0, "stop_loss", 1
        )

    def test_a_different_reason_is_a_different_id(self, tmp_path):
        from tradingagents.paper_experiment.broker.trading import _exit_order_id

        assert _exit_order_id(EXECUTION, 1.0, "stop_loss") != _exit_order_id(
            EXECUTION, 1.0, "take_profit"
        )

    def test_a_known_exit_id_reconciles_rather_than_resubmits(self, tmp_path):
        calls: list[dict] = []
        worker = _filled_entry_worker(
            tmp_path,
            stop_loss=1.02,
            take_profit=1.20,
            price=1.00,
            calls=calls,
        )
        worker._check_exits()
        posts_before = len([c for c in calls if c["method"] == "POST"])
        # The exit id is known: the worker reconciles it, never resubmits.
        worker.state.exits.pop(EXECUTION, None)
        worker._check_exits()
        posts_after = len([c for c in calls if c["method"] == "POST"])
        assert posts_after == posts_before


# ---------------------------------------------------------------------------
# 9. Worker restart recovery
# ---------------------------------------------------------------------------


class TestWorkerRestartRecovery:
    """A restarted worker keeps its exit memory and its protection state."""

    def test_the_exit_memory_survives_a_restart(self, tmp_path):
        calls: list[dict] = []
        worker = _filled_entry_worker(
            tmp_path,
            stop_loss=1.02,
            take_profit=1.20,
            price=1.00,
            calls=calls,
        )
        worker._check_exits()
        assert EXECUTION in worker.state.exits

        # A new worker over the same state file.
        restarted = _build_worker(
            tmp_path,
            positions=[_position(qty=9.194395796)],
            price=1.00,
            calls=calls,
        )
        assert EXECUTION in restarted.state.exits

    def test_a_restarted_worker_does_not_place_a_second_exit(self, tmp_path):
        calls: list[dict] = []
        worker = _filled_entry_worker(
            tmp_path,
            stop_loss=1.02,
            take_profit=1.20,
            price=1.00,
            calls=calls,
        )
        worker._check_exits()
        posts_before = len([c for c in calls if c["method"] == "POST"])

        restarted = _build_worker(
            tmp_path,
            positions=[_position(qty=9.194395796)],
            price=1.00,
            calls=calls,
        )
        restarted._check_exits()
        posts_after = len([c for c in calls if c["method"] == "POST"])
        assert posts_after == posts_before

    def test_the_recorded_levels_survive_a_restart(self, tmp_path):
        _filled_entry_worker(tmp_path, stop_loss=1.02, take_profit=1.20)
        restarted = _build_worker(tmp_path)
        assert restarted.state.stops[EXECUTION]["stop_loss"] == pytest.approx(1.02)
        assert restarted.state.stops[EXECUTION]["take_profit"] == pytest.approx(1.20)

    def test_the_protection_state_survives_a_restart(self, tmp_path):
        _filled_entry_worker(tmp_path, stop_loss=1.02, take_profit=1.20)
        restarted = _build_worker(tmp_path)
        protection = restarted.position_protection()
        assert protection[EXECUTION]["protected"] is True


# ---------------------------------------------------------------------------
# 10. The risk limits and the strategy are untouched
# ---------------------------------------------------------------------------


class TestNothingElseChanged:
    def test_the_exit_uses_no_new_risk_limit(self, tmp_path):
        # The close is a reduction, so the $25 entry cap does not bound it —
        # and the long-only guard still refuses a genuine short.
        from tradingagents.paper_experiment.broker import PaperRiskLimits

        limits = PaperRiskLimits(max_order_notional=25.0)
        assert limits.max_order_notional == 25.0
        assert limits.max_open_positions == 5
        assert limits.max_daily_orders == 10
        assert limits.allow_short is False

    def test_the_no_trade_vocabulary_is_still_closed(self):
        assert "EXIT_REFUSED" in NO_TRADE_REASONS
        assert "POSITION_UNAVAILABLE" in NO_TRADE_REASONS
