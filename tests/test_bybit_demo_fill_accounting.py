"""The Bybit Demo broker accounts for a fill exactly once.

These tests pin the accounting that a **growing** fill depends on, and the
safety rules that follow from it. A broker that counts a fill twice inflates
what it believes it holds; a broker that counts it once and then ignores the
extra quantity understates it. Both are wrong in the direction that matters:
the second one leaves the real quantity unguarded.

The rule under test is the same throughout: **the quantity, not the id**, is
what makes the accounting exactly-once, and the venue's own records are the
authority on what actually traded.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tradingagents.paper_experiment.broker.bybit_demo import (
    DEMO_BASE_URL,
    BrokerHttpResponse,
    BrokerOrder,
    BybitDemoBroker,
    PaperRiskRejected,
)
from tradingagents.paper_experiment.broker.state import BrokerStateStore

# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class RecordingTransport:
    """A mock transport that records every request and replays canned answers."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses: dict[tuple[str, str], list[dict]] = {}
        self._failures: dict[tuple[str, str], list[Exception]] = {}
        self._default: dict = {"retCode": 0, "retMsg": "OK", "result": {}}

    def add(self, method: str, path: str, response: dict) -> None:
        self._responses.setdefault((method.upper(), path), []).append(response)

    def fail(self, method: str, path: str, error: Exception) -> None:
        """Make the next call to a path raise, to model an unreadable venue."""
        self._failures.setdefault((method.upper(), path), []).append(error)

    def calls_to(self, method: str, path: str) -> list[dict[str, Any]]:
        return [
            call
            for call in self.calls
            if call["method"].upper() == method.upper() and call["path"] == path
        ]

    def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: dict,
        params: dict,
        json_body: dict,
        timeout: float,
    ) -> BrokerHttpResponse:
        path = url.split("?")[0].replace(DEMO_BASE_URL, "")
        self.calls.append(
            {
                "method": method,
                "url": url,
                "path": path,
                "headers": dict(headers),
                "params": dict(params),
                "json_body": dict(json_body or {}),
            }
        )
        failures = self._failures.get((method.upper(), path))
        if failures:
            raise failures.pop(0)
        queue = self._responses.get((method.upper(), path))
        payload = self._default
        if queue:
            payload = queue.pop(0) if len(queue) > 1 else queue[0]
        return BrokerHttpResponse(200, json.dumps(payload))


# ---------------------------------------------------------------------------
# Canned venue answers
# ---------------------------------------------------------------------------


def wallet_response(coins: dict[str, dict[str, str]]) -> dict:
    return {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "list": [
                {
                    "accountId": "demo-account",
                    "accountType": "SPOT",
                    "coin": [
                        {
                            "coin": coin,
                            "walletBalance": values["walletBalance"],
                            "availableToWithdraw": values.get(
                                "availableToWithdraw", values["walletBalance"]
                            ),
                            "locked": values.get("locked", "0"),
                        }
                        for coin, values in coins.items()
                    ],
                }
            ]
        },
    }


def instruments_response(entries: list[dict]) -> dict:
    return {"retCode": 0, "retMsg": "OK", "result": {"category": "spot", "list": entries}}


def spot_instrument(symbol: str = "BTCUSDT", **overrides: Any) -> dict:
    entry = {
        "symbol": symbol,
        "status": "Trading",
        "baseCoin": symbol[:-4],
        "quoteCoin": "USDT",
        "lotSizeFilter": {
            "minOrderQty": "0.00001",
            "maxOrderQty": "1000000",
            "qtyStep": "0.00001",
            "minOrderAmt": "5",
        },
        "priceFilter": {"tickSize": "0.1"},
    }
    entry.update(overrides)
    return entry


def order_row(
    *,
    order_link_id: str,
    status: str = "New",
    symbol: str = "BTCUSDT",
    side: str = "Buy",
    qty: str = "0.0005",
    cum_exec_qty: str = "0",
    avg_price: str = "0",
    order_filter: str = "",
    stop_order_type: str = "",
    trigger_price: str = "0",
    order_id: str = "venue-1",
) -> dict:
    row = {
        "orderId": order_id,
        "orderLinkId": order_link_id,
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": qty,
        "price": "0",
        "avgPrice": avg_price,
        "cumExecQty": cum_exec_qty,
        "leavesQty": qty,
        "timeInForce": "GTC",
        "orderStatus": status,
        "createdTime": "1700000000000",
        "updatedTime": "1700000000000",
    }
    if order_filter:
        row["orderFilter"] = order_filter
    if stop_order_type:
        row["stopOrderType"] = stop_order_type
    if trigger_price != "0":
        row["triggerPrice"] = trigger_price
    return row


def execution_row(
    *,
    exec_id: str,
    order_link_id: str,
    exec_qty: str,
    exec_price: str = "50000",
    symbol: str = "BTCUSDT",
) -> dict:
    """One row of the venue's per-trade execution record."""
    return {
        "execId": exec_id,
        "orderId": "venue-order",
        "orderLinkId": order_link_id,
        "symbol": symbol,
        "execQty": exec_qty,
        "execPrice": exec_price,
        "execType": "Trade",
        "execTime": "1700000000000",
        "isMaker": "false",
        "fee": "0",
        "feeRate": "0",
    }


def executions_response(rows: list[dict]) -> dict:
    return {
        "retCode": 0,
        "retMsg": "OK",
        "result": {"list": rows, "nextPageCursor": ""},
    }


def make_broker(transport: RecordingTransport, **kwargs: Any) -> BybitDemoBroker:
    return BybitDemoBroker(
        api_key="demo-key",
        api_secret="demo-secret",
        base_url=DEMO_BASE_URL,
        transport=transport,
        **kwargs,
    )


def broker_order(
    *,
    client_order_id: str = "o1",
    filled_quantity: float = 0.0,
    quantity: float = 0.1,
    status: str = "partially_filled",
    symbol: str = "BTC/USDT",
    side: str = "sell",
) -> BrokerOrder:
    return BrokerOrder(
        client_order_id=client_order_id,
        broker_order_id="venue-1",
        symbol=symbol,
        side=side,
        quantity=quantity,
        filled_quantity=filled_quantity,
        filled_avg_price=50000.0,
        status=status,
    )


# ---------------------------------------------------------------------------
# The cumulative quantity, not the id, is what makes the accounting once
# ---------------------------------------------------------------------------


class TestGrowingFillIsAccountedOnce:
    def test_the_same_fill_seen_twice_is_applied_once(self):
        transport = RecordingTransport()
        broker = make_broker(transport)

        order = broker_order(filled_quantity=0.04)
        assert broker.apply_fill(order) is True
        # The same order, read back with the same quantity: nothing new.
        assert broker.apply_fill(order) is False
        assert broker._state.applied_fill_quantities["o1"] == pytest.approx(0.04)

    def test_a_growing_fill_is_applied_again(self):
        transport = RecordingTransport()
        broker = make_broker(transport)

        assert broker.apply_fill(broker_order(filled_quantity=0.04)) is True
        # The order traded more. The **same** client order id, a larger
        # cumulative quantity: this is new and has to be accounted for.
        assert broker.apply_fill(broker_order(filled_quantity=0.06)) is True
        assert broker._state.applied_fill_quantities["o1"] == pytest.approx(0.06)

    def test_a_growing_fill_does_not_look_like_a_duplicate(self):
        """The old guard keyed on the id alone, so this reported "duplicate"."""
        transport = RecordingTransport()
        broker = make_broker(transport)

        broker.apply_fill(broker_order(filled_quantity=0.04))
        # The old implementation returned False here, because the id was
        # already in ``applied_fills``. The extra 0.02 never reached the
        # accounting.
        assert broker.apply_fill(broker_order(filled_quantity=0.06)) is True

    def test_a_shrinking_report_is_never_un_applied(self):
        transport = RecordingTransport()
        broker = make_broker(transport)

        broker.apply_fill(broker_order(filled_quantity=0.06))
        # The venue cannot un-trade. A smaller report is a stale read, and it
        # must not roll the accounting back.
        assert broker.apply_fill(broker_order(filled_quantity=0.04)) is False
        assert broker._state.applied_fill_quantities["o1"] == pytest.approx(0.06)

    def test_a_non_finite_quantity_is_refused(self):
        transport = RecordingTransport()
        broker = make_broker(transport)

        assert broker.apply_fill(broker_order(filled_quantity=float("nan"))) is False
        assert broker.apply_fill(broker_order(filled_quantity=float("inf"))) is False
        assert "o1" not in broker._state.applied_fill_quantities

    def test_an_order_that_did_not_trade_is_not_applied(self):
        transport = RecordingTransport()
        broker = make_broker(transport)

        order = broker_order(filled_quantity=0.0, status="accepted")
        assert broker.apply_fill(order) is False


# ---------------------------------------------------------------------------
# The execution id is the unit of exactly-once accounting
# ---------------------------------------------------------------------------


class TestExecutionIdsAreAppliedOnce:
    def test_the_same_execution_id_is_never_applied_twice(self):
        transport = RecordingTransport()
        broker = make_broker(transport)

        rows = [execution_row(exec_id="e1", order_link_id="o1", exec_qty="0.04")]
        first = broker.apply_executions(rows)
        assert first["newly_applied"] == ["e1"]
        assert first["already_applied"] == []

        second = broker.apply_executions(rows)
        assert second["newly_applied"] == []
        assert second["already_applied"] == ["e1"]

    def test_two_trades_of_one_order_are_both_applied(self):
        transport = RecordingTransport()
        broker = make_broker(transport)

        rows = [
            execution_row(exec_id="e1", order_link_id="o1", exec_qty="0.04"),
            execution_row(exec_id="e2", order_link_id="o1", exec_qty="0.02"),
        ]
        result = broker.apply_executions(rows)
        assert sorted(result["newly_applied"]) == ["e1", "e2"]

        # Re-reading the same list applies nothing new.
        again = broker.apply_executions(rows)
        assert again["newly_applied"] == []
        assert sorted(again["already_applied"]) == ["e1", "e2"]

    def test_a_record_with_no_execution_id_is_skipped_not_guessed(self):
        transport = RecordingTransport()
        broker = make_broker(transport)

        row = execution_row(exec_id="", order_link_id="o1", exec_qty="0.04")
        result = broker.apply_executions([row])
        assert result["newly_applied"] == []
        assert result["skipped"] == ["o1"]

    def test_the_applied_execution_ids_survive_a_restart(self):
        transport = RecordingTransport()
        path = _state_path()
        broker = make_broker(transport, state_path=path)

        rows = [execution_row(exec_id="e1", order_link_id="o1", exec_qty="0.04")]
        broker.apply_executions(rows)

        # A restart reads the state back.
        reloaded = BrokerStateStore(path).load()
        assert "e1" in reloaded.applied_execution_ids.get("o1", [])

        # The restarted broker does not re-apply the trade.
        restarted = make_broker(transport, state_path=path)
        again = restarted.apply_executions(rows)
        assert again["newly_applied"] == []
        assert again["already_applied"] == ["e1"]


def _state_path():
    """A throwaway path for the broker state, unique to the test."""
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp()) / "broker_state.json"


# ---------------------------------------------------------------------------
# A restart keeps the accounting
# ---------------------------------------------------------------------------


class TestTheAccountingSurvivesARestart:
    def test_a_growing_fill_after_a_restart_is_still_applied(self):
        transport = RecordingTransport()
        path = _state_path()
        broker = make_broker(transport, state_path=path)

        assert broker.apply_fill(broker_order(filled_quantity=0.04)) is True

        reloaded = BrokerStateStore(path).load()
        assert reloaded.applied_fill_quantities["o1"] == pytest.approx(0.04)
        restarted = make_broker(transport, state_path=path)

        # The order traded more while the worker was down.
        assert restarted.apply_fill(broker_order(filled_quantity=0.06)) is True
        assert restarted._state.applied_fill_quantities["o1"] == pytest.approx(0.06)

    def test_the_same_fill_after_a_restart_is_not_applied_again(self):
        transport = RecordingTransport()
        path = _state_path()
        broker = make_broker(transport, state_path=path)

        assert broker.apply_fill(broker_order(filled_quantity=0.04)) is True

        reloaded = BrokerStateStore(path).load()
        assert reloaded.applied_fill_quantities["o1"] == pytest.approx(0.04)
        restarted = make_broker(transport, state_path=path)

        assert restarted.apply_fill(broker_order(filled_quantity=0.04)) is False


# ---------------------------------------------------------------------------
# The sell is sized against the venue's balance, not a stale cache
# ---------------------------------------------------------------------------


class TestTheSellIsSizedAgainstTheVenue:
    def _broker_with_balance(self, balance: str) -> tuple[BybitDemoBroker, RecordingTransport]:
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response({"BTC": {"walletBalance": balance}, "USDT": {"walletBalance": "1000"}}),
        )
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT")]),
        )
        transport.add(
            "GET",
            "/v5/market/tickers",
            {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "50000"}]}},
        )
        transport.add("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add(
            "GET",
            "/v5/order/history",
            {"retCode": 0, "result": {"list": [], "nextPageCursor": ""}},
        )
        transport.add(
            "GET",
            "/v5/execution/list",
            {"retCode": 0, "result": {"list": [], "nextPageCursor": ""}},
        )
        broker = make_broker(transport)
        broker._state.confirmed = True
        broker.refresh_account()
        return broker, transport

    def test_a_sell_larger_than_the_venue_balance_is_refused(self):
        broker, _ = self._broker_with_balance("0.06")

        # The cached position set says more than the venue holds, which is
        # exactly the stale state a partial fill leaves behind.
        broker._positions = {
            "BTC/USDT": _position("BTC/USDT", quantity=0.10),
        }

        with pytest.raises(PaperRiskRejected) as exc:
            broker.submit_order(
                symbol="BTC/USDT",
                quantity=0.10,
                side="sell",
                reference_price=50000.0,
                client_order_id="sell-1",
                allow_orders=True,
            )
        assert "short" in str(exc.value).lower()

    def test_a_sell_within_the_venue_balance_is_allowed(self):
        broker, transport = self._broker_with_balance("0.06")

        # The order-create route answers "success", so the sell reaches the
        # venue. The quantity is covered by the balance, so it is **not**
        # refused as a short.
        transport.add(
            "POST",
            "/v5/order/create",
            {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "orderId": "venue-1",
                    "orderLinkId": "sell-2",
                    "symbol": "BTCUSDT",
                    "side": "Sell",
                    "qty": "0.06",
                    "price": "0",
                    "avgPrice": "0",
                    "cumExecQty": "0",
                    "orderStatus": "New",
                    "createdTime": "1700000000000",
                },
            },
        )

        order = broker.submit_order(
            symbol="BTC/USDT",
            quantity=0.06,
            side="sell",
            reference_price=50000.0,
            client_order_id="sell-2",
            allow_orders=True,
        )
        assert order.client_order_id == "sell-2"
        assert order.side == "sell"

    def test_an_unreadable_balance_refuses_the_sell(self):
        broker, transport = self._broker_with_balance("0.06")
        transport.fail(
            "GET",
            "/v5/account/wallet-balance",
            OSError("the venue could not be read"),
        )

        with pytest.raises(PaperRiskRejected) as exc:
            broker.submit_order(
                symbol="BTC/USDT",
                quantity=0.06,
                side="sell",
                reference_price=50000.0,
                client_order_id="sell-3",
                allow_orders=True,
            )
        message = str(exc.value)
        assert "could not be read" in message
        assert "unknown" in message


def _position(symbol: str, *, quantity: float):
    from tradingagents.paper_experiment.broker.bybit_demo import BrokerPosition

    return BrokerPosition(
        symbol=symbol,
        quantity=quantity,
        available_quantity=quantity,
        avg_price=None,
        mark_price=None,
        unrealised_pnl=None,
        side="buy",
        venue_symbol=symbol.replace("/", ""),
    )


# ---------------------------------------------------------------------------
# The reconciliation accounts for the trades it reads
# ---------------------------------------------------------------------------


class TestTheReconciliationAccountsForTheTrades:
    def test_the_reconciliation_accounts_for_each_execution_once(self):
        transport = RecordingTransport()
        broker = make_broker(transport)
        broker._state.confirmed = True

        transport.add("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add(
            "GET",
            "/v5/order/history",
            {"retCode": 0, "result": {"list": [], "nextPageCursor": ""}},
        )
        transport.add(
            "GET",
            "/v5/execution/list",
            executions_response(
                [
                    execution_row(exec_id="e1", order_link_id="p-sl", exec_qty="0.04"),
                ]
            ),
        )

        first = broker.reconcile_protection_orders("BTC/USDT")
        assert first["executions"]["newly_applied"] == ["e1"]

        second = broker.reconcile_protection_orders("BTC/USDT")
        assert second["executions"]["newly_applied"] == []
        assert second["executions"]["already_applied"] == ["e1"]

    def test_an_unreadable_execution_record_is_reported_not_hidden(self):
        transport = RecordingTransport()
        broker = make_broker(transport)
        broker._state.confirmed = True

        transport.add("GET", "/v5/order/realtime", {"retCode": 0, "result": {"list": []}})
        transport.add(
            "GET",
            "/v5/order/history",
            {"retCode": 0, "result": {"list": [], "nextPageCursor": ""}},
        )
        transport.fail("GET", "/v5/execution/list", OSError("the venue could not be read"))

        result = broker.reconcile_protection_orders("BTC/USDT")
        assert result["complete"] is False
        assert any("could not be accounted for" in error for error in result["errors"])


# ---------------------------------------------------------------------------
# The worker tells the broker about the fill
# ---------------------------------------------------------------------------


class TestTheWorkerAccountsForTheFill:
    """The worker's reconciliation is where the venue's answer arrives.

    A broker that owns protection orders has work to do when one of them
    trades: the counterpart has to be cancelled and the residual accounted
    for. That work is the broker's, and it is triggered by the broker being
    told that a fill landed. The worker's reconciliation is the only place
    the venue's answer reaches the worker, so it is the only place the
    broker can learn about the fill.
    """

    def test_the_worker_tells_the_broker_about_a_fill(self):
        from tradingagents.paper_experiment.broker.trading import PaperTradingWorker

        seen: list[BrokerOrder] = []

        class _Broker:
            """A broker that records the fills it is told about."""

            def __init__(self) -> None:
                self._order = broker_order(client_order_id="o1", filled_quantity=0.04)

            def _reconcile(self, client_order_id: str) -> BrokerOrder:
                return self._order

            def apply_fill(self, order: BrokerOrder) -> bool:
                seen.append(order)
                return True

        worker = PaperTradingWorker.__new__(PaperTradingWorker)
        worker._broker = _Broker()
        worker.state = _WorkerState()
        worker.state.orders = {"o1": {"client_order_id": "o1", "status": "accepted"}}
        worker._save = lambda: None

        outcomes = worker.reconcile_orders()

        assert outcomes == {"o1": "partially_filled"}
        # The broker was told, and it was told about **this** order.
        assert [order.client_order_id for order in seen] == ["o1"]
        assert seen[0].filled_quantity == pytest.approx(0.04)

    def test_a_broker_without_the_seam_is_not_broken_by_it(self):
        """Alpaca Paper carries the seam, but a broker that does not is fine."""
        from tradingagents.paper_experiment.broker.trading import PaperTradingWorker

        class _Broker:
            def _reconcile(self, client_order_id: str) -> BrokerOrder:
                return broker_order(client_order_id=client_order_id, filled_quantity=0.04)

        worker = PaperTradingWorker.__new__(PaperTradingWorker)
        worker._broker = _Broker()
        worker.state = _WorkerState()
        worker.state.orders = {"o1": {"client_order_id": "o1", "status": "accepted"}}
        worker._save = lambda: None

        outcomes = worker.reconcile_orders()
        assert outcomes == {"o1": "partially_filled"}

    def test_a_failed_accounting_is_reported_not_hidden(self):
        from tradingagents.paper_experiment.broker.trading import PaperTradingWorker

        class _Broker:
            def _reconcile(self, client_order_id: str) -> BrokerOrder:
                return broker_order(client_order_id=client_order_id, filled_quantity=0.04)

            def apply_fill(self, order: BrokerOrder) -> bool:
                raise RuntimeError("the accounting failed")

        worker = PaperTradingWorker.__new__(PaperTradingWorker)
        worker._broker = _Broker()
        worker.state = _WorkerState()
        worker.state.orders = {"o1": {"client_order_id": "o1", "status": "accepted"}}
        worker._save = lambda: None

        worker.reconcile_orders()
        assert "the accounting failed" in (worker.state.last_error or "")


class _WorkerState:
    """The slice of the worker state that the reconciliation touches."""

    def __init__(self) -> None:
        self.orders: dict[str, dict] = {}
        self.last_error: str | None = None
