"""Regression tests for the two safety fixes in the Bybit Demo broker.

Both fixes close the same kind of hole: a state the broker could not
**prove** was being treated as a state it could.

1. **An unknown counterpart state must not resolve an OCO pair.** When the
   venue's answer about the other leg could not be read, the pair stays
   unresolved. "Not seen" is not "finished".

2. **A partial fill must not leave the residual position unprotected.** When
   one leg of a protection pair trades partly, the quantity that is still
   held has no guard once the counterpart is cancelled. The residual is
   either re-protected and **verified** at the venue, or the position is
   reported unprotected and the entry gate closes.

Every test drives the real broker code through a mock transport. No test
makes a network call and no test needs a credential.
"""

from __future__ import annotations

import pytest

from tests.test_bybit_demo_audit import (
    RecordingTransport,
    instruments_response,
    make_broker,
    order_row,
    spot_instrument,
    wallet_response,
)
from tradingagents.paper_experiment.broker.bybit_demo import (
    PROTECTION_CANCEL_CONFIRMED_TERMINAL,
    PROTECTION_CANCEL_STILL_ACTIVE,
    PROTECTION_CANCEL_UNKNOWN,
    BrokerOrder,
    BybitDemoBroker,
)
from tradingagents.paper_experiment.broker.state import BrokerAccountState


def _execution_row(order_link_id: str, exec_qty: str, exec_price: str = "50000") -> dict:
    """One row of the venue's per-trade execution record."""
    return {
        "orderId": "venue-order",
        "orderLinkId": order_link_id,
        "symbol": "BTCUSDT",
        "execQty": exec_qty,
        "execPrice": exec_price,
        "execType": "Trade",
        "execTime": "1700000000000",
        "isMaker": "false",
        "fee": "0",
        "feeRate": "0",
    }


def _cancelled_tp_in_history() -> list[dict]:
    """The venue's history after the counterpart take-profit was cancelled.

    A cancelled order leaves the live list and shows up in the **history**
    with a terminal status. That history row is what confirms the
    cancellation: without it the counterpart's fate is unknown.
    """
    return [
        order_row(
            order_link_id="p-tp",
            status="Cancelled",
            side="Sell",
            qty="0.1",
            order_filter="tpslOrder",
            stop_order_type="TakeProfit",
            trigger_price="55000",
            order_id="v-tp",
        )
    ]


def _base_transport(
    *,
    btc_balance: str = "0.06",
    sl_status: str = "Untriggered",
    tp_status: str = "Untriggered",
    sl_qty: str = "0.1",
    tp_qty: str = "0.1",
    executions: list[dict] | None = None,
    executions_cursor: str = "",
    sl_cum_exec: str = "0",
    tp_cum_exec: str = "0",
    residual_legs: list[dict] | None = None,
    history_legs: list[dict] | None = None,
    counterpart_cancelled: bool = False,
) -> RecordingTransport:
    """A transport that models a BTC/USDT position with a protection pair.

    ``residual_legs`` are the replacement orders the venue carries once the
    residual pair has been created. They are what a real venue's live list
    would show after the replacement is accepted.

    ``history_legs`` are the finished orders the venue's **order history**
    carries. A counterpart that was cancelled leaves the live list and shows
    up in the history with a terminal status, and that history read is what
    confirms the cancellation.

    ``counterpart_cancelled`` models exactly that: the take-profit is removed
    from the live list and appears in the history as ``Cancelled``. It is the
    state the venue is in after a stop loss traded and its counterpart was
    cancelled.
    """
    transport = RecordingTransport()
    transport.add(
        "GET",
        "/v5/account/wallet-balance",
        wallet_response(
            {"USDT": {"walletBalance": "1000"}, "BTC": {"walletBalance": btc_balance}}
        ),
    )
    transport.add(
        "GET",
        "/v5/market/instruments-info",
        instruments_response([spot_instrument("BTCUSDT")]),
    )
    live_legs = [
        order_row(
            order_link_id="p-sl",
            status=sl_status,
            side="Sell",
            qty=sl_qty,
            cum_exec_qty=sl_cum_exec,
            order_filter="tpslOrder",
            stop_order_type="StopLoss",
            trigger_price="45000",
            order_id="v-sl",
        ),
    ]
    if not counterpart_cancelled:
        live_legs.append(
            order_row(
                order_link_id="p-tp",
                status=tp_status,
                side="Sell",
                qty=tp_qty,
                cum_exec_qty=tp_cum_exec,
                order_filter="tpslOrder",
                stop_order_type="TakeProfit",
                trigger_price="55000",
                order_id="v-tp",
            )
        )
    live_legs.extend(residual_legs or [])
    transport.add(
        "GET",
        "/v5/order/realtime",
        {
            "retCode": 0,
            "result": {
                "list": live_legs,
            },
        },
    )
    history = list(history_legs or [])
    if counterpart_cancelled:
        history.extend(_cancelled_tp_in_history())
    transport.add(
        "GET",
        "/v5/order/history",
        {
            "retCode": 0,
            "result": {"list": history, "nextPageCursor": ""},
        },
    )
    transport.add(
        "GET",
        "/v5/market/tickers",
        {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "lastPrice": "50000"}]}},
    )
    transport.add(
        "GET",
        "/v5/execution/list",
        {
            "retCode": 0,
            "result": {"list": executions or [], "nextPageCursor": executions_cursor},
        },
    )
    return transport


class TestUnknownCounterpartDoesNotResolveThePair:
    """An unreadable counterpart state is unknown, never terminal."""

    def test_a_missing_counterpart_keeps_the_pair_unresolved(self):
        """The counterpart is nowhere to be found: the pair stays open.

        The venue's live list does not carry the target and the history does
        not either. That is **unknown**: the target may still be working at
        the venue, so the pair must not be declared settled.
        """
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response(
                {"USDT": {"walletBalance": "1000"}, "BTC": {"walletBalance": "0.1"}}
            ),
        )
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT")]),
        )
        # The live list carries only the stop; the target is not in it.
        transport.add(
            "GET",
            "/v5/order/realtime",
            {
                "retCode": 0,
                "result": {
                    "list": [
                        order_row(
                            order_link_id="p-sl",
                            status="Filled",
                            side="Sell",
                            qty="0.1",
                            order_filter="tpslOrder",
                            stop_order_type="StopLoss",
                            trigger_price="45000",
                            order_id="v-sl",
                        )
                    ]
                },
            },
        )
        # The history carries the stop too, but not the target.
        transport.add(
            "GET",
            "/v5/order/history",
            {
                "retCode": 0,
                "result": {
                    "list": [
                        order_row(
                            order_link_id="p-sl",
                            status="Filled",
                            side="Sell",
                            qty="0.1",
                            cum_exec_qty="0.1",
                            order_filter="tpslOrder",
                            stop_order_type="StopLoss",
                            trigger_price="45000",
                            order_id="v-sl",
                        )
                    ],
                    "nextPageCursor": "",
                },
            },
        )
        transport.add(
            "GET",
            "/v5/execution/list",
            {"retCode": 0, "result": {"list": [_execution_row("p-sl", "0.1")]}},
        )

        broker = make_broker(transport)
        broker.connect()
        broker._state.protection_order_pairs["p"] = {
            "sl_client_id": "p-sl",
            "tp_client_id": "p-tp",
            "symbol": "BTC/USDT",
            "position_qty": 0.1,
        }

        broker._cancel_counterpart_protection("p", "p-tp", "BTC/USDT", "test")

        # The pair is still recorded: the counterpart was never proven gone.
        assert "p" in broker._state.protection_order_pairs

    def test_an_unknown_counterpart_status_keeps_the_pair_unresolved(self):
        """A status the broker cannot classify is unknown, not terminal."""
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response(
                {"USDT": {"walletBalance": "1000"}, "BTC": {"walletBalance": "0.1"}}
            ),
        )
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT")]),
        )
        # A status the project does not know: it cannot be called terminal.
        transport.add(
            "GET",
            "/v5/order/realtime",
            {
                "retCode": 0,
                "result": {
                    "list": [
                        order_row(
                            order_link_id="p-tp",
                            status="SomeBrandNewVenueStatus",
                            side="Sell",
                            qty="0.1",
                            order_filter="tpslOrder",
                            stop_order_type="TakeProfit",
                            trigger_price="55000",
                            order_id="v-tp",
                        )
                    ]
                },
            },
        )
        transport.add(
            "GET",
            "/v5/order/history",
            {"retCode": 0, "result": {"list": [], "nextPageCursor": ""}},
        )
        transport.add(
            "GET",
            "/v5/execution/list",
            {"retCode": 0, "result": {"list": []}},
        )

        broker = make_broker(transport)
        broker.connect()
        broker._state.protection_order_pairs["p"] = {
            "sl_client_id": "p-sl",
            "tp_client_id": "p-tp",
            "symbol": "BTC/USDT",
            "position_qty": 0.1,
        }

        broker._cancel_counterpart_protection("p", "p-tp", "BTC/USDT", "test")

        assert "p" in broker._state.protection_order_pairs


class TestTheCancelOutcomeDecidesTheResidual:
    """The cancellation's outcome decides whether a replacement may be made.

    A cancel request that was **accepted** is not the same fact as a
    cancelled order. The venue accepting the request says only that it will
    try. Creating a replacement pair while the original counterpart is still
    working would put two live sells against the same quantity, and whichever
    triggered second would sell more than the account holds. So the outcome
    has to be explicit, and only a confirmed terminal status permits a
    replacement.
    """

    def _broker(self, transport: RecordingTransport) -> BybitDemoBroker:
        broker = make_broker(transport)
        broker.connect()
        broker._state.protection_order_pairs["p"] = {
            "sl_client_id": "p-sl",
            "tp_client_id": "p-tp",
            "symbol": "BTC/USDT",
            "position_qty": 0.1,
        }
        broker._state.orders["p-tp"] = {"status": "accepted"}
        return broker

    def _transport_with_tp_status(self, tp_status: str) -> RecordingTransport:
        transport = RecordingTransport()
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            wallet_response(
                {"USDT": {"walletBalance": "1000"}, "BTC": {"walletBalance": "0.1"}}
            ),
        )
        transport.add(
            "GET",
            "/v5/market/instruments-info",
            instruments_response([spot_instrument("BTCUSDT")]),
        )
        transport.add(
            "GET",
            "/v5/order/realtime",
            {
                "retCode": 0,
                "result": {
                    "list": [
                        order_row(
                            order_link_id="p-tp",
                            status=tp_status,
                            side="Sell",
                            qty="0.1",
                            order_filter="tpslOrder",
                            stop_order_type="TakeProfit",
                            trigger_price="55000",
                            order_id="v-tp",
                        )
                    ]
                },
            },
        )
        transport.add(
            "GET",
            "/v5/order/history",
            {"retCode": 0, "result": {"list": [], "nextPageCursor": ""}},
        )
        transport.add(
            "GET",
            "/v5/execution/list",
            {"retCode": 0, "result": {"list": []}},
        )
        transport.add(
            "POST",
            "/v5/order/cancel",
            {"retCode": 0, "retMsg": "OK", "result": {"orderId": "v-tp"}},
        )
        return transport

    def test_a_confirmed_terminal_counterpart_permits_a_replacement(self):
        transport = self._transport_with_tp_status("Cancelled")
        broker = self._broker(transport)

        outcome = broker._cancel_counterpart_protection("p", "p-tp", "BTC/USDT", "test")
        assert outcome == PROTECTION_CANCEL_CONFIRMED_TERMINAL
        assert broker._state.protection_cancel_outcomes["p"]["outcome"] == (
            PROTECTION_CANCEL_CONFIRMED_TERMINAL
        )

    def test_a_counterpart_still_working_forbids_a_replacement(self):
        transport = self._transport_with_tp_status("Untriggered")
        broker = self._broker(transport)

        outcome = broker._cancel_counterpart_protection("p", "p-tp", "BTC/USDT", "test")
        assert outcome == PROTECTION_CANCEL_STILL_ACTIVE
        assert broker._state.protection_cancel_outcomes["p"]["outcome"] == (
            PROTECTION_CANCEL_STILL_ACTIVE
        )

    def test_an_unreadable_venue_is_unknown_not_terminal(self):
        transport = self._transport_with_tp_status("Untriggered")
        broker = self._broker(transport)
        # The venue becomes unreadable **after** the account was confirmed.
        # Both reads that establish the counterpart's state fail: the one
        # inside the cancellation and the one that confirms it. An
        # unreadable venue is unknown, and unknown is not terminal.
        transport.fail(
            "GET",
            "/v5/order/realtime",
            OSError("the venue could not be read"),
        )
        transport.fail(
            "GET",
            "/v5/order/realtime",
            OSError("the venue could not be read"),
        )

        outcome = broker._cancel_counterpart_protection("p", "p-tp", "BTC/USDT", "test")
        assert outcome == PROTECTION_CANCEL_UNKNOWN

    def test_a_missing_counterpart_is_unknown_not_terminal(self):
        transport = self._transport_with_tp_status("Untriggered")
        broker = self._broker(transport)
        # The counterpart is nowhere: neither live nor in the history.
        transport.add(
            "GET",
            "/v5/order/realtime",
            {"retCode": 0, "result": {"list": []}},
        )

        outcome = broker._cancel_counterpart_protection("p", "p-tp", "BTC/USDT", "test")
        assert outcome == PROTECTION_CANCEL_UNKNOWN

    def test_a_still_working_counterpart_leaves_the_position_unprotected(self):
        """The whole point: no replacement while the original still works."""
        transport = _base_transport(
            sl_status="PartiallyFilled",
            sl_cum_exec="0.04",
            # The counterpart was NOT cancelled: the venue still shows it.
            counterpart_cancelled=False,
            executions=[_execution_row("p-sl", "0.04")],
        )
        broker = make_broker(transport)
        broker.connect()
        broker._state.protection_order_pairs["p"] = {
            "sl_client_id": "p-sl",
            "tp_client_id": "p-tp",
            "symbol": "BTC/USDT",
            "position_qty": 0.1,
        }
        broker._state.orders["p-tp"] = {"status": "accepted"}
        broker._state.protection_levels["BTC/USDT"] = {
            "stop_loss": 45000.0,
            "take_profit": 55000.0,
        }

        partial = BrokerOrder(
            client_order_id="p-sl",
            broker_order_id="v-sl",
            symbol="BTC/USDT",
            side="sell",
            quantity=0.1,
            filled_quantity=0.04,
            filled_avg_price=50000.0,
            status="partially_filled",
        )
        broker.apply_fill(partial)

        residual = broker._state.protection_residuals.get("p")
        assert residual is not None
        # No replacement pair was created, so the held quantity has no guard.
        assert residual["protected"] is False
        assert residual["created"] is False
        assert "not confirmed finished" in residual["reason"]

        # And the position is reported unprotected.
        broker.positions(refresh=True)
        protected, reason = broker.is_position_protected("BTC/USDT")
        assert protected is False


class TestPartialFillLeavesNoResidualUnprotected:
    """A partial fill must not leave the held quantity without a guard."""

    def _broker_with_partial_fill(self, transport: RecordingTransport) -> BybitDemoBroker:
        broker = make_broker(transport)
        broker.connect()
        broker._state.protection_order_pairs["p"] = {
            "sl_client_id": "p-sl",
            "tp_client_id": "p-tp",
            "symbol": "BTC/USDT",
            "position_qty": 0.1,
        }
        broker._state.protection_levels["BTC/USDT"] = {
            "stop_loss": 45000.0,
            "take_profit": 55000.0,
            "position_qty": 0.1,
        }
        return broker

    def test_a_failed_residual_creation_leaves_the_position_unprotected(self):
        """The residual cannot be guarded: the position is unprotected.

        The venue's executions say 0.04 of the stop traded and the venue
        still holds 0.06. The replacement pair cannot be created, so the
        quantity that is still held has no guard at all — and the system has
        to say so.
        """
        transport = _base_transport(
            btc_balance="0.06",
            sl_status="PartiallyFilled",
            sl_cum_exec="0.04",
            counterpart_cancelled=True,
            executions=[_execution_row("p-sl", "0.04")],
        )
        # The replacement pair cannot be created: the venue refuses it.
        transport.add(
            "POST",
            "/v5/order/create",
            {"retCode": 10001, "retMsg": "order creation failed", "result": {}},
        )
        broker = self._broker_with_partial_fill(transport)

        partial = BrokerOrder(
            client_order_id="p-sl",
            broker_order_id="v-sl",
            symbol="BTC/USDT",
            side="sell",
            quantity=0.1,
            filled_quantity=0.04,
            filled_avg_price=50000.0,
            status="partially_filled",
        )
        broker.apply_fill(partial)

        residual = broker._state.protection_residuals.get("p")
        assert residual is not None
        assert residual["protected"] is False
        assert residual["created"] is False
        assert "unprotected" in residual["reason"]

        # The position the venue still holds is reported unprotected.
        broker.positions(refresh=True)
        protected, reason = broker.is_position_protected("BTC/USDT")
        assert protected is False
        assert "residual" in reason

    def test_a_truncated_execution_read_places_no_new_order(self):
        """An unproven traded quantity must not produce a new order."""
        # The execution read is truncated: the venue hands back a cursor, so
        # the walk stops with the read unproven.
        transport = _base_transport(
            btc_balance="0.06",
            sl_status="PartiallyFilled",
            sl_cum_exec="0.04",
            counterpart_cancelled=True,
            executions=[_execution_row("p-sl", "0.04")],
            executions_cursor="page-2",
        )
        broker = self._broker_with_partial_fill(transport)

        partial = BrokerOrder(
            client_order_id="p-sl",
            broker_order_id="v-sl",
            symbol="BTC/USDT",
            side="sell",
            quantity=0.1,
            filled_quantity=0.04,
            filled_avg_price=50000.0,
            status="partially_filled",
        )
        broker.apply_fill(partial)

        residual = broker._state.protection_residuals.get("p")
        assert residual is not None
        assert residual["protected"] is False
        assert "truncated" in residual["reason"]
        # No new order was placed: an unproven quantity never reaches the venue.
        assert transport.calls_to("POST", "/v5/order/create") == []

    def test_an_unreadable_coin_balance_places_no_new_order(self):
        """An unknown balance cannot be sized against, so nothing is placed."""
        transport = _base_transport(
            sl_status="PartiallyFilled",
            sl_cum_exec="0.04",
            counterpart_cancelled=True,
            executions=[_execution_row("p-sl", "0.04")],
        )
        # The coin list is gone: the balance cannot be read.
        transport.add(
            "GET",
            "/v5/account/wallet-balance",
            {"retCode": 0, "result": {"list": []}},
        )
        broker = self._broker_with_partial_fill(transport)

        partial = BrokerOrder(
            client_order_id="p-sl",
            broker_order_id="v-sl",
            symbol="BTC/USDT",
            side="sell",
            quantity=0.1,
            filled_quantity=0.04,
            filled_avg_price=50000.0,
            status="partially_filled",
        )
        broker.apply_fill(partial)

        residual = broker._state.protection_residuals.get("p")
        assert residual is not None
        assert residual["protected"] is False
        assert "balance" in residual["reason"]
        assert transport.calls_to("POST", "/v5/order/create") == []

    def test_a_missing_execution_record_places_no_new_order(self):
        """No execution record means the traded quantity is unproven."""
        transport = _base_transport(
            btc_balance="0.06",
            sl_status="PartiallyFilled",
            sl_cum_exec="0.04",
            counterpart_cancelled=True,
            executions=[],
        )
        broker = self._broker_with_partial_fill(transport)

        partial = BrokerOrder(
            client_order_id="p-sl",
            broker_order_id="v-sl",
            symbol="BTC/USDT",
            side="sell",
            quantity=0.1,
            filled_quantity=0.04,
            filled_avg_price=50000.0,
            status="partially_filled",
        )
        broker.apply_fill(partial)

        residual = broker._state.protection_residuals.get("p")
        assert residual is not None
        assert residual["protected"] is False
        assert transport.calls_to("POST", "/v5/order/create") == []

    def test_a_venue_confirmed_residual_is_protected(self):
        """The residual is re-protected and the venue confirms it.

        The whole point of the fix: the quantity the venue still holds gets a
        guard of its own, and the guard is **proved** at the venue before the
        position is called protected.
        """
        residual_prefix = "p-residual-0p04"
        transport = _base_transport(
            btc_balance="0.06",
            sl_status="PartiallyFilled",
            sl_cum_exec="0.04",
            counterpart_cancelled=True,
            executions=[_execution_row("p-sl", "0.04")],
            residual_legs=[
                order_row(
                    order_link_id=f"{residual_prefix}-sl",
                    status="Untriggered",
                    side="Sell",
                    qty="0.06",
                    order_filter="tpslOrder",
                    stop_order_type="StopLoss",
                    trigger_price="45000",
                    order_id="v-rsl",
                ),
                order_row(
                    order_link_id=f"{residual_prefix}-tp",
                    status="Untriggered",
                    side="Sell",
                    qty="0.06",
                    order_filter="tpslOrder",
                    stop_order_type="TakeProfit",
                    trigger_price="55000",
                    order_id="v-rtp",
                ),
            ],
        )
        # The venue accepts the replacement pair.
        transport.add(
            "POST",
            "/v5/order/create",
            {
                "retCode": 0,
                "retMsg": "OK",
                "result": {"orderId": "new-order", "orderLinkId": "ignored"},
            },
        )
        broker = self._broker_with_partial_fill(transport)

        partial = BrokerOrder(
            client_order_id="p-sl",
            broker_order_id="v-sl",
            symbol="BTC/USDT",
            side="sell",
            quantity=0.1,
            filled_quantity=0.04,
            filled_avg_price=50000.0,
            status="partially_filled",
        )
        broker.apply_fill(partial)

        residual = broker._state.protection_residuals.get("p")
        assert residual is not None
        assert residual["protected"] is True
        assert residual["created"] is True
        assert residual["verified"] is True
        assert residual["executed_quantity"] == pytest.approx(0.04)
        assert residual["residual_quantity"] == pytest.approx(0.06)

        # A replacement pair really was asked for.
        creates = transport.calls_to("POST", "/v5/order/create")
        assert len(creates) == 2
        # And it covers the residual, not the original quantity. The venue's
        # own step formats the number, so the value is compared as a number.
        residual_bodies = [c["json_body"] for c in creates]
        quantities = [float(body["qty"]) for body in residual_bodies]
        assert all(qty == pytest.approx(0.06) for qty in quantities)

    def test_a_residual_without_confirmed_legs_stays_unprotected(self):
        """A created order the venue does not confirm proves nothing."""
        transport = _base_transport(
            btc_balance="0.06",
            sl_status="PartiallyFilled",
            sl_cum_exec="0.04",
            counterpart_cancelled=True,
            executions=[_execution_row("p-sl", "0.04")],
        )
        transport.add(
            "POST",
            "/v5/order/create",
            {
                "retCode": 0,
                "retMsg": "OK",
                "result": {"orderId": "new-order", "orderLinkId": "ignored"},
            },
        )
        broker = self._broker_with_partial_fill(transport)

        # The venue's live list carries no replacement leg, so the pair
        # cannot be confirmed. The read above only knows the original legs.
        partial = BrokerOrder(
            client_order_id="p-sl",
            broker_order_id="v-sl",
            symbol="BTC/USDT",
            side="sell",
            quantity=0.1,
            filled_quantity=0.04,
            filled_avg_price=50000.0,
            status="partially_filled",
        )
        broker.apply_fill(partial)

        residual = broker._state.protection_residuals.get("p")
        assert residual is not None
        # Either the venue confirmed it (it did not here) or it is unprotected.
        assert residual["protected"] is False
        assert residual["created"] is True
        assert residual["verified"] is False


class TestResidualStateSurvivesARestart:
    """The residual verdict is durable: a restart must not forget it."""

    def test_the_residual_is_written_to_the_state_and_read_back(self):
        state = BrokerAccountState()
        state.protection_residuals["p"] = {
            "symbol": "BTC/USDT",
            "protected": False,
            "reason": "the residual is unprotected",
        }
        payload = state.to_dict()
        assert payload["protection_residuals"]["p"]["protected"] is False

        restored = BrokerAccountState.from_dict(payload)
        assert restored.protection_residuals["p"]["reason"] == "the residual is unprotected"

    def test_an_older_state_without_the_field_reads_back_empty(self):
        payload = BrokerAccountState().to_dict()
        payload.pop("protection_residuals")
        payload.pop("protection_levels")
        restored = BrokerAccountState.from_dict(payload)
        assert restored.protection_residuals == {}
        assert restored.protection_levels == {}
