"""Phase 35: the final review of the venue reconciliation and the protection roll-up.

The canonical-symbol fix in phase 34 was accepted. This file pins the three
things that were still wrong around it:

1. **The reconciliation was one-directional and read a cache.** It compared the
   worker's positions against the broker's *cached* positions, so a position
   that appeared at the venue after the cache was filled could not be seen — the
   comparison was the worker against its own earlier self. And it only ever
   asked "is a worker position missing from the venue?", never the other way
   round, so a position the venue held and the worker did not know about was
   invisible. The orders request also short-circuited the positions read
   entirely, so a successful order list could never tell the operator whether
   the positions had been verified.

2. **The protection roll-up let a closed history mask a live position.** The
   "every record is CLOSED" branch was evaluated before the "an open position
   has no protection record" branch, so a worker holding a newly opened,
   entirely unguarded position reported ``CLOSED`` whenever its older records
   were all closed. ``CLOSED`` says *no risk is left*, which is the opposite of
   what was true.

3. **The order list was read one page and dropped what it could not parse.**
   ``GET /v2/orders`` is paginated, so a single request saw only the first
   page. A record the venue sent that carried no usable client order id was
   silently skipped, so the reconciliation could report "everything agrees"
   about an order nobody could identify.

Every test here is offline and deterministic: the venue transport is injected,
so no test reaches the network and none needs a venue credential. The TJR rule
chain, the risk limits and the order path are **not** touched — these tests pin
the *reading and the reporting* around them.

Nothing here fabricates a venue answer. A test that wants a position the venue
holds asks the mocked venue for one; a test that wants the venue unreadable
makes the mocked venue fail. The worker never invents a position, a fill, a
protection record or an exit level.
"""

from __future__ import annotations

import json

import pytest

from tradingagents.paper_experiment.broker.alpaca_paper import (
    AlpacaPaperBroker,
    VenueOrderPage,
)
from tradingagents.paper_experiment.broker.trading import (
    PROTECTION_STATUS_CLOSED,
    PROTECTION_STATUS_EXIT_PENDING,
    PROTECTION_STATUS_NO_OPEN_POSITIONS,
    PROTECTION_STATUS_PENDING,
    PROTECTION_STATUS_PROTECTED,
    PROTECTION_STATUS_UNPROTECTED,
    PaperTradingConfig,
    PaperTradingWorker,
    StaticMarketData,
)

CRYPTO = "AVAX/USD"
VENUE_SPELLING = "AVAXUSD"
OTHER = "XRP/USD"
OTHER_VENUE_SPELLING = "XRPUSD"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)


_ACCOUNT = {
    "id": "paper-acct-1",
    "status": "ACTIVE",
    "cash": "40.51",
    "equity": "50.51",
    "buying_power": "40.51",
    "currency": "USD",
    "trading_blocked": False,
    "account_blocked": False,
    "shorting_enabled": False,
}


def _venue_position(**over):
    """A crypto position as Alpaca spells it — with no slash."""
    payload = {
        "symbol": VENUE_SPELLING,
        "qty": "1.180856",
        "market_value": "10.00",
        "avg_entry_price": "8.46859",
        "side": "long",
    }
    payload.update(over)
    return payload


def _venue_order(**over):
    payload = {
        "id": "broker-order-1",
        "client_order_id": "c-1",
        "symbol": CRYPTO,
        "side": "buy",
        "qty": "1.180856",
        "filled_qty": "1.180856",
        "filled_avg_price": "8.46859",
        "status": "filled",
        "type": "market",
        "time_in_force": "gtc",
        "submitted_at": "2026-11-10T13:30:00Z",
    }
    payload.update(over)
    return payload


def _transport(routes, *, calls=None):
    """A venue transport over a fixed routing table.

    A route may be a single response or a **list** of responses, in which case
    each call takes the next one. That is how the pagination tests answer
    successive pages of ``GET /v2/orders``.
    """

    def send(*, method, url, headers, params, json_body, timeout):
        if calls is not None:
            calls.append(
                {
                    "method": method,
                    "url": url,
                    "params": dict(params or {}),
                    "json": json_body,
                }
            )
        for (route_method, suffix), response in routes.items():
            if method == route_method and url.endswith(suffix):
                if isinstance(response, list):
                    if not response:
                        raise AssertionError(f"no response left for {method} {url}")
                    return response.pop(0)
                return response
        raise AssertionError(f"unexpected request: {method} {url}")

    return send


def _routes(*, account=None, positions=None, orders=None, order_list=None):
    return {
        ("GET", "/v2/account"): account or _Resp(200, _ACCOUNT),
        ("GET", "/v2/positions"): positions or _Resp(200, [_venue_position()]),
        ("GET", "/v2/orders"): order_list or _Resp(200, [_venue_order()]),
        ("POST", "/v2/orders"): orders or _Resp(200, _venue_order()),
    }


class _Source:
    """A decision source that yields a fixed list, one per call."""

    def __init__(self, decisions=()):
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


def _worker(tmp_path, *, decisions=(), routes=None, calls=None, name="w.json"):
    """A worker over a mocked venue, with the real broker adapter."""
    config = PaperTradingConfig(
        order_notional=25.0,
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
        market_data=StaticMarketData({CRYPTO: 8.47, OTHER: 2.71}),
        broker_factory=build,
    )


def _broker(routes, *, calls=None) -> AlpacaPaperBroker:
    """A broker adapter over a mocked venue, with no worker around it."""
    return AlpacaPaperBroker(
        api_key="paper-key",
        api_secret="paper-secret",
        transport=_transport(routes, calls=calls),
    )


# ---------------------------------------------------------------------------
# 1. The bidirectional position reconciliation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestThePositionsAreComparedInBothDirections:
    def test_a_venue_position_the_worker_does_not_know_is_reported(self, tmp_path):
        """The venue holds a position the worker has no record of.

        This is the direction the old reconciliation never asked about. The
        worker saw an empty position map and the venue held a real position;
        nothing in the report said so, so the operator's view implied the
        account was flat.
        """
        routes = _routes(
            positions=_Resp(200, [_venue_position(symbol=OTHER_VENUE_SPELLING, qty="5.0")])
        )
        worker = _worker(tmp_path, routes=routes)
        worker.start()
        worker.run_once()
        try:
            # The worker's own view is empty — the position it never recorded.
            worker.state.positions = {}
            report = worker.reconcile_with_venue()
            assert report["positions_readable"] is True
            missing = report["venue_positions_missing_from_worker"]
            assert len(missing) == 1
            assert missing[0]["symbol"] == OTHER
            assert missing[0]["quantity"] == pytest.approx(5.0)
            # ...and it is not in the other direction: the worker does not hold it.
            assert report["worker_positions_missing_from_venue"] == []
        finally:
            worker.stop()

    def test_a_worker_position_the_venue_does_not_report_is_reported(self, tmp_path):
        """The worker believes it holds a position the venue does not show."""
        routes = _routes(positions=_Resp(200, []))
        worker = _worker(tmp_path, routes=routes)
        worker.start()
        worker.run_once()
        try:
            worker.state.positions = {
                CRYPTO: {"symbol": CRYPTO, "quantity": 1.180856}
            }
            report = worker.reconcile_with_venue()
            missing = report["worker_positions_missing_from_venue"]
            assert len(missing) == 1
            assert missing[0]["symbol"] == CRYPTO
            assert missing[0]["quantity"] == pytest.approx(1.180856)
            assert report["venue_positions_missing_from_worker"] == []
        finally:
            worker.stop()

    def test_the_two_spellings_are_compared_as_one_instrument(self, tmp_path):
        """The venue's ``AVAXUSD`` is the worker's ``AVAX/USD``.

        Both sides are canonicalised before they are compared, so the same
        instrument is not reported as two positions that do not exist.
        """
        routes = _routes(positions=_Resp(200, [_venue_position()]))
        worker = _worker(tmp_path, routes=routes)
        worker.start()
        worker.run_once()
        try:
            report = worker.reconcile_with_venue()
            assert report["venue_positions_missing_from_worker"] == []
            assert report["worker_positions_missing_from_venue"] == []
            assert report["position_quantity_mismatches"] == []
        finally:
            worker.stop()

    def test_a_quantity_disagreement_is_reported_on_its_own(self, tmp_path):
        """Both sides hold the symbol, but in different quantities.

        This is a different fact from either side being absent, so it is
        reported separately rather than folded into the missing lists.
        """
        routes = _routes(positions=_Resp(200, [_venue_position(qty="2.5")]))
        worker = _worker(tmp_path, routes=routes)
        worker.start()
        worker.run_once()
        try:
            worker.state.positions = {
                CRYPTO: {"symbol": CRYPTO, "quantity": 1.180856}
            }
            report = worker.reconcile_with_venue()
            assert report["venue_positions_missing_from_worker"] == []
            assert report["worker_positions_missing_from_venue"] == []
            mismatches = report["position_quantity_mismatches"]
            assert len(mismatches) == 1
            assert mismatches[0]["symbol"] == CRYPTO
            assert mismatches[0]["worker_quantity"] == pytest.approx(1.180856)
            assert mismatches[0]["venue_quantity"] == pytest.approx(2.5)
        finally:
            worker.stop()

    def test_the_positions_are_read_fresh_not_from_the_cache(self, tmp_path):
        """A cached response cannot verify the venue.

        The old reconciliation compared the worker against the broker's
        **cached** positions — the worker's own earlier view. A position that
        appeared at the venue after the cache was filled was therefore
        impossible to see. The reconciliation now asks the venue again.

        The cycle reads the positions several times, and every one of those
        answers is the same. The reconciliation's read is the one that sees
        the venue's new position — which is the whole point: it asks.
        """
        calls: list[dict] = []
        unchanged = _Resp(200, [_venue_position()])
        changed = _Resp(
            200,
            [
                _venue_position(),
                _venue_position(symbol=OTHER_VENUE_SPELLING, qty="3.0"),
            ],
        )
        routes = _routes(
            positions=[
                unchanged,
                unchanged,
                unchanged,
                changed,
            ]
        )
        worker = _worker(tmp_path, routes=routes, calls=calls)
        worker.start()
        worker.run_once()
        try:
            position_reads = [
                call for call in calls if call["url"].endswith("/v2/positions")
            ]
            assert position_reads, "the cycle did not read the positions"
            # The worker's own view is the first answer. The venue has since
            # gained a position the worker has never seen.
            worker.state.positions = {
                CRYPTO: {"symbol": CRYPTO, "quantity": 1.180856}
            }
            calls.clear()
            report = worker.reconcile_with_venue()
            assert [
                call for call in calls if call["url"].endswith("/v2/positions")
            ], "the reconciliation did not ask the venue for the positions"
            # The fresh venue answer holds a position the earlier one did not.
            assert len(report["venue_positions_missing_from_worker"]) == 1
            assert report["venue_positions_missing_from_worker"][0]["symbol"] == OTHER
        finally:
            worker.stop()

    def test_an_empty_positions_response_is_readable(self, tmp_path):
        """A venue that answers "no positions" has been asked and said no."""
        routes = _routes(positions=_Resp(200, []))
        worker = _worker(tmp_path, routes=routes)
        worker.start()
        worker.run_once()
        try:
            worker.state.positions = {}
            report = worker.reconcile_with_venue()
            assert report["positions_readable"] is True
            assert report["venue_positions_missing_from_worker"] == []
            assert report["worker_positions_missing_from_venue"] == []
        finally:
            worker.stop()

    def test_a_failed_positions_request_is_unknown_not_empty(self, tmp_path):
        """A venue that cannot be asked has not been shown to hold nothing.

        A failed read and a successful empty read are different facts, and only
        one of them means "the account is flat". The venue answers the cycle's
        read and then fails the reconciliation's read, which is what a venue
        outage part-way through a run looks like.
        """
        routes = _routes(
            positions=[
                _Resp(200, [_venue_position()]),
                _Resp(503, {"message": "unavailable"}),
            ]
        )
        worker = _worker(tmp_path, routes=routes)
        worker.start()
        worker.run_once()
        try:
            report = worker.reconcile_with_venue()
            assert report["positions_readable"] is False
            assert report["venue_positions_missing_from_worker"] == []
            assert report["worker_positions_missing_from_venue"] == []
            assert report["detail"]
        finally:
            worker.stop()

    def test_a_successful_order_list_does_not_imply_the_positions_were_read(
        self, tmp_path
    ):
        """The two reads are separate, and the report keeps them apart.

        The orders request used to short-circuit the whole reconciliation, so
        a successful order list said nothing about the positions — and a
        *failed* order list silently skipped them.
        """
        routes = _routes(
            order_list=_Resp(200, [_venue_order()]),
            positions=[
                _Resp(200, [_venue_position()]),
                _Resp(503, {"message": "unavailable"}),
            ],
        )
        worker = _worker(tmp_path, routes=routes)
        worker.start()
        worker.run_once()
        try:
            report = worker.reconcile_with_venue()
            assert report["orders_readable"] is True
            assert report["positions_readable"] is False
        finally:
            worker.stop()

    def test_a_failed_order_list_still_reads_the_positions(self, tmp_path):
        """One failed request does not cancel the other."""
        routes = _routes(
            order_list=_Resp(503, {"message": "unavailable"}),
            positions=_Resp(200, [_venue_position()]),
        )
        worker = _worker(tmp_path, routes=routes)
        worker.start()
        worker.run_once()
        try:
            worker.state.positions = {}
            report = worker.reconcile_with_venue()
            assert report["orders_readable"] is False
            assert report["positions_readable"] is True
            assert len(report["venue_positions_missing_from_worker"]) == 1
        finally:
            worker.stop()

    def test_the_reconciliation_adopts_closes_and_submits_nothing(self, tmp_path):
        """An unmatched position is reported, never acted on.

        The report is a read. It does not adopt the venue's position into the
        worker's state, does not close it, does not invent an exit level for
        it and does not place an order.
        """
        calls: list[dict] = []
        routes = _routes(
            positions=_Resp(200, [_venue_position(symbol=OTHER_VENUE_SPELLING, qty="5.0")])
        )
        worker = _worker(tmp_path, routes=routes, calls=calls)
        worker.start()
        worker.run_once()
        try:
            worker.state.positions = {}
            calls.clear()
            report = worker.reconcile_with_venue()
            methods = [call["method"] for call in calls]
            assert "POST" not in methods
            assert "DELETE" not in methods
            assert "PUT" not in methods
            assert "GET" in methods
            # The position was reported, not adopted.
            assert report["venue_positions_missing_from_worker"]
            assert OTHER not in (worker.state.positions or {})
            # No exit levels were invented for it.
            assert OTHER not in (worker.state.stops or {})
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# 2. The protection roll-up precedence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheProtectionPrecedence:
    def test_a_closed_record_never_hides_a_new_open_position(self, tmp_path):
        """The regression: a closed history must not report ``CLOSED``.

        A worker that had closed one position kept its ``CLOSED`` record, then
        opened a second position that had no protection record at all. The
        roll-up evaluated "every record is CLOSED" first and reported
        ``CLOSED`` — a claim that no risk is left, while a position with no
        guard was open.
        """
        worker = _worker(tmp_path)
        worker.start()
        worker.run_once()
        try:
            # The closed position's history survives.
            worker.state.protections = {
                CRYPTO: {
                    "symbol": CRYPTO,
                    "status": PROTECTION_STATUS_CLOSED,
                    "detail": "the position was closed by an exit of 1.180856",
                }
            }
            # A different symbol is open now, with no protection record.
            worker.state.positions = {
                OTHER: {"symbol": OTHER, "quantity": 5.0}
            }
            worker._roll_up_protection_status()
            assert worker.state.protection_status != PROTECTION_STATUS_CLOSED
            assert worker.state.protection_status == PROTECTION_STATUS_PENDING
            assert OTHER in (worker.state.protection_detail or "")
        finally:
            worker.stop()

    def test_a_closed_record_on_the_open_symbol_is_also_not_closed(self, tmp_path):
        """The broker says it is held; its own record says it is closed.

        The two disagree, and the disagreement is the finding. The position is
        open, so the verdict is not ``CLOSED`` — and not ``PROTECTED`` either,
        because a record that says the episode is over does not confirm an
        executable exit.
        """
        worker = _worker(tmp_path)
        worker.start()
        worker.run_once()
        try:
            worker.state.positions = {
                CRYPTO: {"symbol": CRYPTO, "quantity": 1.180856}
            }
            worker.state.protections = {
                CRYPTO: {"symbol": CRYPTO, "status": PROTECTION_STATUS_CLOSED}
            }
            worker._roll_up_protection_status()
            assert worker.state.protection_status == PROTECTION_STATUS_PENDING
            assert worker.state.protection_status != PROTECTION_STATUS_CLOSED
            assert worker.state.protection_status != PROTECTION_STATUS_PROTECTED
        finally:
            worker.stop()

    def test_an_unprotected_position_wins_over_everything(self, tmp_path):
        """The worst verdict decides the roll-up."""
        worker = _worker(tmp_path)
        worker.start()
        worker.run_once()
        try:
            worker.state.positions = {
                CRYPTO: {"symbol": CRYPTO, "quantity": 1.180856},
                OTHER: {"symbol": OTHER, "quantity": 5.0},
            }
            worker.state.protections = {
                CRYPTO: {"symbol": CRYPTO, "status": PROTECTION_STATUS_UNPROTECTED},
                OTHER: {"symbol": OTHER, "status": PROTECTION_STATUS_PROTECTED},
            }
            worker._roll_up_protection_status()
            assert worker.state.protection_status == PROTECTION_STATUS_UNPROTECTED
        finally:
            worker.stop()

    def test_a_pending_exit_is_reported_as_exit_pending(self, tmp_path):
        """A pending exit keeps its own status."""
        worker = _worker(tmp_path)
        worker.start()
        worker.run_once()
        try:
            worker.state.positions = {
                CRYPTO: {"symbol": CRYPTO, "quantity": 1.180856}
            }
            worker.state.protections = {
                CRYPTO: {"symbol": CRYPTO, "status": PROTECTION_STATUS_EXIT_PENDING}
            }
            worker._roll_up_protection_status()
            assert worker.state.protection_status == PROTECTION_STATUS_EXIT_PENDING
        finally:
            worker.stop()

    def test_closed_is_reported_only_when_nothing_is_open(self, tmp_path):
        """The healthy closed case still reports ``CLOSED``."""
        worker = _worker(tmp_path)
        worker.start()
        worker.run_once()
        try:
            worker.state.positions = {}
            worker.state.protections = {
                CRYPTO: {"symbol": CRYPTO, "status": PROTECTION_STATUS_CLOSED}
            }
            worker._roll_up_protection_status()
            assert worker.state.protection_status == PROTECTION_STATUS_CLOSED
        finally:
            worker.stop()

    def test_no_positions_and_no_records_is_not_a_closed_verdict(self, tmp_path):
        """A worker that never opened anything says so plainly."""
        worker = _worker(tmp_path)
        worker.start()
        worker.run_once()
        try:
            worker.state.positions = {}
            worker.state.protections = {}
            worker._roll_up_protection_status()
            assert worker.state.protection_status == PROTECTION_STATUS_NO_OPEN_POSITIONS
            assert worker.state.protection_status != PROTECTION_STATUS_CLOSED
        finally:
            worker.stop()

    def test_protected_is_reported_only_when_every_open_position_is_covered(
        self, tmp_path
    ):
        """Every open position must have a broker-verified guard."""
        worker = _worker(tmp_path)
        worker.start()
        worker.run_once()
        try:
            worker.state.positions = {
                CRYPTO: {"symbol": CRYPTO, "quantity": 1.180856},
                OTHER: {"symbol": OTHER, "quantity": 5.0},
            }
            worker.state.protections = {
                CRYPTO: {"symbol": CRYPTO, "status": PROTECTION_STATUS_PROTECTED},
                OTHER: {"symbol": OTHER, "status": PROTECTION_STATUS_PROTECTED},
            }
            worker._roll_up_protection_status()
            assert worker.state.protection_status == PROTECTION_STATUS_PROTECTED
        finally:
            worker.stop()

    def test_a_closed_position_leaves_the_open_position_map(self, tmp_path):
        """A closed position does not linger as if it were still held.

        ``state.positions`` is a snapshot of the broker's holdings and is only
        refreshed by the next position read. The exit's filled quantity already
        proves the position is closed at the venue, so the snapshot must not
        keep claiming otherwise — otherwise the roll-up sees an open position
        whose own record says it is closed, and reports a contradiction
        instead of the truth.
        """
        worker = _worker(tmp_path)
        worker.start()
        worker.run_once()
        try:
            worker.state.positions = {
                CRYPTO: {"symbol": CRYPTO, "quantity": 1.180856}
            }
            worker.state.protections = {
                CRYPTO: {"symbol": CRYPTO, "status": PROTECTION_STATUS_PROTECTED}
            }
            worker._mark_protection_closed(CRYPTO, 1.180856)
            assert CRYPTO not in worker.state.positions
            assert worker.state.protection_status == PROTECTION_STATUS_CLOSED
        finally:
            worker.stop()

    def test_closing_one_position_does_not_close_another(self, tmp_path):
        """One symbol's exit does not touch another symbol's position."""
        worker = _worker(tmp_path)
        worker.start()
        worker.run_once()
        try:
            worker.state.positions = {
                CRYPTO: {"symbol": CRYPTO, "quantity": 1.180856},
                OTHER: {"symbol": OTHER, "quantity": 5.0},
            }
            worker.state.protections = {
                CRYPTO: {"symbol": CRYPTO, "status": PROTECTION_STATUS_PROTECTED},
                OTHER: {"symbol": OTHER, "status": PROTECTION_STATUS_UNPROTECTED},
            }
            worker._mark_protection_closed(CRYPTO, 1.180856)
            assert CRYPTO not in worker.state.positions
            assert OTHER in worker.state.positions
            # The other position's verdict still decides the roll-up.
            assert worker.state.protection_status == PROTECTION_STATUS_UNPROTECTED
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# 3. The order list: pagination and unclassified records
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheOrderListIsComplete:
    def test_the_pages_are_followed_to_the_end(self):
        """``GET /v2/orders`` is paginated, so one page is not the list."""
        calls: list[dict] = []
        page_one = _Resp(
            200,
            [_venue_order(id="o-1", client_order_id="c-1"),
             _venue_order(id="o-2", client_order_id="c-2")],
        )
        page_two = _Resp(
            200,
            [_venue_order(id="o-3", client_order_id="c-3"),
             _venue_order(id="o-4", client_order_id="c-4")],
        )
        page_three = _Resp(200, [])
        broker = _broker(
            {("GET", "/v2/orders"): [page_one, page_two, page_three]}, calls=calls
        )
        page = broker.list_orders(limit=2)
        assert isinstance(page, VenueOrderPage)
        assert [order.client_order_id for order in page.orders] == [
            "c-1",
            "c-2",
            "c-3",
            "c-4",
        ]
        assert page.pages_read == 3
        assert page.truncated is False
        # Each request after the first carried the venue's cursor.
        assert calls[1]["params"]["after"] == "o-2"
        assert calls[2]["params"]["after"] == "o-4"

    def test_a_short_page_ends_the_read(self):
        """A page shorter than the limit is the last page."""
        calls: list[dict] = []
        broker = _broker(
            {("GET", "/v2/orders"): _Resp(200, [_venue_order()])}, calls=calls
        )
        page = broker.list_orders(limit=50)
        assert len(page.orders) == 1
        assert page.pages_read == 1
        assert page.truncated is False
        assert len(calls) == 1

    def test_a_bounded_read_says_it_was_truncated(self):
        """A crawl that stops at its bound says so, rather than claiming all."""
        calls: list[dict] = []
        full_page = [_venue_order(id=f"o-{index}") for index in range(2)]
        broker = _broker(
            {("GET", "/v2/orders"): _Resp(200, full_page)}, calls=calls
        )
        page = broker.list_orders(limit=2, max_pages=1)
        assert page.truncated is True
        assert page.pages_read == 1
        assert len(calls) == 1
        # The orders that *were* read are still returned.
        assert len(page.orders) == 2

    def test_a_record_without_a_client_order_id_is_reported_not_dropped(self):
        """An order nobody can identify is a finding, not a gap.

        The old reader skipped such a record, so a venue order with no client
        order id disappeared from the comparison entirely and the
        reconciliation could report that everything agreed.
        """
        broker = _broker(
            {
                ("GET", "/v2/orders"): _Resp(
                    200,
                    [
                        _venue_order(id="o-1", client_order_id="c-1"),
                        _venue_order(id="o-2", client_order_id=None),
                    ],
                )
            }
        )
        page = broker.list_orders()
        assert [order.client_order_id for order in page.orders] == ["c-1"]
        assert len(page.unclassified) == 1
        assert page.unclassified[0]["broker_order_id"] == "o-2"
        assert page.unclassified[0]["symbol"] == CRYPTO
        # Nothing was thrown away.
        assert page.total_records == 2

    def test_a_record_that_is_not_an_object_is_reported(self):
        """A malformed entry is reported, not silently skipped."""
        broker = _broker(
            {("GET", "/v2/orders"): _Resp(200, ["not-an-order", _venue_order()])}
        )
        page = broker.list_orders()
        assert len(page.orders) == 1
        assert len(page.unclassified) == 1
        assert page.total_records == 2

    def test_the_unclassified_records_reach_the_report(self, tmp_path):
        """The worker's report carries the records it could not classify."""
        routes = _routes(
            order_list=_Resp(
                200,
                [_venue_order(client_order_id=None, id="broker-order-x")],
            )
        )
        worker = _worker(tmp_path, routes=routes)
        worker.start()
        worker.run_once()
        try:
            report = worker.reconcile_with_venue()
            assert report["orders_readable"] is True
            assert len(report["unclassified_orders"]) == 1
            assert report["unclassified_orders"][0]["broker_order_id"] == (
                "broker-order-x"
            )
            # It is not counted as an unknown order either: it could not be
            # matched at all, which is a different fact.
            assert report["unknown_orders"] == []
        finally:
            worker.stop()

    def test_reading_the_order_list_changes_nothing(self):
        """The read submits nothing, cancels nothing and adopts nothing."""
        calls: list[dict] = []
        broker = _broker(
            {("GET", "/v2/orders"): _Resp(200, [_venue_order()])}, calls=calls
        )
        broker.list_orders()
        methods = [call["method"] for call in calls]
        assert methods == ["GET"]
        assert broker._state.orders == {}
