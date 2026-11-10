"""Phase 33: the runtime audit of the idle worker — a current price, a truthful
protection verdict, and an order whose real status is read back.

A Railway deployment reported three things that did not agree with each other:

* ``last_snapshot_age_seconds = 863.94`` beside ``market_data_status = stale``
  and ``is_current_price = false``, while the Binance diagnostics still showed
  **HTTP 200** — the seam was not broken, it was simply *never asked again*;
* ``protection_status = UNPROTECTED`` beside ``positions = {}`` and
  ``protection_checked_at = null`` — a claim about an open position that did
  not exist;
* ``AVAX/USD accepted, filled_quantity=0`` and ``XRP/USD accepted,
  filled_quantity=0`` with ``positions={}`` — an order record that was a
  *submit-time snapshot* and was never reconciled, so it reported the moment of
  submission forever.

Each of those is a distinct defect with a distinct cause, and each is pinned
here at the code path that caused it:

1. **the refresh** — the market-data seam is re-read every cycle, independently
   of whether the cycle produced a decision or holds a position, so the stored
   observation does not age out while the worker idles;
2. **the protection verdict** — a worker that holds nothing reports
   ``NO_OPEN_POSITIONS``, never ``UNPROTECTED``, which is a claim about an open
   position; and a position that is open but not yet verified is
   ``PROTECTION_PENDING``, never a confirmed breach;
3. **the reconciliation** — an unsettled order is read back from the broker and
   its real status is recorded, the read never re-submits, and an unreadable
   venue leaves the record exactly as it was instead of inventing a result.

Everything here is offline and deterministic: the clock, the Binance client and
the venue transport are injected, so no test reaches the network and none needs
a venue credential. The TJR rule chain, the risk limits and the order path are
**not** modified here — these tests pin the *observability around* them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.crypto.types import Ticker
from tradingagents.paper_experiment.broker.trading import (
    MARKET_DATA_STATUS_OK,
    MARKET_DATA_STATUS_STALE,
    PROTECTION_STATUS_NO_OPEN_POSITIONS,
    PROTECTION_STATUS_PENDING,
    PROTECTION_STATUS_PROTECTED,
    BinanceSnapshotMarketData,
    MarketDataError,
)

SYMBOL = "AAPL"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _epoch_ms(stamp: datetime) -> int:
    return int(stamp.timestamp() * 1000)


def _ticker(symbol: str, *, observed_at: datetime | None, price: float = 100.0) -> Ticker:
    """A ticker whose instant comes from the venue's own timestamp.

    ``observed_at`` is what the venue stated, not what the test's clock says —
    that distinction is the whole point of the freshness check.
    """
    return Ticker.from_binance(
        {
            "symbol": symbol,
            "lastPrice": str(price),
            "bidPrice": str(price),
            "askPrice": str(price + 0.01),
            "closeTime": None if observed_at is None else _epoch_ms(observed_at),
        }
    )


class _FakeBinance:
    """A Binance client that serves the symbols it is given and counts its calls.

    The rows are normalized through the **real** ``Ticker.from_binance``, so the
    test exercises the production normalizer rather than a hand-built object.

    ``observed_at`` is the venue's timestamp. When it is a callable, the venue
    states the time of *this* request — which is what Binance does, and what
    makes the freshness check meaningful: a fresh read carries a fresh
    timestamp, while the stored observation from an earlier read keeps its
    older one.
    """

    def __init__(self, *, observed_at=None, price: float = 100.0) -> None:
        self.observed_at = observed_at
        self.price = price
        self.calls = 0
        self.last_diagnostics: list[dict] = []

    def _instant(self):
        if callable(self.observed_at):
            return self.observed_at()
        return self.observed_at

    def ticker_24h(self, symbols):
        self.calls += 1
        self.last_diagnostics.append({"status": 200})
        instant = self._instant()
        return {
            symbol: _ticker(symbol, observed_at=instant, price=self.price)
            for symbol in symbols
        }


class _FailingBinance:
    """A Binance client whose request fails, the way an unreachable venue does."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_diagnostics: list[dict] = []

    def ticker_24h(self, symbols):
        self.calls += 1
        self.last_diagnostics.append({"status": 503})
        raise RuntimeError("the venue is unreachable")


# ---------------------------------------------------------------------------
# 1. The refresh: the observation does not age out while the worker idles
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheMarketDataRefresh:
    def test_the_seam_is_re_read_so_the_observation_does_not_age_out(self):
        """A cycle that produces no decision still refreshes the price.

        The symptom: ``last_snapshot_age_seconds = 863.94`` beside HTTP 200.
        The cause was not a broken seam — it was a seam that was **never asked
        again**. The entry path reads a price when a decision exists and the
        exit path reads one when a position is open; a worker with neither read
        nothing, so the stored observation aged out while the diagnostics kept
        showing the successful read of an earlier cycle.
        """
        now = datetime(2026, 11, 10, 8, 0, 0, tzinfo=UTC)
        clock = [now]
        # The venue states the time of *this* request, as Binance does.
        client = _FakeBinance(observed_at=lambda: clock[0])
        seam = BinanceSnapshotMarketData(
            [SYMBOL], client=client, max_age_seconds=15, clock=lambda: clock[0]
        )

        # The first read: the price is current.
        seam.refresh()
        described = seam.describe()
        assert described["market_data_status"] == MARKET_DATA_STATUS_OK
        assert described["is_current_price"] is True
        assert described["last_snapshot_age_seconds"] == pytest.approx(0.0)
        assert client.calls == 1

        # The worker idles for far longer than the age bound.
        clock[0] = now + timedelta(seconds=863.94)

        # The next cycle refreshes it, so the observation describes *now*.
        seam.refresh()
        described = seam.describe()
        assert described["last_snapshot_age_seconds"] == pytest.approx(0.0)
        assert described["market_data_status"] == MARKET_DATA_STATUS_OK
        assert described["is_current_price"] is True
        assert client.calls == 2

    def test_a_refresh_that_never_happened_leaves_the_price_visibly_stale(self):
        """Without the refresh the same clock produces exactly the symptom.

        This is the control for the test above: it proves the staleness came
        from the missing refresh and not from the seam's own arithmetic.
        """
        now = datetime(2026, 11, 10, 8, 0, 0, tzinfo=UTC)
        clock = [now]
        client = _FakeBinance(observed_at=lambda: clock[0])
        seam = BinanceSnapshotMarketData(
            [SYMBOL], client=client, max_age_seconds=15, clock=lambda: clock[0]
        )

        seam.refresh()
        clock[0] = now + timedelta(seconds=863.94)
        # No refresh: the observation is still the one from the first read, so
        # it still carries the *first* read's timestamp.
        described = seam.describe()
        assert described["last_snapshot_age_seconds"] == pytest.approx(863.94)
        assert described["market_data_status"] == MARKET_DATA_STATUS_STALE
        assert described["is_current_price"] is False
        # The diagnostics show the *earlier* successful request. They describe
        # the past, not the present.
        assert client.last_diagnostics == [{"status": 200}]
        assert client.calls == 1

    def test_http_200_alone_does_not_prove_the_price_is_current(self):
        """A successful request is not evidence that its price is fresh.

        The venue answered 200 and the payload carries a timestamp that is well
        past the age bound. The read succeeded; the *price* is stale. So the
        status is ``stale``, the price is not current, and the trade path
        refuses it — the HTTP status never overrides the age.

        The stale observation is **not stored**, so there is no age to report:
        the seam has no usable price at all. That is the fail-closed direction,
        and it is why the age field is ``None`` rather than a number.
        """
        now = datetime(2026, 11, 10, 8, 0, 0, tzinfo=UTC)
        # The venue answers 200 with a price it dated 863 seconds ago.
        client = _FakeBinance(observed_at=now - timedelta(seconds=863.94))
        seam = BinanceSnapshotMarketData(
            [SYMBOL], client=client, max_age_seconds=15, clock=lambda: now
        )

        assert client.last_diagnostics == []  # not read yet
        assert seam.refresh() is False
        described = seam.describe()
        assert client.last_diagnostics == [{"status": 200}]
        assert described["market_data_status"] == MARKET_DATA_STATUS_STALE
        assert described["is_current_price"] is False
        # Nothing usable was stored, so there is no age to report.
        assert described["last_snapshot_age_seconds"] is None
        # And no order can be sized from it.
        with pytest.raises(MarketDataError):
            seam.price(SYMBOL)

    def test_a_failed_refresh_leaves_the_previous_observation_and_reports_it_as_stale(self):
        """A refresh that fails does not make an old price look current.

        The observation is not replaced by a failed read, so the age keeps
        growing and the status keeps saying ``stale``. The seam never raises
        out of ``refresh()``: observing the market must not fail a cycle.
        """
        now = datetime(2026, 11, 10, 8, 0, 0, tzinfo=UTC)
        clock = [now]
        client = _FakeBinance(observed_at=now)
        seam = BinanceSnapshotMarketData(
            [SYMBOL], client=client, max_age_seconds=15, clock=lambda: clock[0]
        )
        seam.refresh()

        # The venue goes away and the clock moves on.
        failing = _FailingBinance()
        seam._client = failing
        clock[0] = now + timedelta(seconds=120)

        assert seam.refresh() is False
        described = seam.describe()
        assert described["last_snapshot_age_seconds"] == pytest.approx(120.0)
        assert described["is_current_price"] is False
        assert described["market_data_status"] != MARKET_DATA_STATUS_OK

    def test_an_unrefreshed_price_cannot_be_traded(self):
        """A stale price is refused at the moment of the order, not just reported.

        The refresh keeps the *view* current; it does not soften the trade
        path. A price past its age bound still raises, so no order is sized
        from it.
        """
        now = datetime(2026, 11, 10, 8, 0, 0, tzinfo=UTC)
        clock = [now]
        client = _FakeBinance(observed_at=now)
        seam = BinanceSnapshotMarketData(
            [SYMBOL], client=client, max_age_seconds=15, clock=lambda: clock[0]
        )
        seam.refresh()

        clock[0] = now + timedelta(seconds=30)
        with pytest.raises(MarketDataError):
            seam.price(SYMBOL)

    def test_a_missing_timestamp_is_refused(self):
        """A ticker with no instant has no age, so it is not a usable price."""
        now = datetime(2026, 11, 10, 8, 0, 0, tzinfo=UTC)
        client = _FakeBinance(observed_at=None)
        seam = BinanceSnapshotMarketData(
            [SYMBOL], client=client, max_age_seconds=15, clock=lambda: now
        )
        assert seam.refresh() is False
        with pytest.raises(MarketDataError):
            seam.price(SYMBOL)

    def test_a_future_timestamp_is_refused(self):
        """A price dated in the future is refused, fail-closed."""
        now = datetime(2026, 11, 10, 8, 0, 0, tzinfo=UTC)
        client = _FakeBinance(observed_at=now + timedelta(seconds=60))
        seam = BinanceSnapshotMarketData(
            [SYMBOL], client=client, max_age_seconds=15, clock=lambda: now
        )
        assert seam.refresh() is False
        with pytest.raises(MarketDataError):
            seam.price(SYMBOL)

    def test_the_worker_cycle_refreshes_the_price_when_nothing_is_decided(self, tmp_path):
        """The production path: an idle worker still observes the market.

        The seam's own ``refresh`` is only half the fix — what matters is that
        the **worker** calls it on a cycle that produced no decision and holds
        no position, because that is exactly the cycle the runtime report
        described. So this drives the real worker over a real cycle.
        """
        now = datetime(2026, 11, 10, 8, 0, 0, tzinfo=UTC)
        clock = [now]
        client = _FakeBinance(observed_at=lambda: clock[0])
        seam = BinanceSnapshotMarketData(
            [SYMBOL], client=client, max_age_seconds=15, clock=lambda: clock[0]
        )
        worker = _worker(tmp_path, routes=_routes(positions=_Resp(200, [])), seam=seam)
        worker.start()

        # The first cycle reads the price.
        worker.run_once()
        described = seam.describe()
        assert described["is_current_price"] is True
        assert client.calls >= 1
        calls_after_first = client.calls

        # The worker idles for far longer than the age bound.
        clock[0] = now + timedelta(seconds=863.94)

        # A cycle with **no decision due** and **no position open** still
        # refreshes the observation.
        worker.run_once()
        described = seam.describe()
        assert client.calls > calls_after_first, "the price was not re-read"
        assert described["last_snapshot_age_seconds"] == pytest.approx(0.0)
        assert described["market_data_status"] == MARKET_DATA_STATUS_OK
        assert described["is_current_price"] is True
        worker.stop()


# ---------------------------------------------------------------------------
# 2. The protection verdict: truthful about a position that does not exist
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
        "submitted_at": "2026-11-10T13:30:00Z",
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


def _routes(*, account=None, positions=None, orders=None):
    return {
        ("GET", "/v2/account"): account or _Resp(200, _ACCOUNT),
        ("GET", "/v2/positions"): positions or _Resp(200, []),
        ("POST", "/v2/orders"): orders or _Resp(200, _venue_order()),
    }


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


def _worker(tmp_path, *, decisions=(), routes=None, calls=None, name="w.json", seam=None):
    """A worker over a mocked venue, with the real broker adapter."""
    from tradingagents.paper_experiment.broker import (
        AlpacaPaperBroker,
        PaperRiskLimits,
        PaperTradingConfig,
        PaperTradingWorker,
    )
    from tradingagents.paper_experiment.broker.trading import StaticMarketData

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
        market_data=seam if seam is not None else StaticMarketData({SYMBOL: 100.0}),
        broker_factory=build,
    )


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


@pytest.mark.unit
class TestTheProtectionVerdict:
    def test_a_worker_with_no_position_does_not_report_an_unprotected_position(self, tmp_path):
        """``UNPROTECTED`` is a claim about an *open* position.

        The reported symptom was ``protection_status = UNPROTECTED`` beside
        ``positions = {}`` and ``protection_checked_at = null``: a position
        that did not exist, reported as unguarded, with no evidence of any
        check. The verdict for "nothing is open" says exactly that instead.
        """
        worker = _worker(tmp_path, routes=_routes(positions=_Resp(200, [])))
        worker.start()
        worker.run_once()

        assert worker.state.positions == {}
        assert worker.state.protection_status == PROTECTION_STATUS_NO_OPEN_POSITIONS
        # The verdict was actually derived, not left at a default.
        assert worker.state.protection_checked_at is not None
        worker.stop()

    def test_a_position_the_broker_holds_is_never_hidden_by_the_no_positions_verdict(self, tmp_path):
        """The empty-verdict state must never swallow a real position.

        The safety direction matters more than the tidiness: a position the
        broker reports open always produces a per-position verdict, and the
        roll-up takes the worst of those. So a position that is open and not
        yet verified is ``PROTECTION_PENDING`` — not ``NO_OPEN_POSITIONS``.
        """
        worker = _worker(
            tmp_path,
            routes=_routes(positions=_Resp(200, [_venue_position()])),
        )
        worker.start()
        worker.run_once()

        assert SYMBOL in worker.state.positions
        assert worker.state.protection_status != PROTECTION_STATUS_NO_OPEN_POSITIONS
        worker.stop()

    def test_the_verdict_is_re_derived_after_every_position_read(self, tmp_path):
        """The verdict follows the positions, not the last thing that happened.

        A worker whose position closed must not keep reporting the verdict it
        had while that position was open.
        """
        # The venue's answer changes: the position is open, then it is closed.
        venue_positions = [_venue_position()]

        class _Changing:
            def __init__(self):
                self.status_code = 200

            @property
            def text(self):
                return json.dumps(venue_positions)

        worker = _worker(
            tmp_path,
            routes=_routes(positions=_Changing()),
        )
        worker.start()
        worker.run_once()
        # The position is open, so the verdict is about an open position.
        assert SYMBOL in worker.state.positions
        assert worker.state.protection_status != PROTECTION_STATUS_NO_OPEN_POSITIONS

        # The broker now reports the account flat.
        venue_positions.clear()
        worker.sync_positions()
        assert worker.state.positions == {}
        assert worker.state.protection_status == PROTECTION_STATUS_NO_OPEN_POSITIONS
        worker.stop()

    def test_an_open_position_that_is_not_verified_is_pending_not_a_confirmed_breach(self, tmp_path):
        """``PENDING`` and ``UNPROTECTED`` are different facts.

        "Not verified yet" is not "verified and found unguarded". The
        distinction is what lets an operator read the report: a pending
        verification says the check has not happened, while an unprotected
        position says it happened and the guard does not work.

        Either way the position is **not** protected, so a new entry is
        refused — the gate reads the per-position verdict, not this summary.
        """
        worker = _worker(
            tmp_path,
            routes=_routes(positions=_Resp(200, [_venue_position()])),
        )
        worker.start()
        worker.run_once()

        # A position the worker opened — it recorded the exit levels — but
        # whose protection was never verified against the broker.
        worker.state.stops[SYMBOL] = {
            "symbol": SYMBOL,
            "stop_loss": 90.0,
            "take_profit": 115.0,
            "decision_id": "d-0",
        }
        worker.state.protections[SYMBOL] = {
            "symbol": SYMBOL,
            "status": PROTECTION_STATUS_PENDING,
            "reason_code": None,
            "detail": "the entry filled; the protection has not been verified yet",
            "quantity": 0.25,
            "checked_at": None,
        }
        worker._roll_up_protection_status()

        assert worker.state.protection_status == PROTECTION_STATUS_PENDING
        assert worker.state.protection_status != PROTECTION_STATUS_PROTECTED
        # And the entry gate still refuses.
        assert worker._open_position_protection_refusal() is not None
        worker.stop()


# ---------------------------------------------------------------------------
# 3. The reconciliation: the order's real status is read back
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheOrderReconciliation:
    def test_an_unsettled_order_is_read_back_even_when_no_decision_is_offered(self, tmp_path):
        """The record is a submit-time snapshot, and the broker is the authority.

        The symptom was ``AVAX/USD accepted, filled_quantity=0`` in the report
        while the account held nothing: the stored record was the answer at the
        moment of submission, and nothing ever asked the venue what became of
        it. So the venue is asked — on a cycle where **no decision is due**,
        which is exactly the case that was not covered.
        """
        calls: list[dict] = []
        routes = _routes(positions=_Resp(200, []))
        routes[("GET", "/v2/orders:by_client_order_id")] = _Resp(
            200,
            _venue_order(
                id="broker-order-avax",
                client_order_id="c-avax",
                symbol="AVAX/USD",
                qty="0.25",
                filled_qty="0.25",
                filled_avg_price="35.0",
                status="filled",
            ),
        )
        worker = _worker(tmp_path, routes=routes, calls=calls)
        worker.start()

        # The worker already knows this order: it was submitted on an earlier
        # cycle and the venue accepted it without filling it.
        worker.state.orders["c-avax"] = {
            "client_order_id": "c-avax",
            "broker_order_id": "broker-order-avax",
            "symbol": "AVAX/USD",
            "side": "buy",
            "quantity": 0.25,
            "filled_quantity": 0.0,
            "filled_avg_price": None,
            "status": "accepted",
            "submitted_at": "2026-11-10T13:30:00+00:00",
        }

        # A cycle with no decision due still reconciles.
        worker.run_once()

        # The venue was asked, and its answer is what is recorded now.
        assert any(
            call["method"] == "GET" and "by_client_order_id" in call["url"]
            for call in calls
        )
        record = worker.state.orders["c-avax"]
        assert record["status"] == "filled"
        assert record["filled_quantity"] == pytest.approx(0.25)
        assert record["filled_avg_price"] == pytest.approx(35.0)
        assert record["broker_order_id"] == "broker-order-avax"
        assert record["client_order_id"] == "c-avax"
        worker.stop()

    def test_reconciliation_never_submits_an_order_again(self, tmp_path):
        """A read is a read. The reconciliation must not place a second order.

        A second submission would double the position when the first one
        fills, so the reconciliation is checked against the request log.
        """
        calls: list[dict] = []
        routes = _routes(positions=_Resp(200, []))
        routes[("GET", "/v2/orders:by_client_order_id")] = _Resp(
            200,
            _venue_order(
                client_order_id="c-avax", status="accepted", filled_qty="0"
            ),
        )
        worker = _worker(tmp_path, routes=routes, calls=calls)
        worker.start()
        worker.state.orders["c-avax"] = {
            "client_order_id": "c-avax",
            "broker_order_id": "broker-order-avax",
            "symbol": "AVAX/USD",
            "side": "buy",
            "quantity": 0.25,
            "filled_quantity": 0.0,
            "filled_avg_price": None,
            "status": "accepted",
            "submitted_at": "2026-11-10T13:30:00+00:00",
        }
        worker.run_once()

        assert [call for call in calls if call["method"] == "POST"] == []
        worker.stop()

    def test_a_terminal_order_is_not_asked_about_again(self, tmp_path):
        """A final answer is final, so re-asking would only spend a request."""
        calls: list[dict] = []
        routes = _routes(positions=_Resp(200, []))
        worker = _worker(tmp_path, routes=routes, calls=calls)
        worker.start()
        worker.state.orders["c-1"] = {
            "client_order_id": "c-1",
            "broker_order_id": "broker-order-1",
            "symbol": SYMBOL,
            "side": "buy",
            "quantity": 0.25,
            "filled_quantity": 0.25,
            "filled_avg_price": 100.0,
            "status": "filled",
            "submitted_at": "2026-11-10T13:30:00+00:00",
        }
        worker.run_once()

        assert not any(
            "by_client_order_id" in call["url"] for call in calls
        ), "a filled order was asked about again"
        worker.stop()

    def test_an_unreadable_venue_leaves_the_record_exactly_as_it_was(self, tmp_path):
        """A failed read invents nothing.

        The stored order keeps its last known state, because that is the truth
        the worker has: it does not know whether the order filled. It is not
        turned into ``filled`` and not turned into ``cancelled``.
        """
        calls: list[dict] = []
        routes = _routes(positions=_Resp(200, []))
        routes[("GET", "/v2/orders:by_client_order_id")] = _Resp(404, {"message": "not found"})
        worker = _worker(tmp_path, routes=routes, calls=calls)
        worker.start()
        worker.state.orders["c-avax"] = {
            "client_order_id": "c-avax",
            "broker_order_id": "broker-order-avax",
            "symbol": "AVAX/USD",
            "side": "buy",
            "quantity": 0.25,
            "filled_quantity": 0.0,
            "filled_avg_price": None,
            "status": "accepted",
            "submitted_at": "2026-11-10T13:30:00+00:00",
        }
        worker.run_once()

        record = worker.state.orders["c-avax"]
        assert record["status"] == "accepted"
        assert record["filled_quantity"] == pytest.approx(0.0)
        # And no order was sent in its place.
        assert [call for call in calls if call["method"] == "POST"] == []
        worker.stop()

    def test_an_accepted_order_is_not_reported_as_a_position(self, tmp_path):
        """``accepted`` is not ``filled``: an unfilled order is not a holding.

        The report paired ``accepted, filled_quantity=0`` with
        ``positions={}``. That pairing was correct — the venue had accepted the
        order without filling it — and the fix is to report the venue's own
        answer, not to promote an accepted order to a filled one.
        """
        calls: list[dict] = []
        routes = _routes(positions=_Resp(200, []))
        routes[("GET", "/v2/orders:by_client_order_id")] = _Resp(
            200,
            _venue_order(
                client_order_id="c-avax",
                status="accepted",
                filled_qty="0",
                filled_avg_price=None,
            ),
        )
        worker = _worker(tmp_path, routes=routes, calls=calls)
        worker.start()
        worker.state.orders["c-avax"] = {
            "client_order_id": "c-avax",
            "broker_order_id": "broker-order-avax",
            "symbol": "AVAX/USD",
            "side": "buy",
            "quantity": 0.25,
            "filled_quantity": 0.0,
            "filled_avg_price": None,
            "status": "accepted",
            "submitted_at": "2026-11-10T13:30:00+00:00",
        }
        worker.run_once()

        record = worker.state.orders["c-avax"]
        assert record["status"] == "accepted"
        assert record["filled_quantity"] == pytest.approx(0.0)
        # The account holds nothing, and the view says so.
        assert worker.state.positions == {}
        worker.stop()

