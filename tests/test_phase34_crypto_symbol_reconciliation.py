"""Phase 34: the crypto symbol reconciliation — one spelling for one instrument.

A Railway deployment reported this, and the pieces did not agree:

* ``AVAX/USD accepted, filled_quantity=0`` and ``XRP/USD accepted,
  filled_quantity=0`` beside ``positions = {}``;
* the Alpaca paper account really held AVAX — about ``1.180856`` units — and
  the worker still reported no positions, no exits and no stops;
* ``protection_status = PROTECTED`` while that position was open.

The cause was a **spelling**, not a fill. Alpaca spells a crypto position
``AVAXUSD``; the order that opened it was sent as ``AVAX/USD``. They name the
same instrument, but as dictionary keys they are different strings, so:

* the broker's position map was keyed ``AVAXUSD`` while every lookup was made
  with ``AVAX/USD``, so the held quantity read as ``0.0`` and the exit could
  never be sized;
* the broker's own "is this a short?" check read the holding as zero, so the
  exit order would have been refused as opening a short;
* the entry gate looked for the open position under the execution spelling,
  found nothing, and allowed a new entry while the first position's guard was
  in doubt;
* the protection roll-up counted the position the worker could not see, so it
  reported ``PROTECTED`` for a position that had no confirmed exit at all.

The fix normalises the venue's spelling **once**, at the boundary where the
position enters the system, so every consumer downstream sees one symbol.

This file also pins the **observation cadence**: the decision cadence (60s) and
the observation cadence (the freshness bound) answer different questions, and
letting the observation ride on the decision cadence left a 15-second freshness
bound permanently unmet.

Everything here is offline and deterministic: the venue transport and the clock
are injected, so no test reaches the network and none needs a venue credential.
The TJR rule chain, the risk limits and the order path are **not** modified
here — these tests pin the *matching around* them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.crypto.types import Ticker
from tradingagents.paper_experiment.broker.alpaca_paper import (
    canonical_position_symbol,
)
from tradingagents.paper_experiment.broker.trading import (
    NO_TRADE_PROTECTION_FAILED,
    PROTECTION_STATUS_PENDING,
    PROTECTION_STATUS_PROTECTED,
    BinanceSnapshotMarketData,
    PaperTradingConfig,
    PaperTradingWorker,
    StaticMarketData,
)

CRYPTO = "AVAX/USD"
VENUE_SPELLING = "AVAXUSD"


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


def _worker(tmp_path, *, decisions=(), routes=None, calls=None, prices=None, name="w.json"):
    """A worker over a mocked venue, with the real broker adapter."""
    from tradingagents.paper_experiment.broker import AlpacaPaperBroker

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
        market_data=StaticMarketData(prices or {CRYPTO: 8.47}),
        broker_factory=build,
    )


# ---------------------------------------------------------------------------
# 1. The canonical spelling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheCanonicalSymbol:
    def test_a_venue_crypto_symbol_is_canonicalised(self):
        """Alpaca's ``AVAXUSD`` is the same instrument as ``AVAX/USD``."""
        assert canonical_position_symbol("AVAXUSD") == CRYPTO
        assert canonical_position_symbol("avaxusd") == CRYPTO
        assert canonical_position_symbol("AVAX/USD") == CRYPTO
        assert canonical_position_symbol("AVAX-USD") == CRYPTO

    def test_an_equity_symbol_is_never_rewritten(self):
        """``AAPL`` is not a crypto pair, so it keeps its own spelling."""
        assert canonical_position_symbol("AAPL") == "AAPL"
        assert canonical_position_symbol("AAPL") == "AAPL"
        assert canonical_position_symbol("BRK.B") == "BRK.B"

    def test_an_unknown_symbol_keeps_the_venue_spelling(self):
        """An unparseable instrument is reported as the venue wrote it."""
        assert canonical_position_symbol("ZZZZ") == "ZZZZ"
        assert canonical_position_symbol("") == ""


# ---------------------------------------------------------------------------
# 2 & 9. The position is found under the execution spelling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestThePositionIsMatched:
    def test_a_venue_position_is_keyed_by_the_canonical_symbol(self, tmp_path):
        """The broker's position map uses one spelling, so every lookup works."""
        worker = _worker(tmp_path)
        worker.start()
        worker.run_once()
        try:
            assert list(worker.state.positions) == [CRYPTO]
            position = worker.state.positions[CRYPTO]
            assert position["quantity"] == pytest.approx(1.180856)
            # The venue's own spelling is kept, so the raw answer stays auditable.
            assert position["venue_symbol"] == VENUE_SPELLING
        finally:
            worker.stop()

    def test_the_held_quantity_is_found_under_the_execution_spelling(self, tmp_path):
        """The exit monitor's lookup finds the holding it must size from."""
        worker = _worker(tmp_path)
        worker.start()
        worker.run_once()
        try:
            assert worker._held_quantity(CRYPTO) == pytest.approx(1.180856)
        finally:
            worker.stop()

    def test_the_broker_sees_the_holding_when_it_checks_for_a_short(self, tmp_path):
        """The broker's own short-check reads the holding, not zero.

        The check asks "is this quantity more than we hold?". With the venue's
        spelling as the key it read ``0.0``, so an exit would have been refused
        as opening a short — the position could never be closed.
        """
        from tradingagents.paper_experiment.broker import AlpacaPaperBroker

        transport = _transport(_routes())
        broker = AlpacaPaperBroker(
            api_key="paper-key", api_secret="paper-secret", transport=transport
        )
        broker.connect()
        broker.positions(refresh=True)
        # A sell of the held quantity is a close, not a short, so it is allowed.
        broker._check_sell_is_not_a_short(symbol=CRYPTO, quantity=1.180856)
        # A sell of more than we hold would open a short, so it is still refused.
        from tradingagents.paper_experiment.broker import PaperRiskRejected

        with pytest.raises(PaperRiskRejected):
            broker._check_sell_is_not_a_short(symbol=CRYPTO, quantity=2.0)

    def test_an_equity_position_keeps_its_symbol(self, tmp_path):
        """The canonicalisation does not touch the equity path."""
        routes = _routes(positions=_Resp(200, [_venue_position(symbol="AAPL", qty="2")]))
        worker = _worker(tmp_path, routes=routes)
        worker.start()
        worker.run_once()
        try:
            assert list(worker.state.positions) == ["AAPL"]
        finally:
            worker.stop()

    def test_the_risk_equity_sees_the_crypto_position(self, tmp_path):
        """The account's equity is not understated by an invisible holding.

        Alpaca keeps the crypto USD cash on the crypto side of the account, so
        ``/v2/account`` reports the cash without the value of the crypto
        position. The RiskEngine therefore marks the holding and adds it to the
        equity itself. That only works if the holding is **visible** in the
        worker's position map — which is exactly what the spelling mismatch
        broke: the position was absent, so the equity was the cash alone and
        every fraction-of-equity check was computed against a halved account.
        """
        from tradingagents.paper_experiment.dashboard.paper_control import (
            _book_snapshot_from_worker,
        )
        from tradingagents.paper_experiment.tjr_source import ledger_from_snapshot

        routes = _routes(
            positions=_Resp(200, [_venue_position(symbol=VENUE_SPELLING, qty="1.180856")])
        )
        worker = _worker(tmp_path, routes=routes)
        worker.start()
        worker.run_once()
        try:
            snapshot = _book_snapshot_from_worker(worker)
            # The holding is visible, under the canonical symbol.
            assert snapshot.positions == {CRYPTO: pytest.approx(1.180856)}
            # ...so the equity the RiskEngine reads includes its value.
            ledger = ledger_from_snapshot(
                snapshot, prices={CRYPTO: 8.47}
            )
            expected = snapshot.cash + 1.180856 * 8.47
            assert ledger.equity == pytest.approx(expected)
            assert ledger.equity > snapshot.cash
        finally:
            worker.stop()

    def test_a_crypto_position_can_actually_be_closed(self, tmp_path):
        """The end-to-end proof: a crypto exit is sized from the holding.

        The spelling mismatch defeated this. The exit monitor asked for the
        held quantity under the execution spelling, the broker's map was keyed
        by the venue spelling, so the read came back as zero — and the venue's
        own short-check then read the holding as zero too, so the close would
        have been refused as opening a short. The position could never be
        closed.
        """
        calls: list[dict] = []
        routes = _routes()
        transport = _transport(routes, calls=calls)
        from tradingagents.paper_experiment.broker import AlpacaPaperBroker

        config = PaperTradingConfig(
            order_notional=25.0,
            state_path=tmp_path / "w.json",
            crypto=True,
        )

        def build():
            return AlpacaPaperBroker(
                api_key="paper-key",
                api_secret="paper-secret",
                transport=transport,
            )

        worker = PaperTradingWorker(
            config=config,
            decision_source=_Source(),
            market_data=StaticMarketData({CRYPTO: 7.90}),
            broker_factory=build,
        )
        worker.start()
        worker.run_once()
        try:
            # The worker opened this position, so it recorded the levels.
            worker.state.stops[CRYPTO] = {"stop_loss": 8.20, "take_profit": 10.0}
            # The price has crossed the stop, so the exit is due.
            result = worker._check_exits()
            assert result is not None, "the stop was crossed, so an exit was due"
            assert result.ran is True, f"the exit was refused: {result.error}"
            # The exit order was placed for the whole holding, under the
            # canonical symbol, as a sell.
            posts = [call for call in calls if call["method"] == "POST"]
            assert len(posts) == 1
            body = posts[0]["json"]
            assert body["symbol"] == CRYPTO
            assert body["side"] == "sell"
            assert float(body["qty"]) == pytest.approx(1.180856)
            # The venue's crypto vocabulary: only ``gtc`` and ``ioc`` are valid.
            assert body["time_in_force"] in ("gtc", "ioc")
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# 5 & 10. The protection verdict and the entry gate
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheProtectionVerdict:
    def test_an_open_position_with_no_record_is_never_reported_as_protected(self, tmp_path):
        """The absence of a verdict is not a verdict of "safe"."""
        worker = _worker(tmp_path)
        worker.start()
        worker.run_once()
        try:
            # One position is open and nothing has confirmed its exit.
            assert list(worker.state.positions) == [CRYPTO]
            worker.state.protections = {}
            worker._roll_up_protection_status()
            assert worker.state.protection_status == PROTECTION_STATUS_PENDING
            assert worker.state.protection_status != PROTECTION_STATUS_PROTECTED
            assert CRYPTO in (worker.state.protection_detail or "")
        finally:
            worker.stop()

    def test_a_protected_position_is_reported_as_protected(self, tmp_path):
        """The healthy case still reports ``PROTECTED``."""
        worker = _worker(tmp_path)
        worker.start()
        worker.run_once()
        try:
            worker.state.protections = {
                CRYPTO: {"symbol": CRYPTO, "status": PROTECTION_STATUS_PROTECTED}
            }
            worker._roll_up_protection_status()
            assert worker.state.protection_status == PROTECTION_STATUS_PROTECTED
        finally:
            worker.stop()

    def test_a_new_entry_is_refused_while_an_open_position_is_unconfirmed(self, tmp_path):
        """An open position with an unconfirmed guard blocks a new entry.

        This is the guard the spelling mismatch defeated: the gate looked for
        the position under the execution spelling, found nothing, and allowed a
        new entry while the first position's exit was unconfirmed.
        """
        worker = _worker(tmp_path)
        worker.start()
        worker.run_once()
        try:
            # The worker opened it, so it recorded its exit levels.
            worker.state.stops[CRYPTO] = {"stop_loss": 7.5, "take_profit": 10.0}
            worker.state.protections[CRYPTO] = {
                "symbol": CRYPTO,
                "status": PROTECTION_STATUS_PENDING,
            }
            worker._roll_up_protection_status()
            refusal = worker._open_position_protection_refusal()
            assert refusal is not None
            assert refusal[0] == NO_TRADE_PROTECTION_FAILED
            assert CRYPTO in refusal[1]
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# 9. The venue's own order list
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheVenueReconciliation:
    def test_an_order_the_worker_does_not_know_about_is_reported(self, tmp_path):
        """An order the venue holds and the worker does not is *reported*.

        It is never submitted, cancelled or adopted: an order this worker did
        not send has no decision behind it, so it cannot be traded against.
        """
        routes = _routes(
            order_list=_Resp(
                200,
                [
                    _venue_order(client_order_id="c-known"),
                    _venue_order(client_order_id="c-unknown", id="broker-order-2"),
                ],
            )
        )
        worker = _worker(tmp_path, routes=routes)
        worker.start()
        worker.run_once()
        try:
            worker.state.orders = {
                "c-known": {
                    "client_order_id": "c-known",
                    "broker_order_id": "broker-order-1",
                    "symbol": CRYPTO,
                    "side": "buy",
                    "quantity": 1.180856,
                    "filled_quantity": 1.180856,
                    "filled_avg_price": 8.46859,
                    "status": "filled",
                    "submitted_at": "2026-11-10T13:30:00Z",
                }
            }
            report = worker.reconcile_with_venue()
            assert report["venue_readable"] is True
            unknown = report["unknown_orders"]
            assert len(unknown) == 1
            assert unknown[0]["client_order_id"] == "c-unknown"
            assert unknown[0]["broker_order_id"] == "broker-order-2"
            assert unknown[0]["status"] == "filled"
            assert unknown[0]["filled_quantity"] == pytest.approx(1.180856)
        finally:
            worker.stop()

    def test_a_status_divergence_is_reported(self, tmp_path):
        """The worker said ``accepted``; the venue says ``filled``."""
        routes = _routes(order_list=_Resp(200, [_venue_order(status="filled")]))
        worker = _worker(tmp_path, routes=routes)
        worker.start()
        worker.run_once()
        try:
            worker.state.orders = {
                "c-1": {
                    "client_order_id": "c-1",
                    "broker_order_id": "broker-order-1",
                    "symbol": CRYPTO,
                    "side": "buy",
                    "quantity": 1.180856,
                    "filled_quantity": 0.0,
                    "filled_avg_price": None,
                    "status": "accepted",
                    "submitted_at": "2026-11-10T13:30:00Z",
                }
            }
            report = worker.reconcile_with_venue()
            assert len(report["status_mismatches"]) == 1
            mismatch = report["status_mismatches"][0]
            assert mismatch["worker_status"] == "accepted"
            assert mismatch["venue_status"] == "filled"
            assert mismatch["filled_quantity"] == pytest.approx(1.180856)
        finally:
            worker.stop()

    def test_an_unreadable_venue_is_reported_as_unreadable(self, tmp_path):
        """A venue that cannot be read is unknown, never "no divergence"."""
        routes = _routes(order_list=_Resp(503, {"message": "unavailable"}))
        worker = _worker(tmp_path, routes=routes)
        worker.start()
        worker.run_once()
        try:
            report = worker.reconcile_with_venue()
            assert report["venue_readable"] is False
            assert report["unknown_orders"] == []
            assert report["detail"]
        finally:
            worker.stop()

    def test_the_reconciliation_never_submits_anything(self, tmp_path):
        """The report is a read: it places no order and cancels none."""
        calls: list[dict] = []
        routes = _routes(
            order_list=_Resp(
                200, [_venue_order(client_order_id="c-unknown", id="broker-order-2")]
            )
        )
        worker = _worker(tmp_path, routes=routes, calls=calls)
        worker.start()
        worker.run_once()
        calls.clear()
        try:
            worker.reconcile_with_venue()
            methods = [call["method"] for call in calls]
            assert "POST" not in methods
            assert "DELETE" not in methods
            assert "GET" in methods
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# 4. The observation cadence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheObservationCadence:
    @staticmethod
    def _refreshable_seam():
        """A current-price seam that supports ``refresh()``, like the real one."""

        class _Client:
            def __init__(self) -> None:
                self.calls = 0
                self.last_diagnostics: list[dict] = []

            def ticker_24h(self, symbols):
                self.calls += 1
                self.last_diagnostics.append({"status": 200})
                now = datetime.now(UTC)
                return {
                    symbol: Ticker.from_binance(
                        {
                            "symbol": symbol,
                            "lastPrice": "8.47",
                            "bidPrice": "8.46",
                            "askPrice": "8.48",
                            "closeTime": int(now.timestamp() * 1000),
                        }
                    )
                    for symbol in symbols
                }

        return BinanceSnapshotMarketData(
            [VENUE_SPELLING], client=_Client(), max_age_seconds=15.0
        )

    @staticmethod
    def _refreshable_client():
        """A Binance client that counts the reads it serves."""
        seam = TestTheObservationCadence._refreshable_seam()
        return seam, seam._client

    def test_the_observation_interval_is_faster_than_the_freshness_bound(self):
        """The observation is refreshed before it can age past the bound.

        With a 15-second bound and a 60-second decision cadence, the reported
        price was stale for 44 seconds out of every minute.
        """
        config = PaperTradingConfig(
            market_data_max_age_seconds=15.0, poll_interval_seconds=60.0
        )
        worker = PaperTradingWorker(
            config=config,
            decision_source=_Source(),
            market_data=StaticMarketData({CRYPTO: 8.47}),
            broker_factory=lambda: None,
        )
        assert worker._observation_interval() == pytest.approx(7.5)
        assert worker._next_interval() == pytest.approx(7.5)

    def test_the_observation_interval_is_never_longer_than_the_decision_cadence(self):
        """The loop waits for whichever comes round first."""
        config = PaperTradingConfig(
            market_data_max_age_seconds=300.0, poll_interval_seconds=60.0
        )
        worker = PaperTradingWorker(
            config=config,
            decision_source=_Source(),
            market_data=StaticMarketData({CRYPTO: 8.47}),
            broker_factory=lambda: None,
        )
        assert worker._observation_interval() == pytest.approx(150.0)
        assert worker._next_interval() == pytest.approx(60.0)

    def test_an_explicit_observation_interval_wins(self):
        config = PaperTradingConfig(
            market_data_max_age_seconds=15.0,
            poll_interval_seconds=60.0,
            observation_interval_seconds=5.0,
        )
        worker = PaperTradingWorker(
            config=config,
            decision_source=_Source(),
            market_data=StaticMarketData({CRYPTO: 8.47}),
            broker_factory=lambda: None,
        )
        assert worker._observation_interval() == pytest.approx(5.0)

    def test_the_observation_interval_never_hammers_the_venue(self):
        """A very tight freshness bound still gets a one-second floor.

        A 0.4-second bound would otherwise mean two requests a second, for
        every symbol, forever. The floor keeps the read rate sane without ever
        letting the observation age past the bound in production, where the
        bound is 15 seconds.
        """
        config = PaperTradingConfig(
            market_data_max_age_seconds=0.4, poll_interval_seconds=60.0
        )
        worker = PaperTradingWorker(
            config=config,
            decision_source=_Source(),
            market_data=StaticMarketData({CRYPTO: 8.47}),
            broker_factory=lambda: None,
        )
        assert worker._observation_interval() == pytest.approx(1.0)

    def test_an_observation_tick_never_produces_a_decision(self):
        """A faster observation is not a reason to trade.

        The observation tick refreshes the price and nothing else: it does not
        ask the decision source, so a symbol that already produced its decision
        is still not asked again.
        """
        source = _Source([])
        config = PaperTradingConfig(
            market_data_max_age_seconds=15.0,
            poll_interval_seconds=60.0,
            observation_interval_seconds=5.0,
        )
        seam = self._refreshable_seam()
        worker = PaperTradingWorker(
            config=config,
            decision_source=source,
            market_data=seam,
            broker_factory=lambda: None,
        )
        worker._refresh_market_data()
        assert source._i == 0  # the source was never asked
        assert worker._observation_due() is False

    def test_the_observation_becomes_due_again_after_its_interval(self):
        config = PaperTradingConfig(
            market_data_max_age_seconds=15.0,
            poll_interval_seconds=60.0,
            observation_interval_seconds=5.0,
        )
        worker = PaperTradingWorker(
            config=config,
            decision_source=_Source(),
            market_data=self._refreshable_seam(),
            broker_factory=lambda: None,
        )
        assert worker._observation_due() is True  # never read yet
        worker._refresh_market_data()
        assert worker._observation_due() is False
        # Age the worker's own knowledge of its last read past the interval.
        worker._last_market_data_at = (
            datetime.now(UTC) - timedelta(seconds=30)
        ).isoformat()
        assert worker._observation_due() is True

    def test_a_seam_without_a_refresh_is_left_alone(self):
        """A seam that cannot be refreshed is not an error; it is just old."""

        class _NoRefresh:
            def describe(self):
                return {}

            def available(self):
                return False

        config = PaperTradingConfig(
            market_data_max_age_seconds=15.0, poll_interval_seconds=60.0
        )
        worker = PaperTradingWorker(
            config=config,
            decision_source=_Source(),
            market_data=_NoRefresh(),
            broker_factory=lambda: None,
        )
        worker._refresh_market_data()  # must not raise
        assert worker._last_market_data_at is None

    def test_the_loop_refreshes_more_often_than_it_evaluates(self):
        """The observation cadence is faster than the decision cadence.

        The symptom this pins: a 15-second freshness bound served by a
        60-second poll, so the reported price was stale for 44 seconds out of
        every minute. The two cadences are separate because they answer
        different questions.
        """
        import threading
        import time

        seam = self._refreshable_seam()
        client = seam._client
        source = _Source()
        config = PaperTradingConfig(
            market_data_max_age_seconds=0.4,
            poll_interval_seconds=30.0,
            state_path=None,
        )
        worker = PaperTradingWorker(
            config=config,
            decision_source=source,
            market_data=seam,
            broker_factory=lambda: None,
        )
        # The decision cadence is far in the future, so every tick in this
        # window is an observation-only tick.
        worker._next_cycle_at = worker._in_seconds(30.0)

        def stop_soon():
            time.sleep(2.5)
            worker._stop_event.set()

        stopper = threading.Thread(target=stop_soon, daemon=True)
        stopper.start()
        worker._loop_body()
        stopper.join(timeout=10)

        # The price was re-read, and the observation is newer than the read
        # that preceded the loop...
        assert client.calls >= 1, "the observation must be refreshed"
        assert worker._last_market_data_at is not None
        # ...while no decision cycle ran and the source was never asked.
        assert worker.state.cycles_completed == 0
        assert source._i == 0

    def test_an_observation_tick_does_not_count_as_a_cycle(self):
        """A refresh is not a decision cycle.

        The cycle counter answers "how many trading decisions were evaluated?".
        An observation tick must not inflate it, or the dashboard would claim
        the worker was evaluating far more often than it was.
        """
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                market_data_max_age_seconds=15.0,
                poll_interval_seconds=60.0,
                observation_interval_seconds=5.0,
            ),
            decision_source=_Source(),
            market_data=self._refreshable_seam(),
            broker_factory=lambda: None,
        )
        before = worker.state.cycles_completed
        worker._refresh_market_data()
        assert worker.state.cycles_completed == before
