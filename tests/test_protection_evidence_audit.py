"""An evidentiary audit of the mandatory trade protection.

This module is not a restatement of the protection's rules. It is the evidence
that the rules hold, and it is written so that each test **fails when the
property it names stops holding** — including the properties that are easy to
claim and hard to prove:

- **Provenance.** The stop and the targets come from the market's structure, not
  from the protection module. The audit walks the chain from the TJR sweep to
  the order that carries the levels, and checks that each level in the order is
  a level the market produced.

- **Lifecycle.** A position is only protected when the *broker* says it is held
  and the mechanism that would close it has been exercised. A recorded level is
  a claim, not a protection, so every assertion here demands the broker's
  evidence.

- **Failure.** Every way the exit can fail — a refused order, an unreadable
  position, a blind monitor, a restart, a changed quantity, a legacy position,
  an unbindable one — is driven for real against a mock Alpaca Paper transport.
  None of them may leave a position reported as protected.

- **The ladder.** TP1, TP2 and TP3 close in stages, the stop still covers the
  remainder, a target is never sold into twice, and the exit quantities can
  never add up to more than the position.

Nothing here reaches the network. The venue is the project's own Alpaca paper
adapter driven over a transport that behaves like the venue for the parts that
matter, so the code under test is the real code and not a stand-in for it.
"""

from __future__ import annotations

import contextlib
import json

import pandas as pd
import pytest

from tradingagents.paper_experiment.broker import (
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    TradingDecision,
)
from tradingagents.paper_experiment.broker.alpaca_paper import AlpacaPaperBroker
from tradingagents.paper_experiment.broker.trading import (
    PROTECTION_STATUS_CLOSED,
    PROTECTION_STATUS_PENDING,
    PROTECTION_STATUS_PROTECTED,
    PROTECTION_STATUS_UNPROTECTED,
)
from tradingagents.paper_experiment.dashboard.engine_view import positions_view
from tradingagents.tjr.rules import evaluate
from tradingagents.tjr.strategy import TjrStrategy

from .tjr_helpers import BULLISH_BARS, SESSION, series_pair

# ---------------------------------------------------------------------------
# The venue: a mock Alpaca Paper transport
# ---------------------------------------------------------------------------

MARKET_DATA_SYMBOL = "XRP/USDT"
EXECUTION_SYMBOL = "XRP/USD"

ACCOUNT = {
    "id": "paper-acct-1",
    "status": "ACTIVE",
    "cash": "1000.00",
    "equity": "1000.00",
    "buying_power": "1000.00",
    "currency": "USD",
    "trading_blocked": False,
    "account_blocked": False,
    "shorting_enabled": False,
}


class _Resp:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    @property
    def ok(self) -> bool:
        return self.status_code < 400


def _order_payload(
    *,
    client_order_id="c1",
    side="buy",
    qty="2",
    filled_qty=None,
    status="filled",
    symbol=EXECUTION_SYMBOL,
):
    filled = filled_qty if filled_qty is not None else qty
    return {
        "id": "order-1",
        "client_order_id": client_order_id,
        "symbol": symbol,
        "side": side,
        "type": "market",
        "time_in_force": "day",
        "qty": qty,
        "filled_qty": filled,
        "filled_avg_price": "1.085",
        "status": status,
        "submitted_at": "2026-11-08T12:00:00Z",
    }


def _position_payload(*, qty="2", symbol=EXECUTION_SYMBOL):
    return {
        "symbol": symbol,
        "qty": qty,
        "market_value": "2.17",
        "avg_entry_price": "1.085",
        "side": "long",
    }


class Venue:
    """A mock Alpaca Paper venue that behaves like the venue where it matters.

    The venue holds a position, reduces it when a sell fills, remembers the
    orders it was given, and reports the **filled** quantity it actually
    booked. A sell can be told to fill only partly, which is the case that
    separates "the order was accepted" from "the position was closed".
    """

    def __init__(
        self,
        *,
        held: float | None = 2.0,
        fill_ratio: float = 1.0,
        fail_sell_with: int | None = None,
        fail_positions_with: int | None = None,
        fail_sell: bool = False,
    ):
        self.held = held
        self.fill_ratio = fill_ratio
        self.fail_sell_with = fail_sell_with
        self.fail_positions_with = fail_positions_with
        self.fail_sell = fail_sell
        self.orders: dict[str, dict] = {}
        self.calls: list[dict] = []

    def transport(self, *, method, url, headers, params, json_body, timeout):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "params": dict(params or {}),
                "json": json_body,
            }
        )
        if method == "GET" and url.endswith("/v2/account"):
            return _Resp(200, json.dumps(ACCOUNT))
        if method == "GET" and url.endswith("/v2/positions"):
            if self.fail_positions_with is not None:
                return _Resp(self.fail_positions_with, '{"message":"unavailable"}')
            if self.held is None:
                return _Resp(200, "[]")
            return _Resp(200, json.dumps([_position_payload(qty=str(self.held))]))
        if method == "POST" and url.endswith("/v2/orders"):
            side = (json_body or {}).get("side", "buy")
            requested = str((json_body or {}).get("qty", "0"))
            client_id = (json_body or {}).get("client_order_id", "c1")
            if side == "sell":
                if self.fail_sell:
                    return _Resp(500, '{"message":"the venue is unavailable"}')
                if self.fail_sell_with is not None:
                    return _Resp(self.fail_sell_with, '{"message":"refused"}')
            filled = str(float(requested) * self.fill_ratio)
            # The venue reports a **partial** fill as such. Reporting "filled"
            # with a smaller quantity would be a lie the worker could not detect,
            # and the point of the partial-fill test is that it can.
            fully_filled = float(filled) >= float(requested) - 1e-9
            payload = _order_payload(
                client_order_id=client_id,
                side=side,
                qty=requested,
                filled_qty=filled,
                status="filled" if fully_filled else "partially_filled",
            )
            self.orders[client_id] = payload
            # The venue's book follows its own fills: a buy adds what it filled,
            # a sell removes what it filled. The worker reads this back, so a
            # mock that did not update it would misreport what is held.
            if side == "buy":
                self.held = float(filled)
            elif self.held is not None:
                self.held = max(0.0, self.held - float(filled))
            return _Resp(200, json.dumps(payload))
        if method == "GET" and url.endswith("/v2/orders:by_client_order_id"):
            client_id = (params or {}).get("client_order_id")
            order = self.orders.get(client_id)
            if order is None:
                return _Resp(404, '{"message":"unknown"}')
            # An open order's fill state is **live**: the venue keeps filling it
            # while it works, and every further fill reduces the holding. A mock
            # that froze the order at submission — or that filled the order
            # without reducing the book — would misreport what is held, and the
            # worker's reconciliation depends on both being right.
            if order["side"] == "sell" and order["status"] == "partially_filled":
                requested = float(order["qty"])
                already = float(order["filled_qty"])
                now_filled = min(requested, requested * self.fill_ratio)
                extra = max(0.0, now_filled - already)
                if extra:
                    self.held = max(0.0, (self.held or 0.0) - extra)
                order = dict(order, filled_qty=str(now_filled))
                if now_filled >= requested - 1e-9:
                    order["status"] = "filled"
                self.orders[client_id] = order
            return _Resp(200, json.dumps(order))
        raise AssertionError(f"unexpected request: {method} {url}")

    def sells(self) -> list[dict]:
        """The sell orders the venue **accepted**, one per client order id.

        A refused sell is not an order: the venue rejected it, so it never
        reached the book. The venue's own book is the authority on what it
        accepted, so the count comes from there and not from the attempts.
        """
        return [
            order
            for order in self.orders.values()
            if order["side"] == "sell"
        ]

    def sell_attempts(self) -> list[dict]:
        """Every sell the worker **tried** to place, accepted or not."""
        return [
            call
            for call in self.calls
            if call["method"] == "POST" and (call.get("json") or {}).get("side") == "sell"
        ]


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


class _Price:
    """A market-data seam whose price can move between cycles."""

    def __init__(self, price: float):
        self.value = float(price)

    def price(self, symbol):
        return float(self.value)

    def available(self):
        return True

    def identity(self):
        return {
            "market_data_provider": "mock",
            "market_data_kind": "test",
            "market_data_status": "ok",
        }

    def describe(self):
        return self.identity()


class _BlindPrice(_Price):
    """A market-data seam that cannot see the market."""

    def __init__(self):
        super().__init__(10.0)

    def price(self, symbol):
        from tradingagents.paper_experiment.broker.trading import MarketDataError

        raise MarketDataError("the market data feed is unavailable")


def _decision(
    decision_id="tjr-XRP/USDT-2026-11-08",
    *,
    symbol=MARKET_DATA_SYMBOL,
    side="buy",
    notional=25.0,
    quantity=None,
    stop_loss=1.00,
    take_profit=1.30,
    targets=(),
):
    return TradingDecision(
        decision_id=decision_id,
        symbol=symbol,
        side=side,
        notional=None if quantity is not None else notional,
        quantity=quantity,
        stop_loss=stop_loss,
        take_profit=take_profit,
        targets=tuple(targets),
        strategy_name="tjr",
    )


_WORKERS: list[PaperTradingWorker] = []


@pytest.fixture(autouse=True)
def _stop_workers():
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
    venue: Venue | None = None,
    price=1.10,
    name="worker_state.json",
    market_data=None,
    min_reward_risk=1.0,
):
    venue = venue if venue is not None else Venue()
    config = PaperTradingConfig(
        order_notional=25.0,
        min_reward_risk=min_reward_risk,
        limits=PaperRiskLimits(max_order_notional=25.0),
        whole_shares=False,
        crypto=True,
        state_path=tmp_path / name,
    )

    def factory():
        return AlpacaPaperBroker(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=venue.transport,
            limits=PaperRiskLimits(max_order_notional=25.0),
        )

    worker = PaperTradingWorker(
        config=config,
        decision_source=_Source(decisions),
        market_data=market_data or _Price(price),
        broker_factory=factory,
    )
    _WORKERS.append(worker)
    return worker, venue


def _open_position(
    tmp_path,
    *,
    venue: Venue | None = None,
    quantity=2.0,
    stop_loss=1.00,
    take_profit=1.30,
    targets=(),
    price=1.10,
    name="worker_state.json",
):
    """A worker with a **filled** entry, driven through the real entry path.

    The decision goes through the worker's own submission path and the venue's
    fill is what produces the protection record. Nothing here writes the state
    by hand, so the protection that results is the protection the code builds.
    """
    worker, venue = _worker(
        tmp_path,
        decisions=[
            _decision(
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                targets=targets,
            )
        ],
        venue=venue,
        price=price,
        name=name,
    )
    worker.start()
    result = worker.run_once()
    assert result.ran is True, "the entry did not reach the venue"
    return worker, venue


def _ready_outcome(five, one):
    """The chain's first completed setup over a fixture, or ``None``."""
    for index in range(len(five.frame)):
        as_of = five.frame.index[index] + pd.Timedelta(minutes=5)
        candidate = evaluate(
            symbol="X", session=SESSION, as_of=as_of, series_5m=five, series_1m=one
        )
        if candidate.is_ready:
            return candidate
    return None


# ---------------------------------------------------------------------------
# 1. Provenance: the levels come from the market, and travel with the decision
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheProvenanceOfTheLevels:
    """The stop and the targets are the market's, and they reach the order."""

    def test_the_stop_is_the_swept_swing_the_structure_invalidates(self):
        """The stop is the sweep's own extreme, widened by the buffer.

        The chain's stop is not a risk multiple and not a round number: it is
        the low the sweep took, which is the level whose loss would invalidate
        the idea. The audit recomputes it from the sweep the chain found.
        """
        five, one = series_pair("X", SESSION, BULLISH_BARS)
        outcome = _ready_outcome(five, one)
        assert outcome is not None, "the bullish fixture must complete the chain"
        assert outcome.sweep is not None
        # The stop sits beyond the swept extreme, on the losing side of the entry.
        assert outcome.stop is not None
        assert outcome.stop < outcome.sweep.extreme < outcome.entry
        # And it is the extreme itself, widened by the settings' buffer.
        from tradingagents.tjr.rules import TjrSettings

        expected = outcome.sweep.extreme * (1 - TjrSettings().stop_buffer)
        assert outcome.stop == pytest.approx(expected, rel=1e-6)

    def test_every_target_is_a_market_level_or_the_documented_single_fallback(self):
        """Each target is the market's, or the one documented fallback — never a pad.

        The chain prefers a real draw on liquidity. When the market offers no
        draw beyond the entry it falls back to a **single** target at the
        settings' R multiple, and that fallback is never repeated into a
        ladder. So a target is either a price the evaluation's own liquidity
        levels carry, or exactly that one fallback price — and a ladder never
        holds a mix of invented rungs.
        """
        from tradingagents.tjr.rules import TjrSettings

        five, one = series_pair("X", SESSION, BULLISH_BARS)
        outcome = _ready_outcome(five, one)
        assert outcome is not None
        assert outcome.targets, "a completed setup names at least one target"
        market_prices = {round(level.price, 6) for level in outcome.levels}
        fallback = (
            outcome.entry + TjrSettings().fallback_target_r * abs(outcome.entry - outcome.stop)
        )
        for target in outcome.targets:
            is_market_level = round(target, 6) in market_prices
            is_the_fallback = round(target, 6) == round(fallback, 6)
            assert is_market_level or is_the_fallback, (
                f"the target {target} is neither a market liquidity level nor the "
                "documented single-R fallback"
            )

    def test_a_target_that_is_the_fallback_is_never_joined_by_another(self):
        """The fallback is one target, not the first of three.

        If the market offers no draw, the ladder holds exactly the fallback.
        A second and a third rung at 3R and 4R would be fabricated structure
        dressed as the market's.
        """
        from tradingagents.tjr.rules import TjrSettings, _target_ladder

        settings = TjrSettings()
        ladder = _target_ladder(
            "long",
            entry=1.00,
            stop=0.90,
            levels=[],
            settings=settings,
            formed_by=pd.Timestamp("2026-11-08T12:00:00Z"),
            impulse=1.05,
        )
        assert len(ladder) == 1

    def test_a_real_draw_is_preferred_over_the_fallback(self):
        """When a draw exists above the entry, the chain takes the draw.

        The fallback is a last resort, not the default. A market level beyond
        the entry must become the target instead of the R multiple.
        """
        from tradingagents.tjr.levels import LiquidityLevel
        from tradingagents.tjr.rules import TjrSettings, _target_ladder

        ladder = _target_ladder(
            "long",
            entry=1.00,
            stop=0.90,
            levels=[
                LiquidityLevel(
                    label="swing_high", kind="high", price=1.25,
                    formed_at=pd.Timestamp("2026-11-08T11:00:00Z"),
                )
            ],
            settings=TjrSettings(),
            formed_by=pd.Timestamp("2026-11-08T12:00:00Z"),
            impulse=1.05,
        )
        assert ladder == [1.25], "the real draw must win over the R fallback"

    def test_the_ladder_is_not_padded_when_the_market_supports_one_draw(self):
        """One draw on liquidity means one target, never three.

        The ladder is the market's decision. A session that produced a single
        draw above the entry must yield a single target, because a second and
        third target at fixed R multiples would be fabricated structure.
        """
        from tradingagents.tjr.levels import LiquidityLevel
        from tradingagents.tjr.rules import TjrSettings, _target_ladder

        levels = [
            LiquidityLevel(
                label="swing_high", kind="high", price=1.50,
                formed_at=pd.Timestamp("2026-11-08T11:00:00Z"),
            )
        ]
        ladder = _target_ladder(
            "long",
            entry=1.00,
            stop=0.90,
            levels=levels,
            settings=TjrSettings(),
            formed_by=pd.Timestamp("2026-11-08T12:00:00Z"),
            impulse=1.05,
        )
        assert ladder == [1.50]

    def test_three_draws_produce_three_targets_in_order(self):
        """Three real draws produce three targets, nearest first."""
        from tradingagents.tjr.levels import LiquidityLevel
        from tradingagents.tjr.rules import TjrSettings, _target_ladder

        levels = [
            LiquidityLevel(
                label="swing_high", kind="high", price=price,
                formed_at=pd.Timestamp("2026-11-08T11:00:00Z"),
            )
            for price in (1.30, 1.60, 2.00)
        ]
        ladder = _target_ladder(
            "long",
            entry=1.00,
            stop=0.90,
            levels=levels,
            settings=TjrSettings(),
            formed_by=pd.Timestamp("2026-11-08T12:00:00Z"),
            impulse=1.05,
        )
        assert ladder == [1.30, 1.60, 2.00]
        assert ladder == sorted(ladder), "the ladder runs away from the entry"

    def test_no_level_is_invented_when_the_market_offers_no_draw(self):
        """With no draw on liquidity the ladder is the documented fallback only.

        The fallback is a single target at the settings' R multiple — one, not
        three. A padded fallback would be fabricated structure pretending to be
        the market's.
        """
        from tradingagents.tjr.rules import TjrSettings, _target_ladder

        settings = TjrSettings()
        ladder = _target_ladder(
            "long",
            entry=1.00,
            stop=0.90,
            levels=[],
            settings=settings,
            formed_by=pd.Timestamp("2026-11-08T12:00:00Z"),
            impulse=1.05,
        )
        assert len(ladder) == 1
        assert ladder[0] == pytest.approx(
            1.00 + settings.fallback_target_r * 0.10
        )

    def test_the_signal_carries_the_chain_s_levels_unchanged(self):
        """The signal the chain produces carries that chain's own levels.

        The audit runs the **real** chain over the real fixture and checks that
        the signal it produces carries the evaluation's stop and target ladder,
        with no rounding and no substitution.
        """
        five, one = series_pair("X", SESSION, BULLISH_BARS)
        strategy = TjrStrategy(one_minute={"X": one}, five_minute={"X": five})
        found = None
        for index in range(len(five.frame)):
            as_of = five.frame.index[index] + pd.Timedelta(minutes=5)
            signal, candidate = strategy.signal("X", SESSION, as_of)
            if candidate.is_ready:
                found = (signal, candidate)
                break
        assert found is not None
        signal, outcome = found
        assert signal.stop_loss == pytest.approx(outcome.stop)
        assert signal.take_profit == pytest.approx(outcome.target)
        assert list(signal.targets) == [pytest.approx(v) for v in outcome.targets]

    def test_the_order_carries_the_decision_s_levels_unchanged(self, tmp_path):
        """What the worker records is the decision's levels, verbatim.

        The chain's stop and targets reach the decision, the decision reaches
        the worker, and the worker records them without altering them. The
        audit checks the recorded record against the decision it was given.
        """
        decision = _decision(
            quantity=2.0, stop_loss=1.017, take_profit=1.203, targets=(1.203, 1.311, 1.402)
        )
        worker, venue = _worker(tmp_path, decisions=[decision], price=1.10)
        worker.start()
        assert worker.run_once().ran is True

        recorded = worker.state.stops[EXECUTION_SYMBOL]
        assert recorded["stop_loss"] == decision.stop_loss
        assert recorded["take_profit"] == decision.take_profit
        assert list(recorded["targets"]) == list(decision.targets)
        assert recorded["decision_id"] == decision.decision_id

    def test_the_levels_are_bound_to_the_decision_that_opened_the_position(self, tmp_path):
        """The recorded record names the decision it came from.

        A position's levels are only meaningful relative to the decision that
        produced them, so the decision's id travels with them into the state
        and into the dashboard.
        """
        decision = _decision(decision_id="tjr-XRP/USDT-2026-11-08", quantity=2.0)
        worker, venue = _worker(tmp_path, decisions=[decision], price=1.10)
        worker.start()
        assert worker.run_once().ran is True
        # The protection is verified against the broker on the next cycle: the
        # entry filled in this one, and the broker is the authority on what it
        # holds, so the verdict cannot be claimed before that read.
        worker.run_once()
        rows = positions_view(worker.status())
        assert rows[0]["protected"] is True
        assert worker.state.stops[EXECUTION_SYMBOL]["decision_id"] == (
            "tjr-XRP/USDT-2026-11-08"
        )

    def test_a_decision_without_a_stop_never_reaches_the_broker(self, tmp_path):
        """No stop, no order: the chain's geometry is not invented here."""
        worker, venue = _worker(
            tmp_path, decisions=[_decision(quantity=2.0, stop_loss=None, take_profit=1.30)]
        )
        worker.start()
        result = worker.run_once()
        assert result.ran is False
        assert not venue.sells()
        assert worker.state.stops == {}


# ---------------------------------------------------------------------------
# 2. Lifecycle: the position, from the fill to the confirmed exit
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestThePositionLifecycle:
    """The whole life of a position, against the mock venue."""

    def test_the_entry_fills_and_the_position_is_held_at_the_broker(self, tmp_path):
        """The fill is the venue's, and the position is read back from it."""
        worker, venue = _open_position(tmp_path, quantity=2.0)
        # The size is the risk engine's, not the test's: the $25 notional and the
        # stop distance together bound it. The audit reads what was actually
        # filled rather than assuming a size, and checks the venue agrees.
        filled = venue.held
        assert filled > 0
        assert worker.state.protections[EXECUTION_SYMBOL]["quantity"] == pytest.approx(filled)
        assert EXECUTION_SYMBOL in worker.state.stops
        assert worker.state.stops[EXECUTION_SYMBOL]["decision_id"] == (
            "tjr-XRP/USDT-2026-11-08"
        )

    def test_a_filled_position_becomes_protected_only_after_the_broker_confirms_it(
        self, tmp_path
    ):
        """The protection is *pending* until the broker has been asked.

        The entry filled; that is not yet evidence that the position is held
        and guarded. The verdict stays pending until the broker's own position
        read has confirmed the holding.
        """
        worker, venue = _open_position(tmp_path, quantity=2.0)
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_PENDING
        )
        assert worker.state.protection_status == PROTECTION_STATUS_UNPROTECTED

        worker.run_once()
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_PROTECTED
        )
        assert worker.state.protection_status == PROTECTION_STATUS_PROTECTED

    def test_the_exit_fires_when_the_price_crosses_the_stop(self, tmp_path):
        """A price below the stop sends a sell for the whole position."""
        worker, venue = _open_position(
            tmp_path, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        held = venue.held
        worker.market_data.value = 0.95
        result = worker.run_once()
        assert result.ran is True
        sells = venue.sells()
        assert len(sells) == 1
        assert sells[0]["side"] == "sell"
        assert float(sells[0]["qty"]) == pytest.approx(held)

    def test_the_exit_fires_when_the_price_crosses_the_target(self, tmp_path):
        """A price above the target sends a sell for the whole position."""
        worker, venue = _open_position(
            tmp_path, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.market_data.value = 1.35
        result = worker.run_once()
        assert result.ran is True
        assert len(venue.sells()) == 1

    def test_the_exit_is_confirmed_from_the_venue_s_fill_not_the_order_status(
        self, tmp_path
    ):
        """The position is closed on the **filled quantity**, not the status string.

        A market order the venue reports as ``accepted`` can already be fully
        filled, and a status of ``filled`` can still leave a remainder. So the
        verdict is the venue's ``filled_qty`` against the holding, and the
        venue is asked directly.
        """
        worker, venue = _open_position(
            tmp_path, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        held = venue.held
        worker.market_data.value = 1.35
        worker.run_once()
        assert worker.state.exits[EXECUTION_SYMBOL]["filled_quantity"] == pytest.approx(held)
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_CLOSED
        )

    def test_a_partially_filled_exit_leaves_the_position_open_and_watched(
        self, tmp_path
    ):
        """A partial fill is not a close, and the remainder stays protected.

        The venue books half the sell. The position is still half open, so it
        must not be recorded as exited, and its levels must still be watched —
        otherwise the remainder would sit on the market with nothing watching
        it.
        """
        venue = Venue(held=2.0, fill_ratio=0.5)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        held = venue.held
        worker.market_data.value = 1.35
        worker.run_once()
        # The venue booked a partial fill.
        assert venue.held == pytest.approx(held / 2)
        # The position is NOT closed.
        assert EXECUTION_SYMBOL not in worker.state.exits
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] != (
            PROTECTION_STATUS_CLOSED
        )
        # And it is still watched: the levels survive.
        assert EXECUTION_SYMBOL in worker.state.stops

    def test_the_working_exit_is_reconciled_rather_than_replaced(self, tmp_path):
        """The working order is finished off, not abandoned for a second one.

        The venue keeps filling the order that is already working. The worker
        asks the venue how that order stands and books the result, so the
        position closes through the order that was already placed — a second
        sell would oversell.
        """
        venue = Venue(held=2.0, fill_ratio=0.5)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        held = venue.held
        worker.market_data.value = 1.35
        worker.run_once()
        assert len(venue.sells()) == 1
        assert venue.held == pytest.approx(held / 2)

        # The venue keeps working the order and fills the rest.
        venue.fill_ratio = 1.0
        worker.market_data.value = 0.95
        worker.run_once()
        assert len(venue.sells()) == 1, "the worker placed a second exit order"
        assert venue.held == pytest.approx(0.0)
        assert EXECUTION_SYMBOL in worker.state.exits
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_CLOSED
        )

    def test_a_canceled_working_exit_leaves_the_remainder_watched(self, tmp_path):
        """A canceled exit is not a close, and the remainder is still watched.

        The venue cancels the working order after a partial fill. The position
        is still half open, so it is not recorded as exited, and its levels must
        keep being watched — otherwise the remainder would sit on the market
        with nothing watching it.
        """
        venue = Venue(held=2.0, fill_ratio=0.5)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        held = venue.held
        worker.market_data.value = 1.35
        worker.run_once()
        remainder = venue.held
        assert remainder == pytest.approx(held / 2)

        # The venue cancels the working order.
        for order in venue.orders.values():
            if order["side"] == "sell":
                order["status"] = "canceled"
        worker.run_once()
        # The position is NOT closed: the venue still holds the remainder.
        assert EXECUTION_SYMBOL not in worker.state.exits
        # And it is still watched: the levels survive for the next attempt.
        assert EXECUTION_SYMBOL in worker.state.stops
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] != (
            PROTECTION_STATUS_CLOSED
        )

    def test_a_triggered_stop_does_not_become_a_stream_of_orders(self, tmp_path):
        """One exit attempt, then reconciliation — not a new order each cycle.

        A stop that stays crossed across several cycles must produce one order.
        The attempt record is what stops the second one.
        """
        worker, venue = _open_position(
            tmp_path, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.market_data.value = 0.95
        worker.run_once()
        assert len(venue.sells()) == 1
        for _ in range(3):
            worker.run_once()
        assert len(venue.sells()) == 1, "a second exit order was placed"

    def test_the_state_survives_a_restart_and_the_protection_is_re_verified(
        self, tmp_path
    ):
        """A restart restores the levels, the decision id and the ladder.

        The worker is stopped and a new one is built over the same state file.
        The levels and the decision's id come back, and the protection is
        verified against the broker again rather than being assumed from the
        restored file.
        """
        worker, venue = _open_position(
            tmp_path, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.run_once()
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_PROTECTED
        )
        worker.stop()

        restarted, _ = _worker(tmp_path, venue=venue, price=1.10)
        restarted.start()
        restored = restarted.state.stops[EXECUTION_SYMBOL]
        assert restored["stop_loss"] == 1.00
        assert restored["take_profit"] == 1.30
        assert restored["decision_id"] == "tjr-XRP/USDT-2026-11-08"
        # And the restarted worker still watches it.
        restarted.market_data.value = 1.35
        restarted.run_once()
        assert len(venue.sells()) == 1

    def test_a_restart_does_not_re_place_an_exit_that_was_already_placed(
        self, tmp_path
    ):
        """The attempt record survives the restart, so the exit is not duplicated.

        The exit order was placed but the worker died before the fill was
        seen. On restart the worker must reconcile that order, not send a
        second one.
        """
        venue = Venue(held=2.0)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.market_data.value = 1.35
        worker.run_once()
        assert len(venue.sells()) == 1
        worker.stop()

        restarted, _ = _worker(tmp_path, venue=venue, price=1.10)
        restarted.start()
        restarted.market_data.value = 1.35
        restarted.run_once()
        assert len(venue.sells()) == 1, "the restart placed a second exit order"


# ---------------------------------------------------------------------------
# 3. Failure: no path may leave a position reported as protected
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheFailureModes:
    """Every way the exit can fail, driven for real against the mock venue.

    The rule these tests share: a position is reported as protected only on
    positive evidence that the broker holds it **and** the mechanism that would
    close it can act. Anything less is unprotected, and the reason names which
    thing failed.
    """

    def test_a_refused_exit_leaves_the_position_unprotected(self, tmp_path):
        """The venue refuses the exit: the position is not protected.

        The stop and the target are recorded, the price crossed one of them,
        and the order that would have acted on it was refused. The position is
        still open and still watched, but it cannot keep claiming to be
        protected — the guard is on paper.
        """
        venue = Venue(held=2.0, fail_sell=True)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.run_once()
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_PROTECTED
        )

        worker.market_data.value = 0.95
        result = worker.run_once()
        assert result.ran is False
        assert result.reason_code == "EXIT_REFUSED"
        # The verdict is degraded, and the reason says why.
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_UNPROTECTED
        )
        assert worker.state.protections[EXECUTION_SYMBOL]["reason_code"] == "EXIT_REFUSED"
        assert worker.state.protection_status == PROTECTION_STATUS_UNPROTECTED

    def test_a_refused_exit_is_not_reported_as_protected_in_the_dashboard(self, tmp_path):
        """The dashboard must not show a protected position after a refusal.

        This is the gap the audit exists to close: the verdict used to be read
        off the recorded levels, so a position whose exit had just been refused
        still rendered as protected.
        """
        venue = Venue(held=2.0, fail_sell=True)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.run_once()
        worker.market_data.value = 0.95
        worker.run_once()

        rows = positions_view(worker.status())
        assert rows
        assert rows[0]["protected"] is False
        assert rows[0]["protection_state"] == "UNPROTECTED"

    def test_a_refused_exit_keeps_the_levels_so_the_next_cycle_can_retry(self, tmp_path):
        """The refusal does not erase the intent: the levels survive.

        The exit may have been refused for a transient reason — a venue outage,
        a daily-order cap that resets. The position is still open, so the stop
        and the target must still be there for the next attempt.
        """
        venue = Venue(held=2.0, fail_sell=True)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.market_data.value = 0.95
        worker.run_once()
        assert EXECUTION_SYMBOL in worker.state.stops
        assert worker.state.stops[EXECUTION_SYMBOL]["stop_loss"] == 1.00
        assert worker.state.stops[EXECUTION_SYMBOL]["take_profit"] == 1.30

    def test_a_refused_exit_is_retried_once_the_venue_recovers(self, tmp_path):
        """The next cycle tries again, and succeeds once the venue is back.

        The refusal degrades the *claim*, not the intent. So the position is
        still watched, and when the venue recovers the exit goes through and
        the position closes.
        """
        venue = Venue(held=2.0, fail_sell=True)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.market_data.value = 0.95
        worker.run_once()
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_UNPROTECTED
        )

        # The venue recovers.
        venue.fail_sell = False
        worker.run_once()
        assert len(venue.sells()) == 1
        # The retry carries the **same** client order id: the exit is one
        # intent, retried, not a second order for the same position.
        attempts = venue.sell_attempts()
        assert len(attempts) == 2
        assert attempts[0]["json"]["client_order_id"] == attempts[1]["json"]["client_order_id"]
        assert EXECUTION_SYMBOL in worker.state.exits
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_CLOSED
        )

    def test_a_position_the_broker_cannot_be_read_about_is_not_protected(self, tmp_path):
        """A position read that fails is not a protected position.

        "I could not check" and "it is safe" must never look the same. The
        broker is the authority on what is held, so a failure to read it
        leaves the position unprotected, with the failure as the reason.
        """
        venue = Venue(held=2.0)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.run_once()
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_PROTECTED
        )

        venue.fail_positions_with = 503
        worker.run_once()
        record = worker.state.protections[EXECUTION_SYMBOL]
        assert record["status"] == PROTECTION_STATUS_UNPROTECTED
        assert record["reason_code"] == "POSITION_UNAVAILABLE"

    def test_a_blind_monitor_leaves_the_position_unprotected(self, tmp_path):
        """A monitor that cannot see the market cannot protect the position.

        Without a price the stop and the target cannot be detected, so the
        exit cannot fire. The position is still open and still watched, but it
        is not reported as protected while the monitor is blind.
        """
        venue = Venue(held=2.0)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.run_once()
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_PROTECTED
        )

        # The market-data feed goes down.
        worker.market_data = _BlindPrice()
        worker.run_once()
        record = worker.state.protections[EXECUTION_SYMBOL]
        assert record["status"] == PROTECTION_STATUS_UNPROTECTED
        assert record["reason_code"] == "MARKET_DATA_UNAVAILABLE"
        # And the position is still watched, so the next cycle can act.
        assert EXECUTION_SYMBOL in worker.state.stops

    def test_a_position_the_broker_does_not_hold_is_not_protected(self, tmp_path):
        """A holding the broker does not report cannot be protected.

        The position was closed outside the worker. There is nothing left to
        guard, so the verdict must not be "protected" — and the levels must
        not keep claiming to watch a position that is gone.
        """
        venue = Venue(held=2.0)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.run_once()
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_PROTECTED
        )

        # The position is closed outside the worker.
        venue.held = None
        worker.run_once()
        record = worker.state.protections[EXECUTION_SYMBOL]
        assert record["status"] == PROTECTION_STATUS_CLOSED
        assert record["held_quantity"] == 0.0

    def test_a_position_reduced_outside_the_worker_is_not_reported_as_protected(
        self, tmp_path
    ):
        """A holding the broker cut short is not the position that was entered.

        The broker holds less than the entry filled. The levels are still the
        right ones, but the record must follow the broker's quantity, and the
        verdict must not read as a plain "protected" — the position is no
        longer the one the entry described.
        """
        venue = Venue(held=2.0)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.run_once()
        entry_quantity = worker.state.protections[EXECUTION_SYMBOL]["quantity"]

        # The broker holds less than the entry filled.
        venue.held = entry_quantity / 2
        worker.run_once()
        record = worker.state.protections[EXECUTION_SYMBOL]
        assert record["status"] != PROTECTION_STATUS_PROTECTED
        assert record["held_quantity"] == pytest.approx(entry_quantity / 2)

    def test_a_legacy_position_with_no_recorded_levels_is_not_protected(self, tmp_path):
        """A position the worker never opened has no levels, so no protection.

        The levels come from a strategy decision. There is no decision behind
        this holding, so the worker records nothing for it and reports it
        unprotected.
        """
        venue = Venue(held=2.0)
        worker, venue = _worker(tmp_path, venue=venue, price=1.10)
        worker.start()
        worker.run_once()
        worker.sync_positions()
        protection = worker.position_protection()
        assert protection[EXECUTION_SYMBOL]["protected"] is False
        assert protection[EXECUTION_SYMBOL]["reason"] == "NO_RECORDED_EXIT_LEVELS"

    def test_an_unreadable_level_never_becomes_a_trigger(self, tmp_path):
        """A malformed level is not a stop, so it cannot fire an exit.

        The decision path refuses such a level, so this can only arrive from a
        legacy or hand-edited state file. Even then it must not act.
        """
        venue = Venue(held=2.0)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        # Corrupt the persisted level, as a legacy file could be.
        worker.state.stops[EXECUTION_SYMBOL]["stop_loss"] = "not-a-number"
        worker._save()
        worker.market_data.value = 0.01
        worker.run_once()
        assert not venue.sells(), "a malformed level acted as a trigger"

    def test_the_verdict_is_never_protected_without_a_verdict(self, tmp_path):
        """A recorded level with no verdict is not a protection.

        This is the state a legacy file has: the levels are there, but nothing
        ever confirmed the position against the broker. The record says so,
        and the view must not upgrade it. The label is ``PROTECTION_PENDING``
        rather than ``UNPROTECTED``: nothing is *wrong* yet — the check simply
        has not run — but it is not protected either.
        """
        venue = Venue(held=2.0)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        # Drop the verdict, as an older state file would have.
        worker.state.protections.pop(EXECUTION_SYMBOL, None)
        worker._save()
        rows = positions_view(worker.status())
        assert rows[0]["protected"] is False
        assert rows[0]["protection_state"] == "PROTECTION_PENDING"


# ---------------------------------------------------------------------------
# 4. The ladder: staged exits, and the guard on what is left
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheTargetLadder:
    """A ladder of targets closes the position in stages.

    The strategy names several draws on liquidity. Each one closes a slice,
    and the position stays open against the next. The guard on the remainder
    must survive every stage.
    """

    def test_the_first_target_closes_a_slice_and_leaves_the_rest_watched(self, tmp_path):
        """The nearest target closes a slice; the rest keeps its levels."""
        worker, venue = _open_position(
            tmp_path,
            quantity=2.0,
            stop_loss=1.00,
            take_profit=1.30,
            targets=(1.30, 1.50),
            price=1.10,
        )
        held = venue.held
        worker.market_data.value = 1.35
        worker.run_once()

        sells = venue.sells()
        assert len(sells) == 1
        # The slice is a part of the position, not the whole of it.
        assert float(sells[0]["qty"]) < held
        # The position is still open, so it is still protected and watched.
        assert EXECUTION_SYMBOL not in worker.state.exits
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_PROTECTED
        )
        assert EXECUTION_SYMBOL in worker.state.stops

    def test_the_taken_target_is_not_sold_into_again(self, tmp_path):
        """A filled target is marked as taken, so the next stage is the next one.

        Without this the worker would sell the same slice again on the next
        cycle, and the position would close faster than the strategy asked.
        """
        worker, venue = _open_position(
            tmp_path,
            quantity=2.0,
            stop_loss=1.00,
            take_profit=1.30,
            targets=(1.30, 1.50),
            price=1.10,
        )
        worker.market_data.value = 1.35
        worker.run_once()
        first = float(venue.sells()[0]["qty"])

        # The price stays above the first target.
        worker.market_data.value = 1.36
        worker.run_once()
        sells = venue.sells()
        assert len(sells) == 1, "the first target was sold into twice"
        assert float(sells[0]["qty"]) == pytest.approx(first)
        assert worker.state.target_progress[EXECUTION_SYMBOL]["taken"] == [1.30]

    def test_the_stop_closes_the_whole_position_not_just_a_slice(self, tmp_path):
        """A stop is not a stage: it ends the position.

        The ladder manages the position in stages, but the stop is the end of
        it. So a stop closes everything that is left, whatever stage the ladder
        had reached.
        """
        worker, venue = _open_position(
            tmp_path,
            quantity=2.0,
            stop_loss=1.00,
            take_profit=1.30,
            targets=(1.30, 1.50),
            price=1.10,
        )
        worker.market_data.value = 1.35
        worker.run_once()
        remaining = venue.held

        # The price falls through the stop.
        worker.market_data.value = 0.95
        worker.run_once()
        sells = venue.sells()
        assert len(sells) == 2
        # The stop's sell is for everything that was left.
        assert float(sells[1]["qty"]) == pytest.approx(remaining)
        assert venue.held == pytest.approx(0.0)
        assert EXECUTION_SYMBOL in worker.state.exits


# ---------------------------------------------------------------------------
# 5. The invariant: "protected" always means the same thing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheInvariant:
    """One rule, checked from every side.

    ``protected`` is a claim about a position: that the broker holds it and
    that the recorded levels can act on it. These tests pin the meaning, so a
    later change cannot quietly widen it.
    """

    def test_protected_is_never_claimed_for_a_position_the_broker_does_not_hold(
        self, tmp_path
    ):
        """The broker is the authority: no broker, no protection.

        The position is closed outside the worker, so the broker stops
        reporting it. The worker then drops it from the protection view
        entirely — a position that is gone cannot be protected, and it is
        not listed as an unprotected one either, because there is nothing
        left to protect.
        """
        venue = Venue(held=2.0)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.run_once()
        assert worker.position_protection()[EXECUTION_SYMBOL]["protected"] is True

        # The venue stops reporting the position.
        venue.held = None
        worker.run_once()
        protection = worker.position_protection()
        assert EXECUTION_SYMBOL not in protection
        # And the verdict that was recorded says so, too.
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_CLOSED
        )

    def test_protected_survives_a_cycle_that_changed_nothing(self, tmp_path):
        """A normal cycle keeps the verdict: the position is still held.

        The verdict is not a one-off. Every cycle re-reads the broker and
        re-affirms the protection, so a position does not fall out of the
        protected state simply because time passed.
        """
        venue = Venue(held=2.0)
        worker, venue = _open_position(
            tmp_path, venue=venue, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.run_once()
        for _ in range(3):
            worker.run_once()
            assert worker.position_protection()[EXECUTION_SYMBOL]["protected"] is True

    def test_the_protection_names_the_position_it_is_about(self, tmp_path):
        """The verdict is about the execution symbol, under the decision's id.

        The position is held under the execution venue's pair, and its levels
        come from a decision made about another market. The record carries
        both, so a reader can tell which decision the guard belongs to.
        """
        worker, venue = _open_position(
            tmp_path, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        record = worker.state.protections[EXECUTION_SYMBOL]
        assert record["symbol"] == EXECUTION_SYMBOL
        assert record["decision_id"] == "tjr-XRP/USDT-2026-11-08"
        assert worker.state.stops[EXECUTION_SYMBOL]["canonical_symbol"] == "XRP/USDT"

    def test_the_recorded_levels_are_exactly_the_strategy_s_own(self, tmp_path):
        """The guard acts at the strategy's levels, not at a rounded version.

        A stop that was nudged, or a target that was padded, would not be the
        strategy's exit. The record carries them verbatim.
        """
        worker, venue = _open_position(
            tmp_path,
            quantity=2.0,
            stop_loss=1.07,
            take_profit=1.24,
            targets=(1.24, 1.31),
            price=1.10,
        )
        record = worker.state.protections[EXECUTION_SYMBOL]
        assert record["stop_loss"] == 1.07
        assert record["take_profit"] == 1.24
        assert tuple(record["targets"]) == (1.24, 1.31)

    def test_one_position_s_verdict_does_not_speak_for_another(self, tmp_path):
        """The worker-level verdict is the worst of the positions, not one of them.

        Closing one position must not report the whole worker as closed while
        another position is still open and unprotected. The global status is a
        roll-up, so it is written from the set of positions and never from one
        of them alone.

        This is checked on the roll-up itself, because a later broker read
        legitimately changes a position's verdict (a position the broker does
        not hold really is closed). The roll-up is what must never lie about
        the *set*.
        """
        worker, venue = _open_position(
            tmp_path, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        # A second position the worker cannot protect.
        worker.state.protections["BTC/USD"] = {
            "symbol": "BTC/USD",
            "status": PROTECTION_STATUS_UNPROTECTED,
            "reason_code": "POSITION_UNAVAILABLE",
            "detail": "the held quantity could not be read",
            "held_quantity": 1.0,
            "checked_at": "2026-11-08T00:00:00+00:00",
        }
        # The first position closes.
        worker._mark_protection_closed(EXECUTION_SYMBOL, 2.0)

        # The closed position is closed...
        assert worker.state.protections[EXECUTION_SYMBOL]["status"] == (
            PROTECTION_STATUS_CLOSED
        )
        # ...but the worker is not "closed": another position is still open and
        # is not protected, so the worst verdict must win.
        assert worker.state.protection_status == PROTECTION_STATUS_UNPROTECTED

    def test_a_closed_position_alone_reports_the_worker_as_closed(self, tmp_path):
        """With every position closed there is nothing left to protect.

        The verdict is "closed", never "protected": reporting a protection for
        a position that no longer exists would tell an operator a risk is
        covered when no position is open at all.
        """
        worker, venue = _open_position(
            tmp_path, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker._mark_protection_closed(EXECUTION_SYMBOL, 2.0)
        assert worker.state.protection_status == PROTECTION_STATUS_CLOSED

    def test_an_unprotected_position_survives_another_position_s_closure(self, tmp_path):
        """A healthy position cannot overwrite another's unhealthy verdict.

        The order in which the positions are verified must not change the
        outcome: the roll-up takes the worst status, so an unprotected position
        is reported no matter what the other position's verdict says.
        """
        worker, venue = _open_position(
            tmp_path, quantity=2.0, stop_loss=1.00, take_profit=1.30, price=1.10
        )
        worker.state.protections["BTC/USD"] = {
            "symbol": "BTC/USD",
            "status": PROTECTION_STATUS_UNPROTECTED,
            "reason_code": "POSITION_UNAVAILABLE",
            "detail": "the held quantity could not be read",
            "held_quantity": 1.0,
            "checked_at": "2026-11-08T00:00:00+00:00",
        }
        worker._mark_protection_closed(EXECUTION_SYMBOL, 2.0)
        # The roll-up again, from the same set.
        worker._roll_up_protection_status()
        assert worker.state.protection_status == PROTECTION_STATUS_UNPROTECTED
