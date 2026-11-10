"""Phase 36: the TJR target ladder, and the refusals the worker could not explain.

Two defects, both found in the Railway runtime report and both reproduced here
offline, with no network and no credential.

**The ladder.** A long setup whose target ladder had **more than one rung**
raised ``ValidationError`` inside the TJR source, and the source reported it as
``STRATEGY_UNAVAILABLE`` with the bare detail "the TJR chain could not evaluate
AVAX/USDT: ValidationError". The rule chain was fine; the **contract** the
completed setup was written into disagreed with the chain about the order of the
ladder. Every other part of the project — ``TradingDecision.targets``,
``protection.py``, and the existing three-rung ladder test — says "nearest
first", and ``protection.py`` sorts the ladder and reads ``[0]`` as *the nearest
target*. The validator alone required the reverse.

The failure was **symbol-agnostic**: it was the *shape* of the ladder, not
AVAX. A symbol with one target was fine; a symbol with two was refused, which
is why one symbol in a report failed while its neighbours succeeded.

**The refusals.** The worker counted six ``BROKER_REJECTED`` outcomes and could
not say which symbol each one was, which side it was for, or what the venue
said. ``decision_outcomes`` held only the code, and ``last_error`` held only the
*most recent* error, so the earlier refusals were gone by the time an operator
looked.

What these tests pin:

1. a multi-rung ladder produces a **valid** BUY signal — the chain's answer
   reaches the contract instead of being refused by it;
2. the ladder's order is the project's convention: nearest first, ascending for
   a long;
3. a ladder in the *wrong* order is still refused — the fix relaxes the
   contradiction, it does not remove the constraint;
4. a failed evaluation is reported with its **stage** and its **field**, not as
   a bare exception name, and stays a refusal rather than becoming a no-signal;
5. every refused decision leaves a row naming its symbol, its side and the
   venue's own message;
6. that row survives a restart, and is bounded.

The negative controls matter most here: a test that only proves the happy path
would pass against the broken code, so each fix is paired with a test that
fails when the fix is reverted.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingagents.paper_experiment.broker.trading import (
    DECISION_OUTCOME_ORDERED,
    NO_TRADE_BROKER_REJECTED,
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    StaticMarketData,
    WorkerState,
)
from tradingagents.paper_experiment.tjr_diagnostics import (
    STAGE_RULE_CHAIN,
    STAGE_SIGNAL_CONTRACT,
    classify_evaluation_failure,
    format_evaluation_failure,
)
from tradingagents.paper_experiment.tjr_intraday_data import resample_series
from tradingagents.signals.contract import SignalAction, TradingSignal
from tradingagents.tjr.frames import IntradaySeries, session_open_utc
from tradingagents.tjr.strategy import TjrStrategy

from .tjr_helpers import (
    BULLISH_BARS,
    SESSION,
    frame_from_bars,
    one_minute_from_five,
)

SYMBOL = "AVAX/USDT"
OTHER_SYMBOL = "XRP/USDT"

# The history that produces a **multi-rung** ladder.
#
# The rule chain derives its targets from the draws on liquidity that sit above
# the entry and predate the sweep. A single session rarely offers two of them
# above the entry, so the fixture gives the chain:
#
# - the immediately-previous session, whose high (103.5) forms at the previous
#   close and is therefore *before* the sweep;
# - older sessions, whose swing highs (108.0) also predate the sweep;
#
# and the current session carries the canonical bullish structure. The result is
# a ladder of two rungs — [103.5, 108.0], nearest first — which is the exact
# shape that the broken validator refused.
_PRIOR_DAYS = ("2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-26")
_LAST_PRIOR_DAY = "2026-10-27"
_PRIOR_HIGH = 108.0
_LAST_PRIOR_HIGH = 103.5


def _multi_ladder_frames():
    """The (5m, 1m) frames whose chain produces a two-rung ladder."""
    frames = [
        frame_from_bars(day, [(100.0, _PRIOR_HIGH, 99.0, 107.0)] * 20)
        for day in _PRIOR_DAYS
    ]
    frames.append(
        frame_from_bars(_LAST_PRIOR_DAY, [(100.0, _LAST_PRIOR_HIGH, 99.0, 103.0)] * 20)
    )
    frames.append(frame_from_bars(SESSION, BULLISH_BARS))
    five_frame = pd.concat(frames)
    return five_frame, one_minute_from_five(five_frame)


def _as_of(bar_index: int = 15) -> pd.Timestamp:
    """The decision instant at the close of 5m bar ``bar_index``."""
    return session_open_utc(SESSION) + pd.Timedelta(minutes=5 * bar_index)


def _multi_ladder_strategy() -> TjrStrategy:
    five_frame, one_frame = _multi_ladder_frames()
    five = IntradaySeries(SYMBOL, "5m", five_frame)
    one = IntradaySeries(SYMBOL, "1m", one_frame)
    hourly = resample_series(five, "60m")
    four_hour = resample_series(five, "4h")
    return TjrStrategy(
        one_minute={SYMBOL: one},
        five_minute={SYMBOL: five},
        hourly={SYMBOL: hourly} if hourly is not None else None,
        four_hour={SYMBOL: four_hour} if four_hour is not None else None,
    )


# ---------------------------------------------------------------------------
# 1. The multi-rung ladder reaches the contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheLadderReachesTheContract:
    def test_the_chain_derives_more_than_one_target(self):
        """The premise: this fixture really does produce a multi-rung ladder.

        A test that asserted "the signal is valid" would pass on a fixture with
        one target, and would prove nothing about the defect. This pins the
        *shape* that was refused.
        """
        _, evaluation = _multi_ladder_strategy().signal(SYMBOL, SESSION, _as_of())
        assert evaluation.stage == "READY"
        assert len(evaluation.targets) >= 2, evaluation.targets
        assert evaluation.targets == sorted(evaluation.targets), evaluation.targets

    def test_a_multi_rung_ladder_builds_a_valid_buy_signal(self):
        """The regression: the chain's answer reaches the contract.

        Before the fix this raised ``ValidationError`` — "the targets must be
        ordered away from the entry" — because the validator demanded the
        reverse of the order the chain produces.
        """
        signal, evaluation = _multi_ladder_strategy().signal(SYMBOL, SESSION, _as_of())

        assert signal.action is SignalAction.BUY
        assert len(signal.targets) >= 2
        assert signal.targets == evaluation.targets
        # The nearest target is the one the exit policy watches first.
        assert signal.take_profit == signal.targets[0]
        assert signal.take_profit == evaluation.targets[0]

    def test_the_ladder_is_ordered_away_from_the_entry(self):
        """A long's ladder ascends: the nearest target first, the furthest last."""
        signal, _ = _multi_ladder_strategy().signal(SYMBOL, SESSION, _as_of())
        assert signal.targets == sorted(signal.targets)
        assert all(level > signal.entry_price for level in signal.targets)

    def test_the_defect_is_not_specific_to_one_symbol(self):
        """The refusal was about the ladder's *shape*, not about AVAX.

        A symbol with one target was never refused, which is why one symbol in
        the report failed while its neighbours succeeded. The same shape must
        build a signal for every symbol.
        """
        for symbol in (SYMBOL, OTHER_SYMBOL, "BTC/USDT"):
            five_frame, one_frame = _multi_ladder_frames()
            five = IntradaySeries(symbol, "5m", five_frame)
            one = IntradaySeries(symbol, "1m", one_frame)
            hourly = resample_series(five, "60m")
            strategy = TjrStrategy(
                one_minute={symbol: one},
                five_minute={symbol: five},
                hourly={symbol: hourly} if hourly is not None else None,
            )
            signal, _ = strategy.signal(symbol, SESSION, _as_of())
            assert signal.action is SignalAction.BUY, symbol
            assert len(signal.targets) >= 2, symbol


# ---------------------------------------------------------------------------
# 2. The constraint still holds
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheConstraintStillHolds:
    """The fix removes a *contradiction*; it does not remove the constraint."""

    def _signal(self, targets, *, action=SignalAction.BUY):
        entry = 10.0
        stop = 9.0 if action is SignalAction.BUY else 11.0
        return TradingSignal(
            symbol=SYMBOL,
            action=action,
            confidence=0.8,
            timestamp=datetime.now(UTC),
            as_of_date="2026-11-10",
            entry_price=entry,
            stop_loss=stop,
            take_profit=targets[0],
            targets=list(targets),
            holding_period_days=1,
            source="tjr",
        )

    def test_a_longs_nearest_first_ladder_is_accepted(self):
        assert self._signal([12.0, 14.0, 16.0]).targets == [12.0, 14.0, 16.0]

    def test_a_longs_furthest_first_ladder_is_refused(self):
        with pytest.raises(ValueError, match="ordered away from the entry"):
            self._signal([16.0, 14.0, 12.0])

    def test_a_shorts_nearest_first_ladder_is_accepted(self):
        signal = self._signal([8.0, 6.0], action=SignalAction.SELL)
        assert signal.targets == [8.0, 6.0]

    def test_a_shorts_ascending_ladder_is_refused(self):
        with pytest.raises(ValueError, match="ordered away from the entry"):
            self._signal([6.0, 8.0], action=SignalAction.SELL)

    def test_duplicate_targets_are_still_refused(self):
        with pytest.raises(ValueError, match="must be distinct"):
            self._signal([12.0, 12.0])

    def test_a_take_profit_outside_the_ladder_is_still_refused(self):
        with pytest.raises(ValueError, match="must be one of the listed targets"):
            TradingSignal(
                symbol=SYMBOL,
                action=SignalAction.BUY,
                confidence=0.8,
                timestamp=datetime.now(UTC),
                as_of_date="2026-11-10",
                entry_price=10.0,
                stop_loss=9.0,
                take_profit=99.0,
                targets=[12.0, 14.0],
                holding_period_days=1,
                source="tjr",
            )

    def test_the_error_names_the_expected_direction(self):
        """The old message said "ordered away from the entry" and gave no hint.

        That ambiguity is what let the contradiction sit unnoticed, so the
        message now says which direction it wanted.
        """
        with pytest.raises(ValueError, match="ascending"):
            self._signal([16.0, 14.0, 12.0])


# ---------------------------------------------------------------------------
# 3. A failed evaluation says where and what
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheFailureIsDiagnosable:
    def test_a_contract_failure_names_its_stage_and_field(self):
        failure = _validation_failure()
        diagnostics = classify_evaluation_failure(failure)

        assert diagnostics["stage"] == STAGE_SIGNAL_CONTRACT
        assert diagnostics["error_type"] == "ValidationError"
        assert diagnostics["constraint"]
        assert "stop_loss" in diagnostics["constraint"]

    def test_a_contract_failure_names_the_field_that_was_refused(self):
        """A model-level validator reports an empty location, and the field is
        then honestly reported as unknown rather than guessed at."""
        failure = _validation_failure()
        diagnostics = classify_evaluation_failure(failure)

        # The fixture breaks a model-level constraint, so there is no single
        # field to name; the constraint text carries the detail instead.
        assert diagnostics["field"] is None
        assert "stop_loss" in diagnostics["constraint"]

    def test_a_chain_failure_is_told_apart_from_a_contract_failure(self):
        """A rule that raises and a contract that refuses are different faults."""
        diagnostics = classify_evaluation_failure(ZeroDivisionError("division by zero"))
        assert diagnostics["stage"] == STAGE_RULE_CHAIN
        assert diagnostics["error_type"] == "ZeroDivisionError"

    def test_the_detail_names_the_symbol_and_the_stage(self):
        diagnostics = classify_evaluation_failure(_validation_failure())
        detail = format_evaluation_failure(SYMBOL, diagnostics)

        assert SYMBOL in detail
        assert "stage=signal_contract" in detail
        assert "ValidationError" in detail

    def test_the_detail_does_not_carry_the_offending_price(self):
        """The diagnostics are sanitized: no market data leaves the source.

        A pydantic message quotes the value it refused. The detail is for an
        operator, not for a price feed, so the value is not reported.
        """
        failure = _validation_failure()
        diagnostics = classify_evaluation_failure(failure)
        detail = format_evaluation_failure(SYMBOL, diagnostics)

        # The fixture's entry (10.0) and stop (11.0) are the offending values.
        assert "11.0" not in detail
        assert "10.0" not in detail

    def test_a_plain_exception_is_still_described(self):
        """A non-pydantic failure must not crash the diagnostics."""
        diagnostics = classify_evaluation_failure(RuntimeError("the feed vanished"))
        assert diagnostics["stage"] == STAGE_RULE_CHAIN
        assert "the feed vanished" in diagnostics["constraint"]

    def test_an_exception_with_a_broken_error_accessor_is_survivable(self):
        class Broken(Exception):
            def errors(self):
                raise RuntimeError("no errors here")

        diagnostics = classify_evaluation_failure(Broken("boom"))
        assert diagnostics["stage"] == STAGE_RULE_CHAIN
        assert diagnostics["error_type"] == "Broken"


# ---------------------------------------------------------------------------
# 4. Every refusal leaves a row
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEveryRefusalLeavesARow:
    def _worker(self, tmp_path, *, decisions, calls=None):
        config = PaperTradingConfig(
            order_notional=25.0,
            limits=PaperRiskLimits(),
            state_path=tmp_path / "worker_state.json",
        )
        worker = PaperTradingWorker(
            config=config,
            decision_source=_Source(decisions),
            market_data=StaticMarketData({SYMBOL: 10.0, OTHER_SYMBOL: 20.0}),
            broker_factory=lambda: _refusing_broker(calls),
        )
        return worker

    def _decision(self, symbol, side="buy"):
        from tradingagents.paper_experiment.broker.trading import TradingDecision

        # The levels must clear the protection gate for the symbol's own price,
        # or the decision is refused *before* it reaches the venue — and this
        # test is about what happens when the **venue** refuses it.
        price = 10.0 if symbol == SYMBOL else 20.0
        return TradingDecision(
            decision_id=f"tjr-{symbol}-2026-11-10",
            symbol=symbol,
            side=side,
            notional=25.0,
            stop_loss=price * 0.9,
            take_profit=price * 1.2,
        )

    def test_a_refusal_records_its_symbol_side_and_reason(self, tmp_path):
        worker = self._worker(tmp_path, decisions=[self._decision(SYMBOL)])
        worker.start()
        result = worker.run_once()

        assert result.ran is False
        assert result.reason_code == NO_TRADE_BROKER_REJECTED

        rejections = worker.state.rejections
        assert len(rejections) == 1
        row = next(iter(rejections.values()))
        assert row["symbol"] == SYMBOL
        assert row["side"] == "buy"
        assert row["code"] == NO_TRADE_BROKER_REJECTED
        assert row["error_type"] == "BrokerRejectedError"
        assert "HTTP 422" in row["detail"]
        assert row["at"]

    def test_several_refusals_are_told_apart(self, tmp_path):
        """The whole point: a count cannot be acted on, a list can.

        Six refusals that only said ``BROKER_REJECTED`` were the reported
        problem. Six refusals that each name their symbol are a diagnosis.
        """
        worker = self._worker(
            tmp_path,
            decisions=[
                self._decision(SYMBOL),
                self._decision(OTHER_SYMBOL),
            ],
        )
        worker.start()
        worker.run_once()
        worker.run_once()

        symbols = {row["symbol"] for row in worker.state.rejections.values()}
        assert symbols == {SYMBOL, OTHER_SYMBOL}

    def test_a_refusal_records_the_side_it_was_for(self, tmp_path):
        """The side is not recoverable from the decision id, so it is recorded.

        A TJR decision id names the symbol and the session; it says nothing
        about whether the order was to buy or to sell. Without the side, a
        refusal cannot be traced to the order it belongs to.
        """
        worker = self._worker(tmp_path, decisions=[self._decision(SYMBOL)])
        worker.start()
        worker.run_once()

        row = next(iter(worker.state.rejections.values()))
        assert row["side"] == "buy"

    def test_a_refusal_still_places_no_order(self, tmp_path):
        """The refusal is a refusal: the worker records no order.

        The order *reached* the venue — the venue is the one that refused it —
        so the request was made. What must not happen is the worker counting
        that as an order it placed.
        """
        calls = []
        worker = self._worker(tmp_path, decisions=[self._decision(SYMBOL)], calls=calls)
        worker.start()
        worker.run_once()

        assert worker.state.decision_outcomes
        assert all(
            value != DECISION_OUTCOME_ORDERED
            for value in worker.state.decision_outcomes.values()
        )
        assert worker.state.orders == {}

    def test_the_row_survives_a_restart(self, tmp_path):
        worker = self._worker(tmp_path, decisions=[self._decision(SYMBOL)])
        worker.start()
        worker.run_once()

        reloaded = WorkerState.from_dict(json.loads((tmp_path / "worker_state.json").read_text()))
        assert len(reloaded.rejections) == 1
        assert next(iter(reloaded.rejections.values()))["symbol"] == SYMBOL

    def test_an_older_state_file_without_the_field_still_loads(self):
        """The field is additive: an old file reads back, it does not raise."""
        payload = {
            "schema_version": WorkerState().schema_version,
            "strategy_id": "alpaca-paper",
            "status": "STOPPED",
        }
        state = WorkerState.from_dict(payload)
        assert state.rejections == {}

    def test_the_rows_are_bounded(self, tmp_path):
        """A long-running worker must not grow the state file without limit."""
        worker = self._worker(tmp_path, decisions=[])
        worker.state.rejection_limit = 3
        for index in range(10):
            worker._remember_rejection(
                f"tjr-{SYMBOL}-2026-11-{index:02d}",
                symbol=SYMBOL,
                side="buy",
                code=NO_TRADE_BROKER_REJECTED,
                exc=RuntimeError("the venue said no"),
            )
        assert len(worker.state.rejections) == 3
        # The newest survive; the oldest are the ones dropped.
        assert "tjr-AVAX/USDT-2026-11-09" in worker.state.rejections

    def test_the_tally_and_the_rows_agree(self, tmp_path):
        """The count says how many; the rows say which. Neither contradicts."""
        worker = self._worker(
            tmp_path,
            decisions=[self._decision(SYMBOL), self._decision(OTHER_SYMBOL)],
        )
        worker.start()
        worker.run_once()
        worker.run_once()

        statistics = worker.status()["decision_statistics"]
        assert statistics["broker_rejections"] == len(worker.state.rejections)


# ---------------------------------------------------------------------------
# 5. The risk limits are untouched
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheRiskLimitsAreUntouched:
    def test_the_paper_caps_are_unchanged(self):
        """The caps the worker's own broker layer enforces are untouched.

        These are the paper account's limits, read from the same class the
        worker uses, and they are the numbers the task pinned: $25, 5 open
        positions, 10 daily orders, long only.
        """
        limits = PaperRiskLimits()
        assert limits.max_order_notional == 25.0
        assert limits.max_open_positions == 5
        assert limits.max_daily_orders == 10
        assert limits.allow_short is False

    def test_the_risk_engine_caps_are_unchanged(self):
        from tradingagents.risk.paper_parity import (
            MAX_DAILY_ORDERS,
            MAX_OPEN_POSITIONS,
            MAX_ORDER_NOTIONAL,
            paper_parity_limits,
        )

        limits = paper_parity_limits()
        assert MAX_ORDER_NOTIONAL == 25.0
        assert MAX_OPEN_POSITIONS == 5
        assert MAX_DAILY_ORDERS == 10
        assert limits.max_order_value == 25.0
        assert limits.long_only is True

    def test_the_chain_itself_was_not_changed(self):
        """The fix is in the contract, not in the rule chain.

        The chain's own settings and reason codes are what they were, so a
        strategy that behaved one way before still behaves the same way.
        """
        from tradingagents.tjr.rules import TjrSettings

        settings = TjrSettings()
        assert settings.min_reward_risk == 1.0
        assert settings.fallback_target_r == 2.0
        assert settings.allow_short is False if hasattr(settings, "allow_short") else True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _validation_failure():
    """A real pydantic ``ValidationError``, of the kind the source can meet.

    The fixture builds the failure from a constraint that is **still** enforced
    — a stop on the wrong side of the entry — rather than from the ladder
    ordering the fix relaxed. That matters: a fixture built from the relaxed
    constraint would stop raising the moment the fix landed, and the
    diagnostics tests would then be asserting against nothing.

    The failure is a genuine pydantic error, so the diagnostics are exercised
    against the real structure they have to read.
    """
    try:
        TradingSignal(
            symbol=SYMBOL,
            action=SignalAction.BUY,
            confidence=0.8,
            timestamp=datetime.now(UTC),
            as_of_date="2026-11-10",
            entry_price=10.0,
            # A BUY's stop must be BELOW its entry. This one is above it.
            stop_loss=11.0,
            take_profit=12.0,
            targets=[12.0, 14.0],
            holding_period_days=1,
            source="tjr",
        )
    except Exception as exc:  # noqa: BLE001 - the failure is the fixture
        return exc
    raise AssertionError("the fixture must produce a validation failure")


def _refusing_broker(calls=None):
    """A broker whose venue **refuses** every order with a 422.

    The venue serves the account and an empty position book, so the worker gets
    as far as the order and is refused **by the venue** — which is the fault
    the report showed and the one the worker must record.
    """
    from tradingagents.paper_experiment.broker.alpaca_paper import AlpacaPaperBroker

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
            return _Resp(
                422,
                '{"code":42210000,"message":"invalid crypto time_in_force"}',
            )
        if method == "GET" and url.endswith("/v2/account"):
            return _Resp(200, _ACCOUNT)
        if method == "GET" and url.endswith("/v2/positions"):
            return _Resp(200, "[]")
        if method == "GET" and url.endswith("/v2/orders:by_client_order_id"):
            return _Resp(404, '{"code":40410000,"message":"order not found"}')
        raise AssertionError(f"unexpected request: {method} {url}")

    return AlpacaPaperBroker(
        api_key="paper-key",
        api_secret="paper-secret",
        transport=send,
        limits=PaperRiskLimits(),
    )


_ACCOUNT = (
    '{"id":"acct","status":"ACTIVE","currency":"USD",'
    '"buying_power":"10000.00","cash":"10000.00",'
    '"portfolio_value":"10000.00","equity":"10000.00",'
    '"last_equity":"10000.00","multiplier":"1"}'
)
"""The venue's account, as the broker adapter reads it."""


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


class _Resp:
    """A minimal HTTP response, as the broker adapter expects it."""

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text
