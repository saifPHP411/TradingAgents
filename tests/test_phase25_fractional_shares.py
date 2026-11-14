"""Phase 25: fractional shares, so a high-priced symbol fits a small notional.

The reported blocker: with whole shares a $25 notional at SPY's ~$580 price floors
to **zero** shares, and the worker raised ``WorkerError: the notional ... buys no
shares``. This phase pins the safety properties around sizing:

- fractional sizing is the default (the one parity constant, ``WHOLE_SHARES``), so
  the worker sizes the same way the TJR backtest does, and a symbol that floors to
  zero under whole shares is refused with a message naming the fix (never a
  zero-quantity order);
- fractional sizing produces the expected quantity;
- the $25 notional cap still binds — fractional sizing never lets an order exceed
  it, including by a rounding artefact;
- shorting stays blocked, and the decision shape is unchanged.
"""

from __future__ import annotations

import pytest

from tradingagents.paper_experiment.broker import (
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    StaticMarketData,
    TradingDecision,
    WorkerError,
)
from tradingagents.risk import WHOLE_SHARES

SPY_PRICE = 580.0


def _worker(tmp_path, *, whole_shares: bool, prices=None, order_notional=25.0,
            limits=None):
    config = PaperTradingConfig(
        order_notional=order_notional,
        whole_shares=whole_shares,
        limits=limits or PaperRiskLimits(),
        state_path=tmp_path / "worker_state.json",
    )
    return PaperTradingWorker(
        config=config,
        decision_source=_NoDecisions(),
        market_data=StaticMarketData(prices or {"SPY": SPY_PRICE}),
    )


def _size(worker, *, notional=25.0, symbol="SPY", side="buy"):
    decision = TradingDecision(
        decision_id="d1", symbol=symbol, side=side, notional=notional
    )
    return worker._size(decision, worker.market_data.price(symbol))  # noqa: SLF001


class _NoDecisions:
    def next_decision(self):
        return None


# ---------------------------------------------------------------------------
# Whole shares: a clear refusal, never a zero-quantity order
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWholeSharesRefusal:
    def test_spy_under_a_small_notional_is_refused_clearly(self, tmp_path):
        worker = _worker(tmp_path, whole_shares=True)
        with pytest.raises(WorkerError) as excinfo:
            _size(worker)
        message = str(excinfo.value)
        assert "no whole shares" in message
        assert "fractional" in message

    def test_the_default_config_matches_the_parity_constant(self):
        """The worker's default sizing is the TJR backtest's, not a second policy."""
        from tradingagents.risk import FRACTIONAL_SHARES

        assert PaperTradingConfig().whole_shares is (not FRACTIONAL_SHARES)

    def test_a_symbol_that_fits_still_sizes_to_whole_shares(self, tmp_path):
        worker = _worker(tmp_path, whole_shares=True, prices={"XYZ": 10.0})
        assert _size(worker, symbol="XYZ") == 2.0


# ---------------------------------------------------------------------------
# Fractional shares: opt-in, correct quantity, within the cap
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFractionalSizing:
    def test_spy_fractional_quantity_is_correct(self, tmp_path):
        worker = _worker(tmp_path, whole_shares=False)
        quantity = _size(worker)
        # Floored to six decimals, so it is the exact ratio truncated, never above it.
        assert quantity == 0.043103
        assert quantity <= 25.0 / SPY_PRICE
        assert quantity > 0

    def test_the_notional_never_exceeds_the_cap(self, tmp_path):
        """The quantity is floored, so the notional cannot round past the cap."""
        worker = _worker(tmp_path, whole_shares=False)
        quantity = _size(worker)
        assert quantity * SPY_PRICE <= 25.0 + 1e-9

    def test_fractional_is_the_default(self, tmp_path):
        worker = _worker(tmp_path, whole_shares=WHOLE_SHARES)
        quantity = _size(worker)
        assert quantity > 0
        assert quantity * SPY_PRICE <= 25.0 + 1e-9

    def test_a_tiny_notional_that_floors_to_zero_is_refused(self, tmp_path):
        worker = _worker(tmp_path, whole_shares=False, order_notional=0.0000001)
        with pytest.raises(WorkerError):
            _size(worker, notional=0.0000001)

    def test_fractional_sizing_does_not_change_the_decision_shape(self, tmp_path):
        worker = _worker(tmp_path, whole_shares=False)
        decision = TradingDecision(
            decision_id="d1", symbol="SPY", side="buy", notional=25.0
        )
        quantity = worker._size(decision, SPY_PRICE)  # noqa: SLF001
        assert decision.to_dict()["notional"] == 25.0
        assert decision.to_dict()["quantity"] is None
        assert quantity > 0


# ---------------------------------------------------------------------------
# The cap and long-only rule are unchanged
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCapsUnchanged:
    def test_the_notional_cap_is_still_25(self):
        assert PaperRiskLimits().max_order_notional == 25.0

    def test_an_order_notional_above_the_cap_is_refused_at_config_time(self, tmp_path):
        with pytest.raises(WorkerError):
            PaperTradingConfig(order_notional=30.0, limits=PaperRiskLimits())

    def test_shorting_is_still_disabled_by_default(self):
        assert PaperRiskLimits().allow_short is False

    def test_a_sell_beyond_the_holding_is_still_refused_by_the_broker_gate(self):
        """The long-only rule lives in the broker's own risk check, unchanged here."""
        from tradingagents.paper_experiment.broker import AlpacaPaperBroker

        broker = AlpacaPaperBroker(api_key="k", api_secret="s")
        # No positions read: a sell against an unknown holding is refused.
        from tradingagents.paper_experiment.broker import PaperRiskRejected

        with pytest.raises(PaperRiskRejected):
            broker._check_sell_is_not_a_short(symbol="SPY", quantity=1.0)  # noqa: SLF001

    def test_the_decision_notional_is_still_capped_by_the_config(self, tmp_path):
        """A decision asking for more than the cap is refused by the broker check."""
        from tradingagents.paper_experiment.broker import AlpacaPaperBroker, PaperRiskRejected

        broker = AlpacaPaperBroker(api_key="k", api_secret="s")
        with pytest.raises(PaperRiskRejected):
            broker._check_risk(  # noqa: SLF001
                symbol="SPY", quantity=1.0, side="buy", reference_price=100.0
            )
