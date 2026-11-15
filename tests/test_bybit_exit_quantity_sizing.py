"""Bybit exit sizing must respect venue steps and minimum order notional."""

from types import SimpleNamespace

import pytest

from tradingagents.paper_experiment.broker.trading import PaperTradingWorker


class _InstrumentBroker:
    def __init__(self, *, step=0.0001, min_order_amt=5.0):
        self.instrument = SimpleNamespace(qty_step=step, min_order_amt=min_order_amt)

    def get_instrument(self, symbol):
        assert symbol == "BNB/USDT"
        return self.instrument


def _worker(*, step=0.0001, min_order_amt=5.0):
    worker = object.__new__(PaperTradingWorker)
    worker.config = SimpleNamespace(whole_shares=False)
    worker._broker = _InstrumentBroker(step=step, min_order_amt=min_order_amt)
    return worker


def _exit(worker, *, held, ladder, taken=(), target=1.0, price=1000.0):
    return worker._exit_quantity(
        symbol="BNB/USDT",
        held=held,
        reason="take_profit",
        ladder=ladder,
        taken=list(taken),
        target=target,
        reference_price=price,
    )


@pytest.mark.unit
def test_partial_target_is_floored_to_the_actual_bybit_quantity_step():
    worker = _worker(step=0.0001)
    quantity = _exit(worker, held=0.0153, ladder=[1.0, 2.0, 3.0], price=1000.0)

    assert quantity == pytest.approx(0.0051)
    assert quantity / 0.0001 == pytest.approx(round(quantity / 0.0001))


@pytest.mark.unit
def test_partial_exit_keeps_both_remaining_slices_above_minimum_notional():
    worker = _worker(step=0.0001, min_order_amt=5.0)
    # Split the currently held quantity evenly across the two remaining rungs.
    # Flooring the slice leaves a remainder at least as large as the slice.
    quantity = _exit(
        worker,
        held=0.0102,
        ladder=[1.0, 2.0, 3.0],
        taken=[1.0],
        target=2.0,
        price=1000.0,
    )

    assert quantity == pytest.approx(0.0051)


@pytest.mark.unit
def test_partial_slice_below_minimum_notional_collapses_to_full_close():
    worker = _worker(step=0.0001, min_order_amt=5.0)
    quantity = _exit(worker, held=0.01, ladder=[1.0, 2.0, 3.0], price=1000.0)

    assert quantity == pytest.approx(0.01)


@pytest.mark.unit
def test_last_target_closes_the_entire_remaining_position():
    worker = _worker(step=0.0001)
    quantity = _exit(
        worker,
        held=0.0102,
        ladder=[1.0, 2.0, 3.0],
        taken=[1.0, 2.0],
        target=3.0,
        price=1000.0,
    )

    assert quantity == pytest.approx(0.0102)


@pytest.mark.unit
def test_a_slice_smaller_than_one_quantity_step_never_rounds_up():
    worker = _worker(step=0.0001)
    quantity = _exit(worker, held=0.0002, ladder=[1.0, 2.0, 3.0], price=100000.0)

    # The requested rung is below one step, so it safely becomes a full close.
    assert quantity == pytest.approx(0.0002)
