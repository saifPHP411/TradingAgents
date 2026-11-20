"""Regression checks for isolated collateral and forced exits on synthetic quotes."""

import numpy as np
from leverage_replay import position, replay


def market(shock=False):
    ts = np.arange(190) * 5000 + 10000
    bid = np.full(190, 100.0)
    ask = bid + 0.01
    if shock:
        bid[2:] -= 3
        ask[2:] -= 3
    depth = np.full(190, 10000.0)
    return ts, bid, ask, depth, depth


def test_forced_exit_consumes_isolated_margin():
    arrays = market(shock=True)
    times, pnl, _, value, leverage, breached = position(
        arrays, 0, len(arrays[0]), 1, 0.001, 50, 50, "50", 0.1, False
    )
    assert breached and times[-1] == arrays[0][2]
    assert np.isclose(-pnl[-1], value / leverage + value * 0.00055)


def test_shared_cash_limits_overlapping_positions():
    arrays = market()
    sig = np.zeros(190, dtype=int)
    sig[0] = 1
    series = arrays, sig, np.full(190, 0.0005), 0.1
    a = replay({"A": series}, 0, "50", False)
    both = replay({"A": series, "B": series}, 0, "50", False)
    assert a["trades"] == 1 and both["max_open"] == both["trades"] == 2
    # The second trade cannot reserve another 60% of the starting account.
    assert 0 < 50 - a["end"] < 50 - both["end"] < 2 * (50 - a["end"])
