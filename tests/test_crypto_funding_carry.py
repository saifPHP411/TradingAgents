from tradingagents.crypto.backtest.funding_carry import run_study, simulate_symbol


def _fixture(next_rate=0.001, basis_change=0.0):
    t0 = 1_800_000_000_000
    t1 = t0 + 8 * 3_600_000
    t2 = t1 + 8 * 3_600_000
    return {
        "spot_1h": {
            str(t0 + 3_600_000): {"open": 100.0, "close": 100.0},
            str(t1 + 3_600_000): {"open": 100.0, "close": 100.0},
            str(t2 + 3_600_000): {"open": 100.0, "close": 100.0},
        },
        "perp_1h": {
            str(t0 + 3_600_000): {"open": 100.0, "close": 100.0},
            str(t1 + 3_600_000): {"open": 100.0 - basis_change, "close": 100.0},
            str(t2 + 3_600_000): {"open": 100.0, "close": 100.0},
        },
        "funding": [
            {"timestamp": t0, "rate": 0.001},
            {"timestamp": t1, "rate": next_rate},
            {"timestamp": t2, "rate": next_rate},
        ],
    }


def test_funding_carry_uses_lagged_signal_and_next_realized_payment():
    result = simulate_symbol(
        "BTCUSDT", _fixture(next_rate=0.001),
        annualized_entry_threshold=0.10,
        costs={"commission_bps_per_fill": 0, "half_spread_bps_per_fill": 0,
               "slippage_bps_per_fill": 0},
    )
    assert result["trades"] == 2
    assert result["trade_log"][0]["observed_funding_rate"] == 0.001
    assert result["trade_log"][0]["realized_next_funding_rate"] == 0.001
    assert result["trade_log"][0]["funding_return_pct"] > 0


def test_funding_signal_does_not_use_current_future_funding_to_enter():
    data = _fixture(next_rate=0.01)
    data["funding"][0]["rate"] = 0.0
    result = simulate_symbol(
        "BTCUSDT", data, annualized_entry_threshold=0.10,
        costs={"commission_bps_per_fill": 0, "half_spread_bps_per_fill": 0,
               "slippage_bps_per_fill": 0},
    )
    assert result["trades"] == 1
    assert result["trade_log"][0]["observed_funding_rate"] == 0.01


def test_funding_costs_charge_four_legs_and_can_reverse_small_carry():
    result = simulate_symbol(
        "BTCUSDT", _fixture(next_rate=0.0001),
        annualized_entry_threshold=0.10,
        costs={"commission_bps_per_fill": 10, "half_spread_bps_per_fill": 1,
               "slippage_bps_per_fill": 2},
    )
    assert result["trades"] == 2
    assert result["trade_log"][0]["cost_pct"] == 0.52
    assert result["trade_log"][0]["net_return_pct"] < 0


def test_funding_basis_change_is_included_in_pnl():
    data = _fixture(next_rate=0.001, basis_change=1.0)
    result = simulate_symbol(
        "BTCUSDT", data, annualized_entry_threshold=0.10,
        costs={"commission_bps_per_fill": 0, "half_spread_bps_per_fill": 0,
               "slippage_bps_per_fill": 0},
    )
    assert result["trade_log"][0]["basis_return_pct"] == 1.0


def test_run_study_reports_realistic_and_conservative_and_no_demo():
    result = run_study({"data_source": "synthetic", "symbols": {"BTCUSDT": _fixture()}})
    assert set(result["scenarios"]) == {"realistic", "conservative"}
    assert result["decision"] == "research_only_no_demo_approval"
