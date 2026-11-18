from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from tradingagents.paper_experiment.dashboard.server import Dashboard


@pytest.mark.parametrize(
    ("metric", "value", "expected_gate"),
    [
        ("trades", 12, "minimum_trade_count"),
        ("win_rate_pct", 74.99, "target_win_rate"),
    ],
)
def test_saved_lab_report_recomputes_current_gate_instead_of_trusting_stale_pass_flags(
    tmp_path, metric, value, expected_gate
):
    candidate = "selective_bollinger_rsi_range"
    leverage = 2.0
    profit_factors = {
        "realistic_estimate": 1.30,
        "conservative": 1.10,
        "stress": 1.02,
    }
    report = {
        "excluded_previous_holdout_start": "2026-01-01T00:00:00+00:00",
        "rankings": [
            {
                "candidate": candidate,
                "leverage": leverage,
                # Simulate an old report whose gate was more permissive.
                "development_gate_passed_all_costs": True,
                "minimum_scenario_profit_factor": 1.02,
                "conservative_return_pct": 0.5,
            }
        ],
        "reports": {},
    }
    for scenario, profit_factor in profit_factors.items():
        metrics = {
            "trades": 40,
            "win_rate_pct": 80.0,
            "expectancy_per_trade": 0.25,
            "profit_factor": profit_factor,
            "equal_weight_compounded_return_pct": 0.5,
            "positive_folds": 7,
            "fold_count": 10,
            "positive_symbols": 3,
            "symbol_count": 4,
            "worst_fold_drawdown_pct": 4.0,
            "liquidations": 0,
        }
        metrics[metric] = value
        report["reports"][f"{candidate}/{leverage:g}x/{scenario}"] = {
            "development_only": metrics,
            # Saved gate flags must not override the current policy.
            "acceptance_gate": {"passed": True},
        }

    (tmp_path / "perpetual_strategy_lab.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    dashboard = SimpleNamespace(
        root=tmp_path,
        _perpetual_lab_lock=threading.Lock(),
        _perpetual_lab_status={"status": "idle"},
    )

    result = Dashboard.perpetual_lab_status(dashboard)

    assert result["status"] == "completed"
    assert result["target_win_rate_pct"] == 75.0
    assert "75% net-of-cost win-rate target" in result["selection_rule"]
    candidate_result = result["results"][0]
    assert candidate_result["development_gate_passed_all_costs"] is False
    assert candidate_result["selection_eligible"] is False
    for scenario in ("realistic_estimate", "conservative", "stress"):
        current_gate = candidate_result["scenarios"][scenario]["acceptance_gate"]
        assert current_gate[expected_gate] is False
        assert candidate_result["scenarios"][scenario]["passed"] is False
    assert result["best_candidate"] is None
    assert result["rerun_blocked"] is True
