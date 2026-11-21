"""Conservative, auditable statistics for the Gemini desk's modeled shadow closes.

These are *not* exchange fills, account equity, a copy-trading score, or proof
of future profitability. Each settled shadow trade is normalized to equal
notional, and net_pct already includes the desk's stress-cost deduction.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime
from statistics import mean

MIN_TRADES = 30
MIN_DAYS = 3
MIN_WIN_RATE = 0.75
MIN_PROFIT_FACTOR = 1.3
MAX_MODELED_DRAWDOWN = 0.15


def shadow_scorecard(rows: list[dict], *, as_of_ms: int) -> dict:
    """Score *previously settled* Mark-model outcomes without future leakage.

    A null PF for a no-loss sample is deliberate: it cannot estimate the
    severity of losing trades. Modeled drawdown compounds equal-notional
    per-trade returns, not the Bybit account's realized equity.
    """
    if type(as_of_ms) is not int or as_of_ms <= 0:
        raise ValueError("as_of_ms must be a positive integer")
    observations: list[tuple[int, float]] = []
    for row in rows:
        exit_ms = row["exit_ms"]
        net = row["net_pct"]
        if type(exit_ms) is not int or exit_ms <= 0 or exit_ms >= as_of_ms:
            raise ValueError("shadow close must be a prior timestamp")
        if type(net) not in (int, float) or not math.isfinite(net) or net <= -100:
            raise ValueError("shadow net percentage must be finite and above -100")
        observations.append((exit_ms, float(net)))
    observations.sort(key=lambda item: item[0])
    values = [net for _, net in observations]
    count = len(values)
    wins = sum(net > 0 for net in values)
    losses = sum(net < 0 for net in values)
    gross_profit = sum(net for net in values if net > 0)
    gross_loss = -sum(net for net in values if net < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    win_rate = wins / count if count else None
    # Wilson score lower bound, 95% two-sided. Descriptive, not a promise.
    if count:
        z = 1.96
        denominator = 1 + z * z / count
        lower_win_rate = (
            win_rate + z * z / (2 * count)
            - z * math.sqrt(win_rate * (1 - win_rate) / count + z * z / (4 * count * count))
        ) / denominator
    else:
        lower_win_rate = None
    equity = peak = 1.0
    drawdown = 0.0
    for net in values:
        equity *= 1 + net / 100
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak)
    days_observed = len({
        datetime.fromtimestamp(exit_ms / 1000, UTC).date()
        for exit_ms, _ in observations
    })
    expectancy = mean(values) if values else None
    failures = []
    if count < MIN_TRADES:
        failures.append("insufficient_settled_trades")
    if days_observed < MIN_DAYS:
        failures.append("insufficient_independent_days")
    if win_rate is None or win_rate < MIN_WIN_RATE:
        failures.append("win_rate_below_threshold")
    if profit_factor is None:
        failures.append("profit_factor_undefined_without_losses")
    elif profit_factor < MIN_PROFIT_FACTOR:
        failures.append("profit_factor_below_threshold")
    if expectancy is None or expectancy <= 0:
        failures.append("nonpositive_net_expectancy")
    if drawdown > MAX_MODELED_DRAWDOWN:
        failures.append("modeled_drawdown_exceeds_threshold")
    return {
        "source": "settled_shadow_mark_model_not_venue_fills",
        "settled_trades": count,
        "winning_trades": wins,
        "losing_trades": losses,
        "days_observed": days_observed,
        "win_rate": win_rate,
        "win_rate_lower_95": lower_win_rate,
        "profit_factor": profit_factor,
        "net_expectancy_pct_per_trade": expectancy,
        "modeled_max_drawdown_fraction": drawdown,
        "modeled_equal_notional_compounded_return": equity - 1,
        "eligible_for_demo": not failures,
        "failed_gates": failures,
        "thresholds": {
            "min_settled_trades": MIN_TRADES,
            "min_distinct_utc_days": MIN_DAYS,
            "min_win_rate": MIN_WIN_RATE,
            "min_profit_factor": MIN_PROFIT_FACTOR,
            "max_modeled_drawdown_fraction": MAX_MODELED_DRAWDOWN,
            "positive_net_expectancy_required": True,
        },
    }
