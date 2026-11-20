"""Measure net P&L in units of initial stop risk and quantify weekly concentration."""

import json
from pathlib import Path

import pandas as pd
from binance_confirmation import PERIODS as EXTRA_PERIODS
from gated_momentum import PERIODS, build_gates, settings
from relative_weekly import prepare
from weekly import episode


def risk_multiple(trade, atr):
    """Net P&L divided by the initial price risk of the position."""
    return trade["net"] / (trade["qty"] * atr * 1.5)


def main():
    symbols, specs = settings()
    periods = [*PERIODS, *(("confirmation", *period) for period in EXTRA_PERIODS)]
    rows = []
    totals = []
    instrument_index = {symbol: i for i, symbol in enumerate(symbols)}
    for phase, months, anchor, count in periods:
        index, tape, scores = prepare(months, symbols)
        candidates = dict(build_gates(index, tape, scores, symbols.index("BTCUSDT")))
        for week in pd.date_range(anchor, periods=count, freq="7D", tz="UTC"):
            for stress in (False, True):
                trades = []
                result = episode(
                    tape,
                    candidates["strong_signal"],
                    index.get_loc(week),
                    0,
                    50,
                    stress,
                    specs=specs,
                    symbols=symbols,
                    audit=trades,
                    stop_atr=1.5,
                    reward_risk=1.5,
                    max_hold_bars=96,
                )
                common = {"phase": phase, "week": week.date().isoformat(), "stress": stress}
                units = []
                for trade in trades:
                    atr = scores[instrument_index[trade["symbol"]], trade["entry"], -1]
                    multiple = float(risk_multiple(trade, atr))
                    units.append(multiple)
                    rows.append(
                        {
                            **common,
                            "net_r": multiple,
                            "symbol": trade["symbol"],
                            "entry": index[trade["entry"]].isoformat(),
                            "net_usdt": trade["net"],
                        }
                    )
                totals.append(
                    {
                        **common,
                        "end_usdt": result["end"],
                        "trades": len(trades),
                        "net_r": sum(units),
                    }
                )
    detail = pd.DataFrame(rows)
    weeks = pd.DataFrame(totals)
    detail.to_csv("trade-risk-units.csv", index=False)
    weeks.to_csv("weekly-risk-units.csv", index=False)
    summary = {}
    for stress, block in weeks.groupby("stress"):
        label = "stress" if stress else "base"
        best = block.loc[block.net_r.idxmax()]
        residual = block.drop(index=best.name)
        summary[label] = {
            "weeks": len(block),
            "trades": int(block.trades.sum()),
            "net_r_total": float(block.net_r.sum()),
            "best_week": best.week,
            "best_week_net_r": float(best.net_r),
            "net_r_without_best_week": float(residual.net_r.sum()),
            "profitable_weeks": int((block.end_usdt > 50).sum()),
        }
    Path("risk-units-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
