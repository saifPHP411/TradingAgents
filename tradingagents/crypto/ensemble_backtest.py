"""Conservative, next-bar-open backtest for the research ensemble.

This is a single-symbol research harness, not a portfolio simulator and not wired
to any broker. Signals use completed bars only; entries/exits execute at the next
bar open, stops are checked intrabar with gap-aware fills, and both sides pay fees
and slippage. Results must not be treated as live performance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import pandas as pd

from .ensemble import (
    EnsembleSettings,
    _evaluate_indicator_row,
    _indicators,
    _validate_settings,
)


@dataclass(frozen=True)
class BacktestSettings:
    initial_cash: float = 10_000.0
    risk_fraction: float = 0.005
    max_notional_fraction: float = 0.10
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    warmup_bars: int = 400
    max_hold_bars: int = 168
    trailing_stop_atr_multiple: float = 3.0


@dataclass
class BacktestResult:
    mode: str
    settings: dict
    final_equity: float
    net_return_pct: float
    max_drawdown_pct: float
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    profit_factor: float | None
    gross_profit: float
    gross_loss: float
    net_profit: float
    net_loss: float
    fees_paid: float
    slippage_paid: float
    slippage_bps: float
    equity_curve: list[dict]
    trade_log: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _validate(frame: pd.DataFrame, settings: BacktestSettings) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    if required.difference(frame.columns):
        raise ValueError(f"missing OHLCV columns: {sorted(required.difference(frame.columns))}")
    if len(frame) < settings.warmup_bars + 2:
        raise ValueError("not enough bars for warm-up and next-bar execution")
    data = frame.loc[:, ["open", "high", "low", "close", "volume"]].astype(float).copy()
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None:
        raise ValueError("backtest bars require a timezone-aware DatetimeIndex")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("bar index must be strictly increasing and unique")
    intervals = data.index.to_series().diff().dropna()
    if not (intervals == pd.Timedelta(hours=1)).all():
        raise ValueError("backtest requires contiguous 1h bars without gaps")
    if not all(isfinite(float(v)) for v in data.to_numpy().ravel()):
        raise ValueError("OHLCV values must be finite")
    if (data[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("prices must be positive")
    if (data["volume"] < 0).any():
        raise ValueError("volume must be non-negative")
    if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("invalid high in OHLC data")
    if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("invalid low in OHLC data")
    return data


def _stop_risk_per_unit(entry: float, stop: float, fee_rate: float, slippage_rate: float) -> float:
    """Estimate stop loss per unit after both fees and adverse exit slippage."""
    stop_fill = stop * (1.0 - slippage_rate)
    return max(entry * (1.0 + fee_rate) - stop_fill * (1.0 - fee_rate), 0.0)


def run_backtest(
    frame: pd.DataFrame,
    *,
    mode: str = "ensemble",
    signal_settings: EnsembleSettings | None = None,
    settings: BacktestSettings = BacktestSettings(),
    btc_frame: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run one symbol; each signal is acted on at the following bar's open."""
    if mode not in {"ensemble", "trend", "range", "breakout", "trend_pullback"}:
        raise ValueError("unsupported strategy mode")
    if type(settings.max_hold_bars) is not int or settings.max_hold_bars < 1:
        raise ValueError("max hold bars must be a positive integer")
    if type(settings.warmup_bars) is not int or settings.warmup_bars < 220:
        raise ValueError("warm-up must be an integer of at least 220 bars")
    numeric_settings = (
        settings.initial_cash, settings.risk_fraction, settings.max_notional_fraction,
        settings.fee_bps, settings.slippage_bps,
        settings.trailing_stop_atr_multiple,
    )
    if not all(isfinite(float(value)) for value in numeric_settings):
        raise ValueError("backtest settings must be finite numbers")
    if settings.initial_cash <= 0 or not 0 < settings.risk_fraction <= 0.05:
        raise ValueError("initial cash and risk fraction are outside safe bounds")
    if not 0 < settings.max_notional_fraction <= 1:
        raise ValueError("max notional fraction must be in (0, 1]")
    if settings.fee_bps < 0 or settings.slippage_bps < 0:
        raise ValueError("fees and slippage cannot be negative")
    if settings.trailing_stop_atr_multiple <= 0:
        raise ValueError("trailing stop ATR multiple must be positive")

    signal_settings = signal_settings or EnsembleSettings()
    _validate_settings(signal_settings)
    data = _validate(frame, settings)
    signal_indicators = _indicators(data, signal_settings)
    btc_indicators = None
    if btc_frame is not None:
        btc_data = _validate(btc_frame, settings)
        btc_indicators = _indicators(btc_data, signal_settings)
    cash = float(settings.initial_cash)
    position = None
    pending = None
    trades: list[dict] = []
    curve: list[dict] = []
    total_fees = 0.0
    total_slippage = 0.0
    cooldown_until = -1
    slip = settings.slippage_bps / 10_000.0
    fee_rate = settings.fee_bps / 10_000.0

    def close_position(timestamp, raw_price: float, reason: str) -> None:
        nonlocal cash, position, total_fees, total_slippage
        if position is None:
            return
        fill = raw_price * (1.0 - slip)
        fee = position["qty"] * fill * fee_rate
        exit_slippage = position["qty"] * (raw_price - fill)
        total_slippage += exit_slippage
        cash += position["qty"] * fill - fee
        total_fees += fee
        pnl = (
            position["qty"] * (fill - position["entry"])
            - position["entry_fee"]
            - fee
        )
        trades.append({
            "entry_time": position["entry_time"],
            "exit_time": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
            "entry_price": position["entry"],
            "raw_entry_price": position["raw_entry"],
            "exit_price": fill,
            "raw_exit_price": raw_price,
            "quantity": position["qty"],
            "entry_equity": position["entry_equity"],
            "estimated_stop_risk": position["estimated_stop_risk"],
            "initial_stop": position["initial_stop"],
            "final_stop": position["stop"],
            "gross_pnl": position["qty"] * (raw_price - position["raw_entry"]),
            "net_pnl": pnl,
            "entry_fee": position["entry_fee"],
            "exit_fee": fee,
            "slippage_paid": position["entry_slippage"] + exit_slippage,
            "reason": reason,
        })
        position = None

    for j in range(settings.warmup_bars, len(data)):
        row = data.iloc[j]
        timestamp = data.index[j]

        # Pending actions come from the previous completed bar.
        if pending is not None:
            if pending["action"] == "exit" and position is not None:
                close_position(timestamp, float(row.open), pending["reason"])
                cooldown_until = j + 1
            elif pending["action"] == "entry" and position is None:
                entry = float(row.open) * (1.0 + slip)
                stop = float(pending["stop"])
                equity_at_open = cash
                risk_per_unit = _stop_risk_per_unit(entry, stop, fee_rate, slip)
                if stop > 0 and risk_per_unit > 0 and equity_at_open > 0:
                    risk_budget = equity_at_open * settings.risk_fraction
                    notional_cap = equity_at_open * settings.max_notional_fraction
                    qty = min(risk_budget / risk_per_unit, notional_cap / entry)
                    if qty > 0 and isfinite(qty):
                        entry_fee = qty * entry * fee_rate
                        entry_slippage = qty * (entry - float(row.open))
                        total_slippage += entry_slippage
                        cash -= qty * entry + entry_fee
                        total_fees += entry_fee
                        position = {
                            "entry": entry,
                            "raw_entry": float(row.open),
                            "entry_time": timestamp.isoformat(),
                            "entry_bar": j,
                            "qty": qty,
                            "stop": stop,
                            "initial_stop": stop,
                            "highest_high": max(entry, float(row.high)),
                            "entry_fee": entry_fee,
                            "entry_slippage": entry_slippage,
                            "entry_equity": equity_at_open,
                            "estimated_stop_risk": qty * risk_per_unit,
                        }
            pending = None

        # BTC is a fail-closed risk overlay for altcoins. Its state is read only
        # at the same completed timestamp; missing BTC bars block new risk.
        btc_risk_off = False
        if btc_indicators is not None:
            btc_position = btc_indicators.index.get_indexer([timestamp])[0]
            if btc_position < 0:
                btc_risk_off = True
            else:
                btc_row = btc_indicators.iloc[btc_position]
                btc_required = ("close", "fast_ema", "slow_ema", "momentum",
                                "plus_di", "minus_di", "adx")
                btc_risk_off = any(pd.isna(btc_row[key]) for key in btc_required) or not (
                    btc_row.close > btc_row.slow_ema
                    and btc_row.fast_ema > btc_row.slow_ema
                    and btc_row.momentum > 0
                    and btc_row.plus_di > btc_row.minus_di
                    and btc_row.adx >= signal_settings.adx_trend_min
                )

        # Intrabar stop: if the bar gaps through the stop, use its open.
        if position is not None and float(row.low) <= position["stop"]:
            stop_fill = min(float(row.open), position["stop"])
            close_position(timestamp, stop_fill, "atr_stop")
            cooldown_until = j + 1

        # Tighten the Chandelier-style stop after this bar's stop check.
        # The updated level is active only from the next candle onward.
        if position is not None:
            position["highest_high"] = max(position["highest_high"], float(row.high))
            current_atr = float(signal_indicators.atr.iloc[j])
            if isfinite(current_atr) and current_atr > 0:
                trail = position["highest_high"] - settings.trailing_stop_atr_multiple * current_atr
                position["stop"] = max(position["stop"], trail)

        # Count the entry candle as the first completed holding bar.
        time_stop_due = position is not None and j - position["entry_bar"] + 1 >= settings.max_hold_bars

        # Mark-to-market equity at this bar's close.
        equity = cash + (position["qty"] * float(row.close) if position else 0.0)
        curve.append({"time": timestamp.isoformat(), "equity": equity})

        if j == len(data) - 1:
            if position is not None:
                close_position(timestamp, float(row.close), "end_of_data")
                curve[-1]["equity"] = cash
            break

        # Only completed bars through j are visible to the signal function.
        signal = _evaluate_indicator_row(signal_indicators, j, signal_settings, mode)
        if time_stop_due and position is not None:
            pending = {"action": "exit", "reason": "max_hold_bars"}
        elif btc_risk_off and position is not None:
            pending = {"action": "exit", "reason": "btc_risk_filter"}
        elif position is not None and signal.should_exit:
            pending = {"action": "exit", "reason": "signal_flat"}
        elif position is None and j > cooldown_until and not btc_risk_off and signal.action == "long" and signal.stop is not None:
            pending = {"action": "entry", "stop": signal.stop}

    # If a position remains after the loop, close it at the last available close.
    if position is not None:
        close_position(data.index[-1], float(data.close.iloc[-1]), "end_of_data")
        if curve:
            curve[-1]["equity"] = cash

    equity_values = [point["equity"] for point in curve]
    peak = settings.initial_cash
    max_dd = 0.0
    for value in equity_values:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)

    pnls = [trade["net_pnl"] for trade in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_pnls = [trade["gross_pnl"] for trade in trades]
    gross_profit = sum(value for value in gross_pnls if value > 0)
    gross_loss = abs(sum(value for value in gross_pnls if value < 0))
    net_profit = sum(wins)
    net_loss = abs(sum(losses))
    profit_factor = net_profit / net_loss if net_loss > 0 else None
    final_equity = cash
    return BacktestResult(
        mode=mode,
        settings={**asdict(settings), "signal_settings": asdict(signal_settings)},
        final_equity=final_equity,
        net_return_pct=(final_equity / settings.initial_cash - 1.0) * 100.0,
        max_drawdown_pct=max_dd * 100.0,
        trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=(len(wins) / len(trades) * 100.0) if trades else 0.0,
        profit_factor=profit_factor,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
        net_loss=net_loss,
        fees_paid=total_fees,
        slippage_paid=total_slippage,
        slippage_bps=settings.slippage_bps,
        equity_curve=curve,
        trade_log=trades,
    )


def compare_modes(
    frame: pd.DataFrame,
    *,
    signal_settings: EnsembleSettings | None = None,
    settings: BacktestSettings = BacktestSettings(),
    btc_frame: pd.DataFrame | None = None,
) -> dict:
    """Run four comparable variants on identical bars and execution assumptions."""
    signal_settings = signal_settings or EnsembleSettings()
    modes = ("trend", "trend_pullback", "range", "breakout", "ensemble")
    return {
        "schema_version": 1,
        "kind": "crypto_regime_ensemble_comparison",
        "bar_count": len(frame),
        "first_bar": frame.index[0].isoformat(),
        "last_bar": frame.index[-1].isoformat(),
        "execution": "signal_on_close_next_bar_open_stop_gap_aware",
        "btc_risk_overlay": btc_frame is not None,
        "results": {
            mode: run_backtest(
                frame, mode=mode, signal_settings=signal_settings, settings=settings,
                btc_frame=btc_frame,
            ).to_dict()
            for mode in modes
        },
    }
