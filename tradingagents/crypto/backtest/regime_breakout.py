"""Research-only regime-aware spot breakout strategy.

This module is isolated from the paper worker. It provides a causal long-only
signal and a single-symbol OHLCV backtest with next-bar-open entries, conservative
intrabar stop handling, explicit execution costs and risk-based sizing.

It is a research candidate, not a claim of profitability and not wired to trading.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class RegimeBreakoutSettings:
    fast_ema: int = 20
    slow_ema: int = 50
    breakout_lookback: int = 20
    volume_lookback: int = 20
    atr_period: int = 14
    volume_multiple: float = 1.20
    stop_atr_multiple: float = 2.5
    reward_risk: float = 3.0
    max_holding_bars: int = 48

    def __post_init__(self) -> None:
        for name in ("fast_ema", "slow_ema", "breakout_lookback",
                     "volume_lookback", "atr_period", "max_holding_bars"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least 2")
        if self.fast_ema >= self.slow_ema:
            raise ValueError("fast_ema must be smaller than slow_ema")
        if self.volume_multiple <= 0 or self.stop_atr_multiple <= 0 or self.reward_risk <= 0:
            raise ValueError("volume_multiple, stop_atr_multiple and reward_risk must be positive")


@dataclass(frozen=True)
class ExecutionCosts:
    """Per-fill assumptions; these are scenarios, not verified account fees."""
    commission_bps: float = 10.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0

    def __post_init__(self) -> None:
        if min(self.commission_bps, self.half_spread_bps, self.slippage_bps) < 0:
            raise ValueError("execution costs cannot be negative")


@dataclass(frozen=True)
class BacktestSettings:
    initial_cash: float = 10_000.0
    risk_fraction: float = 0.005
    max_notional_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not 0 < self.risk_fraction <= 0.02:
            raise ValueError("risk_fraction must be in (0, 0.02]")
        if not 0 < self.max_notional_fraction <= 1:
            raise ValueError("max_notional_fraction must be in (0, 1]")


@dataclass
class BreakoutTrade:
    entry_time: str
    exit_time: str
    quantity: float
    entry_price: float
    exit_price: float
    stop: float
    target: float
    exit_reason: str
    fees: float
    gross_pnl: float
    net_pnl: float


@dataclass
class BreakoutBacktestResult:
    initial_cash: float
    final_equity: float
    trades: list[BreakoutTrade] = field(default_factory=list)
    total_fees: float = 0.0
    total_slippage_spread: float = 0.0
    equity_curve: list[float] = field(default_factory=list)

    def metrics(self) -> dict:
        wins = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        losses = [t.net_pnl for t in self.trades if t.net_pnl < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        peak = self.initial_cash
        max_drawdown = 0.0
        for equity in self.equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
        return {
            "initial_equity": self.initial_cash,
            "final_equity": self.final_equity,
            "net_pnl": self.final_equity - self.initial_cash,
            "net_return_pct": (self.final_equity / self.initial_cash - 1.0) * 100.0,
            "trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": (len(wins) / len(self.trades) * 100.0) if self.trades else 0.0,
            "profit_factor": (gross_profit / gross_loss) if gross_loss else (float("inf") if gross_profit else 0.0),
            "expectancy_per_trade": (
                sum(t.net_pnl for t in self.trades) / len(self.trades) if self.trades else 0.0
            ),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "fees": self.total_fees,
            "spread_and_slippage": self.total_slippage_spread,
        }


def indicators(frame: pd.DataFrame, settings: RegimeBreakoutSettings = RegimeBreakoutSettings()) -> pd.DataFrame:
    """Return causal indicators for a chronological OHLCV frame."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    if frame.empty:
        return pd.DataFrame(index=frame.index)
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("OHLCV index must be strictly chronological and unique")
    out = frame.loc[:, ["open", "high", "low", "close", "volume"]].astype(float).copy()
    if (out[["open", "high", "low", "close"]] <= 0).any().any() or (out["volume"] < 0).any():
        raise ValueError("prices must be positive and volume cannot be negative")
    if (out["high"] < out[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high must be the greatest OHLC price")
    if (out["low"] > out[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low must be the smallest OHLC price")
    previous_close = out["close"].shift(1)
    true_range = pd.concat(
        [out["high"] - out["low"],
         (out["high"] - previous_close).abs(),
         (out["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr"] = true_range.rolling(settings.atr_period, min_periods=settings.atr_period).mean()
    out["ema_fast"] = out["close"].ewm(span=settings.fast_ema, adjust=False,
                                         min_periods=settings.fast_ema).mean()
    out["ema_slow"] = out["close"].ewm(span=settings.slow_ema, adjust=False,
                                         min_periods=settings.slow_ema).mean()
    out["ema_slow_rising"] = out["ema_slow"] > out["ema_slow"].shift(5)
    out["breakout_level"] = out["high"].rolling(
        settings.breakout_lookback, min_periods=settings.breakout_lookback
    ).max().shift(1)
    out["volume_baseline"] = out["volume"].rolling(
        settings.volume_lookback, min_periods=settings.volume_lookback
    ).mean().shift(1)
    out["long_entry"] = (
        (out["close"] > out["breakout_level"])
        & (out["ema_fast"] > out["ema_slow"])
        & out["ema_slow_rising"]
        & (out["close"] > out["ema_fast"])
        & (out["volume"] >= out["volume_baseline"] * settings.volume_multiple)
        & out["atr"].notna()
    )
    return out


def entry_levels(row: pd.Series, settings: RegimeBreakoutSettings = RegimeBreakoutSettings()) -> tuple[float, float, float] | None:
    """Return (reference entry, stop, target) for a completed signal bar."""
    entry = float(row["close"])
    atr = float(row["atr"])
    if not pd.notna(atr) or atr <= 0 or entry <= 0:
        return None
    stop = entry - settings.stop_atr_multiple * atr
    target = entry + settings.reward_risk * (entry - stop)
    if stop <= 0 or not (stop < entry < target):
        return None
    return entry, stop, target


def run_backtest(
    frame: pd.DataFrame,
    *,
    strategy: RegimeBreakoutSettings = RegimeBreakoutSettings(),
    execution: ExecutionCosts = ExecutionCosts(),
    portfolio: BacktestSettings = BacktestSettings(),
    start_index: int = 0,
    end_index: int | None = None,
) -> BreakoutBacktestResult:
    """Backtest one spot symbol; signals use closed bars and fill at next open.

    If one bar touches both stop and target, stop wins. A position is closed at
    the last close if the data ends before its stop/target/time exit.
    """
    data = indicators(frame, strategy)
    if data.empty:
        return BreakoutBacktestResult(initial_cash=portfolio.initial_cash, final_equity=portfolio.initial_cash)
    end_index = len(data) - 1 if end_index is None else end_index
    if start_index < 0 or end_index >= len(data) or start_index > end_index:
        raise ValueError("start_index/end_index must select a valid non-empty window")
    cash = float(portfolio.initial_cash)
    result = BreakoutBacktestResult(initial_cash=cash, final_equity=cash)
    position: dict | None = None
    impact = (execution.half_spread_bps + execution.slippage_bps) / 10_000.0

    for i in range(start_index, end_index + 1):
        row = data.iloc[i]
        stamp = data.index[i]
        if position is not None:
            held = i - position["entry_index"]
            exit_price = None
            reason = None
            if float(row["low"]) <= position["stop"]:
                # A stop-market order cannot fill at the stop after a gap below it.
                exit_price, reason = min(position["stop"], float(row["open"])), "stop"
            elif float(row["high"]) >= position["target"]:
                exit_price, reason = position["target"], "target"
            elif held >= strategy.max_holding_bars:
                exit_price, reason = float(row["close"]), "time"
            if exit_price is not None:
                fill = exit_price * (1.0 - impact)
                notional = position["quantity"] * fill
                fee = notional * execution.commission_bps / 10_000.0
                cash += notional - fee
                gross = (fill - position["entry_price"]) * position["quantity"]
                net = gross - position["entry_fee"] - fee
                result.total_fees += fee
                result.total_slippage_spread += (
                    abs(exit_price - fill) * position["quantity"]
                )
                result.trades.append(BreakoutTrade(
                    entry_time=position["entry_time"],
                    exit_time=stamp.isoformat(),
                    quantity=position["quantity"],
                    entry_price=position["entry_price"],
                    exit_price=fill,
                    stop=position["stop"],
                    target=position["target"],
                    exit_reason=reason,
                    fees=position["entry_fee"] + fee,
                    gross_pnl=gross,
                    net_pnl=net,
                ))
                position = None

        # Signal at this close; the earliest fill is the next bar's open.
        if position is None and i + 1 <= end_index and bool(row["long_entry"]):
            levels = entry_levels(row, strategy)
            if levels is None:
                continue
            _, signal_stop, _ = levels
            next_open = float(data.iloc[i + 1]["open"])
            fill = next_open * (1.0 + impact)
            risk_per_unit = fill - signal_stop
            if risk_per_unit <= 0:
                continue
            risk_budget = cash * portfolio.risk_fraction
            quantity_by_risk = risk_budget / risk_per_unit
            quantity_by_notional = cash * portfolio.max_notional_fraction / fill
            quantity = min(quantity_by_risk, quantity_by_notional)
            if quantity <= 0:
                continue
            # Keep the signal-bar invalidation price fixed across an overnight/bar gap.
            # Sizing above uses this exact distance, so the risk budget remains honest.
            stop = signal_stop
            target = fill + strategy.reward_risk * (fill - stop)
            entry_notional = quantity * fill
            entry_fee = entry_notional * execution.commission_bps / 10_000.0
            total_debit = entry_notional + entry_fee
            if total_debit > cash:
                quantity = cash / (fill * (1.0 + execution.commission_bps / 10_000.0))
                entry_notional = quantity * fill
                entry_fee = entry_notional * execution.commission_bps / 10_000.0
                total_debit = entry_notional + entry_fee
            cash -= total_debit
            result.total_fees += entry_fee
            result.total_slippage_spread += abs(fill - next_open) * quantity
            position = {
                "entry_index": i + 1,
                "entry_time": data.index[i + 1].isoformat(),
                "entry_price": fill,
                "quantity": quantity,
                "stop": stop,
                "target": target,
                "entry_fee": entry_fee,
            }
        marked_equity = cash + (position["quantity"] * float(row["close"]) if position else 0.0)
        result.equity_curve.append(marked_equity)

    if position is not None and len(data):
        stamp = data.index[end_index]
        raw_exit = float(data.iloc[end_index]["close"])
        fill = raw_exit * (1.0 - impact)
        notional = position["quantity"] * fill
        fee = notional * execution.commission_bps / 10_000.0
        cash += notional - fee
        gross = (fill - position["entry_price"]) * position["quantity"]
        net = gross - position["entry_fee"] - fee
        result.total_fees += fee
        result.total_slippage_spread += abs(raw_exit - fill) * position["quantity"]
        result.trades.append(BreakoutTrade(
            entry_time=position["entry_time"], exit_time=stamp.isoformat(),
            quantity=position["quantity"], entry_price=position["entry_price"],
            exit_price=fill, stop=position["stop"], target=position["target"],
            exit_reason="data_end", fees=position["entry_fee"] + fee,
            gross_pnl=gross, net_pnl=net,
        ))

    result.final_equity = cash
    if result.equity_curve:
        result.equity_curve[-1] = cash
    else:
        result.equity_curve.append(cash)
    return result


__all__ = [
    "BacktestSettings", "BreakoutBacktestResult", "BreakoutTrade",
    "ExecutionCosts", "RegimeBreakoutSettings", "entry_levels",
    "indicators", "run_backtest",
]
