"""Research-only 3-minute EMA(9)/UTC-session VWAP crossover screen.

Strictly causal: signals use completed 3-minute bars, entries/exits fill at the
next bar open, and all reported returns include explicit per-fill costs. Spot
long-only proxy; no options, leverage, exchange integration, or orders.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EmaVwapCosts:
    commission_bps: float = 10.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0

    def __post_init__(self) -> None:
        values = (self.commission_bps, self.half_spread_bps, self.slippage_bps)
        if any(pd.isna(v) or not float("-inf") < v < float("inf") or v < 0 for v in values):
            raise ValueError("costs must be finite and non-negative")

    @property
    def per_fill_bps(self) -> float:
        return self.commission_bps + self.half_spread_bps + self.slippage_bps


def _validate_1m(frame: pd.DataFrame, symbol: str) -> None:
    required = {"open", "high", "low", "close", "volume"}
    if frame is None or frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"{symbol} has missing OHLCV data")
    if frame.index.tz is None:
        raise ValueError(f"{symbol} timestamps must be timezone-aware")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError(f"{symbol} timestamps must be strictly increasing and unique")
    values = frame[list(required)].astype(float)
    if values.isna().any().any():
        raise ValueError(f"{symbol} contains missing OHLCV values")
    if not (values[["open", "high", "low", "close"]] > 0).all().all():
        raise ValueError(f"{symbol} prices must be positive")
    if (values.volume < 0).any():
        raise ValueError(f"{symbol} volume must be non-negative")
    if (values.high < values[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError(f"{symbol} contains invalid highs")
    if (values.low > values[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError(f"{symbol} contains invalid lows")


def resample_3m(frame: pd.DataFrame, symbol: str = "unknown") -> pd.DataFrame:
    """Aggregate only complete, consecutive UTC-aligned groups of three 1m bars."""
    _validate_1m(frame, symbol)
    source = frame[["open", "high", "low", "close", "volume"]].astype(float).copy()
    source.index = source.index.tz_convert("UTC")
    groups = source.resample("3min", origin="start_day", label="left", closed="left")
    counts = groups["close"].count()
    bars = pd.DataFrame({
        "open": groups["open"].first(),
        "high": groups["high"].max(),
        "low": groups["low"].min(),
        "close": groups["close"].last(),
        "volume": groups["volume"].sum(),
        "minute_count": counts,
    })
    # A group is valid only if its exact UTC minute starts are present.
    valid_groups = []
    for bucket, group in source.groupby(source.index.floor("3min")):
        expected = pd.date_range(bucket, periods=3, freq="min")
        if len(group) == 3 and group.index.equals(expected):
            valid_groups.append(bucket)
    bars = bars.loc[(bars.minute_count == 3) & bars.index.isin(valid_groups)]
    return bars.drop(columns="minute_count")


def _indicators(bars: pd.DataFrame) -> pd.DataFrame:
    result = bars.copy()
    result["ema9"] = result.close.ewm(span=9, adjust=False, min_periods=9).mean()
    typical = (result.high + result.low + result.close) / 3.0
    session = result.index.floor("D")
    pv = typical * result.volume
    cumulative_pv = pv.groupby(session).cumsum()
    cumulative_volume = result.volume.groupby(session).cumsum()
    result["vwap"] = cumulative_pv / cumulative_volume.where(cumulative_volume > 0)
    result["cross_up"] = (
        (result.ema9.shift(1) <= result.vwap.shift(1))
        & (result.ema9 > result.vwap)
    )
    return result


def _simulate_symbol(
    symbol: str, bars: pd.DataFrame, costs: EmaVwapCosts
) -> tuple[list[dict], list[tuple[pd.Timestamp, float]]]:
    data = _indicators(bars)
    trades: list[dict] = []
    equity_events: list[tuple[pd.Timestamp, float]] = []
    cost = costs.per_fill_bps / 10_000.0
    i = 1
    position: dict | None = None
    while i < len(data) - 1:
        row = data.iloc[i]
        next_row = data.iloc[i + 1]
        timestamp = data.index[i]
        if position is not None:
            if pd.notna(row.ema9) and float(row.close) < float(row.ema9):
                exit_raw = float(next_row.open)
                exit_fill = exit_raw * (1.0 - cost)
                net = exit_fill / position["entry_fill"] - 1.0
                trade = {
                    "symbol": symbol,
                    "signal_time": position["signal_time"].isoformat(),
                    "entry_time": position["entry_time"].isoformat(),
                    "exit_signal_time": timestamp.isoformat(),
                    "exit_time": data.index[i + 1].isoformat(),
                    "entry_price": position["entry_raw"],
                    "exit_price": exit_raw,
                    "net_return_pct": net * 100.0,
                    "round_trip_cost_bps": 2 * costs.per_fill_bps,
                    "exit_reason": "closed_below_ema9_next_open",
                }
                trades.append(trade)
                equity_events.append((data.index[i + 1], net))
                position = None
                i += 1
                continue
        elif bool(row.cross_up):
            entry_raw = float(next_row.open)
            position = {
                "signal_time": timestamp,
                "entry_time": data.index[i + 1],
                "entry_raw": entry_raw,
                "entry_fill": entry_raw * (1.0 + cost),
            }
            i += 1
            continue
        i += 1

    # Close any remaining exposure at the final available close; no future bar is assumed.
    if position is not None:
        exit_raw = float(data.close.iloc[-1])
        exit_fill = exit_raw * (1.0 - cost)
        net = exit_fill / position["entry_fill"] - 1.0
        trades.append({
            "symbol": symbol,
            "signal_time": position["signal_time"].isoformat(),
            "entry_time": position["entry_time"].isoformat(),
            "exit_signal_time": data.index[-1].isoformat(),
            "exit_time": data.index[-1].isoformat(),
            "entry_price": position["entry_raw"],
            "exit_price": exit_raw,
            "net_return_pct": net * 100.0,
            "round_trip_cost_bps": 2 * costs.per_fill_bps,
            "exit_reason": "end_of_data",
        })
        equity_events.append((data.index[-1], net))
    return trades, equity_events


def run_ema_vwap_study(
    frames: Mapping[str, pd.DataFrame], *, costs: EmaVwapCosts | None = None
) -> dict:
    """Run the strict long-only EMA9/VWAP crossover on cached 1m Spot OHLCV."""
    if not frames:
        raise ValueError("at least one symbol is required")
    costs = costs or EmaVwapCosts()
    symbols = sorted(frames)
    all_trades: list[dict] = []
    per_symbol: dict[str, dict] = {}
    sleeve_equities: dict[str, float] = {}
    for symbol in symbols:
        bars = resample_3m(frames[symbol], symbol)
        if len(bars) < 30:
            raise ValueError(f"{symbol} has fewer than 30 complete 3-minute bars")
        trades, _ = _simulate_symbol(symbol, bars, costs)
        equity = 1.0
        wins = 0
        gains = 0.0
        losses = 0.0
        peak = 1.0
        max_dd = 0.0
        for trade in trades:
            ret = trade["net_return_pct"] / 100.0
            equity *= 1.0 + ret
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / peak - 1.0)
            wins += ret > 0
            gains += max(ret, 0.0)
            losses += max(-ret, 0.0)
        sleeve_equities[symbol] = equity
        per_symbol[symbol] = {
            "bars_3m": len(bars),
            "first_bar": bars.index[0].isoformat(),
            "last_bar": bars.index[-1].isoformat(),
            "trades": len(trades),
            "win_rate_pct": wins / len(trades) * 100 if trades else 0.0,
            "profit_factor": gains / losses if losses else ("infinite" if gains else 0.0),
            "compounded_return_pct": (equity - 1.0) * 100.0,
            "max_closed_trade_drawdown_pct": max_dd * 100.0,
        }
        all_trades.extend(trades)

    # Equal, fixed capital sleeves per symbol; each sleeve compounds independently.
    portfolio_equity = sum(sleeve_equities.values()) / len(symbols)
    all_trades.sort(key=lambda trade: (trade["exit_time"], trade["symbol"]))
    positive = sum(max(t["net_return_pct"], 0.0) for t in all_trades)
    negative = -sum(min(t["net_return_pct"], 0.0) for t in all_trades)
    # Chronological thirds are diagnostics, not independent validation folds.
    first = min(pd.Timestamp(v["first_bar"]) for v in per_symbol.values())
    last = max(pd.Timestamp(v["last_bar"]) for v in per_symbol.values())
    boundaries = [first + (last - first) * k / 3 for k in (1, 2)]
    slices = []
    for start, end in zip([first, *boundaries], [*boundaries, last]):
        returns = []
        for symbol in symbols:
            sleeve = 1.0
            for trade in all_trades:
                entry = pd.Timestamp(trade["entry_time"])
                if trade["symbol"] == symbol and start <= entry < end:
                    sleeve *= 1.0 + trade["net_return_pct"] / 100.0
            returns.append(sleeve)
        slices.append((sum(returns) / len(returns) - 1.0) * 100.0)

    return {
        "strategy": "ema9_crosses_utc_session_vwap_long_only",
        "timeframe": "3m aggregated from complete 1m bars",
        "symbols": symbols,
        "per_fill_cost_bps": costs.per_fill_bps,
        "round_trip_cost_bps": 2 * costs.per_fill_bps,
        "trades": len(all_trades),
        "win_rate_pct": (
            sum(t["net_return_pct"] > 0 for t in all_trades) / len(all_trades) * 100
            if all_trades else 0.0
        ),
        "profit_factor": positive / negative if negative else ("infinite" if positive else 0.0),
        "equal_weight_sleeve_compounded_return_pct": (portfolio_equity - 1.0) * 100.0,
        "chronological_thirds_pct": slices,
        "per_symbol": per_symbol,
        "trade_log": all_trades,
        "rules": {
            "vwap": "UTC-day session VWAP, typical price (H+L+C)/3 weighted by base volume",
            "entry": "completed 3m EMA9 crosses above VWAP; fill next 3m open",
            "exit": "completed 3m candle closes below EMA9; fill next 3m open",
            "direction": "long-only Spot proxy; no shorting or options",
            "stop_loss": "no 25-30% underlying stop; video stop refers to option premium",
            "positioning": "one position per symbol; equal fixed initial capital sleeves",
        },
        "research_only": True,
        "execution_note": "Binance public Spot OHLCV proxy; not Bybit fills or account-specific fees.",
    }
