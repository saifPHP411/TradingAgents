"""Fast, research-only cross-sectional Spot momentum screen.

Signals use completed hourly closes, execute at the next open, and exit after
one fixed 24-hour holding period. No exchange/order integration exists here.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MomentumCosts:
    commission_bps: float = 10.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0

    def __post_init__(self) -> None:
        values = (self.commission_bps, self.half_spread_bps, self.slippage_bps)
        if any(pd.isna(v) or not float("-inf") < v < float("inf") or v < 0 for v in values):
            raise ValueError("costs must be finite and non-negative")

    @property
    def per_side_bps(self) -> float:
        return self.commission_bps + self.half_spread_bps + self.slippage_bps


@dataclass(frozen=True)
class MomentumConfig:
    lookback_bars: int = 72
    trend_bars: int = 96
    rebalance_bars: int = 24
    top_n: int = 2
    allocation_per_asset: float = 0.10
    min_universe: int = 3

    def __post_init__(self) -> None:
        for name in ("lookback_bars", "trend_bars", "rebalance_bars", "top_n", "min_universe"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.allocation_per_asset <= 0.20:
            raise ValueError("allocation_per_asset must be in (0, 0.20]")


def _validate_frames(frames: Mapping[str, pd.DataFrame]) -> tuple[list[str], pd.DatetimeIndex]:
    if len(frames) < 3:
        raise ValueError("at least three symbols are required")
    symbols = sorted(frames)
    common = None
    for symbol, frame in frames.items():
        required = {"open", "high", "low", "close", "volume"}
        if frame is None or frame.empty or not required.issubset(frame.columns):
            raise ValueError(f"{symbol} has missing OHLCV data")
        if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError(f"{symbol} timestamps must be strictly increasing and unique")
        if frame.index.tz is None:
            raise ValueError(f"{symbol} timestamps must be timezone-aware")
        values = frame[["open", "high", "low", "close", "volume"]].astype(float)
        if values.isna().any().any() or not (values[["open", "high", "low", "close"]] > 0).all().all():
            raise ValueError(f"{symbol} contains invalid OHLCV values")
        if (values.volume < 0).any() or (values.high < values[["open", "low", "close"]].max(axis=1)).any():
            raise ValueError(f"{symbol} contains invalid OHLCV values")
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or len(common) < 200:
        raise ValueError("at least 200 common hourly bars are required")
    common = common.sort_values()
    expected = pd.Timedelta(hours=1)
    if not (common.to_series().diff().dropna() == expected).all():
        raise ValueError("common timestamps contain missing or irregular hourly bars")
    return symbols, common


def run_momentum_screen(
    frames: Mapping[str, pd.DataFrame],
    *,
    costs: MomentumCosts | None = None,
    config: MomentumConfig | None = None,
    benchmark_symbol: str = "BTC/USDT",
) -> dict:
    """Run a short, fixed-rule hourly cross-sectional momentum screen."""
    costs = costs or MomentumCosts()
    config = config or MomentumConfig()
    symbols, common = _validate_frames(frames)
    if benchmark_symbol not in frames:
        raise ValueError(f"benchmark {benchmark_symbol} is missing")
    aligned = {s: frames[s].reindex(common) for s in symbols}
    if any(frame[["open", "high", "low", "close", "volume"]].isna().any().any()
           for frame in aligned.values()):
        raise ValueError("symbols must share every evaluated timestamp")
    warmup = max(config.lookback_bars, config.trend_bars) + 1
    if len(common) <= warmup + config.rebalance_bars:
        raise ValueError("not enough bars after indicator warm-up")

    close = {s: f.close.astype(float) for s, f in aligned.items()}
    ema = {s: close[s].ewm(span=config.trend_bars, adjust=False,
                           min_periods=config.trend_bars).mean() for s in symbols}
    benchmark_ok = close[benchmark_symbol] > ema[benchmark_symbol]
    total_return = 1.0
    fold_returns: list[float] = []
    trades: list[dict] = []
    periods = list(range(warmup, len(common) - config.rebalance_bars, config.rebalance_bars))
    if not periods:
        raise ValueError("no complete rebalance periods available")
    period_returns: list[float] = []
    cost_rate = costs.per_side_bps / 10_000

    for signal_i in periods:
        entry_i = signal_i + 1
        exit_i = signal_i + config.rebalance_bars + 1
        if exit_i >= len(common):
            continue
        picks: list[tuple[float, str]] = []
        if bool(benchmark_ok.iloc[signal_i]):
            for symbol in symbols:
                if symbol == benchmark_symbol:
                    pass
                momentum = float(close[symbol].iloc[signal_i] /
                                 close[symbol].iloc[signal_i - config.lookback_bars] - 1)
                trend_ok = close[symbol].iloc[signal_i] > ema[symbol].iloc[signal_i]
                if momentum > 0 and bool(trend_ok):
                    picks.append((momentum, symbol))
        picks.sort(reverse=True)
        selected = [symbol for _, symbol in picks[:config.top_n]]
        weighted_net = 0.0
        for symbol in selected:
            entry_raw = float(aligned[symbol].open.iloc[entry_i])
            exit_raw = float(aligned[symbol].open.iloc[exit_i])
            entry_fill = entry_raw * (1 + cost_rate)
            exit_fill = exit_raw * (1 - cost_rate)
            net_return = exit_fill / entry_fill - 1
            weighted_net += config.allocation_per_asset * net_return
            trades.append({
                "symbol": symbol,
                "signal_time": common[signal_i].isoformat(),
                "entry_time": common[entry_i].isoformat(),
                "exit_time": common[exit_i].isoformat(),
                "momentum_pct": round(next(v for v, s in picks if s == symbol) * 100, 5),
                "net_return_pct": round(net_return * 100, 5),
                "cost_round_trip_bps": round(2 * costs.per_side_bps, 3),
            })
        # Unallocated capital remains in cash; no leverage or reinvestment of idle cash.
        total_return *= 1 + weighted_net
        period_returns.append(weighted_net)

    if not period_returns:
        raise ValueError("no completed trades or rebalance periods")
    # Report contiguous thirds as quick chronological diagnostic slices.
    chunk = max(1, len(period_returns) // 3)
    for start in range(0, len(period_returns), chunk):
        value = 1.0
        for r in period_returns[start:start + chunk]:
            value *= 1 + r
        fold_returns.append((value - 1) * 100)

    wins = sum(t["net_return_pct"] > 0 for t in trades)
    net_values = [t["net_return_pct"] for t in trades]
    gross_positive = sum(v for v in net_values if v > 0)
    gross_negative = -sum(v for v in net_values if v < 0)
    return {
        "strategy": "cross_sectional_momentum_market_filter",
        "data_interval": "1h",
        "symbols": symbols,
        "bars": len(common),
        "first_bar": common[0].isoformat(),
        "last_bar": common[-1].isoformat(),
        "parameters": {
            "momentum_lookback_bars": config.lookback_bars,
            "trend_ema_bars": config.trend_bars,
            "rebalance_bars": config.rebalance_bars,
            "top_n": config.top_n,
            "allocation_per_asset": config.allocation_per_asset,
            "benchmark_filter": benchmark_symbol + " close above EMA",
        },
        "costs_per_fill_bps": costs.per_side_bps,
        "round_trip_cost_bps": 2 * costs.per_side_bps,
        "trades": len(trades),
        "wins": wins,
        "win_rate_pct": wins / len(trades) * 100 if trades else 0.0,
        "profit_factor": gross_positive / gross_negative if gross_negative else (
            "infinite" if gross_positive else 0.0
        ),
        "compounded_portfolio_return_pct": (total_return - 1) * 100,
        "quick_chronological_slices_pct": fold_returns,
        "trade_log": trades,
        "research_only": True,
        "execution_note": "Simulated next-open fills on Binance public Spot proxy data; not Bybit fills.",
    }
