"""Research-only 4h Donchian trend-following Spot strategy.

Signals are based on completed 4h bars, enter at the following bar open,
and use a causal ATR chandelier stop plus a Donchian exit. This module is
isolated from paper/live execution and never submits exchange orders.
"""
from __future__ import annotations

import dataclasses
import json
import math
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class TurtleTrendSettings:
    entry_lookback: int = 55
    exit_lookback: int = 20
    atr_period: int = 20
    ema_period: int = 200
    ema_slope_bars: int = 5
    atr_stop_multiple: float = 3.0
    max_holding_bars: int = 180
    max_entry_gap_atr: float = 1.0
    adx_period: int = 14
    volume_period: int = 20
    min_adx: float | None = None
    min_relative_volume: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "entry_lookback", "exit_lookback", "atr_period",
            "ema_period", "ema_slope_bars", "max_holding_bars",
            "adx_period", "volume_period",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if self.exit_lookback >= self.entry_lookback:
            raise ValueError("exit_lookback must be smaller than entry_lookback")
        for name in ("atr_stop_multiple", "max_entry_gap_atr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_adx is not None and (
            isinstance(self.min_adx, bool)
            or not math.isfinite(self.min_adx)
            or not 0 <= self.min_adx <= 100
        ):
            raise ValueError("min_adx must be in [0, 100] when supplied")
        if self.min_relative_volume is not None and (
            isinstance(self.min_relative_volume, bool)
            or not math.isfinite(self.min_relative_volume)
            or self.min_relative_volume <= 0
        ):
            raise ValueError("min_relative_volume must be finite and positive")


@dataclass(frozen=True)
class TurtleExecutionCosts:
    commission_bps: float = 10.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0

    def __post_init__(self) -> None:
        values = (self.commission_bps, self.half_spread_bps, self.slippage_bps)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("execution costs must be finite and non-negative")

    @property
    def impact_rate(self) -> float:
        return (self.half_spread_bps + self.slippage_bps) / 10_000.0

    @property
    def fee_rate(self) -> float:
        return self.commission_bps / 10_000.0


@dataclass(frozen=True)
class TurtlePortfolioSettings:
    initial_cash: float = 10_000.0
    risk_fraction: float = 0.005
    max_notional_fraction: float = 0.10

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        if not math.isfinite(self.risk_fraction) or not 0 < self.risk_fraction <= 0.02:
            raise ValueError("risk_fraction must be in (0, 0.02]")
        if not math.isfinite(self.max_notional_fraction) or not 0 < self.max_notional_fraction <= 1:
            raise ValueError("max_notional_fraction must be in (0, 1]")


@dataclass
class TurtleTrade:
    entry_time: str
    exit_time: str
    quantity: float
    entry_price: float
    exit_price: float
    stop_price: float
    exit_reason: str
    fees: float
    gross_pnl: float
    net_pnl: float


@dataclass
class TurtleBacktestResult:
    initial_cash: float
    final_equity: float
    trades: list[TurtleTrade] = field(default_factory=list)
    total_fees: float = 0.0
    total_spread_slippage: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)

    def metrics(self) -> dict:
        winners = [trade.net_pnl for trade in self.trades if trade.net_pnl > 0]
        losers = [trade.net_pnl for trade in self.trades if trade.net_pnl < 0]
        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
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
            "wins": len(winners),
            "losses": len(losers),
            "win_rate_pct": len(winners) / len(self.trades) * 100.0 if self.trades else 0.0,
            "profit_factor": (
                gross_profit / gross_loss if gross_loss
                else ("infinite" if gross_profit else 0.0)
            ),
            "expectancy_per_trade": (
                sum(trade.net_pnl for trade in self.trades) / len(self.trades)
                if self.trades else 0.0
            ),
            "net_profit_on_winners": gross_profit,
            "net_loss_on_losers": gross_loss,
            "max_drawdown_pct": max_drawdown * 100.0,
            "fees": self.total_fees,
            "spread_and_slippage": self.total_spread_slippage,
            "rejections": dict(self.rejections),
        }


COST_SCENARIOS = {
    "frictionless": TurtleExecutionCosts(0, 0, 0),
    "realistic_estimate": TurtleExecutionCosts(10, 1, 2),
    "conservative_estimate": TurtleExecutionCosts(15, 3, 5),
}


def _validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("OHLCV timestamps must be timezone-aware")
    if frame.empty or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("OHLCV data must be non-empty, chronological, and unique")
    data = frame.loc[:, ["open", "high", "low", "close", "volume"]].astype(float).copy()
    if data.isna().any().any() or not data.map(math.isfinite).all().all():
        raise ValueError("OHLCV values must be finite and non-null")
    if (data[["open", "high", "low", "close"]] <= 0).any().any() or (data.volume < 0).any():
        raise ValueError("prices must be positive and volume non-negative")
    if (data.high < data[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high must be the greatest OHLC price")
    if (data.low > data[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low must be the smallest OHLC price")
    return data


def resample_hourly_to_4h(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only complete, contiguous UTC 4-hour candles from 1h data."""
    data = _validate_ohlcv(frame)
    if len(data) < 4:
        raise ValueError("at least four hourly bars are required")
    if not data.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("hourly input contains missing or irregular 1h bars")
    grouped = data.resample("4h", origin="epoch", label="right", closed="left")
    counts = grouped["close"].count()
    complete = counts.eq(4)
    candles = grouped.agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    candles = candles.loc[complete].dropna()
    if candles.empty:
        raise ValueError("no complete 4-hour candles could be formed")
    return candles


def _features(
    frame: pd.DataFrame,
    benchmark_close: pd.Series,
    settings: TurtleTrendSettings,
) -> pd.DataFrame:
    data = _validate_ohlcv(frame)
    benchmark = benchmark_close.reindex(data.index).astype(float)
    if benchmark.isna().any() or (benchmark <= 0).any() or not benchmark.map(math.isfinite).all():
        raise ValueError("benchmark closes must be finite, positive, and timestamp-aligned")
    previous_close = data.close.shift(1)
    true_range = pd.concat(
        [
            data.high - data.low,
            (data.high - previous_close).abs(),
            (data.low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    features = data.copy()
    features["atr"] = true_range.rolling(
        settings.atr_period, min_periods=settings.atr_period
    ).mean()
    features["ema"] = data.close.ewm(
        span=settings.ema_period,
        adjust=False,
        min_periods=settings.ema_period,
    ).mean()
    features["entry_high"] = data.high.rolling(
        settings.entry_lookback, min_periods=settings.entry_lookback
    ).max().shift(1)
    features["exit_low"] = data.low.rolling(
        settings.exit_lookback, min_periods=settings.exit_lookback
    ).min().shift(1)
    benchmark_ema = benchmark.ewm(
        span=settings.ema_period,
        adjust=False,
        min_periods=settings.ema_period,
    ).mean()
    features["benchmark_close"] = benchmark
    features["benchmark_ema"] = benchmark_ema
    # Wilder-style ADX: the filter is optional, so the baseline signal remains
    # unchanged unless a refinement explicitly supplies min_adx.
    up_move = data.high.diff()
    down_move = -data.low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    alpha = 1.0 / settings.adx_period
    smoothed_tr = true_range.ewm(
        alpha=alpha, adjust=False, min_periods=settings.adx_period
    ).mean()
    plus_di = 100.0 * plus_dm.ewm(
        alpha=alpha, adjust=False, min_periods=settings.adx_period
    ).mean() / smoothed_tr.replace(0, math.nan)
    minus_di = 100.0 * minus_dm.ewm(
        alpha=alpha, adjust=False, min_periods=settings.adx_period
    ).mean() / smoothed_tr.replace(0, math.nan)
    directional_sum = (plus_di + minus_di).replace(0, math.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / directional_sum
    features["adx"] = dx.ewm(
        alpha=alpha, adjust=False, min_periods=settings.adx_period
    ).mean()
    prior_volume_mean = data.volume.rolling(
        settings.volume_period, min_periods=settings.volume_period
    ).mean().shift(1)
    features["relative_volume"] = data.volume / prior_volume_mean.replace(0, math.nan)

    features["long_entry"] = (
        (features.close > features.entry_high)
        & (features.close > features.ema)
        & (features.ema > features.ema.shift(settings.ema_slope_bars))
        & (features.benchmark_close > features.benchmark_ema)
        & (
            features.adx >= settings.min_adx
            if settings.min_adx is not None else True
        )
        & (
            features.relative_volume >= settings.min_relative_volume
            if settings.min_relative_volume is not None else True
        )
    )
    return features


def _record_exit(
    result: TurtleBacktestResult,
    position: dict,
    raw_exit: float,
    stamp: pd.Timestamp,
    reason: str,
    cash: float,
    costs: TurtleExecutionCosts,
) -> float:
    exit_fill = raw_exit * (1.0 - costs.impact_rate)
    exit_notional = position["quantity"] * exit_fill
    exit_fee = exit_notional * costs.fee_rate
    cash += exit_notional - exit_fee
    gross = (exit_fill - position["entry_price"]) * position["quantity"]
    net = gross - position["entry_fee"] - exit_fee
    result.total_fees += exit_fee
    result.total_spread_slippage += abs(raw_exit - exit_fill) * position["quantity"]
    result.trades.append(TurtleTrade(
        entry_time=position["entry_time"],
        exit_time=stamp.isoformat(),
        quantity=position["quantity"],
        entry_price=position["entry_price"],
        exit_price=exit_fill,
        stop_price=position["stop"],
        exit_reason=reason,
        fees=position["entry_fee"] + exit_fee,
        gross_pnl=gross,
        net_pnl=net,
    ))
    return cash


def run_backtest(
    frame: pd.DataFrame,
    *,
    benchmark_close: pd.Series | None = None,
    settings: TurtleTrendSettings | None = None,
    execution: TurtleExecutionCosts | None = None,
    portfolio: TurtlePortfolioSettings | None = None,
    start_index: int = 0,
    end_index: int | None = None,
) -> TurtleBacktestResult:
    """Backtest one 4h Spot symbol using next-open entries and causal exits."""
    settings = settings or TurtleTrendSettings()
    execution = execution or TurtleExecutionCosts()
    portfolio = portfolio or TurtlePortfolioSettings()
    data = _validate_ohlcv(frame)
    benchmark_close = benchmark_close if benchmark_close is not None else data.close
    features = _features(data, benchmark_close, settings)
    end_index = len(data) - 1 if end_index is None else end_index
    if start_index < 0 or end_index >= len(data) or start_index > end_index:
        raise ValueError("start_index/end_index must select a valid non-empty window")

    cash = portfolio.initial_cash
    result = TurtleBacktestResult(cash, cash)
    position: dict | None = None
    pending_entry: dict | None = None
    for i in range(start_index, end_index + 1):
        row = features.iloc[i]
        stamp = features.index[i]

        if pending_entry is not None:
            next_open = float(row.open)
            signal_close = pending_entry["signal_close"]
            signal_atr = pending_entry["signal_atr"]
            upper_gap = signal_close + settings.max_entry_gap_atr * signal_atr
            if next_open < signal_close or next_open > upper_gap:
                result.rejections["next-open gap outside entry band"] = (
                    result.rejections.get("next-open gap outside entry band", 0) + 1
                )
            else:
                entry_fill = next_open * (1.0 + execution.impact_rate)
                stop = entry_fill - settings.atr_stop_multiple * signal_atr
                stop_exit_fill = stop * (1.0 - execution.impact_rate)
                risk_per_unit = (
                    entry_fill - stop_exit_fill
                    + execution.fee_rate * (entry_fill + stop_exit_fill)
                )
                if stop <= 0 or risk_per_unit <= 0:
                    result.rejections["invalid stop geometry"] = (
                        result.rejections.get("invalid stop geometry", 0) + 1
                    )
                else:
                    quantity = min(
                        cash * portfolio.risk_fraction / risk_per_unit,
                        cash * portfolio.max_notional_fraction / entry_fill,
                    )
                    entry_notional = quantity * entry_fill
                    entry_fee = entry_notional * execution.fee_rate
                    if quantity > 0 and entry_notional + entry_fee <= cash:
                        cash -= entry_notional + entry_fee
                        result.total_fees += entry_fee
                        result.total_spread_slippage += abs(entry_fill - next_open) * quantity
                        position = {
                            "entry_index": i,
                            "entry_time": stamp.isoformat(),
                            "entry_price": entry_fill,
                            "entry_fee": entry_fee,
                            "quantity": quantity,
                            "stop": stop,
                            "high_watermark": next_open,
                            "pending_exit": None,
                        }
                    else:
                        result.rejections["insufficient cash after fees"] = (
                            result.rejections.get("insufficient cash after fees", 0) + 1
                        )
            pending_entry = None

        if position is not None:
            if position["pending_exit"] is not None:
                reason = position["pending_exit"]
                cash = _record_exit(
                    result, position, float(row.open), stamp, reason, cash, execution
                )
                position = None
            elif float(row.low) <= position["stop"]:
                raw_exit = min(float(row.open), position["stop"])
                cash = _record_exit(
                    result, position, raw_exit, stamp, "atr_stop", cash, execution
                )
                position = None
            else:
                held_bars = i - position["entry_index"] + 1
                if held_bars >= settings.max_holding_bars:
                    position["pending_exit"] = "max_holding"
                elif float(row.close) <= float(row.exit_low):
                    position["pending_exit"] = "donchian_exit"
                else:
                    position["high_watermark"] = max(
                        position["high_watermark"], float(row.high)
                    )
                    if pd.notna(row.atr) and row.atr > 0:
                        trailing_stop = (
                            position["high_watermark"]
                            - settings.atr_stop_multiple * float(row.atr)
                        )
                        position["stop"] = max(position["stop"], trailing_stop)

        if (
            position is None
            and pending_entry is None
            and i < end_index
            and bool(row.long_entry)
            and pd.notna(row.atr)
            and row.atr > 0
        ):
            pending_entry = {
                "signal_close": float(row.close),
                "signal_atr": float(row.atr),
            }

        marked = cash
        if position is not None:
            marked += position["quantity"] * float(row.close)
        result.equity_curve.append(marked)

    if position is not None:
        last = features.iloc[end_index]
        cash = _record_exit(
            result, position, float(last.close), features.index[end_index],
            "end_of_window", cash, execution,
        )
    result.final_equity = cash
    if result.equity_curve:
        result.equity_curve[-1] = cash
    return result


def _buy_and_hold_return(
    frame: pd.DataFrame,
    start: int,
    end: int,
    initial_cash: float,
    costs: TurtleExecutionCosts,
) -> float:
    entry_index = start + 1
    buy = float(frame.iloc[entry_index].open) * (1.0 + costs.impact_rate)
    sell = float(frame.iloc[end].close) * (1.0 - costs.impact_rate)
    quantity = initial_cash / (buy * (1.0 + costs.fee_rate))
    final_cash = quantity * sell * (1.0 - costs.fee_rate)
    return (final_cash / initial_cash - 1.0) * 100.0


def run_walk_forward_study(
    frames: Mapping[str, pd.DataFrame],
    *,
    warmup_bars: int = 300,
    test_bars: int = 180,
    settings: TurtleTrendSettings | None = None,
    portfolio: TurtlePortfolioSettings | None = None,
    scenarios: Mapping[str, TurtleExecutionCosts] = COST_SCENARIOS,
    benchmark_symbol: str = "BTC/USDT",
) -> dict:
    """Run fixed-rule, disjoint chronological OOS folds over common 4h candles."""
    if warmup_bars < 210 or test_bars < 24:
        raise ValueError("warmup_bars must be >= 210 and test_bars must be >= 24")
    if not frames or benchmark_symbol not in frames:
        raise ValueError("frames must include the benchmark symbol")
    settings = settings or TurtleTrendSettings()
    portfolio = portfolio or TurtlePortfolioSettings()
    hourly = {symbol: _validate_ohlcv(frame) for symbol, frame in sorted(frames.items())}
    four_hour = {symbol: resample_hourly_to_4h(frame) for symbol, frame in hourly.items()}
    common_index = next(iter(four_hour.values())).index
    for frame in list(four_hour.values())[1:]:
        common_index = common_index.intersection(frame.index, sort=False)
    common_index = common_index.sort_values()
    if len(common_index) < warmup_bars + test_bars:
        raise ValueError("not enough common 4h bars for one complete OOS fold")
    if not common_index.to_series().diff().dropna().eq(pd.Timedelta(hours=4)).all():
        raise ValueError("common 4h history contains missing or irregular candles")
    four_hour = {symbol: frame.loc[common_index].copy() for symbol, frame in four_hour.items()}
    benchmark = four_hour[benchmark_symbol].close
    folds = []
    for fold_number, start in enumerate(
        range(warmup_bars, len(common_index) - test_bars + 1, test_bars), 1
    ):
        end = start + test_bars - 1
        fold = {
            "fold": fold_number,
            "oos_start": common_index[start].isoformat(),
            "oos_end": common_index[end].isoformat(),
            "scenarios": {},
        }
        for scenario_name, costs in scenarios.items():
            symbol_rows = {}
            for symbol, frame in four_hour.items():
                result = run_backtest(
                    frame,
                    benchmark_close=benchmark,
                    settings=settings,
                    execution=costs,
                    portfolio=portfolio,
                    start_index=start,
                    end_index=end,
                )
                metrics = result.metrics()
                metrics["buy_hold_return_pct"] = _buy_and_hold_return(
                    frame, start, end, portfolio.initial_cash, costs
                )
                symbol_rows[symbol] = metrics
            returns = [row["net_return_pct"] for row in symbol_rows.values()]
            benchmarks = [row["buy_hold_return_pct"] for row in symbol_rows.values()]
            fold["scenarios"][scenario_name] = {
                "equal_weight_net_return_pct": sum(returns) / len(returns),
                "equal_weight_buy_hold_return_pct": sum(benchmarks) / len(benchmarks),
                "symbols": symbol_rows,
            }
        folds.append(fold)

    summary = {}
    for scenario_name in scenarios:
        returns = [
            fold["scenarios"][scenario_name]["equal_weight_net_return_pct"]
            for fold in folds
        ]
        equity = portfolio.initial_cash
        peak = equity
        drawdown = 0.0
        all_rows = [
            row
            for fold in folds
            for row in fold["scenarios"][scenario_name]["symbols"].values()
        ]
        for value in returns:
            equity *= 1.0 + value / 100.0
            peak = max(peak, equity)
            if peak > 0:
                drawdown = max(drawdown, (peak - equity) / peak)
        winners = sum(row["net_profit_on_winners"] for row in all_rows)
        losers = sum(row["net_loss_on_losers"] for row in all_rows)
        trades = sum(row["trades"] for row in all_rows)
        per_symbol = {}
        for symbol in four_hour:
            symbol_equity = portfolio.initial_cash
            for fold in folds:
                value = fold["scenarios"][scenario_name]["symbols"][symbol]["net_return_pct"]
                symbol_equity *= 1.0 + value / 100.0
            per_symbol[symbol] = (symbol_equity / portfolio.initial_cash - 1.0) * 100.0
        benchmarks = [
            fold["scenarios"][scenario_name]["equal_weight_buy_hold_return_pct"]
            for fold in folds
        ]
        summary[scenario_name] = {
            "folds": len(folds),
            "trades": trades,
            "compounded_oos_return_pct": (equity / portfolio.initial_cash - 1.0) * 100.0,
            "aggregate_profit_factor": winners / losers if losers else (
                "infinite" if winners else 0.0
            ),
            "expectancy_per_trade": (
                sum(row["net_pnl"] for row in all_rows) / trades if trades else 0.0
            ),
            "positive_folds": sum(value > 0 for value in returns),
            "max_drawdown_across_fold_returns_pct": drawdown * 100.0,
            "mean_buy_hold_return_pct": sum(benchmarks) / len(benchmarks),
            "per_symbol_compounded_oos_return_pct": per_symbol,
        }
    return {
        "study": "4h_donchian_trend_following_chronological_oos",
        "data_source": "Binance public Spot hourly candles resampled to complete UTC 4h bars",
        "execution_target": "Bybit Demo Spot proxy only; no Bybit fills are used",
        "parameters_fitted_on_oos": False,
        "symbols": list(four_hour),
        "bar_count_per_symbol_4h": len(common_index),
        "first_bar": common_index[0].isoformat(),
        "last_bar": common_index[-1].isoformat(),
        "settings": dataclasses.asdict(settings),
        "portfolio": dataclasses.asdict(portfolio),
        "cost_scenarios_per_fill_bps": {
            name: {
                "commission": costs.commission_bps,
                "half_spread": costs.half_spread_bps,
                "slippage": costs.slippage_bps,
            }
            for name, costs in scenarios.items()
        },
        "summary": summary,
        "folds": folds,
        "selection_gate": (
            "Research candidate only. Do not promote unless both realistic and "
            "conservative after-cost returns, expectancy, drawdown, and symbol "
            "breadth pass pre-registered criteria on untouched data."
        ),
    }


def run_refinement_comparison(
    frames: Mapping[str, pd.DataFrame],
    *,
    warmup_bars: int = 300,
    test_bars: int = 180,
    portfolio: TurtlePortfolioSettings | None = None,
    scenarios: Mapping[str, TurtleExecutionCosts] = COST_SCENARIOS,
    benchmark_symbol: str = "BTC/USDT",
) -> dict:
    """Compare a small, predeclared filter ablation set on historical development data.

    This is a diagnostic comparison, not an untouched holdout or permission to
    promote whichever variant happens to rank first on the inspected history.
    """
    variants = {
        "baseline": TurtleTrendSettings(),
        "adx_20": TurtleTrendSettings(min_adx=20.0),
        "relative_volume_1_2": TurtleTrendSettings(min_relative_volume=1.2),
        "adx_20_and_relative_volume_1_2": TurtleTrendSettings(
            min_adx=20.0, min_relative_volume=1.2
        ),
    }
    reports = {}
    for name, candidate_settings in variants.items():
        report = run_walk_forward_study(
            frames,
            warmup_bars=warmup_bars,
            test_bars=test_bars,
            settings=candidate_settings,
            portfolio=portfolio,
            scenarios=scenarios,
            benchmark_symbol=benchmark_symbol,
        )
        reports[name] = {
            "settings": dataclasses.asdict(candidate_settings),
            "summary": report["summary"],
            "folds": report["folds"],
        }
    return {
        "study": "4h_donchian_filter_refinement_comparison",
        "data_scope": "Previously inspected Binance Spot development history; not a locked holdout",
        "parameters_fitted_on_oos": False,
        "selection_warning": (
            "All variants are exploratory. Do not select/promote based on these same "
            "historical folds; reserve future untouched data and measured Bybit costs."
        ),
        "variants": reports,
    }


def load_hourly_frames(cache_dir: str, symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Load completed cached UTC 1h candles; no network calls are made."""
    from .history import HistoryCache

    cache = HistoryCache(cache_dir)
    now = pd.Timestamp.now(tz="UTC")
    frames = {}
    for symbol in symbols:
        frame = cache.load(symbol, "1h")
        if frame is None or frame.empty:
            raise ValueError(f"missing cached 1h history for {symbol}")
        frame = frame.sort_index()
        frame = frame.loc[
            frame.index + pd.Timedelta(hours=1) <= now
        ].dropna(subset=["open", "high", "low", "close", "volume"])
        if frame.empty:
            raise ValueError(f"no completed hourly candles for {symbol}")
        frames[symbol] = frame
    return frames


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the fixed-rule 4h Donchian trend-following OOS study."
    )
    parser.add_argument("--symbols", nargs="+", default=[
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT", "BCH/USDT",
    ])
    parser.add_argument("--cache", default="./crypto-cache")
    parser.add_argument("--warmup-bars", type=int, default=300)
    parser.add_argument("--test-bars", type=int, default=180)
    parser.add_argument(
        "--compare-refinements", action="store_true",
        help="Compare baseline with predeclared ADX and relative-volume filters.",
    )
    parser.add_argument(
        "--out", type=pathlib.Path,
        default=pathlib.Path("artifacts/turtle_trend_oos.json"),
    )
    args = parser.parse_args(argv)
    frames = load_hourly_frames(args.cache, tuple(args.symbols))
    if args.compare_refinements:
        report = run_refinement_comparison(
            frames, warmup_bars=args.warmup_bars, test_bars=args.test_bars
        )
        printable = {
            name: details["summary"] for name, details in report["variants"].items()
        }
    else:
        report = run_walk_forward_study(
            frames, warmup_bars=args.warmup_bars, test_bars=args.test_bars
        )
        printable = report["summary"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(printable, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
