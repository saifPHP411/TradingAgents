"""Research-only EMA trend backtester for linear USDT perpetual contracts.

Uses completed 4h candles, next-bar-open entries, isolated margin, one-way
positions and historical funding events. This module never submits exchange orders.
The liquidation price is an approximation, not Bybit's tiered formula.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import pandas as pd

from tradingagents.crypto.perpetuals import (
    LinearPerpetualInstrument,
    PerpetualRiskSettings,
    estimate_isolated_liquidation_price,
    linear_funding_cashflow,
    size_linear_position,
)


@dataclass(frozen=True)
class PerpetualEMAStrategy:
    enabled: bool = False
    fast: int = 14
    slow: int = 50
    trend: int = 200
    atr_period: int = 14
    atr_multiple: float = 2.5
    min_adx: float | None = 18.0
    signal_mode: str = "ema_cross"
    breakout_period: int = 20
    max_holding_bars: int = 180
    mean_reversion_period: int = 20
    mean_reversion_std: float = 2.0
    mean_reversion_rsi_period: int = 14
    mean_reversion_rsi_long: float = 32.0
    mean_reversion_rsi_short: float = 68.0
    mean_reversion_max_adx: float = 18.0
    interval_minutes: int = 240
    risk: PerpetualRiskSettings = field(default_factory=PerpetualRiskSettings)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        if not (2 <= self.fast < self.slow < self.trend):
            raise ValueError("EMA periods must satisfy 2 <= fast < slow < trend")
        if self.atr_period < 2 or self.max_holding_bars < 1:
            raise ValueError("ATR period and max holding bars must be positive")
        if self.interval_minutes != 240:
            raise ValueError("strategy uses completed 4-hour candles only")
        if not math.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError("ATR multiple must be finite and positive")
        if self.min_adx is not None and (not math.isfinite(self.min_adx) or not 0 <= self.min_adx <= 100):
            raise ValueError("min_adx must be in [0, 100]")
        if self.signal_mode not in {"ema_cross", "donchian_breakout", "bollinger_reversion"}:
            raise ValueError("unsupported signal_mode")
        if (
            isinstance(self.mean_reversion_period, bool)
            or not isinstance(self.mean_reversion_period, int)
            or not 10 <= self.mean_reversion_period <= 100
        ):
            raise ValueError("mean_reversion_period must be an integer in [10, 100]")
        if (
            isinstance(self.mean_reversion_rsi_period, bool)
            or not isinstance(self.mean_reversion_rsi_period, int)
            or not 5 <= self.mean_reversion_rsi_period <= 50
        ):
            raise ValueError("mean_reversion_rsi_period must be an integer in [5, 50]")
        if not math.isfinite(self.mean_reversion_std) or self.mean_reversion_std <= 0:
            raise ValueError("mean_reversion_std must be finite and positive")
        if not (
            0 <= self.mean_reversion_rsi_long < 50
            < self.mean_reversion_rsi_short <= 100
        ):
            raise ValueError("mean-reversion RSI thresholds must straddle 50")
        if not math.isfinite(self.mean_reversion_max_adx) or not 0 <= self.mean_reversion_max_adx <= 100:
            raise ValueError("mean_reversion_max_adx must be in [0, 100]")
        if (
            isinstance(self.breakout_period, bool)
            or not isinstance(self.breakout_period, int)
            or not 2 <= self.breakout_period <= 200
        ):
            raise ValueError("breakout_period must be an integer in [2, 200]")


@dataclass(frozen=True)
class PerpetualCosts:
    commission_bps: float = 5.5
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0

    def __post_init__(self) -> None:
        for name in ("commission_bps", "half_spread_bps", "slippage_bps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def fee_rate(self) -> float:
        return self.commission_bps / 10_000

    @property
    def impact_rate(self) -> float:
        return (self.half_spread_bps + self.slippage_bps) / 10_000

    @property
    def per_fill_rate(self) -> float:
        return self.fee_rate + self.impact_rate


@dataclass
class PerpetualTrade:
    side: str
    entry_time: str
    exit_time: str
    quantity: float
    leverage: float
    margin: float
    entry_price: float
    exit_price: float
    liquidation_price_estimate: float
    exit_reason: str
    gross_pnl: float
    fees: float
    funding: float
    net_pnl: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PerpetualBacktestResult:
    initial_equity: float
    final_equity: float
    trades: list[PerpetualTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    assumptions: dict = field(default_factory=dict)

    def metrics(self) -> dict:
        pnls = [t.net_pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        peak, drawdown = self.initial_equity, 0.0
        for equity in self.equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                drawdown = max(drawdown, (peak - equity) / peak)
        gross_loss = abs(sum(losses))
        return {
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "net_pnl": self.final_equity - self.initial_equity,
            "net_return_pct": (self.final_equity / self.initial_equity - 1) * 100,
            "trades": len(pnls),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": len(wins) / len(pnls) * 100 if pnls else None,
            "profit_factor": sum(wins) / gross_loss if gross_loss else ("infinite" if wins else None),
            "max_drawdown_pct": drawdown * 100,
            "liquidations": sum(t.exit_reason == "liquidation" for t in self.trades),
            "rejections": dict(self.rejections),
        }


def _validate(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    if not isinstance(frame, pd.DataFrame) or frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"{name} must contain non-empty OHLC data")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError(f"{name} index must be timezone-aware")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError(f"{name} timestamps must be strictly increasing and unique")
    data = frame.loc[:, ["open", "high", "low", "close"]].astype(float)
    if data.isna().any().any() or not data.map(math.isfinite).all().all() or (data <= 0).any().any():
        raise ValueError(f"{name} prices must be finite and positive")
    if (data.high < data[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError(f"{name} high is inconsistent")
    if (data.low > data[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError(f"{name} low is inconsistent")
    return data


def _features(data: pd.DataFrame, cfg: PerpetualEMAStrategy) -> pd.DataFrame:
    x = data.copy()
    prev = x.close.shift(1)
    tr = pd.concat([x.high - x.low, (x.high - prev).abs(), (x.low - prev).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(cfg.atr_period, min_periods=cfg.atr_period).mean()
    for key, span in (("fast", cfg.fast), ("slow", cfg.slow), ("trend", cfg.trend)):
        x[key] = x.close.ewm(span=span, adjust=False, min_periods=span).mean()
    up, down = x.high.diff(), -x.low.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    alpha = 1 / cfg.atr_period
    smooth_tr = tr.ewm(alpha=alpha, adjust=False, min_periods=cfg.atr_period).mean().replace(0, math.nan)
    pdi = 100 * plus.ewm(alpha=alpha, adjust=False, min_periods=cfg.atr_period).mean() / smooth_tr
    mdi = 100 * minus.ewm(alpha=alpha, adjust=False, min_periods=cfg.atr_period).mean() / smooth_tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, math.nan)
    x["adx"] = dx.ewm(alpha=alpha, adjust=False, min_periods=cfg.atr_period).mean()
    adx_ok = x.adx >= cfg.min_adx if cfg.min_adx is not None else True
    if cfg.signal_mode == "bollinger_reversion":
        center = x.close.rolling(
            cfg.mean_reversion_period,
            min_periods=cfg.mean_reversion_period,
        ).mean()
        deviation = x.close.rolling(
            cfg.mean_reversion_period,
            min_periods=cfg.mean_reversion_period,
        ).std(ddof=0)
        x["mean_target"] = center
        x["mean_lower"] = center - cfg.mean_reversion_std * deviation
        x["mean_upper"] = center + cfg.mean_reversion_std * deviation
        delta = x.close.diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        avg_gain = gains.ewm(
            alpha=1 / cfg.mean_reversion_rsi_period,
            adjust=False,
            min_periods=cfg.mean_reversion_rsi_period,
        ).mean()
        avg_loss = losses.ewm(
            alpha=1 / cfg.mean_reversion_rsi_period,
            adjust=False,
            min_periods=cfg.mean_reversion_rsi_period,
        ).mean()
        denominator = (avg_gain + avg_loss).replace(0, math.nan)
        x["rsi"] = (100.0 * avg_gain / denominator).fillna(50.0)
    x["signal"] = 0
    if cfg.signal_mode == "ema_cross":
        x.loc[
            (x.fast > x.slow)
            & (x.fast.shift(1) <= x.slow.shift(1))
            & (x.close > x.trend)
            & adx_ok,
            "signal",
        ] = 1
        x.loc[
            (x.fast < x.slow)
            & (x.fast.shift(1) >= x.slow.shift(1))
            & (x.close < x.trend)
            & adx_ok,
            "signal",
        ] = -1
    else:
        # Previous-window channels are shifted before comparing the current
        # completed close, preventing the current candle from entering its own
        # breakout threshold.
        prior_high = x.high.rolling(
            cfg.breakout_period, min_periods=cfg.breakout_period
        ).max().shift(1)
        prior_low = x.low.rolling(
            cfg.breakout_period, min_periods=cfg.breakout_period
        ).min().shift(1)
        x["prior_channel_high"] = prior_high
        x["prior_channel_low"] = prior_low
        x.loc[(x.close > prior_high) & (x.close > x.trend) & adx_ok, "signal"] = 1
        x.loc[(x.close < prior_low) & (x.close < x.trend) & adx_ok, "signal"] = -1
    else:
        # Enter only after price closes back inside the prior extreme band and
        # RSI turns with it. Freeze the signal candle's mid-band before the
        # next-open fill. The ADX ceiling avoids fading strong trends.
        range_ok = x.adx <= cfg.mean_reversion_max_adx
        long_reentry = (
            (x.close.shift(1) <= x.mean_lower.shift(1))
            & (x.close > x.mean_lower)
            & (x.rsi.shift(1) <= cfg.mean_reversion_rsi_long)
            & (x.rsi > x.rsi.shift(1))
            & range_ok
        )
        short_reentry = (
            (x.close.shift(1) >= x.mean_upper.shift(1))
            & (x.close < x.mean_upper)
            & (x.rsi.shift(1) >= cfg.mean_reversion_rsi_short)
            & (x.rsi < x.rsi.shift(1))
            & range_ok
        )
        x.loc[long_reentry, "signal"] = 1
        x.loc[short_reentry, "signal"] = -1
    return x


def run_backtest(
    traded: pd.DataFrame,
    mark: pd.DataFrame,
    funding_rates: pd.Series,
    instrument: LinearPerpetualInstrument,
    *,
    initial_equity: float = 1_000.0,
    strategy: PerpetualEMAStrategy | None = None,
    costs: PerpetualCosts | None = None,
    trade_start_index: int = 1,
) -> PerpetualBacktestResult:
    """Simulate a single symbol; missing or misaligned funding/mark data fail closed.

    trade_start_index lets a walk-forward fold use preceding candles only for
    indicator warm-up. No signal is accepted before this index.
    """
    cfg, fees = strategy or PerpetualEMAStrategy(), costs or PerpetualCosts()
    if not cfg.enabled:
        raise ValueError("strategy is disabled by default; enable explicitly for research")
    if not math.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial_equity must be finite and positive")
    if isinstance(trade_start_index, bool) or not isinstance(trade_start_index, int):
        raise ValueError("trade_start_index must be an integer")
    if cfg.risk.leverage > instrument.max_leverage or cfg.risk.leverage < instrument.min_leverage:
        raise ValueError("configured leverage is outside current instrument limits")
    trade = _validate(traded, "traded candles")
    marks = _validate(mark, "mark candles")
    if trade_start_index < 1 or trade_start_index >= len(trade):
        raise ValueError("trade_start_index must be within the candle window")
    trade.index = trade.index.tz_convert("UTC")
    marks.index = marks.index.tz_convert("UTC")
    if not trade.index.equals(marks.index):
        raise ValueError("traded and mark candles must have identical timestamps")
    delta = trade.index.to_series().diff().dropna()
    if delta.empty or not (delta == pd.Timedelta(minutes=cfg.interval_minutes)).all():
        raise ValueError("candles must be contiguous completed 4-hour bars")
    if not isinstance(funding_rates, pd.Series) or funding_rates.empty or not isinstance(funding_rates.index, pd.DatetimeIndex) or funding_rates.index.tz is None:
        raise ValueError("complete timezone-aware historical funding series is required")
    rates = funding_rates.astype(float).copy()
    rates.index = rates.index.tz_convert("UTC")
    if rates.index.has_duplicates or not rates.index.is_monotonic_increasing or rates.isna().any() or not rates.map(math.isfinite).all() or (rates.abs() > 0.05).any():
        raise ValueError("funding rates must be finite, unique, ordered and within +/-5%")
    funding_gap = pd.Timedelta(minutes=instrument.funding_interval_minutes)
    if len(rates) > 1 and not (rates.index.to_series().diff().dropna() == funding_gap).all():
        raise ValueError("funding history contains missing or irregular settlement events")
    if rates.index[0] - trade.index[0] > funding_gap or trade.index[-1] - rates.index[-1] > funding_gap:
        raise ValueError("funding history does not cover the candle window within one venue settlement interval")
    if not (rates.index.isin(trade.index)).all():
        raise ValueError("each funding event must align exactly to a candle open")
    if len(trade) < cfg.trend + cfg.atr_period + 3:
        raise ValueError("insufficient candles for EMA/ATR warm-up")

    features = _features(trade, cfg)
    cash = float(initial_equity)
    trades: list[PerpetualTrade] = []
    curve: list[float] = [cash]
    rejected: dict[str, int] = {}
    position: dict | None = None
    pending_exit: str | None = None
    pending_signal, pending_atr = 0, math.nan
    pending_target = math.nan
    peak = cash
    day_start = cash
    day = trade.index[0].date()
    daily_halt = global_halt = False
    consecutive_losses = 0

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    def close(raw_price: float, stamp: pd.Timestamp, reason: str) -> None:
        nonlocal cash, position, consecutive_losses, global_halt, pending_exit
        if position is None:
            return
        # A queued risk exit belongs only to the position that triggered it.
        # Clear it on every closure so it cannot leak into a later position.
        pending_exit = None
        direction, qty = position["direction"], position["qty"]
        fill = raw_price * (1 - direction * fees.impact_rate)
        exit_fee = qty * fill * fees.fee_rate
        gross = direction * (fill - position["entry"]) * qty
        funding = position["funding"]
        liq_fee = qty * fill * cfg.risk.liquidation_fee_rate if reason == "liquidation" else 0.0
        net = gross - position["entry_fee"] - exit_fee - liq_fee + funding
        cash += gross - exit_fee - liq_fee
        total_fee = position["entry_fee"] + exit_fee + liq_fee
        trades.append(PerpetualTrade(
            side="long" if direction > 0 else "short",
            entry_time=position["time"], exit_time=stamp.isoformat(), quantity=qty,
            leverage=cfg.risk.leverage, margin=position["margin"], entry_price=position["entry"],
            exit_price=fill, liquidation_price_estimate=position["liq"], exit_reason=reason,
            gross_pnl=gross, fees=total_fee, funding=funding, net_pnl=net,
        ))
        consecutive_losses = consecutive_losses + 1 if net < 0 else 0
        if consecutive_losses >= cfg.risk.max_consecutive_losses:
            global_halt = True
        position = None

    for i in range(1, len(trade)):
        stamp, row, mr = trade.index[i], trade.iloc[i], marks.iloc[i]
        if stamp.date() != day:
            day, day_start, daily_halt = stamp.date(), curve[-1], False
        if stamp in rates.index and position is not None:
            amount = linear_funding_cashflow(direction=position["direction"], quantity=position["qty"], mark_price=float(mr.open), funding_rate=float(rates.loc[stamp]))
            cash += amount
            position["funding"] += amount

        closed_intrabar = False
        if position is not None:
            d, liq = position["direction"], position["liq"]
            # A mark-price liquidation already breached at the bar open has priority.
            opened_beyond_liq = (d > 0 and mr.open <= liq) or (d < 0 and mr.open >= liq)
            if opened_beyond_liq:
                raw = min(float(mr.open), liq) if d > 0 else max(float(mr.open), liq)
                close(raw, stamp, "liquidation")
                closed_intrabar = True
            elif pending_exit:
                close(float(row.open), stamp, pending_exit)
                pending_exit = None
            elif pending_signal and pending_signal != position["direction"]:
                close(float(row.open), stamp, "ema_reversal")
            elif i - position["index"] >= cfg.max_holding_bars:
                close(float(row.open), stamp, "max_holding_bars")
                pending_signal, pending_atr = 0, math.nan
            elif (d > 0 and mr.low <= liq) or (d < 0 and mr.high >= liq):
                close(liq, stamp, "liquidation")
                closed_intrabar = True
            elif (d > 0 and row.low <= position["stop"]) or (d < 0 and row.high >= position["stop"]):
                raw = min(float(row.open), position["stop"]) if d > 0 and row.open <= position["stop"] else max(float(row.open), position["stop"]) if d < 0 and row.open >= position["stop"] else position["stop"]
                close(raw, stamp, "atr_stop")
                closed_intrabar = True
            elif cfg.signal_mode == "bollinger_reversion" and (
                (d > 0 and row.high >= position["target"])
                or (d < 0 and row.low <= position["target"])
            ):
                target = float(position["target"])
                raw = max(float(row.open), target) if d > 0 else min(float(row.open), target)
                close(raw, stamp, "mean_reversion_target")
                closed_intrabar = True
        if closed_intrabar:
            # A stop/liquidation within this candle cannot be followed by a
            # re-entry priced at the same candle's open. Wait for a fresh signal.
            pending_signal, pending_atr = 0, math.nan

        if position is None and pending_signal and not daily_halt and not global_halt:
            raw_entry = float(row.open)
            direction = pending_signal
            fill = raw_entry * (1 + direction * fees.impact_rate)
            stop = fill - direction * cfg.atr_multiple * pending_atr
            target = (
                float(pending_target)
                if cfg.signal_mode == "bollinger_reversion" and math.isfinite(pending_target)
                else None
            )
            target_is_valid = (
                cfg.signal_mode != "bollinger_reversion"
                or (
                    target is not None
                    and ((direction > 0 and target > fill) or (direction < 0 and target < fill))
                )
            )
            if not target_is_valid:
                reject("target_not_favorable_after_gap")
            else:
                sizing = size_linear_position(
                    equity=max(cash, 1e-9), entry_price=fill, stop_price=stop,
                    direction=direction, cost_rate_per_fill=fees.per_fill_rate,
                    instrument=instrument, risk=cfg.risk,
                )
                if not sizing.approved:
                    reject(sizing.rejection_reason or "sizing_rejected")
                else:
                    liq = estimate_isolated_liquidation_price(
                        entry_price=fill,
                        leverage=cfg.risk.leverage,
                        maintenance_margin_rate=cfg.risk.maintenance_margin_rate,
                        direction=direction,
                    )
                    if (direction > 0 and liq >= fill) or (direction < 0 and liq <= fill):
                        reject("invalid_liquidation_estimate")
                    else:
                        buffer = cfg.risk.liquidation_buffer_fraction
                        liquidation_clears_stop = (
                            liq < stop * (1.0 - buffer)
                            if direction > 0
                            else liq > stop * (1.0 + buffer)
                        )
                        if not liquidation_clears_stop:
                            reject("liquidation_too_close_to_atr_stop")
                        else:
                            entry_fee = sizing.notional * fees.fee_rate
                            cash -= entry_fee
                            position = {"direction": direction, "qty": sizing.quantity, "entry": fill,
                                "time": stamp.isoformat(), "index": i, "entry_fee": entry_fee,
                                "margin": sizing.initial_margin, "stop": stop, "liq": liq,
                                "target": target, "funding": 0.0}
            pending_signal, pending_atr, pending_target = 0, math.nan, math.nan

        if position is not None:
            d = position["direction"]
            mtm = cash + d * (float(mr.close) - position["entry"]) * position["qty"]
            atr_now = features.iloc[i].atr
            if cfg.signal_mode != "bollinger_reversion" and pd.notna(atr_now) and atr_now > 0:
                candidate = float(row.close) - d * cfg.atr_multiple * float(atr_now)
                position["stop"] = max(position["stop"], candidate) if d > 0 else min(position["stop"], candidate)
            if (
                cfg.signal_mode == "bollinger_reversion"
                and pending_exit is None
                and pd.notna(features.iloc[i].adx)
                and float(features.iloc[i].adx) > cfg.mean_reversion_max_adx
            ):
                pending_exit = "trend_regime_exit"
        else:
            mtm = cash
        if not math.isfinite(mtm) or mtm <= 0:
            raise RuntimeError("perpetual simulation equity became non-positive")
        curve.append(mtm)
        peak = max(peak, mtm)
        if mtm <= day_start * (1 - cfg.risk.max_daily_loss_fraction):
            daily_halt, pending_signal, pending_atr = True, 0, math.nan
            if position is not None:
                pending_exit = "daily_loss_limit"
        if mtm <= peak * (1 - cfg.risk.max_drawdown_fraction):
            global_halt, pending_signal, pending_atr = True, 0, math.nan
            if position is not None:
                pending_exit = "max_drawdown_limit"
        if i >= trade_start_index and not global_halt and not daily_halt:
            sig = int(features.iloc[i].signal)
            if sig:
                pending_signal, pending_atr = sig, float(features.iloc[i].atr)
                if cfg.signal_mode == "bollinger_reversion":
                    pending_target = float(features.iloc[i].mean_target)

    if position is not None:
        close(float(trade.iloc[-1].close), trade.index[-1], "end_of_window")
        curve[-1] = cash
    return PerpetualBacktestResult(
        initial_equity=initial_equity, final_equity=cash, trades=trades,
        equity_curve=curve, rejections=rejected,
        assumptions={"instrument": instrument.symbol, "contract": "linear USDT perpetual", "margin_mode": "isolated",
            "position_mode": "one-way", "leverage": cfg.risk.leverage, "research_leverage_ceiling": 2.0,
            "strategy": (
                "EMA14/EMA50 cross + EMA200 trend + ADX, ATR trailing stop"
                if cfg.signal_mode == "ema_cross"
                else (
                    f"Donchian {cfg.breakout_period} breakout + EMA{cfg.trend} regime + ADX, ATR trailing stop"
                    if cfg.signal_mode == "donchian_breakout"
                    else (
                        f"Bollinger({cfg.mean_reversion_period}, {cfg.mean_reversion_std:g}) re-entry + "
                        f"RSI({cfg.mean_reversion_rsi_period}) recovery + ADX <= "
                        f"{cfg.mean_reversion_max_adx:g}; frozen mid-band target, ATR stop"
                    )
                )
            ),
            "signal_timing": "completed 4h candle, next candle open", "funding_series_required": True,
            "mark_price_liquidation": True, "liquidation_estimate": "approximate, not Bybit tier formula",
            "liquidation_buffer_fraction": cfg.risk.liquidation_buffer_fraction,
            "demo_ready": False},
    )
