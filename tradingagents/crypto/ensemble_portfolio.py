"""Shared-equity, multi-symbol portfolio backtest for the RAE research system.

All symbols trade one cash account. It applies position-count, altcoin
concentration, gross-exposure, daily-loss and peak-drawdown entry gates. It is
research-only and is not connected to any broker or worker.
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
from .ensemble_backtest import BacktestSettings, _validate


@dataclass(frozen=True)
class PortfolioSettings:
    max_open_positions: int = 2
    max_alt_positions: int = 1
    max_gross_exposure_fraction: float = 0.25
    daily_loss_limit_fraction: float | None = 0.02
    max_drawdown_fraction: float | None = 0.10


@dataclass
class PortfolioBacktestResult:
    mode: str
    symbols: list[str]
    settings: dict
    portfolio_settings: dict
    final_equity: float
    net_return_pct: float
    max_drawdown_pct: float
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    profit_factor: float | None
    fees_paid: float
    slippage_paid: float
    blocked_entries: dict[str, int]
    equity_curve: list[dict]
    trade_log: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_portfolio_settings(settings: PortfolioSettings) -> None:
    if type(settings.max_open_positions) is not int or settings.max_open_positions < 1:
        raise ValueError("max_open_positions must be a positive integer")
    if type(settings.max_alt_positions) is not int or settings.max_alt_positions < 1:
        raise ValueError("max_alt_positions must be a positive integer")
    if (
        not isfinite(settings.max_gross_exposure_fraction)
        or not 0 < settings.max_gross_exposure_fraction <= 1
    ):
        raise ValueError("max gross exposure fraction must be in (0, 1]")
    for name, value in (
        ("daily loss", settings.daily_loss_limit_fraction),
        ("maximum drawdown", settings.max_drawdown_fraction),
    ):
        if value is not None and (not isfinite(value) or not 0 < value < 1):
            raise ValueError(f"{name} limit must be in (0, 1) or None")


def _btc_risk_off(indicators: pd.DataFrame, position: int, adx_min: float) -> bool:
    row = indicators.iloc[position]
    required = ("close", "fast_ema", "slow_ema", "momentum", "plus_di", "minus_di", "adx")
    if any(pd.isna(row[key]) for key in required):
        return True
    return not (
        row.close > row.slow_ema
        and row.fast_ema > row.slow_ema
        and row.momentum > 0
        and row.plus_di > row.minus_di
        and row.adx >= adx_min
    )


def run_portfolio_backtest(
    frames: dict[str, pd.DataFrame],
    *,
    btc_symbol: str = "BTC/USDT",
    mode: str = "ensemble",
    signal_settings: EnsembleSettings | None = None,
    settings: BacktestSettings = BacktestSettings(),
    portfolio_settings: PortfolioSettings = PortfolioSettings(),
) -> PortfolioBacktestResult:
    """Backtest a shared-cash spot portfolio using only completed 1h bars.

    Pending decisions are filled at the following bar open. BTC trend state
    blocks new altcoin risk and schedules existing altcoin exits for the next
    open. The simulator is long-only and charges both-side fees and slippage.
    """
    allowed_modes = {"ensemble", "trend", "trend_pullback", "range", "breakout"}
    if mode not in allowed_modes:
        raise ValueError("unsupported strategy mode")
    _validate_portfolio_settings(portfolio_settings)

    numeric_settings = (
        settings.initial_cash,
        settings.risk_fraction,
        settings.max_notional_fraction,
        settings.fee_bps,
        settings.slippage_bps,
    )
    if (
        type(settings.max_hold_bars) is not int
        or settings.max_hold_bars < 1
        or type(settings.warmup_bars) is not int
        or settings.warmup_bars < 220
    ):
        raise ValueError("hold and warm-up bars must be positive integers")
    if not all(isfinite(float(value)) for value in numeric_settings):
        raise ValueError("backtest settings must be finite numbers")
    if settings.initial_cash <= 0 or not 0 < settings.risk_fraction <= 0.05:
        raise ValueError("initial cash and risk fraction are outside safe bounds")
    if not 0 < settings.max_notional_fraction <= 1:
        raise ValueError("max notional fraction must be in (0, 1]")
    if settings.fee_bps < 0 or settings.slippage_bps < 0:
        raise ValueError("fees and slippage cannot be negative")

    signal_settings = signal_settings or EnsembleSettings()
    _validate_settings(signal_settings)
    if not frames or btc_symbol not in frames:
        raise ValueError(f"frames must include the BTC risk-filter symbol {btc_symbol}")

    symbols = sorted(frames)
    data = {symbol: _validate(frames[symbol], settings) for symbol in symbols}
    index = data[btc_symbol].index
    for symbol in symbols:
        if not data[symbol].index.equals(index):
            raise ValueError("all portfolio symbols must have identical hourly timestamps")

    indicators = {
        symbol: _indicators(frame, signal_settings)
        for symbol, frame in data.items()
    }

    cash = float(settings.initial_cash)
    positions: dict[str, dict] = {}
    pending: dict[str, dict] = {}
    cooldown_until = {symbol: -1 for symbol in symbols}
    trades: list[dict] = []
    curve: list[dict] = []
    blocked = {
        "max_open_positions": 0,
        "max_alt_positions": 0,
        "gross_exposure": 0,
        "daily_loss_limit": 0,
        "max_drawdown": 0,
        "btc_risk_filter": 0,
        "insufficient_cash_or_risk": 0,
    }
    total_fees = 0.0
    total_slippage = 0.0
    fee_rate = settings.fee_bps / 10_000.0
    slip = settings.slippage_bps / 10_000.0
    peak_equity = settings.initial_cash
    max_drawdown = 0.0
    daily_start_equity = settings.initial_cash
    current_day = None
    daily_blocked = False
    drawdown_blocked = False
    previous_equity = settings.initial_cash

    def close_position(symbol: str, timestamp, raw_price: float, reason: str) -> None:
        nonlocal cash, total_fees, total_slippage
        position = positions.pop(symbol, None)
        if position is None:
            return
        fill = raw_price * (1.0 - slip)
        fee = position["qty"] * fill * fee_rate
        exit_slippage = position["qty"] * (raw_price - fill)
        total_slippage += position["entry_slippage"] + exit_slippage
        total_fees += fee
        cash += position["qty"] * fill - fee
        net_pnl = (
            position["qty"] * (fill - position["entry"])
            - position["entry_fee"]
            - fee
        )
        trades.append({
            "symbol": symbol,
            "entry_time": position["entry_time"],
            "exit_time": timestamp.isoformat(),
            "entry_price": position["entry"],
            "raw_entry_price": position["raw_entry"],
            "exit_price": fill,
            "raw_exit_price": raw_price,
            "quantity": position["qty"],
            "gross_pnl": position["qty"] * (raw_price - position["raw_entry"]),
            "net_pnl": net_pnl,
            "entry_fee": position["entry_fee"],
            "exit_fee": fee,
            "slippage_paid": position["entry_slippage"] + exit_slippage,
            "reason": reason,
        })

    for j in range(settings.warmup_bars, len(index)):
        timestamp = index[j]

        # Execute all exits before entries so released cash/exposure is reusable.
        for symbol in symbols:
            action = pending.get(symbol)
            if action and action["action"] == "exit" and symbol in positions:
                close_position(symbol, timestamp, float(data[symbol].open.iloc[j]), action["reason"])
                pending.pop(symbol, None)

        # Deterministic symbol order prevents a simultaneous signal from
        # exceeding portfolio caps. BTC sorts before most altcoins.
        for symbol in symbols:
            action = pending.get(symbol)
            if not action or action["action"] != "entry":
                continue
            pending.pop(symbol, None)
            if symbol in positions:
                continue
            if len(positions) >= portfolio_settings.max_open_positions:
                blocked["max_open_positions"] += 1
                continue
            alt_count = sum(1 for held in positions if held != btc_symbol)
            if symbol != btc_symbol and alt_count >= portfolio_settings.max_alt_positions:
                blocked["max_alt_positions"] += 1
                continue

            entry_raw = float(data[symbol].open.iloc[j])
            entry = entry_raw * (1.0 + slip)
            stop = float(action["stop"])
            risk_per_unit = entry - stop
            if risk_per_unit <= 0:
                blocked["insufficient_cash_or_risk"] += 1
                continue

            equity_at_open = cash + sum(
                position["qty"] * float(data[held].open.iloc[j])
                for held, position in positions.items()
            )
            gross_exposure = sum(
                position["qty"] * float(data[held].open.iloc[j])
                for held, position in positions.items()
            )
            exposure_capacity = (
                equity_at_open * portfolio_settings.max_gross_exposure_fraction
                - gross_exposure
            )
            if exposure_capacity <= 0:
                blocked["gross_exposure"] += 1
                continue

            risk_budget = equity_at_open * settings.risk_fraction
            notional_cap = equity_at_open * settings.max_notional_fraction
            cash_cap_qty = cash / (entry * (1.0 + fee_rate))
            qty = min(
                risk_budget / risk_per_unit,
                notional_cap / entry,
                exposure_capacity / entry,
                cash_cap_qty,
            )
            if qty <= 0 or not isfinite(qty):
                blocked["insufficient_cash_or_risk"] += 1
                continue

            entry_fee = qty * entry * fee_rate
            entry_slippage = qty * (entry - entry_raw)
            cash -= qty * entry + entry_fee
            total_fees += entry_fee
            positions[symbol] = {
                "entry": entry,
                "raw_entry": entry_raw,
                "entry_time": timestamp.isoformat(),
                "entry_bar": j,
                "qty": qty,
                "stop": stop,
                "entry_fee": entry_fee,
                "entry_slippage": entry_slippage,
            }

        # Intrabar protective stops, including positions opened at this bar's
        # open. A gap through the stop is filled at the worse opening price.
        for symbol in list(positions):
            position = positions[symbol]
            if float(data[symbol].low.iloc[j]) <= position["stop"]:
                raw_stop = min(float(data[symbol].open.iloc[j]), position["stop"])
                close_position(symbol, timestamp, raw_stop, "atr_stop")
                cooldown_until[symbol] = j + 1

        btc_risk_off = _btc_risk_off(
            indicators[btc_symbol], j, signal_settings.adx_trend_min
        )

        # Time, trend and cross-asset exits are decisions at this completed
        # close; they are submitted for the next bar's open.
        for symbol, position in positions.items():
            if j - position["entry_bar"] >= settings.max_hold_bars:
                pending[symbol] = {"action": "exit", "reason": "max_hold_bars"}
                continue
            if symbol != btc_symbol and btc_risk_off:
                pending[symbol] = {"action": "exit", "reason": "btc_risk_filter"}
                continue
            signal = _evaluate_indicator_row(indicators[symbol], j, signal_settings, mode)
            if signal.should_exit:
                pending[symbol] = {"action": "exit", "reason": "signal_flat"}

        # Mark to market at the completed bar close.
        equity = cash + sum(
            position["qty"] * float(data[symbol].close.iloc[j])
            for symbol, position in positions.items()
        )
        day = timestamp.date()
        if current_day != day:
            current_day = day
            daily_start_equity = previous_equity
            daily_blocked = False
        if (
            portfolio_settings.daily_loss_limit_fraction is not None
            and equity <= daily_start_equity * (1.0 - portfolio_settings.daily_loss_limit_fraction)
        ):
            daily_blocked = True

        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)
        if (
            portfolio_settings.max_drawdown_fraction is not None
            and drawdown >= portfolio_settings.max_drawdown_fraction
        ):
            drawdown_blocked = True

        curve.append({"time": timestamp.isoformat(), "equity": equity})
        previous_equity = equity

        if j == len(index) - 1:
            for symbol in list(positions):
                close_position(symbol, timestamp, float(data[symbol].close.iloc[j]), "end_of_data")
            curve[-1]["equity"] = cash
            break

        # Generate fresh entries only after all risk gates have evaluated the
        # completed bar. Pending entries are always filled on the next bar.
        for symbol in symbols:
            if symbol in positions or symbol in pending or j <= cooldown_until[symbol]:
                continue
            if daily_blocked:
                blocked["daily_loss_limit"] += 1
                continue
            if drawdown_blocked:
                blocked["max_drawdown"] += 1
                continue
            if symbol != btc_symbol and btc_risk_off:
                blocked["btc_risk_filter"] += 1
                continue
            signal = _evaluate_indicator_row(indicators[symbol], j, signal_settings, mode)
            if signal.action == "long" and signal.stop is not None:
                pending[symbol] = {"action": "entry", "stop": signal.stop}

    peak = settings.initial_cash
    max_drawdown = 0.0
    for point in curve:
        value = float(point["equity"])
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak)
    wins = [trade["net_pnl"] for trade in trades if trade["net_pnl"] > 0]
    losses = [trade["net_pnl"] for trade in trades if trade["net_pnl"] < 0]
    net_loss = abs(sum(losses))
    profit_factor = sum(wins) / net_loss if net_loss > 0 else None
    return PortfolioBacktestResult(
        mode=mode,
        symbols=symbols,
        settings={**asdict(settings), "signal_settings": asdict(signal_settings)},
        portfolio_settings=asdict(portfolio_settings),
        final_equity=cash,
        net_return_pct=(cash / settings.initial_cash - 1.0) * 100.0,
        max_drawdown_pct=max_drawdown * 100.0,
        trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=(len(wins) / len(trades) * 100.0) if trades else 0.0,
        profit_factor=profit_factor,
        fees_paid=total_fees,
        slippage_paid=total_slippage,
        blocked_entries=blocked,
        equity_curve=curve,
        trade_log=trades,
    )
