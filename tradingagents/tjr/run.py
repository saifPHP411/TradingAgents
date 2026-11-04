"""Run the TJR backtest over fetched intraday frames and print the report.

Offline once the frames exist: it reads the CSV files ``fetch_data`` wrote and
never touches the network. The 5-minute series is the vendor's if it was fetched,
otherwise it is resampled from the 1-minute series — the coarser bars are always
built from bars that had closed, so the closed-bar rule survives either path.

Usage:

    python -m tradingagents.tjr.fetch_data --symbols AAPL MSFT --days 20
    python -m tradingagents.tjr.run --symbols AAPL MSFT --out docs/tjr_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import TjrBacktester, TjrConfig
from .fetch_data import DEFAULT_OUT, DEFAULT_SYMBOLS, load_frames
from .frames import IntradayFrameError, IntradaySeries, resample
from .report import build_report


def build_run(
    data_dir: Path,
    symbols: tuple[str, ...],
    *,
    initial_cash: float = 100_000.0,
    allow_short: bool = False,
) -> dict:
    """Load the frames, run the backtest, and return the report as a dict."""
    one_minute_frames = load_frames(data_dir, symbols, "1m")
    one_minute = {
        symbol: IntradaySeries(symbol, "1m", frame) for symbol, frame in one_minute_frames.items()
    }
    five_minute: dict[str, IntradaySeries] = {}
    for symbol, series in one_minute.items():
        try:
            fetched = load_frames(data_dir, (symbol,), "5m")
        except FileNotFoundError:
            fetched = {}
        five_minute[symbol] = (
            IntradaySeries(symbol, "5m", fetched[symbol])
            if symbol in fetched
            else resample(series, "5m")
        )
    hourly = _resample_if_possible(five_minute, "60m")
    four_hour = _resample_if_possible(five_minute, "4h")

    config = TjrConfig(initial_cash=initial_cash, allow_short=allow_short)
    backtester = TjrBacktester(
        one_minute=one_minute,
        five_minute=five_minute,
        hourly=hourly,
        four_hour=four_hour,
        config=config,
    )
    result = backtester.run()
    report = build_report(result)
    payload = report.to_dict()
    payload["trades"] = [
        {
            "symbol": trade.symbol,
            "session": trade.session,
            "side": trade.side,
            "quantity": trade.quantity,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "exit_reason": trade.exit_reason,
            "net_pnl": trade.net_pnl,
            "reason_code": trade.reason_code,
        }
        for trade in result.trades
    ]
    payload["rejections"] = result.rejections
    return payload


def _resample_if_possible(
    five_minute: dict[str, IntradaySeries], interval: str
) -> dict[str, IntradaySeries]:
    """Resample each symbol, dropping any that leaves no complete bucket.

    A short window may not fill a single 60m or 4h bucket. That is an absence of a
    level source, not an error: the rule chain simply has no hourly or 4h level for
    that symbol, which is the honest state of a study this short.
    """
    out: dict[str, IntradaySeries] = {}
    for symbol, series in five_minute.items():
        try:
            out[symbol] = resample(series, interval)
        except IntradayFrameError:
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the TJR backtest over fetched frames.")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--data", type=Path, default=DEFAULT_OUT, help="directory of fetched frames")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    payload = build_run(args.data, tuple(args.symbols), initial_cash=args.initial_cash)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - a maintainer entry point
    raise SystemExit(main())
