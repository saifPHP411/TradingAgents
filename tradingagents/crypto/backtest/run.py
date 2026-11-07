"""Run the crypto TJR study over the cached bars and write its report. Offline.

Reads only the bars ``...backtest.fetch`` cached; it never touches the network. It
resolves the universe, builds every interval series by resampling the cached
1-minute bars, runs the study, and writes a JSON report and a Markdown summary.

    python -m tradingagents.crypto.backtest.run --out docs/crypto_tjr_study.json
    python -m tradingagents.crypto.backtest.run --periods 7 30 --no-sensitivity

The report states its own coverage, its session policy and its entry cut, so a
reader can tell what was actually run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..binance import BinancePublicClient
from ..symbols import parse_pair
from ..types import CANDLE_INTERVALS
from ..universe import DEFAULT_SYMBOLS, resolve_universe
from .history import HistoryCache
from .report import render_markdown
from .study import (
    STANDARD_PERIODS,
    CryptoBacktestConfig,
    run_study,
    series_from_minutes,
)


def load_cached_series(
    cache: HistoryCache,
    symbols: tuple[str, ...],
    *,
    intervals: tuple[str, ...] = CANDLE_INTERVALS,
):
    """Build every interval series for ``symbols`` from the cached 1-minute bars."""
    from tradingagents.tjr.sessions import utc_day_session

    policy = utc_day_session()
    series: dict[str, dict] = {}
    coverage: dict[str, dict] = {}
    for raw in symbols:
        symbol = parse_pair(raw).canonical
        minutes = cache.load(symbol, "1m")
        if minutes is None or minutes.empty:
            continue
        series[symbol] = series_from_minutes(symbol, minutes, policy, intervals=intervals)
        coverage[symbol] = {
            "bars": len(minutes),
            "first": str(minutes.index[0]),
            "last": str(minutes.index[-1]),
        }
    return series, coverage


def build_study(
    *,
    symbols: tuple[str, ...] | None = None,
    cache_dir: str | None = None,
    initial_cash: float = 100_000.0,
    periods: tuple[int, ...] = STANDARD_PERIODS,
    allow_substitution: bool = True,
    include_sensitivity: bool = True,
    client: BinancePublicClient | None = None,
):
    """Resolve the universe, load the cache, and run the study."""
    client = client or BinancePublicClient()
    registry = client.exchange_info()
    requested = symbols or DEFAULT_SYMBOLS
    resolution = resolve_universe(
        registry,
        tuple(parse_pair(s).canonical for s in requested),
        allow_substitution=allow_substitution,
    )
    cache = HistoryCache(cache_dir) if cache_dir else HistoryCache()
    series, coverage = load_cached_series(cache, resolution.symbols)
    if not series:
        raise SystemExit(
            "no cached 1m bars found; run `python -m tradingagents.crypto.backtest.fetch` first"
        )
    config = CryptoBacktestConfig(
        symbols=tuple(sorted(series)), initial_cash=initial_cash, periods=periods
    )
    return run_study(
        series,
        config,
        coverage=coverage,
        universe=resolution.to_dict(),
        include_sensitivity=include_sensitivity,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the crypto TJR study over cached bars.")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--periods", nargs="+", type=int, default=list(STANDARD_PERIODS))
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON study here")
    parser.add_argument("--markdown", type=Path, default=None, help="write a Markdown summary here")
    args = parser.parse_args(argv)

    study = build_study(
        symbols=tuple(args.symbols) if args.symbols else None,
        cache_dir=args.cache,
        initial_cash=args.initial_cash,
        periods=tuple(args.periods),
        include_sensitivity=not args.no_sensitivity,
    )
    payload = study.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(study), encoding="utf-8")
        print(f"wrote {args.markdown}", flush=True)
    if args.out is None and args.markdown is None:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - a maintainer entry point
    raise SystemExit(main())
