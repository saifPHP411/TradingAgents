"""Fetch a reproducible 1-hour OHLCV history for the RAE research study.

Public Binance market data only; no API key and no trading endpoint. The cache
contains completed candles only. Defaults target roughly two years per symbol.
"""
from __future__ import annotations  # noqa: I001

import argparse
import json

from .binance import BinancePublicClient
from .backtest.history import HistoryCache
from .symbols import parse_pair
from .universe import DEFAULT_SYMBOLS


def fetch_ensemble_history(
    symbols: tuple[str, ...],
    *,
    bars: int = 17_520,
    cache_dir: str | None = None,
    client: BinancePublicClient | None = None,
) -> dict:
    if bars < 1:
        raise ValueError("bars must be positive")
    client = client or BinancePublicClient()
    cache = HistoryCache(cache_dir) if cache_dir else HistoryCache()
    coverage = {}
    for raw in symbols:
        symbol = parse_pair(raw).canonical
        candles = client.klines(symbol, "1h", limit=bars)
        closed = [candle for candle in candles if candle.is_closed]
        if not closed:
            raise RuntimeError(f"Binance returned no completed 1h candles for {symbol}")
        count = cache.store(symbol, "1h", closed)
        frame = cache.load(symbol, "1h")
        coverage[symbol] = {
            "bars_returned_this_fetch": len(closed),
            "bars_in_cache": count,
            "first": frame.index[0].isoformat(),
            "last": frame.index[-1].isoformat(),
        }
    return {"interval": "1h", "requested_bars": bars, "coverage": coverage}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch completed 1h RAE research bars.")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--bars", type=int, default=17_520)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args(argv)
    result = fetch_ensemble_history(tuple(args.symbols), bars=args.bars, cache_dir=args.cache)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
