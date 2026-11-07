"""Fetch the real historical bars a crypto study reads. The one network entry point.

Run once by a human, like the equity TJR's ``fetch_data``. It resolves the
requested universe against the venue, fetches 1-minute bars for each symbol (the
base interval every coarser one is resampled from), caches them, and prints the
coverage the venue actually served — so a short window is visible rather than
assumed.

    python -m tradingagents.crypto.backtest.fetch --limit 2500
    python -m tradingagents.crypto.backtest.fetch --symbols BTC/USD ETH/USD --limit 2000

The cache directory defaults to ``$TRADINGAGENTS_CRYPTO_CACHE_DIR`` or
``~/.tradingagents/crypto``. The offline study (``...backtest.run``) reads only
this cache; it never touches the network.
"""

from __future__ import annotations

import argparse
import json

from ..binance import BinancePublicClient
from ..symbols import parse_pair
from ..universe import DEFAULT_SYMBOLS, resolve_universe
from .history import HistoryCache, fetch_history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch real crypto bars into the study cache.")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument(
        "--limit", type=int, default=2500, help="1m bars per symbol (venue cap 1000/page)"
    )
    parser.add_argument(
        "--cache", default=None, help="cache directory (default: env or ~/.tradingagents/crypto)"
    )
    parser.add_argument(
        "--no-substitute", action="store_true", help="do not substitute a quote the venue serves"
    )
    args = parser.parse_args(argv)

    client = BinancePublicClient()
    registry = client.exchange_info()
    resolution = resolve_universe(
        registry,
        tuple(parse_pair(s).canonical for s in args.symbols),
        allow_substitution=not args.no_substitute,
    )
    print(f"universe: {resolution.describe()}", flush=True)
    if not resolution.symbols:
        print("no tradable symbols; nothing fetched", flush=True)
        return 1

    cache = HistoryCache(args.cache) if args.cache else HistoryCache()
    frames = fetch_history(
        resolution.symbols, intervals=("1m",), limit=args.limit, client=client, cache=cache
    )
    coverage = {}
    for symbol, bundle in frames.items():
        frame = bundle.get("1m")
        if frame is None or frame.empty:
            continue
        coverage[symbol] = {
            "bars": len(frame),
            "first": str(frame.index[0]),
            "last": str(frame.index[-1]),
        }
        print(
            f"  {symbol:12} {len(frame):5} bars  {frame.index[0]} .. {frame.index[-1]}", flush=True
        )
    summary = {"universe": resolution.to_dict(), "coverage": coverage}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - a maintainer entry point
    raise SystemExit(main())
