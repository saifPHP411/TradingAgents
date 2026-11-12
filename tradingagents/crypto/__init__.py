"""Crypto-first market data, instruments and pairs for a 24/7 spot market.

This package is the crypto layer's **data** side: canonical pair symbols
(:mod:`.symbols`), the venue's own trading rules (:mod:`.instruments`), normalized
market-data values (:mod:`.types`), Binance's public REST and WebSocket feeds
(:mod:`.binance`, :mod:`.stream`), and the engine that unifies them
(:mod:`.engine`).

It sits *below* the strategy and the paper experiment: nothing here decides a
trade, sizes an order or touches a ledger. A consumer (the TJR source, the
backtest, the dashboard) reads normalized values from here and applies its own
rules.

Why Binance's public **data** hosts: they need no API key and no registration, and
the ``data-api.binance.vision`` / ``data-stream.binance.vision`` hosts are served
where the main ``api.binance.com`` host answers HTTP 451. That is a genuinely free,
genuinely live source, which is what the market-data requirement asks for.
"""

from __future__ import annotations

from .binance import (
    REST_HOST,
    WS_HOST,
    BinancePublicClient,
    kline_stream_names,
    trade_stream_names,
)
from .bybit import (
    DEMO_REST_HOST,
    LIVE_REST_HOST,
    BybitPublicClient,
    bybit_symbol,
)
from .engine import CryptoMarketDataEngine, SymbolState, build_frame
from .instruments import (
    ORDER_REASONS,
    STATUS_TRADING,
    InstrumentError,
    InstrumentRegistry,
    OrderCheck,
    TradingRules,
    parse_exchange_info,
    validate_order,
)
from .stream import (
    CONNECTED,
    DISCONNECTED,
    RECONNECTING,
    STALE,
    MarketStream,
    StreamCallbacks,
    StreamStatus,
)
from .symbols import (
    CryptoPair,
    SymbolError,
    canonical_symbol,
    is_crypto_symbol,
    parse_pair,
    try_parse_pair,
)
from .tjr_settings import (
    CRYPTO_ENTRY_CUT,
    crypto_tjr_policy,
    crypto_tjr_settings,
)
from .transport import (
    CryptoAuthError,
    CryptoDataError,
    CryptoRateLimitedError,
    HttpResponse,
    fetch_json,
)
from .types import (
    CANDLE_INTERVALS,
    BookLevel,
    Candle,
    OrderBook,
    PublicTrade,
    Ticker,
    candles_to_frame,
)
from .universe import (
    DEFAULT_SYMBOLS,
    UniverseResolution,
    resolve_universe,
)

__all__ = [
    "CANDLE_INTERVALS",
    "CONNECTED",
    "CRYPTO_ENTRY_CUT",
    "DEFAULT_SYMBOLS",
    "DISCONNECTED",
    "ORDER_REASONS",
    "RECONNECTING",
    "REST_HOST",
    "STALE",
    "STATUS_TRADING",
    "WS_HOST",
    "BinancePublicClient",
    "BookLevel",
    "Candle",
    "CryptoAuthError",
    "CryptoDataError",
    "CryptoMarketDataEngine",
    "CryptoPair",
    "CryptoRateLimitedError",
    "HttpResponse",
    "InstrumentError",
    "InstrumentRegistry",
    "MarketStream",
    "OrderBook",
    "OrderCheck",
    "PublicTrade",
    "StreamCallbacks",
    "StreamStatus",
    "SymbolError",
    "SymbolState",
    "Ticker",
    "TradingRules",
    "UniverseResolution",
    "build_frame",
    "candles_to_frame",
    "canonical_symbol",
    "crypto_tjr_policy",
    "crypto_tjr_settings",
    "fetch_json",
    "is_crypto_symbol",
    "kline_stream_names",
    "parse_exchange_info",
    "parse_pair",
    "resolve_universe",
    "trade_stream_names",
    "try_parse_pair",
    "validate_order",
    # Bybit
    "DEMO_REST_HOST",
    "LIVE_REST_HOST",
    "BybitPublicClient",
    "bybit_symbol",
]
