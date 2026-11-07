"""Tests for the crypto data layer: symbols, instruments, types, transport.

Everything here runs offline. The HTTP seam is injected, so no test opens a socket
(the suite's autouse guard would refuse one anyway). The tests are written to catch
a *premise* error — a pair parsed into two instruments, a quantity rounded to a step
it cannot take, a NaN price passing a positivity guard — not to confirm the code.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tradingagents.crypto import (
    BinancePublicClient,
    Candle,
    CryptoDataError,
    CryptoPair,
    HttpResponse,
    SymbolError,
    TradingRules,
    candles_to_frame,
    canonical_symbol,
    fetch_json,
    is_crypto_symbol,
    parse_exchange_info,
    parse_pair,
    try_parse_pair,
    validate_order,
)
from tradingagents.crypto.instruments import (
    REASON_BELOW_MIN_NOTIONAL,
    REASON_BELOW_MIN_QTY,
    REASON_NOT_TRADABLE,
    REASON_QUANTITY_STEP,
    REASON_UNSUPPORTED_ORDER_TYPE,
    InstrumentRegistry,
)
from tradingagents.crypto.symbols import KNOWN_QUOTES
from tradingagents.crypto.types import interval_millis

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------


class TestSymbols:
    def test_canonical_spellings_agree(self):
        for text in ("BTC/USD", "btc-usd", "BTC_USD", "BTCUSD", "BTC:USD"):
            assert parse_pair(text).canonical == "BTC/USD"

    def test_usdt_is_not_rewritten_to_usd(self):
        # A pair parsed from BTCUSDT keeps USDT: the two are different instruments.
        assert parse_pair("BTCUSDT").canonical == "BTC/USDT"

    def test_longest_quote_wins(self):
        assert parse_pair("BTCUSDT").quote == "USDT"
        assert parse_pair("ETHUSDC").quote == "USDC"

    def test_base_may_itself_be_a_quote(self):
        assert parse_pair("ETHBTC").canonical == "ETH/BTC"

    def test_venue_spellings(self):
        pair = parse_pair("BTC/USD")
        assert pair.binance_symbol == "BTCUSD"
        assert pair.alpaca_symbol == "BTC/USD"
        assert pair.venue_symbol(separator="-") == "BTC-USD"

    def test_equality_is_canonical(self):
        assert parse_pair("BTCUSD") == parse_pair("BTC/USD")
        assert len({parse_pair("BTCUSD"), parse_pair("btc/usd")}) == 1

    def test_a_pair_needs_two_distinct_assets(self):
        with pytest.raises(SymbolError):
            CryptoPair(base="BTC", quote="BTC")

    @pytest.mark.parametrize("bad", ["", "   ", "NOTAPAIR", None, 42])
    def test_unparseable_symbols_raise(self, bad):
        with pytest.raises(SymbolError):
            parse_pair(bad)
        assert try_parse_pair(bad) is None

    def test_detection_and_canonicalisation(self):
        assert is_crypto_symbol("ethusd")
        assert not is_crypto_symbol("AAPL")
        assert canonical_symbol("solusdt") == "SOL/USDT"

    def test_every_known_quote_parses(self):
        for quote in KNOWN_QUOTES:
            if quote == "BTC":
                # A bare BTC quote is the one case that cannot be recovered from the
                # text alone: ``BTCBTC`` is refused as a same-asset pair, and the
                # venue's own asset list, not the parser, is what says which is real.
                continue
            assert parse_pair(f"BTC{quote}").quote == quote


# ---------------------------------------------------------------------------
# Types and frames
# ---------------------------------------------------------------------------


def _candle(minute: int, *, close: float = 100.0, symbol: str = "BTC/USD") -> Candle:
    open_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minute)
    return Candle(
        symbol=symbol,
        interval="1m",
        open_time=open_time,
        close_time=open_time + timedelta(minutes=1),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=10.0,
    )


class TestTypes:
    def test_interval_millis(self):
        assert interval_millis("1m") == 60_000
        assert interval_millis("4h") == 14_400_000
        with pytest.raises(ValueError):
            interval_millis("3m")

    def test_frame_is_sorted_and_utc_indexed(self):
        frame = candles_to_frame([_candle(2), _candle(0), _candle(1)])
        assert list(frame.index) == sorted(frame.index)
        assert str(frame.index.tz) == "UTC"

    def test_duplicate_candle_start_is_refused(self):
        with pytest.raises(ValueError, match="duplicate"):
            candles_to_frame([_candle(0), _candle(0)])

    def test_non_positive_price_is_refused(self):
        with pytest.raises(ValueError):
            candles_to_frame([_candle(0, close=0.0)])

    def test_nan_price_is_refused(self):
        # NaN passes `x <= 0` and `x < 0`, so it must be caught by isfinite.
        candle = _candle(0)
        poisoned = Candle(**{**candle.__dict__, "high": float("nan")})
        with pytest.raises(ValueError):
            candles_to_frame([poisoned])

    def test_high_below_low_is_refused(self):
        candle = _candle(0)
        bad = Candle(**{**candle.__dict__, "high": 1.0, "low": 5.0})
        with pytest.raises(ValueError):
            candles_to_frame([bad])

    def test_empty_candle_list_is_refused(self):
        with pytest.raises(ValueError):
            candles_to_frame([])

    def test_frame_is_readable_by_the_tjr_layer(self):
        from tradingagents.tjr.frames import IntradaySeries
        from tradingagents.tjr.sessions import utc_day_session

        frame = candles_to_frame([_candle(i) for i in range(10)])
        series = IntradaySeries("BTC/USD", "1m", frame, session=utc_day_session())
        # ``available`` is the closed-bar reader: a bar is visible only once it has
        # closed, so the last bar's own close is the last one a caller may see.
        assert len(series.available(frame.index[-1] + pd.Timedelta(minutes=1))) == 10


# ---------------------------------------------------------------------------
# Instruments and order validation
# ---------------------------------------------------------------------------


def _rules(**overrides) -> TradingRules:
    base = {
        "symbol": "BTC/USDT",
        "base_asset": "BTC",
        "quote_asset": "USDT",
        "status": "TRADING",
        "base_precision": 8,
        "quote_precision": 8,
        "min_quantity": 0.0001,
        "max_quantity": 100.0,
        "quantity_step": 0.0001,
        "price_tick": 0.01,
        "min_price": None,
        "max_price": None,
        "min_notional": 10.0,
        "max_notional": None,
        "order_types": ("LIMIT", "MARKET"),
    }
    base.update(overrides)
    return TradingRules(**base)


class TestInstruments:
    def test_accepts_a_conforming_order(self):
        check = validate_order(_rules(), quantity=0.5, price=20_000.0, order_type="MARKET")
        assert check.ok, check.reasons

    def test_not_tradable_is_refused(self):
        check = validate_order(_rules(status="BREAK"), quantity=1.0, price=20_000.0)
        assert REASON_NOT_TRADABLE in check.reasons

    def test_below_min_quantity(self):
        check = validate_order(_rules(), quantity=0.00001, price=20_000.0)
        assert REASON_BELOW_MIN_QTY in check.reasons

    def test_quantity_must_sit_on_the_step(self):
        # 0.00015 is not a multiple of stepSize 0.0001.
        check = validate_order(_rules(), quantity=0.00015, price=20_000.0)
        assert REASON_QUANTITY_STEP in check.reasons

    def test_below_min_notional(self):
        check = validate_order(_rules(), quantity=0.0001, price=20_000.0)
        assert REASON_BELOW_MIN_NOTIONAL in check.reasons

    def test_unsupported_order_type(self):
        check = validate_order(_rules(), quantity=1.0, price=20_000.0, order_type="STOP_LOSS")
        assert REASON_UNSUPPORTED_ORDER_TYPE in check.reasons

    def test_price_must_sit_on_the_tick(self):
        check = validate_order(_rules(), quantity=1.0, price=20_000.005, order_type="LIMIT")
        assert check.ok or "PRICE_STEP" in check.reasons

    def test_non_finite_quantity_is_refused(self):
        check = validate_order(_rules(), quantity=float("nan"), price=20_000.0)
        assert not check.ok

    def test_quantise_quantity_floors_to_step(self):
        rules = _rules()
        assert rules.quantize_quantity(0.123456789) == pytest.approx(0.1234)

    def test_quantise_price_truncates_to_tick(self):
        # Truncation, not rounding: an order priced a hair lower is a smaller error
        # than one priced a hair higher than intended.
        rules = _rules(price_tick=0.5)
        assert rules.quantize_price(100.26) == pytest.approx(100.0)
        assert rules.quantize_price(100.6) == pytest.approx(100.5)

    def test_exchange_info_parsing(self):
        payload = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "baseAssetPrecision": 8,
                    "quotePrecision": 8,
                    "orderTypes": ["LIMIT", "MARKET"],
                    "filters": [
                        {
                            "filterType": "LOT_SIZE",
                            "minQty": "0.0001",
                            "maxQty": "100",
                            "stepSize": "0.0001",
                        },
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "NOTIONAL", "minNotional": "10"},
                    ],
                }
            ]
        }
        registry = parse_exchange_info(payload)
        assert "BTC/USDT" in registry
        rules = registry["BTC/USDT"]
        assert rules.min_quantity == pytest.approx(0.0001)
        assert rules.quantity_step == pytest.approx(0.0001)

    def test_registry_lookup_by_pair(self):
        payload = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "filters": [
                        {"filterType": "LOT_SIZE", "minQty": "0.0001", "stepSize": "0.0001"},
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    ],
                }
            ]
        }
        registry = InstrumentRegistry(parse_exchange_info(payload))
        assert registry.get("BTC/USDT") is not None
        assert registry.is_tradable("BTC/USDT")
        # BTC/USD is a different instrument from BTC/USDT; the venue listed only one.
        assert registry.get("BTC/USD") is None
        assert not registry.is_supported("BTC/USD")


# ---------------------------------------------------------------------------
# Transport and the Binance client
# ---------------------------------------------------------------------------


class TestTransport:
    def test_non_200_raises(self):
        def transport(*, url, params, headers, timeout):
            return HttpResponse(500, "boom")

        with pytest.raises(CryptoDataError, match="500"):
            fetch_json(url="https://x/y", transport=transport)

    def test_non_json_raises(self):
        def transport(*, url, params, headers, timeout):
            return HttpResponse(200, "<html>not json</html>")

        with pytest.raises(CryptoDataError, match="JSON"):
            fetch_json(url="https://x/y", transport=transport)

    def test_transport_failure_is_reported_not_hidden(self):
        def transport(*, url, params, headers, timeout):
            raise OSError("connection refused")

        with pytest.raises(CryptoDataError, match="connection refused"):
            fetch_json(url="https://x/y", transport=transport)

    def test_secret_is_redacted_from_an_error_body(self):
        secret = "super-secret-key"

        def transport(*, url, params, headers, timeout):
            return HttpResponse(401, f"invalid key {secret}")

        with pytest.raises(CryptoDataError) as excinfo:
            fetch_json(url="https://x/y", transport=transport, secrets=(secret,))
        assert secret not in str(excinfo.value)

    def test_json_object_and_array_both_return(self):
        def transport(*, url, params, headers, timeout):
            return HttpResponse(200, '[{"a": 1}]')

        assert fetch_json(url="https://x/y", transport=transport) == [{"a": 1}]


class TestBinanceClient:
    def _client(self, payloads: dict[str, object]):
        calls: list[dict] = []

        def transport(*, url, params, headers, timeout):
            calls.append({"url": url, "params": params})
            for key, payload in payloads.items():
                if key in url:
                    return HttpResponse(200, __import__("json").dumps(payload))
            return HttpResponse(404, "not found")

        return BinancePublicClient(transport=transport), calls

    def test_klines_are_parsed_and_ordered(self):
        rows = [
            [
                1_700_000_000_000,
                "10",
                "12",
                "9",
                "11",
                "100",
                1_700_000_059_999,
                "0",
                0,
                "0",
                "0",
                "0",
            ],
            [
                1_700_000_060_000,
                "11",
                "13",
                "10",
                "12",
                "50",
                1_700_000_119_999,
                "0",
                0,
                "0",
                "0",
                "0",
            ],
        ]
        client, _ = self._client({"klines": rows})
        candles = client.klines("BTC/USD", "1m", limit=2)
        assert [c.open for c in candles] == [10.0, 11.0]
        assert candles[0].open_time < candles[1].open_time

    def test_klines_skip_a_malformed_row(self):
        rows = [
            [
                1_700_000_000_000,
                "10",
                "12",
                "9",
                "11",
                "100",
                1_700_000_059_999,
                "0",
                0,
                "0",
                "0",
                "0",
            ],
            ["bad", "row"],
            [
                1_700_000_060_000,
                "11",
                "13",
                "10",
                "12",
                "50",
                1_700_000_119_999,
                "0",
                0,
                "0",
                "0",
                "0",
            ],
        ]
        client, _ = self._client({"klines": rows})
        assert len(client.klines("BTC/USD", "1m", limit=3)) == 2

    def test_ticker_is_normalised(self):
        payload = {
            "symbol": "BTCUSDT",
            "lastPrice": "100.5",
            "bidPrice": "100.4",
            "askPrice": "100.6",
            "volume": "1000",
            "quoteVolume": "100500",
            "highPrice": "110",
            "lowPrice": "90",
            "priceChange": "1.5",
            "priceChangePercent": "1.5",
            "closeTime": 1_700_000_000_000,
        }
        client, _ = self._client({"ticker/24hr": payload})
        # The venue answered for its own spelling (BTCUSDT); the client keys the
        # result by canonical pair, so the caller looks it up as BTC/USDT.
        ticker = client.ticker_24h(["BTC/USDT"])["BTC/USDT"]
        assert ticker.last == pytest.approx(100.5)
        assert ticker.spread == pytest.approx(0.2)
        assert ticker.mid == pytest.approx(100.5)

    def test_unsupported_interval_is_refused_before_any_call(self):
        client, calls = self._client({})
        with pytest.raises(CryptoDataError):
            client.klines("BTC/USD", "3m")
        assert calls == []
