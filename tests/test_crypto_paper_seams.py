"""Tests for the crypto paper seams: the 24/7 market clock and the intraday window.

Both seams are driven offline. The market clock is exercised with an injected
clock (open, naive, and a raising clock that must read *unknown*, never *open*);
the intraday seam is exercised with a fake Bybit client so the closed-bar,
freshness, future-dated, malformed and transport branches all run for real.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tradingagents.paper_experiment.crypto_intraday_data import (
    CryptoIntradayBars,
    CryptoIntradayDataError,
    StaticCryptoIntradayBars,
)
from tradingagents.paper_experiment.crypto_market_hours import (
    CryptoMarketHours,
    StaticCryptoMarketHours,
)
from tradingagents.paper_experiment.market_hours import MARKET_OPEN, MARKET_UNKNOWN
from tradingagents.tjr.sessions import utc_day_session


class _FakeClient:
    provider_name = "fake"
    kind = "vendor_rest"

    def __init__(self, frames, *, error=None):
        self._frames = frames
        self._error = error

    def describe(self):
        return {"crypto_data_provider": self.provider_name, "crypto_data_requires_key": False}

    def klines(self, symbol, interval, *, limit=1000, start=None, end=None):
        if self._error is not None:
            from tradingagents.crypto.transport import CryptoDataError

            raise CryptoDataError(self._error)
        from tradingagents.crypto.types import Candle

        frame = self._frames[symbol][interval]
        candles = []
        for stamp, row in frame.iterrows():
            step = timedelta(minutes=1 if interval == "1m" else 5)
            candles.append(
                Candle(
                    symbol=symbol,
                    interval=interval,
                    open_time=stamp.to_pydatetime(),
                    close_time=(stamp + step).to_pydatetime(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 1.0)),
                )
            )
        return candles


def _frame(start: datetime, bars: int, *, minutes: int, price: float = 100.0) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [start + timedelta(minutes=minutes * i) for i in range(bars)], tz="UTC"
    )
    return pd.DataFrame(
        {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1.0},
        index=index,
    )


class TestCryptoMarketHours:
    def test_always_open_with_a_utc_session(self):
        now = datetime(2026, 3, 4, 13, 5, tzinfo=UTC)
        gate = CryptoMarketHours(clock=lambda: now)
        status = gate.status()
        assert status.state == MARKET_OPEN
        assert status.session == "2026-03-04"
        assert status.is_open is True

    def test_a_raising_clock_reads_unknown_never_open(self):
        def boom():
            raise RuntimeError("clock down")

        status = CryptoMarketHours(clock=boom).status()
        assert status.state == MARKET_UNKNOWN
        assert status.is_open is None
        assert status.session is None

    def test_a_naive_clock_reads_unknown(self):
        status = CryptoMarketHours(clock=lambda: datetime(2026, 3, 4, 13, 5)).status()
        assert status.state == MARKET_UNKNOWN

    def test_describe_states_it_is_always_open(self):
        described = CryptoMarketHours().describe()
        assert described["market_hours_always_open"] is True
        assert described["market_hours_session_basis"] == "utc_day"

    def test_the_static_clock_pins_the_session(self):
        status = StaticCryptoMarketHours(session="2026-01-01").status()
        assert status.state == MARKET_OPEN
        assert status.session == "2026-01-01"


class TestCryptoIntradayClosedBars:
    def test_returns_only_bars_closed_by_the_instant(self):
        as_of = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)
        five = _frame(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), 24, minutes=5)
        one = _frame(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), 120, minutes=1)
        client = _FakeClient({"BTC/USDT": {"1m": one, "5m": five}})
        seam = CryptoIntradayBars(
            client=client, max_stale_minutes=None, clock=lambda: as_of
        )
        five_series, one_series = seam.series("BTC/USDT", session="2026-01-01", as_of=as_of)
        # No bar whose close is after the decision instant may be returned.
        assert (five_series.frame.index + pd.Timedelta(minutes=5) <= as_of).all()
        assert (one_series.frame.index + pd.Timedelta(minutes=1) <= as_of).all()

    def test_a_stale_newest_bar_is_refused(self):
        as_of = datetime(2026, 1, 1, 3, 0, tzinfo=UTC)
        five = _frame(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), 12, minutes=5)
        one = _frame(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), 60, minutes=1)
        client = _FakeClient({"BTC/USDT": {"1m": one, "5m": five}})
        seam = CryptoIntradayBars(client=client, max_stale_minutes=20.0, clock=lambda: as_of)
        with pytest.raises(CryptoIntradayDataError) as err:
            seam.series("BTC/USDT", session="2026-01-01", as_of=as_of)
        assert err.value.kind == "stale"

    def test_a_future_dated_bar_is_refused(self):
        as_of = datetime(2026, 1, 2, 0, 30, tzinfo=UTC)
        five = _frame(datetime(2026, 1, 2, 0, 0, tzinfo=UTC), 12, minutes=5)
        one = _frame(datetime(2026, 1, 2, 0, 0, tzinfo=UTC), 60, minutes=1)
        client = _FakeClient({"BTC/USDT": {"1m": one, "5m": five}})
        seam = CryptoIntradayBars(client=client, max_stale_minutes=None, clock=lambda: as_of)
        with pytest.raises(CryptoIntradayDataError) as err:
            seam.series("BTC/USDT", session="2026-01-01", as_of=as_of)
        assert err.value.kind == "future"

    def test_a_transport_failure_is_a_data_unavailable_code(self):
        as_of = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)
        client = _FakeClient({}, error="network down")
        seam = CryptoIntradayBars(client=client, max_stale_minutes=None, clock=lambda: as_of)
        with pytest.raises(CryptoIntradayDataError) as err:
            seam.series("BTC/USDT", session="2026-01-01", as_of=as_of)
        assert err.value.kind == "transport"
        from tradingagents.paper_experiment.broker.trading import (
            NO_TRADE_MARKET_DATA_UNAVAILABLE,
        )

        assert err.value.code == NO_TRADE_MARKET_DATA_UNAVAILABLE

    def test_no_closed_bars_by_the_instant_is_refused(self):
        as_of = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        five = _frame(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), 12, minutes=5)
        one = _frame(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), 60, minutes=1)
        client = _FakeClient({"BTC/USDT": {"1m": one, "5m": five}})
        seam = CryptoIntradayBars(client=client, max_stale_minutes=None, clock=lambda: as_of)
        with pytest.raises(CryptoIntradayDataError) as err:
            seam.series("BTC/USDT", session="2026-01-01", as_of=as_of)
        assert err.value.kind == "not_started"

    def test_a_naive_instant_is_refused(self):
        seam = CryptoIntradayBars(client=_FakeClient({}), max_stale_minutes=None)
        with pytest.raises(CryptoIntradayDataError) as err:
            seam.series("BTC/USDT", session="2026-01-01", as_of=datetime(2026, 1, 1, 1, 0))
        assert err.value.kind == "malformed"

    def test_describe_names_the_session_policy(self):
        seam = CryptoIntradayBars(client=_FakeClient({}))
        described = seam.describe()
        assert described["tjr_data_session_policy"] == utc_day_session().label
        assert described["crypto_data_requires_key"] is False


class TestStaticCryptoIntraday:
    def test_the_static_seam_serves_a_pinned_window(self):
        as_of = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)
        frames = {
            "BTC/USDT": {
                "1m": _frame(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), 120, minutes=1),
                "5m": _frame(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), 24, minutes=5),
            }
        }
        seam = StaticCryptoIntradayBars(frames, max_stale_minutes=None, clock=lambda: as_of)
        five, one = seam.series("BTC/USDT", session="2026-01-01", as_of=as_of)
        assert five.symbol == "BTC/USDT"
        assert not one.frame.empty

    def test_the_static_seam_refuses_a_future_bar(self):
        as_of = datetime(2026, 1, 2, 0, 30, tzinfo=UTC)
        frames = {
            "BTC/USDT": {
                "1m": _frame(datetime(2026, 1, 2, 0, 0, tzinfo=UTC), 60, minutes=1),
                "5m": _frame(datetime(2026, 1, 2, 0, 0, tzinfo=UTC), 12, minutes=5),
            }
        }
        seam = StaticCryptoIntradayBars(frames, max_stale_minutes=None, clock=lambda: as_of)
        with pytest.raises(CryptoIntradayDataError) as err:
            seam.series("BTC/USDT", session="2026-01-01", as_of=as_of)
        assert err.value.kind == "future"
