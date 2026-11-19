"""Fast point-in-time intraday slicing must preserve the old closed-bar semantics."""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.tjr.frames import IntradaySeries
from tradingagents.tjr.sessions import SessionPolicy, policy_named, utc_day_session

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("interval,frequency,bar_minutes", [("1m", "1min", 1), ("5m", "5min", 5)])
@pytest.mark.parametrize("policy_name", ["utc_day", "crypto_named", "us_equity"])
def test_available_and_session_bars_match_the_reference_mask(
    interval: str, frequency: str, bar_minutes: int, policy_name: str
) -> None:
    """The searchsorted implementation returns exactly the bars the old masks did."""
    policy = policy_named(policy_name)
    index = pd.date_range("2026-10-31T00:00:00Z", periods=3 * 24 * 60 // bar_minutes, freq=frequency)
    close = [100.0 + i / 1000.0 for i in range(len(index))]
    frame = pd.DataFrame(
        {
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [1_000.0] * len(index),
        },
        index=index,
    )
    series = IntradaySeries("BTC/USDT", interval, frame, session=policy)

    for session in ("2026-10-31", "2026-11-01", "2026-11-02"):
        session_open, session_close = policy.session_bounds(session)
        delta = pd.Timedelta(minutes=bar_minutes)
        instants = (
            session_open - delta,
            session_open,
            session_open + pd.Timedelta(hours=2),
            session_close - delta,
            session_close,
            session_close + 2 * delta,
        )
        for as_of in instants:
            cutoff = as_of - delta
            expected_available = series.frame[series.frame.index <= cutoff]
            expected_session = expected_available[
                policy.index_mask(expected_available.index, session)
            ]
            pd.testing.assert_frame_equal(series.available(as_of), expected_available)
            pd.testing.assert_frame_equal(series.session_bars(session, as_of), expected_session)


def test_session_bars_masks_only_the_current_session_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 120-day cache must not be rescanned in full for each intraday decision."""
    policy = utc_day_session()
    index = pd.date_range("2026-07-01T00:00:00Z", periods=120 * 24 * 60, freq="1min")
    close = [100.0 + i / 100_000.0 for i in range(len(index))]
    frame = pd.DataFrame(
        {
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [1_000.0] * len(index),
        },
        index=index,
    )
    series = IntradaySeries("BTC/USDT", "1m", frame, session=policy)
    scanned_lengths: list[int] = []
    original = SessionPolicy.index_mask

    def record_mask(self: SessionPolicy, selected_index: pd.DatetimeIndex, session: str):
        scanned_lengths.append(len(selected_index))
        return original(self, selected_index, session)

    monkeypatch.setattr(SessionPolicy, "index_mask", record_mask)
    session = "2026-10-28"
    as_of = pd.Timestamp("2026-10-29T00:00:00Z")
    result = series.session_bars(session, as_of)

    assert len(result) == 24 * 60
    assert scanned_lengths == [24 * 60]
    assert scanned_lengths[0] < len(series.frame) // 100



def test_cached_swing_points_match_each_point_in_time_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Precomputing swings is safe only when each confirmation bar has closed."""
    import math

    from tradingagents.tjr import levels
    from tradingagents.tjr.structure import swing_points as reference_swing_points

    index = pd.date_range("2026-11-16T00:00:00Z", periods=240, freq="5min")
    close = [100.0 + math.sin(i / 3.0) * 2.0 + i * 0.001 for i in range(len(index))]
    frame = pd.DataFrame(
        {
            "open": close,
            "high": [value + 0.4 for value in close],
            "low": [value - 0.4 for value in close],
            "close": close,
            "volume": [1_000.0] * len(index),
        },
        index=index,
    )
    series = IntradaySeries("BTC/USDT", "5m", frame, session=utc_day_session())
    calls: list[int] = []
    original = levels.swing_points

    def count_full_frame_calls(data: pd.DataFrame, *, lookback: int = 2):
        calls.append(len(data))
        return original(data, lookback=lookback)

    monkeypatch.setattr(levels, "swing_points", count_full_frame_calls)
    as_of_values = (
        index[10] + pd.Timedelta(minutes=5),
        index[40] + pd.Timedelta(minutes=5),
        index[100] + pd.Timedelta(minutes=5),
        index[-1] + pd.Timedelta(minutes=5),
    )
    for as_of in as_of_values:
        expected = reference_swing_points(series.available(as_of), lookback=2)
        actual = levels._confirmed_swing_points(series, as_of, lookback=2)
        assert actual == expected

    assert calls == [len(series.frame)]
    assert series._analysis_cache[("tjr_swing_points", 2)]


@pytest.mark.parametrize("policy_name", ["utc_day", "crypto_named", "us_equity"])
def test_previous_session_lookup_matches_the_full_history_reference(policy_name: str) -> None:
    """The bounded lookup must preserve local-day semantics, including DST."""
    from tradingagents.tjr.levels import _previous_session

    policy = policy_named(policy_name)
    index = pd.date_range("2026-10-31T00:00:00Z", periods=3 * 24 * 60, freq="1min")
    close = [100.0 + i / 10_000.0 for i in range(len(index))]
    frame = pd.DataFrame(
        {
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [1_000.0] * len(index),
        },
        index=index,
    )
    series = IntradaySeries("BTC/USDT", "1m", frame, session=policy)

    for session in ("2026-10-31", "2026-11-01", "2026-11-02"):
        session_open, session_close = policy.session_bounds(session)
        instants = (
            session_open - pd.Timedelta(minutes=1),
            session_open,
            session_open + pd.Timedelta(hours=2),
            session_close,
            session_close + pd.Timedelta(minutes=2),
        )
        for as_of in instants:
            available = series.available(as_of)
            days = available.index.tz_convert(policy.timezone).strftime("%Y-%m-%d")
            previous_days = sorted({day for day in days if day < session})
            expected = previous_days[-1] if previous_days else None
            assert _previous_session(series, session, as_of) == expected
