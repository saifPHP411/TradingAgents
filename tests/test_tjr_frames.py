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
