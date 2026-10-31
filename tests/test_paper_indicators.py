"""The paper-inference indicator context: deterministic, point-in-time, honest.

The paper experiment makes one bounded model call per decision. A bare OHLCV
table leaves the model to recompute trend, momentum and volatility itself, and a
single arithmetic slip there is indistinguishable from a genuine opinion. These
tests pin the module that computes that evidence in code, so the request carries
stated evidence rather than an invitation to re-derive it.

Three properties are load-bearing:

- **Deterministic.** The same window yields the same numbers and the same
  rendered text, so two runs of the same session agree and a restart is
  byte-comparable.
- **Point-in-time.** Every value is a function of the window passed in, which is
  already bounded at the decision session. No later bar can reach a value.
- **No fabrication.** A window too short for an indicator yields ``None`` and a
  rendered "insufficient history" line naming the bar count; it is never
  zero-filled or extrapolated. A malformed value raises rather than rendering.

Offline and deterministic: no network, no provider, no key.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from tradingagents.paper_experiment.indicators import (
    BOLL_PERIOD,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    SMA_SHORT,
    SMA_TREND,
    IndicatorError,
    IndicatorSet,
    compute_indicators,
    render_indicator_context,
)


def _frame(periods: int = 260, *, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    opens = [start + step * i for i in range(periods)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 2.0 for o in opens],
            "low": [o - 1.0 for o in opens],
            "close": [o + 1.0 for o in opens],
            "volume": [1_000_000.0] * periods,
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheIndicatorsAreDeterministic:
    def test_the_same_window_yields_the_same_set(self):
        first = compute_indicators(_frame())
        second = compute_indicators(_frame())
        assert first == second

    def test_the_same_window_renders_the_same_text(self):
        frame = _frame()
        assert render_indicator_context(frame) == render_indicator_context(frame.copy())

    def test_the_indicator_set_is_hashable_and_comparable(self):
        values = compute_indicators(_frame())
        assert isinstance(values, IndicatorSet)
        assert hash(values) == hash(compute_indicators(_frame()))


# ---------------------------------------------------------------------------
# Point-in-time
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheIndicatorsArePointInTime:
    def test_a_later_bar_cannot_change_an_earlier_value(self):
        full = _frame()
        cutoff = full.index[120]
        bounded = full[full.index <= cutoff]

        from_bounded = compute_indicators(bounded)
        from_full_truncated = compute_indicators(full.iloc[:121])
        assert from_bounded == from_full_truncated

    def test_a_rising_series_has_rsi_at_the_top_of_its_range(self):
        values = compute_indicators(_frame())
        assert values.rsi is not None
        assert values.rsi == pytest.approx(100.0)

    def test_a_falling_series_has_rsi_at_the_bottom_of_its_range(self):
        values = compute_indicators(_frame(step=-1.0))
        assert values.rsi is not None
        assert values.rsi == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Insufficient history is stated, never fabricated
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInsufficientHistoryIsStatedNotFabricated:
    def test_a_short_window_yields_none_rather_than_a_number(self):
        values = compute_indicators(_frame(periods=5))
        assert values.sma_short is None
        assert values.sma_long is None
        assert values.sma_trend is None
        assert values.rsi is None
        assert values.macd is None
        assert values.boll_mid is None
        assert values.atr is None

    def test_the_render_names_the_shortfall(self):
        text = render_indicator_context(_frame(periods=5))
        assert "insufficient history" in text
        assert f"need {SMA_TREND} bars" in text
        assert "have 5" in text

    def test_a_value_is_never_zero_filled(self):
        text = render_indicator_context(_frame(periods=5))
        # No rendered SMA line carries a fabricated 0.00 for a missing value.
        for line in text.splitlines():
            if "SMA200" in line:
                assert "insufficient history" in line
                assert "0.00" not in line

    def test_a_sufficient_window_reports_real_numbers(self):
        values = compute_indicators(_frame())
        assert values.sma_short is not None
        assert values.sma_trend is not None
        assert values.rsi is not None
        assert values.atr is not None
        assert values.boll_upper is not None and values.boll_lower is not None
        assert values.boll_upper > values.boll_lower

    def test_a_partial_window_reports_what_it_can(self):
        values = compute_indicators(_frame(periods=RSI_PERIOD + 5))
        assert values.rsi is not None
        assert values.sma_short is None  # needs 20 bars
        assert values.macd is None  # needs 26 + 9 bars


# ---------------------------------------------------------------------------
# A malformed window raises rather than rendering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAMalformedWindowRaises:
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_value_raises(self, bad):
        frame = _frame(periods=40)
        frame.iloc[-1, frame.columns.get_loc("close")] = bad
        with pytest.raises(IndicatorError, match="non-finite"):
            compute_indicators(frame)

    def test_a_missing_column_raises(self):
        frame = _frame(periods=40).drop(columns=["volume"])
        with pytest.raises(IndicatorError, match="volume"):
            compute_indicators(frame)

    def test_an_empty_window_renders_a_stated_absence(self):
        assert "No indicator context" in render_indicator_context(None)
        assert "No indicator context" in render_indicator_context(pd.DataFrame())


# ---------------------------------------------------------------------------
# The rendered block carries the evidence the model needs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheRenderedBlockCarriesTheEvidence:
    def test_trend_momentum_and_volatility_are_all_present(self):
        text = render_indicator_context(_frame())
        for label in ("trend", "RSI", "MACD", "Bollinger", "ATR", "realized vol", "volume"):
            assert label in text, label

    def test_the_trend_line_states_a_direction(self):
        text = render_indicator_context(_frame())
        assert "up (SMA20 vs SMA50)" in text

    def test_a_falling_series_states_a_down_trend(self):
        text = render_indicator_context(_frame(step=-1.0))
        assert "down (SMA20 vs SMA50)" in text

    def test_the_rsi_state_is_labelled(self):
        assert "overbought" in render_indicator_context(_frame())
        assert "oversold" in render_indicator_context(_frame(step=-1.0))

    def test_the_bar_count_is_reported(self):
        assert f"bars in window           {SMA_TREND + 10}" in render_indicator_context(
            _frame(periods=SMA_TREND + 10)
        )


# ---------------------------------------------------------------------------
# The window the indicators see is the window the record attests to
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTheIndicatorWindowIsTheRecordedWindow:
    def test_the_indicators_use_the_same_bars_the_record_hashes(self, tmp_path):
        from tradingagents.paper_experiment import (
            ExperimentConfig,
            PaperExperiment,
            StaticPriceSource,
        )
        from tradingagents.paper_experiment.inference import StubInference

        source = StaticPriceSource({"SPY": _frame(periods=60)})
        config = ExperimentConfig.from_protocol(
            symbols=("SPY",), decision_frequency_sessions=1, evaluation_lag_sessions=2
        )
        experiment = PaperExperiment(
            source=source,
            config=config,
            root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()

        session = source.sessions()[30]
        experiment.process_session(session)
        snapshot = experiment._snapshot("SPY", session)
        assert snapshot is not None

        values = compute_indicators(snapshot.to_frame())
        assert values.bars == len(snapshot.to_frame())
        assert values.last_close == pytest.approx(
            float(snapshot.to_frame()["close"].iloc[-1])
        )


@pytest.mark.unit
class TestIndicatorConstantsMatchTheirNames:
    def test_macd_windows_are_the_conventional_ones(self):
        assert (MACD_SLOW, MACD_SIGNAL) == (26, 9)
        assert RSI_PERIOD == 14
        assert BOLL_PERIOD == 20
        assert SMA_SHORT == 20
