from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.crypto.backtest.learning import OnlineExpertLearner


def _stamp(day):
    return (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=day)).isoformat()


def test_learner_starts_in_shadow_mode_and_uses_only_matured_outcomes():
    learner = OnlineExpertLearner(min_observations=3)
    for day, value in enumerate((-1.0, -1.0, -1.0)):
        learner.update("trend_pullback", _stamp(day), value)
    assert learner.allows("trend_pullback", _stamp(2))
    assert not learner.allows("trend_pullback", _stamp(3))
    assert not learner.allows("trend_pullback", _stamp(4))


def test_learner_blocks_expert_only_after_confident_negative_net_results():
    learner = OnlineExpertLearner(min_observations=4)
    for day in range(4):
        learner.update("breakout", _stamp(day), -1.0)
    state = learner.snapshot(_stamp(5))["breakout"]
    assert state["status"] == "blocked"
    assert state["lower_95pct_bound_pct"] == pytest.approx(-1.0)
    assert not learner.allows("breakout", _stamp(5))


def test_learner_keeps_expert_when_lower_confidence_bound_is_positive():
    learner = OnlineExpertLearner(min_observations=4)
    for day, value in enumerate((0.8, 1.0, 1.1, 0.9)):
        learner.update("range_reversion", _stamp(day), value)
    assert learner.allows("range_reversion", _stamp(5))
    assert learner.snapshot(_stamp(5))["range_reversion"]["status"] == "validated"


def test_learner_caps_memory_and_rejects_nonchronological_or_invalid_updates():
    learner = OnlineExpertLearner(min_observations=2, max_observations=3)
    for day in range(5):
        learner.update("breakout", _stamp(day), 0.1)
    assert len(learner.observations["breakout"]) == 3
    with pytest.raises(ValueError, match="chronologically"):
        learner.update("breakout", _stamp(3), 0.1)
    with pytest.raises(ValueError, match="finite"):
        learner.update("breakout", _stamp(6), float("nan"))


def test_learner_requires_timezone_aware_timestamps_and_bounded_config():
    with pytest.raises(ValueError, match="timezone"):
        OnlineExpertLearner().update("breakout", "2026-01-01T00:00:00", 1.0)
    with pytest.raises(ValueError, match="min_observations"):
        OnlineExpertLearner(min_observations=1)
    with pytest.raises(ValueError, match="max_observations"):
        OnlineExpertLearner(min_observations=3, max_observations=2)
