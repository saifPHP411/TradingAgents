"""Three independent, fail-closed votes for Donchian 15m research signals.

Roles are deliberately deterministic and auditable, NOT three LLMs making
uncorroborated calls each second. All three must approve the *same* symbol,
15-minute candle and direction before a shadow opportunity is recorded.

NO order method, credential, position mutation, or Demo/live permission lives
in this module. Research consensus is never exchange execution approval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MAX_QUOTE_AGE_MS = 5_000
MAX_CANDIDATE_AGE_MS = 60_000
MAX_ASK_BID_SPREAD_BPS = 10.0
MAX_NOTIONAL_FRACTION = 0.20
MAX_LEVERAGE = 2.0
COST_PER_SIDE = 0.00080 + 0.00070  # Stress estimate, per-side
DONCHIAN_STOP = 0.008
DONCHIAN_TARGET = 0.018


@dataclass(frozen=True)
class TriadInput:
    symbol: str
    candle_end_ms: int
    now_ms: int
    direction: int
    close: float
    close_15m_ago: float
    ema60: float
    ema240: float
    rsi: float
    mark: float
    mark_ts_ms: int
    best_bid: float
    best_ask: float
    book_ts_ms: int
    account_equity_usdt: float = 50.0
    requested_notional_usdt: float = 10.0
    modeled_cost_per_side: float = COST_PER_SIDE
    max_leverage: float = MAX_LEVERAGE
    # Historical candles and public quotes DO NOT prove venue order size.
    venue_filters_verified: bool = False


def evaluate_triad(inp: TriadInput) -> dict:
    """Evaluate the sniper, analyst, professor independently; unanimity only."""
    if not isinstance(inp.symbol, str) or not inp.symbol.endswith("USDT"):
        raise ValueError("expected a USDT perpetual symbol")
    if any(
        isinstance(v, bool) or not isinstance(v, int)
        for v in (inp.candle_end_ms, inp.now_ms, inp.mark_ts_ms, inp.book_ts_ms)
    ):
        raise ValueError("timestamps must be integer milliseconds")
    if inp.direction not in (-1, 0, 1):
        raise ValueError("direction must be -1, 0 or 1")
    prices = (
        inp.close, inp.close_15m_ago, inp.ema60, inp.ema240, inp.mark,
        inp.best_bid, inp.best_ask, inp.account_equity_usdt,
    )
    if not all(math.isfinite(v) and v > 0 for v in prices):
        raise ValueError("prices and account equity must be finite positive")
    if not (0 <= inp.rsi <= 100 and math.isfinite(inp.rsi)):
        raise ValueError("invalid RSI")
    if (not math.isfinite(inp.requested_notional_usdt)
            or inp.requested_notional_usdt <= 0
            or not math.isfinite(inp.modeled_cost_per_side)
            or not 0 <= inp.modeled_cost_per_side <= 0.02
            or not math.isfinite(inp.max_leverage)
            or inp.max_leverage <= 0):
        raise ValueError("invalid risk or cost input")

    event_id = f"{inp.symbol}:{inp.candle_end_ms}:{inp.direction}"
    candle_is_complete_15m = inp.candle_end_ms % 900_000 == 0
    candle_age = inp.now_ms - inp.candle_end_ms
    candidate_fresh = 0 <= candle_age < MAX_CANDIDATE_AGE_MS
    mark_age = inp.now_ms - inp.mark_ts_ms
    book_age = inp.now_ms - inp.book_ts_ms
    quote_fresh = 0 <= mark_age <= MAX_QUOTE_AGE_MS and 0 <= book_age <= MAX_QUOTE_AGE_MS
    spread_bps = ((inp.best_ask - inp.best_bid)
                  / ((inp.best_ask + inp.best_bid) / 2) * 10000)
    spread_ok = 0 <= spread_bps <= MAX_ASK_BID_SPREAD_BPS
    market_coherent = inp.best_bid <= inp.mark <= inp.best_ask
    # The sniper owns the already independently defined Donchian 15m trigger.
    sniper_ok = inp.direction != 0 and candle_is_complete_15m and candidate_fresh
    sniper_reason = (
        "candidate_ready" if sniper_ok else "no_fresh_completed_15m_trigger"
    )
    # The analyst owns independent trend-strength / 15m impulse confirmation.
    ema_gap = (inp.ema60 - inp.ema240) / inp.close
    impulse = (inp.close - inp.close_15m_ago) / inp.close_15m_ago
    analyst_ok = (
        inp.direction != 0
        and inp.direction * ema_gap >= 0.0004
        and inp.direction * impulse >= 0.0003
        and (50 <= inp.rsi <= 85 if inp.direction == 1 else
             15 <= inp.rsi <= 50)
    )
    analyst_reason = "trend_confirmed" if analyst_ok else "trend_not_confirmed"
    # Professor rejects adverse costs, stale quotes, untradeably large
    # notionals, spread spikes, leverage changes, and negative net expectancy
    # in the specified stop/target scenario. This is a *screen*, not a proof
    # that positive expectancy exists.
    round_trip_cost = 2 * inp.modeled_cost_per_side
    after_cost_reward = DONCHIAN_TARGET - round_trip_cost
    after_cost_risk = DONCHIAN_STOP + round_trip_cost
    professor_ok = (
        quote_fresh and spread_ok and market_coherent
        and inp.requested_notional_usdt
        <= MAX_NOTIONAL_FRACTION * inp.account_equity_usdt + 1e-9
        and inp.max_leverage <= MAX_LEVERAGE
        and after_cost_risk > 0
        and after_cost_reward / after_cost_risk >= 1.3
    )
    professor_reason = (
        "modeled_cost_and_risk_screen_passed" if professor_ok
        else "stale_quote_spread_or_risk_limit_veto"
    )
    votes = {
        "sniper": {"approve": sniper_ok, "direction": inp.direction, "reason": sniper_reason},
        "analyst": {"approve": analyst_ok, "direction": inp.direction, "reason": analyst_reason},
        "professor": {
            "approve": professor_ok, "direction": inp.direction,
            "reason": professor_reason,
        },
    }
    unanimous = all(v["approve"] for v in votes.values())
    return {
        "event_id": event_id,
        "symbol": inp.symbol,
        "direction": inp.direction,
        "candle_end_ms": inp.candle_end_ms,
        "sampled_at_ms": inp.now_ms,
        "all_three_approve": unanimous,
        "votes": votes,
        "observed_spread_bps": spread_bps,
        "mark_age_ms": mark_age,
        "book_age_ms": book_age,
        "modeled_net_reward_risk": after_cost_reward / after_cost_risk,
        "venue_filters_verified": inp.venue_filters_verified,
        "execution_authorized": False,
        "research_only": True,
        "reason": "three_voters_agree_shadow_only" if unanimous else "one_or_more_vetoes",
    }
