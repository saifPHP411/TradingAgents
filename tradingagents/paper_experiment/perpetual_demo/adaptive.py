"""Persistent forward learning and a bounded Gemini vote for the Demo desk."""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tradingagents.crypto.backtest.learning import OnlineExpertLearner
from tradingagents.crypto.scalping import ScalpingConfig

CANDLE_MS = 60_000
MIN_TRADES = 30
MIN_WIN_RATE = 0.75
MAX_HISTORY = 100
MODEL = "gemini-3.8-flash"
# Observed Mark is a proxy. The stress deduction represents fees + adverse impact.
STRESS_COST_PER_SIDE = 0.0015
MAX_REFERENCE_DRIFT = 0.00075
SCHEMA = {
    "type": "object",
    "properties": {
        "approve": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["approve", "reason"],
}


def _iso(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, UTC)


def _positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


class GeminiAnalyst:
    """A stateless provider call with a small, structured response."""

    def __init__(self, api_key: str, model: str = MODEL):
        if not re.fullmatch(r"gemini-[a-z0-9.-]{1,60}", model):
            raise ValueError("Invalid Gemini model ID")
        self._api_key = api_key.strip()
        self.model = model

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def vote(self, evidence: dict) -> dict:
        instruction = (
            "You are the Analyst in a Bybit Demo research system. Review only the supplied "
            "perpetual market snapshot and prior settled shadow outcomes. Approve the given "
            "direction only when the trend agrees and the evidence is sufficient. Abstain "
            "on weak, mixed, or insufficient evidence. You cannot choose a different trade, "
            "change sizing, change rules, or place orders. Give one brief reason in Arabic."
        )
        data = {
            "systemInstruction": {"parts": [{"text": instruction}]},
            "contents": [{"parts": [{"text": json.dumps(evidence, allow_nan=False)}]}],
            "generationConfig": {
                "maxOutputTokens": 512,
                "responseMimeType": "application/json",
                "responseJsonSchema": SCHEMA,
            },
        }
        request = Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        request.add_unredirected_header("x-goog-api-key", self._api_key)
        with urlopen(request, timeout=8) as response:
            raw = json.loads(response.read(65_537))
        candidate = raw["candidates"][0]
        if candidate.get("finishReason") != "STOP":
            raise ValueError("Incomplete Gemini response")
        content = "".join(
            part.get("text", "") for part in candidate["content"]["parts"]
            if not part.get("thought")
        )
        vote = json.loads(content)
        if (
            not isinstance(vote, dict) or set(vote) != {"approve", "reason"}
            or type(vote["approve"]) is not bool or not isinstance(vote["reason"], str)
            or not 0 < len(vote["reason"].strip()) <= 360
        ):
            raise ValueError("Invalid Gemini vote")
        return {**vote, "model_version": raw.get("modelVersion") or self.model}


class AdaptiveDesk:
    """Collect baseline outcomes independently of the Gemini or performance veto."""

    def __init__(
        self, root: Path, analyst: GeminiAnalyst, strategy: ScalpingConfig, *,
        calls_per_day: int = 24, poll_seconds: int = 15,
    ):
        if not 1 <= calls_per_day <= 250:
            raise ValueError("Gemini daily limit must be between 1 and 250")
        self.path = root / "gemini_scalper_v1.sqlite3"
        self.analyst = analyst
        self.strategy = strategy
        self.calls_per_day = calls_per_day
        self.max_gap_ms = (2 * poll_seconds + 5) * 1000
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemini-analyst")
        self._inflight: Future | None = None
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS shadow (
                    id TEXT PRIMARY KEY, symbol TEXT NOT NULL, side TEXT NOT NULL,
                    entry_ms INTEGER NOT NULL, last_quote_ms INTEGER NOT NULL,
                    entry REAL NOT NULL, stop REAL NOT NULL, target REAL NOT NULL,
                    filled_ms INTEGER, exit_ms INTEGER, net_pct REAL, exit_reason TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS shadow_one_open
                    ON shadow(symbol) WHERE exit_ms IS NULL;
                CREATE INDEX IF NOT EXISTS shadow_exits ON shadow(exit_ms);
                CREATE TABLE IF NOT EXISTS votes (
                    id TEXT PRIMARY KEY, symbol TEXT NOT NULL, side TEXT NOT NULL,
                    requested_ms INTEGER NOT NULL, expires_ms INTEGER NOT NULL,
                    model TEXT NOT NULL, evidence TEXT NOT NULL, result TEXT,
                    completed_ms INTEGER
                );
                CREATE INDEX IF NOT EXISTS votes_requested ON votes(requested_ms);
            """)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _performance(self, db, now_ms: int) -> dict:
        learner = OnlineExpertLearner(min_observations=MIN_TRADES, max_observations=MAX_HISTORY)
        rows = db.execute("""
            SELECT side, exit_ms, net_pct FROM shadow
            WHERE net_pct IS NOT NULL AND exit_ms < ?
            ORDER BY exit_ms DESC, id DESC LIMIT ?
        """, (now_ms, MAX_HISTORY)).fetchall()
        for row in reversed(rows):
            learner.update("baseline", _iso(row["exit_ms"]).isoformat(), row["net_pct"])
            learner.update(row["side"], _iso(row["exit_ms"]).isoformat(), row["net_pct"])
        metrics = learner.snapshot(_iso(now_ms))
        for name, summary in metrics.items():
            returns = [row["net_pct"] for row in rows if name in ("baseline", row["side"])]
            summary["win_rate"] = sum(value > 0 for value in returns) / len(returns)
            gross_loss = -sum(value for value in returns if value < 0)
            summary["profit_factor"] = sum(value for value in returns if value > 0) / gross_loss if gross_loss else None
        return metrics

    @staticmethod
    def _qualified(performance: dict, side: str | None = None) -> bool:
        baseline = performance.get("baseline", {})
        side_metrics = performance.get(side, {})
        return (
            baseline.get("status") == "validated"
            and baseline["win_rate"] >= MIN_WIN_RATE
            and side_metrics.get("status") != "blocked"
        )

    def ready(self, now_ms: int) -> bool:
        with self._db() as db:
            return self.analyst.configured and self._qualified(self._performance(db, now_ms))

    @staticmethod
    def _day_start(now_ms: int) -> int:
        return now_ms // 86_400_000 * 86_400_000

    def observe(self, symbol: str, market: dict, candle_ms: int, quote_ms: int, now_ms: int) -> str:
        """Update independent research fills and queue at most one model call per signal."""
        price = market.get("mark_price")
        if not _positive(price) or not candle_ms <= quote_ms <= now_ms < candle_ms + CANDLE_MS:
            return "stale_or_invalid_market"
        if now_ms - quote_ms > 5_000:
            return "stale_or_invalid_market"

        signal = market.get("signal")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            position = db.execute(
                "SELECT * FROM shadow WHERE symbol = ? AND exit_ms IS NULL", (symbol,)
            ).fetchone()
            if position and quote_ms > position["last_quote_ms"]:
                self._track(db, position, signal, price, candle_ms, quote_ms)
            if signal not in (-1, 1):
                return "no_sniper_signal"

            side = "long" if signal == 1 else "short"
            stop = market.get(f"suggested_{side}_stop")
            target = market.get(f"suggested_{side}_target")
            if not all(_positive(level) for level in (stop, target)):
                return "professor_invalid_bracket"
            if not (stop < price < target if signal == 1 else target < price < stop):
                return "professor_invalid_bracket"
            if signal * (target - price) / price <= 2 * STRESS_COST_PER_SIDE:
                return "professor_cost_veto"

            event_id = f"{symbol}:{candle_ms}:{side}"
            next_entry = db.execute(
                "SELECT MAX(exit_ms) FROM shadow WHERE symbol = ?", (symbol,)
            ).fetchone()[0]
            if not position and (
                next_entry is None or quote_ms >= next_entry + self.strategy.cooldown_minutes * CANDLE_MS
            ):
                db.execute("""
                    INSERT OR IGNORE INTO shadow
                    (id, symbol, side, entry_ms, last_quote_ms, entry, stop, target)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (event_id, symbol, side, quote_ms, quote_ms, price, stop, target))

            performance = self._performance(db, now_ms)
            row = db.execute("SELECT result, completed_ms, evidence FROM votes WHERE id = ?", (event_id,)).fetchone()
            if row and row["result"] is not None:
                result = json.loads(row["result"])
                if not result.get("approve"):
                    return result.get("status", "analyst_veto")
                if not row["completed_ms"] < quote_ms <= now_ms:
                    return "waiting_for_quote_after_vote"
                reference = json.loads(row["evidence"])["mark"]
                if abs(price / reference - 1) > MAX_REFERENCE_DRIFT:
                    return "professor_price_moved"
                return "unanimous" if self._qualified(performance, side) else "professor_learning_gate"
            if row:
                return "analyst_pending"
            if not self.analyst.configured:
                return "gemini_key_missing"
            if self._inflight is not None and not self._inflight.done():
                return "analyst_busy"
            if now_ms + 10_000 >= candle_ms + CANDLE_MS:
                return "insufficient_time_for_vote"
            count = db.execute("SELECT COUNT(*) FROM votes WHERE requested_ms >= ?", (
                self._day_start(now_ms),
            )).fetchone()[0]
            if count >= self.calls_per_day:
                return "gemini_daily_budget"
            context = {
                "symbol": symbol, "direction": side, "observed_at_utc": _iso(quote_ms).isoformat(),
                "mark": price, "stop": stop, "target": target,
                "rsi": market.get("rsi") if _positive(market.get("rsi")) else None,
                "ema9": market.get("ema9") if _positive(market.get("ema9")) else None,
                "trend_5m_15m_agrees": bool(market.get(f"{side}_bias")),
                "stress_cost_fraction_per_side": STRESS_COST_PER_SIDE,
                "prior_settled_shadow_mark_model": performance,
            }
            db.execute("INSERT INTO votes VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)", (
                event_id, symbol, side, now_ms, candle_ms + CANDLE_MS,
                self.analyst.model, json.dumps(context, allow_nan=False),
            ))

        self._inflight = self._pool.submit(self._review, event_id, context, candle_ms + CANDLE_MS)
        return "analyst_requested"

    def _track(self, db, position, signal: int, price: float, candle_ms: int, quote_ms: int) -> None:
        direction = 1 if position["side"] == "long" else -1
        net_pct = None
        if quote_ms - position["last_quote_ms"] > self.max_gap_ms:
            reason = "missing_quotes"
        elif position["filled_ms"] is None:
            if (
                signal == direction and position["entry_ms"] >= candle_ms
                and abs(price / position["entry"] - 1) <= MAX_REFERENCE_DRIFT
                and direction * (price - position["stop"]) > 0
                and direction * (position["target"] - price) > 0
            ):
                db.execute("""
                    UPDATE shadow SET filled_ms = ?, last_quote_ms = ?, entry = ? WHERE id = ?
                """, (quote_ms, quote_ms, price, position["id"]))
                return
            reason = "entry_not_fillable"
        elif direction * (price - position["stop"]) <= 0:
            reason = "stop"
        elif direction * (price - position["target"]) >= 0:
            reason = "target"
            price = position["target"]  # Ignore favorable overshoot.
        elif signal == -direction:
            reason = "opposite_signal"
        elif quote_ms - position["filled_ms"] >= self.strategy.max_hold_minutes * CANDLE_MS:
            reason = "time_exit"
        else:
            db.execute("UPDATE shadow SET last_quote_ms = ? WHERE id = ?", (quote_ms, position["id"]))
            return
        if reason not in {"missing_quotes", "entry_not_fillable"}:
            entry = position["entry"]
            gross = direction * (price - entry)
            fees = (entry + price) * STRESS_COST_PER_SIDE
            net_pct = (gross - fees) / entry * 100
        db.execute("UPDATE shadow SET exit_ms = ?, net_pct = ?, exit_reason = ? WHERE id = ?", (
            quote_ms, net_pct, reason, position["id"],
        ))

    def _review(self, event_id: str, context: dict, expires_ms: int) -> None:
        try:
            result = self.analyst.vote(context)
            result["status"] = "analyst_approved" if result["approve"] else "analyst_veto"
        except HTTPError as exc:
            result = {"approve": False, "status": f"gemini_http_{exc.code}"}
        except (URLError, OSError, KeyError, TypeError, ValueError, IndexError, AttributeError):
            result = {"approve": False, "status": "gemini_unavailable_or_invalid"}
        now_ms = time.time_ns() // 1_000_000
        if now_ms >= expires_ms:
            result = {**result, "approve": False, "status": "gemini_vote_expired"}
        with self._db() as db:
            db.execute("UPDATE votes SET result = ?, completed_ms = ? WHERE id = ?", (
                json.dumps(result, allow_nan=False), now_ms, event_id,
            ))

    def status(self, now_ms: int) -> dict:
        with self._db() as db:
            performance = self._performance(db, now_ms)
            counts = dict(db.execute("""
                SELECT COALESCE(exit_reason, 'open'), COUNT(*) FROM shadow GROUP BY exit_reason
            """).fetchall())
            votes = db.execute("""
                SELECT symbol, side, model, requested_ms, expires_ms, result
                FROM votes ORDER BY requested_ms DESC LIMIT 12
            """).fetchall()
            calls = db.execute("SELECT COUNT(*) FROM votes WHERE requested_ms >= ?", (
                self._day_start(now_ms),
            )).fetchone()[0]
        recent = []
        for row in votes:
            vote = dict(row)
            result = vote.pop("result")
            recent.append({
                **vote, **(json.loads(result) if result else {
                    "status": "pending" if now_ms < row["expires_ms"] else "expired_without_response",
                }),
            })
        return {
            "configured": self.analyst.configured, "model": self.analyst.model,
            "model_requests_today": calls, "model_daily_limit": self.calls_per_day,
            "sample_counts": counts, "performance": performance, "recent_votes": recent,
            "shadow_only_metrics": True,
            "demo_evidence_ready": self._qualified(performance),
            "min_evidence_trades": MIN_TRADES, "min_evidence_win_rate": MIN_WIN_RATE,
        }

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
