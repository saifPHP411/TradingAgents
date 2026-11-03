"""Phase 27: the independent current-market decision path.

The defect these tests answer: the paper worker's decision source replayed a
*historical* experiment, so once its sessions were consumed it reported
``HISTORY_EXHAUSTED`` and the worker could never trade on the live market. This
suite pins a *separate* source that reasons on current Alpaca data, is gated on
Alpaca's market clock, reuses the same strategy/inference seam, and reaches the
*same* worker, risk layer and broker.

Everything here is offline and deterministic: the broker transport, the market
clock and the decision-data fetch are all injected, so no test reaches the
network (the suite's autouse guard refuses it) and none needs an Alpaca key. The
tests drive the real code paths — the real ``CurrentMarketDecisionSource``, the
real ``PaperTradingWorker``, the real risk checks — not stubs of them.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tradingagents.paper_experiment import ExperimentConfig, PaperExperiment
from tradingagents.paper_experiment.broker import (
    DECISION_SOURCE_CURRENT_MARKET,
    DECISION_SOURCE_HISTORY,
    NO_TRADE_CURRENT_MARKET_NO_NEW_DATA,
    NO_TRADE_HISTORY_EXHAUSTED,
    NO_TRADE_MARKET_CLOSED,
    NO_TRADE_MARKET_DATA_UNAVAILABLE,
    NO_TRADE_NO_SIGNAL,
    NO_TRADE_REASONS,
    NO_TRADE_RISK_REJECTED,
    AlpacaPaperBroker,
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    StaticMarketData,
)
from tradingagents.paper_experiment.current_market_data import (
    AlpacaCurrentBars,
    CurrentMarketDataError,
)
from tradingagents.paper_experiment.current_source import CurrentMarketDecisionSource
from tradingagents.paper_experiment.dashboard.engine_view import engine_view
from tradingagents.paper_experiment.inference import StubInference
from tradingagents.paper_experiment.market_data_wiring import (
    DECISION_SOURCE_ENV,
    describe_decision_source,
    env_decision_source,
    resolve_decision_source,
)
from tradingagents.paper_experiment.market_hours import (
    MARKET_CLOSED,
    MARKET_OPEN,
    MARKET_UNKNOWN,
    AlpacaMarketClock,
    MarketHoursError,
    StaticMarketHours,
)
from tradingagents.paper_experiment.paper_trading import ExperimentDecisionSource
from tradingagents.paper_experiment.providers.base import ProviderError
from tradingagents.paper_experiment.sources import StaticPriceSource

SYMBOL = "SPY"
SESSION = "2026-10-29"

_ACCOUNT = {
    "id": "paper-acct-1",
    "status": "ACTIVE",
    "cash": "5000.00",
    "equity": "5000.00",
    "buying_power": "5000.00",
    "currency": "USD",
    "trading_blocked": False,
    "account_blocked": False,
    "shorting_enabled": False,
}


def _frame(periods: int = 40, *, start: float = 100.0, end: str = SESSION) -> pd.DataFrame:
    index = pd.date_range(end=end, periods=periods, freq="B")
    opens = [start + i for i in range(periods)]
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


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)


def _bars_payload(frame: pd.DataFrame, symbol: str = SYMBOL) -> dict:
    bars = []
    for stamp, row in frame.iterrows():
        bars.append(
            {
                "t": f"{str(stamp)[:10]}T04:00:00Z",
                "o": float(row["open"]),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
                "v": float(row["volume"]),
            }
        )
    return {"bars": {symbol: bars}, "next_page_token": None}


def _clock(is_open: bool, session: str = SESSION, **over) -> dict:
    payload = {
        "timestamp": f"{session}T14:00:00-04:00",
        "is_open": is_open,
        "next_open": f"{session}T09:30:00-04:00",
        "next_close": f"{session}T16:00:00-04:00",
    }
    payload.update(over)
    return payload


def _clock_transport(payload_or_error):
    def send(*, url, params, headers, timeout):
        if isinstance(payload_or_error, Exception):
            raise payload_or_error
        return _Resp(200, payload_or_error)

    return send


def _bars_transport(payload_or_error, *, calls=None):
    def send(*, url, params, headers, timeout):
        if calls is not None:
            calls.append({"url": url, "params": dict(params)})
        if isinstance(payload_or_error, Exception):
            raise payload_or_error
        return _Resp(200, payload_or_error)

    return send


def _bars_source(frame: pd.DataFrame | None = None, **kwargs) -> AlpacaCurrentBars:
    return AlpacaCurrentBars(
        api_key="paper-key",
        api_secret="paper-secret",
        transport=_bars_transport(_bars_payload(frame if frame is not None else _frame())),
        **kwargs,
    )


def _current_source(
    *,
    frame: pd.DataFrame | None = None,
    is_open: bool = True,
    session: str = SESSION,
    rating: str = "Buy",
    clock_error: Exception | None = None,
    bars_error: Exception | None = None,
    market_data=None,
    is_consumed=None,
    state_path=None,
    order_notional: float = 25.0,
):
    clock = StaticMarketHours(is_open=is_open, session=session, error=(
        str(clock_error) if clock_error is not None else None
    ))
    if market_data is None:
        market_data = (
            AlpacaCurrentBars(
                api_key="paper-key",
                api_secret="paper-secret",
                transport=_bars_transport(
                    bars_error if bars_error is not None else _bars_payload(
                        frame if frame is not None else _frame()
                    )
                ),
            )
            if bars_error is not None or frame is not None
            else _bars_source(frame)
        )
    return CurrentMarketDecisionSource(
        symbols=(SYMBOL,),
        inference=StubInference(rating=rating),
        market_data=market_data,
        market_hours=clock,
        order_notional=order_notional,
        state_path=state_path,
        is_consumed=is_consumed,
    )


# ---------------------------------------------------------------------------
# 1. The history source is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHistorySourceUnchanged:
    def test_history_source_still_reports_history_exhausted(self, tmp_path):
        source = StaticPriceSource({SYMBOL: _frame(periods=6)}, name="static_test")
        config = ExperimentConfig.from_protocol(
            symbols=(SYMBOL,), decision_frequency_sessions=1, evaluation_lag_sessions=1
        )
        experiment = PaperExperiment(
            source=source, config=config, root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()
        for session in list(source.sessions()):
            experiment.process_session(session)

        history = ExperimentDecisionSource(experiment, order_notional=25.0)
        # Consume everything the source will offer.
        while history.next_decision() is not None:
            pass
        assert history.skip_code == NO_TRADE_HISTORY_EXHAUSTED

    def test_default_decision_source_is_history(self):
        assert PaperTradingConfig().decision_source == DECISION_SOURCE_HISTORY
        assert env_decision_source() == DECISION_SOURCE_HISTORY


# ---------------------------------------------------------------------------
# 2. Current source works when history is exhausted
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCurrentSourceIndependentOfHistory:
    def test_current_source_works_when_history_is_exhausted(self, tmp_path):
        # A fully-consumed historical experiment.
        source = StaticPriceSource({SYMBOL: _frame(periods=4)}, name="static_test")
        config = ExperimentConfig.from_protocol(
            symbols=(SYMBOL,), decision_frequency_sessions=1, evaluation_lag_sessions=1
        )
        experiment = PaperExperiment(
            source=source, config=config, root=tmp_path / "exp",
            inference=StubInference(rating="Buy"),
        )
        experiment.ensure_created()
        for session in list(source.sessions()):
            experiment.process_session(session)
        history = ExperimentDecisionSource(experiment, order_notional=25.0)
        while history.next_decision() is not None:
            pass
        assert history.skip_code == NO_TRADE_HISTORY_EXHAUSTED

        # The current source, on its own data, still yields a genuine decision.
        current = _current_source()
        decision = current.next_decision()
        assert decision is not None
        assert decision.side == "buy"
        assert decision.symbol == SYMBOL

    def test_history_exhausted_code_does_not_block_current_mode(self):
        current = _current_source()
        decision = current.next_decision()
        assert decision is not None
        assert current.skip_code is None
        assert current.skip_code != NO_TRADE_HISTORY_EXHAUSTED


# ---------------------------------------------------------------------------
# 3. Valid current data reaches strategy evaluation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCurrentDataReachesTheStrategy:
    def test_valid_alpaca_current_data_reaches_the_strategy(self):
        calls = []
        recorded: list = []

        class _Recording:
            def __call__(self, *, symbol, session, window=None):
                recorded.append((symbol, session, window))
                return "Buy", "recorded"

        bars = AlpacaCurrentBars(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=_bars_transport(_bars_payload(_frame()), calls=calls),
        )
        source = CurrentMarketDecisionSource(
            symbols=(SYMBOL,),
            inference=_Recording(),
            market_data=bars,
            market_hours=StaticMarketHours(is_open=True, session=SESSION),
            order_notional=25.0,
        )
        decision = source.next_decision()
        assert decision is not None
        # The strategy was handed the *real* fetched window, not an empty one.
        assert len(recorded) == 1
        symbol, session, window = recorded[0]
        assert symbol == SYMBOL
        assert session == SESSION
        assert len(window) == 40
        assert calls and calls[0]["url"].endswith("/v2/stocks/bars")
        assert calls[0]["params"].get("feed") == "iex"

    def test_the_window_carries_the_current_session_bar(self):
        frame = _frame()
        assert str(frame.index[-1])[:10] == SESSION
        window = _bars_source(frame).fetch(SYMBOL, session=SESSION)
        assert window.last_bar_date == SESSION
        assert window.bar_count == 40


# ---------------------------------------------------------------------------
# 4. Stale / malformed / future data is refused
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFreshness:
    def test_stale_current_data_is_refused(self):
        # The newest bar is a month before the session: not current.
        frame = _frame(end="2026-09-29")
        source = _bars_source(frame)
        with pytest.raises(CurrentMarketDataError) as exc:
            source.fetch(SYMBOL, session=SESSION)
        assert exc.value.kind == "stale"

    def test_stale_data_returns_market_data_unavailable(self):
        frame = _frame(end="2026-09-29")
        source = _current_source(frame=frame)
        decision = source.next_decision()
        assert decision is None
        assert source.skip_code == NO_TRADE_MARKET_DATA_UNAVAILABLE

    def test_future_dated_data_is_refused(self):
        frame = _frame(end="2026-11-30")  # after the decision session
        source = _bars_source(frame)
        with pytest.raises(CurrentMarketDataError) as exc:
            source.fetch(SYMBOL, session=SESSION)
        assert exc.value.kind == "future"

    def test_future_data_returns_market_data_unavailable(self):
        source = _current_source(frame=_frame(end="2026-11-30"))
        decision = source.next_decision()
        assert decision is None
        assert source.skip_code == NO_TRADE_MARKET_DATA_UNAVAILABLE

    def test_malformed_bar_is_refused(self):
        payload = _bars_payload(_frame())
        payload["bars"][SYMBOL][-1]["c"] = "not-a-number"
        bars = AlpacaCurrentBars(
            api_key="k", api_secret="s",
            transport=_bars_transport(payload),
        )
        with pytest.raises(CurrentMarketDataError) as exc:
            bars.fetch(SYMBOL, session=SESSION)
        assert exc.value.kind == "malformed"

    def test_duplicate_session_is_refused(self):
        payload = _bars_payload(_frame())
        payload["bars"][SYMBOL].append(dict(payload["bars"][SYMBOL][-1]))
        bars = AlpacaCurrentBars(api_key="k", api_secret="s", transport=_bars_transport(payload))
        with pytest.raises(CurrentMarketDataError) as exc:
            bars.fetch(SYMBOL, session=SESSION)
        assert exc.value.kind == "malformed"

    def test_non_finite_price_is_refused(self):
        payload = _bars_payload(_frame())
        payload["bars"][SYMBOL][-1]["h"] = float("nan")
        bars = AlpacaCurrentBars(api_key="k", api_secret="s", transport=_bars_transport(payload))
        with pytest.raises(CurrentMarketDataError) as exc:
            bars.fetch(SYMBOL, session=SESSION)
        assert exc.value.kind == "malformed"

    def test_a_transport_failure_is_unavailable_not_a_decision(self):
        source = _current_source(bars_error=OSError("connection reset"))
        decision = source.next_decision()
        assert decision is None
        assert source.skip_code == NO_TRADE_MARKET_DATA_UNAVAILABLE

    def test_missing_credentials_are_reported_not_defaulted(self, monkeypatch):
        monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_PAPER_API_SECRET", raising=False)
        bars = AlpacaCurrentBars(transport=_bars_transport(_bars_payload(_frame())))
        with pytest.raises(CurrentMarketDataError) as exc:
            bars.fetch(SYMBOL, session=SESSION)
        assert exc.value.kind == "not_configured"

    def test_an_empty_bar_series_is_refused(self):
        bars = AlpacaCurrentBars(
            api_key="k", api_secret="s",
            transport=_bars_transport({"bars": {SYMBOL: []}}),
        )
        with pytest.raises(CurrentMarketDataError) as exc:
            bars.fetch(SYMBOL, session=SESSION)
        assert exc.value.kind == "no_data"


# ---------------------------------------------------------------------------
# 5. Market hours
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMarketHours:
    def test_market_closed_returns_market_closed(self):
        source = _current_source(is_open=False)
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_MARKET_CLOSED

    def test_a_weekend_is_closed(self):
        # The clock is the authority; a weekend session simply reports closed.
        source = _current_source(is_open=False, session="2026-10-31")
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_MARKET_CLOSED

    def test_an_unreadable_clock_is_market_data_unavailable(self):
        source = _current_source(clock_error=MarketHoursError("clock 500"))
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_MARKET_DATA_UNAVAILABLE

    def test_open_market_but_unavailable_data_is_market_data_unavailable(self):
        source = _current_source(is_open=True, bars_error=ProviderError("503"))
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_MARKET_DATA_UNAVAILABLE

    def test_the_clock_never_places_an_order(self):
        calls = []

        def send(*, url, params, headers, timeout):
            calls.append(url)
            return _Resp(200, _clock(True))

        clock = AlpacaMarketClock(api_key="k", api_secret="s", transport=send)
        status = clock.status()
        assert status.state == MARKET_OPEN
        assert status.session == SESSION
        assert calls == ["https://paper-api.alpaca.markets/v2/clock"]

    def test_the_clock_reports_closed_and_unknown(self):
        closed = AlpacaMarketClock(
            api_key="k", api_secret="s", transport=_clock_transport(_clock(False))
        )
        assert closed.status().state == MARKET_CLOSED
        unknown = AlpacaMarketClock(
            api_key="k", api_secret="s", transport=_clock_transport(OSError("down"))
        )
        with pytest.raises(MarketHoursError):
            unknown.status()
        assert MARKET_UNKNOWN not in (MARKET_OPEN, MARKET_CLOSED)

    def test_a_naive_clock_timestamp_is_refused(self):
        payload = _clock(True, timestamp="2026-10-29T14:00:00")
        clock = AlpacaMarketClock(api_key="k", api_secret="s", transport=_clock_transport(payload))
        with pytest.raises(MarketHoursError):
            clock.status()


# ---------------------------------------------------------------------------
# 6. An honest NO_TRADE
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHonestNoTrade:
    def test_a_hold_rating_is_an_honest_no_trade(self):
        source = _current_source(rating="Hold")
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_NO_SIGNAL

    def test_an_actionless_rating_is_a_no_trade(self):
        source = _current_source(rating="REVIEW")
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_NO_SIGNAL

    def test_a_sell_rating_is_a_sell_decision(self):
        source = _current_source(rating="Sell")
        decision = source.next_decision()
        assert decision is not None and decision.side == "sell"

    def test_no_signal_is_never_a_fabricated_side(self):
        source = _current_source(rating="Hold")
        assert source.next_decision() is None
        # Nothing was recorded as a tradeable decision.
        assert source.last_decision_for(SYMBOL)["side"] is None


# ---------------------------------------------------------------------------
# 7. Cadence, determinism and idempotency
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCadenceAndIdempotency:
    def test_one_decision_per_symbol_per_session(self):
        source = _current_source()
        first = source.next_decision()
        assert first is not None
        # A second poll in the same session yields nothing new.
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_CURRENT_MARKET_NO_NEW_DATA

    def test_decision_id_is_deterministic(self):
        source = _current_source()
        decision = source.next_decision()
        assert decision.decision_id == f"current-{SYMBOL}-{SESSION}"
        # A fresh source in the same session produces the same id.
        other = _current_source()
        assert other.next_decision().decision_id == decision.decision_id

    def test_a_new_session_yields_a_new_decision(self):
        clock = StaticMarketHours(is_open=True, session=SESSION)
        source = CurrentMarketDecisionSource(
            symbols=(SYMBOL,),
            inference=StubInference(rating="Buy"),
            market_data=_bars_source(),
            market_hours=clock,
            order_notional=25.0,
        )
        first = source.next_decision()
        assert first is not None
        clock._session = "2026-10-30"  # the market rolls to the next session
        second = source.next_decision()
        assert second is not None
        assert second.decision_id != first.decision_id

    def test_a_restart_within_a_session_replays_the_same_decision(self, tmp_path):
        state = tmp_path / "current_source_state.json"
        first = _current_source(state_path=state)
        decision = first.next_decision()
        assert decision is not None

        # A fresh source (a restart) reads its own state and replays the id.
        restarted = _current_source(state_path=state)
        replayed = restarted.next_decision()
        assert replayed is not None
        assert replayed.decision_id == decision.decision_id

    def test_the_model_is_not_called_twice_for_one_session(self, tmp_path):
        state = tmp_path / "current_source_state.json"
        calls = []

        class _Counting:
            def __call__(self, *, symbol, session, window=None):
                calls.append(session)
                return "Buy", "x"

        source = CurrentMarketDecisionSource(
            symbols=(SYMBOL,),
            inference=_Counting(),
            market_data=_bars_source(),
            market_hours=StaticMarketHours(is_open=True, session=SESSION),
            order_notional=25.0,
            state_path=state,
        )
        assert source.next_decision() is not None
        assert source.next_decision() is None
        assert len(calls) == 1

    def test_a_consumed_decision_is_not_re_offered(self):
        offered = {f"current-{SYMBOL}-{SESSION}"}
        source = _current_source(is_consumed=lambda did: did in offered)
        assert source.next_decision() is None
        assert source.skip_code == NO_TRADE_CURRENT_MARKET_NO_NEW_DATA

    def test_the_current_source_does_not_fabricate_size_or_stop(self):
        source = _current_source()
        decision = source.next_decision()
        assert decision.notional == 25.0  # the worker's configured target
        assert decision.stop_loss is None
        assert decision.take_profit is None


# ---------------------------------------------------------------------------
# 8. The current decision reaches the existing worker, risk and broker
# ---------------------------------------------------------------------------


def _order(**over):
    payload = {
        "id": "broker-order-1",
        "client_order_id": "c1",
        "symbol": SYMBOL,
        "side": "buy",
        "qty": "0.25",
        "filled_qty": "0.0",
        "filled_avg_price": None,
        "status": "accepted",
        "type": "market",
        "time_in_force": "day",
        "submitted_at": "2026-10-29T13:30:00Z",
    }
    payload.update(over)
    return payload


def _transport(routes, *, calls=None):
    def send(*, method, url, headers, params, json_body, timeout):
        if calls is not None:
            calls.append({"method": method, "url": url, "json": json_body})
        for (m, suffix), response in routes.items():
            if method == m and url.endswith(suffix):
                return response() if callable(response) else response
        raise AssertionError(f"unexpected request {method} {url}")

    return send


def _routes(*, orders=None, limits=None, account=None):
    return {
        ("GET", "/v2/account"): _Resp(200, account or _ACCOUNT),
        ("GET", "/v2/positions"): _Resp(200, []),
        ("POST", "/v2/orders"): orders or _Resp(200, _order()),
        ("GET", "/v2/orders:by_client_order_id"): _Resp(200, _order()),
    }


def _broker_factory(routes, *, limits=None):
    def build():
        return AlpacaPaperBroker(
            api_key="paper-key",
            api_secret="paper-secret",
            transport=_transport(routes),
            limits=limits or PaperRiskLimits(),
        )

    return build


def _worker(root, source, *, routes=None, market_price=101.0, limits=None, whole_shares=False):
    config = PaperTradingConfig(
        strategy_id="current-market",
        order_notional=25.0,
        whole_shares=whole_shares,
        state_path=root / "worker_state.json",
        market_data_provider="bar_open",
        decision_source=DECISION_SOURCE_CURRENT_MARKET,
        limits=limits or PaperRiskLimits(),
    )
    worker = PaperTradingWorker(
        config=config,
        decision_source=source,
        market_data=StaticMarketData(prices={SYMBOL: market_price}),
        broker_factory=_broker_factory(routes or _routes()),
    )
    return worker


@pytest.mark.unit
class TestTheCurrentChain:
    def test_a_current_buy_reaches_the_existing_risk_layer_and_broker(self, tmp_path):
        source = _current_source()
        worker = _worker(tmp_path, source)
        worker.start()
        result = worker.run_once()
        worker.stop()

        assert result.ran is True
        assert result.reason_code is None
        assert result.order is not None
        assert result.order.symbol == SYMBOL
        assert result.order.side == "buy"
        assert result.reference_price == pytest.approx(101.0)
        assert result.market_data_provider == "static"

    def test_a_risk_rejection_prevents_order_submission(self, tmp_path):
        calls = []
        # The broker's own limit refuses the $25 order before it is submitted. The
        # config's notional is left at its default, so it is the *risk layer* — not
        # configuration validation — that refuses the order.
        routes = _routes()
        source = _current_source()
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                strategy_id="current-market",
                order_notional=25.0,
                whole_shares=False,
                state_path=tmp_path / "worker_state.json",
                decision_source=DECISION_SOURCE_CURRENT_MARKET,
            ),
            decision_source=source,
            market_data=StaticMarketData(prices={SYMBOL: 101.0}),
            broker_factory=lambda: AlpacaPaperBroker(
                api_key="paper-key",
                api_secret="paper-secret",
                transport=_transport(routes, calls=calls),
                limits=PaperRiskLimits(max_order_notional=5.0),
            ),
        )
        worker.start()
        result = worker.run_once()
        worker.stop()
        assert result.ran is False
        assert result.reason_code == NO_TRADE_RISK_REJECTED
        assert not any(c["method"] == "POST" for c in calls)

    def test_an_approved_current_decision_reaches_the_paper_broker(self, tmp_path):
        calls = []
        routes = _routes()
        source = _current_source()
        worker = PaperTradingWorker(
            config=PaperTradingConfig(
                strategy_id="current-market",
                order_notional=25.0,
                whole_shares=False,
                state_path=tmp_path / "worker_state.json",
                decision_source=DECISION_SOURCE_CURRENT_MARKET,
            ),
            decision_source=source,
            market_data=StaticMarketData(prices={SYMBOL: 101.0}),
            broker_factory=lambda: AlpacaPaperBroker(
                api_key="paper-key",
                api_secret="paper-secret",
                transport=_transport(routes, calls=calls),
                limits=PaperRiskLimits(),
            ),
        )
        worker.start()
        worker.run_once()
        worker.stop()
        assert any(c["method"] == "POST" and c["url"].endswith("/v2/orders") for c in calls)

    def test_a_duplicate_cannot_be_ordered_after_a_restart(self, tmp_path):
        source = _current_source()
        worker = _worker(tmp_path, source)
        worker.start()
        first = worker.run_once()
        worker.stop()
        assert first.order is not None
        client_order_id = first.order.client_order_id

        # A restart, with the same source object and the same persisted state.
        restarted = PaperTradingWorker(
            config=worker.config,
            decision_source=source,
            market_data=worker.market_data,
            broker_factory=_broker_factory(_routes()),
        )
        restarted.start()
        second = restarted.run_once()
        restarted.stop()
        assert second.reason_code in (
            NO_TRADE_CURRENT_MARKET_NO_NEW_DATA,
            "ALREADY_PROCESSED",
            "ORDER_EXISTS",
        )
        # No second order id was created.
        assert set(restarted.state.orders) == {client_order_id}

    def test_the_current_source_does_not_write_the_experiment_state(self, tmp_path):
        root = tmp_path / "exp"
        root.mkdir()
        before = sorted(p.name for p in root.iterdir())
        source = _current_source(state_path=root / "current_source_state.json")
        source.next_decision()
        after = sorted(p.name for p in root.iterdir())
        # It writes only its *own* state file, never session_state.json/journal.json.
        assert "current_source_state.json" in after
        assert "session_state.json" not in after
        assert "journal.json" not in after
        assert before == []

    def test_the_current_source_state_is_separate_from_the_experiment(self, tmp_path):
        state = tmp_path / "current_source_state.json"
        source = _current_source(state_path=state)
        source.next_decision()
        payload = json.loads(state.read_text())
        assert payload["schema_version"] == 1
        assert SYMBOL in payload["decisions"]


# ---------------------------------------------------------------------------
# 9. Configuration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfiguration:
    def test_default_mode_remains_history(self):
        assert PaperTradingConfig().decision_source == DECISION_SOURCE_HISTORY
        assert env_decision_source() == DECISION_SOURCE_HISTORY

    def test_current_market_is_opt_in(self):
        assert resolve_decision_source("current_market") == DECISION_SOURCE_CURRENT_MARKET

    def test_an_unknown_decision_source_is_refused_not_defaulted(self):
        from tradingagents.paper_experiment.market_data_wiring import MarketDataWiringError

        with pytest.raises(MarketDataWiringError):
            resolve_decision_source("live")
        with pytest.raises(MarketDataWiringError):
            resolve_decision_source(None)

    def test_the_env_var_selects_the_source(self, monkeypatch):
        monkeypatch.setenv(DECISION_SOURCE_ENV, "current_market")
        assert env_decision_source() == DECISION_SOURCE_CURRENT_MARKET

    def test_an_unknown_config_value_is_refused(self):
        from tradingagents.paper_experiment.broker import WorkerError

        with pytest.raises(WorkerError):
            PaperTradingConfig(decision_source="live")

    def test_describe_decision_source_names_both(self):
        assert "history" in describe_decision_source("history").lower()
        assert "current" in describe_decision_source("current_market").lower()

    def test_the_cli_exposes_the_flag_defaulting_to_history(self):
        from tradingagents.paper_experiment.__main__ import _build_parser

        help_text = _build_parser().format_help()
        assert "--paper-decision-source" in help_text
        assert "current_market" in help_text and "history" in help_text


# ---------------------------------------------------------------------------
# 10. No live trading path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoLivePath:
    def test_the_current_data_host_refuses_a_live_host(self):
        from tradingagents.paper_experiment.providers.alpaca_iex import assert_data_host
        from tradingagents.paper_experiment.providers.base import ProviderUnsupportedError

        with pytest.raises(ProviderUnsupportedError):
            assert_data_host("https://api.alpaca.markets")

    def test_the_market_clock_refuses_a_live_host(self):
        from tradingagents.paper_experiment.broker import LiveEndpointRefusedError

        with pytest.raises(LiveEndpointRefusedError):
            AlpacaMarketClock(api_key="k", api_secret="s", base_url="https://api.alpaca.markets")

    def test_no_live_decision_source_choice_exists(self):
        assert "live" not in (DECISION_SOURCE_HISTORY, DECISION_SOURCE_CURRENT_MARKET)
        assert "live" not in NO_TRADE_REASONS

    def test_the_config_refuses_a_live_endpoint(self):
        from tradingagents.paper_experiment.broker import LiveEndpointRefusedError

        with pytest.raises(LiveEndpointRefusedError):
            PaperTradingConfig(base_url="https://api.alpaca.markets")


# ---------------------------------------------------------------------------
# 11. Observability
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestObservability:
    def test_the_status_view_names_the_decision_source(self, tmp_path):
        source = _current_source(state_path=tmp_path / "s.json")
        worker = _worker(tmp_path, source)
        worker.start()
        worker.run_once()
        status = worker.status()
        worker.stop()
        assert status["decision_source_configured"] == DECISION_SOURCE_CURRENT_MARKET
        assert status["decision_source_kind"] == "current_market"
        assert status["decision_data_provider"] == "alpaca_current_bars"
        assert status["decision_data_feed"] == "iex"

    def test_the_engine_view_exposes_the_current_path(self, tmp_path):
        source = _current_source(state_path=tmp_path / "s.json")
        worker = _worker(tmp_path, source)
        worker.start()
        worker.run_once()
        view = engine_view(worker.status())
        worker.stop()
        assert view["decision_source"]["kind"] == "current_market"
        assert view["decision_source"]["decision_data_provider"] == "alpaca_current_bars"
        assert view["states"]["order_submitted"] is True

    def test_a_closed_market_surfaces_as_a_named_blocker(self, tmp_path):
        source = _current_source(is_open=False)
        worker = _worker(tmp_path, source)
        worker.start()
        worker.run_once()
        view = engine_view(worker.status())
        worker.stop()
        assert view["no_trade_code"] == NO_TRADE_MARKET_CLOSED
        assert view["blocker"]["code"] == NO_TRADE_MARKET_CLOSED

    def test_the_status_view_exposes_no_secret(self, tmp_path):
        source = _current_source(state_path=tmp_path / "s.json")
        worker = _worker(tmp_path, source)
        worker.start()
        worker.run_once()
        blob = json.dumps(worker.status())
        worker.stop()
        assert "paper-key" not in blob and "paper-secret" not in blob


# ---------------------------------------------------------------------------
# 12. The controller wires the current source
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestControllerWiring:
    def _experiment_root(self, tmp_path):
        root = tmp_path / "exp"
        source = StaticPriceSource({SYMBOL: _frame(periods=6)}, name="static_test")
        config = ExperimentConfig.from_protocol(
            symbols=(SYMBOL,), decision_frequency_sessions=1, evaluation_lag_sessions=1
        )
        experiment = PaperExperiment(
            source=source, config=config, root=root, inference=StubInference(rating="Buy")
        )
        experiment.ensure_created()
        experiment.process_session(list(source.sessions())[0])
        return root

    def test_history_is_the_default_controller_source(self, tmp_path):
        from tradingagents.paper_experiment.dashboard.paper_control import PaperTradingController

        root = self._experiment_root(tmp_path)
        controller = PaperTradingController(root, enabled=True)
        worker = controller._ensure_worker()  # noqa: SLF001
        assert isinstance(worker.decision_source, ExperimentDecisionSource)

    def test_current_market_builds_the_current_source(self, tmp_path):
        from tradingagents.paper_experiment.dashboard.paper_control import PaperTradingController

        root = self._experiment_root(tmp_path)
        controller = PaperTradingController(
            root,
            enabled=True,
            decision_source_kind=DECISION_SOURCE_CURRENT_MARKET,
            market_hours_gate=lambda: StaticMarketHours(is_open=True, session=SESSION),
            decision_data_factory=lambda: _bars_source(),
        )
        worker = controller._ensure_worker()  # noqa: SLF001
        assert isinstance(worker.decision_source, CurrentMarketDecisionSource)
        assert worker.config.decision_source == DECISION_SOURCE_CURRENT_MARKET

    def test_the_current_source_uses_the_same_strategy_seam(self, tmp_path):
        from tradingagents.paper_experiment.dashboard.paper_control import PaperTradingController
        from tradingagents.paper_experiment.inference import ProviderInference

        root = self._experiment_root(tmp_path)
        controller = PaperTradingController(
            root,
            enabled=True,
            decision_source_kind=DECISION_SOURCE_CURRENT_MARKET,
            market_hours_gate=lambda: StaticMarketHours(is_open=True, session=SESSION),
            decision_data_factory=lambda: _bars_source(),
        )
        worker = controller._ensure_worker()  # noqa: SLF001
        # The current source reuses the *same* ProviderInference the historical
        # path builds from the persisted config — no second strategy.
        assert isinstance(worker.decision_source.inference, ProviderInference)


@pytest.mark.unit
class TestPromptReuse:
    def test_the_extracted_builder_is_the_one_provider_inference_sends(self):
        from tradingagents.paper_experiment.inference import (
            ProviderInference,
            render_strategy_prompt,
        )

        class _Client:
            def __init__(self):
                self.prompts = []

            def get_llm(self):
                outer = self

                class _LLM:
                    def invoke(self, prompt, **kwargs):
                        outer.prompts.append(prompt)
                        return type("R", (), {"content": "**Rating**: Buy"})()

                return _LLM()

        client = _Client()
        inference = ProviderInference(
            provider="groq", model="m", client_factory=lambda **k: client
        )
        import os

        os.environ["GROQ_API_KEY"] = "g" * 32
        try:
            frame = _frame()
            inference(symbol=SYMBOL, session=SESSION, window=frame)
        finally:
            os.environ.pop("GROQ_API_KEY", None)
        expected = render_strategy_prompt(symbol=SYMBOL, session=SESSION, window=frame)
        assert client.prompts[0] == expected
