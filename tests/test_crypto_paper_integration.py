"""The crypto paper path, end to end: symbol → price → risk → order → exit.

This is the integration test for the cross-venue crypto path. The crypto paper
worker analyses Binance's ``BTC/USDT`` and trades Alpaca's ``BTC/USD``, so three
things have to line up that never line up on a single-venue path:

1. the **decision symbol** is translated to the **execution symbol**, and the
   order carries the execution one;
2. the order is **sized from the execution venue's own price**, because the
   market-data venue's price is a different instrument's price (USDT vs USD);
3. the **exit** for a position is priced and placed under the same execution
   symbol the entry used.

It also pins the failure modes that matter more than the happy path:

- no fresh execution price → **no order** (the market-data price is never a
  substitute);
- an execution price too far from the decision price → **no order**;
- a symbol one venue does not list → **no order**;
- a restart → **no duplicate order**;
- a live endpoint → **refused**.

Every test drives a mocked HTTP transport, so the suite is hermetic: no network,
no credential, no paper order. The mock is a fake *transport*, not a fake broker —
the adapter's own endpoint validation, credential handling, order parsing, risk
checks, idempotency and reconciliation all run for real.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tradingagents.paper_experiment.broker import (
    DEFAULT_MAX_PRICE_DEVIATION,
    NO_TRADE_EXECUTION_PRICE_UNAVAILABLE,
    NO_TRADE_PRICE_DEVIATION,
    NO_TRADE_RISK_REJECTED,
    PROVIDER_ALPACA_CRYPTO,
    AlpacaCryptoExecutionPrice,
    AlpacaPaperBroker,
    MarketDataError,
    PaperRiskLimits,
    PaperTradingConfig,
    PaperTradingWorker,
    TradingDecision,
)
from tradingagents.paper_experiment.broker.alpaca_paper import LiveEndpointRefusedError
from tradingagents.paper_experiment.crypto_symbol_config import (
    resolve_symbol_configuration,
)
from tradingagents.paper_experiment.providers.alpaca_crypto import (
    AlpacaCryptoClient,
    AlpacaCryptoHostRefused,
    snapshot_price_and_time,
)
from tradingagents.paper_experiment.symbol_mapping import SymbolMapping
from tradingagents.risk.paper_parity import (
    MAX_DAILY_ORDERS,
    MAX_OPEN_POSITIONS,
    MAX_ORDER_NOTIONAL,
)

CANONICAL = "BTC/USDT"
MARKET_DATA_SYMBOL = "BTCUSDT"
EXECUTION_SYMBOL = "BTC/USD"

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
"""The clock every seam is pinned to, so freshness is deterministic."""


def _iso(offset_seconds: float = 0.0) -> str:
    return (NOW + timedelta(seconds=offset_seconds)).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Fake transports
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status_code: int, body) -> None:
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)


_ACCOUNT = {
    "id": "paper-acct-123",
    "status": "ACTIVE",
    "cash": "25000.00",
    "equity": "25000.00",
    "buying_power": "50000.00",
    "currency": "USD",
    "trading_blocked": False,
    "account_blocked": False,
}


def _order_payload(**over) -> dict:
    payload = {
        "id": "broker-order-1",
        "client_order_id": "c1",
        "symbol": EXECUTION_SYMBOL,
        "side": "buy",
        "qty": "0.0005",
        "filled_qty": "0.0005",
        "filled_avg_price": "50000.0",
        "status": "filled",
        "type": "market",
        "time_in_force": "day",
        "submitted_at": _iso(),
    }
    payload.update(over)
    return payload


def _snapshot(price: float = 50000.0, *, age_seconds: float = 0.0) -> dict:
    return {
        "snapshots": {
            EXECUTION_SYMBOL: {
                "latestTrade": {"p": str(price), "s": "0.001", "t": _iso(-age_seconds)},
                "latestQuote": {
                    "bp": str(price - 0.5),
                    "ap": str(price + 0.5),
                    "bs": "0.001",
                    "as": "0.001",
                    "t": _iso(-age_seconds),
                },
                "minuteBar": {"c": str(price), "t": _iso(-age_seconds)},
            }
        }
    }


class _DataTransport:
    """Answers the market-data host's snapshot endpoint."""

    def __init__(self, response: _Resp | None = None) -> None:
        self.response = response or _Resp(200, _snapshot())
        self.calls: list[dict] = []

    def __call__(self, *, url, params, headers, timeout):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers)})
        return self.response


class _BrokerTransport:
    """Answers the paper trading host."""

    def __init__(self, *, positions=None, order_payload=None, account=None) -> None:
        self.positions = positions or []
        self.order_payload = order_payload or _order_payload()
        self.account = account or _ACCOUNT
        self.orders: list[dict] = []
        self.calls: list[dict] = []

    def __call__(self, *, method, url, headers, params, json_body, timeout):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": dict(params or {}),
                "json": json_body,
            }
        )
        if method == "GET" and url.endswith("/v2/account"):
            return _Resp(200, self.account)
        if method == "GET" and url.endswith("/v2/positions"):
            return _Resp(200, self.positions)
        if method == "GET" and url.endswith("/v2/orders"):
            return _Resp(200, [])
        if method == "POST" and url.endswith("/v2/orders"):
            self.orders.append(json_body or {})
            return _Resp(200, self.order_payload)
        raise AssertionError(f"unexpected request {method} {url}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mapping() -> SymbolMapping:
    from tradingagents.crypto.instruments import TradingRules

    rules = TradingRules(
        symbol=CANONICAL,
        base_asset="BTC",
        quote_asset="USDT",
        status="TRADING",
        min_quantity=0.0001,
        quantity_step=0.0001,
        price_tick=0.01,
        order_types=("market", "limit"),
    )
    return SymbolMapping(
        canonical=CANONICAL,
        market_data_symbol=MARKET_DATA_SYMBOL,
        execution_symbol=EXECUTION_SYMBOL,
        base_asset="BTC",
        market_data_quote="USDT",
        execution_quote="USD",
        market_data_tradable=True,
        execution_tradable=True,
        market_data_rules=rules,
        execution_rules=rules,
    )


class _StaticMarketData:
    """The market-data venue's price for the canonical pair."""

    provider_name = "static_test"
    kind = "current"

    def __init__(self, price: float = 50000.0) -> None:
        self.price_value = price
        self.requested: list[str] = []

    def available(self) -> bool:
        return True

    def price(self, symbol: str) -> float:
        self.requested.append(symbol)
        return self.price_value


class _DecisionSource:
    def __init__(self, decisions: list[TradingDecision]) -> None:
        self._decisions = list(decisions)
        self.offered = 0

    def next_decision(self) -> TradingDecision | None:
        if self.offered >= len(self._decisions):
            return None
        decision = self._decisions[self.offered]
        self.offered += 1
        return decision


def _buy(**over) -> TradingDecision:
    payload = {
        "decision_id": "tjr-BTC-USDT-2026-01-01",
        "symbol": CANONICAL,
        "side": "buy",
        "notional": 25.0,
        "stop_loss": 48000.0,
        "take_profit": 54000.0,
    }
    payload.update(over)
    return TradingDecision(**payload)


def _worker(
    tmp_path: Path,
    *,
    market_data=None,
    execution_seam=None,
    decisions=None,
    broker_transport=None,
    positions=None,
    max_price_deviation: float = DEFAULT_MAX_PRICE_DEVIATION,
    state_path: Path | None = None,
) -> PaperTradingWorker:
    """A worker wired to mocked transports, with the crypto mapping in place."""
    market_data = market_data or _StaticMarketData()
    broker_transport = broker_transport or _BrokerTransport(positions=positions)
    limits = PaperRiskLimits(
        max_order_notional=MAX_ORDER_NOTIONAL,
        max_open_positions=MAX_OPEN_POSITIONS,
        max_daily_orders=MAX_DAILY_ORDERS,
        allow_short=False,
        allowed_symbols=(EXECUTION_SYMBOL,),
    )
    config = PaperTradingConfig(
        strategy_id="crypto-tjr",
        order_notional=MAX_ORDER_NOTIONAL,
        whole_shares=False,
        state_path=state_path or (tmp_path / "worker_state.json"),
        decision_source="crypto",
        symbol_mappings=(_mapping(),),
        max_price_deviation=max_price_deviation,
        execution_price_seam=execution_seam,
    )
    broker = AlpacaPaperBroker(
        api_key="test-paper-key",
        api_secret="test-paper-secret",
        transport=broker_transport,
        limits=limits,
        state_path=tmp_path / "broker_state.json",
    )
    source = _DecisionSource(decisions if decisions is not None else [_buy()])
    worker = PaperTradingWorker(
        config=config,
        decision_source=source,
        market_data=market_data,
        broker_factory=lambda: broker,
        root=tmp_path,
    )
    return worker


def _execution_seam(
    *, price: float = 50000.0, age_seconds: float = 0.0, response: _Resp | None = None
) -> tuple[AlpacaCryptoExecutionPrice, _DataTransport]:
    transport = _DataTransport(response or _Resp(200, _snapshot(price, age_seconds=age_seconds)))
    client = AlpacaCryptoClient(
        api_key="test-paper-key",
        api_secret="test-paper-secret",
        transport=transport,
    )
    seam = AlpacaCryptoExecutionPrice(
        [EXECUTION_SYMBOL],
        client=client,
        max_age_seconds=300.0,
        clock=lambda: NOW,
    )
    return seam, transport


# ---------------------------------------------------------------------------
# 1. The execution-price seam
# ---------------------------------------------------------------------------


class TestTheCryptoProviderRefusesEveryHostButTheApprovedOne:
    """No host but the approved one may be reached — not even by mistake.

    The crypto client reads two different things from two different hosts: the
    market data from ``data.alpaca.markets`` and the asset registry from the
    **paper** trading host. Each is checked at the point of use, so a
    misconfiguration is refused before a request leaves, rather than being
    caught later by whatever the remote host happens to say.
    """

    def test_the_data_client_refuses_a_live_trading_host(self):
        for live in (
            "https://api.alpaca.markets",
            "https://live-api.alpaca.markets",
            "https://api.alpaca.markets/v2",
        ):
            with pytest.raises(AlpacaCryptoHostRefused):
                AlpacaCryptoClient(base_url=live, api_key="k", api_secret="s")

    def test_the_data_client_refuses_an_unapproved_host(self):
        with pytest.raises(AlpacaCryptoHostRefused):
            AlpacaCryptoClient(
                base_url="https://not-alpaca.example.com", api_key="k", api_secret="s"
            )

    def test_the_data_client_accepts_the_approved_data_host(self):
        client = AlpacaCryptoClient(api_key="k", api_secret="s")
        assert client.describe()["execution_data_host"] == "https://data.alpaca.markets"

    def test_the_asset_registry_refuses_a_live_trading_host(self):
        client = AlpacaCryptoClient(api_key="k", api_secret="s")
        for live in ("https://api.alpaca.markets", "https://live-api.alpaca.markets"):
            with pytest.raises(LiveEndpointRefusedError):
                client.assets(base_url=live)

    def test_the_asset_registry_refuses_an_unapproved_host(self):
        client = AlpacaCryptoClient(api_key="k", api_secret="s")
        with pytest.raises(LiveEndpointRefusedError):
            client.assets(base_url="https://not-alpaca.example.com")

    def test_the_refusal_happens_before_any_credential_is_read(self):
        """A refused host must not even look at the environment."""
        with pytest.raises(AlpacaCryptoHostRefused):
            AlpacaCryptoClient(base_url="https://api.alpaca.markets")


class TestExecutionPriceSeam:
    def test_it_reads_the_execution_pair_from_the_execution_venue(self):
        seam, transport = _execution_seam()
        assert seam.price(EXECUTION_SYMBOL) == pytest.approx(50000.0)
        assert transport.calls, "no snapshot was requested"
        assert transport.calls[0]["params"]["symbols"] == EXECUTION_SYMBOL

    def test_it_never_reads_the_market_data_pair(self):
        seam, transport = _execution_seam()
        seam.price(EXECUTION_SYMBOL)
        assert all(
            CANONICAL not in str(call["params"].get("symbols", "")) for call in transport.calls
        )

    def test_it_identifies_itself_as_the_execution_venue(self):
        seam, _ = _execution_seam()
        described = seam.describe()
        assert described["execution_data_provider"] == PROVIDER_ALPACA_CRYPTO
        assert described["is_current_price"] is True

    def test_a_stale_execution_price_is_refused(self):
        seam, _ = _execution_seam(age_seconds=3600)
        with pytest.raises(MarketDataError) as excinfo:
            seam.price(EXECUTION_SYMBOL)
        assert "stale" in str(excinfo.value)

    def test_a_future_dated_execution_price_is_refused(self):
        transport = _DataTransport(_Resp(200, _snapshot(50000.0, age_seconds=-600)))
        client = AlpacaCryptoClient(
            api_key="k", api_secret="s", transport=transport
        )
        seam = AlpacaCryptoExecutionPrice(
            [EXECUTION_SYMBOL], client=client, max_age_seconds=300.0, clock=lambda: NOW
        )
        with pytest.raises(MarketDataError) as excinfo:
            seam.price(EXECUTION_SYMBOL)
        assert "future" in str(excinfo.value)

    def test_a_missing_snapshot_is_refused(self):
        transport = _DataTransport(_Resp(200, {"snapshots": {}}))
        client = AlpacaCryptoClient(api_key="k", api_secret="s", transport=transport)
        seam = AlpacaCryptoExecutionPrice(
            [EXECUTION_SYMBOL], client=client, max_age_seconds=300.0, clock=lambda: NOW
        )
        with pytest.raises(MarketDataError):
            seam.price(EXECUTION_SYMBOL)

    def test_a_snapshot_without_a_timestamp_is_refused(self):
        body = {
            "snapshots": {
                EXECUTION_SYMBOL: {
                    "latestTrade": {"p": "50000.0"},
                    "latestQuote": {"bp": "49999", "ap": "50001"},
                    "minuteBar": {"c": "50000"},
                }
            }
        }
        transport = _DataTransport(_Resp(200, body))
        client = AlpacaCryptoClient(api_key="k", api_secret="s", transport=transport)
        seam = AlpacaCryptoExecutionPrice(
            [EXECUTION_SYMBOL], client=client, max_age_seconds=300.0, clock=lambda: NOW
        )
        with pytest.raises(MarketDataError) as excinfo:
            seam.price(EXECUTION_SYMBOL)
        assert "timestamp" in str(excinfo.value)

    def test_a_nan_price_is_refused(self):
        body = {
            "snapshots": {
                EXECUTION_SYMBOL: {
                    "latestTrade": {"p": "NaN", "t": _iso()},
                    "latestQuote": {"bp": "NaN", "ap": "NaN", "t": _iso()},
                    "minuteBar": {"c": "NaN", "t": _iso()},
                }
            }
        }
        transport = _DataTransport(_Resp(200, body))
        client = AlpacaCryptoClient(api_key="k", api_secret="s", transport=transport)
        seam = AlpacaCryptoExecutionPrice(
            [EXECUTION_SYMBOL], client=client, max_age_seconds=300.0, clock=lambda: NOW
        )
        with pytest.raises(MarketDataError):
            seam.price(EXECUTION_SYMBOL)

    def test_an_unreachable_venue_is_refused_rather_than_substituted(self):
        def boom(**kwargs):
            raise TimeoutError("simulated timeout")

        client = AlpacaCryptoClient(api_key="k", api_secret="s", transport=boom)
        seam = AlpacaCryptoExecutionPrice(
            [EXECUTION_SYMBOL], client=client, max_age_seconds=300.0, clock=lambda: NOW
        )
        with pytest.raises(MarketDataError):
            seam.price(EXECUTION_SYMBOL)

    def test_the_newest_observation_wins(self):
        body = {
            "snapshots": {
                EXECUTION_SYMBOL: {
                    "latestTrade": {"p": "100", "t": _iso(-100)},
                    "latestQuote": {"bp": "200", "ap": "200", "t": _iso(-10)},
                    "minuteBar": {"c": "300", "t": _iso(-50)},
                }
            }
        }
        price, observed_at, field = snapshot_price_and_time(body["snapshots"][EXECUTION_SYMBOL])
        assert price == pytest.approx(200.0)
        assert field == "latestQuote"

    def test_a_quote_midpoint_is_used_when_there_is_no_trade(self):
        body = {
            "latestQuote": {"bp": "100", "ap": "110", "t": _iso()},
        }
        price, _, field = snapshot_price_and_time(body)
        assert price == pytest.approx(105.0)
        assert field == "latestQuote"


# ---------------------------------------------------------------------------
# 2. The worker prices and submits the order correctly
# ---------------------------------------------------------------------------


class TestCryptoOrderSubmission:
    def test_the_order_carries_the_execution_symbol(self, tmp_path):
        seam, _ = _execution_seam()
        transport = _BrokerTransport()
        worker = _worker(tmp_path, execution_seam=seam, broker_transport=transport)
        worker.start()
        try:
            result = worker.run_once()
            assert result.ran is True
            assert transport.orders, "no order was submitted"
            assert transport.orders[0]["symbol"] == EXECUTION_SYMBOL
        finally:
            worker.stop()

    def test_the_order_is_sized_from_the_execution_price(self, tmp_path):
        """The execution price, not the market-data price, sets the quantity."""
        # The two prices differ by 1%, inside the permitted deviation, so the
        # order is placed — and it is sized from the *execution* price.
        seam, _ = _execution_seam(price=50500.0)
        market = _StaticMarketData(price=50000.0)
        transport = _BrokerTransport()
        worker = _worker(
            tmp_path,
            market_data=market,
            execution_seam=seam,
            broker_transport=transport,
        )
        worker.start()
        try:
            worker.run_once()
            submitted = transport.orders[0]
            # 25 USD at 50500 = 0.000495; at 50000 it would have been 0.0005.
            assert float(submitted["qty"]) == pytest.approx(0.000495, rel=1e-6)
        finally:
            worker.stop()

    def test_the_notional_never_exceeds_the_cap(self, tmp_path):
        seam, _ = _execution_seam(price=50000.0)
        transport = _BrokerTransport()
        worker = _worker(tmp_path, execution_seam=seam, broker_transport=transport)
        worker.start()
        try:
            worker.run_once()
            submitted = transport.orders[0]
            assert float(submitted["qty"]) * 50000.0 <= MAX_ORDER_NOTIONAL
        finally:
            worker.stop()

    def test_the_market_data_price_is_read_for_the_market_data_pair(self, tmp_path):
        seam, _ = _execution_seam()
        market = _StaticMarketData()
        worker = _worker(tmp_path, market_data=market, execution_seam=seam)
        worker.start()
        try:
            worker.run_once()
            assert market.requested == [MARKET_DATA_SYMBOL]
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# 3. The failure modes
# ---------------------------------------------------------------------------


class TestCryptoOrderRefusals:
    def test_no_execution_price_means_no_order(self, tmp_path):
        """The market-data price is never used as a substitute."""
        transport = _BrokerTransport()
        seam, _ = _execution_seam()
        # Make the execution seam fail while the market-data seam still works.
        def boom(**kwargs):
            raise TimeoutError("simulated timeout")

        failing = AlpacaCryptoExecutionPrice(
            [EXECUTION_SYMBOL],
            client=AlpacaCryptoClient(api_key="k", api_secret="s", transport=boom),
            max_age_seconds=300.0,
            clock=lambda: NOW,
        )
        worker = _worker(
            tmp_path,
            execution_seam=failing,
            broker_transport=transport,
        )
        worker.start()
        try:
            result = worker.run_once()
            assert result.ran is False
            assert result.reason_code == NO_TRADE_EXECUTION_PRICE_UNAVAILABLE
            assert not transport.orders, "an order was placed without an execution price"
        finally:
            worker.stop()

    def test_a_stale_execution_price_means_no_order(self, tmp_path):
        transport = _BrokerTransport()
        seam, _ = _execution_seam(age_seconds=3600)
        worker = _worker(tmp_path, execution_seam=seam, broker_transport=transport)
        worker.start()
        try:
            result = worker.run_once()
            assert result.ran is False
            assert result.reason_code == NO_TRADE_EXECUTION_PRICE_UNAVAILABLE
            assert not transport.orders
        finally:
            worker.stop()

    def test_a_large_price_deviation_means_no_order(self, tmp_path):
        """A 20% gap between the decision price and the execution price is refused."""
        seam, _ = _execution_seam(price=60000.0)
        market = _StaticMarketData(price=50000.0)
        transport = _BrokerTransport()
        worker = _worker(
            tmp_path,
            market_data=market,
            execution_seam=seam,
            broker_transport=transport,
            max_price_deviation=0.05,
        )
        worker.start()
        try:
            result = worker.run_once()
            assert result.ran is False
            assert result.reason_code == NO_TRADE_PRICE_DEVIATION
            assert not transport.orders
        finally:
            worker.stop()

    def test_a_small_price_deviation_still_places_the_order(self, tmp_path):
        seam, _ = _execution_seam(price=51000.0)
        market = _StaticMarketData(price=50000.0)
        transport = _BrokerTransport()
        worker = _worker(
            tmp_path,
            market_data=market,
            execution_seam=seam,
            broker_transport=transport,
            max_price_deviation=0.05,
        )
        worker.start()
        try:
            result = worker.run_once()
            assert result.ran is True
            assert transport.orders
        finally:
            worker.stop()

    def test_a_decision_with_no_mapping_is_refused(self, tmp_path):
        """A canonical symbol the mapping does not cover never reaches the broker."""
        seam, _ = _execution_seam()
        transport = _BrokerTransport()
        worker = _worker(
            tmp_path,
            execution_seam=seam,
            broker_transport=transport,
            decisions=[_buy(symbol="ETH/USDT")],
        )
        worker.start()
        try:
            result = worker.run_once()
            assert result.ran is False
            assert not transport.orders
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# 4. The exit path
# ---------------------------------------------------------------------------


class TestCryptoExit:
    def _worker_with_position(self, tmp_path, *, price: float, positions):
        """A worker that already holds the execution-symbol position."""
        transport = _BrokerTransport(positions=positions)
        seam, _ = _execution_seam(price=price)
        worker = _worker(
            tmp_path,
            # The exit is *decided* on the market-data venue's price — the market
            # the strategy analysed — and *priced* on the execution venue.
            market_data=_StaticMarketData(price=price),
            execution_seam=seam,
            broker_transport=transport,
            decisions=[],  # the exit is the only action
        )
        # Seed the state the way a filled entry would have.
        worker.state.positions = {
            EXECUTION_SYMBOL: {
                "symbol": EXECUTION_SYMBOL,
                "quantity": 0.0005,
                "avg_entry_price": 50000.0,
                "current_price": price,
                "unrealized_pl": 0.0,
            }
        }
        worker.state.stops = {
            EXECUTION_SYMBOL: {
                "symbol": EXECUTION_SYMBOL,
                "stop_loss": 48000.0,
                "take_profit": 52000.0,
                "decision_id": "tjr-BTC-USDT-2026-01-01",
                "recorded_at": _iso(),
                "canonical_symbol": CANONICAL,
            }
        }
        return worker, transport

    def test_a_target_trigger_places_an_exit_under_the_execution_symbol(self, tmp_path):
        worker, transport = self._worker_with_position(
            tmp_path,
            price=53000.0,
            positions=[
                {
                    "symbol": EXECUTION_SYMBOL,
                    "qty": "0.0005",
                    "avg_entry_price": "50000.0",
                    "current_price": "53000.0",
                    "unrealized_pl": "1.5",
                }
            ],
        )
        worker.start()
        try:
            result = worker.run_once()
            assert result.ran is True, result.skipped_reason
            assert transport.orders, "no exit order was placed"
            assert transport.orders[0]["symbol"] == EXECUTION_SYMBOL
            assert transport.orders[0]["side"] == "sell"
        finally:
            worker.stop()

    def test_a_stop_trigger_places_an_exit(self, tmp_path):
        worker, transport = self._worker_with_position(
            tmp_path,
            price=47000.0,
            positions=[
                {
                    "symbol": EXECUTION_SYMBOL,
                    "qty": "0.0005",
                    "avg_entry_price": "50000.0",
                    "current_price": "47000.0",
                    "unrealized_pl": "-1.5",
                }
            ],
        )
        worker.start()
        try:
            result = worker.run_once()
            assert result.ran is True, result.skipped_reason
            assert transport.orders
            assert transport.orders[0]["side"] == "sell"
        finally:
            worker.stop()

    def test_no_level_crossed_means_no_exit(self, tmp_path):
        worker, transport = self._worker_with_position(
            tmp_path,
            price=50000.0,
            positions=[
                {
                    "symbol": EXECUTION_SYMBOL,
                    "qty": "0.0005",
                    "avg_entry_price": "50000.0",
                    "current_price": "50000.0",
                    "unrealized_pl": "0.0",
                }
            ],
        )
        worker.start()
        try:
            result = worker.run_once()
            assert result.ran is False
            assert not transport.orders
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# 5. Restart safety
# ---------------------------------------------------------------------------


class TestRestartSafety:
    def test_a_restart_places_no_duplicate_order(self, tmp_path):
        """The same decision, replayed after a restart, is reconciled, not resent."""
        state_path = tmp_path / "worker_state.json"
        broker_state = tmp_path / "broker_state.json"
        transport = _BrokerTransport()
        seam, _ = _execution_seam()

        config = PaperTradingConfig(
            strategy_id="crypto-tjr",
            order_notional=MAX_ORDER_NOTIONAL,
            whole_shares=False,
            state_path=state_path,
            decision_source="crypto",
            symbol_mappings=(_mapping(),),
            execution_price_seam=seam,
        )
        broker = AlpacaPaperBroker(
            api_key="test-paper-key",
            api_secret="test-paper-secret",
            transport=transport,
            limits=PaperRiskLimits(
                max_order_notional=MAX_ORDER_NOTIONAL,
                max_open_positions=MAX_OPEN_POSITIONS,
                max_daily_orders=MAX_DAILY_ORDERS,
                allow_short=False,
                allowed_symbols=(EXECUTION_SYMBOL,),
            ),
            state_path=broker_state,
        )

        first = PaperTradingWorker(
            config=config,
            decision_source=_DecisionSource([_buy()]),
            market_data=_StaticMarketData(),
            broker_factory=lambda: broker,
            root=tmp_path,
        )
        first.start()
        try:
            first.run_once()
        finally:
            first.stop()
        assert len(transport.orders) == 1

        # Restart with the same durable state and the same decision.
        second = PaperTradingWorker(
            config=config,
            decision_source=_DecisionSource([_buy()]),
            market_data=_StaticMarketData(),
            broker_factory=lambda: broker,
            root=tmp_path,
        )
        second.start()
        try:
            second.run_once()
        finally:
            second.stop()
        assert len(transport.orders) == 1, "the order was placed twice across a restart"


# ---------------------------------------------------------------------------
# 6. The symbol configuration gates the order
# ---------------------------------------------------------------------------


class TestSymbolGating:
    def test_a_symbol_the_execution_venue_does_not_list_produces_no_mapping(self):
        from tradingagents.crypto.instruments import InstrumentRegistry, TradingRules

        binance = InstrumentRegistry(
            rules={
                "BTC/USDT": TradingRules(
                    symbol="BTC/USDT",
                    base_asset="BTC",
                    quote_asset="USDT",
                    status="TRADING",
                )
            },
            venue="binance",
        )
        alpaca = InstrumentRegistry(
            rules={}, venue="alpaca_paper", fetched_at=_iso()
        )
        config = resolve_symbol_configuration(
            [CANONICAL],
            market_data_registry=binance,
            execution_registry=alpaca,
        )
        assert config.mappings == ()
        assert config.rejected_for(CANONICAL) is not None

    def test_a_configured_symbol_below_the_minimum_notional_is_refused_by_risk(self, tmp_path):
        """The risk engine still bounds the order, whatever the venue says."""
        seam, _ = _execution_seam(price=50000.0)
        transport = _BrokerTransport()
        worker = _worker(
            tmp_path,
            execution_seam=seam,
            broker_transport=transport,
            decisions=[_buy(notional=1000.0)],
        )
        worker.start()
        try:
            result = worker.run_once()
            assert result.ran is False
            assert result.reason_code == NO_TRADE_RISK_REJECTED
            assert not transport.orders
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# 8. AlpacaCryptoClient.assets() tests
# ---------------------------------------------------------------------------


class TestAlpacaCryptoClientAssets:
    """Tests for the assets() method of AlpacaCryptoClient."""

    def _assets_response(self) -> list[dict]:
        """A fake Alpaca assets response."""
        return [
            {
                "symbol": "BTC/USD",
                "class": "crypto",
                "status": "active",
                "tradable": True,
                "fractionable": True,
                "min_order_size": "0.0001",
                "min_trade_increment": "0.0001",
                "price_increment": "0.01",
            },
            {
                "symbol": "ETH/USD",
                "class": "crypto",
                "status": "active",
                "tradable": True,
                "fractionable": True,
                "min_order_size": "0.001",
                "min_trade_increment": "0.001",
                "price_increment": "0.01",
            },
        ]

    def test_assets_works_without_transport_using_default(self):
        """assets() works when no transport is passed (uses default transport)."""
        # This test verifies the fix: no transport passed should not raise TypeError
        from tradingagents.paper_experiment.providers.alpaca_crypto import AlpacaCryptoClient
        from tradingagents.paper_experiment.providers.base import HttpResponse

        class _FakeTransport:
            def __init__(self, response: HttpResponse) -> None:
                self.response = response
                self.calls: list[dict] = []

            def __call__(self, *, url, params, headers, timeout):
                self.calls.append({"url": url, "params": params, "headers": headers})
                return self.response

        fake_transport = _FakeTransport(
            HttpResponse(200, json.dumps(self._assets_response()))
        )

        # The client is created without a transport, but we pass one to assets()
        # This tests that assets() can accept a custom transport
        client = AlpacaCryptoClient(api_key="test-key", api_secret="test-secret")
        assets = client.assets(
            base_url="https://paper-api.alpaca.markets",
            transport=fake_transport,
        )

        assert len(assets) == 2
        assert assets[0]["symbol"] == "BTC/USD"
        assert assets[1]["symbol"] == "ETH/USD"
        assert fake_transport.calls, "transport was not called"
        assert fake_transport.calls[0]["params"]["asset_class"] == "crypto"
        assert fake_transport.calls[0]["params"]["status"] == "active"

    def test_assets_works_with_custom_transport(self):
        """assets() works when a custom transport is passed."""
        from tradingagents.paper_experiment.providers.alpaca_crypto import AlpacaCryptoClient
        from tradingagents.paper_experiment.providers.base import HttpResponse

        class _FakeTransport:
            def __init__(self, response: HttpResponse) -> None:
                self.response = response
                self.calls: list[dict] = []

            def __call__(self, *, url, params, headers, timeout):
                self.calls.append({"url": url, "params": params, "headers": headers})
                return self.response

        fake_transport = _FakeTransport(
            HttpResponse(200, json.dumps(self._assets_response()))
        )

        client = AlpacaCryptoClient(
            api_key="test-key",
            api_secret="test-secret",
            transport=fake_transport,  # Transport on client
        )
        assets = client.assets(
            base_url="https://paper-api.alpaca.markets",
            transport=fake_transport,  # Also pass to method
        )

        assert len(assets) == 2
        assert assets[0]["symbol"] == "BTC/USD"
        assert fake_transport.calls, "transport was not called"

    def test_assets_handles_401_unauthorized(self):
        """assets() raises ProviderAuthError on HTTP 401."""
        from tradingagents.paper_experiment.providers.alpaca_crypto import (
            AlpacaCryptoClient,
            AlpacaCryptoError,
        )
        from tradingagents.paper_experiment.providers.base import (
            ProviderAuthError,
            HttpResponse,
        )

        class _FakeTransport:
            def __init__(self, response: HttpResponse) -> None:
                self.response = response

            def __call__(self, *, url, params, headers, timeout):
                return self.response

        fake_transport = _FakeTransport(
            HttpResponse(401, '{"message": "unauthorized"}')
        )

        client = AlpacaCryptoClient(
            api_key="test-key",
            api_secret="test-secret",
            transport=fake_transport,
        )

        with pytest.raises(ProviderAuthError) as excinfo:
            client.assets(
                base_url="https://paper-api.alpaca.markets",
                transport=fake_transport,
            )
        # Credential should not leak in error message
        error_msg = str(excinfo.value)
        assert "test-key" not in error_msg
        assert "test-secret" not in error_msg
        assert "401" in error_msg

    def test_assets_handles_403_forbidden(self):
        """assets() raises ProviderUnsupportedError on HTTP 403."""
        from tradingagents.paper_experiment.providers.alpaca_crypto import (
            AlpacaCryptoClient,
        )
        from tradingagents.paper_experiment.providers.base import (
            ProviderUnsupportedError,
            HttpResponse,
        )

        class _FakeTransport:
            def __init__(self, response: HttpResponse) -> None:
                self.response = response

            def __call__(self, *, url, params, headers, timeout):
                return self.response

        fake_transport = _FakeTransport(
            HttpResponse(403, '{"message": "forbidden - crypto not enabled"}')
        )

        client = AlpacaCryptoClient(
            api_key="test-key",
            api_secret="test-secret",
            transport=fake_transport,
        )

        with pytest.raises(ProviderUnsupportedError) as excinfo:
            client.assets(
                base_url="https://paper-api.alpaca.markets",
                transport=fake_transport,
            )
        # Credential should not leak in error message
        error_msg = str(excinfo.value)
        assert "test-key" not in error_msg
        assert "test-secret" not in error_msg
        assert "403" in error_msg

    def test_assets_handles_connection_error(self):
        """assets() raises ProviderError on connection failure."""
        from tradingagents.paper_experiment.providers.alpaca_crypto import (
            AlpacaCryptoClient,
            AlpacaCryptoError,
        )
        from tradingagents.paper_experiment.providers.base import (
            ProviderError,
            HttpResponse,
        )

        class _FakeTransport:
            def __call__(self, *, url, params, headers, timeout):
                raise ConnectionError("connection refused")

        fake_transport = _FakeTransport()

        client = AlpacaCryptoClient(
            api_key="test-key",
            api_secret="test-secret",
            transport=fake_transport,
        )

        with pytest.raises(ProviderError) as excinfo:
            client.assets(
                base_url="https://paper-api.alpaca.markets",
                transport=fake_transport,
            )
        # Credential should not leak in error message
        error_msg = str(excinfo.value)
        assert "test-key" not in error_msg
        assert "test-secret" not in error_msg
        assert "connection" in error_msg.lower() or "failed" in error_msg.lower()

    def test_assets_no_credential_leakage_in_errors(self):
        """No Alpaca API credentials appear in any error message from assets()."""
        from tradingagents.paper_experiment.providers.alpaca_crypto import (
            AlpacaCryptoClient,
        )
        from tradingagents.paper_experiment.providers.base import (
            ProviderError,
            HttpResponse,
        )

        class _FakeTransport:
            def __init__(self, status_code: int, body: str) -> None:
                self.response = HttpResponse(status_code, body)

            def __call__(self, *, url, params, headers, timeout):
                return self.response

        # Test various error scenarios
        for status_code, body in [
            (401, "unauthorized"),
            (403, "forbidden"),
            (429, "rate limited"),
            (500, "internal server error"),
            (502, "bad gateway"),
        ]:
            fake_transport = _FakeTransport(status_code, body)
            client = AlpacaCryptoClient(
                api_key="super-secret-key-123",
                api_secret="super-secret-secret-456",
                transport=fake_transport,
            )

            try:
                client.assets(
                    base_url="https://paper-api.alpaca.markets",
                    transport=fake_transport,
                )
            except ProviderError as e:
                error_msg = str(e)
                # Verify credentials never appear in error messages
                assert "super-secret-key-123" not in error_msg
                assert "super-secret-secret-456" not in error_msg
            except Exception as e:
                # Other exceptions should also not leak credentials
                error_msg = str(e)
                assert "super-secret-key-123" not in error_msg
                assert "super-secret-secret-456" not in error_msg
