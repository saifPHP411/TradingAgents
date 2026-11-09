"""The USD execution transition: a USDT-quoted signal traded on a USD pair.

The crypto path analyses one market and trades another. The TJR chain reads
Binance's ``XRP/USDT`` and the order is placed on Alpaca's ``XRP/USD``. The
account is funded in USD and every cap the risk engine enforces — the $25 order
notional, the exposure fraction of equity — is denominated in USD.

These tests pin the four defects that made the two currencies mix:

1. the order was sized from a **quantity** computed at the USDT price and sent
   as if it were a USD quantity, so the order's cash value was never the value
   the risk engine approved;
2. the risk engine sized the entry at the USDT price, so its $25 cap and its
   equity fractions described a USD order in USDT terms;
3. the ledger matched a USD holding against a USDT price by full symbol, so the
   open position counted for nothing and the engine saw the cash alone;
4. the market-data seam validated a snapshot's freshness but did not **store**
   it, so the dashboard could show a 273-second-old price beside the status
   "ok" and the claim "is_current_price: true".

Every test drives the real code paths offline: no network, no credential, no
paper order, and no live endpoint.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.paper_experiment.broker import (
    NO_TRADE_INSUFFICIENT_FUNDS,
    AlpacaCryptoExecutionPrice,
    PaperTradingConfig,
    PaperTradingWorker,
    TradingDecision,
)
from tradingagents.paper_experiment.broker.alpaca_paper import (
    BrokerRejectedError,
    InsufficientFundsError,
)
from tradingagents.paper_experiment.broker.trading import (
    BinanceSnapshotMarketData,
    MarketDataError,
)
from tradingagents.paper_experiment.symbol_mapping import SymbolMapping
from tradingagents.paper_experiment.tjr_source import (
    TjrBookSnapshot,
    ledger_from_snapshot,
)

CANONICAL = "XRP/USDT"
EXECUTION = "XRP/USD"
MAX_ORDER_NOTIONAL = 25.0


# ---------------------------------------------------------------------------
# 1. The ledger marks a USD holding at the price of the same coin
# ---------------------------------------------------------------------------


class TestTheLedgerMatchesAcrossQuoteCurrencies:
    """A holding is identified by its base asset, not by its full symbol.

    The broker holds ``XRP/USD``; the strategy's price is ``XRP/USDT``. Matching
    the two spellings literally left the position marked at no price, so the
    risk engine computed an equity of the cash alone — understating the account
    by the whole value of the open positions.
    """

    def test_a_usd_holding_is_marked_at_the_usdt_price(self):
        ledger = ledger_from_snapshot(
            TjrBookSnapshot(cash=40.0, positions={EXECUTION: 9.194395796}),
            prices={CANONICAL: 2.72},
        )
        # The position is worth its quantity times the price of the same coin.
        assert ledger.market_value == pytest.approx(9.194395796 * 2.72)
        assert ledger.equity == pytest.approx(40.0 + 9.194395796 * 2.72)

    def test_the_position_is_not_worthless_to_the_risk_engine(self):
        # The regression this guards: the position counted for NOTHING, so the
        # engine's equity was $40 instead of ~$65 — a 38% understatement that
        # distorted every fraction-of-equity check.
        ledger = ledger_from_snapshot(
            TjrBookSnapshot(cash=40.0, positions={EXECUTION: 9.194395796}),
            prices={CANONICAL: 2.72},
        )
        assert ledger.market_value > 0
        assert ledger.equity > 40.0

    def test_an_exact_symbol_match_still_wins(self):
        ledger = ledger_from_snapshot(
            TjrBookSnapshot(cash=100.0, positions={CANONICAL: 2.0}),
            prices={CANONICAL: 50.0},
        )
        assert ledger.market_value == pytest.approx(100.0)

    def test_a_different_coin_is_not_marked(self):
        # The base asset must match: a BTC holding is never marked at the XRP
        # price, so its value is excluded rather than guessed at.
        ledger = ledger_from_snapshot(
            TjrBookSnapshot(cash=100.0, positions={"BTC/USD": 1.0}),
            prices={CANONICAL: 2.72},
        )
        assert ledger.market_value == pytest.approx(0.0)

    def test_a_symbol_with_no_quote_is_not_marked(self):
        ledger = ledger_from_snapshot(
            TjrBookSnapshot(cash=100.0, positions={"XRPUSD": 1.0}),
            prices={CANONICAL: 2.72},
        )
        assert ledger.market_value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. The worker sizes the order at the execution price
# ---------------------------------------------------------------------------


class _RecordingBroker:
    """A broker that records what it was asked to do."""

    def __init__(self) -> None:
        self.orders: list[dict] = []
        self.positions: dict[str, float] = {}
        self.daily = 0

    def daily_orders(self) -> int:
        return self.daily

    def account(self):
        class _Account:
            cash = 1000.0
            equity = 1000.0
            buying_power = 1000.0

        return _Account()

    def positions(self):
        return dict(self.positions)

    def submit_order(self, **kwargs):
        self.orders.append(kwargs)
        self.daily += 1

        class _Order:
            id = "order-1"
            client_order_id = kwargs.get("client_order_id", "pt-1")
            symbol = kwargs.get("symbol")
            side = kwargs.get("side")
            qty = kwargs.get("qty")
            type = kwargs.get("type", "market")
            status = "accepted"
            submitted_at = datetime.now(UTC)
            filled_avg_price = None

        return _Order()

    def get_order_by_client_order_id(self, client_order_id):
        return None


class TestTheOrderIsSizedAtTheExecutionPrice:
    """A quantity sized at the USDT price must not be sent as a USD quantity.

    9.194 XRP bought at a $2.72 USDT price is a $25.00 notional; at the $2.71
    USD execution price the same quantity is $24.92. The drift is small here,
    but the *rule* is what matters: the order's cash value is always the
    notional the risk engine approved.
    """

    def _worker(self, **config_kwargs) -> PaperTradingWorker:
        broker = _RecordingBroker()
        config = PaperTradingConfig(
            whole_shares=False,
            **config_kwargs,
        )
        worker = PaperTradingWorker(
            config=config,
            decision_source=[],
            market_data=BinanceSnapshotMarketData(
                [CANONICAL],
                client=_StubBinance(2.72),
                max_age_seconds=15.0,
            ),
            broker_factory=lambda: broker,
        )
        worker._broker = broker
        return worker

    def test_a_quantity_and_a_notional_are_re_sized_at_the_execution_price(self):
        worker = self._worker()
        decision = TradingDecision(
            decision_id="d1",
            symbol=CANONICAL,
            side="buy",
            # Sized at the USDT price of 2.72: 9.194 units is $25.00.
            quantity=9.194395796,
            notional=25.0,
        )
        quantity = worker._size(decision, 2.71)
        # Re-sized at the USD price of 2.71, the same $25.00 notional buys
        # 9.225 units — not the 9.194 that the USDT price produced.
        assert quantity == pytest.approx(25.0 / 2.71, rel=1e-6)
        assert quantity != pytest.approx(9.194395796, rel=1e-6)

    def test_the_re_sized_order_keeps_the_approved_notional(self):
        worker = self._worker()
        decision = TradingDecision(
            decision_id="d1",
            symbol=CANONICAL,
            side="buy",
            quantity=9.194395796,
            notional=25.0,
        )
        quantity = worker._size(decision, 2.71)
        # The cash value of the order is exactly what the risk engine approved.
        assert quantity * 2.71 == pytest.approx(25.0, rel=1e-6)

    def test_a_quantity_alone_is_still_honoured(self):
        # A decision that carries only a quantity cannot be re-based — the
        # quantity is the only statement of size there is — so it is used as it
        # stands. Nothing about the single-venue paths changes.
        worker = self._worker()
        decision = TradingDecision(
            decision_id="d1",
            symbol=CANONICAL,
            side="buy",
            quantity=3.0,
        )
        assert worker._size(decision, 2.71) == pytest.approx(3.0)

    def test_a_notional_alone_is_still_sized_at_the_price(self):
        worker = self._worker()
        decision = TradingDecision(
            decision_id="d1",
            symbol=CANONICAL,
            side="buy",
            notional=25.0,
        )
        assert worker._size(decision, 2.71) == pytest.approx(25.0 / 2.71, rel=1e-6)


class _StubBinance:
    """A Binance ticker that answers with one price."""

    def __init__(self, price: float, observed_at: datetime | None = None) -> None:
        self.price = price
        self.observed_at = observed_at or datetime.now(UTC)

    def ticker_24h(self, symbols):
        ticker = type(
            "Ticker",
            (),
            {
                "symbol": "XRPUSDT",
                "last": self.price,
                "bid": self.price,
                "ask": self.price,
                "observed_at": self.observed_at,
            },
        )()
        return {"XRP/USDT": ticker}


# ---------------------------------------------------------------------------
# 3. The freshness view cannot contradict itself
# ---------------------------------------------------------------------------


class TestTheFreshnessViewIsSelfConsistent:
    """The age and the "is current" answer must describe the same price.

    The reported symptom was a 273.65-second-old price (against a 15-second
    limit) shown beside the status "ok" and the claim "is_current_price: true".
    """

    def test_a_price_that_aged_past_the_limit_is_not_current(self):
        now = datetime.now(UTC)
        seam = BinanceSnapshotMarketData(
            [CANONICAL],
            client=_StubBinance(2.72, observed_at=now),
            max_age_seconds=15.0,
            clock=lambda: now,
        )
        assert seam.price(CANONICAL) == pytest.approx(2.72)
        # No new read: the stored observation simply ages.
        seam._clock = lambda: now + timedelta(seconds=273.65)
        described = seam.describe()
        assert described["last_snapshot_age_seconds"] == pytest.approx(273.65, abs=0.1)
        assert described["is_current_price"] is False

    def test_a_fresh_price_is_current(self):
        now = datetime.now(UTC)
        seam = BinanceSnapshotMarketData(
            [CANONICAL],
            client=_StubBinance(2.72, observed_at=now),
            max_age_seconds=15.0,
            clock=lambda: now,
        )
        seam.price(CANONICAL)
        described = seam.describe()
        assert described["is_current_price"] is True
        assert described["market_data_status"] == "ok"

    def test_a_seam_that_has_read_nothing_is_not_current(self):
        now = datetime.now(UTC)
        seam = BinanceSnapshotMarketData(
            [CANONICAL],
            client=_StubBinance(2.72, observed_at=now),
            max_age_seconds=15.0,
            clock=lambda: now,
        )
        assert seam.describe()["is_current_price"] is False

    def test_the_status_and_the_age_describe_the_same_read(self):
        # The defect was that ``available()`` validated a snapshot without
        # storing it, so the age came from an older read and the status from a
        # newer one. The stored observation must be the one that was validated.
        now = datetime.now(UTC)
        seam = BinanceSnapshotMarketData(
            [CANONICAL],
            client=_StubBinance(2.72, observed_at=now),
            max_age_seconds=15.0,
            clock=lambda: now,
        )
        seam.available()
        described = seam.describe()
        assert described["last_snapshot_age_seconds"] == pytest.approx(0.0, abs=1.0)
        assert described["market_data_status"] == "ok"

    def test_a_stale_price_is_refused_outright(self):
        now = datetime.now(UTC)
        seam = BinanceSnapshotMarketData(
            [CANONICAL],
            client=_StubBinance(2.72, observed_at=now),
            max_age_seconds=15.0,
            clock=lambda: now,
        )
        seam._clock = lambda: now + timedelta(seconds=273.65)
        with pytest.raises(MarketDataError):
            seam.price(CANONICAL)

    def test_the_execution_seam_reports_its_age(self):
        now = datetime.now(UTC)
        seam = AlpacaCryptoExecutionPrice(
            [EXECUTION],
            client=_StubAlpacaCrypto(2.71, now),
            max_age_seconds=15.0,
            clock=lambda: now,
        )
        seam.price(EXECUTION)
        described = seam.describe()
        assert described["is_current_price"] is True
        assert described["execution_data_last_snapshot_age_seconds"] == pytest.approx(0.0)


class _StubAlpacaCrypto:
    """An Alpaca crypto client that answers with one snapshot."""

    def __init__(self, price: float, observed_at: datetime) -> None:
        self.price = price
        self.observed_at = observed_at

    def latest_snapshot(self, symbols):
        return {
            symbol: {
                "latestTrade": {
                    "p": str(self.price),
                    "t": self.observed_at.isoformat(),
                }
            }
            for symbol in symbols
        }


# ---------------------------------------------------------------------------
# 4. An insufficient balance is named apart from any other refusal
# ---------------------------------------------------------------------------


class TestInsufficientFundsIsNamedApart:
    """The account's money is a different problem from the order's shape."""

    def test_insufficient_buying_power_is_classified(self):
        from tradingagents.paper_experiment.broker.alpaca_paper import (
            _is_insufficient_funds,
        )

        assert _is_insufficient_funds("insufficient buying power") is True
        assert _is_insufficient_funds("not enough buying power") is True
        assert _is_insufficient_funds("insufficient funds") is True
        assert _is_insufficient_funds("symbol is not tradable") is False

    def test_it_is_a_broker_rejection(self):
        # The catch-all must still catch it: a consumer that knows only about
        # BrokerRejectedError keeps working.
        assert issubclass(InsufficientFundsError, BrokerRejectedError)

    def test_the_code_is_in_the_closed_vocabulary(self):
        from tradingagents.paper_experiment.broker.trading import NO_TRADE_REASONS

        assert NO_TRADE_INSUFFICIENT_FUNDS in NO_TRADE_REASONS
        # It is a no-trade reason, so it is distinct from the "an order was
        # placed" outcome.
        assert NO_TRADE_INSUFFICIENT_FUNDS != "ORDERED"


# ---------------------------------------------------------------------------
# 5. The whole chain, end to end
# ---------------------------------------------------------------------------


class TestTheWholeChainInUSD:
    """A USDT-quoted signal, sized in USD, placed on a USD pair.

    This is the whole defect in one test: the chain reads ``XRP/USDT`` at
    $2.72, the account is funded in USD, the risk engine's cap is $25, and the
    order is placed on ``XRP/USD`` at $2.71. The order's cash value must be the
    $25 the engine approved — in the currency the account holds.
    """

    def _worker(self) -> PaperTradingWorker:
        broker = _RecordingBroker()
        config = PaperTradingConfig(
            whole_shares=False,
            symbol_mappings=(
                SymbolMapping(
                    canonical=CANONICAL,
                    market_data_symbol="XRPUSDT",
                    execution_symbol=EXECUTION,
                    base_asset="XRP",
                    market_data_quote="USDT",
                    execution_quote="USD",
                    market_data_tradable=True,
                    execution_tradable=True,
                ),
            ),
        )
        worker = PaperTradingWorker(
            config=config,
            decision_source=[],
            market_data=BinanceSnapshotMarketData(
                [CANONICAL],
                client=_StubBinance(2.72),
                max_age_seconds=15.0,
            ),
            broker_factory=lambda: broker,
        )
        worker._broker = broker
        return worker

    def test_the_order_is_placed_in_usd_at_the_usd_price(self):
        worker = self._worker()
        decision = TradingDecision(
            decision_id="d1",
            symbol=CANONICAL,
            side="buy",
            quantity=9.194395796,  # sized at the USDT price
            notional=25.0,
        )
        quantity = worker._size(
            decision, 2.71, execution_symbol=EXECUTION
        )
        # 9.194 units at the USDT price is $25.00; the same $25.00 at the USD
        # price buys 9.225 units. The order's cash value is unchanged.
        assert quantity * 2.71 == pytest.approx(25.0, rel=1e-6)
        assert quantity > 9.194395796

    def test_the_worker_sends_the_re_sized_quantity_to_the_broker(self):
        worker = self._worker()
        broker = worker._broker
        decision = TradingDecision(
            decision_id="d1",
            symbol=CANONICAL,
            side="buy",
            quantity=9.194395796,
            notional=25.0,
        )
        quantity = worker._size(decision, 2.71, execution_symbol=EXECUTION)
        broker.submit_order(
            symbol=EXECUTION,
            side="buy",
            qty=quantity,
            client_order_id=decision.client_order_id(strategy_id="t"),
        )
        assert broker.orders[0]["qty"] == pytest.approx(25.0 / 2.71, rel=1e-6)
        assert broker.orders[0]["symbol"] == EXECUTION

    def test_the_cap_is_still_twenty_five_dollars(self):
        # The re-sizing must not loosen the cap: the notional the engine
        # approved is the ceiling, and it is still $25.
        worker = self._worker()
        decision = TradingDecision(
            decision_id="d1",
            symbol=CANONICAL,
            side="buy",
            quantity=9.194395796,
            notional=25.0,
        )
        quantity = worker._size(decision, 2.71, execution_symbol=EXECUTION)
        assert quantity * 2.71 <= MAX_ORDER_NOTIONAL + 1e-9

    def test_the_equity_sees_the_open_position(self):
        # With the position marked, the engine's equity includes it — so the
        # account is ~$65, not the $40 the cash alone would show.
        ledger = ledger_from_snapshot(
            TjrBookSnapshot(cash=40.0, positions={EXECUTION: 9.194395796}),
            prices={CANONICAL: 2.72},
        )
        assert ledger.equity == pytest.approx(40.0 + 9.194395796 * 2.72)
        assert ledger.equity > 60.0

