"""A rejected symbol must be visible, with its reason — never silently missing.

The crypto path reads two venues with separate listings, so a pair can be
rejected by either one. The rules this file pins:

1. every rejection carries a stable, machine-readable code;
2. the dashboard's status reports the validation state, the accepted symbols and
   every rejection;
3. before the venues' listings have been read, the state is an explicit
   "not validated yet" — never an implied "everything is fine";
4. a venue that cannot be read is reported as a failure, not as a registry that
   happens to be empty;
5. a venue that refuses fractional quantities is rejected only when a fractional
   quantity is actually required.

Every test is offline: the registries are built from the venues' own answer
shapes.
"""

from __future__ import annotations

from tradingagents.crypto.instruments import InstrumentRegistry, TradingRules
from tradingagents.paper_experiment.crypto_symbol_config import (
    REJECTION_EXECUTION_NOT_FRACTIONABLE,
    REJECTION_EXECUTION_NOT_TRADABLE,
    REJECTION_EXECUTION_REGISTRY_UNAVAILABLE,
    REJECTION_EXECUTION_UNAVAILABLE,
    REJECTION_MARKET_DATA_NOT_TRADABLE,
    REJECTION_MARKET_DATA_UNAVAILABLE,
    resolve_symbol_configuration,
)


def _rules(
    symbol: str,
    *,
    status: str = "TRADING",
    fractionable: bool | None = True,
) -> TradingRules:
    base, _, quote = symbol.partition("/")
    return TradingRules(
        symbol=symbol,
        base_asset=base,
        quote_asset=quote,
        status=status,
        min_quantity=0.0001,
        quantity_step=0.001,
        price_tick=0.01,
        order_types=("market", "limit"),
        fractionable=fractionable,
    )


def _registry(venue: str, symbols: dict[str, str], *, fractionable=True) -> InstrumentRegistry:
    return InstrumentRegistry(
        rules={
            s: _rules(s, status=status, fractionable=fractionable)
            for s, status in symbols.items()
        },
        venue=venue,
        fetched_at="2026-01-01T00:00:00+00:00",
    )


BINANCE = _registry(
    "binance",
    {
        "BTC/USDT": "TRADING",
        "ETH/USDT": "TRADING",
        # On the market-data venue but not on the execution venue, so it
        # isolates the execution-side rejection.
        "LINK/USDT": "TRADING",
        "FAKE/USDT": "BREAK",
    },
)
ALPACA = _registry(
    "alpaca_paper",
    {
        "BTC/USD": "TRADING",
        "ETH/USD": "TRADING",
        "DOGE/USD": "TRADING",
    },
)


class TestRejectionCodes:
    def test_a_pair_the_market_data_venue_does_not_list_is_rejected(self):
        config = resolve_symbol_configuration(
            ["NOPE/USDT"], market_data_registry=BINANCE, execution_registry=ALPACA
        )
        assert config.accepted == ()
        rejection = config.rejected[0]
        assert rejection.symbol == "NOPE/USDT"
        assert rejection.code == REJECTION_MARKET_DATA_UNAVAILABLE
        assert "not listed" in rejection.market_data_reason

    def test_a_pair_the_market_data_venue_does_not_trade_is_rejected(self):
        config = resolve_symbol_configuration(
            ["FAKE/USDT"], market_data_registry=BINANCE, execution_registry=ALPACA
        )
        assert config.accepted == ()
        assert config.rejected[0].code == REJECTION_MARKET_DATA_NOT_TRADABLE

    def test_a_pair_the_execution_venue_does_not_list_is_rejected(self):
        config = resolve_symbol_configuration(
            ["LINK/USDT"], market_data_registry=BINANCE, execution_registry=ALPACA
        )
        assert config.accepted == ()
        rejection = config.rejected[0]
        assert rejection.code == REJECTION_EXECUTION_UNAVAILABLE
        assert "not listed" in rejection.execution_reason
        # The market-data venue was happy with it, so the reason names only the
        # execution venue.
        assert rejection.market_data_reason is None

    def test_when_both_venues_refuse_the_market_data_code_comes_first(self):
        """The gates run in pipeline order, and the code names the first failure.

        The prose ``reason`` still carries both venues' words, so the operator
        sees everything; the code is the single stable identifier of the first
        gate that failed.
        """
        config = resolve_symbol_configuration(
            ["NOPE/USDT"], market_data_registry=BINANCE, execution_registry=ALPACA
        )
        rejection = config.rejected[0]
        assert rejection.code == REJECTION_MARKET_DATA_UNAVAILABLE
        assert rejection.market_data_reason is not None
        assert rejection.execution_reason is not None

    def test_an_unavailable_execution_registry_rejects_everything(self):
        config = resolve_symbol_configuration(
            ["BTC/USDT"], market_data_registry=BINANCE, execution_registry=None
        )
        assert config.accepted == ()
        rejection = config.rejected[0]
        assert rejection.code == REJECTION_EXECUTION_REGISTRY_UNAVAILABLE
        assert "not available" in rejection.execution_reason

    def test_a_pair_listed_but_not_traded_is_rejected(self):
        alpaca = _registry(
            "alpaca_paper", {"BTC/USD": "TRADING", "ETH/USD": "INACTIVE"}
        )
        config = resolve_symbol_configuration(
            ["ETH/USDT"], market_data_registry=BINANCE, execution_registry=alpaca
        )
        assert config.accepted == ()
        assert config.rejected[0].code == REJECTION_EXECUTION_NOT_TRADABLE

    def test_every_rejection_carries_a_code_and_a_reason(self):
        config = resolve_symbol_configuration(
            ["NOPE/USDT", "FAKE/USDT", "LINK/USDT"],
            market_data_registry=BINANCE,
            execution_registry=ALPACA,
        )
        assert len(config.rejected) == 3
        for rejection in config.rejected:
            assert rejection.code
            assert rejection.reason


class TestFractionalRequirement:
    def test_a_non_fractionable_pair_is_rejected_when_fractional_is_required(self):
        alpaca = _registry("alpaca_paper", {"BTC/USD": "TRADING"}, fractionable=False)
        config = resolve_symbol_configuration(
            ["BTC/USDT"],
            market_data_registry=BINANCE,
            execution_registry=alpaca,
            require_fractional=True,
        )
        assert config.accepted == ()
        assert config.rejected[0].code == REJECTION_EXECUTION_NOT_FRACTIONABLE

    def test_a_non_fractionable_pair_is_accepted_when_a_whole_unit_is_enough(self):
        """The requirement decides, not an assumption about crypto."""
        alpaca = _registry("alpaca_paper", {"BTC/USD": "TRADING"}, fractionable=False)
        config = resolve_symbol_configuration(
            ["BTC/USDT"],
            market_data_registry=BINANCE,
            execution_registry=alpaca,
            require_fractional=False,
        )
        assert config.accepted == ("BTC/USDT",)

    def test_an_unstated_fractionability_is_not_a_refusal(self):
        """An unknown answer is unknown, not "not fractionable"."""
        alpaca = InstrumentRegistry(
            rules={"BTC/USD": _rules("BTC/USD", fractionable=None)},
            venue="alpaca_paper",
            fetched_at="2026-01-01T00:00:00+00:00",
        )
        config = resolve_symbol_configuration(
            ["BTC/USDT"],
            market_data_registry=BINANCE,
            execution_registry=alpaca,
            require_fractional=True,
        )
        assert config.accepted == ("BTC/USDT",)


class TestTheConfigurationSerialises:
    def test_a_rejection_is_serialised_with_its_code(self):
        config = resolve_symbol_configuration(
            ["NOPE/USDT"], market_data_registry=BINANCE, execution_registry=ALPACA
        )
        payload = config.to_dict()
        assert payload["rejected"][0]["code"] == REJECTION_MARKET_DATA_UNAVAILABLE
        assert payload["rejected"][0]["reason"]

    def test_the_accepted_symbols_are_serialised(self):
        config = resolve_symbol_configuration(
            ["BTC/USDT"], market_data_registry=BINANCE, execution_registry=ALPACA
        )
        payload = config.to_dict()
        assert payload["accepted"] == ["BTC/USDT"]
        assert payload["rejected"] == []


class TestTheDashboardReportsTheValidation:
    def test_the_status_reports_not_validated_before_the_registries_are_read(self):
        """Before the worker is built, the honest answer is "not yet"."""
        import tempfile
        from pathlib import Path

        from tradingagents.paper_experiment.dashboard.paper_control import (
            PaperTradingController,
        )

        root = Path(tempfile.mkdtemp())
        controller = PaperTradingController(root=root, enabled=True)
        status = controller.status()
        validation = status["symbol_validation"]
        assert validation["state"] == "NOT_VALIDATED"
        assert validation["accepted"] == []
        assert validation["rejected"] == []
        assert "not validated" in validation["detail"].lower() or (
            "when the paper worker starts" in validation["detail"]
        )

    def test_the_status_reports_the_validation_even_when_paper_is_disabled(self):
        import tempfile
        from pathlib import Path

        from tradingagents.paper_experiment.dashboard.paper_control import (
            PaperTradingController,
        )

        root = Path(tempfile.mkdtemp())
        controller = PaperTradingController(root=root, enabled=False)
        status = controller.status()
        assert status["status"] == "DISABLED"
        assert status["symbol_validation"]["state"] == "NOT_VALIDATED"

    def test_the_engine_view_carries_the_symbol_validation(self):
        from tradingagents.paper_experiment.dashboard.engine_view import engine_view

        view = engine_view(
            {
                "status": "STOPPED",
                "symbol_validation": {
                    "state": "VALIDATED",
                    "detail": "1 accepted, 1 rejected",
                    "validated_at": "2026-01-01T00:00:00+00:00",
                    "requested": ["BTC/USDT", "NOPE/USDT"],
                    "accepted": ["BTC/USDT"],
                    "rejected": [
                        {
                            "symbol": "NOPE/USDT",
                            "code": REJECTION_MARKET_DATA_UNAVAILABLE,
                            "reason": "not listed on market-data venue",
                        }
                    ],
                },
            }
        )
        validation = view["symbols"]["validation"]
        assert validation["state"] == "VALIDATED"
        assert validation["rejected"][0]["symbol"] == "NOPE/USDT"
        assert validation["rejected"][0]["code"] == REJECTION_MARKET_DATA_UNAVAILABLE


class TestTheDashboardDoesNotReadTheRegistriesAtStartup:
    def test_the_controller_does_not_build_a_worker_when_status_is_read(self):
        """The status path must stay read-only and network-free."""
        import tempfile
        from pathlib import Path

        from tradingagents.paper_experiment.dashboard.paper_control import (
            PaperTradingController,
        )

        root = Path(tempfile.mkdtemp())
        controller = PaperTradingController(root=root, enabled=True)
        controller.status()
        assert controller._worker is None
        assert controller.symbol_configuration is None
