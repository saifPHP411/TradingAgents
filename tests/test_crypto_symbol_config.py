"""The crypto symbol configuration: one request, two venue registries, one answer.

The crypto path reads two venues that do not share a symbol list — Binance quotes
USDT pairs, Alpaca quotes USD pairs — so a single mixed list is not a safe
configuration. These tests pin the properties that make the configuration safe:

1. a symbol is accepted only when **both** venues list it as trading;
2. a symbol only one venue serves is **rejected with that venue's reason**, never
   silently dropped and never substituted for a different pair;
3. an unavailable execution registry rejects everything (fail closed);
4. the environment variables keep their separate jobs, and a malformed symbol is a
   loud error rather than a silently skipped entry;
5. the deployment configuration carries no mixed USDT/USD list.

Every test is offline: the registries are built from the venues' own documented
answer shapes, and the deployment file is read from disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingagents.crypto.instruments import InstrumentRegistry, TradingRules
from tradingagents.paper_experiment.crypto_symbol_config import (
    CRYPTO_SYMBOLS_ENV,
    DEFAULT_EXECUTION_SYMBOLS,
    DEFAULT_MARKET_DATA_SYMBOLS,
    EXPERIMENT_SYMBOLS_ENV,
    SymbolConfigurationError,
    env_symbols,
    parse_symbol_list,
    resolve_symbol_configuration,
)


def _rules(symbol: str, *, status: str = "TRADING", step: float = 0.001) -> TradingRules:
    base, _, quote = symbol.partition("/")
    return TradingRules(
        symbol=symbol,
        base_asset=base,
        quote_asset=quote,
        status=status,
        min_quantity=0.0001,
        quantity_step=step,
        price_tick=0.01,
        order_types=("market", "limit"),
    )


def _registry(venue: str, symbols: dict[str, str]) -> InstrumentRegistry:
    """A registry whose entries carry the given status per symbol."""
    return InstrumentRegistry(
        rules={s: _rules(s, status=status) for s, status in symbols.items()},
        venue=venue,
        fetched_at="2026-01-01T00:00:00+00:00",
    )


BINANCE = _registry(
    "binance",
    {
        "BTC/USDT": "TRADING",
        "ETH/USDT": "TRADING",
        "SOL/USDT": "TRADING",
        "DOGE/USDT": "TRADING",
        "FAKE/USDT": "BREAK",
    },
)

ALPACA = _registry(
    "alpaca_paper",
    {
        "BTC/USD": "TRADING",
        "ETH/USD": "TRADING",
        "SOL/USD": "TRADING",
        "DOGE/USD": "TRADING",
    },
)


# ---------------------------------------------------------------------------
# 1. Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    def test_a_space_separated_string_is_split(self):
        assert parse_symbol_list("BTC/USDT ETH/USDT") == ("BTC/USDT", "ETH/USDT")

    def test_a_comma_separated_string_is_split(self):
        assert parse_symbol_list("BTC/USDT,ETH/USDT") == ("BTC/USDT", "ETH/USDT")

    def test_a_venue_spelling_is_canonicalised(self):
        assert parse_symbol_list("btcusdt") == ("BTC/USDT",)
        assert parse_symbol_list("btc-usdt") == ("BTC/USDT",)

    def test_duplicates_keep_the_first_mention_and_collapse(self):
        assert parse_symbol_list("BTC/USDT btc/usdt ETH/USDT") == (
            "BTC/USDT",
            "ETH/USDT",
        )

    def test_a_malformed_symbol_is_a_loud_error(self):
        with pytest.raises(SymbolConfigurationError):
            parse_symbol_list("XYZ")

    def test_an_empty_list_is_empty(self):
        assert parse_symbol_list("") == ()
        assert parse_symbol_list(None) == ()


# ---------------------------------------------------------------------------
# 2. Two-venue validation
# ---------------------------------------------------------------------------


class TestTwoVenueValidation:
    def test_a_symbol_on_both_venues_is_accepted(self):
        config = resolve_symbol_configuration(
            ["BTC/USDT"],
            market_data_registry=BINANCE,
            execution_registry=ALPACA,
        )
        assert config.accepted == ("BTC/USDT",)
        assert config.rejected == ()

    def test_the_accepted_symbol_carries_both_venue_spellings(self):
        config = resolve_symbol_configuration(
            ["BTC/USDT"],
            market_data_registry=BINANCE,
            execution_registry=ALPACA,
        )
        mapping = config.mapping_for("BTC/USDT")
        assert mapping is not None
        assert mapping.market_data_symbol == "BTCUSDT"
        assert mapping.execution_symbol == "BTC/USD"
        assert mapping.market_data_quote == "USDT"
        assert mapping.execution_quote == "USD"
        assert mapping.fully_tradable is True

    def test_a_symbol_the_market_data_venue_does_not_list_is_rejected(self):
        config = resolve_symbol_configuration(
            ["XRP/USDT"],
            market_data_registry=BINANCE,
            execution_registry=ALPACA,
        )
        assert config.accepted == ()
        rejection = config.rejected_for("XRP/USDT")
        assert rejection is not None
        assert "market-data venue" in rejection.reason

    def test_a_symbol_the_execution_venue_does_not_list_is_rejected(self):
        config = resolve_symbol_configuration(
            ["DOGE/USDT"],
            market_data_registry=BINANCE,
            execution_registry=ALPACA,
        )
        # DOGE is on both registries above, so use a market-data-only pair.
        market_only = _registry("binance", {"ETH/USDT": "TRADING"})
        config = resolve_symbol_configuration(
            ["ETH/USDT"],
            market_data_registry=market_only,
            execution_registry=_registry("alpaca_paper", {"BTC/USD": "TRADING"}),
        )
        assert config.accepted == ()
        rejection = config.rejected_for("ETH/USDT")
        assert rejection is not None
        assert "execution venue" in rejection.reason

    def test_a_halted_pair_is_rejected_with_its_status(self):
        config = resolve_symbol_configuration(
            ["FAKE/USDT"],
            market_data_registry=BINANCE,
            execution_registry=ALPACA,
        )
        assert config.accepted == ()
        rejection = config.rejected_for("FAKE/USDT")
        assert rejection is not None
        assert "BREAK" in rejection.reason

    def test_an_unavailable_execution_registry_rejects_everything(self):
        config = resolve_symbol_configuration(
            ["BTC/USDT"],
            market_data_registry=BINANCE,
            execution_registry=None,
        )
        assert config.accepted == ()
        rejection = config.rejected_for("BTC/USDT")
        assert rejection is not None
        assert "not available" in rejection.reason

    def test_accepted_and_rejected_are_reported_separately(self):
        config = resolve_symbol_configuration(
            ["BTC/USDT", "XRP/USDT", "ETH/USDT"],
            market_data_registry=BINANCE,
            execution_registry=ALPACA,
        )
        assert config.accepted == ("BTC/USDT", "ETH/USDT")
        assert [r.symbol for r in config.rejected] == ["XRP/USDT"]

    def test_a_rejected_symbol_is_never_substituted_for_another(self):
        config = resolve_symbol_configuration(
            ["XRP/USDT"],
            market_data_registry=BINANCE,
            execution_registry=ALPACA,
        )
        assert config.accepted == ()
        assert config.mappings == ()

    def test_the_description_names_what_was_dropped(self):
        config = resolve_symbol_configuration(
            ["BTC/USDT", "XRP/USDT"],
            market_data_registry=BINANCE,
            execution_registry=ALPACA,
        )
        described = config.describe()
        assert "1 accepted" in described
        assert "XRP/USDT" in described

    def test_the_dict_form_is_serialisable(self):
        config = resolve_symbol_configuration(
            ["BTC/USDT"],
            market_data_registry=BINANCE,
            execution_registry=ALPACA,
        )
        payload = json.dumps(config.to_dict())
        assert "BTC/USDT" in payload
        assert "BTC/USD" in payload


# ---------------------------------------------------------------------------
# 3. The environment variables keep their separate jobs
# ---------------------------------------------------------------------------


class TestEnvironmentVariables:
    def test_the_market_data_variable_is_read(self, monkeypatch):
        monkeypatch.setenv(CRYPTO_SYMBOLS_ENV, "BTC/USDT ETH/USDT")
        assert env_symbols(CRYPTO_SYMBOLS_ENV) == ("BTC/USDT", "ETH/USDT")

    def test_the_experiment_variable_is_read(self, monkeypatch):
        monkeypatch.setenv(EXPERIMENT_SYMBOLS_ENV, "SOL/USDT")
        assert env_symbols(EXPERIMENT_SYMBOLS_ENV) == ("SOL/USDT",)

    def test_an_unset_variable_is_empty(self, monkeypatch):
        monkeypatch.delenv(CRYPTO_SYMBOLS_ENV, raising=False)
        assert env_symbols(CRYPTO_SYMBOLS_ENV) == ()

    def test_the_two_default_lists_are_not_mixed(self):
        """The defaults must be single-venue lists, not one mixed list.

        Binance quotes USDT pairs and Alpaca quotes USD pairs. A single list that
        mixed the two would ask one venue for a pair it does not serve.
        """
        for symbol in DEFAULT_MARKET_DATA_SYMBOLS:
            assert symbol.endswith("/USDT"), symbol
        for symbol in DEFAULT_EXECUTION_SYMBOLS:
            assert symbol.endswith("/USD"), symbol

    def test_the_market_data_default_has_no_usd_pairs(self):
        assert not any(s.endswith("/USD") for s in DEFAULT_MARKET_DATA_SYMBOLS)

    def test_the_execution_default_has_no_usdt_pairs(self):
        assert not any(s.endswith("/USDT") for s in DEFAULT_EXECUTION_SYMBOLS)

    def test_the_two_defaults_cover_the_same_base_assets(self):
        assert {s.split("/")[0] for s in DEFAULT_MARKET_DATA_SYMBOLS} == {
            s.split("/")[0] for s in DEFAULT_EXECUTION_SYMBOLS
        }


# ---------------------------------------------------------------------------
# 4. The deployment configuration
# ---------------------------------------------------------------------------


class TestDeploymentConfiguration:
    @pytest.fixture
    def railway(self) -> dict:
        root = Path(__file__).resolve().parents[1]
        return json.loads((root / "railway.json").read_text())

    def test_the_deployment_symbol_list_is_not_mixed(self, railway):
        variables = railway["environments"]["production"]["variables"]
        raw = variables["TRADINGAGENTS_CRYPTO_SYMBOLS"]
        symbols = parse_symbol_list(raw)
        assert symbols, "the deployment names no crypto symbols"
        for symbol in symbols:
            assert symbol.endswith("/USDT"), (
                f"{symbol} is not a Binance pair; the market-data venue quotes USDT "
                "pairs, so a USD pair here would never resolve"
            )

    def test_the_deployment_execution_list_matches_the_market_data_list(self, railway):
        variables = railway["environments"]["production"]["variables"]
        display = parse_symbol_list(variables["TRADINGAGENTS_CRYPTO_SYMBOLS"])
        experiment = parse_symbol_list(variables["TRADINGAGENTS_EXPERIMENT_SYMBOLS"])
        assert display == experiment

    def test_the_deployment_names_no_usd_pair_in_the_market_data_list(self, railway):
        variables = railway["environments"]["production"]["variables"]
        raw = variables["TRADINGAGENTS_CRYPTO_SYMBOLS"]
        symbols = parse_symbol_list(raw)
        assert not any(s.endswith("/USD") for s in symbols)

    def test_the_deployment_keeps_the_risk_limits(self, railway):
        variables = railway["environments"]["production"]["variables"]
        assert variables["TRADINGAGENTS_PAPER_ORDER_NOTIONAL"] == "25"

    def test_the_deployment_never_names_a_live_endpoint(self, railway):
        raw = json.dumps(railway)
        assert "api.alpaca.markets" not in raw
        assert "broker-api.alpaca.markets" not in raw
