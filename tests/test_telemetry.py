"""Telemetry: cost accounting, budgets enforced before the spend, and run records.

The budget tests are the important ones. A limit that is only checked after a
call is a report, not a limit: it tells you that you overspent. These tests pin
that the check happens *before* the call is made, by counting calls that actually
ran.

The run-metadata tests pin the other convention that matters: a field the system
could not determine is ``None``, never a plausible-looking default, and any
nondeterminism is flagged rather than left implicit.
"""

from __future__ import annotations

import json

import pytest

from tradingagents.telemetry import (
    Budget,
    BudgetExceededError,
    LLMTelemetry,
    ModelConfig,
    ModelPrice,
    PriceTable,
    RunMetadata,
)


def _table() -> PriceTable:
    table = PriceTable()
    table.set("openai", "gpt-5", ModelPrice(input_per_million=2.0, output_per_million=8.0))
    return table


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPricing:
    def test_a_cost_is_per_million_tokens(self):
        price = ModelPrice(input_per_million=2.0, output_per_million=8.0)
        # 1M in + 1M out
        assert price.cost(input_tokens=1_000_000, output_tokens=1_000_000) == pytest.approx(10.0)

    def test_a_small_call_costs_a_fraction_of_a_cent(self):
        price = ModelPrice(input_per_million=2.0, output_per_million=8.0)
        assert price.cost(input_tokens=1_000, output_tokens=500) == pytest.approx(0.006)

    def test_the_table_is_case_insensitive(self):
        table = _table()
        assert table.get("OpenAI", "GPT-5") is not None

    def test_an_unknown_model_has_no_price(self):
        assert _table().get("openai", "gpt-9-unknown") is None

    def test_an_unknown_model_has_no_cost_not_a_zero(self):
        """Zero would look like a free call; None says the price is unknown."""
        assert _table().cost(provider="openai", model="mystery", input_tokens=1_000,
                             output_tokens=1_000) is None

    def test_there_are_no_built_in_prices(self):
        """A stale hardcoded rate is worse than none, so the default table is empty."""
        from tradingagents.telemetry import EMPTY_PRICE_TABLE

        assert len(EMPTY_PRICE_TABLE) == 0
        assert EMPTY_PRICE_TABLE.cost(provider="openai", model="gpt-5", input_tokens=1,
                                      output_tokens=1) is None


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecording:
    def test_a_call_is_recorded_with_its_tokens(self):
        telemetry = LLMTelemetry(price_table=_table())
        record = telemetry.record(provider="openai", model="gpt-5",
                                  input_tokens=1_000, output_tokens=500)
        assert record.total_tokens == 1_500
        assert record.estimated_cost == pytest.approx(0.006)

    def test_an_unpriced_call_records_a_none_cost(self):
        telemetry = LLMTelemetry()
        record = telemetry.record(provider="openai", model="gpt-5",
                                  input_tokens=100, output_tokens=50)
        assert record.estimated_cost is None

    def test_records_are_sequenced(self):
        telemetry = LLMTelemetry()
        for _ in range(3):
            telemetry.record(provider="p", model="m", input_tokens=1, output_tokens=1)
        assert [r.sequence for r in telemetry.records] == [0, 1, 2]

    def test_a_record_holds_no_prompt_or_response_text(self):
        """A telemetry log must not become a second copy of research content."""
        telemetry = LLMTelemetry()
        record = telemetry.record(provider="openai", model="gpt-5", input_tokens=1,
                                  output_tokens=1, purpose="fundamental_analyst")
        fields = set(record.to_dict())
        assert not {"prompt", "response", "messages", "content"} & fields
        assert record.purpose == "fundamental_analyst"

    def test_an_error_is_recorded(self):
        telemetry = LLMTelemetry()
        record = telemetry.record(provider="openai", model="gpt-5", input_tokens=0,
                                  output_tokens=0, error="RateLimitError: too many")
        assert not record.succeeded
        assert "RateLimitError" in record.error


# ---------------------------------------------------------------------------
# The timed call context manager
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimedCall:
    def test_a_call_is_recorded_on_exit(self):
        telemetry = LLMTelemetry(price_table=_table())
        with telemetry.timed_call(provider="openai", model="gpt-5") as call:
            call["input_tokens"] = 1_000
            call["output_tokens"] = 500
        assert len(telemetry.records) == 1
        assert telemetry.records[0].estimated_cost == pytest.approx(0.006)

    def test_a_failed_call_is_still_recorded(self):
        """A failed call cost latency, and possibly tokens; it must not vanish."""
        telemetry = LLMTelemetry()
        with (
            pytest.raises(ValueError),
            telemetry.timed_call(provider="openai", model="gpt-5"),
        ):
            raise ValueError("boom")
        assert len(telemetry.records) == 1
        assert "boom" in telemetry.records[0].error

    def test_latency_is_measured(self):
        clock_values = iter([100.0, 100.25])
        telemetry = LLMTelemetry(clock=lambda: next(clock_values))
        with telemetry.timed_call(provider="p", model="m"):
            pass
        assert telemetry.records[0].latency_seconds == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBudgets:
    def test_no_budget_is_unbounded(self):
        telemetry = LLMTelemetry()
        for _ in range(50):
            telemetry.check_budget()
            telemetry.record(provider="p", model="m", input_tokens=1, output_tokens=1)
        assert telemetry.summary().calls == 50

    def test_the_call_limit_stops_before_the_call(self):
        """The check happens first, so the overspending call is never made."""
        telemetry = LLMTelemetry(budget=Budget(max_llm_calls=2))
        executed = 0
        for _ in range(5):
            try:
                with telemetry.timed_call(provider="p", model="m"):
                    executed += 1
            except BudgetExceededError:
                break
        assert executed == 2
        assert len(telemetry.records) == 2

    def test_the_cost_limit_stops_before_the_call(self):
        telemetry = LLMTelemetry(
            price_table=_table(), budget=Budget(max_run_cost=0.01)
        )
        executed = 0
        for _ in range(10):
            try:
                with telemetry.timed_call(provider="openai", model="gpt-5") as call:
                    call["input_tokens"] = 1_000
                    call["output_tokens"] = 500  # 0.006 each
                    executed += 1
            except BudgetExceededError:
                break
        # Two calls cost 0.012, so the third is refused before it runs.
        assert executed == 2
        assert telemetry.estimated_cost() == pytest.approx(0.012)

    def test_the_error_names_the_limit_that_was_hit(self):
        telemetry = LLMTelemetry(budget=Budget(max_llm_calls=1))
        telemetry.record(provider="p", model="m", input_tokens=1, output_tokens=1)
        with pytest.raises(BudgetExceededError) as excinfo:
            telemetry.check_budget()
        assert excinfo.value.limit == "max_llm_calls"
        assert "before the call was made" in str(excinfo.value)

    def test_a_retry_limit_is_enforced(self):
        telemetry = LLMTelemetry(budget=Budget(max_retries=2))
        telemetry.check_retries(2)
        with pytest.raises(BudgetExceededError):
            telemetry.check_retries(3)

    def test_an_unpriced_run_cannot_hit_a_cost_limit(self):
        """An unknown cost cannot be shown to breach a limit, so it is not assumed to."""
        telemetry = LLMTelemetry(budget=Budget(max_run_cost=0.0001))
        for _ in range(5):
            telemetry.check_budget()
            telemetry.record(provider="unknown", model="unknown", input_tokens=1_000,
                             output_tokens=1_000)
        assert telemetry.estimated_cost() is None
        assert telemetry.summary().unpriced_calls == 5

    def test_a_negative_limit_is_rejected(self):
        with pytest.raises(ValueError):
            Budget(max_run_cost=-1.0)

    def test_a_zero_call_limit_stops_immediately(self):
        telemetry = LLMTelemetry(budget=Budget(max_llm_calls=0))
        with pytest.raises(BudgetExceededError):
            telemetry.check_budget()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSummary:
    def test_totals_are_aggregated(self):
        telemetry = LLMTelemetry(price_table=_table())
        telemetry.record(provider="openai", model="gpt-5", input_tokens=1_000, output_tokens=500)
        telemetry.record(provider="openai", model="gpt-5", input_tokens=2_000, output_tokens=0)
        summary = telemetry.summary()
        assert summary.calls == 2
        assert summary.input_tokens == 3_000
        assert summary.output_tokens == 500
        assert summary.total_tokens == 3_500
        assert summary.estimated_cost == pytest.approx(0.006 + 0.004)

    def test_an_unpriced_call_makes_the_total_unknown(self):
        """A partial total would understate the run, which is the wrong direction."""
        telemetry = LLMTelemetry(price_table=_table())
        telemetry.record(provider="openai", model="gpt-5", input_tokens=1_000, output_tokens=0)
        telemetry.record(provider="mystery", model="mystery", input_tokens=1_000, output_tokens=0)
        assert telemetry.summary().estimated_cost is None

    def test_the_breakdown_is_per_model(self):
        telemetry = LLMTelemetry(price_table=_table())
        telemetry.record(provider="openai", model="gpt-5", input_tokens=1_000, output_tokens=0)
        telemetry.record(provider="anthropic", model="claude", input_tokens=500, output_tokens=0)
        by_model = telemetry.summary().by_model
        assert set(by_model) == {"openai/gpt-5", "anthropic/claude"}
        assert by_model["openai/gpt-5"]["input_tokens"] == 1_000

    def test_failures_and_retries_are_counted(self):
        telemetry = LLMTelemetry()
        telemetry.record(provider="p", model="m", input_tokens=0, output_tokens=0,
                         error="timeout", retries=2)
        telemetry.record(provider="p", model="m", input_tokens=0, output_tokens=0)
        summary = telemetry.summary()
        assert summary.failures == 1
        assert summary.retries == 2

    def test_an_empty_run_has_zero_totals(self):
        summary = LLMTelemetry().summary()
        assert summary.calls == 0
        assert summary.estimated_cost == 0.0
        assert summary.mean_latency_seconds is None

    def test_the_summary_is_json_safe(self):
        telemetry = LLMTelemetry(price_table=_table())
        telemetry.record(provider="openai", model="gpt-5", input_tokens=1, output_tokens=1)
        json.dumps(telemetry.summary().to_dict())

    def test_a_configured_budget_is_reported(self):
        telemetry = LLMTelemetry(budget=Budget(max_run_cost=1.0))
        assert telemetry.summary().budget == {
            "max_run_cost": 1.0, "max_llm_calls": None, "max_retries": None
        }

    def test_an_unconfigured_budget_is_absent(self):
        assert LLMTelemetry().summary().budget is None


@pytest.mark.unit
class TestTelemetryFiles:
    def test_the_log_is_written_as_json(self, tmp_path):
        telemetry = LLMTelemetry(price_table=_table())
        telemetry.record(provider="openai", model="gpt-5", input_tokens=1_000, output_tokens=500)
        path = telemetry.write_json(tmp_path / "nested" / "telemetry.json")
        payload = json.loads(path.read_text())
        assert payload["summary"]["calls"] == 1
        assert len(payload["calls"]) == 1

    def test_the_written_log_contains_no_secret_shaped_keys(self, tmp_path):
        telemetry = LLMTelemetry()
        telemetry.record(provider="openai", model="gpt-5", input_tokens=1, output_tokens=1)
        path = telemetry.write_json(tmp_path / "t.json")
        text = path.read_text().lower()
        for forbidden in ("api_key", "apikey", "secret", "token\":", "authorization"):
            assert forbidden not in text


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunMetadata:
    def test_a_run_id_is_generated_and_unique(self):
        assert RunMetadata().run_id != RunMetadata().run_id

    def test_the_code_version_and_commit_are_recorded(self):
        metadata = RunMetadata()
        assert metadata.python_version
        assert metadata.code_version  # the package is installed in this environment

    def test_an_unknown_value_is_none_not_a_default(self):
        """A fabricated configuration looks like evidence; a missing one does not."""
        metadata = RunMetadata()
        assert metadata.model.temperature is None
        assert metadata.model.model is None
        assert metadata.research_depth is None

    def test_an_unrecorded_temperature_flags_nondeterminism(self):
        metadata = RunMetadata()
        metadata.assess_determinism()
        assert metadata.nondeterministic
        assert any("temperature" in note for note in metadata.nondeterminism_notes)

    def test_a_zero_temperature_is_deterministic(self):
        metadata = RunMetadata(model=ModelConfig(temperature=0.0))
        metadata.assess_determinism()
        assert not any("temperature 0" in note for note in metadata.nondeterminism_notes)

    def test_a_non_zero_temperature_is_flagged(self):
        metadata = RunMetadata(model=ModelConfig(temperature=0.7))
        metadata.assess_determinism()
        assert metadata.nondeterministic
        assert any("0.7" in note for note in metadata.nondeterminism_notes)

    def test_an_explicit_reason_can_be_added(self):
        metadata = RunMetadata()
        metadata.flag_nondeterminism("the news vendor backfills articles")
        assert "the news vendor backfills articles" in metadata.nondeterminism_notes

    def test_a_reason_is_not_duplicated(self):
        metadata = RunMetadata()
        metadata.flag_nondeterminism("same")
        metadata.flag_nondeterminism("same")
        assert metadata.nondeterminism_notes.count("same") == 1

    def test_the_record_round_trips_through_a_file(self, tmp_path):
        metadata = RunMetadata(
            model=ModelConfig(provider="openai", model="gpt-5", temperature=0.0),
            symbols=["AAPL"], research_depth=2, random_seed=7,
            configuration={"holding_period_days": 5},
        )
        path = metadata.write_json(tmp_path / "run.json")
        loaded = RunMetadata.load(path)
        assert loaded.run_id == metadata.run_id
        assert loaded.model.model == "gpt-5"
        assert loaded.symbols == ["AAPL"]
        assert loaded.configuration == {"holding_period_days": 5}

    def test_loading_tolerates_unknown_fields(self, tmp_path):
        payload = RunMetadata().to_dict()
        payload["a_field_from_the_future"] = 1
        path = tmp_path / "run.json"
        path.write_text(json.dumps(payload))
        assert RunMetadata.load(path).run_id == payload["run_id"]

    def test_the_record_is_json_safe(self):
        metadata = RunMetadata()
        metadata.telemetry = {"calls": 3}
        json.dumps(metadata.to_dict())

    def test_the_configuration_is_recorded_verbatim(self):
        """The run's settings must be recoverable from the record, not summarised."""
        config = {"risk": {"max_position_size": 0.1}, "costs": {"commission_rate": 0.001}}
        metadata = RunMetadata(configuration=config)
        assert metadata.to_dict()["configuration"] == config
