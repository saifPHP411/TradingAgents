"""The runnable surface for a paper experiment: start, resume, run, report.

A thin command layer over :class:`~.experiment.PaperExperiment`, so an operator can
drive the cycle from a shell without writing Python. It is deliberately small:
the experiment does the work, this decides *which sessions* to run and *when to
stop*.

Commands:

- ``start`` — create an experiment directory and its journal.
- ``run`` — process the next ``--sessions N`` sessions (or every remaining one),
  resuming if the directory already exists.
- ``report`` — write the report from the current state.
- ``verify`` — check the journal's hash chain.
- ``sources`` — list the selectable price sources.

The default price source is the frozen historical dataset; a vendor source is
opt-in via ``--source``. A vendor source is fetched once, when the experiment is
created, and never while resuming a run: a restart reads the source from the
saved state, so no network call happens on ``report``/``verify`` and a resumed
``run`` does not refetch. No command places a live order or contacts a broker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ExperimentConfig
from .experiment import ExperimentError, PaperExperiment
from .inference import ProviderInference, StubInference
from .journal import JournalError
from .providers import (
    KNOWN_SOURCES,
    ProviderError,
    SourceSelection,
    SourceSelectionError,
    provider_registry,
    select_source,
)
from .report import build_report
from .sources import FrozenPriceSource, PriceSource, PriceSourceError


class RunnerError(RuntimeError):
    """A command could not proceed."""


def _frozen_source() -> PriceSource:
    from tradingagents.baseline.data import load_feed
    from tradingagents.llm_evaluation.protocol import DEFAULT_PROTOCOL

    try:
        feed = load_feed(DEFAULT_PROTOCOL)
    except Exception as exc:  # a missing dataset is reported, not hidden
        raise RunnerError(f"the frozen dataset is unavailable: {exc}") from exc
    return FrozenPriceSource(feed, name="frozen_baseline")


def _make_inference(config: ExperimentConfig, *, stub: bool):
    if stub:
        return StubInference()
    return ProviderInference(provider=config.provider, model=config.model)


def _resolve_source(args, config: ExperimentConfig) -> SourceSelection:
    """Select the price source for a *new* run, honouring an explicit fallback."""
    requested = getattr(args, "source", None) or "frozen"
    fallback = getattr(args, "fallback", None)
    return select_source(
        requested,
        symbols=config.symbols,
        frozen_feed_factory=_frozen_source,
        fallback=fallback,
    )


def _restored_source(root: Path) -> PriceSource:
    """The source a saved experiment recorded, rebuilt without any network call.

    A resumed run must not refetch: the source is a *view* of already-fetched data
    (a daily-bar provider freezes its frame at construction; the frozen dataset is
    local). The bars a frame-backed source fetched are persisted in the state, so
    a vendor run resumes on the *same* data without a second, quota-consuming
    call. A source that cannot be persisted (a quote-only source) is refused
    rather than silently swapped for the frozen dataset.
    """
    state_path = root / "session_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"experiment state {state_path} is not usable: {exc}") from exc
    source = state.get("source", {})
    name = source.get("name", "frozen_baseline")
    if source.get("restore"):
        try:
            return PaperExperiment._restore_source(state)
        except ExperimentError as exc:
            raise RunnerError(str(exc)) from exc
    # No persisted bars (e.g. a quote-only source). Reloading the frozen dataset
    # here would silently change the data source, so refuse instead.
    if name.startswith("frozen"):
        reloaded = _frozen_source()
        _check_frozen_window(reloaded, source.get("provenance") or {})
        return reloaded
    raise RunnerError(
        f"this experiment ran on the source {name!r}, whose bars are not stored in "
        "the experiment directory; it cannot be resumed without re-fetching, and "
        "the CLI will not silently substitute the frozen dataset"
    )


def _check_frozen_window(source: PriceSource, recorded: dict) -> None:
    """Refuse a resume when the committed dataset no longer matches the run.

    The frozen dataset is reloaded from disk on resume rather than stored in the
    state. If the repository's dataset changed since the run began, the reloaded
    sessions would differ from the ones the run was built on, so a silent
    continue would mix two datasets. The recorded window is checked instead.
    """
    sessions = tuple(source.sessions())
    start, end = recorded.get("dataset_start"), recorded.get("dataset_end")
    if start and end and (sessions[0] != start or sessions[-1] != end):
        raise RunnerError(
            f"the committed dataset now spans {sessions[0]}..{sessions[-1]}, but this "
            f"experiment was built on {start}..{end}; refusing to resume on a "
            "different dataset"
        )


def _start(args) -> int:
    root = Path(args.root)
    if (root / "session_state.json").exists():
        raise RunnerError(
            f"an experiment already exists at {root}; use `run` to resume it"
        )
    config = ExperimentConfig.from_protocol(
        symbols=(args.symbol,) if args.symbol else ("SPY",),
        provider=args.provider,
        model=args.model,
    )
    selection = _resolve_source(args, config)
    experiment = PaperExperiment(
        source=selection.source,
        config=config,
        root=root,
        inference=_make_inference(config, stub=args.stub),
    )
    experiment.ensure_created()
    experiment.save_state()
    print(f"started experiment {experiment.experiment_id} at {root}")
    print(f"  source: {selection.source.name} (kind {getattr(selection.source, 'kind', '?')})")
    if selection.switched:
        print(f"  source switched: {selection.reason}")
    return 0


def _run(args) -> int:
    root = Path(args.root)
    if (root / "session_state.json").exists():
        source = _restored_source(root)
        inference = StubInference() if args.stub else ProviderInference(
            provider=args.provider, model=args.model
        )
        experiment = PaperExperiment.load(root, source=source, inference=inference)
    else:
        config = ExperimentConfig.from_protocol(
            symbols=(args.symbol,) if args.symbol else ("SPY",),
            provider=args.provider,
            model=args.model,
        )
        selection = _resolve_source(args, config)
        experiment = PaperExperiment(
            source=selection.source,
            config=config,
            root=root,
            inference=_make_inference(config, stub=args.stub),
        )
        experiment.ensure_created()
        if selection.switched:
            print(f"source switched: {selection.reason}")

    source = experiment.source
    sessions = list(source.sessions())
    if args.sessions:
        # Start after the last processed session, so a resume continues.
        processed = experiment.processed_sessions
        remaining = [s for s in sessions if s not in processed]
        to_run = remaining[: args.sessions]
    else:
        to_run = [s for s in sessions if s not in experiment.processed_sessions]

    for session in to_run:
        outcome = experiment.process_session(session)
        state = outcome.decision_state or "?"
        print(f"{session}: {state}")

    report = build_report(experiment)
    print(report.summary())
    if args.report_out:
        print(f"report written to {report.write(args.report_out)}")
    return 0


def _report(args) -> int:
    root = Path(args.root)
    source = _restored_source(root)
    experiment = PaperExperiment.load(root, source=source)
    report = build_report(experiment)
    if args.out:
        print(f"report written to {report.write(args.out)}")
    else:
        print(json.dumps(report.to_dict(), indent=2))
    return 0


def _sources(args) -> int:
    for name, cls in provider_registry().items():
        kind = getattr(cls, "kind", "unknown")
        current = name == "finnhub"
        print(f"{name:16s} kind={kind:14s} current_market={current}")
    print(f"\nknown source names: {', '.join(KNOWN_SOURCES)}")
    return 0


def _verify(args) -> int:
    from .journal import ExperimentJournal

    root = Path(args.root)
    try:
        journal = ExperimentJournal.load(root / "journal.json")
    except JournalError as exc:
        print(f"JOURNAL INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"journal OK: {len(journal)} entries, chain verified")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.paper_experiment",
        description=(
            "Prospective paper experiment. Paper only: no live order is ever placed. "
            "The default price source is the frozen historical dataset; a vendor "
            "source is opt-in with --source."
        ),
    )
    parser.add_argument("command", choices=("start", "run", "report", "verify", "sources"))
    parser.add_argument("--root", required=True, help="experiment directory")
    parser.add_argument("--symbol", default=None, help="symbol (default SPY)")
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--source",
        default="frozen",
        choices=KNOWN_SOURCES,
        help=(
            "price source for a new run: frozen (default, local dataset), "
            "alpha_vantage (free daily bars), or finnhub (free current-day quote; "
            "cannot drive a multi-session run)"
        ),
    )
    parser.add_argument(
        "--fallback",
        default=None,
        choices=KNOWN_SOURCES,
        help=(
            "an explicit source to use if --source fails to build; the switch is "
            "logged. Omit it to make a source failure stop the run rather than "
            "silently change the data."
        ),
    )
    parser.add_argument("--sessions", type=int, default=None, help="process at most N sessions")
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "use a non-genuine stub decision source (for offline demonstration); the "
            "decisions are labelled synthetic_fixture and are not real inferences"
        ),
    )
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    handlers = {
        "start": _start,
        "run": _run,
        "report": _report,
        "verify": _verify,
        "sources": _sources,
    }
    try:
        return handlers[args.command](args)
    except (
        RunnerError,
        ExperimentError,
        PriceSourceError,
        ProviderError,
        SourceSelectionError,
        JournalError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
