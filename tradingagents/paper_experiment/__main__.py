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

The default price source is the frozen historical dataset; there is no
current-market source, so ``run`` reports the experiment as offline paper. No
command places a live order or contacts a broker.
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
from .report import build_report
from .sources import (
    FrozenPriceSource,
    PriceSource,
    PriceSourceError,
)


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
    source = _frozen_source()
    experiment = PaperExperiment(
        source=source,
        config=config,
        root=root,
        inference=_make_inference(config, stub=args.stub),
    )
    experiment.ensure_created()
    experiment.save_state()
    print(f"started experiment {experiment.experiment_id} at {root}")
    return 0


def _run(args) -> int:
    root = Path(args.root)
    config = ExperimentConfig.from_protocol(
        symbols=(args.symbol,) if args.symbol else ("SPY",),
        provider=args.provider,
        model=args.model,
    )
    source = _frozen_source()
    if (root / "session_state.json").exists():
        experiment = PaperExperiment.load(
            root, source=source, inference=_make_inference(config, stub=args.stub)
        )
    else:
        experiment = PaperExperiment(
            source=source,
            config=config,
            root=root,
            inference=_make_inference(config, stub=args.stub),
        )
        experiment.ensure_created()

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
    source = _frozen_source()
    experiment = PaperExperiment.load(root, source=source)
    report = build_report(experiment)
    if args.out:
        print(f"report written to {report.write(args.out)}")
    else:
        print(json.dumps(report.to_dict(), indent=2))
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
            "Prospective paper experiment. Paper only: no live order is ever placed, "
            "and the default price source is the frozen historical dataset."
        ),
    )
    parser.add_argument("command", choices=("start", "run", "report", "verify"))
    parser.add_argument("--root", required=True, help="experiment directory")
    parser.add_argument("--symbol", default=None, help="symbol (default SPY)")
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
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

    handlers = {"start": _start, "run": _run, "report": _report, "verify": _verify}
    try:
        return handlers[args.command](args)
    except (RunnerError, ExperimentError, PriceSourceError, JournalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
