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
- ``diagnose`` — report, without revealing a secret, whether a provider's
  credential env var is present and whether the request would carry it.
- ``broker-check`` (alias ``alpaca-diagnose``) — read-only Alpaca **paper**
  diagnostic. Reports whether the paper credential env vars are present (presence
  only) and, with ``--connect``, reads the paper account and reports its figures
  and restrictions. It places no order and changes nothing.
- ``serve`` — run the local monitoring dashboard (a small stdlib HTTP server).
  Read-only by default; ``--allow-control`` adds the ability to advance the
  simulation, still simulation-only. It binds loopback by default and refuses a
  public bind without a token.

The default price source is the frozen historical dataset; a vendor source is
opt-in via ``--source``. A vendor source is fetched once, when the experiment is
created, and never while resuming a run: a restart reads the source from the
saved state, so no network call happens on ``report``/``verify`` and a resumed
``run`` does not refetch. No command places a live order or contacts a broker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .broker import BrokerError
from .config import ExperimentConfig
from .experiment import ExperimentError, PaperExperiment
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


def _root(args) -> Path:
    root = getattr(args, "root", None)
    if not root:
        raise RunnerError("this command needs --root <experiment directory>")
    return Path(root)


def _frozen_source() -> PriceSource:
    from tradingagents.baseline.data import load_feed
    from tradingagents.llm_evaluation.protocol import DEFAULT_PROTOCOL

    try:
        feed = load_feed(DEFAULT_PROTOCOL)
    except Exception as exc:  # a missing dataset is reported, not hidden
        raise RunnerError(f"the frozen dataset is unavailable: {exc}") from exc
    return FrozenPriceSource(feed, name="frozen_baseline")


def _make_inference(config: ExperimentConfig, *, stub: bool):
    from .runner import build_inference

    return build_inference(config, stub=stub)


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
    root = _root(args)
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
    root = _root(args)
    if (root / "session_state.json").exists():
        source = _restored_source(root)
        # Build the seam from the *persisted* config, not the CLI defaults, so a
        # resumed run decides through the provider the experiment was started with.
        experiment = PaperExperiment.load(root, source=source)
        experiment.inference = _make_inference(experiment.config, stub=args.stub)
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
    from .runner import process_pending

    if args.sessions:
        outcomes = process_pending(experiment, sessions=args.sessions)
    else:
        outcomes = process_pending(experiment)

    for outcome in outcomes:
        state = outcome.decision_state or "?"
        print(f"{outcome.session}: {state}")

    report = build_report(experiment)
    print(report.summary())
    if args.report_out:
        print(f"report written to {report.write(args.report_out)}")
    return 0


def _report(args) -> int:
    root = _root(args)
    source = _restored_source(root)
    experiment = PaperExperiment.load(root, source=source)
    report = build_report(experiment)
    if args.out:
        print(f"report written to {report.write(args.out)}")
    else:
        print(json.dumps(report.to_dict(), indent=2))
    return 0


def _serve(args) -> int:
    """Run the local dashboard. Simulation-only monitoring and control.

    The server is read-only unless ``--allow-control`` is passed. It binds
    loopback by default; a non-loopback bind requires a token
    (``TRADINGAGENTS_DASHBOARD_TOKEN`` or ``--token``), refused otherwise.

    ``--allow-paper-trading`` additionally enables the START/STOP control for the
    Alpaca **paper** worker. It is off by default, so the dashboard cannot place a
    paper order unless the operator opts in, and it never enables live trading.
    """
    from .dashboard import DashboardAuthError, serve

    root = _root(args)
    # Honour a platform-provided port (Railway sets $PORT) when --port is omitted.
    port = args.port if args.port is not None else int(os.environ.get("PORT", "8000"))
    try:
        serve(
            root,
            host=args.host,
            port=port,
            token=args.token,
            allow_control=args.allow_control,
            stub=args.stub,
            auto_advance=getattr(args, "auto_advance", 0) or 0,
            auto_interval=getattr(args, "auto_interval", 60.0) or 60.0,
            allow_paper_trading=getattr(args, "allow_paper_trading", False),
            paper_order_notional=getattr(args, "paper_order_notional", 25.0) or 25.0,
        )
    except DashboardAuthError as exc:
        raise RunnerError(str(exc)) from exc
    return 0


def _sources(args) -> int:
    for name, cls in provider_registry().items():
        kind = getattr(cls, "kind", "unknown")
        current = name == "finnhub"
        print(f"{name:16s} kind={kind:14s} current_market={current}")
    print(f"\nknown source names: {', '.join(KNOWN_SOURCES)}")
    return 0


def _diagnose(args) -> int:
    """Report, without revealing a secret, whether a provider credential is set.

    Answers the two questions a deploy needs: is the env var present, and would
    the outgoing request carry the ``apikey`` parameter? It never prints the key,
    any part of it, or its length — only presence and the parameter *name*.
    """
    from .providers.alpha_vantage import (
        API_BASE_URL as AV_URL,
        API_KEY_ENV as AV_ENV,
        build_request_params as av_params,
    )
    from .providers.finnhub import API_KEY_ENV as FH_ENV

    requested = args.source or "frozen"
    print(f"requested source: {requested}")
    print(f"known sources: {', '.join(KNOWN_SOURCES)}")

    env_name = {"alpha_vantage": AV_ENV, "finnhub": FH_ENV}.get(requested)
    if env_name is None:
        print(f"{requested}: no credential required")
        return 0

    value = os.environ.get(env_name, "")
    present = bool(value.strip())
    print(f"credential env var: {env_name}")
    print(f"env var present and non-blank: {present}")
    if not present:
        print(
            f"the request would fail before it is sent: {env_name} is unset or blank, "
            "and the provider refuses rather than sending an unauthenticated call"
        )
        return 0

    if requested == "alpha_vantage":
        # Only parameter *names* are printed, never values. The key is passed
        # through the same builder the real request uses, so this check cannot
        # drift from the outgoing call.
        params = av_params(symbol="SPY", api_key=value)
        print(f"request URL: {AV_URL}")
        print(f"request carries apikey parameter: {'apikey' in params}")
        print(f"request parameter names: {', '.join(sorted(params))}")
    else:
        print("request carries token parameter: True (Finnhub uses 'token')")
    return 0


def _verify(args) -> int:
    from .journal import ExperimentJournal

    root = _root(args)
    try:
        journal = ExperimentJournal.load(root / "journal.json")
    except JournalError as exc:
        print(f"JOURNAL INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"journal OK: {len(journal)} entries, chain verified")
    return 0


def _paper_trade(args) -> int:
    """Run one explicit Alpaca **paper** trading cycle from the CLI.

    This is the deliberate, opt-in path: it loads the experiment, builds the
    paper worker, runs preflight (broker connection, account status, buying power,
    market data, risk config), processes **at most one** decision, and stops. It
    places an order only if the decision source yields a tradeable decision *and*
    every local risk check passes.

    ``--dry-run`` performs preflight and reports what would happen without placing
    an order. There is no flag that reaches a live endpoint: the broker's gate
    refuses every live host, and the endpoint is not configurable here.
    """
    from .dashboard.paper_control import PaperTradingController

    root = _root(args)
    controller = PaperTradingController(
        root,
        enabled=True,
        order_notional=getattr(args, "paper_order_notional", 25.0) or 25.0,
    )
    if getattr(args, "dry_run", False):
        worker = controller._ensure_worker()  # noqa: SLF001 - the CLI owns the worker
        result = worker.preflight()
        print(json.dumps({"preflight": result, "placed_order": False}, indent=2, default=str))
        return 0

    started = controller.start()
    print(f"paper worker: {started['status']}")
    for check in started.get("checks", []):
        print(f"  preflight {check['check']}: {'ok' if check['ok'] else 'FAILED'}")
    try:
        cycle = controller.run_once()
    finally:
        controller.stop()
    print(json.dumps(cycle, indent=2, default=str))
    return 0


def _broker_check(args) -> int:
    """Read-only Alpaca **paper** diagnostic. Places no order, changes nothing.

    Reports, without any secret value (never the key, any part of it, or its
    length):

    - that the only endpoint used is the approved Alpaca paper host;
    - whether the paper credential env vars are present (presence only);
    - whether any *live* credential variable is present (a misconfiguration it
      will not use).

    With ``--connect`` it performs the single read-only ``GET /v2/account`` call
    and reports the account's cash, equity, buying power, currency and active
    restrictions. It never submits, modifies or cancels an order, and there is no
    flag that does: a paper order is placed only from code, deliberately, with
    ``allow_orders=True``. It fails safe (a non-zero exit with a clear message) if
    the credentials are absent, authentication fails, or the paper environment
    cannot be proven.
    """
    from .broker import (
        PAPER_BASE_URL,
        AlpacaPaperBroker,
        BrokerError,
        BrokerNotConfiguredError,
        broker_status,
    )

    status = broker_status()
    print(f"approved paper endpoint: {status.paper_endpoint}")
    print(f"endpoint is the approved paper host: {status.endpoint_ok}")
    print(f"paper key env var: {status.key_env} present: {status.key_present}")
    print(f"paper secret env var: {status.secret_env} present: {status.secret_present}")
    if status.live_credentials_present:
        print(
            "note: live credential variables are present and will NOT be used: "
            + ", ".join(status.live_credentials_present)
        )
    configured = status.key_present and status.secret_present
    if not configured:
        message = (
            f"the broker path is not configured: set {status.key_env} and "
            f"{status.secret_env} (values are never printed)."
        )
        if getattr(args, "connect", False):
            # An explicit request to connect cannot be honoured; fail safely.
            print(f"broker check failed: {message}", file=sys.stderr)
            return 2
        print(f"{message} No order is placed by this command.")
        return 0
    if not getattr(args, "connect", False):
        print("configuration OK (pass --connect to read the account; still no order)")
        return 0

    try:
        broker = AlpacaPaperBroker(base_url=PAPER_BASE_URL)
        account = broker.connect()
    except (BrokerNotConfiguredError, BrokerError) as exc:
        # Safe, secret-free failure: a missing key, a refused credential, a
        # timeout, or a non-object response. Never a fallback to a live host.
        print(f"broker check failed: {exc}", file=sys.stderr)
        return 2

    # Reaching here proves the paper environment: the only URL this adapter can
    # request is the validated paper host, so a successful account read is proof
    # that the credentials are paper credentials.
    print("paper environment: CONFIRMED (account read from the approved paper host)")
    print(f"account id: {account.account_id} status: {account.status}")
    print(f"currency: {account.currency or 'unknown'}")
    print(
        f"cash: {account.cash} equity: {account.equity} "
        f"buying_power: {account.buying_power}"
    )
    if account.portfolio_value is not None:
        print(f"portfolio_value: {account.portfolio_value}")
    if account.multiplier is not None:
        print(f"multiplier: {account.multiplier}")
    restrictions = account.restrictions()
    print(
        "restrictions: " + (", ".join(restrictions) if restrictions else "none reported")
    )
    print("read-only check complete; no order was placed")
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
    parser.add_argument(
        "command",
        choices=(
            "start",
            "run",
            "report",
            "verify",
            "sources",
            "diagnose",
            "broker-check",
            "alpaca-diagnose",
            "paper-trade",
            "serve",
        ),
    )
    parser.add_argument(
        "--root", default=None, help="experiment directory (required except for diagnose)"
    )
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
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "serve only: interface to bind. Defaults to loopback. A non-loopback "
            "host requires a token, and is refused without one."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="serve only: port to listen on (default: $PORT, else 8000)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "serve only: dashboard token. Defaults to the "
            "TRADINGAGENTS_DASHBOARD_TOKEN environment variable. Required for a "
            "non-loopback bind; never logged."
        ),
    )
    parser.add_argument(
        "--allow-control",
        action="store_true",
        help=(
            "serve only: enable the dashboard's advance-the-simulation action. "
            "Off by default, so the dashboard is read-only. It stays "
            "simulation-only; no broker order can be placed."
        ),
    )
    parser.add_argument(
        "--allow-paper-trading",
        action="store_true",
        help=(
            "serve only: enable the dashboard's START/STOP control for the Alpaca "
            "PAPER worker. Off by default. Paper only: the broker endpoint is fixed "
            "to Alpaca's paper host and every live endpoint is refused. Requires "
            "ALPACA_PAPER_API_KEY / ALPACA_PAPER_API_SECRET in the environment."
        ),
    )
    parser.add_argument(
        "--paper-order-notional",
        type=float,
        default=float(os.environ.get("TRADINGAGENTS_PAPER_ORDER_NOTIONAL", "25") or 25),
        help=(
            "paper trading: target cash per order (default: "
            "$TRADINGAGENTS_PAPER_ORDER_NOTIONAL, else 25)"
        ),
    )
    parser.add_argument(
        "--auto-advance",
        type=int,
        default=int(os.environ.get("TRADINGAGENTS_AUTO_ADVANCE_SESSIONS", "0") or 0),
        help=(
            "serve only: advance this many sessions every --auto-interval seconds, "
            "in-process, so one service both runs and monitors the experiment "
            "(default: $TRADINGAGENTS_AUTO_ADVANCE_SESSIONS, else 0 = monitor only). "
            "Requires --allow-control."
        ),
    )
    parser.add_argument(
        "--auto-interval",
        type=float,
        default=float(os.environ.get("TRADINGAGENTS_AUTO_ADVANCE_INTERVAL", "60") or 60),
        help="serve only: seconds between auto-advance ticks (default: 60)",
    )
    parser.add_argument(
        "--connect",
        action="store_true",
        help=(
            "broker-check / alpaca-diagnose only: also read the Alpaca paper "
            "account (read-only; places no order). There is no CLI flag that "
            "submits an order."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "paper-trade only: run preflight and report, without processing a "
            "decision or placing an order."
        ),
    )
    args = parser.parse_args(argv)

    handlers = {
        "start": _start,
        "run": _run,
        "report": _report,
        "verify": _verify,
        "sources": _sources,
        "diagnose": _diagnose,
        "broker-check": _broker_check,
        "alpaca-diagnose": _broker_check,  # an explicit, discoverable alias
        "paper-trade": _paper_trade,
        "serve": _serve,
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
        BrokerError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
