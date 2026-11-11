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

from .broker import (
    DECISION_SOURCE_CHOICES,
    DEFAULT_MAX_SNAPSHOT_AGE_SECONDS,
    EXECUTION_PROVIDER_ALPACA_PAPER,
    EXECUTION_PROVIDER_CHOICES,
    EXPERIMENT_SYMBOLS_ENV,
    MARKET_DATA_CHOICES,
    BrokerError,
)
from .config import ExperimentConfig
from .experiment import ExperimentError, PaperExperiment
from .journal import JournalError
from .market_data_wiring import (
    DECISION_SOURCE_ENV,
    EXECUTION_PROVIDER_ENV,
    FRACTIONAL_SHARES_ENV,
    MARKET_DATA_ENV,
    SNAPSHOT_MAX_AGE_ENV,
    MarketDataWiringError,
    describe_decision_source,
    describe_execution_provider,
    describe_provider,
    market_data_factory_for,
    resolve_execution_provider,
    resolve_paper_settings,
    resolve_settings,
)
from .providers import (
    KNOWN_SOURCES,
    ProviderError,
    SourceSelection,
    SourceSelectionError,
    provider_registry,
    select_source,
)
from .refresh import REFRESH_STATE_FILENAME, REFRESHABLE_KINDS
from .report import build_report
from .sources import FrozenPriceSource, PriceSource, PriceSourceError


class RunnerError(RuntimeError):
    """A command could not proceed."""


DEFAULT_EXPERIMENT_SYMBOL = "SPY"
"""The symbol an experiment uses when nothing names one.

The frozen dataset — the default source — carries exactly this one symbol. A
crypto deployment names its own pairs through ``--symbols`` or
``TRADINGAGENTS_EXPERIMENT_SYMBOLS``, so this default never silently applies to
a crypto run.
"""


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


def _parse_experiment_symbols(args) -> tuple[str, ...]:
    """Parse the configured experiment symbols from CLI or environment.

    Reads from --symbols CLI flag (if provided) or TRADINGAGENTS_EXPERIMENT_SYMBOLS
    environment variable. Returns a tuple of canonical symbols.

    With nothing configured, the single default symbol is used, exactly as before
    the multi-symbol option existed: an experiment needs *some* symbol, and the
    default source (the frozen dataset) carries exactly one. A crypto deployment
    names its own pairs through ``--symbols`` or the environment variable, so the
    default never silently applies to a crypto run.

    Raises RunnerError if configuration is invalid or empty.
    """
    # Check CLI --symbols first
    cli_symbols = getattr(args, "symbols", None)
    if cli_symbols is not None:
        if not cli_symbols:
            raise RunnerError("--symbols cannot be empty; provide at least one symbol")
        return tuple(str(s).strip().upper() for s in cli_symbols)

    # Check environment variable
    env_symbols = os.environ.get(EXPERIMENT_SYMBOLS_ENV, "").strip()
    if env_symbols:
        symbols = [s.strip().upper() for s in env_symbols.replace(",", " ").split() if s.strip()]
        if not symbols:
            raise RunnerError(f"{EXPERIMENT_SYMBOLS_ENV} is set but contains no valid symbols")
        return tuple(symbols)

    # Fall back to single --symbol for backward compatibility
    if getattr(args, "symbol", None):
        return (str(args.symbol).strip().upper(),)

    # The frozen dataset carries exactly one symbol, so an unconfigured experiment
    # uses it. A crypto deployment always names its own pairs.
    return (DEFAULT_EXPERIMENT_SYMBOL,)


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


def _record_initial_fetch(root: Path, source: PriceSource) -> None:
    """Record the experiment's first vendor fetch in the refresh sidecar.

    Only a refreshable vendor source counts; a frozen dataset or a fixture makes no
    network call and needs no bookkeeping. Best-effort: a failure here must not
    abort a successful `start`.
    """
    if getattr(source, "kind", None) not in REFRESHABLE_KINDS:
        return
    from datetime import UTC, datetime

    from .refresh import STATUS_INITIAL_FETCH, RefreshState

    provenance = getattr(source, "provenance", None)
    now = datetime.now(UTC)
    state = RefreshState(
        day=now.date().isoformat(),
        fetches_today=1,
        last_attempt_at=now.isoformat(),
        last_success_at=now.isoformat(),
        last_status=STATUS_INITIAL_FETCH,
        last_bar_date=getattr(provenance, "last_bar_date", None),
    )
    state.save(root / REFRESH_STATE_FILENAME)


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
    symbols = _parse_experiment_symbols(args)
    config = ExperimentConfig.from_protocol(
        symbols=symbols,
        provider=args.provider,
        model=args.model,
        fallback_provider=getattr(args, "fallback_provider", None),
        fallback_model=getattr(args, "fallback_model", None),
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
    # Record the initial fetch in the refresh sidecar, so the daily quota count is
    # accurate (the initial fetch is a vendor call) and the throttle holds, rather
    # than a `run` immediately spending a second call.
    _record_initial_fetch(root, selection.source)
    print(f"started experiment {experiment.experiment_id} at {root}")
    print(f"  symbols: {', '.join(symbols)}")
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
        symbols = _parse_experiment_symbols(args)
        config = ExperimentConfig.from_protocol(
            symbols=symbols,
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
    from .runner import advance

    advanced = (
        advance(experiment, sessions=args.sessions)
        if args.sessions
        else advance(experiment)
    )
    outcomes = advanced.outcomes

    if advanced.refresh is not None:
        print(advanced.refresh.summary_line())

    for outcome in outcomes:
        state = outcome.decision_state or "?"
        if outcome.decision_skip_reason:
            print(f"{outcome.session}: no decision ({outcome.decision_skip_reason})")
        else:
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


def _paper_market_data_settings(args) -> tuple[str, float | None, bool]:
    """Resolve the paper worker's market-data settings from the CLI/environment.

    Returns ``(provider, max_age_seconds, whole_shares)``. Each CLI flag is passed
    as ``None`` when it was not given, so the environment is consulted only then
    and an explicit flag always wins. A bad value raises :class:`RunnerError`
    rather than falling back, so a typo in a provider name or a freshness value can
    never silently change which price a paper order is sized from.
    """
    try:
        return resolve_settings(
            provider=getattr(args, "paper_market_data", None),
            max_age=getattr(args, "paper_snapshot_max_age_seconds", None),
            fractional=getattr(args, "paper_fractional_shares", None),
        )
    except MarketDataWiringError as exc:
        raise RunnerError(str(exc)) from exc


def _paper_decision_source(args) -> str:
    """Resolve the paper worker's decision source from the CLI/environment.

    ``history`` (the default) keeps the existing historical path; ``current_market``
    opts into the independent live path. A bad value raises rather than falling
    back, so a typo cannot silently switch which decisions the worker acts on.
    """
    try:
        _, _, _, source = resolve_paper_settings(
            decision_source=getattr(args, "paper_decision_source", None),
        )
    except MarketDataWiringError as exc:
        raise RunnerError(str(exc)) from exc
    return source


def _announce_paper_market_data(provider: str, max_age: float | None) -> None:
    """Print the effective price seam to stderr, so a start log names it."""
    suffix = (
        f"; snapshot max age {max_age:.0f}s"
        if provider == "alpaca_snapshot" and max_age is not None
        else ""
    )
    print(f"paper market data: {describe_provider(provider)}{suffix}", file=sys.stderr)


def _announce_paper_decision_source(source: str) -> None:
    """Print the effective decision source to stderr, so a start log names it."""
    print(f"paper decision source: {describe_decision_source(source)}", file=sys.stderr)


def _paper_execution_provider(args) -> str:
    """Resolve the paper worker's execution broker from the CLI/environment.

    ``alpaca_paper`` (the default) keeps the existing path; ``bybit_demo``
    opts into the Bybit Demo Spot broker. A bad value raises rather than
    falling back, so a typo cannot silently switch which broker the worker
    places orders with.
    """
    try:
        return resolve_execution_provider(
            getattr(args, "paper_execution_provider", None)
            or os.environ.get(EXECUTION_PROVIDER_ENV, "").strip()
            or EXECUTION_PROVIDER_ALPACA_PAPER
        )
    except MarketDataWiringError as exc:
        raise RunnerError(str(exc)) from exc


def _announce_paper_execution_provider(provider: str) -> None:
    """Print the effective execution broker to stderr, so a start log names it."""
    print(
        f"paper execution provider: {describe_execution_provider(provider)}",
        file=sys.stderr,
    )


def _serve(args) -> int:
    """Run the local dashboard. Simulation-only monitoring and control.

    The server is read-only unless ``--allow-control`` is passed. It binds
    loopback by default; a non-loopback bind requires a token
    (``TRADINGAGENTS_DASHBOARD_TOKEN`` or ``--token``), refused otherwise.

    ``--allow-paper-trading`` additionally enables the START/STOP control for the
    Alpaca **paper** worker. It is off by default, so the dashboard cannot place a
    paper order unless the operator opts in, and it never enables live trading.

    ``--paper-market-data`` selects the price seam the worker sizes from, and is
    passed through as the same factory the module-form server uses, so the two
    entrypoints cannot diverge. ``bar_open`` keeps the historical default;
    ``alpaca_snapshot`` reads a current price and never falls back to a bar open.

    ``--paper-decision-source`` selects which decisions the worker acts on.
    ``history`` (the default) keeps the existing historical path; ``current_market``
    opts into the independent live path. The two are separate objects on separate
    data, so a consumed historical history does not block the current path.
    """
    from .dashboard import DashboardAuthError, serve

    root = _root(args)
    provider, max_age, whole_shares = _paper_market_data_settings(args)
    decision_source = _paper_decision_source(args)
    execution_provider = _paper_execution_provider(args)
    _announce_paper_market_data(provider, max_age)
    _announce_paper_decision_source(decision_source)
    _announce_paper_execution_provider(execution_provider)
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
            market_data_factory=market_data_factory_for(provider, max_age_seconds=max_age),
            paper_whole_shares=whole_shares,
            paper_market_data_provider=provider,
            paper_market_data_max_age_seconds=max_age,
            paper_decision_source=decision_source,
            paper_execution_provider=execution_provider,
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


def _probe_source(source_name: str, env_name: str, value: str) -> tuple[str, str]:
    """One real, read-only request to classify a credential. Never prints the key.

    Returns ``(verdict, detail)`` where verdict is ``OK``, ``INVALID`` or
    ``UNREACHABLE``. A rejection that names the credential is INVALID; a transport
    failure is UNREACHABLE; anything else is reported with its message.
    """
    from .providers.base import (
        ProviderAuthError,
        ProviderNotConfiguredError,
        ProviderRateLimitedError,
    )

    try:
        if source_name == "alpha_vantage":
            from .providers.alpha_vantage import AlphaVantageDailySource

            AlphaVantageDailySource("SPY", api_key=value, max_stale_days=None)
        elif source_name == "finnhub":
            from .providers.finnhub import FinnhubQuoteSource

            FinnhubQuoteSource("SPY", api_key=value)
        else:
            return "SKIPPED", "no probe is defined for this source"
    except (ProviderAuthError, ProviderNotConfiguredError) as exc:
        return "INVALID", f"the vendor rejected the credential: {exc}"
    except ProviderRateLimitedError as exc:
        # The key is accepted; the plan is out of quota. Not an auth failure.
        return "OK", f"credential accepted, but the plan is throttled: {exc}"
    except Exception as exc:  # noqa: BLE001 - a probe reports, it never raises
        return "UNREACHABLE", f"{type(exc).__name__}: {exc}"
    return "OK", f"{env_name} is set and the vendor accepted the request"


def _diagnose(args) -> int:
    """Report, without revealing a secret, whether a provider credential is set.

    Answers the two questions a deploy needs: is the env var present, and would
    the outgoing request carry the ``apikey`` parameter? It never prints the key,
    any part of it, or its length — only presence and the parameter *name*.

    ``--probe`` additionally makes one real, read-only request per credential, so a
    key that is SET but REJECTED (the failure that reads "the parameter apikey is
    invalid or missing" in a deploy log) is reported as INVALID rather than left as
    an ambiguous "present". Without ``--probe`` no vendor call is made.
    """
    from .providers.alpha_vantage import (
        API_BASE_URL as AV_URL,
        API_KEY_ENV as AV_ENV,
        build_request_params as av_params,
    )
    from .providers.finnhub import API_KEY_ENV as FH_ENV

    requested = args.source or "frozen"
    probe = bool(getattr(args, "probe", False))
    print(f"requested source: {requested}")
    print(f"known sources: {', '.join(KNOWN_SOURCES)}")

    env_name = {"alpha_vantage": AV_ENV, "finnhub": FH_ENV}.get(requested)
    if env_name is None:
        print(f"{requested}: no credential required")
        return 0

    value = os.environ.get(env_name, "")
    present = bool(value.strip())
    # A three-way verdict: MISSING (unset/blank), INVALID (set but the vendor
    # rejects it), or SET (present; only a probe can promote it to a working key).
    verdict = "MISSING" if not present else "SET"
    print(f"credential env var: {env_name}")
    print(f"env var present and non-blank: {present}")
    print(f"credential status: {verdict}")
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

    if probe:
        outcome, detail = _probe_source(requested, env_name, value)
        print(f"live probe: {outcome}")
        print(f"  {detail}")
        if outcome == "INVALID":
            print(
                "  the credential is present but the vendor rejected it: replace "
                f"{env_name} with a valid key"
            )
    else:
        print("live probe: skipped (pass --probe to make one read-only request)")
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

    ``--paper-market-data`` selects the price seam, exactly as ``serve`` does, so
    the CLI and the deployed dashboard cannot diverge: ``bar_open`` is the
    historical default and ``alpaca_snapshot`` reads a current price that is
    refused when stale. ``--paper-decision-source`` selects which decisions the
    worker acts on — ``history`` (the default) or the independent ``current_market``.
    ``--dry-run`` performs preflight and reports what would happen without placing
    an order. There is no flag that reaches a live endpoint: the broker's gate
    refuses every live host, and the endpoint is not configurable here.
    """
    from .dashboard.paper_control import PaperTradingController

    root = _root(args)
    provider, max_age, whole_shares = _paper_market_data_settings(args)
    decision_source = _paper_decision_source(args)
    execution_provider = _paper_execution_provider(args)
    _announce_paper_market_data(provider, max_age)
    _announce_paper_decision_source(decision_source)
    _announce_paper_execution_provider(execution_provider)
    controller = PaperTradingController(
        root,
        enabled=True,
        order_notional=getattr(args, "paper_order_notional", 25.0) or 25.0,
        market_data_factory=market_data_factory_for(provider, max_age_seconds=max_age),
        whole_shares=whole_shares,
        market_data_provider=provider,
        market_data_max_age_seconds=max_age,
        decision_source_kind=decision_source,
        execution_provider=execution_provider,
    )
    if getattr(args, "dry_run", False):
        worker = controller._ensure_worker()  # noqa: SLF001 - the CLI owns the worker
        result = worker.preflight()
        print(
            json.dumps(
                {
                    "preflight": result,
                    "placed_order": False,
                    "market_data_provider": provider,
                    "decision_source": decision_source,
                },
                indent=2,
                default=str,
            )
        )
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


def _build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help=(
            "one or more symbols for the experiment (e.g., BTC/USDT ETH/USDT). "
            "Overrides --symbol and TRADINGAGENTS_EXPERIMENT_SYMBOLS. "
            "Required for multi-symbol crypto experiments."
        ),
    )
    parser.add_argument("--symbol", default=None, help="single symbol for backward compatibility (default SPY)")
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--fallback-provider",
        default=os.environ.get("TRADINGAGENTS_FALLBACK_PROVIDER") or None,
        help=(
            "an optional second LLM provider, tried only when the primary cannot "
            "answer (missing credential, error, or unreadable response). Recorded "
            "per decision. Default: $TRADINGAGENTS_FALLBACK_PROVIDER, else none."
        ),
    )
    parser.add_argument(
        "--fallback-model",
        default=os.environ.get("TRADINGAGENTS_FALLBACK_MODEL") or None,
        help="model for --fallback-provider (default: same as --model)",
    )
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
        "--paper-market-data",
        choices=MARKET_DATA_CHOICES,
        default=None,
        help=(
            "serve / paper-trade: how the paper worker reads a current price "
            f"(default: ${MARKET_DATA_ENV}, else bar_open, the experiment's latest "
            "bar open). 'alpaca_snapshot' reads a genuinely current price from "
            "Alpaca market data (feed=iex) and requires ALPACA_PAPER_API_KEY / "
            "ALPACA_PAPER_API_SECRET. 'binance_snapshot' reads a genuinely current "
            "CRYPTO price from Binance's public market data (no API key). There is "
            "no silent fallback: if the snapshot cannot be read or is stale, no "
            "order is placed."
        ),
    )
    parser.add_argument(
        "--paper-snapshot-max-age-seconds",
        type=float,
        default=None,
        help=(
            "serve / paper-trade: the maximum age of a snapshot price before it is "
            f"refused, in seconds (default: ${SNAPSHOT_MAX_AGE_ENV}, else "
            f"{DEFAULT_MAX_SNAPSHOT_AGE_SECONDS:.0f}). Only used by alpaca_snapshot."
        ),
    )
    parser.add_argument(
        "--paper-decision-source",
        choices=DECISION_SOURCE_CHOICES,
        default=None,
        help=(
            "serve / paper-trade: which decisions the paper worker acts on "
            f"(default: ${DECISION_SOURCE_ENV}, else history). 'history' is the "
            "existing paper experiment's recorded, historical decisions, bounded by "
            "its sessions (it reports HISTORY_EXHAUSTED once they are consumed). "
            "'current_market' is an INDEPENDENT decision taken live over current "
            "Alpaca market data, gated on Alpaca's market clock; it is not blocked by "
            "a consumed history, and needs ALPACA_PAPER_API_KEY / "
            "ALPACA_PAPER_API_SECRET and an LLM credential. "
            "'tjr' is the DETERMINISTIC TJR rule chain (no model) over current Alpaca "
            "INTRADAY 1m/5m data, gated on Alpaca's market clock and the RiskEngine; "
            "it needs ALPACA_PAPER_API_KEY / ALPACA_PAPER_API_SECRET and is best "
            "paired with --paper-market-data alpaca_snapshot. "
            "'crypto' is that same unchanged TJR rule chain over current Binance "
            "public INTRADAY 1m/5m crypto bars (no API key), gated on a 24/7 clock "
            "and the RiskEngine; pair it with --paper-market-data binance_snapshot."
        ),
    )
    parser.add_argument(
        "--paper-execution-provider",
        choices=EXECUTION_PROVIDER_CHOICES,
        default=None,
        help=(
            "serve / paper-trade: which broker the paper worker places orders with "
            f"(default: ${EXECUTION_PROVIDER_ENV}, else alpaca_paper). 'alpaca_paper' "
            "is the existing Alpaca Paper path and stays the default. 'bybit_demo' is "
            "the Bybit DEMO Spot broker (crypto Spot only, demo trading, long-only); "
            "it needs BYBIT_DEMO_API_KEY / BYBIT_DEMO_API_SECRET and never falls back "
            "to Alpaca when they are missing."
        ),
    )
    parser.add_argument(
        "--paper-fractional-shares",
        action="store_true",
        default=None,
        help=(
            "serve / paper-trade: allow a fractional quantity (default: "
            f"${FRACTIONAL_SHARES_ENV}, else fractional, the one parity constant). "
            "Needed for a high-priced symbol such as SPY under a small notional, "
            "where whole shares would floor to zero. The $25 notional cap and "
            "long-only rule are unchanged."
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
    parser.add_argument(
        "--probe",
        action="store_true",
        help=(
            "diagnose only: make one real, read-only request per credential to tell "
            "a SET-but-rejected key (INVALID) from a working one. Off by default, so "
            "a plain diagnose spends no vendor call."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
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
