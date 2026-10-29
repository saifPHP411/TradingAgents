"""A small, phone-friendly HTTP dashboard for a paper experiment.

This is a **control and monitoring surface for the simulation**, not a trading
surface. It serves:

- ``GET /`` — a single self-contained HTML page (no external assets) sized for a
  phone;
- ``GET /healthz`` — an unauthenticated liveness probe for Railway;
- ``GET /api/summary``, ``/api/decisions``, ``/api/journal`` — read-only views of
  the persisted experiment;
- ``POST /api/run`` — *optionally* advance the simulation by N sessions.

Every ``/api`` route requires a bearer token. The ``POST /api/run`` route is
additionally gated behind ``--allow-control`` **and** a confirmation header, and
is refused while another run is in progress. There is no route that places a
broker order, and none can be added here without an explicit code change: the
only action available drives ``PaperExperiment.process_session``, which is
simulation-only.

The server uses only the standard library, so it adds no dependency to a project
whose runtime is deliberately small.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ..broker import BANNER, BrokerError
from .auth import DashboardAuthError, check_bearer, require_bindable, resolve_token
from .paper_control import PaperTradingController, PaperTradingDisabled
from .state import DashboardStateError, ExperimentView

DEFAULT_MAX_SESSIONS_PER_CALL = 50
"""The most sessions one control call may advance, so a request stays bounded."""

CONTROL_CONFIRM_HEADER = "X-Control-Confirm"

PAPER_CONFIRM_HEADER = "X-Paper-Confirm"
"""The explicit confirmation header a paper start/stop must carry.

Distinct from the run-confirm header on purpose: a stray "advance the simulation"
request must not be able to start the broker-backed worker. Starting the worker is
the one dashboard action that can lead to a broker order, so it requires its own
deliberate confirmation.
"""

ROOT_ENV = "TRADINGAGENTS_EXPERIMENT_ROOT"
"""Environment variable the module entry point reads ``--root`` from."""


class Dashboard:
    """The dashboard's configuration and shared state, one per server."""

    def __init__(
        self,
        root: str | Path,
        *,
        token: str | None = None,
        allow_control: bool = False,
        max_sessions_per_call: int = DEFAULT_MAX_SESSIONS_PER_CALL,
        stub: bool = False,
        allow_paper_trading: bool = False,
        paper_order_notional: float = 25.0,
        broker_factory=None,
    ) -> None:
        self.root = Path(root)
        self.token = resolve_token(token)
        self.allow_control = allow_control
        self.max_sessions_per_call = max_sessions_per_call
        self.stub = stub
        self._lock = threading.Lock()
        self._run_in_progress = False
        self.last_run: dict[str, Any] | None = None
        self.paper = PaperTradingController(
            self.root,
            enabled=allow_paper_trading,
            order_notional=paper_order_notional,
            stub=stub,
            broker_factory=broker_factory,
        )

    def view(self) -> ExperimentView:
        return ExperimentView(self.root)

    # -- control ----------------------------------------------------------

    def run_sessions(self, sessions: int) -> dict[str, Any]:
        """Advance the simulation by ``sessions`` sessions. Simulation-only.

        Serialised with a lock: a second call while one is running is refused
        rather than run concurrently, because two writers to one experiment
        directory would race on the state file.

        The experiment is loaded with a real decision seam built from the
        *persisted* config, so a session advanced here decides through the same
        provider a terminal ``run`` would. Without one every decision would be
        recorded as blocked — and, because a blocked session is still marked
        processed, that would silently burn the session. The provider reads its
        own credential; this method neither reads nor logs one.
        """
        if not self.allow_control:
            raise PermissionError(
                "control is disabled; start the dashboard with --allow-control to "
                "enable advancing the simulation"
            )
        if sessions < 0:
            raise ValueError("sessions must not be negative")
        bounded = min(sessions, self.max_sessions_per_call)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("a run is already in progress")
        try:
            self._run_in_progress = True
            from ..runner import build_inference, pending_sessions, process_pending

            experiment = self.view().load()
            experiment.inference = build_inference(experiment.config, stub=self.stub)
            remaining_before = len(pending_sessions(experiment))
            outcomes = process_pending(experiment, sessions=bounded)
            result = {
                "processed": [o.session for o in outcomes],
                "processed_count": len(outcomes),
                "remaining_before": remaining_before,
                "remaining_after": len(pending_sessions(experiment)),
                "bounded_to": bounded,
            }
            self.last_run = result
            return result
        finally:
            self._run_in_progress = False
            self._lock.release()


def make_handler(dashboard: Dashboard):
    """Build a request handler bound to ``dashboard``."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "TradingAgentsDashboard/1.0"

        # -- helpers --------------------------------------------------

        def _send_json(self, code: int, payload: Any) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, code: int, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _authed(self) -> bool:
            if check_bearer(self.headers.get("Authorization"), dashboard.token):
                return True
            self._send_json(401, {"error": "unauthorized"})
            return False

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            # Default logging writes the request line, which can include a query
            # string; the token is only ever in a header, but keep the log terse.
            return

        # -- routes ---------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            path = urlsplit(self.path).path
            if path == "/healthz":
                self._send_json(200, {"status": "ok"})
                return
            if path == "/":
                self._send_html(200, INDEX_HTML)
                return
            if not self._authed():
                return
            try:
                if path == "/api/summary":
                    self._send_json(200, dashboard.view().summary())
                elif path == "/api/decisions":
                    self._send_json(200, {"decisions": dashboard.view().decisions()})
                elif path == "/api/journal":
                    query = parse_qs(urlsplit(self.path).query)
                    limit = int((query.get("limit") or ["50"])[0])
                    self._send_json(
                        200, {"entries": dashboard.view().journal_tail(limit=limit)}
                    )
                elif path == "/api/control/status":
                    self._send_json(
                        200,
                        {
                            "control_enabled": dashboard.allow_control,
                            "run_in_progress": dashboard._run_in_progress,
                            "max_sessions_per_call": dashboard.max_sessions_per_call,
                            "last_run": dashboard.last_run,
                        },
                    )
                elif path == "/api/paper/status":
                    self._send_json(200, dashboard.paper.status())
                else:
                    self._send_json(404, {"error": "not found"})
            except DashboardStateError as exc:
                self._send_json(404, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - surface, never crash the server
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib name
            path = urlsplit(self.path).path
            if path in ("/api/paper/start", "/api/paper/stop"):
                self._handle_paper_control(path)
                return
            if path != "/api/run":
                self._send_json(404, {"error": "not found"})
                return
            if not self._authed():
                return
            # A deliberate, explicit header so a stray form post or a CSRF-style
            # request cannot trigger a run.
            if (self.headers.get(CONTROL_CONFIRM_HEADER) or "").lower() != "yes":
                self._send_json(
                    403,
                    {
                        "error": "confirmation header required",
                        "hint": f"send {CONTROL_CONFIRM_HEADER}: yes",
                    },
                )
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
                sessions = int(payload.get("sessions", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                self._send_json(400, {"error": "body must be JSON with an integer 'sessions'"})
                return
            try:
                result = dashboard.run_sessions(sessions)
                self._send_json(200, result)
            except PermissionError as exc:
                self._send_json(403, {"error": str(exc)})
            except RuntimeError as exc:
                self._send_json(409, {"error": str(exc)})
            except DashboardStateError as exc:
                self._send_json(404, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

        # -- paper trading control ------------------------------------

        def _handle_paper_control(self, path: str) -> None:
            """Start or stop the Alpaca **paper** worker. Auth + confirmation required.

            The endpoint cannot change the broker endpoint: the worker builds its
            broker from the server's own configuration, and the broker's gate
            refuses any non-paper host. No credential is accepted from the body or
            any header — the broker reads its own environment variables.
            """
            if not self._authed():
                return
            if (self.headers.get(PAPER_CONFIRM_HEADER) or "").lower() != "yes":
                self._send_json(
                    403,
                    {
                        "error": "confirmation header required",
                        "hint": f"send {PAPER_CONFIRM_HEADER}: yes",
                    },
                )
                return
            try:
                if path == "/api/paper/start":
                    result = dashboard.paper.start()
                    self._send_json(200, {"started": True, **result})
                else:
                    result = dashboard.paper.stop()
                    self._send_json(200, {"stopped": True, **result})
            except PaperTradingDisabled as exc:
                self._send_json(403, {"error": str(exc)})
            except BrokerError as exc:
                # A preflight/broker failure is a safe, secret-free 502; the
                # worker has recorded ERROR and is not running.
                self._send_json(502, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main(argv: list[str] | None = None) -> int:
    """Run the dashboard as ``python -m ...dashboard.server``.

    A thin wrapper over :func:`serve` for a platform (Railway, Docker) that runs
    a module directly rather than the ``serve`` subcommand. ``--root`` falls back
    to ``TRADINGAGENTS_EXPERIMENT_ROOT`` and the port to ``$PORT``, so the start
    command can be as short as ``python -m ...dashboard.server``.
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.paper_experiment.dashboard.server",
        description=(
            "Paper-experiment dashboard. Simulation only: no broker order is ever "
            "placed. Read-only unless --allow-control is passed."
        ),
    )
    parser.add_argument(
        "--root",
        default=os.environ.get(ROOT_ENV) or None,
        help=f"experiment directory (default: ${ROOT_ENV})",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("TRADINGAGENTS_DASHBOARD_HOST", "127.0.0.1"),
        help="interface to bind (default: $TRADINGAGENTS_DASHBOARD_HOST, else loopback)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="port to listen on (default: $PORT, else 8000)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "dashboard token (default: $TRADINGAGENTS_DASHBOARD_TOKEN). Required "
            "for a non-loopback bind; never logged."
        ),
    )
    parser.add_argument(
        "--allow-control",
        action="store_true",
        help="enable the advance-the-simulation action (off by default)",
    )
    parser.add_argument(
        "--allow-paper-trading",
        action="store_true",
        help=(
            "enable the START/STOP control for the Alpaca PAPER worker (off by "
            "default). Paper only: the broker endpoint is fixed to Alpaca's paper "
            "host and every live endpoint is refused. Requires "
            "ALPACA_PAPER_API_KEY / ALPACA_PAPER_API_SECRET in the environment."
        ),
    )
    parser.add_argument(
        "--paper-order-notional",
        type=float,
        default=float(os.environ.get("TRADINGAGENTS_PAPER_ORDER_NOTIONAL", "25") or 25),
        help="target cash per paper order (default: $TRADINGAGENTS_PAPER_ORDER_NOTIONAL, else 25)",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "use the non-genuine stub decision source (offline demonstration); "
            "decisions are labelled synthetic_fixture"
        ),
    )
    parser.add_argument(
        "--auto-advance",
        type=int,
        default=int(os.environ.get("TRADINGAGENTS_AUTO_ADVANCE_SESSIONS", "0") or 0),
        help=(
            "advance this many sessions every interval, in-process (default: "
            "$TRADINGAGENTS_AUTO_ADVANCE_SESSIONS, else 0 = monitor only). "
            "Requires --allow-control."
        ),
    )
    parser.add_argument(
        "--auto-interval",
        type=float,
        default=float(os.environ.get("TRADINGAGENTS_AUTO_ADVANCE_INTERVAL", "60") or 60),
        help="seconds between auto-advance ticks (default: 60)",
    )
    args = parser.parse_args(argv)

    if not args.root:
        parser.error(f"an experiment directory is required: pass --root or set {ROOT_ENV}")
    port = args.port if args.port is not None else int(os.environ.get("PORT", "8000"))
    try:
        serve(
            args.root,
            host=args.host,
            port=port,
            token=args.token,
            allow_control=args.allow_control,
            stub=args.stub,
            auto_advance=args.auto_advance,
            auto_interval=args.auto_interval,
            allow_paper_trading=args.allow_paper_trading,
            paper_order_notional=args.paper_order_notional,
        )
    except DashboardAuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def serve(
    root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    token: str | None = None,
    allow_control: bool = False,
    max_sessions_per_call: int = DEFAULT_MAX_SESSIONS_PER_CALL,
    stub: bool = False,
    auto_advance: int = 0,
    auto_interval: float = 60.0,
    allow_paper_trading: bool = False,
    paper_order_notional: float = 25.0,
    broker_factory=None,
) -> None:
    """Run the dashboard until interrupted.

    Refuses to bind a non-loopback host with no token, so the control surface is
    never served unauthenticated on a public interface.

    ``auto_advance`` turns the dashboard into the *engine* as well as the monitor:
    when it is a positive N, a background thread advances N unprocessed sessions
    every ``auto_interval`` seconds, through the same locked ``run_sessions`` the
    Run button uses. One process, one writer, so it cannot race the state file. It
    is off by default (``0``), which leaves the server a pure monitor.

    ``allow_paper_trading`` enables the START/STOP control for the Alpaca paper
    worker. It is off by default, so the dashboard cannot place a paper order
    unless the operator opts in; it never enables live trading.
    """
    resolved = resolve_token(token)
    require_bindable(host, resolved)
    dashboard = Dashboard(
        root,
        token=resolved,
        allow_control=allow_control,
        max_sessions_per_call=max_sessions_per_call,
        stub=stub,
        allow_paper_trading=allow_paper_trading,
        paper_order_notional=paper_order_notional,
        broker_factory=broker_factory,
    )
    httpd = ThreadingHTTPServer((host, port), make_handler(dashboard))
    httpd.daemon_threads = True
    print(f"dashboard on http://{host}:{port}  (root: {Path(root).resolve()})")
    print(f"  auth: {'token required' if resolved else 'loopback only, no token'}")
    print(f"  control: {'enabled' if allow_control else 'read-only'}")
    print(
        "  paper trading: "
        + (
            f"enabled ({BANNER}, order notional {paper_order_notional:g})"
            if allow_paper_trading
            else "disabled (pass --allow-paper-trading to enable)"
        )
    )
    if auto_advance:
        if not allow_control:
            print("  auto-advance: refused (needs --allow-control)", flush=True)
        else:
            _start_auto_advance(dashboard, auto_advance, auto_interval)
            print(
                f"  auto-advance: {auto_advance} session(s) every {auto_interval:g}s",
                flush=True,
            )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the paper worker's loop before the process exits, so a shutdown
        # never leaves a cycle mid-flight and never places a new order. This is
        # best-effort: a stop failure must not mask the server's own shutdown.
        try:
            dashboard.paper.shutdown()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            print(f"  paper trading: stop on shutdown failed: {type(exc).__name__}", flush=True)
        httpd.server_close()


def _start_auto_advance(dashboard: Dashboard, sessions: int, interval: float) -> threading.Thread:
    """Advance the simulation on a timer, in-process, behind the control lock.

    A daemon thread so it never keeps the process alive. Each tick logs one line,
    so an operator can see from the platform's logs that sessions are progressing.
    A tick that finds nothing to do, or that loses the lock to a manual Run, says
    so rather than failing silently.
    """

    def loop() -> None:
        while True:
            time.sleep(interval)
            try:
                result = dashboard.run_sessions(sessions)
            except PermissionError:
                print("[auto] control is disabled; stopping auto-advance", flush=True)
                return
            except RuntimeError:
                print("[auto] a run is already in progress; skipping this tick", flush=True)
                continue
            except Exception as exc:  # noqa: BLE001 - a tick failure must not kill the thread
                print(f"[auto] tick failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            done = result["processed"]
            if not done:
                print(
                    "[auto] no unprocessed sessions remain; nothing to advance",
                    flush=True,
                )
                continue
            print(
                f"[auto] advanced {result['processed_count']} session(s): "
                f"{done[0]}..{done[-1]}; {result['remaining_after']} remaining",
                flush=True,
            )

    thread = threading.Thread(target=loop, name="auto-advance", daemon=True)
    thread.start()
    return thread


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="dark light">
<title>TradingAgents · Paper</title>
<style>
  :root{
    --bg:#0b0f14; --card:#141b23; --line:#232d38; --fg:#e7edf3; --mut:#8ea0b3;
    --acc:#4ea1ff; --ok:#3ddc84; --warn:#ffcc4d; --bad:#ff6b6b; --radius:14px;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font:16px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    padding:env(safe-area-inset-top) 12px calc(env(safe-area-inset-bottom) + 12px)}
  header{position:sticky;top:0;background:linear-gradient(180deg,var(--bg),var(--bg) 70%,transparent);
    padding:14px 2px 8px;z-index:5}
  h1{font-size:18px;margin:0 0 2px}
  .sub{color:var(--mut);font-size:12.5px;word-break:break-all}
  .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .grow{flex:1}
  .badge{display:inline-block;padding:3px 9px;border-radius:999px;font-size:12px;font-weight:600}
  .b-ok{background:rgba(61,220,132,.15);color:var(--ok)}
  .b-warn{background:rgba(255,204,77,.15);color:var(--warn)}
  .b-bad{background:rgba(255,107,107,.15);color:var(--bad)}
  .b-mut{background:rgba(142,160,179,.15);color:var(--mut)}
  nav{display:flex;gap:6px;overflow:auto;margin:10px 0;position:sticky;top:58px;z-index:4;
    background:var(--bg);padding:6px 0}
  nav button{flex:0 0 auto;background:var(--card);color:var(--mut);border:1px solid var(--line);
    padding:8px 14px;border-radius:999px;font-size:14px;font-weight:600}
  nav button.on{color:var(--fg);border-color:var(--acc);background:rgba(78,161,255,.12)}
  section{display:none}
  section.on{display:block}
  .card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
    padding:12px 14px;margin:10px 0}
  .card h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);
    margin:0 0 10px}
  .kv{display:grid;grid-template-columns:1fr auto;gap:6px 12px;font-size:15px}
  .kv .k{color:var(--mut)}
  .kv .v{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
  .pill{display:inline-block;padding:2px 8px;border-radius:8px;font-size:12px;font-weight:600}
  .s-pending{background:rgba(142,160,179,.18);color:var(--mut)}
  .s-executed{background:rgba(78,161,255,.18);color:var(--acc)}
  .s-evaluated{background:rgba(61,220,132,.18);color:var(--ok)}
  .s-blocked{background:rgba(255,107,107,.18);color:var(--bad)}
  .s-unevaluable{background:rgba(255,204,77,.18);color:var(--warn)}
  ul.clean{list-style:none;margin:0;padding:0}
  ul.clean li{border-bottom:1px solid var(--line);padding:9px 0;font-size:14px}
  ul.clean li:last-child{border-bottom:0}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:var(--mut);
    word-break:break-all}
  .err{background:rgba(255,107,107,.12);border:1px solid rgba(255,107,107,.4);
    border-radius:var(--radius);padding:10px 12px;margin:10px 0;font-size:14px;display:none}
  button.act{background:var(--acc);color:#06121f;border:0;padding:11px 16px;border-radius:11px;
    font-size:15px;font-weight:700}
  button.act[disabled]{opacity:.45}
  input{background:#0e141b;color:var(--fg);border:1px solid var(--line);border-radius:10px;
    padding:11px 12px;font-size:16px;width:100%}
  .muted{color:var(--mut);font-size:12.5px}
  .hide{display:none}
  .livewarn{background:rgba(255,107,107,.14);border:1px solid rgba(255,107,107,.5);
    color:#ffb3b3;border-radius:var(--radius);padding:9px 12px;margin:10px 0;font-size:13px;
    font-weight:700;letter-spacing:.04em;text-align:center}
  .state-RUNNING{background:rgba(61,220,132,.18);color:var(--ok)}
  .state-STARTING,.state-STOPPING{background:rgba(255,204,77,.18);color:var(--warn)}
  .state-STOPPED,.state-DISABLED{background:rgba(142,160,179,.18);color:var(--mut)}
  .state-ERROR{background:rgba(255,107,107,.18);color:var(--bad)}
  button.stop{background:var(--bad);color:#1a0505}
</style>
</head>
<body>
<header>
  <div class="row">
    <div class="grow">
      <h1>TradingAgents · Paper</h1>
      <div class="sub" id="expid">loading…</div>
    </div>
    <span id="status" class="badge b-mut">…</span>
  </div>
</header>

<div id="auth" class="card">
  <h2>Access token</h2>
  <input id="token" type="password" placeholder="paste dashboard token" autocomplete="off">
  <div class="row" style="margin-top:10px">
    <button class="act" onclick="saveToken()">Connect</button>
    <span class="muted">stored only in this browser</span>
  </div>
</div>

<div id="err" class="err"></div>

<div id="livewarn" class="livewarn">ALPACA PAPER — NOT LIVE</div>

<nav id="nav" class="hide">
  <button data-t="overview" class="on">Overview</button>
  <button data-t="book">Book</button>
  <button data-t="decisions">Decisions</button>
  <button data-t="journal">Journal</button>
  <button data-t="paper">Paper trading</button>
  <button data-t="control">Control</button>
</nav>

<main>
  <section id="overview" class="on">
    <div class="card"><h2>Readiness</h2><div id="readiness"></div></div>
    <div class="card"><h2>Counts</h2><div id="counts" class="kv"></div></div>
    <div class="card"><h2>Performance</h2><div id="perf" class="kv"></div></div>
    <div class="card"><h2>Data source</h2><div id="data" class="kv"></div></div>
  </section>

  <section id="book">
    <div class="card"><h2>Book</h2><div id="bookkv" class="kv"></div></div>
    <div class="card"><h2>Positions</h2><div id="positions"></div></div>
    <div class="card"><h2>Resting orders</h2><div id="resting"></div></div>
  </section>

  <section id="decisions">
    <div class="card"><h2>Decisions</h2><ul class="clean" id="declist"></ul></div>
  </section>

  <section id="journal">
    <div class="card"><h2>Journal (latest)</h2><ul class="clean" id="jlist"></ul></div>
  </section>

  <section id="paper">
    <div class="card">
      <h2>Alpaca paper worker</h2>
      <div class="livewarn" style="margin-top:0">ALPACA PAPER — NOT LIVE</div>
      <div class="row" style="margin-top:6px">
        <span id="pstate" class="pill state-STOPPED">…</span>
        <span class="grow"></span>
        <button class="act" id="pstart" onclick="doPaperStart()">START PAPER TRADING</button>
        <button class="act stop" id="pstop" onclick="doPaperStop()">STOP PAPER TRADING</button>
      </div>
      <p class="muted" id="pnote" style="margin-bottom:0"></p>
    </div>
    <div class="card"><h2>Account</h2><div id="paccount" class="kv"></div></div>
    <div class="card"><h2>Positions</h2><div id="ppositions"></div></div>
    <div class="card"><h2>Orders</h2><div id="porders"></div></div>
    <div class="card"><h2>Last cycle</h2><div id="pcycle" class="mono">—</div></div>
    <div class="card"><h2>Risk limits</h2><div id="prisk" class="kv"></div></div>
  </section>

  <section id="control">
    <div class="card">
      <h2>Advance simulation</h2>
      <p class="muted" id="ctlstate"></p>
      <div class="row">
        <input id="nsessions" type="number" min="0" value="10" style="max-width:120px">
        <button class="act" id="runbtn" onclick="doRun()">Run sessions</button>
      </div>
      <p class="muted" style="margin-bottom:0">Simulation only. No broker order is ever placed.</p>
    </div>
    <div class="card"><h2>Last run</h2><div id="lastrun" class="mono">—</div></div>
  </section>
</main>

<script>
const $ = (id) => document.getElementById(id);
let TOKEN = localStorage.getItem("ta_token") || "";
let timer = null;

function showErr(msg){
  const e = $("err");
  if(!msg){ e.style.display="none"; return; }
  e.textContent = msg; e.style.display="block";
}
function saveToken(){
  TOKEN = $("token").value.trim();
  localStorage.setItem("ta_token", TOKEN);
  boot();
}
async function api(path, opts){
  const o = Object.assign({headers:{}}, opts||{});
  o.headers["Authorization"] = "Bearer " + TOKEN;
  const r = await fetch(path, o);
  if(r.status === 401){ throw new Error("unauthorized — check the token"); }
  if(!r.ok){
    let m = "HTTP " + r.status;
    try{ const j = await r.json(); if(j.error) m = j.error; }catch(e){}
    throw new Error(m);
  }
  return r.json();
}
function badge(status){
  const cls = /READY/i.test(status) ? "b-ok" : /BLOCKED/i.test(status) ? "b-bad" : "b-warn";
  return `<span class="badge ${cls}">${status}</span>`;
}
function kv(obj, fmt){
  return Object.entries(obj).map(([k,v]) =>
    `<div class="k">${k}</div><div class="v">${fmt ? fmt(k,v) : (v===null?"N/A":v)}</div>`
  ).join("");
}
const money = (v) => v===null||v===undefined ? "N/A" : Number(v).toLocaleString(undefined,{maximumFractionDigits:2});
const pct = (v) => v===null||v===undefined ? "N/A" : (Number(v)*100).toFixed(2)+"%";

async function refresh(){
  try{
    const s = await api("/api/summary");
    showErr("");
    $("auth").classList.add("hide");
    $("nav").classList.remove("hide");
    $("expid").textContent = s.experiment_id + "  ·  " + s.strategy_id;
    $("status").textContent = s.readiness.status;
    $("status").className = "badge " + (/READY/i.test(s.readiness.status)?"b-ok":
      /BLOCKED/i.test(s.readiness.status)?"b-bad":"b-warn");

    $("readiness").innerHTML = badge(s.readiness.status) +
      (s.readiness.blockers.length
        ? '<ul class="clean">'+s.readiness.blockers.map(b=>`<li>${b}</li>`).join("")+"</ul>"
        : '<p class="muted">No blockers.</p>');

    $("counts").innerHTML = kv({
      "Sessions processed": s.counts.sessions_processed,
      "Decisions": s.counts.decisions_total,
      "Genuine inferences": s.counts.genuine_inferences,
      "Settled (executed)": s.counts.executed,
      "Evaluated": s.counts.evaluated,
      "Blocked": s.counts.blocked,
      "Pending": s.counts.pending,
      "Orders submitted": s.orders.submitted,
      "Filled orders": s.counts.filled_orders,
      "Without an order": s.orders.without_order,
    });
    $("perf").innerHTML = kv({
      "Total return": pct(s.performance.total_return),
      "Max drawdown": pct(s.performance.max_drawdown),
      "Sharpe": s.performance.sharpe_ratio===null?"N/A":s.performance.sharpe_ratio.toFixed(2),
      "Closed outcomes": s.performance.sample.closed_outcomes,
      "Min for metrics": s.performance.min_trades_for_metrics,
      "Sample limited": s.performance.sample_limited ? "yes" : "no",
    });
    $("data").innerHTML = kv({
      "Source": s.data.source_name,
      "Kind": s.data.source_kind,
      "Current market": s.data.is_current_market ? "yes" : "no",
      "Dataset": (s.data.dataset_start||"?")+" → "+(s.data.dataset_end||"?"),
      "Sessions available": s.data.sessions_available,
      "Last processed": s.last_session_processed||"—",
    }) + `<div class="k" style="grid-column:1/3;color:var(--mut);font-size:12.5px;margin-top:6px">${s.data.note}</div>`;

    $("bookkv").innerHTML = kv({
      "Initial cash": money(s.book.initial_cash),
      "Cash": money(s.book.cash),
      "Market value": money(s.book.market_value),
      "Equity": money(s.book.equity),
      "Realized P&L": money(s.book.realized_pnl),
      "Unrealized P&L": money(s.book.unrealized_pnl),
      "Total fees": money(s.book.total_fees),
      "Resting orders": s.book.resting_orders,
    });
    const pos = Object.entries(s.book.positions);
    $("positions").innerHTML = pos.length
      ? '<ul class="clean">'+pos.map(([sym,p])=>`<li><b>${sym}</b> · qty ${p.quantity} @ ${money(p.average_entry_price)}
          <div class="mono">mv ${money(p.market_value)} · uPnL ${money(p.unrealized_pnl)}</div></li>`).join("")+"</ul>"
      : '<p class="muted">No open positions.</p>';
    $("resting").innerHTML = s.resting_orders.length
      ? '<ul class="clean">'+s.resting_orders.map(o=>`<li><b>${o.symbol}</b> ${o.side} ${o.order_type}
          <span class="pill s-${(o.status||"").toLowerCase()}">${o.status}</span>
          <div class="mono">qty ${o.quantity} · filled ${o.filled_quantity} · remaining ${o.remaining_quantity}
          · limit ${money(o.limit_price)} · ${o.time_in_force}${o.expires_session?" → "+o.expires_session:""}</div></li>`).join("")+"</ul>"
      : '<p class="muted">No resting orders.</p>';
  }catch(e){
    if(/unauthorized/i.test(e.message)){
      $("auth").classList.remove("hide");
      $("nav").classList.add("hide");
    }
    showErr(e.message);
  }
}

async function refreshDecisions(){
  try{
    const d = await api("/api/decisions");
    $("declist").innerHTML = d.decisions.length ? d.decisions.map(x=>`<li>
      <b>${x.symbol}</b> ${x.decision_session}
      <span class="pill s-${x.state}">${x.state}</span>
      <div class="mono">src ${x.source} · exec ${x.execution_session} · ${x.order_status||"—"}
      · filled ${x.filled_quantity}${x.average_fill_price?" @ "+money(x.average_fill_price):""}
      · fees ${money(x.fees)}</div>
      ${x.blocked_reason?`<div class="mono" style="color:var(--bad)">${x.blocked_reason}</div>`:""}
      ${x.evaluation?`<div class="mono">net ${money(x.evaluation.net_pnl)}</div>`:""}
    </li>`).join("") : '<li class="muted">No decisions yet.</li>';
  }catch(e){ showErr(e.message); }
}
async function refreshJournal(){
  try{
    const j = await api("/api/journal?limit=50");
    $("jlist").innerHTML = j.entries.length ? j.entries.map(x=>`<li>
      <b>#${x.seq}</b> ${x.kind} <span class="muted">${x.timestamp}</span>
      <div class="mono">${x.event_id}</div></li>`).join("") : '<li class="muted">Journal empty.</li>';
  }catch(e){ showErr(e.message); }
}
async function refreshControl(){
  try{
    const c = await api("/api/control/status");
    $("ctlstate").textContent = c.control_enabled
      ? `Control is enabled. Max ${c.max_sessions_per_call} sessions per call.`
      : "Control is disabled (read-only dashboard). Restart with --allow-control to enable.";
    $("runbtn").disabled = !c.control_enabled || c.run_in_progress;
    $("lastrun").textContent = c.last_run ? JSON.stringify(c.last_run, null, 2) : "—";
  }catch(e){ showErr(e.message); }
}
async function doRun(){
  const n = parseInt($("nsessions").value || "0", 10);
  $("runbtn").disabled = true;
  try{
    const r = await api("/api/run", {
      method:"POST",
      headers:{"Content-Type":"application/json", "X-Control-Confirm":"yes"},
      body: JSON.stringify({sessions:n}),
    });
    $("lastrun").textContent = JSON.stringify(r, null, 2);
    await refresh();
  }catch(e){ showErr(e.message); }
  finally{ $("runbtn").disabled = false; }
}

function paperState(status){
  const known = ["STOPPED","STARTING","RUNNING","STOPPING","ERROR","DISABLED"];
  const s = known.includes(status) ? status : "STOPPED";
  const el = $("pstate");
  el.textContent = status || "STOPPED";
  el.className = "pill state-" + s;
}
async function refreshPaper(){
  try{
    const p = await api("/api/paper/status");
    paperState(p.status);
    $("pnote").textContent = p.enabled
      ? `Mode ${p.mode} · endpoint ${p.endpoint} · paper-only ${p.paper_endpoint_only?"yes":"NO"}`
      : (p.reason || "paper trading is disabled on this dashboard");
    $("pstart").disabled = !p.enabled || p.status==="RUNNING" || p.status==="STARTING";
    $("pstop").disabled = !p.enabled || p.status==="STOPPED" || p.status==="STOPPING";
    const a = p.account || {};
    $("paccount").innerHTML = kv({
      "Status": a.status || "—",
      "Cash": money(a.cash),
      "Equity": money(a.equity),
      "Buying power": money(a.buying_power),
      "Currency": a.currency || "—",
      "Restrictions": (a.restrictions && a.restrictions.length) ? a.restrictions.join(", ") : "none reported",
      "Positions read": p.positions_at || "never",
    });
    const pos = Object.entries(p.positions || {});
    $("ppositions").innerHTML = pos.length
      ? '<ul class="clean">'+pos.map(([sym,x])=>`<li><b>${sym}</b> · qty ${x.quantity}
          <div class="mono">mv ${money(x.market_value)} · avg ${money(x.average_entry_price)}</div></li>`).join("")+"</ul>"
      : '<p class="muted">No positions reported.</p>';
    const orders = Object.entries(p.orders || {});
    $("porders").innerHTML = orders.length
      ? '<ul class="clean">'+orders.map(([cid,o])=>`<li><b>${o.symbol}</b> ${o.side} ${o.quantity}
          <span class="pill">${o.status}</span>
          <div class="mono">${cid}<br>filled ${o.filled_quantity}${o.filled_avg_price?" @ "+money(o.filled_avg_price):""}
          · broker ${o.broker_order_id||"—"}</div></li>`).join("")+"</ul>"
      : '<p class="muted">No orders.</p>';
    $("pcycle").textContent = p.last_cycle ? JSON.stringify(p.last_cycle, null, 2)
      : (p.last_error ? ("error: "+p.last_error) : "—");
    $("prisk").innerHTML = kv(p.risk_limits || {});
  }catch(e){ showErr(e.message); }
}
async function doPaperStart(){
  $("pstart").disabled = true;
  try{
    await api("/api/paper/start", {
      method:"POST", headers:{"X-Paper-Confirm":"yes"},
    });
    await refreshPaper();
  }catch(e){ showErr(e.message); }
  finally{ refreshPaper(); }
}
async function doPaperStop(){
  $("pstop").disabled = true;
  try{
    await api("/api/paper/stop", {
      method:"POST", headers:{"X-Paper-Confirm":"yes"},
    });
    await refreshPaper();
  }catch(e){ showErr(e.message); }
  finally{ refreshPaper(); }
}

document.querySelectorAll("nav button").forEach(b=>{
  b.onclick = () => {
    document.querySelectorAll("nav button").forEach(x=>x.classList.toggle("on", x===b));
    document.querySelectorAll("section").forEach(s=>s.classList.toggle("on", s.id===b.dataset.t));
    if(b.dataset.t==="decisions") refreshDecisions();
    if(b.dataset.t==="journal") refreshJournal();
    if(b.dataset.t==="paper") refreshPaper();
    if(b.dataset.t==="control") refreshControl();
  };
});

function boot(){
  if(!TOKEN){ $("auth").classList.remove("hide"); return; }
  refresh();
  if(timer) clearInterval(timer);
  timer = setInterval(refresh, 15000);
}
boot();
</script>
</body>
</html>
"""


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
