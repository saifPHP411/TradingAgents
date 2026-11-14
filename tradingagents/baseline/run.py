"""Assemble the official baseline artifact. Run offline, reproducibly.

    python -m tradingagents.baseline.run            # print + write the artifact
    python -m tradingagents.baseline.run --out PATH # write to a specific path

The artifact is JSON: protocol, data provenance, strategy id/version, seed,
execution assumptions, per-window metrics, the independent cross-check, regime and
concentration analyses, cost sensitivity, the walk-forward result, and metadata.
A future strategy is compared against this file.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

from tradingagents import __version__ as package_version

from . import harness
from .data import load_feed
from .protocol import PROTOCOL_VERSION, BaselineProtocol
from .reference import SmaCrossover

ARTIFACT_SCHEMA_VERSION = "1.0.0"


def _regime_and_concentration(window, feed, protocol):
    labels = harness.regime_labels(feed, protocol, list(window.result.sessions))
    equity = window.result.equity_series()
    regimes = harness.regime_breakdown(equity, labels)
    net_pnl = window.metrics.final_equity - window.metrics.initial_capital
    conc = harness.concentration(window.result.trades, net_pnl)
    return regimes, conc


def build_baseline(protocol: BaselineProtocol | None = None, strategy=None) -> dict:
    """Run the full baseline and return the artifact dict. Deterministic."""
    protocol = protocol or BaselineProtocol()
    feed = load_feed(protocol)
    strategy = strategy or SmaCrossover(position_size=protocol.position_size)

    splits = harness.run_split(feed, protocol, strategy)
    walkforward = harness.run_walkforward(feed, protocol, strategy)
    cost = harness.cost_sensitivity(feed, protocol, strategy, label="test")

    regimes, conc = _regime_and_concentration(splits["test"], feed, protocol)
    dev_regimes, _ = _regime_and_concentration(splits["development"], feed, protocol)

    windows = {label: harness.summarize_window(w) for label, w in splits.items()}
    cross_checks = {
        label: [c.to_dict() for c in w.checks] for label, w in splits.items()
    }

    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "metadata": {
            "package_version": package_version,
            "protocol_version": PROTOCOL_VERSION,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "protocol": protocol.to_dict(),
        "strategy": {
            "id": strategy.name,
            "version": "1.0.0",
            "kind": "deterministic_reference",
            "description": (
                "Long-only SMA(20) above SMA(100) trend filter, entered at "
                "position_size of equity on the next session's open, with a fixed "
                "holding-period exit and a 20% protective stop. Conventional "
                "parameters, chosen a priori; not tuned on this data."
            ),
            "fast": strategy.fast,
            "slow": strategy.slow,
            "stop_loss_pct": strategy.stop_loss_pct,
        },
        "benchmark": {
            "symbol": protocol.benchmark,
            "kind": "buy_and_hold",
        },
        "results": windows,
        "independent_cross_check": cross_checks,
        "cross_check_passes": all(
            all(c["agrees"] for c in checks) for checks in cross_checks.values()
        ),
        "regime_analysis": {"test": regimes, "development": dev_regimes},
        "concentration": conc,
        "cost_sensitivity": cost,
        "walk_forward": walkforward.to_dict(),
    }
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="path for the JSON artifact")
    args = parser.parse_args()

    artifact = build_baseline()
    text = json.dumps(artifact, indent=2, default=str)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)

    test = artifact["results"]["test"]
    print(
        f"\ntest: return {test['total_return']:+.2%}  "
        f"benchmark {test['benchmark_return']:+.2%}  "
        f"maxDD {test['max_drawdown']:.2%}  trades {test['num_trades']}  "
        f"cross-check {'PASS' if artifact['cross_check_passes'] else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
