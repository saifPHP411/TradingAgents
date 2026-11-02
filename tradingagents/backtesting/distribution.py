"""Outcome distribution: how much of a result is signal, and how much is one path.

A backtest returns a single equity curve and a single set of trades. That is one
realisation of a stochastic process, and it says nothing about how much the
result would have moved had the sessions fallen in a different order. This module
answers that question by resampling the result it is handed — and nothing else.

What is resampled, and why. The default metric is the series of **per-session
simple returns of the equity curve**, and it is resampled with a **moving-block
bootstrap** (:func:`bootstrap_mean_ci`). Sessions are resampled in contiguous
blocks rather than one at a time because financial returns are not independent:
volatility clusters, a drawdown spans several sessions, and a market-wide move
hits every position at once. Drawing sessions one at a time would break that
dependence and report a confidence interval narrower than the data supports. A
block keeps a run of consecutive sessions together, so the short-range structure
survives the resample.

The trade-level view is offered as ``method="trade_iid"`` and carries a warning.
Trades from one book are **not** independent observations: positions overlap in
time, they share a common market factor, and the exit of one is uncorrelated with
the entry of the next only by accident. Bootstrapping them one at a time treats
them as exchangeable, which understates the true uncertainty. It is available
because a per-trade P&L distribution is the unit a trader reasons in, and it is
honest only when the warning it emits is read. When the schema does not support a
defensible block structure, this module does the analysis the data *does* support
and says so rather than inventing one.

What this is not. It is not a second portfolio simulator: it never creates an
order, never touches a ledger, never fetches a market price, and never re-derives
a cost. It reads ``net_pnl`` and the equity curve, which are already net of every
fee the :class:`~tradingagents.execution.costs.CostModel` charged, so costs are
counted exactly once — the ledger's. It is not a prediction and not a guarantee:
a confidence interval describes the variability of *this sample under resampling*,
not the range of future returns. And the number of bootstrap iterations is not a
sample size: ten thousand resamples of thirty trades are still thirty trades.

Point-in-time. The caller is responsible for handing in-sample and out-of-sample
results separately; :func:`in_sample_vs_out_of_sample` keeps them apart and there
is deliberately no function that pools them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from tradingagents.backtesting.engine import BacktestResult

DEFAULT_ITERATIONS = 2_000
"""Bootstrap resamples. Not a sample size: see the module docstring."""

DEFAULT_CONFIDENCE = 0.95

METHOD_SESSION_BLOCK = "session_block"
METHOD_TRADE_IID = "trade_iid"

SMALL_SAMPLE_THRESHOLD = 20
"""Below this many observations, any interval is reported with a warning."""

MIN_SESSIONS_FOR_ESS = 3
"""Fewer than this and the autocorrelation-adjusted sample size is not estimated."""


class DistributionError(ValueError):
    """The inputs to a distribution analysis were empty or not finite."""


# ---------------------------------------------------------------------------
# Numeric guards
# ---------------------------------------------------------------------------


def _finite(value: float, what: str) -> float:
    """Return ``value`` as a float, refusing anything non-finite.

    NaN is the dangerous case: it passes an ``is not None`` check and then poisons
    every comparison, so it is checked explicitly rather than by a range test.
    """
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DistributionError(f"{what} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise DistributionError(f"{what} must be finite, got {value!r}")
    return number


def _finite_series(values, what: str) -> list[float]:
    return [_finite(v, f"{what}[{i}]") for i, v in enumerate(values)]


# ---------------------------------------------------------------------------
# Bootstrap primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """The bootstrap distribution of a sample mean.

    ``sample_count`` is how many real observations went in; ``iterations`` is how
    many resamples were drawn from them. The two are reported apart so neither is
    mistaken for the other.
    """

    sample_mean: float
    sample_median: float
    sample_stdev: float
    bootstrap_mean: float
    ci_low: float
    ci_high: float
    confidence: float
    probability_positive: float
    """Share of resample means strictly above zero. A probability about *this
    sample*, not a forecast of the next trade."""
    sample_count: int
    iterations: int
    block_size: int
    seed: int | None

    def to_dict(self) -> dict:
        return {
            "sample_mean": self.sample_mean,
            "sample_median": self.sample_median,
            "sample_stdev": self.sample_stdev,
            "bootstrap_mean": self.bootstrap_mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "probability_positive": self.probability_positive,
            "sample_count": self.sample_count,
            "iterations": self.iterations,
            "block_size": self.block_size,
            "seed": self.seed,
        }


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile of an already-sorted, non-empty list."""
    if not sorted_values:
        raise DistributionError("percentile of an empty distribution")
    if q <= 0.0:
        return sorted_values[0]
    if q >= 1.0:
        return sorted_values[-1]
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    variance = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _lag1_autocorrelation(values: list[float]) -> float:
    """Lag-1 autocorrelation, or 0.0 when it is not estimable."""
    n = len(values)
    if n < 3:
        return 0.0
    mu = _mean(values)
    denom = sum((v - mu) ** 2 for v in values)
    if denom <= 0.0:
        return 0.0
    numer = sum((values[i] - mu) * (values[i - 1] - mu) for i in range(1, n))
    return max(-1.0, min(1.0, numer / denom))


def effective_sample_size(values: list[float]) -> float | None:
    """Independent observations the series is worth, from its own autocorrelation.

    A serially correlated series carries less information than its length
    suggests. For a first-order autoregressive series the standard adjustment is
    ``n * (1 - rho) / (1 + rho)`` with ``rho`` the lag-1 autocorrelation, which
    collapses to ``n`` when the series is uncorrelated and to a small number when
    it is strongly persistent. It is a clearly justified equivalent to a sample
    size, computed from the data rather than assumed.

    ``None`` when there are too few points to estimate ``rho`` at all, so a caller
    cannot read a fabricated number.
    """
    n = len(values)
    if n < MIN_SESSIONS_FOR_ESS:
        return None
    rho = _lag1_autocorrelation(values)
    if rho <= 0.0:
        return float(n)
    estimate = n * (1.0 - rho) / (1.0 + rho)
    return float(max(1.0, min(float(n), estimate)))


def _resample_blocks(values: list[float], block_size: int, rng: random.Random) -> list[float]:
    """One moving-block resample of ``values``, the same length as the input.

    Blocks are contiguous runs of consecutive observations, so within-block order
    and adjacency are preserved; only the blocks are drawn independently. With
    ``block_size == 1`` this is the ordinary i.i.d. bootstrap.
    """
    n = len(values)
    if block_size <= 1:
        return [values[rng.randrange(n)] for _ in range(n)]
    block = min(block_size, n)
    starts = n - block + 1
    out: list[float] = []
    while len(out) < n:
        start = rng.randrange(starts)
        out.extend(values[start:start + block])
    return out[:n]


def bootstrap_mean_ci(
    samples,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    block_size: int = 1,
) -> BootstrapResult:
    """Moving-block bootstrap confidence interval for the mean of ``samples``.

    ``block_size`` is the number of consecutive observations drawn together.
    ``1`` is the i.i.d. bootstrap; larger values preserve short-range temporal
    dependence and are the honest choice for a serially correlated series such as
    session returns. The interval is the percentile interval of the bootstrap
    distribution of the mean, taken from the resamples themselves — not a normal
    approximation.

    The result is deterministic for a given ``seed`` and the same inputs. Every
    sample must be finite; an empty input is refused rather than returning a zero.
    """
    if iterations < 1:
        raise DistributionError(f"iterations must be at least 1, got {iterations}")
    if not 0.0 < confidence < 1.0:
        raise DistributionError(f"confidence must be between 0 and 1, got {confidence}")
    if block_size < 1:
        raise DistributionError(f"block_size must be at least 1, got {block_size}")
    if seed is not None and not isinstance(seed, int):
        raise DistributionError(f"seed must be an int or None, got {seed!r}")

    values = _finite_series(samples, "sample")
    n = len(values)
    if n == 0:
        raise DistributionError("cannot bootstrap an empty sample")

    rng = random.Random(seed)
    means = [
        _mean(_resample_blocks(values, block_size, rng)) for _ in range(iterations)
    ]
    ordered = sorted(means)
    alpha = (1.0 - confidence) / 2.0
    positive = sum(1 for m in means if m > 0.0) / float(iterations)
    return BootstrapResult(
        sample_mean=_mean(values),
        sample_median=_median(values),
        sample_stdev=_stdev(values),
        bootstrap_mean=_mean(means),
        ci_low=_percentile(ordered, alpha),
        ci_high=_percentile(ordered, 1.0 - alpha),
        confidence=confidence,
        probability_positive=positive,
        sample_count=n,
        iterations=iterations,
        block_size=min(block_size, n),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Reading a backtest result
# ---------------------------------------------------------------------------


def session_returns(result: BacktestResult) -> list[float]:
    """Simple per-session returns of the equity curve, in session order.

    ``equity[i] / equity[i-1] - 1``. A non-positive equity point cannot produce a
    meaningful return, so it is refused rather than divided by. The series keeps
    its order, which is what makes a block resample meaningful.
    """
    points = result.equity_curve
    if len(points) < 2:
        return []
    equities = [_finite(p.equity, "equity") for p in points]
    returns: list[float] = []
    for previous, current in zip(equities, equities[1:], strict=False):
        if previous <= 0.0:
            raise DistributionError(
                f"equity {previous!r} is not positive, so a session return cannot be formed"
            )
        returns.append(current / previous - 1.0)
    return returns


def trade_net_pnl(result: BacktestResult) -> list[float]:
    """Per-trade net P&L, exactly as the ledger recorded it.

    ``Trade.net_pnl`` already subtracts the round trip's fees, so this is the
    net figure and no cost is applied a second time here.
    """
    return [_finite(t.net_pnl, "trade net_pnl") for t in result.trades]


@dataclass(frozen=True)
class OutcomeDistribution:
    """A resampled view of one metric from one run.

    ``available`` is ``False`` for a run with no observations, in which case every
    statistic is ``None`` rather than a flattering zero. ``metric`` names what was
    measured and ``unit`` what it is in, so a return and a currency amount are
    never read as the same thing.
    """

    available: bool
    metric: str
    unit: str
    method: str
    sample_mean: float | None = None
    sample_median: float | None = None
    sample_stdev: float | None = None
    bootstrap_mean: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    confidence: float | None = None
    probability_positive: float | None = None
    effective_sample_size: float | None = None
    sample_count: int = 0
    iterations: int = 0
    block_size: int | None = None
    seed: int | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "metric": self.metric,
            "unit": self.unit,
            "method": self.method,
            "sample_mean": self.sample_mean,
            "sample_median": self.sample_median,
            "sample_stdev": self.sample_stdev,
            "bootstrap_mean": self.bootstrap_mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "probability_positive": self.probability_positive,
            "effective_sample_size": self.effective_sample_size,
            "sample_count": self.sample_count,
            "iterations": self.iterations,
            "block_size": self.block_size,
            "seed": self.seed,
            "warnings": list(self.warnings),
        }


def _default_block_size(n: int) -> int:
    """``round(n ** (1/3))``, the usual rule of thumb, floored at 1 and capped at n."""
    if n <= 1:
        return 1
    return max(1, min(n, round(n ** (1.0 / 3.0))))


def outcome_distribution(
    result: BacktestResult,
    *,
    method: str = METHOD_SESSION_BLOCK,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    block_size: int | None = None,
) -> OutcomeDistribution:
    """Resample one run's outcome into a distribution with a confidence interval.

    ``method="session_block"`` (the default) resamples the equity curve's
    per-session returns in contiguous blocks, preserving the temporal grouping
    that makes an i.i.d. resample invalid. ``method="trade_iid"`` resamples
    per-trade net P&L one at a time and always carries a dependence warning.

    An empty run returns ``available=False`` with every statistic ``None``. It is
    never reported as a zero, which would read as a real, flat result.
    """
    if method not in (METHOD_SESSION_BLOCK, METHOD_TRADE_IID):
        raise DistributionError(
            f"unknown method {method!r}; expected {METHOD_SESSION_BLOCK!r} or {METHOD_TRADE_IID!r}"
        )

    warnings: list[str] = []
    if method == METHOD_SESSION_BLOCK:
        samples = session_returns(result)
        metric, unit = "mean per-session return", "fraction"
        chosen_block = block_size if block_size is not None else _default_block_size(len(samples))
    else:
        samples = trade_net_pnl(result)
        metric, unit = "mean per-trade net P&L", "account currency"
        chosen_block = 1
        warnings.append(
            "trades are resampled as if independent, but trades from one book are not: "
            "positions overlap in time and share a market factor, so this interval is "
            "narrower than the data can support. Prefer method='session_block'."
        )

    if not samples:
        return OutcomeDistribution(
            available=False,
            metric=metric,
            unit=unit,
            method=method,
            confidence=confidence,
            iterations=iterations,
            block_size=chosen_block,
            seed=seed,
            warnings=warnings + [f"no observations to resample for {metric}; the result is unavailable"],
        )

    boot = bootstrap_mean_ci(
        samples,
        iterations=iterations,
        seed=seed,
        confidence=confidence,
        block_size=chosen_block,
    )
    if boot.sample_count < SMALL_SAMPLE_THRESHOLD:
        warnings.append(
            f"only {boot.sample_count} observations: the interval is wide and unstable, "
            "and should be read as indicative rather than precise"
        )
    ess = effective_sample_size(samples) if method == METHOD_SESSION_BLOCK else float(boot.sample_count)
    if method == METHOD_SESSION_BLOCK and ess is not None and ess < boot.sample_count:
        warnings.append(
            f"serial correlation reduces the {boot.sample_count} sessions to about "
            f"{ess:.1f} independent observations"
        )
    if method == METHOD_TRADE_IID:
        warnings.append(
            "the effective sample size is the trade count only under the independence "
            "assumption the warning above already disputes"
        )

    return OutcomeDistribution(
        available=True,
        metric=metric,
        unit=unit,
        method=method,
        sample_mean=boot.sample_mean,
        sample_median=boot.sample_median,
        sample_stdev=boot.sample_stdev,
        bootstrap_mean=boot.bootstrap_mean,
        ci_low=boot.ci_low,
        ci_high=boot.ci_high,
        confidence=boot.confidence,
        probability_positive=boot.probability_positive,
        effective_sample_size=ess,
        sample_count=boot.sample_count,
        iterations=boot.iterations,
        block_size=boot.block_size,
        seed=boot.seed,
        warnings=warnings,
    )


@dataclass(frozen=True)
class InOutComparison:
    """In-sample and out-of-sample distributions, held apart and never pooled.

    There is no combined field on purpose: pooling the two is exactly the mistake
    walk-forward exists to prevent, so the type makes it impossible to do by
    accident.
    """

    in_sample: OutcomeDistribution
    out_of_sample: OutcomeDistribution
    note: str = (
        "in-sample and out-of-sample results are reported separately and are not "
        "combined; a difference between them measures overfitting, not skill"
    )

    def to_dict(self) -> dict:
        return {
            "in_sample": self.in_sample.to_dict(),
            "out_of_sample": self.out_of_sample.to_dict(),
            "note": self.note,
        }


def in_sample_vs_out_of_sample(
    in_sample: BacktestResult,
    out_of_sample: BacktestResult,
    *,
    method: str = METHOD_SESSION_BLOCK,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    block_size: int | None = None,
) -> InOutComparison:
    """Summarise two runs separately, so they can be compared but not merged.

    Each side gets its own distribution, its own sample count and its own seed
    stream. Nothing here concatenates the two, and no statistic is computed across
    both, because a number that mixes them would describe neither window.
    """
    return InOutComparison(
        in_sample=outcome_distribution(
            in_sample,
            method=method,
            iterations=iterations,
            seed=seed,
            confidence=confidence,
            block_size=block_size,
        ),
        out_of_sample=outcome_distribution(
            out_of_sample,
            method=method,
            iterations=iterations,
            seed=seed,
            confidence=confidence,
            block_size=block_size,
        ),
    )


__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_ITERATIONS",
    "METHOD_SESSION_BLOCK",
    "METHOD_TRADE_IID",
    "BootstrapResult",
    "DistributionError",
    "InOutComparison",
    "OutcomeDistribution",
    "bootstrap_mean_ci",
    "effective_sample_size",
    "in_sample_vs_out_of_sample",
    "outcome_distribution",
    "session_returns",
    "trade_net_pnl",
]
