"""The real-inference seam for the paper experiment, and the offline stub.

The experiment takes an ``InferenceFn``: a callable that returns ``(rating,
reasoning)`` for a symbol, session and the point-in-time price window, or raises
:class:`~.experiment.InferenceUnavailable` when no genuine call is possible. Two
implementations live here:

- :class:`ProviderInference` makes **one** bounded real request per session,
  through the same provider client the rest of the system uses, with
  ``max_retries=0``. It checks the credential *before* the request and raises
  :class:`InferenceUnavailable` when there is none, so a missing key records a
  blocked decision rather than a fabricated one. It never places an order.
- :class:`StubInference` returns a fixed decision and declares ``genuine = False``,
  so the record is labelled ``synthetic_fixture`` and cannot be counted as a real
  inference. It exists so the whole cycle can be driven offline in tests.

The request carries the session-bounded price window (the same window the
decision is later anchored to), because asking a model for a directional call
while withholding the data is a question it can only answer with ``Hold``. The
window is rendered oldest-first, capped at ``WINDOW_BARS`` trailing bars, and it
can never contain a bar after the decision session: the frame comes from the
source's point-in-time ``window`` slice, and a non-finite value is refused rather
than rendered into the prompt.

Neither implementation prints, logs or stores a credential.
:class:`ProviderInference` reads the variable *name* only.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from tradingagents.llm_clients.api_key_env import get_api_key_env

from .experiment import InferenceUnavailable
from .indicators import IndicatorError, render_indicator_context
from .protocol import PAPER_INFERENCE_PROMPT_VERSION

__all__ = [
    "PAPER_INFERENCE_PROMPT_VERSION",
    "WINDOW_BARS",
    "InferenceUnavailable",
    "ProviderInference",
    "StubInference",
    "render_strategy_prompt",
]

WINDOW_BARS = 30
"""How many trailing bars are shown to the model, matching the verified market
snapshot's own 30-row cap. Bounded so one request stays small and the cost of a
decision does not grow with the length of the dataset."""

_BAR_COLUMNS = ("open", "high", "low", "close", "volume")

_PROMPT = (
    "You are a portfolio manager. Decide a position for {symbol} as of {session}.\n"
    "Use only the evidence below, which is computed from daily bars up to and\n"
    "including the decision session and contains no later bar.\n"
    "\n"
    "Indicators for {symbol} (computed from the same window):\n"
    "{indicators}\n"
    "\n"
    "Daily bars for {symbol} (oldest first):\n"
    "{window}\n"
    "\n"
    "Weigh the trend, momentum and volatility evidence. State the factors that\n"
    "support your call and the factors that oppose it; if you choose Hold, give a\n"
    "concrete reason grounded in the evidence above rather than a general caution.\n"
    "Reply in exactly this format and nothing else:\n"
    "**Rating**: <one of Buy, Overweight, Hold, Underweight, Sell>\n"
    "**Supporting factors**: <one short sentence>\n"
    "**Opposing factors**: <one short sentence>\n"
    "**Executive Summary**: <one short sentence>\n"
)


def _render_window(window, *, limit: int = WINDOW_BARS) -> str:
    """Render a point-in-time price window as prompt text, oldest bar first.

    Refuses a non-finite value instead of writing ``nan`` into the request: a
    price that is not a real number cannot be reasoned about, and letting it
    through would put a fabricated number in front of the model. Raises
    :class:`InferenceUnavailable` so the caller records a blocked decision.
    """
    if window is None or len(window) == 0:
        return "No price history is available for this session."
    tail = window.tail(limit)
    lines = ["date,open,high,low,close,volume"]
    for stamp, row in tail.iterrows():
        values = []
        for column in _BAR_COLUMNS:
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError) as exc:
                raise InferenceUnavailable(
                    f"the price window has no usable {column!r} value: {exc}"
                ) from exc
            if not math.isfinite(value):
                raise InferenceUnavailable(
                    f"the price window carries a non-finite {column!r} value "
                    f"({value!r}); a decision cannot be taken on it"
                )
            values.append(f"{value:g}")
        lines.append(",".join([str(stamp)[:10], *values]))
    return "\n".join(lines)


def _render_indicators(window) -> str:
    """Render the indicator block, refusing a malformed window before any request.

    The same fail-closed rule as :func:`_render_window`: a non-finite or missing
    OHLCV value raises :class:`InferenceUnavailable` so the caller records a
    blocked decision, rather than sending the model evidence computed from a
    number that is not real.
    """
    try:
        return render_indicator_context(window)
    except IndicatorError as exc:
        raise InferenceUnavailable(
            f"the indicator context could not be computed from the price window: {exc}"
        ) from exc


def render_strategy_prompt(*, symbol: str, session: str, window=None) -> str:
    """The one strategy prompt both the historical and current paths send.

    Extracted verbatim from :class:`ProviderInference` so the current-market
    path reuses the *same* request shape and the *same* deterministic indicator
    block rather than growing a second strategy. The two paths differ only in how
    the window is acquired (a historical session slice versus a current-market
    fetch), never in what the model is asked or shown. A malformed window raises
    :class:`InferenceUnavailable` before any request, exactly as before.
    """
    return _PROMPT.format(
        symbol=symbol,
        session=session,
        indicators=_render_indicators(window),
        window=_render_window(window),
    )


@dataclass
class StubInference:
    """A deterministic, non-genuine decision source for offline tests.

    ``genuine`` is ``False``, so a record built from it is a ``synthetic_fixture``
    and the report never counts it as a real inference.

    ``stop_loss`` and ``take_profit`` are optional. When set, the decision carries
    them as the levels the strategy proposed; when unset, it carries none — and
    the mandatory-protection gate then refuses the trade, because a position
    without levels cannot be protected. Both cases are exercised by the suite.
    """

    rating: str = "Hold"
    reasoning: str = "stub decision for offline testing"
    genuine: bool = False
    calls: int = 0
    last_window: object | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    targets: tuple[float, ...] = ()

    def __call__(self, *, symbol: str, session: str, window=None) -> tuple[str, str]:
        self.calls += 1
        self.last_window = window
        return self.rating, self.reasoning


class ProviderInference:
    """One bounded real provider call per session, or a blocked marker.

    The credential is checked before the call; the client is built with
    ``max_retries=0`` so one session is one request. A response with no readable
    rating raises :class:`InferenceUnavailable` rather than guessing one.

    An optional **fallback provider** is tried only when the primary cannot
    answer — a missing credential, or a call that raised or returned no readable
    rating. The switch is one-directional (primary then fallback, never back) and
    is recorded on ``last_fallback_reason``, so a record can name the provider
    that *actually* answered rather than the one that was configured. A fallback
    is never silent and never produces a second decision for the same session: the
    first readable rating is returned, and if every provider fails the session is
    blocked, not decided.

    Neither implementation prints, logs or stores a credential.
    :class:`ProviderInference` reads the variable *name* only.
    """

    genuine = True

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        timeout: float = 30.0,
        client_factory=None,
        fallback_provider: str | None = None,
        fallback_model: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self._client_factory = client_factory
        self.fallback_provider = fallback_provider
        self.fallback_model = fallback_model
        # The provider and model that answered the most recent call, so a record
        # names what happened rather than what was requested. ``None`` until a
        # call succeeds.
        self.last_provider: str | None = None
        self.last_model: str | None = None
        self.last_fallback_reason: str | None = None

    def _credential_present(self, provider: str) -> tuple[bool, str]:
        env_name = get_api_key_env(provider)
        if env_name is None:
            return False, ""
        value = os.environ.get(env_name)
        return bool(value and value.strip()), env_name

    def _call(self, provider: str, model: str, prompt: str) -> tuple[str, str]:
        from tradingagents.llm_clients import factory
        from tradingagents.llm_evaluation.capture import observed_decision

        make_client = self._client_factory or factory.create_llm_client
        client = make_client(
            provider=provider, model=model, timeout=self.timeout, max_retries=0
        )
        llm = client.get_llm()
        response = llm.invoke(prompt)
        text = getattr(response, "content", response)
        if isinstance(text, list):
            text = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in text
            )
        return observed_decision(text)

    def _attempts(self) -> list[tuple[str, str]]:
        """The providers to try, in order: the primary, then the named fallback."""
        attempts = [(self.provider, self.model)]
        if self.fallback_provider:
            attempts.append((self.fallback_provider, self.fallback_model or self.model))
        return attempts

    def __call__(self, *, symbol: str, session: str, window=None) -> tuple[str, str]:
        # Build the prompt first: a malformed window is refused before any request,
        # so no provider is asked to decide on evidence that is not real. The
        # builder is the shared seam the current-market path also uses, so the two
        # ask the model the identical question.
        prompt = render_strategy_prompt(symbol=symbol, session=session, window=window)
        errors: list[str] = []
        for index, (provider, model) in enumerate(self._attempts()):
            present, env_name = self._credential_present(provider)
            if not present:
                errors.append(
                    f"{provider}: no credential "
                    f"(set {env_name})" if env_name else f"{provider}: no credential"
                )
                continue
            try:
                rating, reasoning = self._call(provider, model, prompt)
            except Exception as exc:  # a provider failure becomes a blocked decision
                errors.append(f"{provider}/{model}: {type(exc).__name__}: {exc}")
                continue
            self.last_provider = provider
            self.last_model = model
            self.last_fallback_reason = errors[-1] if index > 0 and errors else None
            return rating, reasoning

        detail = "; ".join(errors) if errors else "no provider is configured"
        raise InferenceUnavailable(f"no provider produced a decision ({detail})")
