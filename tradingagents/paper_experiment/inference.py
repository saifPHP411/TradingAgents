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

PAPER_INFERENCE_PROMPT_VERSION = "paper-inference-2"
"""Bumped when the request shape changes. ``paper-inference-1`` was the data-blind
prompt that could only elicit ``Hold``; a record captured under it saw no bars."""

WINDOW_BARS = 30
"""How many trailing bars are shown to the model, matching the verified market
snapshot's own 30-row cap. Bounded so one request stays small and the cost of a
decision does not grow with the length of the dataset."""

_BAR_COLUMNS = ("open", "high", "low", "close", "volume")

_PROMPT = (
    "You are a portfolio manager. Decide a position for {symbol} as of {session}.\n"
    "Use only the daily price history below, which ends at the decision session and\n"
    "contains no later bar. Reply in exactly this format and nothing else:\n"
    "**Rating**: <one of Buy, Overweight, Hold, Underweight, Sell>\n"
    "**Executive Summary**: <one short sentence>\n"
    "\n"
    "Daily bars for {symbol} (oldest first):\n"
    "{window}"
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


@dataclass
class StubInference:
    """A deterministic, non-genuine decision source for offline tests.

    ``genuine`` is ``False``, so a record built from it is a ``synthetic_fixture``
    and the report never counts it as a real inference.
    """

    rating: str = "Hold"
    reasoning: str = "stub decision for offline testing"
    genuine: bool = False
    calls: int = 0
    last_window: object | None = None

    def __call__(self, *, symbol: str, session: str, window=None) -> tuple[str, str]:
        self.calls += 1
        self.last_window = window
        return self.rating, self.reasoning


class ProviderInference:
    """One bounded real provider call per session, or a blocked marker.

    The credential is checked before the call; the client is built with
    ``max_retries=0`` so one session is one request. A response with no readable
    rating raises :class:`InferenceUnavailable` rather than guessing one.
    """

    genuine = True

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        timeout: float = 30.0,
        client_factory=None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self._client_factory = client_factory

    def _credential_present(self) -> tuple[bool, str]:
        env_name = get_api_key_env(self.provider)
        if env_name is None:
            return False, ""
        value = os.environ.get(env_name)
        return bool(value and value.strip()), env_name

    def __call__(self, *, symbol: str, session: str, window=None) -> tuple[str, str]:
        present, env_name = self._credential_present()
        if not present:
            raise InferenceUnavailable(
                f"no credential for provider {self.provider!r}: set the {env_name} "
                "environment variable to produce a genuine decision"
            )
        from tradingagents.llm_clients import factory
        from tradingagents.llm_evaluation.capture import observed_decision

        prompt = _PROMPT.format(
            symbol=symbol, session=session, window=_render_window(window)
        )
        make_client = self._client_factory or factory.create_llm_client
        try:
            client = make_client(
                provider=self.provider, model=self.model, timeout=self.timeout, max_retries=0
            )
            llm = client.get_llm()
            response = llm.invoke(prompt)
        except Exception as exc:  # a provider failure becomes a blocked decision
            raise InferenceUnavailable(
                f"provider call failed: {type(exc).__name__}: {exc}"
            ) from exc

        text = getattr(response, "content", response)
        if isinstance(text, list):
            text = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in text
            )
        try:
            return observed_decision(text)
        except Exception as exc:  # noqa: BLE001 - an unreadable response is not a decision
            raise InferenceUnavailable(
                f"the provider response had no readable rating: {exc}"
            ) from exc
