"""The real-inference seam for the paper experiment, and the offline stub.

The experiment takes an ``InferenceFn``: a callable that returns ``(rating,
reasoning)`` for a symbol and session, or raises
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

Neither prints, logs or stores a credential. :class:`ProviderInference` reads the
variable *name* only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from tradingagents.llm_clients.api_key_env import get_api_key_env

from .experiment import InferenceUnavailable

_PROMPT = (
    "You are a portfolio manager. For {symbol} on {session}, given no further "
    "data, reply in exactly this format and nothing else:\n"
    "**Rating**: <one of Buy, Overweight, Hold, Underweight, Sell>\n"
    "**Executive Summary**: <one short sentence>"
)


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

    def __call__(self, *, symbol: str, session: str) -> tuple[str, str]:
        self.calls += 1
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

    def __call__(self, *, symbol: str, session: str) -> tuple[str, str]:
        present, env_name = self._credential_present()
        if not present:
            raise InferenceUnavailable(
                f"no credential for provider {self.provider!r}: set the {env_name} "
                "environment variable to produce a genuine decision"
            )
        from tradingagents.llm_clients import factory
        from tradingagents.llm_evaluation.capture import observed_decision

        make_client = self._client_factory or factory.create_llm_client
        try:
            client = make_client(
                provider=self.provider, model=self.model, timeout=self.timeout, max_retries=0
            )
            llm = client.get_llm()
            response = llm.invoke(_PROMPT.format(symbol=symbol, session=session))
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
