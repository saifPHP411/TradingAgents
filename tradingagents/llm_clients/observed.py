"""Observe whether a real model call actually happened.

The decision-capture path must label a record ``llm_inference`` only when a real
provider call produced the decision. It cannot take the caller's word for that,
and it must not be fooled by a scripted or stubbed model in a test. This module
provides the discriminator.

How it works, and why it is honest:

- An :class:`InferenceObserver` is a LangChain callback handler. It flips to
  ``observed`` when ``on_llm_end`` fires, which happens only after a model
  returns.
- The graph attaches one observer to the callbacks it hands its *real* clients
  (``create_tier_client`` forwards callbacks to the provider chat model). So a
  graph built through the real provider factory reports observed inference.
- A test that stubs the model — by monkeypatching ``create_tier_client`` or by
  passing a bare ``BaseChatModel`` — never routes through that path, so its
  observer never fires and the run reports *not* observed. A mock therefore
  cannot be mislabelled as genuine inference, which is the whole point.

The flag is per-observer, so two graphs in one process do not contaminate each
other. It is never set from an argument.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class InferenceObserver(BaseCallbackHandler):
    """Records whether any model call completed on the run it is attached to."""

    def __init__(self) -> None:
        super().__init__()
        self.observed = False
        self.calls = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: ARG002
        # Fires once per completed model call. The response body is never read,
        # so no model output or credential can leak through this handler.
        self.observed = True
        self.calls += 1


__all__ = ["InferenceObserver"]
