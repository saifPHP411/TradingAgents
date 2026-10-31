"""The shared seam for a vendor data provider: errors, and one bounded HTTP call.

Every vendor provider goes through :func:`fetch_json` so the network behaviour is
written once and can be reasoned about in one place:

- **Bounded.** A timeout is always passed; there is no unbounded call.
- **Redacting.** A credential that appears in a URL, a header or an error message
  is replaced with ``***`` before anything is raised, so a key cannot reach a log,
  a traceback or a report. (Same rule as ``dataflows/net.py``.)
- **Injectable.** The transport is a parameter, so a test drives a mocked response
  and never touches a socket. The default is ``requests``.
- **Non-retrying by default.** ``max_retries=0`` so one session is one request; a
  caller that wants a retry asks for it explicitly and gets a bounded number.

A provider that cannot answer raises a :class:`ProviderError` subclass, never
returns an empty frame dressed up as data. A rate-limit notice that arrives inside
an HTTP 200 body is detected by the *provider* (it knows its own schema) and
raised as :class:`ProviderRateLimitedError`, because a vendor that throttles
often answers 200 with a notice rather than a 429.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Callable, Mapping

from ..sources import PriceSourceError


class ProviderError(PriceSourceError):
    """A vendor provider could not serve a request.

    A :class:`~tradingagents.paper_experiment.sources.PriceSourceError` (and thus a
    ``ValueError``), so existing callers and the experiment's own error handling
    keep working, while a provider-specific failure is still distinguishable.
    """


class ProviderNotConfiguredError(ProviderError):
    """The provider was selected but its credential is missing."""


class ProviderAuthError(ProviderError):
    """The provider rejected the credential (invalid or unauthorised)."""


class ProviderRateLimitedError(ProviderError):
    """The provider throttled the request, or the plan's quota is exhausted."""


class ProviderUnsupportedError(ProviderError):
    """The provider cannot serve this request (unsupported symbol or data kind)."""


class ProviderStaleDataError(ProviderError):
    """The provider returned data whose latest bar is older than the requested session.

    Distinct from "no data": the vendor answered, but what it answered with is too
    old to be a bar for the session being processed, so using it would silently
    serve a stale price as a current one.
    """


class HttpResponse:
    """The part of an HTTP response a provider needs, and nothing else."""

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


Transport = Callable[..., HttpResponse]


def _redact(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def _default_transport(*, url: str, params: Mapping, headers: Mapping, timeout: float) -> HttpResponse:
    import requests

    response = requests.get(url, params=dict(params), headers=dict(headers), timeout=timeout)
    return HttpResponse(response.status_code, response.text)


def fetch_json(
    *,
    url: str,
    params: Mapping,
    timeout: float,
    secrets: tuple[str, ...] = (),
    headers: Mapping | None = None,
    transport: Transport | None = None,
    max_retries: int = 0,
) -> dict:
    """One bounded JSON GET, with the credential scrubbed from any failure.

    Raises :class:`ProviderError` on a transport failure, a non-2xx status, or a
    body that is not a JSON object. It never returns a partially parsed body and
    never swallows the failure.
    """
    send = transport or _default_transport
    attempt = 0
    while True:
        try:
            response = send(url=url, params=params, headers=headers or {}, timeout=timeout)
        except ProviderError:
            raise
        except Exception as exc:  # a transport failure is reported, never hidden
            message = _redact(f"{type(exc).__name__}: {exc}", secrets)
            if attempt < max_retries:
                attempt += 1
                continue
            raise ProviderError(f"the request failed: {message}") from None

        if response.status_code != 200:
            body = _redact((response.text or "")[:200], secrets)
            raise ProviderError(
                f"the provider answered HTTP {response.status_code}: {body}"
            )
        try:
            payload = json.loads(response.text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ProviderError(
                f"the provider did not answer with JSON: {_redact(str(exc), secrets)}"
            ) from None
        if not isinstance(payload, dict):
            raise ProviderError("the provider answered with a non-object JSON body")
        return payload


def redact_url(url: str, secrets: tuple[str, ...]) -> str:
    """A URL with any query-string secret replaced, for safe display."""
    if not secrets:
        return url
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    scrubbed = [
        (key, "***" if value in secrets else value) for key, value in query
    ]
    return urllib.parse.urlunsplit(
        parsed._replace(query=urllib.parse.urlencode(scrubbed))
    )
