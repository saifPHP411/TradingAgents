"""One bounded, redacting JSON GET, shared by the crypto data sources.

The crypto layer is below the paper experiment (it is data, not trading), so it
carries its own small HTTP seam rather than importing the experiment's. The shape
is the same on purpose: bounded, injectable, non-retrying by default, and it
scrubs any secret from a failure before raising, so a key cannot reach a log.

The default transport is ``requests``. A test injects its own and never opens a
socket; the suite's autouse network guard would refuse one anyway.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Callable, Mapping


class HttpResponse:
    """The part of an HTTP response a data source needs, and nothing else."""

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


Transport = Callable[..., HttpResponse]


class CryptoDataError(RuntimeError):
    """A crypto data source could not serve a request. Never partial."""


class CryptoAuthError(CryptoDataError):
    """The venue rejected a credential (for a source that needs one)."""


class CryptoRateLimitedError(CryptoDataError):
    """The venue throttled the request."""


def _redact(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def _default_transport(
    *, url: str, params: Mapping, headers: Mapping, timeout: float
) -> HttpResponse:
    import requests

    response = requests.get(url, params=dict(params), headers=dict(headers), timeout=timeout)
    return HttpResponse(response.status_code, response.text)


def fetch_json(
    *,
    url: str,
    params: Mapping | None = None,
    timeout: float = 15.0,
    secrets: tuple[str, ...] = (),
    headers: Mapping | None = None,
    transport: Transport | None = None,
    classify_status: Callable[[int, str], None] | None = None,
) -> object:
    """One bounded JSON GET. Raises :class:`CryptoDataError` on any failure.

    Returns whatever JSON the body held (an object or an array, because the crypto
    APIs return both). ``classify_status`` lets a source give a specific error for
    a status it can interpret (a 429 as rate-limiting, a 401 as auth); it runs
    after redaction, so a secret in the body cannot leak through it.
    """
    send = transport or _default_transport
    try:
        response = send(url=url, params=params or {}, headers=headers or {}, timeout=timeout)
    except CryptoDataError:
        raise
    except Exception as exc:  # noqa: BLE001 - a transport failure is reported, never hidden
        raise CryptoDataError(
            f"the request failed: {_redact(f'{type(exc).__name__}: {exc}', secrets)}"
        ) from None

    if response.status_code != 200:
        body = _redact((response.text or "")[:200], secrets)
        if classify_status is not None:
            classify_status(response.status_code, body)
        raise CryptoDataError(f"the venue answered HTTP {response.status_code}: {body}")
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise CryptoDataError(
            f"the venue did not answer with JSON: {_redact(str(exc), secrets)}"
        ) from None


def redact_url(url: str, secrets: tuple[str, ...]) -> str:
    """A URL with any query-string secret replaced, for safe display."""
    if not secrets:
        return url
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    scrubbed = [(key, "***" if value in secrets else value) for key, value in query]
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(scrubbed)))


__all__ = [
    "CryptoAuthError",
    "CryptoDataError",
    "CryptoRateLimitedError",
    "HttpResponse",
    "Transport",
    "fetch_json",
    "redact_url",
]
