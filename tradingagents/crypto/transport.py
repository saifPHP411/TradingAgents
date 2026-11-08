"""One bounded, redacting JSON GET, shared by the crypto data sources.

The crypto layer is below the paper experiment (it is data, not trading), so it
carries its own small HTTP seam rather than importing the experiment's. The shape
is the same on purpose: bounded, injectable, and it scrubs any secret from a
failure before raising, so a key cannot reach a log.

The default transport is ``requests``. A test injects its own and never opens a
socket; the suite's autouse network guard would refuse one anyway.

**Retries.** A throttled (HTTP 429/418) or momentarily unreachable venue is
retried a *bounded* number of times with exponential backoff, because a single
transient failure should not become a "no price" cycle. The bound is small and
the delay grows, so this is never an aggressive polling loop. A response that is
*correct but unwelcome* (HTTP 400 for an invalid symbol, HTTP 404) is **not**
retried: repeating it would only repeat the same answer.
"""

from __future__ import annotations

import json
import time
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

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        """The venue's HTTP status, when the failure was a status and not a fault.

        ``None`` means the request never produced a status (a timeout, a refused
        connection, a body that was not JSON). A caller uses the difference to
        tell "the venue answered and refused" from "the venue could not answer".
        """


class CryptoAuthError(CryptoDataError):
    """The venue rejected a credential (for a source that needs one)."""


class CryptoRateLimitedError(CryptoDataError):
    """The venue throttled the request."""


MAX_ATTEMPTS = 3
"""Total attempts for a retryable failure (one try plus two retries)."""

RETRY_BACKOFF_SECONDS = 0.5
"""The first backoff delay. Each further attempt doubles it."""

RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
"""Statuses worth one more try: throttling and transient server/network faults.

HTTP 400 and 404 are deliberately absent: they are the venue's considered answer
to "that symbol is not valid", and repeating them would only repeat the answer.
"""


def _is_transient_failure(exc: BaseException) -> bool:
    """Whether a transport exception looks like a temporary network fault.

    A timeout, a refused connection or a DNS failure is transient and worth one
    more try. Anything else (a programming error, a bad argument) is not: the
    retry would fail the same way, so it is reported at once.
    """
    import requests

    if isinstance(exc, requests.exceptions.Timeout):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    return isinstance(exc, (ConnectionError, TimeoutError))


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
    diagnose: Callable[[dict[str, object]], None] | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> object:
    """One bounded JSON GET. Raises :class:`CryptoDataError` on any failure.

    Returns whatever JSON the body held (an object or an array, because the crypto
    APIs return both). ``classify_status`` lets a source give a specific error for
    a status it can interpret (a 429 as rate-limiting, a 401 as auth); it runs
    after redaction, so a secret in the body cannot leak through it.

    ``diagnose`` receives a structured record of every attempt — the path, the
    HTTP status, the sanitized venue error code and message, the response's JSON
    type and whether the requested symbol was present — so a production failure
    can be read from the log instead of guessed at. The record is sanitized before
    it is handed over: only a venue's own numeric code and a short message are
    kept, never a header, a key or a query-string secret.

    A throttled or momentarily unreachable venue is retried up to ``max_attempts``
    times with exponential backoff; a status the venue means (400, 404) is not.
    """
    send = transport or _default_transport
    attempt = 0
    delay = RETRY_BACKOFF_SECONDS
    last_error: CryptoDataError | None = None

    while attempt < max(1, max_attempts):
        attempt += 1
        try:
            response = send(
                url=url, params=params or {}, headers=headers or {}, timeout=timeout
            )
        except CryptoDataError:
            raise
        except Exception as exc:  # noqa: BLE001 - a transport failure is reported, never hidden
            sanitized = _redact(f"{type(exc).__name__}: {exc}", secrets)
            transient = _is_transient_failure(exc)
            _diagnose(
                diagnose,
                url=url,
                attempt=attempt,
                status=None,
                venue_code=None,
                venue_message=sanitized[:200],
                response_type=None,
                symbol_present=None,
                outcome="transport_failure",
                retrying=transient and attempt < max(1, max_attempts),
            )
            last_error = CryptoDataError(f"the request failed: {sanitized}")
            if transient and attempt < max(1, max_attempts):
                sleep(delay)
                delay *= 2
                continue
            raise last_error from None

        body = _redact((response.text or "")[:200], secrets)
        if response.status_code != 200:
            code, message = _venue_error(response.text)
            _diagnose(
                diagnose,
                url=url,
                attempt=attempt,
                status=response.status_code,
                venue_code=code,
                venue_message=_redact(message, secrets) if message else message,
                response_type=None,
                symbol_present=None,
                outcome="http_error",
                retrying=response.status_code in RETRYABLE_STATUSES
                and attempt < max(1, max_attempts),
            )
            last_error = CryptoDataError(
                f"the venue answered HTTP {response.status_code}: {body}",
                status_code=response.status_code,
            )
            if response.status_code in RETRYABLE_STATUSES and attempt < max(1, max_attempts):
                sleep(delay)
                delay *= 2
                continue
            if classify_status is not None:
                _classify(classify_status, response.status_code, body)
            raise last_error from None

        try:
            payload = json.loads(response.text)
        except (json.JSONDecodeError, TypeError) as exc:
            _diagnose(
                diagnose,
                url=url,
                attempt=attempt,
                status=response.status_code,
                venue_code=None,
                venue_message=_redact(str(exc), secrets)[:200],
                response_type=None,
                symbol_present=None,
                outcome="malformed_json",
                retrying=False,
            )
            raise CryptoDataError(
                f"the venue did not answer with JSON: {_redact(str(exc), secrets)}"
            ) from None

        _diagnose(
            diagnose,
            url=url,
            attempt=attempt,
            status=response.status_code,
            venue_code=None,
            venue_message=None,
            response_type=type(payload).__name__,
            symbol_present=_symbol_present(payload),
            outcome="ok",
            retrying=False,
        )
        return payload

    raise last_error or CryptoDataError("the request failed")  # pragma: no cover


def _classify(
    classify_status: Callable[[int, str], None], status: int, body: str
) -> None:
    """Let the source name a status, keeping the status on the error it raises.

    The source's own hook raises a *specific* error (a throttle, an auth
    refusal), which is more useful than the generic one. The status is carried
    on that error too, so a caller can still tell a refusal from a fault.
    """
    try:
        classify_status(status, body)
    except CryptoDataError as exc:
        if exc.status_code is None:
            exc.status_code = status
        raise


def _venue_error(text: str) -> tuple[int | None, str | None]:
    """The venue's own numeric error code and message, when it sent a JSON one.

    Binance answers a bad request with ``{"code": -1100, "msg": "..."}``. Only
    those two fields are surfaced, so a diagnostic record cannot carry an
    arbitrary body (and therefore cannot carry a secret) into a log.
    """
    try:
        payload = json.loads(text or "")
    except (json.JSONDecodeError, TypeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    code = payload.get("code")
    message = payload.get("msg")
    return (
        code if isinstance(code, int) else None,
        message[:200] if isinstance(message, str) else None,
    )


def _symbol_present(payload: object) -> bool | None:
    """Whether the response carried a ticker row for the requested symbol.

    ``None`` when the response is not a ticker-shaped object or array, so a
    diagnostic never claims a symbol was present when it cannot tell.
    """
    if isinstance(payload, dict):
        return bool(payload.get("symbol"))
    if isinstance(payload, list):
        return any(isinstance(row, dict) and row.get("symbol") for row in payload)
    return None


def _diagnose(
    diagnose: Callable[[dict[str, object]], None] | None,
    **fields: object,
) -> None:
    """Hand a structured record to the caller's sink, never raising from it."""
    if diagnose is None:
        return
    record = {"path": _path_of(str(fields.get("url", ""))), **fields}
    record.pop("url", None)
    try:
        diagnose(record)
    except Exception:  # noqa: BLE001 - a diagnostic sink must not break the read
        return


def _path_of(url: str) -> str:
    """The URL's path, without its host or its query string."""
    return urllib.parse.urlsplit(url).path


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
