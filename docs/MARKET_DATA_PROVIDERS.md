# Market-data providers

This document describes the real vendor providers the paper experiment can draw
prices from, and — just as importantly — what each one can and cannot provide on
its **free** plan. It exists because "the provider has a free API" does not mean
"the endpoint I want is free", and a paper experiment that silently used a
premium-only or stale endpoint would be lying about what it measured.

Everything below is stated from the vendors' own documentation. Each claim is
quoted with its source; where a fact could not be verified it is marked as such.

## What ships

| Source name | Class | Kind | Current market? | Can drive a multi-session run? |
|---|---|---|---|---|
| `frozen` (default) | `FrozenPriceSource` | `frozen` | no | yes (historical, reproducible) |
| `static` | `StaticPriceSource` | `static` | no | tests only |
| `alpha_vantage` | `AlphaVantageDailySource` | `vendor_daily` | only if the newest bar is recent | yes (daily OHLCV) |
| `finnhub` | `FinnhubQuoteSource` | `vendor_quote` | yes | **no** (a single current-day quote) |

A provider is **opt-in**: nothing is fetched unless `--source` names it. A vendor
source is fetched once, at construction, and frozen; a restart never refetches.
A frame-backed source writes its validated bars and provenance into
`session_state.json`, so a resumed run rebuilds the same source locally instead
of spending a second vendor call.

## Alpha Vantage — daily OHLCV

### The endpoint, and why it is the free one

The provider uses `TIME_SERIES_DAILY`. Alpha Vantage documents that function as
free with `outputsize=compact`:

> "TIME_SERIES_DAILY This API returns raw (as-traded) daily time series (date,
> daily open, daily high, daily low, daily close, daily volume) of the global
> equity specified, covering 25+ years of historical data."
>
> "The "compact" outputsize is available to both free and premium API keys. The
> "full" outputsize is available to premium keys. Please subscribe to any premium
> membership plan to unlock the outputsize=full capability."

Source: <https://www.alphavantage.co/documentation/> (Time Series Stock Data APIs).

The **adjusted** variant is premium, so it is deliberately not used:

> "TIME_SERIES_DAILY_ADJUSTED Trending Premium This API returns raw (as-traded)
> daily open/high/low/close/volume values, adjusted close values, and historical
> split/dividend events…"

Source: <https://www.alphavantage.co/documentation/>.

Consequences in code:

- `outputsize="full"` raises `ProviderUnsupportedError` unless the caller passes
  `allow_premium_outputsize=True` and thereby accepts a paid call.
- Because the free function is unadjusted, the source declares `adjusted=False`
  and never mixes adjusted and unadjusted bars.

### Rate limit and quota

The free plan allows 25 requests per day:

> "We are pleased to provide free stock API service covering the majority of our
> datasets for 25 API requests per day and unlimited API requests for verified
> open-source or educational projects."

Source: <https://www.alphavantage.co/support/>.

Alpha Vantage reports a quota or a bad key **inside an HTTP 200 body** (a
`"Note"` / `"Information"` field), and a rejected call as `"Error Message"`. The
provider classifies each:

- a quota/limit notice → `ProviderRateLimitedError`;
- an invalid/missing key notice → `ProviderAuthError`;
- a rejected call → `ProviderUnsupportedError`.

It never parses such a body as data. (This matches the existing
`dataflows/vendors/alpha_vantage/common.py` classification.)

### Freshness

A bar is only "recent data" if its date is within `max_stale_days` (default 10
calendar days) of the fetch date; otherwise the fetch raises
`ProviderStaleDataError`. `is_current_market` is `True` only when the newest bar
is within that window. Staleness can be disabled explicitly
(`max_stale_days=None`), in which case the source reports `is_current_market =
False`, because "unknown recency" is not "current".

## Finnhub — a current-day quote, not a bar history

### The candle endpoint is premium

The historical OHLCV endpoint is **not** on the free plan:

> `"/stock/candle" … "premium":"Premium Access Required"`

Source: <https://finnhub.io/docs/api/stock-candles> (API spec, `premium` field for
`/stock/candle`).

The pricing page leaves the Free column's OHLC row blank and shows OHLC history
only under the paid plan:

> "| US Market Data | | | … | OHLC | | 30+ years |"

Source: <https://finnhub.io/pricing> (Free vs. All-In-One comparison table).

Therefore `FinnhubQuoteSource` **does not call the candle endpoint** and does not
pretend to have a bar history. Its `bars`, `window`, `sessions` and `covers`
methods raise `ProviderUnsupportedError`.

### What the free endpoint does provide

`/quote` is free and returns a real-time quote:

> "Get real-time quote data for US stocks. Constant polling is not recommended.
> Use websocket if you need real-time updates."

Source: <https://finnhub.io/docs/api/quote>.

The provider parses `c` (current price), `o`/`h`/`l` (day open/high/low), `pc`
(previous close) and `t` (Unix timestamp), validates them finite and positive, and
checks that the quote's own date matches the expected session — a quote for the
wrong day raises `ProviderStaleDataError` rather than being served as this day's
price.

### Rate limit

The free plan allows 60 API calls per minute:

> "| Limit | 60 API calls/minute | …"

Source: <https://finnhub.io/pricing>.

## Safety rules the providers obey

1. **No live path.** A provider fetches data, never an order. There is no broker
   client and no `ExecutionMode.LIVE`.
2. **Explicit fallback only.** `select_source(requested, fallback=…)` switches to
   the fallback only when one is named, and logs it. Without a fallback, a source
   failure raises — the run stops rather than changing its data.
3. **No mixing.** An experiment runs on exactly one source's bars, so a snapshot
   never blends two vendors. There is no per-session fallback.
4. **No silent 2024 substitution.** The CLI refuses to resume a vendor run on the
   frozen dataset, so a vendor failure cannot be hidden behind historical data. A
   frame-backed vendor run is resumed from the bars stored in its own state, so
   the refusal only applies to a source that carries no persistable bars (a
   quote-only source).
5. **No fabricated data.** A missing price is an error or a blocked decision, never
   an invented bar. A malformed payload is refused by the shared validator
   (`sources._validate_frame`): NaN, infinity, a non-positive price, a high below
   low, an open/close outside the range, a duplicate session, an out-of-order
   session and a non-numeric column are all rejected.
6. **No credential in any artifact.** A key is read from the environment, passed
   as a `secret` to the HTTP layer (which redacts it from any error), and never
   written to a frame, a snapshot, a report or the journal.

## Provenance recorded for every source

Each source declares a `SourceProvenance`, written into the experiment state, the
journal's creation entry and the report's `data` block:

`name`, `kind` (`frozen` / `static` / `vendor_daily` / `vendor_quote`),
`is_current_market`, `symbol`, `fetched_at`, `last_bar_date`, `timezone`,
`adjusted`, `dataset_start`, `dataset_end`, `snapshot_id`, `content_hash`, and
`notes`. The snapshot fingerprint itself is the existing
`llm_evaluation.replay.window_hash`, so a snapshot and the record that names it
must agree.

## Testing

Every provider test drives a **mocked HTTP transport**; no test reaches the
network (the suite's autouse `_no_network` fixture would refuse it) and no test
needs a real key. The mock is a fake transport, not a fake provider, so the
provider's own parsing, validation, classification and provenance code runs for
real. See `tests/test_phase11_providers.py`.

## Unverified / open items

- The exact free-plan request-per-day figure for Finnhub beyond the 60/minute
  rate limit is not published on the pages fetched here; only the per-minute limit
  is asserted.
- Live behaviour against a real key has **not** been exercised in this
  environment (no key present). All provider tests are simulated. A live check
  needs the key entered directly in the secure environment, with explicit
  approval.
