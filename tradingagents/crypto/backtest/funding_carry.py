"""Research-only delta-neutral spot/perpetual funding-rate carry study.

Uses public Binance Spot + USD-M perpetual history as a cross-venue proxy.
No credentials, account access, orders, or trading integration.
Funding observations are lagged for decisions: the rate settled at t can only
select exposure for the *next* funding interval. The next interval's realized
rate is credited only after the position has remained open through settlement.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

SPOT_HOSTS = ("https://data-api.binance.vision", "https://api.binance.com")
PERP_HOSTS = (
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
)
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
HOUR_MS = 3_600_000
FUNDING_INTERVAL_MS = 8 * HOUR_MS


def _get_json(session: requests.Session, hosts: tuple[str, ...], path: str,
              params: dict[str, Any]) -> Any:
    errors: list[str] = []
    for host in hosts:
        url = host + path
        try:
            response = session.get(url, params=params, timeout=20)
            if response.status_code in (403, 418, 429, 451):
                errors.append(f"{url}: HTTP {response.status_code}")
                continue
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError:
                preview = response.text[:160].replace("\n", " ")
                errors.append(
                    f"{url}: non-JSON response "
                    f"(content-type={response.headers.get('Content-Type')!r}, body={preview!r})"
                )
                continue
            if not isinstance(payload, list):
                errors.append(f"{url}: unexpected JSON payload type {type(payload).__name__}")
                continue
            return payload
        except requests.RequestException as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    detail = "; ".join(errors) if errors else "no hosts configured"
    raise RuntimeError(
        f"public market-data endpoint unavailable for {path}; "
        f"tried {len(hosts)} hosts: {detail}"
    ) from None


def _parse_futures_kline_archive(
    content: bytes, *, start_ms: int, end_ms: int
) -> dict[int, dict[str, float]]:
    """Parse Binance USD-M Futures 1h kline CSV files from a public ZIP archive."""
    rows: dict[int, dict[str, float]] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"corrupt ZIP member: {bad_member}")
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not members:
                raise ValueError("archive contains no CSV data file")
            for name in members:
                with archive.open(name) as stream:
                    reader = csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
                    for item in reader:
                        if len(item) < 5:
                            continue
                        try:
                            timestamp = int(float(item[0]))
                            open_price, close_price = float(item[1]), float(item[4])
                        except (TypeError, ValueError, OverflowError):
                            continue
                        if timestamp > 100_000_000_000_000:
                            timestamp //= 1000
                        if not start_ms <= timestamp < end_ms:
                            continue
                        if (
                            not math.isfinite(open_price)
                            or not math.isfinite(close_price)
                            or open_price <= 0
                            or close_price <= 0
                        ):
                            raise ValueError(f"invalid OHLC price in archive row at {timestamp}")
                        rows[timestamp] = {"open": open_price, "close": close_price}
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid Binance archive ZIP: {exc}") from None
    return rows


def _archive_futures_klines(
    session: requests.Session, symbol: str, start_ms: int, end_ms: int,
    *, now: datetime | None = None,
) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    """Fallback to Binance's official USD-M Futures monthly and daily public archives."""
    now = now or datetime.now(UTC)
    start_date = datetime.fromtimestamp(start_ms / 1000, UTC).date()
    end_date = datetime.fromtimestamp((end_ms - 1) / 1000, UTC).date()
    current_month = now.date().replace(day=1)
    month_cursor = start_date.replace(day=1)
    rows: dict[int, dict[str, float]] = {}
    downloaded: list[str] = []
    missing: list[str] = []
    failures: list[str] = []

    while month_cursor <= end_date:
        if month_cursor == current_month:
            first_day = max(start_date, month_cursor)
            last_day = min(end_date, now.date() - timedelta(days=1))
            days = (
                first_day + timedelta(days=offset)
                for offset in range(max(0, (last_day - first_day).days + 1))
            ) if last_day >= first_day else ()
            periods = (
                ("daily", f"{symbol}-1h-{day.isoformat()}.zip")
                for day in days
            )
        else:
            periods = iter((
                ("monthly", f"{symbol}-1h-{month_cursor.year}-{month_cursor.month:02d}.zip"),
            ))

        for frequency, filename in periods:
            url = (
                "https://data.binance.vision/data/futures/um/"
                f"{frequency}/klines/{symbol}/1h/{filename}"
            )
            try:
                response = session.get(url, timeout=30)
                if response.status_code == 404:
                    missing.append(url)
                    continue
                response.raise_for_status()
                rows.update(
                    _parse_futures_kline_archive(
                        response.content, start_ms=start_ms, end_ms=end_ms
                    )
                )
                downloaded.append(url)
            except (requests.RequestException, ValueError, OSError) as exc:
                failures.append(f"{url}: {type(exc).__name__}: {exc}")

        month_cursor = date(
            month_cursor.year + (1 if month_cursor.month == 12 else 0),
            1 if month_cursor.month == 12 else month_cursor.month + 1,
            1,
        )

    archive_end = min(
        end_ms,
        int(datetime.combine(now.date(), datetime.min.time(), UTC).timestamp() * 1000),
    )
    expected_bars = max(1, int((archive_end - start_ms) / HOUR_MS))
    minimum_bars = int(expected_bars * 0.90)
    if len(rows) < minimum_bars:
        detail = "; ".join(failures[:8]) or "no archive download errors recorded"
        raise RuntimeError(
            f"Binance archive fallback incomplete for {symbol}: "
            f"{len(rows)}/{expected_bars} hourly bars (minimum {minimum_bars}); "
            f"downloaded={len(downloaded)}, missing={len(missing)}; {detail}"
        )
    return rows, {
        "source": "binance_public_data_archive",
        "downloaded_files": len(downloaded),
        "missing_files": len(missing),
        "first_bar_ms": min(rows),
        "last_bar_ms": max(rows),
    }


def _paged_klines(session: requests.Session, hosts: tuple[str, ...], path: str,
                  symbol: str, start_ms: int, end_ms: int) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    cursor = start_ms
    while cursor < end_ms:
        page = _get_json(session, hosts, path, {
            "symbol": symbol, "interval": "1h", "startTime": cursor,
            "endTime": end_ms - 1, "limit": 1000,
        })
        if not isinstance(page, list) or not page:
            break
        for item in page:
            ts = int(item[0])
            if start_ms <= ts < end_ms:
                rows[ts] = {"open": float(item[1]), "close": float(item[4])}
        next_cursor = int(page[-1][0]) + HOUR_MS
        if next_cursor <= cursor:
            raise RuntimeError(f"non-advancing kline pagination for {symbol}")
        cursor = next_cursor
        if len(page) < 1000:
            break
        time.sleep(0.08)
    return rows


def _funding_history(session: requests.Session, symbol: str,
                     start_ms: int, end_ms: int) -> list[dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    cursor = start_ms
    while cursor < end_ms:
        page = _get_json(session, PERP_HOSTS, "/fapi/v1/fundingRate", {
            "symbol": symbol, "startTime": cursor, "endTime": end_ms - 1, "limit": 1000,
        })
        if not isinstance(page, list) or not page:
            break
        for item in page:
            ts = int(item["fundingTime"])
            if start_ms <= ts < end_ms:
                rows[ts] = {"timestamp": float(ts), "rate": float(item["fundingRate"])}
        next_cursor = int(page[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            raise RuntimeError(f"non-advancing funding pagination for {symbol}")
        cursor = next_cursor
        if len(page) < 1000:
            break
        time.sleep(0.08)
    return [rows[key] for key in sorted(rows)]


def _parse_funding_archive(content: bytes, *, start_ms: int, end_ms: int) -> list[dict[str, float]]:
    """Parse a Binance Vision monthly USD-M funding-rate CSV archive."""
    rows: dict[int, dict[str, float]] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"corrupt ZIP member: {bad_member}")
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not members:
                raise ValueError("funding archive contains no CSV data file")
            for name in members:
                with archive.open(name) as stream:
                    reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
                    if not reader.fieldnames:
                        raise ValueError("funding archive CSV is missing its header")
                    fields = {field.strip().lower(): field for field in reader.fieldnames}
                    if not {"calc_time", "last_funding_rate"}.issubset(fields):
                        raise ValueError(
                            "funding archive CSV must contain calc_time and last_funding_rate"
                        )
                    interval_field = fields.get("funding_interval_hours")
                    for item in reader:
                        try:
                            timestamp = int(float(item[fields["calc_time"]]))
                            rate = float(item[fields["last_funding_rate"]])
                            interval_hours = float(item[interval_field]) if interval_field else 8.0
                        except (TypeError, ValueError, OverflowError, KeyError):
                            raise ValueError("invalid row in funding archive CSV") from None
                        if timestamp > 100_000_000_000_000:
                            timestamp //= 1000
                        if not start_ms <= timestamp < end_ms:
                            continue
                        if (
                            not math.isfinite(rate)
                            or not math.isfinite(interval_hours)
                            or interval_hours <= 0
                            or interval_hours > 24
                        ):
                            raise ValueError(f"invalid funding observation at {timestamp}")
                        rows[timestamp] = {
                            "timestamp": float(timestamp),
                            "rate": rate,
                            "interval_hours": interval_hours,
                        }
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid funding archive ZIP: {exc}") from None
    return [rows[key] for key in sorted(rows)]


def _archive_funding_history(
    session: requests.Session, symbol: str, start_ms: int, end_ms: int,
    *, now: datetime | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Load completed monthly funding archives; the current partial month is excluded."""
    now = now or datetime.now(UTC)
    start_date = datetime.fromtimestamp(start_ms / 1000, UTC).date()
    requested_end_date = datetime.fromtimestamp((end_ms - 1) / 1000, UTC).date()
    current_month_start = now.date().replace(day=1)
    archive_end_ms = min(
        end_ms,
        int(datetime.combine(current_month_start, datetime.min.time(), UTC).timestamp() * 1000),
    )
    if start_ms >= archive_end_ms:
        raise RuntimeError(
            f"no completed monthly funding archive is available for {symbol} in the requested range"
        )

    month_cursor = start_date.replace(day=1)
    last_archive_month = datetime.fromtimestamp(
        (archive_end_ms - 1) / 1000, UTC
    ).date().replace(day=1)
    rows: dict[int, dict[str, float]] = {}
    downloaded: list[str] = []
    missing: list[str] = []
    failures: list[str] = []

    while month_cursor <= last_archive_month:
        filename = f"{symbol}-fundingRate-{month_cursor.year}-{month_cursor.month:02d}.zip"
        url = (
            "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
            f"{symbol}/{filename}"
        )
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 404:
                missing.append(url)
            else:
                response.raise_for_status()
                for item in _parse_funding_archive(
                    response.content, start_ms=start_ms, end_ms=archive_end_ms
                ):
                    rows[int(item["timestamp"])] = item
                downloaded.append(url)
        except (requests.RequestException, ValueError, OSError) as exc:
            failures.append(f"{url}: {type(exc).__name__}: {exc}")

        month_cursor = date(
            month_cursor.year + (1 if month_cursor.month == 12 else 0),
            1 if month_cursor.month == 12 else month_cursor.month + 1,
            1,
        )

    expected_min_events = int(((archive_end_ms - start_ms) / (12 * HOUR_MS)) * 0.90)
    minimum_events = max(20, expected_min_events)
    if len(rows) < minimum_events:
        detail = "; ".join(failures[:8]) or "no archive download errors recorded"
        raise RuntimeError(
            f"Binance funding archive incomplete for {symbol}: {len(rows)} events "
            f"(expected at least {minimum_events}), downloaded={len(downloaded)}, "
            f"missing={len(missing)}; {detail}"
        )
    return [rows[key] for key in sorted(rows)], {
        "source": "binance_public_data_archive_monthly",
        "downloaded_files": len(downloaded),
        "missing_files": len(missing),
        "first_funding_ms": min(rows),
        "last_funding_ms": max(rows),
        "requested_end_date": requested_end_date.isoformat(),
        "archive_end_exclusive_utc": datetime.fromtimestamp(
            archive_end_ms / 1000, UTC
        ).isoformat(),
        "current_month_excluded": True,
    }


def fetch_market_history(symbols: tuple[str, ...], days: int = 180,
                         now: datetime | None = None) -> dict[str, Any]:
    """Fetch synchronized public Spot, perpetual and realized funding history."""
    if not symbols or days < 30 or days > 730:
        raise ValueError("symbols required and days must be between 30 and 730")
    now = now or datetime.now(UTC)
    end_ms = int(now.timestamp() * 1000)
    start_ms = end_ms - days * 24 * HOUR_MS
    session = requests.Session()
    output: dict[str, Any] = {
        "study": "spot_perpetual_funding_carry",
        "data_source": "Binance public Spot + USD-M perpetual; cross-venue proxy, not Bybit",
        "requested_days": days,
        "requested_start_utc": datetime.fromtimestamp(start_ms / 1000, UTC).isoformat(),
        "requested_end_utc": datetime.fromtimestamp(end_ms / 1000, UTC).isoformat(),
        "symbols": {},
    }
    for symbol in symbols:
        spot = _paged_klines(session, SPOT_HOSTS, "/api/v3/klines", symbol, start_ms, end_ms)
        perp_source = "binance_usdm_rest_api"
        archive_metadata: dict[str, Any] | None = None
        try:
            perp = _paged_klines(
                session, PERP_HOSTS, "/fapi/v1/klines", symbol, start_ms, end_ms
            )
        except RuntimeError as rest_error:
            try:
                perp, archive_metadata = _archive_futures_klines(
                    session, symbol, start_ms, end_ms, now=now
                )
                perp_source = "binance_public_data_archive"
                print(
                    f"{symbol}: USD-M REST unavailable; recovered {len(perp)} "
                    "hourly bars from Binance public archive",
                    flush=True,
                )
            except RuntimeError as archive_error:
                raise RuntimeError(
                    f"USD-M perpetual klines unavailable for {symbol}; "
                    f"REST failed ({rest_error}); archive fallback failed ({archive_error})"
                ) from None
        funding_source = "binance_usdm_rest_api"
        funding_archive_metadata: dict[str, Any] | None = None
        try:
            funding = _funding_history(session, symbol, start_ms, end_ms)
        except RuntimeError as rest_error:
            try:
                funding, funding_archive_metadata = _archive_funding_history(
                    session, symbol, start_ms, end_ms, now=now
                )
                funding_source = "binance_public_data_archive_monthly"
                # Align strategy and benchmark on the same effective dates.
                # The monthly archive omits the current incomplete month.
                first_price_ms = int(funding[0]["timestamp"]) + HOUR_MS
                last_price_ms = int(funding[-1]["timestamp"]) + HOUR_MS
                spot = {
                    timestamp: bar for timestamp, bar in spot.items()
                    if first_price_ms <= timestamp <= last_price_ms
                }
                perp = {
                    timestamp: bar for timestamp, bar in perp.items()
                    if first_price_ms <= timestamp <= last_price_ms
                }
                print(
                    f"{symbol}: funding REST unavailable; loaded {len(funding)} "
                    "settlements from completed monthly archive",
                    flush=True,
                )
            except RuntimeError as archive_error:
                raise RuntimeError(
                    f"funding history unavailable for {symbol}; "
                    f"REST failed ({rest_error}); archive fallback failed ({archive_error})"
                ) from None
        if not spot or not perp or len(funding) < 20:
            raise RuntimeError(f"insufficient synchronized history for {symbol}: "
                               f"spot={len(spot)} perp={len(perp)} funding={len(funding)}")
        output["symbols"][symbol] = {
            "spot_1h": spot, "perp_1h": perp, "funding": funding,
            "perp_1h_source": perp_source,
            "perp_archive": archive_metadata,
            "funding_source": funding_source,
            "funding_archive": funding_archive_metadata,
            "coverage": {
                "spot_bars": len(spot), "perp_bars": len(perp),
                "funding_events": len(funding),
                "funding_source": funding_source,
                "spot_first": min(spot), "spot_last": max(spot),
                "perp_first": min(perp), "perp_last": max(perp),
                "funding_first": int(funding[0]["timestamp"]),
                "funding_last": int(funding[-1]["timestamp"]),
            },
        }
        print(f"{symbol}: Spot {len(spot)} bars, Perp {len(perp)} bars, "
              f"funding {len(funding)} events", flush=True)
    return output


def _bar_at_or_after(bars: dict[int, dict[str, float]], timestamp: int) -> tuple[int, dict[str, float]] | None:
    candidates = [key for key in bars if key >= timestamp]
    if not candidates:
        return None
    key = min(candidates)
    if key - timestamp > HOUR_MS:
        return None
    return key, bars[key]


def _metrics(returns: list[float], initial_capital: float, elapsed_days: float) -> dict[str, Any]:
    equity = initial_capital
    peak = equity
    max_dd = 0.0
    wins = losses = 0
    gross_profit = gross_loss = 0.0
    monthly: dict[str, float] = {}
    for row in returns:
        value = row["net_return"]
        pnl = equity * value
        if value > 0:
            wins += 1
            gross_profit += pnl
        elif value < 0:
            losses += 1
            gross_loss += -pnl
        equity *= 1 + value
        peak = max(peak, equity)
        if peak:
            max_dd = max(max_dd, (peak - equity) / peak)
        month = datetime.fromtimestamp(row["exit_ms"] / 1000, UTC).strftime("%Y-%m")
        monthly[month] = monthly.get(month, 1.0) * (1 + value)
    total_return = equity / initial_capital - 1
    pf = gross_profit / gross_loss if gross_loss else ("infinite" if gross_profit else 0.0)
    return {
        "initial_equity": initial_capital, "final_equity": round(equity, 8),
        "compounded_return_pct": round(total_return * 100, 6),
        "profit_factor": round(pf, 6) if isinstance(pf, float) else pf,
        "trades": len(returns), "wins": wins, "losses": losses,
        "win_rate_pct": round(100 * wins / len(returns), 4) if returns else 0.0,
        "max_drawdown_pct": round(max_dd * 100, 6),
        "annualized_return_pct": round(((equity / initial_capital) ** (365.25 / elapsed_days) - 1) * 100, 6)
            if elapsed_days > 0 and equity > 0 else None,
        "monthly_compounded_returns_pct": {
            key: round((value - 1) * 100, 5) for key, value in sorted(monthly.items())
        },
    }


def simulate_symbol(symbol: str, data: dict[str, Any], *, annualized_entry_threshold: float,
                    costs: dict[str, float], initial_capital: float = 10_000.0) -> dict[str, Any]:
    """Long Spot + short Perp, with lagged entry signal and next-settlement cashflow."""
    if annualized_entry_threshold < 0 or initial_capital <= 0:
        raise ValueError("invalid threshold or capital")
    for key in ("commission_bps_per_fill", "half_spread_bps_per_fill", "slippage_bps_per_fill"):
        if key not in costs or not math.isfinite(float(costs[key])) or float(costs[key]) < 0:
            raise ValueError(f"invalid cost: {key}")
    spot = {int(k): v for k, v in data["spot_1h"].items()}
    perp = {int(k): v for k, v in data["perp_1h"].items()}
    funding = sorted(data["funding"], key=lambda item: int(item["timestamp"]))
    rows: list[dict[str, Any]] = []
    total_cost_bps = sum(float(costs[k]) for k in costs)
    index = 0
    while index < len(funding) - 1:
        current = funding[index]
        t0 = int(current["timestamp"])
        next_ts = int(funding[index + 1]["timestamp"])
        interval_ms = next_ts - t0
        if interval_ms <= 0 or interval_ms > 12 * HOUR_MS:
            index += 1
            continue
        # The rate settled at t0 is observable and selects only the next
        # interval. A qualifying carry position stays open across consecutive
        # qualifying intervals so entry/exit costs are paid once per holding run.
        observed_rate = float(current["rate"])
        annualized = observed_rate * (365.25 * 24 * HOUR_MS / interval_ms)
        if annualized < annualized_entry_threshold:
            index += 1
            continue
        entry = _bar_at_or_after(spot, t0 + HOUR_MS)
        entry_perp = _bar_at_or_after(perp, t0 + HOUR_MS)
        if not entry or not entry_perp:
            index += 1
            continue
        entry_ms, sb0 = entry
        entry_perp_ms, pb0 = entry_perp
        if entry_ms != entry_perp_ms:
            index += 1
            continue
        spot_entry, perp_entry = sb0["open"], pb0["open"]
        if min(spot_entry, perp_entry) <= 0:
            index += 1
            continue

        # Stay in the hedge until the newly settled rate no longer meets the
        # predeclared threshold. The rate at each later settlement is credited
        # for the interval just completed before deciding whether to continue.
        exit_index = index + 1
        funding_return = 0.0
        while exit_index < len(funding):
            settled = funding[exit_index]
            settled_ts = int(settled["timestamp"])
            funding_mark = _bar_at_or_after(perp, settled_ts)
            if not funding_mark or funding_mark[1]["open"] <= 0:
                raise RuntimeError(f"missing valid perpetual mark at funding settlement for {symbol}")
            funding_return += float(settled["rate"]) * (funding_mark[1]["open"] / spot_entry)
            if exit_index == len(funding) - 1:
                break
            following_ts = int(funding[exit_index + 1]["timestamp"])
            following_interval = following_ts - settled_ts
            if following_interval <= 0 or following_interval > 12 * HOUR_MS:
                break
            next_annualized = float(settled["rate"]) * (
                365.25 * 24 * HOUR_MS / following_interval
            )
            if next_annualized < annualized_entry_threshold:
                break
            exit_index += 1

        exit_ts = int(funding[exit_index]["timestamp"])
        exit_ = _bar_at_or_after(spot, exit_ts + HOUR_MS)
        exit_perp = _bar_at_or_after(perp, exit_ts + HOUR_MS)
        if not exit_ or not exit_perp:
            index = exit_index + 1
            continue
        exit_ms, sb1 = exit_
        exit_perp_ms, pb1 = exit_perp
        if exit_ms != exit_perp_ms:
            index = exit_index + 1
            continue
        spot_exit, perp_exit = sb1["open"], pb1["open"]
        if min(spot_exit, perp_exit) <= 0:
            index = exit_index + 1
            continue

        # Equal base quantity on both legs; normalize actual price PnL by
        # initial Spot notional. This captures basis drift without assuming
        # Spot and Perp entry prices are identical.
        basis_pnl_fraction = ((spot_exit - spot_entry) - (perp_exit - perp_entry)) / spot_entry
        # Four executions per complete holding run: Spot buy/sell and Perp sell/buy.
        traded_notionals_over_initial = (spot_entry + spot_exit + perp_entry + perp_exit) / spot_entry
        cost_fraction = traded_notionals_over_initial * total_cost_bps / 10_000
        net_return = basis_pnl_fraction + funding_return - cost_fraction
        rows.append({
            "symbol": symbol, "entry_ms": entry_ms, "exit_ms": exit_ms,
            "observed_funding_rate": observed_rate,
            "realized_next_funding_rate": float(funding[index + 1]["rate"]),
            "realized_funding_return_pct": funding_return * 100,
            "funding_events_held": exit_index - index,
            "observed_annualized_rate_pct": annualized * 100,
            "basis_return_pct": basis_pnl_fraction * 100,
            "funding_return_pct": funding_return * 100,
            "cost_pct": cost_fraction * 100,
            "net_return": net_return,
            "net_return_pct": net_return * 100,
            "holding_hours": (exit_ms - entry_ms) / HOUR_MS,
        })
        index = exit_index + 1
    elapsed_days = 0.0
    if funding:
        elapsed_days = max(0.0, (int(funding[-1]["timestamp"]) -
                                  int(funding[0]["timestamp"])) / (24 * HOUR_MS))
    summary = _metrics(rows, initial_capital, elapsed_days)
    spot_times = sorted(spot)
    benchmark_return = (spot[spot_times[-1]]["close"] / spot[spot_times[0]]["close"] - 1) * 100 if len(spot_times) > 1 else None
    summary.update({
        "spot_buy_and_hold_return_pct": round(benchmark_return, 6) if benchmark_return is not None else None,
        "symbol": symbol,
        "entry_threshold_annualized_pct": annualized_entry_threshold * 100,
        "costs_per_fill_bps": costs,
        "trade_log": rows,
        "limitations": [
            "Binance Spot and USD-M perpetual are proxies, not Bybit fills or account fees.",
            "Entry uses the prior settled funding rate; next settlement rate is credited only after holding through settlement.",
            "Hourly opens approximate execution; spread, market impact, queue, partial fills and liquidation paths are not observed.",
            "Positive funding only; no short-Spot/long-Perp reverse carry or borrow costs.",
            "Funding basis is not risk-free; collateral, margin, liquidation and exchange risk require separate validation.",
        ],
    })
    return summary


def run_study(history: dict[str, Any], *, threshold: float = 0.10,
              initial_capital: float = 10_000.0) -> dict[str, Any]:
    if not 0 <= threshold <= 5:
        raise ValueError("threshold must be a decimal annual rate between 0 and 5")
    scenarios = {
        "realistic": {"commission_bps_per_fill": 10.0, "half_spread_bps_per_fill": 1.0,
                      "slippage_bps_per_fill": 2.0},
        "conservative": {"commission_bps_per_fill": 15.0, "half_spread_bps_per_fill": 3.0,
                         "slippage_bps_per_fill": 5.0},
    }
    reports: dict[str, Any] = {}
    for scenario, costs in scenarios.items():
        reports[scenario] = {
            symbol: simulate_symbol(symbol, data, annualized_entry_threshold=threshold,
                                     costs=costs, initial_capital=initial_capital)
            for symbol, data in history["symbols"].items()
        }
    data_coverage: dict[str, dict[str, Any]] = {}
    for symbol, data in history["symbols"].items():
        coverage = data.get("coverage", {})
        data_coverage[symbol] = {
            "spot_hourly_bars": coverage.get("spot_bars", len(data.get("spot_1h", {}))),
            "perpetual_hourly_bars": coverage.get("perp_bars", len(data.get("perp_1h", {}))),
            "funding_events": coverage.get("funding_events", len(data.get("funding", []))),
            "spot_first_utc": (
                datetime.fromtimestamp(int(coverage["spot_first"]) / 1000, UTC).isoformat()
                if coverage.get("spot_first") is not None else None
            ),
            "spot_last_utc": (
                datetime.fromtimestamp(int(coverage["spot_last"]) / 1000, UTC).isoformat()
                if coverage.get("spot_last") is not None else None
            ),
            "perpetual_first_utc": (
                datetime.fromtimestamp(int(coverage["perp_first"]) / 1000, UTC).isoformat()
                if coverage.get("perp_first") is not None else None
            ),
            "perpetual_last_utc": (
                datetime.fromtimestamp(int(coverage["perp_last"]) / 1000, UTC).isoformat()
                if coverage.get("perp_last") is not None else None
            ),
            "funding_first_utc": (
                datetime.fromtimestamp(int(coverage["funding_first"]) / 1000, UTC).isoformat()
                if coverage.get("funding_first") is not None else None
            ),
            "funding_last_utc": (
                datetime.fromtimestamp(int(coverage["funding_last"]) / 1000, UTC).isoformat()
                if coverage.get("funding_last") is not None else None
            ),
            "perpetual_source": data.get("perp_1h_source", "unknown"),
            "funding_source": data.get("funding_source", "unknown"),
            "perpetual_archive": data.get("perp_archive"),
            "funding_archive": data.get("funding_archive"),
        }
    return {
        "study": "spot_perpetual_funding_carry",
        "data_source": history.get("data_source"),
        "requested_start_utc": history.get("requested_start_utc"),
        "requested_end_utc": history.get("requested_end_utc"),
        "data_coverage": data_coverage,
        "entry_rule": "use previous settled funding rate; enter after settlement; exit after next settlement",
        "threshold_annualized_rate_pct": threshold * 100,
        "initial_capital_per_symbol": initial_capital,
        "cost_model": "four fills per round trip: Spot buy/sell and Perp sell/buy",
        "scenarios": reports,
        "decision": "research_only_no_demo_approval",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded spot/perpetual funding carry research screen.")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="minimum prior settled funding rate annualized as a decimal (0.10 = 10%%)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/funding_carry.json"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        history = fetch_market_history(tuple(args.symbols), days=args.days)
        report = run_study(history, threshold=args.threshold)
    except Exception as exc:
        # Keep a machine-readable diagnostic when a public endpoint is blocked.
        # An empty/missing report must never be mistaken for a completed study.
        report = {
            "study": "spot_perpetual_funding_carry",
            "status": "data_unavailable",
            "decision": "no_strategy_result_no_demo_approval",
            "requested_days": args.days,
            "symbols": list(args.symbols),
            "threshold_annualized_rate": args.threshold,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "note": "No backtest was run because synchronized market history was unavailable.",
        }
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"market-data fetch failed; wrote diagnostic artifact to {args.out}: {exc}", flush=True)
        return 2

    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for scenario, symbols in report["scenarios"].items():
        print(f"\n{scenario.upper()}")
        for symbol, metrics in symbols.items():
            print(f"{symbol}: {metrics['compounded_return_pct']:.4f}% "
                  f"PF={metrics['profit_factor']} trades={metrics['trades']} "
                  f"DD={metrics['max_drawdown_pct']:.3f}%")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
