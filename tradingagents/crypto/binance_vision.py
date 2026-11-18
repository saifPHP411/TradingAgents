"""Read-only Binance USD-M historical archive adapter for research.

The public Vision archive is independent of the geographically restricted
futures REST API. This adapter reads monthly ZIP archives for traded candles,
mark-price candles, and realized funding events. It never submits orders.

Venue quantity filters are deliberately permissive research placeholders:
the archive contains market history, not the live exchangeInfo filter snapshot.
Results using this adapter are not proof of executable order sizing.
"""
from __future__ import annotations

import io
import math
import re
import zipfile
from collections.abc import Callable
from urllib.parse import urlsplit

import pandas as pd

from tradingagents.crypto.perpetuals import LinearPerpetualInstrument

BASE_URL = "https://data.binance.vision"
ALLOWED_HOSTS = {"data.binance.vision"}
INTERVAL = "4h"
INTERVAL_MS = 4 * 60 * 60 * 1000
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{3,30}USDT$")


class BinanceVisionDataError(RuntimeError):
    """A Binance historical archive could not be read or validated."""


class BinanceVisionPublicClient:
    """Fetch validated monthly USD-M archive files through an injectable seam."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        transport: Callable[..., bytes] | None = None,
        timeout: float = 30.0,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in ALLOWED_HOSTS
            or parsed.port not in (None, 443)
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError("only the official Binance Vision HTTPS archive is allowed")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.timeout = timeout
        self._archive_cache: dict[str, bytes] = {}

    @staticmethod
    def _symbol(symbol: str) -> str:
        value = str(symbol).strip().upper()
        if not SYMBOL_PATTERN.fullmatch(value):
            raise ValueError("only uppercase USDT-margined perpetual symbols are supported")
        return value

    @staticmethod
    def _validate_range(start_ms: int, end_ms: int) -> None:
        if (
            isinstance(start_ms, bool)
            or isinstance(end_ms, bool)
            or not isinstance(start_ms, int)
            or not isinstance(end_ms, int)
            or start_ms <= 0
            or end_ms <= start_ms
        ):
            raise ValueError("start_ms/end_ms must be positive ordered integers")

    def _download(self, path: str) -> bytes:
        if path in self._archive_cache:
            return self._archive_cache[path]
        url = self.base_url + "/" + path.lstrip("/")
        try:
            if self.transport is not None:
                content = self.transport(url=url, timeout=self.timeout)
            else:
                import requests

                response = requests.get(url, timeout=self.timeout)
                if response.status_code != 200:
                    raise BinanceVisionDataError(
                        f"Binance Vision archive returned HTTP {response.status_code}"
                    )
                content = response.content
        except BinanceVisionDataError:
            raise
        except Exception as exc:
            raise BinanceVisionDataError(
                f"could not download Binance Vision archive: {type(exc).__name__}"
            ) from None
        if not isinstance(content, bytes) or not content:
            raise BinanceVisionDataError("Binance Vision returned an empty archive")
        self._archive_cache[path] = content
        return content

    def _csv_bytes(self, path: str) -> bytes:
        content = self._download(path)
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if len(names) != 1:
                    raise BinanceVisionDataError(
                        "archive must contain exactly one CSV data file"
                    )
                return archive.read(names[0])
        except (zipfile.BadZipFile, OSError, KeyError):
            raise BinanceVisionDataError("Binance Vision archive is not a valid ZIP/CSV") from None

    @staticmethod
    def _months(start_ms: int, end_ms: int) -> list[str]:
        first = pd.Timestamp(start_ms, unit="ms", tz="UTC").to_period("M")
        last = pd.Timestamp(end_ms - 1, unit="ms", tz="UTC").to_period("M")
        return [str(period) for period in pd.period_range(first, last, freq="M")]

    def _monthly_path(
        self, dataset: str, symbol: str, month: str, interval: str | None = None
    ) -> str:
        year, number = month.split("-")
        if interval is None:
            filename = f"{symbol}-{dataset}-{year}-{number}.zip"
            return f"data/futures/um/monthly/{dataset}/{symbol}/{filename}"
        filename = f"{symbol}-{interval}-{year}-{number}.zip"
        return (
            f"data/futures/um/monthly/{dataset}/{symbol}/"
            f"{interval}/{filename}"
        )

    @staticmethod
    def _parse_candle_csv(
        csv_data: bytes,
        market: str,
        dataset: str,
        *,
        include_volume: bool,
    ) -> pd.DataFrame:
        """Parse both headered and legacy headerless Binance candle CSV files."""
        raw = pd.read_csv(io.BytesIO(csv_data))
        names = {
            str(column).strip().lower().lstrip(chr(65279)): column
            for column in raw.columns
        }
        if {"open_time", "open", "high", "low", "close"}.issubset(names):
            selected = pd.DataFrame(
                {
                    "timestamp": raw[names["open_time"]],
                    "open": raw[names["open"]],
                    "high": raw[names["high"]],
                    "low": raw[names["low"]],
                    "close": raw[names["close"]],
                }
            )
            if "close_time" in names:
                selected["close_time"] = raw[names["close_time"]]
            else:
                selected["close_time"] = (
                    pd.to_numeric(selected["timestamp"], errors="raise")
                    + INTERVAL_MS - 1
                )
            if include_volume and "volume" in names:
                selected["volume"] = raw[names["volume"]]
            elif include_volume:
                raise BinanceVisionDataError(
                    f"{market} traded kline CSV omitted volume"
                )
        else:
            raw = pd.read_csv(io.BytesIO(csv_data), header=None)
            if raw.shape[1] < 7:
                raise BinanceVisionDataError(
                    f"{market} {dataset} CSV has too few columns"
                )
            selected = raw.iloc[:, [0, 1, 2, 3, 4, 5, 6]].copy()
            selected.columns = [
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time",
            ]
            if not include_volume:
                selected = selected.drop(columns=["volume"])
        for column in selected.columns:
            selected[column] = pd.to_numeric(selected[column], errors="raise")
        return selected

    def _daily_path(self, dataset: str, symbol: str, day: str) -> str:
        filename = f"{symbol}-{INTERVAL}-{day}.zip"
        return (
            f"data/futures/um/daily/{dataset}/{symbol}/"
            f"{INTERVAL}/{filename}"
        )

    def instrument_info(self, symbol: str) -> LinearPerpetualInstrument:
        market = self._symbol(symbol)
        # Historical archive files do not contain the exchange's live lot-size,
        # tick-size or leverage-bracket snapshot. These values allow the research
        # simulator to evaluate fractional sizing; they are not trading filters.
        return LinearPerpetualInstrument(
            symbol=market,
            base_coin=market.removesuffix("USDT"),
            quote_coin="USDT",
            settle_coin="USDT",
            status="Trading",
            contract_type="LinearPerpetual",
            min_quantity=1e-8,
            max_market_quantity=1_000_000_000.0,
            quantity_step=1e-8,
            min_notional=5.0,
            tick_size=1e-8,
            min_leverage=1.0,
            max_leverage=2.0,
            leverage_step=1.0,
            funding_interval_minutes=480,
        )

    def _candles(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        dataset: str,
        *,
        include_volume: bool,
    ) -> pd.DataFrame:
        market = self._symbol(symbol)
        self._validate_range(start_ms, end_ms)
        frames = []
        for month in self._months(start_ms, end_ms):
            path = self._monthly_path(dataset, market, month, INTERVAL)
            selected = self._parse_candle_csv(
                self._csv_bytes(path),
                market,
                dataset,
                include_volume=include_volume,
            )
            selected = selected[
                (selected["timestamp"] >= start_ms)
                & (selected["timestamp"] < end_ms)
                & (selected["close_time"] < end_ms)
            ]
            if selected.empty:
                continue
            selected["timestamp"] = selected["timestamp"].astype("int64")
            frames.append(selected)

        if not frames:
            raise BinanceVisionDataError(
                f"no {dataset} archive candles returned for {market}"
            )
        frame = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        expected = pd.date_range(
            start=pd.to_datetime(start_ms, unit="ms", utc=True),
            end=pd.to_datetime(end_ms, unit="ms", utc=True),
            freq="4h",
            inclusive="left",
        )
        present = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        missing = expected.difference(present)
        if not missing.empty:
            # Binance occasionally publishes a monthly mark-price file with
            # isolated missing sessions. Recover only those missing dates from
            # the corresponding official daily archives; never synthesize prices.
            recovered = []
            for day in sorted({stamp.strftime("%Y-%m-%d") for stamp in missing}):
                daily_path = self._daily_path(dataset, market, day)
                try:
                    daily = self._parse_candle_csv(
                        self._csv_bytes(daily_path),
                        market,
                        dataset,
                        include_volume=include_volume,
                    )
                except BinanceVisionDataError:
                    continue
                daily = daily[
                    daily["timestamp"].isin(
                        [stamp.value // 1_000_000 for stamp in missing]
                    )
                ]
                if not daily.empty:
                    recovered.append(daily)
            if recovered:
                frame = pd.concat([frame, *recovered], ignore_index=True)
                frame = frame.sort_values("timestamp")
            present = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
            missing = expected.difference(present)
            if not missing.empty:
                raise BinanceVisionDataError(
                    f"{market} {dataset} archive is missing completed 4-hour bars "
                    f"after monthly and daily recovery: {list(missing[:6])}"
                )
        if frame["timestamp"].duplicated().any():
            raise BinanceVisionDataError(f"duplicate {dataset} timestamps for {market}")
        if (frame[["open", "high", "low", "close"]] <= 0).any().any():
            raise BinanceVisionDataError(f"non-positive {dataset} OHLC value for {market}")
        if (frame["high"] < frame["low"]).any():
            raise BinanceVisionDataError(f"invalid {dataset} high/low range for {market}")
        index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        columns = {
            "open": frame["open"].astype(float).to_numpy(),
            "high": frame["high"].astype(float).to_numpy(),
            "low": frame["low"].astype(float).to_numpy(),
            "close": frame["close"].astype(float).to_numpy(),
        }
        if include_volume:
            columns["volume"] = frame["volume"].astype(float).to_numpy()
        return pd.DataFrame(columns, index=index)

    def klines(self, symbol: str, *, start_ms: int, end_ms: int) -> pd.DataFrame:
        return self._candles(
            symbol, start_ms, end_ms, "klines", include_volume=True
        )

    def mark_price_klines(
        self, symbol: str, *, start_ms: int, end_ms: int
    ) -> pd.DataFrame:
        return self._candles(
            symbol, start_ms, end_ms, "markPriceKlines", include_volume=False
        )

    def funding_rates(
        self,
        symbol: str,
        *,
        start_ms: int,
        end_ms: int,
        interval_minutes: int = 480,
    ) -> pd.Series:
        market = self._symbol(symbol)
        self._validate_range(start_ms, end_ms)
        if (
            isinstance(interval_minutes, bool)
            or not isinstance(interval_minutes, int)
            or not 1 <= interval_minutes <= 480
        ):
            raise ValueError("interval_minutes must be an integer in [1, 480]")

        frames = []
        for month in self._months(start_ms, end_ms):
            path = self._monthly_path("fundingRate", market, month)
            raw = pd.read_csv(io.BytesIO(self._csv_bytes(path)))
            columns = {str(column).strip().lower(): column for column in raw.columns}
            if "calc_time" not in columns or "last_funding_rate" not in columns:
                raw = pd.read_csv(
                    io.BytesIO(self._csv_bytes(path)),
                    header=None,
                    names=["calc_time", "funding_interval_hours", "last_funding_rate"],
                )
                columns = {str(column).strip().lower(): column for column in raw.columns}
            if "calc_time" not in columns or "last_funding_rate" not in columns:
                raise BinanceVisionDataError(f"{market} funding CSV schema is invalid")
            if "funding_interval_hours" in columns:
                part = raw[
                    [
                        columns["calc_time"],
                        columns["funding_interval_hours"],
                        columns["last_funding_rate"],
                    ]
                ].copy()
            else:
                part = raw[
                    [columns["calc_time"], columns["last_funding_rate"]]
                ].copy()
                part["funding_interval_hours"] = interval_minutes / 60
            part.columns = [
                "timestamp", "funding_interval_hours", "funding_rate"
            ]
            part["timestamp"] = pd.to_numeric(part["timestamp"], errors="raise")
            part["funding_interval_hours"] = pd.to_numeric(
                part["funding_interval_hours"], errors="raise"
            )
            part["funding_rate"] = pd.to_numeric(part["funding_rate"], errors="raise")
            if (
                not part["funding_interval_hours"].map(math.isfinite).all()
                or (part["funding_interval_hours"] <= 0).any()
                or (part["funding_interval_hours"] > 8).any()
            ):
                raise BinanceVisionDataError(
                    f"invalid historical funding interval for {market}"
                )
            part = part[
                (part["timestamp"] >= start_ms) & (part["timestamp"] <= end_ms)
            ]
            if not part.empty:
                frames.append(part)

        if not frames:
            raise BinanceVisionDataError(f"no funding archive records returned for {market}")
        frame = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        if frame["timestamp"].duplicated().any():
            raise BinanceVisionDataError(f"duplicate funding timestamps for {market}")
        if not frame["funding_rate"].map(math.isfinite).all():
            raise BinanceVisionDataError(f"non-finite funding rate for {market}")
        raw_index = pd.to_datetime(
            frame["timestamp"].astype("int64"), unit="ms", utc=True
        )
        # Funding settlement timestamps can contain a few milliseconds of
        # clock jitter. The 4h simulator can apply a payment only at a bar open,
        # so normalize sub-second differences and reject larger deviations.
        index = raw_index.round("4h")
        jitter = (raw_index - index).to_series().abs()
        if (jitter > pd.Timedelta(seconds=1)).any():
            raise BinanceVisionDataError(
                f"funding settlement for {market} does not align to the 4h "
                "candle grid within one second"
            )
        if index.has_duplicates:
            raise BinanceVisionDataError(
                f"multiple funding settlements map to one 4h candle for {market}"
            )
        rates = pd.Series(
            frame["funding_rate"].astype(float).to_numpy(),
            index=index,
            name="funding_rate",
        )
        interval_minutes_by_event = pd.Series(
            (frame["funding_interval_hours"].astype(float) * 60).round().astype(int).to_numpy(),
            index=index,
            name="funding_interval_minutes",
        )
        rates.attrs["funding_interval_minutes"] = interval_minutes_by_event
        if len(rates) > 1:
            actual_gaps = rates.index.to_series().diff().dropna().reset_index(drop=True)
            previous_intervals = interval_minutes_by_event.iloc[:-1].reset_index(drop=True)
            next_intervals = interval_minutes_by_event.iloc[1:].reset_index(drop=True)
            expected_previous = pd.to_timedelta(previous_intervals, unit="m")
            expected_next = pd.to_timedelta(next_intervals, unit="m")
            valid = (actual_gaps == expected_previous) | (actual_gaps == expected_next)
            if not valid.all():
                bad = actual_gaps[~valid].head(3)
                raise BinanceVisionDataError(
                    f"funding archive has missing or irregular settlement events for "
                    f"{market}: {list(bad)}"
                )
        return rates


__all__ = ["BinanceVisionDataError", "BinanceVisionPublicClient"]
