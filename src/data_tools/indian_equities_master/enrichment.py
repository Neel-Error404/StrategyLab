from __future__ import annotations

import datetime as dt
import logging
import math
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yfinance as yf

from .config import EnrichConfig


LOGGER = logging.getLogger(__name__)

COLUMN_ORDER = [
    # Identification
    "symbol",
    "exchange",
    "currency",
    "listing_status",
    "first_trade_date",
    # Descriptive
    "long_name",
    "short_name",
    "sector",
    "industry",
    # Market structure
    "instrument_type",
    "lot_size",
    "tick_size",
    "market_state",
    "trading_session",
    "exchange_timezone",
    # Fundamentals
    "market_cap",
    "enterprise_value",
    "shares_outstanding",
    "book_value",
    "beta",
    "trailing_pe",
    "forward_pe",
    "price_to_book",
    "dividend_yield",
    "profit_margins",
    # Daily metrics
    "regular_market_price",
    "regular_market_previous_close",
    "regular_market_open",
    "regular_market_day_high",
    "regular_market_day_low",
    "regular_market_volume",
    "regular_market_change_percent",
    "fifty_two_week_high",
    "fifty_two_week_low",
    # Derived flags
    "is_fno_eligible",
    "is_etf",
    "is_psu",
    "has_options_chain",
    "data_quality_score",
    "source_timestamp",
]

FUNDAMENTAL_KEYS = [
    "market_cap",
    "enterprise_value",
    "shares_outstanding",
    "book_value",
    "beta",
    "trailing_pe",
    "forward_pe",
    "price_to_book",
    "dividend_yield",
    "profit_margins",
]

DAILY_KEYS = [
    "regular_market_price",
    "regular_market_previous_close",
    "regular_market_open",
    "regular_market_day_high",
    "regular_market_day_low",
    "regular_market_volume",
    "regular_market_change_percent",
    "fifty_two_week_high",
    "fifty_two_week_low",
]


def build_dataset(
    listings: List[Dict[str, object]],
    config: EnrichConfig,
) -> pd.DataFrame:
    """
    Enrich the raw Yahoo screener quotes with fundamentals and produce a DataFrame.
    """
    symbol_to_listing = {item["symbol"]: item for item in listings if "symbol" in item}
    symbols = list(symbol_to_listing.keys())
    rows: List[Dict[str, object]] = []

    for batch_symbols in _chunk(symbols, config.batch_size):
        LOGGER.info("Enriching batch of %s tickers", len(batch_symbols))
        tickers = yf.Tickers(" ".join(batch_symbols))
        for symbol in batch_symbols:
            listing = symbol_to_listing[symbol]
            ticker = tickers.tickers.get(symbol)
            row = _compose_row(symbol, listing, ticker, config)
            rows.append(row)

        time.sleep(config.request_pause_seconds)

    frame = pd.DataFrame(rows)
    # Enforce column ordering and ensure missing columns exist
    for column in COLUMN_ORDER:
        if column not in frame.columns:
            frame[column] = None

    frame = frame[COLUMN_ORDER]
    frame.sort_values(by=["symbol", "exchange"], inplace=True, ignore_index=True)
    return frame


def _chunk(symbols: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(symbols), size):
        yield symbols[idx : idx + size]


def _compose_row(
    symbol: str,
    listing: Dict[str, object],
    ticker: Optional[yf.ticker.Ticker],
    config: EnrichConfig,
) -> Dict[str, object]:
    info: Dict[str, object] = {}
    fast_info: Dict[str, object] = {}
    options_flag: Optional[bool] = None

    if ticker is not None:
        info = _safe_get_info(ticker)
        fast_info = _safe_get_fast_info(ticker)
        if config.enable_options_lookup:
            options_flag = _check_options_chain(ticker, config.max_retries)

    row: Dict[str, object] = {
        # Identification
        "symbol": symbol,
        "exchange": listing.get("fullExchangeName") or listing.get("exchange"),
        "currency": listing.get("currency") or info.get("currency"),
        "listing_status": _derive_listing_status(listing.get("marketState")),
        "first_trade_date": _epoch_to_date(
            info.get("firstTradeDateEpochUtc")
            or info.get("firstTradeDateTimestamp")
            or info.get("firstTradeDateEpoch")
        ),
        # Descriptive
        "long_name": info.get("longName") or listing.get("longName"),
        "short_name": listing.get("shortName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        # Market structure
        "instrument_type": listing.get("quoteType"),
        "lot_size": info.get("sharesPerLot") or info.get("lotSize"),
        "tick_size": info.get("tickSize"),
        "market_state": listing.get("marketState"),
        "trading_session": listing.get("quoteSourceName"),
        "exchange_timezone": listing.get("exchangeTimezoneName") or info.get("exchangeTimezoneShortName"),
        # Fundamentals
        "market_cap": _safe_number(info.get("marketCap")),
        "enterprise_value": _safe_number(info.get("enterpriseValue")),
        "shares_outstanding": _safe_number(info.get("sharesOutstanding")),
        "book_value": _safe_number(info.get("bookValue")),
        "beta": _safe_number(info.get("beta")),
        "trailing_pe": _safe_number(info.get("trailingPE")),
        "forward_pe": _safe_number(info.get("forwardPE")),
        "price_to_book": _safe_number(info.get("priceToBook")),
        "dividend_yield": _safe_number(info.get("dividendYield")),
        "profit_margins": _safe_number(info.get("profitMargins")),
        # Daily metrics (prefer fast info but fall back to listing)
        "regular_market_price": _safe_number(
            fast_info.get("last_price") or listing.get("regularMarketPrice")
        ),
        "regular_market_previous_close": _safe_number(
            fast_info.get("previous_close") or listing.get("regularMarketPreviousClose")
        ),
        "regular_market_open": _safe_number(
            fast_info.get("open") or listing.get("regularMarketOpen")
        ),
        "regular_market_day_high": _safe_number(
            fast_info.get("day_high") or listing.get("regularMarketDayHigh")
        ),
        "regular_market_day_low": _safe_number(
            fast_info.get("day_low") or listing.get("regularMarketDayLow")
        ),
        "regular_market_volume": _safe_number(
            fast_info.get("last_volume") or listing.get("regularMarketVolume")
        ),
        "regular_market_change_percent": _safe_number(
            listing.get("regularMarketChangePercent")
        ),
        "fifty_two_week_high": _safe_number(
            info.get("fiftyTwoWeekHigh") or listing.get("fiftyTwoWeekHigh")
        ),
        "fifty_two_week_low": _safe_number(
            info.get("fiftyTwoWeekLow") or listing.get("fiftyTwoWeekLow")
        ),
        # Derived flags
        "is_fno_eligible": info.get("fnoEligible"),
        "is_etf": listing.get("quoteType") == "ETF",
        "is_psu": info.get("governmentStake") is not None or info.get("isPublicSectorUndertaking"),
        "has_options_chain": options_flag,
        "data_quality_score": None,  # placeholder; computed later
        "source_timestamp": _source_timestamp(listing),
    }

    row["data_quality_score"] = _compute_quality_score(row)
    return row


def _safe_get_info(ticker: yf.ticker.Ticker) -> Dict[str, object]:
    try:
        info = ticker.get_info()
        if isinstance(info, dict):
            return info
    except Exception as exc:  # noqa: BLE001 - upstream network issues
        LOGGER.debug("ticker.get_info failed for %s: %s", ticker.ticker, exc)
    return {}


def _safe_get_fast_info(ticker: yf.ticker.Ticker) -> Dict[str, object]:
    try:
        fast = ticker.fast_info
        if hasattr(fast, "items"):
            return dict(fast.items())
        if isinstance(fast, dict):
            return fast
        if hasattr(fast, "__dict__"):
            return dict(fast.__dict__)
    except Exception as exc:  # noqa: BLE001 - upstream network issues
        LOGGER.debug("ticker.fast_info failed for %s: %s", ticker.ticker, exc)
    return {}


def _check_options_chain(ticker: yf.ticker.Ticker, max_retries: int) -> Optional[bool]:
    for attempt in range(1, max_retries + 1):
        try:
            options = ticker.options
            return bool(options)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Options lookup failed attempt=%s for %s: %s", attempt, ticker.ticker, exc)
            time.sleep(1.0 * attempt)
    return None


def _derive_listing_status(market_state: Optional[str]) -> Optional[str]:
    if not market_state:
        return None
    market_state = str(market_state).upper()
    if market_state in {"REGULAR", "PRE", "POST", "CLOSED"}:
        return "active"
    if market_state in {"HALT", "DELAYED"}:
        return "suspended"
    if market_state in {"POSTPOST", "PREPRE"}:
        return "inactive"
    return market_state.lower()


def _epoch_to_date(value: Optional[object]) -> Optional[str]:
    if value in (None, "", 0):
        return None
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None

    if ts > 1e12:  # handle milliseconds
        ts /= 1000

    try:
        return dt.datetime.utcfromtimestamp(ts).date().isoformat()
    except (OverflowError, ValueError):
        return None


def _safe_number(value: Optional[object]) -> Optional[float]:
    if value in (None, "", "na", "NA", "None"):
        return None
    try:
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return None
        return num
    except (TypeError, ValueError):
        try:
            num = int(value)  # type: ignore[arg-type]
            return float(num)
        except (TypeError, ValueError):
            return None


def _source_timestamp(listing: Dict[str, object]) -> str:
    # Prefer Yahoo market timestamp, fallback to current UTC time.
    market_time = listing.get("regularMarketTime")
    if isinstance(market_time, (int, float)) and market_time > 0:
        try:
            return dt.datetime.utcfromtimestamp(float(market_time)).replace(microsecond=0).isoformat() + "Z"
        except (OverflowError, ValueError):
            pass
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _compute_quality_score(row: Dict[str, object]) -> float:
    fields = FUNDAMENTAL_KEYS + DAILY_KEYS
    available = sum(1 for key in fields if row.get(key) is not None)
    return round(available / len(fields), 3)
