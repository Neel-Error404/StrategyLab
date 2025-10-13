from __future__ import annotations

import logging
import time
from typing import Dict, List, Set

import yfinance as yf
from yfinance import EquityQuery

from .config import ScreenerConfig


LOGGER = logging.getLogger(__name__)

EXCHANGE_MAP = {
    "NSE": "NSI",
    "NSI": "NSI",
    "BSE": "BSE",
    "BOMBAY": "BSE",
}


def fetch_listings(config: ScreenerConfig) -> List[Dict[str, object]]:
    """
    Use yfinance's screener API to list Indian equities + ETFs.

    Fetches all available listings by paginating through results until
    no more data is available.
    """
    query = _build_query(config)

    listings: List[Dict[str, object]] = []
    seen: Set[str] = set()
    offset = 0
    allowed_types = {qt.upper() for qt in config.quote_types}
    consecutive_empty_pages = 0  # Track empty pages to detect end of results

    while True:
        try:
            result = yf.screen(
                query,
                offset=offset,
                count=min(config.page_size, 250),
                sortField="ticker",
                sortAsc=True,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if "too many requests" in message or "rate limit" in message:
                LOGGER.warning("Yahoo rate limit hit, sleeping before retry: %s", exc)
                time.sleep(2.0)
                continue
            raise RuntimeError(f"Yahoo screener request failed: {exc}") from exc

        quotes = result.get("quotes", [])
        LOGGER.info(
            "Fetched %s quotes at offset=%s (total so far=%s)",
            len(quotes),
            offset,
            len(listings),
        )

        # Check if we got zero results
        if not quotes or len(quotes) == 0:
            consecutive_empty_pages += 1
            LOGGER.warning(
                "Received empty page at offset=%s (consecutive empty: %s)",
                offset,
                consecutive_empty_pages
            )

            # If we got 2 consecutive empty pages, we're done
            if consecutive_empty_pages >= 2:
                LOGGER.info(
                    "Received %s consecutive empty pages, stopping pagination",
                    consecutive_empty_pages
                )
                break

            # Otherwise increment offset and try next page
            offset += min(config.page_size, 250)
            time.sleep(0.5)
            continue

        # Reset empty page counter when we get results
        consecutive_empty_pages = 0

        new_items = 0
        for quote in quotes:
            symbol = quote.get("symbol")
            exchange = quote.get("fullExchangeName") or quote.get("exchange")
            if not symbol or not exchange:
                continue

            # Filter out test symbols
            if any(keyword in symbol.upper() for keyword in ["TEST", "DUMMY", "SAMPLE"]):
                LOGGER.debug("Skipping test symbol: %s", symbol)
                continue

            quote_type = (quote.get("quoteType") or "").upper()
            if allowed_types and quote_type not in allowed_types:
                continue
            if symbol in seen:
                continue
            listings.append(quote)
            seen.add(symbol)
            new_items += 1

        # Always increment offset by number of quotes received
        offset += len(quotes)

        # Stop if we've reached max pages safety limit
        if offset >= config.page_size * config.max_pages:
            LOGGER.warning(
                "Stopping pagination after %s pages (safety guard).", config.max_pages
            )
            break

        time.sleep(0.5)

    if not listings:
        raise RuntimeError("Yahoo screener returned zero listings for the query.")

    LOGGER.info("Total unique listings discovered: %s", len(listings))

    # Validate and filter results
    valid_listings = _validate_listings(listings)
    LOGGER.info("Valid listings after filtering: %s", len(valid_listings))

    return valid_listings


def _build_query(config: ScreenerConfig) -> EquityQuery:
    """
    Build Yahoo Finance screener query for Indian equities.

    Returns a query that filters for:
    - Region: India (IN)
    - Exchanges: NSE and/or BSE
    - Quote types: EQUITY and/or ETF
    - Market cap > 10 million INR to exclude penny stocks and test symbols
    """
    operands = []

    # Region filter
    region = config.region.lower()
    operands.append(EquityQuery("EQ", ["region", region]))

    # Exchange filter
    if config.exchanges:
        mapped = []
        for exchange in config.exchanges:
            code = EXCHANGE_MAP.get(exchange.upper())
            if code:
                mapped.append(code)
            else:
                LOGGER.warning("Unknown exchange '%s' for Yahoo screener.", exchange)
        if mapped:
            operands.append(EquityQuery("IS-IN", ["exchange", *mapped]))

    # Market cap filter to exclude penny stocks and test symbols
    # Market cap > 10 million INR (~$120k USD)
    try:
        operands.append(EquityQuery("GT", ["marketcap", 10000000]))
        LOGGER.debug("Added market cap filter: > 10M INR")
    except Exception as exc:
        LOGGER.warning("Could not add market cap filter: %s", exc)

    # Quote type filter
    if config.quote_types:
        try:
            operands.append(EquityQuery("IS-IN", ["quotetype", *[qt.upper() for qt in config.quote_types]]))
            LOGGER.debug("Added quote type filter: %s", config.quote_types)
        except Exception as exc:
            LOGGER.warning("Could not add quote type filter: %s", exc)

    if len(operands) == 1:
        return operands[0]

    return EquityQuery("AND", operands)


def _validate_listings(listings: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Validate and filter screener results to ensure data quality.

    Filters out:
    - Test symbols (*TEST*, *DUMMY*, *SAMPLE*)
    - Symbols without valid exchange info
    - Symbols with incomplete data
    """
    valid = []
    filtered_reasons = {}

    for listing in listings:
        symbol = listing.get("symbol", "")
        exchange = listing.get("fullExchangeName") or listing.get("exchange")
        long_name = listing.get("longName") or listing.get("shortName")

        # Filter test symbols
        if any(keyword in symbol.upper() for keyword in ["TEST", "DUMMY", "SAMPLE", "XXXXX", "ZZZZZ"]):
            filtered_reasons.setdefault("test_symbol", 0)
            filtered_reasons["test_symbol"] += 1
            continue

        # Filter symbols without valid exchange
        valid_exchanges = ["NSE", "BSE", "NATIONAL STOCK EXCHANGE OF INDIA", "BOMBAY STOCK EXCHANGE"]
        if not exchange or not any(ex in str(exchange).upper() for ex in valid_exchanges):
            filtered_reasons.setdefault("invalid_exchange", 0)
            filtered_reasons["invalid_exchange"] += 1
            continue

        # Filter symbols without names (likely invalid)
        if not long_name or str(long_name).strip() == "":
            filtered_reasons.setdefault("missing_name", 0)
            filtered_reasons["missing_name"] += 1
            continue

        valid.append(listing)

    # Log filtering summary
    if filtered_reasons:
        total_filtered = sum(filtered_reasons.values())
        LOGGER.info("Filtered out %s invalid listings:", total_filtered)
        for reason, count in filtered_reasons.items():
            LOGGER.info("  - %s: %s", reason, count)

    return valid
