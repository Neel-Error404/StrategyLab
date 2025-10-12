from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "symbol",
    "exchange",
    "currency",
    "listing_status",
]


def run_validations(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Apply lightweight validation rules to the dataset.
    """
    if frame.empty:
        raise ValueError("Dataset is empty after enrichment.")

    _assert_required_columns(frame)
    _drop_duplicates(frame)
    _validate_currency(frame)
    _validate_listing_status(frame)
    _validate_numeric_non_negative(frame)

    return frame


def _assert_required_columns(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def _drop_duplicates(frame: pd.DataFrame) -> None:
    before = len(frame)
    frame.drop_duplicates(subset=["symbol", "exchange"], inplace=True)
    after = len(frame)
    if after < before:
        LOGGER.info("Dropped %s duplicate rows.", before - after)


def _validate_currency(frame: pd.DataFrame) -> None:
    allowed = {"INR", "USD", "INR=X"}
    unique_currencies = set(str(x) for x in frame["currency"].dropna().unique())
    unknown = unique_currencies - allowed
    if unknown:
        LOGGER.warning("Detected currencies outside allowlist: %s", sorted(unknown))


def _validate_listing_status(frame: pd.DataFrame) -> None:
    allowed = {"active", "suspended", "inactive", "delisted", None}
    unique_status = set(frame["listing_status"].astype(str))
    for status in unique_status:
        if status not in allowed and status.lower() not in allowed:
            LOGGER.warning("Unexpected listing_status encountered: %s", status)


def _validate_numeric_non_negative(frame: pd.DataFrame) -> None:
    numeric_cols: Iterable[str] = [
        "market_cap",
        "enterprise_value",
        "shares_outstanding",
        "regular_market_price",
        "regular_market_previous_close",
        "regular_market_open",
        "regular_market_day_high",
        "regular_market_day_low",
        "regular_market_volume",
        "fifty_two_week_high",
        "fifty_two_week_low",
    ]

    for column in numeric_cols:
        if column not in frame.columns:
            continue
        negatives = frame[column].dropna()
        if not negatives.empty and (negatives < 0).any():
            LOGGER.warning("Column %s contains negative values.", column)
