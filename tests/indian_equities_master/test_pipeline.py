from __future__ import annotations

import pandas as pd

from src.data_tools.indian_equities_master import enrichment, qa
from src.data_tools.indian_equities_master.config import EnrichConfig


class DummyFastInfo(dict):
    def items(self):
        return super().items()


class DummyTicker:
    def __init__(self, symbol: str):
        self.ticker = symbol

    def get_info(self):
        return {
            "longName": "Infosys Limited",
            "shortName": "Infosys",
            "sector": "Technology",
            "industry": "IT Services",
            "marketCap": 123_456_789,
            "enterpriseValue": 120_000_000,
            "sharesOutstanding": 1_234_000_000,
            "bookValue": 100.5,
            "beta": 0.9,
            "trailingPE": 25.1,
            "forwardPE": 22.3,
            "priceToBook": 4.2,
            "dividendYield": 0.015,
            "profitMargins": 0.22,
            "fiftyTwoWeekHigh": 1700.0,
            "fiftyTwoWeekLow": 1200.0,
            "firstTradeDateEpochUtc": 915148800,
            "fnoEligible": True,
            "isPublicSectorUndertaking": False,
        }

    @property
    def fast_info(self):
        return DummyFastInfo(
            {
                "last_price": 1500.0,
                "previous_close": 1490.0,
                "open": 1485.0,
                "day_high": 1510.0,
                "day_low": 1475.0,
                "last_volume": 1_000_000,
            }
        )

    @property
    def options(self):
        return []


def test_build_dataset_uses_listing_and_info(monkeypatch):
    listings = [
        {
            "symbol": "INFY.NS",
            "fullExchangeName": "NSE",
            "currency": "INR",
            "marketState": "REGULAR",
            "quoteType": "EQUITY",
            "shortName": "Infosys",
            "regularMarketPrice": 1498.0,
            "regularMarketPreviousClose": 1490.0,
            "regularMarketOpen": 1485.0,
            "regularMarketDayHigh": 1505.0,
            "regularMarketDayLow": 1470.0,
            "regularMarketVolume": 950_000,
            "regularMarketChangePercent": 0.0067,
            "regularMarketTime": 1712121600,
            "fiftyTwoWeekHigh": 1705.0,
            "fiftyTwoWeekLow": 1195.0,
            "exchangeTimezoneName": "Asia/Kolkata",
            "quoteSourceName": "Delayed Quote",
        }
    ]

    def fake_tickers(_):
        return type("Dummy", (), {"tickers": {"INFY.NS": DummyTicker("INFY.NS")}})()

    monkeypatch.setattr(enrichment.yf, "Tickers", fake_tickers)

    frame = enrichment.build_dataset(listings, EnrichConfig(batch_size=5))
    assert list(frame.columns) == enrichment.COLUMN_ORDER
    row = frame.iloc[0].to_dict()
    assert row["symbol"] == "INFY.NS"
    assert row["market_cap"] == 123_456_789
    assert row["regular_market_price"] == 1500.0
    assert row["listing_status"] == "active"
    assert row["data_quality_score"] > 0


def test_run_validations_deduplicates_rows():
    data = {
        "symbol": ["ABC.NS", "ABC.NS"],
        "exchange": ["NSE", "NSE"],
        "currency": ["INR", "INR"],
        "listing_status": ["active", "active"],
        "market_cap": [1_000_000, 1_000_000],
    }
    frame = pd.DataFrame(data)
    cleaned = qa.run_validations(frame)
    assert len(cleaned) == 1
