import hashlib
import types
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.core.options.data.schemas import OptionType
from src.core.options.replay.config import OptionsReplayConfig
from src.core.options.replay.data_loader import OptionDataStore
from src.core.options.replay.engine import OptionsReplayEngine, TickerContext
from src.core.options.replay.models import (
    EquityTrade,
    OptionContractSpec,
    OptionPositionSnapshot,
    PricingEvent,
)


def _load_options_config(tmp_path: Path) -> OptionsReplayConfig:
    config_path = Path("src/core/options/config/options_config.yaml")
    raw = yaml.safe_load(config_path.read_text())
    raw["output"]["output_dir"] = str(tmp_path / "replay_outputs")
    raw["inputs"]["run_label"] = "unit_test"
    raw["risk"]["max_portfolio_allocation"] = 1.0
    raw["risk"]["max_position_size_per_trade"] = 1.0
    raw["risk"]["max_concurrent_positions"] = 10
    return OptionsReplayConfig.from_dict(raw, base_dir=Path.cwd())


class StubPricer:
    def price(self, contract, timestamp, underlying_price):
        event = PricingEvent(
            timestamp=timestamp,
            price=10.0,
            pricing_mode="actual",
            implied_vol=0.2,
            underlying_price=underlying_price,
            notes={"synthetic_price": 10.0, "actual_price": 10.0},
        )
        return event, None

    def price_path(self, contract, timestamps, underlying_prices, include_greeks=True):
        snapshots = [
            OptionPositionSnapshot(
                timestamp=ts,
                option_price=10.0,
                underlying_price=spot,
                dte=1.0,
                greeks={},
                pricing_mode="actual",
            )
            for ts, spot in zip(timestamps, underlying_prices)
        ]
        return snapshots, []


def test_option_data_store_intraday_fallback(tmp_path):
    store = OptionDataStore(base_dir=tmp_path, timeframes=("1minute", "1day"))
    expiry = pd.Timestamp("2024-01-25", tz="Asia/Kolkata")
    key = expiry.normalize().strftime("%Y-%m-%d")

    minute_df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01 09:30", tz="Asia/Kolkata")],
            "strike": [2000.0],
            "option_type": ["CE"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.5],
            "close": [10.2],
            "volume": [100],
            "open_interest": [200],
            "bid": [10.1],
            "ask": [10.3],
            "ticker": ["TEST"],
            "expiry": [expiry.date()],
            "lot_size": [25],
        }
    )
    daily_df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01 15:30", tz="Asia/Kolkata")],
            "strike": [2000.0],
            "option_type": ["CE"],
            "open": [10.0],
            "high": [10.6],
            "low": [9.4],
            "close": [10.4],
            "volume": [500],
            "open_interest": [500],
            "bid": [10.2],
            "ask": [10.6],
            "ticker": ["TEST"],
            "expiry": [expiry.date()],
            "lot_size": [25],
        }
    )
    store._option_cache["1minute"][key] = minute_df
    store._option_cache["1day"][key] = daily_df

    ts = pd.Timestamp("2024-01-01 09:30", tz="Asia/Kolkata")
    row, meta = store.find_price_bar(expiry, OptionType.CALL, 2000.0, ts)
    assert meta["timeframe"] == "1minute"
    assert row["close"] == pytest.approx(10.2)

    # Remove minute data to force fallback
    store._option_cache["1minute"][key] = None
    row, meta = store.find_price_bar(expiry, OptionType.CALL, 2000.0, ts)
    assert meta["timeframe"] == "1day"
    assert meta["alignment"] == "session_close"
    assert row["close"] == pytest.approx(10.4)


def test_replay_engine_multi_ticker_ordering(monkeypatch, tmp_path):
    config = _load_options_config(tmp_path)

    base_ts = pd.Timestamp("2024-01-01 09:30", tz="Asia/Kolkata")
    trades = [
        EquityTrade(
            trade_id="T1",
            ticker="RELIANCE",
            side="LONG",
            entry_time=base_ts + pd.Timedelta(minutes=15),
            exit_time=base_ts + pd.Timedelta(minutes=45),
            entry_price=100.0,
            exit_price=102.0,
            quantity=1,
            pnl=2.0,
        ),
        EquityTrade(
            trade_id="T2",
            ticker="TCS",
            side="LONG",
            entry_time=base_ts,
            exit_time=base_ts + pd.Timedelta(minutes=30),
            entry_price=200.0,
            exit_price=205.0,
            quantity=1,
            pnl=5.0,
        ),
        EquityTrade(
            trade_id="T3",
            ticker="RELIANCE",
            side="LONG",
            entry_time=base_ts + pd.Timedelta(minutes=60),
            exit_time=base_ts + pd.Timedelta(minutes=120),
            entry_price=101.0,
            exit_price=103.0,
            quantity=1,
            pnl=2.0,
        ),
    ]

    monkeypatch.setattr(
        "src.core.options.replay.engine.load_equity_trades",
        lambda trades_path, tz, ticker_whitelist: trades,
    )

    def fake_map_trade_to_option(config, option_store, trade, underlying_entry_price):
        expiry = trade.exit_time + pd.Timedelta(days=30)
        contract = OptionContractSpec(
            ticker=trade.ticker,
            expiry=expiry,
            strike=float(underlying_entry_price),
            option_type=OptionType.CALL,
            lot_size=25,
        )
        return types.SimpleNamespace(contract=contract, metadata={})

    monkeypatch.setattr("src.core.options.replay.engine.map_trade_to_option", fake_map_trade_to_option)

    def fake_prepare(self, trades_by_ticker):
        contexts = {}
        for ticker, items in trades_by_ticker.items():
            cache = {}
            for equity_trade, label in items:
                df = pd.DataFrame(
                    {
                        "timestamp": [
                            equity_trade.entry_time,
                            equity_trade.exit_time,
                        ],
                        "close": [equity_trade.entry_price, equity_trade.exit_price],
                    }
                )
                if label in cache:
                    cache[label] = (
                        pd.concat([cache[label], df])
                        .drop_duplicates(subset="timestamp")
                        .sort_values("timestamp")
                        .reset_index(drop=True)
                    )
                else:
                    cache[label] = df
            contexts[ticker] = TickerContext(
                ticker=ticker,
                pricer=StubPricer(),
                underlying_cache=cache,
                option_store=object(),
                date_ranges=sorted({label for _, label in items}),
            )
        return contexts, 1

    monkeypatch.setattr(OptionsReplayEngine, "_prepare_ticker_contexts", fake_prepare)

    engine = OptionsReplayEngine(config)
    artefacts = engine.run(
        tickers=["RELIANCE", "TCS"],
        date_ranges=["2024-01-01_to_2024-01-01"],
        verify_hash=False,
    )

    ordered_ids = [trade.equity_trade.trade_id for trade in artefacts.trades]
    assert ordered_ids == ["T2", "T1", "T3"]

    ticker_summary = artefacts.metadata["ticker_summary"]
    assert ticker_summary["RELIANCE"]["processed"] == 2
    assert ticker_summary["TCS"]["processed"] == 1

    metrics_hash = engine.hash_records["options_metrics.json"]
    metrics_path = Path(engine.output_dir) / "options_metrics.json"
    assert metrics_hash == hashlib.sha256(metrics_path.read_bytes()).hexdigest()


def test_write_json_deterministic_hash(monkeypatch, tmp_path):
    config = _load_options_config(tmp_path)
    engine = OptionsReplayEngine(config)
    payload = {"foo": 1, "bar": "baz"}
    engine._write_json(payload, "snapshot.json")
    first_hash = engine.hash_records["snapshot.json"]
    engine._write_json(payload, "snapshot.json")
    second_hash = engine.hash_records["snapshot.json"]
    path = Path(engine.output_dir) / "snapshot.json"
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert first_hash == second_hash == expected_hash
