import pandas as pd

from src.strategies.strategy_open_source_baseline import OpenSourceBaselineStrategy
from src.strategies.register_strategies import register_all_strategies
from src.strategies.strategy_factory import StrategyFactory


def _sample_ohlcv(rows=20, trend=1.0):
    index = pd.date_range("2024-01-01 09:15", periods=rows, freq="5min", tz="Asia/Kolkata")
    base = pd.Series((1 + trend / rows) ** pd.RangeIndex(rows), index=index)
    close = 100 * base
    data = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": pd.Series(5_000, index=index),
        }
    )
    return data


def test_prepare_data_adds_indicators():
    strategy = OpenSourceBaselineStrategy(
        "open_source_baseline",
        {"short_window": 3, "long_window": 5, "momentum_window": 4},
    )
    raw = _sample_ohlcv()
    prepared = strategy.prepare_data({"5m": raw}, "RELIANCE", "2024-01-01")
    assert "short_sma" in prepared
    assert "long_sma" in prepared
    assert "momentum" in prepared
    assert prepared["volume_ok"].all()


def test_generate_signals_for_trend_transitions():
    strategy = OpenSourceBaselineStrategy(
        "open_source_baseline",
        {"short_window": 3, "long_window": 5, "momentum_window": 4, "min_volume": 0},
    )

    up = _sample_ohlcv(trend=5.0)
    prepared_up = strategy.prepare_data({"5m": up}, "RELIANCE", "2024-01-01")
    signals_up = strategy.generate_signals(prepared_up)
    assert signals_up["entry_signal_buy"].any()

    down = _sample_ohlcv(trend=-5.0)
    prepared_down = strategy.prepare_data({"5m": down}, "RELIANCE", "2024-01-02")
    signals_down = strategy.generate_signals(prepared_down)
    assert signals_down["entry_signal_sell"].any()


def test_strategy_factory_registers_open_source_baseline():
    register_all_strategies()
    instance = StrategyFactory.create_strategy("open_source_baseline")
    assert isinstance(instance, OpenSourceBaselineStrategy)
