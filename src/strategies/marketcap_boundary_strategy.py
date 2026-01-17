"""
Market-cap boundary strategy (semiannual AMFI-style buckets).

Signals:
- Entry (long-only): when close <= boundary_price * buffer for the active bucket.
- Exit: first bar of the next semiannual period (rebalance).

Data:
- Requires daily timeframe (`day`) with columns: timestamp, open, high, low, close, volume.
- Requires precomputed snapshots from analysis/tools/build_marketcap_rankings.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .support.strategy_base import StrategyBase


@dataclass
class MarketCapBoundaryParameters:
    snapshot_dir: str = "analysis/derived/marketcap_rankings"
    snapshot_index: str = "analysis/derived/marketcap_rankings/snapshot_index.json"
    buffer: float = 1.0  # 1.0 = exact boundary; use 0.98/0.95 to enter earlier
    start_date: str = "2022-10-01"  # first trading day after initial cutoff
    min_volume: Optional[float] = None  # optional liquidity filter
    stop_loss_pct: Optional[float] = 0.10  # e.g., 0.10 = risk 10%
    take_profit_pct: Optional[float] = 0.20  # e.g., 0.20 = target 20%


class MarketCapBoundaryStrategy(StrategyBase):
    def __init__(
        self,
        name: str = "marketcap_boundary",
        parameters: Optional[Dict[str, Any]] = None,
        config: Optional[Any] = None,
    ) -> None:
        super().__init__(name=name, parameters=parameters or {}, config=config)

        self.params = MarketCapBoundaryParameters(**self.parameters)
        self.required_timeframes = ["day"]
        self.warmup_period = 0

        self.logger = logging.getLogger(f"strategy.{name}")
        self._snapshot_cache: Dict[str, pd.DataFrame] = {}
        self._snapshot_index = self._load_snapshot_index(Path(self.params.snapshot_index))
        self._start_date = pd.to_datetime(self.params.start_date).date()
        self._active_ticker: Optional[str] = None

    def _load_snapshot_index(self, index_path: Path) -> List[Dict[str, Any]]:
        if not index_path.exists():
            raise FileNotFoundError(f"Snapshot index not found: {index_path}")
        with index_path.open() as f:
            raw_index = json.load(f)
        parsed = []
        for item in raw_index:
            parsed.append(
                {
                    "cutoff_date": pd.to_datetime(item["cutoff_date"]).date(),
                    "valid_from": pd.to_datetime(item["valid_from"]).date(),
                    "valid_to": pd.to_datetime(item["valid_to"]).date() if item.get("valid_to") else None,
                    "snapshot_path": Path(item["snapshot_path"]),
                    "window_start": pd.to_datetime(item.get("window_start", item["valid_from"])).date(),
                }
            )
        return parsed

    def _load_snapshot(self, cutoff_date: datetime.date) -> pd.DataFrame:
        key = cutoff_date.isoformat()
        if key in self._snapshot_cache:
            return self._snapshot_cache[key]
        record = next((r for r in self._snapshot_index if r["cutoff_date"] == cutoff_date), None)
        if not record:
            raise ValueError(f"No snapshot found for cutoff {cutoff_date}")
        df = pd.read_parquet(record["snapshot_path"])
        self._snapshot_cache[key] = df
        return df

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------
    def prepare_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        ticker: str,
        pull_date: str,
    ) -> pd.DataFrame:
        self._active_ticker = ticker
        if isinstance(data, dict):
            if "day" not in data:
                raise ValueError("MarketCapBoundaryStrategy requires 'day' timeframe")
            df = data["day"].copy()
        else:
            df = data.copy()

        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        if not self.validate_data(df):
            raise ValueError("Data validation failed for day timeframe")

        df = df[df["date"] >= self._start_date].reset_index(drop=True)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["entry_signal_buy"] = False
        df["entry_signal_sell"] = False
        df["exit_signal_buy"] = False
        df["exit_signal_sell"] = False

        if df.empty:
            return df

        df["period_id"] = self._assign_periods(df["date"])
        df = df[df["period_id"].notna()].reset_index(drop=True)

        # Precompute entry signals per period using boundary prices
        for pid in sorted(df["period_id"].unique()):
            period_mask = df["period_id"] == pid
            record = self._snapshot_index[int(pid)]
            snapshot = self._load_snapshot(record["cutoff_date"])
            row = snapshot.loc[snapshot["ticker"] == self._active_ticker]

            if row.empty:
                continue

            boundary_price = float(row["boundary_price"].iloc[0]) * float(self.params.buffer)
            period_slice = df.loc[period_mask]

            cond = period_slice["close"] <= boundary_price
            if self.params.min_volume:
                cond = cond & (period_slice["volume"] >= float(self.params.min_volume))

            entry_mask = cond & (~cond.shift(1, fill_value=False))
            df.loc[period_slice.index, "entry_signal_buy"] = entry_mask

        # State machine to add stop/target exits and rebalance exits
        in_trade = False
        entry_price: Optional[float] = None

        for idx, row in df.iterrows():
            period_change = idx > 0 and df.loc[idx, "period_id"] != df.loc[idx - 1, "period_id"]

            if period_change and in_trade:
                df.at[idx, "exit_signal_buy"] = True
                in_trade = False
                entry_price = None
                continue

            if not in_trade and df.at[idx, "entry_signal_buy"]:
                in_trade = True
                entry_price = row["close"]
                continue

            if in_trade and entry_price:
                stop_hit = (
                    self.params.stop_loss_pct is not None
                    and row["close"] <= entry_price * (1 - float(self.params.stop_loss_pct))
                )
                tp_hit = (
                    self.params.take_profit_pct is not None
                    and row["close"] >= entry_price * (1 + float(self.params.take_profit_pct))
                )
                if stop_hit or tp_hit:
                    df.at[idx, "exit_signal_buy"] = True
                    in_trade = False
                    entry_price = None

        # Exit on first bar of each new period if still flagged by period change (guard)
        period_change = df["period_id"] != df["period_id"].shift(1)
        exit_mask = period_change & (df.index > df.index.min())
        df.loc[exit_mask, "exit_signal_buy"] = True

        return df.drop(columns=["period_id"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _assign_periods(self, dates: pd.Series) -> pd.Series:
        period_ids: List[Optional[int]] = []
        for current_date in dates:
            pid = None
            for idx, rec in enumerate(self._snapshot_index):
                valid_from = rec["valid_from"]
                valid_to = rec["valid_to"]
                if current_date >= valid_from and (valid_to is None or current_date <= valid_to):
                    pid = idx
                    break
            period_ids.append(pid)
        return pd.Series(period_ids, index=dates.index)
