"""
Pricing validation module for Phase 2.

Loads cached historical option prices, generates synthetic prices via
configurable volatility models, and compares them across multiple error
metrics and diagnostic plots.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # Headless plotting for reproducibility
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
except ImportError:  # pragma: no cover - seaborn is optional
    sns = None

from src.core.options.pricing import SyntheticPricingEngine, build_volatility_model
from src.core.options.validation.config_loader import get_validation_config
from src.core.options.validation.data_storage import OptionsDataStorage


logger = logging.getLogger(__name__)

IST = "Asia/Kolkata"


def _ensure_ist(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST)
    else:
        ts = ts.dt.tz_convert(IST)
    return ts


@dataclass
class SyntheticModelConfig:
    """Structured representation of a synthetic pricing model."""

    key: str
    name: str
    risk_free_rate: float
    volatility: Dict[str, Any]
    enabled: bool = True


class PricingValidator:
    """Main entry point for pricing validation and reporting."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        options_config_path: str = "src/core/options/config/options_config.yaml",
    ) -> None:
        self.config = get_validation_config(config_path)
        self.options_data_storage = OptionsDataStorage()
        self.validation_cfg = self.config.to_dict()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.options_config = self._load_options_config(options_config_path)
        pricing_cfg = self.options_config.get("pricing", {}).get("synthetic", {})
        self.default_risk_free_rate = pricing_cfg.get("risk_free_rate", 0.06)
        self.vol_floor = pricing_cfg.get("vol_floor", 0.10)
        self.vol_cap = pricing_cfg.get("vol_cap", 1.50)

        self.equity_pool = self.config.equity_pool or self._auto_detect_equity_pool()

        self.moneyness_bins = self.validation_cfg.get("segmentation", {}).get("moneyness_bins", [])
        self.dte_bins = self.validation_cfg.get("segmentation", {}).get("dte_bins", [])
        self.volatility_bins = self.validation_cfg.get("segmentation", {}).get("volatility_bins", [])

        self.summary_records: List[Dict[str, Any]] = []
        self.detail_records: List[Dict[str, Any]] = []
        self.row_records: List[pd.DataFrame] = []

    def run(
        self,
        tickers: Optional[Iterable[str]] = None,
        timeframe: Optional[str] = None,
        date_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the pricing validation workflow for the requested tickers.
        """
        timeframe = timeframe or self.config.timeframe
        date_range = date_range or self.config.date_range
        if date_range is None:
            raise ValueError("Validation date_range is required; configure validation.date_range")

        tickers = list(tickers) if tickers is not None else list(self.config.tickers)
        if not tickers:
            raise ValueError("No tickers provided for pricing validation")

        model_specs = self._prepare_models()
        if not model_specs:
            raise ValueError("No enabled synthetic models found in validation config")

        for ticker in tickers:
            logger.info("Starting pricing validation for %s", ticker)
            underlying_df = self._load_underlying_data(ticker)
            if underlying_df is None or underlying_df.empty:
                logger.error("Skipping %s - unable to load underlying equity data", ticker)
                continue

            expiries = self.options_data_storage.list_expiries(ticker, timeframe, date_range)
            if not expiries:
                logger.warning("No option expiries found for %s in %s/%s", ticker, timeframe, date_range)
                continue

            for expiry in expiries:
                options_df = self.options_data_storage.load_expiry_data(ticker, expiry, timeframe, date_range)
                if options_df is None or options_df.empty:
                    logger.warning("Empty options data for %s expiry %s", ticker, expiry)
                    continue

                for model_spec in model_specs:
                    row_df = self._price_with_model(
                        ticker=ticker,
                        expiry=expiry,
                        options_df=options_df,
                        underlying_df=underlying_df,
                        model_spec=model_spec,
                    )

                    if row_df.empty:
                        continue

                    self._append_metrics(row_df)
                    self.row_records.append(row_df)

        if not self.row_records:
            raise RuntimeError("Pricing validation produced no comparable rows; check data availability")

        combined_df = pd.concat(self.row_records, ignore_index=True)
        summary_df = pd.DataFrame(self.summary_records)
        detail_df = pd.DataFrame(self.detail_records)

        self._persist_outputs(combined_df, summary_df, detail_df)
        decision = self._derive_recommendation(summary_df, detail_df, combined_df)

        metrics_path = self.output_dir / self.config.get("output.metrics_json", "validation_metrics.json")
        summary_json = summary_df.copy()
        if "expiry" in summary_json.columns:
            summary_json["expiry"] = summary_json["expiry"].astype(str)
        detail_json = detail_df.copy()
        if "expiry" in detail_json.columns:
            detail_json["expiry"] = detail_json["expiry"].astype(str)

        metrics_payload = {
            "summary": summary_json.to_dict(orient="records"),
            "detail": detail_json.to_dict(orient="records"),
            "decision": decision,
        }
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics_payload, fh, indent=2)

        logger.info("Pricing validation completed. Summary rows: %d", len(summary_df))
        return {
            "combined": combined_df,
            "summary": summary_df,
            "detail": detail_df,
            "decision": decision,
            "metrics_path": metrics_path,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_models(self) -> List[SyntheticModelConfig]:
        models_cfg = self.validation_cfg.get("synthetic_models", {})
        models: List[SyntheticModelConfig] = []
        for key, spec in models_cfg.items():
            enabled = spec.get("enabled", True)
            if not enabled:
                continue
            name = spec.get("name", key)
            risk_free_rate = spec.get("risk_free_rate", self.default_risk_free_rate)
            volatility_cfg = spec.get("volatility", {})
            models.append(
                SyntheticModelConfig(
                    key=key,
                    name=name,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility_cfg,
                    enabled=enabled,
                )
            )
        return models

    def _auto_detect_equity_pool(self) -> Optional[str]:
        pools_dir = Path("data/pools")
        if not pools_dir.exists():
            logger.warning("data/pools directory not found while auto-detecting equity pool")
            return None

        import re

        pattern = re.compile(r"\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}")
        candidates = [d for d in pools_dir.iterdir() if d.is_dir() and pattern.match(d.name) and d.name != "options"]
        if not candidates:
            logger.warning("No equity pools discovered in %s", pools_dir)
            return None

        def end_date(dir_name: str) -> datetime:
            try:
                end_str = dir_name.split('_to_')[-1]
                return datetime.strptime(end_str, '%Y-%m-%d')
            except Exception:  # pragma: no cover - defensive
                return datetime.min

        candidates.sort(key=lambda d: end_date(d.name), reverse=True)
        chosen = candidates[0]
        logger.info("Auto-detected equity pool: %s", chosen)
        return str(chosen)

    def _load_options_config(self, path: str) -> Dict[str, Any]:
        cfg_path = Path(path)
        if not cfg_path.exists():
            logger.warning("Options config not found at %s; using defaults", path)
            return {}

        import yaml

        with open(cfg_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _load_underlying_data(self, ticker: str) -> Optional[pd.DataFrame]:
        equity_pool = self.equity_pool
        if equity_pool is None:
            logger.error("No equity_pool configured; cannot source underlying for %s", ticker)
            return None

        pool_path = Path(equity_pool)
        if not pool_path.exists():
            logger.error("Equity pool path %s does not exist", pool_path)
            return None

        candidates: List[Path] = []
        ticker_dir = pool_path / ticker
        if ticker_dir.exists():
            candidates.extend(sorted(ticker_dir.glob("*.parquet")))
            candidates.extend(sorted(ticker_dir.glob("*.csv")))
        else:
            for subdir in pool_path.iterdir():
                if not subdir.is_dir():
                    continue
                candidates.extend(sorted(subdir.glob(f"{ticker}_*.parquet")))
                candidates.extend(sorted(subdir.glob(f"{ticker}_*.csv")))

        if not candidates:
            logger.error("No equity files found for %s in %s", ticker, pool_path)
            return None

        source = candidates[0]
        logger.info("Loading underlying data for %s from %s", ticker, source)
        if source.suffix == ".parquet":
            df = pd.read_parquet(source)
        else:
            df = pd.read_csv(source)

        if "timestamp" not in df.columns:
            ts_column = None
            for cand in ("datetime", "date", "Date", "time"):
                if cand in df.columns:
                    ts_column = cand
                    break
            if ts_column is None:
                raise ValueError(f"Underlying data {source} lacks a timestamp column")
            df["timestamp"] = df[ts_column]

        # Ensure OHLC columns exist for volatility models
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                df[col] = df["close"]

        return df[["timestamp", "open", "high", "low", "close"]]

    def _price_with_model(
        self,
        ticker: str,
        expiry: date,
        options_df: pd.DataFrame,
        underlying_df: pd.DataFrame,
        model_spec: SyntheticModelConfig,
    ) -> pd.DataFrame:
        option_work = options_df.copy()
        option_work["timestamp"] = _ensure_ist(option_work["timestamp"])
        underlying_ts = _ensure_ist(underlying_df["timestamp"])
        min_ts = underlying_ts.min()
        max_ts = underlying_ts.max()
        mask = (option_work["timestamp"] >= min_ts) & (option_work["timestamp"] <= max_ts)
        if not mask.all():
            dropped = int((~mask).sum())
            logger.debug("Dropped %d rows outside underlying range for %s %s", dropped, ticker, expiry)
        option_work = option_work.loc[mask]
        if option_work.empty:
            logger.warning("No overlapping dates between options and underlying for %s %s", ticker, expiry)
            return pd.DataFrame()

        model_instance = build_volatility_model(model_spec.volatility)
        engine = SyntheticPricingEngine(
            volatility_model=model_instance,
            risk_free_rate=model_spec.risk_free_rate,
            vol_floor=self.vol_floor,
            vol_cap=self.vol_cap,
        )
        engine.set_underlying_data(underlying_df)

        context = {"expiry": expiry.isoformat(), "ticker": ticker}
        try:
            engine.fit_volatility_surface(option_data=option_work, context=context)
            priced = engine.price_options(option_work, option_metadata=context)
        except ValueError as exc:
            logger.error("Pricing failed for %s %s model %s: %s", ticker, expiry, model_spec.key, exc)
            return pd.DataFrame()

        df = priced.dataframe.copy()
        df["model_key"] = model_spec.key
        df["model_name"] = model_spec.name
        df["ticker"] = ticker
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
        df["error"] = df["synthetic_price"] - df["close"]
        df["abs_error"] = df["error"].abs()

        denom = df["close"].replace(0, np.nan)
        df["pct_error"] = df["error"] / denom
        df["abs_pct_error"] = df["pct_error"].abs()

        df["moneyness"] = df.apply(self._compute_moneyness, axis=1)
        df["dte_days"] = df["time_to_expiry"] * 365.0

        df["moneyness_bucket"] = df["moneyness"].apply(lambda v: self._assign_bucket(v, self.moneyness_bins))
        df["dte_bucket"] = df["dte_days"].apply(lambda v: self._assign_bucket(v, self.dte_bins))
        df["vol_bucket"] = df["model_vol"].apply(lambda v: self._assign_bucket(v, self.volatility_bins))

        return df

    def _compute_moneyness(self, row: pd.Series) -> float:
        spot = float(row.get("underlying_close", np.nan))
        strike = float(row.get("strike", np.nan))
        option_type = row.get("option_type")
        if not np.isfinite(spot) or not np.isfinite(strike) or strike == 0:
            return np.nan
        if option_type == "CE":
            return spot / strike
        return strike / spot

    def _assign_bucket(self, value: float, bins: List[Dict[str, Any]]) -> str:
        if not np.isfinite(value):
            return "missing"
        for entry in bins:
            min_val = entry.get("min", -np.inf)
            max_val = entry.get("max", np.inf)
            if min_val <= value < max_val:
                return entry.get("name", f"{min_val}-{max_val}")
        return "out_of_range"

    def _append_metrics(self, df: pd.DataFrame) -> None:
        summary_metrics = self._compute_metrics(df)
        summary_metrics.update(
            {
                "ticker": df["ticker"].iloc[0],
                "model_key": df["model_key"].iloc[0],
                "model_name": df["model_name"].iloc[0],
                "expiry": df["expiry"].iloc[0],
                "segment_type": "overall",
                "segment_name": "all",
                "count": len(df),
            }
        )
        self.summary_records.append(summary_metrics)

        for segment_col, segment_type in [
            ("moneyness_bucket", "moneyness"),
            ("dte_bucket", "dte"),
            ("vol_bucket", "volatility"),
        ]:
            for segment_name, segment_df in df.groupby(segment_col):
                if segment_df.empty:
                    continue
                segment_metrics = self._compute_metrics(segment_df)
                segment_metrics.update(
                    {
                        "ticker": df["ticker"].iloc[0],
                        "model_key": df["model_key"].iloc[0],
                        "model_name": df["model_name"].iloc[0],
                        "expiry": df["expiry"].iloc[0],
                        "segment_type": segment_type,
                        "segment_name": segment_name,
                        "count": len(segment_df),
                    }
                )
                self.detail_records.append(segment_metrics)

    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        errors = df["error"].to_numpy(dtype=float)
        abs_errors = np.abs(errors)
        valid_mask = df["close"].to_numpy(dtype=float) > 0
        pct_errors = df.loc[valid_mask, "pct_error"].to_numpy(dtype=float)

        if len(abs_errors):
            metrics["mean_absolute_error"] = float(np.mean(abs_errors))
            metrics["rmse"] = float(np.sqrt(np.mean(errors ** 2)))
            metrics["std_error"] = float(np.std(errors))
            metrics["p95_error"] = float(np.percentile(abs_errors, 95))
            metrics["p99_error"] = float(np.percentile(abs_errors, 99))
            metrics["max_error"] = float(np.max(abs_errors))
        else:
            metrics.update({
                "mean_absolute_error": np.nan,
                "rmse": np.nan,
                "std_error": np.nan,
                "p95_error": np.nan,
                "p99_error": np.nan,
                "max_error": np.nan,
            })

        if len(pct_errors):
            metrics["mean_percentage_error"] = float(np.mean(pct_errors))
            metrics["median_percentage_error"] = float(np.median(np.abs(pct_errors)))
            metrics["systematic_bias"] = float(np.mean(pct_errors))
        else:
            metrics["mean_percentage_error"] = np.nan
            metrics["median_percentage_error"] = np.nan
            metrics["systematic_bias"] = np.nan

        # Directional accuracy based on day-over-day moves
        df_sorted = df.sort_values("timestamp")
        actual_change = df_sorted["close"].diff()
        synthetic_change = df_sorted["synthetic_price"].diff()
        move_mask = (actual_change != 0) & (synthetic_change != 0)
        if move_mask.sum() > 0:
            accuracy = np.mean(np.sign(actual_change[move_mask]) == np.sign(synthetic_change[move_mask]))
            metrics["directional_accuracy"] = float(accuracy)
        else:
            metrics["directional_accuracy"] = np.nan

        return metrics

    def _persist_outputs(
        self,
        combined_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        detail_df: pd.DataFrame,
    ) -> None:
        summary_path = self.output_dir / self.config.get("output.summary_report", "pricing_validation_summary.csv")
        detail_path = self.output_dir / self.config.get("output.detail_report", "pricing_validation_detail.csv")

        combined_df.to_parquet(self.output_dir / "pricing_validation_rows.parquet", index=False)
        summary_df.to_csv(summary_path, index=False)
        detail_df.to_csv(detail_path, index=False)

        plots_cfg = self.validation_cfg.get("output", {})
        if not plots_cfg.get("generate_plots", True):
            return

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_format = plots_cfg.get("plot_format", "png")
        dpi = plots_cfg.get("plot_dpi", 200)

        self._plot_error_distribution(combined_df, plots_dir / f"error_distribution_by_model.{plot_format}", dpi)
        self._plot_heatmap(combined_df, plots_dir / f"error_heatmap_moneyness_dte.{plot_format}", dpi)
        self._plot_error_timeseries(combined_df, plots_dir / f"error_timeseries.{plot_format}", dpi)
        self._plot_model_boxplot(combined_df, plots_dir / f"model_comparison_boxplot.{plot_format}", dpi)
        self._plot_bias_analysis(combined_df, plots_dir / f"bias_analysis.{plot_format}", dpi)

    def _plot_error_distribution(self, df: pd.DataFrame, path: Path, dpi: int) -> None:
        if df.empty:
            logger.warning("Skipping error distribution plot; no data available")
            return
        plt.figure(figsize=(10, 6))
        for model_name, group in df.groupby("model_name"):
            data = (group["pct_error"].dropna() * 100).clip(-200, 200)
            plt.hist(data, bins=50, alpha=0.4, label=model_name, density=True)
        plt.xlabel("Percentage Error (%)")
        plt.ylabel("Density")
        plt.title("Synthetic vs Actual Error Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()

    def _plot_heatmap(self, df: pd.DataFrame, path: Path, dpi: int) -> None:
        if df.empty:
            logger.warning("Skipping heatmap plot; no data available")
            return
        pivot = (
            df.pivot_table(
                index="moneyness_bucket",
                columns="dte_bucket",
                values="abs_pct_error",
                aggfunc="mean",
            )
            * 100
        )
        plt.figure(figsize=(8, 6))
        if sns is not None:
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
        else:  # pragma: no cover - fallback for environments without seaborn
            plt.imshow(pivot, aspect="auto", cmap="viridis")
            plt.colorbar(label="Mean Abs % Error")
            plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
            plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title("Mean Absolute Percentage Error by Moneyness & DTE")
        plt.xlabel("Days to Expiry Bucket")
        plt.ylabel("Moneyness Bucket")
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()

    def _plot_error_timeseries(self, df: pd.DataFrame, path: Path, dpi: int) -> None:
        if df.empty:
            logger.warning("Skipping timeseries plot; no data available")
            return
        daily = (
            df.assign(error_pct=lambda x: x["pct_error"] * 100)
            .groupby([df["timestamp"].dt.date, "model_name"])
            ["error_pct"]
            .median()
            .reset_index()
        )
        plt.figure(figsize=(10, 6))
        for model_name, group in daily.groupby("model_name"):
            plt.plot(group["timestamp"], group["error_pct"], label=model_name)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("Date")
        plt.ylabel("Median % Error")
        plt.title("Daily Median Percentage Error by Model")
        plt.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()

    def _plot_model_boxplot(self, df: pd.DataFrame, path: Path, dpi: int) -> None:
        if df.empty:
            logger.warning("Skipping boxplot; no data available")
            return
        plt.figure(figsize=(10, 6))
        data = [group["abs_pct_error"].dropna() * 100 for _, group in df.groupby("model_name")]
        labels = [name for name, _ in df.groupby("model_name")]
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.ylabel("Absolute Percentage Error (%)")
        plt.title("Model Comparison - Absolute Percentage Error")
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()

    def _plot_bias_analysis(self, df: pd.DataFrame, path: Path, dpi: int) -> None:
        if df.empty:
            logger.warning("Skipping bias analysis plot; no data available")
            return
        plt.figure(figsize=(10, 6))
        if len(df) > 5000:
            sample = df.sample(n=5000, random_state=42)
        else:
            sample = df
        scatter = plt.scatter(
            sample["moneyness"],
            sample["pct_error"] * 100,
            c=sample["dte_days"],
            cmap="coolwarm",
            alpha=0.6,
        )
        plt.xlabel("Moneyness")
        plt.ylabel("Percentage Error (%)")
        plt.title("Bias Analysis by Moneyness and DTE")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Days to Expiry")
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()

    def _derive_recommendation(self, summary_df: pd.DataFrame, detail_df: pd.DataFrame, combined_df: pd.DataFrame) -> Dict[str, Any]:
        thresholds = self.validation_cfg.get("decision_criteria", {}).get("thresholds", {})
        recommendations = []

        atm_data = combined_df[combined_df["moneyness_bucket"] == "ATM"]
        for (ticker, model_name), group in atm_data.groupby(["ticker", "model_name"]):
            if group.empty:
                continue
            atm_median_error = float(group["abs_pct_error"].median())
            atm_mape = float(group["abs_pct_error"].mean())
            recommendation = self._map_threshold(atm_median_error, thresholds)
            recommendations.append(
                {
                    "ticker": ticker,
                    "model_name": model_name,
                    "atm_median_error": atm_median_error,
                    "atm_mean_abs_pct": atm_mape,
                    "atm_count": int(len(group)),
                    "recommendation": recommendation,
                }
            )

        return {"evaluations": recommendations}

    def _map_threshold(self, value: float, thresholds: Dict[str, Dict[str, Any]]) -> str:
        ordered = [
            (name, cfg["atm_median_error"], cfg["recommendation"])
            for name, cfg in thresholds.items()
        ]
        ordered.sort(key=lambda x: x[1])
        for _, limit, recommendation in ordered:
            if value <= limit:
                return recommendation
        return ordered[-1][2] if ordered else "No recommendation configured"


def run_pricing_validation(**kwargs) -> Dict[str, Any]:
    validator = PricingValidator(**kwargs)
    return validator.run()
