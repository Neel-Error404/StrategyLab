import hashlib
import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.core.output.enhanced_output_orchestrator import EnhancedOutputOrchestrator


def _make_base_frame(ticker: str) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1_000, 1_200, 1_100, 900, 950],
            "entry_signal_buy": [0, 1, 0, 0, 1],
            "ticker": [ticker] * len(dates),
        }
    )


def _make_strategy_trades(ticker: str) -> list[dict]:
    return [
        {
            "Trade Type": "LONG",
            "Entry Time": "2025-01-02 09:15:00",
            "Exit Time": "2025-01-03 15:30:00",
            "Entry Price": 100.0,
            "Exit Price": 105.0,
            "Profit (Currency)": 500.0,
            "Profit (%)": 5.0,
            "Trade Duration (min)": 900,
            "ticker": ticker,
        },
        {
            "Trade Type": "LONG",
            "Entry Time": "2025-01-04 09:15:00",
            "Exit Time": "2025-01-05 15:30:00",
            "Entry Price": 104.0,
            "Exit Price": 102.5,
            "Profit (Currency)": -300.0,
            "Profit (%)": -1.44,
            "Trade Duration (min)": 900,
            "ticker": ticker,
        },
    ]


def _make_results_payload(
    strategy: str,
    date_range: str,
    ticker: str,
    *,
    strategy_trades: list[dict] | None = None,
    approved_trades: list[dict] | None = None,
    risk_report: dict | None = None,
    options_metrics: dict | None = None,
) -> dict:
    strategy_trades = strategy_trades if strategy_trades is not None else _make_strategy_trades(ticker)
    approved_trades = approved_trades if approved_trades is not None else strategy_trades[:1]
    base_df = _make_base_frame(ticker)
    if risk_report is None:
        risk_report = {
            "original_trade_count": len(strategy_trades),
            "approved_trade_count": len(approved_trades),
            "rejected_trade_count": len(strategy_trades) - len(approved_trades),
            "approval_rate": 50.0,
        }
    if options_metrics is None:
        options_metrics = {"options_total_pnl": 150.0, "options_delta": 0.25}
    metrics_block = {"Parameters": {"fast": 5, "slow": 20}}
    if "options_total_pnl" in options_metrics:
        metrics_block["options_total_pnl"] = options_metrics["options_total_pnl"]
    if "options_delta" in options_metrics:
        metrics_block["options_delta"] = options_metrics["options_delta"]

    return {
        "ticker": ticker,
        "strategy": strategy,
        "date_range": date_range,
        "base_data": base_df,
        "strategy_trades": strategy_trades,
        "trades": approved_trades,
        "risk_report": risk_report,
        "bias_report": {"violations": []},
        "metrics": metrics_block,
        "options_metrics": options_metrics,
    }


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@pytest.mark.parametrize("tickers", [["RELIANCE", "INFY"]])
def test_enhanced_orchestrator_generates_real_reports(tmp_path: Path, tickers: list[str]) -> None:
    strategy_name = "demo_strategy"
    date_range = "2025-01-01_to_2025-01-05"

    results_data = {
        strategy_name: {
            date_range: {
                ticker: _make_results_payload(strategy_name, date_range, ticker)
                for ticker in tickers
            }
        }
    }

    orchestrator = EnhancedOutputOrchestrator(base_output_dir=tmp_path)
    strategy_run_dir = tmp_path / "demo_strategy_run"

    processing = orchestrator.process_complete_backtest_results(
        strategy_name=strategy_name,
        date_range=date_range,
        tickers=tickers,
        results_data=results_data,
        run_id="test_run",
        strategy_run_dir=strategy_run_dir,
    )

    assert processing["ticker_reports"], "ticker reports should be generated"

    reliance_metrics = processing["ticker_reports"]["RELIANCE"]["metrics"]
    assert reliance_metrics["trade_metrics"]["generated_trades"] == 2
    assert reliance_metrics["trade_metrics"]["approved_trades"] == 1
    assert reliance_metrics["risk_metrics"]["approval_rate_pct"] == pytest.approx(50.0)
    assert reliance_metrics["data_metrics"]["signal_events"] == 2
    assert reliance_metrics["pnl_metrics"]["total_profit_currency"] == pytest.approx(500.0)
    assert reliance_metrics["options_metrics"]["options_total_pnl"] == pytest.approx(150.0)

    metrics_path = processing["ticker_reports"]["RELIANCE"]["metrics_file"]
    assert metrics_path.exists()
    metrics_on_disk = json.loads(metrics_path.read_text())
    assert metrics_on_disk["trade_metrics"]["approved_trades"] == 1

    manifest_path = processing["manifest_file"]
    manifest = json.loads(manifest_path.read_text())
    assert "RELIANCE" in manifest["components"]["three_file_outputs"]
    assert any("RELIANCE_Base" in path for path in manifest["file_inventory"]["csv_files"])
    assert manifest["portfolio_summary"]["portfolio_overview"]["total_approved_trades"] == 2
    options_summary = manifest["portfolio_summary"]["portfolio_overview"]["options_metrics"]
    assert options_summary["options_total_pnl"] == pytest.approx(150.0 * len(tickers))

    risk_file = processing["ticker_reports"]["RELIANCE"]["risk_report_file"]
    assert risk_file.exists()
    risk_report = json.loads(risk_file.read_text())
    assert risk_report["approved_trade_count"] == 1

    visuals = processing["visualizations"]
    assert visuals["portfolio_level"], "portfolio visuals should be generated"
    portfolio_sample = next(iter(visuals["portfolio_level"].values()))
    assert Path(portfolio_sample).exists()
    inventory = manifest["file_inventory"]
    assert inventory["visualization_files"], "visualizations should be inventoried"
    assert inventory["visualization_hashes"], "visualization hashes should be tracked"
    expected_hash = _hash_file(Path(portfolio_sample))
    viz_hashes = processing["visualization_hashes"]
    assert viz_hashes[str(portfolio_sample)] == expected_hash
    assert manifest["visualization_hashes"][str(portfolio_sample)] == expected_hash
    assert inventory["visualization_hashes"][str(portfolio_sample)] == expected_hash

    portfolio_report_path = processing["reports"]["portfolio_performance"]
    portfolio_report = json.loads(portfolio_report_path.read_text())
    assert portfolio_report["options_summary"]["options_total_pnl"] == pytest.approx(150.0 * len(tickers))


def test_orchestrator_handles_zero_trades(tmp_path: Path) -> None:
    strategy_name = "demo_strategy"
    date_range = "2025-02-01_to_2025-02-05"
    ticker = "RELIANCE"

    results_data = {
        strategy_name: {
            date_range: {
                ticker: _make_results_payload(
                    strategy_name,
                    date_range,
                    ticker,
                    strategy_trades=[],
                    approved_trades=[],
                    risk_report={
                        "original_trade_count": 0,
                        "approved_trade_count": 0,
                        "rejected_trade_count": 0,
                        "approval_rate": 0.0,
                    },
                    options_metrics={},
                )
            }
        }
    }

    orchestrator = EnhancedOutputOrchestrator(base_output_dir=tmp_path)
    processing = orchestrator.process_complete_backtest_results(
        strategy_name=strategy_name,
        date_range=date_range,
        tickers=[ticker],
        results_data=results_data,
        run_id="zero_case",
        strategy_run_dir=tmp_path / "zero_trades_run",
    )

    metrics = processing["ticker_reports"][ticker]["metrics"]
    assert metrics["trade_metrics"]["generated_trades"] == 0
    assert metrics["trade_metrics"]["approved_trades"] == 0
    assert metrics["risk_metrics"]["approval_rate_pct"] == 0
    assert metrics["pnl_metrics"]["total_profit_currency"] == 0
    assert metrics.get("options_metrics", {}) == {}

    manifest = json.loads(processing["manifest_file"].read_text())
    assert manifest["portfolio_summary"]["portfolio_overview"]["total_generated_trades"] == 0
    assert manifest["file_inventory"]["csv_files"], "CSV inventory should still enumerate data artifacts"
    assert manifest["visualization_hashes"] == processing["visualization_hashes"]
    assert manifest["file_inventory"]["visualization_hashes"] == processing["visualization_hashes"]


def test_orchestrator_handles_all_rejections(tmp_path: Path) -> None:
    strategy_name = "demo_strategy"
    date_range = "2025-03-01_to_2025-03-05"
    ticker = "RELIANCE"
    strategy_trades = _make_strategy_trades(ticker)[:1]

    results_data = {
        strategy_name: {
            date_range: {
                ticker: _make_results_payload(
                    strategy_name,
                    date_range,
                    ticker,
                    strategy_trades=strategy_trades,
                    approved_trades=[],
                    risk_report={
                        "original_trade_count": 1,
                        "approved_trade_count": 0,
                        "rejected_trade_count": 1,
                        "approval_rate": 0.0,
                    },
                    options_metrics={"options_total_pnl": 0.0},
                )
            }
        }
    }

    orchestrator = EnhancedOutputOrchestrator(base_output_dir=tmp_path)
    processing = orchestrator.process_complete_backtest_results(
        strategy_name=strategy_name,
        date_range=date_range,
        tickers=[ticker],
        results_data=results_data,
        run_id="rejection_case",
        strategy_run_dir=tmp_path / "rejection_run",
    )

    metrics = processing["ticker_reports"][ticker]["metrics"]
    assert metrics["trade_metrics"]["generated_trades"] == 1
    assert metrics["trade_metrics"]["approved_trades"] == 0
    assert metrics["risk_metrics"]["rejected_trade_count"] == 1
    assert metrics["risk_metrics"]["approval_rate_pct"] == 0

    manifest = json.loads(processing["manifest_file"].read_text())
    assert manifest["portfolio_summary"]["portfolio_overview"]["total_approved_trades"] == 0
    assert manifest["visualization_hashes"] == processing["visualization_hashes"]


def test_orchestrator_ignores_additional_date_ranges(tmp_path: Path) -> None:
    strategy_name = "demo_strategy"
    primary_range = "2025-04-01_to_2025-04-05"
    extra_range = "2025-01-01_to_2025-01-03"
    ticker = "RELIANCE"

    results_data = {
        strategy_name: {
            primary_range: {ticker: _make_results_payload(strategy_name, primary_range, ticker)},
            extra_range: {ticker: _make_results_payload(strategy_name, extra_range, ticker)},
        }
    }

    orchestrator = EnhancedOutputOrchestrator(base_output_dir=tmp_path)
    run_dir = tmp_path / "multi_range_run"
    processing = orchestrator.process_complete_backtest_results(
        strategy_name=strategy_name,
        date_range=primary_range,
        tickers=[ticker],
        results_data=results_data,
        run_id="multi_range",
        strategy_run_dir=run_dir,
    )

    # Ensure only primary range artifacts were materialised
    base_file_primary = run_dir / "data" / "base_data" / f"{ticker}_Base_{primary_range}.csv"
    base_file_extra = run_dir / "data" / "base_data" / f"{ticker}_Base_{extra_range}.csv"
    assert base_file_primary.exists()
    assert not base_file_extra.exists()

    manifest = json.loads(processing["manifest_file"].read_text())
    assert manifest["portfolio_summary"]["portfolio_overview"]["total_approved_trades"] == 1
    assert manifest["visualization_hashes"] == processing["visualization_hashes"]
