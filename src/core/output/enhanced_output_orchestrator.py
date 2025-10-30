# src/core/output/enhanced_output_orchestrator.py
"""
Enhanced Output Orchestrator for Comprehensive Backtesting Infrastructure.

This module orchestrates the complete output system including:
1. Three-file CSV system (base, strategy trades, risk-approved trades)
2. Portfolio-level visualizations
3. Risk management analytics
4. Bias analysis integration
5. Transaction cost analysis
6. Comprehensive reporting

This is the main coordination layer that ensures all components work together.
"""

import hashlib
import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set

import numpy as np
import pandas as pd

from .three_file_system import ThreeFileOutputSystem

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from ..analysis.portfolio_visualization import PortfolioVisualizer
from ..analysis.visualization import StrategyVisualizer

class EnhancedOutputOrchestrator:
    """
    Orchestrates the complete enhanced output system for backtesting.
    """
    
    def __init__(self, base_output_dir: Union[str, Path] = "outputs"):
        """
        Initialize the enhanced output orchestrator.
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = logging.getLogger("EnhancedOutputOrchestrator")

        # Visualization components are initialised lazily when required
        self.portfolio_visualizer: Optional[PortfolioVisualizer] = None
        self.strategy_visualizer: Optional[StrategyVisualizer] = None

        self.logger.debug("EnhancedOutputOrchestrator initialised with base directory %s", self.base_output_dir)

    # ------------------------------------------------------------------
    # Helper loaders
    # ------------------------------------------------------------------
    def _load_dataframe(self, path: Path) -> pd.DataFrame:
        """Safely load a CSV file into a DataFrame."""
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as exc:
            self.logger.error("Failed to read dataframe from %s: %s", path, exc)
            return pd.DataFrame()

    def _load_analysis_json(self, strategy_run_dir: Path, ticker: str, date_range: str) -> Dict[str, Any]:
        """Load per-ticker analysis JSON if available."""
        analysis_path = strategy_run_dir / "analysis_reports" / "individual" / f"{ticker}_Analysis_{date_range}.json"
        if not analysis_path.exists():
            return {}
        try:
            with open(analysis_path, "r") as fh:
                return json.load(fh)
        except Exception as exc:
            self.logger.error("Failed to read analysis report %s: %s", analysis_path, exc)
            return {}

    def _load_metrics_json(self, strategy_run_dir: Path, ticker: str) -> Dict[str, Any]:
        """Load ticker metrics JSON if already materialised."""
        metrics_path = strategy_run_dir / "tickers" / ticker / "metrics.json"
        if not metrics_path.exists():
            return {}
        try:
            with open(metrics_path, "r") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _load_risk_json(self, strategy_run_dir: Path, ticker: str) -> Dict[str, Any]:
        """Load ticker risk report JSON if available."""
        risk_path = strategy_run_dir / "tickers" / ticker / "risk_report.json"
        if not risk_path.exists():
            return {}
        try:
            with open(risk_path, "r") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _compute_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Compute maximum drawdown (as positive percentage) from trade-level returns."""
        if trades_df.empty or 'Profit (%)' not in trades_df.columns:
            return 0.0
        try:
            returns = trades_df['Profit (%)'].astype(float) / 100.0
        except Exception:
            return 0.0
        equity = (1.0 + returns).cumprod()
        if equity.empty:
            return 0.0
        peak = equity.cummax()
        drawdown = (equity / peak) - 1.0
        if drawdown.empty:
            return 0.0
        return float(abs(drawdown.min()) * 100.0)

    def _sum_signal_columns(self, base_df: pd.DataFrame) -> int:
        """Count total signal activations across signal columns."""
        if base_df.empty:
            return 0
        signal_cols = [col for col in base_df.columns if 'signal' in col.lower()]
        if not signal_cols:
            return 0
        try:
            signal_df = base_df[signal_cols].fillna(0)
            # Some signals are boolean; ensure numeric
            signal_df = signal_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            return int(signal_df.abs().sum().sum())
        except Exception:
            return 0
    
    def process_complete_backtest_results(self,
                                          strategy_name: str,
                                          date_range: str,
                                          tickers: List[str],
                                          results_data: Dict[str, Any],
                                          run_id: Optional[str] = None,
                                          strategy_run_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Process complete backtest results through the enhanced output system.
        
        Args:
            strategy_name: Name of the strategy
            date_range: Date range string
            tickers: List of tickers processed
            results_data: Complete results data from backtest
            run_id: Optional run identifier
            strategy_run_dir: Optional pre-created strategy run directory
            
        Returns:
            Dictionary with all output paths and analysis results
        """
        
        # Use provided strategy_run_dir or create new one
        if strategy_run_dir:
            strategy_run_dir = Path(strategy_run_dir)
        else:
            from src.runners.utils.naming import create_monolith_directory_structure
            if not run_id:
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_run_dir = Path(create_monolith_directory_structure(
                str(self.base_output_dir), strategy_name, date_range, run_id
            ))
        
        self.logger.info(f"Processing complete backtest results for {strategy_name} in {strategy_run_dir}")
          # Initialize systems
        three_file_system = ThreeFileOutputSystem(strategy_run_dir)
        # Use 'auto' trade source to fallback from risk-approved to strategy trades when needed
        self.portfolio_visualizer = PortfolioVisualizer(output_dir=strategy_run_dir, trade_source='auto')
        self.strategy_visualizer = StrategyVisualizer(strategy_run_dir / "visualizations")
        
        # Process results
        processing_results = {
            'strategy_run_dir': strategy_run_dir,
            'three_file_outputs': {},
            'visualizations': {},
            'analytics': {},
            'reports': {},
            'summary': {}
        }
        
        try:
            # 1. Process three-file system for each ticker
            self.logger.info("Processing three-file system...")
            for ticker in tickers:
                ticker_results = self._process_ticker_three_files(
                    three_file_system, ticker, date_range, results_data
                )
                processing_results['three_file_outputs'][ticker] = ticker_results
            
            # 2. Create comprehensive analytics
            self.logger.info("Creating comprehensive analytics...")
            for ticker in tickers:
                analytics = three_file_system.create_comprehensive_analysis(ticker, date_range)
                processing_results['analytics'][ticker] = analytics
            
            # 3. Create portfolio-level analysis
            self.logger.info("Creating portfolio-level analysis...")
            portfolio_analysis = three_file_system.create_portfolio_three_file_analysis(date_range, tickers)
            processing_results['analytics']['portfolio'] = portfolio_analysis
            
            # 4. Create ticker-level reports (bias, metrics, risk, config)
            self.logger.info("Creating ticker-level reports...")
            ticker_reports_results = self._create_ticker_level_reports(
                strategy_run_dir, strategy_name, date_range, tickers, results_data
            )
            processing_results['ticker_reports'] = ticker_reports_results
            
            # 5. Generate visualizations using enriched metrics
            self.logger.info("Generating visualization suite...")
            visualization_results = self._generate_all_visualizations(
                strategy_run_dir,
                strategy_name,
                date_range,
                tickers,
                results_data,
                ticker_reports_results
            )
            processing_results['visualizations'] = visualization_results
            processing_results['visualization_hashes'] = self._compute_visualization_hashes(visualization_results)
            
            # 6. Create enhanced reports
            self.logger.info("Creating enhanced reports...")
            report_results = self._create_enhanced_reports(
                strategy_run_dir, strategy_name, date_range, tickers, ticker_reports_results
            )
            processing_results['reports'] = report_results
            
            # 7. Generate executive summary
            self.logger.info("Generating executive summary...")
            executive_summary = self._generate_executive_summary(
                strategy_name, date_range, tickers, processing_results
            )
            processing_results['summary'] = executive_summary
            
            # 8. Create final manifest file
            manifest = self._create_output_manifest(processing_results)
            manifest_file = strategy_run_dir / "output_manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            processing_results['manifest_file'] = manifest_file
            
            self.logger.info(f"Successfully processed complete backtest results")
            
        except Exception as e:
            self.logger.error(f"Error processing backtest results: {e}")
            processing_results['error'] = str(e)
    
        return processing_results

    def _extract_ticker_results(self,
                                 results_data: Dict[str, Any],
                                 strategy_name: str,
                                 date_range: str,
                                 ticker: str) -> Dict[str, Any]:
        """Extract nested ticker results from execution payload."""
        if not results_data or not strategy_name:
            return {}
        strategy_bucket = results_data.get(strategy_name, {})
        if not isinstance(strategy_bucket, dict):
            return {}
        date_bucket = strategy_bucket.get(date_range, {})
        if not isinstance(date_bucket, dict):
            return {}
        ticker_payload = date_bucket.get(ticker, {})
        return ticker_payload if isinstance(ticker_payload, dict) else {}

    def _ensure_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert arbitrary payloads to a DataFrame for analysis."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if data is None:
            return pd.DataFrame()
        if isinstance(data, dict):
            try:
                return pd.DataFrame(data)
            except ValueError:
                return pd.DataFrame()
        if isinstance(data, list):
            try:
                return pd.DataFrame(data)
            except ValueError:
                return pd.DataFrame()
        return pd.DataFrame()

    def _hash_file(self, file_path: Path) -> Optional[str]:
        """Compute a deterministic SHA-256 hash for the given file."""
        try:
            path = Path(file_path)
            if not path.exists() or not path.is_file():
                return None
            hasher = hashlib.sha256()
            with open(path, 'rb') as fh:
                for chunk in iter(lambda: fh.read(8192), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as exc:
            self.logger.warning("Failed to hash visualization %s: %s", file_path, exc)
            return None

    def _collect_visualization_paths(self, visualizations: Any) -> Set[Path]:
        """Extract all file paths from the visualization structure."""
        paths: Set[Path] = set()

        if isinstance(visualizations, dict):
            for value in visualizations.values():
                paths.update(self._collect_visualization_paths(value))
        elif isinstance(visualizations, (list, tuple, set)):
            for item in visualizations:
                paths.update(self._collect_visualization_paths(item))
        elif isinstance(visualizations, (Path, str)):
            candidate = Path(visualizations)
            paths.add(candidate)

        return paths

    def _compute_visualization_hashes(self, visualizations: Dict[str, Any]) -> Dict[str, str]:
        """Compute hashes for all generated visualization files."""
        hashes: Dict[str, str] = {}
        for path in self._collect_visualization_paths(visualizations):
            digest = self._hash_file(path)
            if digest:
                hashes[str(path)] = digest
        return hashes

    def _build_ticker_metrics(self,
                              ticker: str,
                              strategy_name: str,
                              date_range: str,
                              base_df: pd.DataFrame,
                              strategy_trades_df: pd.DataFrame,
                              approved_trades_df: pd.DataFrame,
                              ticker_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble structured metrics for a ticker."""
        metrics_payload: Dict[str, Any] = {
            'ticker': ticker,
            'strategy': strategy_name,
            'date_range': date_range,
            'generated_at': datetime.now().isoformat()
        }

        # Data coverage
        data_metrics: Dict[str, Any] = {
            'base_data_points': int(base_df.shape[0]) if base_df is not None else 0,
            'signal_events': self._sum_signal_columns(base_df),
            'columns': list(base_df.columns) if not base_df.empty else []
        }
        if not base_df.empty and 'timestamp' in base_df.columns:
            timestamps = pd.to_datetime(base_df['timestamp'], errors='coerce').dropna()
            if not timestamps.empty:
                data_metrics['date_start'] = timestamps.min().isoformat()
                data_metrics['date_end'] = timestamps.max().isoformat()
                data_metrics['session_count'] = int(timestamps.dt.normalize().nunique())
        metrics_payload['data_metrics'] = data_metrics

        # Trade level metrics
        generated_trades = int(strategy_trades_df.shape[0])
        approved_trades = int(approved_trades_df.shape[0])
        rejected_trades = max(generated_trades - approved_trades, 0)

        profits_series = pd.Series(dtype=float)
        if 'Profit (%)' in approved_trades_df.columns:
            profits_series = pd.to_numeric(approved_trades_df['Profit (%)'], errors='coerce')
        valid_profits = profits_series.dropna()
        winning_trades = int((valid_profits > 0).sum())
        losing_trades = int((valid_profits <= 0).sum())
        win_rate_pct = float(round((winning_trades / len(valid_profits)) * 100, 2)) if len(valid_profits) else 0.0

        trade_metrics: Dict[str, Any] = {
            'generated_trades': generated_trades,
            'approved_trades': approved_trades,
            'rejected_trades': rejected_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate_pct
        }
        if 'Trade Duration (min)' in approved_trades_df.columns:
            durations = pd.to_numeric(approved_trades_df['Trade Duration (min)'], errors='coerce').dropna()
            if not durations.empty:
                trade_metrics['avg_trade_duration_min'] = float(round(durations.mean(), 2))
        metrics_payload['trade_metrics'] = trade_metrics

        # P&L analytics
        pnl_metrics: Dict[str, Any] = {
            'max_drawdown_pct': float(round(self._compute_max_drawdown(approved_trades_df), 2))
        }
        if 'Profit (Currency)' in approved_trades_df.columns:
            pnl_currency = pd.to_numeric(approved_trades_df['Profit (Currency)'], errors='coerce').dropna()
            pnl_metrics['total_profit_currency'] = float(round(pnl_currency.sum(), 2))
            if not pnl_currency.empty:
                pnl_metrics['avg_profit_currency'] = float(round(pnl_currency.mean(), 2))
                pnl_metrics['best_trade_currency'] = float(round(pnl_currency.max(), 2))
                pnl_metrics['worst_trade_currency'] = float(round(pnl_currency.min(), 2))
        else:
            pnl_metrics['total_profit_currency'] = 0.0

        if len(valid_profits):
            pnl_metrics['total_profit_pct'] = float(round(valid_profits.sum(), 2))
            pnl_metrics['avg_profit_pct'] = float(round(valid_profits.mean(), 2))
            pnl_metrics['best_trade_pct'] = float(round(valid_profits.max(), 2))
            pnl_metrics['worst_trade_pct'] = float(round(valid_profits.min(), 2))
        else:
            pnl_metrics.setdefault('total_profit_pct', 0.0)
        metrics_payload['pnl_metrics'] = pnl_metrics

        # Risk lens
        risk_report = ticker_data.get('risk_report', {}) or {}
        risk_metrics: Dict[str, Any] = {
            'risk_management_enabled': not risk_report.get('risk_management_disabled', False),
            'original_trade_count': int(risk_report.get('original_trade_count', generated_trades)),
            'approved_trade_count': int(risk_report.get('approved_trade_count', approved_trades)),
            'rejected_trade_count': int(risk_report.get('rejected_trade_count', rejected_trades))
        }
        approval_rate = risk_report.get('approval_rate')
        if approval_rate is None:
            approval_rate = risk_report.get('approval_rate_pct')
        if approval_rate is None and risk_metrics['original_trade_count']:
            approval_rate = risk_metrics['approved_trade_count'] / max(risk_metrics['original_trade_count'], 1)
        try:
            if approval_rate is not None:
                approval_rate_val = float(approval_rate)
                if approval_rate_val <= 1:
                    approval_rate_val *= 100.0
                risk_metrics['approval_rate_pct'] = float(round(approval_rate_val, 2))
        except (TypeError, ValueError):
            pass

        rejection_rate = risk_report.get('rejection_rate')
        try:
            if rejection_rate is not None:
                rejection_rate_val = float(rejection_rate)
                if rejection_rate_val <= 1:
                    rejection_rate_val *= 100.0
                risk_metrics['rejection_rate_pct'] = float(round(rejection_rate_val, 2))
        except (TypeError, ValueError):
            pass
        if 'sample_rejections' in risk_report:
            risk_metrics['sample_rejections'] = risk_report['sample_rejections']
        if 'risk_management_error' in risk_report:
            risk_metrics['risk_management_error'] = risk_report['risk_management_error']
        metrics_payload['risk_metrics'] = risk_metrics

        # Bias snapshot
        bias_report = ticker_data.get('bias_report')
        if bias_report:
            if isinstance(bias_report, dict):
                bias_summary = {
                    'violation_count': len(bias_report.get('violations', [])),
                    'issues': bias_report.get('issues') or bias_report.get('violations') or bias_report
                }
            elif isinstance(bias_report, list):
                bias_summary = {
                    'violation_count': len(bias_report),
                    'issues': bias_report
                }
            else:
                bias_summary = {'issues': bias_report}
            metrics_payload['bias_summary'] = bias_summary

        strategy_metrics = ticker_data.get('metrics', {})
        if strategy_metrics:
            metrics_payload['strategy_metrics'] = strategy_metrics

        options_payload = ticker_data.get('options_metrics')
        if not options_payload and isinstance(strategy_metrics, dict):
            inferred_options = {
                key: value for key, value in strategy_metrics.items()
                if isinstance(key, str) and key.lower().startswith(('options_', 'option_'))
            }
            options_payload = inferred_options if inferred_options else None
        if options_payload:
            normalized_options: Dict[str, Any] = {}
            for key, value in options_payload.items():
                try:
                    normalized_options[key] = float(value)
                except (TypeError, ValueError):
                    normalized_options[key] = value
            metrics_payload['options_metrics'] = normalized_options

        return metrics_payload

    def _create_ticker_level_reports(self,
                                     strategy_run_dir: Path,
                                     strategy_name: str,
                                     date_range: str,
                                     tickers: List[str],
                                     results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Materialise per-ticker analytics, bias reports, and configs."""
        ticker_reports: Dict[str, Any] = {}

        for ticker in tickers:
            try:
                ticker_payload = self._extract_ticker_results(results_data, strategy_name, date_range, ticker)
                ticker_dir = strategy_run_dir / "tickers" / ticker
                ticker_dir.mkdir(parents=True, exist_ok=True)

                base_df = self._ensure_dataframe(ticker_payload.get('base_data'))
                strategy_trades_df = self._ensure_dataframe(ticker_payload.get('strategy_trades'))
                approved_trades_df = self._ensure_dataframe(ticker_payload.get('trades'))
                if approved_trades_df.empty and strategy_trades_df.empty:
                    # Fall back to whatever trade payload exists
                    approved_trades_df = self._ensure_dataframe(ticker_payload.get('approved_trades'))

                metrics_payload = self._build_ticker_metrics(
                    ticker, strategy_name, date_range, base_df,
                    strategy_trades_df, approved_trades_df, ticker_payload
                )

                metrics_file = ticker_dir / "metrics.json"
                with open(metrics_file, 'w') as fh:
                    json.dump(metrics_payload, fh, indent=2, default=str)

                ticker_entry: Dict[str, Any] = {
                    'metrics_file': metrics_file,
                    'metrics': metrics_payload
                }

                risk_report = ticker_payload.get('risk_report')
                if risk_report:
                    risk_file = ticker_dir / "risk_report.json"
                    with open(risk_file, 'w') as fh:
                        json.dump(risk_report, fh, indent=2, default=str)
                    ticker_entry['risk_report_file'] = risk_file
                    ticker_entry['risk_report'] = risk_report

                bias_report = ticker_payload.get('bias_report')
                if bias_report:
                    bias_file = ticker_dir / "bias_report.json"
                    with open(bias_file, 'w') as fh:
                        json.dump(bias_report, fh, indent=2, default=str)
                    ticker_entry['bias_report_file'] = bias_file
                    ticker_entry['bias_report'] = bias_report

                strategy_metrics = ticker_payload.get('metrics', {})
                config_payload = {
                    'strategy': strategy_name,
                    'ticker': ticker,
                    'date_range': date_range,
                    'parameters': strategy_metrics.get('Parameters'),
                    'generated_at': datetime.now().isoformat()
                }
                config_file = ticker_dir / "config.json"
                with open(config_file, 'w') as fh:
                    json.dump(config_payload, fh, indent=2, default=str)
                ticker_entry['config_file'] = config_file

                ticker_reports[ticker] = ticker_entry

            except Exception as exc:
                self.logger.error("Failed to create ticker report for %s: %s", ticker, exc)
                ticker_reports[ticker] = {'error': str(exc)}

        return ticker_reports
    
    def _process_ticker_three_files(self, three_file_system: ThreeFileOutputSystem, 
                                    ticker: str, date_range: str, 
                                    results_data: Dict[str, Any]) -> Dict[str, Path]:
        """Process three-file system for a single ticker."""
        ticker_outputs = {}
        
        try:
            # Extract ticker-specific data from the nested results structure
            # Navigate: strategy -> date_range -> ticker
            strategy_name = list(results_data.keys())[0] if results_data else None
            if strategy_name and strategy_name in results_data:
                strategy_results = results_data[strategy_name]
                if date_range in strategy_results and ticker in strategy_results[date_range]:
                    ticker_data = strategy_results[date_range][ticker]
                else:
                    ticker_data = {}
            else:
                ticker_data = {}
              # 1. Save base file (price data + signals + indicators)
            base_data_dict = ticker_data.get('base_data', {})
            if base_data_dict is not None and len(base_data_dict) > 0:
                # Convert dict to DataFrame if needed
                if isinstance(base_data_dict, dict):
                    base_data = pd.DataFrame(base_data_dict)
                else:
                    base_data = base_data_dict
                    
                if not base_data.empty:
                    base_file = three_file_system.save_base_file(ticker, date_range, base_data)
                    ticker_outputs['base_file'] = base_file
            
            # 2. Save strategy trades file (all trades generated by strategy)
            strategy_trades = ticker_data.get('strategy_trades', [])
            if not strategy_trades:
                strategy_trades = ticker_data.get('trades', [])
            strategy_metadata = ticker_data.get('strategy_metadata', {})
            strategy_trades_file = three_file_system.save_strategy_trades_file(
                ticker, date_range, strategy_trades, strategy_metadata
            )
            ticker_outputs['strategy_trades_file'] = strategy_trades_file
            
            # 3. Save risk-approved trades file (trades that passed risk management)
            # For now, assume all trades are risk-approved (can be enhanced later)
            approved_trades = ticker_data.get('trades', strategy_trades)
            risk_analysis = ticker_data.get('risk_report', {})
            risk_approved_file = three_file_system.save_risk_approved_trades_file(
                ticker, date_range, approved_trades, risk_analysis
            )
            ticker_outputs['risk_approved_file'] = risk_approved_file
            
        except Exception as e:
            self.logger.error(f"Error processing three-files for {ticker}: {e}")
            ticker_outputs['error'] = str(e)
        
        return ticker_outputs
    
    def _generate_all_visualizations(self,
                                     strategy_run_dir: Path,
                                     strategy_name: str,
                                     date_range: str,
                                     tickers: List[str],
                                     results_data: Dict[str, Any],
                                     ticker_reports: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all visualization outputs."""
        visualizations = {
            'portfolio_level': {},
            'individual_tickers': {},
            'strategy_analysis': {},
            'risk_analysis': {}
        }
        
        try:
            # Portfolio-level visualizations
            portfolio_viz = self.portfolio_visualizer.create_portfolio_dashboard(
                strategy_run_dir, date_range, tickers
            )
            visualizations['portfolio_level'] = portfolio_viz
            
            # Individual ticker visualizations
            for ticker in tickers:
                ticker_viz = self.portfolio_visualizer.create_individual_ticker_dashboard(
                    strategy_run_dir, ticker, date_range
                )
                visualizations['individual_tickers'][ticker] = ticker_viz
            
            # Strategy analysis visualizations (if trade data available)
            strategy_visualizations = self._create_strategy_visualizations(
                strategy_run_dir, date_range, strategy_name, tickers, results_data
            )
            visualizations['strategy_analysis'] = strategy_visualizations
            
            # Risk analysis visualizations
            risk_visualizations = self._create_risk_visualizations(
                strategy_run_dir, date_range, tickers, ticker_reports
            )
            visualizations['risk_analysis'] = risk_visualizations
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _create_strategy_visualizations(self,
                                        strategy_run_dir: Path,
                                        date_range: str,
                                        strategy_name: str,
                                        tickers: List[str],
                                        results_data: Dict[str, Any]) -> Dict[str, Path]:
        """Create strategy-specific visualizations."""
        strategy_viz = {}
        
        try:
            # Collect all trades for strategy analysis
            all_trades = []
            for ticker in tickers:
                ticker_data = self._extract_ticker_results(results_data, strategy_name, date_range, ticker)
                trades_source = (
                    ticker_data.get('trades')
                    or ticker_data.get('approved_trades')
                    or ticker_data.get('strategy_trades')
                    or []
                )
                for trade in trades_source:
                    trade_record = dict(trade)
                    trade_record['Ticker'] = ticker
                    all_trades.append(trade_record)
            
            if all_trades:
                trades_df = pd.DataFrame(all_trades)
                
                # Convert timestamp columns
                for col in ['Entry Time', 'Exit Time']:
                    if col in trades_df.columns:
                        trades_df[col] = pd.to_datetime(trades_df[col], errors='coerce')
                
                # Create equity curves
                equity_curves = self.strategy_visualizer.calculate_equity_curve(trades_df)
                
                # Generate strategy visualizations
                viz_dir = strategy_run_dir / "visualizations" / "strategy"
                viz_dir.mkdir(parents=True, exist_ok=True)
                
                # Equity curve
                equity_file = viz_dir / f"strategy_equity_curve_{date_range}.png"
                self.strategy_visualizer.plot_equity_curve(
                    equity_curves, 
                    title=f"Strategy Equity Curve - {date_range}",
                    save_path=str(equity_file)
                )
                strategy_viz['equity_curve'] = equity_file
                
                # Trade distribution
                trade_dist_file = viz_dir / f"trade_distribution_{date_range}.png"
                self.strategy_visualizer.plot_trade_distribution(trades_df, save_path=str(trade_dist_file))
                strategy_viz['trade_distribution'] = trade_dist_file
                
                # Correlation heatmap
                corr_file = viz_dir / f"correlation_heatmap_{date_range}.png"
                self.strategy_visualizer.plot_correlation_heatmap(trades_df, save_path=str(corr_file))
                strategy_viz['correlation_heatmap'] = corr_file
                
                # Performance metrics
                summary_data = []
                for ticker in tickers:
                    ticker_trades = trades_df[trades_df['Ticker'] == ticker]
                    if not ticker_trades.empty:
                        metrics = {
                            'Ticker': ticker,
                            'Total Trades': len(ticker_trades),
                            'Average Profit (%)': ticker_trades['Profit (%)'].mean() if 'Profit (%)' in ticker_trades.columns else 0,
                            'Wins': len(ticker_trades[ticker_trades['Profit (%)'] > 0]) if 'Profit (%)' in ticker_trades.columns else 0
                        }
                        summary_data.append(metrics)
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    perf_file = viz_dir / f"performance_metrics_{date_range}.png"
                    self.strategy_visualizer.plot_performance_metrics(summary_df, save_path=str(perf_file))
                    strategy_viz['performance_metrics'] = perf_file
            
        except Exception as e:
            self.logger.error(f"Error creating strategy visualizations: {e}")
            strategy_viz['error'] = str(e)
        
        return strategy_viz
    
    def _create_risk_visualizations(self,
                                    strategy_run_dir: Path,
                                    date_range: str,
                                    tickers: List[str],
                                    ticker_reports: Dict[str, Any]) -> Dict[str, Path]:
        """Create risk analysis visualizations."""
        risk_viz = {}
        
        try:
            import matplotlib.pyplot as plt
            
            # Create risk analysis directory
            viz_dir = strategy_run_dir / "visualizations" / "risk"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect risk data
            risk_data = []
            for ticker in tickers:
                report_entry = ticker_reports.get(ticker, {})
                metrics_payload = report_entry.get('metrics', {})
                trade_metrics = metrics_payload.get('trade_metrics', {})
                risk_metrics = metrics_payload.get('risk_metrics', {})
                risk_report = report_entry.get('risk_report', {})

                original_trades = int(risk_metrics.get(
                    'original_trade_count',
                    trade_metrics.get('generated_trades', trade_metrics.get('total_trades', 0))
                ))
                approved_trades = int(risk_metrics.get(
                    'approved_trade_count',
                    trade_metrics.get('approved_trades', trade_metrics.get('total_trades', 0))
                ))
                rejected_trades = int(risk_metrics.get('rejected_trade_count', max(original_trades - approved_trades, 0)))

                approval_rate_pct = risk_metrics.get('approval_rate_pct')
                if approval_rate_pct is None:
                    approval_rate_pct = risk_report.get('approval_rate_pct')
                if approval_rate_pct is None:
                    approval_rate = risk_report.get('approval_rate', 0)
                    approval_rate_pct = approval_rate * 100 if approval_rate <= 1 else approval_rate
                if approval_rate_pct is None and original_trades:
                    approval_rate_pct = (approved_trades / max(original_trades, 1)) * 100
                approval_rate_pct = float(approval_rate_pct or 0.0)
                approval_rate_ratio = approval_rate_pct / 100.0

                risk_data.append({
                    'ticker': ticker,
                    'original_trades': original_trades,
                    'approved_trades': approved_trades,
                    'rejected_trades': rejected_trades,
                    'approval_rate': approval_rate_ratio
                })
            
            if risk_data:
                # Risk approval rates chart
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                df_risk = pd.DataFrame(risk_data)
                
                # Approval rates bar chart
                bars = ax1.bar(df_risk['ticker'], df_risk['approval_rate'] * 100, 
                              color='green', alpha=0.7)
                ax1.set_title('Risk Approval Rates by Ticker')
                ax1.set_ylabel('Approval Rate (%)')
                ax1.tick_params(axis='x', rotation=45)
                
                # Add percentage labels
                for bar, rate in zip(bars, df_risk['approval_rate'] * 100):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                            f'{rate:.1f}%', ha='center', va='bottom')
                
                # Trade approval/rejection stacked bar
                x = range(len(df_risk))
                ax2.bar(x, df_risk['approved_trades'], label='Approved', color='green', alpha=0.7)
                ax2.bar(x, df_risk['rejected_trades'], bottom=df_risk['approved_trades'], 
                       label='Rejected', color='red', alpha=0.7)
                
                ax2.set_title('Trade Approval vs Rejection')
                ax2.set_ylabel('Number of Trades')
                ax2.set_xticks(x)
                ax2.set_xticklabels(df_risk['ticker'], rotation=45)
                ax2.legend()
                
                plt.tight_layout()
                
                risk_file = viz_dir / f"risk_analysis_{date_range}.png"
                plt.savefig(risk_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                risk_viz['risk_analysis'] = risk_file
        
        except Exception as e:
            self.logger.error(f"Error creating risk visualizations: {e}")
            risk_viz['error'] = str(e)
        
        return risk_viz
    
    def _create_enhanced_reports(self, strategy_run_dir: Path, strategy_name: str,
                                 date_range: str, tickers: List[str],
                                 ticker_reports: Dict[str, Any]) -> Dict[str, Path]:
        """Create enhanced analytical reports derived from ticker metrics."""
        reports: Dict[str, Path] = {}

        ticker_metrics_map = {
            ticker: ticker_reports.get(ticker, {}).get('metrics', {})
            for ticker in tickers
        }
        ticker_risk_map = {
            ticker: ticker_reports.get(ticker, {}).get('risk_report', {})
            for ticker in tickers
        }
        ticker_analysis_map = {
            ticker: self._load_analysis_json(strategy_run_dir, ticker, date_range)
            for ticker in tickers
        }

        try:
            reports_dir = strategy_run_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            exec_summary = self._create_executive_summary_report(
                strategy_name, date_range, tickers, ticker_metrics_map
            )
            exec_file = reports_dir / f"executive_summary_{date_range}.json"
            with open(exec_file, 'w') as fh:
                json.dump(exec_summary, fh, indent=2, default=str)
            reports['executive_summary'] = exec_file

            risk_report = self._create_risk_management_report(
                tickers, ticker_metrics_map, ticker_risk_map
            )
            risk_file = reports_dir / f"risk_management_report_{date_range}.json"
            with open(risk_file, 'w') as fh:
                json.dump(risk_report, fh, indent=2, default=str)
            reports['risk_management'] = risk_file

            portfolio_report = self._create_portfolio_performance_report(
                tickers, ticker_metrics_map
            )
            portfolio_file = reports_dir / f"portfolio_performance_{date_range}.json"
            with open(portfolio_file, 'w') as fh:
                json.dump(portfolio_report, fh, indent=2, default=str)
            reports['portfolio_performance'] = portfolio_file

            signal_report = self._create_signal_analysis_report(
                tickers, ticker_analysis_map
            )
            signal_file = reports_dir / f"signal_analysis_{date_range}.json"
            with open(signal_file, 'w') as fh:
                json.dump(signal_report, fh, indent=2, default=str)
            reports['signal_analysis'] = signal_file

            recommendations = self._generate_recommendations_report(
                tickers, ticker_metrics_map, ticker_risk_map
            )
            rec_file = reports_dir / f"recommendations_{date_range}.json"
            with open(rec_file, 'w') as fh:
                json.dump(recommendations, fh, indent=2, default=str)
            reports['recommendations'] = rec_file

        except Exception as exc:
            self.logger.error("Error creating enhanced reports: %s", exc)
            reports['error'] = str(exc)

        return reports

    def _generate_executive_summary(self,
                                    strategy_name: str,
                                    date_range: str,
                                    tickers: List[str],
                                    processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Produce high-level summary for manifests and human review."""
        ticker_reports = processing_results.get('ticker_reports', {})
        metrics_payloads = [
            report.get('metrics', {}) for report in ticker_reports.values()
            if isinstance(report, dict) and report.get('metrics')
        ]

        total_data_points = sum(
            metric.get('data_metrics', {}).get('base_data_points', 0) for metric in metrics_payloads
        )
        total_generated_trades = sum(
            metric.get('trade_metrics', {}).get('generated_trades', 0) for metric in metrics_payloads
        )
        total_approved_trades = sum(
            metric.get('trade_metrics', {}).get('approved_trades', 0) for metric in metrics_payloads
        )
        total_profit_currency = sum(
            metric.get('pnl_metrics', {}).get('total_profit_currency', 0.0) for metric in metrics_payloads
        )
        total_profit_pct = sum(
            metric.get('pnl_metrics', {}).get('total_profit_pct', 0.0) for metric in metrics_payloads
        )

        drawdowns = [
            metric.get('pnl_metrics', {}).get('max_drawdown_pct', 0.0) for metric in metrics_payloads
        ]
        max_drawdown = max(drawdowns) if drawdowns else 0.0

        options_totals: Dict[str, float] = {}
        for metric in metrics_payloads:
            for key, value in (metric.get('options_metrics') or {}).items():
                try:
                    options_totals[key] = options_totals.get(key, 0.0) + float(value)
                except (TypeError, ValueError):
                    continue

        performance_ranking = sorted(
            (
                (
                    metric.get('ticker'),
                    metric.get('pnl_metrics', {}).get('total_profit_currency', 0.0)
                )
                for metric in metrics_payloads if metric.get('ticker')
            ),
            key=lambda item: item[1],
            reverse=True
        )

        alerts: List[str] = []
        for metric in metrics_payloads:
            trade_metrics = metric.get('trade_metrics', {})
            if trade_metrics.get('generated_trades', 0) == 0:
                alerts.append(f"{metric.get('ticker', 'UNKNOWN')}: no trades generated.")
            elif trade_metrics.get('approved_trades', 0) == 0:
                alerts.append(f"{metric.get('ticker', 'UNKNOWN')}: all trades rejected by risk.")

        summary = {
            'report_type': 'executive_summary',
            'strategy_name': strategy_name,
            'date_range': date_range,
            'generated_at': datetime.now().isoformat(),
            'tickers_covered': tickers,
            'portfolio_overview': {
                'total_tickers': len(tickers),
                'total_data_points': total_data_points,
                'total_generated_trades': total_generated_trades,
                'total_approved_trades': total_approved_trades,
                'total_profit_currency': float(round(total_profit_currency, 2)),
                'total_profit_pct': float(round(total_profit_pct, 2)),
                'max_drawdown_pct': float(round(max_drawdown, 2))
            },
            'leaders': performance_ranking[:3],
            'laggards': performance_ranking[-3:] if performance_ranking else [],
            'alerts': alerts
        }
        if options_totals:
            summary['portfolio_overview']['options_metrics'] = {
                key: float(round(value, 4)) for key, value in options_totals.items()
            }
        return summary

    def _create_executive_summary_report(self, strategy_name: str, date_range: str,
                                         tickers: List[str],
                                         ticker_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create executive summary report."""
        total_strategy_trades = 0
        total_approved_trades = 0
        total_base_points = 0

        for ticker in tickers:
            metrics = ticker_metrics.get(ticker, {})
            trade_metrics = metrics.get('trade_metrics', {})
            data_metrics = metrics.get('data_metrics', {})

            total_strategy_trades += int(trade_metrics.get('generated_trades', trade_metrics.get('total_trades', 0)))
            total_approved_trades += int(trade_metrics.get('approved_trades', trade_metrics.get('total_trades', 0)))
            total_base_points += int(data_metrics.get('base_data_points', 0))

        overall_approval_rate = float(round((total_approved_trades / total_strategy_trades) * 100, 2)) if total_strategy_trades else 0.0

        return {
            'report_type': 'executive_summary',
            'strategy_name': strategy_name,
            'date_range': date_range,
            'generated_at': datetime.now().isoformat(),
            'portfolio_overview': {
                'total_tickers': len(tickers),
                'tickers_list': tickers,
                'total_base_data_points': total_base_points,
                'total_strategy_trades': total_strategy_trades,
                'total_approved_trades': total_approved_trades,
                'overall_approval_rate': overall_approval_rate
            },
            'key_insights': [
                f"Processed {len(tickers)} tickers with {total_base_points:,} total data points",
                f"Strategy generated {total_strategy_trades} trades across all tickers",
                f"Risk management approved {total_approved_trades} trades ({overall_approval_rate:.1f}% approval rate)"
            ],
            'system_status': {
                'three_file_system': 'Operational',
                'visualization_system': 'Operational',
                'risk_management': 'Operational',
                'portfolio_analysis': 'Operational'
            }
        }

    def _create_risk_management_report(self, tickers: List[str],
                                        ticker_metrics: Dict[str, Dict[str, Any]],
                                        ticker_risk: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create detailed risk management report."""
        risk_summary = {
            'report_type': 'risk_management',
            'generated_at': datetime.now().isoformat(),
            'overall_metrics': {},
            'ticker_breakdown': {},
            'rejection_analysis': {},
            'recommendations': []
        }

        total_generated = 0
        total_approved = 0
        all_rejection_reasons: Dict[str, int] = {}

        for ticker in tickers:
            metrics = ticker_metrics.get(ticker, {}).get('trade_metrics', {})
            generated = int(metrics.get('generated_trades', metrics.get('total_trades', 0)))
            approved = int(metrics.get('approved_trades', metrics.get('total_trades', 0)))
            rejected = max(generated - approved, 0)
            approval_rate = float(round((approved / generated) * 100, 2)) if generated else 0.0

            risk_info = ticker_risk.get(ticker, {}).get('risk_management', {})
            rejection_reasons = risk_info.get('rejection_reasons', {})
            for reason, count in rejection_reasons.items():
                all_rejection_reasons[reason] = all_rejection_reasons.get(reason, 0) + int(count)

            risk_summary['ticker_breakdown'][ticker] = {
                'trades_generated': generated,
                'trades_approved': approved,
                'trades_rejected': rejected,
                'approval_rate_pct': approval_rate,
                'rejection_reasons': rejection_reasons
            }

            total_generated += generated
            total_approved += approved

        overall_approval_rate = float(round((total_approved / total_generated) * 100, 2)) if total_generated else 0.0
        overall_rejection_rate = float(round(((total_generated - total_approved) / total_generated) * 100, 2)) if total_generated else 0.0

        risk_summary['overall_metrics'] = {
            'total_trades_generated': total_generated,
            'total_trades_approved': total_approved,
            'overall_approval_rate': overall_approval_rate,
            'overall_rejection_rate': overall_rejection_rate
        }

        risk_summary['rejection_analysis'] = {
            'most_common_reasons': dict(sorted(all_rejection_reasons.items(), key=lambda x: x[1], reverse=True)),
            'total_rejections': int(sum(all_rejection_reasons.values()))
        }

        if overall_approval_rate < 10:
            risk_summary['recommendations'].append("Very low approval rate - consider relaxing risk parameters")
        elif overall_approval_rate > 90:
            risk_summary['recommendations'].append("Very high approval rate - consider tightening risk parameters")
        else:
            risk_summary['recommendations'].append("Risk parameters appear well-calibrated")

        return risk_summary

    def _create_portfolio_performance_report(self, tickers: List[str],
                                              ticker_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create portfolio performance analysis report."""
        total_trades = 0
        total_wins = 0
        total_losses = 0
        total_profit_ccy = 0.0
        total_profit_pct = 0.0
        total_drawdown = 0.0
        options_totals: Dict[str, float] = {}

        ticker_breakdown = {}

        for ticker in tickers:
            metrics = ticker_metrics.get(ticker, {})
            trade_metrics = metrics.get('trade_metrics', {})
            pnl_metrics = metrics.get('pnl_metrics', {})
            options_metrics = metrics.get('options_metrics', {})

            trades = int(trade_metrics.get('total_trades', 0))
            wins = int(trade_metrics.get('winning_trades', 0))
            losses = int(trade_metrics.get('losing_trades', 0))
            profit_ccy = float(pnl_metrics.get('total_profit_currency', 0.0))
            profit_pct = float(pnl_metrics.get('total_profit_pct', 0.0))
            drawdown = float(pnl_metrics.get('max_drawdown_pct', 0.0))

            total_trades += trades
            total_wins += wins
            total_losses += losses
            total_profit_ccy += profit_ccy
            total_profit_pct += profit_pct
            total_drawdown = max(total_drawdown, drawdown)

            ticker_breakdown[ticker] = {
                'total_trades': trades,
                'winning_trades': wins,
                'losing_trades': losses,
                'total_profit_currency': profit_ccy,
                'total_profit_pct': profit_pct,
                'max_drawdown_pct': drawdown
            }
            if options_metrics:
                normalized_options: Dict[str, float] = {}
                for key, value in options_metrics.items():
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        continue
                    normalized_options[key] = numeric_value
                    options_totals[key] = options_totals.get(key, 0.0) + numeric_value
                if normalized_options:
                    ticker_breakdown[ticker]['options_metrics'] = normalized_options

        win_rate = float(round((total_wins / total_trades) * 100, 2)) if total_trades else 0.0

        diversification = {
            ticker: float(round((data['total_trades'] / total_trades) * 100, 2)) if total_trades else 0.0
            for ticker, data in ticker_breakdown.items()
        }

        return {
            'report_type': 'portfolio_performance',
            'generated_at': datetime.now().isoformat(),
            'ticker_count': len(tickers),
            'performance_metrics': {
                'total_trades': total_trades,
                'winning_trades': total_wins,
                'losing_trades': total_losses,
                'win_rate_pct': win_rate,
                'total_profit_currency': float(round(total_profit_ccy, 4)),
                'total_profit_pct': float(round(total_profit_pct, 4)),
                'max_drawdown_pct': float(round(total_drawdown, 4))
            },
            'diversification_analysis': diversification,
            'risk_adjusted_returns': {
                ticker: {
                    'total_return_pct': float(round(data['total_profit_pct'], 4)),
                    'max_drawdown_pct': float(round(data['max_drawdown_pct'], 4))
                }
                for ticker, data in ticker_breakdown.items()
            },
            'options_summary': {
                key: float(round(value, 4)) for key, value in options_totals.items()
            } if options_totals else {}
        }

    def _create_signal_analysis_report(self, tickers: List[str],
                                        ticker_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create signal analysis report."""
        total_signal_counts: Dict[str, int] = {}
        signal_frequency: Dict[str, float] = {}

        for ticker in tickers:
            analysis = ticker_analysis.get(ticker, {})
            signal_info = analysis.get('signal_analysis', {})
            counts = signal_info.get('signal_counts', {})
            frequency = signal_info.get('signal_frequency', {})

            for signal, count in counts.items():
                total_signal_counts[signal] = total_signal_counts.get(signal, 0) + int(count)
            for signal, freq in frequency.items():
                signal_frequency.setdefault(signal, []).append(float(freq))

        averaged_frequency = {
            signal: float(round(np.mean(freqs), 4)) for signal, freqs in signal_frequency.items() if freqs
        }

        return {
            'report_type': 'signal_analysis',
            'generated_at': datetime.now().isoformat(),
            'signal_generation_summary': total_signal_counts,
            'signal_quality_metrics': averaged_frequency,
            'conversion_analysis': {}
        }

    def _generate_recommendations_report(self, tickers: List[str],
                                          ticker_metrics: Dict[str, Dict[str, Any]],
                                          ticker_risk: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate actionable recommendations."""
        strategy_recommendations: List[str] = []
        risk_recommendations: List[str] = []
        system_recommendations: List[str] = []

        # Identify top/bottom performers by profit
        profit_map = {
            ticker: float(metrics.get('pnl_metrics', {}).get('total_profit_currency', 0.0))
            for ticker, metrics in ticker_metrics.items()
        }
        if profit_map:
            best_ticker = max(profit_map, key=profit_map.get)
            worst_ticker = min(profit_map, key=profit_map.get)
            strategy_recommendations.append(
                f"Focus on scaling {best_ticker} which generated the highest absolute P&L ({profit_map[best_ticker]:.2f})."
            )
            strategy_recommendations.append(
                f"Review parameters for {worst_ticker}; it delivered the lowest P&L ({profit_map[worst_ticker]:.2f})."
            )

        # Risk recommendations based on approval rates
        for ticker in ticker_metrics.keys():
            risk_info = ticker_risk.get(ticker, {}).get('risk_management', {})
            approval_rate = risk_info.get('approval_rate_pct')
            if approval_rate is None:
                approval_rate = ticker_metrics[ticker].get('trade_metrics', {}).get('win_rate_pct')
            if approval_rate is None:
                continue
            if approval_rate < 10:
                risk_recommendations.append(f"{ticker}: Approval rate below 10% - consider relaxing risk limits or reviewing signals.")
            elif approval_rate > 90:
                risk_recommendations.append(f"{ticker}: Approval rate above 90% - consider tightening risk parameters to challenge trade quality.")

        system_recommendations.append("Ensure reporting pipeline regression tests cover metrics, risk, and manifests to prevent placeholder drift.")

        return {
            'report_type': 'recommendations',
            'generated_at': datetime.now().isoformat(),
            'strategy_recommendations': strategy_recommendations,
            'risk_recommendations': risk_recommendations,
            'system_recommendations': system_recommendations,
            'priority_actions': strategy_recommendations[:1] + risk_recommendations[:1]
        }

    def _create_output_manifest(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive manifest of all outputs."""
        three_file_outputs = processing_results.get('three_file_outputs', {})
        analytics_outputs = processing_results.get('analytics', {})
        reports_outputs = processing_results.get('reports', {})

        components = {
            'three_file_outputs': {
                ticker: {
                    name: str(path) for name, path in outputs.items()
                    if isinstance(path, (Path, str))
                }
                for ticker, outputs in three_file_outputs.items()
            },
            'ticker_reports': {
                ticker: {
                    key: str(value) for key, value in report.items()
                    if key.endswith('_file') and isinstance(value, (Path, str))
                }
                for ticker, report in processing_results.get('ticker_reports', {}).items()
                if isinstance(report, dict)
            },
            'portfolio_reports': {
                name: str(path) for name, path in reports_outputs.items()
                if isinstance(path, (Path, str))
            },
            'analytics': {
                ticker: {
                    name: str(path) for name, path in analytics.items()
                    if isinstance(path, (Path, str))
                }
                for ticker, analytics in analytics_outputs.items()
                if isinstance(analytics, dict)
            }
        }

        manifest = {
            'manifest_version': '1.0',
            'created_at': datetime.now().isoformat(),
            'strategy_run_dir': str(processing_results.get('strategy_run_dir', '')),
            'component_counts': {
                'tickers': len(three_file_outputs),
                'portfolio_reports': len(reports_outputs),
                'analytics_items': sum(len(v) for v in components['analytics'].values())
            },
            'components': components,
            'portfolio_summary': processing_results.get('summary', {}),
            'visualization_hashes': dict(sorted((processing_results.get('visualization_hashes', {}) or {}).items())),
            'file_inventory': self._create_file_inventory(processing_results),
            'usage_guide': {
                'start_here': 'Review reports/portfolio_performance_*.json for aggregate view',
                'ticker_drilldown': 'Open tickers/<TICKER>/metrics.json for per-ticker insights',
                'risk_focus': 'Combine tickers/<TICKER>/risk_report.json with reports/risk_management_report_*.json',
                'three_file_system': 'CSV files under data/ provide raw parity across pipeline stages'
            }
        }
        return manifest
    
    def _create_file_inventory(self, processing_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create inventory of all generated files."""
        csv_files: Set[str] = set()
        json_files: Set[str] = set()
        visualization_files: Set[str] = set()

        for ticker_outputs in processing_results.get('three_file_outputs', {}).values():
            if isinstance(ticker_outputs, dict):
                for path in ticker_outputs.values():
                    if isinstance(path, (Path, str)):
                        path_str = str(path)
                        if path_str.endswith('.csv'):
                            csv_files.add(path_str)
                        elif path_str.endswith('.json'):
                            json_files.add(path_str)

        for ticker_report in processing_results.get('ticker_reports', {}).values():
            if isinstance(ticker_report, dict):
                for key, path in ticker_report.items():
                    if key.endswith('_file') and isinstance(path, (Path, str)):
                        path_str = str(path)
                        if path_str.endswith('.json'):
                            json_files.add(path_str)

        for analytics in processing_results.get('analytics', {}).values():
            if isinstance(analytics, dict):
                for path in analytics.values():
                    if isinstance(path, (Path, str)):
                        path_str = str(path)
                        if path_str.endswith('.json'):
                            json_files.add(path_str)

        for report_path in processing_results.get('reports', {}).values():
            if isinstance(report_path, (Path, str)):
                path_str = str(report_path)
                if path_str.endswith('.json'):
                    json_files.add(path_str)

        visualizations = processing_results.get('visualizations', {})
        if isinstance(visualizations, dict):
            for value in visualizations.values():
                if isinstance(value, dict):
                    for path in value.values():
                        if isinstance(path, (Path, str)):
                            path_str = str(path)
                            if path_str.endswith(('.png', '.svg')):
                                visualization_files.add(path_str)
                elif isinstance(value, (Path, str)):
                    path_str = str(value)
                    if path_str.endswith(('.png', '.svg')):
                        visualization_files.add(path_str)

        visualization_hashes = dict(sorted((processing_results.get('visualization_hashes', {}) or {}).items()))
        return {
            'csv_files': sorted(csv_files),
            'json_files': sorted(json_files),
            'visualization_files': sorted(visualization_files),
            'visualization_hashes': visualization_hashes
        }
