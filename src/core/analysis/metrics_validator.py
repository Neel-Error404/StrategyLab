# src/etl/analysis/metrics_validator.py
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Any, Optional, Tuple
import os

from src.strat_stats.statistics import calculate_metrics

class MetricsValidator:
    """
    Validates metrics calculation against known test cases and provides
    a comprehensive report of all calculated metrics.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the metrics validator.
        
        Args:
            output_dir: Directory to save validation results (optional)
        """
        self.logger = logging.getLogger("MetricsValidator")
        self.output_dir = output_dir
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_metrics(self, strategy_name: str, pull_date: str, 
                         strat_output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Validates metrics calculation against summary file and provides
        a comprehensive report.
        
        Args:
            strategy_name: Name of the strategy
            pull_date: Date of the backtest
            strat_output_dir: Directory containing strategy outputs (defaults to "Backtester/Strat_out/{strategy_name}/{pull_date}")
            
        Returns:
            Dictionary with validation results
        """
        if not strat_output_dir:
            strat_output_dir = Path(f"Backtester/Strat_out/{strategy_name}/{pull_date}")
        
        if not strat_output_dir.exists():
            self.logger.error(f"Strategy output directory not found: {strat_output_dir}")
            return {'error': f"Directory not found: {strat_output_dir}"}
        
        # Load summary file
        summary_file = strat_output_dir / f"{pull_date}_Summary.csv"
        if not summary_file.exists():
            self.logger.error(f"Summary file not found: {summary_file}")
            return {'error': f"Summary file not found: {summary_file}"}
        
        summary_df = pd.read_csv(summary_file)
        
        # Get list of all ticker trade files
        trade_files = list(strat_output_dir.glob("*_Trades_*.csv"))
        
        if not trade_files:
            self.logger.error(f"No trade files found in {strat_output_dir}")
            return {'error': f"No trade files found in {strat_output_dir}"}
        
        # Recompute metrics for a sample of tickers to validate
        validation_results = []
        ticker_metrics = {}
        
        for trade_file in trade_files:
            ticker = trade_file.name.split("_")[0]
            self.logger.info(f"Validating metrics for {ticker}...")
            
            # Load trades
            trades_df = pd.read_csv(trade_file)
            if trades_df.empty:
                self.logger.warning(f"No trades found for {ticker}")
                validation_results.append({
                    'ticker': ticker,
                    'valid': None,
                    'diffs': {},
                    'error': "No trades found"
                })
                continue
                
            # Convert to list of dicts for calculate_metrics
            trades = trades_df.to_dict('records')
            
            # Recalculate metrics
            recalculated = calculate_metrics(trades)
            ticker_metrics[ticker] = recalculated
            
            # Get original metrics from summary
            original = summary_df[summary_df['Ticker'] == ticker].iloc[0].to_dict() if not summary_df[summary_df['Ticker'] == ticker].empty else {}
            
            if not original:
                self.logger.warning(f"No summary data found for {ticker}")
                validation_results.append({
                    'ticker': ticker,
                    'valid': False,
                    'diffs': {},
                    'error': "No summary data found"
                })
                continue
            
            # Compare key metrics
            diffs = {}
            validation_keys = [
                'Total Trades', 'Wins', 'Losses', 'Win/Loss Ratio', 
                'Accuracy (%)', 'Average Profit (%)', 'Average Profit (Currency)',
                'Max Drawdown (%)', 'Average Drawdown (%)', 'Average Trade Duration (min)',
                'Average RRR', 'Sharpe Ratio'
            ]
            
            for key in validation_keys:
                if key in original and key in recalculated:
                    orig_val = original[key]
                    recalc_val = recalculated[key]
                    
                    # Handle non-numeric values
                    if not isinstance(orig_val, (int, float)) or not isinstance(recalc_val, (int, float)):
                        diffs[key] = {
                            'original': orig_val,
                            'recalculated': recalc_val,
                            'diff': None,
                            'pct_diff': None,
                            'match': orig_val == recalc_val
                        }
                    else:
                        # Calculate absolute and percentage differences
                        abs_diff = orig_val - recalc_val
                        pct_diff = (abs_diff / orig_val * 100) if orig_val != 0 else float('inf')
                        
                        # Determine if values match (within tolerance)
                        # Use tighter tolerance for integer metrics
                        tolerance = 0.001 if isinstance(orig_val, int) else 0.01
                        match = abs(abs_diff) < tolerance
                        
                        diffs[key] = {
                            'original': orig_val,
                            'recalculated': recalc_val,
                            'diff': abs_diff,
                            'pct_diff': pct_diff,
                            'match': match
                        }
            
            # Determine overall validity
            valid = all(d['match'] for d in diffs.values() if d['match'] is not None)
            
            validation_results.append({
                'ticker': ticker,
                'valid': valid,
                'diffs': diffs
            })
        
        # Create overall validation report
        total_tickers = len(validation_results)
        valid_tickers = sum(1 for r in validation_results if r.get('valid') is True)
        invalid_tickers = sum(1 for r in validation_results if r.get('valid') is False)
        error_tickers = sum(1 for r in validation_results if r.get('valid') is None)
        
        validation_report = {
            'strategy_name': strategy_name,
            'pull_date': pull_date,
            'total_tickers': total_tickers,
            'valid_tickers': valid_tickers,
            'invalid_tickers': invalid_tickers,
            'error_tickers': error_tickers,
            'validation_rate': (valid_tickers / total_tickers) * 100 if total_tickers > 0 else 0,
            'ticker_results': validation_results
        }
        
        # Save validation report if output directory is provided
        if self.output_dir:
            report_file = self.output_dir / f"{strategy_name}_{pull_date}_validation.json"
            with open(report_file, 'w') as f:
                json.dump(validation_report, f, indent=2)
            self.logger.info(f"Validation report saved to {report_file}")
        
        return validation_report
    
    def print_validation_report(self, report: Dict[str, Any]) -> None:
        """
        Print a formatted validation report to the console.
        
        Args:
            report: Validation report from validate_metrics
        """
        if 'error' in report:
            print(f"Validation Error: {report['error']}")
            return
        
        print("\n" + "="*80)
        print(f"METRICS VALIDATION REPORT: {report['strategy_name']} - {report['pull_date']}")
        print("="*80)
        
        print(f"\nSummary:")
        print(f"  Total Tickers: {report['total_tickers']}")
        print(f"  Valid Tickers: {report['valid_tickers']} ({report['validation_rate']:.2f}%)")
        print(f"  Invalid Tickers: {report['invalid_tickers']}")
        print(f"  Error Tickers: {report['error_tickers']}")
        
        print("\nDetailed Results:")
        for result in report['ticker_results']:
            ticker = result['ticker']
            valid = result['valid']
            
            status = "✅ Valid" if valid is True else "❌ Invalid" if valid is False else "⚠️ Error"
            print(f"\n{ticker}: {status}")
            
            if 'error' in result:
                print(f"  Error: {result['error']}")
                continue
            
            for key, values in result['diffs'].items():
                match_str = "✓" if values.get('match', False) else "✗"
                
                # Format the difference information
                if values.get('diff') is not None:
                    diff_str = f"Diff: {values['diff']:.4f} ({values['pct_diff']:.2f}%)"
                else:
                    diff_str = "Diff: N/A"
                
                print(f"  {key}: {match_str} {values['original']} (original) vs {values['recalculated']} (recalculated) - {diff_str}")
    
    def plot_metric_comparison(self, report: Dict[str, Any], metric: str, 
                               save_path: Optional[str] = None) -> None:
        """
        Plot a comparison of original vs recalculated values for a specific metric.
        
        Args:
            report: Validation report from validate_metrics
            metric: Name of the metric to compare
            save_path: Path to save the plot (optional)
        """
        if 'error' in report:
            self.logger.error(f"Cannot plot metric comparison: {report['error']}")
            return
        
        # Extract metric values from report
        tickers = []
        original_values = []
        recalculated_values = []
        
        for result in report['ticker_results']:
            if 'error' in result or 'diffs' not in result:
                continue
                
            if metric in result['diffs']:
                tickers.append(result['ticker'])
                original_values.append(result['diffs'][metric]['original'])
                recalculated_values.append(result['diffs'][metric]['recalculated'])
        
        if not tickers:
            self.logger.warning(f"No valid data found for metric: {metric}")
            return
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Ticker': tickers,
            'Original': original_values,
            'Recalculated': recalculated_values
        })
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Bar plot for original and recalculated values
        x = np.arange(len(tickers))
        width = 0.35
        
        plt.bar(x - width/2, df['Original'], width, label='Original')
        plt.bar(x + width/2, df['Recalculated'], width, label='Recalculated')
        
        plt.xlabel('Ticker')
        plt.ylabel(metric)
        plt.title(f'Original vs Recalculated {metric} by Ticker')
        plt.xticks(x, df['Ticker'], rotation=45)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()


def validate_all_strategies(base_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Validate metrics for all strategies and dates in the base directory.
    
    Args:
        base_dir: Base directory containing strategy output folders
        output_dir: Directory to save validation results (optional)
        
    Returns:
        Dictionary mapping strategy_date to validation reports
    """
    validator = MetricsValidator(output_dir)
    all_reports = {}
    
    # Find all strategy directories
    strategy_dirs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for strategy_dir in strategy_dirs:
        strategy_name = strategy_dir.name
        
        # Find all date directories
        date_dirs = [d for d in strategy_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for date_dir in date_dirs:
            pull_date = date_dir.name
            
            # Skip if not a valid date directory
            if not (date_dir / f"{pull_date}_Summary.csv").exists():
                continue
                
            # Validate metrics
            report = validator.validate_metrics(strategy_name, pull_date, date_dir)
            all_reports[f"{strategy_name}_{pull_date}"] = report
            
            # Print report
            validator.print_validation_report(report)
    
    return all_reports


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate strategy metrics calculations")
    parser.add_argument("--strategy", type=str, help="Strategy name")
    parser.add_argument("--date", type=str, help="Pull date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, help="Output directory for validation results")
    parser.add_argument("--base-dir", type=str, default="Backtester/Strat_out", help="Base directory containing strategy outputs")
    parser.add_argument("--validate-all", action="store_true", help="Validate all strategies and dates")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output) if args.output else None
    
    if args.validate_all:
        validate_all_strategies(base_dir, output_dir)
    elif args.strategy and args.date:
        validator = MetricsValidator(output_dir)
        report = validator.validate_metrics(args.strategy, args.date)
        validator.print_validation_report(report)
    else:
        print("Please provide --strategy and --date, or use --validate-all")