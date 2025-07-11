# src/core/output/three_file_system.py
"""
Three-File CSV Output System for Backtesting Framework.

This module implements the comprehensive three-file output structure:
1. Base File: Price data with signals and indicators (everything)
2. Real Trades File: All trades generated by strategy (irrespective of risk)
3. Risk-Allowed Trades File: Only trades that pass risk management

This enables full transparency in strategy analysis and risk management effectiveness.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

class ThreeFileOutputSystem:
    """
    Manages the three-file CSV output system for comprehensive trade analysis.
    """
    
    def __init__(self, strategy_run_dir: Path):
        """
        Initialize the three-file output system.
        
        Args:
            strategy_run_dir: Path to strategy run directory
        """
        self.strategy_run_dir = strategy_run_dir
        self.logger = logging.getLogger("ThreeFileOutputSystem")
          # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories for the three-file system."""        # Use nested data structure to match naming utility
        directories = [
            'data/base_data',
            'data/strategy_trades',  # All trades generated by strategy
            'data/risk_approved_trades',  # Only trades that passed risk management
            'analysis_reports',
            'analysis_reports/individual',
            'analysis_reports/portfolio',
            'reports'
        ]
        
        for directory in directories:
            (self.strategy_run_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def save_base_file(self, ticker: str, date_range: str, base_data: pd.DataFrame) -> Path:
        """
        Save the base file with price data, signals, and indicators.
        
        Args:
            ticker: Ticker symbol
            date_range: Date range string
            base_data: DataFrame with price data, signals, and indicators
            
        Returns:
            Path to saved base file
        """
        file_path = self.strategy_run_dir / 'data' / 'base_data' / f"{ticker}_Base_{date_range}.csv"
        
        # Ensure all required columns are present
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ticker']
        
        # Add missing columns with default values if necessary
        for col in required_columns:
            if col not in base_data.columns:
                if col == 'ticker':
                    base_data[col] = ticker
                else:
                    base_data[col] = 0
        
        # Save with comprehensive metadata
        metadata = {
            'file_type': 'base_data',
            'ticker': ticker,
            'date_range': date_range,
            'total_rows': len(base_data),
            'columns': list(base_data.columns),
            'created_at': datetime.now().isoformat(),
            'data_summary': {
                'date_start': str(base_data['timestamp'].min()) if 'timestamp' in base_data.columns else 'N/A',
                'date_end': str(base_data['timestamp'].max()) if 'timestamp' in base_data.columns else 'N/A',
                'signal_counts': self._count_signals(base_data)
            }
        }
        
        # Save data
        base_data.to_csv(file_path, index=False)
        
        # Save metadata
        metadata_path = file_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Saved base file for {ticker}: {file_path}")
        return file_path
    
    def save_strategy_trades_file(self, ticker: str, date_range: str, 
                                  all_trades: List[Dict], 
                                  strategy_metadata: Dict[str, Any]) -> Path:
        """
        Save all trades generated by strategy (before risk filtering).
        
        Args:
            ticker: Ticker symbol
            date_range: Date range string
            all_trades: List of all trades generated by strategy
            strategy_metadata: Metadata about strategy execution
            
        Returns:
            Path to saved strategy trades file
        """
        file_path = self.strategy_run_dir / 'data' / 'strategy_trades' / f"{ticker}_StrategyTrades_{date_range}.csv"
        
        # Prepare trades DataFrame
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
        else:
            trades_df = self._create_empty_trades_df()
        
        # Add strategy metadata columns
        trades_df['ticker'] = ticker
        trades_df['strategy_generated'] = True
        trades_df['risk_processed'] = False  # These are pre-risk processing
        
        # Save data
        trades_df.to_csv(file_path, index=False)
        
        # Create comprehensive metadata
        metadata = {
            'file_type': 'strategy_trades',
            'ticker': ticker,
            'date_range': date_range,
            'total_trades': len(all_trades),
            'strategy_metadata': strategy_metadata,
            'created_at': datetime.now().isoformat(),
            'trade_summary': self._analyze_trades(all_trades) if all_trades else {}
        }
        
        # Save metadata
        metadata_path = file_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Saved strategy trades for {ticker}: {len(all_trades)} trades")
        return file_path
    
    def save_risk_approved_trades_file(self, ticker: str, date_range: str,
                                       approved_trades: List[Dict],
                                       risk_analysis: Dict[str, Any]) -> Path:
        """
        Save trades that passed risk management.
        
        Args:
            ticker: Ticker symbol
            date_range: Date range string
            approved_trades: List of trades that passed risk management
            risk_analysis: Risk management analysis results
            
        Returns:
            Path to saved risk-approved trades file
        """
        file_path = self.strategy_run_dir / 'data' / 'risk_approved_trades' / f"{ticker}_RiskApprovedTrades_{date_range}.csv"
        
        # Prepare trades DataFrame
        if approved_trades:
            trades_df = pd.DataFrame(approved_trades)
        else:
            trades_df = self._create_empty_trades_df()
        
        # Add risk management metadata columns
        trades_df['ticker'] = ticker
        trades_df['strategy_generated'] = True
        trades_df['risk_processed'] = True
        trades_df['risk_approved'] = True
        
        # Save data
        trades_df.to_csv(file_path, index=False)
        
        # Create comprehensive metadata
        metadata = {
            'file_type': 'risk_approved_trades',
            'ticker': ticker,
            'date_range': date_range,
            'approved_trades': len(approved_trades),
            'risk_analysis': risk_analysis,
            'created_at': datetime.now().isoformat(),
            'trade_summary': self._analyze_trades(approved_trades) if approved_trades else {},
            'risk_efficiency': {
                'approval_rate': risk_analysis.get('approval_rate', 0),
                'rejection_rate': risk_analysis.get('rejection_rate', 0),
                'most_common_rejection': risk_analysis.get('most_common_rejection', 'N/A')
            }
        }
        
        # Save metadata
        metadata_path = file_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Saved risk-approved trades for {ticker}: {len(approved_trades)} trades")
        return file_path
    
    def create_comprehensive_analysis(self, ticker: str, date_range: str) -> Dict[str, Path]:
        """
        Create comprehensive analysis comparing all three files.
        
        Args:
            ticker: Ticker symbol
            date_range: Date range string
            
        Returns:
            Dictionary of created analysis files
        """
        analysis_files = {}        
        # Load all three files
        base_file = self.strategy_run_dir / 'data' / 'base_data' / f"{ticker}_Base_{date_range}.csv"
        strategy_trades_file = self.strategy_run_dir / 'data' / 'strategy_trades' / f"{ticker}_StrategyTrades_{date_range}.csv"
        risk_trades_file = self.strategy_run_dir / 'data' / 'risk_approved_trades' / f"{ticker}_RiskApprovedTrades_{date_range}.csv"
        
        try:
            # Load data
            base_data = pd.read_csv(base_file) if base_file.exists() else pd.DataFrame()
            strategy_trades = pd.read_csv(strategy_trades_file) if strategy_trades_file.exists() else pd.DataFrame()
            risk_trades = pd.read_csv(risk_trades_file) if risk_trades_file.exists() else pd.DataFrame()
            
            # Create comprehensive analysis
            analysis = {
                'ticker': ticker,
                'date_range': date_range,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_summary': {
                    'base_data_points': len(base_data),
                    'strategy_trades_generated': len(strategy_trades),
                    'risk_approved_trades': len(risk_trades),
                    'risk_rejection_count': len(strategy_trades) - len(risk_trades),
                    'risk_approval_rate': len(risk_trades) / len(strategy_trades) if len(strategy_trades) > 0 else 0
                },
                'signal_analysis': self._analyze_signals(base_data),
                'strategy_performance': self._analyze_strategy_trades(strategy_trades),
                'risk_impact': self._analyze_risk_impact(strategy_trades, risk_trades),
                'recommendations': self._generate_recommendations(base_data, strategy_trades, risk_trades)
            }            
            # Save comprehensive analysis
            analysis_file = self.strategy_run_dir / 'analysis_reports' / 'individual' / f"{ticker}_Analysis_{date_range}.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            analysis_files['comprehensive_analysis'] = analysis_file
            
            # Create summary report
            summary_file = self._create_summary_report(ticker, date_range, analysis)
            analysis_files['summary_report'] = summary_file
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive analysis for {ticker}: {e}")
        
        return analysis_files
    
    def _create_empty_trades_df(self) -> pd.DataFrame:
        """Create empty trades DataFrame with proper structure."""
        columns = [
            "Trade Type", "Entry Time", "Exit Time", "Entry Price", "Exit Price",
            "Profit (Currency)", "Profit (%)", "High During Trade", "Low During Trade",
            "High Time", "Low Time", "Trade Duration (min)", "Target (%)",
            "Drawdown (%)", "RRR", "Recovery Time (min)"
        ]
        return pd.DataFrame(columns=columns)
    
    def _count_signals(self, base_data: pd.DataFrame) -> Dict[str, int]:
        """Count different types of signals in base data."""
        signal_counts = {}
        
        # Common signal columns
        signal_columns = [
            'entry_signal_buy', 'entry_signal_sell',
            'exit_signal_buy', 'exit_signal_sell',
            'long_entry', 'long_exit', 'short_entry', 'short_exit'        ]
        
        for col in signal_columns:
            if col in base_data.columns:
                signal_counts[col] = int(base_data[col].sum()) if pd.api.types.is_numeric_dtype(base_data[col]) else 0
        
        return signal_counts
    
    def _analyze_trades(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze basic trade statistics."""
        if not trades:
            return {'total_trades': 0}
        
        df = pd.DataFrame(trades)
        
        analysis = {
            'total_trades': len(trades),
            'profitable_trades': len(df[df.get('Profit (%)', 0) > 0]) if 'Profit (%)' in df.columns else 0,
            'losing_trades': len(df[df.get('Profit (%)', 0) <= 0]) if 'Profit (%)' in df.columns else 0,
        }
        
        if 'Profit (%)' in df.columns:
            analysis.update({
                'avg_profit_pct': float(df['Profit (%)'].mean()),
                'max_profit_pct': float(df['Profit (%)'].max()),
                'min_profit_pct': float(df['Profit (%)'].min()),
                'win_rate': analysis['profitable_trades'] / len(trades) if len(trades) > 0 else 0
            })
        
        return analysis
    
    def _analyze_signals(self, base_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze signal generation patterns."""
        if base_data.empty:
            return {'total_data_points': 0, 'signal_counts': {}, 'signal_frequency': {}}
        
        return {
            'total_data_points': len(base_data),
            'signal_counts': self._count_signals(base_data),
            'signal_frequency': {
                col: round(base_data[col].sum() / len(base_data), 4)
                for col in base_data.columns
                if 'signal' in col.lower() and pd.api.types.is_numeric_dtype(base_data[col]) and len(base_data) > 0
            }
        }
    
    def _analyze_strategy_trades(self, strategy_trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze strategy-generated trades."""
        if strategy_trades.empty:
            return {'total_strategy_trades': 0}
        
        return {
            'total_strategy_trades': len(strategy_trades),
            'trade_types': strategy_trades.get('Trade Type', pd.Series()).value_counts().to_dict() if 'Trade Type' in strategy_trades.columns else {},
            'performance_metrics': self._analyze_trades(strategy_trades.to_dict('records'))
        }
    
    def _analyze_risk_impact(self, strategy_trades: pd.DataFrame, risk_trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the impact of risk management."""
        strategy_count = len(strategy_trades)
        risk_count = len(risk_trades)
        
        return {
            'trades_generated_by_strategy': strategy_count,
            'trades_approved_by_risk': risk_count,
            'trades_rejected_by_risk': strategy_count - risk_count,
            'approval_rate': risk_count / strategy_count if strategy_count > 0 else 0,
            'rejection_rate': (strategy_count - risk_count) / strategy_count if strategy_count > 0 else 0,
            'risk_efficiency': 'High' if risk_count / strategy_count > 0.7 else 'Medium' if risk_count / strategy_count > 0.3 else 'Low'
        }
    
    def _generate_recommendations(self, base_data: pd.DataFrame, 
                                  strategy_trades: pd.DataFrame, 
                                  risk_trades: pd.DataFrame) -> List[str]:
        """Generate recommendations based on three-file analysis."""
        recommendations = []
        
        # Signal analysis recommendations
        if not base_data.empty:
            signal_cols = [col for col in base_data.columns if 'signal' in col.lower()]
            if len(signal_cols) == 0:
                recommendations.append("Consider adding signal columns to base data for better analysis")
        
        # Strategy performance recommendations
        strategy_count = len(strategy_trades)
        risk_count = len(risk_trades)
        
        if strategy_count == 0:
            recommendations.append("No trades generated by strategy - review signal generation logic")
        elif risk_count == 0:
            recommendations.append("All trades rejected by risk management - review risk parameters")
        elif risk_count / strategy_count < 0.1:
            recommendations.append("Very low trade approval rate - consider relaxing risk parameters")
        elif risk_count / strategy_count > 0.9:
            recommendations.append("Very high trade approval rate - consider tightening risk parameters")
        
        return recommendations
    
    def _create_summary_report(self, ticker: str, date_range: str, analysis: Dict) -> Path:
        """Create human-readable summary report."""
        summary_file = self.strategy_run_dir / 'reports' / f"{ticker}_ThreeFileReport_{date_range}.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"Three-File System Analysis Report\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Ticker: {ticker}\n")
            f.write(f"Date Range: {date_range}\n")
            f.write(f"Analysis Time: {analysis['analysis_timestamp']}\n\n")
            
            # Data Summary
            data_summary = analysis['data_summary']
            f.write(f"Data Summary:\n")
            f.write(f"- Base data points: {data_summary['base_data_points']:,}\n")
            f.write(f"- Strategy trades generated: {data_summary['strategy_trades_generated']}\n")
            f.write(f"- Risk approved trades: {data_summary['risk_approved_trades']}\n")
            f.write(f"- Risk rejection count: {data_summary['risk_rejection_count']}\n")
            f.write(f"- Risk approval rate: {data_summary['risk_approval_rate']:.1%}\n\n")
            
            # Recommendations
            if analysis.get('recommendations'):
                f.write(f"Recommendations:\n")
                for i, rec in enumerate(analysis['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
        
        return summary_file

    def create_portfolio_three_file_analysis(self, date_range: str, tickers: List[str]) -> Dict[str, Any]:
        """
        Create portfolio-level analysis across all tickers using three-file system.
        
        Args:
            date_range: Date range string
            tickers: List of tickers to analyze
            
        Returns:
            Portfolio analysis results
        """
        portfolio_analysis = {
            'analysis_type': 'portfolio_three_file_analysis',
            'date_range': date_range,
            'tickers': tickers,
            'analysis_timestamp': datetime.now().isoformat(),
            'portfolio_summary': {},
            'ticker_breakdowns': {},
            'cross_ticker_analysis': {}
        }
        
        # Aggregate data across all tickers
        total_base_points = 0
        total_strategy_trades = 0
        total_risk_approved = 0
        
        for ticker in tickers:
            try:                # Load ticker analysis
                analysis_file = self.strategy_run_dir / 'analysis_reports' / 'individual' / f"{ticker}_Analysis_{date_range}.json"
                if analysis_file.exists():
                    with open(analysis_file, 'r') as f:
                        ticker_analysis = json.load(f)
                    
                    portfolio_analysis['ticker_breakdowns'][ticker] = ticker_analysis
                    
                    # Aggregate numbers
                    data_summary = ticker_analysis.get('data_summary', {})
                    total_base_points += data_summary.get('base_data_points', 0)
                    total_strategy_trades += data_summary.get('strategy_trades_generated', 0)
                    total_risk_approved += data_summary.get('risk_approved_trades', 0)
                    
            except Exception as e:
                self.logger.error(f"Error loading analysis for {ticker}: {e}")
        
        # Portfolio summary
        portfolio_analysis['portfolio_summary'] = {
            'total_tickers': len(tickers),
            'total_base_data_points': total_base_points,
            'total_strategy_trades': total_strategy_trades,
            'total_risk_approved_trades': total_risk_approved,
            'portfolio_approval_rate': total_risk_approved / total_strategy_trades if total_strategy_trades > 0 else 0,
            'avg_trades_per_ticker': total_strategy_trades / len(tickers) if tickers else 0
        }
        
        # Cross-ticker analysis
        portfolio_analysis['cross_ticker_analysis'] = {
            'most_active_ticker': max(portfolio_analysis['ticker_breakdowns'].items(), 
                                    key=lambda x: x[1].get('data_summary', {}).get('strategy_trades_generated', 0))[0] if portfolio_analysis['ticker_breakdowns'] else 'N/A',
            'highest_approval_rate_ticker': max(portfolio_analysis['ticker_breakdowns'].items(),
                                              key=lambda x: x[1].get('data_summary', {}).get('risk_approval_rate', 0))[0] if portfolio_analysis['ticker_breakdowns'] else 'N/A'
        }
        
        # Save portfolio analysis
        portfolio_file = self.strategy_run_dir / 'analysis_reports' / 'portfolio' / f"Portfolio_Analysis_{date_range}.json"
        with open(portfolio_file, 'w') as f:
            json.dump(portfolio_analysis, f, indent=2, default=str)
        
        self.logger.info(f"Created portfolio three-file analysis for {len(tickers)} tickers")
        return portfolio_analysis
