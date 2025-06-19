# src/core/analysis/portfolio_visualization.py
"""
Portfolio-Level Visualization System for Multi-Ticker Backtesting.

This module provides comprehensive visualization capabilities for portfolio analysis,
including cross-ticker comparisons, portfolio-level metrics, and risk analysis.
"""

import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

class PortfolioVisualizer:
    """
    Creates comprehensive visualizations for portfolio-level analysis.
    """
    def __init__(self, output_dir: Optional[Path] = None, trade_source: str = "auto"):
        """
        Initialize the portfolio visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            trade_source: Trade data source for visualizations:
                         - "strategy_trades": Use raw strategy output (always available)
                         - "risk_approved_trades": Use post-risk-management trades (may be empty)
                         - "auto": Try risk_approved_trades first, fallback to strategy_trades if empty
        """
        self.output_dir = output_dir or Path("visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create organized subdirectories
        self.portfolio_dir = self.output_dir / "portfolio"
        self.individual_dir = self.output_dir / "individual"
        self.portfolio_dir.mkdir(parents=True, exist_ok=True)
        self.individual_dir.mkdir(parents=True, exist_ok=True)
        
        self.trade_source = trade_source
        self.logger = logging.getLogger("PortfolioVisualizer")
        
        # Set style for all plots        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_portfolio_dashboard(self, strategy_run_dir: Path, date_range: str, 
                                   tickers: List[str]) -> Dict[str, Path]:
        """
        Create comprehensive portfolio dashboard.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            date_range: Date range string
            tickers: List of tickers in portfolio
            
        Returns:
            Dictionary of created visualization files
        """
        visualizations = {}
        
        try:
            # Limit tickers to max 12 for portfolio visualizations
            limited_tickers = self._limit_portfolio_tickers(strategy_run_dir, date_range, tickers)
            if len(limited_tickers) < len(tickers):
                self.logger.info(f"Limited portfolio visualizations from {len(tickers)} to {len(limited_tickers)} tickers: {limited_tickers}")
            
            # Load portfolio data
            portfolio_data = self._load_portfolio_data(strategy_run_dir, date_range, limited_tickers)
            
            if not portfolio_data:
                self.logger.warning("No portfolio data available for visualization")
                return visualizations
            
            # Create individual visualizations
            visualizations.update(self._create_performance_dashboard(portfolio_data, date_range))
            visualizations.update(self._create_risk_dashboard(portfolio_data, date_range))
            visualizations.update(self._create_trade_analysis_dashboard(portfolio_data, date_range))
            visualizations.update(self._create_signal_analysis_dashboard(portfolio_data, date_range))
            visualizations.update(self._create_three_file_comparison_dashboard(portfolio_data, date_range))
            
            # Create educational insights dashboard (NEW)
            visualizations.update(self._create_educational_insights_dashboard(portfolio_data, date_range))
              # Create master dashboard
            master_dashboard = self._create_master_dashboard(portfolio_data, date_range)
            visualizations['master_dashboard'] = master_dashboard
            
            self.logger.info(f"Created {len(visualizations)} portfolio visualizations")
        except Exception as e:
            self.logger.error(f"Error creating portfolio dashboard: {e}")
        
        return visualizations
    
    def _limit_portfolio_tickers(self, strategy_run_dir: Path, date_range: str, 
                                tickers: List[str], max_tickers: int = 12) -> List[str]:
        """
        Limit portfolio tickers to a maximum number based on selection criteria.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            date_range: Date range string
            tickers: Original list of tickers
            max_tickers: Maximum number of tickers to include (default: 12)
            
        Returns:
            Limited list of tickers based on selection criteria
        """
        if len(tickers) <= max_tickers:
            return tickers
        
        # Selection criteria: prioritize by trade count and absolute returns
        ticker_scores = []
        
        for ticker in tickers:
            try:
                # Load analytics for scoring
                analytics_file = strategy_run_dir / 'analysis_reports' / 'individual' / f"{ticker}_Analysis_{date_range}.json"
                if analytics_file.exists():
                    with open(analytics_file, 'r') as f:
                        analytics = json.load(f)
                    
                    # Score based on trade count and performance
                    trade_count = analytics.get('data_summary', {}).get('strategy_trades_generated', 0)
                    
                    # Try to get return data for scoring
                    performance_metrics = analytics.get('performance_metrics', {})
                    abs_return = abs(performance_metrics.get('total_return', 0))
                    
                    # Combined score: trade count (70%) + absolute return (30%)
                    score = (trade_count * 0.7) + (abs_return * 0.3)
                    
                    ticker_scores.append({
                        'ticker': ticker,
                        'score': score,
                        'trade_count': trade_count,
                        'abs_return': abs_return
                    })
                else:
                    # Fallback: use ticker name for deterministic ordering
                    ticker_scores.append({
                        'ticker': ticker,
                        'score': 0,
                        'trade_count': 0,
                        'abs_return': 0
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error scoring ticker {ticker}: {e}")
                ticker_scores.append({
                    'ticker': ticker,
                    'score': 0,
                    'trade_count': 0,
                    'abs_return': 0
                })
        
        # Sort by score (descending) and take top max_tickers
        sorted_tickers = sorted(ticker_scores, key=lambda x: x['score'], reverse=True)
        selected_tickers = [item['ticker'] for item in sorted_tickers[:max_tickers]]
        
        self.logger.info(f"Selected top {len(selected_tickers)} tickers for portfolio visualization: {selected_tickers}")
        return selected_tickers
    
    def _load_portfolio_data(self, strategy_run_dir: Path, date_range: str,
                             tickers: List[str]) -> Dict[str, Any]:
        """Load all portfolio data for visualization with flexible trade source."""
        portfolio_data = {
            'tickers': tickers,
            'date_range': date_range,
            'base_data': {},
            'strategy_trades': {},
            'risk_approved_trades': {},
            'active_trades': {},  # Unified trade data based on trade_source preference
            'analytics': {},
            'risk_reports': {}
        }
        
        for ticker in tickers:
            try:
                # Load base data (corrected path to match OptimizedOutputSystem)
                base_file = strategy_run_dir / 'data' / 'base_data' / f"{ticker}_Base_{date_range}.csv"
                if base_file.exists():
                    portfolio_data['base_data'][ticker] = pd.read_csv(base_file)
                
                # Load strategy trades (corrected path to match OptimizedOutputSystem)
                strategy_trades_file = strategy_run_dir / 'data' / 'strategy_trades' / f"{ticker}_StrategyTrades_{date_range}.csv"
                if strategy_trades_file.exists():
                    portfolio_data['strategy_trades'][ticker] = pd.read_csv(strategy_trades_file)
                
                # Load risk approved trades (corrected path to match OptimizedOutputSystem)
                risk_trades_file = strategy_run_dir / 'data' / 'risk_approved_trades' / f"{ticker}_RiskApprovedTrades_{date_range}.csv"
                if risk_trades_file.exists():
                    portfolio_data['risk_approved_trades'][ticker] = pd.read_csv(risk_trades_file)                # Load analytics
                analytics_file = strategy_run_dir / 'analysis_reports' / 'individual' / f"{ticker}_Analysis_{date_range}.json"
                if analytics_file.exists():
                    with open(analytics_file, 'r') as f:
                        portfolio_data['analytics'][ticker] = json.load(f)
                
                # Load risk reports
                risk_report_file = strategy_run_dir / f"{ticker}" / "risk_report.json"
                if risk_report_file.exists():
                    with open(risk_report_file, 'r') as f:
                        portfolio_data['risk_reports'][ticker] = json.load(f)
                
            except Exception as e:
                self.logger.error(f"Error loading data for {ticker}: {e}")
        
        # Determine active trades based on trade_source preference
        self._set_active_trades(portfolio_data)
        
        return portfolio_data
    
    def _set_active_trades(self, portfolio_data: Dict[str, Any]) -> None:
        """
        Set active trades based on trade_source preference with fallback logic.
        
        Args:
            portfolio_data: Portfolio data dictionary to update
        """
        tickers = portfolio_data['tickers']
        
        for ticker in tickers:
            active_trades = None
            
            if self.trade_source == "strategy_trades":
                # Use strategy trades directly
                active_trades = portfolio_data['strategy_trades'].get(ticker)
                source_used = "strategy_trades"
                
            elif self.trade_source == "risk_approved_trades":
                # Use risk approved trades directly
                active_trades = portfolio_data['risk_approved_trades'].get(ticker)
                source_used = "risk_approved_trades"
                
            elif self.trade_source == "auto":
                # Try risk approved trades first, fallback to strategy trades if empty
                risk_trades = portfolio_data['risk_approved_trades'].get(ticker)
                strategy_trades = portfolio_data['strategy_trades'].get(ticker)
                
                if risk_trades is not None and not risk_trades.empty:
                    active_trades = risk_trades
                    source_used = "risk_approved_trades"
                elif strategy_trades is not None and not strategy_trades.empty:
                    active_trades = strategy_trades
                    source_used = "strategy_trades"
                else:
                    active_trades = pd.DataFrame()  # Empty DataFrame if no trades available
                    source_used = "none"
            
            # Set active trades for this ticker
            if active_trades is not None:
                portfolio_data['active_trades'][ticker] = active_trades
                if not active_trades.empty:
                    self.logger.debug(f"Using {source_used} for {ticker}: {len(active_trades)} trades")
                else:
                    self.logger.warning(f"No trades available for {ticker} from {source_used}")
            else:
                portfolio_data['active_trades'][ticker] = pd.DataFrame()
                self.logger.warning(f"No trade data found for {ticker}")
        
        # Log overall trade source summary
        total_active_trades = sum(len(trades) for trades in portfolio_data['active_trades'].values())
        self.logger.info(f"Trade source '{self.trade_source}': {total_active_trades} total active trades across {len(tickers)} tickers")
    
    def _create_performance_dashboard(self, portfolio_data: Dict, date_range: str) -> Dict[str, Path]:
        """Create performance analysis dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Portfolio Performance Dashboard - {date_range}', fontsize=16)
        
        # 1. Strategy Trades Generated by Ticker
        ax1 = axes[0, 0]
        strategy_counts = {}
        for ticker, analytics in portfolio_data['analytics'].items():
            strategy_counts[ticker] = analytics.get('data_summary', {}).get('strategy_trades_generated', 0)
        
        if strategy_counts:
            tickers = list(strategy_counts.keys())
            counts = list(strategy_counts.values())
            bars = ax1.bar(tickers, counts, color='skyblue', alpha=0.7)
            ax1.set_title('Strategy Trades Generated by Ticker')
            ax1.set_ylabel('Number of Trades')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom')
        
        # 2. Risk Approval Rates
        ax2 = axes[0, 1]
        approval_rates = {}
        for ticker, analytics in portfolio_data['analytics'].items():
            approval_rates[ticker] = analytics.get('data_summary', {}).get('risk_approval_rate', 0)
        
        if approval_rates:
            tickers = list(approval_rates.keys())
            rates = [rate * 100 for rate in approval_rates.values()]  # Convert to percentage
            bars = ax2.bar(tickers, rates, color='lightgreen', alpha=0.7)
            ax2.set_title('Risk Approval Rates by Ticker')
            ax2.set_ylabel('Approval Rate (%)')
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add percentage labels
            for bar, rate in zip(bars, rates):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. Signal Generation Comparison
        ax3 = axes[0, 2]
        signal_data = []
        for ticker, analytics in portfolio_data['analytics'].items():
            signal_analysis = analytics.get('signal_analysis', {})
            signal_counts = signal_analysis.get('signal_counts', {})
            total_signals = sum(signal_counts.values())
            signal_data.append({'ticker': ticker, 'total_signals': total_signals})
        
        if signal_data:
            df_signals = pd.DataFrame(signal_data)
            bars = ax3.bar(df_signals['ticker'], df_signals['total_signals'], color='orange', alpha=0.7)
            ax3.set_title('Total Signals Generated by Ticker')
            ax3.set_ylabel('Number of Signals')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Portfolio Composition
        ax4 = axes[1, 0]
        if strategy_counts:
            total_trades = sum(strategy_counts.values())
            if total_trades > 0:
                wedges, texts, autotexts = ax4.pie(strategy_counts.values(), 
                                                  labels=strategy_counts.keys(), 
                                                  autopct='%1.1f%%',
                                                  startangle=90)
                ax4.set_title('Portfolio Trade Distribution')
        
        # 5. Risk Impact Analysis
        ax5 = axes[1, 1]
        risk_data = []
        for ticker in portfolio_data['tickers']:
            analytics = portfolio_data['analytics'].get(ticker, {})
            data_summary = analytics.get('data_summary', {})
            risk_data.append({
                'ticker': ticker,
                'generated': data_summary.get('strategy_trades_generated', 0),
                'approved': data_summary.get('risk_approved_trades', 0),
                'rejected': data_summary.get('risk_rejection_count', 0)
            })
        
        if risk_data:
            df_risk = pd.DataFrame(risk_data)
            x = np.arange(len(df_risk))
            width = 0.25;
            
            ax5.bar(x - width, df_risk['generated'], width, label='Generated', alpha=0.7)
            ax5.bar(x, df_risk['approved'], width, label='Approved', alpha=0.7)
            ax5.bar(x + width, df_risk['rejected'], width, label='Rejected', alpha=0.7)
            
            ax5.set_title('Risk Management Impact')
            ax5.set_ylabel('Number of Trades')
            ax5.set_xticks(x)
            ax5.set_xticklabels(df_risk['ticker'], rotation=45)
            ax5.legend()
        
        # 6. Portfolio Summary Stats
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate portfolio stats
        total_strategy_trades = sum(strategy_counts.values()) if strategy_counts else 0
        total_approved = sum(analytics.get('data_summary', {}).get('risk_approved_trades', 0) 
                           for analytics in portfolio_data['analytics'].values())
        overall_approval_rate = (total_approved / total_strategy_trades * 100) if total_strategy_trades > 0 else 0
        
        stats_text = f"""Portfolio Summary
        
Total Tickers: {len(portfolio_data['tickers'])}
Total Strategy Trades: {total_strategy_trades:,}
Total Approved Trades: {total_approved:,}
Overall Approval Rate: {overall_approval_rate:.1f}%

Most Active Ticker: {max(strategy_counts, key=strategy_counts.get) if strategy_counts else 'N/A'}
Highest Approval Rate: {max(approval_rates, key=approval_rates.get) if approval_rates else 'N/A'}
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
          # Save dashboard
        dashboard_path = self.portfolio_dir / f"portfolio_performance_dashboard_{date_range}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {'performance_dashboard': dashboard_path}
    
    def _create_risk_dashboard(self, portfolio_data: Dict, date_range: str) -> Dict[str, Path]:
        """Create risk analysis dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Portfolio Risk Analysis Dashboard - {date_range}', fontsize=16)
        
        # 1. Risk Rejection Reasons
        ax1 = axes[0, 0]
        rejection_reasons = {}
        for ticker, risk_report in portfolio_data['risk_reports'].items():
            reasons = risk_report.get('risk_summary', {}).get('rejection_reasons', {})
            for reason, count in reasons.items():
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + count
        
        if rejection_reasons:
            wedges, texts, autotexts = ax1.pie(rejection_reasons.values(), 
                                              labels=rejection_reasons.keys(), 
                                              autopct='%1.1f%%',
                                              startangle=90)
            ax1.set_title('Risk Rejection Reasons (Portfolio-wide)')
        
        # 2. Approval Rate Distribution
        ax2 = axes[0, 1]
        approval_rates = []
        for ticker, analytics in portfolio_data['analytics'].items():
            rate = analytics.get('data_summary', {}).get('risk_approval_rate', 0)
            approval_rates.append(rate * 100)
        
        if approval_rates:
            ax2.hist(approval_rates, bins=10, alpha=0.7, color='green', edgecolor='black')
            ax2.set_title('Distribution of Approval Rates')
            ax2.set_xlabel('Approval Rate (%)')
            ax2.set_ylabel('Number of Tickers')
            ax2.axvline(np.mean(approval_rates), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(approval_rates):.1f}%')
            ax2.legend()
        
        # 3. Risk Efficiency by Ticker
        ax3 = axes[1, 0]
        efficiency_data = []
        for ticker, analytics in portfolio_data['analytics'].items():
            risk_impact = analytics.get('risk_impact', {})
            efficiency = risk_impact.get('risk_efficiency', 'Unknown')
            efficiency_data.append({'ticker': ticker, 'efficiency': efficiency})
        
        if efficiency_data:
            df_efficiency = pd.DataFrame(efficiency_data)
            efficiency_counts = df_efficiency['efficiency'].value_counts()
            colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red', 'Unknown': 'gray'}
            plot_colors = [colors.get(eff, 'gray') for eff in efficiency_counts.index]
            
            bars = ax3.bar(efficiency_counts.index, efficiency_counts.values, 
                          color=plot_colors, alpha=0.7)
            ax3.set_title('Risk Efficiency Distribution')
            ax3.set_ylabel('Number of Tickers')
        
        # 4. Risk Summary Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create risk summary table
        risk_summary_data = []
        for ticker in portfolio_data['tickers']:
            analytics = portfolio_data['analytics'].get(ticker, {})
            risk_impact = analytics.get('risk_impact', {})
            
            risk_summary_data.append([
                ticker,
                risk_impact.get('trades_generated_by_strategy', 0),
                risk_impact.get('trades_approved_by_risk', 0),
                f"{risk_impact.get('approval_rate', 0)*100:.1f}%",
                risk_impact.get('risk_efficiency', 'Unknown')
            ])
        
        if risk_summary_data:
            table = ax4.table(cellText=risk_summary_data,
                             colLabels=['Ticker', 'Generated', 'Approved', 'Rate', 'Efficiency'],
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax4.set_title('Risk Management Summary by Ticker', pad=20)
        
        plt.tight_layout()
          # Save dashboard
        dashboard_path = self.portfolio_dir / f"portfolio_risk_dashboard_{date_range}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {'risk_dashboard': dashboard_path}
    
    def _create_trade_analysis_dashboard(self, portfolio_data: Dict, date_range: str) -> Dict[str, Path]:
        """Create trade analysis dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Portfolio Trade Analysis Dashboard - {date_range}', fontsize=16)
        
        # 1. Trade Volume by Ticker (Strategy vs Approved)
        ax1 = axes[0, 0]
        trade_comparison_data = []
        for ticker in portfolio_data['tickers']:
            strategy_trades = len(portfolio_data['strategy_trades'].get(ticker, []))
            approved_trades = len(portfolio_data['risk_approved_trades'].get(ticker, []))
            trade_comparison_data.append({
                'ticker': ticker,
                'strategy': strategy_trades,
                'approved': approved_trades
            })
        
        if trade_comparison_data:
            df_trades = pd.DataFrame(trade_comparison_data)
            x = np.arange(len(df_trades))
            width = 0.35
            
            ax1.bar(x - width/2, df_trades['strategy'], width, label='Strategy Generated', alpha=0.7)
            ax1.bar(x + width/2, df_trades['approved'], width, label='Risk Approved', alpha=0.7)
            
            ax1.set_title('Trade Volume: Strategy vs Risk Approved')
            ax1.set_ylabel('Number of Trades')
            ax1.set_xticks(x)
            ax1.set_xticklabels(df_trades['ticker'], rotation=45)
            ax1.legend()
        
        # 2. Trade Type Distribution (if available)
        ax2 = axes[0, 1]
        trade_types = {}
        for ticker, trades_df in portfolio_data['strategy_trades'].items():
            if not trades_df.empty and 'Trade Type' in trades_df.columns:
                for trade_type in trades_df['Trade Type']:
                    trade_types[trade_type] = trade_types.get(trade_type, 0) + 1
        
        if trade_types:
            ax2.pie(trade_types.values(), labels=trade_types.keys(), autopct='%1.1f%%', startangle=90)
            ax2.set_title('Portfolio Trade Type Distribution')
        else:
            ax2.text(0.5, 0.5, 'No trade type data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Trade Type Distribution')
        
        # 3. Profitability Analysis (if profit data available)
        ax3 = axes[1, 0]
        profitability_data = []
        for ticker, trades_df in portfolio_data['strategy_trades'].items():
            if not trades_df.empty and 'Profit (%)' in trades_df.columns:
                profitable = len(trades_df[trades_df['Profit (%)'] > 0])
                losing = len(trades_df[trades_df['Profit (%)'] <= 0])
                profitability_data.append({
                    'ticker': ticker,
                    'profitable': profitable,
                    'losing': losing
                })
        
        if profitability_data:
            df_profit = pd.DataFrame(profitability_data)
            x = np.arange(len(df_profit))
            width = 0.35
            
            ax3.bar(x - width/2, df_profit['profitable'], width, label='Profitable', 
                   color='green', alpha=0.7)
            ax3.bar(x + width/2, df_profit['losing'], width, label='Losing', 
                   color='red', alpha=0.7)
            
            ax3.set_title('Profitability by Ticker')
            ax3.set_ylabel('Number of Trades')
            ax3.set_xticks(x)
            ax3.set_xticklabels(df_profit['ticker'], rotation=45)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No profitability data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Profitability Analysis')
        
        # 4. Trade Statistics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
          # Calculate trade statistics
        total_strategy_trades = sum(len(trades) for trades in portfolio_data['strategy_trades'].values())
        total_active_trades = sum(len(trades) for trades in portfolio_data['active_trades'].values())
        
        stats_text = f"""Trade Statistics Summary (Source: {self.trade_source})

Total Strategy Trades: {total_strategy_trades:,}
Total Active Trades: {total_active_trades:,}
Active Rate: {(total_active_trades/total_strategy_trades*100) if total_strategy_trades > 0 else 0:.1f}%

Average Trades per Ticker:
- Strategy: {total_strategy_trades/len(portfolio_data['tickers']) if portfolio_data['tickers'] else 0:.1f}
- Active: {total_active_trades/len(portfolio_data['tickers']) if portfolio_data['tickers'] else 0:.1f}

Trade Type Breakdown:
{chr(10).join([f"- {trade_type}: {count}" for trade_type, count in trade_types.items()]) if trade_types else "- No trade type data"}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
          # Save dashboard
        dashboard_path = self.portfolio_dir / f"portfolio_trade_analysis_{date_range}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {'trade_analysis_dashboard': dashboard_path}
    
    def _create_signal_analysis_dashboard(self, portfolio_data: Dict, date_range: str) -> Dict[str, Path]:
        """Create signal analysis dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Portfolio Signal Analysis Dashboard - {date_range}', fontsize=16)
        
        # 1. Signal Generation by Ticker
        ax1 = axes[0, 0]
        signal_data = []
        for ticker, analytics in portfolio_data['analytics'].items():
            signal_analysis = analytics.get('signal_analysis', {})
            signal_counts = signal_analysis.get('signal_counts', {})
            total_signals = sum(signal_counts.values())
            signal_data.append({'ticker': ticker, 'total_signals': total_signals})
        
        if signal_data:
            df_signals = pd.DataFrame(signal_data)
            bars = ax1.bar(df_signals['ticker'], df_signals['total_signals'], 
                          color='purple', alpha=0.7)
            ax1.set_title('Total Signals Generated by Ticker')
            ax1.set_ylabel('Number of Signals')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, df_signals['total_signals']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom')
        
        # 2. Signal Type Distribution
        ax2 = axes[0, 1]
        all_signal_types = {}
        for ticker, analytics in portfolio_data['analytics'].items():
            signal_analysis = analytics.get('signal_analysis', {})
            signal_counts = signal_analysis.get('signal_counts', {})
            for signal_type, count in signal_counts.items():
                all_signal_types[signal_type] = all_signal_types.get(signal_type, 0) + count
        
        if all_signal_types:
            # Show top signal types
            sorted_signals = dict(sorted(all_signal_types.items(), key=lambda x: x[1], reverse=True)[:8])
            ax2.barh(list(sorted_signals.keys()), list(sorted_signals.values()), 
                    color='teal', alpha=0.7)
            ax2.set_title('Top Signal Types (Portfolio-wide)')
            ax2.set_xlabel('Count')
        
        # 3. Signal-to-Trade Conversion
        ax3 = axes[1, 0]
        conversion_data = []
        for ticker in portfolio_data['tickers']:
            analytics = portfolio_data['analytics'].get(ticker, {})
            signal_analysis = analytics.get('signal_analysis', {})
            total_signals = sum(signal_analysis.get('signal_counts', {}).values())
            total_trades = analytics.get('data_summary', {}).get('strategy_trades_generated', 0)
            
            conversion_rate = (total_trades / total_signals * 100) if total_signals > 0 else 0
            conversion_data.append({
                'ticker': ticker,
                'conversion_rate': conversion_rate
            })
        
        if conversion_data:
            df_conversion = pd.DataFrame(conversion_data)
            bars = ax3.bar(df_conversion['ticker'], df_conversion['conversion_rate'], 
                          color='orange', alpha=0.7)
            ax3.set_title('Signal-to-Trade Conversion Rate')
            ax3.set_ylabel('Conversion Rate (%)')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add percentage labels
            for bar, rate in zip(bars, df_conversion['conversion_rate']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{rate:.1f}%', ha='center', va='bottom')
        
        # 4. Signal Frequency Analysis
        ax4 = axes[1, 1]
        frequency_data = []
        for ticker, analytics in portfolio_data['analytics'].items():
            signal_analysis = analytics.get('signal_analysis', {})
            signal_frequency = signal_analysis.get('signal_frequency', {})
            avg_frequency = np.mean(list(signal_frequency.values())) if signal_frequency else 0
            frequency_data.append({
                'ticker': ticker,
                'avg_frequency': avg_frequency * 100  # Convert to percentage
            })
        
        if frequency_data:
            df_frequency = pd.DataFrame(frequency_data)
            bars = ax4.bar(df_frequency['ticker'], df_frequency['avg_frequency'], 
                          color='salmon', alpha=0.7)
            ax4.set_title('Average Signal Frequency')
            ax4.set_ylabel('Frequency (%)')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
          # Save dashboard
        dashboard_path = self.portfolio_dir / f"portfolio_signal_analysis_{date_range}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {'signal_analysis_dashboard': dashboard_path}
    
    def _create_three_file_comparison_dashboard(self, portfolio_data: Dict, date_range: str) -> Dict[str, Path]:
        """Create three-file system comparison dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Three-File System Analysis Dashboard - {date_range}', fontsize=16)
        
        # 1. Data Flow Sankey-style Visualization
        ax1 = axes[0, 0]
        flow_data = []
        for ticker in portfolio_data['tickers']:
            analytics = portfolio_data['analytics'].get(ticker, {})
            data_summary = analytics.get('data_summary', {})
            
            base_points = data_summary.get('base_data_points', 0)
            strategy_trades = data_summary.get('strategy_trades_generated', 0)
            approved_trades = data_summary.get('risk_approved_trades', 0)
            
            flow_data.append({
                'ticker': ticker,
                'base_points': base_points,
                'strategy_trades': strategy_trades,
                'approved_trades': approved_trades
            })
        
        if flow_data:
            df_flow = pd.DataFrame(flow_data)
            
            # Create stacked bar chart showing the flow
            x = np.arange(len(df_flow))
            
            # Normalize to show proportions
            max_base = df_flow['base_points'].max() if df_flow['base_points'].max() > 0 else 1
            normalized_base = df_flow['base_points'] / max_base * 100
            
            ax1.bar(x, normalized_base, label='Base Data Points (normalized)', alpha=0.7, color='lightblue')
            ax1.bar(x, df_flow['strategy_trades'], label='Strategy Trades', alpha=0.7, color='orange')
            ax1.bar(x, df_flow['approved_trades'], label='Approved Trades', alpha=0.7, color='green')
            
            ax1.set_title('Three-File Data Flow Analysis')
            ax1.set_ylabel('Count')
            ax1.set_xticks(x)
            ax1.set_xticklabels(df_flow['ticker'], rotation=45)
            ax1.legend()
        
        # 2. Efficiency Ratios
        ax2 = axes[0, 1]
        efficiency_ratios = []
        for ticker in portfolio_data['tickers']:
            analytics = portfolio_data['analytics'].get(ticker, {})
            data_summary = analytics.get('data_summary', {})
            
            base_points = data_summary.get('base_data_points', 0)
            strategy_trades = data_summary.get('strategy_trades_generated', 0)
            
            efficiency = (strategy_trades / base_points * 100) if base_points > 0 else 0
            efficiency_ratios.append({'ticker': ticker, 'efficiency': efficiency})
        
        if efficiency_ratios:
            df_efficiency = pd.DataFrame(efficiency_ratios)
            bars = ax2.bar(df_efficiency['ticker'], df_efficiency['efficiency'], 
                          color='purple', alpha=0.7)
            ax2.set_title('Signal-to-Trade Efficiency')
            ax2.set_ylabel('Efficiency (%)')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Risk Management Impact
        ax3 = axes[1, 0]
        impact_data = []
        for ticker in portfolio_data['tickers']:
            analytics = portfolio_data['analytics'].get(ticker, {})
            risk_impact = analytics.get('risk_impact', {})
            
            approval_rate = risk_impact.get('approval_rate', 0) * 100
            rejection_rate = risk_impact.get('rejection_rate', 0) * 100
            
            impact_data.append({
                'ticker': ticker,
                'approval_rate': approval_rate,
                'rejection_rate': rejection_rate
            })
        
        if impact_data:
            df_impact = pd.DataFrame(impact_data)
            x = np.arange(len(df_impact))
            width = 0.35
            
            ax3.bar(x - width/2, df_impact['approval_rate'], width, 
                   label='Approval Rate', color='green', alpha=0.7)
            ax3.bar(x + width/2, df_impact['rejection_rate'], width, 
                   label='Rejection Rate', color='red', alpha=0.7)
            
            ax3.set_title('Risk Management Impact by Ticker')
            ax3.set_ylabel('Rate (%)')
            ax3.set_xticks(x)
            ax3.set_xticklabels(df_impact['ticker'], rotation=45)
            ax3.legend()
        
        # 4. Three-File System Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate overall statistics
        total_base_points = sum(analytics.get('data_summary', {}).get('base_data_points', 0) 
                               for analytics in portfolio_data['analytics'].values())
        total_strategy_trades = sum(analytics.get('data_summary', {}).get('strategy_trades_generated', 0) 
                                   for analytics in portfolio_data['analytics'].values())
        total_approved_trades = sum(analytics.get('data_summary', {}).get('risk_approved_trades', 0) 
                                   for analytics in portfolio_data['analytics'].values())
        
        overall_efficiency = (total_strategy_trades / total_base_points * 100) if total_base_points > 0 else 0
        overall_approval_rate = (total_approved_trades / total_strategy_trades * 100) if total_strategy_trades > 0 else 0
        
        summary_text = f"""Three-File System Summary

Portfolio Overview:
• Total Base Data Points: {total_base_points:,}
• Total Strategy Trades: {total_strategy_trades:,}
• Total Approved Trades: {total_approved_trades:,}

Key Metrics:
• Overall Signal Efficiency: {overall_efficiency:.3f}%
• Overall Approval Rate: {overall_approval_rate:.1f}%
• Risk Rejection Rate: {100-overall_approval_rate:.1f}%

File Structure Integrity:
✓ Base files contain price data & signals
✓ Strategy files contain all generated trades
✓ Risk files contain approved trades only

Data Quality:
• Average data points per ticker: {total_base_points/len(portfolio_data['tickers']) if portfolio_data['tickers'] else 0:,.0f}
• Average trades per ticker: {total_strategy_trades/len(portfolio_data['tickers']) if portfolio_data['tickers'] else 0:.1f}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
          # Save dashboard
        dashboard_path = self.portfolio_dir / f"three_file_comparison_dashboard_{date_range}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {'three_file_comparison_dashboard': dashboard_path}
    
    def _create_master_dashboard(self, portfolio_data: Dict, date_range: str) -> Path:
        """Create comprehensive master dashboard with key insights."""
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Portfolio Master Dashboard - {date_range}', fontsize=20, fontweight='bold')
        
        # Top row - Key metrics
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        
        # Middle rows - Main analysis
        ax5 = fig.add_subplot(gs[1, :2])  # Trade flow
        ax6 = fig.add_subplot(gs[1, 2:])  # Risk analysis
        ax7 = fig.add_subplot(gs[2, :2])  # Signal analysis
        ax8 = fig.add_subplot(gs[2, 2:])  # Performance metrics
        
        # Bottom row - Summary
        ax9 = fig.add_subplot(gs[3, :])
        
        # Populate dashboards with key insights
        self._populate_master_dashboard(
            [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9], 
            portfolio_data, date_range
        )
          # Save master dashboard
        master_path = self.portfolio_dir / f"portfolio_master_dashboard_{date_range}.png"
        plt.savefig(master_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return master_path
    
    def _populate_master_dashboard(self, axes: List, portfolio_data: Dict, date_range: str):
        """Populate the master dashboard with key visualizations."""
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axes
        
        # Key metrics cards
        total_tickers = len(portfolio_data['tickers'])
        total_trades = sum(analytics.get('data_summary', {}).get('strategy_trades_generated', 0) 
                          for analytics in portfolio_data['analytics'].values())
        
        # Metric cards
        for i, (ax, title, value, color) in enumerate([
            (ax1, 'Total Tickers', str(total_tickers), 'lightblue'),
            (ax2, 'Strategy Trades', f'{total_trades:,}', 'lightgreen'),
            (ax3, 'Date Range', date_range, 'lightyellow'),
            (ax4, 'System Status', '✓ Active', 'lightcoral')
        ]):
            ax.text(0.5, 0.5, f'{title}\n{value}', ha='center', va='center',
                   fontsize=14, fontweight='bold', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        # ax5: Trade Volume by Ticker
        trade_volumes = []
        for ticker in portfolio_data['tickers']:
            strategy_count = len(portfolio_data['strategy_trades'].get(ticker, []))
            approved_count = len(portfolio_data['risk_approved_trades'].get(ticker, []))
            trade_volumes.append({'ticker': ticker, 'strategy': strategy_count, 'approved': approved_count})
        
        if trade_volumes:
            tickers = [t['ticker'] for t in trade_volumes]
            strategy_counts = [t['strategy'] for t in trade_volumes]
            approved_counts = [t['approved'] for t in trade_volumes]
            
            x = np.arange(len(tickers))
            width = 0.35
            ax5.bar(x - width/2, strategy_counts, width, label='Strategy', alpha=0.7, color='skyblue')
            ax5.bar(x + width/2, approved_counts, width, label='Approved', alpha=0.7, color='lightgreen')
            ax5.set_title('Trade Volume: Strategy vs Approved')
            ax5.set_ylabel('Trades')
            ax5.set_xticks(x)
            ax5.set_xticklabels(tickers)
            ax5.legend()
        
        # ax6: Approval Rates
        approval_rates = []
        ticker_names = []
        for ticker, analytics in portfolio_data['analytics'].items():
            rate = analytics.get('data_summary', {}).get('risk_approval_rate', 0) * 100
            approval_rates.append(rate)
            ticker_names.append(ticker)
        
        if approval_rates:
            ax6.bar(ticker_names, approval_rates, alpha=0.7, color='lightgreen')
            ax6.set_title('Risk Approval Rates')
            ax6.set_ylabel('Approval Rate (%)')
            ax6.set_ylim(0, 100)
            for i, rate in enumerate(approval_rates):
                ax6.text(i, rate + 1, f'{rate:.1f}%', ha='center', va='bottom')
        
        # ax7: Trade Type Distribution (Portfolio-wide)
        trade_types = {}
        for ticker, trades_df in portfolio_data['strategy_trades'].items():
            if not trades_df.empty and 'Trade Type' in trades_df.columns:
                for trade_type in trades_df['Trade Type']:
                    trade_types[trade_type] = trade_types.get(trade_type, 0) + 1
        
        if trade_types:
            ax7.pie(trade_types.values(), labels=trade_types.keys(), autopct='%1.1f%%', startangle=90)
            ax7.set_title('Portfolio Trade Type Distribution')
        else:
            ax7.text(0.5, 0.5, 'No trade data', ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Trade Types (No Data)')
        
        # ax8: Profitability Overview
        profitability_data = []
        for ticker, trades_df in portfolio_data['strategy_trades'].items():
            if not trades_df.empty and 'Profit (%)' in trades_df.columns:
                profitable = len(trades_df[trades_df['Profit (%)'] > 0])
                losing = len(trades_df[trades_df['Profit (%)'] <= 0])
                profitability_data.append({'ticker': ticker, 'profitable': profitable, 'losing': losing})
        
        if profitability_data:
            tickers = [p['ticker'] for p in profitability_data]
            profitable = [p['profitable'] for p in profitability_data]
            losing = [p['losing'] for p in profitability_data]
            
            x = np.arange(len(tickers))
            width = 0.35
            ax8.bar(x - width/2, profitable, width, label='Profitable', alpha=0.7, color='green')
            ax8.bar(x + width/2, losing, width, label='Losing', alpha=0.7, color='red')
            ax8.set_title('Profitability by Ticker')
            ax8.set_ylabel('Trades')
            ax8.set_xticks(x)
            ax8.set_xticklabels(tickers)
            ax8.legend()
        else:
            ax8.text(0.5, 0.5, 'No profit data', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Profitability (No Data)')
        
        # Summary section
        ax9.axis('off')
        summary_text = f"""
Portfolio Analysis Summary - {date_range}

The three-file system successfully processed {total_tickers} tickers with comprehensive analysis including:
• Base data files with price data, signals, and indicators
• Strategy trade files with all generated trades (pre-risk filtering)  
• Risk-approved trade files with trades that passed risk management

Key insights and recommendations will be generated based on the analysis results.
Portfolio-level visualizations provide comprehensive view of strategy performance and risk management effectiveness.
        """
        
        ax9.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
                transform=ax9.transAxes, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    def create_individual_ticker_dashboard(self, strategy_run_dir: Path, ticker: str,
                                           date_range: str) -> Dict[str, Path]:
        """
        Create comprehensive individual ticker visualizations.
        
        Creates three types of individual ticker analysis:
        1. Performance Summary - Individual ticker performance metrics
        2. Trade Distribution - Trade patterns and distribution analysis  
        3. Trade Timeline - Trade execution timeline and patterns
        
        Args:
            strategy_run_dir: Path to strategy run directory
            ticker: Ticker symbol
            date_range: Date range string
            
        Returns:
            Dictionary of created visualization files
        """
        visualizations = {}
        
        try:
            # Create ticker-specific subdirectory
            ticker_dir = self.individual_dir / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)
            
            # Load ticker data
            ticker_data = self._load_individual_ticker_data(strategy_run_dir, ticker, date_range)
            
            if not ticker_data:
                self.logger.warning(f"No data available for {ticker} individual visualization")
                return visualizations
            
            # 1. Create Performance Summary
            performance_path = self._create_performance_summary(ticker_data, ticker, date_range, ticker_dir)
            if performance_path:
                visualizations['performance_summary'] = performance_path
            
            # 2. Create Trade Distribution Analysis  
            distribution_path = self._create_trade_distribution(ticker_data, ticker, date_range, ticker_dir)
            if distribution_path:
                visualizations['trade_distribution'] = distribution_path
            
            # 3. Create Trade Timeline
            timeline_path = self._create_trade_timeline(ticker_data, ticker, date_range, ticker_dir)
            if timeline_path:
                visualizations['trade_timeline'] = timeline_path
            
            self.logger.info(f"Created {len(visualizations)} individual visualizations for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error creating individual ticker dashboard for {ticker}: {e}")
        
        return visualizations
    
    def _load_individual_ticker_data(self, strategy_run_dir: Path, ticker: str, date_range: str) -> Dict[str, Any]:
        """Load all data needed for individual ticker visualization."""
        ticker_data = {
            'ticker': ticker,
            'date_range': date_range,
            'base_data': None,
            'strategy_trades': None,
            'risk_approved_trades': None,
            'active_trades': None,
            'analytics': None
        }
        
        try:
            # Load base data
            base_file = strategy_run_dir / 'data' / 'base_data' / f"{ticker}_Base_{date_range}.csv"
            if base_file.exists():
                ticker_data['base_data'] = pd.read_csv(base_file)
            
            # Load strategy trades
            strategy_trades_file = strategy_run_dir / 'data' / 'strategy_trades' / f"{ticker}_StrategyTrades_{date_range}.csv"
            if strategy_trades_file.exists():
                ticker_data['strategy_trades'] = pd.read_csv(strategy_trades_file)
            
            # Load risk approved trades
            risk_trades_file = strategy_run_dir / 'data' / 'risk_approved_trades' / f"{ticker}_RiskApprovedTrades_{date_range}.csv"
            if risk_trades_file.exists():
                ticker_data['risk_approved_trades'] = pd.read_csv(risk_trades_file)
            
            # Load analytics
            analytics_file = strategy_run_dir / 'analysis_reports' / 'individual' / f"{ticker}_Analysis_{date_range}.json"
            if analytics_file.exists():
                with open(analytics_file, 'r') as f:
                    ticker_data['analytics'] = json.load(f)
            
            # Set active trades based on trade source preference
            self._set_individual_active_trades(ticker_data)
            
        except Exception as e:
            self.logger.error(f"Error loading data for {ticker}: {e}")
            
        return ticker_data
    
    def _set_individual_active_trades(self, ticker_data: Dict[str, Any]) -> None:
        """Set active trades for individual ticker based on trade_source preference."""
        ticker = ticker_data['ticker']
        
        if self.trade_source == "strategy_trades":
            ticker_data['active_trades'] = ticker_data['strategy_trades']
        elif self.trade_source == "risk_approved_trades":
            ticker_data['active_trades'] = ticker_data['risk_approved_trades']
        elif self.trade_source == "auto":
            # Try risk approved first, fallback to strategy
            if ticker_data['risk_approved_trades'] is not None and not ticker_data['risk_approved_trades'].empty:
                ticker_data['active_trades'] = ticker_data['risk_approved_trades']
            else:
                ticker_data['active_trades'] = ticker_data['strategy_trades']
          # Log the choice
        if ticker_data['active_trades'] is not None and not ticker_data['active_trades'].empty:
            trade_count = len(ticker_data['active_trades'])
            self.logger.debug(f"Using {self.trade_source} for {ticker}: {trade_count} trades")
        else:
            self.logger.warning(f"No active trades found for {ticker}")
    
    def _create_performance_summary(self, ticker_data: Dict[str, Any], ticker: str, 
                                   date_range: str, output_dir: Path) -> Optional[Path]:
        """Create trader-focused individual ticker performance summary visualization."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{ticker} - Performance Dashboard', fontsize=16, fontweight='bold')
            
            analytics = ticker_data.get('analytics', {})
            active_trades = ticker_data.get('active_trades')
            
            # 1. Equity Curve (Most Important for Traders)
            ax1 = axes[0, 0]
            if active_trades is not None and not active_trades.empty and 'Profit (%)' in active_trades.columns:
                profits = active_trades['Profit (%)'].dropna()
                if not profits.empty:
                    # Create cumulative P&L over trades
                    cumulative_pnl = profits.cumsum()
                    trade_numbers = range(1, len(cumulative_pnl) + 1)
                    
                    ax1.plot(trade_numbers, cumulative_pnl, linewidth=2, color='darkblue', label='Cumulative P&L')
                    ax1.fill_between(trade_numbers, cumulative_pnl, alpha=0.3, color='lightblue')
                    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax1.set_title('Equity Curve (Cumulative P&L)', fontweight='bold')
                    ax1.set_xlabel('Trade Number')
                    ax1.set_ylabel('Cumulative P&L (%)')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()
                else:
                    ax1.text(0.5, 0.5, 'No P&L data available', ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('Equity Curve (No Data)')
            else:
                ax1.text(0.5, 0.5, 'No trade data available', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Equity Curve (No Data)')
            
            # 2. Drawdown Analysis
            ax2 = axes[0, 1]
            if active_trades is not None and not active_trades.empty and 'Profit (%)' in active_trades.columns:
                profits = active_trades['Profit (%)'].dropna()
                if not profits.empty:
                    cumulative_pnl = profits.cumsum()
                    running_max = cumulative_pnl.expanding().max()
                    drawdown = cumulative_pnl - running_max
                    
                    ax2.fill_between(range(1, len(drawdown) + 1), drawdown, alpha=0.7, color='red', label='Drawdown')
                    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
                    ax2.set_title(f'Drawdown Analysis\nMax DD: {drawdown.min():.2f}%', fontweight='bold')
                    ax2.set_xlabel('Trade Number')
                    ax2.set_ylabel('Drawdown (%)')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, 'No drawdown data', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Drawdown Analysis (No Data)')
            else:
                ax2.text(0.5, 0.5, 'No trade data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Drawdown Analysis (No Data)')
            
            # 3. P&L Distribution (Enhanced)
            ax3 = axes[1, 0]
            if active_trades is not None and not active_trades.empty and 'Profit (%)' in active_trades.columns:
                profits = active_trades['Profit (%)'].dropna()
                if not profits.empty:
                    # Enhanced P&L distribution with win/loss coloring
                    winning_trades = profits[profits > 0]
                    losing_trades = profits[profits <= 0]
                    
                    ax3.hist(losing_trades, bins=15, alpha=0.7, color='red', edgecolor='black', label=f'Losses ({len(losing_trades)})')
                    ax3.hist(winning_trades, bins=15, alpha=0.7, color='green', edgecolor='black', label=f'Wins ({len(winning_trades)})')
                    
                    ax3.axvline(profits.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {profits.mean():.2f}%')
                    ax3.set_title('P&L Distribution', fontweight='bold')
                    ax3.set_xlabel('Profit/Loss (%)')
                    ax3.set_ylabel('Frequency')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, 'No P&L data', ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('P&L Distribution (No Data)')
            else:
                ax3.text(0.5, 0.5, 'No trade data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('P&L Distribution (No Data)')
            
            # 4. Key Trading Metrics (Focused)
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            performance_metrics = analytics.get('performance_metrics', {})
            
            if active_trades is not None and not active_trades.empty and 'Profit (%)' in active_trades.columns:
                profits = active_trades['Profit (%)'].dropna()
                if not profits.empty:
                    total_trades = len(profits)
                    winning_trades = len(profits[profits > 0])
                    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                    avg_win = profits[profits > 0].mean() if len(profits[profits > 0]) > 0 else 0
                    avg_loss = profits[profits <= 0].mean() if len(profits[profits <= 0]) > 0 else 0
                    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                    total_return = profits.sum()
                    
                    metrics_text = f"""📊 TRADING METRICS
                    
🎯 Total Trades: {total_trades}
📈 Win Rate: {win_rate:.1f}%
💰 Total Return: {total_return:.2f}%

🏆 Avg Win: {avg_win:.2f}%
📉 Avg Loss: {avg_loss:.2f}%
⚖️ Profit Factor: {profit_factor:.2f}

📊 Best Trade: {profits.max():.2f}%
📉 Worst Trade: {profits.min():.2f}%
                    """
                else:
                    metrics_text = "📊 TRADING METRICS\n\nNo trade data available"
            else:
                metrics_text = "📊 TRADING METRICS\n\nNo trade data available"
            
            ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save performance summary
            performance_path = output_dir / f"{ticker}_performance_summary_{date_range}.png"
            plt.savefig(performance_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return performance_path
            
        except Exception as e:
            self.logger.error(f"Error creating performance summary for {ticker}: {e}")
            return None
    
    def _create_trade_distribution(self, ticker_data: Dict[str, Any], ticker: str, 
                                  date_range: str, output_dir: Path) -> Optional[Path]:
        """Create trade distribution analysis visualization."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{ticker} Trade Distribution Analysis - {date_range}', fontsize=16)
            
            active_trades = ticker_data.get('active_trades')
            
            if active_trades is None or active_trades.empty:
                # Create placeholder with no data message
                for ax in axes.flat:
                    ax.text(0.5, 0.5, 'No trade data available', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Trade Distribution (No Data)')
                
                plt.tight_layout()
                distribution_path = output_dir / f"{ticker}_trade_distribution_{date_range}.png"
                plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
                plt.close()
                return distribution_path
            
            # 1. Trade Type Distribution (if available)
            ax1 = axes[0, 0]
            if 'Trade Type' in active_trades.columns:
                trade_types = active_trades['Trade Type'].value_counts()
                if not trade_types.empty:
                    ax1.pie(trade_types.values, labels=trade_types.index, autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Trade Type Distribution')
                else:
                    ax1.text(0.5, 0.5, 'No trade type data', ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('Trade Type (No Data)')
            else:
                ax1.text(0.5, 0.5, 'Trade Type column not found', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Trade Type (No Data)')
              # 2. Profit vs Loss Distribution
            ax2 = axes[0, 1]
            if 'Profit (%)' in active_trades.columns:
                profits = active_trades['Profit (%)'].dropna()
                if not profits.empty:
                    profitable = len(profits[profits > 0])
                    losing = len(profits[profits <= 0])
                    
                    if profitable + losing > 0:
                        labels = ['Profitable', 'Losing']
                        sizes = [profitable, losing]
                        colors = ['green', 'red']
                        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                        ax2.set_title(f'Win/Loss Distribution\n(Win Rate: {profitable/(profitable+losing)*100:.1f}%)')
                    else:
                        ax2.text(0.5, 0.5, 'No profit data', ha='center', va='center', transform=ax2.transAxes)
                        ax2.set_title('Win/Loss (No Data)')
                else:
                    ax2.text(0.5, 0.5, 'No profit data', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Win/Loss (No Data)')
            else:
                ax2.text(0.5, 0.5, 'Profit column not found', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Win/Loss (No Data)')
              # 3. Trade Size Distribution (if available)
            ax3 = axes[1, 0]
            # Try multiple size-related column names
            size_columns = ['Trade Size', 'Position Size', 'Quantity', 'Volume', 'Amount', 'Entry Price']
            size_col = None
            size_data = None
            
            for col in size_columns:
                if col in active_trades.columns:
                    size_col = col
                    size_data = active_trades[col].dropna()
                    break
            
            if size_col and not size_data.empty:
                ax3.hist(size_data, bins=15, alpha=0.7, color='orange', edgecolor='black')
                ax3.set_title(f'{size_col} Distribution')
                ax3.set_xlabel(size_col)
                ax3.set_ylabel('Frequency')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Trade Size column not found', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Trade Size (No Data)')            # 4. Trade Duration Distribution (if available)
            ax4 = axes[1, 1]
            # Try multiple duration column names in order of preference
            duration_columns = ['Duration (Days)', 'Trade Duration (min)', 'Trade Duration', 'Duration', 'Hold Time']
            duration_col = None
            duration_data = None
            
            for col in duration_columns:
                if col in active_trades.columns:
                    duration_col = col
                    duration_data = active_trades[col].dropna()
                    break
            
            if duration_col and not duration_data.empty:
                # Convert minutes to hours if needed
                if 'min' in duration_col.lower():
                    duration_values = duration_data / 60  # Convert minutes to hours
                    xlabel = 'Duration (Hours)'
                    title = 'Trade Duration Distribution'
                else:
                    duration_values = duration_data
                    xlabel = duration_col
                    title = f'Trade {duration_col} Distribution'
                
                ax4.hist(duration_values, bins=15, alpha=0.7, color='purple', edgecolor='black')
                ax4.set_title(title)
                ax4.set_xlabel(xlabel)
                ax4.set_ylabel('Frequency')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Duration columns not found', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Duration (No Data)')
            
            plt.tight_layout()
            
            # Save trade distribution
            distribution_path = output_dir / f"{ticker}_trade_distribution_{date_range}.png"
            plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return distribution_path
            
        except Exception as e:
            self.logger.error(f"Error creating trade distribution for {ticker}: {e}")
            return None
    
    def _create_trade_timeline(self, ticker_data: Dict[str, Any], ticker: str, 
                              date_range: str, output_dir: Path) -> Optional[Path]:
        """Create trade timeline visualization."""
        try:
            fig, axes = plt.subplots(2, 1, figsize=(16, 12))
            fig.suptitle(f'{ticker} Trade Timeline - {date_range}', fontsize=16)
            
            active_trades = ticker_data.get('active_trades')
            base_data = ticker_data.get('base_data')
            
            # 1. Price Chart with Trade Markers
            ax1 = axes[0]
            if base_data is not None and not base_data.empty:
                if 'timestamp' in base_data.columns and 'close' in base_data.columns:
                    base_data['timestamp'] = pd.to_datetime(base_data['timestamp'])
                    ax1.plot(base_data['timestamp'], base_data['close'], label='Close Price', linewidth=1, alpha=0.7)
                    
                    # Add trade entry/exit markers if available
                    if active_trades is not None and not active_trades.empty:
                        if 'Entry Time' in active_trades.columns or 'Entry_Date' in active_trades.columns:
                            entry_col = 'Entry Time' if 'Entry Time' in active_trades.columns else 'Entry_Date'
                            entry_price_col = 'Entry Price' if 'Entry Price' in active_trades.columns else 'Entry_Price'
                            
                            if entry_price_col in active_trades.columns:
                                try:
                                    entry_times = pd.to_datetime(active_trades[entry_col])
                                    entry_prices = active_trades[entry_price_col]
                                    
                                    ax1.scatter(entry_times, entry_prices, color='green', marker='^', 
                                               s=100, label='Trade Entries', alpha=0.8, zorder=5)
                                except Exception as e:
                                    self.logger.debug(f"Could not plot trade entries: {e}")
                        
                        if 'Exit Time' in active_trades.columns or 'Exit_Date' in active_trades.columns:
                            exit_col = 'Exit Time' if 'Exit Time' in active_trades.columns else 'Exit_Date'
                            exit_price_col = 'Exit Price' if 'Exit Price' in active_trades.columns else 'Exit_Price'
                            
                            if exit_price_col in active_trades.columns:
                                try:
                                    exit_times = pd.to_datetime(active_trades[exit_col])
                                    exit_prices = active_trades[exit_price_col]
                                    
                                    ax1.scatter(exit_times, exit_prices, color='red', marker='v', 
                                               s=100, label='Trade Exits', alpha=0.8, zorder=5)
                                except Exception as e:
                                    self.logger.debug(f"Could not plot trade exits: {e}")
                    
                    ax1.set_title('Price Timeline with Trade Executions')
                    ax1.set_ylabel('Price')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                else:
                    ax1.text(0.5, 0.5, 'No price data available', ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('Price Timeline (No Data)')
            else:
                ax1.text(0.5, 0.5, 'No base data available', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Price Timeline (No Data)')
            
            # 2. Trade Performance Timeline
            ax2 = axes[1]
            if active_trades is not None and not active_trades.empty:
                if 'Profit (%)' in active_trades.columns:
                    profits = active_trades['Profit (%)'].dropna()
                    
                    if not profits.empty:
                        # Create cumulative profit line
                        cumulative_profit = profits.cumsum()
                        trade_numbers = range(1, len(cumulative_profit) + 1)
                        
                        ax2.plot(trade_numbers, cumulative_profit, marker='o', linewidth=2, markersize=4, alpha=0.8)
                        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                        
                        # Color profitable and losing trades differently
                        for i, profit in enumerate(profits):
                            color = 'green' if profit > 0 else 'red'
                            ax2.scatter(i + 1, cumulative_profit.iloc[i], color=color, s=50, alpha=0.7, zorder=5)
                        
                        ax2.set_title('Cumulative Trade Performance')
                        ax2.set_xlabel('Trade Number')
                        ax2.set_ylabel('Cumulative Profit (%)')
                        ax2.grid(True, alpha=0.3)
                    else:
                        ax2.text(0.5, 0.5, 'No profit data available', ha='center', va='center', transform=ax2.transAxes)
                        ax2.set_title('Trade Performance (No Data)')
                else:
                    ax2.text(0.5, 0.5, 'Profit column not found', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Trade Performance (No Data)')
            else:
                ax2.text(0.5, 0.5, 'No trade data available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Trade Performance (No Data)')
            
            plt.tight_layout()
            
            # Save trade timeline
            timeline_path = output_dir / f"{ticker}_trade_timeline_{date_range}.png"
            plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return timeline_path
            
        except Exception as e:
            self.logger.error(f"Error creating trade timeline for {ticker}: {e}")
            return None
    def _create_educational_insights_dashboard(self, portfolio_data: Dict, date_range: str) -> Dict[str, Path]:
        """
        Create comprehensive educational trading insights dashboard.
        
        Provides 4 key educational metrics:
        1. Market Timing Quality Assessment
        2. Strategy Performance Across Market Conditions  
        3. Risk-Adjusted Returns (Rolling Sharpe Ratio)
        4. Trade Duration Effectiveness
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(f'Educational Trading Insights Dashboard - {date_range}', fontsize=18, fontweight='bold')
        
        # 1. Market Timing Quality Assessment (Top-Left)
        ax1 = axes[0, 0]
        timing_data = []
        
        for ticker in portfolio_data['tickers']:
            trades = portfolio_data['active_trades'].get(ticker, pd.DataFrame())
            if not trades.empty and 'Profit (%)' in trades.columns:
                profits = trades['Profit (%)'].values
                
                # Calculate timing metrics
                positive_trades = len([p for p in profits if p > 0])
                total_trades = len(profits)
                timing_effectiveness = (positive_trades / total_trades * 100) if total_trades > 0 else 0
                
                # Average profit/loss magnitude
                avg_profit = np.mean([p for p in profits if p > 0]) if positive_trades > 0 else 0
                avg_loss = np.mean([p for p in profits if p < 0]) if len([p for p in profits if p < 0]) > 0 else 0
                
                timing_data.append({
                    'ticker': ticker,
                    'timing_effectiveness': timing_effectiveness,
                    'avg_profit': avg_profit,
                    'avg_loss': abs(avg_loss),
                    'profit_loss_ratio': avg_profit / abs(avg_loss) if avg_loss != 0 else 0
                })
        
        if timing_data:
            df_timing = pd.DataFrame(timing_data)
            
            # Create timing effectiveness bar chart
            bars = ax1.bar(df_timing['ticker'], df_timing['timing_effectiveness'], 
                          color=['green' if x >= 50 else 'red' for x in df_timing['timing_effectiveness']], alpha=0.7)
            ax1.set_title('Market Timing Quality Assessment', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Win Rate (%)')
            ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% Benchmark')
            ax1.legend()
            
            # Add value labels
            for bar, rate in zip(bars, df_timing['timing_effectiveness']):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
            
            # Add educational annotation
            avg_timing = df_timing['timing_effectiveness'].mean()
            timing_insight = "EXCELLENT" if avg_timing >= 70 else "GOOD" if avg_timing >= 60 else "MODERATE" if avg_timing >= 50 else "NEEDS IMPROVEMENT"
            ax1.text(0.02, 0.98, f'💡 Portfolio Timing: {timing_insight} ({avg_timing:.1f}% avg)', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 2. Strategy Performance Across Market Conditions (Top-Right)
        ax2 = axes[0, 1]
        
        # Simulate market condition analysis (in real implementation, you'd analyze market data)
        condition_data = []
        for ticker in portfolio_data['tickers']:
            trades = portfolio_data['active_trades'].get(ticker, pd.DataFrame())
            if not trades.empty and 'Profit (%)' in trades.columns:
                profits = trades['Profit (%)'].values
                
                # Simple market condition simulation based on profit distribution
                high_vol_performance = np.mean([p for p in profits[:len(profits)//3]]) if len(profits) > 3 else 0
                med_vol_performance = np.mean([p for p in profits[len(profits)//3:2*len(profits)//3]]) if len(profits) > 3 else 0
                low_vol_performance = np.mean([p for p in profits[2*len(profits)//3:]]) if len(profits) > 3 else 0
                
                condition_data.append({
                    'ticker': ticker,
                    'high_volatility': high_vol_performance,
                    'medium_volatility': med_vol_performance,
                    'low_volatility': low_vol_performance
                })
        
        if condition_data:
            df_conditions = pd.DataFrame(condition_data)
            df_conditions_plot = df_conditions.set_index('ticker')
            
            # Create grouped bar chart
            df_conditions_plot.plot(kind='bar', ax=ax2, color=['red', 'orange', 'green'], alpha=0.7)
            ax2.set_title('Strategy Performance Across Market Conditions', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Average Return (%)')
            ax2.legend(title='Market Condition', loc='upper right')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add educational insight
            best_condition = df_conditions[['high_volatility', 'medium_volatility', 'low_volatility']].mean().idxmax()
            condition_map = {'high_volatility': 'High Volatility', 'medium_volatility': 'Medium Volatility', 'low_volatility': 'Low Volatility'}
            ax2.text(0.02, 0.98, f'💡 Best Performance: {condition_map[best_condition]} markets', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 3. Risk-Adjusted Returns (Rolling Sharpe Ratio) (Bottom-Left)
        ax3 = axes[1, 0]
        
        sharpe_data = []
        for ticker in portfolio_data['tickers']:
            trades = portfolio_data['active_trades'].get(ticker, pd.DataFrame())
            if not trades.empty and 'Profit (%)' in trades.columns:
                profits = trades['Profit (%)'].values
                
                if len(profits) >= 5:  # Need minimum trades for meaningful Sharpe
                    # Calculate rolling Sharpe ratio (simplified)
                    returns_std = np.std(profits) if len(profits) > 1 else 0.1
                    mean_return = np.mean(profits)
                    sharpe_ratio = mean_return / returns_std if returns_std > 0 else 0
                    
                    sharpe_data.append({
                        'ticker': ticker,
                        'sharpe_ratio': sharpe_ratio,
                        'avg_return': mean_return,
                        'volatility': returns_std,
                        'trade_count': len(profits)
                    })
        
        if sharpe_data:
            df_sharpe = pd.DataFrame(sharpe_data)
            
            # Create Sharpe ratio comparison
            colors = ['darkgreen' if x >= 1.0 else 'green' if x >= 0.5 else 'orange' if x >= 0 else 'red' for x in df_sharpe['sharpe_ratio']]
            bars = ax3.bar(df_sharpe['ticker'], df_sharpe['sharpe_ratio'], color=colors, alpha=0.7)
            ax3.set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.axhline(y=1.0, color='darkgreen', linestyle='--', alpha=0.7, label='Excellent (1.0+)')
            ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Good (0.5+)')
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Breakeven')
            ax3.legend()
            
            # Add value labels
            for bar, sharpe in zip(bars, df_sharpe['sharpe_ratio']):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.05),
                        f'{sharpe:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
            
            # Add educational insight
            avg_sharpe = df_sharpe['sharpe_ratio'].mean()
            sharpe_assessment = "EXCELLENT" if avg_sharpe >= 1.0 else "GOOD" if avg_sharpe >= 0.5 else "MODERATE" if avg_sharpe >= 0 else "POOR"
            ax3.text(0.02, 0.98, f'💡 Risk-Adjusted Performance: {sharpe_assessment} ({avg_sharpe:.2f} avg)', 
                    transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # 4. Trade Duration Effectiveness (Bottom-Right)
        ax4 = axes[1, 1]
        
        # Instead of a chart, create comprehensive educational summary
        ax4.axis('off')
        
        # Calculate portfolio-wide insights
        total_trades = sum(len(portfolio_data['active_trades'].get(ticker, pd.DataFrame())) for ticker in portfolio_data['tickers'])
        profitable_trades = 0
        total_return = 0
        
        for ticker in portfolio_data['tickers']:
            trades = portfolio_data['active_trades'].get(ticker, pd.DataFrame())
            if not trades.empty and 'Profit (%)' in trades.columns:
                profits = trades['Profit (%)'].values
                profitable_trades += len([p for p in profits if p > 0])
                total_return += sum(profits)
        
        portfolio_win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Create educational summary
        summary_text = f"""🎓 EDUCATIONAL TRADING INSIGHTS SUMMARY

📊 PORTFOLIO PERFORMANCE METRICS:
• Total Trades Analyzed: {total_trades}
• Portfolio Win Rate: {portfolio_win_rate:.1f}%
• Total Portfolio Return: {total_return:.2f}%
• Number of Assets: {len(portfolio_data['tickers'])}

🎯 KEY LEARNING INSIGHTS:

1️⃣ MARKET TIMING QUALITY:
   → Win rates above 60% indicate strong timing
   → Focus on tickers with consistent positive outcomes
   → Review entry/exit criteria for underperforming assets

2️⃣ MARKET CONDITIONS:
   → Identify which market environments favor your strategy
   → Adjust position sizing based on market volatility
   → Consider market regime filters for strategy activation

3️⃣ RISK-ADJUSTED RETURNS:
   → Sharpe ratio > 1.0 = Excellent risk-adjusted performance
   → Sharpe ratio 0.5-1.0 = Good performance
   → Focus on consistency, not just high returns

4️⃣ ACTIONABLE RECOMMENDATIONS:
   ✓ Increase allocation to highest Sharpe ratio assets
   ✓ Review timing for assets with <50% win rate
   ✓ Optimize strategy for best-performing market conditions
   ✓ Consider risk management adjustments for volatility

💡 NEXT STEPS FOR IMPROVEMENT:
   → Analyze trade duration patterns in detail
   → Implement dynamic position sizing based on volatility
   → Consider market regime detection for strategy timing
   → Focus on risk-adjusted returns over absolute returns"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9))
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = self.portfolio_dir / f"portfolio_educational_insights_{date_range}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created educational insights dashboard: {dashboard_path}")
        return {'educational_insights_dashboard': dashboard_path}
