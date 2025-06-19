"""
Performance Summary Chart for Individual Tickers.

This module provides trader-focused performance visualization for individual tickers,
extracted and enhanced from the original monolith.
"""

import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.base_visualizer import BaseVisualizer


class PerformanceSummaryChart(BaseVisualizer):
    """
    Creates trader-focused individual ticker performance summary visualization.
    
    This chart provides the most critical performance metrics that traders need
    to evaluate individual ticker performance, including equity curves, drawdowns,
    P&L distribution, and key trading metrics.    """
    
    def __init__(self, trade_source: str = "auto", use_enhanced_naming: bool = False):
        """
        Initialize the performance summary chart.
        
        Args:
            trade_source: Trade data source preference
            use_enhanced_naming: Whether to use enhanced trader-focused naming
        """
        super().__init__(trade_source=trade_source, use_enhanced_naming=use_enhanced_naming)
        
    def create(self, ticker_data: Dict[str, Any], ticker: str, 
               date_range: str, output_dir: Path) -> Optional[Path]:
        """
        Create trader-focused individual ticker performance summary visualization.
        
        Args:
            ticker_data: Dictionary containing all ticker data
            ticker: Ticker symbol
            date_range: Date range string
            output_dir: Directory to save the chart
            
        Returns:
            Path to saved chart file, or None if creation failed
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{ticker} - Performance Dashboard', fontsize=16, fontweight='bold')
            
            analytics = ticker_data.get('analytics', {})
            active_trades = ticker_data.get('active_trades')
            
            # 1. Equity Curve (Most Important for Traders)
            self._create_equity_curve(axes[0, 0], active_trades)
            
            # 2. Drawdown Analysis
            self._create_drawdown_analysis(axes[0, 1], active_trades)
            
            # 3. P&L Distribution (Enhanced)
            self._create_pnl_distribution(axes[1, 0], active_trades)
              # 4. Key Trading Metrics (Focused)
            self._create_metrics_summary(axes[1, 1], active_trades, analytics)
            
            # Apply enhanced naming if enabled
            if self.use_enhanced_naming and self.naming_scheme:
                filename, enhanced_title = self.naming_scheme.get_chart_names(
                    "performance_summary", ticker, date_range, self.trade_source
                )
                # Update the figure title with enhanced version
                fig.suptitle(enhanced_title, fontsize=14, fontweight='bold', y=0.95)
                
                # Add educational subtitle
                subtitle = self.naming_scheme.get_enhanced_subtitle_text("performance_summary")
                if subtitle:
                    fig.text(0.5, 0.02, subtitle, ha='center', va='bottom', 
                            fontsize=10, style='italic', wrap=True)
            else:
                filename = f"{ticker}_performance_summary_{date_range}.png"
            
            plt.tight_layout()
            
            # Save performance summary
            performance_path = output_dir / filename
            return self._safe_save_plot(fig, performance_path)
            
        except Exception as e:
            self.logger.error(f"Error creating performance summary for {ticker}: {e}")
            return None
            
    def _create_equity_curve(self, ax, active_trades):
        """Create equity curve subplot."""
        if active_trades is not None and not active_trades.empty and 'Profit (%)' in active_trades.columns:
            profits = active_trades['Profit (%)'].dropna()
            if not profits.empty:
                # Create cumulative P&L over trades
                cumulative_pnl = profits.cumsum()
                trade_numbers = range(1, len(cumulative_pnl) + 1)
                
                ax.plot(trade_numbers, cumulative_pnl, linewidth=2, color='darkblue', label='Cumulative P&L')
                ax.fill_between(trade_numbers, cumulative_pnl, alpha=0.3, color='lightblue')
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.set_title('Equity Curve (Cumulative P&L)', fontweight='bold')
                ax.set_xlabel('Trade Number')
                ax.set_ylabel('Cumulative P&L (%)')
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No P&L data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Equity Curve (No Data)')
        else:
            ax.text(0.5, 0.5, 'No trade data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Equity Curve (No Data)')
            
    def _create_drawdown_analysis(self, ax, active_trades):
        """Create drawdown analysis subplot."""
        if active_trades is not None and not active_trades.empty and 'Profit (%)' in active_trades.columns:
            profits = active_trades['Profit (%)'].dropna()
            if not profits.empty:
                cumulative_pnl = profits.cumsum()
                running_max = cumulative_pnl.expanding().max()
                drawdown = cumulative_pnl - running_max
                
                ax.fill_between(range(1, len(drawdown) + 1), drawdown, alpha=0.7, color='red', label='Drawdown')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
                ax.set_title(f'Drawdown Analysis\\nMax DD: {drawdown.min():.2f}%', fontweight='bold')
                ax.set_xlabel('Trade Number')
                ax.set_ylabel('Drawdown (%)')
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No drawdown data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Drawdown Analysis (No Data)')
        else:
            ax.text(0.5, 0.5, 'No trade data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Drawdown Analysis (No Data)')
            
    def _create_pnl_distribution(self, ax, active_trades):
        """Create P&L distribution subplot."""
        if active_trades is not None and not active_trades.empty and 'Profit (%)' in active_trades.columns:
            profits = active_trades['Profit (%)'].dropna()
            if not profits.empty:
                # Enhanced P&L distribution with win/loss coloring
                winning_trades = profits[profits > 0]
                losing_trades = profits[profits <= 0]
                
                ax.hist(losing_trades, bins=15, alpha=0.7, color='red', edgecolor='black', label=f'Losses ({len(losing_trades)})')
                ax.hist(winning_trades, bins=15, alpha=0.7, color='green', edgecolor='black', label=f'Wins ({len(winning_trades)})')
                
                ax.axvline(profits.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {profits.mean():.2f}%')
                ax.set_title('P&L Distribution', fontweight='bold')
                ax.set_xlabel('Profit/Loss (%)')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No P&L data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('P&L Distribution (No Data)')
        else:
            ax.text(0.5, 0.5, 'No trade data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('P&L Distribution (No Data)')
            
    def _create_metrics_summary(self, ax, active_trades, analytics):
        """Create key trading metrics summary."""
        ax.axis('off')
        
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
                metrics_text = "📊 TRADING METRICS\\n\\nNo trade data available"
        else:
            metrics_text = "📊 TRADING METRICS\\n\\nNo trade data available"
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
