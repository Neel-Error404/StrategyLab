"""
Trade Timeline Chart for Individual Tickers.

This module provides trade execution timeline and pattern analysis for individual tickers,
extracted and enhanced from the original monolith.
"""

import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.base_visualizer import BaseVisualizer


class TradeTimelineChart(BaseVisualizer):
    """
    Creates trade timeline visualization for individual tickers.    
    This chart shows the chronological progression of trades alongside price data,
    helping traders understand timing, entry/exit points, and cumulative performance
    over time.
    """
    
    def __init__(self, trade_source: str = "auto", use_enhanced_naming: bool = False):
        """
        Initialize the trade timeline chart.
        
        Args:
            trade_source: Trade data source preference
            use_enhanced_naming: Whether to use enhanced trader-focused naming scheme
        """
        super().__init__(trade_source=trade_source, use_enhanced_naming=use_enhanced_naming)
        
    def create(self, ticker_data: Dict[str, Any], ticker: str, 
               date_range: str, output_dir: Path) -> Optional[Path]:
        """
        Create trade timeline visualization.
        
        Args:
            ticker_data: Dictionary containing all ticker data
            ticker: Ticker symbol
            date_range: Date range string
            output_dir: Directory to save the chart
            
        Returns:
            Path to saved chart file, or None if creation failed
        """
        try:
            fig, axes = plt.subplots(2, 1, figsize=(16, 12))
            fig.suptitle(f'{ticker} Trade Timeline - {date_range}', fontsize=16)
            
            active_trades = ticker_data.get('active_trades')
            base_data = ticker_data.get('base_data')
            
            # 1. Price Chart with Trade Markers
            self._create_price_timeline(axes[0], base_data, active_trades)
              # 2. Trade Performance Timeline
            self._create_performance_timeline(axes[1], active_trades)
            
            # Apply enhanced naming if enabled
            if self.use_enhanced_naming and self.naming_scheme:
                filename, enhanced_title = self.naming_scheme.get_chart_names(
                    "trade_timeline", ticker, date_range, self.trade_source
                )
                # Update the figure title with enhanced version
                fig.suptitle(enhanced_title, fontsize=14, fontweight='bold', y=0.95)
                
                # Add educational subtitle
                subtitle = self.naming_scheme.get_enhanced_subtitle_text("trade_timeline")
                if subtitle:
                    fig.text(0.5, 0.02, subtitle, ha='center', va='bottom', 
                            fontsize=10, style='italic', wrap=True)
            else:
                filename = f"{ticker}_trade_timeline_{date_range}.png"
            
            plt.tight_layout()
            
            # Save trade timeline
            timeline_path = output_dir / filename
            return self._safe_save_plot(fig, timeline_path)
            
        except Exception as e:
            self.logger.error(f"Error creating trade timeline for {ticker}: {e}")
            return None
            
    def _create_price_timeline(self, ax, base_data, active_trades):
        """Create price timeline with trade markers subplot."""
        if base_data is not None and not base_data.empty:
            if 'timestamp' in base_data.columns and 'close' in base_data.columns:
                base_data['timestamp'] = pd.to_datetime(base_data['timestamp'])
                ax.plot(base_data['timestamp'], base_data['close'], label='Close Price', linewidth=1, alpha=0.7)
                
                # Add trade entry/exit markers if available
                if active_trades is not None and not active_trades.empty:
                    self._add_trade_markers(ax, active_trades)
                
                ax.set_title('Price Timeline with Trade Executions')
                ax.set_ylabel('Price')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No price data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Price Timeline (No Data)')
        else:
            ax.text(0.5, 0.5, 'No base data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Price Timeline (No Data)')
            
    def _add_trade_markers(self, ax, active_trades):
        """Add trade entry and exit markers to price chart."""
        # Add entry markers
        if 'Entry Time' in active_trades.columns or 'Entry_Date' in active_trades.columns:
            entry_col = 'Entry Time' if 'Entry Time' in active_trades.columns else 'Entry_Date'
            entry_price_col = 'Entry Price' if 'Entry Price' in active_trades.columns else 'Entry_Price'
            
            if entry_price_col in active_trades.columns:
                try:
                    entry_times = pd.to_datetime(active_trades[entry_col])
                    entry_prices = active_trades[entry_price_col]
                    
                    ax.scatter(entry_times, entry_prices, color='green', marker='^', 
                               s=100, label='Trade Entries', alpha=0.8, zorder=5)
                except Exception as e:
                    self.logger.debug(f"Could not plot trade entries: {e}")
        
        # Add exit markers
        if 'Exit Time' in active_trades.columns or 'Exit_Date' in active_trades.columns:
            exit_col = 'Exit Time' if 'Exit Time' in active_trades.columns else 'Exit_Date'
            exit_price_col = 'Exit Price' if 'Exit Price' in active_trades.columns else 'Exit_Price'
            
            if exit_price_col in active_trades.columns:
                try:
                    exit_times = pd.to_datetime(active_trades[exit_col])
                    exit_prices = active_trades[exit_price_col]                    
                    ax.scatter(exit_times, exit_prices, color='red', marker='v', 
                               s=100, label='Trade Exits', alpha=0.8, zorder=5)
                except Exception as e:
                    self.logger.debug(f"Could not plot trade exits: {e}")
    
    def _create_performance_timeline(self, ax, active_trades):
        """Create trade performance timeline subplot."""
        if active_trades is not None and not active_trades.empty:
            if 'Profit (%)' in active_trades.columns:
                profits = active_trades['Profit (%)'].dropna()
                
                if not profits.empty:
                    # Create cumulative profit line
                    cumulative_profit = profits.cumsum()
                    trade_numbers = range(1, len(cumulative_profit) + 1)
                    
                    ax.plot(trade_numbers, cumulative_profit, marker='o', linewidth=2, markersize=4, alpha=0.8)
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    
                    # Color profitable and losing trades differently
                    for i, profit in enumerate(profits):
                        color = 'green' if profit > 0 else 'red'
                        ax.scatter(i + 1, cumulative_profit.iloc[i], color=color, s=50, alpha=0.7, zorder=5)
                    
                    ax.set_title('Cumulative Trade Performance')
                    ax.set_xlabel('Trade Number')
                    ax.set_ylabel('Cumulative Profit (%)')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No profit data available', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Trade Performance (No Data)')
            else:
                ax.text(0.5, 0.5, 'Profit column not found', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Trade Performance (No Data)')
        else:
            ax.text(0.5, 0.5, 'No trade data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trade Performance (No Data)')
