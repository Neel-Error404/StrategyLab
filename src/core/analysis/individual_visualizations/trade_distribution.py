"""
Trade Distribution Chart for Individual Tickers.

This module provides trade pattern and distribution analysis for individual tickers,
extracted and enhanced from the original monolith.
"""

import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.base_visualizer import BaseVisualizer


class TradeDistributionChart(BaseVisualizer):
    """
    Creates trade distribution analysis visualization for individual tickers.    
    This chart analyzes trade patterns including trade types, win/loss distribution,
    trade sizes, and trade duration patterns to help traders understand their
    trading behavior and identify areas for improvement.
    """
    
    def __init__(self, trade_source: str = "auto", use_enhanced_naming: bool = False):
        """
        Initialize the trade distribution chart.
        
        Args:
            trade_source: Trade data source preference
            use_enhanced_naming: Whether to use enhanced trader-focused naming scheme
        """
        super().__init__(trade_source=trade_source, use_enhanced_naming=use_enhanced_naming)
        
    def create(self, ticker_data: Dict[str, Any], ticker: str, 
               date_range: str, output_dir: Path) -> Optional[Path]:
        """
        Create trade distribution analysis visualization.
        
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
            fig.suptitle(f'{ticker} Trade Distribution Analysis - {date_range}', fontsize=16)
            
            active_trades = ticker_data.get('active_trades')
            
            if active_trades is None or active_trades.empty:
                # Create placeholder with no data message
                for ax in axes.flat:
                    ax.text(0.5, 0.5, 'No trade data available', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Trade Distribution (No Data)')
                
                plt.tight_layout()
                distribution_path = output_dir / f"{ticker}_trade_distribution_{date_range}.png"
                return self._safe_save_plot(fig, distribution_path)
            
            # 1. Trade Type Distribution (if available)
            self._create_trade_type_distribution(axes[0, 0], active_trades)
            
            # 2. Profit vs Loss Distribution
            self._create_win_loss_distribution(axes[0, 1], active_trades)
            
            # 3. Trade Size Distribution (if available)
            self._create_trade_size_distribution(axes[1, 0], active_trades)
              # 4. Trade Duration Distribution (if available)
            self._create_trade_duration_distribution(axes[1, 1], active_trades)
            
            # Apply enhanced naming if enabled
            if self.use_enhanced_naming and self.naming_scheme:
                filename, enhanced_title = self.naming_scheme.get_chart_names(
                    "trade_distribution", ticker, date_range, self.trade_source
                )
                # Update the figure title with enhanced version
                fig.suptitle(enhanced_title, fontsize=14, fontweight='bold', y=0.95)
                
                # Add educational subtitle
                subtitle = self.naming_scheme.get_enhanced_subtitle_text("trade_distribution")
                if subtitle:
                    fig.text(0.5, 0.02, subtitle, ha='center', va='bottom', 
                            fontsize=10, style='italic', wrap=True)
            else:
                filename = f"{ticker}_trade_distribution_{date_range}.png"
            
            plt.tight_layout()
            
            # Save trade distribution
            distribution_path = output_dir / filename
            return self._safe_save_plot(fig, distribution_path)
            
        except Exception as e:
            self.logger.error(f"Error creating trade distribution for {ticker}: {e}")
            return None
            
    def _create_trade_type_distribution(self, ax, active_trades):
        """Create trade type distribution subplot."""
        if 'Trade Type' in active_trades.columns:
            trade_types = active_trades['Trade Type'].value_counts()
            if not trade_types.empty:
                ax.pie(trade_types.values, labels=trade_types.index, autopct='%1.1f%%', startangle=90)
                ax.set_title('Trade Type Distribution')
            else:
                ax.text(0.5, 0.5, 'No trade type data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Trade Type (No Data)')
        else:
            ax.text(0.5, 0.5, 'Trade Type column not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trade Type (No Data)')
            
    def _create_win_loss_distribution(self, ax, active_trades):
        """Create win/loss distribution subplot."""
        if 'Profit (%)' in active_trades.columns:
            profits = active_trades['Profit (%)'].dropna()
            if not profits.empty:
                profitable = len(profits[profits > 0])
                losing = len(profits[profits <= 0])
                
                if profitable + losing > 0:
                    labels = ['Profitable', 'Losing']
                    sizes = [profitable, losing]
                    colors = ['green', 'red']
                    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax.set_title(f'Win/Loss Distribution\\n(Win Rate: {profitable/(profitable+losing)*100:.1f}%)')
                else:
                    ax.text(0.5, 0.5, 'No profit data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Win/Loss (No Data)')
            else:
                ax.text(0.5, 0.5, 'No profit data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Win/Loss (No Data)')
        else:
            ax.text(0.5, 0.5, 'Profit column not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Win/Loss (No Data)')
            
    def _create_trade_size_distribution(self, ax, active_trades):
        """Create trade size distribution subplot."""
        if 'Trade Size' in active_trades.columns:
            trade_sizes = active_trades['Trade Size'].dropna()
            if not trade_sizes.empty:
                ax.hist(trade_sizes, bins=15, alpha=0.7, color='orange', edgecolor='black')
                ax.set_title('Trade Size Distribution')
                ax.set_xlabel('Trade Size')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No trade size data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Trade Size (No Data)')
        else:
            ax.text(0.5, 0.5, 'Trade Size column not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trade Size (No Data)')
            
    def _create_trade_duration_distribution(self, ax, active_trades):
        """Create trade duration distribution subplot."""
        if 'Duration (Days)' in active_trades.columns:
            durations = active_trades['Duration (Days)'].dropna()
            if not durations.empty:
                ax.hist(durations, bins=15, alpha=0.7, color='purple', edgecolor='black')
                ax.set_title('Trade Duration Distribution')
                ax.set_xlabel('Duration (Days)')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No duration data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Duration (No Data)')
        else:
            # Try alternative column names
            alt_columns = ['Trade Duration', 'Duration', 'Hold Time']
            duration_col = None
            for col in alt_columns:
                if col in active_trades.columns:
                    duration_col = col
                    break
            
            if duration_col:
                durations = active_trades[duration_col].dropna()
                if not durations.empty:
                    ax.hist(durations, bins=15, alpha=0.7, color='purple', edgecolor='black')
                    ax.set_title(f'Trade {duration_col} Distribution')
                    ax.set_xlabel(duration_col)
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'No {duration_col} data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{duration_col} (No Data)')
            else:
                ax.text(0.5, 0.5, 'Duration columns not found', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Duration (No Data)')
