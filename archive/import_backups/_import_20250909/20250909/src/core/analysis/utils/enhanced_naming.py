"""
Enhanced naming utilities for trader-focused visualization file names and titles.

This module provides enhanced naming schemas that make chart purposes and insights
immediately clear to traders, replacing generic technical names with descriptive,
educational terminology.
"""

from typing import Dict, Tuple
from datetime import datetime
import re


class EnhancedNamingScheme:
    """
    Provides trader-focused naming for visualization files and chart titles.
    
    Transforms technical chart names into descriptive, educational names that
    immediately communicate the chart's purpose and key insights to traders.
    """
    
    def __init__(self, use_enhanced_naming: bool = True):
        """
        Initialize the enhanced naming scheme.
        
        Args:
            use_enhanced_naming: Whether to use enhanced names or fall back to original
        """
        self.use_enhanced_naming = use_enhanced_naming
        
    def get_chart_names(self, chart_type: str, ticker: str, date_range: str, 
                       trade_source: str = "auto") -> Tuple[str, str]:
        """
        Get enhanced file name and chart title for a given chart type.
        
        Args:
            chart_type: Type of chart ("performance_summary", "trade_distribution", "trade_timeline")
            ticker: Ticker symbol
            date_range: Date range string
            trade_source: Trade data source being used
            
        Returns:
            Tuple of (enhanced_filename, enhanced_title)
        """
        if not self.use_enhanced_naming:
            return self._get_original_names(chart_type, ticker, date_range)
            
        return self._get_enhanced_names(chart_type, ticker, date_range, trade_source)
        
    def _get_original_names(self, chart_type: str, ticker: str, date_range: str) -> Tuple[str, str]:
        """Get original naming scheme for backward compatibility."""
        original_names = {
            "performance_summary": (
                f"{ticker}_performance_summary_{date_range}.png",
                f"{ticker} - Performance Dashboard"
            ),
            "trade_distribution": (
                f"{ticker}_trade_distribution_{date_range}.png", 
                f"{ticker} Trade Distribution Analysis - {date_range}"
            ),
            "trade_timeline": (
                f"{ticker}_trade_timeline_{date_range}.png",
                f"{ticker} Trade Timeline - {date_range}"
            )
        }
        
        return original_names.get(chart_type, (f"{ticker}_{chart_type}_{date_range}.png", f"{ticker} {chart_type}"))
        
    def _get_enhanced_names(self, chart_type: str, ticker: str, date_range: str, 
                           trade_source: str) -> Tuple[str, str]:
        """Get enhanced trader-focused naming scheme."""
        
        # Parse date range for better formatting
        formatted_period = self._format_date_range(date_range)
        trade_source_label = self._get_trade_source_label(trade_source)
        
        enhanced_names = {
            "performance_summary": (
                f"{ticker}_Equity_Curve_and_Trading_Metrics_{date_range}.png",
                f"{ticker} Trading Performance Analysis - Equity Curve, Drawdowns & Key Metrics\\n"
                f"Period: {formatted_period} | Data: {trade_source_label}"
            ),
            "trade_distribution": (
                f"{ticker}_Trade_Pattern_Analysis_{date_range}.png",
                f"{ticker} Trade Pattern & Risk Distribution Analysis\\n"
                f"Period: {formatted_period} | Data: {trade_source_label}"
            ),
            "trade_timeline": (
                f"{ticker}_Trade_Execution_Timeline_{date_range}.png", 
                f"{ticker} Trade Execution Timeline & Cumulative Performance\\n"
                f"Period: {formatted_period} | Data: {trade_source_label}"
            )
        }
        
        return enhanced_names.get(chart_type, self._get_original_names(chart_type, ticker, date_range))
        
    def _format_date_range(self, date_range: str) -> str:
        """Format date range into readable period description."""
        try:
            # Handle format like "2024-01-01_2024-06-01"
            if '_' in date_range:
                start_str, end_str = date_range.split('_')
                start_date = datetime.strptime(start_str, '%Y-%m-%d')
                end_date = datetime.strptime(end_str, '%Y-%m-%d')
                
                # Calculate period length
                period_days = (end_date - start_date).days
                
                if period_days <= 31:
                    return f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')} ({period_days} days)"
                elif period_days <= 92:
                    return f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')} (~{period_days//30} months)"
                else:
                    return f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')} ({period_days} days)"
            else:
                return date_range
        except (ValueError, AttributeError):
            # Fallback to original if parsing fails
            return date_range
            
    def _get_trade_source_label(self, trade_source: str) -> str:
        """Get human-readable label for trade data source."""
        source_labels = {
            "strategy_trades": "Strategy Signals", 
            "risk_approved_trades": "Risk-Approved Trades",
            "auto": "Auto-Selected Trades"
        }
        return source_labels.get(trade_source, trade_source.title())
        
    def get_enhanced_subtitle_text(self, chart_type: str) -> str:
        """
        Get educational subtitle text that explains what traders should focus on.
        
        Args:
            chart_type: Type of chart
            
        Returns:
            Educational subtitle text
        """
        if not self.use_enhanced_naming:
            return ""
            
        subtitles = {
            "performance_summary": 
                "🎯 Focus: Equity curve consistency, drawdown control, win rate vs. profit factor balance",
            "trade_distribution":
                "🔍 Focus: Trade size consistency, win/loss ratio patterns, position duration analysis", 
            "trade_timeline":
                "⏱️ Focus: Entry/exit timing, cumulative performance trends, trade clustering patterns"
        }
        
        return subtitles.get(chart_type, "")


class ChartMetadataEnhancer:
    """
    Enhances chart metadata and annotations for educational value.
    """
    
    def __init__(self, naming_scheme: EnhancedNamingScheme):
        self.naming_scheme = naming_scheme
        
    def get_chart_annotations(self, chart_type: str, metrics: Dict = None) -> Dict[str, str]:
        """
        Get educational annotations to add to charts.
        
        Args:
            chart_type: Type of chart
            metrics: Optional metrics dictionary for dynamic annotations
            
        Returns:
            Dictionary of annotation keys and text
        """
        if not self.naming_scheme.use_enhanced_naming:
            return {}
            
        base_annotations = {
            "performance_summary": {
                "equity_curve_tip": "💡 Smooth upward slope = consistent strategy",
                "drawdown_tip": "⚠️ Max drawdown shows worst-case scenario", 
                "distribution_tip": "📊 More wins than losses doesn't guarantee profitability"
            },
            "trade_distribution": {
                "win_rate_tip": "🎯 High win rate with small wins can be unprofitable",
                "size_tip": "📏 Consistent position sizing shows discipline",
                "duration_tip": "⏰ Trade duration affects strategy scalability"
            },
            "trade_timeline": {
                "timeline_tip": "📈 Look for clustering of wins/losses over time",
                "execution_tip": "🎯 Entry/exit marks show timing precision", 
                "trend_tip": "📊 Cumulative line shows strategy momentum"
            }
        }
        
        return base_annotations.get(chart_type, {})
