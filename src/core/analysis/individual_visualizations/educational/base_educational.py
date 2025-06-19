"""
Base class for educational visualizations.

This module provides the foundational class for all educational visualizations,
with built-in learning features, annotations, and improvement suggestions.
"""

import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from ...utils.base_visualizer import BaseVisualizer


class EducationalVisualization(BaseVisualizer):
    """
    Base class for educational visualizations with built-in learning features.
    
    This class extends BaseVisualizer to provide educational functionality
    specifically designed to help new traders learn critical trading concepts.    All educational visualizations default to enhanced naming for maximum
    educational value.
    """
    
    def __init__(self, output_dir: Optional[Path] = None, trade_source: str = "auto", 
                 use_enhanced_naming: bool = True):
        """
        Initialize educational visualization.
        
        Args:
            output_dir: Directory to save visualizations
            trade_source: Trade data source preference
            use_enhanced_naming: Whether to use enhanced naming (defaults to True for education)
        """
        # Educational visualizations default to enhanced naming for maximum learning value
        super().__init__(output_dir=output_dir, trade_source=trade_source, use_enhanced_naming=use_enhanced_naming)
        self.educational_annotations = True
        self.learning_level = "beginner"  # Can be "beginner", "intermediate", "advanced"
        
    def add_learning_annotations(self, fig: plt.Figure, chart_type: str, 
                               metrics: Dict[str, Any] = None) -> None:
        """
        Add educational annotations to help new traders learn.
        
        Args:
            fig: Matplotlib figure object
            chart_type: Type of educational chart
            metrics: Optional metrics for dynamic annotations
        """
        if not self.educational_annotations:
            return
            
        # Get educational annotations from enhanced naming if available
        if self.use_enhanced_naming and self.metadata_enhancer:
            annotations = self.metadata_enhancer.get_chart_annotations(chart_type, metrics)
            
            # Add annotations to the figure
            y_position = 0.98
            for key, annotation in annotations.items():
                fig.text(0.02, y_position, annotation, transform=fig.transFigure,
                        fontsize=9, style='italic', alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
                y_position -= 0.03
                
    def create_educational_summary(self, metrics: Dict[str, Any], chart_type: str) -> str:
        """
        Generate educational summary of key insights.
        
        Args:
            metrics: Dictionary of calculated metrics
            chart_type: Type of educational chart
            
        Returns:
            Educational summary text
        """
        summaries = {
            "risk_management": self._create_risk_management_summary(metrics),
            "trading_psychology": self._create_psychology_summary(metrics),
            "execution_quality": self._create_execution_summary(metrics),
            "learning_progress": self._create_progress_summary(metrics)
        }
        
        return summaries.get(chart_type, "Educational insights based on your trading data.")
        
    def get_improvement_suggestions(self, data: Dict[str, Any], chart_type: str) -> List[str]:
        """
        Provide specific improvement suggestions based on data analysis.
        
        Args:
            data: Trading data and metrics
            chart_type: Type of educational chart
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        if chart_type == "risk_management":
            suggestions.extend(self._get_risk_management_suggestions(data))
        elif chart_type == "trading_psychology":
            suggestions.extend(self._get_psychology_suggestions(data))
        elif chart_type == "execution_quality":
            suggestions.extend(self._get_execution_suggestions(data))
        elif chart_type == "learning_progress":
            suggestions.extend(self._get_progress_suggestions(data))
            
        return suggestions
        
    def _create_risk_management_summary(self, metrics: Dict[str, Any]) -> str:
        """Create educational summary for risk management."""
        avg_position_size = metrics.get('avg_position_size_pct', 0)
        risk_reward_ratio = metrics.get('avg_risk_reward_ratio', 0)
        stop_loss_usage = metrics.get('stop_loss_usage_pct', 0)
        
        return f"""
🛡️ RISK MANAGEMENT EDUCATION:

Your average position size is {avg_position_size:.1f}% of account value.
Risk/Reward ratio averages {risk_reward_ratio:.2f}:1.
Stop losses used in {stop_loss_usage:.0f}% of trades.

KEY LEARNING: Proper position sizing (1-2% risk per trade) and consistent stop loss usage 
are the foundation of successful trading. Focus on preserving capital over maximizing profits.
        """.strip()
        
    def _create_psychology_summary(self, metrics: Dict[str, Any]) -> str:
        """Create educational summary for trading psychology."""
        max_winning_streak = metrics.get('max_winning_streak', 0)
        max_losing_streak = metrics.get('max_losing_streak', 0)
        drawdown_recovery_avg = metrics.get('avg_drawdown_recovery_days', 0)
        
        return f"""
🧠 TRADING PSYCHOLOGY EDUCATION:

Longest winning streak: {max_winning_streak} trades
Longest losing streak: {max_losing_streak} trades  
Average drawdown recovery: {drawdown_recovery_avg:.1f} days

KEY LEARNING: Streaks are normal in trading. The key is maintaining discipline during 
both winning and losing periods. Emotional control determines long-term success.
        """.strip()
        
    def _create_execution_summary(self, metrics: Dict[str, Any]) -> str:
        """Create educational summary for execution quality."""
        avg_slippage = metrics.get('avg_slippage_pct', 0)
        timing_score = metrics.get('timing_quality_score', 0)
        missed_opportunities = metrics.get('missed_trades_count', 0)
        
        return f"""
⚡ EXECUTION QUALITY EDUCATION:

Average slippage: {avg_slippage:.3f}%
Timing quality score: {timing_score:.1f}/10
Missed trading opportunities: {missed_opportunities}

KEY LEARNING: Quality execution can significantly impact returns. Focus on timing, 
minimize slippage, and don't let good setups pass by due to hesitation.
        """.strip()
        
    def _create_progress_summary(self, metrics: Dict[str, Any]) -> str:
        """Create educational summary for learning progress."""
        skill_improvement = metrics.get('skill_improvement_pct', 0)
        mistake_reduction = metrics.get('mistake_reduction_pct', 0)
        consistency_score = metrics.get('consistency_score', 0)
        
        return f"""
📈 LEARNING PROGRESS EDUCATION:

Skill improvement: {skill_improvement:.1f}% over period
Mistake reduction: {mistake_reduction:.1f}% 
Consistency score: {consistency_score:.1f}/10

KEY LEARNING: Trading is a skill that improves with practice and self-awareness.
Track your progress, learn from mistakes, and focus on consistent application of rules.
        """.strip()
        
    def _get_risk_management_suggestions(self, data: Dict[str, Any]) -> List[str]:
        """Get improvement suggestions for risk management."""
        suggestions = []
        
        trades = data.get('active_trades')
        if trades is not None and not trades.empty:
            # Check position sizing consistency
            if 'Position_Size_Pct' in trades.columns:
                pos_sizes = trades['Position_Size_Pct'].dropna()
                if not pos_sizes.empty:
                    if pos_sizes.std() > pos_sizes.mean() * 0.5:
                        suggestions.append("🎯 Consider more consistent position sizing for better risk control")
                    if pos_sizes.mean() > 5:
                        suggestions.append("⚠️ Your average position size may be too large - consider 1-2% risk per trade")
                        
            # Check risk/reward ratios
            if 'Risk_Reward_Ratio' in trades.columns:
                rr_ratios = trades['Risk_Reward_Ratio'].dropna()
                if not rr_ratios.empty and rr_ratios.mean() < 1.5:
                    suggestions.append("📊 Try to achieve risk/reward ratios of at least 1.5:1 or higher")
                    
            # Check stop loss usage
            if 'Stop_Loss_Used' in trades.columns:
                stop_usage = trades['Stop_Loss_Used'].mean() * 100
                if stop_usage < 80:
                    suggestions.append("🛡️ Consider using stop losses more consistently to protect capital")
                    
        return suggestions
        
    def _get_psychology_suggestions(self, data: Dict[str, Any]) -> List[str]:
        """Get improvement suggestions for trading psychology."""
        suggestions = []
        
        trades = data.get('active_trades')
        if trades is not None and not trades.empty:
            # Check for revenge trading patterns
            if 'Profit (%)' in trades.columns:
                profits = trades['Profit (%)']
                # Look for large position sizes after losses
                for i in range(1, len(profits)):
                    if profits.iloc[i-1] < -2 and profits.iloc[i] > 5:  # Loss followed by big win
                        suggestions.append("🧠 Watch for revenge trading - avoid increasing position size after losses")
                        break
                        
            # Check drawdown recovery
            if len(profits) > 10:
                cumulative = profits.cumsum()
                peak = cumulative.expanding().max()
                drawdown = (cumulative - peak) / peak * 100
                max_dd = drawdown.min()
                if max_dd < -15:
                    suggestions.append("💪 Consider reducing position sizes during drawdown periods to protect psychology")
                    
        return suggestions
        
    def _get_execution_suggestions(self, data: Dict[str, Any]) -> List[str]:
        """Get improvement suggestions for execution quality."""
        suggestions = []
        
        trades = data.get('active_trades')
        if trades is not None and not trades.empty:
            # Check timing patterns
            if 'Entry_Time' in trades.columns and 'Exit_Time' in trades.columns:
                # Suggest timing improvements based on patterns
                suggestions.append("⏰ Consider analyzing your best performing entry times to improve timing")
                
            # Check for missed opportunities
            if len(trades) < 10:  # Assuming monthly data should have more trades
                suggestions.append("🎯 Look for missed trading opportunities - you might be too selective")
                
        return suggestions
        
    def _get_progress_suggestions(self, data: Dict[str, Any]) -> List[str]:
        """Get improvement suggestions for learning progress."""
        suggestions = []
        
        suggestions.extend([
            "📚 Keep a trading journal to track decisions and emotions",
            "📊 Review your worst trades weekly to identify improvement areas",
            "🎯 Focus on one skill at a time rather than trying to improve everything",
            "📈 Set specific, measurable goals for your trading development"
        ])
        
        return suggestions
        
    def set_learning_level(self, level: str) -> None:
        """
        Set the learning level for appropriate educational content.
        
        Args:
            level: Learning level ("beginner", "intermediate", "advanced")
        """
        if level in ["beginner", "intermediate", "advanced"]:
            self.learning_level = level
        else:
            self.logger.warning(f"Invalid learning level: {level}. Using 'beginner'.")
            self.learning_level = "beginner"
