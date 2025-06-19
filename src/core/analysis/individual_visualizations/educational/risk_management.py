"""
Risk Management Dashboard for Individual Tickers.

This module provides comprehensive risk management analysis and education for new traders,
focusing on position sizing, risk/reward ratios, stop loss effectiveness, and overall
risk control practices.
"""

import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from .base_educational import EducationalVisualization


class RiskManagementDashboard(EducationalVisualization):
    """
    Creates comprehensive risk management dashboard for individual tickers.
    
    This educational visualization teaches new traders critical risk management
    concepts including position sizing, risk/reward ratios, stop loss usage,
    and overall risk control practices.
    """
    
    def __init__(self, trade_source: str = "auto", use_enhanced_naming: bool = True):
        """
        Initialize the risk management dashboard.
        
        Args:
            trade_source: Trade data source preference
            use_enhanced_naming: Whether to use enhanced trader-focused naming scheme
        """
        super().__init__(trade_source=trade_source, use_enhanced_naming=use_enhanced_naming)
        
    def create(self, ticker_data: Dict[str, Any], ticker: str, 
               date_range: str, output_dir: Path) -> Optional[Path]:
        """
        Create comprehensive risk management dashboard.
        
        Args:
            ticker_data: Dictionary containing all ticker data
            ticker: Ticker symbol
            date_range: Date range string
            output_dir: Directory to save the chart
            
        Returns:
            Path to saved chart file, or None if creation failed
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'{ticker} Risk Management Analysis - {date_range}', fontsize=16, fontweight='bold')
            
            active_trades = ticker_data.get('active_trades')
            analytics = ticker_data.get('analytics', {})
            
            if active_trades is None or active_trades.empty:
                # Create placeholder with educational message
                for ax in axes.flat:
                    ax.text(0.5, 0.5, 
                           'No trade data available for risk analysis\\n\\n'
                           '🛡️ Risk management is the foundation of successful trading.\\n'
                           'Focus on: Position sizing, Stop losses, Risk/Reward ratios',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow'))
                    ax.set_title('Risk Management Education')
                
                # Apply enhanced naming for empty data
                if self.use_enhanced_naming and self.naming_scheme:
                    filename, enhanced_title = self.naming_scheme.get_chart_names(
                        "risk_management", ticker, date_range, self.trade_source
                    )
                    fig.suptitle(enhanced_title, fontsize=14, fontweight='bold', y=0.95)
                else:
                    filename = f"{ticker}_risk_management_{date_range}.png"
                
                plt.tight_layout()
                filepath = output_dir / filename
                return self._safe_save_plot(fig, filepath)
            
            # Calculate risk management metrics
            risk_metrics = self._calculate_risk_metrics(active_trades, analytics)
            
            # 1. Position Sizing Analysis
            self._create_position_sizing_analysis(axes[0, 0], active_trades, risk_metrics)
            
            # 2. Risk/Reward Ratio Analysis
            self._create_risk_reward_analysis(axes[0, 1], active_trades, risk_metrics)
            
            # 3. Stop Loss Effectiveness
            self._create_stop_loss_analysis(axes[0, 2], active_trades, risk_metrics)
            
            # 4. Risk Distribution Over Time
            self._create_risk_timeline(axes[1, 0], active_trades, risk_metrics)
            
            # 5. Portfolio Risk Heatmap
            self._create_portfolio_risk_heatmap(axes[1, 1], active_trades, risk_metrics)
            
            # 6. Risk Management Scorecard
            self._create_risk_scorecard(axes[1, 2], risk_metrics)
            
            # Apply enhanced naming if enabled
            if self.use_enhanced_naming and self.naming_scheme:
                filename, enhanced_title = self.naming_scheme.get_chart_names(
                    "risk_management", ticker, date_range, self.trade_source
                )
                fig.suptitle(enhanced_title, fontsize=14, fontweight='bold', y=0.95)
                
                # Add educational subtitle
                subtitle = "🛡️ Focus: Position sizing discipline, risk/reward optimization, stop loss effectiveness"
                fig.text(0.5, 0.02, subtitle, ha='center', va='bottom', 
                        fontsize=10, style='italic', wrap=True)
            else:
                filename = f"{ticker}_risk_management_{date_range}.png"
            
            # Add educational annotations
            self.add_learning_annotations(fig, "risk_management", risk_metrics)
            
            plt.tight_layout()
            
            # Save the dashboard
            filepath = output_dir / filename
            return self._safe_save_plot(fig, filepath)
            
        except Exception as e:
            self.logger.error(f"Error creating risk management dashboard for {ticker}: {e}")
            return None
            
    def _calculate_risk_metrics(self, active_trades: pd.DataFrame, 
                               analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk management metrics."""
        metrics = {}
        
        try:
            # Basic trade metrics
            metrics['total_trades'] = len(active_trades)
            
            # Position sizing metrics
            if 'Position_Size_Pct' in active_trades.columns:
                pos_sizes = active_trades['Position_Size_Pct'].dropna()
                metrics['avg_position_size_pct'] = pos_sizes.mean()
                metrics['position_size_std'] = pos_sizes.std()
                metrics['max_position_size_pct'] = pos_sizes.max()
                metrics['position_size_consistency'] = 1 - (pos_sizes.std() / pos_sizes.mean() if pos_sizes.mean() > 0 else 0)
            else:
                # Estimate position sizing if not available
                metrics['avg_position_size_pct'] = 2.0  # Default assumption
                metrics['position_size_std'] = 0.5
                metrics['max_position_size_pct'] = 3.0
                metrics['position_size_consistency'] = 0.75
                
            # Risk/Reward metrics
            if 'Risk_Reward_Ratio' in active_trades.columns:
                rr_ratios = active_trades['Risk_Reward_Ratio'].dropna()
                metrics['avg_risk_reward_ratio'] = rr_ratios.mean()
                metrics['median_risk_reward_ratio'] = rr_ratios.median()
                metrics['rr_ratio_consistency'] = (rr_ratios >= 1.0).mean()
            else:
                # Calculate from profit/loss if available
                if 'Profit (%)' in active_trades.columns:
                    profits = active_trades['Profit (%)'].dropna()
                    winning_trades = profits[profits > 0]
                    losing_trades = profits[profits < 0]
                    
                    if len(winning_trades) > 0 and len(losing_trades) > 0:
                        avg_win = winning_trades.mean()
                        avg_loss = abs(losing_trades.mean())
                        metrics['avg_risk_reward_ratio'] = avg_win / avg_loss if avg_loss > 0 else 1.0
                    else:
                        metrics['avg_risk_reward_ratio'] = 1.0
                        
                    metrics['median_risk_reward_ratio'] = metrics['avg_risk_reward_ratio']
                    metrics['rr_ratio_consistency'] = 0.6
                    
            # Stop loss metrics
            if 'Stop_Loss_Used' in active_trades.columns:
                stop_loss_usage = active_trades['Stop_Loss_Used']
                metrics['stop_loss_usage_pct'] = stop_loss_usage.mean() * 100
                metrics['stop_loss_effectiveness'] = self._calculate_stop_loss_effectiveness(active_trades)
            else:
                # Estimate based on trade outcomes
                if 'Profit (%)' in active_trades.columns:
                    profits = active_trades['Profit (%)']
                    # Assume stop losses were used if losses are limited
                    large_losses = (profits < -5).sum()
                    total_losses = (profits < 0).sum()
                    metrics['stop_loss_usage_pct'] = ((total_losses - large_losses) / total_losses * 100 
                                                    if total_losses > 0 else 80)
                    metrics['stop_loss_effectiveness'] = 0.7
                else:
                    metrics['stop_loss_usage_pct'] = 50
                    metrics['stop_loss_effectiveness'] = 0.5
                    
            # Overall risk score (0-10 scale)
            metrics['overall_risk_score'] = self._calculate_overall_risk_score(metrics)
            
            # Risk trends
            metrics['risk_trend'] = self._calculate_risk_trend(active_trades)
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            # Provide default metrics
            metrics = {
                'total_trades': len(active_trades) if active_trades is not None else 0,
                'avg_position_size_pct': 2.0,
                'position_size_std': 0.5,
                'max_position_size_pct': 3.0,
                'position_size_consistency': 0.75,
                'avg_risk_reward_ratio': 1.2,
                'median_risk_reward_ratio': 1.1,
                'rr_ratio_consistency': 0.6,
                'stop_loss_usage_pct': 70,
                'stop_loss_effectiveness': 0.7,
                'overall_risk_score': 6.5,
                'risk_trend': 'stable'
            }
            
        return metrics
        
    def _calculate_stop_loss_effectiveness(self, active_trades: pd.DataFrame) -> float:
        """Calculate how effective stop losses are at limiting losses."""
        try:
            if 'Stop_Loss_Used' in active_trades.columns and 'Profit (%)' in active_trades.columns:
                stop_loss_trades = active_trades[active_trades['Stop_Loss_Used'] == True]
                no_stop_trades = active_trades[active_trades['Stop_Loss_Used'] == False]
                
                if len(stop_loss_trades) > 0 and len(no_stop_trades) > 0:
                    stop_loss_avg_loss = stop_loss_trades[stop_loss_trades['Profit (%)'] < 0]['Profit (%)'].mean()
                    no_stop_avg_loss = no_stop_trades[no_stop_trades['Profit (%)'] < 0]['Profit (%)'].mean()
                    
                    # Effectiveness = how much stop losses limit losses compared to no stop
                    if pd.notna(stop_loss_avg_loss) and pd.notna(no_stop_avg_loss) and no_stop_avg_loss < 0:
                        effectiveness = 1 - (abs(stop_loss_avg_loss) / abs(no_stop_avg_loss))
                        return max(0, min(1, effectiveness))
                        
        except Exception as e:
            self.logger.debug(f"Error calculating stop loss effectiveness: {e}")
            
        return 0.7  # Default moderate effectiveness
        
    def _calculate_overall_risk_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall risk management score (0-10 scale)."""
        score = 0
        
        # Position sizing score (0-3 points)
        pos_size = metrics.get('avg_position_size_pct', 5)
        if pos_size <= 2:
            score += 3
        elif pos_size <= 3:
            score += 2
        elif pos_size <= 5:
            score += 1
            
        # Risk/reward score (0-3 points)
        rr_ratio = metrics.get('avg_risk_reward_ratio', 1)
        if rr_ratio >= 2:
            score += 3
        elif rr_ratio >= 1.5:
            score += 2
        elif rr_ratio >= 1:
            score += 1
            
        # Stop loss score (0-2 points)
        stop_usage = metrics.get('stop_loss_usage_pct', 0)
        if stop_usage >= 90:
            score += 2
        elif stop_usage >= 70:
            score += 1
            
        # Consistency score (0-2 points)
        consistency = metrics.get('position_size_consistency', 0)
        if consistency >= 0.8:
            score += 2
        elif consistency >= 0.6:
            score += 1
            
        return min(10, score)
        
    def _calculate_risk_trend(self, active_trades: pd.DataFrame) -> str:
        """Calculate if risk management is improving, declining, or stable."""
        try:
            if len(active_trades) < 10:
                return 'insufficient_data'
                
            # Split trades into first and second half
            mid_point = len(active_trades) // 2
            first_half = active_trades.iloc[:mid_point]
            second_half = active_trades.iloc[mid_point:]
            
            # Compare risk metrics between halves
            first_half_risk = self._calculate_period_risk_score(first_half)
            second_half_risk = self._calculate_period_risk_score(second_half)
            
            diff = second_half_risk - first_half_risk
            
            if diff > 0.5:
                return 'improving'
            elif diff < -0.5:
                return 'declining'
            else:
                return 'stable'
                
        except Exception:
            return 'stable'
            
    def _calculate_period_risk_score(self, trades: pd.DataFrame) -> float:
        """Calculate risk score for a period of trades."""
        if trades.empty:
            return 5.0
            
        score = 5.0  # Base score
        
        # Check position sizing if available
        if 'Position_Size_Pct' in trades.columns:
            avg_pos_size = trades['Position_Size_Pct'].mean()
            if avg_pos_size <= 2:
                score += 1
            elif avg_pos_size > 5:
                score -= 1
                
        # Check stop loss usage if available
        if 'Stop_Loss_Used' in trades.columns:
            stop_usage = trades['Stop_Loss_Used'].mean()
            if stop_usage > 0.8:
                score += 1
            elif stop_usage < 0.5:
                score -= 1
                
        return max(0, min(10, score))
        
    def _create_position_sizing_analysis(self, ax, active_trades: pd.DataFrame, 
                                       risk_metrics: Dict[str, Any]) -> None:
        """Create position sizing analysis visualization."""
        try:
            if 'Position_Size_Pct' in active_trades.columns:
                pos_sizes = active_trades['Position_Size_Pct'].dropna()
                
                if not pos_sizes.empty:
                    # Create histogram of position sizes
                    ax.hist(pos_sizes, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                    
                    # Add optimal range shading
                    ax.axvspan(1, 2, alpha=0.3, color='green', label='Optimal Range (1-2%)')
                    ax.axvspan(2, 3, alpha=0.2, color='yellow', label='Acceptable Range (2-3%)')
                    ax.axvspan(3, 10, alpha=0.2, color='red', label='High Risk (>3%)')
                    
                    # Add average line
                    avg_size = pos_sizes.mean()
                    ax.axvline(avg_size, color='red', linestyle='--', linewidth=2, 
                              label=f'Your Average: {avg_size:.1f}%')
                    
                    ax.set_title('Position Sizing Distribution')
                    ax.set_xlabel('Position Size (% of Account)')
                    ax.set_ylabel('Number of Trades')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    # Add educational note
                    ax.text(0.02, 0.98, '💡 Tip: Risk 1-2% per trade for optimal capital preservation',
                           transform=ax.transAxes, fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No position sizing data available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Position Sizing (No Data)')
            else:
                # Create educational placeholder
                ax.text(0.5, 0.7, 'Position Sizing Education', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14, fontweight='bold')
                ax.text(0.5, 0.5, '🎯 Optimal Position Size: 1-2% of account value per trade\\n'
                                  '⚠️ Never risk more than 5% on a single trade\\n'
                                  '💡 Consistent sizing = Better risk control',
                       ha='center', va='center', transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
                ax.set_title('Position Sizing Education')
                
        except Exception as e:
            self.logger.error(f"Error creating position sizing analysis: {e}")
            ax.text(0.5, 0.5, 'Error creating position sizing analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Position Sizing (Error)')
            
    def _create_risk_reward_analysis(self, ax, active_trades: pd.DataFrame, 
                                   risk_metrics: Dict[str, Any]) -> None:
        """Create risk/reward ratio analysis visualization."""
        try:
            if 'Risk_Reward_Ratio' in active_trades.columns:
                rr_ratios = active_trades['Risk_Reward_Ratio'].dropna()
                
                if not rr_ratios.empty:
                    # Create scatter plot of risk/reward ratios
                    colors = ['green' if ratio >= 1.5 else 'orange' if ratio >= 1.0 else 'red' 
                             for ratio in rr_ratios]
                    
                    ax.scatter(range(len(rr_ratios)), rr_ratios, c=colors, alpha=0.7, s=50)
                    
                    # Add reference lines
                    ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Break-even (1:1)')
                    ax.axhline(y=1.5, color='green', linestyle='--', alpha=0.7, label='Good (1.5:1)')
                    ax.axhline(y=2.0, color='blue', linestyle='--', alpha=0.7, label='Excellent (2:1)')
                    
                    # Add average line
                    avg_rr = rr_ratios.mean()
                    ax.axhline(y=avg_rr, color='red', linestyle='-', linewidth=2,
                              label=f'Your Average: {avg_rr:.2f}:1')
                    
                    ax.set_title('Risk/Reward Ratio Analysis')
                    ax.set_xlabel('Trade Number')
                    ax.set_ylabel('Risk/Reward Ratio')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    # Add educational note
                    good_trades = (rr_ratios >= 1.5).sum()
                    total_trades = len(rr_ratios)
                    pct_good = (good_trades / total_trades) * 100
                    
                    ax.text(0.02, 0.98, f'📊 {pct_good:.0f}% of trades have good R/R (≥1.5:1)',
                           transform=ax.transAxes, fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No risk/reward data available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Risk/Reward Ratios (No Data)')
            else:
                # Create educational placeholder
                ax.text(0.5, 0.7, 'Risk/Reward Education', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14, fontweight='bold')
                ax.text(0.5, 0.5, '🎯 Target Risk/Reward: At least 1.5:1\\n'
                                  '💰 Better R/R = Profitable with <50% win rate\\n'
                                  '📈 Focus on reward potential vs. risk taken',
                       ha='center', va='center', transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
                ax.set_title('Risk/Reward Education')
                
        except Exception as e:
            self.logger.error(f"Error creating risk/reward analysis: {e}")
            ax.text(0.5, 0.5, 'Error creating risk/reward analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Risk/Reward (Error)')
            
    def _create_stop_loss_analysis(self, ax, active_trades: pd.DataFrame, 
                                 risk_metrics: Dict[str, Any]) -> None:
        """Create stop loss effectiveness analysis."""
        try:
            if 'Stop_Loss_Used' in active_trades.columns:
                stop_usage = active_trades['Stop_Loss_Used']
                usage_pct = stop_usage.mean() * 100
                
                # Create pie chart
                sizes = [usage_pct, 100 - usage_pct]
                labels = [f'Used Stop Loss\\n({usage_pct:.0f}%)', f'No Stop Loss\\n({100-usage_pct:.0f}%)']
                colors = ['lightgreen', 'lightcoral']
                explode = (0.1, 0) if usage_pct < 80 else (0, 0.1)
                
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                                 explode=explode, autopct='%1.0f%%',
                                                 startangle=90, textprops={'fontsize': 10})
                
                ax.set_title('Stop Loss Usage')
                
                # Add educational assessment
                if usage_pct >= 90:
                    assessment = '🛡️ Excellent discipline!'
                    color = 'lightgreen'
                elif usage_pct >= 70:
                    assessment = '✅ Good risk control'
                    color = 'lightyellow'
                else:
                    assessment = '⚠️ Improve stop loss usage'
                    color = 'lightcoral'
                    
                ax.text(0.5, -1.3, assessment, ha='center', va='center', 
                       transform=ax.transAxes, fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
                
            else:
                # Create educational placeholder
                ax.text(0.5, 0.7, 'Stop Loss Education', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14, fontweight='bold')
                ax.text(0.5, 0.5, '🛡️ Always use stop losses to limit losses\\n'
                                  '📍 Set stops before entering trades\\n'
                                  '💡 Stops preserve capital for future opportunities',
                       ha='center', va='center', transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
                ax.set_title('Stop Loss Education')
                
        except Exception as e:
            self.logger.error(f"Error creating stop loss analysis: {e}")
            ax.text(0.5, 0.5, 'Error creating stop loss analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Stop Loss (Error)')
            
    def _create_risk_timeline(self, ax, active_trades: pd.DataFrame, 
                            risk_metrics: Dict[str, Any]) -> None:
        """Create risk evolution over time."""
        try:
            if len(active_trades) >= 5:
                # Calculate rolling risk metrics
                window_size = max(5, len(active_trades) // 10)
                
                if 'Position_Size_Pct' in active_trades.columns:
                    rolling_pos_size = active_trades['Position_Size_Pct'].rolling(window=window_size).mean()
                    trade_numbers = range(len(rolling_pos_size))
                    
                    ax.plot(trade_numbers, rolling_pos_size, marker='o', linewidth=2, 
                           markersize=4, alpha=0.8, label='Avg Position Size')
                    
                    # Add risk zones
                    ax.axhspan(0, 2, alpha=0.2, color='green', label='Safe Zone (0-2%)')
                    ax.axhspan(2, 3, alpha=0.2, color='yellow', label='Caution Zone (2-3%)')
                    ax.axhspan(3, 10, alpha=0.2, color='red', label='Danger Zone (>3%)')
                    
                    ax.set_title('Risk Evolution Over Time')
                    ax.set_xlabel('Trade Number')
                    ax.set_ylabel('Position Size (%)')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    # Trend analysis
                    trend = risk_metrics.get('risk_trend', 'stable')
                    trend_colors = {'improving': 'green', 'stable': 'blue', 'declining': 'red'}
                    ax.text(0.02, 0.98, f'Trend: {trend.title()}', 
                           transform=ax.transAxes, fontsize=10, verticalalignment='top',
                           color=trend_colors.get(trend, 'black'), fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'Insufficient data for risk timeline', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Risk Timeline (Insufficient Data)')
            else:
                ax.text(0.5, 0.5, 'Need more trades for timeline analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Risk Timeline (Need More Data)')
                
        except Exception as e:
            self.logger.error(f"Error creating risk timeline: {e}")
            ax.text(0.5, 0.5, 'Error creating risk timeline', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Risk Timeline (Error)')
            
    def _create_portfolio_risk_heatmap(self, ax, active_trades: pd.DataFrame, 
                                     risk_metrics: Dict[str, Any]) -> None:
        """Create portfolio risk distribution heatmap."""
        try:
            # Create a simplified risk heatmap based on trade timing and size
            if len(active_trades) >= 10:
                # Group trades by week/period and calculate risk exposure
                if 'Entry_Date' in active_trades.columns:
                    trades_copy = active_trades.copy()
                    trades_copy['Entry_Date'] = pd.to_datetime(trades_copy['Entry_Date'])
                    trades_copy['Week'] = trades_copy['Entry_Date'].dt.isocalendar().week
                    
                    # Calculate weekly risk exposure
                    weekly_risk = trades_copy.groupby('Week').agg({
                        'Position_Size_Pct': 'sum' if 'Position_Size_Pct' in trades_copy.columns else 'count'
                    }).reset_index()
                    
                    # Create simple heatmap representation
                    weeks = weekly_risk['Week'].values
                    risks = weekly_risk.iloc[:, 1].values
                    
                    # Reshape into grid for heatmap
                    grid_size = int(np.ceil(np.sqrt(len(weeks))))
                    risk_grid = np.zeros((grid_size, grid_size))
                    
                    for i, risk in enumerate(risks[:grid_size*grid_size]):
                        row, col = divmod(i, grid_size)
                        risk_grid[row, col] = risk
                        
                    # Create heatmap
                    im = ax.imshow(risk_grid, cmap='RdYlGn_r', alpha=0.8)
                    ax.set_title('Portfolio Risk Distribution\\n(Darker = Higher Risk)')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, shrink=0.6)
                    
                    # Remove ticks for cleaner look
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                else:
                    # Simplified risk distribution chart
                    risk_categories = ['Low Risk\\n(1-2%)', 'Medium Risk\\n(2-3%)', 'High Risk\\n(>3%)']
                    
                    if 'Position_Size_Pct' in active_trades.columns:
                        pos_sizes = active_trades['Position_Size_Pct']
                        low_risk = (pos_sizes <= 2).sum()
                        med_risk = ((pos_sizes > 2) & (pos_sizes <= 3)).sum()
                        high_risk = (pos_sizes > 3).sum()
                        risk_counts = [low_risk, med_risk, high_risk]
                    else:
                        # Default distribution for educational purposes
                        risk_counts = [60, 30, 10]
                        
                    colors = ['green', 'orange', 'red']
                    bars = ax.bar(risk_categories, risk_counts, color=colors, alpha=0.7)
                    
                    ax.set_title('Risk Distribution by Category')
                    ax.set_ylabel('Number of Trades')
                    
                    # Add value labels on bars
                    for bar, count in zip(bars, risk_counts):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{count}', ha='center', va='bottom', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Portfolio Risk Distribution\\n\\nNeed more trades for analysis', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('Portfolio Risk Distribution')
                
        except Exception as e:
            self.logger.error(f"Error creating risk heatmap: {e}")
            ax.text(0.5, 0.5, 'Error creating risk heatmap', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Risk Heatmap (Error)')
            
    def _create_risk_scorecard(self, ax, risk_metrics: Dict[str, Any]) -> None:
        """Create risk management scorecard with grades."""
        try:
            # Define scoring criteria and get grades
            criteria = {
                'Position Sizing': self._grade_position_sizing(risk_metrics),
                'Risk/Reward': self._grade_risk_reward(risk_metrics),
                'Stop Loss Usage': self._grade_stop_loss(risk_metrics),
                'Consistency': self._grade_consistency(risk_metrics),
                'Overall Risk Mgmt': self._grade_overall_risk(risk_metrics)
            }
            
            # Create scorecard visualization
            y_positions = list(range(len(criteria)))
            grade_colors = {'A': 'green', 'B': 'lightgreen', 'C': 'yellow', 
                           'D': 'orange', 'F': 'red'}
            
            ax.set_xlim(0, 10)
            ax.set_ylim(-0.5, len(criteria) - 0.5)
            
            for i, (criterion, grade) in enumerate(criteria.items()):
                y_pos = len(criteria) - 1 - i
                
                # Draw grade box
                color = grade_colors.get(grade, 'gray')
                rect = mpatches.Rectangle((8, y_pos - 0.3), 1.5, 0.6, 
                                        facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                
                # Add criterion name
                ax.text(0.5, y_pos, criterion, va='center', ha='left', fontsize=11, fontweight='bold')
                
                # Add grade
                ax.text(8.75, y_pos, grade, va='center', ha='center', 
                       fontsize=14, fontweight='bold', color='white' if grade == 'F' else 'black')
                
            ax.set_title('Risk Management Scorecard', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add overall score
            overall_score = risk_metrics.get('overall_risk_score', 5)
            ax.text(5, -0.8, f'Overall Score: {overall_score:.1f}/10', 
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)
                
        except Exception as e:
            self.logger.error(f"Error creating risk scorecard: {e}")
            ax.text(0.5, 0.5, 'Error creating risk scorecard', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Risk Scorecard (Error)')
            
    def _grade_position_sizing(self, metrics: Dict[str, Any]) -> str:
        """Grade position sizing practices."""
        avg_size = metrics.get('avg_position_size_pct', 5)
        if avg_size <= 2:
            return 'A'
        elif avg_size <= 3:
            return 'B'
        elif avg_size <= 4:
            return 'C'
        elif avg_size <= 5:
            return 'D'
        else:
            return 'F'
            
    def _grade_risk_reward(self, metrics: Dict[str, Any]) -> str:
        """Grade risk/reward ratio practices."""
        avg_rr = metrics.get('avg_risk_reward_ratio', 1)
        if avg_rr >= 2:
            return 'A'
        elif avg_rr >= 1.5:
            return 'B'
        elif avg_rr >= 1.2:
            return 'C'
        elif avg_rr >= 1:
            return 'D'
        else:
            return 'F'
            
    def _grade_stop_loss(self, metrics: Dict[str, Any]) -> str:
        """Grade stop loss usage."""
        usage = metrics.get('stop_loss_usage_pct', 0)
        if usage >= 90:
            return 'A'
        elif usage >= 75:
            return 'B'
        elif usage >= 60:
            return 'C'
        elif usage >= 40:
            return 'D'
        else:
            return 'F'
            
    def _grade_consistency(self, metrics: Dict[str, Any]) -> str:
        """Grade consistency in risk management."""
        consistency = metrics.get('position_size_consistency', 0)
        if consistency >= 0.9:
            return 'A'
        elif consistency >= 0.8:
            return 'B'
        elif consistency >= 0.7:
            return 'C'
        elif consistency >= 0.6:
            return 'D'
        else:
            return 'F'
            
    def _grade_overall_risk(self, metrics: Dict[str, Any]) -> str:
        """Grade overall risk management."""
        score = metrics.get('overall_risk_score', 5)
        if score >= 9:
            return 'A'
        elif score >= 7:
            return 'B'
        elif score >= 5:
            return 'C'
        elif score >= 3:
            return 'D'
        else:
            return 'F'
