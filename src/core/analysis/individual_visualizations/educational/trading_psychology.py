"""
Trading Psychology Insights Visualization
Educational visualization for new traders to understand emotional and psychological patterns.

This module creates comprehensive analysis of trading psychology patterns including:
- Winning/losing streak analysis
- Drawdown recovery patterns  
- Emotional trading indicators
- Performance vs market conditions
- Consistency scoring

Educational Focus:
- Understanding emotional discipline in trading
- Recognizing psychological patterns that impact performance
- Learning to manage drawdowns effectively
- Identifying signs of emotional trading (revenge trading, overconfidence)
- Building consistency in trading approach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path

from .base_educational import EducationalVisualization
from ...utils.chart_styling import ChartStyler
from ...utils.chart_styling import ChartStyler


class TradingPsychologyVisualizer(EducationalVisualization):
    """
    Creates trading psychology insights visualization focused on emotional patterns,
    streaks, drawdowns, and psychological trading indicators.
    """
    
    def __init__(self, trade_source: str = "auto", use_enhanced_naming: bool = True):
        """
        Initialize the trading psychology visualizer.
        
        Args:
            trade_source: Trade data source preference
            use_enhanced_naming: Whether to use enhanced trader-focused naming scheme
        """
        super().__init__(trade_source=trade_source, use_enhanced_naming=use_enhanced_naming)
        self.logger = logging.getLogger(__name__)
        self.chart_styler = ChartStyler()
        
    def apply_style_config(self, style_config: Optional[Dict] = None):
        """Apply styling configuration to charts."""
        if style_config is None:
            self.chart_styler.apply_global_style()
        else:
            # Apply custom styling if provided
            for key, value in style_config.items():
                plt.rcParams[key] = value
    
    def get_color_palette(self) -> Dict[str, str]:
        """Get color palette for consistent chart styling."""
        base_palette = self.chart_styler.get_color_palette()
        # Add educational-specific colors
        base_palette.update({
            'primary': '#2E86C1',      # Professional blue
            'secondary': '#28B463',     # Success green  
            'warning': '#F39C12',       # Warning orange
            'caution': '#E74C3C',       # Alert red
            'info': '#8E44AD',          # Information purple
            'background': '#F8F9FA'     # Light background
        })
        return base_palette
        
    def get_visualization_info(self) -> Dict[str, str]:
        """Get metadata about this visualization."""
        return {
            'name': 'Trading Psychology Insights',
            'description': 'Analysis of emotional patterns, streaks, and psychological trading indicators',
            'educational_focus': 'Understanding trading psychology, emotional discipline, and consistency patterns',
            'target_audience': 'New traders learning to manage emotions and build discipline',
            'key_concepts': 'Winning/losing streaks, drawdown recovery, emotional trading signs, consistency scoring'
        }
    
    def get_enhanced_filename(self, ticker: str, start_date: str, end_date: str) -> str:
        """Generate enhanced filename for trading psychology visualization."""
        return f"{ticker}_Trading_Psychology_and_Behavioral_Analysis_{start_date}_to_{end_date}.png"
    
    def get_enhanced_title(self, ticker: str) -> str:
        """Generate enhanced title for trading psychology visualization."""
        return f"{ticker} Trading Psychology Analysis - Streaks, Drawdowns & Emotional Patterns"
    
    def get_educational_focus(self) -> str:
        """Get the educational focus description."""
        return "🧠 Focus: Emotional discipline, drawdown management, consistency patterns"
    
    def analyze_streak_patterns(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze winning and losing streak patterns.
        
        Args:
            trades_df: DataFrame with trade data
            
        Returns:
            Dictionary with streak analysis data
        """
        # Create win/loss sequence
        trades_df['Win'] = trades_df['Profit'] > 0
        trades_df['Streak_Group'] = (trades_df['Win'] != trades_df['Win'].shift()).cumsum()
        
        # Calculate streak lengths
        streak_analysis = trades_df.groupby('Streak_Group').agg({
            'Win': ['first', 'size'],
            'Profit': 'sum'
        }).reset_index()
        
        streak_analysis.columns = ['Group', 'Is_Win', 'Length', 'Total_Profit']
        
        # Separate winning and losing streaks
        win_streaks = streak_analysis[streak_analysis['Is_Win']]['Length']
        loss_streaks = streak_analysis[~streak_analysis['Is_Win']]['Length']
        
        return {
            'win_streaks': win_streaks.tolist(),
            'loss_streaks': loss_streaks.tolist(),
            'max_win_streak': win_streaks.max() if len(win_streaks) > 0 else 0,
            'max_loss_streak': loss_streaks.max() if len(loss_streaks) > 0 else 0,
            'avg_win_streak': win_streaks.mean() if len(win_streaks) > 0 else 0,
            'avg_loss_streak': loss_streaks.mean() if len(loss_streaks) > 0 else 0,
            'streak_data': streak_analysis
        }
    
    def analyze_drawdown_recovery(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze drawdown patterns and recovery behavior.
        
        Args:
            trades_df: DataFrame with trade data
            
        Returns:
            Dictionary with drawdown recovery analysis
        """
        # Calculate cumulative returns
        trades_df['Cumulative_Profit'] = trades_df['Profit'].cumsum()
        trades_df['Running_Max'] = trades_df['Cumulative_Profit'].expanding().max()
        trades_df['Drawdown'] = trades_df['Cumulative_Profit'] - trades_df['Running_Max']
        
        # Find drawdown periods
        in_drawdown = trades_df['Drawdown'] < 0
        drawdown_groups = (in_drawdown != in_drawdown.shift()).cumsum()
        
        drawdown_periods = []
        recovery_times = []
        
        for group in drawdown_groups.unique():
            group_data = trades_df[drawdown_groups == group]
            if group_data['Drawdown'].min() < 0:  # This is a drawdown period
                max_dd = group_data['Drawdown'].min()
                duration = len(group_data)
                
                # Find recovery time (if any)
                recovery_idx = group_data.index[-1]
                post_recovery = trades_df.loc[recovery_idx:]['Drawdown'] >= 0
                if post_recovery.any():
                    recovery_time = post_recovery.idxmax() - group_data.index[0]
                else:
                    recovery_time = np.nan
                
                drawdown_periods.append({
                    'max_drawdown': max_dd,
                    'duration': duration,
                    'recovery_time': recovery_time,
                    'start_idx': group_data.index[0],
                    'end_idx': group_data.index[-1]
                })
                
                if not np.isnan(recovery_time):
                    recovery_times.append(recovery_time)
        
        return {
            'drawdown_periods': drawdown_periods,
            'max_drawdown': trades_df['Drawdown'].min(),
            'avg_recovery_time': np.mean(recovery_times) if recovery_times else np.nan,
            'drawdown_curve': trades_df['Drawdown'].tolist(),
            'cumulative_returns': trades_df['Cumulative_Profit'].tolist(),
            'total_drawdown_periods': len(drawdown_periods)
        }
    
    def detect_emotional_trading_signals(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect potential emotional trading patterns.
        
        Args:
            trades_df: DataFrame with trade data
            
        Returns:
            Dictionary with emotional trading indicators
        """
        # Revenge trading detection (quick trades after losses)
        trades_df['Time_Since_Last'] = trades_df['Entry_Date'].diff()
        trades_df['Previous_Loss'] = (trades_df['Profit'].shift(1) < 0)
        
        # Quick trades after losses (within 1 day)
        revenge_trades = trades_df[
            (trades_df['Previous_Loss']) & 
            (trades_df['Time_Since_Last'] <= timedelta(days=1))
        ]
        
        # Overconfidence detection (larger positions after wins)
        trades_df['Previous_Win'] = (trades_df['Profit'].shift(1) > 0)
        if 'Position_Size' in trades_df.columns:
            trades_df['Position_Size_Change'] = trades_df['Position_Size'] / trades_df['Position_Size'].shift(1)
            
            overconfidence_trades = trades_df[
                (trades_df['Previous_Win']) & 
                (trades_df['Position_Size_Change'] > 1.5)  # 50% increase in position size
            ]
        else:
            overconfidence_trades = pd.DataFrame()
        
        # Volatility clustering (emotional periods with rapid trading)
        trades_per_day = trades_df.groupby(trades_df['Entry_Date'].dt.date).size()
        high_activity_days = trades_per_day[trades_per_day > trades_per_day.quantile(0.9)]
        
        return {
            'revenge_trade_count': len(revenge_trades),
            'revenge_trade_performance': revenge_trades['Profit'].mean() if len(revenge_trades) > 0 else 0,
            'overconfidence_trade_count': len(overconfidence_trades),
            'overconfidence_performance': overconfidence_trades['Profit'].mean() if len(overconfidence_trades) > 0 else 0,
            'high_activity_days': len(high_activity_days),
            'avg_daily_trades': trades_per_day.mean(),
            'emotional_periods': high_activity_days.index.tolist()
        }
    
    def calculate_consistency_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading consistency and discipline metrics.
        
        Args:
            trades_df: DataFrame with trade data
            
        Returns:
            Dictionary with consistency metrics
        """
        # Win rate consistency over time
        trades_df['Month'] = trades_df['Entry_Date'].dt.to_period('M')
        monthly_stats = trades_df.groupby('Month').agg({
            'Profit': ['count', lambda x: (x > 0).sum(), 'mean', 'std']
        })
        
        monthly_stats.columns = ['Total_Trades', 'Winning_Trades', 'Avg_Profit', 'Profit_Std']
        monthly_stats['Win_Rate'] = monthly_stats['Winning_Trades'] / monthly_stats['Total_Trades']
        
        # Consistency scores
        win_rate_consistency = 1 - monthly_stats['Win_Rate'].std()
        profit_consistency = 1 - (monthly_stats['Avg_Profit'].std() / abs(monthly_stats['Avg_Profit'].mean()))
        
        # Risk consistency (if position sizing data available)
        if 'Position_Size' in trades_df.columns:
            risk_consistency = 1 - (trades_df['Position_Size'].std() / trades_df['Position_Size'].mean())
        else:
            risk_consistency = np.nan
        
        return {            'win_rate_consistency': max(0, win_rate_consistency),
            'profit_consistency': max(0, profit_consistency) if not np.isnan(profit_consistency) else 0,
            'risk_consistency': max(0, risk_consistency) if not np.isnan(risk_consistency) else np.nan,
            'monthly_stats': monthly_stats,
            'overall_consistency_score': np.nanmean([win_rate_consistency, profit_consistency, risk_consistency])
        }
    
    def create(self, ticker_data: Dict[str, Any], ticker: str, 
               date_range: str, output_dir: Path) -> Optional[Path]:
        """
        Create comprehensive trading psychology insights visualization.
        
        Args:
            ticker_data: Dictionary containing all ticker data
            ticker: Ticker symbol
            date_range: Date range string
            output_dir: Directory to save the charts
            
        Returns:
            Path to the created chart file or None if creation failed
        """
        try:
            trades_df = ticker_data.get('trades_df')
            if trades_df is None or trades_df.empty:
                self.logger.warning(f"No trade data available for {ticker}")
                return None
            
            # Extract date components from date_range
            # Expected format: "YYYY-MM-DD_to_YYYY-MM-DD"
            if "_to_" in date_range:
                start_date, end_date = date_range.split("_to_")
            else:
                # Fallback: use first and last trade dates
                start_date = trades_df['Entry_Date'].min().strftime('%Y-%m-%d')
                end_date = trades_df['Exit_Date'].max().strftime('%Y-%m-%d')
            
            # Save current output directory if needed
            original_output_dir = getattr(self, 'output_dir', None)
            self.output_dir = output_dir
            
            # Use the existing create_visualization method
            chart_path = self.create_visualization(
                ticker=ticker,
                trades_df=trades_df,
                start_date=start_date,
                end_date=end_date,
                style_config=None,
                enhanced_naming=self.use_enhanced_naming
            )
            
            # Restore original output directory
            if original_output_dir is not None:
                self.output_dir = original_output_dir
            
            return Path(chart_path) if chart_path else None
            
        except Exception as e:
            self.logger.error(f"Error creating trading psychology visualization for {ticker}: {e}")
            return None
    
    def create_visualization(self, ticker: str, trades_df: pd.DataFrame,
                           start_date: str, end_date: str, 
                           style_config: Optional[Dict] = None,
                           enhanced_naming: bool = True) -> str:
        """
        Create the complete trading psychology insights visualization.
        
        Args:
            ticker: Stock ticker symbol
            trades_df: DataFrame with trade data
            start_date: Start date string
            end_date: End date string  
            style_config: Optional styling configuration
            enhanced_naming: Whether to use enhanced naming conventions
            
        Returns:
            Path to the generated visualization file
        """
        try:
            # Ensure required columns
            required_columns = ['Entry_Date', 'Exit_Date', 'Profit']
            missing_columns = [col for col in required_columns if col not in trades_df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert date columns
            trades_df = trades_df.copy()
            trades_df['Entry_Date'] = pd.to_datetime(trades_df['Entry_Date'])
            trades_df['Exit_Date'] = pd.to_datetime(trades_df['Exit_Date'])
            
            # Perform analysis
            self.logger.info(f"Analyzing trading psychology patterns for {ticker}")
            
            streak_analysis = self.analyze_streak_patterns(trades_df)
            drawdown_analysis = self.analyze_drawdown_recovery(trades_df)
            emotional_analysis = self.detect_emotional_trading_signals(trades_df)
            consistency_analysis = self.calculate_consistency_metrics(trades_df)
            
            # Create visualization
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
            
            # Apply styling
            self.apply_style_config(style_config)
            
            # Get colors from style
            colors = self.get_color_palette()
            
            # 1. Winning/Losing Streak Analysis (Top Left)
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_streak_analysis(ax1, streak_analysis, colors)
            
            # 2. Drawdown Recovery Pattern (Top Center)
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_drawdown_recovery(ax2, drawdown_analysis, trades_df, colors)
            
            # 3. Emotional Trading Indicators (Top Right)
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_emotional_indicators(ax3, emotional_analysis, colors)
            
            # 4. Consistency Over Time (Middle, spans 2 columns)
            ax4 = fig.add_subplot(gs[1, :2])
            self._plot_consistency_timeline(ax4, consistency_analysis, colors)
            
            # 5. Psychology Metrics Summary (Middle Right)
            ax5 = fig.add_subplot(gs[1, 2])
            self._plot_psychology_summary(ax5, streak_analysis, emotional_analysis, consistency_analysis, colors)
            
            # 6. Cumulative Performance vs Drawdowns (Bottom, spans all)
            ax6 = fig.add_subplot(gs[2:, :])
            self._plot_performance_psychology(ax6, trades_df, drawdown_analysis, colors)
            
            # Set overall title
            if enhanced_naming:
                title = self.get_enhanced_title(ticker)
                focus = self.get_educational_focus()
                fig.suptitle(f"{title}\n{focus}", fontsize=16, fontweight='bold', y=0.98)
            else:
                fig.suptitle(f"{ticker} Trading Psychology Analysis", fontsize=16, fontweight='bold', y=0.98)
            
            # Generate filename and save
            if enhanced_naming:
                filename = self.get_enhanced_filename(ticker, start_date, end_date)
            else:
                filename = f"{ticker}_trading_psychology_{start_date}_to_{end_date}.png"
            
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            self.logger.info(f"Trading psychology visualization saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error creating trading psychology visualization: {str(e)}")
            plt.close('all')
            raise
    
    def _plot_streak_analysis(self, ax, streak_analysis: Dict, colors: Dict):
        """Plot winning/losing streak patterns."""
        win_streaks = streak_analysis['win_streaks']
        loss_streaks = streak_analysis['loss_streaks']
        
        # Create histogram data
        max_streak = max(max(win_streaks) if win_streaks else 0,
                        max(loss_streaks) if loss_streaks else 0)
        
        bins = range(1, max_streak + 2)
        
        # Plot histograms
        if win_streaks:
            ax.hist(win_streaks, bins=bins, alpha=0.7, color=colors['profit'], 
                   label=f'Win Streaks (Max: {streak_analysis["max_win_streak"]})', edgecolor='black')
        
        if loss_streaks:
            ax.hist(loss_streaks, bins=bins, alpha=0.7, color=colors['loss'], 
                   label=f'Loss Streaks (Max: {streak_analysis["max_loss_streak"]})', edgecolor='black')
        
        ax.set_xlabel('Streak Length')
        ax.set_ylabel('Frequency')
        ax.set_title('Winning vs Losing Streak Distribution\n📈 Understanding Streak Patterns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add educational note
        avg_win = streak_analysis['avg_win_streak']
        avg_loss = streak_analysis['avg_loss_streak']
        ax.text(0.02, 0.98, f'Avg Win: {avg_win:.1f}\nAvg Loss: {avg_loss:.1f}', 
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _plot_drawdown_recovery(self, ax, drawdown_analysis: Dict, trades_df: pd.DataFrame, colors: Dict):
        """Plot drawdown and recovery patterns."""
        # Plot drawdown curve
        trade_numbers = range(len(trades_df))
        drawdowns = drawdown_analysis['drawdown_curve']
        
        ax.fill_between(trade_numbers, drawdowns, 0, alpha=0.3, color=colors['loss'], 
                       label='Drawdown')
        ax.plot(trade_numbers, drawdowns, color=colors['loss'], linewidth=2)
        
        # Mark drawdown periods
        for dd_period in drawdown_analysis['drawdown_periods']:
            start_idx = dd_period['start_idx']
            end_idx = dd_period['end_idx']
            ax.axvspan(start_idx, end_idx, alpha=0.2, color='red')
        
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Drawdown ($)')
        ax.set_title('Drawdown Recovery Analysis\n🛡️ Recovery Time & Patterns')
        ax.grid(True, alpha=0.3)
        
        # Add recovery stats
        max_dd = drawdown_analysis['max_drawdown']
        avg_recovery = drawdown_analysis.get('avg_recovery_time', 0)
        dd_periods = drawdown_analysis['total_drawdown_periods']
        
        stats_text = f'Max DD: ${max_dd:.2f}\nAvg Recovery: {avg_recovery:.1f} trades\nDD Periods: {dd_periods}'
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, va='bottom', ha='left',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    def _plot_emotional_indicators(self, ax, emotional_analysis: Dict, colors: Dict):
        """Plot emotional trading indicators."""
        categories = ['Revenge\nTrades', 'Overconfident\nTrades', 'High Activity\nDays']
        values = [
            emotional_analysis['revenge_trade_count'],
            emotional_analysis['overconfidence_trade_count'],
            emotional_analysis['high_activity_days']
        ]
        
        bar_colors = [colors['loss'], colors['warning'], colors['caution']]
        bars = ax.bar(categories, values, color=bar_colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Count')
        ax.set_title('Emotional Trading Signals\n🧠 Psychological Warning Signs')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add performance impact
        revenge_perf = emotional_analysis['revenge_trade_performance']
        overconf_perf = emotional_analysis['overconfidence_performance']
        
        perf_text = f'Revenge P&L: ${revenge_perf:.2f}\nOverconf P&L: ${overconf_perf:.2f}'
        ax.text(0.98, 0.98, perf_text, transform=ax.transAxes, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def _plot_consistency_timeline(self, ax, consistency_analysis: Dict, colors: Dict):
        """Plot consistency metrics over time."""
        monthly_stats = consistency_analysis['monthly_stats']
        
        if len(monthly_stats) > 1:
            months = [str(period) for period in monthly_stats.index]
            win_rates = monthly_stats['Win_Rate'].values
            
            # Plot win rate over time
            ax.plot(months, win_rates, marker='o', linewidth=2, markersize=6,
                   color=colors['primary'], label='Monthly Win Rate')
            
            # Add trend line
            x_numeric = range(len(win_rates))
            z = np.polyfit(x_numeric, win_rates, 1)
            p = np.poly1d(z)
            ax.plot(months, p(x_numeric), '--', color=colors['secondary'], 
                   alpha=0.7, label='Trend')
            
            # Highlight inconsistent months
            std_threshold = np.std(win_rates)
            mean_wr = np.mean(win_rates)
            
            for i, (month, wr) in enumerate(zip(months, win_rates)):
                if abs(wr - mean_wr) > std_threshold:
                    ax.scatter(month, wr, s=100, color='red', alpha=0.7, zorder=5)
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Win Rate')
            ax.set_title('Trading Consistency Over Time\n📊 Performance Stability Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add consistency score
            consistency_score = consistency_analysis['overall_consistency_score']
            ax.text(0.02, 0.98, f'Consistency Score: {consistency_score:.2f}', 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Insufficient data for\nconsistency analysis\n(Need >1 month)', 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax.set_title('Trading Consistency Over Time')
    
    def _plot_psychology_summary(self, ax, streak_analysis: Dict, emotional_analysis: Dict, 
                                consistency_analysis: Dict, colors: Dict):
        """Plot summary of psychology metrics."""
        # Calculate psychology scores (0-100)
        max_loss_streak = streak_analysis['max_loss_streak']
        
        # Streak control score (better when loss streaks are shorter)
        streak_score = max(0, 100 - (max_loss_streak * 10)) if max_loss_streak > 0 else 100
        
        # Emotional control score (fewer emotional trades = better)
        total_trades = emotional_analysis['revenge_trade_count'] + emotional_analysis['overconfidence_trade_count']
        emotional_score = max(0, 100 - (total_trades * 5))
        
        # Consistency score
        consistency_score = (consistency_analysis['overall_consistency_score'] * 100) if not np.isnan(consistency_analysis['overall_consistency_score']) else 50
        
        scores = [streak_score, emotional_score, consistency_score]
        labels = ['Streak\nControl', 'Emotional\nControl', 'Consistency']
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        scores_plot = scores + [scores[0]]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, scores_plot, 'o-', linewidth=2, color=colors['primary'])
        ax.fill(angles, scores_plot, alpha=0.25, color=colors['primary'])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        
        # Add score grid
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.grid(True, alpha=0.3)
        
        ax.set_title('Psychology Score Summary\n🎯 Overall Mental Performance')
        
        # Add overall score
        overall_score = np.mean(scores)
        ax.text(0.5, -0.15, f'Overall Psychology Score: {overall_score:.1f}/100', 
               ha='center', va='top', transform=ax.transAxes, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    def _plot_performance_psychology(self, ax, trades_df: pd.DataFrame, 
                                   drawdown_analysis: Dict, colors: Dict):
        """Plot cumulative performance with psychological overlays."""
        trade_numbers = range(len(trades_df))
        cumulative_returns = drawdown_analysis['cumulative_returns']
        drawdowns = drawdown_analysis['drawdown_curve']
        
        # Main performance line
        ax.plot(trade_numbers, cumulative_returns, linewidth=2, color=colors['primary'], 
               label='Cumulative P&L')        # Fill areas for different psychological states
        # Color code the line based on psychological state
        for i in range(len(trade_numbers) - 1):
            if drawdowns[i] == 0:  # At new high - green
                ax.plot([trade_numbers[i], trade_numbers[i+1]], 
                       [cumulative_returns[i], cumulative_returns[i+1]], 
                       color=colors['profit'], linewidth=3, alpha=0.7)
            elif drawdowns[i] < -abs(max(cumulative_returns) * 0.1):  # Deep drawdown - red
                ax.plot([trade_numbers[i], trade_numbers[i+1]], 
                       [cumulative_returns[i], cumulative_returns[i+1]], 
                       color=colors['loss'], linewidth=3, alpha=0.7)
        
        # Mark major psychological events
        for dd_period in drawdown_analysis['drawdown_periods']:
            if abs(dd_period['max_drawdown']) > abs(max(cumulative_returns) * 0.05):  # Significant drawdown
                start_idx = dd_period['start_idx']
                ax.axvline(start_idx, color='red', linestyle='--', alpha=0.5)
                ax.text(start_idx, max(cumulative_returns) * 0.9, 'Major DD', 
                       rotation=90, ha='right', va='top', color='red')
        
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative P&L ($)')
        ax.set_title('Performance Journey with Psychological Context\n💭 Emotional State Throughout Trading Period')
        ax.grid(True, alpha=0.3)
        
        # Add legend for color coding
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=colors['profit'], lw=3, label='At New Highs (Confident)'),
            Line2D([0], [0], color=colors['primary'], lw=2, label='Normal Trading'),
            Line2D([0], [0], color=colors['loss'], lw=3, label='Deep Drawdown (Stressed)')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        # Add final performance stats
        final_return = cumulative_returns[-1]
        max_dd = min(drawdowns)
        
        stats_text = f'Final P&L: ${final_return:.2f}\nMax Drawdown: ${max_dd:.2f}\nReturn/DD Ratio: {abs(final_return/max_dd):.2f}' if max_dd != 0 else f'Final P&L: ${final_return:.2f}\nMax Drawdown: $0.00'
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, va='bottom', ha='right',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
