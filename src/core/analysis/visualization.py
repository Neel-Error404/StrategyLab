# src/etl/analysis/visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

class StrategyVisualizer:
    """
    Provides comprehensive visualization capabilities for backtesting results.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the strategy visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs (optional)
        """
        self.logger = logging.getLogger("StrategyVisualizer")
        self.output_dir = output_dir
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Set style for all plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
    
    def load_trade_data(self, strategy_name: str, pull_date: str, 
                        strat_output_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load trade and summary data for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            pull_date: Date of the backtest
            strat_output_dir: Directory containing strategy outputs (defaults to "Backtester/Strat_out/{strategy_name}/{pull_date}")
            
        Returns:
            Tuple of (trades_df, summary_df)
        """
        if not strat_output_dir:
            strat_output_dir = Path(f"Backtester/Strat_out/{strategy_name}/{pull_date}")
        
        if not strat_output_dir.exists():
            self.logger.error(f"Strategy output directory not found: {strat_output_dir}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Load summary file
        summary_file = strat_output_dir / f"{pull_date}_Summary.csv"
        if not summary_file.exists():
            self.logger.error(f"Summary file not found: {summary_file}")
            return pd.DataFrame(), pd.DataFrame()
        
        summary_df = pd.read_csv(summary_file)
        
        # Get list of all ticker trade files
        trade_files = list(strat_output_dir.glob("*_Trades_*.csv"))
        
        if not trade_files:
            self.logger.error(f"No trade files found in {strat_output_dir}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Load and combine all trade data
        all_trades = []
        
        for trade_file in trade_files:
            ticker = trade_file.name.split("_")[0]
            
            try:
                df = pd.read_csv(trade_file)
                if not df.empty:
                    df['Ticker'] = ticker  # Ensure ticker column exists
                    all_trades.append(df)
            except Exception as e:
                self.logger.error(f"Error loading trade file for {ticker}: {e}")
        
        if not all_trades:
            self.logger.error(f"No trade data found")
            return pd.DataFrame(), summary_df
        
        # Combine all trade data
        trades_df = pd.concat(all_trades, ignore_index=True)
        
        # Convert timestamp columns to datetime
        for col in ['Entry Time', 'Exit Time', 'High Time', 'Low Time']:
            if col in trades_df.columns:
                trades_df[col] = pd.to_datetime(trades_df[col])
        
        return trades_df, summary_df
    
    def calculate_equity_curve(self, trades_df: pd.DataFrame, initial_capital: float = 100000, 
                               position_size_pct: float = 0.1, include_tickers: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Calculate equity curves from trade data.
        
        Args:
            trades_df: DataFrame containing trade data
            initial_capital: Initial capital amount
            position_size_pct: Position size as percentage of capital
            include_tickers: Whether to calculate equity curves for individual tickers
            
        Returns:
            Dictionary mapping curve names to equity curve DataFrames
        """
        if trades_df.empty:
            self.logger.error("No trade data available for equity curve calculation")
            return {}
        
        # Sort trades by entry time
        trades_df = trades_df.sort_values('Entry Time')
        
        # Initialize equity curves dictionary
        equity_curves = {'Portfolio': pd.DataFrame({
            'timestamp': [trades_df['Entry Time'].min()],
            'equity': [initial_capital]
        })}
        
        # Initialize equity for each ticker if requested
        if include_tickers:
            for ticker in trades_df['Ticker'].unique():
                equity_curves[ticker] = pd.DataFrame({
                    'timestamp': [trades_df['Entry Time'].min()],
                    'equity': [initial_capital]
                })
        
        # Calculate running equity for portfolio
        portfolio_capital = initial_capital
        
        # Process each trade chronologically
        for idx, trade in trades_df.iterrows():
            ticker = trade['Ticker']
            position_size = initial_capital * position_size_pct
            profit_pct = trade['Profit (%)'] / 100  # Convert percentage to decimal
            profit_amount = position_size * profit_pct
            
            # Update portfolio equity
            portfolio_capital += profit_amount
            equity_curves['Portfolio'] = pd.concat([
                equity_curves['Portfolio'],
                pd.DataFrame({
                    'timestamp': [trade['Exit Time']],
                    'equity': [portfolio_capital]
                })
            ], ignore_index=True)
            
            # Update ticker equity if requested
            if include_tickers and ticker in equity_curves:
                ticker_equity = equity_curves[ticker].iloc[-1]['equity']
                ticker_equity += profit_amount
                equity_curves[ticker] = pd.concat([
                    equity_curves[ticker],
                    pd.DataFrame({
                        'timestamp': [trade['Exit Time']],
                        'equity': [ticker_equity]
                    })
                ], ignore_index=True)
        
        # Calculate drawdowns for each equity curve
        for key in equity_curves:
            df = equity_curves[key]
            df['cummax'] = df['equity'].cummax()
            df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax'] * 100
            df['drawdown_dollars'] = df['equity'] - df['cummax']
            
            # Calculate additional metrics
            df['return'] = df['equity'] / initial_capital - 1
            df['return_pct'] = df['return'] * 100
            
            equity_curves[key] = df
        
        return equity_curves
    
    def plot_equity_curve(self, equity_curves: Dict[str, pd.DataFrame], title: str = "Equity Curve", 
                         include_drawdown: bool = True, save_path: Optional[str] = None) -> None:
        """
        Plot equity curves.
        
        Args:
            equity_curves: Dictionary mapping curve names to equity curve DataFrames
            title: Plot title
            include_drawdown: Whether to include drawdown subplot
            save_path: Path to save the plot (optional)
        """
        if not equity_curves:
            self.logger.error("No equity curves available for plotting")
            return
        
        # Determine plot layout
        if include_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot equity curves
        for name, curve in equity_curves.items():
            if curve.empty:
                continue
                
            if name == 'Portfolio':
                ax1.plot(curve['timestamp'], curve['equity'], linewidth=2, label=name)
            else:
                ax1.plot(curve['timestamp'], curve['equity'], linewidth=1, alpha=0.7, label=name)
        
        # Format equity curve plot
        ax1.set_title(title)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity')
        ax1.legend()
        ax1.grid(True)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Format y-axis with comma separator for thousands
        ax1.get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, loc: f"${int(x):,}")
        )
        
        # Add drawdown subplot if requested
        if include_drawdown and 'Portfolio' in equity_curves:
            portfolio = equity_curves['Portfolio']
            ax2.fill_between(portfolio['timestamp'], portfolio['drawdown'], 0, 
                            color='red', alpha=0.3)
            ax2.plot(portfolio['timestamp'], portfolio['drawdown'], color='red', linewidth=1)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Invert y-axis for drawdown (negative values at top)
            ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Equity curve saved to {save_path}")
        else:
            plt.show()
    
    def plot_trade_distribution(self, trades_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Plot trade distribution by various factors.
        
        Args:
            trades_df: DataFrame containing trade data
            save_path: Path to save the plot (optional)
        """
        if trades_df.empty:
            self.logger.error("No trade data available for distribution plotting")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Trade distribution by hour
        trades_df['Hour'] = trades_df['Entry Time'].dt.hour
        hourly_counts = trades_df.groupby('Hour').size()
        
        sns.barplot(x=hourly_counts.index, y=hourly_counts.values, ax=ax1)
        ax1.set_title('Trade Distribution by Hour')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Trades')
        
        # 2. Trade profitability by ticker
        ticker_profit = trades_df.groupby('Ticker')['Profit (%)'].mean().sort_values(ascending=False)
        
        sns.barplot(x=ticker_profit.index, y=ticker_profit.values, ax=ax2)
        ax2.set_title('Average Profit (%) by Ticker')
        ax2.set_xlabel('Ticker')
        ax2.set_ylabel('Average Profit (%)')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # 3. Trade duration distribution
        if 'Trade Duration (min)' in trades_df.columns:
            sns.histplot(trades_df['Trade Duration (min)'], bins=20, kde=True, ax=ax3)
            ax3.set_title('Trade Duration Distribution')
            ax3.set_xlabel('Duration (minutes)')
            ax3.set_ylabel('Frequency')
        
        # 4. Profit distribution
        sns.histplot(trades_df['Profit (%)'], bins=20, kde=True, ax=ax4)
        ax4.set_title('Profit Distribution')
        ax4.set_xlabel('Profit (%)')
        ax4.set_ylabel('Frequency')
        
        # Add line at zero
        ax4.axvline(0, color='red', linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Trade distribution plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_correlation_heatmap(self, trades_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Plot correlation heatmap between ticker performances.
        
        Args:
            trades_df: DataFrame containing trade data
            save_path: Path to save the plot (optional)
        """
        if trades_df.empty:
            self.logger.error("No trade data available for correlation analysis")
            return
        
        # Pivot trades to get ticker performance over time
        daily_returns = {}
        
        for ticker in trades_df['Ticker'].unique():
            ticker_trades = trades_df[trades_df['Ticker'] == ticker]
            if ticker_trades.empty:
                continue
                
            # Group by day and calculate daily returns
            ticker_trades['Date'] = ticker_trades['Exit Time'].dt.date
            daily_pnl = ticker_trades.groupby('Date')['Profit (%)'].sum().reset_index()
            daily_returns[ticker] = daily_pnl.set_index('Date')['Profit (%)']
        
        # Create a DataFrame with all tickers' daily returns
        returns_df = pd.DataFrame(daily_returns)
        
        if returns_df.empty or returns_df.shape[1] < 2:
            self.logger.warning("Insufficient data for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True,
            cmap=cmap,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8},
            fmt=".2f"
        )
        
        plt.title('Correlation Between Ticker Performances')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Correlation heatmap saved to {save_path}")
        else:
            plt.show()
    
    def plot_performance_metrics(self, summary_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Plot various performance metrics from summary data.
        
        Args:
            summary_df: DataFrame containing summary data
            save_path: Path to save the plot (optional)
        """
        if summary_df.empty:
            self.logger.error("No summary data available for metrics plotting")
            return
        
        # Sort tickers by average profit
        sorted_df = summary_df.sort_values('Average Profit (%)', ascending=False)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Average Profit (%) by Ticker
        sns.barplot(x='Ticker', y='Average Profit (%)', data=sorted_df, ax=ax1)
        ax1.set_title('Average Profit (%) by Ticker')
        ax1.set_xlabel('Ticker')
        ax1.set_ylabel('Average Profit (%)')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # 2. Win Rate (%) by Ticker
        if 'Wins' in sorted_df.columns and 'Total Trades' in sorted_df.columns:
            sorted_df['Win Rate (%)'] = sorted_df['Wins'] / sorted_df['Total Trades'] * 100
            win_rate_df = sorted_df.sort_values('Win Rate (%)', ascending=False)
            
            sns.barplot(x='Ticker', y='Win Rate (%)', data=win_rate_df, ax=ax2)
            ax2.set_title('Win Rate (%) by Ticker')
            ax2.set_xlabel('Ticker')
            ax2.set_ylabel('Win Rate (%)')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # 3. Number of Trades by Ticker
        trade_count_df = sorted_df.sort_values('Total Trades', ascending=False)
        
        sns.barplot(x='Ticker', y='Total Trades', data=trade_count_df, ax=ax3)
        ax3.set_title('Number of Trades by Ticker')
        ax3.set_xlabel('Ticker')
        ax3.set_ylabel('Total Trades')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        
        # 4. Profit vs Drawdown Scatter
        if 'Max Drawdown (%)' in sorted_df.columns:
            ax4.scatter(sorted_df['Max Drawdown (%)'], sorted_df['Average Profit (%)'])
            
            for idx, row in sorted_df.iterrows():
                ax4.annotate(row['Ticker'], 
                          (row['Max Drawdown (%)'], row['Average Profit (%)']),
                          fontsize=8)
            
            ax4.set_title('Profit vs Drawdown')
            ax4.set_xlabel('Max Drawdown (%)')
            ax4.set_ylabel('Average Profit (%)')
            ax4.grid(True)
            
            # Add quadrant lines
            ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax4.axvline(0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Performance metrics plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_trades_over_time(self, trades_df: pd.DataFrame, tickers: Optional[List[str]] = None, 
                             save_path: Optional[str] = None) -> None:
        """
        Plot trade entry/exit points over time.
        
        Args:
            trades_df: DataFrame containing trade data
            tickers: List of tickers to include (optional, defaults to all)
            save_path: Path to save the plot (optional)
        """
        if trades_df.empty:
            self.logger.error("No trade data available for time analysis")
            return
        
        # Filter tickers if provided
        if tickers:
            trades_df = trades_df[trades_df['Ticker'].isin(tickers)]
            
            if trades_df.empty:
                self.logger.error(f"No trade data available for specified tickers: {tickers}")
                return
        
        # Get unique tickers
        unique_tickers = trades_df['Ticker'].unique()
        
        # Create one subplot per ticker
        n_tickers = len(unique_tickers)
        n_cols = min(3, n_tickers)
        n_rows = (n_tickers + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), squeeze=False)
        
        # Flatten axes array for easy indexing
        axes = axes.flatten()
        
        for i, ticker in enumerate(unique_tickers):
            ax = axes[i]
            ticker_trades = trades_df[trades_df['Ticker'] == ticker]
            
            if ticker_trades.empty:
                continue
            
            # Plot profitable trades in green, losing trades in red
            profitable_trades = ticker_trades[ticker_trades['Profit (%)'] > 0]
            losing_trades = ticker_trades[ticker_trades['Profit (%)'] <= 0]
            
            # Plot entry points
            ax.scatter(profitable_trades['Entry Time'], profitable_trades['Entry Price'], 
                      marker='^', color='green', alpha=0.7, label='Profitable Entry')
            ax.scatter(losing_trades['Entry Time'], losing_trades['Entry Price'], 
                      marker='^', color='red', alpha=0.7, label='Losing Entry')
            
            # Plot exit points
            ax.scatter(profitable_trades['Exit Time'], profitable_trades['Exit Price'], 
                      marker='v', color='green', alpha=0.7, label='Profitable Exit')
            ax.scatter(losing_trades['Exit Time'], losing_trades['Exit Price'], 
                      marker='v', color='red', alpha=0.7, label='Losing Exit')
            
            # Connect entry and exit points with lines
            for _, trade in ticker_trades.iterrows():
                color = 'green' if trade['Profit (%)'] > 0 else 'red'
                ax.plot([trade['Entry Time'], trade['Exit Time']], 
                       [trade['Entry Price'], trade['Exit Price']], 
                       color=color, alpha=0.5, linestyle='-')
            
            ax.set_title(f"{ticker} Trades")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.grid(True)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Add legend for first plot only
            if i == 0:
                ax.legend()
        
        # Hide unused subplots
        for i in range(n_tickers, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Trades over time plot saved to {save_path}")
        else:
            plt.show()
    
    def create_strategy_dashboard(self, strategy_name: str, pull_date: str, 
                                 strat_output_dir: Optional[Path] = None) -> None:
        """
        Create a comprehensive dashboard for strategy performance.
        
        Args:
            strategy_name: Name of the strategy
            pull_date: Date of the backtest
            strat_output_dir: Directory containing strategy outputs (defaults to "Backtester/Strat_out/{strategy_name}/{pull_date}")
        """
        if not strat_output_dir:
            strat_output_dir = Path(f"Backtester/Strat_out/{strategy_name}/{pull_date}")
        
        # Create output directory for plots
        plot_dir = self.output_dir or (strat_output_dir / "plots")
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Load trade and summary data
        trades_df, summary_df = self.load_trade_data(strategy_name, pull_date, strat_output_dir)
        
        if trades_df.empty:
            self.logger.error("No trade data available for dashboard creation")
            return
        
        # Calculate equity curves
        equity_curves = self.calculate_equity_curve(trades_df)
        
        # Generate plots
        # 1. Equity curve
        self.plot_equity_curve(
            equity_curves, 
            title=f"{strategy_name} Equity Curve - {pull_date}",
            save_path=str(plot_dir / "equity_curve.png")
        )
        
        # 2. Trade distribution
        self.plot_trade_distribution(
            trades_df,
            save_path=str(plot_dir / "trade_distribution.png")
        )
        
        # 3. Correlation heatmap
        self.plot_correlation_heatmap(
            trades_df,
            save_path=str(plot_dir / "correlation_heatmap.png")
        )
        
        # 4. Performance metrics
        self.plot_performance_metrics(
            summary_df,
            save_path=str(plot_dir / "performance_metrics.png")
        )
        
        # 5. Trades over time for top 10 tickers by profit
        if len(summary_df) > 0:
            top_tickers = summary_df.sort_values('Average Profit (%)', ascending=False)['Ticker'].head(10).tolist()
            
            self.plot_trades_over_time(
                trades_df,
                tickers=top_tickers,
                save_path=str(plot_dir / "trades_over_time.png")
            )
        
        self.logger.info(f"Strategy dashboard created in {plot_dir}")
    
    # Additional visualization methods for production-grade reporting

def plot_advanced_metrics_dashboard(self, trades_df: pd.DataFrame, 
                                  equity_curve: pd.DataFrame,
                                  save_path: Optional[str] = None) -> None:
    """Create comprehensive metrics dashboard with multiple subplots."""
    
    fig = plt.figure(figsize=(20, 24))
    
    # Create grid for subplots
    gs = plt.GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # 1. Equity Curve with Drawdown Shading
    ax1 = fig.add_subplot(gs[0, :])
    self._plot_equity_with_drawdown_shading(equity_curve, ax1)
    
    # 2. Returns Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    self._plot_returns_distribution(equity_curve, ax2)
    
    # 3. Rolling Sharpe Ratio
    ax3 = fig.add_subplot(gs[1, 1])
    self._plot_rolling_sharpe(equity_curve, ax3)
    
    # 4. Monthly Returns Heatmap
    ax4 = fig.add_subplot(gs[1, 2])
    self._plot_monthly_returns_heatmap(equity_curve, ax4)
    
    # 5. Trade Analysis
    ax5 = fig.add_subplot(gs[2, :2])
    self._plot_trade_analysis(trades_df, ax5)
    
    # 6. Risk Metrics
    ax6 = fig.add_subplot(gs[2, 2])
    self._plot_risk_metrics_table(trades_df, equity_curve, ax6)
    
    # 7. Underwater Curve
    ax7 = fig.add_subplot(gs[3, :])
    self._plot_underwater_curve(equity_curve, ax7)
    
    # 8. Win/Loss Distribution
    ax8 = fig.add_subplot(gs[4, 0])
    self._plot_win_loss_distribution(trades_df, ax8)
    
    # 9. Trade Duration Analysis
    ax9 = fig.add_subplot(gs[4, 1])
    self._plot_trade_duration_analysis(trades_df, ax9)
    
    # 10. Time of Day Analysis
    ax10 = fig.add_subplot(gs[4, 2])
    self._plot_time_of_day_analysis(trades_df, ax10)
    
    # 11. Cumulative Profit by Strategy Component
    ax11 = fig.add_subplot(gs[5, :])
    self._plot_profit_decomposition(trades_df, ax11)
    
    plt.suptitle(f'Strategy Performance Dashboard - {self.strategy_name}', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def _plot_equity_with_drawdown_shading(self, equity_curve: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot equity curve with drawdown periods shaded."""
    # Calculate drawdown
    rolling_max = equity_curve['equity'].expanding().max()
    drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
    
    # Plot equity curve
    ax.plot(equity_curve.index, equity_curve['equity'], 'b-', linewidth=2, label='Equity')
    
    # Shade drawdown periods
    ax.fill_between(equity_curve.index, 
                   equity_curve['equity'], 
                   rolling_max,
                   where=(equity_curve['equity'] < rolling_max),
                   color='red', alpha=0.3, label='Drawdown')
    
    ax.set_title('Equity Curve with Drawdown Periods')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

def _plot_returns_distribution(self, equity_curve: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot returns distribution with statistics."""
    returns = equity_curve['equity'].pct_change().dropna()
    
    # Plot histogram
    n, bins, patches = ax.hist(returns, bins=50, density=True, alpha=0.7, color='blue')
    
    # Fit normal distribution
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
    
    # Add statistics
    ax.axvline(returns.mean(), color='green', linestyle='--', label=f'Mean: {mu:.3f}')
    ax.axvline(returns.median(), color='orange', linestyle='--', label=f'Median: {returns.median():.3f}')
    
    # Add text statistics
    stats_text = f'Skew: {returns.skew():.2f}\nKurtosis: {returns.kurtosis():.2f}'
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title('Returns Distribution')
    ax.set_xlabel('Daily Returns')
    ax.set_ylabel('Density')
    ax.legend()

def _plot_monthly_returns_heatmap(self, equity_curve: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot monthly returns heatmap."""
    returns = equity_curve['equity'].pct_change().dropna()
    
    # Calculate monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Reshape for heatmap
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    pivot_returns = monthly_returns.groupby([monthly_returns.index.year, 
                                           monthly_returns.index.month]).first().unstack()
    
    # Plot heatmap
    sns.heatmap(pivot_returns * 100, annot=True, fmt='.1f', cmap='RdYlGn', 
                center=0, ax=ax, cbar_kws={'label': 'Return %'})
    
    ax.set_title('Monthly Returns Heatmap (%)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')

def _plot_risk_metrics_table(self, trades_df: pd.DataFrame, 
                           equity_curve: pd.DataFrame, ax: plt.Axes) -> None:
    """Display risk metrics in a table format."""
    # Calculate metrics
    returns = equity_curve['equity'].pct_change().dropna()
    metrics = calculate_advanced_metrics(trades_df.to_dict('records'))
    
    # Create table data
    table_data = [
        ['Metric', 'Value'],
        ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"],
        ['Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"],
        ['Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}"],
        ['Max Drawdown', f"{metrics.get('Max Drawdown (%)', 0):.1f}%"],
        ['VaR (95%)', f"{metrics.get('var_95', 0)*100:.1f}%"],
        ['CVaR (95%)', f"{metrics.get('cvar_95', 0)*100:.1f}%"],
        ['Win Rate', f"{metrics.get('Accuracy (%)', 0):.1f}%"],
        ['Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"],
        ['Expectancy', f"${metrics.get('expectancy', 0):.2f}"],
        ['Stability', f"{metrics.get('stability_of_returns', 0):.2f}"]
    ]
    
    # Remove axis
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Risk Metrics Summary')


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualizations for strategy performance")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name")
    parser.add_argument("--date", type=str, required=True, help="Pull date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, help="Output directory for visualizations")
    parser.add_argument("--input", type=str, help="Input directory containing strategy outputs")
    parser.add_argument("--dashboard", action="store_true", help="Create comprehensive dashboard")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    input_dir = Path(args.input) if args.input else None
    
    visualizer = StrategyVisualizer(output_dir)
    
    if args.dashboard:
        visualizer.create_strategy_dashboard(args.strategy, args.date, input_dir)
    else:
        # Load trade and summary data
        trades_df, summary_df = visualizer.load_trade_data(args.strategy, args.date, input_dir)
        
        if not trades_df.empty:
            # Calculate equity curves
            equity_curves = visualizer.calculate_equity_curve(trades_df)
            
            # Plot equity curve
            visualizer.plot_equity_curve(equity_curves, title=f"{args.strategy} Equity Curve - {args.date}")
            
            # Plot trade distribution
            visualizer.plot_trade_distribution(trades_df)
            
            # Plot correlation heatmap
            visualizer.plot_correlation_heatmap(trades_df)
            
            # Plot performance metrics
            visualizer.plot_performance_metrics(summary_df)