# analysis/portfolio_analyzer.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta

class PortfolioAnalyzer:
    """
    Analyzes strategy performance across multiple tickers and provides portfolio-level insights.
    """
    def __init__(self, strategy_name, pull_date, output_dir=None):
        self.strategy_name = strategy_name
        self.pull_date = pull_date
        self.output_dir = output_dir or Path(f"Backtester/Strat_out/{strategy_name}/{pull_date}")
        self.summary_df = None
        self.trades_df = None
        self.tickers = []
        self.equity_curves = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load summary and trade data for all tickers"""
        summary_file = self.output_dir / f"{self.pull_date}_Summary.csv"
        if summary_file.exists():
            self.summary_df = pd.read_csv(summary_file)
            self.tickers = self.summary_df['Ticker'].unique().tolist()
        
        # Load all trade files and combine
        trade_files = list(self.output_dir.glob("*_Trades_*.csv"))
        all_trades = []
        
        for file in trade_files:
            ticker = file.name.split("_")[0]
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    df['Ticker'] = ticker  # Ensure ticker is included
                    all_trades.append(df)
            except Exception as e:
                print(f"Error loading trade file for {ticker}: {e}")
        
        if all_trades:
            self.trades_df = pd.concat(all_trades, ignore_index=True)
            
            # Convert date columns to datetime
            for col in ['Entry Time', 'Exit Time', 'High Time', 'Low Time']:
                if col in self.trades_df.columns:
                    self.trades_df[col] = pd.to_datetime(self.trades_df[col])
        else:
            print("No trade data found.")
    
    def generate_portfolio_summary(self):
        """Generate overall portfolio performance summary"""
        if self.summary_df is None:
            print("No summary data available.")
            return
        
        # Calculate portfolio-level metrics
        total_trades = self.summary_df['Total Trades'].sum()
        total_wins = self.summary_df['Wins'].sum()
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        # Weighted average profit
        weighted_profit = np.average(
            self.summary_df['Average Profit (%)'], 
            weights=self.summary_df['Total Trades'],
            axis=0
        )
        
        # Find best and worst performing tickers
        best_ticker = self.summary_df.loc[self.summary_df['Average Profit (%)'].idxmax()]
        worst_ticker = self.summary_df.loc[self.summary_df['Average Profit (%)'].idxmin()]
        
        # Portfolio statistics
        portfolio_stats = {
            'Total Tickers': len(self.tickers),
            'Total Trades': total_trades,
            'Total Wins': total_wins,
            'Portfolio Win Rate (%)': round(win_rate, 2),
            'Weighted Average Profit (%)': round(weighted_profit, 2),
            'Best Performing Ticker': {
                'Ticker': best_ticker['Ticker'],
                'Average Profit (%)': round(best_ticker['Average Profit (%)'], 2),
                'Win Rate (%)': round((best_ticker['Wins'] / best_ticker['Total Trades'] * 100), 2) if best_ticker['Total Trades'] > 0 else 0,
                'Total Trades': best_ticker['Total Trades']
            },
            'Worst Performing Ticker': {
                'Ticker': worst_ticker['Ticker'],
                'Average Profit (%)': round(worst_ticker['Average Profit (%)'], 2),
                'Win Rate (%)': round((worst_ticker['Wins'] / worst_ticker['Total Trades'] * 100), 2) if worst_ticker['Total Trades'] > 0 else 0,
                'Total Trades': worst_ticker['Total Trades']
            }
        }
        
        return portfolio_stats
    
    def calculate_equity_curves(self, initial_capital=100000, position_size_pct=0.1):
        """Calculate equity curves for portfolio and individual tickers"""
        if self.trades_df is None or self.trades_df.empty:
            print("No trade data available.")
            return
        
        # Sort trades by entry time
        self.trades_df = self.trades_df.sort_values('Entry Time')
        
        # Initialize equity dictionary with starting capital
        equity = {'Portfolio': [initial_capital]}
        equity_timestamps = {'Portfolio': [self.trades_df['Entry Time'].min()]}
        
        # Initialize equity for each ticker
        for ticker in self.tickers:
            equity[ticker] = [initial_capital]
            equity_timestamps[ticker] = [self.trades_df['Entry Time'].min()]
        
        # Calculate running equity for portfolio
        portfolio_capital = initial_capital
        
        # Process each trade chronologically
        for idx, trade in self.trades_df.iterrows():
            ticker = trade['Ticker']
            position_size = initial_capital * position_size_pct
            profit_amount = position_size * (trade['Profit (%)'] / 100)
            
            # Update portfolio equity
            portfolio_capital += profit_amount
            equity['Portfolio'].append(portfolio_capital)
            equity_timestamps['Portfolio'].append(trade['Exit Time'])
            
            # Update ticker equity (reset for each ticker to see performance in isolation)
            ticker_capital = equity[ticker][-1]
            ticker_capital += profit_amount
            equity[ticker].append(ticker_capital)
            equity_timestamps[ticker].append(trade['Exit Time'])
        
        # Convert to DataFrame for easier plotting
        for key in equity:
            self.equity_curves[key] = pd.DataFrame({
                'timestamp': equity_timestamps[key],
                'equity': equity[key]
            })
            
        return self.equity_curves
    
    def plot_portfolio_performance(self, save_path=None):
        """Plot various portfolio performance visualizations"""
        if self.summary_df is None or self.trades_df is None:
            print("No data available for analysis.")
            return
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Equity curves
        if not self.equity_curves:
            self.calculate_equity_curves()
            
        ax1 = plt.subplot(2, 2, 1)
        for ticker, curve in self.equity_curves.items():
            if ticker == 'Portfolio':
                ax1.plot(curve['timestamp'], curve['equity'], linewidth=2, label=ticker)
            
        ax1.set_title('Portfolio Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Performance by ticker
        ax2 = plt.subplot(2, 2, 2)
        ticker_perf = self.summary_df.sort_values('Average Profit (%)', ascending=False)
        sns.barplot(x='Ticker', y='Average Profit (%)', data=ticker_perf.head(10), ax=ax2)
        ax2.set_title('Top 10 Tickers by Average Profit (%)')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # 3. Trade distribution
        ax3 = plt.subplot(2, 2, 3)
        self.trades_df['hour'] = self.trades_df['Entry Time'].dt.hour
        trade_hours = self.trades_df.groupby('hour').size()
        trade_hours.plot(kind='bar', ax=ax3)
        ax3.set_title('Trade Distribution by Hour')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Number of Trades')
        
        # 4. Win/Loss by ticker
        ax4 = plt.subplot(2, 2, 4)
        win_rate_by_ticker = self.summary_df.copy()
        win_rate_by_ticker['Win Rate (%)'] = (win_rate_by_ticker['Wins'] / win_rate_by_ticker['Total Trades'] * 100)
        win_rate_by_ticker = win_rate_by_ticker.sort_values('Win Rate (%)', ascending=False)
        sns.barplot(x='Ticker', y='Win Rate (%)', data=win_rate_by_ticker.head(10), ax=ax4)
        ax4.set_title('Top 10 Tickers by Win Rate (%)')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Portfolio analysis saved to {save_path}")
        else:
            plt.show()
            
    def calculate_correlations(self):
        """Calculate correlation between ticker performances"""
        if self.trades_df is None or self.trades_df.empty:
            print("No trade data available.")
            return
            
        # Pivot trades to get ticker performance over time
        daily_returns = {}
        
        for ticker in self.tickers:
            ticker_trades = self.trades_df[self.trades_df['Ticker'] == ticker]
            if ticker_trades.empty:
                continue
                
            # Group by day and calculate daily returns
            ticker_trades['date'] = ticker_trades['Exit Time'].dt.date
            daily_pnl = ticker_trades.groupby('date')['Profit (%)'].sum().reset_index()
            daily_returns[ticker] = daily_pnl.set_index('date')['Profit (%)']
        
        # Create a DataFrame with all tickers' daily returns
        returns_df = pd.DataFrame(daily_returns)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        return corr_matrix
    
    def plot_correlation_heatmap(self, save_path=None):
        """Plot correlation heatmap between ticker performances"""
        corr_matrix = self.calculate_correlations()
        
        if corr_matrix is None or corr_matrix.empty:
            print("No correlation data available.")
            return
            
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=.5, fmt=".2f")
        plt.title('Correlation Between Ticker Performances')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Correlation heatmap saved to {save_path}")
        else:
            plt.show()
            
    def filter_tickers(self, min_trades=5, min_win_rate=50, min_profit=0):
        """Filter tickers based on performance criteria"""
        if self.summary_df is None:
            print("No summary data available.")
            return
            
        filtered_df = self.summary_df.copy()
        
        # Add win rate column
        filtered_df['Win Rate (%)'] = (filtered_df['Wins'] / filtered_df['Total Trades'] * 100)
        
        # Apply filters
        filtered_df = filtered_df[
            (filtered_df['Total Trades'] >= min_trades) & 
            (filtered_df['Win Rate (%)'] >= min_win_rate) &
            (filtered_df['Average Profit (%)'] >= min_profit)
        ]
        
        return filtered_df
    
    def generate_analysis_report(self, output_file=None):
        """Generate comprehensive analysis report in JSON format"""
        portfolio_summary = self.generate_portfolio_summary()

        # Check if trades_df exists and is not empty
        if self.trades_df is None or self.trades_df.empty:
            report = {
                'Strategy': self.strategy_name,
                'Date': self.pull_date,
                'Status': 'No trade data available',
                'Analysis Timestamp': datetime.now().isoformat()
            }

            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=4)
                print(f"Empty analysis report saved to {output_file}")

            return report

        # Calculate additional metrics
        trade_duration_stats = {
            'Mean (min)': float(self.trades_df['Trade Duration (min)'].mean()),
            'Median (min)': float(self.trades_df['Trade Duration (min)'].median()),
            'Min (min)': float(self.trades_df['Trade Duration (min)'].min()),
            'Max (min)': float(self.trades_df['Trade Duration (min)'].max()),
        }

        profit_distribution = {
            'Mean (%)': float(self.trades_df['Profit (%)'].mean()),
            'Median (%)': float(self.trades_df['Profit (%)'].median()),
            'Std Dev (%)': float(self.trades_df['Profit (%)'].std()),
            'Min (%)': float(self.trades_df['Profit (%)'].min()),
            'Max (%)': float(self.trades_df['Profit (%)'].max()),
        }

        # Trade type analysis
        trade_types = {k: int(v) for k, v in self.trades_df['Trade Type'].value_counts().to_dict().items()}
        buy_trades = self.trades_df[self.trades_df['Trade Type'] == 'Buy']
        sell_trades = self.trades_df[self.trades_df['Trade Type'] == 'Sell']

        trade_type_analysis = {
            'Buy Trades': {
                'Count': int(len(buy_trades)),
                'Win Rate (%)': float((buy_trades['Profit (%)'] > 0).mean() * 100) if not buy_trades.empty else 0,
                'Average Profit (%)': float(buy_trades['Profit (%)'].mean()) if not buy_trades.empty else 0,
            },
            'Sell Trades': {
                'Count': int(len(sell_trades)),
                'Win Rate (%)': float((sell_trades['Profit (%)'] > 0).mean() * 100) if not sell_trades.empty else 0,
                'Average Profit (%)': float(sell_trades['Profit (%)'].mean()) if not sell_trades.empty else 0,
            }
        }

        # Ticker filters
        top_performers = self.filter_tickers(min_win_rate=60, min_profit=0.5)
        top_tickers = top_performers['Ticker'].tolist() if not top_performers.empty else []

        # Compile the report
        report = {
            'Strategy': self.strategy_name,
            'Date': self.pull_date,
            'Portfolio Summary': portfolio_summary,
            'Trade Duration Statistics': trade_duration_stats,
            'Profit Distribution': profit_distribution,
            'Trade Type Analysis': trade_type_analysis,
            'Top Performing Tickers': top_tickers,
            'Analysis Timestamp': datetime.now().isoformat()
        }

        # Helper function to convert NumPy types to Python types
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Convert all NumPy types to Python types
        report = convert_numpy_types(report)

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"Analysis report saved to {output_file}")

        return report