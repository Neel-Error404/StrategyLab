"""
Visualization Module
====================

Purpose:
--------
Generate visualizations for baseline metrics, MAE/MFE analysis, and optimization results.

Visualizations:
---------------
- MAE/MFE scatter plot
- Exit efficiency distribution
- Capture ratio by trade duration
- Equity curve
- Drawdown chart
- Win/Loss distribution

Author: Strategy Optimization Pipeline
Date: 2025-10-04
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_mae_mfe_scatter(enhanced_data: pd.DataFrame,
                         save_path: str = None,
                         title: str = "MAE vs MFE Scatter Plot"):
    """
    Create scatter plot of MAE vs MFE with color-coded efficiency scores.

    Parameters:
    -----------
    enhanced_data : pd.DataFrame
        Trade data with MAE/MFE columns
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """

    # Filter to valid data
    valid_data = enhanced_data.dropna(subset=['MFE_pct', 'MAE_pct', 'Exit_Efficiency_Score'])

    if len(valid_data) == 0:
        print("⚠️  No valid MAE/MFE data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create scatter plot with color-coded efficiency scores
    scatter = ax.scatter(
        valid_data['MFE_pct'],
        valid_data['MAE_pct'],
        c=valid_data['Exit_Efficiency_Score'],
        cmap='RdYlGn',
        s=50,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Exit Efficiency Score', rotation=270, labelpad=20)

    # Add diagonal reference line (MAE = MFE)
    max_val = max(valid_data['MFE_pct'].max(), valid_data['MAE_pct'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='MAE = MFE')

    # Add efficiency zones
    ax.axhline(y=valid_data['MAE_pct'].median(), color='orange', linestyle=':', alpha=0.5, label=f'Median MAE: {valid_data["MAE_pct"].median():.2f}%')
    ax.axvline(x=valid_data['MFE_pct'].median(), color='blue', linestyle=':', alpha=0.5, label=f'Median MFE: {valid_data["MFE_pct"].median():.2f}%')

    ax.set_xlabel('Maximum Favorable Excursion (MFE) %', fontsize=12)
    ax.set_ylabel('Maximum Adverse Excursion (MAE) %', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")

    plt.close()


def plot_exit_efficiency_distribution(enhanced_data: pd.DataFrame,
                                      save_path: str = None,
                                      title: str = "Exit Efficiency Score Distribution"):
    """
    Create histogram of exit efficiency scores.

    Parameters:
    -----------
    enhanced_data : pd.DataFrame
        Trade data with Exit_Efficiency_Score column
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """

    valid_data = enhanced_data.dropna(subset=['Exit_Efficiency_Score'])

    if len(valid_data) == 0:
        print("⚠️  No valid exit efficiency data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create histogram
    ax.hist(valid_data['Exit_Efficiency_Score'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')

    # Add vertical lines for zones
    ax.axvline(x=70, color='green', linestyle='--', linewidth=2, label='Excellent (>70)')
    ax.axvline(x=50, color='orange', linestyle='--', linewidth=2, label='Good (50-70)')
    ax.axvline(x=30, color='red', linestyle='--', linewidth=2, label='Poor (30-50)')

    # Add mean and median lines
    mean_score = valid_data['Exit_Efficiency_Score'].mean()
    median_score = valid_data['Exit_Efficiency_Score'].median()

    ax.axvline(x=mean_score, color='blue', linestyle='-', linewidth=2, label=f'Mean: {mean_score:.2f}')
    ax.axvline(x=median_score, color='purple', linestyle='-', linewidth=2, label=f'Median: {median_score:.2f}')

    ax.set_xlabel('Exit Efficiency Score', fontsize=12)
    ax.set_ylabel('Number of Trades', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")

    plt.close()


def plot_capture_ratio_by_duration(enhanced_data: pd.DataFrame,
                                   save_path: str = None,
                                   title: str = "MFE Capture Ratio by Trade Duration"):
    """
    Create scatter plot of capture ratio vs trade duration.

    Parameters:
    -----------
    enhanced_data : pd.DataFrame
        Trade data with MFE_Capture_Ratio and duration columns
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """

    # Calculate duration if not present
    if 'duration_hours' not in enhanced_data.columns:
        enhanced_data_copy = enhanced_data.copy()
        enhanced_data_copy['Entry Time'] = pd.to_datetime(enhanced_data_copy['Entry Time'])
        enhanced_data_copy['Exit Time'] = pd.to_datetime(enhanced_data_copy['Exit Time'])
        enhanced_data_copy['duration_hours'] = (
            enhanced_data_copy['Exit Time'] - enhanced_data_copy['Entry Time']
        ).dt.total_seconds() / 3600
    else:
        enhanced_data_copy = enhanced_data

    valid_data = enhanced_data_copy.dropna(subset=['MFE_Capture_Ratio', 'duration_hours'])

    if len(valid_data) == 0:
        print("⚠️  No valid capture ratio/duration data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create scatter plot
    ax.scatter(
        valid_data['duration_hours'],
        valid_data['MFE_Capture_Ratio'],
        c=valid_data['Exit_Efficiency_Score'],
        cmap='RdYlGn',
        s=50,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    # Add colorbar
    scatter = ax.scatter(
        valid_data['duration_hours'],
        valid_data['MFE_Capture_Ratio'],
        c=valid_data['Exit_Efficiency_Score'],
        cmap='RdYlGn',
        s=50,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Exit Efficiency Score', rotation=270, labelpad=20)

    # Add reference lines
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='70% Capture (Target)')
    ax.axhline(y=valid_data['MFE_Capture_Ratio'].median(), color='orange', linestyle=':', alpha=0.5,
               label=f'Median: {valid_data["MFE_Capture_Ratio"].median():.1f}%')

    ax.set_xlabel('Trade Duration (hours)', fontsize=12)
    ax.set_ylabel('MFE Capture Ratio (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")

    plt.close()


def plot_equity_curve(trade_data: pd.DataFrame,
                     initial_capital: float = 100000,
                     save_path: str = None,
                     title: str = "Equity Curve"):
    """
    Create equity curve plot.

    Parameters:
    -----------
    trade_data : pd.DataFrame
        Trade data with percentage_return column
    initial_capital : float
        Starting capital
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """

    if 'percentage_return' not in trade_data.columns:
        print("⚠️  No percentage_return column found")
        return

    # Calculate equity curve
    returns = trade_data['percentage_return'].values
    equity = (1 + returns / 100).cumprod() * initial_capital

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Equity curve
    ax1.plot(equity, linewidth=2, color='steelblue')
    ax1.fill_between(range(len(equity)), equity, initial_capital, alpha=0.3, color='steelblue')
    ax1.axhline(y=initial_capital, color='red', linestyle='--', label=f'Initial Capital: ${initial_capital:,.0f}')

    # Calculate and show final equity
    final_equity = equity[-1] if len(equity) > 0 else initial_capital
    total_return = ((final_equity - initial_capital) / initial_capital) * 100

    ax1.axhline(y=final_equity, color='green', linestyle='--', label=f'Final Equity: ${final_equity:,.0f} (+{total_return:.2f}%)')

    ax1.set_ylabel('Equity ($)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Drawdown chart
    running_max = pd.Series(equity).expanding().max()
    drawdown = (equity - running_max) / running_max * 100

    ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.5)
    ax2.plot(drawdown, linewidth=1, color='darkred')

    max_dd = drawdown.min()
    ax2.axhline(y=max_dd, color='darkred', linestyle='--', label=f'Max Drawdown: {max_dd:.2f}%')

    ax2.set_xlabel('Trade Number', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")

    plt.close()


def plot_win_loss_distribution(trade_data: pd.DataFrame,
                               save_path: str = None,
                               title: str = "Win/Loss Distribution"):
    """
    Create histogram of win/loss distribution.

    Parameters:
    -----------
    trade_data : pd.DataFrame
        Trade data with percentage_return column
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """

    if 'percentage_return' not in trade_data.columns:
        print("⚠️  No percentage_return column found")
        return

    returns = trade_data['percentage_return']
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Wins histogram
    if len(wins) > 0:
        ax1.hist(wins, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax1.axvline(x=wins.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {wins.mean():.2f}%')
        ax1.axvline(x=wins.median(), color='lime', linestyle='--', linewidth=2, label=f'Median: {wins.median():.2f}%')
        ax1.set_xlabel('Winning Trade Return (%)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Winning Trades (n={len(wins):,})', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

    # Losses histogram
    if len(losses) > 0:
        ax2.hist(losses, bins=30, color='red', alpha=0.7, edgecolor='black')
        ax2.axvline(x=losses.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {losses.mean():.2f}%')
        ax2.axvline(x=losses.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {losses.median():.2f}%')
        ax2.set_xlabel('Losing Trade Return (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Losing Trades (n={len(losses):,})', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")

    plt.close()


def create_baseline_visualizations(enhanced_data: pd.DataFrame,
                                   output_dir: str,
                                   prefix: str = "baseline"):
    """
    Create all baseline visualizations.

    Parameters:
    -----------
    enhanced_data : pd.DataFrame
        Trade data with MAE/MFE and metrics
    output_dir : str
        Directory to save visualizations
    prefix : str
        Filename prefix
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 Generating visualizations...")

    # 1. MAE/MFE Scatter
    plot_mae_mfe_scatter(
        enhanced_data,
        save_path=str(output_path / f"{prefix}_mae_mfe_scatter.png"),
        title=f"{prefix.upper()}: MAE vs MFE"
    )

    # 2. Exit Efficiency Distribution
    plot_exit_efficiency_distribution(
        enhanced_data,
        save_path=str(output_path / f"{prefix}_exit_efficiency_distribution.png"),
        title=f"{prefix.upper()}: Exit Efficiency Distribution"
    )

    # 3. Capture Ratio by Duration
    plot_capture_ratio_by_duration(
        enhanced_data,
        save_path=str(output_path / f"{prefix}_capture_ratio_by_duration.png"),
        title=f"{prefix.upper()}: MFE Capture Ratio by Duration"
    )

    # 4. Equity Curve
    plot_equity_curve(
        enhanced_data,
        save_path=str(output_path / f"{prefix}_equity_curve.png"),
        title=f"{prefix.upper()}: Equity Curve"
    )

    # 5. Win/Loss Distribution
    plot_win_loss_distribution(
        enhanced_data,
        save_path=str(output_path / f"{prefix}_win_loss_distribution.png"),
        title=f"{prefix.upper()}: Win/Loss Distribution"
    )

    print(f"   ✓ All visualizations saved to: {output_dir}")
