"""
Chart styling utilities for consistent visualization appearance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import logging


class ChartStyler:
    """Provides consistent styling across all visualization modules."""
    
    def __init__(self):
        self.logger = logging.getLogger("ChartStyler")
        
    @staticmethod
    def apply_global_style():
        """Apply global matplotlib and seaborn styling."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Set default figure parameters
        plt.rcParams.update({
            'figure.figsize': (16, 12),
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
    
    @staticmethod
    def get_color_palette() -> Dict[str, str]:
        """Get consistent color palette for charts."""
        return {
            'profit': 'green',
            'loss': 'red', 
            'neutral': 'blue',
            'background': 'lightgray',
            'grid': 'gray',
            'text': 'black',
            'accent': 'orange'
        }
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format value as percentage with consistent decimals."""
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def format_currency(value: float, decimals: int = 2) -> str:
        """Format value as currency with consistent decimals."""
        return f"₹{value:.{decimals}f}"
    
    @staticmethod
    def apply_grid_style(ax, alpha: float = 0.3):
        """Apply consistent grid styling to axes."""
        ax.grid(True, alpha=alpha, linestyle='-', linewidth=0.5)
    
    @staticmethod
    def add_chart_annotations(ax, title: str, subtitle: Optional[str] = None):
        """Add consistent title and subtitle formatting."""
        ax.set_title(title, fontweight='bold', pad=20)
        if subtitle:
            ax.text(0.5, 0.98, subtitle, transform=ax.transAxes, 
                   ha='center', va='top', style='italic', fontsize=9)
