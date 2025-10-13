"""
Generic Analysis Modules
========================

Reusable modules for strategy-agnostic analysis.

Available modules:
- config_loader: Load and validate YAML configuration
- data_loader: Load trade and base data from config
- metrics_calculator: Calculate performance metrics
- visualizer: Generate charts and plots
"""

from .config_loader import (
    load_config,
    resolve_paths,
    get_analysis_config,
    get_module_spec,
    get_output_dir,
    get_report_dir,
    resolve_artifact_path,
)
from .data_loader import load_trades, load_base_data

__all__ = [
    'load_config',
    'resolve_paths',
    'get_analysis_config',
    'get_module_spec',
    'get_output_dir',
    'get_report_dir',
    'resolve_artifact_path',
    'load_trades',
    'load_base_data'
]
