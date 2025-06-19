# src/output/__init__.py
"""
Output Management Module for Backtesting Framework.

This module provides comprehensive output structure management and organization
for backtesting results, ensuring consistent naming conventions, consolidated
analysis packages, and easy navigation of results.
"""

from .output_manager import OutputManager, convert_numpy_types

__all__ = ['OutputManager', 'convert_numpy_types']
