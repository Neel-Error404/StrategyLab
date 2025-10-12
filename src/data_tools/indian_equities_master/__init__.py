"""
Yahoo Finance powered Indian equities master dataset pipeline.

This package exposes a CLI entry point (`python -m src.data_tools.indian_equities_master.cli update`)
that downloads and normalises the daily master file into `data/indian_equities_master.csv`.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
