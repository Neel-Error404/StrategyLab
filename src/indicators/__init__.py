"""
Indicator library package.

Exposes helper utilities that compute technical indicators and provide
metadata (warm-up periods, default sensitivities) for the declarative
strategy pipeline.
"""

from .indicator_catalog import INDICATOR_FUNCTIONS  # noqa: F401
from .library import IndicatorLibrary  # noqa: F401
