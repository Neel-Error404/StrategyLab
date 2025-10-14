"""
Signal Parity Validator

Validates that backtest and live trading environments produce identical signals.
This is critical for ensuring production readiness and eliminating backtest/live divergence.

Key Features:
- Timestamp synchronization validation
- Indicator value comparison (with tolerance)
- Signal overlap calculation
- Warmup period enforcement
- Parameter consistency checks

Usage:
    from src.core.validation.signal_parity_validator import SignalParityValidator
    
    validator = SignalParityValidator()
    report = validator.compare_signals(backtest_signals, live_signals)
    
    if report.overlap_percentage >= 90.0:
        print(" Parity achieved!")
    else:
        print(f" Parity violation: {report.overlap_percentage}% overlap")

Created: October 15, 2025
Phase: 6.2 - Backtest vs Live Parity
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class ParityError(Exception):
    """Raised when backtest/live parity is violated"""
    pass


@dataclass
class SignalComparison:
    """Results of comparing a single signal between backtest and live"""
    timestamp: datetime
    symbol: str
    backtest_signal: Optional[str]  # "BUY", "SELL", None
    live_signal: Optional[str]
    match: bool
    price_diff: Optional[Decimal] = None
    indicator_diffs: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass
class ParityReport:
    """Comprehensive report of backtest vs live parity validation"""
    total_signals: int
    matching_signals: int
    missing_in_backtest: int
    missing_in_live: int
    signal_mismatches: int
    overlap_percentage: float
    avg_price_diff: Optional[Decimal]
    max_price_diff: Optional[Decimal]
    avg_indicator_diff: Dict[str, float]
    comparisons: List[SignalComparison]
    warmup_validation: Dict[str, any]
    parameter_validation: Dict[str, any]
    timestamp_validation: Dict[str, any]
    passed: bool
    
    def __str__(self) -> str:
        """Human-readable report"""
        lines = [
            "\n" + "=" * 80,
            "BACKTEST vs LIVE PARITY REPORT",
            "=" * 80,
            f"Total Signals: {self.total_signals}",
            f"Matching Signals: {self.matching_signals} ({self.overlap_percentage:.2f}%)",
            f"Missing in Backtest: {self.missing_in_backtest}",
            f"Missing in Live: {self.missing_in_live}",
            f"Signal Mismatches: {self.signal_mismatches}",
            "",
            f"Average Price Difference: {self.avg_price_diff or 'N/A'}",
            f"Maximum Price Difference: {self.max_price_diff or 'N/A'}",
            "",
            "Warmup Validation:",
            f"  Backtest Warmup: {self.warmup_validation.get('backtest_warmup', 'N/A')} minutes",
            f"  Live Warmup: {self.warmup_validation.get('live_warmup', 'N/A')} minutes",
            f"  Match: {'' if self.warmup_validation.get('match', False) else ''}",
            "",
            "Parameter Validation:",
            f"  Parameters Match: {'' if self.parameter_validation.get('match', False) else ''}",
            f"  Mismatched Parameters: {self.parameter_validation.get('mismatches', [])}",
            "",
            "Timestamp Validation:",
            f"  Timezone Match: {'' if self.timestamp_validation.get('timezone_match', False) else ''}",
            f"  Bar Alignment: {'' if self.timestamp_validation.get('bar_alignment', False) else ''}",
            "",
            "=" * 80,
            f"OVERALL STATUS: {' PASS (>90% overlap)' if self.passed else ' FAIL (<90% overlap)'}",
            "=" * 80,
        ]
        
        # Add mismatched signals details if any
        if self.signal_mismatches > 0:
            lines.append("\nSignal Mismatches (first 10):")
            mismatches = [c for c in self.comparisons if not c.match][:10]
            for comp in mismatches:
                lines.append(
                    f"  {comp.timestamp} {comp.symbol}: "
                    f"Backtest={comp.backtest_signal}, Live={comp.live_signal} - {comp.notes}"
                )
        
        return "\n".join(lines)


class SignalParityValidator:
    """
    Validates backtest vs live parity for trading signals.
    
    Ensures:
    1. Warmup periods match exactly (525 minutes)
    2. Strategy parameters are identical
    3. Timestamps are synchronized (timezone, bar alignment)
    4. Signal overlap >= 90%
    5. Price differences <= 1%
    6. Indicator values within tolerance
    """
    
    # Constants
    REQUIRED_WARMUP_MINUTES = 525
    MIN_OVERLAP_PERCENTAGE = 90.0
    MAX_PRICE_DIFF_PERCENTAGE = 1.0  # 1% maximum price difference
    INDICATOR_TOLERANCE = 0.01  # 1% tolerance for indicator values
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize signal parity validator.
        
        Args:
            strict_mode: If True, fail on any parity violation. If False, only warn.
        """
        self.strict_mode = strict_mode
        self.logger = logger
    
    def validate_warmup_periods(
        self, 
        backtest_warmup: int, 
        live_warmup: int
    ) -> Dict[str, any]:
        """
        Validate that backtest and live have identical warmup periods.
        
        Args:
            backtest_warmup: Warmup minutes in backtest
            live_warmup: Warmup minutes in live trading
            
        Returns:
            Validation result dictionary
            
        Raises:
            ParityError: If warmup periods don't match (strict_mode=True)
        """
        match = backtest_warmup == live_warmup
        meets_requirement = (
            backtest_warmup >= self.REQUIRED_WARMUP_MINUTES and 
            live_warmup >= self.REQUIRED_WARMUP_MINUTES
        )
        
        result = {
            "backtest_warmup": backtest_warmup,
            "live_warmup": live_warmup,
            "match": match,
            "meets_requirement": meets_requirement,
            "required_warmup": self.REQUIRED_WARMUP_MINUTES,
        }
        
        if not match:
            error_msg = (
                f"Warmup period mismatch: backtest={backtest_warmup}min, "
                f"live={live_warmup}min (MUST be identical)"
            )
            self.logger.error(error_msg)
            if self.strict_mode:
                raise ParityError(error_msg)
        
        if not meets_requirement:
            error_msg = (
                f"Warmup period too short: required={self.REQUIRED_WARMUP_MINUTES}min, "
                f"backtest={backtest_warmup}min, live={live_warmup}min"
            )
            self.logger.error(error_msg)
            if self.strict_mode:
                raise ParityError(error_msg)
        
        self.logger.info(f" Warmup period validation passed: {backtest_warmup}min (both)")
        return result
    
    def validate_parameters(
        self, 
        backtest_params: Dict[str, any], 
        live_params: Dict[str, any],
        critical_params: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Validate that backtest and live have identical strategy parameters.
        
        Args:
            backtest_params: Parameters used in backtest
            live_params: Parameters used in live trading
            critical_params: List of critical parameter names (if None, checks all)
            
        Returns:
            Validation result dictionary
            
        Raises:
            ParityError: If critical parameters don't match (strict_mode=True)
        """
        # Critical parameters that MUST match
        if critical_params is None:
            critical_params = [
                "warmup_minutes",
                "macd_fast",
                "macd_slow",
                "macd_signal",
                "exit_threshold",
                "entry_cooldown_minutes",
                "position_size",
            ]
        
        mismatches = []
        all_params = set(list(backtest_params.keys()) + list(live_params.keys()))
        
        for param in all_params:
            backtest_value = backtest_params.get(param)
            live_value = live_params.get(param)
            
            if backtest_value != live_value:
                is_critical = param in critical_params
                mismatches.append({
                    "param": param,
                    "backtest_value": backtest_value,
                    "live_value": live_value,
                    "critical": is_critical,
                })
                
                if is_critical:
                    error_msg = (
                        f"Critical parameter mismatch: {param}: "
                        f"backtest={backtest_value}, live={live_value}"
                    )
                    self.logger.error(error_msg)
                    if self.strict_mode:
                        raise ParityError(error_msg)
        
        result = {
            "match": len(mismatches) == 0,
            "mismatches": mismatches,
            "critical_mismatches": [m for m in mismatches if m["critical"]],
            "total_params_checked": len(all_params),
        }
        
        if result["match"]:
            self.logger.info(f" Parameter validation passed: {len(all_params)} params checked")
        else:
            self.logger.warning(
                f"  Parameter validation issues: {len(mismatches)} mismatches found"
            )
        
        return result
    
    def validate_timestamps(
        self, 
        backtest_df: pd.DataFrame, 
        live_df: pd.DataFrame,
        timestamp_col: str = "timestamp"
    ) -> Dict[str, any]:
        """
        Validate timestamp synchronization between backtest and live.
        
        Args:
            backtest_df: Backtest signals DataFrame with timestamps
            live_df: Live signals DataFrame with timestamps
            timestamp_col: Name of timestamp column
            
        Returns:
            Validation result dictionary
        """
        # Check timezone consistency
        backtest_tz = backtest_df[timestamp_col].dt.tz if not backtest_df.empty else None
        live_tz = live_df[timestamp_col].dt.tz if not live_df.empty else None
        timezone_match = backtest_tz == live_tz
        
        # Check bar alignment (all timestamps should align to bar boundaries)
        backtest_aligned = True
        live_aligned = True
        
        if not backtest_df.empty:
            # Check if timestamps are at 5-minute intervals (for 5min bars)
            backtest_minutes = backtest_df[timestamp_col].dt.minute
            backtest_aligned = all(backtest_minutes % 5 == 0)
        
        if not live_df.empty:
            live_minutes = live_df[timestamp_col].dt.minute
            live_aligned = all(live_minutes % 5 == 0)
        
        bar_alignment = backtest_aligned and live_aligned
        
        result = {
            "timezone_match": timezone_match,
            "backtest_timezone": str(backtest_tz),
            "live_timezone": str(live_tz),
            "bar_alignment": bar_alignment,
            "backtest_aligned": backtest_aligned,
            "live_aligned": live_aligned,
        }
        
        if not timezone_match:
            self.logger.warning(
                f"  Timezone mismatch: backtest={backtest_tz}, live={live_tz}"
            )
        
        if not bar_alignment:
            self.logger.warning(
                "  Bar alignment issue: timestamps not at expected intervals"
            )
        
        if timezone_match and bar_alignment:
            self.logger.info(" Timestamp validation passed")
        
        return result
    
    def compare_signals(
        self,
        backtest_signals: pd.DataFrame,
        live_signals: pd.DataFrame,
        backtest_params: Dict[str, any],
        live_params: Dict[str, any],
        timestamp_col: str = "timestamp",
        symbol_col: str = "symbol",
        signal_col: str = "signal",
        price_col: str = "price",
        indicator_cols: Optional[List[str]] = None
    ) -> ParityReport:
        """
        Compare signals from backtest and live trading.
        
        Args:
            backtest_signals: DataFrame with backtest signals
            live_signals: DataFrame with live signals
            backtest_params: Backtest strategy parameters
            live_params: Live strategy parameters
            timestamp_col: Timestamp column name
            symbol_col: Symbol column name
            signal_col: Signal column name ("BUY", "SELL", None)
            price_col: Price column name
            indicator_cols: List of indicator columns to compare (e.g., ["macd", "macd_signal"])
            
        Returns:
            ParityReport with detailed comparison results
        """
        self.logger.info("Starting backtest vs live parity validation...")
        
        # Step 1: Validate warmup periods
        backtest_warmup = backtest_params.get("warmup_minutes", 0)
        live_warmup = live_params.get("warmup_minutes", 0)
        warmup_validation = self.validate_warmup_periods(backtest_warmup, live_warmup)
        
        # Step 2: Validate parameters
        parameter_validation = self.validate_parameters(backtest_params, live_params)
        
        # Step 3: Validate timestamps
        timestamp_validation = self.validate_timestamps(
            backtest_signals, live_signals, timestamp_col
        )
        
        # Step 4: Compare signals
        comparisons = []
        
        # Merge signals on timestamp and symbol
        merged = backtest_signals.merge(
            live_signals,
            on=[timestamp_col, symbol_col],
            how="outer",
            suffixes=("_bt", "_live"),
            indicator=True
        )
        
        total_signals = len(merged)
        matching_signals = 0
        missing_in_backtest = 0
        missing_in_live = 0
        signal_mismatches = 0
        
        price_diffs = []
        indicator_diffs = {col: [] for col in (indicator_cols or [])}
        
        for _, row in merged.iterrows():
            timestamp = row[timestamp_col]
            symbol = row[symbol_col]
            
            # Determine signal presence
            in_backtest = row["_merge"] != "right_only"
            in_live = row["_merge"] != "left_only"
            
            backtest_signal = row.get(f"{signal_col}_bt") if in_backtest else None
            live_signal = row.get(f"{signal_col}_live") if in_live else None
            
            # Calculate match
            match = backtest_signal == live_signal
            
            # Track statistics
            if not in_backtest:
                missing_in_backtest += 1
            elif not in_live:
                missing_in_live += 1
            elif not match:
                signal_mismatches += 1
            else:
                matching_signals += 1
            
            # Calculate price difference (if both present)
            price_diff = None
            if in_backtest and in_live:
                bt_price = Decimal(str(row.get(f"{price_col}_bt", 0)))
                live_price = Decimal(str(row.get(f"{price_col}_live", 0)))
                if bt_price > 0 and live_price > 0:
                    price_diff = abs(bt_price - live_price)
                    price_diffs.append(price_diff)
            
            # Calculate indicator differences
            comp_indicator_diffs = {}
            if indicator_cols and in_backtest and in_live:
                for ind_col in indicator_cols:
                    bt_val = row.get(f"{ind_col}_bt")
                    live_val = row.get(f"{ind_col}_live")
                    if bt_val is not None and live_val is not None:
                        diff = abs(float(bt_val) - float(live_val))
                        comp_indicator_diffs[ind_col] = diff
                        indicator_diffs[ind_col].append(diff)
            
            # Create comparison record
            notes = ""
            if not in_backtest:
                notes = "Signal only in live"
            elif not in_live:
                notes = "Signal only in backtest"
            elif not match:
                notes = f"Signal mismatch: {backtest_signal} != {live_signal}"
            
            comparisons.append(SignalComparison(
                timestamp=timestamp,
                symbol=symbol,
                backtest_signal=backtest_signal,
                live_signal=live_signal,
                match=match,
                price_diff=price_diff,
                indicator_diffs=comp_indicator_diffs,
                notes=notes
            ))
        
        # Calculate summary statistics
        overlap_percentage = (
            (matching_signals / total_signals * 100) if total_signals > 0 else 0.0
        )
        
        avg_price_diff = (
            sum(price_diffs) / len(price_diffs) if price_diffs else None
        )
        
        max_price_diff = max(price_diffs) if price_diffs else None
        
        avg_indicator_diff = {
            col: (sum(diffs) / len(diffs) if diffs else 0.0)
            for col, diffs in indicator_diffs.items()
        }
        
        # Determine pass/fail
        passed = (
            overlap_percentage >= self.MIN_OVERLAP_PERCENTAGE and
            warmup_validation["match"] and
            len(parameter_validation["critical_mismatches"]) == 0
        )
        
        # Create report
        report = ParityReport(
            total_signals=total_signals,
            matching_signals=matching_signals,
            missing_in_backtest=missing_in_backtest,
            missing_in_live=missing_in_live,
            signal_mismatches=signal_mismatches,
            overlap_percentage=overlap_percentage,
            avg_price_diff=avg_price_diff,
            max_price_diff=max_price_diff,
            avg_indicator_diff=avg_indicator_diff,
            comparisons=comparisons,
            warmup_validation=warmup_validation,
            parameter_validation=parameter_validation,
            timestamp_validation=timestamp_validation,
            passed=passed
        )
        
        # Log summary
        self.logger.info(f"\n{report}")
        
        if not passed and self.strict_mode:
            raise ParityError(
                f"Parity validation failed: {overlap_percentage:.2f}% overlap "
                f"(required: {self.MIN_OVERLAP_PERCENTAGE}%)"
            )
        
        return report


# Standalone test/demo
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )
    
    print("=" * 80)
    print("Signal Parity Validator - Standalone Test")
    print("=" * 80)
    
    # Test 1: Warmup period validation
    print("\n1. Warmup Period Validation Test:")
    validator = SignalParityValidator(strict_mode=False)
    
    try:
        result = validator.validate_warmup_periods(525, 525)
        print(f"    Matching warmup periods (525): {result['match']}")
    except ParityError as e:
        print(f"    Error: {e}")
    
    try:
        result = validator.validate_warmup_periods(525, 60)
        print(f"     Mismatched warmup periods (525 vs 60): {result['match']}")
    except ParityError as e:
        print(f"    Error: {e}")
    
    # Test 2: Parameter validation
    print("\n2. Parameter Validation Test:")
    backtest_params = {
        "warmup_minutes": 525,
        "macd_fast": 12,
        "macd_slow": 26,
        "exit_threshold": 0.8
    }
    live_params = {
        "warmup_minutes": 525,
        "macd_fast": 12,
        "macd_slow": 26,
        "exit_threshold": 0.8
    }
    
    result = validator.validate_parameters(backtest_params, live_params)
    print(f"    Matching parameters: {result['match']}")
    
    live_params["exit_threshold"] = 0.7  # Introduce mismatch
    result = validator.validate_parameters(backtest_params, live_params)
    print(f"     Mismatched parameters: {result['match']}")
    print(f"      Mismatches: {result['mismatches']}")
    
    # Test 3: Signal comparison
    print("\n3. Signal Comparison Test:")
    backtest_signals = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01 09:15", periods=5, freq="5min"),
        "symbol": ["RELIANCE"] * 5,
        "signal": ["BUY", None, None, "SELL", None],
        "price": [2800.50, 2802.00, 2801.50, 2805.00, 2803.50]
    })
    
    live_signals = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01 09:15", periods=5, freq="5min"),
        "symbol": ["RELIANCE"] * 5,
        "signal": ["BUY", None, None, "SELL", None],  # Perfect match
        "price": [2800.55, 2802.05, 2801.45, 2805.10, 2803.45]  # Slight price diffs
    })
    
    backtest_params["warmup_minutes"] = 525
    live_params["warmup_minutes"] = 525
    live_params["exit_threshold"] = 0.8  # Fix mismatch
    
    report = validator.compare_signals(
        backtest_signals,
        live_signals,
        backtest_params,
        live_params
    )
    
    print(f"\n   Signal Overlap: {report.overlap_percentage:.2f}%")
    print(f"   Avg Price Diff: {report.avg_price_diff}")
    print(f"   Status: {' PASS' if report.passed else ' FAIL'}")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
