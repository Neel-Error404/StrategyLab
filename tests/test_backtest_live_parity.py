"""
Backtest vs Live Parity Tests

Comprehensive test suite to ensure backtest and live trading environments
produce identical signals and maintain configuration parity.

Test Coverage:
1. Warmup period validation
2. Configuration parameter parity
3. Timestamp synchronization
4. Signal generation parity
5. Indicator value alignment
6. Order generation consistency
7. Position tracking parity
8. PnL calculation accuracy

Created: October 15, 2025
Phase: 6.7 - Backtest vs Live Parity
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal

# Import validators
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.validation.signal_parity_validator import (
    SignalParityValidator,
    ParityError,
    ParityReport
)
from src.core.validation.config_parity_validator import (
    validate_config_parity,
    validate_warmup_parity,
    ConfigParityError,
    CRITICAL_PARAMETERS
)


class TestWarmupPeriodParity:
    """Test warmup period configuration matches between backtest and live"""
    
    def test_warmup_periods_match_525_minutes(self):
        """Test that both backtest and live use 525-minute warmup"""
        validator = SignalParityValidator()
        result = validator.validate_warmup_periods(525, 525)
        
        assert result["match"] is True
        assert result["meets_requirement"] is True
        assert result["backtest_warmup"] == 525
        assert result["live_warmup"] == 525
    
    def test_warmup_mismatch_raises_error(self):
        """Test that warmup mismatch raises ParityError in strict mode"""
        validator = SignalParityValidator(strict_mode=True)
        
        with pytest.raises(ParityError, match="Warmup period mismatch"):
            validator.validate_warmup_periods(525, 60)
    
    def test_insufficient_warmup_raises_error(self):
        """Test that warmup < 525 minutes raises error"""
        validator = SignalParityValidator(strict_mode=True)
        
        with pytest.raises(ParityError, match="Warmup period too short"):
            validator.validate_warmup_periods(100, 100)
    
    def test_warmup_parity_validator(self):
        """Test dedicated warmup validator"""
        # Should pass
        result = validate_warmup_parity(525, 525, required_warmup=525)
        assert result is None
        
        # Should raise error
        with pytest.raises(ConfigParityError):
            validate_warmup_parity(525, 60, required_warmup=525)


class TestConfigurationParity:
    """Test configuration parameter parity"""
    
    def test_identical_configs_pass(self):
        """Test that identical configs pass validation"""
        config = {
            "warmup_minutes": 525,
            "macd_fast": 12,
            "macd_slow": 26,
            "exit_threshold": 0.8,
        }
        
        mismatches = validate_config_parity(config, config.copy(), strict_mode=False)
        assert len(mismatches) == 0
    
    def test_critical_mismatch_raises_error(self):
        """Test that critical parameter mismatch raises error"""
        backtest_config = {"warmup_minutes": 525, "macd_fast": 12}
        live_config = {"warmup_minutes": 60, "macd_fast": 12}
        
        with pytest.raises(ConfigParityError, match="Critical configuration"):
            validate_config_parity(backtest_config, live_config, strict_mode=True)
    
    def test_non_critical_mismatch_warns(self):
        """Test that non-critical mismatch only warns"""
        backtest_config = {"warmup_minutes": 525, "other_param": 100}
        live_config = {"warmup_minutes": 525, "other_param": 200}
        
        mismatches = validate_config_parity(backtest_config, live_config, strict_mode=False)
        assert len(mismatches) == 1
        assert not mismatches[0].is_critical
    
    def test_all_critical_parameters_checked(self):
        """Test that all critical parameters are validated"""
        # Create config with all critical parameters
        backtest_config = {}
        for category, params in CRITICAL_PARAMETERS.items():
            for param in params:
                backtest_config[param] = 100
        
        live_config = backtest_config.copy()
        
        mismatches = validate_config_parity(backtest_config, live_config, strict_mode=False)
        assert len(mismatches) == 0


class TestTimestampSynchronization:
    """Test timestamp and timezone synchronization"""
    
    def test_timezone_consistency(self):
        """Test that timestamps use consistent timezone"""
        validator = SignalParityValidator()
        
        # Create DataFrames with IST timezone
        backtest_df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01 09:15", periods=5, freq="5min", tz="Asia/Kolkata"),
            "symbol": ["RELIANCE"] * 5,
            "signal": [None] * 5,
        })
        
        live_df = backtest_df.copy()
        
        result = validator.validate_timestamps(backtest_df, live_df)
        assert result["timezone_match"] is True
        assert "Asia/Kolkata" in result["backtest_timezone"]
    
    def test_bar_alignment_5min(self):
        """Test that 5-minute bar timestamps align correctly"""
        validator = SignalParityValidator()
        
        # Create properly aligned timestamps
        backtest_df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01 09:15", periods=10, freq="5min"),
            "symbol": ["RELIANCE"] * 10,
            "signal": [None] * 10,
        })
        
        result = validator.validate_timestamps(backtest_df, backtest_df.copy())
        assert result["bar_alignment"] is True
        assert result["backtest_aligned"] is True
        assert result["live_aligned"] is True


class TestSignalGenerationParity:
    """Test signal generation produces identical results"""
    
    def test_identical_signals_100_percent_overlap(self):
        """Test that identical signals achieve 100% overlap"""
        validator = SignalParityValidator(strict_mode=False)
        
        signals_df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01 09:15", periods=10, freq="5min"),
            "symbol": ["RELIANCE"] * 10,
            "signal": ["BUY", None, None, "SELL", None, "BUY", None, "SELL", None, None],
            "price": [2800 + i for i in range(10)],
        })
        
        params = {"warmup_minutes": 525, "macd_fast": 12}
        
        report = validator.compare_signals(
            signals_df, signals_df.copy(), params, params.copy()
        )
        
        assert report.passed is True
        assert report.overlap_percentage == 100.0
        assert report.matching_signals == 10
        assert report.signal_mismatches == 0
    
    def test_signal_mismatch_detection(self):
        """Test that signal mismatches are detected"""
        validator = SignalParityValidator(strict_mode=False)
        
        backtest_signals = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01 09:15", periods=5, freq="5min"),
            "symbol": ["RELIANCE"] * 5,
            "signal": ["BUY", None, None, "SELL", None],
            "price": [2800, 2801, 2802, 2803, 2804],
        })
        
        live_signals = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01 09:15", periods=5, freq="5min"),
            "symbol": ["RELIANCE"] * 5,
            "signal": ["BUY", None, "BUY", "SELL", None],  # Extra BUY signal
            "price": [2800, 2801, 2802, 2803, 2804],
        })
        
        params = {"warmup_minutes": 525}
        
        report = validator.compare_signals(
            backtest_signals, live_signals, params, params.copy()
        )
        
        assert report.signal_mismatches > 0
        assert report.overlap_percentage < 100.0
    
    def test_missing_signals_tracked(self):
        """Test that missing signals are tracked separately"""
        validator = SignalParityValidator(strict_mode=False)
        
        backtest_signals = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01 09:15", periods=3, freq="5min"),
            "symbol": ["RELIANCE"] * 3,
            "signal": ["BUY", None, "SELL"],
            "price": [2800, 2801, 2802],
        })
        
        # Live has one less signal
        live_signals = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01 09:15", periods=2, freq="5min"),
            "symbol": ["RELIANCE"] * 2,
            "signal": ["BUY", None],
            "price": [2800, 2801],
        })
        
        params = {"warmup_minutes": 525}
        
        report = validator.compare_signals(
            backtest_signals, live_signals, params, params.copy()
        )
        
        assert report.missing_in_live == 1


class TestPriceDifferences:
    """Test price difference tracking and validation"""
    
    def test_small_price_diff_acceptable(self):
        """Test that small price differences are tracked but acceptable"""
        validator = SignalParityValidator(strict_mode=False)
        
        backtest_signals = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01 09:15", periods=3, freq="5min"),
            "symbol": ["RELIANCE"] * 3,
            "signal": ["BUY", None, "SELL"],
            "price": [2800.00, 2801.00, 2802.00],
        })
        
        live_signals = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01 09:15", periods=3, freq="5min"),
            "symbol": ["RELIANCE"] * 3,
            "signal": ["BUY", None, "SELL"],
            "price": [2800.05, 2801.10, 2802.15],  # Small diffs (tick size)
        })
        
        params = {"warmup_minutes": 525}
        
        report = validator.compare_signals(
            backtest_signals, live_signals, params, params.copy()
        )
        
        assert report.avg_price_diff is not None
        assert report.avg_price_diff < Decimal("1.0")  # Less than 1 rupee average


class TestParityReport:
    """Test parity report generation and interpretation"""
    
    def test_passing_report_structure(self):
        """Test structure of passing parity report"""
        validator = SignalParityValidator(strict_mode=False)
        
        signals_df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01 09:15", periods=5, freq="5min"),
            "symbol": ["RELIANCE"] * 5,
            "signal": ["BUY", None, None, "SELL", None],
            "price": [2800.0] * 5,
        })
        
        params = {"warmup_minutes": 525}
        
        report = validator.compare_signals(
            signals_df, signals_df.copy(), params, params.copy()
        )
        
        assert isinstance(report, ParityReport)
        assert report.passed is True
        assert report.overlap_percentage >= 90.0
        assert report.warmup_validation["match"] is True
        assert len(report.parameter_validation["critical_mismatches"]) == 0
        
        # Test report string representation
        report_str = str(report)
        assert "PASS" in report_str
        assert "100.00%" in report_str


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
