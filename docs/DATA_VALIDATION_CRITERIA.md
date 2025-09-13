# Data Validation Criteria

**Date**: September 12, 2025  
**Version**: 1.0  
**Component**: `src/runners/components/validator.py`

## Overview

This document defines the comprehensive data validation criteria used by the backtesting system to ensure data quality while allowing for normal market anomalies. The validation system distinguishes between **critical errors** (that block execution) and **warnings** (that log issues but allow execution).

## Validation Philosophy

The validation system follows these principles:

1. **Real Market Data First**: Validation should accommodate normal market anomalies
2. **Threshold-Based**: Use statistical thresholds instead of zero-tolerance
3. **Error vs Warning**: Distinguish between critical and minor issues
4. **Context Awareness**: Consider data size and market context

## Validation Categories

### 1. Required Columns Validation

**Purpose**: Ensure essential columns exist for strategy execution  
**Level**: CRITICAL ERROR  
**Threshold**: Zero tolerance

#### Required Columns
- `timestamp` or `datetime` or `time`
- `open`
- `high` 
- `low`
- `close`
- `volume` (optional but recommended)

```python
def _validate_required_columns(self, data: pd.DataFrame) -> List[str]:
    """Validate required columns exist."""
    required_base = ['open', 'high', 'low', 'close']
    timestamp_cols = ['timestamp', 'datetime', 'time']
    
    # Must have timestamp column
    if not any(col in data.columns for col in timestamp_cols):
        return ["Missing timestamp column"]
    
    # Must have OHLC columns  
    missing_cols = [col for col in required_base if col not in data.columns]
    if missing_cols:
        return [f"Missing required columns: {missing_cols}"]
    
    return []
```

### 2. Data Completeness Validation

**Purpose**: Check for missing data and gaps  
**Level**: CRITICAL ERROR  
**Threshold**: Zero tolerance for null values

#### Validation Rules
- No null/NaN values in OHLC columns
- No infinite values
- Minimum data size requirements

```python
def _check_missing_data(self, data: pd.DataFrame, issues: List[str]) -> None:
    """Check for missing data in critical columns."""
    critical_columns = ['open', 'high', 'low', 'close']
    
    for col in critical_columns:
        if col in data.columns:
            null_count = data[col].isnull().sum()
            inf_count = np.isinf(data[col]).sum()
            
            if null_count > 0:
                issues.append(f"Found {null_count} null values in {col}")
            if inf_count > 0:
                issues.append(f"Found {inf_count} infinite values in {col}")
```

### 3. Price Consistency Validation

**Purpose**: Validate OHLC relationships  
**Level**: ERROR (major inconsistencies) / WARNING (minor inconsistencies)  
**Threshold**: 0.1% of data or minimum 10 rows

#### Validation Rules

##### Critical Relationships (Always Errors)
- `high >= low` (fundamental market rule)
- No negative prices in any column

##### Threshold-Based Relationships (Error > threshold, Warning ≤ threshold)
- `high >= open` and `high >= close`
- `low <= open` and `low <= close`

```python
def _check_price_consistency(self, data: pd.DataFrame, issues: List[str]) -> None:
    """Check OHLC price relationships with reasonable thresholds."""
    
    # Critical: High must be >= Low (always error)
    high_low_issues = (data['high'] < data['low']).sum()
    if high_low_issues > 0:
        issues.append(f"Found {high_low_issues} rows where high < low")
    
    # Threshold-based: High should be >= open and close
    high_open_issues = (data['high'] < data['open']).sum()
    high_close_issues = (data['high'] < data['close']).sum()
    
    threshold = max(10, len(data) * 0.001)  # 0.1% or minimum 10 rows
    
    if high_open_issues > threshold or high_close_issues > threshold:
        issues.append(f"High price inconsistency: {high_open_issues} rows where high < open, "
                    f"{high_close_issues} rows where high < close")
    elif high_open_issues > 0 or high_close_issues > 0:
        # Minor inconsistencies - warning only
        self.warnings.append(f"Minor high price inconsistency: {high_open_issues} rows where high < open, "
                           f"{high_close_issues} rows where high < close")
```

### 4. Volume Validation

**Purpose**: Check volume data quality  
**Level**: WARNING only  
**Threshold**: Statistical outlier detection

#### Validation Rules
- Volume outliers (>10x median) → Warning
- Zero volume periods → Warning (if >5% of data)
- Negative volume → Error

```python
def _check_volume_quality(self, data: pd.DataFrame) -> None:
    """Check volume data quality with outlier detection."""
    if 'volume' not in data.columns:
        return
    
    # Negative volumes are errors
    negative_volume = (data['volume'] < 0).sum()
    if negative_volume > 0:
        self.issues.append(f"Found {negative_volume} negative volume values")
    
    # Volume outliers are warnings
    median_volume = data['volume'].median()
    if median_volume > 0:
        outliers = (data['volume'] > median_volume * 10).sum()
        if outliers > 0:
            self.warnings.append(f"Found {outliers} volume outliers (>10x median)")
    
    # Excessive zero volume periods
    zero_volume = (data['volume'] == 0).sum()
    zero_threshold = len(data) * 0.05  # 5% of data
    if zero_volume > zero_threshold:
        self.warnings.append(f"Found {zero_volume} zero volume periods ({zero_volume/len(data)*100:.1f}% of data)")
```

### 5. Price Movement Validation

**Purpose**: Detect unrealistic price movements  
**Level**: WARNING (outliers) / ERROR (extreme cases)  
**Threshold**: Statistical analysis based

#### Validation Rules
- Large price gaps (>10% change) → Warning
- Extreme gaps (>50% change) → Error
- Price spike detection → Warning

```python
def _check_price_movements(self, data: pd.DataFrame) -> None:
    """Check for unrealistic price movements."""
    if len(data) < 2:
        return
    
    # Calculate price changes
    price_changes = data['close'].pct_change().dropna()
    
    # Large gaps (>10%) are warnings
    large_gaps = (abs(price_changes) > 0.10).sum()
    if large_gaps > 0:
        max_gap = abs(price_changes).max() * 100
        self.warnings.append(f"Found {large_gaps} large price gaps (>10%), max gap: {max_gap:.1f}%")
    
    # Extreme gaps (>50%) are errors
    extreme_gaps = (abs(price_changes) > 0.50).sum()
    if extreme_gaps > 0:
        max_extreme = abs(price_changes[abs(price_changes) > 0.50]).max() * 100
        self.issues.append(f"Found {extreme_gaps} extreme price gaps (>50%), max: {max_extreme:.1f}%")
```

### 6. Timestamp Validation

**Purpose**: Ensure proper time series structure  
**Level**: ERROR (critical) / WARNING (minor issues)  
**Threshold**: Context dependent

#### Validation Rules
- Monotonic timestamps (non-decreasing) → Error
- Duplicate timestamps → Error
- Large time gaps → Warning
- Weekend/holiday detection → Info

```python
def _check_timestamp_quality(self, data: pd.DataFrame) -> None:
    """Validate timestamp data quality."""
    
    # Monotonic check - critical
    if not data['timestamp'].is_monotonic_increasing:
        non_monotonic = (data['timestamp'].diff() < pd.Timedelta(0)).sum()
        self.issues.append(f"Found {non_monotonic} non-monotonic timestamp sequences")
    
    # Duplicate timestamps - critical
    duplicates = data['timestamp'].duplicated().sum()
    if duplicates > 0:
        self.issues.append(f"Found {duplicates} duplicate timestamps")
    
    # Large gaps - warning
    time_diffs = data['timestamp'].diff().dropna()
    median_diff = time_diffs.median()
    large_gaps = (time_diffs > median_diff * 10).sum()
    
    if large_gaps > 0:
        self.warnings.append(f"Found {large_gaps} large time gaps (>10x median interval)")
```

## Validation Thresholds Summary

| Validation Type | Threshold | Action | Reasoning |
|-----------------|-----------|---------|-----------|
| **Missing Columns** | 0 tolerance | ERROR | Critical for execution |
| **Null Values** | 0 tolerance | ERROR | Cannot process null OHLC |
| **High < Low** | 0 tolerance | ERROR | Violates market fundamentals |
| **Negative Prices** | 0 tolerance | ERROR | Impossible in real markets |
| **Price Inconsistencies** | >0.1% OR >10 rows | ERROR | Minor anomalies expected |
| **Price Inconsistencies** | ≤0.1% AND ≤10 rows | WARNING | Allow minor anomalies |
| **Volume Outliers** | >10x median | WARNING | Normal in volatile markets |
| **Large Price Gaps** | >10% change | WARNING | Can occur in real markets |
| **Extreme Price Gaps** | >50% change | ERROR | Likely data errors |
| **Zero Volume** | >5% of data | WARNING | May indicate data issues |
| **Time Gaps** | >10x median interval | WARNING | Market closures expected |

## Configuration Options

### Validation Strictness Levels

```python
class ValidationStrictness:
    STRICT = "strict"        # Zero tolerance for any anomaly
    NORMAL = "normal"        # Current thresholds (default)
    LENIENT = "lenient"      # Higher thresholds for noisy data
    
    @staticmethod
    def get_thresholds(level: str) -> Dict:
        thresholds = {
            "strict": {"price_inconsistency": 0, "volume_outlier": 5},
            "normal": {"price_inconsistency": 0.001, "volume_outlier": 10},  # 0.1%
            "lenient": {"price_inconsistency": 0.005, "volume_outlier": 20}  # 0.5%
        }
        return thresholds.get(level, thresholds["normal"])
```

### Customizable Parameters

```python
# Example configuration in unified_config.py
validation:
  enabled: true
  strictness: "normal"  # strict, normal, lenient
  custom_thresholds:
    price_inconsistency_percent: 0.1  # 0.1% of data
    price_inconsistency_min_rows: 10  # Minimum 10 rows
    volume_outlier_multiplier: 10     # 10x median
    large_gap_threshold: 0.10         # 10% price change
    extreme_gap_threshold: 0.50       # 50% price change
    zero_volume_threshold: 0.05       # 5% of data
```

## Real-World Examples

### Successful Validation (Warnings Only)
```
[INFO] DataLoader: Loading multi-timeframe data for RELIANCE: ['5m', '15m']
[INFO] DataLoader: ✅ Successfully loaded 5m data: 67793 records
[WARNING] Data validation warning for RELIANCE 5m: Found 522 volume outliers (>10x median)
[WARNING] Data validation warning for RELIANCE 5m: Minor low price inconsistency: 1 rows where low > open
[INFO] MSEStrategyBacktesting: Strategy execution completed successfully
```

### Failed Validation (Errors Block Execution)  
```
[ERROR] Data validation failed for CORRUPT_DATA 5m: ['High price inconsistency: 1250 rows where high < open (1.8% of data)']
[ERROR] Data validation failed for CORRUPT_DATA 5m: ['Found 50 extreme price gaps (>50%), max: 250.0%']
[WARNING] No data found for CORRUPT_DATA in 2022-01-01_to_2025-08-31
```

## Impact on Strategy Execution

### Before Validation Fixes
- **1 row inconsistency** out of 67,000 → **BLOCKED ALL TRADES**
- Volume spikes → **BLOCKED ALL TRADES**  
- Minor anomalies → **ZERO STRATEGY EXECUTION**

### After Validation Fixes
- **Minor inconsistencies** → **WARNINGS + EXECUTION CONTINUES**
- **Major data corruption** → **ERROR + EXECUTION BLOCKED**
- **Normal market anomalies** → **TRADES GENERATED SUCCESSFULLY**

## Monitoring and Metrics

### Validation Health Metrics
- **Validation Pass Rate**: % of tickers that pass validation
- **Warning Rate**: % of tickers with warnings but successful execution  
- **Error Rate**: % of tickers blocked by validation errors
- **Common Issues**: Most frequent validation warnings/errors

### Example Monitoring Output
```
Validation Summary for 2022-01-01_to_2025-08-31:
├── Total Tickers Processed: 26
├── Validation Pass Rate: 96.2% (25/26 tickers)
├── Warning Rate: 84.6% (22/26 tickers had warnings)
├── Error Rate: 3.8% (1/26 tickers blocked)
└── Most Common Warnings:
    ├── Volume outliers: 85% of tickers
    ├── Minor price inconsistencies: 12% of tickers
    └── Large price gaps: 8% of tickers
```

## Best Practices

### For Strategy Developers
1. **Test with Real Data**: Always test strategies with actual market data
2. **Handle Warnings**: Don't assume data is perfect, handle edge cases
3. **Monitor Validation**: Check validation reports for data quality insights

### For Data Providers
1. **Clean but Realistic**: Remove obvious errors but preserve market anomalies
2. **Consistent Formatting**: Maintain consistent column names and formats
3. **Gap Documentation**: Document known gaps or data issues

### for System Operators  
1. **Monitor Error Rates**: Track validation error rates over time
2. **Adjust Thresholds**: Fine-tune thresholds based on data characteristics
3. **Data Quality Reports**: Generate regular data quality summaries

## Testing and Validation

### Unit Tests
```python
def test_price_consistency_thresholds():
    """Test that minor inconsistencies are warnings, major ones are errors."""
    # 1 row inconsistency out of 1000 = 0.1% = WARNING
    data_minor = create_test_data(1000, price_inconsistencies=1)
    result = validator.validate_market_data(data_minor)
    assert result['is_valid'] == True
    assert len(result['warnings']) > 0
    
    # 50 row inconsistencies out of 1000 = 5% = ERROR  
    data_major = create_test_data(1000, price_inconsistencies=50)
    result = validator.validate_market_data(data_major)
    assert result['is_valid'] == False
    assert len(result['issues']) > 0
```

### Integration Tests
```python
def test_real_market_data_validation():
    """Test validation with actual market data."""
    # Load real RELIANCE data (known to have minor anomalies)
    data = load_market_data("RELIANCE", "2022-01-01", "2025-08-31")
    result = validator.validate_market_data(data)
    
    # Should pass validation with warnings
    assert result['is_valid'] == True
    assert len(result['warnings']) > 0  # Volume outliers expected
    assert "volume outliers" in str(result['warnings'])
```

## Related Documentation

- [Signal Handling and Validation Fixes](SIGNAL_HANDLING_AND_VALIDATION_FIXES.md)
- [Strategy Development Guide](STRATEGY_DEVELOPMENT_GUIDE.md)  
- [Data Requirements](DATA_REQUIREMENTS.md)
- [System Architecture](SYSTEM_ARCHITECTURE.md)

---

**Last Updated**: September 12, 2025  
**Validation Version**: 1.0  
**Status**: ✅ Production Ready