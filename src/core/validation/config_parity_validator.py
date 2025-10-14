"""
Config Parity Validator

Ensures backtest and live trading configurations are identical for critical parameters.
Prevents configuration drift that could cause signal divergence.

Key Features:
- Validates warmup periods match (525 minutes required)
- Validates strategy parameters (MACD periods, thresholds, etc.)
- Validates risk limits (position size, max loss, etc.)
- Fails fast on critical mismatches

Usage:
    from src.core.validation.config_parity_validator import validate_config_parity
    
    errors = validate_config_parity(backtest_config, live_config)
    if errors:
        raise ConfigurationError(f"Config parity violations: {errors}")

Created: October 15, 2025
Phase: 6.5 - Backtest vs Live Parity
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ConfigParityError(Exception):
    """Raised when configuration parity is violated"""
    pass


@dataclass
class ConfigMismatch:
    """Records a single configuration mismatch"""
    parameter: str
    backtest_value: Any
    live_value: Any
    is_critical: bool
    category: str  # "warmup", "strategy", "risk", "execution"


# Critical parameters that MUST match between backtest and live
CRITICAL_PARAMETERS = {
    "warmup": [
        "warmup_minutes",
    ],
    "strategy": [
        "macd_fast",
        "macd_slow",
        "macd_signal",
        "exit_threshold",
        "entry_cooldown_minutes",
        "use_previous_bar",
        "enable_two_bar_rule",
    ],
    "risk": [
        "max_position_size",
        "max_daily_loss",
        "max_total_exposure",
    ],
    "execution": [
        "execution_delay",
        "slippage_model",
    ],
}


def validate_config_parity(
    backtest_config: Dict[str, Any],
    live_config: Dict[str, Any],
    strict_mode: bool = True,
    critical_only: bool = False
) -> List[ConfigMismatch]:
    """
    Validate that backtest and live configurations match.
    
    Args:
        backtest_config: Configuration dictionary from backtest
        live_config: Configuration dictionary from live trading
        strict_mode: If True, raise exception on critical mismatches
        critical_only: If True, only check critical parameters
        
    Returns:
        List of ConfigMismatch objects (empty if all match)
        
    Raises:
        ConfigParityError: If critical parameters mismatch (strict_mode=True)
    """
    mismatches = []
    
    # Build list of parameters to check
    params_to_check = {}
    for category, params in CRITICAL_PARAMETERS.items():
        for param in params:
            params_to_check[param] = category
    
    if not critical_only:
        # Add all other parameters from both configs
        all_params = set(backtest_config.keys()) | set(live_config.keys())
        for param in all_params:
            if param not in params_to_check:
                params_to_check[param] = "other"
    
    # Check each parameter
    for param, category in params_to_check.items():
        backtest_value = backtest_config.get(param)
        live_value = live_config.get(param)
        
        # Check if values differ
        if backtest_value != live_value:
            is_critical = param in [p for params in CRITICAL_PARAMETERS.values() for p in params]
            
            mismatch = ConfigMismatch(
                parameter=param,
                backtest_value=backtest_value,
                live_value=live_value,
                is_critical=is_critical,
                category=category
            )
            mismatches.append(mismatch)
            
            # Log the mismatch
            log_level = logging.ERROR if is_critical else logging.WARNING
            logger.log(
                log_level,
                f"{'CRITICAL ' if is_critical else ''}Config mismatch [{category}]: "
                f"{param}: backtest={backtest_value}, live={live_value}"
            )
    
    # Check critical mismatches
    critical_mismatches = [m for m in mismatches if m.is_critical]
    
    if critical_mismatches and strict_mode:
        error_msg = f"Critical configuration mismatches found ({len(critical_mismatches)}):\n"
        for m in critical_mismatches:
            error_msg += f"  - {m.parameter}: backtest={m.backtest_value}, live={m.live_value}\n"
        raise ConfigParityError(error_msg)
    
    # Log summary
    if mismatches:
        logger.warning(
            f"Config parity check: {len(mismatches)} mismatches "
            f"({len(critical_mismatches)} critical)"
        )
    else:
        logger.info(" Config parity check passed: All parameters match")
    
    return mismatches


def validate_warmup_parity(
    backtest_warmup: int,
    live_warmup: int,
    required_warmup: int = 525
) -> Optional[ConfigMismatch]:
    """
    Validate warmup period parity (specialized check).
    
    Args:
        backtest_warmup: Warmup minutes in backtest
        live_warmup: Warmup minutes in live
        required_warmup: Required minimum warmup (default 525)
        
    Returns:
        ConfigMismatch if mismatch found, None otherwise
        
    Raises:
        ConfigParityError: If warmup periods don't match
    """
    if backtest_warmup != live_warmup:
        mismatch = ConfigMismatch(
            parameter="warmup_minutes",
            backtest_value=backtest_warmup,
            live_value=live_warmup,
            is_critical=True,
            category="warmup"
        )
        
        logger.error(
            f"CRITICAL: Warmup period mismatch: "
            f"backtest={backtest_warmup}min, live={live_warmup}min "
            f"(MUST be identical)"
        )
        
        raise ConfigParityError(
            f"Warmup period mismatch: backtest={backtest_warmup}min, "
            f"live={live_warmup}min (required: {required_warmup}min)"
        )
    
    if backtest_warmup < required_warmup or live_warmup < required_warmup:
        logger.error(
            f"CRITICAL: Warmup period too short: "
            f"required={required_warmup}min, "
            f"backtest={backtest_warmup}min, live={live_warmup}min"
        )
        
        raise ConfigParityError(
            f"Warmup period too short: required={required_warmup}min, "
            f"got backtest={backtest_warmup}min, live={live_warmup}min"
        )
    
    logger.info(f" Warmup parity validated: {backtest_warmup}min (both)")
    return None


def generate_parity_report(mismatches: List[ConfigMismatch]) -> str:
    """
    Generate human-readable parity report.
    
    Args:
        mismatches: List of configuration mismatches
        
    Returns:
        Formatted report string
    """
    if not mismatches:
        return " CONFIG PARITY: All parameters match"
    
    critical = [m for m in mismatches if m.is_critical]
    non_critical = [m for m in mismatches if not m.is_critical]
    
    report = []
    report.append("\n" + "=" * 80)
    report.append("CONFIG PARITY REPORT")
    report.append("=" * 80)
    report.append(f"Total Mismatches: {len(mismatches)}")
    report.append(f"Critical Mismatches: {len(critical)}")
    report.append(f"Non-Critical Mismatches: {len(non_critical)}")
    report.append("")
    
    if critical:
        report.append(" CRITICAL MISMATCHES (MUST FIX):")
        for m in critical:
            report.append(
                f"  [{m.category}] {m.parameter}: "
                f"backtest={m.backtest_value}, live={m.live_value}"
            )
        report.append("")
    
    if non_critical:
        report.append("  NON-CRITICAL MISMATCHES (Review):")
        for m in non_critical:
            report.append(
                f"  [{m.category}] {m.parameter}: "
                f"backtest={m.backtest_value}, live={m.live_value}"
            )
        report.append("")
    
    report.append("=" * 80)
    status = " FAIL" if critical else "  WARN"
    report.append(f"STATUS: {status}")
    report.append("=" * 80)
    
    return "\n".join(report)


# Standalone test
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )
    
    print("=" * 80)
    print("Config Parity Validator - Standalone Test")
    print("=" * 80)
    
    # Test 1: Matching configs
    print("\n1. Matching Configs Test:")
    backtest_config = {
        "warmup_minutes": 525,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "exit_threshold": 0.8,
        "max_position_size": 500,
    }
    
    live_config = backtest_config.copy()
    
    mismatches = validate_config_parity(backtest_config, live_config, strict_mode=False)
    print(f"   Mismatches: {len(mismatches)}")
    
    # Test 2: Critical mismatch (warmup)
    print("\n2. Critical Mismatch Test (Warmup):")
    live_config["warmup_minutes"] = 60
    
    try:
        mismatches = validate_config_parity(backtest_config, live_config, strict_mode=True)
    except ConfigParityError as e:
        print(f"    Expected error caught: {str(e)[:100]}...")
    
    # Test 3: Non-critical mismatch
    print("\n3. Non-Critical Mismatch Test:")
    live_config["warmup_minutes"] = 525  # Fix critical
    live_config["some_other_param"] = "different"
    
    mismatches = validate_config_parity(backtest_config, live_config, strict_mode=False)
    print(f"   Mismatches: {len(mismatches)} (non-critical)")
    
    # Test 4: Warmup validation
    print("\n4. Warmup Validation Test:")
    try:
        validate_warmup_parity(525, 525, required_warmup=525)
        print("    Warmup parity passed")
    except ConfigParityError as e:
        print(f"    Error: {e}")
    
    try:
        validate_warmup_parity(525, 60, required_warmup=525)
    except ConfigParityError as e:
        print(f"    Expected error: Mismatch detected")
    
    # Test 5: Report generation
    print("\n5. Report Generation Test:")
    mismatches = [
        ConfigMismatch("warmup_minutes", 525, 60, True, "warmup"),
        ConfigMismatch("exit_threshold", 0.8, 0.7, True, "strategy"),
        ConfigMismatch("some_param", 100, 200, False, "other"),
    ]
    
    report = generate_parity_report(mismatches)
    print(report)
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
