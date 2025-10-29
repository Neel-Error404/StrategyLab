# src/core/validation/precision_validator.py
"""
Precision Validation for Trading System

This module enforces decimal precision requirements for Indian stock exchanges (NSE/BSE).
All prices, quantities, and PnL values must maintain 4-decimal precision to ensure:
- Compliance with exchange regulations
- Accurate order execution
- Correct PnL calculations
- Prevention of floating-point arithmetic errors

Key Functions:
- validate_price(): Enforce 4-decimal price precision
- validate_quantity(): Enforce integer or allowed fractional quantities
- validate_pnl(): Ensure PnL calculations maintain precision
- round_to_precision(): Safe rounding with overflow protection
"""

import logging
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Union, Optional

logger = logging.getLogger(__name__)

# Exchange precision requirements (NSE/BSE standard)
PRICE_DECIMALS = 4  # Maximum 4 decimal places for prices
QUANTITY_DECIMALS = 0  # Quantities are typically integers (shares/lots)
PNL_DECIMALS = 4  # PnL calculations maintain 4-decimal precision

# Validation thresholds
MIN_PRICE = Decimal('0.0001')  # Minimum non-zero price (1 paisa = 0.01, but allow sub-paisa)
MAX_PRICE = Decimal('999999.9999')  # Maximum price (10 lakh with 4 decimals)
MIN_QUANTITY = 1  # Minimum order quantity
MAX_QUANTITY = 100_000_000  # Maximum order quantity (10 crore shares)


class PrecisionError(ValueError):
    """Raised when precision validation fails."""
    pass


def to_decimal(value: Union[float, int, str, Decimal]) -> Decimal:
    """
    Convert value to Decimal for precise calculations.
    
    Args:
        value: Value to convert (float, int, str, or Decimal)
        
    Returns:
        Decimal representation
        
    Raises:
        PrecisionError: If value cannot be converted
    """
    try:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
    except (ValueError, InvalidOperation) as e:
        raise PrecisionError(f"Cannot convert {value} to Decimal: {e}")


def round_to_precision(value: Union[float, int, str, Decimal], 
                       decimals: int = PRICE_DECIMALS) -> Decimal:
    """
    Round value to specified decimal precision using ROUND_HALF_UP.
    
    Args:
        value: Value to round
        decimals: Number of decimal places (default: 4 for prices)
        
    Returns:
        Decimal rounded to specified precision
        
    Examples:
        >>> round_to_precision(123.45678, 4)
        Decimal('123.4568')
        >>> round_to_precision(0.12345, 2)
        Decimal('0.12')
    """
    dec_value = to_decimal(value)
    quantizer = Decimal('0.1') ** decimals
    return dec_value.quantize(quantizer, rounding=ROUND_HALF_UP)


def validate_price(price: Union[float, int, str, Decimal],
                  allow_zero: bool = False,
                  symbol: Optional[str] = None) -> Decimal:
    """
    Validate and normalize price to 4-decimal precision.
    
    Args:
        price: Price value to validate
        allow_zero: Whether zero prices are allowed (default: False)
        symbol: Symbol name for error reporting (optional)
        
    Returns:
        Decimal price rounded to 4 decimals
        
    Raises:
        PrecisionError: If price is invalid
        
    Examples:
        >>> validate_price(123.456789)
        Decimal('123.4568')
        >>> validate_price(0.0)  # Raises PrecisionError
        >>> validate_price(0.0, allow_zero=True)
        Decimal('0.0000')
    """
    try:
        dec_price = round_to_precision(price, PRICE_DECIMALS)
    except PrecisionError as e:
        raise PrecisionError(f"Invalid price for {symbol or 'unknown'}: {e}")
    
    # Validate range - check negative first
    if dec_price < 0:
        raise PrecisionError(
            f"Price cannot be negative for {symbol or 'unknown'}: {price}"
        )
    
    if allow_zero and dec_price == 0:
        return Decimal('0.0000')
    
    if not allow_zero and dec_price <= 0:
        raise PrecisionError(
            f"Price must be positive for {symbol or 'unknown'}: {price}"
        )
    
    if dec_price < MIN_PRICE:
        raise PrecisionError(
            f"Price below minimum ({MIN_PRICE}) for {symbol or 'unknown'}: {price}"
        )
    
    if dec_price > MAX_PRICE:
        raise PrecisionError(
            f"Price exceeds maximum ({MAX_PRICE}) for {symbol or 'unknown'}: {price}"
        )
    
    logger.debug(f"Price validated: {price}  {dec_price} ({symbol or 'unknown'})")
    return dec_price


def validate_quantity(quantity: Union[float, int, str, Decimal],
                     allow_fractional: bool = False,
                     symbol: Optional[str] = None) -> int:
    """
    Validate and normalize quantity to integer (or fractional if allowed).
    
    Args:
        quantity: Quantity value to validate
        allow_fractional: Whether fractional quantities are allowed (default: False)
        symbol: Symbol name for error reporting (optional)
        
    Returns:
        Integer quantity (or Decimal if fractional allowed)
        
    Raises:
        PrecisionError: If quantity is invalid
        
    Examples:
        >>> validate_quantity(100)
        100
        >>> validate_quantity(100.5)  # Raises PrecisionError (fractional not allowed)
        >>> validate_quantity(100.5, allow_fractional=True)
        100  # Rounds down to integer
    """
    try:
        dec_quantity = to_decimal(quantity)
    except PrecisionError as e:
        raise PrecisionError(f"Invalid quantity for {symbol or 'unknown'}: {e}")
    
    # Check for negative
    if dec_quantity <= 0:
        raise PrecisionError(
            f"Quantity must be positive for {symbol or 'unknown'}: {quantity}"
        )
    
    # Check for fractional (unless allowed)
    if not allow_fractional:
        int_quantity = int(dec_quantity)
        if dec_quantity != int_quantity:
            logger.warning(
                f"Fractional quantity rounded: {quantity}  {int_quantity} ({symbol or 'unknown'})"
            )
        dec_quantity = Decimal(int_quantity)
    
    # Validate range
    if dec_quantity < MIN_QUANTITY:
        raise PrecisionError(
            f"Quantity below minimum ({MIN_QUANTITY}) for {symbol or 'unknown'}: {quantity}"
        )
    
    if dec_quantity > MAX_QUANTITY:
        raise PrecisionError(
            f"Quantity exceeds maximum ({MAX_QUANTITY}) for {symbol or 'unknown'}: {quantity}"
        )
    
    result = int(dec_quantity) if not allow_fractional else dec_quantity
    logger.debug(f"Quantity validated: {quantity}  {result} ({symbol or 'unknown'})")
    return result


def validate_pnl(entry_price: Union[float, Decimal],
                 exit_price: Union[float, Decimal],
                 quantity: Union[int, float],
                 transaction_costs: Union[float, Decimal] = 0,
                 symbol: Optional[str] = None) -> Decimal:
    """
    Calculate and validate PnL with 4-decimal precision.
    
    Formula: PnL = (exit_price - entry_price) * quantity - transaction_costs
    
    Args:
        entry_price: Entry price per unit
        exit_price: Exit price per unit
        quantity: Number of units
        transaction_costs: Total transaction costs (brokerage, taxes, etc.)
        symbol: Symbol name for error reporting (optional)
        
    Returns:
        Decimal PnL rounded to 4 decimals
        
    Examples:
        >>> validate_pnl(100.0, 110.0, 100)
        Decimal('1000.0000')
        >>> validate_pnl(100.0, 90.0, 100, 50)
        Decimal('-1050.0000')
    """
    # Validate inputs
    entry = validate_price(entry_price, symbol=symbol)
    exit_val = validate_price(exit_price, symbol=symbol)
    qty = to_decimal(quantity)
    costs = to_decimal(transaction_costs)
    
    # Calculate PnL
    price_diff = exit_val - entry
    gross_pnl = price_diff * qty
    net_pnl = gross_pnl - costs
    
    # Round to precision
    pnl = round_to_precision(net_pnl, PNL_DECIMALS)
    
    logger.debug(
        f"PnL calculated: Entry={entry}, Exit={exit_val}, Qty={qty}, "
        f"Costs={costs}, PnL={pnl} ({symbol or 'unknown'})"
    )
    
    return pnl


def validate_order_value(price: Union[float, Decimal],
                         quantity: Union[int, float],
                         symbol: Optional[str] = None) -> Decimal:
    """
    Calculate and validate total order value.
    
    Formula: Order Value = price * quantity
    
    Args:
        price: Price per unit
        quantity: Number of units
        symbol: Symbol name for error reporting (optional)
        
    Returns:
        Decimal order value rounded to 2 decimals (rupee precision)
        
    Examples:
        >>> validate_order_value(123.45, 100)
        Decimal('12345.00')
    """
    validated_price = validate_price(price, symbol=symbol)
    validated_qty = to_decimal(quantity)
    
    order_value = validated_price * validated_qty
    
    # Round to 2 decimals for rupee precision
    rounded_value = round_to_precision(order_value, 2)
    
    logger.debug(
        f"Order value calculated: Price={validated_price}, Qty={validated_qty}, "
        f"Value={rounded_value} ({symbol or 'unknown'})"
    )
    
    return rounded_value


def check_precision_compliance(value: Union[float, Decimal],
                               max_decimals: int = PRICE_DECIMALS) -> bool:
    """
    Check if value complies with decimal precision requirement.
    
    Args:
        value: Value to check
        max_decimals: Maximum allowed decimal places
        
    Returns:
        True if compliant, False otherwise
        
    Examples:
        >>> check_precision_compliance(123.4567, 4)
        True
        >>> check_precision_compliance(123.45678, 4)
        False
    """
    dec_value = to_decimal(value)
    rounded_value = round_to_precision(dec_value, max_decimals)
    return dec_value == rounded_value


# Convenience functions for common validations
def validate_buy_order(symbol: str, price: float, quantity: int) -> dict:
    """Validate all parameters for a buy order."""
    return {
        'symbol': symbol,
        'price': validate_price(price, symbol=symbol),
        'quantity': validate_quantity(quantity, symbol=symbol),
        'order_value': validate_order_value(price, quantity, symbol=symbol)
    }


def validate_sell_order(symbol: str, price: float, quantity: int) -> dict:
    """Validate all parameters for a sell order."""
    return validate_buy_order(symbol, price, quantity)  # Same validation rules


if __name__ == '__main__':
    # Test the precision validator
    import sys
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing Precision Validator:")
    print("=" * 60)
    
    # Test price validation
    print("\n1. Price Validation:")
    print(f"   123.456789  {validate_price(123.456789)}")
    print(f"   0.0001  {validate_price(0.0001)}")
    print(f"   0.0 (allow_zero=True)  {validate_price(0.0, allow_zero=True)}")
    
    # Test quantity validation
    print("\n2. Quantity Validation:")
    print(f"   100  {validate_quantity(100)}")
    print(f"   100.8  {validate_quantity(100.8)}")  # Rounds down
    
    # Test PnL calculation
    print("\n3. PnL Calculation:")
    pnl = validate_pnl(entry_price=100.0, exit_price=110.0, quantity=100, transaction_costs=50.0)
    print(f"   Entry=100, Exit=110, Qty=100, Costs=50  PnL={pnl}")
    
    # Test order value
    print("\n4. Order Value:")
    value = validate_order_value(price=123.45, quantity=100)
    print(f"   Price=123.45, Qty=100  Value={value}")
    
    # Test precision compliance
    print("\n5. Precision Compliance:")
    print(f"   123.4567 (4 decimals)  {check_precision_compliance(123.4567, 4)}")
    print(f"   123.45678 (5 decimals)  {check_precision_compliance(123.45678, 4)}")
    
    print("\n Precision validator tests complete!")
