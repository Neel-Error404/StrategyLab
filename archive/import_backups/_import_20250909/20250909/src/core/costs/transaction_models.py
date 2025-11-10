# src/costs/transaction_models.py
import pandas as pd
import numpy as np
from typing import Dict, Optional, Callable
from abc import ABC, abstractmethod

class TransactionCostModel(ABC):
    """Abstract base class for transaction cost models."""

    @abstractmethod
    def calculate_cost(self, trade: Dict, market_state: Dict) -> Dict[str, float]:
        """Calculate transaction costs for a trade."""
        pass

class AdvancedTransactionCosts:
    """
    Comprehensive transaction cost modeling including:
    - Spread costs
    - Market impact
    - Commissions
    - Slippage based on liquidity
    """

    def __init__(self,
                 commission_rate: float = 0.0003,  # 3 bps
                 min_commission: float = 20,
                 spread_model: Optional[Callable] = None,
                 impact_model: Optional[Callable] = None):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.spread_model = spread_model or self._default_spread_model
        self.impact_model = impact_model or self._default_impact_model

    def calculate_total_cost(self, trade: Dict, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate all transaction costs for a trade.

        Args:
            trade: Dictionary with trade details
            market_data: Market data around trade time

        Returns:
            Dictionary with cost breakdown
        """
        # Extract trade details
        size = trade['size']
        price = trade['price']
        timestamp = trade['timestamp']
        ticker = trade['ticker']

        # Get market state
        market_state = self._get_market_state(market_data, timestamp, ticker)

        # Calculate liquidity metrics
        liquidity = self._calculate_liquidity(market_state)

        # Cost components
        costs = {
            'commission': self._calculate_commission(size * price),
            'spread': self._calculate_spread_cost(size, price, liquidity, market_state),
            'market_impact': self._calculate_market_impact(size, price, liquidity, market_state),
            'timing_cost': self._calculate_timing_cost(trade, market_state)
        }

        costs['total'] = sum(costs.values())
        costs['percentage'] = (costs['total'] / (size * price)) * 100

        return costs

    def calculate_cost(self, trade: Dict, market_state: Dict) -> Dict[str, float]:
        """
        Calculate transaction costs - alias for calculate_total_cost for backward compatibility.

        Args:
            trade: Dictionary with trade details
            market_state: Market state dictionary or DataFrame

        Returns:
            Dictionary with cost breakdown
        """
        # Convert market_state dict to DataFrame if needed for compatibility
        if isinstance(market_state, dict):
            # Create a simple DataFrame from the market state dict
            import pandas as pd
            market_data = pd.DataFrame([market_state])
            if 'timestamp' in trade:
                market_data.index = [trade['timestamp']]
        else:
            market_data = market_state

        return self.calculate_total_cost(trade, market_data)

    def _calculate_liquidity(self, market_state: Dict) -> Dict[str, float]:
        """
        Calculate liquidity metrics from market state.

        Liquidity factors:
        - Average daily volume (ADV)
        - Bid-ask spread
        - Order book depth
        - Volatility
        """
        volume = market_state.get('volume', 0)
        spread = market_state.get('spread', 0.01)  # Default 1 cent
        volatility = market_state.get('volatility', 0.02)  # Default 2% daily vol

        # Normalize volume to get liquidity score (0-1)
        adv = market_state.get('adv', volume)
        participation_rate = volume / adv if adv > 0 else 0.1

        liquidity = {
            'score': 1 / (1 + spread * 100 + volatility * 10),  # Simple liquidity score
            'adv': adv,
            'spread_bps': spread / market_state.get('price', 100) * 10000,
            'volatility': volatility,
            'participation_rate': participation_rate
        }

        return liquidity

    def _calculate_commission(self, trade_value: float) -> float:
        """Calculate commission costs."""
        return max(self.min_commission, trade_value * self.commission_rate)

    def _calculate_spread_cost(self, size: float, price: float,
                               liquidity: Dict, market_state: Dict) -> float:
        """
        Calculate spread cost based on liquidity.

        Less liquid stocks have wider spreads.
        """
        base_spread = market_state.get('spread', 0.01)

        # Adjust spread based on trade size relative to liquidity
        size_factor = size / liquidity['adv'] if liquidity['adv'] > 0 else 0.01
        adjusted_spread = base_spread * (1 + size_factor * 10)

        return size * adjusted_spread / 2  # Pay half spread

    def _calculate_market_impact(self, size: float, price: float,
                                liquidity: Dict, market_state: Dict) -> float:
        """
        Calculate market impact using square-root model.

        Impact = spread * (size/ADV)^0.5
        """
        if liquidity['adv'] == 0:
            return 0

        # Square-root market impact model
        participation = size / liquidity['adv']
        impact_bps = liquidity['spread_bps'] * np.sqrt(participation)

        return size * price * impact_bps / 10000

    def _calculate_timing_cost(self, trade: Dict, market_state: Dict) -> float:
        """
        Calculate timing cost (implementation shortfall).
          Difference between decision price and execution price.
        """
        decision_price = trade.get('decision_price', trade['price'])
        execution_price = trade['price']

        return abs(execution_price - decision_price) * trade['size']

    def _get_market_state(self, market_data: pd.DataFrame,
                         timestamp: pd.Timestamp, ticker: str) -> Dict:
        """Extract market state at trade time."""
        # Find the row closest to trade timestamp
        if hasattr(market_data.index, 'get_loc') and hasattr(market_data.index, 'searchsorted'):
            try:
                idx = market_data.index.get_loc(timestamp, method='nearest')
            except (TypeError, KeyError):
                # Fallback for RangeIndex or when timestamp not found
                if 'timestamp' in market_data.columns:
                    time_diffs = abs(market_data['timestamp'] - timestamp)
                    idx = time_diffs.idxmin()
                else:
                    idx = len(market_data) // 2  # Use middle row as fallback
        else:
            # For RangeIndex, use the last available row
            idx = len(market_data) - 1

        row = market_data.iloc[idx]        # Calculate additional metrics
        volatility = self._calculate_volatility(market_data, idx)
        adv = self._calculate_adv(market_data, idx)

        return {
            'price': row['close'],
            'volume': row.get('volume', 1000000),  # Default volume if missing
            'spread': row.get('spread', self._calculate_default_spread(row)),
            'volatility': volatility,
            'adv': adv,
            'timestamp': timestamp
        }
    
    def _calculate_default_spread(self, row):
        """Calculate default spread when high/low not available."""
        if 'high' in row and 'low' in row and 'close' in row:
            return (row['high'] - row['low']) / row['close']
        else:
            # Default to 0.1% spread if high/low not available
            return 0.001

    def _calculate_volatility(self, data: pd.DataFrame, current_idx: int,
                            lookback: int = 20) -> float:
        """Calculate realized volatility."""
        start_idx = max(0, current_idx - lookback)
        returns = data.iloc[start_idx:current_idx]['close'].pct_change().dropna()

        if len(returns) < 2:
            return 0.02  # Default 2% daily volatility

        return returns.std()

    def _calculate_adv(self, data: pd.DataFrame, current_idx: int,
                      lookback: int = 20) -> float:
        """Calculate average daily volume."""
        start_idx = max(0, current_idx - lookback)
        volumes = data.iloc[start_idx:current_idx]['volume']

        return volumes.mean() if len(volumes) > 0 else 0

    def _default_spread_model(self, ticker: str, timestamp: pd.Timestamp) -> float:
        """Default spread model - can be overridden with actual data."""
        # In production, this would query actual bid-ask data
        return 0.01  # 1 cent default spread

    def _default_impact_model(self, size: float, adv: float) -> float:
        """Default market impact model."""
        if adv == 0:
            return 0
        return 0.1 * np.sqrt(size / adv)  # 10bps * sqrt(participation)