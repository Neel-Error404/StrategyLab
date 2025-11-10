# src/options/options_engine.py
import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta

class BlackScholesEngine:
    """
    Black-Scholes options pricing engine for backtesting.
    Handles option pricing, Greeks calculation, and synthetic option creation.
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
    
    def calculate_option_price(self, S: float, K: float, T: float, r: float, 
                             sigma: float, option_type: str = 'call') -> float:
        """
        Calculate option price using Black-Scholes formula.
        
        Args:
            S: Current price of underlying
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility of underlying
            option_type: 'call' or 'put'
        """
        if T <= 0:
            # Option expired
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        
        return price
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks for an option.
        """
        if T <= 0:
            return {
                'delta': 1.0 if (option_type == 'call' and S > K) else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = stats.norm.cdf(d1)
        else:
            delta = stats.norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type == 'call':
            theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                    - r * K * np.exp(-r * T) * stats.norm.cdf(d2)) / 365
        else:
            theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                    + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)) / 365
        
        # Vega (same for calls and puts)
        vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def calculate_implied_volatility(self, option_price: float, S: float, K: float, 
                                   T: float, r: float, option_type: str = 'call',
                                   tol: float = 1e-5, max_iter: int = 100) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        """
        # Initial guess
        sigma = 0.3
        
        for i in range(max_iter):
            # Calculate option price and vega
            price = self.calculate_option_price(S, K, T, r, sigma, option_type)
            greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
            vega = greeks['vega'] * 100  # Convert back from percentage
            
            # Newton-Raphson update
            price_diff = price - option_price
            
            if abs(price_diff) < tol:
                return sigma
            
            if vega == 0:
                return np.nan
            
            sigma = sigma - price_diff / vega
            
            # Ensure sigma stays positive
            sigma = max(sigma, 0.001)
        
        return np.nan  # Failed to converge

class OptionsBacktester:
    """
    Backtesting engine for options strategies.
    """
    
    def __init__(self, underlying_data: pd.DataFrame, 
                 options_data: Optional[pd.DataFrame] = None,
                 use_synthetic: bool = True):
        """
        Initialize options backtester.
        
        Args:
            underlying_data: Historical data for underlying asset
            options_data: Historical options data (if available)
            use_synthetic: Whether to use synthetic options via Black-Scholes
        """
        self.underlying_data = underlying_data
        self.options_data = options_data
        self.use_synthetic = use_synthetic
        self.bs_engine = BlackScholesEngine()
        
        # Calculate historical volatility if needed
        if use_synthetic:
            self._calculate_historical_volatility()
    
    def _calculate_historical_volatility(self, window: int = 20) -> None:
        """Calculate rolling historical volatility for synthetic options."""
        returns = self.underlying_data['close'].pct_change()
        self.underlying_data['hist_vol'] = returns.rolling(window).std() * np.sqrt(252)
        self.underlying_data['hist_vol'] = self.underlying_data['hist_vol'].fillna(0.3)  # Default 30%
    
    def get_option_price(self, date: datetime, strike: float, expiry: datetime,
                        option_type: str = 'call') -> Tuple[float, Dict[str, float]]:
        """
        Get option price and Greeks for a specific date.
        """
        # Get underlying price and volatility
        idx = self.underlying_data.index.get_loc(date, method='nearest')
        S = self.underlying_data.iloc[idx]['close']
        
        if self.use_synthetic:
            # Use historical volatility for synthetic options
            sigma = self.underlying_data.iloc[idx]['hist_vol']
            T = (expiry - date).days / 365.0
            
            price = self.bs_engine.calculate_option_price(
                S, strike, T, self.bs_engine.risk_free_rate, sigma, option_type
            )
            greeks = self.bs_engine.calculate_greeks(
                S, strike, T, self.bs_engine.risk_free_rate, sigma, option_type
            )
        else:
            # Look up actual options data
            # This would query historical options prices
            price, greeks = self._lookup_historical_option(date, strike, expiry, option_type)
        
        return price, greeks
    
    def backtest_options_strategy(self, strategy_func, start_date: datetime, 
                                end_date: datetime, initial_capital: float = 100000) -> pd.DataFrame:
        """
        Backtest an options strategy.
        
        Args:
            strategy_func: Function that generates options trades
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Starting capital
            
        Returns:
            DataFrame with backtest results
        """
        results = []
        positions = {}
        capital = initial_capital
        
        # Filter data for backtest period
        mask = (self.underlying_data.index >= start_date) & (self.underlying_data.index <= end_date)
        backtest_data = self.underlying_data[mask]
        
        for date, row in backtest_data.iterrows():
            # Update existing positions
            positions, pnl = self._update_positions(positions, date, row)
            capital += pnl
            
            # Generate new signals
            signals = strategy_func(date, row, positions)
            
            # Execute trades
            for signal in signals:
                if signal['action'] == 'open':
                    positions[signal['id']] = {
                        'type': signal['option_type'],
                        'strike': signal['strike'],
                        'expiry': signal['expiry'],
                        'quantity': signal['quantity'],
                        'entry_price': signal['price'],
                        'entry_date': date
                    }
                    capital -= signal['quantity'] * signal['price'] * 100  # Options multiplier
                    
                elif signal['action'] == 'close' and signal['id'] in positions:
                    position = positions[signal['id']]
                    exit_price = signal['price']
                    pnl = (exit_price - position['entry_price']) * position['quantity'] * 100
                    capital += pnl
                    del positions[signal['id']]
            
            # Record results
            results.append({
                'date': date,
                'underlying_price': row['close'],
                'capital': capital,
                'positions': len(positions),
                'daily_pnl': pnl
            })
        
        return pd.DataFrame(results)
    
    def _update_positions(self, positions: Dict, date: datetime, 
                         underlying_row: pd.Series) -> Tuple[Dict, float]:
        """Update option positions and calculate P&L."""
        total_pnl = 0
        
        for pos_id, position in list(positions.items()):
            # Check if option expired
            if date >= position['expiry']:
                # Calculate expiration P&L
                S = underlying_row['close']
                K = position['strike']
                
                if position['type'] == 'call':
                    intrinsic = max(S - K, 0)
                else:
                    intrinsic = max(K - S, 0)
                
                pnl = (intrinsic - position['entry_price']) * position['quantity'] * 100
                total_pnl += pnl
                
                # Remove expired position
                del positions[pos_id]
        
        return positions, total_pnl