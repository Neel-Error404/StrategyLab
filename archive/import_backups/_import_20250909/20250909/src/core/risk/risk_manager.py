# src/risk/risk_manager.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

class RiskManager:
    """
    Comprehensive risk management framework for backtesting and live trading.
    """
    
    def __init__(self, config):
        """Initialize RiskManager with RiskConfig dataclass or dictionary."""
        self.config = config
        self.logger = logging.getLogger("RiskManager")
        
        # Handle both dataclass and dictionary configurations
        if hasattr(config, 'bypass_mode'):
            # It's a RiskConfig dataclass
            self.bypass_mode = getattr(config, 'bypass_mode', False)
            self.max_position_size = getattr(config, 'max_position_size', 0.1)
            self.max_drawdown = getattr(config, 'max_drawdown', 0.2)
            self.max_daily_loss = getattr(config, 'max_daily_loss', 0.02)
            self.stop_loss_pct = getattr(config, 'stop_loss_pct', 0.05)
            self.take_profit_pct = getattr(config, 'take_profit_pct', 0.10)
            self.position_timeout_minutes = getattr(config, 'position_timeout_minutes', 240)
            self.enable_stop_loss = getattr(config, 'enable_stop_loss', True)
            self.enable_take_profit = getattr(config, 'enable_take_profit', True)
            self.enable_timeout = getattr(config, 'enable_timeout', True)
        else:
            # It's a dictionary (legacy support)
            self.bypass_mode = config.get('bypass_mode', False)
            self.max_position_size = config.get('max_position_size', 0.1)
            self.max_drawdown = config.get('max_drawdown', 0.2)
            self.max_daily_loss = config.get('max_daily_loss', 0.02)
            self.stop_loss_pct = config.get('stop_loss_pct', 0.05)
            self.take_profit_pct = config.get('take_profit_pct', 0.10)
            self.position_timeout_minutes = config.get('position_timeout_minutes', 240)
            self.enable_stop_loss = config.get('enable_stop_loss', True)
            self.enable_take_profit = config.get('enable_take_profit', True)
            self.enable_timeout = config.get('enable_timeout', True)
          # Additional legacy attributes for compatibility
        self.max_sector_exposure = 0.3  # 30% sector limit  
        self.max_leverage = 1.0  # No leverage by default
        self.max_concentration = config.get('max_concentration', 0.5)  # 50% default for diversification
        self.position_limits = {}
        
        # Risk metrics tracking
        self.risk_metrics = {
            'current_drawdown': 0,
            'max_drawdown_reached': 0,
            'var_95': 0,
            'cvar_95': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0        }
          # NEW: Risk rejection tracking
        self.rejection_stats = {
            'total_trades_evaluated': 0,
            'total_trades_approved': 0,
            'total_trades_rejected': 0,
            'rejection_reasons': {},  # Count by reason
            'detailed_rejections': []  # Detailed rejection log
        }
        
        # NEW: Portfolio State Management for Multi-Ticker Coordination
        self.portfolio_state = {
            'initialized': False,
            'total_capital': 0,
            'allocated_capital': 0,
            'available_capital': 0,
            'ticker_allocations': {},  # ticker -> allocated amount
            'ticker_positions': {},    # ticker -> current position value
            'ticker_count': 0,
            'max_per_ticker': 0,      # maximum capital per ticker
            'portfolio_exposure': 0,   # total portfolio exposure
            'portfolio_concentration': {}  # ticker -> concentration ratio
        }
    
    def validate_trade(self, proposed_trade: Dict, 
                      portfolio: Dict, 
                      market_data: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate a proposed trade against risk limits.
        
        Args:
            proposed_trade: Trade to validate
            portfolio: Current portfolio state
            market_data: Current market data
            
        Returns:
            Tuple of (is_valid, risk_report)
        """
        # Track statistics
        self.rejection_stats['total_trades_evaluated'] += 1
        
        risk_checks = {
            'position_size': self._check_position_size(proposed_trade, portfolio),
            'portfolio_exposure': self._check_portfolio_exposure(proposed_trade, portfolio),
            'drawdown': self._check_drawdown_limit(portfolio),
            'leverage': self._check_leverage(proposed_trade, portfolio),
            'concentration': self._check_concentration_risk(proposed_trade, portfolio),
            'liquidity': self._check_liquidity_risk(proposed_trade, market_data),
            'volatility': self._check_volatility_risk(proposed_trade, market_data)
        }
        
        # Check if all risk checks pass
        all_passed = all(check['passed'] for check in risk_checks.values())
        
        # Generate risk report
        risk_report = {
            'approved': all_passed,
            'checks': risk_checks,
            'risk_score': self._calculate_risk_score(risk_checks),
            'recommended_size': self._calculate_recommended_size(proposed_trade, portfolio, market_data)
        }
          # Update rejection statistics
        if all_passed:
            self.rejection_stats['total_trades_approved'] += 1
            
            # Update portfolio state if trade is approved and portfolio mode is active
            if self.portfolio_state['initialized']:
                trade_value = proposed_trade['size'] * proposed_trade['price']
                self.update_portfolio_position(proposed_trade['ticker'], trade_value)
        else:
            self.rejection_stats['total_trades_rejected'] += 1
            failed_checks = [name for name, check in risk_checks.items() if not check['passed']]
            
            # Count rejections by reason
            for reason in failed_checks:
                if reason not in self.rejection_stats['rejection_reasons']:
                    self.rejection_stats['rejection_reasons'][reason] = 0
                self.rejection_stats['rejection_reasons'][reason] += 1
            
            # Log detailed rejection
            rejection_detail = {
                'ticker': proposed_trade.get('ticker', 'UNKNOWN'),
                'timestamp': proposed_trade.get('timestamp', pd.Timestamp.now()),
                'price': proposed_trade.get('price', 0),
                'size': proposed_trade.get('size', 0),
                'failed_checks': failed_checks,
                'check_details': {name: check['message'] for name, check in risk_checks.items() if not check['passed']}
            }
            self.rejection_stats['detailed_rejections'].append(rejection_detail)
            
            self.logger.warning(f"Trade rejected for {proposed_trade.get('ticker', 'UNKNOWN')}. "
                              f"Failed checks: {failed_checks}")
            for name, check in risk_checks.items():
                if not check['passed']:
                    self.logger.warning(f"  {name}: {check['message']}")
        
        return all_passed, risk_report
    
    def initialize_portfolio_state(self, total_capital: float, tickers: List[str], 
                                 allocation_method: str = 'equal_weight') -> None:
        """
        Initialize portfolio state for multi-ticker risk management.
        
        Args:
            total_capital: Total available capital for the portfolio
            tickers: List of ticker symbols to allocate capital to
            allocation_method: Method for capital allocation ('equal_weight', 'risk_parity', etc.)
        """
        self.logger.info(f"Initializing portfolio state: {total_capital:,.0f} capital across {len(tickers)} tickers")
        
        ticker_count = len(tickers)
        if ticker_count == 0:
            raise ValueError("Cannot initialize portfolio with zero tickers")
        
        # Calculate per-ticker allocation based on method
        if allocation_method == 'equal_weight':
            max_per_ticker = total_capital / ticker_count
        else:
            # Future: implement risk parity and other allocation methods
            max_per_ticker = total_capital / ticker_count
            
        # Initialize portfolio state
        self.portfolio_state.update({
            'initialized': True,
            'total_capital': total_capital,
            'allocated_capital': 0,
            'available_capital': total_capital,
            'ticker_count': ticker_count,
            'max_per_ticker': max_per_ticker,
            'portfolio_exposure': 0,
            'allocation_method': allocation_method
        })
        
        # Initialize ticker-specific allocations
        for ticker in tickers:
            self.portfolio_state['ticker_allocations'][ticker] = 0
            self.portfolio_state['ticker_positions'][ticker] = 0
            self.portfolio_state['portfolio_concentration'][ticker] = 0
            
        self.logger.info(f"Portfolio state initialized: {max_per_ticker:,.0f} max capital per ticker")
        
    def update_portfolio_position(self, ticker: str, position_value: float) -> None:
        """
        Update portfolio state when a position is taken.
        
        Args:
            ticker: Ticker symbol
            position_value: Value of the new position
        """
        if not self.portfolio_state['initialized']:
            self.logger.warning("Portfolio state not initialized, treating as single ticker")
            return
            
        # Update ticker position
        old_position = self.portfolio_state['ticker_positions'].get(ticker, 0)
        self.portfolio_state['ticker_positions'][ticker] = position_value
        
        # Update portfolio-wide metrics
        position_change = position_value - old_position
        self.portfolio_state['allocated_capital'] += position_change
        self.portfolio_state['available_capital'] -= position_change
        self.portfolio_state['portfolio_exposure'] = (
            self.portfolio_state['allocated_capital'] / self.portfolio_state['total_capital']
        )
        
        # Recalculate concentrations
        self._recalculate_portfolio_concentrations()
        
        self.logger.debug(f"Updated {ticker} position: {position_value:,.0f}, "
                         f"Portfolio exposure: {self.portfolio_state['portfolio_exposure']:.1%}")
    
    def get_available_capital_for_ticker(self, ticker: str) -> float:
        """
        Get remaining capital available for a specific ticker.
        
        Args:
            ticker: Ticker symbol
              Returns:
            Available capital for this ticker
        """
        if not self.portfolio_state['initialized']:
            # Fallback to original behavior for single ticker
            return self.portfolio_state.get('total_capital', 1000000)
        
        current_position = self.portfolio_state['ticker_positions'].get(ticker, 0)
        max_allowed = self.portfolio_state['max_per_ticker']
        
        return max(0, max_allowed - current_position)
    
    def get_portfolio_concentration(self, ticker: str, proposed_trade_value: float) -> float:
        """
        Calculate what portfolio concentration would be after a proposed trade.
        For portfolio diversification, this calculates concentration against total capital,
        not just allocated capital, to properly support equal-weight allocation.
        
        Args:
            ticker: Ticker symbol
            proposed_trade_value: Value of the proposed trade
            
        Returns:
            Concentration ratio after the trade (0-1)
        """
        if not self.portfolio_state['initialized']:
            # Fallback for single ticker mode
            return 1.0
            
        current_position = self.portfolio_state['ticker_positions'].get(ticker, 0)
        new_position_value = current_position + proposed_trade_value
        
        # For portfolio diversification, calculate concentration against total capital
        # This allows proper equal-weight allocation across tickers
        total_capital = self.portfolio_state['total_capital']
        
        if total_capital <= 0:
            return 1.0
            
        return new_position_value / total_capital
    
    def _recalculate_portfolio_concentrations(self) -> None:
        """Recalculate concentration ratios for all tickers."""
        total_allocated = self.portfolio_state['allocated_capital']
        
        if total_allocated <= 0:
            # No positions, all concentrations are zero
            for ticker in self.portfolio_state['portfolio_concentration']:
                self.portfolio_state['portfolio_concentration'][ticker] = 0.0
            return
            
        for ticker, position_value in self.portfolio_state['ticker_positions'].items():
            concentration = position_value / total_allocated if total_allocated > 0 else 0
            self.portfolio_state['portfolio_concentration'][ticker] = concentration
    
    def reset_portfolio_state(self) -> None:
        """Reset portfolio state for a new backtest run."""
        self.portfolio_state = {
            'initialized': False,
            'total_capital': 0,
            'allocated_capital': 0,
            'available_capital': 0,
            'ticker_allocations': {},
            'ticker_positions': {},
            'ticker_count': 0,
            'max_per_ticker': 0,
            'portfolio_exposure': 0,
            'portfolio_concentration': {}
        }
        self.logger.info("Portfolio state reset")
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio state summary."""
        if not self.portfolio_state['initialized']:
            return {'portfolio_mode': False, 'message': 'Single ticker mode'}
            
        return {
            'portfolio_mode': True,
            'total_capital': self.portfolio_state['total_capital'],
            'allocated_capital': self.portfolio_state['allocated_capital'],
            'available_capital': self.portfolio_state['available_capital'],
            'portfolio_exposure': self.portfolio_state['portfolio_exposure'],
            'ticker_count': self.portfolio_state['ticker_count'],
            'max_per_ticker': self.portfolio_state['max_per_ticker'],
            'allocation_method': self.portfolio_state.get('allocation_method', 'equal_weight'),
            'ticker_positions': dict(self.portfolio_state['ticker_positions']),
            'portfolio_concentrations': dict(self.portfolio_state['portfolio_concentration'])
        }
    
    def calculate_portfolio_risk_metrics(self, portfolio: Dict, 
                                       returns: pd.Series,
                                       benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate comprehensive portfolio risk metrics.
        """
        if len(returns) < 2:
            return self.risk_metrics
        
        # Basic statistics
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Drawdown analysis
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        
        # Value at Risk (VaR) and Conditional VaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Risk-adjusted returns
        risk_free_rate = 0.05 / 252  # Daily risk-free rate
        excess_returns = returns - risk_free_rate
        
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if std_return > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else std_return
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        max_dd = drawdown.min()
        calmar_ratio = mean_return * 252 / abs(max_dd) if max_dd < 0 else 0
        
        # Update risk metrics
        self.risk_metrics.update({
            'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0,
            'max_drawdown_reached': max_dd,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'volatility': std_return * np.sqrt(252),
            'downside_volatility': downside_std * np.sqrt(252),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'max_daily_loss': returns.min(),
            'max_daily_gain': returns.max()
        })
        
        # Calculate beta and alpha if benchmark provided
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            alpha = (mean_return - beta * benchmark_returns.mean()) * 252
            
            self.risk_metrics.update({
                'beta': beta,
                'alpha': alpha,
                'correlation': returns.corr(benchmark_returns),
                'tracking_error': (returns - benchmark_returns).std() * np.sqrt(252)
            })
        
        return self.risk_metrics
    
    def calculate_position_size(self, signal_strength: float,
                              portfolio_value: float,
                              volatility: float,
                              method: str = 'kelly') -> float:
        """
        Calculate optimal position size based on risk.
        
        Methods:
        - kelly: Kelly Criterion
        - fixed_fractional: Fixed percentage
        - volatility_targeting: Target specific volatility
        """
        if method == 'kelly':
            # Simplified Kelly Criterion
            # f = (p * b - q) / b
            # where p = win probability, b = win/loss ratio, q = 1-p
            win_rate = 0.55  # Assumed from historical data
            avg_win_loss_ratio = 1.5  # Assumed
            
            kelly_fraction = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            position_size = portfolio_value * kelly_fraction * signal_strength
            
        elif method == 'fixed_fractional':
            position_size = portfolio_value * self.max_position_size * signal_strength
            
        elif method == 'volatility_targeting':
            target_volatility = 0.15  # 15% annual volatility target
            position_volatility = volatility
            
            if position_volatility > 0:
                position_size = (portfolio_value * target_volatility / position_volatility) * signal_strength
            else:
                position_size = 0
        
        else:
            position_size = portfolio_value * 0.02  # Default 2% position
        
        # Apply maximum position size limit
        max_allowed = portfolio_value * self.max_position_size
        position_size = min(position_size, max_allowed)
        
        return position_size
    
    def _check_position_size(self, trade: Dict, portfolio: Dict) -> Dict:
        """Check if position size is within limits."""
        portfolio_value = portfolio.get('total_value', 0)
        trade_value = trade['size'] * trade['price']
        
        position_pct = trade_value / portfolio_value if portfolio_value > 0 else 1
        
        return {
            'passed': position_pct <= self.max_position_size,
            'current': position_pct,
            'limit': self.max_position_size,
            'message': f"Position size: {position_pct:.2%} of portfolio"
        }
    
    def _check_portfolio_exposure(self, trade: Dict, portfolio: Dict) -> Dict:
        """Check total portfolio exposure."""
        current_exposure = sum(pos['value'] for pos in portfolio.get('positions', {}).values())
        new_exposure = current_exposure + trade['size'] * trade['price']
        portfolio_value = portfolio.get('total_value', 0)
        
        exposure_pct = new_exposure / portfolio_value if portfolio_value > 0 else 1
        
        return {
            'passed': exposure_pct <= 1.0,  # No more than 100% invested
            'current': exposure_pct,
            'limit': 1.0,
            'message': f"Total exposure: {exposure_pct:.2%}"
        }
    
    def _check_drawdown_limit(self, portfolio: Dict) -> Dict:
        """Check if current drawdown exceeds limit."""
        current_dd = abs(self.risk_metrics.get('current_drawdown', 0))
        
        return {
            'passed': current_dd < self.max_drawdown,
            'current': current_dd,
            'limit': self.max_drawdown,
            'message': f"Current drawdown: {current_dd:.2%}"
        }
    
    def _check_leverage(self, trade: Dict, portfolio: Dict) -> Dict:
        """Check leverage constraints."""
        current_leverage = portfolio.get('leverage', 1.0)
        
        return {
            'passed': current_leverage <= self.max_leverage,
            'current': current_leverage,        'limit': self.max_leverage,
            'message': f"Current leverage: {current_leverage:.2f}x"
        }
    
    def _check_concentration_risk(self, trade: Dict, portfolio: Dict) -> Dict:
        """Check portfolio concentration risk (portfolio-aware)."""
        ticker = trade['ticker']
        trade_value = trade['size'] * trade['price']
        
        # Use configurable max concentration (default 50% for portfolio diversification)
        max_concentration_limit = getattr(self, 'max_concentration', 0.5)
        
        # Portfolio-aware concentration calculation
        if self.portfolio_state['initialized']:
            # Calculate concentration across the entire portfolio
            new_concentration = self.get_portfolio_concentration(ticker, trade_value)
            
            # Get current portfolio summary for detailed reporting
            portfolio_summary = self.get_portfolio_summary()
            
            self.logger.debug(f"Portfolio concentration check for {ticker}: "
                            f"new_concentration={new_concentration:.1%}, "
                            f"limit={max_concentration_limit:.1%}, "
                            f"portfolio_exposure={portfolio_summary['portfolio_exposure']:.1%}")
            
            return {
                'passed': new_concentration <= max_concentration_limit,
                'current': new_concentration,
                'limit': max_concentration_limit,
                'message': f"Portfolio concentration: {new_concentration:.2%} (limit: {max_concentration_limit:.2%})",
                'portfolio_mode': True,
                'portfolio_exposure': portfolio_summary['portfolio_exposure'],
                'ticker_positions': len([t for t, v in portfolio_summary['ticker_positions'].items() if v > 0])
            }
        else:
            # Fallback to original single-ticker logic
            positions = portfolio.get('positions', {})
            
            # Calculate current concentration
            position_values = [pos['value'] for pos in positions.values()]
            total_value = sum(position_values)
            
            if total_value > 0:
                max_concentration = max(position_values) / total_value if position_values else 0
            else:
                max_concentration = 0
            
            # Check what concentration would be after trade
            new_position_value = positions.get(ticker, {}).get('value', 0) + trade_value
            new_total_value = total_value + trade_value
            new_concentration = new_position_value / new_total_value if new_total_value > 0 else 1
            
            return {
                'passed': new_concentration <= max_concentration_limit,
                'current': new_concentration,
                'limit': max_concentration_limit,
                'message': f"Single ticker concentration: {new_concentration:.2%}",
                'portfolio_mode': False
            }
    
    def _check_liquidity_risk(self, trade: Dict, market_data: pd.DataFrame) -> Dict:
        """Check if trade size is appropriate for market liquidity."""
        # Get average daily volume
        adv = market_data['volume'].rolling(20).mean().iloc[-1]
        trade_volume = trade['size']
        
        participation_rate = trade_volume / adv if adv > 0 else 1
        
        return {
            'passed': participation_rate <= 0.1,  # Max 10% of ADV
            'current': participation_rate,
            'limit': 0.1,
            'message': f"Trade is {participation_rate:.2%} of ADV"
        }
    
    def _check_volatility_risk(self, trade: Dict, market_data: pd.DataFrame) -> Dict:
        """Check if volatility is within acceptable range."""
        returns = market_data['close'].pct_change().dropna()
        current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        max_acceptable_vol = 0.5  # 50% annualized volatility
        
        return {
            'passed': current_vol <= max_acceptable_vol,
            'current': current_vol,
            'limit': max_acceptable_vol,
            'message': f"Current volatility: {current_vol:.2%} annualized"
        }
    
    def _calculate_risk_score(self, risk_checks: Dict) -> float:
        """Calculate overall risk score (0-100)."""
        scores = []
        
        for check in risk_checks.values():
            if check['passed']:
                # Score based on how close to limit
                utilization = check['current'] / check['limit'] if check['limit'] > 0 else 0
                scores.append(utilization * 100)
            else:
                scores.append(100)  # Max risk score for failed checks
        
        return np.mean(scores) if scores else 0
    
    def _calculate_recommended_size(self, trade: Dict, portfolio: Dict, 
                                  market_data: pd.DataFrame) -> float:
        """Calculate recommended position size based on risk constraints."""
        portfolio_value = portfolio.get('total_value', 0)
        
        # Get volatility
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Calculate size using selected method
        recommended_size = self.calculate_position_size(
            signal_strength=1.0,  # Full signal
            portfolio_value=portfolio_value,
            volatility=volatility,
            method='volatility_targeting'
        )
        
        return recommended_size
    
    def get_rejection_summary(self) -> Dict:
        """Get comprehensive risk rejection statistics."""
        return {
            'total_evaluated': self.rejection_stats['total_trades_evaluated'],
            'total_approved': self.rejection_stats['total_trades_approved'],
            'total_rejected': self.rejection_stats['total_trades_rejected'],
            'approval_rate': (
                self.rejection_stats['total_trades_approved'] / 
                self.rejection_stats['total_trades_evaluated'] 
                if self.rejection_stats['total_trades_evaluated'] > 0 else 0
            ),
            'rejection_rate': (
                self.rejection_stats['total_trades_rejected'] / 
                self.rejection_stats['total_trades_evaluated'] 
                if self.rejection_stats['total_trades_evaluated'] > 0 else 0
            ),
            'rejection_reasons': self.rejection_stats['rejection_reasons'].copy(),
            'most_common_rejection': max(
                self.rejection_stats['rejection_reasons'].items(), 
                key=lambda x: x[1]
            ) if self.rejection_stats['rejection_reasons'] else None
        }
        
    def reset_rejection_stats(self):
        """Reset rejection statistics for new run."""
        self.rejection_stats = {
            'total_trades_evaluated': 0,
            'total_trades_approved': 0,
            'total_trades_rejected': 0,
            'rejection_reasons': {},
            'detailed_rejections': []
        }
    
    def get_detailed_rejection_report(self) -> Dict:
        """Get detailed risk rejection report for final reporting."""
        summary = self.get_rejection_summary()
        
        # Add detailed breakdown by rule
        rule_breakdown = {}
        for rule, count in self.rejection_stats['rejection_reasons'].items():
            rule_breakdown[rule] = {
                'count': count,
                'percentage': (count / self.rejection_stats['total_trades_evaluated'] * 100) 
                             if self.rejection_stats['total_trades_evaluated'] > 0 else 0,
                'description': self._get_rule_description(rule)
            }
        
        return {
            'summary': summary,
            'rule_breakdown': rule_breakdown,
            'sample_rejections': self.rejection_stats['detailed_rejections'][-5:],  # Last 5 rejections
            'has_rejections': self.rejection_stats['total_trades_rejected'] > 0
        }
    
    def _get_rule_description(self, rule_name: str) -> str:
        """Get human-readable description of risk rule."""
        descriptions = {
            'position_size': 'Trade size exceeds maximum position size limit',
            'portfolio_exposure': 'Total portfolio exposure exceeds limit',
            'drawdown': 'Portfolio drawdown exceeds maximum allowed',
            'leverage': 'Leverage constraint violation',
            'concentration': 'Portfolio concentration risk too high',
            'liquidity': 'Trade size inappropriate for market liquidity',
            'volatility': 'Market volatility exceeds acceptable range'
        }
        return descriptions.get(rule_name, f'Risk rule: {rule_name}')
    
    def get_zero_trades_attribution(self, strategy_generated_trades: int) -> Dict:
        """
        Determine why there are zero trades in final results.
        
        Args:
            strategy_generated_trades: Number of trades generated by strategy
            
        Returns:
            Attribution report explaining zero trades
        """
        if strategy_generated_trades == 0:
            return {
                'cause': 'strategy',
                'explanation': 'Strategy did not generate any trade signals',
                'recommendation': 'Review strategy parameters and market conditions'
            }
        elif self.rejection_stats['total_trades_rejected'] == strategy_generated_trades:
            most_common = max(
                self.rejection_stats['rejection_reasons'].items(),
                key=lambda x: x[1]
            ) if self.rejection_stats['rejection_reasons'] else ('unknown', 0)
            
            return {
                'cause': 'risk_manager',
                'explanation': f'All {strategy_generated_trades} trades rejected by risk management',
                'primary_reason': most_common[0],
                'primary_reason_count': most_common[1],
                'all_rejection_reasons': dict(self.rejection_stats['rejection_reasons']),
                'recommendation': f'Consider relaxing {most_common[0]} constraints or review risk parameters'
            }
        else:
            return {
                'cause': 'partial_rejection',
                'explanation': f'Strategy generated {strategy_generated_trades} trades, '
                             f'{self.rejection_stats["total_trades_rejected"]} rejected by risk management',
                'approved_trades': self.rejection_stats['total_trades_approved'],
                'rejection_reasons': dict(self.rejection_stats['rejection_reasons'])
            }