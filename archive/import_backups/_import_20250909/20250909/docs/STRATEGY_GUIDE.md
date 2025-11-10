# 🎯 Strategy Development Guide

**Complete guide to creating, testing, and deploying custom trading strategies.**

---

## 🚀 **Quick Start: 4-Step Strategy Creation**

### **1️⃣ Copy Template**
```bash
# Copy the strategy template
cp src/strategies/strategy_template.py src/strategies/strategy_my_strategy.py
```

### **2️⃣ Implement Logic**
```python
# Edit strategy_my_strategy.py
class MyStrategy(StrategyBase):
    def __init__(self, **params):
        super().__init__(**params)
        self.lookback = params.get('lookback', 20)
        self.threshold = params.get('threshold', 0.02)
    
    def generate_signals(self, data, current_time, ticker):
        # Your strategy logic here
        return signals
```

### **3️⃣ Register Strategy**
```python
# Add to src/strategies/register_strategies.py
from .strategy_my_strategy import MyStrategy

def register_all_strategies():
    factory = StrategyFactory()
    factory.register('my_strategy', MyStrategy)
    # ... existing registrations
```

### **4️⃣ Test Strategy**
```bash
# Test with minimal template
python src/runners/unified_runner.py \
  --mode backtest \
  --template minimal \
  --strategies my_strategy \
  --dates 2024-01-01 \
  --tickers RELIANCE
```

---

## 🧬 **Strategy Framework Architecture**

### **Base Class: StrategyBase**

All strategies inherit from `StrategyBase` which provides:

```python
from src.strategies.strategy_base import StrategyBase

class MyStrategy(StrategyBase):
    def __init__(self, **params):
        super().__init__(**params)
        # Your initialization
    
    def generate_signals(self, data, current_time, ticker):
        # Core method: implement your logic here
        pass
    
    def get_required_history(self):
        # Return number of historical periods needed
        return self.lookback
```

### **Built-in Indicators** 

StrategyBase provides market-standard indicators:

```python
# Moving Averages
sma_20 = self.calculate_sma(data['close'], 20)
ema_12 = self.calculate_ema(data['close'], 12)

# Momentum Indicators  
rsi_14 = self.calculate_rsi(data['close'], 14)
macd_line, signal_line, histogram = self.calculate_macd(data['close'], 12, 26, 9)

# Volatility Indicators
upper_band, lower_band = self.calculate_bollinger_bands(data['close'], 20, 2.0)
atr_14 = self.calculate_atr(data, 14)

# Oscillators
stoch_k, stoch_d = self.calculate_stochastic(data, 14)
```

### **Data Access Methods**

```python
# Historical data access
historical_data = self.get_historical_data(ticker, current_time, periods=50)

# Data validation
if self.validate_market_data(data):
    # Proceed with signal generation
    pass
```

---

## 📝 **Complete Strategy Template**

```python
# src/strategies/strategy_example.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .strategy_base import StrategyBase

class ExampleStrategy(StrategyBase):
    """
    Example strategy demonstrating best practices.
    
    Strategy Logic:
    - Buy when price crosses above SMA and RSI < 70
    - Sell when price crosses below SMA or RSI > 80
    - Use ATR for position sizing
    """
    
    def __init__(self, **params):
        """Initialize strategy with parameters."""
        super().__init__(**params)
        
        # Strategy parameters with defaults
        self.sma_period = params.get('sma_period', 20)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_overbought = params.get('rsi_overbought', 80)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.atr_period = params.get('atr_period', 14)
        self.atr_multiplier = params.get('atr_multiplier', 2.0)
        
        # Strategy state
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = None
        self.stop_loss = None
        
    def get_required_history(self) -> int:
        """Return minimum historical periods needed."""
        return max(self.sma_period, self.rsi_period, self.atr_period) + 5
    
    def generate_signals(self, data: pd.DataFrame, current_time: datetime, 
                        ticker: str) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on strategy logic.
        
        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            current_time: Current timestamp
            ticker: Stock symbol
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        # Validate data quality
        if not self.validate_market_data(data):
            self.logger.warning(f"Invalid data for {ticker} at {current_time}")
            return signals
            
        # Ensure sufficient data
        if len(data) < self.get_required_history():
            return signals
            
        try:
            # Calculate indicators
            sma = self.calculate_sma(data['close'], self.sma_period)
            rsi = self.calculate_rsi(data['close'], self.rsi_period)
            atr = self.calculate_atr(data, self.atr_period)
            
            # Current values
            current_price = data['close'].iloc[-1]
            current_sma = sma.iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_atr = atr.iloc[-1]
            
            # Previous values for trend detection
            prev_price = data['close'].iloc[-2]
            prev_sma = sma.iloc[-2]
            
            # Entry Signals
            if self.position == 0:  # No current position
                
                # Long Entry: Price crosses above SMA and RSI not overbought
                if (prev_price <= prev_sma and current_price > current_sma and 
                    current_rsi < self.rsi_overbought):
                    
                    # Calculate position size using ATR
                    stop_distance = current_atr * self.atr_multiplier
                    position_size = self.calculate_position_size(
                        current_price, stop_distance, ticker
                    )
                    
                    signals.append({
                        'action': 'buy',
                        'ticker': ticker,
                        'price': current_price,
                        'quantity': position_size,
                        'timestamp': current_time,
                        'signal_strength': self._calculate_signal_strength(current_rsi),
                        'stop_loss': current_price - stop_distance,
                        'take_profit': current_price + (stop_distance * 1.5),
                        'reasoning': f"SMA crossover, RSI: {current_rsi:.1f}"
                    })
                    
                    # Update internal state
                    self.position = 1
                    self.entry_price = current_price
                    self.stop_loss = current_price - stop_distance
            
            # Exit Signals
            elif self.position == 1:  # Long position
                
                should_exit = False
                exit_reason = ""
                
                # Exit condition 1: Price crosses below SMA
                if prev_price >= prev_sma and current_price < current_sma:
                    should_exit = True
                    exit_reason = "SMA cross below"
                
                # Exit condition 2: RSI overbought
                elif current_rsi > self.rsi_overbought:
                    should_exit = True
                    exit_reason = f"RSI overbought: {current_rsi:.1f}"
                
                # Exit condition 3: Stop loss hit
                elif current_price <= self.stop_loss:
                    should_exit = True
                    exit_reason = "Stop loss triggered"
                
                if should_exit:
                    signals.append({
                        'action': 'sell',
                        'ticker': ticker,
                        'price': current_price,
                        'quantity': 'all',  # Close entire position
                        'timestamp': current_time,
                        'reasoning': exit_reason
                    })
                    
                    # Reset internal state
                    self.position = 0
                    self.entry_price = None
                    self.stop_loss = None
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {ticker}: {e}")
            
        return signals
    
    def calculate_position_size(self, price: float, stop_distance: float, 
                               ticker: str) -> int:
        """
        Calculate position size based on risk management rules.
        
        Args:
            price: Current price
            stop_distance: Distance to stop loss
            ticker: Stock symbol
            
        Returns:
            Number of shares to buy
        """
        # Risk 1% of capital per trade
        risk_per_trade = 0.01
        available_capital = self.get_available_capital()
        risk_amount = available_capital * risk_per_trade
        
        # Position size = Risk amount / Stop distance
        if stop_distance > 0:
            position_value = risk_amount / (stop_distance / price)
            position_size = int(position_value / price)
        else:
            position_size = 0
            
        return max(position_size, 0)
    
    def _calculate_signal_strength(self, rsi: float) -> float:
        """Calculate signal strength based on RSI."""
        if rsi < self.rsi_oversold:
            return 1.0  # Strong buy
        elif rsi < 50:
            return 0.7  # Moderate buy
        else:
            return 0.3  # Weak buy
    
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """Return current strategy parameters."""
        return {
            'sma_period': self.sma_period,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'atr_period': self.atr_period,
            'atr_multiplier': self.atr_multiplier
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy metadata."""
        return {
            'name': 'Example SMA-RSI Strategy',
            'description': 'Trend following with momentum confirmation',
            'version': '1.0.0',
            'parameters': self.get_strategy_parameters(),
            'timeframes': ['1minute', '5minute', '15minute'],
            'asset_classes': ['equity'],
            'risk_level': 'moderate'
        }
```

---

## 🛠️ **Advanced Strategy Features**

### **Multi-Timeframe Analysis**
```python
def generate_signals(self, data, current_time, ticker):
    # Access different timeframes
    daily_data = self.get_historical_data(ticker, current_time, 
                                         periods=50, timeframe='1day')
    hourly_data = self.get_historical_data(ticker, current_time, 
                                          periods=24, timeframe='1hour')
    
    # Combine insights from multiple timeframes
    daily_trend = self.calculate_sma(daily_data['close'], 20)
    hourly_momentum = self.calculate_rsi(hourly_data['close'], 14)
    
    # Generate signals based on multi-timeframe analysis
    signals = []
    if daily_trend.iloc[-1] > daily_trend.iloc[-2] and hourly_momentum.iloc[-1] < 70:
        # Daily uptrend + hourly not overbought = buy signal
        signals.append(self._create_buy_signal(data, current_time, ticker))
    
    return signals
```

### **Portfolio-Aware Strategies**
```python
def generate_signals(self, data, current_time, ticker):
    # Check portfolio exposure
    current_exposure = self.get_portfolio_exposure(ticker)
    total_exposure = self.get_total_portfolio_exposure()
    
    # Limit position size based on portfolio rules
    if current_exposure > 0.1 or total_exposure > 0.8:
        return []  # Skip if overexposed
    
    # Generate normal signals
    return self._generate_base_signals(data, current_time, ticker)
```

### **Dynamic Parameter Adjustment**
```python
def generate_signals(self, data, current_time, ticker):
    # Adjust parameters based on market conditions
    volatility = self.calculate_atr(data, 14).iloc[-1]
    
    if volatility > self.high_volatility_threshold:
        # Use shorter periods in high volatility
        self.sma_period = 10
        self.rsi_period = 7
    else:
        # Use standard periods in normal conditions
        self.sma_period = 20
        self.rsi_period = 14
    
    return self._generate_base_signals(data, current_time, ticker)
```

---

## 🧪 **Testing & Validation**

### **Unit Testing Your Strategy**
```python
# tests/test_my_strategy.py
import unittest
import pandas as pd
from datetime import datetime
from src.strategies.strategy_my_strategy import MyStrategy

class TestMyStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = MyStrategy(sma_period=20, rsi_period=14)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
    
    def test_signal_generation(self):
        signals = self.strategy.generate_signals(
            self.sample_data, datetime.now(), 'TEST'
        )
        self.assertIsInstance(signals, list)
    
    def test_parameter_validation(self):
        params = self.strategy.get_strategy_parameters()
        self.assertIn('sma_period', params)
        self.assertEqual(params['sma_period'], 20)

if __name__ == '__main__':
    unittest.main()
```

### **Strategy Validation Commands**
```bash
# Quick validation
python src/runners/unified_runner.py \
  --mode validate \
  --strategies my_strategy \
  --dates 2024-01-01

# Minimal backtest
python src/runners/unified_runner.py \
  --mode backtest \
  --template minimal \
  --strategies my_strategy \
  --dates 2024-01-01 \
  --tickers RELIANCE

# Extended validation
python src/runners/unified_runner.py \
  --mode backtest \
  --template conservative \
  --strategies my_strategy \
  --date-ranges 2024-01-01_to_2024-03-31 \
  --tickers RELIANCE TCS
```

---

## 🎯 **Strategy Examples**

### **1. Simple Moving Average Crossover**
```python
class SMAStrategy(StrategyBase):
    def __init__(self, **params):
        super().__init__(**params)
        self.fast_period = params.get('fast_period', 10)
        self.slow_period = params.get('slow_period', 30)
    
    def generate_signals(self, data, current_time, ticker):
        fast_sma = self.calculate_sma(data['close'], self.fast_period)
        slow_sma = self.calculate_sma(data['close'], self.slow_period)
        
        signals = []
        if fast_sma.iloc[-1] > slow_sma.iloc[-1] and fast_sma.iloc[-2] <= slow_sma.iloc[-2]:
            signals.append(self._create_buy_signal(data, current_time, ticker))
        elif fast_sma.iloc[-1] < slow_sma.iloc[-1] and fast_sma.iloc[-2] >= slow_sma.iloc[-2]:
            signals.append(self._create_sell_signal(data, current_time, ticker))
        
        return signals
```

### **2. Mean Reversion Strategy**
```python
class MeanReversionStrategy(StrategyBase):
    def __init__(self, **params):
        super().__init__(**params)
        self.bb_period = params.get('bb_period', 20)
        self.bb_std = params.get('bb_std', 2.0)
        self.rsi_period = params.get('rsi_period', 14)
    
    def generate_signals(self, data, current_time, ticker):
        # Calculate Bollinger Bands and RSI
        upper_band, lower_band = self.calculate_bollinger_bands(
            data['close'], self.bb_period, self.bb_std
        )
        rsi = self.calculate_rsi(data['close'], self.rsi_period)
        
        current_price = data['close'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        signals = []
        
        # Buy when price touches lower band and RSI oversold
        if current_price <= lower_band.iloc[-1] and current_rsi < 30:
            signals.append(self._create_buy_signal(data, current_time, ticker))
        
        # Sell when price touches upper band and RSI overbought  
        elif current_price >= upper_band.iloc[-1] and current_rsi > 70:
            signals.append(self._create_sell_signal(data, current_time, ticker))
        
        return signals
```

### **3. Momentum Strategy**
```python
class MomentumStrategy(StrategyBase):
    def __init__(self, **params):
        super().__init__(**params)
        self.lookback = params.get('lookback', 20)
        self.threshold = params.get('threshold', 0.02)
        
    def generate_signals(self, data, current_time, ticker):
        # Calculate momentum
        returns = data['close'].pct_change(self.lookback)
        current_momentum = returns.iloc[-1]
        
        signals = []
        
        # Buy on strong positive momentum
        if current_momentum > self.threshold:
            signals.append(self._create_buy_signal(data, current_time, ticker))
        
        # Sell on strong negative momentum
        elif current_momentum < -self.threshold:
            signals.append(self._create_sell_signal(data, current_time, ticker))
        
        return signals
```

---

## 📊 **Signal Structure**

### **Buy Signal Format**
```python
buy_signal = {
    'action': 'buy',
    'ticker': 'RELIANCE',
    'price': 2450.50,
    'quantity': 100,
    'timestamp': datetime(2024, 1, 1, 9, 30),
    'signal_strength': 0.8,          # 0.0 to 1.0
    'stop_loss': 2400.00,            # Optional
    'take_profit': 2500.00,          # Optional
    'reasoning': 'SMA crossover with RSI confirmation',
    'confidence': 0.75,              # Optional
    'indicators': {                  # Optional
        'sma_20': 2440.0,
        'rsi_14': 35.5,
        'volume_ratio': 1.2
    }
}
```

### **Sell Signal Format**
```python
sell_signal = {
    'action': 'sell',
    'ticker': 'RELIANCE',
    'price': 2475.50,
    'quantity': 'all',               # or specific number
    'timestamp': datetime(2024, 1, 1, 10, 30),
    'reasoning': 'Take profit target reached',
    'exit_type': 'take_profit'       # 'stop_loss', 'take_profit', 'signal'
}
```

---

## 🎛️ **Configuration Integration**

### **Strategy-Specific YAML Config**
```yaml
# config/templates/my_strategy.yaml
strategy:
  name: "my_strategy"
  parameters:
    sma_period: 20
    rsi_period: 14
    rsi_overbought: 80
    rsi_oversold: 30
    atr_multiplier: 2.0
  
risk:
  max_position_size: 0.10
  stop_loss_pct: 0.03
  take_profit_pct: 0.06
```

### **Using Custom Config**
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --config config/templates/my_strategy.yaml \
  --dates 2024-01-01 \
  --tickers RELIANCE
```

---

## 🚀 **Deployment Checklist**

### **✅ Before Production**

1. **Code Quality**
   - [ ] Strategy class inherits from `StrategyBase`
   - [ ] All required methods implemented
   - [ ] Proper error handling
   - [ ] Logging statements added

2. **Testing**
   - [ ] Unit tests written and passing
   - [ ] Validation tests completed
   - [ ] Minimal backtest successful
   - [ ] Extended backtest completed

3. **Registration**
   - [ ] Strategy registered in `register_strategies.py`
   - [ ] Strategy appears in available strategies list
   - [ ] Configuration template created

4. **Documentation**
   - [ ] Strategy logic documented
   - [ ] Parameters explained
   - [ ] Usage examples provided

### **✅ Production Deployment**

```bash
# 1. Final validation
python src/runners/unified_runner.py --mode validate --strategies my_strategy --dates 2024-01-01

# 2. Test with conservative template
python src/runners/unified_runner.py --mode backtest --template conservative --strategies my_strategy --dates 2024-01-01 --tickers RELIANCE

# 3. Production backtest
python src/runners/unified_runner.py --mode backtest --config my_strategy.yaml --date-ranges 2024-01-01_to_2024-12-31 --tickers RELIANCE TCS INFY --parallel
```

---

## 🤖 **AI-Assisted Strategy Development**

### **Strategy Creation Prompt**

Use this prompt with any AI model to get help creating strategies:

```
I want to create a custom trading strategy for the backtester system.

STRATEGY REQUIREMENTS:
- Strategy type: [trend following/mean reversion/momentum/breakout]
- Indicators to use: [SMA, EMA, RSI, MACD, Bollinger Bands, etc.]
- Entry conditions: [describe when to buy]
- Exit conditions: [describe when to sell]
- Risk management: [stop loss, take profit preferences]
- Timeframe: [1minute, 5minute, 15minute, 1hour, 1day]

SYSTEM CONTEXT:
- Framework: StrategyBase class with built-in indicators
- Available methods: calculate_sma, calculate_ema, calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_atr, calculate_stochastic
- Signal format: Dictionary with action, ticker, price, quantity, timestamp
- Data format: Pandas DataFrame with OHLCV columns

Please provide:
1. Complete strategy class implementation
2. Parameter recommendations  
3. Testing suggestions
4. Configuration YAML template
```

---

## 🛠️ **Advanced Topics**

### **Custom Indicators**
```python
def custom_indicator(self, data, period):
    """Implement your custom technical indicator."""
    # Your calculation logic
    return indicator_values

def generate_signals(self, data, current_time, ticker):
    custom_values = self.custom_indicator(data['close'], 20)
    # Use in signal generation
```

### **Machine Learning Integration**
```python
from sklearn.ensemble import RandomForestClassifier

class MLStrategy(StrategyBase):
    def __init__(self, **params):
        super().__init__(**params)
        self.model = RandomForestClassifier()
        self.is_trained = False
    
    def train_model(self, historical_data):
        # Prepare features and train model
        pass
    
    def generate_signals(self, data, current_time, ticker):
        if not self.is_trained:
            return []
        
        # Use model for predictions
        features = self.prepare_features(data)
        prediction = self.model.predict(features)
        # Generate signals based on ML predictions
```

### **Options Strategies**
```python
class OptionsStrategy(StrategyBase):
    def generate_signals(self, data, current_time, ticker):
        # Access options data if available
        if hasattr(self, 'options_data'):
            # Implement options-specific logic
            return self._generate_options_signals(data, current_time, ticker)
        
        return []
```

---

## 📞 **Getting Help**

### **Common Issues**
- **Strategy not found**: Check registration in `register_strategies.py`
- **Insufficient data**: Increase `get_required_history()` return value
- **Signal errors**: Validate signal dictionary format
- **Performance issues**: Use vectorized operations, avoid loops

### **Resources**
- **Framework docs**: See `docs/` directory
- **Example strategies**: Check `src/strategies/` folder  
- **Testing guide**: See `docs/TESTING.md`
- **Configuration help**: See `docs/TEMPLATE_GUIDE.md`

### **AI Assistance**
Use the strategy creation prompt above with AI models for personalized help!

---

**Happy Strategy Development! 📈**
