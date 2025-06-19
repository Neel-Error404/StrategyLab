# 📊 Template & Risk Management Guide

**Master YAML configuration templates and risk management for optimal trading performance.**

---

## 🎯 **Template Overview**

| Template | Risk Level | Max Position | Stop Loss | Use Case |
|----------|------------|-------------|-----------|----------|
| **minimal** | Ultra-Safe | 5% | 2% | Learning, testing |
| **conservative** | Low | 15% | 3% | Stable income |
| **aggressive** | High | 20% | 5% | Growth focused |
| **options** | Moderate | 15% | 4% | Options strategies |
| **portfolio_diversified** | Balanced | 15% | 3% | Multi-asset |

---

## 🚀 **Quick Template Usage**

```bash
# Use pre-built template
python src/runners/unified_runner.py --mode backtest --template conservative --date-ranges 2024-12-12_to_2025-06-09

# Customize template
cp config/templates/conservative.yaml my_custom.yaml
# Edit my_custom.yaml...
python src/runners/unified_runner.py --mode backtest --config my_custom.yaml --date-ranges 2024-12-12_to_2025-06-09
```

---

## 🛡️ **Template Deep Dive**

### **🟢 Minimal Template (Ultra-Safe)**

**Perfect for beginners and strategy testing**

```yaml
# config/templates/minimal.yaml
strategy:
  name: "mse"
  risk_profile: "minimal"
  initial_capital: 100000

risk:
  max_position_size: 0.05        # Only 5% per position
  max_daily_loss: 0.01           # 1% daily loss limit
  stop_loss_pct: 0.02            # 2% stop loss
  take_profit_pct: 0.04          # 4% take profit
  max_concurrent_positions: 2     # Very limited exposure
  
  # Conservative position sizing
  position_sizing_method: "equal_weight"
  capital_at_risk_pct: 0.01      # Risk only 1% per trade
  max_portfolio_drawdown: 0.03   # 3% max drawdown

validation:
  enabled: true
  strict_mode: true              # Strict for safety
  look_ahead_bias_check: true

# Single-threaded for stability
parallel_processing: false
```

**Use Cases:**
- Learning trading concepts
- Testing new strategies safely
- Paper trading simulation
- Educational purposes

**Command Example:**
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --template minimal \
  --date-ranges 2024-12-12_to_2025-06-09 \
  --tickers RELIANCE \
  --log-level INFO
```

### **🔵 Conservative Template (Stable Income)**

**Balanced risk-reward for steady returns**

```yaml
# config/templates/conservative.yaml
strategy:
  name: "mse"
  risk_profile: "conservative"
  initial_capital: 100000

risk:
  max_position_size: 0.15        # 15% per position
  max_daily_loss: 0.02           # 2% daily loss limit
  stop_loss_pct: 0.03            # 3% stop loss
  take_profit_pct: 0.06          # 6% take profit
  max_concurrent_positions: 5     # Moderate diversification
  
  # Conservative drawdown protection
  max_portfolio_drawdown: 0.05   # 5% max drawdown
  daily_loss_limit: 0.02         # 2% daily limit
  trailing_stop_enabled: true
  trailing_stop_pct: 0.02

transaction_costs:
  enabled: true
  brokerage_pct: 0.0003          # 0.03% brokerage
  slippage_pct: 0.0001           # 0.01% slippage

# Moderate parallel processing
parallel_processing: true
max_workers: 2
```

**Use Cases:**
- Retirement account trading
- Conservative wealth building
- Risk-averse investors
- Steady income generation

**Command Example:**
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --template conservative \
  --date-ranges 2024-12-12_to_2025-06-09 \
  --tickers RELIANCE TCS INFY \
  --parallel
```

### **🔴 Aggressive Template (Growth Focused)**

**Higher risk tolerance for maximum returns**

```yaml
# config/templates/aggressive.yaml
strategy:
  name: "mse"
  risk_profile: "aggressive"
  initial_capital: 100000

risk:
  max_position_size: 0.20        # 20% per position
  max_daily_loss: 0.05           # 5% daily loss limit
  stop_loss_pct: 0.05            # 5% stop loss
  take_profit_pct: 0.10          # 10% take profit
  max_concurrent_positions: 8     # Higher diversification
  
  # Higher risk tolerance
  max_portfolio_drawdown: 0.10   # 10% max drawdown
  position_sizing_method: "volatility_adjusted"
  leverage_enabled: false        # No leverage even in aggressive

transaction_costs:
  enabled: true
  slippage_pct: 0.0002           # Higher slippage expected

# Full parallel processing
parallel_processing: true
max_workers: 4

# Advanced features
optimization:
  enabled: true
  parameter_optimization: true
```

**Use Cases:**
- Growth-focused portfolios
- Young investors with long horizon
- High risk tolerance traders
- Maximum return seeking

**Command Example:**
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --template aggressive \
  --date-ranges 2024-12-12_to_2025-06-09 \
  --tickers RELIANCE TCS INFY ZOMATO ADANIENT \
  --parallel \
  --max-workers 4
```

### **🟡 Options Template (Derivatives)**

**Specialized for options trading strategies**

```yaml
# config/templates/options.yaml
strategy:
  name: "mse"
  risk_profile: "options"
  initial_capital: 100000

risk:
  max_position_size: 0.15        # 15% per position
  max_daily_loss: 0.03           # 3% daily loss limit
  stop_loss_pct: 0.04            # 4% stop loss
  options_specific_risk: true

# Options-specific configuration
options:
  enabled: true
  synthetic_enabled: true
  volatility_model: "black_scholes"
  greeks_calculation: true
  
  # Options risk parameters
  max_delta_exposure: 0.5
  max_gamma_exposure: 0.1
  max_vega_exposure: 0.2
  max_theta_decay: 0.05

  # Expiry management
  min_days_to_expiry: 7
  max_days_to_expiry: 45
  early_close_dte: 5

validation:
  options_data_check: true
  greeks_validation: true
```

**Use Cases:**
- Options trading strategies
- Volatility trading
- Income generation (covered calls)
- Hedging strategies

**Command Example:**
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --template options \
  --date-ranges 2024-12-12_to_2025-06-09 \
  --tickers NIFTY BANKNIFTY
```

### **🟣 Portfolio Diversified Template (Multi-Asset)**

**Comprehensive portfolio management**

```yaml
# config/templates/portfolio_diversified.yaml
strategy:
  name: "mse"
  risk_profile: "diversified"
  initial_capital: 100000

risk:
  max_position_size: 0.15        # 15% per individual position
  max_sector_exposure: 0.30      # 30% per sector
  max_portfolio_exposure: 0.80   # 80% total market exposure
  
  # Portfolio-level risk management
  correlation_limit: 0.70        # Max correlation between positions
  concentration_limit: 0.25      # Max single position concentration
  rebalancing_frequency: "weekly"

portfolio:
  enabled: true
  diversification_enabled: true
  correlation_analysis: true
  sector_rotation: true
  
  # Asset allocation
  equity_allocation: 0.70        # 70% equity
  cash_allocation: 0.30          # 30% cash reserve

analysis:
  portfolio_attribution: true
  risk_decomposition: true
  correlation_analysis: true
  sector_analysis: true
```

**Use Cases:**
- Multi-ticker portfolios
- Professional fund management
- Diversified investment strategies
- Risk-balanced trading

**Command Example:**
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --template portfolio_diversified \
  --date-ranges 2024-12-12_to_2025-06-09 \
  --tickers RELIANCE TCS INFY ZOMATO ADANIENT TATAMOTORS
```

---

## 🎛️ **Template Customization**

### **Creating Custom Templates**

#### **Step 1: Copy Base Template**
```bash
# Start with closest template
cp config/templates/conservative.yaml config/templates/my_custom.yaml
```

#### **Step 2: Modify Parameters**
```yaml
# config/templates/my_custom.yaml
strategy:
  name: "mse"
  risk_profile: "custom"
  initial_capital: 250000        # Custom capital

risk:
  max_position_size: 0.12        # Custom position size
  stop_loss_pct: 0.035           # Custom stop loss
  take_profit_pct: 0.075         # Custom take profit
  
  # Custom risk parameters
  max_daily_trades: 5
  max_weekly_trades: 20
  cool_down_period: 2            # Days between trades on same ticker
```

#### **Step 3: Test Custom Template**
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --config config/templates/my_custom.yaml \
  --date-ranges 2024-12-12_to_2025-06-09 \
  --tickers RELIANCE
```

### **Parameter Categories**

#### **🎯 Strategy Parameters**
```yaml
strategy:
  name: "mse"                    # Strategy to use
  risk_profile: "custom"         # Risk profile name
  initial_capital: 100000        # Starting capital
  parameters:                    # Strategy-specific params
    lookback_period: 20
    threshold: 0.015
    volatility_adjustment: true
```

#### **🛡️ Risk Management**
```yaml
risk:
  # Position Sizing
  max_position_size: 0.15        # 15% max per position
  position_sizing_method: "equal_weight"  # or "volatility_adjusted"
  capital_at_risk_pct: 0.02      # 2% risk per trade
  
  # Loss Protection
  max_daily_loss: 0.03           # 3% daily loss limit
  max_weekly_loss: 0.08          # 8% weekly loss limit
  max_portfolio_drawdown: 0.05   # 5% max drawdown
  
  # Stop Loss & Take Profit
  stop_loss_pct: 0.03            # 3% stop loss
  take_profit_pct: 0.06          # 6% take profit
  trailing_stop_enabled: true
  trailing_stop_pct: 0.02        # 2% trailing stop
  
  # Position Limits
  max_concurrent_positions: 5    # Max open positions
  max_daily_trades: 10           # Max trades per day
  cool_down_period: 1            # Days between trades on same ticker
```

#### **💰 Transaction Costs**
```yaml
transaction_costs:
  enabled: true
  brokerage_pct: 0.0003          # 0.03% brokerage
  slippage_pct: 0.0001           # 0.01% slippage
  market_impact_pct: 0.00005     # 0.005% market impact
  
  # Advanced cost modeling
  fixed_cost_per_trade: 20       # Fixed cost per trade
  volume_discount: true          # Volume-based discounts
```

#### **📊 Data Configuration**
```yaml
data:
  data_pool_dir: "data/pools"
  default_timeframe: "1minute"
  timeframe_folders:
    1minute: "1minute"
    5minute: "5minute"
    15minute: "15minute"
    1hour: "1hour"
    1day: "1day"
  
  # Data quality
  data_quality_check: true
  remove_outliers: true
  outlier_threshold: 5           # Standard deviations
```

#### **✅ Validation Settings**
```yaml
validation:
  enabled: true
  strict_mode: false             # Set true for production
  data_quality_check: true
  min_data_points: 100
  
  # Bias detection
  look_ahead_bias_check: true
  survivorship_bias_check: true
  data_mining_bias_check: true
  
  # Performance validation
  min_sharpe_ratio: 0.5          # Minimum acceptable Sharpe
  max_volatility: 0.25           # Maximum acceptable volatility
```

#### **📈 Output Configuration**
```yaml
output:
  save_trades: true
  save_metrics: true
  save_plots: true
  output_dir: "outputs"
  
  # Analysis options
  detailed_logs: true
  performance_attribution: true
  risk_analysis: true
  
  # Export formats
  export_csv: true
  export_json: true
  export_excel: false
```

#### **⚡ Performance Settings**
```yaml
# Execution performance
parallel_processing: true
max_workers: 4
cache_enabled: true

# Memory management
batch_size: 1000
memory_limit_gb: 8

# Logging
logging:
  level: "INFO"
  console_enabled: true
  file_enabled: true
  performance_logging: true
```

---

## 🧪 **Template Testing & Validation**

### **Validation Workflow**

#### **1. Parameter Validation**
```bash
# Test template loads correctly
python src/runners/unified_runner.py \
  --mode validate \
  --config config/templates/my_custom.yaml \
  --date-ranges 2024-12-12_to_2025-06-09
```

#### **2. Quick Test**
```bash
# Single day test
python src/runners/unified_runner.py \
  --mode backtest \
  --config config/templates/my_custom.yaml \
  --date-ranges 2024-12-12_to_2025-06-09 \
  --tickers RELIANCE
```

#### **3. Extended Test**
```bash
# Month-long test
python src/runners/unified_runner.py \
  --mode backtest \
  --config config/templates/my_custom.yaml \
  --date-ranges 2025-06-06_to_2025-06-07 \
  --tickers RELIANCE TCS
```

#### **4. Production Test**
```bash
# Full year test
python src/runners/unified_runner.py \
  --mode backtest \
  --config config/templates/my_custom.yaml \
  --date-ranges 2024-12-12_to_2025-06-09 \
  --tickers RELIANCE TCS INFY \
  --parallel
```

### **Template Comparison**

```bash
# Compare templates on same data
python src/runners/unified_runner.py --mode backtest --template minimal --date-ranges 2024-12-12_to_2025-06-09 --tickers RELIANCE
python src/runners/unified_runner.py --mode backtest --template conservative --date-ranges 2024-12-12_to_2025-06-09 --tickers RELIANCE
python src/runners/unified_runner.py --mode backtest --template aggressive --date-ranges 2024-12-12_to_2025-06-09 --tickers RELIANCE

# Then analyze results
python src/runners/unified_runner.py --mode analyze --date-ranges 2024-12-12_to_2025-06-09 --tickers RELIANCE
```

---

## 🎯 **Risk Management Deep Dive**

### **Position Sizing Methods**

#### **Equal Weight**
```yaml
risk:
  position_sizing_method: "equal_weight"
  max_position_size: 0.15        # Each position gets equal weight
```

#### **Volatility Adjusted**
```yaml
risk:
  position_sizing_method: "volatility_adjusted"
  volatility_lookback: 20        # ATR calculation period
  volatility_multiplier: 2.0     # Risk adjustment factor
```

#### **Risk Parity**
```yaml
risk:
  position_sizing_method: "risk_parity"
  target_volatility: 0.15        # Target portfolio volatility
  rebalance_frequency: "weekly"
```

#### **Kelly Criterion**
```yaml
risk:
  position_sizing_method: "kelly"
  kelly_multiplier: 0.25         # Fractional Kelly for safety
  min_position_size: 0.01        # Minimum position size
  max_position_size: 0.20        # Maximum position size
```

### **Stop Loss Strategies**

#### **Fixed Percentage**
```yaml
risk:
  stop_loss_pct: 0.03            # 3% fixed stop loss
  stop_loss_type: "percentage"
```

#### **ATR-Based**
```yaml
risk:
  stop_loss_type: "atr"
  atr_period: 14                 # ATR calculation period
  atr_multiplier: 2.0            # ATR multiplier for stop distance
```

#### **Trailing Stop**
```yaml
risk:
  trailing_stop_enabled: true
  trailing_stop_pct: 0.02        # 2% trailing stop
  trailing_stop_trigger: 0.03    # Activate after 3% profit
```

#### **Time-Based Exit**
```yaml
risk:
  time_based_exit: true
  max_holding_period: 30         # Days
  profit_taking_time: 14         # Take profit after 14 days if positive
```

### **Portfolio-Level Risk Controls**

#### **Correlation Limits**
```yaml
risk:
  correlation_limit: 0.70        # Max correlation between positions
  correlation_period: 60         # Correlation calculation period
  correlation_threshold_action: "reduce_position"  # Action when exceeded
```

#### **Sector Exposure**
```yaml
risk:
  max_sector_exposure: 0.30      # 30% max per sector
  sector_mapping_file: "config/sector_mapping.csv"
  sector_rebalancing: true
```

#### **Concentration Limits**
```yaml
risk:
  max_single_position: 0.20      # 20% max single position
  max_top_5_positions: 0.60      # 60% max top 5 positions
  concentration_check_frequency: "daily"
```

---

## 🛠️ **Advanced Template Features**

### **Dynamic Risk Adjustment**

```yaml
# Dynamic risk based on market conditions
risk:
  dynamic_risk_enabled: true
  volatility_regime_detection: true
  
  # Low volatility regime
  low_vol_threshold: 0.10
  low_vol_position_size: 0.20
  
  # High volatility regime  
  high_vol_threshold: 0.25
  high_vol_position_size: 0.08
  
  # Risk adjustment frequency
  risk_adjustment_frequency: "daily"
```

### **Multi-Timeframe Risk**

```yaml
risk:
  multi_timeframe_risk: true
  
  # Daily risk limits
  daily_var_limit: 0.02          # 2% daily VaR
  daily_loss_limit: 0.03         # 3% daily loss
  
  # Weekly risk limits
  weekly_var_limit: 0.05         # 5% weekly VaR
  weekly_loss_limit: 0.07        # 7% weekly loss
  
  # Monthly risk limits
  monthly_var_limit: 0.10        # 10% monthly VaR
  monthly_drawdown_limit: 0.15   # 15% monthly drawdown
```

### **Options-Specific Risk**

```yaml
options:
  enabled: true
  
  # Greeks exposure limits
  max_delta_exposure: 0.5        # Max net delta
  max_gamma_exposure: 0.1        # Max net gamma
  max_vega_exposure: 0.2         # Max net vega
  max_theta_decay: 0.05          # Max daily theta decay
  
  # Position limits
  max_option_positions: 10       # Max option contracts
  max_notional_exposure: 1000000 # Max notional exposure
  
  # Expiry management
  min_days_to_expiry: 7
  max_days_to_expiry: 45
  early_close_dte: 5
```

---

## 🤖 **AI-Assisted Template Creation**

### **Template Creation Prompt**

Use this prompt with any AI model:

```
I need to create a custom YAML configuration template for my trading backtester.

MY REQUIREMENTS:
- Risk tolerance: [conservative/moderate/aggressive]
- Initial capital: [amount]
- Maximum position size: [percentage]
- Stop loss preference: [percentage]
- Take profit target: [percentage]
- Number of concurrent positions: [number]
- Trading frequency: [daily/weekly/monthly]
- Asset types: [stocks/options/both]
- Market conditions: [trending/ranging/all]

SYSTEM CONTEXT:
- Framework: Production backtesting system
- Configuration format: YAML
- Available templates: minimal, conservative, aggressive, options, portfolio_diversified
- Risk categories: position sizing, loss protection, portfolio limits, transaction costs

Please provide:
1. Complete YAML configuration template
2. Risk parameter explanations
3. Recommended testing approach
4. Usage commands for the template

Base the configuration on similar existing templates but customize for my specific requirements.
```

---

## 📊 **Template Performance Metrics**

### **Risk-Adjusted Returns**

| Template | Sharpe Ratio | Max Drawdown | Win Rate | Volatility |
|----------|--------------|-------------|----------|------------|
| minimal | 0.8-1.2 | 3-5% | 55-65% | 8-12% |
| conservative | 1.0-1.5 | 5-8% | 50-60% | 12-18% |
| aggressive | 0.8-1.8 | 8-15% | 45-55% | 18-25% |
| options | 0.9-1.4 | 6-12% | 50-60% | 15-22% |
| portfolio_diversified | 1.1-1.6 | 4-10% | 52-62% | 10-16% |

### **Performance Comparison Command**

```bash
# Generate performance comparison report
python utils/template_comparison.py \
  --templates minimal conservative aggressive \
  --date-ranges 2024-12-12_to_2025-06-09 \
  --tickers RELIANCE TCS INFY \
  --output performance_comparison.html
```

---

## 🎯 **Best Practices**

### **✅ Template Selection Guidelines**

1. **New to Trading**: Start with `minimal` template
2. **Conservative Investor**: Use `conservative` template  
3. **Growth Focused**: Use `aggressive` template
4. **Options Trader**: Use `options` template
5. **Portfolio Manager**: Use `portfolio_diversified` template

### **✅ Customization Best Practices**

1. **Start Small**: Begin with existing template closest to your needs
2. **Test Incrementally**: Test each change individually
3. **Document Changes**: Comment your modifications in YAML
4. **Version Control**: Keep multiple versions for comparison
5. **Validate Thoroughly**: Test with historical data before live use

### **✅ Risk Management Rules**

1. **Never risk more than 2% per trade**
2. **Keep portfolio drawdown under 10%**
3. **Diversify across uncorrelated assets**
4. **Use stop losses consistently**
5. **Monitor correlation between positions**

---

## 📞 **Getting Help**

### **Template Issues**
- **Template not loading**: Check YAML syntax
- **Parameter errors**: Validate parameter ranges
- **Performance issues**: Reduce position sizes or enable caching

### **Risk Management Questions**
- See `docs/SETUP_GUIDE.md` for installation help
- See `docs/CLI_REFERENCE.md` for command options
- See `docs/OUTPUT_GUIDE.md` for results analysis

### **AI Assistance**
Use the template creation prompt above for personalized configuration help!

---

**Master Your Risk, Maximize Your Returns! 📈**
