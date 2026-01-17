# SMA Crossover Strategy Research for Indian Equities

## Executive Summary

This document provides comprehensive research on Simple Moving Average (SMA) crossover strategies specifically for Indian equity markets (NSE/BSE). The research covers strategy mechanics, Indian market considerations, performance evidence, and actionable implementation recommendations for backtesting.

**Key Findings:**
- SMA crossover strategies show mixed results in Indian markets with no statistically significant advantage over buy-and-hold
- Golden Cross (50/200 SMA) generates infrequent signals (~33 signals in 66 years on S&P 500) with average trade duration of 350 days
- Transaction costs (0.035% per trade) and slippage significantly impact profitability in Indian equities
- Filtering false signals using RSI, volume, and ADX is critical for reducing whipsaw losses in sideways markets
- Large-cap stocks (Nifty 50) show better technical positioning than mid/small caps for SMA strategies

---

## 1. Core Strategy Mechanics

### 1.1 How SMA Crossover Works

The Simple Moving Average (SMA) calculates the average price of a stock over a specified number of days to smooth out daily fluctuations. The crossover strategy generates signals when two SMAs of different periods intersect:

**Long Entry Signal:** Short-period SMA crosses above long-period SMA (indicates potential uptrend)
**Exit/Short Signal:** Short-period SMA crosses below long-period SMA (indicates potential downtrend)

### 1.2 Golden Cross and Death Cross

**Golden Cross:**
- Most common: 50-day SMA crosses above 200-day SMA
- Signals potential long-term uptrend
- Average return: 2.12% after 3 months, 3.43% after 6 months (1896-2016 study)
- Market average gain: 9.9% in year following Golden Cross

**Death Cross:**
- 50-day SMA crosses below 200-day SMA
- Signals potential long-term downtrend
- Average decline: 7.8% in 6 months following Death Cross

### 1.3 Common Parameter Combinations

| SMA Combination | Trading Style | Typical Holding Period | Use Case |
|-----------------|---------------|------------------------|----------|
| 5/10 or 10/20 | Very Short-term | Days to 1-2 weeks | Day trading, scalping |
| 10/30 | Short-term | 1-3 weeks | Short-term momentum |
| 20/50 | Intermediate | 1-3 months | Swing trading |
| 50/100 | Intermediate-Long | 2-6 months | Position trading |
| 50/200 | Long-term | 6-12+ months | Golden Cross strategy |
| 100/200 | Long-term | 6-12+ months | Conservative trend following |

**Research-backed optimal ranges:**
- Fast SMA: 40-60 periods
- Slow SMA: 100-120 periods
- Note: These require adjustment for specific instruments and market conditions

### 1.4 Entry and Exit Signal Rules

**Basic Entry Rules:**
1. Generate buy signal when short SMA crosses above long SMA
2. Confirm with volume spike (1.5x average recommended)
3. Check that session time is before market cutoff (3:15 PM IST recommended)
4. Verify minimum liquidity requirements

**Enhanced Entry Rules (Reduce False Signals):**
1. RSI confirmation: RSI < 70 for longs, RSI > 30 for shorts
2. Volume filter: Breakout volume must exceed 1.5x 10-period average
3. ADX filter: ADX > 25 to confirm trending market (avoid sideways)
4. Multi-timeframe alignment: Confirm trend on higher timeframe

**Exit Signal Rules:**
1. SMA crossover in opposite direction
2. Intraday cutoff: Force close all positions at 3:15 PM IST
3. Stop-loss: 1.5-2.0 x ATR below entry
4. Target: 2.0-3.0 x ATR above entry (1:2 or 1:3 risk-reward ratio)

### 1.5 Position Sizing and Risk Management

**Position Sizing Formula:**
```
Position Size = (Account Risk Amount) / (Stop Loss per Share)
```

**Risk Management Rules:**
1. **Maximum Risk per Trade:** 1-2% of total trading capital
   - Example: ₹5,00,000 capital → Risk ₹5,000-₹10,000 per trade

2. **Risk-Reward Ratio:** Minimum 1:3
   - Risk ₹1 to potentially make ₹3

3. **Stop-Loss Methods:**
   - **Fixed:** Set predetermined price level
   - **Percentage-Based:** 1-3% below purchase price
   - **Moving Average-Based:** Just below key SMA (e.g., below 50-day SMA)
   - **Trailing Stop-Loss:** Adjust stop as price moves favorably (e.g., ₹10 below highest price)

4. **Position Size Limits:**
   - Maximum 10% of equity per position to prevent catastrophic losses during gaps
   - Reduce position size during high volatility periods

---

## 2. Indian Market Considerations

### 2.1 NSE/BSE Market Characteristics

**Market Structure:**
- NSE (National Stock Exchange): Primary electronic exchange
- BSE (Bombay Stock Exchange): Oldest exchange in Asia
- Nifty 50: Blue-chip index, most liquid
- BSE Sensex: 30-stock benchmark
- Market breadth currently showing divergence (Nifty 50 strong, mid/small caps lagging)

**Market Hours:**
- Pre-market: 9:00 AM - 9:15 AM IST
- Regular session: 9:15 AM - 3:30 PM IST
- Recommended entry cutoff: 3:15 PM IST (avoid end-of-day volatility)

**Liquidity Characteristics:**
- Large-cap stocks: Higher liquidity, lower slippage
- Mid-cap stocks: Moderate liquidity, moderate slippage risk
- Small-cap stocks: Lower volumes, higher slippage risk
- Nifty 50 stocks: Ideal for SMA strategies due to liquidity

### 2.2 Transaction Costs

**Brokerage and Fees:**
- Futures trading: ~0.01% brokerage
- Equity delivery: 0.01-0.50% depending on broker
- STT (Securities Transaction Tax): 0.1% on sell side
- GST and other charges: ~0.02%
- **Total estimated cost:** 0.035% per trade (one-way for futures)

**Impact on Strategy:**
- High transaction costs reduce profitability
- Frequent signals (short-period SMAs) incur more costs
- Longer-period SMAs (50/200) generate fewer but more significant signals

### 2.3 Slippage in Indian Markets

**Slippage Sources:**
- Rapid price movements on BSE and NSE
- High volatility periods
- Low liquidity in mid/small caps
- Market gaps beyond stop prices

**Typical Slippage:**
- Futures: ~0.025% (average)
- Large-cap equities: 0.02-0.05%
- Mid-cap equities: 0.05-0.15%
- Small-cap equities: 0.10-0.30%

**Backtesting Considerations:**
- **Critical:** Factor in both transaction costs and slippage
- Many strategies appear profitable but fail in live trading when costs are ignored
- Use realistic assumptions: 0.035% transaction + 0.025% slippage = 0.06% per trade

### 2.4 Settlement and Circuit Breakers

**T+2 Settlement:**
- All trades settled on T+2 day
- Funds and securities credited after 2 working days
- Impacts: Position holding requirements, capital planning

**Circuit Breakers:**

**Index-Level (Market-Wide):**
- Triggered at 10%, 15%, and 20% index movement
- Applies to both BSE Sensex and Nifty 50 (whichever breaches first)
- Trading halt across all equity and derivatives markets

**Stock-Level:**
- Individual stocks: 2%, 5%, 10%, or 20% daily limits
- Stocks with derivatives contracts: NO circuit limits
- Nifty 50 constituents: Generally no individual limits due to derivatives

**Impact on SMA Strategies:**
1. Trapped positions during circuit breaker halts
2. Price discovery issues post-halt
3. Volatile price movements when trading resumes
4. Gap risk beyond stop-loss levels
5. Algo traders must include circuit breaker contingencies

**Risk Management for Circuit Breakers:**
- Reduce position sizes during high volatility
- Use market-on-open orders cautiously
- Monitor pre-market indicators
- Maintain adequate margin buffers

### 2.5 Sector-Specific Performance

**Current Market Dynamics (2024-2025):**
- Strong sectors: Industrials, Healthcare, Communication
- Large-cap performance: Leading the rally
- Mid-cap performance: Moderate strength, not fully aligned with Nifty 50
- Small-cap performance: Significantly lagging (Nifty Smallcap 250 still 10% below ATH)

**SMA Strategy Implications:**
- Large-caps show stronger technical positioning (more stocks above key SMAs)
- Mid/small caps: Less than one-third above major moving averages
- Narrow rally: Market breadth weakness suggests selective stock picking critical

**Sector Rotation Considerations:**
- SMA strategies work best in trending sectors
- Monitor sector rotation to adjust stock universe
- Avoid sectors in consolidation/sideways movement

---

## 3. Performance Evidence

### 3.1 Academic Research on Indian Markets

**Study 1: Nifty50 Moving Average Analysis**
- Research: Utilized MA crossover on Nifty50 indices
- Comparison: MA crossover vs. buy-and-hold (pre and during COVID-19)
- Finding: EMA (50,200) and EMA (15,100) generated better CAGR than buy-and-hold for Sun Pharma
- Reference: Multiple studies on Indian technical analysis (Bahl & Goyal, 2017)

**Study 2: T-Test Analysis of Golden Cross/Death Cross**
- Method: Statistical T-Test on Indian equities
- Finding: **No statistically significant evidence** that MA crossovers offer more reliable signals than passive investment
- Conclusion: No significant evidence that crossover improves returns
- Implication: SMA crossover may not provide edge over buy-and-hold in Indian markets

**Study 3: EMA Crossover with Volume Confirmation**
- Research: Machine learning-based study on Indian and American markets
- Indicators: Moving Average, Stochastic RSI, Price-Volume analysis
- Finding: Moving averages widely used but effectiveness varies
- Emphasis: Volume confirmation improves signal reliability

**Study 4: Impact of Moving Averages on Prediction**
- Scope: Predictive power of various MA periods
- Commonly used: 50-day, 100-day, 200-day SMAs
- Finding: Mixed results on predictive power

### 3.2 General Strategy Performance (Global Context)

**S&P 500 Backtest (1960-Present):**
- Starting capital: $100,000
- End result: $7.2 million (66 years, no dividend reinvestment)
- Signals generated: 33 (extremely infrequent)
- Average trade duration: ~350 days
- CAGR: Slightly lags buy-and-hold
- **Risk-adjusted return:** Higher than buy-and-hold (invested only 2/3 of time)
- **Maximum drawdown:** Cut roughly in half vs. buy-and-hold

**Alternative Study (2000-2020):**
- Return: 118%
- Buy-and-hold return: 263%
- Conclusion: Underperformed buy-and-hold in strong bull market

**Chart Report Study (1896-2016):**
- Golden Cross average return: 2.12% after 3 months
- 6-month return: 3.43%
- Indicates modest gains, not exceptional

**MarketWatch Study:**
- Average gain following Golden Cross: 9.9% in next year
- Death Cross average decline: 7.8% in next 6 months

### 3.3 Comparison to Buy-and-Hold

**Advantages of SMA Crossover:**
- Lower maximum drawdown (approximately 50% reduction)
- Higher risk-adjusted returns (Sharpe ratio)
- Capital preservation during bear markets
- Reduced time in market (lower risk exposure)

**Disadvantages of SMA Crossover:**
- Lower absolute returns in strong bull markets
- Misses early portions of trends (lagging indicator)
- Transaction costs erode profits
- Whipsaw losses in sideways markets

**Verdict for Indian Markets:**
- Statistical evidence suggests no significant advantage over buy-and-hold
- Risk reduction benefit may justify lower returns for conservative traders
- Best suited for trending markets, not sideways/choppy conditions

### 3.4 Drawdown Characteristics

**Maximum Drawdown:**
- SMA strategies: Approximately 50% lower than buy-and-hold
- Initial risk per position: 2-3% of equity typical
- Without stop-loss: Greater drawdown risk during extreme volatility
- With proper risk management: Controlled drawdowns

**Drawdown Risk Factors:**
- Lack of volatility filtering: Exposes to high-volatility periods
- Market gaps beyond stop prices: Losses exceed planned 2-3%
- Position sizing errors: Oversized positions amplify drawdowns
- Whipsaw trades: Multiple consecutive losing trades

**Drawdown Mitigation:**
- Position sizing based on volatility (ATR-based)
- Stop-loss mechanisms (trailing stops recommended)
- Volatility filters (ADX, ATR thresholds)
- Portfolio diversification across low-correlation stocks

**Expected Drawdown Metrics (Typical):**
- Average drawdown: 5-10% (with risk management)
- Maximum drawdown: 15-25% (depending on parameters)
- Recovery time: Varies by market conditions

---

## 4. Implementation Recommendations

### 4.1 Suggested Parameter Ranges for Indian Equities

**Recommended for Backtesting:**

| Parameter Combination | Description | Expected Signals | Risk Level |
|-----------------------|-------------|------------------|------------|
| 10/30 SMA | Short-term momentum | High frequency | High |
| 20/50 SMA | Swing trading | Moderate frequency | Moderate |
| 50/100 SMA | Position trading | Low-moderate frequency | Moderate-Low |
| 50/200 SMA | Golden Cross (long-term) | Very low frequency | Low |
| 40/100 SMA | Optimized intermediate | Moderate frequency | Moderate |
| 60/120 SMA | Optimized long-term | Low frequency | Low |

**Optimization Strategy:**
1. Start with standard combinations (50/200, 20/50)
2. Grid search across ranges:
   - Fast SMA: 10, 20, 30, 40, 50, 60
   - Slow SMA: 50, 100, 120, 150, 200
3. Evaluate on both trending and sideways periods
4. Walk-forward analysis to prevent overfitting
5. Validate on out-of-sample data

**Universe Selection:**
- **Priority 1:** Nifty 50 stocks (highest liquidity, lowest slippage)
- **Priority 2:** Nifty 100 stocks (good liquidity)
- **Priority 3:** Nifty 200 / BSE 500 large-caps
- **Avoid:** Small-caps and illiquid stocks (high slippage negates strategy edge)

### 4.2 Additional Filters to Reduce False Signals

**1. RSI (Relative Strength Index) Filter**

**Purpose:** Avoid overbought/oversold extremes
**Implementation:**
- Long entries: RSI < 70 (not overbought), ideally RSI between 40-60
- Short entries: RSI > 30 (not oversold), ideally RSI between 40-60
- Parameters: 14-period RSI (standard)
- Logic: When fast SMA crosses above slow SMA AND RSI confirms → Buy signal

**Code Logic:**
```python
# Long entry
long_signal = (fast_sma > slow_sma) & (rsi < 70) & (rsi > 40)

# Short entry
short_signal = (fast_sma < slow_sma) & (rsi > 30) & (rsi < 60)
```

**2. Volume Filter**

**Purpose:** Confirm breakout strength
**Implementation:**
- Require volume > 1.5x average volume on crossover day
- Calculate: 10-period average volume as baseline
- Rationale: High volume breakouts more reliable than low-volume crosses

**Code Logic:**
```python
volume_avg = volume.rolling(10).mean()
volume_spike = volume > (1.5 * volume_avg)
entry_signal = sma_crossover & volume_spike
```

**3. ADX (Average Directional Index) Filter**

**Purpose:** Avoid trading in sideways/choppy markets
**Implementation:**
- Only take signals when ADX > 20-25 (trending market)
- ADX < 20: Sideways market, skip signals
- Parameters: 14-period ADX (standard)
- Strategy: Use 65sma_3cc approach with ADX filter

**Code Logic:**
```python
adx = calculate_adx(high, low, close, period=14)
trending_market = adx > 25
entry_signal = sma_crossover & trending_market
```

**4. RAVI (Range Action Verification Index) Filter**

**Purpose:** Distinguish trending vs. sideways markets
**Implementation:**
- RAVI < 3%: Sideways market (avoid trades)
- RAVI > 3%: Trending market (take trades)
- Calculation: RAVI = (Fast SMA - Slow SMA) / Slow SMA * 100

**5. Envelope and Band Filters**

**Purpose:** Reduce noise and whipsaw
**Implementation:**
- SMA Envelopes: Set ±1% bands around 200-day SMA
- Entry requirement: Price must move > 1% beyond SMA to trigger signal
- Benefit: Filters minor fluctuations

**Code Logic:**
```python
upper_envelope = sma_200 * 1.01
lower_envelope = sma_200 * 0.99
bullish_breakout = (close > upper_envelope) & (fast_sma > slow_sma)
```

**6. Hysteresis (Consecutive Confirmation)**

**Purpose:** Require sustained breakout
**Implementation:**
- Require 2 consecutive bars confirming crossover
- Reduces false signals from single-bar spikes
- Trade-off: Slightly later entries but higher reliability

**Code Logic:**
```python
crossover_today = fast_sma > slow_sma
crossover_yesterday = fast_sma.shift(1) > slow_sma.shift(1)
confirmed_signal = crossover_today & crossover_yesterday
```

**7. Multi-Timeframe Confirmation**

**Purpose:** Align with higher timeframe trend
**Implementation:**
- Entry timeframe: 5-minute or daily
- Confirmation timeframe: 15-minute or weekly
- Rule: Only take trades when both timeframes bullish/bearish

**Example:**
```python
# Daily chart crossover (entry timeframe)
daily_bullish = fast_sma_daily > slow_sma_daily

# Weekly chart confirmation
weekly_bullish = fast_sma_weekly > slow_sma_weekly

# Trade only when aligned
entry_signal = daily_bullish & weekly_bullish
```

**Recommended Filter Combination:**
```
Entry Signal = SMA Crossover
               AND Volume > 1.5x Average
               AND ADX > 25
               AND RSI in range [40, 70] for longs / [30, 60] for shorts
               AND Multi-timeframe alignment (optional but recommended)
```

### 4.3 Risk Management Rules

**Stop-Loss Rules:**

1. **ATR-Based Stops (Recommended for Indian Markets):**
   - Stop-loss: Entry price - (1.5 to 2.0 x ATR)
   - Take-profit: Entry price + (2.0 to 3.0 x ATR)
   - ATR period: 14 days (standard)
   - Benefit: Adapts to volatility

2. **Moving Average-Based Stops:**
   - Place stop just below key SMA (e.g., 50-day SMA)
   - Rationale: Breaking below SMA invalidates bullish thesis
   - Risk: Stops may be far from entry in trending markets

3. **Trailing Stops:**
   - Initial stop: 1.5-2.0 x ATR
   - Trail stop as price moves favorably
   - Example: Lock in profits by moving stop to breakeven + ₹10 per move

4. **Time-Based Stops:**
   - Intraday strategies: Close all positions by 3:15 PM IST
   - Multi-day strategies: Review positions weekly, close stale trades

**Position Sizing Rules:**

1. **Fixed Percentage Risk:**
   - Risk 1-2% of capital per trade
   - Calculate position size based on stop distance
   - Formula: `Position Size = (Account × Risk %) / (Entry - Stop)`

2. **Volatility-Based Sizing:**
   - Use ATR to adjust position size
   - High volatility → Smaller positions
   - Low volatility → Larger positions (within limits)

3. **Maximum Position Limits:**
   - No single position > 10% of equity
   - Prevents catastrophic loss from gaps

4. **Portfolio Heat:**
   - Total risk across all open positions < 6-8% of capital
   - Example: 4 positions @ 2% risk each = 8% total heat

**Risk-Reward Requirements:**

- Minimum 1:2 risk-reward ratio (risk ₹1 to make ₹2)
- Target 1:3 for optimal profitability
- Skip trades with poor risk-reward profiles

**Drawdown Management:**

- Daily loss limit: Stop trading if -3% of capital in one day
- Weekly loss limit: Reduce position sizes if -5% of capital in one week
- Monthly drawdown review: If -10%, reassess strategy parameters

### 4.4 Suitable Stock Universes

**Tier 1: Nifty 50 (Highest Priority)**

**Characteristics:**
- 50 large-cap blue-chip stocks
- Highest liquidity on NSE
- Lowest slippage
- Most stocks above key SMAs (currently leading rally)

**Why Ideal for SMA Strategy:**
- Smooth price action (less noise)
- Liquid enough for quick entries/exits
- Derivatives available (hedging options)
- Most research and analysis available

**Recommended Stocks (Examples):**
- Reliance Industries, TCS, Infosys, HDFC Bank, ICICI Bank
- IT sector: Strong trend characteristics
- Banking: High liquidity

**Tier 2: Nifty 100 (Good Alternative)**

**Characteristics:**
- Includes Nifty 50 + next 50 large caps
- Good liquidity
- Moderate slippage

**Why Suitable:**
- Broader opportunity set
- Still maintains liquidity requirements
- Diversification benefits

**Tier 3: Nifty 200 / BSE 500 Large-Caps (Selective)**

**Characteristics:**
- Includes mid-cap exposure
- Variable liquidity
- Higher slippage risk

**Caution:**
- Screen for minimum average daily volume (e.g., > ₹10 crore turnover)
- Avoid stocks with wide bid-ask spreads
- Test slippage assumptions carefully

**NOT Recommended: Mid-cap and Small-cap (Unless High Conviction)**

**Current Market Context:**
- Mid-caps: Moderate strength but not aligned with Nifty 50
- Small-caps: Significantly lagging (< 1/3 above major SMAs)
- Nifty Smallcap 250: Still 10% below all-time high

**Why Avoid:**
- Lower liquidity → Higher slippage (0.10-0.30%)
- Wider bid-ask spreads
- Circuit breaker risk (2-5% limits on many stocks)
- SMA strategies less effective in choppy, illiquid markets

**Exception:** Consider if:
- Stock has derivatives (more liquidity)
- Part of strong trending sector
- Average daily turnover > ₹5 crore
- Testing shows acceptable slippage

**Sector Selection Strategy:**

1. **Trending Sectors (Priority):**
   - Current strong sectors: Industrials, Healthcare, Communication
   - Monitor sector rotation
   - Allocate more capital to trending sectors

2. **Avoid Consolidating Sectors:**
   - Sectors in sideways range
   - High whipsaw risk

3. **Diversification:**
   - Select stocks across 3-5 sectors
   - Reduces correlation risk
   - Smoother equity curve

**Dynamic Universe Approach:**

1. **Monthly Review:**
   - Evaluate which Nifty 50 stocks are in strong trends
   - Remove stocks below 200-day SMA
   - Add stocks breaking above 200-day SMA

2. **Market Breadth Monitoring:**
   - Track % of Nifty 50 above 50-day and 200-day SMA
   - High breadth (> 70%): Favorable for SMA strategies
   - Low breadth (< 40%): Reduce trading, increase selectivity

3. **Liquidity Screening:**
   - Minimum average daily turnover: ₹10 crore
   - Minimum average daily volume: 500,000 shares
   - Maximum bid-ask spread: 0.1%

**Recommended Initial Universe for Backtesting:**

Start with Nifty 50 stocks filtered by:
1. Average daily turnover > ₹10 crore (last 30 days)
2. Currently above 200-day SMA (in uptrend)
3. Sector diversification (max 3-4 stocks per sector)
4. Has active derivatives (F&O segment)

This gives a clean, liquid universe of 20-30 stocks ideal for SMA crossover backtesting.

---

## 5. Backtesting Framework Integration

### 5.1 Strategy Configuration Template

Based on the existing framework structure (config/templates/), here's how to configure an SMA crossover strategy:

**File:** `config/templates/strategy_sma_crossover.yaml`

```yaml
strategy:
  name: sma_crossover
  description: "Simple Moving Average Crossover Strategy with Volume and RSI Filters"
  risk_profile: moderate
  timeframes:
    entry: [1d]  # Daily timeframe for swing/position trading
    confirmation: [1w]  # Weekly for trend confirmation (optional)

  parameters:
    # SMA parameters
    fast_sma_period: 50
    slow_sma_period: 200

    # Filter parameters
    rsi_period: 14
    rsi_long_min: 40
    rsi_long_max: 70
    rsi_short_min: 30
    rsi_short_max: 60

    # Volume filter
    volume_multiplier: 1.5
    volume_lookback: 10

    # ADX trend filter
    adx_period: 14
    adx_threshold: 25

    # ATR-based exits
    atr_period: 14
    stop_loss_atr_multiplier: 1.5
    take_profit_atr_multiplier: 2.5

    # Trading controls
    enable_short: false  # Long-only for Indian equities
    min_volume: 500000  # Minimum daily volume (shares)
    min_turnover: 100000000  # Minimum ₹10 crore daily turnover

    # Warmup periods
    warmup_bars: 220  # ~200 for SMA + buffer

    # Indicator column mappings
    indicator_columns:
      fast_sma: 1d_sma_50
      slow_sma: 1d_sma_200
      rsi: 1d_rsi_14
      adx: 1d_adx_14
      atr: 1d_atr_14
      volume_avg: 1d_volume_avg_10

  indicators:
    entry:
      # Fast SMA
      - name: 1d_sma_50
        type: sma
        timeframe: 1d
        params:
          period: 50

      # Slow SMA
      - name: 1d_sma_200
        type: sma
        timeframe: 1d
        params:
          period: 200

      # RSI Filter
      - name: 1d_rsi_14
        type: rsi
        timeframe: 1d
        params:
          period: 14

      # ADX Trend Filter
      - name: 1d_adx_14
        type: adx
        timeframe: 1d
        params:
          period: 14

      # ATR for stops
      - name: 1d_atr_14
        type: atr
        timeframe: 1d
        params:
          period: 14

      # Volume average
      - name: 1d_volume_avg_10
        type: sma
        timeframe: 1d
        params:
          period: 10
          column: volume

  exit:
    template_path: config/templates/exits/exit_stop_target.yaml

risk:
  enabled: true
  max_risk_per_trade: 0.02  # 2% of capital per trade
  max_portfolio_heat: 0.08  # 8% total risk
  max_position_size: 0.10   # 10% of capital per position

transaction:
  enabled: true
  commission: 0.0001  # 0.01% brokerage
  slippage: 0.00025   # 0.025% slippage assumption

validation:
  enabled: true
  lookahead_bias_check: true

execution:
  parallel_processing: false
  max_workers: 1

output:
  save_trades: true
  save_signals: true
  save_metrics: true
  save_visualizations: true
```

### 5.2 Entry Logic Pseudocode

```python
def generate_signals(data):
    """
    Generate SMA crossover signals with filters.
    """
    # Calculate indicators (if not already from registry)
    fast_sma = calculate_sma(data['close'], period=50)
    slow_sma = calculate_sma(data['close'], period=200)
    rsi = calculate_rsi(data['close'], period=14)
    adx = calculate_adx(data['high'], data['low'], data['close'], period=14)
    volume_avg = calculate_sma(data['volume'], period=10)
    atr = calculate_atr(data['high'], data['low'], data['close'], period=14)

    # Entry conditions
    bullish_cross = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
    rsi_ok = (rsi >= 40) & (rsi <= 70)
    trending = adx > 25
    volume_ok = data['volume'] > (1.5 * volume_avg)

    # Long entry signal
    data['entry_signal_buy'] = bullish_cross & rsi_ok & trending & volume_ok

    # Bearish cross (exit or short)
    bearish_cross = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))
    data['exit_signal_buy'] = bearish_cross

    # Calculate stop and target levels
    data['stop_loss'] = data['close'] - (1.5 * atr)
    data['take_profit'] = data['close'] + (2.5 * atr)

    return data
```

### 5.3 Key Backtesting Considerations

**1. Warmup Period:**
- Minimum: 200 bars for 200-day SMA
- Recommended: 220 bars (buffer for other indicators)
- Ensure no signals generated during warmup

**2. Transaction Costs:**
- Commission: 0.01% (realistic for discount brokers)
- Slippage: 0.025% (Nifty 50 stocks)
- STT and taxes: Include if modeling delivery trades
- **Critical:** Don't skip this or backtest will be overly optimistic

**3. Lookahead Bias Prevention:**
- Signal on bar N → Execute on bar N+1 open
- Don't use future data in indicator calculations
- Framework's validation module should catch this

**4. Market Regime Testing:**
- Test on trending periods (2020-2021 bull run)
- Test on sideways periods (2018, 2022 consolidation)
- Test on bear markets (2020 COVID crash, 2022 correction)
- Evaluate performance across all regimes

**5. Walk-Forward Analysis:**
- Train on 70% of data
- Test on 30% out-of-sample
- Roll forward every 6-12 months
- Ensures parameters not overfitted

**6. Performance Metrics to Track:**
- Total return vs. buy-and-hold
- Sharpe ratio (risk-adjusted return)
- Maximum drawdown
- Win rate
- Average win vs. average loss
- Profit factor (gross profit / gross loss)
- Number of trades
- Average trade duration

### 5.4 Expected Performance Benchmarks

Based on research, realistic expectations for Indian equities:

| Metric | Conservative | Moderate | Aggressive |
|--------|--------------|----------|------------|
| Annual Return | 8-12% | 12-18% | 18-25% |
| Sharpe Ratio | 0.5-0.8 | 0.8-1.2 | 1.0-1.5 |
| Max Drawdown | 10-15% | 15-20% | 20-30% |
| Win Rate | 40-45% | 45-50% | 50-55% |
| Profit Factor | 1.2-1.5 | 1.5-2.0 | 2.0-3.0 |
| Trades per Year | 5-10 | 10-20 | 20-40 |

**Notes:**
- Conservative: 50/200 SMA, strict filters, Nifty 50 only
- Moderate: 20/50 or 50/100 SMA, RSI + volume filters
- Aggressive: 10/30 or 20/50 SMA, fewer filters, broader universe

**Reality Check:**
- If backtest shows > 30% annual return, likely overfitted or unrealistic assumptions
- If win rate > 60%, question signal filtering or lookahead bias
- Compare to Nifty 50 buy-and-hold return (~12-15% long-term CAGR)

---

## 6. Limitations and Risks

### 6.1 Strategy Limitations

**1. Lagging Indicator:**
- SMA calculates past prices, reacts after trend starts
- Misses early portions of moves
- Late exits give back profits

**2. Poor Performance in Sideways Markets:**
- Generates false signals in range-bound conditions
- "Whipsaw" losses: Enter long, immediately reverse
- Indian markets spend ~40-50% of time in consolidation

**3. Infrequent Signals (Long-Period SMAs):**
- 50/200 Golden Cross: Very rare signals (decades between some crosses)
- Low trade frequency reduces learning opportunities
- Capital sits idle for long periods

**4. Transaction Cost Sensitivity:**
- Short-period SMAs generate frequent signals
- Each round-trip trade costs 0.06-0.12% (entry + exit)
- High-frequency variants quickly eroded by costs

**5. No Statistical Edge in Indian Markets:**
- T-Test studies show no significant advantage over buy-and-hold
- Works better for risk reduction than absolute returns
- Requires discipline to accept underperformance in bull markets

### 6.2 Risk Factors

**1. Gap Risk:**
- Indian markets prone to overnight gaps (global news, earnings)
- Stops may not execute at planned prices
- Circuit breakers compound gap risk

**2. Market Regime Sensitivity:**
- Excellent in trends (2020-2021)
- Terrible in chop (2018, parts of 2022)
- No advance warning of regime changes

**3. Overfitting Risk:**
- Easy to optimize parameters to historical data
- Optimized parameters often fail in live trading
- Walk-forward testing essential

**4. Psychological Challenges:**
- Watching buy-and-hold outperform during bull runs
- Staying disciplined during whipsaw periods
- Exiting winners "too early"

**5. Liquidity Risk:**
- Mid/small caps: Slippage can exceed profit per trade
- Market stress: Even Nifty 50 stocks can gap
- Limit orders may not fill, market orders costly

### 6.3 When NOT to Use SMA Crossover

**Avoid in These Conditions:**

1. **Sideways/Choppy Markets:**
   - ADX < 20 consistently
   - Price oscillating around SMAs
   - High whipsaw probability

2. **High Volatility Periods:**
   - VIX > 30-35 (extreme fear)
   - Post-crash recovery (violent swings)
   - Earnings season for individual stocks

3. **Low Liquidity Stocks:**
   - Average daily turnover < ₹5 crore
   - Wide bid-ask spreads
   - Small-cap stocks in weak markets

4. **Overbought/Oversold Extremes:**
   - RSI > 80 or < 20 (exhaustion zones)
   - Parabolic moves (unsustainable trends)

5. **Major Event Risk:**
   - Budget day, RBI policy announcements
   - Elections (volatility spikes)
   - Global crisis events (COVID, financial crisis)

**Alternative Strategies for These Conditions:**
- Mean reversion strategies (sideways markets)
- Momentum strategies (trending markets with confirmation)
- Volatility strategies (high VIX environments)
- Cash / buy-and-hold (uncertain conditions)

---

## 7. Sources and References

### Academic Research

- [An Empirical Study on Investment and Trading Decision](https://americanscholarspress.us/journals/IMR/pdf/IMR-2-2022/IMR2022FAllspecil-art2.pdf)
- [THE MOVING AVERAGE CROSSOVER STRATEGY: A STUDY](https://www.inspirajournals.com/uploads/Issues/1938789358.pdf)
- [Simple Moving Average (SMA) Crossover Strategy with Buy Sell Indicator](https://ejournal.lincolnrpl.org/index.php/ajmt/article/download/103/138/412)
- [Exponential Moving Average Crossover with Traded Volume Confirmation](https://www.researchgate.net/publication/394814669_Exponential_Moving_Average_Crossover_with_Traded_Volume_Confirmation_Based_Automated_Algorithmic_Stock_Market_Trading_strategies_for_Equities_Commodities_and_Indices_in_Indian_and_American_Stock_Marke)
- [A Study of the Impact of Moving Averages on Predicting](https://jier.org/index.php/journal/article/download/3254/2626/5883)

### Indian Market Technical Analysis

- [Moving Average Crossover Strategy | TrueData](https://www.truedata.in/blog/moving-average-crossover-strategy/)
- [Simple Moving Average Trading Strategy | Kotak Securities](https://www.kotaksecurities.com/investing-guide/share-market/what-is-simple-moving-trading-strategy/)
- [Moving averages in stock market | Zerodha Varsity](https://zerodha.com/varsity/chapter/moving-averages/)
- [How to Use Golden Crossover Strategy | IIFL](https://www.indiainfoline.com/knowledge-center/share-market/how-to-use-golden-crossover-strategy)
- [SMA Screeners For Indian Market](https://www.topstockresearch.com/rt/Screener/Technical/SMAScreener)

### Strategy Performance and Backtesting

- [Golden Cross Trading Strategy | Capital.com](https://capital.com/en-int/learn/trading-strategies/golden-cross-trading-strategy)
- [Golden Cross Trading Strategy Backtest | QuantifiedStrategies](https://www.quantifiedstrategies.com/golden-cross-trading-strategy/)
- [Death Cross Trading Strategy | QuantifiedStrategies](https://www.quantifiedstrategies.com/death-cross-in-trading/)
- [Moving Average Crossover Strategies | QuantInsti](https://blog.quantinsti.com/moving-average-trading-strategies/)
- [Moving Average Crossover Strategies | TrendSpider](https://trendspider.com/learning-center/moving-average-crossover-strategies/)

### Transaction Costs and Slippage

- [Slippage in Trading | Kotak Securities](https://www.kotaksecurities.com/investing-guide/articles/what-is-slippage-in-trading/)
- [Essential Data for Backtesting | Marketfeed](https://www.marketfeed.com/read/en/essential-data-for-backtesting-in-algo-trading-a-simple-guide-2)
- [What is Slippage in Trading | Lares Algotech](https://laresalgotech.com/what-is-slippage-in-trading-meaning-examples/)

### Risk Management

- [Risk Management in Trading | MasterTrust](https://www.mastertrust.co.in/blog/risk-management-in-trading-stop-losses-and-position-sizing-explained)
- [Stop Loss Secrets | Argyles Restaurant](https://argylesrestaurant.com/en/lifestyle-en/stop-loss-secrets-strategies-used-by-indias-most-successful-traders/)
- [How to Calculate Stop Loss | Bajaj Finserv](https://www.bajajfinserv.in/how-to-calculate-stop-loss-in-intraday-trading)
- [Position Sizing Calculator | Sharekhan](https://www.sharekhan.com/financial-blog/blogs/position-sizing-calculator-for-stocks)
- [Risk Management in Intraday Trading | Bajaj Finserv](https://www.bajajfinserv.in/risk-management-in-intraday-trading)

### Filtering False Signals

- [Moving Average and RSI Crossover Strategy | Medium](https://medium.com/@redsword_23261/moving-average-and-rsi-crossover-strategy-6aa3a83b1f16)
- [44 SMA and 9 EMA Crossover Strategy with RSI Filter | Medium](https://medium.com/@redsword_23261/44-sma-and-9-ema-crossover-strategy-with-rsi-filter-and-tp-sl-12f792ea8370)
- [Reducing Moving Average Whipsaws | StockCharts](https://stockcharts.com/articles/arthurhill/2018/10/systemtrader---reducing-moving-average-whipsaws-with-smoothing-and-quantifying-filters-.html)
- [An Indicator to Reduce Whipsaws | StockCharts](https://articles.stockcharts.com/article/articles-arthurhill-2024-09-an-indicator-to-reduce-whipsaw-979/)

### Indian Market Structure

- [Circuit Breakers | NSE India](https://www.nseindia.com/products-services/equity-market-circuit-breakers)
- [Settlement Cycle | NSE India](https://www.nseindia.com/products-services/equity-market-settlement-cycle)
- [Circuit Breakers in Indian Stock Market | ISFM](https://isfm.co.in/circuit-breakers-in-the-indian-stock-market/)
- [Stock Market Circuit Breakers | Groww](https://groww.in/p/stock-market-circuit-breakers)
- [India T+1 Settlement | Deutsche Bank](https://flow.db.com/securities-services/india-trumpets-t1-settlement)

### Market Performance and Analysis

- [Nifty 50 Hits Record High | Samco](https://www.samco.in/knowledge-center/articles/nifty-50-hits-all-time-high-but-market-breadth-shows-a-different-reality/)
- [Golden crossover Screener](https://www.screener.in/screens/333667/golden-crossover/)
- [Golden Cross Stocks in Nifty 100 | Trendlyne](https://trendlyne.com/stock-screeners/simple-moving-average/sma-crossovers/sma-50/above/sma-200/today/index/NIFTY100/nifty-100/)

---

## 8. Conclusion and Next Steps

### Key Takeaways

1. **SMA crossover strategies are simple but not superior:** Research shows no statistically significant advantage over buy-and-hold in Indian markets. The primary benefit is risk reduction (lower drawdowns) rather than higher returns.

2. **Filtering is essential:** Raw SMA crossovers produce too many false signals. Combining RSI, volume, ADX, and multi-timeframe filters significantly improves reliability.

3. **Transaction costs matter:** At 0.06-0.12% per round-trip, costs quickly erode profits from frequent signals. Longer-period SMAs (50/200) are more cost-effective than short-period variants.

4. **Nifty 50 is ideal universe:** Large-cap stocks offer the liquidity and smooth price action needed for SMA strategies. Mid and small caps introduce unacceptable slippage and whipsaw risk.

5. **Risk management is non-negotiable:** Position sizing (1-2% risk per trade), ATR-based stops, and portfolio heat limits are critical for long-term survival.

6. **Market regime awareness:** SMA strategies excel in trends but fail in sideways markets. ADX filtering helps avoid choppy periods but cannot eliminate all whipsaws.

### Recommended Implementation Path

**Phase 1: Basic Backtest (Week 1-2)**
1. Implement 50/200 SMA crossover on Nifty 50 stocks
2. Use realistic transaction costs (0.01% commission, 0.025% slippage)
3. Apply 200-bar warmup period
4. Test on 5+ years of data (2018-2023) covering multiple market regimes
5. Benchmark against Nifty 50 buy-and-hold

**Phase 2: Add Filters (Week 3-4)**
1. Integrate RSI filter (14-period, range 40-70 for longs)
2. Add volume filter (1.5x 10-period average)
3. Implement ADX trend filter (> 25 threshold)
4. Compare filtered vs. unfiltered performance

**Phase 3: Risk Management (Week 5)**
1. Implement ATR-based stops (1.5x ATR)
2. Add position sizing (2% risk per trade)
3. Enforce portfolio heat limits (8% total)
4. Test drawdown characteristics

**Phase 4: Parameter Optimization (Week 6-7)**
1. Grid search across SMA combinations:
   - Fast: [10, 20, 30, 40, 50, 60]
   - Slow: [50, 100, 120, 150, 200]
2. Walk-forward analysis (70/30 train/test split)
3. Validate on out-of-sample data (2024)
4. Select robust parameters (avoid overfitting)

**Phase 5: Live Simulation (Week 8+)**
1. Paper trade selected configuration for 3-6 months
2. Monitor slippage assumptions vs. reality
3. Track psychological challenges (discipline)
4. Gradually transition to live capital if successful

### Final Recommendations

**Use SMA Crossover If:**
- You prioritize risk reduction over maximum returns
- You have discipline to follow systematic rules
- You're trading liquid large-cap stocks (Nifty 50)
- You can tolerate underperformance in strong bull markets
- You have realistic expectations (10-15% annual returns)

**Consider Alternatives If:**
- You want to beat buy-and-hold in all market conditions (no strategy does this)
- You're impatient and want frequent trades (SMA is slow)
- You're trading illiquid mid/small caps (high slippage)
- You cannot accept 15-20% drawdowns
- You expect 30%+ annual returns (unrealistic for SMA)

### Questions for Further Research

1. **Adaptive SMAs:** Can dynamic SMA periods that adjust to volatility (ATR-based) improve performance?

2. **Sector Rotation:** Does applying SMA crossover to sector indices (Nifty IT, Bank Nifty) outperform individual stocks?

3. **Multi-Asset:** Can SMA crossover on Nifty 50 index futures provide better risk-adjusted returns than stock-picking?

4. **Machine Learning Enhancement:** Can ML models predict when SMA signals will succeed vs. fail (regime detection)?

5. **Options Integration:** Can selling options around SMA signals (covered calls on longs) improve returns?

### Contact and Feedback

This research document is intended for internal use within the backtesting framework. For questions, improvements, or additional research requests, please update this document and maintain version control.

**Document Version:** 1.0
**Last Updated:** 2026-01-04
**Research Conducted By:** Claude Code Agent
**Framework Version:** Backtester v1.x

---

**END OF RESEARCH DOCUMENT**
