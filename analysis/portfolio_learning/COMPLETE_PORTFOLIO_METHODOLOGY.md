# 🎯 COMPLETE PORTFOLIO CONSTRUCTION METHODOLOGY

**Date:** September 2025
**Version:** 1.0 - Final Methodology
**Status:** Ready for Implementation
**Teacher:** 15-Year Algotrading Portfolio Manager
**Student:** Portfolio Construction Learning Journey

---

## 🎓 **EXECUTIVE SUMMARY**

This document outlines the **complete systematic portfolio construction methodology** for building optimal trading portfolios from MSE strategy anti-cascading trades. The approach combines **Modern Portfolio Theory** with **practical algotrading constraints** to identify the best risk-adjusted portfolio combinations from our top 50 performing tickers under ₹2,000.

**Key Innovation:** Using **actual daily trade returns** to calculate portfolio metrics instead of theoretical assumptions, ensuring real-world applicability.

---

## 📊 **PHASE 1: DATA FOUNDATION & UNIVERSE SELECTION**

### **1.1 Ticker Universe Definition**

**Source:** MSE Strategy Analysis - Top 50 Performers (Tier 1 + Tier 2)

**Top 50 Tickers:**
```
MRF, PAGEIND, POWERINDIA, HONAUT, LICI, ULTRACEMCO, SHREECEM, BEML, MCX, SBIN,
PNB, OFSS, ABB, TATAPOWER, ABBOTINDIA, LT, RELIANCE, LICHSGFIN, ATUL, ADANIENT,
INDUSINDBK, TATAELXSI, PAYTM, CANBK, BALRAMCHIN, IEX, RECLTD, IRCTC, TCS, M&M,
BIOCON, UNIONBANK, SBICARD, 3MINDIA, JUSTDIAL, GMDCLTD, GODFRYPHLP, BAJAJ-AUTO,
DEEPAKNTR, MARUTI, APOLLOHOSP, BANKINDIA, TATACHEM, SONACOMS, AARTIIND, NIACL,
JUBLFOOD, BANDHANBNK, GSFC, ASHOKLEY
```

### **1.2 Price Filtering Methodology**

**Objective:** Focus on tickers under ₹2,000 for capital efficiency

**Method:**
```python
# Use last trade's Exit Price as current price proxy
last_trades = trades_df.groupby('ticker').last()
current_price = last_trade['Exit Price']
affordable_tickers = tickers[current_price < 2000]
```

**Expected Result:** ~28-30 tickers under ₹2,000 (approximately 60% of top 50)

**Capital Efficiency Rationale:**
- Lower price tickers allow better diversification with limited capital
- Reduced position sizing constraints
- Better portfolio rebalancing flexibility

### **1.3 Anti-Cascading Trade Dataset Creation**

**Critical Foundation:** Create clean dataset with ONLY non-cascading trades

**Anti-Cascading Logic:**
```
INCLUDE (Non-Cascading):
✅ First trades (no previous trade to cascade from)
✅ BUY → SELL (opposite direction)
✅ SELL → BUY (opposite direction)
✅ First trade of new day
✅ First trade for new ticker

EXCLUDE (Cascading):
❌ BUY → BUY (consecutive same direction)
❌ SELL → SELL (consecutive same direction)
```

**Implementation Steps:**
1. Load `all_trade_mereged.csv` (1.15M+ trades)
2. Filter for top 50 tickers only
3. Sort by ticker and entry time
4. Apply cascading detection algorithm
5. Save as `anti_cascading_trades_top50_under2k.csv`

**Expected Dataset Reduction:** 1.15M trades → ~400K-500K anti-cascading trades

---

## 🏗️ **PHASE 2: SECTOR CLASSIFICATION & CORRELATION ANALYSIS**

### **2.1 Sector Mapping Framework**

**Purpose:** Enable sector diversification rules to reduce concentration risk

**Sector Classification:**

| Sector | Tickers | Max Portfolio Weight |
|--------|---------|---------------------|
| **Banking & Financial Services** | SBIN, PNB, CANBK, LICHSGFIN, UNIONBANK, BANKINDIA, INDUSINDBK, SBICARD | 40% |
| **Automotive & Auto Components** | MARUTI, M&M, BAJAJ-AUTO, ASHOKLEY, SONACOMS | 40% |
| **Information Technology** | TCS, TATAELXSI, JUSTDIAL, PAYTM | 40% |
| **Chemicals & Petrochemicals** | ATUL, AARTIIND, DEEPAKNTR, BALRAMCHIN, TATACHEM | 40% |
| **Infrastructure & Construction** | LT, ADANIENT, TATAPOWER, RECLTD | 40% |
| **Cement & Building Materials** | ULTRACEMCO, SHREECEM | 30% |
| **Pharmaceuticals & Healthcare** | BIOCON, APOLLOHOSP, ABBOTINDIA | 40% |
| **Industrial Equipment** | ABB, BEML, HONAUT, GODFRYPHLP | 40% |
| **Consumer Goods & Services** | JUBLFOOD, 3MINDIA, PAGEIND | 40% |
| **Metals & Mining** | GMDCLTD | 20% |
| **Exchange & Financial Markets** | MCX, IEX | 30% |
| **Transportation & Logistics** | IRCTC | 20% |
| **Insurance** | LICI, NIACL | 30% |
| **Specialty & Others** | MRF, POWERINDIA, OFSS, GSFC | 40% |

**Diversification Rules:**
- Maximum 40% allocation to any single sector in a portfolio
- Minimum 2 sectors required for portfolios ≥5 tickers
- Minimum 3 sectors required for portfolios ≥8 tickers

### **2.2 Correlation Matrix Calculation**

**Method:** Calculate correlations from **actual daily trade returns**, not price correlations

**Implementation:**
```python
def calculate_trade_correlation_matrix(anti_cascading_trades):
    """
    Calculate correlation matrix from actual daily trade returns
    This captures strategy-level correlations, not just price correlations
    """

    # Calculate daily returns for each ticker
    daily_returns = {}

    for ticker in affordable_tickers:
        ticker_trades = anti_cascading_trades[anti_cascading_trades['ticker'] == ticker]
        ticker_trades['trade_date'] = ticker_trades['Entry Time'].dt.date
        ticker_trades['daily_return'] = (ticker_trades['Exit Price'] /
                                        ticker_trades['Entry Price'] - 1) * 100

        # Average returns per day (if multiple trades per day)
        daily_returns[ticker] = ticker_trades.groupby('trade_date')['daily_return'].mean()

    # Create correlation matrix
    correlation_df = pd.DataFrame(daily_returns).fillna(0)
    correlation_matrix = correlation_df.corr()

    return correlation_matrix
```

**Correlation-Based Filtering:**
- Maximum average portfolio correlation: 0.75
- Exclude portfolios with >3 ticker pairs having correlation >0.85
- Prioritize portfolios with low inter-sector correlations

---

## 🔧 **PHASE 3: INTELLIGENT COMBINATION GENERATION**

### **3.1 Computational Complexity Analysis**

**Portfolio Size vs Combinations:**

| Portfolio Size | Combinations Formula | Expected Combinations | Computational Feasibility |
|---------------|---------------------|----------------------|---------------------------|
| **5 tickers** | 30C5 | 142,506 | ✅ Full enumeration (2-3 hours) |
| **8 tickers** | 30C8 | 5,852,925 | ⚠️ Smart sampling (50K samples) |
| **10 tickers** | 30C10 | 30,045,015 | ⚠️ Strategic sampling (20K samples) |
| **12 tickers** | 30C12 | 86,493,225 | ⚠️ Guided sampling (10K samples) |

### **3.2 Pre-Filtering Strategy**

**Objective:** Reduce computational load without losing optimal combinations

**Filter 1: Minimum Trade Threshold**
```python
min_trades_per_ticker = 300  # Ensure statistical significance
valid_tickers = [ticker for ticker in affordable_tickers
                if trade_counts[ticker] >= min_trades_per_ticker]
```

**Filter 2: Sector Diversification**
```python
def check_sector_diversification(ticker_combination):
    """
    Ensure portfolio meets sector diversification requirements
    """
    sector_weights = calculate_sector_weights(ticker_combination)
    max_sector_weight = max(sector_weights.values())
    return max_sector_weight <= 0.4  # Max 40% in any sector
```

**Filter 3: Correlation Constraint**
```python
def check_correlation_constraint(ticker_combination, correlation_matrix):
    """
    Filter out highly correlated portfolios
    """
    portfolio_correlations = []
    for i in range(len(ticker_combination)):
        for j in range(i+1, len(ticker_combination)):
            corr = correlation_matrix.loc[ticker_combination[i], ticker_combination[j]]
            portfolio_correlations.append(abs(corr))

    avg_correlation = np.mean(portfolio_correlations)
    return avg_correlation <= 0.75  # Max average correlation
```

**Filter 4: Individual Performance Screening**
```python
def check_individual_performance(ticker_combination, individual_metrics):
    """
    Ensure all tickers meet minimum performance thresholds
    """
    for ticker in ticker_combination:
        if individual_metrics[ticker]['sharpe_ratio'] < 1.0:  # Minimum Sharpe
            return False
        if individual_metrics[ticker]['profit_factor'] < 1.2:  # Minimum profit factor
            return False
    return True
```

### **3.3 Smart Sampling Strategy**

**For Large Combination Spaces (8+ tickers):**

**Method 1: Stratified Sampling**
- Divide correlation space into buckets (low, medium, high correlation)
- Sample proportionally from each bucket
- Ensures coverage across correlation spectrum

**Method 2: Greedy Construction**
- Start with top individual performers
- Add tickers that improve portfolio Sharpe ratio
- Use hill-climbing optimization

**Method 3: Monte Carlo with Bias**
- Random sampling with probability weights
- Higher probability for better individual performers
- Lower probability for highly correlated pairs

---

## ⚙️ **PHASE 4: PORTFOLIO CALCULATION ENGINE**

### **4.1 Daily Portfolio Return Calculation**

**Core Algorithm:**
```python
def calculate_daily_portfolio_returns(portfolio_trades, weights):
    """
    Calculate true portfolio daily returns from actual trade data
    This is the heart of our methodology!
    """

    # Prepare trade data
    portfolio_trades['trade_date'] = portfolio_trades['Entry Time'].dt.date
    portfolio_trades['trade_return'] = ((portfolio_trades['Exit Price'] /
                                       portfolio_trades['Entry Price']) - 1) * 100

    # Get all unique trading dates
    all_dates = sorted(portfolio_trades['trade_date'].unique())
    daily_portfolio_returns = []

    for date in all_dates:
        date_trades = portfolio_trades[portfolio_trades['trade_date'] == date]

        # Calculate weighted return for each ticker on this date
        daily_return = 0.0
        total_weight_used = 0.0

        for i, ticker in enumerate(ticker_combination):
            ticker_trades = date_trades[date_trades['ticker'] == ticker]

            if len(ticker_trades) > 0:
                # Average return if multiple trades per ticker per day
                ticker_daily_return = ticker_trades['trade_return'].mean()
                daily_return += weights[i] * ticker_daily_return
                total_weight_used += weights[i]

        # Normalize if not all tickers traded on this date
        if total_weight_used > 0:
            daily_return = daily_return / total_weight_used
            daily_portfolio_returns.append(daily_return)

    return pd.Series(daily_portfolio_returns)
```

### **4.2 Portfolio Metrics Calculation**

**Risk-Adjusted Performance Metrics:**
```python
def calculate_portfolio_metrics(daily_returns_series):
    """
    Calculate comprehensive portfolio performance metrics
    """

    if len(daily_returns_series) == 0:
        return None

    # Basic statistics
    daily_mean = daily_returns_series.mean()
    daily_std = daily_returns_series.std()

    # Annualized metrics
    annual_return = daily_mean * 252
    annual_volatility = daily_std * np.sqrt(252)

    # Risk-adjusted metrics
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

    # Downside metrics
    negative_returns = daily_returns_series[daily_returns_series < 0]
    downside_deviation = np.sqrt(np.mean(negative_returns**2)) * np.sqrt(252)
    sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0

    # Drawdown analysis
    cumulative_returns = (1 + daily_returns_series/100).cumprod()
    running_max = cumulative_returns.expanding(min_periods=1).max()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Consistency metrics
    win_rate = len(daily_returns_series[daily_returns_series > 0]) / len(daily_returns_series)
    profit_factor = abs(daily_returns_series[daily_returns_series > 0].sum() /
                       daily_returns_series[daily_returns_series < 0].sum()) if len(negative_returns) > 0 else float('inf')

    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trading_days': len(daily_returns_series),
        'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
    }
```

---

## 🏆 **PHASE 5: MULTI-OBJECTIVE PORTFOLIO RANKING**

### **5.1 Composite Scoring Framework**

**Primary Ranking Methodology:**

| Metric | Weight | Rationale |
|--------|--------|-----------|
| **Sharpe Ratio** | 35% | Primary risk-adjusted return measure |
| **Maximum Drawdown** | 25% | Downside protection critical for capital preservation |
| **Sortino Ratio** | 20% | Downside-focused risk adjustment |
| **Profit Factor** | 10% | Consistency of positive performance |
| **Calmar Ratio** | 10% | Long-term risk-adjusted performance |

**Scoring Formula:**
```python
def calculate_composite_score(metrics):
    """
    Multi-objective portfolio scoring
    """
    # Normalize each metric to 0-100 scale
    sharpe_score = min(metrics['sharpe_ratio'] / 5.0 * 100, 100)  # Cap at 5.0 Sharpe
    drawdown_score = max(0, (1 + metrics['max_drawdown']) * 100)  # Higher is better (less negative)
    sortino_score = min(metrics['sortino_ratio'] / 7.0 * 100, 100)  # Cap at 7.0 Sortino
    profit_factor_score = min((metrics['profit_factor'] - 1) / 2.0 * 100, 100)  # Cap at 3.0 PF
    calmar_score = min(metrics['calmar_ratio'] / 10.0 * 100, 100)  # Cap at 10.0 Calmar

    # Weighted composite score
    composite_score = (
        0.35 * sharpe_score +
        0.25 * drawdown_score +
        0.20 * sortino_score +
        0.10 * profit_factor_score +
        0.10 * calmar_score
    )

    return composite_score
```

### **5.2 Portfolio Size Optimization**

**Objective:** Find optimal number of tickers balancing diversification vs alpha dilution

**Analysis Framework:**
- Compare best portfolios across sizes (5, 8, 10, 12 tickers)
- Evaluate marginal benefit of additional diversification
- Consider practical implementation constraints

**Expected Insights:**
- 5-ticker portfolios: High alpha, moderate risk
- 8-ticker portfolios: Balanced risk-return
- 10+ ticker portfolios: Maximum diversification, potential alpha dilution

---

## 💻 **PHASE 6: COMPUTATIONAL IMPLEMENTATION**

### **6.1 Batch Processing Architecture**

**Processing Strategy:**
```python
def process_portfolio_combinations_in_batches(combinations, batch_size=1000):
    """
    Process large combination sets efficiently
    """
    total_combinations = len(combinations)
    results = []

    for i in range(0, total_combinations, batch_size):
        batch = combinations[i:i+batch_size]

        print(f"Processing batch {i//batch_size + 1}: combinations {i+1} to {min(i+batch_size, total_combinations)}")

        batch_results = []
        for combination in batch:
            try:
                portfolio_metrics = analyze_portfolio_combination(combination)
                if portfolio_metrics:
                    batch_results.append({
                        'tickers': combination,
                        'metrics': portfolio_metrics,
                        'composite_score': calculate_composite_score(portfolio_metrics)
                    })
            except Exception as e:
                print(f"Error processing {combination}: {e}")
                continue

        results.extend(batch_results)

        # Save incremental results to prevent data loss
        pd.DataFrame(batch_results).to_csv(f'portfolio_results_batch_{i//batch_size + 1}.csv')

        print(f"Batch {i//batch_size + 1} completed. Found {len(batch_results)} valid portfolios.")

    return results
```

### **6.2 Performance Optimization**

**Memory Management:**
- Process combinations in batches to prevent memory overflow
- Clear intermediate variables after each batch
- Use pandas chunking for large datasets

**Parallel Processing:**
- Utilize multiple CPU cores for independent combination analysis
- Implement proper resource sharing for trade data
- Consider distributed computing for very large combination spaces

**Progress Tracking:**
- Real-time progress reporting
- Estimated time remaining calculations
- Intermediate result checkpointing

---

## 📊 **PHASE 7: RESULTS ANALYSIS & VALIDATION**

### **7.1 Portfolio Comparison Framework**

**Top Portfolio Identification:**
- Best portfolio per size category (5, 8, 10, 12 tickers)
- Best sector-diversified portfolios
- Best risk-adjusted performance portfolios
- Best consistency-focused portfolios

**Performance Attribution Analysis:**
```python
def analyze_portfolio_performance_attribution(top_portfolios):
    """
    Understand what drives portfolio performance
    """
    for portfolio in top_portfolios:
        # Individual ticker contributions
        ticker_contributions = calculate_individual_contributions(portfolio)

        # Sector contributions
        sector_contributions = calculate_sector_contributions(portfolio)

        # Diversification benefit
        diversification_effect = portfolio_return - weighted_individual_returns

        # Risk reduction benefit
        risk_reduction = weighted_individual_volatility - portfolio_volatility
```

### **7.2 Robustness Testing**

**Time Period Analysis:**
- Split performance into different market regimes
- Test portfolio stability across bull/bear/sideways markets
- Identify portfolios with consistent performance

**Sensitivity Analysis:**
- Test impact of different weight allocations
- Analyze performance with rebalancing frequencies
- Evaluate impact of transaction costs

### **7.3 Implementation Readiness**

**Capital Allocation Guidelines:**
- Recommended position sizes per ticker
- Rebalancing frequency recommendations
- Risk management parameters

**Monitoring Framework:**
- Key performance indicators to track
- Warning signals for portfolio degradation
- Criteria for portfolio modification

---

## 🎯 **EXPECTED OUTCOMES & DELIVERABLES**

### **Primary Deliverables:**

1. **Anti-Cascading Trade Dataset:** `anti_cascading_trades_top50_under2k.csv`
2. **Portfolio Analysis Results:** Comprehensive performance metrics for all tested combinations
3. **Top Portfolio Recommendations:** Best 3-5 portfolios per size category
4. **Sector & Correlation Analysis:** Detailed diversification insights
5. **Implementation Guidelines:** Practical deployment recommendations

### **Performance Expectations:**

**Computational Timeline:**
- Data preparation: 30-45 minutes
- 5-ticker analysis: 2-3 hours (142K combinations)
- 8-ticker analysis: 4-6 hours (50K sampled combinations)
- 10+ ticker analysis: 2-3 hours (guided sampling)
- **Total project timeline: 8-12 hours**

**Quality Metrics:**
- Statistical significance: Minimum 200 trading days per portfolio
- Risk-adjusted performance: Target Sharpe ratio >2.0
- Diversification: Maximum 40% sector concentration
- Consistency: Target win rate >50%, profit factor >1.5

---

## 🔄 **CONTINUOUS IMPROVEMENT FRAMEWORK**

### **Model Updates:**
- Monthly recalibration of correlation matrices
- Quarterly performance review and ranking updates
- Annual methodology enhancement based on results

### **Risk Management:**
- Real-time monitoring of portfolio correlations
- Dynamic position sizing based on recent volatility
- Circuit breakers for extreme market conditions

---

## 📝 **CONCLUSION**

This methodology represents a **rigorous, systematic approach** to portfolio construction that:

1. **Uses real trade data** instead of theoretical assumptions
2. **Incorporates practical constraints** (capital efficiency, sector diversification)
3. **Balances computational feasibility** with thorough analysis
4. **Provides actionable results** for actual trading implementation

The approach has been designed to be **teachable, reproducible, and scientifically sound**, ensuring that every step can be explained, verified, and improved upon.

**Next Steps:** Upon student approval, proceed with anti-cascading dataset creation and begin systematic portfolio analysis implementation.

---

*This methodology represents 15 years of algotrading experience combined with academic portfolio theory, adapted for practical implementation with real trade data.*