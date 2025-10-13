# ANALYSIS METHODOLOGY & LOGIC
## First Principles Approach to Portfolio Construction

**Last Updated**: October 8, 2025
**Philosophy**: Every analysis must have a clear "why" backed by statistical reasoning and market assumptions

---

## 📚 **TABLE OF CONTENTS**

1. [Overview & Core Philosophy](#overview--core-philosophy)
2. [Generic Analysis Suite](#generic-analysis-suite)
   - [01. Basic EDA](#01-basic-eda---foundational-statistics)
   - [02. Trade Type Analysis](#02-trade-type-analysis---directional-bias-detection)
   - [03. Cascade Analysis](#03-cascade-analysis---behavioral-pattern-detection)
   - [04. Stop Loss Simulation](#04-stop-loss-simulation---risk-management-optimization)
   - [05. Ticker Ranking](#05-ticker-ranking---quality-scoring-system)
   - [06. Risk-Adjusted Patterns](#06-risk-adjusted-patterns---risk-normalized-performance)
   - [07. Top50 vs Overall Comparison](#07-top50-vs-overall-comparison---selection-validation)
   - [08. Top50 Pattern Breakdown](#08-top50-pattern-breakdown---winner-profiling)
   - [09. Validation Check](#09-validation-check---data-integrity-audit)
3. [Portfolio Construction Suite](#portfolio-construction-suite)
   - [00. Foundation Analysis](#00-foundation-analysis---comprehensive-ticker-ranking)
   - [01. Anti-Cascade Filter](#01-anti-cascade-filter---behavioral-bias-removal)
   - [02. Sector Classification](#02-sector-classification---diversification-framework)
   - [03. Combination Generator](#03-combination-generator---constrained-optimization-space)
   - [04. Portfolio Optimization Engine](#04-portfolio-optimization-engine---equal-weight-evaluation)
   - [05. PyPortfolioOpt Weights](#05-pypfopt-optimal-weights---markowitz-optimization)
   - [06. Equity Curve Generator](#06-equity-curve-generator---visual-validation)
4. [Key Assumptions & Limitations](#key-assumptions--limitations)
5. [Statistical Foundations](#statistical-foundations)

---

## 📖 **OVERVIEW & CORE PHILOSOPHY**

### **Fundamental Question**
*"Given a strategy's historical trades, how do we systematically construct a portfolio that maximizes risk-adjusted returns while minimizing behavioral biases and concentration risks?"*

### **Analysis Hierarchy**

```
LEVEL 1: Raw Trade Data (Backtest Output)
    ↓
LEVEL 2: Generic Analysis (Understand what happened)
    → Statistics, patterns, biases, quality metrics
    ↓
LEVEL 3: Portfolio Construction (Build optimal combinations)
    → Filter, diversify, optimize, validate
    ↓
LEVEL 4: Production Deployment (Execute in live markets)
    → Risk management, monitoring, reconciliation
```

### **Core Principles**

1. **First-Principles Decomposition**: Every metric must answer "so what?"
2. **Assume Nothing, Verify Everything**: Test assumptions with data
3. **Bias Detection Before Optimization**: Remove behavioral flaws first
4. **Risk-Adjusted Always**: Raw returns mean nothing without risk context
5. **Diversification ≠ Diworsification**: Correlation-aware combination generation
6. **Reproducibility**: Same data + same config = same results (deterministic)

---

# 🔬 **GENERIC ANALYSIS SUITE**

**Purpose**: Strategy-agnostic analysis to understand trade quality, identify biases, and establish baseline performance.

**Philosophy**: Before optimizing portfolios, understand the raw material (trades) at a fundamental level.

---

## **01. Basic EDA - Foundational Statistics**

### **Why This Analysis?**
**Core Question**: *"What is the fundamental quality of trades this strategy generates?"*

Before portfolio construction, we must establish:
- Does the strategy have positive expectancy? (Win Rate × Avg Win > Loss Rate × Avg Loss)
- Is performance consistent across time and tickers?
- Are there statistical anomalies requiring investigation?

### **Backend Logic**

**1. Profit Factor Calculation**
```
Profit Factor = Σ(Winning Trade P&L) / |Σ(Losing Trade P&L)|

Logic:
- PF > 1.0: Profitable system (gains exceed losses)
- PF = 1.5: For every ₹1 lost, system makes ₹1.50
- PF < 1.0: Unprofitable system (structural flaw)
```

**Mathematical Foundation**:
- Law of Large Numbers: Over sufficient trades, PF converges to true expectancy
- If PF < 1.1 after 1000+ trades → strategy has low edge, portfolio construction cannot fix this

**2. Win Rate Analysis**
```
Win Rate = (Winning Trades / Total Trades) × 100

Context:
- WR > 50%: More winners than losers (mean-reversion style)
- WR < 50%: Trend-following style (few big wins, many small losses acceptable)
- WR alone is meaningless without Profit Factor
```

**Critical Assumption**: Win rate distribution is NOT uniform across:
- Time periods (market regime dependency)
- Tickers (instrument-specific edge)
- Trade directions (long vs short bias)

**3. Ticker-Level Decomposition**
```
For each ticker:
- Individual PF, WR, Sharpe
- Trade frequency (liquidity proxy)
- Consistency (rolling 30-trade PF)

Purpose: Identify "anchor tickers" (high quality) vs "drag tickers" (dilute portfolio)
```

### **Base Assumptions**

| Assumption | Validation Method | Risk if Invalid |
|------------|-------------------|-----------------|
| Trades are independent | Autocorrelation test | Overstated confidence intervals |
| Past performance indicates edge | Walk-forward validation | Overfitting to historical regime |
| All tickers equally liquid | Volume analysis in base data | Execution slippage in production |
| No survivorship bias | Check delisted tickers | Inflated historical returns |

### **Key Metrics & Interpretation**

| Metric | Formula | Interpretation | Action Threshold |
|--------|---------|----------------|------------------|
| **Profit Factor** | Gross Profit / Gross Loss | System edge | < 1.1: Reject strategy |
| **Win Rate** | Wins / Total Trades | Mean-reversion vs trend | N/A (context-dependent) |
| **Sharpe Ratio** | (Return - Risk-Free) / StdDev | Risk-adjusted return | < 1.0: Poor, > 2.0: Excellent |
| **Max Drawdown** | Peak-to-Trough % | Worst-case loss | > 20%: High risk |
| **Recovery Time** | Days to recover from DD | Capital efficiency | > 90 days: Concerning |

### **Decision Criteria**

**GO Decision** (Proceed to portfolio construction):
- Profit Factor > 1.2
- Win Rate: 45-55% (balanced) OR < 40% but high avg win/loss ratio
- Sharpe Ratio > 1.0
- Max Drawdown < 15%

**NO-GO Decision** (Strategy needs optimization first):
- Profit Factor < 1.1
- Sharpe Ratio < 0.5
- Max Drawdown > 25%

---

## **02. Trade Type Analysis - Directional Bias Detection**

### **Why This Analysis?**
**Core Question**: *"Does the strategy have a structural directional bias that creates hidden risks?"*

**Real-World Problem**:
- A strategy might have PF 1.3 overall but PF 0.9 on shorts → net short portfolio would lose money
- Long bias in bull market ≠ structural edge (may fail in bear market)

### **Backend Logic**

**1. Directional Performance Decomposition**
```
For LONG trades:
- PF_long, WR_long, Avg_Win_long, Avg_Loss_long
- Trade frequency, duration distribution

For SHORT trades:
- PF_short, WR_short, Avg_Win_short, Avg_Loss_short
- Trade frequency, duration distribution

Statistical Test: Two-sample t-test
H0: μ(Long P&L) = μ(Short P&L)
H1: μ(Long P&L) ≠ μ(Short P&L)
```

**Why This Matters**:
- If p-value < 0.05 → statistically significant directional bias
- Portfolio construction must account for this (e.g., only long positions, or hedge with offsetting instruments)

**2. Ticker-Level Directional Preferences**
```
For each ticker, calculate:
Long_Performance_Score = (PF_long × WR_long) / Avg_Duration_long
Short_Performance_Score = (PF_short × WR_short) / Avg_Duration_short

Directional_Bias_Index = (Long_Score - Short_Score) / (Long_Score + Short_Score)

Interpretation:
+1.0: Pure long performer (shorts fail)
-1.0: Pure short performer (longs fail)
 0.0: Direction-neutral (equal performance)
```

### **Base Assumptions**

| Assumption | Implication | Test |
|------------|-------------|------|
| Market is symmetrical (long = short) | No structural bias expected | Compare PF_long vs PF_short |
| Entry signals are direction-neutral | Equal quality longs and shorts | Analyze entry indicator values |
| Transaction costs equal both ways | No execution bias | Check slippage in base data |
| Funding costs negligible | Overnight holding cost ignored | Validate for intraday-only strategies |

### **Key Metrics**

**Directional Bias Ratio (DBR)**:
```
DBR = (Trades_Long - Trades_Short) / Total_Trades

Interpretation:
DBR > +0.3: Long-biased strategy (70% long, 30% short)
DBR < -0.3: Short-biased strategy
-0.2 < DBR < +0.2: Direction-balanced
```

**Risk-Reward Asymmetry**:
```
For each direction:
RRR = Average_Win / Average_Loss

Expected for trend-following: RRR_long > 2.0, RRR_short > 2.0
Expected for mean-reversion: RRR_long ~ 1.0, RRR_short ~ 1.0
```

### **Decision Criteria**

**Portfolio Construction Implications**:

| Finding | Action |
|---------|--------|
| PF_long > 1.3, PF_short < 1.0 | Build long-only portfolio |
| Strong directional bias (DBR > 0.5) | Add market-neutral hedging component |
| Ticker has opposite bias to strategy | Exclude from portfolio (conflicts with edge) |
| Equal performance both directions | Full diversification possible |

---

## **03. Cascade Analysis - Behavioral Pattern Detection**

### **Why This Analysis?**
**Core Question**: *"Are consecutive trades independent, or do they exhibit behavioral coupling (e.g., revenge trading, momentum chasing)?"*

**Critical Insight**: Human/algorithmic behavior often creates trade dependencies:
- **Winning Cascade**: Win → overconfidence → larger position → win → ... (gambler's fallacy)
- **Losing Cascade**: Loss → revenge trade → loss → bigger revenge trade → ... (sunk cost fallacy)
- **Time-Gap Dependency**: Trades <5 minutes apart often share same market microstructure (not independent)

### **Backend Logic**

**1. Trade Sequence Tagging**
```
For each trade T_i, check:
- Is there a previous trade T_(i-1) for same ticker on same day?
- What was the outcome of T_(i-1)? (Win/Loss)
- What is time gap between T_(i-1) exit and T_i entry?

Tag Assignment Logic:
IF no previous trade today → "FIRST_TRADE_OF_DAY" (clean entry)
ELSE IF T_(i-1) was WIN and same direction → "WINNING_CASCADE"
ELSE IF T_(i-1) was LOSS and same direction → "LOSING_CASCADE"
ELSE IF opposite direction → "CONSECUTIVE_OPPOSITE_DIRECTION"
```

**2. Performance Decomposition by Pattern**
```
For each cascade type:
- Win Rate
- Profit Factor
- Average P&L
- Sharpe Ratio

Statistical Test: ANOVA (Analysis of Variance)
H0: μ(First Trade) = μ(Winning Cascade) = μ(Losing Cascade)
H1: At least one mean differs

If p < 0.05 → cascade effect is real, not random
```

**3. Time-Gap Analysis**
```
Time Buckets:
- 0-5 minutes: Microstructure dependency (same price regime)
- 5-15 minutes: Momentum continuation
- 15-30 minutes: Reversion window
- 30-60 minutes: New market context
- 60+ minutes: Independent trade

Hypothesis:
Trades <5 min apart have correlated outcomes (same trend/volatility regime)
Trades >60 min apart are independent
```

### **Base Assumptions**

| Assumption | Reality Check | Bias if Wrong |
|------------|---------------|---------------|
| Strategy has no memory (stateless) | Check if entry logic uses previous trade outcome | Behavioral feedback loop |
| Market microstructure resets every 5 min | Validate with correlation analysis | Overstated independence |
| Winning cascades are luck, not skill | Test if WR increases with streak length | May be real edge (momentum) |
| Losing cascades indicate strategy flaw | Could be market regime shift | May discard valid drawdown periods |

### **Key Metrics**

**Cascade Performance Ratio (CPR)**:
```
CPR = (PF_First_Trade - PF_Losing_Cascade) / PF_First_Trade

Interpretation:
CPR > 0.3: Losing cascades significantly underperform (30% worse)
         → Action: Filter out losing cascades for portfolio
CPR < 0.1: No meaningful cascade effect
         → Action: Use all trades
```

**Temporal Independence Score**:
```
For each time bucket (0-5min, 5-15min, ...):
Calculate correlation of returns:
ρ = Corr(Return_i, Return_(i-1))

Independence if |ρ| < 0.2
Strong dependency if |ρ| > 0.5
```

### **Decision Criteria**

**Portfolio Construction Filter Rules**:

| Finding | Filter Action | Rationale |
|---------|---------------|-----------|
| Losing cascades have PF < 1.0 | **EXCLUDE** all losing cascade trades | Remove behavioral tilt |
| Winning cascades have PF > 1.5 | **INCLUDE** only winning cascades | Capitalize on momentum |
| Time gap <5 min has ρ > 0.5 | **EXCLUDE** rapid consecutive trades | Correlated outcomes inflate trade count |
| First-of-day trades have highest Sharpe | **PREFER** single trade per ticker per day | Quality over quantity |

**Critical Assumption Check**:
If cascade filtering reduces trade count by >50%, verify this isn't overfitting. Use walk-forward validation on unseen data.

---

## **04. Stop Loss Simulation - Risk Management Optimization**

### **Why This Analysis?**
**Core Question**: *"What is the optimal maximum loss threshold to cut losers early while avoiding premature exits?"*

**Fundamental Trade-Off**:
- **Tight Stop Loss** (e.g., 0.5%): Cuts losers fast → Higher WR, but also cuts winners prematurely
- **Loose Stop Loss** (e.g., 3.0%): Lets winners run → Lower WR, but bigger wins when right
- **Optimal Zone**: Minimize $ losses without sacrificing $ wins

### **Backend Logic**

**1. Intra-Trade Excursion Analysis**
```
For each trade:
- Entry Price
- High During Trade (for longs) / Low During Trade (for shorts)
- Exit Price
- Calculate:
  - Max Adverse Excursion (MAE) = Worst drawdown from entry
  - Max Favorable Excursion (MFE) = Best profit from entry

Example (Long trade):
Entry: ₹1,000
Low during trade: ₹985 → MAE = -1.5% (worst drawdown)
High during trade: ₹1,025 → MFE = +2.5% (best profit)
Exit: ₹1,015 → Final P&L = +1.5%
```

**2. Stop Loss Threshold Simulation**
```
For SL_threshold in [0.5%, 1.0%, 1.5%, 2.0%, 2.5%, 3.0%]:
    For each trade:
        IF MAE <= -SL_threshold:
            Simulated_Exit = Entry_Price × (1 - SL_threshold)
            Simulated_PL = -SL_threshold × Entry_Price
        ELSE:
            Simulated_Exit = Actual_Exit
            Simulated_PL = Actual_PL

    Calculate:
    - New Win Rate
    - New Profit Factor
    - New Sharpe Ratio
    - New Max Drawdown
```

**3. Efficiency Ratio**
```
Efficiency = Captured_Profit / Max_Possible_Profit

For each SL threshold:
Efficiency_SL = Σ(Simulated_PL) / Σ(MFE for winning trades)

Interpretation:
High efficiency (>0.8) with SL → Good threshold (preserves wins)
Low efficiency (<0.5) with SL → Too tight (cuts winners)
```

### **Base Assumptions**

| Assumption | Validity Check | Impact if False |
|------------|----------------|-----------------|
| Can execute at exact SL price | Check slippage in base data | Worse realized SL than simulated |
| MAE represents available exit point | Verify tick data timestamps | SL may have been unreachable |
| Strategy has no inherent SL | Review strategy code for existing stops | Double-counting risk management |
| SL applies equally across all tickers | Test by price range (₹50 vs ₹5000 stocks) | % SL may be inadequate for low-priced stocks |

### **Key Metrics**

**1. Win Rate vs Profit Factor Trade-Off**
```
Optimal SL should:
- Increase Win Rate (cut losers)
- Maintain or improve Profit Factor (preserve winners)

If WR increases but PF decreases → SL too tight (false positive)
```

**2. Sharpe Ratio (Risk-Adjusted Performance)**
```
Sharpe = (Avg Return per Trade) / (StdDev of Returns)

Optimal SL maximizes Sharpe by:
- Reducing outlier losses (lowers StdDev)
- Preserving average win size
```

**3. Max Drawdown Reduction**
```
MaxDD_reduction = (MaxDD_original - MaxDD_with_SL) / MaxDD_original

Example:
Original MaxDD: -12%
With 2% SL: MaxDD: -8%
Reduction: 33% improvement
```

### **Decision Criteria**

**Optimal Stop Loss Selection**:

| Metric | Weight | Threshold |
|--------|--------|-----------|
| Sharpe Improvement | 40% | > +10% vs baseline |
| Profit Factor | 30% | > 1.2 minimum |
| Max DD Reduction | 20% | > 20% improvement |
| Win Rate | 10% | Secondary (can decrease if PF compensates) |

**Example Decision Matrix**:
```
SL %  | Sharpe | PF   | MaxDD | WR   | Score
0.5%  | 1.45   | 1.18 | -6%   | 68%  | ❌ Too tight (low PF)
1.0%  | 1.82   | 1.35 | -7%   | 62%  | ✅ OPTIMAL
1.5%  | 1.79   | 1.32 | -8%   | 58%  | ✅ Good
2.0%  | 1.68   | 1.28 | -9%   | 54%  | ⚠️ Marginal
3.0%  | 1.42   | 1.22 | -11%  | 49%  | ❌ Ineffective
```

---

## **05. Ticker Ranking - Quality Scoring System**

### **Why This Analysis?**
**Core Question**: *"Not all tickers are equal - which ones provide the best risk-adjusted edge for portfolio inclusion?"*

**Portfolio Construction Reality**:
- Top 20% of tickers often generate 80% of profits (Pareto principle)
- Including low-quality tickers dilutes portfolio Sharpe ratio
- Need objective, multi-factor scoring system to rank tickers

### **Backend Logic**

**1. Multi-Factor Scoring System**
```
Ticker_Score = Weighted_Sum(Normalized_Metrics)

Components:
1. Profitability (40%):
   - Profit Factor (20%)
   - Total P&L (10%)
   - Average Win Size (10%)

2. Consistency (30%):
   - Win Rate (10%)
   - Sharpe Ratio (15%)
   - Max Drawdown (5%, inverted)

3. Trade Quality (20%):
   - Trade Frequency (10%, capped at threshold)
   - Average Trade Duration (5%, prefer shorter)
   - Recovery Time (5%, prefer faster)

4. Risk-Adjusted (10%):
   - Sortino Ratio (downside deviation only) (10%)

Normalization: Z-score across all tickers
Score_i = Σ(w_i × Z(Metric_i))
```

**2. Z-Score Normalization**
```
For each metric:
Z(x) = (x - μ) / σ

Where:
μ = Mean value across all tickers
σ = Standard deviation

Example (Profit Factor):
Ticker A PF: 1.45
Mean PF: 1.15
StdDev: 0.18
Z(PF_A) = (1.45 - 1.15) / 0.18 = +1.67 (1.67 std devs above average)
```

**Why Z-Scores?**
- Makes metrics comparable (PF is unitless ~1.0-2.0, P&L is ₹ thousands)
- Identifies outliers (Z > +2.0 = top 2.5%, Z < -2.0 = bottom 2.5%)
- Allows weighted aggregation

**3. Tiering System**
```
After scoring all tickers, create tiers:

Tier 1 (Top 20%): Z_composite > +0.8 → "Anchor Holdings"
Tier 2 (Next 30%): 0 < Z_composite < +0.8 → "Core Holdings"
Tier 3 (Next 30%): -0.8 < Z_composite < 0 → "Satellite Holdings"
Tier 4 (Bottom 20%): Z_composite < -0.8 → "Exclude"
```

### **Base Assumptions**

| Assumption | Test | Risk |
|------------|------|------|
| Metrics are independent | Correlation matrix of metrics | Double-counting correlated factors |
| Linear weighting is appropriate | Test non-linear (e.g., Sharpe^2) | Suboptimal weight allocation |
| Past ranking predicts future | Walk-forward ranking stability | Regime change makes ranking obsolete |
| Equal weight across market cap | Compare small vs large cap performance | Size factor bias |

### **Key Metrics**

**Composite Score Formula**:
```
Score = 0.20×Z(PF) + 0.10×Z(Total_PL) + 0.10×Z(Avg_Win)
      + 0.10×Z(WR) + 0.15×Z(Sharpe) - 0.05×Z(MaxDD)
      + 0.10×Z(Trade_Freq) - 0.05×Z(Avg_Duration) - 0.05×Z(Recovery)
      + 0.10×Z(Sortino)
```

**Ranking Stability Test**:
```
Calculate ranking for:
- Period 1: First 50% of trades
- Period 2: Last 50% of trades

Spearman Rank Correlation:
ρ = Corr(Rank_P1, Rank_P2)

ρ > 0.7: Stable ranking (good)
ρ < 0.4: Unstable ranking (regime-dependent)
```

### **Decision Criteria**

**Top 50 Selection**:
```
1. Minimum Thresholds (Hard Filters):
   - Profit Factor > 1.1
   - Win Rate > 40%
   - Total Trades > 50 (statistical significance)
   - Max Drawdown < 25%

2. Ranking:
   - Sort by Composite Score descending
   - Select Top 50

3. Validation:
   - Top 50 must represent >70% of total P&L
   - Top 50 should have >30% higher Sharpe than Bottom 50
```

**Portfolio Allocation**:
```
Tier 1 (Anchor): 50% of capital
Tier 2 (Core): 35% of capital
Tier 3 (Satellite): 15% of capital
Tier 4: 0% (excluded)
```

---

## **06. Risk-Adjusted Patterns - Risk-Normalized Performance**

### **Why This Analysis?**
**Core Question**: *"Which trade patterns deliver alpha after accounting for the risk taken?"*

**Critical Insight**:
- A pattern with +5% avg return but 20% volatility (Sharpe 0.25) is worse than +2% return with 2% volatility (Sharpe 1.0)
- Raw returns without risk context are meaningless
- Portfolio construction must prioritize risk-adjusted patterns

### **Backend Logic**

**1. Pattern Definition**
```
Trade patterns are multi-dimensional classifications:

Time Patterns:
- Hour-of-day (09:15-10:00, 10:00-11:00, ..., 14:30-15:15)
- Day-of-week (Monday, Tuesday, ..., Friday)
- Month-of-year (January, ..., December)

Sequential Patterns:
- First trade of day
- After winning trade
- After losing trade

Duration Patterns:
- Ultra-short (<30 min)
- Short (30-60 min)
- Medium (1-4 hours)
- Long (4+ hours)
```

**2. Sharpe Ratio by Pattern**
```
For each pattern P:

Returns_P = [r1, r2, ..., rn] (all trades matching pattern)

Sharpe_P = (Mean(Returns_P) - Risk_Free_Rate) / StdDev(Returns_P)

Annualized Sharpe (if daily returns):
Sharpe_annual = Sharpe_P × √252

Interpretation:
Sharpe < 0: Pattern loses money on risk-adjusted basis
Sharpe 0-1: Suboptimal (return doesn't compensate for risk)
Sharpe 1-2: Good (1 unit return per 1 unit risk)
Sharpe > 2: Excellent (strong edge)
```

**3. Sortino Ratio (Downside Risk Only)**
```
Sortino_P = (Mean(Returns_P) - Risk_Free_Rate) / Downside_Deviation

Downside_Deviation = √[Σ(min(0, r - MAR)²) / n]

Where MAR = Minimum Acceptable Return (typically 0%)

Why Sortino > Sharpe?
- Penalizes only downside volatility (bad risk)
- Upside volatility is desirable (big wins)
- Better for asymmetric return distributions
```

**4. Risk-Adjusted Return Ratio (RARR)**
```
RARR = (Avg_Profit_per_Trade) / Max_Drawdown_in_Pattern

Example:
Pattern A: Avg +0.8%, Max DD -3% → RARR = 0.267
Pattern B: Avg +0.5%, Max DD -1% → RARR = 0.500 ← BETTER

RARR > 0.5: Excellent (small drawdowns relative to gains)
RARR < 0.2: Poor (large drawdowns eat profits)
```

### **Base Assumptions**

| Assumption | Validation | Consequence if False |
|------------|------------|---------------------|
| Returns are normally distributed | Jarque-Bera test | Sharpe may understate tail risk |
| Pattern edge is stable over time | Split-sample test | May be data mining artifact |
| Patterns are independent | Chi-square test | Overstated diversification benefit |
| Sufficient sample size per pattern | n > 30 trades minimum | Unreliable statistics |

### **Key Metrics**

**Pattern Quality Score**:
```
Quality = 0.4 × Sharpe + 0.3 × Sortino + 0.2 × RARR + 0.1 × Win_Rate

Normalized to 0-100 scale:
Score < 40: Avoid pattern
Score 40-60: Marginal pattern
Score 60-80: Good pattern
Score > 80: Excellent pattern (prioritize in portfolio)
```

### **Decision Criteria**

**Pattern-Based Trade Filtering**:

| Pattern Sharpe | Action | Rationale |
|---------------|--------|-----------|
| > 2.0 | **INCLUDE** in portfolio | Strong risk-adjusted edge |
| 1.0 - 2.0 | **CONDITIONAL** (if diversifying) | Acceptable edge |
| 0.5 - 1.0 | **AVOID** unless fills gap | Weak edge, risk not compensated |
| < 0.5 | **EXCLUDE** | Negative risk-adjusted return |

**Time-of-Day Filtering Example**:
```
If analysis shows:
09:30-10:30 window: Sharpe 2.4 (excellent)
14:00-15:00 window: Sharpe 0.3 (poor)

Action: Filter out afternoon trades from portfolio
Expected Impact: Improve portfolio Sharpe by removing low-quality trades
```

---

## **07. Top50 vs Overall Comparison - Selection Validation**

### **Why This Analysis?**
**Core Question**: *"Does selecting Top 50 tickers actually improve portfolio metrics, or are we overfitting to noise?"*

**Statistical Validation**:
- Ticker ranking is based on historical data
- Risk: We're selecting tickers that "got lucky" (random variance, not true edge)
- Need: Prove Top 50 have statistically significant better performance

### **Backend Logic**

**1. Statistical Hypothesis Testing**
```
Null Hypothesis (H0): Top 50 and Bottom tickers have equal mean returns
Alternative (H1): Top 50 have higher mean returns

Test: Independent samples t-test

t = (μ_top50 - μ_overall) / SE_difference

Where:
SE = √[(s²_top50/n_top50) + (s²_overall/n_overall)]

Decision:
If p-value < 0.05 → Reject H0 (Top 50 are truly better)
If p-value > 0.05 → Cannot reject H0 (might be luck)
```

**2. Effect Size (Cohen's d)**
```
d = (μ_top50 - μ_overall) / σ_pooled

Interpretation (Cohen's conventions):
d < 0.2: Negligible difference
d = 0.2-0.5: Small effect
d = 0.5-0.8: Medium effect
d > 0.8: Large effect

Example:
Top 50 Sharpe: 1.85
Overall Sharpe: 1.12
StdDev: 0.42
d = (1.85 - 1.12) / 0.42 = 1.74 (very large effect)
```

**3. Concentration Risk Metrics**
```
Portfolio Concentration (Herfindahl Index):
HHI = Σ(Weight_i²)

If equal-weighted 50 tickers:
HHI = 50 × (1/50)² = 0.02

Interpretation:
HHI < 0.01: Highly diversified (100+ tickers)
HHI 0.01-0.10: Moderate concentration (10-100 tickers)
HHI > 0.10: High concentration (<10 tickers)

Trade-off:
More tickers → Lower concentration risk, but diluted alpha
Fewer tickers → Higher alpha, but idiosyncratic risk
```

**4. Incremental Value Analysis**
```
For each decile (Top 10, Top 20, ..., Top 50):
Calculate portfolio Sharpe if stopped at that decile

Example:
Top 10: Sharpe 2.1
Top 20: Sharpe 2.0 (dilution starts)
Top 30: Sharpe 1.92
Top 40: Sharpe 1.87
Top 50: Sharpe 1.85 ← Still good, acceptable dilution

Decision: Use Top 50 (balance alpha vs diversification)
```

### **Base Assumptions**

| Assumption | Risk | Mitigation |
|------------|------|------------|
| Ranking is predictive (not luck) | Overfitting | Walk-forward validation |
| Top 50 won't mean-revert | Winner's curse | Monitor live performance |
| Linear relationship (more tickers = worse) | May plateau | Test multiple thresholds |
| Independence of ticker performance | Sector correlation | Sector diversification rules |

### **Key Metrics**

**Improvement Ratio**:
```
IR = (Sharpe_Top50 - Sharpe_Overall) / Sharpe_Overall

IR > 0.3: Significant improvement (30%+ better)
IR 0.1-0.3: Moderate improvement
IR < 0.1: Marginal (may not be worth concentration risk)
```

**Diversification Benefit**:
```
DB = σ_portfolio / Average(σ_individual_tickers)

Perfect diversification: DB → 0.5 (50% risk reduction)
No diversification: DB → 1.0
Negative diversification: DB > 1.0 (correlated tickers increase risk)
```

### **Decision Criteria**

**Validate Top 50 Selection**:
```
Required:
1. p-value < 0.05 (statistically significant)
2. Cohen's d > 0.5 (medium+ effect size)
3. Sharpe improvement > 20%
4. Concentration HHI < 0.05 (max 5% in any ticker)

If all met → Proceed with Top 50
If any failed → Expand to Top 75 or implement stricter quality filters
```

---

## **08. Top50 Pattern Breakdown - Winner Profiling**

### **Why This Analysis?**
**Core Question**: *"What specific characteristics make Top 50 tickers outperform - can we amplify these traits?"*

**Objective**: Decompose WHY top tickers win:
- Time patterns (when they trade)
- Duration patterns (how long they hold)
- Sequential patterns (first trade vs cascades)
- Risk patterns (drawdown behavior)

### **Backend Logic**

**1. Pattern Prevalence Analysis**
```
For each pattern P:

Prevalence_Top50 = (Trades_with_P in Top50) / (Total_Trades_Top50)
Prevalence_Overall = (Trades_with_P overall) / (Total_Trades)

Enrichment_Ratio = Prevalence_Top50 / Prevalence_Overall

Example:
First-trade-of-day pattern:
Top 50: 42% of trades
Overall: 28% of trades
Enrichment: 42/28 = 1.5× (Top 50 use this pattern 50% more)
```

**2. Pattern Contribution to Alpha**
```
For each pattern P in Top 50:

Alpha_P = (Avg_Return_P - Avg_Return_Overall) × Frequency_P

Example:
First-trade pattern in Top 50:
Avg Return: +1.2%
Overall Avg: +0.6%
Frequency: 42% of trades
Alpha contribution: (1.2% - 0.6%) × 0.42 = +0.25% to portfolio return
```

**3. Duration Analysis**
```
Hypothesis: Top 50 may have shorter/longer optimal holding periods

For Top 50 vs Overall:
- Plot duration distribution (histogram)
- Calculate median duration
- Identify mode (most common duration)

Statistical Test: Kolmogorov-Smirnov test
H0: Duration distributions are identical
H1: Top 50 have different duration profile
```

### **Base Assumptions**

| Assumption | Implication | Test |
|------------|-------------|------|
| Patterns are causal, not correlation | Top 50 succeed BECAUSE of patterns | Causality requires theory + data |
| Patterns will persist | Past patterns predict future | Out-of-sample validation |
| Patterns are replicable | Can force trades into these patterns | May require trade filtering |

### **Key Metrics**

**Pattern Significance Score**:
```
For each pattern:
Significance = Enrichment_Ratio × Alpha_Contribution × √(Sample_Size)

High Significance → Pattern is:
1. More prevalent in Top 50 (Enrichment > 1.0)
2. High alpha contribution
3. Statistically valid (large sample)
```

### **Decision Criteria**

**Pattern-Based Portfolio Rules**:
```
If pattern has:
- Enrichment > 1.3 (30% more common in Top 50)
- Alpha contribution > +0.15%
- Sample size > 100 trades

Action: PREFER trades matching this pattern in portfolio
```

**Example Findings**:
```
Pattern: First trade of day, 09:30-10:30, Duration <60 min
Enrichment: 2.1× (Top 50 use this 210% more often)
Alpha: +0.41% per trade
Sample: 1,247 trades

Recommendation: Filter portfolio to prioritize morning first-entry trades
Expected Impact: +5-8% Sharpe improvement
```

---

## **09. Validation Check - Data Integrity Audit**

### **Why This Analysis?**
**Core Question**: *"Can we trust this data for production deployment, or are there data quality issues?"*

**Critical Reality**:
- Garbage in = Garbage out
- One bad data point (e.g., price spike from fat-finger trade) can corrupt entire analysis
- Before portfolio construction, verify data integrity

### **Backend Logic**

**1. Missing Data Detection**
```
For each required field:
- Check null/NaN values
- Check empty strings
- Check placeholder values (0, -999, "N/A")

Critical Fields:
- Entry Time, Exit Time (no nulls allowed)
- Entry Price, Exit Price (must be > 0)
- P&L (must exist, can be negative)

Anomaly Thresholds:
- Missing data >1% of trades → Investigate source
- Missing data >5% → Data quality issue, reject dataset
```

**2. Price Anomaly Detection**
```
For each trade:

Entry_to_Exit_Change = |Exit_Price - Entry_Price| / Entry_Price

Flag anomalies:
- Change > 10% in <5 minutes → Likely data error or flash crash
- Entry Price = Exit Price exactly → Possible execution error
- Price = 0 or negative → Critical data error

Statistical Outlier Detection (Z-score method):
Z = (Price_Change - μ) / σ
If |Z| > 4.0 → Extreme outlier (>99.99% percentile)
```

**3. Timestamp Validation**
```
Market Hours Check (NSE India example):
Valid trading: 09:15 - 15:30 IST, Monday-Friday

Anomalies:
- Trades outside market hours → Data error
- Exit before Entry → Time sequence error
- Duplicate timestamps → Order execution logging issue
- Gaps > 30 days → Missing data period

Sequence Validation:
For each ticker:
Sort trades by Entry Time
Check: Entry_i+1 >= Exit_i (no overlapping trades for same ticker)
```

**4. P&L Reconciliation**
```
Calculate P&L from prices:
Calculated_PL = (Exit_Price - Entry_Price) × Quantity

Compare to Reported_PL:
Discrepancy = |Calculated_PL - Reported_PL| / |Reported_PL|

Tolerance:
- Discrepancy < 1%: Rounding error (acceptable)
- Discrepancy 1-5%: Transaction costs/slippage (acceptable if documented)
- Discrepancy > 5%: Data integrity issue (investigate)
```

### **Base Assumptions**

| Assumption | Validation | Impact if Invalid |
|------------|------------|-------------------|
| Prices reflect executed trades (not quotes) | Check for bid-ask spread consistency | Inflated backtest returns |
| Timestamps are accurate | Verify against market data provider | Incorrect sequence analysis |
| No survivorship bias | Check for delisted tickers | Overstated returns |
| Transaction costs included | Verify fee structure in P&L | Underestimated costs |

### **Key Metrics**

**Data Quality Score (DQS)**:
```
DQS = 100 - (Penalty_Points)

Penalties:
- Missing critical fields: -10 per 1% of trades affected
- Price anomalies: -5 per anomaly
- Timestamp errors: -3 per error
- P&L discrepancies: -8 per 1% affected

DQS > 90: Excellent (production-ready)
DQS 70-90: Good (minor issues, acceptable)
DQS < 70: Poor (needs data cleaning)
```

### **Decision Criteria**

**GO / NO-GO for Portfolio Construction**:

```
PASS if ALL:
✅ Missing data < 1%
✅ Price anomalies < 0.5% of trades
✅ Timestamp errors = 0
✅ P&L reconciliation discrepancy < 2%
✅ DQS > 85

CONDITIONAL if:
⚠️ 1-3% minor issues → Document and proceed with caution
⚠️ DQS 75-85 → Clean data, re-validate

FAIL if ANY:
❌ Missing critical data > 3%
❌ Price anomalies > 2%
❌ Timestamp sequence errors
❌ P&L discrepancies > 5%
❌ DQS < 70
```

---

# 🏗️ **PORTFOLIO CONSTRUCTION SUITE**

**Purpose**: Given validated, high-quality trades, construct optimal portfolios maximizing risk-adjusted returns.

**Philosophy**: Portfolio > sum of parts (diversification benefit, but only if done correctly with correlation constraints).

---

## **00. Foundation Analysis - Comprehensive Ticker Ranking**

### **Why This Analysis?**
**Core Question**: *"Which tickers are portfolio-worthy across ALL trade types (cascading, anti-cascading, first-of-day)?"*

**Strategic Insight**:
- Portfolio construction will filter trades (e.g., exclude cascades)
- Must rank tickers BEFORE filtering to ensure foundations are solid
- Creates three ranking lists:
  1. ALL trades (baseline)
  2. CASCADING trades only (behavioral patterns)
  3. ANTI-CASCADING trades only (clean entries)

### **Backend Logic**

**1. Triple Ranking System**
```
For each ticker, calculate scores across three subsets:

ALL_TRADES:
Score_all = Weighted_Average(PF, WR, Sharpe, Trade_Freq, Total_PL)

CASCADING_ONLY (after winning or losing trade):
Score_cascade = Same weighted formula, but only cascade trades

ANTI_CASCADING_ONLY (first trade of day, or >60 min gap):
Score_anti = Same weighted formula, but only anti-cascade trades

Rationale:
If ticker ranks high on ALL but low on ANTI → Edge comes from cascading (risky)
If ticker ranks high on ANTI → True edge (preferred for portfolio)
```

**2. Consistency Check**
```
Rank_Correlation = Spearman_ρ(Rank_All, Rank_Anti_Cascade)

High correlation (ρ > 0.7):
→ Ticker quality is consistent regardless of trade type (robust)

Low correlation (ρ < 0.4):
→ Ticker is regime-dependent (avoid)
```

**3. Affordability Filter**
```
Price_Filter_Threshold = ₹2,000 (configurable)

Rationale:
- Higher-priced stocks (e.g., ₹10,000) require larger capital per position
- Limits diversification (can't build 5-ticker portfolio with ₹50,000 capital)
- Focus on liquid, affordable tickers for retail trading

Economic Constraint:
Min_Capital_per_Ticker = Price × Lot_Size
Portfolio_Capital = Min_Capital_per_Ticker × Number_of_Tickers
```

### **Base Assumptions**

| Assumption | Validation | Risk |
|------------|------------|------|
| Price threshold (₹2,000) is appropriate | Backtest with different thresholds | May exclude high-quality expensive stocks |
| Rankings are stable across trade types | Cross-validation on splits | Overfitting to specific pattern |
| Anti-cascade trades are "cleaner" | Verify via MAE analysis | May exclude valid continuation trades |

### **Key Metrics**

**Affordable Ticker Universe**:
```
After applying price filter (< ₹2,000):
- Typically: 17-25 tickers from initial 30-50
- Ensures minimum 4-5 ticker portfolio feasibility
- Preserves diversification potential
```

**Ranking Stability**:
```
For Top 10 in each list:
Overlap = |Set(Top10_All) ∩ Set(Top10_Anti)| / 10

Overlap > 0.7 (7+ common tickers): Stable quality
Overlap < 0.5: Inconsistent (investigate)
```

### **Decision Criteria**

**Proceed to Filtering**:
```
Required:
- At least 15 affordable tickers (< ₹2,000)
- Top 10 anti-cascade have PF > 1.2
- Top 10 anti-cascade have Sharpe > 1.0
- Ranking stability (Spearman ρ > 0.6)
```

---

## **01. Anti-Cascade Filter - Behavioral Bias Removal**

### **Why This Analysis?**
**Core Question**: *"Should we use ALL trades or filter to ANTI-CASCADING trades only for portfolio construction?"*

**Behavioral Finance Foundation**:
- **Recency Bias**: Traders (human or algo) over-react to recent wins/losses
- **Gambler's Fallacy**: "I just won, I'm hot" → over-confidence → larger positions
- **Loss Aversion**: "I just lost, must recover" → revenge trading → poor entries

**Statistical Evidence**:
If cascade analysis (Module 03) shows:
- Losing cascades have PF < 1.0 → Clear behavioral bias
- First-of-day trades have Sharpe 2.1 vs cascades 1.4 → Quality difference

Action: Filter dataset to ANTI-CASCADING ONLY

### **Backend Logic**

**1. Trade Classification**
```
For each trade, assign category:

CONSECUTIVE_SAME_DIRECTION:
→ Same ticker, same direction, <60 min after previous trade
→ EXCLUDE (cascading behavior)

FIRST_TRADE_OF_DAY:
→ No previous trade for this ticker today
→ INCLUDE (clean entry)

CONSECUTIVE_OPPOSITE_DIRECTION:
→ Same ticker, opposite direction (long after short, or vice versa)
→ INCLUDE (counter-trend reversal, not cascading)

FIRST_TRADE_FOR_TICKER:
→ First time trading this ticker in entire dataset
→ INCLUDE (no prior bias)
```

**2. Impact Quantification**
```
Before Filtering:
Total Trades: 43,191
Tickers: 24
Avg Sharpe: 1.12

After Anti-Cascade Filtering:
Remaining Trades: 24,546 (56.8% retained)
Tickers: 17 (some excluded due to low trade count)
Avg Sharpe: 1.48 (+32% improvement)

Trade-off:
Lose 43% of trades BUT gain 32% Sharpe
Net: Better risk-adjusted expectancy
```

**3. Trade Frequency Validation**
```
For each ticker after filtering:
Min_Trades_Threshold = 200 (configurable)

Rationale:
- Need sufficient sample size for portfolio optimization
- Ticker with <200 trades has insufficient data
- Law of Large Numbers: More trades → more reliable statistics

Post-Filter Check:
If ticker has <200 anti-cascade trades → EXCLUDE from portfolio universe
```

### **Base Assumptions**

| Assumption | Test | Impact if Wrong |
|------------|------|-----------------|
| Cascading trades underperform | Compare PF_cascade vs PF_first | May discard valid trades |
| 60-minute gap = independence | Autocorrelation test on returns | Arbitrary threshold |
| Direction change breaks cascade | Analyze opposite-direction performance | May miss behavioral coupling |
| Sample size (200) is sufficient | Statistical power analysis | Overly strict (lose tickers) or loose (noise) |

### **Key Metrics**

**Cascade Impact Ratio**:
```
CIR = (Sharpe_Anti_Cascade - Sharpe_All) / Sharpe_All

CIR > 0.2 (20% improvement): Strong filtering benefit → Use anti-cascade
CIR < 0.05: Marginal benefit → Consider using all trades
```

**Trade Retention Rate**:
```
TRR = Trades_After_Filter / Trades_Before_Filter

TRR > 0.5: Acceptable (retain majority of data)
TRR < 0.3: Too aggressive (losing too much data)
```

### **Decision Criteria**

**Use Anti-Cascade Filter if**:
```
ALL of:
✅ Cascade analysis shows PF_cascade < PF_first (at least 10% worse)
✅ Sharpe improvement > 15% (CIR > 0.15)
✅ Trade retention > 50% (TRR > 0.5)
✅ At least 15 tickers remain with >200 trades each
```

---

## **02. Sector Classification - Diversification Framework**

### **Why This Analysis?**
**Core Question**: *"How do we avoid concentration risk where all portfolio tickers are from the same sector?"*

**Systematic Risk Reality**:
- Banking sector crash 2008: All banks moved together (-40 to -60%)
- Tech bubble 2000: All IT stocks collapsed simultaneously
- Sector correlation during crises → 0.9 (move in lockstep)

**Portfolio Theory** (Markowitz):
Diversification benefit ONLY if assets are imperfectly correlated
σ_portfolio = √[Σ(w_i² × σ_i²) + Σ(w_i × w_j × ρ_ij × σ_i × σ_j)]

If ρ_ij → 1.0 (perfect correlation): No diversification benefit

### **Backend Logic**

**1. Sector Mapping**
```
Manual/External Classification:
AXISBANK → Banking & Financial Services
KOTAKBANK → Banking & Financial Services
HCLTECH → Information Technology
INFY → Information Technology
RELIANCE → Energy & Power
NTPC → Unclassified (Utilities, but small)
...

Data Source:
- NSE sector classification
- Bloomberg Industry Classification Standard (BICS)
- Or manual assignment based on business model
```

**2. Correlation Matrix Calculation**
```
For each pair of tickers (i, j):

Daily_Returns_i = [r1, r2, ..., rn] (from trade data)
Daily_Returns_j = [r1, r2, ..., rn] (same dates)

Pearson Correlation:
ρ_ij = Cov(Returns_i, Returns_j) / (σ_i × σ_j)

Interpretation:
ρ = +1.0: Perfect positive correlation (move identically)
ρ = 0.0: No correlation (independent)
ρ = -1.0: Perfect negative correlation (hedge)

Practical:
ρ > 0.7: High correlation (limited diversification benefit)
ρ < 0.3: Low correlation (good diversification)
```

**3. Intra-Sector vs Inter-Sector Correlation**
```
Within-Sector Correlation (e.g., Banking tickers):
Avg_ρ_within = Mean(ρ_AXISBANK_KOTAKBANK, ρ_AXISBANK_SBIN, ...)

Cross-Sector Correlation (e.g., Banking vs IT):
Avg_ρ_cross = Mean(ρ_AXISBANK_HCLTECH, ρ_KOTAKBANK_INFY, ...)

Diversification Benefit:
DB = (Avg_ρ_within - Avg_ρ_cross) / Avg_ρ_within

Example:
Within-sector: 0.65
Cross-sector: 0.28
DB = (0.65 - 0.28) / 0.65 = 56.9% risk reduction from cross-sector diversification
```

### **Base Assumptions**

| Assumption | Reality Check | Risk |
|------------|---------------|------|
| Sector classification is stable | Companies rarely change sectors | Conglomerate tickers (e.g., Reliance) span multiple sectors |
| Correlation is stationary | Test rolling 30-day correlation | Regime shifts during crises |
| Daily returns capture correlation | Use intraday 5-min returns for higher precision | May miss macro correlations |
| Linear correlation (Pearson) is appropriate | Test Spearman (rank correlation) | Non-linear dependencies missed |

### **Key Metrics**

**Sector Concentration Ratio**:
```
SCR = (Tickers_in_Largest_Sector) / (Total_Tickers)

Example:
9 Unclassified, 3 Banking, 2 IT, 1 Energy, 1 FMCG, 1 Infrastructure
Total: 17 tickers
Largest sector (Unclassified): 9/17 = 52.9%

SCR > 0.6 (60%): High concentration (risky)
SCR < 0.4 (40%): Well-diversified
```

**Average Inter-Asset Correlation**:
```
Avg_ρ = Mean(All pairwise correlations)

Avg_ρ < 0.2: Excellent diversification
Avg_ρ 0.2-0.5: Moderate diversification
Avg_ρ > 0.5: Poor diversification (tickers move together)
```

### **Decision Criteria**

**Portfolio Combination Rules**:
```
For 5-ticker portfolio:

Sector Diversification Rule:
Max 3 tickers from same sector (60% max concentration)

Correlation Constraint:
Average pairwise correlation < 0.5

Example Valid Portfolio:
AXISBANK (Banking), KOTAKBANK (Banking), HCLTECH (IT), INFY (IT), RELIANCE (Energy)
→ 2 Banking, 2 IT, 1 Energy (40% max, passes)
→ Avg correlation: 0.31 (passes)

Example Invalid Portfolio:
AXISBANK, KOTAKBANK, SBIN, HCLTECH, INFY
→ 3 Banking (60%, borderline) + 2 IT (40%)
→ If avg correlation > 0.5 → REJECT
```

---

## **03. Combination Generator - Constrained Optimization Space**

### **Why This Analysis?**
**Core Question**: *"How many valid portfolio combinations exist after applying diversification constraints?"*

**Combinatorial Math**:
Without constraints:
C(17, 5) = 17! / (5! × 12!) = 6,188 possible 5-ticker combinations

With constraints (sector max 60%, correlation < 0.75):
Valid combinations: ~5,054 (81.7% pass rate)

**Computational Efficiency**:
Instead of testing ALL 6,188 in portfolio optimizer:
Pre-filter → Only test 5,054 → 18% faster

### **Backend Logic**

**1. Combinatorial Generation**
```python
from itertools import combinations

# Generate all possible 5-ticker combinations
all_combos = list(combinations(ticker_list, 5))

# Example:
tickers = ['AXISBANK', 'KOTAKBANK', 'HCLTECH', 'INFY', 'RELIANCE', ...]
all_combos = [
    ('AXISBANK', 'KOTAKBANK', 'HCLTECH', 'INFY', 'RELIANCE'),
    ('AXISBANK', 'KOTAKBANK', 'HCLTECH', 'INFY', 'NTPC'),
    ...
]
Total: 6,188 combinations for 5-ticker portfolios
```

**2. Sector Filter**
```
For each combination:
    Count tickers per sector:
    sector_counts = {
        'Banking': 2,
        'IT': 2,
        'Energy': 1
    }

    Max_sector_count = max(sector_counts.values())
    Max_allowed = ceil(0.6 × portfolio_size)  # 60% rule

    IF Max_sector_count <= Max_allowed:
        → PASS (proceed to correlation check)
    ELSE:
        → FAIL (reject combination)
```

**3. Correlation Filter**
```
For each combination that passed sector filter:
    Extract pairwise correlations:
    For 5 tickers: 10 pairs (C(5,2) = 10)

    correlations = [
        ρ(AXISBANK, KOTAKBANK),  # 0.68
        ρ(AXISBANK, HCLTECH),    # 0.21
        ρ(AXISBANK, INFY),       # 0.18
        ...
    ]

    Avg_correlation = Mean(correlations)

    IF Avg_correlation < 0.75:
        → PASS (valid combination)
    ELSE:
        → FAIL (too correlated)
```

### **Base Assumptions**

| Assumption | Justification | Risk |
|------------|---------------|------|
| 60% sector max is optimal | Industry standard for diversification | May be too strict (lose good combos) or loose (concentrated risk) |
| 0.75 correlation threshold | Empirical (>0.75 = high correlation in literature) | Arbitrary cutoff |
| Equal weighting for filter (1/N) | Simplification (actual weights determined later) | May pass combos that fail with optimal weights |
| Historical correlation = future correlation | Stationarity assumption | Crisis correlation → 1.0 |

### **Key Metrics**

**Pass Rate**:
```
Pass_Rate = Valid_Combinations / Total_Combinations

Example:
5,054 valid / 6,188 total = 81.7%

High pass rate (>80%): Constraints not too strict
Low pass rate (<50%): Constraints may be overly restrictive
```

**Diversification Score Distribution**:
```
For all valid combinations:
Calculate average pairwise correlation

Distribution:
Min correlation: 0.12
Max correlation: 0.74
Mean: 0.32
Median: 0.29

Interpretation: Most portfolios are well-diversified (low correlation)
```

### **Decision Criteria**

**Constraints Calibration**:
```
IF Pass_Rate < 50%:
→ Relax constraints (e.g., 70% sector max, 0.80 correlation)

IF Pass_Rate > 95%:
→ Tighten constraints (e.g., 50% sector max, 0.65 correlation)

Target: 70-90% pass rate (balance quality vs quantity)
```

---

## **04. Portfolio Optimization Engine - Equal-Weight Evaluation**

### **Why This Analysis?**
**Core Question**: *"Which portfolio combination delivers the best risk-adjusted returns with equal capital allocation?"*

**Equal-Weight Baseline**:
- Simplest allocation: 1/N per ticker (e.g., 20% each for 5-ticker portfolio)
- No optimization bias
- Establishes baseline before Markowitz optimization

**Portfolio-Level Performance**:
- Individual ticker Sharpe may be 1.2
- Portfolio Sharpe may be 1.8 (diversification benefit)
- Or Portfolio Sharpe may be 0.9 (correlation reduces benefit)

### **Backend Logic**

**1. Daily Return Aggregation**
```
For portfolio P = [Ticker1, Ticker2, ..., Ticker5]:

For each trading day D:
    Daily_Return_P(D) = Equal-Weighted Average of daily returns

    Daily_Return_P(D) = Σ(Weight_i × Return_i(D))

    Where Weight_i = 1/N = 0.20 for 5-ticker portfolio

Example:
2024-03-15:
AXISBANK: +0.8%
KOTAKBANK: +1.2%
HCLTECH: -0.3%
INFY: +0.5%
RELIANCE: +0.2%

Portfolio return = (0.8 + 1.2 - 0.3 + 0.5 + 0.2) / 5 = +0.48%
```

**2. Portfolio Sharpe Ratio**
```
Portfolio_Sharpe = (Annualized_Return - Risk_Free_Rate) / Annualized_Volatility

Annualized_Return = Mean(Daily_Returns) × 252
Annualized_Volatility = StdDev(Daily_Returns) × √252

Example:
Daily mean: +0.013%
Daily std: 0.52%
Annual return: 0.013% × 252 = 3.28%
Annual volatility: 0.52% × √252 = 8.25%
Sharpe = 3.28% / 8.25% = 0.398

With diversification benefit:
Portfolio volatility = 4.08% (50% reduction)
Portfolio Sharpe = 3.28% / 4.08% = 0.804
```

**3. Maximum Drawdown**
```
For portfolio equity curve:
Equity(t) = Starting_Capital × Π(1 + Daily_Return_i)

Running_Max = max(Equity[0:t])
Drawdown(t) = (Equity(t) - Running_Max) / Running_Max × 100

Max_Drawdown = min(Drawdown)

Example:
Peak equity: ₹112,000
Trough: ₹106,500
Max DD = (106,500 - 112,000) / 112,000 = -4.91%
```

### **Base Assumptions**

| Assumption | Validation | Risk |
|------------|------------|------|
| Equal-weight is rational | Test against market-cap weighted | May underweight best performers |
| Daily rebalancing implied | Analyze rebalancing frequency impact | Transaction costs |
| No position sizing | Assumes infinite capital | Real-world: position limits |
| Returns are additive | Check for portfolio leverage | Non-linear effects |

### **Key Metrics**

**Ranking Metrics** (for 5,054 portfolios):
```
Primary: Portfolio Sharpe Ratio (risk-adjusted return)
Secondary: Profit Factor (gross profit / gross loss)
Tertiary: Win Rate (consistency)
Quaternary: Max Drawdown (risk)

Weighted Composite Score:
Score = 0.4 × Sharpe + 0.3 × PF + 0.2 × WR + 0.1 × (1 - MaxDD)
```

**Top 50 Selection**:
```
Sort all 5,054 portfolios by Sharpe descending
Select Top 50

Validation:
Top 50 should have:
- Sharpe > 0.6 (minimum acceptable)
- PF > 1.1 (profitable)
- MaxDD < 10% (controlled risk)
```

### **Decision Criteria**

**Portfolio Quality Thresholds**:
```
Tier 1 (Excellent): Sharpe > 0.8, PF > 1.15, MaxDD < 5%
Tier 2 (Good): Sharpe 0.6-0.8, PF 1.1-1.15, MaxDD 5-8%
Tier 3 (Acceptable): Sharpe 0.4-0.6, PF 1.05-1.1, MaxDD 8-10%
Reject: Sharpe < 0.4, PF < 1.05, MaxDD > 10%
```

**Best Portfolio Identification**:
```
Rank #1:
Tickers: AXISBANK, HCLTECH, INFY, SUNPHARMA, KOTAKBANK
Sharpe: 0.839
PF: 1.16
WR: 51.8%
MaxDD: -4.88%
Annual Return: 3.42%
Annual Vol: 4.08%

Verdict: Tier 1 (Excellent) → Use for production
```

---

## **05. PyPortfolioOpt Optimal Weights - Markowitz Optimization**

### **Why This Analysis?**
**Core Question**: *"Can we beat equal-weight (1/N) allocation with optimal weights from mean-variance optimization?"*

**Markowitz Portfolio Theory (1952)**:
- Equal-weight is naive (ignores expected returns and correlations)
- Optimal weights maximize Sharpe ratio
- Mathematically derived from quadratic optimization

**Expected Improvement**:
Equal-weight Sharpe: 0.83
Optimized Sharpe: 0.90-1.05 (+8-26% improvement)

But: **Only if optimization is robust (not overfit to noise)**

### **Backend Logic**

**1. Expected Returns Calculation**
```
For each ticker in portfolio:

Expected_Return_i = Mean(Historical_Daily_Returns) × 252

Example:
AXISBANK: Mean daily = +0.034% → Annual = 0.034% × 252 = 8.57%
HCLTECH: Mean daily = +0.014% → Annual = 3.53%
...

Vector of expected returns:
μ = [8.57%, 3.53%, 4.52%, 3.48%, 5.23%]  # 5 tickers
```

**2. Covariance Matrix**
```
For each pair (i, j):

Cov(i, j) = E[(R_i - μ_i) × (R_j - μ_j)]

Annualized:
Σ = Cov_Matrix × 252

Example 5×5 matrix:
        AXIS   HCL    INFY   SUN    KOT
AXIS    0.21   0.08   0.06   0.05   0.14
HCL     0.08   0.15   0.11   0.07   0.09
INFY    0.06   0.11   0.14   0.06   0.08
SUN     0.05   0.07   0.06   0.11   0.07
KOT     0.14   0.09   0.08   0.07   0.18
```

**3. Optimization Problem**
```
Maximize: Sharpe_Ratio = (μ^T w - r_f) / √(w^T Σ w)

Subject to:
- Σ w_i = 1 (weights sum to 100%)
- w_i >= 0 (no short selling)
- Optional: w_i >= 0.05 (min 5% per ticker, avoid micro-allocations)
- Optional: w_i <= 0.40 (max 40% per ticker, avoid concentration)

Solve via quadratic programming (convex optimization)
```

**4. Regularization (L2 Penalty)**
```
Without regularization:
Optimal weights may be extreme [0%, 0%, 80%, 15%, 5%]
→ Overfits to historical noise
→ High turnover (transaction costs)

With L2 regularization (γ = 0.1):
Penalize large weights:
Objective = Sharpe - γ × Σ(w_i²)

Result: More balanced weights [18%, 22%, 28%, 16%, 16%]
→ Robust to estimation error
→ Lower turnover
```

### **Base Assumptions**

| Assumption | Reality | Mitigation |
|------------|---------|------------|
| Expected returns = historical mean | Past ≠ future | Use robust estimators (shrinkage) |
| Covariance is stationary | Correlation changes in crises | Use exponentially-weighted covariance |
| Normal distribution of returns | Fat tails exist | Use CVaR (Conditional Value at Risk) |
| No transaction costs | Costs erode returns | Model turnover explicitly |
| Continuous rebalancing | Discrete rebalancing in practice | Simulate monthly/quarterly rebalancing |

### **Key Metrics**

**Optimization Methods Tested**:
```
1. Equal Weight (1/N): Baseline
   Weights: [20%, 20%, 20%, 20%, 20%]
   Sharpe: 0.83

2. Max Sharpe (No regularization):
   Weights: [4%, 2%, 78%, 11%, 5%]
   Sharpe: 1.12
   ⚠️ Risk: Overconcentrated (78% in one ticker)

3. Max Sharpe (L2 γ=0.1):
   Weights: [18%, 22%, 28%, 16%, 16%]
   Sharpe: 1.02
   ✅ Preferred: Balanced + improved Sharpe

4. Min Volatility:
   Weights: [25%, 30%, 15%, 20%, 10%]
   Sharpe: 0.89
   Use case: Conservative portfolios

5. Efficient Risk (Target Vol = 15%):
   Weights: [20%, 25%, 20%, 18%, 17%]
   Sharpe: 0.96
   Use case: Risk budgeting
```

### **Decision Criteria**

**Select Optimization Method**:
```
IF improvement > 10% AND concentration < 40%:
→ Use optimized weights

IF improvement < 5%:
→ Use equal-weight (not worth complexity)

IF concentration > 50%:
→ Use equal-weight (overfitting risk)

Example Decision:
Max Sharpe (L2): Sharpe +23%, Max weight 28% → ✅ USE
Equal Weight: Sharpe baseline, Max weight 20% → Baseline
```

**Production Weights**:
```
Best Portfolio: AXISBANK, HCLTECH, INFY, SUNPHARMA, KOTAKBANK

Equal Weight:
[20%, 20%, 20%, 20%, 20%]
Expected Sharpe: 0.83

Optimized (Max Sharpe L2 γ=0.1):
[18%, 24%, 26%, 15%, 17%]
Expected Sharpe: 1.02 (+23%)

Verdict: Use optimized weights (robust improvement)
```

---

## **06. Equity Curve Generator - Visual Validation**

### **Why This Analysis?**
**Core Question**: *"Do the portfolio metrics (Sharpe 0.83, MaxDD -4.9%) match visual reality, or are there hidden issues?"*

**Human Cognition**:
- Metrics are abstract numbers
- Visual equity curves reveal:
  - Smooth growth vs choppy (consistency)
  - Drawdown frequency (is -4.9% worst, or common?)
  - Recovery patterns (V-shape vs U-shape)
  - Regime changes (2022 bull vs 2024 bear)

**Trust but Verify**:
Numbers can lie (calculation errors, data issues)
Charts don't lie (if equity curve shows -20% drop, MaxDD metric is wrong)

### **Backend Logic**

**1. Cumulative Equity Curve**
```
Starting_Capital = ₹100,000

For each trading day D:
    Daily_Return = Portfolio_Return(D)  # From equal-weight or optimized
    Equity(D) = Equity(D-1) × (1 + Daily_Return)

Example:
Day 1: Start ₹100,000, Return +0.5% → Equity = ₹100,500
Day 2: Start ₹100,500, Return -0.2% → Equity = ₹100,299
...
Day 905: Equity = ₹112,746 → Total Return = +12.75%
```

**2. Drawdown Series**
```
For each day:
    Running_Max = max(Equity[0:D])
    Drawdown(D) = (Equity(D) - Running_Max) / Running_Max × 100

Identify drawdown periods:
- Start: When Equity < Running_Max
- End: When Equity = new Running_Max (recovery)
- Duration: Days from start to end
- Depth: Min(Drawdown) during period
```

**3. Rolling Sharpe Ratio**
```
For window W = 63 days (~3 months):

For day D:
    Rolling_Returns = Daily_Returns[D-62:D]
    Rolling_Sharpe(D) = Mean(Rolling_Returns) / StdDev(Rolling_Returns) × √252

Plot Rolling_Sharpe over time:
- Shows consistency (flat line = stable edge)
- Shows regime changes (sharp drops = strategy breakdown)
- Shows recovery (return to baseline after drawdown)
```

**4. Monthly Returns Heatmap**
```
Aggregate returns by month:

2022:
Jan: +1.2%, Feb: -0.3%, Mar: +0.8%, ..., Dec: +1.1%

2023:
Jan: +0.7%, ..., Dec: +0.9%

Heatmap:
Green = Positive months
Red = Negative months
Color intensity = Magnitude

Insight: Seasonality (e.g., Jan effect, Dec rally)
```

**5. Sector Allocation Pie Chart**
```
For portfolio [AXISBANK, HCLTECH, INFY, SUNPHARMA, KOTAKBANK]:

Sector counts:
Banking: 2 tickers (40%)
IT: 2 tickers (40%)
Pharma: 1 ticker (20%)

Pie chart:
Visual check for diversification
```

### **Base Assumptions**

| Assumption | Validation | Risk |
|------------|------------|------|
| Continuous compounding | Equity curve compounds daily returns | Matches backtest reality |
| No withdrawals/deposits | Portfolio is closed (no cash flows) | Real trading has deposits |
| No transaction costs in visual | Charts show gross returns | Net returns lower (but shape same) |
| Daily data resolution | Higher resolution (5-min) available | May miss intraday volatility |

### **Key Metrics**

**Visual Validation Checks**:
```
1. Equity Curve Slope:
   Upward slope → Positive expectancy ✅
   Flat/downward → Strategy failure ❌

2. Drawdown Magnitude:
   Visual max DD matches calculated MaxDD ✅
   If visual worse → Calculation error ❌

3. Recovery Pattern:
   V-shape (fast recovery) → Resilient strategy ✅
   U-shape (slow recovery) → Prolonged pain ⚠️

4. Consistency:
   Steady growth → Stable edge ✅
   Erratic jumps → Regime-dependent ⚠️
```

**Comparison Charts**:
```
Plot all Top 5 portfolios on same chart:
- Portfolio #1 (Sharpe 0.84): Blue line
- Portfolio #2 (Sharpe 0.83): Orange line
- ...

Observation:
- Lines should be close (all high quality)
- Divergence → Different risk profiles
- Correlation → Lack of diversification across portfolios
```

### **Decision Criteria**

**Visual Sign-Off**:
```
Approve for production IF:
✅ Equity curve shows consistent upward trend
✅ Drawdowns match calculated metrics (±0.5%)
✅ Rolling Sharpe mostly positive (>80% of time)
✅ No unexplained jumps (data errors)
✅ Monthly heatmap shows >60% positive months

Reject IF:
❌ Equity curve is flat or declining
❌ Visual DD > calculated DD (+2% difference)
❌ Rolling Sharpe erratic (flips between -1 and +2)
❌ Suspicious spikes (likely data error)
```

---

# 🔬 **KEY ASSUMPTIONS & LIMITATIONS**

## **Universal Assumptions Across All Analyses**

### **1. Historical Performance ≠ Future Performance**
**Assumption**: Past trades are representative of future edge.

**Reality Check**:
- Market regimes change (2022 bull ≠ 2024 bear)
- Strategy may have worked due to lucky period
- Competitors may copy strategy (alpha decay)

**Mitigation**:
- Walk-forward validation (train on 2022-2023, test on 2024-2025)
- Out-of-sample testing (reserve 20% of data)
- Live paper trading (validate in real-time without capital)

---

### **2. No Look-Ahead Bias**
**Assumption**: Analysis uses only information available at trade entry.

**Critical Check**:
- Exit price known AFTER trade → Can't use for entry decision
- Daily correlation calculated from historical data → Can use for portfolio construction
- Future volatility unknown → Must estimate from past

**Validation**:
- Timestamp audit (Entry Time < Exit Time always)
- Indicator lag verification (5min MACD uses data up to Entry Time only)

---

### **3. Transaction Costs Are Negligible**
**Assumption**: Backtest P&L includes realistic costs (brokerage, STT, slippage).

**Real-World Costs** (NSE India example):
- Brokerage: ₹20 per order (flat) or 0.03% (percentage)
- STT (Securities Transaction Tax): 0.025% on sell side
- Exchange charges: ~0.003%
- GST on brokerage: 18%
- **Total**: ~0.05-0.10% per trade

**Impact**:
Strategy with 1000 trades/year:
Gross P&L: +12%
Transaction costs: -5% (0.05% × 1000 × 2 sides)
Net P&L: +7% (41% reduction!)

**Mitigation**:
Include transaction costs in backtest P&L calculation.

---

### **4. Liquidity Is Sufficient**
**Assumption**: Can execute all trades at stated prices (no slippage).

**Reality**:
- Low-volume tickers: Bid-ask spread 0.5-2% (erodes profits)
- Large positions: Market impact (moving price against you)
- Fast markets: Prices gap past stop loss

**Validation**:
Check average volume in base_data:
Min_Volume_Threshold = ₹10 lakh per 5-min candle

If volume < threshold → Execution risk (avoid ticker)

---

### **5. Stationarity of Statistical Properties**
**Assumption**: Mean, variance, correlation are stable over time.

**Reality**:
- Volatility clusters (VIX spikes during crashes)
- Correlation → 1.0 during systemic crises
- Mean-reversion breaks during regime shifts

**Test**:
Augmented Dickey-Fuller test for stationarity
If p-value > 0.05 → Non-stationary (strategy may fail)

---

### **6. Independence of Observations**
**Assumption**: Each trade is independent (no autocorrelation).

**Reality**:
- Cascade trades are dependent (covered in Module 03)
- Intraday trades share microstructure
- Market-wide moves affect all tickers

**Test**:
Ljung-Box Q-test for autocorrelation
If p-value < 0.05 → Significant autocorrelation (violates assumption)

---

### **7. Normal Distribution of Returns**
**Assumption**: Returns follow Gaussian distribution (bell curve).

**Reality**:
- Fat tails (extreme events more common than normal dist predicts)
- Skewness (asymmetric: losses ≠ wins)
- Kurtosis (leptokurtic: peaked center, heavy tails)

**Impact**:
Sharpe ratio underestimates tail risk
Max Drawdown may exceed calculated bounds

**Alternative Metrics**:
- Sortino Ratio (downside risk only)
- CVaR (Conditional Value at Risk) at 95%
- Omega Ratio (probability-weighted gains/losses)

---

### **8. No Regime Shifts**
**Assumption**: Market structure is stable (same rules apply throughout).

**Reality**:
- 2020: Circuit breakers triggered (trading halts)
- 2023: SEBI algo trading rules changed
- Bull market (2022-2023) ≠ Bear market (2024)

**Mitigation**:
- Test strategy across multiple regimes
- Implement regime detection (VIX threshold)
- Dynamic parameter adjustment

---

## **Portfolio Construction Specific Assumptions**

### **9. Diversification Benefit Is Achievable**
**Assumption**: σ_portfolio < Average(σ_individual) due to correlation < 1.0.

**Reality**:
During crises: ρ → 1.0 (all assets move together)
Example: March 2020 COVID crash (all stocks down 30-40%)

**Implication**:
Diversification works in normal times, fails in tail events.

**Mitigation**:
- Include uncorrelated assets (gold, bonds)
- Hedge with options during high VIX
- Cash allocation (dry powder)

---

### **10. Optimal Weights Are Stable**
**Assumption**: PyPortfolioOpt weights persist (no frequent rebalancing).

**Reality**:
- Weights drift as prices change
- Expected returns change (ticker edge fades)
- Covariance changes (correlation breakdown)

**Practical Approach**:
Rebalance monthly or quarterly (balance stability vs optimality)

---

### **11. Equal Information Across Tickers**
**Assumption**: All tickers have equal data quality and sample size.

**Reality**:
- Some tickers have 2000 trades (high confidence)
- Others have 300 trades (low confidence)

**Weighting**:
Use confidence intervals:
Sharpe ± (1.96 × SE)
Where SE = Sharpe / √n

High-sample tickers: Narrow CI (reliable)
Low-sample tickers: Wide CI (uncertain)

---

# 📐 **STATISTICAL FOUNDATIONS**

## **Core Statistical Concepts**

### **1. Sharpe Ratio**
```
Sharpe = (E[R] - R_f) / σ_R

Where:
E[R] = Expected return
R_f = Risk-free rate (typically 6-7% for India)
σ_R = Standard deviation of returns

Annualization:
If daily returns:
Sharpe_annual = Sharpe_daily × √252

Interpretation:
Sharpe > 2.0: Excellent
Sharpe 1.0-2.0: Good
Sharpe 0.5-1.0: Acceptable
Sharpe < 0.5: Poor
```

### **2. Profit Factor**
```
PF = Σ(Winning Trades P&L) / |Σ(Losing Trades P&L)|

Interpretation:
PF = 1.5: For every ₹1 lost, make ₹1.50 (₹0.50 net)
PF = 1.0: Break-even
PF < 1.0: Losing system

Relationship to Win Rate:
PF = (WR × Avg_Win) / ((1-WR) × Avg_Loss)
```

### **3. Maximum Drawdown**
```
DD(t) = (Equity(t) - Peak_Equity) / Peak_Equity

Max_DD = min(DD(t)) for all t

Example:
Peak: ₹120,000
Trough: ₹108,000
Max DD = (108,000 - 120,000) / 120,000 = -10%

Interpretation:
Worst-case loss from peak to trough
```

### **4. Correlation (Pearson)**
```
ρ(X, Y) = Cov(X, Y) / (σ_X × σ_Y)

Range: -1.0 to +1.0

ρ = +1.0: Perfect positive (move together)
ρ = 0.0: No linear relationship
ρ = -1.0: Perfect negative (opposite moves)

Interpretation:
|ρ| < 0.3: Weak correlation
|ρ| 0.3-0.7: Moderate correlation
|ρ| > 0.7: Strong correlation
```

### **5. Z-Score Normalization**
```
Z = (X - μ) / σ

Where:
μ = Mean of population
σ = Standard deviation

Interpretation:
Z = 0: Average (50th percentile)
Z = +1: 1 std dev above (84th percentile)
Z = +2: 2 std dev above (97.5th percentile)
Z = -1: 1 std dev below (16th percentile)
```

### **6. Statistical Significance (t-test)**
```
t = (X̄_1 - X̄_2) / SE

Where:
SE = √[(s²_1/n_1) + (s²_2/n_2)]

Degrees of freedom: df = n_1 + n_2 - 2

p-value from t-distribution:
p < 0.05: Statistically significant (reject H0)
p > 0.05: Not significant (cannot reject H0)
```

---

## **Production Readiness Checklist**

### **Before Live Deployment**

**Data Validation**:
- ✅ Data Quality Score > 85
- ✅ No missing critical fields
- ✅ Timestamp sequence validated
- ✅ P&L reconciliation passed

**Strategy Validation**:
- ✅ Profit Factor > 1.2
- ✅ Sharpe Ratio > 1.0
- ✅ Max Drawdown < 15%
- ✅ Win Rate: 45-55% OR high RRR if <45%

**Portfolio Validation**:
- ✅ Top 50 tickers identified
- ✅ Anti-cascade filtering applied (if beneficial)
- ✅ Sector diversification enforced
- ✅ Correlation constraints applied
- ✅ Optimal weights calculated (if improvement >10%)

**Visual Validation**:
- ✅ Equity curve shows upward trend
- ✅ Drawdowns match calculated metrics
- ✅ Rolling Sharpe mostly positive
- ✅ No unexplained anomalies

**Out-of-Sample Testing**:
- ✅ Walk-forward validation passed (degradation <15%)
- ✅ Performance holds on unseen data
- ✅ No overfitting detected

**Risk Management**:
- ✅ Stop loss rules defined
- ✅ Position sizing rules documented
- ✅ Maximum portfolio drawdown limit set
- ✅ Circuit breaker conditions defined

---

## **Continuous Improvement Cycle**

```
1. Backtest → Generate trades
2. Generic Analysis → Understand quality
3. Portfolio Construction → Optimize combinations
4. Visual Validation → Trust but verify
5. Paper Trading → Live validation (no capital)
6. Performance Monitoring → Track vs backtest
7. Feedback Loop → Identify degradation
8. Strategy Refinement → Optimize parameters
9. Re-backtest → Validate improvements
10. Repeat
```

---

**Document Version**: 1.0
**Last Updated**: October 8, 2025
**Status**: Phase 1 Complete (Portfolio Construction)
**Next**: Phase 2 (MSE Strategy Optimization)

---

**Questions or Clarifications?**
Review `analysis/WORKFLOW_SOP.md` for step-by-step execution guide.
Review `analysis/config_template.yaml` for configuration reference.
