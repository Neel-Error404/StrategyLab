# MSE Strategy Implementation Assumptions
## Audit Trail and Design Decisions

**Strategy:** MSE (Mean Squared Error)  
**Version:** 2.0 - Audit Compliant  
**Date:** 2025-09-10  
**Author:** Senior Data Architect Engineer  

---

## **CORE STRATEGY ASSUMPTIONS**

### **1. Signal Generation Logic**
- **Entry Conditions (BUY):** All conditions must be TRUE simultaneously
  - 5-minute MACD line > 5-minute Signal line (previous bar)
  - 5-minute EMA(9) > 5-minute EMA(20) (previous bar)
  - 15-minute MACD line > 15-minute Signal line (previous bar)  
  - 15-minute EMA(9) > 15-minute EMA(20) (previous bar)

- **Entry Conditions (SELL):** All conditions must be TRUE simultaneously
  - 5-minute MACD line < 5-minute Signal line (previous bar)
  - 5-minute EMA(9) < 5-minute EMA(20) (previous bar)
  - 15-minute MACD line < 15-minute Signal line (previous bar)
  - 15-minute EMA(9) < 15-minute EMA(20) (previous bar)

### **2. Exit Logic (20% MACD Histogram Rule)**
- **BUY Exit:** When current 15m MACD histogram < 20% of maximum histogram since entry
- **SELL Exit:** When current 15m MACD histogram > 20% of minimum histogram since entry
- **Assumption:** This captures momentum decay while avoiding premature exits

### **3. Timing and Execution Rules (AUDIT COMPLIANT)**
- **Signal Detection:** Uses PREVIOUS bar indicators (shift=1) to avoid lookahead bias
- **Two-Candle Rule:** Signal generated on bar N, execution at bar N+1 open price
- **Warmup Period:** 525 minutes (35×15min bars) minimum before any signals
- **Data Lag:** All indicator calculations use completed bars only

### **4. Position Management Rules**
- **Single Position Rule:** Maximum one position (long OR short) per ticker at any time
- **No Overlapping:** Must exit current position before entering opposite position
- **Position State:** Tracked across all data processing cycles
- **Size Management:** Handled by external risk management system

### **5. Timeframe Hierarchy**
- **Primary Timeframes:** 5-minute and 15-minute
- **Data Source:** Native timeframe data (not resampled from 1-minute)
- **Synchronization:** 15-minute indicators take precedence for trend direction
- **Alignment:** All timeframes must agree for entry signals

### **6. Technical Indicator Parameters**
- **MACD:** 12, 26, 9 (fast EMA, slow EMA, signal EMA)
- **EMA Short:** 9 periods
- **EMA Long:** 20 periods
- **Calculation:** Uses close prices, exponential weighting
- **Rounding:** All values rounded to 2 decimal places for consistency

### **7. Data Quality Assumptions**
- **Market Hours:** Indian market hours (9:15 AM - 3:30 PM IST)
- **Gaps:** No major data gaps during calculation periods
- **Outliers:** Handled by upstream data quality filters
- **Timestamp:** All timestamps represent bar close times

### **8. Risk and Compliance Assumptions**
- **Regulatory:** Compliant with Indian market regulations
- **Audit Trail:** All signals timestamped and logged
- **Reproducibility:** Same inputs always produce same outputs
- **Version Control:** Strategy logic locked for production periods

---

## **LIVE vs BACKTEST PARITY REQUIREMENTS**

### **Identical Processing**
- Same indicator calculations
- Same signal generation logic  
- Same execution timing rules
- Same position management rules

### **Data Consistency**
- Same timeframe data sources
- Same data quality filters
- Same timestamp handling
- Same market hours definition

### **Execution Consistency**  
- Same entry/exit price logic (next bar open)
- Same position sizing rules
- Same risk management integration
- Same logging and audit trail

---

## **KNOWN LIMITATIONS AND EDGE CASES**

### **Market Condition Dependencies**
- **Trending Markets:** Strategy performs better in trending conditions
- **Sideways Markets:** May generate false signals in ranging markets
- **High Volatility:** 20% exit rule may trigger prematurely in volatile conditions

### **Technical Limitations**
- **Data Latency:** Requires real-time data feed for live trading
- **Processing Lag:** Indicator calculations add processing time
- **Memory Usage:** Maintains position state and indicator history

### **Operational Constraints**
- **Market Hours:** Only operates during market hours
- **Holidays:** No trading on market holidays
- **Maintenance:** Strategy paused during system maintenance

---

## **VALIDATION AND TESTING REQUIREMENTS**

### **Unit Tests**
- Individual indicator calculations
- Signal generation logic
- Position state management
- Edge case handling

### **Integration Tests**
- End-to-end signal processing
- Live/backtest output comparison
- Data pipeline integration
- Risk management integration

### **Performance Tests**
- Processing speed benchmarks
- Memory usage monitoring
- Scalability limits
- Concurrent ticker handling

---

## **CHANGE CONTROL**
- Any modification to core assumptions requires re-validation
- All changes must maintain live/backtest parity
- Version control required for all strategy updates
- Audit trail must be preserved for regulatory compliance

**Last Updated:** 2025-09-10  
**Next Review:** Monthly or upon significant market regime change