#!/usr/bin/env python3
"""
Nifty 100 Ticker Analyzer
Identifies missing tickers from Nifty 100 index for backtesting expansion
"""

# Complete Nifty 100 ticker list (as of October 2025)
NIFTY_100_TICKERS = [
    "ABB", "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT",
    "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BANKBARODA",
    "BEL", "BPCL", "BHARTIARTL", "BOSCHLTD", "BRITANNIA",
    "CANFINHOME", "CANBK", "CHOLAFIN", "CIPLA", "COALINDIA",
    "DABUR", "DIVISLAB", "DLF", "DRREDDY", "EICHERMOT",
    "GAIL", "GODREJCP", "GRASIM", "HAVELLS", "HCLTECH",
    "HDFCBANK", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "ICICIGI", "ICICIPRULI", "INDUSINDBK", "INFY", "IOC",
    "ITC", "JSWENERGY", "JSWSTEEL", "JINDALSTEL", "KOTAKBANK",
    "LT", "LTIM", "LICHSGFIN", "M&M", "MARUTI",
    "MOTHERSON", "NTPC", "NHPC", "ONGC", "PIDILITE",
    "PFC", "POWERGRID", "PNB", "RELIANCE", "SBILIFE",
    "SBIN", "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SUNPHARMA",
    "TCS", "TATACONSUM", "TATAMOTORS", "TATAPOWER", "TATASTEEL",
    "TECHM", "TITAN", "TORNTPHARM", "TRENT", "TVSMOTOR",
    "ULTRACEMCO", "UNIONBANK", "UPL", "VEDL", "VOLTAS",
    "WIPRO", "ZYDUSLIFE", "NESTLEIND", "ADANIGREEN", "AADHARBF",
    "ADANITRANS", "IRCTC", "IRFC", "HAL", "LTI",
    "VBL", "DMART", "INDIGO", "JINDAL", "RECLTD",
    "ABFRL", "APOLLOTYRE", "BANDHANBNK", "COLPAL", "DLF",
    "GAIL", "INDHOTEL", "MARICO", "MFSL", "PEL"
]

# Your existing 20 tickers
EXISTING_TICKERS = [
    "ABFRL", "ADANIGREEN", "AXISBANK", "BRITANNIA", "CIPLA",
    "DELHIVERY", "EICHERMOT", "FEDERALBNK", "GLAXO", "HERCULES",
    "INFY", "NDRAUTO", "NMDC", "POCL", "PSUBANK",
    "RELIANCE", "TCS", "TECHM", "UBL", "VIMTALABS"
]

def analyze_tickers():
    """Analyze and identify missing Nifty 100 tickers"""
    
    # Normalize existing tickers
    existing_normalized = set(t.upper() for t in EXISTING_TICKERS)
    nifty100_normalized = set(t.upper() for t in NIFTY_100_TICKERS)
    
    # Find matching tickers
    matching = existing_normalized.intersection(nifty100_normalized)
    
    # Find missing tickers
    missing = nifty100_normalized - existing_normalized
    
    # Find tickers in existing but not in Nifty 100
    extra = existing_normalized - nifty100_normalized
    
    print("=" * 80)
    print("NIFTY 100 TICKER ANALYSIS")
    print("=" * 80)
    print(f"\n📊 Total Nifty 100 Tickers: {len(nifty100_normalized)}")
    print(f"✅ Existing Tickers: {len(existing_normalized)}")
    print(f"🔗 Matching with Nifty 100: {len(matching)}")
    print(f"🆕 Missing from Nifty 100: {len(missing)}")
    print(f"⚠️  Extra (not in Nifty 100): {len(extra)}")
    
    print(f"\n\n{'=' * 80}")
    print(f"MATCHING TICKERS ({len(matching)})")
    print("=" * 80)
    print(", ".join(sorted(matching)))
    
    print(f"\n\n{'=' * 80}")
    print(f"MISSING TICKERS TO FETCH ({len(missing)})")
    print("=" * 80)
    missing_list = sorted(missing)
    
    # Print in groups of 10 for readability
    for i in range(0, len(missing_list), 10):
        print(" ".join(missing_list[i:i+10]))
    
    print(f"\n\n{'=' * 80}")
    print(f"EXTRA TICKERS (Not in Nifty 100) ({len(extra)})")
    print("=" * 80)
    print(", ".join(sorted(extra)))
    
    # Generate CLI command for missing tickers
    print(f"\n\n{'=' * 80}")
    print("MISSING TICKERS - SINGLE LINE (for CLI)")
    print("=" * 80)
    print(" ".join(sorted(missing)))
    
    return {
        'total_nifty100': len(nifty100_normalized),
        'existing': len(existing_normalized),
        'matching': len(matching),
        'missing': len(missing),
        'missing_list': sorted(missing),
        'extra': len(extra),
        'extra_list': sorted(extra)
    }


if __name__ == "__main__":
    results = analyze_tickers()
    
    print(f"\n\n{'=' * 80}")
    print("SUMMARY FOR SCALING")
    print("=" * 80)
    print(f"Current Coverage: {results['matching']}/{results['total_nifty100']} tickers ({results['matching']/results['total_nifty100']*100:.1f}%)")
    print(f"Tickers to Add: {results['missing']}")
    print(f"Final Coverage: {results['matching'] + results['missing']}/{results['total_nifty100']} tickers (100%)")
