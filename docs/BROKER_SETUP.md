# 🔑 Broker Setup Guide

**Complete guide to setting up API keys and live data access. Use the AI assistant for interactive help!**

## 📋 **Overview**

Connect your broker account for:
- **Live market data** (real-time prices)
- **Paper trading** (test strategies safely)
- **Live trading** (execute real trades)

**Supported Brokers**: Zerodha Kite, Upstox, Binance (Crypto)

---

## 🏦 **Zerodha Kite API Setup**

### **Step 1: Get API Credentials**
1. Login to [Kite Connect](https://kite.trade/)
2. Go to "My Apps" → "Create New App"
3. Fill details:
   - **App Name**: "Backtester"
   - **App Type**: "Connect"
   - **Redirect URL**: `http://127.0.0.1:5000/`
4. Note down:
   - **API Key** (api_key)
   - **API Secret** (api_secret)

### **Step 2: Generate Access Token**
```bash
# Set environment variables
export ZERODHA_API_KEY="your_api_key"
export ZERODHA_API_SECRET="your_api_secret"

# Generate token (interactive)
python src/core/etl/data_provider/zerodha_provider.py
```

### **Step 3: Test Connection**
```bash
# Test API connection
python src/runners/unified_runner.py --mode validate --date-ranges 2024-12-12_to_2025-06-09 --tickers RELIANCE
```

---

## 📈 **Upstox API Setup**

### **Step 1: Get API Credentials**
1. Login to [Upstox Developer Console](https://upstox.com/developer/)
2. Create new app:
   - **App Name**: "Backtester"
   - **Redirect URI**: `https://127.0.0.1:5000/`
3. Note down:
   - **API Key**
   - **API Secret**

### **Step 2: Configure**
```bash
# Set environment variables
export UPSTOX_API_KEY="your_api_key"
export UPSTOX_API_SECRET="your_api_secret"
export UPSTOX_REDIRECT_URI="https://127.0.0.1:5000/"
```

---

## 🪙 **Binance (Cryptocurrency)**

### **Overview**
Binance integration provides access to **35+ cryptocurrencies** for backtesting without requiring API keys for historical data.

**Supported Assets**: BTC, ETH, XRP, BNB, SOL, DOGE, ADA, TRX, AVAX, SHIB, UNI, LINK, AAVE, and more

**Key Features**:
- ✅ No API key required for historical data
- ✅ 24/7 trading support (crypto markets never close)
- ✅ Multiple timeframes: 1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M
- ✅ 5+ years of historical data available
- ✅ Direct API access (no authentication needed)

### **Step 1: Fetch Crypto Data**

**Short Symbol Format** (auto-converts to USDT pairs):
```bash
# Fetch Bitcoin data (BTC → BTCUSDT)
python src/runners/unified_runner.py --mode fetch --tickers BTC ETH --timeframes 1m,1h --date-ranges 2024-01-01_to_2024-12-31

# Fetch multiple cryptos (last 30 days)
python src/runners/unified_runner.py --mode fetch --tickers BTC ETH XRP --timeframes 5m --days 30
```

**Full Symbol Format** (explicit USDT pairs):
```bash
# Fetch with full symbol names
python src/runners/unified_runner.py --mode fetch --tickers BTCUSDT ETHUSDT SOLUSDT --timeframes 1h --days 90
```

### **Step 2: Backtest Crypto Strategies**

```bash
# Backtest Bitcoin with conservative template
python src/runners/unified_runner.py --mode backtest --template conservative --date-ranges 2024-01-01_to_2024-12-31 --tickers BTCUSDT

# Multi-crypto portfolio backtest
python src/runners/unified_runner.py --mode backtest --template aggressive --date-ranges 2024-Q1 --tickers BTC ETH BNB SOL

# 24/7 trading analysis (no market hour filters)
python src/runners/unified_runner.py --mode analyze --date-ranges 2024-01-01_to_2024-06-30 --tickers BTCUSDT ETHUSDT
```

### **Step 3: Rate Limits & Best Practices**

**Rate Limiting**:
- Binance allows 6,000 weight units per minute
- System uses 0.1s delay between requests (safe default)
- Large date ranges automatically chunked (6+ months → monthly chunks)

**Data Management**:
```bash
# Check existing crypto data
ls data/pools/*/1minute/BTC*.parquet

# Incremental updates (only fetch new data)
python src/runners/unified_runner.py --mode update --pool-path data/pools/2024-01-01_to_2024-12-31 --extend-to 2025-01-31
```

### **Supported Cryptocurrencies**

| Symbol | Name | Market Cap Rank | Notes |
|--------|------|-----------------|-------|
| **BTC** | Bitcoin | #1 | Most liquid, 5+ years data |
| **ETH** | Ethereum | #2 | Smart contract leader |
| **BNB** | Binance Coin | #4 | Exchange token |
| **SOL** | Solana | #5 | High-performance chain |
| **XRP** | Ripple | #6 | Cross-border payments |
| **DOGE** | Dogecoin | #8 | Meme coin, high volatility |
| **ADA** | Cardano | #9 | Academic blockchain |
| **AVAX** | Avalanche | #10 | DeFi platform |
| **SHIB** | Shiba Inu | #11 | Meme token |
| **TRX** | Tron | #12 | Entertainment blockchain |

*Full list: 35+ major cryptocurrencies supported (see `src/core/etl/data_provider/binance_provider.py`)*

### **Crypto vs Equity Differences**

| Feature | Equities | Crypto |
|---------|----------|--------|
| **Trading Hours** | 9:15-15:30 IST (Mon-Fri) | 24/7/365 |
| **Weekends** | Closed | Open |
| **Timeframes** | 1m, 5m, 15m, 1h, 1d | 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M |
| **Volatility** | Moderate (2-5%) | High (5-20%+) |
| **Liquidity** | Business hours peak | Global, continuous |
| **Settlement** | T+2 | Instant |

### **Configuration (Optional)**

**No configuration needed for backtesting!** For live trading or advanced features:

```python
# config/config.py (optional)
BROKER_CONFIG = {
    'binance': {
        'enabled': True,
        'api_key': None,  # Not required for historical data
        'api_secret': None,  # Not required for historical data
        'testnet': False,
        'timeout': 30
    }
}
```

### **Troubleshooting**

**"Symbol not found" error**:
```bash
# Use correct format
✅ Correct: BTC or BTCUSDT
❌ Wrong: Bitcoin, BTC-USD, BTC/USDT
```

**"Rate limit exceeded"**:
- System automatically handles rate limiting
- If issues persist, increase delay in `binance_provider.py` (line ~150)

**"No data returned"**:
- Verify ticker symbol exists on Binance Spot market
- Check date range (Binance data starts ~2017 for BTC)
- Some tokens have limited history

### **Example: Complete Crypto Backtest Workflow**

```bash
# Step 1: Fetch 1 year of BTC + ETH data
python src/runners/unified_runner.py --mode fetch --tickers BTC ETH --timeframes 1m,1h --date-ranges 2024-01-01_to_2024-12-31

# Step 2: Run backtest with conservative template
python src/runners/unified_runner.py --mode backtest --template conservative --date-ranges 2024-01-01_to_2024-12-31 --tickers BTCUSDT ETHUSDT

# Step 3: Analyze results
python src/runners/unified_runner.py --mode analyze --date-ranges 2024-01-01_to_2024-12-31 --tickers BTCUSDT ETHUSDT

# Step 4: Generate visualizations
python src/runners/unified_runner.py --mode visualize --date-ranges 2024-01-01_to_2024-12-31 --tickers BTCUSDT ETHUSDT
```

**Expected Output**:
```
📊 Bitcoin (BTCUSDT)
   - Total Trades: 245
   - Win Rate: 52.3%
   - Sharpe Ratio: 1.45
   - Max Drawdown: -18.2%

📊 Ethereum (ETHUSDT)
   - Total Trades: 312
   - Win Rate: 48.7%
   - Sharpe Ratio: 1.22
   - Max Drawdown: -22.5%
```

### **Pro Tips**

1. **Start with longer timeframes** (1h, 4h) - crypto is very noisy on 1-minute
2. **Use smaller position sizes** - crypto volatility is 3-5x higher than equities
3. **Test across bull and bear markets** - crypto correlation to BTC dominance
4. **Weekend data included** - crypto trades 24/7, unlike equities
5. **Consider stablecoins** - USDT, USDC for low-volatility pairs

---

## ⚙️ **Configuration**

### **Environment Variables**
Create `.env` file in project root:
```bash
# Zerodha
ZERODHA_API_KEY=your_zerodha_api_key
ZERODHA_API_SECRET=your_zerodha_api_secret
ZERODHA_REDIRECT_URI=http://127.0.0.1:5000/

# Upstox  
UPSTOX_API_KEY=your_upstox_api_key
UPSTOX_API_SECRET=your_upstox_api_secret
UPSTOX_REDIRECT_URI=https://127.0.0.1:5000/
```

### **Access Token Storage**
Tokens are automatically saved to:
```
config/access_tokens/
├── zerodha/
│   └── access_token_YYYYMMDD.json
└── upstox/
    └── access_token_YYYYMMDD.json
```

---

## � **Security Best Practices**

### **Environment Variables**
✅ **DO**: Use environment variables for API keys  
✅ **DO**: Add `.env` to `.gitignore`  
❌ **DON'T**: Hardcode API keys in scripts  
❌ **DON'T**: Commit API keys to git  

### **Access Tokens**
- Tokens expire daily - system auto-refreshes
- Tokens are encrypted and stored locally
- Never share access token files

---

## 🔧 **Troubleshooting**

### **Common Issues**

**"Missing API credentials"**
- Error: `❌ Missing Zerodha API credentials!` or `❌ UPSTOX_CLIENT_ID environment variable is not set!`
- **Solution**: Set required environment variables:
  ```bash
  # For Zerodha
  export ZERODHA_API_KEY="your_api_key"
  export ZERODHA_API_SECRET="your_api_secret"
  
  # For Upstox
  export UPSTOX_CLIENT_ID="your_client_id"
  export UPSTOX_CLIENT_SECRET="your_client_secret"
  ```

**"Invalid API key"**
- Check API key is correct
- Verify environment variables are set
- Ensure Kite Connect app is active

**"Token expired"**
- Re-run token generation
- Check system date/time is correct

**"Connection failed"**
- Check internet connection
- Verify redirect URI matches exactly
- Try different network if behind firewall

### **Debug Mode**
```bash
# Enable debug logging
python src/runners/unified_runner.py --mode validate --log-level DEBUG --tickers RELIANCE
```

---

## 💡 **Pro Tips**

1. **Start with Paper Trading**: Test strategies before live trading
2. **Use Multiple Brokers**: Backup in case one has issues
3. **Monitor Token Expiry**: System handles auto-refresh
4. **Ask AI**: Use the AI assistant in README.md for specific setup help

---

**✨ Need Help?** The AI assistant can guide you through broker-specific setup issues!
|--------|-----------|---------------|---------|--------|
| **Zerodha** | ✅ | ✅ | ✅ | Production Ready |
| **Upstox** | ✅ | ✅ | ✅ | Production Ready |
| **Angel Broking** | 🔄 | 🔄 | 🔄 | Coming Soon |
| **IIFL** | 🔄 | 🔄 | 🔄 | Coming Soon |

---

## 🎯 Zerodha Kite API Setup

### Step 1: Create API App

1. **Log into Kite Connect**
   - Visit: https://kite.zerodha.com/
   - Go to **Console** → **Apps**

2. **Create New App**
   ```
   App Name: My Backtester App
   App Type: Connect
   Redirect URL: http://localhost:8080
   Description: Backtesting and trading application
   ```

3. **Get Credentials**
   After app creation, you'll receive:
   - **API Key** (Consumer Key)
   - **API Secret** (Consumer Secret)

### Step 2: Generate Access Token

```python
# Run this script to generate access token
from kiteconnect import KiteConnect

# Your credentials
api_key = "your_api_key_here"
api_secret = "your_api_secret_here"

# Initialize KiteConnect
kite = KiteConnect(api_key=api_key)

# Get login URL
login_url = kite.login_url()
print(f"Login URL: {login_url}")

# After login, you'll get request_token from callback URL
# Extract request_token from URL like: http://localhost:8080?request_token=XXXXXX&action=login&status=success
request_token = "paste_request_token_here"

# Generate session
data = kite.generate_session(request_token, api_secret=api_secret)
access_token = data["access_token"]

print(f"Access Token: {access_token}")
```

### Step 3: Save Configuration

Create file: `config/access_tokens/zerodha/zerodha_config.json`

```json
{
  "api_key": "your_api_key_here",
  "api_secret": "your_api_secret_here",
  "access_token": "your_access_token_here",
  "user_id": "your_user_id",
  "enctoken": "optional_enctoken",
  "created_date": "2025-06-09",
  "expiry_date": "2025-06-10",
  "broker": "zerodha",
  "environment": "live"
}
```

---

## 📊 Upstox API Setup

### Step 1: Developer Registration

1. **Register as Developer**
   - Visit: https://developer.upstox.com/
   - Sign up with your Upstox account
   - Complete developer verification

2. **Create App**
   ```
   App Name: Backtester Framework
   App Type: Web App
   Redirect URI: http://localhost:8080/callback
   Website: http://localhost:8080
   Description: Trading and backtesting application
   ```

3. **Get Credentials**
   - **Client ID** (API Key)
   - **Client Secret**

### Step 2: Generate Access Token

```python
# Upstox token generation
import requests
from urllib.parse import urlencode

# Your credentials
client_id = "your_client_id"
client_secret = "your_client_secret"
redirect_uri = "http://localhost:8080/callback"

# Step 1: Get authorization URL
auth_url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}"
print(f"Visit this URL: {auth_url}")

# Step 2: After authorization, extract code from callback URL
authorization_code = "paste_code_from_callback_url"

# Step 3: Exchange code for access token
token_url = "https://api.upstox.com/v2/login/authorization/token"
token_data = {
    "code": authorization_code,
    "client_id": client_id,
    "client_secret": client_secret,
    "redirect_uri": redirect_uri,
    "grant_type": "authorization_code"
}

response = requests.post(token_url, data=token_data)
token_info = response.json()

access_token = token_info["access_token"]
print(f"Access Token: {access_token}")
```

### Step 3: Save Configuration

Create file: `config/access_tokens/upstox/upstox_config.json`

```json
{
  "client_id": "your_client_id",
  "client_secret": "your_client_secret",
  "access_token": "your_access_token_here",
  "refresh_token": "your_refresh_token",
  "token_type": "Bearer",
  "expires_in": 86400,
  "created_date": "2025-06-09",
  "broker": "upstox",
  "environment": "live"
}
```

---

## ⚙️ Configuration Setup

### Framework Configuration

Update `config/config.py` with your broker settings:

```python
# Broker Configuration
BROKER_CONFIG = {
    'default_broker': 'zerodha',  # or 'upstox'
    'zerodha': {
        'enabled': True,
        'config_path': 'config/access_tokens/zerodha/zerodha_config.json',
        'paper_trading': False,
        'timeout': 30
    },
    'upstox': {
        'enabled': True,
        'config_path': 'config/access_tokens/upstox/upstox_config.json',
        'paper_trading': False,
        'timeout': 30
    }
}
```

### Environment Variables (Optional)

For added security, use environment variables:

```bash
# Windows PowerShell
$env:ZERODHA_API_KEY="your_api_key"
$env:ZERODHA_API_SECRET="your_secret"
$env:ZERODHA_ACCESS_TOKEN="your_token"

# Linux/Mac
export ZERODHA_API_KEY="your_api_key"
export ZERODHA_API_SECRET="your_secret"
export ZERODHA_ACCESS_TOKEN="your_token"
```

---

## 🔒 Security Best Practices

### 1. **Token Management**
- ✅ Store tokens in secure files with restricted permissions
- ✅ Use environment variables for production
- ✅ Rotate tokens regularly
- ❌ Never commit API keys to version control

### 2. **File Permissions**
```bash
# Set restricted permissions on token files
chmod 600 config/access_tokens/zerodha/zerodha_config.json
chmod 600 config/access_tokens/upstox/upstox_config.json
```

### 3. **Gitignore Configuration**
Add to `.gitignore`:
```
# API Keys and Tokens
config/access_tokens/*/
*.token
*.key
.env
```

### 4. **Token Validation**
```python
# Validate tokens before use
def validate_broker_token(broker: str) -> bool:
    """Validate broker API token"""
    if broker == 'zerodha':
        # Implement Zerodha token validation
        pass
    elif broker == 'upstox':
        # Implement Upstox token validation
        pass
    return False
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Invalid API Key**
```
Error: Invalid API credentials
Solution: Verify API key and secret are correct
```

#### 2. **Token Expired**
```
Error: Access token expired
Solution: Regenerate access token using the steps above
```

#### 3. **Rate Limiting**
```
Error: Too many requests
Solution: Implement request throttling and respect API limits
```

#### 4. **Network Issues**
```
Error: Connection timeout
Solution: Check internet connection and API server status
```

### Debug Mode

Enable debug logging in `config/config.py`:

```python
LOGGING_CONFIG = {
    'level': 'DEBUG',
    'broker_debug': True,
    'api_debug': True
}
```

### Testing Connection

Use the built-in test script:

```python
# Test broker connection
python utils/test_broker_connection.py --broker zerodha
python utils/test_broker_connection.py --broker upstox
```

---

## 📞 Support

### Broker Support
- **Zerodha**: https://support.zerodha.com/
- **Upstox**: https://support.upstox.com/

### Framework Support
- **Documentation**: Check `docs/` folder
- **Issues**: Report bugs in repository issues
- **Community**: Join our discussions

---

## 📚 Additional Resources

### API Documentation
- [Zerodha Kite Connect API](https://kite.trade/docs/connect/v3/)
- [Upstox API v2](https://upstox.com/developer/api-documentation/)

### Video Tutorials
- [Setting up Zerodha API](docs/videos/zerodha_setup.md)
- [Upstox Configuration](docs/videos/upstox_setup.md)

---

**⚠️ Important Note**: API access requires active trading accounts and may have associated costs. Always test with paper trading before going live.

**🔄 Last Updated**: June 9, 2025 | **Version**: 1.0.0
