import pandas as pd
from typing import Tuple
import numpy as np

"""
UTILS FOR UTILS
"""

def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

import numpy as np

def lin_reg_slope_intercept(y, idx, window):
    A = np.vstack([idx, np.ones(window)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept

"""
UTILS THAT RETURN SERIES
"""

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA) of a given data series.
    """
    return data.ewm(span=period, adjust=False).mean()

def calculate_ema_smoothed(prices: pd.Series, periods: int, smoothing: int = 2) -> pd.Series:
    ema = list(prices[:periods-1])
    ema.append((sum(prices[:periods]) / periods))
    for price in prices[periods:]:
        ema.append((price * (smoothing / (1 + periods))) + ema[-1] * (1 - (smoothing / (1 + periods))))
    
    return pd.Series(ema)

def calculate_true_range(data: pd.DataFrame) -> pd.Series:
    x = data['high'] - data['low']
    y = abs(data['high'] - data['close'].shift(1))
    z = abs(data['low'] - data['close'].shift(1))

    tr = list(np.maximum(x, y, z))
    
    tr[0] = data['high'].iloc[0] - data['low'].iloc[0]
    
    return pd.Series(tr)

def calculate_movement_series(high: pd.Series, low: pd.Series) -> Tuple[pd.Series, pd.Series]:
    low_diff = low - low.shift(1)
    high_diff = high - high.shift(1)
    
    high_temp = [0.0]
    low_temp = [0.0]

    for i in range(1, len(low_diff)):
        if high_diff.iloc[i]>low_diff.iloc[i] and high_diff.iloc[i]>=0:
            high_temp.append(high_diff.iloc[i])
        else:
            high_temp.append(0)

        if low_diff.iloc[i]>high_diff.iloc[i] and low_diff.iloc[i]>=0:
            low_temp.append(low_diff.iloc[i])
        else:
            low_temp.append(0)

    return pd.Series(low_temp), pd.Series(high_temp)


def smooth_series(data: pd.Series, period: int = 14) -> pd.Series:
    if len(data)<period:
        return data
    
    l = list(data[:period-1])
    l.append(sum(data[:period])/period)

    if len(data)==period:
        return pd.Series(l)
    
    for tr in data[period:]:
        atr = ((l[-1]*(period-1)) + tr)/period
        l.append(atr)

    return pd.Series(l)

def calculate_cmf(data: pd.DataFrame, period: int = 21) -> pd.Series:
    # ((((Close – Low) – (High – Close)) / (High – Low)) * Volume) / Total(Volume, 21)
    vol_rolling = data['volume'].rolling(window= period).sum()
    vol_rolling.reset_index(inplace= True, drop= True)
    cmf = ((((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])) * data['volume']) / data['volume'].rolling(window= period).sum()
    return cmf

def calculate_transitions(data: pd.DataFrame) -> pd.Series:
    for col in ['high', 'low', 'close']:
        data[f'{col}_shift'] = data[col].shift(1)
    
    data['transition_index'] = 0

    mask_pos = (data['high']>data['high_shift']) & (data['low']>data['low_shift']) & (data['close']>data['high_shift'])
    mask_neg = (data['high']<data['high_shift']) & (data['low']<data['low_shift']) & (data['close']<data['low_shift'])

    data.loc[mask_pos, 'transition_index'] = 1
    data.loc[mask_neg, 'transition_index'] = -1

    return data['transition_index']

def calculate_on_balance_volume(data: pd.DataFrame, periods: int=10) -> pd.Series:
    data.reset_index(inplace= True, drop=True)
    close_shift  = (data['close'] - data['close'].shift(1)) / abs(data['close'] - data['close'].shift(1))
    
    close_shift.reset_index(inplace= True, drop=True)

    volume_signed = data['volume'] * close_shift

    return volume_signed.rolling(periods).sum().reset_index(drop= True)

def calculate_williams_r(data: pd.DataFrame, periods: int=14) -> pd.Series:
    highest_high = data['high'].rolling(periods).max()
    lowest_low = data['low'].rolling(periods).min()

    williams_r = (((highest_high - data['close']) / (highest_high - lowest_low) * -2.0) + 1.0)

    return williams_r

def calculate_pivot_points(data: pd.DataFrame) -> pd.Series:
    return ((data['high'] + data['low'] + data['close']) / 3.0).round(3)

def calculate_cci_from_pivot_points(pivot_points: pd.Series, periods: int=20, mean_deviation_coeff: float = 0.015) -> pd.Series:
    mean_dev = pivot_points.rolling(periods).apply(mad)
    sma_vals = calculate_sma(pivot_points, periods)

    cci = (pivot_points - sma_vals) / (mean_deviation_coeff * mean_dev)
    return cci

def calculate_cci(data: pd.DataFrame, periods: int=20, mean_deviation_coeff: float = 0.015) -> pd.Series:
    return calculate_cci_from_pivot_points(calculate_pivot_points(data), periods, mean_deviation_coeff)

def calculate_detrended_price_oscillator(close: pd.Series, periods: int=20) -> pd.Series:
    sma = calculate_sma(close)
    return (close.shift(int(periods/2)+1) - sma) / sma

def calculate_super_trend(data: pd.DataFrame, periods: int = 7, sensitivity_index: float = 3.0) -> Tuple[pd.Series, pd.Series]:

    highest_high = data['high'].rolling(periods).max()
    lowest_low = data['low'].rolling(periods).min()

    atr = calculate_average_true_range(data, periods)

    middle = (highest_high + lowest_low) / 2.
    middle.reset_index(inplace=True, drop= True)
    atr.reset_index(inplace=True, drop= True)

    return (middle + sensitivity_index*atr), (middle - sensitivity_index*atr)

def calculate_average_true_range(data: pd.DataFrame, period: int = 14) -> pd.Series:
    true_range = calculate_true_range(data)
    return smooth_series(true_range, period)

def calculate_sma(data: pd.Series, periods: str=9) -> pd.Series:
    return data.rolling(window= periods).mean()

def calculate_macd(data: pd.Series, params: Tuple[int, int, int] = [8,21,5]) -> Tuple[pd.Series, pd.Series]:
    ema_small = calculate_ema(data, params[0])
    ema_large = calculate_ema(data, params[1])
    
    macd = ema_small - ema_large
    signal = calculate_ema(macd, params[2])
    
    return macd, signal

def calculate_bollinger_bands(close: pd.Series, period: int = 20, bandwidth: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    middle = calculate_sma(close, period)
    stddev = close.rolling(window=period).std()

    upper = middle + (float(bandwidth) * stddev)
    lower = middle - (float(bandwidth) * stddev)

    return lower, middle, upper

def calculate_ichimoku_cloud(high: pd.Series, low: pd.Series, conversion: int= 9, base: int= 26, leading_span_b: int= 52, lagging_span: int = 26, offset: int = 0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    high_9 = high.rolling(window=conversion).max()
    low_9 = low.rolling(window=conversion).min()
    tenkan_sen = (high_9 + low_9) / 2 # conversion

    high_26 = high.rolling(window=base).max()
    low_26 = low.rolling(window=base).min()
    kijun_sen = (high_26 + low_26) / 2

    senkou_span_leading_a = (tenkan_sen + kijun_sen)/2

    senkou_span_a = senkou_span_leading_a.shift(lagging_span)
    senkou_span_b = ((high.rolling(window=leading_span_b).max() + low.rolling(window=leading_span_b).min()) / 2).shift(lagging_span)

    return tenkan_sen, kijun_sen, senkou_span_leading_a, senkou_span_a, senkou_span_b

def calculate_ichimoku_conversion(high: pd.Series, low: pd.Series, conversion: int=9) -> pd.Series:
    high_9 = high.rolling(window=conversion).max()
    low_9 = low.rolling(window=conversion).min()
    tenkan_sen = (high_9 + low_9) / 2 # conversion

    tenkan_sen.reset_index(inplace=True, drop= True)

    return tenkan_sen

def calculate_chande_momentum_oscillator(close: pd.Series, periods: int=5):
    delta = close.diff()

    # Separate the gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    gain_sum = gain.rolling(periods).sum()
    loss_sum = loss.rolling(periods).sum()

    cmo = 100.0 * ((gain_sum - loss_sum) / (gain_sum + loss_sum))

    return cmo.round(2)

def calculate_rsi_ema(data: pd.Series, periods: int = 5) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) using the EMA method for averaging gains and losses.

    :param data: DataFrame with the 'close' column.
    :param periods: Number of periods to calculate RSI, default is 14.
    :return: DataFrame with the RSI values.
    """
    delta = data.diff()

    # Separate the gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Calculate the EMA of gains and losses
    alpha = 1 / periods
    avg_gain = gain.ewm(alpha=alpha).mean()
    avg_loss = loss.ewm(alpha=alpha).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_volume_oscillator(volume_data: pd.Series, short_period: int = 5, long_period: int = 20) -> pd.Series:
    short_ma = volume_data.rolling(window=short_period).mean()
    long_ma = volume_data.rolling(window=long_period).mean()
    volume_oscillator = ((short_ma - long_ma) * 100.0) / long_ma

    return volume_oscillator

def calculate_volatility_oscillator(data: pd.DataFrame, window: int=100) -> Tuple[pd.Series, pd.Series]:
    s = data['close'] - data['open']
    x = s.rolling(window=window).std()
    
    return s, x

def calculate_marubozu_vals(data: pd.DataFrame) -> pd.Series:

    data['marubozu'] = 0
    data['high_close'] = abs(data['high']-data['close'])
    data['high_open'] = abs(data['high']-data['open'])
    data['low_close'] = abs(data['low']-data['close'])
    data['low_open'] = abs(data['low']-data['open'])

    data.loc[(data['high_close']<=0.05) & (data['low_open']<=0.05), 'marubozu'] = 1
    data.loc[(data['low_close']<=0.05) & (data['high_open']<=0.05), 'marubozu'] = -1

    return data['marubozu']

def three_line_strike_series(data: pd.DataFrame) -> pd.Series:
    """
    Returns 1 is bullish or -1 if bearish, 0 if neither
    """
    
    tls_list = [None, None, None]

    for i in range (3, len(data)):
        if data['close'].iloc[i-3] < data['open'].iloc[i-3] and \
            data['close'].iloc[i-2] < data['open'].iloc[i-2] and \
                data['close'].iloc[i-1] < data['open'].iloc[i-1] and \
                        data['close'].iloc[1] > data['open'].iloc[1]:
            tls_list.append(1)
        elif data['close'].iloc[i-3] > data['open'].iloc[i-3] and \
            data['close'].iloc[i-2] > data['open'].iloc[i-2] and \
                data['close'].iloc[i-1] > data['open'].iloc[i-1] and \
                        data['close'].iloc[i] < data['open'].iloc[i]:
            tls_list.append(-1)
        else:
            tls_list.append(0)
        
    return pd.Series(tls_list)

def engulfing_candle_series(open: pd.Series, close: pd.Series) -> pd.Series:
    """
    Returns 1 is bullish or -1 if bearish, 0 if neither
    """
    
    ecs_list = [None]

    for i in range(1, len(open)):
        if open.iloc[i] <= close.iloc[i-1] and \
            open.iloc[i] < open.iloc[i-1] and \
                close.iloc[i] > open.iloc[i-1]:
            ecs_list.append(1)
        elif open.iloc[i] >= close.iloc[i-1] and \
            open.iloc[i] > open.iloc[i-1] and \
                close.iloc[i] < open.iloc[i-1]:
            ecs_list.append(-1)
        else:
            ecs_list.append(0)

    return pd.Series(ecs_list)

def calculate_ma_direction_series(close: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
        ma = calculate_ema(close, period)
        ma_prev = ma.shift(1)
        return ma, ma_prev

def calculate_chaikin_volatility(data: pd.DataFrame, ema_fwindow: int=10, roc_window=10):
    # Calculate the high-low range for each period
    high_low_range = data['high'] - data['low']    
    # Calculate the EMA of the high-low range
    ema_range = calculate_ema(high_low_range, ema_fwindow)
    # Calculate the percentage change in the EMA of the range over the specified ROC window
    volatility = ema_range.pct_change(periods=roc_window) * 100
    return volatility

def calculate_historical_volatility(data: pd.DataFrame, window:int=20):
    log_returns = np.log(data['close'] / data['close'].shift(1))
    volatility = log_returns.rolling(window=window).std() * np.sqrt(252) # Annualized Volatility
    return volatility

def calculate_parkinson_volatility(data: pd.DataFrame, window: int = 20):
    term = (1 / (4 * np.log(2))) * ((data['high'] / data['low']).apply(np.log))**2
    parkinson_vol = np.sqrt(term.rolling(window=window).mean() * (1 / window))
    return parkinson_vol

def calculate_log_relative_volatility_index(data: pd.DataFrame, window: int=14):
    log_return = np.log(data['close'] / data['close'].shift(1))
    std_dev = log_return.rolling(window=window).std()
    mean_std_dev = std_dev.mean()
    rvi = (std_dev - mean_std_dev) / std_dev.std()
    return rvi

def calculate_volatility_stop(data: pd.DataFrame, atr_multiplier: int =3, atr_window: int=14):
    atr = calculate_average_true_range(data, window=atr_window)
    stop_loss = data['close'] - (atr_multiplier * atr)
    return stop_loss

def calculate_average_daily_range(data: pd.DataFrame, window: int=14):
    daily_range = data['high'] - data['low']
    adr = daily_range.rolling(window=window).mean()
    return adr

def calculate_bollinger_band_width(data: pd.DataFrame, window: int=20, width: int=2):
    l, m, h = calculate_bollinger_bands(data['close'], window, width)
    return h-l

def calculate_chandelier_exit(data: pd.DataFrame, atr_multiplier: int=3, atr_window: int=22, long_position: bool=True):
    atr = calculate_average_true_range(data, period=atr_window)
    if long_position:
        ch_exit = data['high'].rolling(window=atr_window).max().reset_index(drop= True) - (atr_multiplier * atr)
    else:
        ch_exit = data['low'].rolling(window=atr_window).min().reset_index(drop= True) + (atr_multiplier * atr)
    return ch_exit

def calculate_trix(data: pd.DataFrame, window: int=15):
    ema1 = calculate_ema(data['close'], window)
    ema2 = calculate_ema(ema1, window)
    ema3 = calculate_ema(ema2, window)
    trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
    return trix

def calculate_normalized_atr(data: pd.DataFrame, window: int=14):
    atr = calculate_average_true_range(data, window=window)
    natr = (atr / data['close']) * 100
    return natr

def calculate_relative_volatility_index(data: pd.DataFrame, window: int=14):
    standard_deviation = data['close'].rolling(window=window).std()
    rvi = standard_deviation / standard_deviation.mean()
    return rvi * 100

def calculate_market_facilitation_index(data: pd.DataFrame):
    mfi = (data['high'] - data['low']) / data['volume']
    return mfi

def calculate_keltner_channels(data: pd.DataFrame, period: int = 14, bandwidth: int =2) -> Tuple[pd.Series, pd.Series]:
    sma = calculate_sma(data['close'], period)
    atr = calculate_average_true_range(data, period)

    keltner_upper = sma + bandwidth * atr
    keltner_lower = sma - bandwidth * atr

    return keltner_lower, sma, keltner_upper

def calculate_squeeze_momentum_indicator(data: pd.DataFrame, length: int=20, mult: int=2):
    bl, bm, bu = calculate_bollinger_bands(data['close'], length, mult)
    kl, km, ku = calculate_keltner_channels(data, length, mult)
    squeeze = (bl > kl) & (bu < ku)
    return squeeze.astype(int)

def calculate_yang_zhang_volatility(data: pd.DataFrame, window: int=30):
    log_oc = (data['open'] / data['close'].shift()).apply(np.log)
    log_oc_squared = log_oc**2
    log_cc = (data['close'] / data['close'].shift()).apply(np.log)
    log_cc_squared = log_cc**2 
    sigma_open = log_oc_squared.rolling(window=window).mean()
    sigma_close = log_cc_squared.rolling(window=window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    sigma_yz = np.sqrt(sigma_open + k * sigma_close + (1 - k) * sigma_open)
    return pd.Series(sigma_yz, index= data.index)

def calculate_stochastic_oscillator(data: pd.DataFrame, period: int=14, sma_period: int=3) -> Tuple[pd.Series, pd.Series]:
    # Calculate %K
    low_min = data['low'].rolling(window=period).min()
    high_max = data['high'].rolling(window=period).max()
    percent_k = 100 * (data['close'] - low_min) / (high_max - low_min)
    # Calculate %D
    percent_d = percent_k.rolling(window=sma_period).mean() # Simple moving average of %K
    return percent_k, percent_d

def calculate_awesome_oscillator(data: pd.DataFrame, low_window: int = 5, high_window: int=34):
    midpoint = (data['high'] + data['low']) / 2
    sma_5 = midpoint.rolling(window=low_window).mean()
    sma_34 = midpoint.rolling(window=high_window).mean()
    ao = sma_5 - sma_34
    return ao

def calculate_stochastic_rsi(data: pd.DataFrame, window: int=14):
    rsi = calculate_rsi_ema(data['close'], window)
    min_rsi = rsi.rolling(window=window).min()
    max_rsi = rsi.rolling(window=window).max()
    stoch_rsi = 100 * (rsi - min_rsi) / (max_rsi - min_rsi)
    return stoch_rsi

def calculate_money_flow_index(data: pd.DataFrame, window: int=14):
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    raw_money_flow = typical_price * data['volume']
    temp = (typical_price - typical_price.shift(1)) / abs(typical_price - typical_price.shift(1))
    temp = temp.multiply(raw_money_flow)
    positive_money_flow = temp.copy()
    positive_money_flow[positive_money_flow < 0] = 0 # Only positive money flows are considered
    negative_money_flow = temp.copy()
    negative_money_flow[negative_money_flow > 0] = 0 # Only negative money flows are considered
    negative_money_flow*=-1.0

    money_flow_ratio = (positive_money_flow.rolling(window).sum() / negative_money_flow.rolling(window).sum())
    mfi = 100.0 - (100.0 / (money_flow_ratio + 1.0))

    return mfi, money_flow_ratio

def calculate_percentage_volume_oscillator(data: pd.DataFrame, fast_period: int=12, slow_period: int=26, signal_period: int=9):
    fast_ema = calculate_ema(data['volume'], fast_period)
    slow_ema = calculate_ema(data['volume'], slow_period)
    pvo = (fast_ema - slow_ema) / slow_ema * 100
    signal_line = calculate_ema(pvo, signal_period)
    pvo_histogram = pvo - signal_line
    return pvo, signal_line, pvo_histogram

def calculate_true_strength_index(data: pd.DataFrame, long_window: int=25, short_window: int=13, signal_window: int=7):
    delta = data['close'].diff(1)
    double_smoothed_pc = calculate_ema(calculate_ema(delta, short_window), long_window)
    double_smoothed_abs_pc = calculate_ema(calculate_ema(abs(delta), short_window), long_window)
    tsi = 100 * double_smoothed_pc / double_smoothed_abs_pc
    signal = calculate_ema(tsi, signal_window)
    return tsi, signal

def calculate_accumulation_distribution_line(data: pd.DataFrame):
    clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    ad = (clv * data['volume']).cumsum()
    return ad

def calculate_dynamic_momentum_index(data: pd.DataFrame, window: int = 14, vol_window: int = 10):
    std_dev = data['close'].rolling(vol_window).std()
    mean_std = std_dev.mean()
    if mean_std==0:
        mean_std=0.01
    
    std_dev.fillna(0, inplace=True)

    var_window = (window * (std_dev/mean_std)).round().astype(int)
    var_window[var_window < 1] = 1

    d_rsi = []
    for idx, win in var_window.items():
        if idx>=win:
            d_rsi.append(calculate_rsi_ema(data['close'].iloc[idx-win+1:idx+1], win).iloc[-1])
        else:
            d_rsi.append(np.nan)
    d_rsi = pd.Series(index= data.index, data= d_rsi)

    return d_rsi
    
def calculate_hawkins_pressure_indicator(data: pd.DataFrame, window: int =14):
    close_change = data['close'].diff()
    volume_change = data['volume'].diff()
    pressure = (close_change / data['close']) * volume_change
    return pressure.rolling(window=window).mean()

def calculate_swing_index(data: pd.DataFrame):
    # Calculate changes and price differences
    close_change = data['close'] - data['close'].shift()
    term1 = close_change
    term2 = 0.5 * (data['close'] - data['open'])
    term3 = 0.25 * (data['close'].shift() - data['close'].shift(1))
    # Calculate the range (avoiding zero division)
    price_range = data['high'] - data['low']
    price_range.replace(0, pd.NA, inplace=True)  # Replace zeros with NA to avoid division by zero
    # Calculate the Swing Index
    si = 50 * (term1 + term2 + term3) / price_range
    return si

def calculate_linear_regressor_indicator(data: pd.DataFrame, window: int =30):
    y = data['close'].rolling(window, min_periods=0).apply(lambda x: (len(x)))
    y.fillna(0, inplace= True)
    y = y.apply(lambda x: np.arange(int(x)))
    x = pd.Series(data['close'].rolling(window=window))
    res = []
    for i in range(len(y)):
        slope, intercept = np.polyfit(x.iloc[i], y.iloc[i], 1)
        res.append(slope * (window - 1) + intercept)
    
    return pd.Series(res, index= data.index)

def calculate_directional_movement_index(data: pd.DataFrame, window: int =14):
    tr = pd.concat([data['high'] - data['low'], abs(data['high'] - data['close'].shift()), abs(data['low'] - data['close'].shift())], axis=1).max(axis=1)
    tr = tr.rolling(window=window).sum()
    up_move = data['high'].diff()
    down_move = data['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0).cumsum()
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0).cumsum()
    plus_di = 100 * (plus_dm / tr)
    minus_di = 100 * (minus_dm / tr)
    return pd.Series(plus_di, index= data.index), pd.Series(minus_di, index= data.index)

def calculate_hull_moving_average(data: pd.DataFrame, window: int =9):
    wma_half = data['close'].rolling(window=int(window/2)).mean() * 2
    wma_full = data['close'].rolling(window=window).mean()
    raw_hma = wma_half - wma_full
    hma = raw_hma.rolling(window=int(np.sqrt(window))).mean()
    return hma

def calculate_T3(data: pd.DataFrame, length: int =5, a: float =0.7):
    e1 = calculate_ema(data['close'], length)
    e2 = calculate_ema(e1, length)
    e3 = calculate_ema(e2, length)
    c1 = -a * a * a
    c2 = 3 * a * a + 3 * a * a * a
    c3 = -6 * a * a - 3 * a - 3 * a * a * a
    c4 = 1 + 3 * a + a * a * a + 3 * a * a
    T3 = c1 * e3 + c2 * e2 + c3 * e1 + c4 * data['close']
    return T3

def calculate_TEMA(data: pd.DataFrame, window: int =30):
    ema1 = calculate_ema(data['close'], window)
    ema2 = calculate_ema(ema1, window)
    ema3 = calculate_ema(ema2, window)
    tema = 3 * (ema1 - ema2) + ema3
    return tema

def calculate_VWAP(data: pd.DataFrame):
    return (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()

import pandas as pd
import numpy as np

def calculate_ZigZag(data: pd.DataFrame, change_threshold: float=0.05):
    """
    Calculates the ZigZag pattern based on the given data and change threshold.

    Args:
        data (pd.DataFrame): The input data containing the 'close' prices.
        change_threshold (float, optional): The minimum price change to consider. Defaults to 0.05.

    Returns:
        np.ndarray: An array containing the pivot points of the ZigZag pattern.
    """
    pivots = np.zeros(len(data))
    direction = 0
    last_pivot = data['close'].iloc[0]
    for i in range(1, len(data)):
        if direction == 0:
            if data['close'].iloc[i] > last_pivot * (1 + change_threshold):
                direction = 1
            elif data['close'].iloc[i] < last_pivot * (1 - change_threshold):
                direction = -1
        elif direction == 1:
            if data['close'].iloc[i] < last_pivot * (1 - change_threshold):
                pivots[i-1] = last_pivot
                direction = -1
                last_pivot = data['close'].iloc[i]
        elif direction == -1:
            if data['close'].iloc[i] > last_pivot * (1 + change_threshold):
                pivots[i-1] = last_pivot
                direction = 1
                last_pivot = data['close'].iloc[i]
    return pivots

def calculate_chande_forecast_oscillator(data: pd.DataFrame, window: int =14):
    idx = np.arange(window)
    
    # Calculate the forecasted values using linear regression
    forecasted_values = np.array([np.nan] * (window - 1))  # Start with NaN values to align size
    
    for i in range(len(data) - window + 1):
        y = data['close'].iloc[i:i+window].values
        slope, intercept = lin_reg_slope_intercept(y, idx, window)
        forecast = slope * (window - 1) + intercept
        forecasted_values = np.append(forecasted_values, forecast)
    
    forecasted_values = pd.Series(forecasted_values)

    # Calculate the CFO
    cfo = (forecasted_values - (data['close']) * 100.0) / forecasted_values
    
    return pd.Series(cfo, index=data.index)

def calculate_vortex_indicator(data: pd.DataFrame, window: int=14):
    tr = pd.concat([data['high'] - data['low'], abs(data['high'] - data['close'].shift()), abs(data['low'] - data['close'].shift())], axis=1).max(axis=1)
    vm_plus = abs(data['high'] - data['low'].shift())
    vm_minus = abs(data['low'] - data['high'].shift())
    tr_n = tr.rolling(window=window).sum()
    vm_plus_n = vm_plus.rolling(window=window).sum()
    vm_minus_n = vm_minus.rolling(window=window).sum()
    vi_plus = vm_plus_n / tr_n
    vi_minus = vm_minus_n / tr_n
    return vi_plus, vi_minus

def calculate_schaff_trend_cycle(data: pd.DataFrame, fast_ema: int=23, slow_ema: int=50, cycle: int=10, roll_trend: int=10):
    macd, macd_ema = calculate_macd(data['close'], [slow_ema, fast_ema, cycle])
    stc = ((macd - macd_ema)  * 100.0) / (macd_ema.rolling(window=roll_trend).max() - macd_ema.rolling(window=roll_trend).min())
    return stc

def calculate_mcclellan_oscillator(data: pd.DataFrame, window: int = 19):
    num_advances = (data['close'] > data['open']).rolling(window).sum()
    num_declines = (data['close'] < data['open']).rolling(window).sum()
    mcclellan_oscillator = num_advances - num_declines
    return mcclellan_oscillator

def calculate_mcclellan_summation_index(data: pd.DataFrame, window: int=19):
    mo = calculate_mcclellan_oscillator(data, window)
    msi = mo.cumsum()
    return msi

def calculate_coppock_curve(data: pd.DataFrame, wma_window: int=10, low_period: int = 11, high_period: int = 14):
    roc_11 = data['close'].pct_change(periods=low_period)
    roc_14 = data['close'].pct_change(periods=high_period)
    coppock = (roc_11 + roc_14).rolling(window=wma_window).mean()
    return coppock

def calculate_prings_know_sure_thing(data: pd.DataFrame):
    rcma1 = calculate_sma(data['close'].pct_change(10), 10)
    rcma2 = calculate_sma(data['close'].pct_change(10), 10)
    rcma3 = calculate_sma(data['close'].pct_change(10), 10)
    rcma4 = calculate_sma(data['close'].pct_change(10), 15)
    kst = (rcma1 * 1.0) + (rcma2 * 2.0) + (rcma3 * 3.0) + (rcma4 * 4.0)
    return kst, calculate_sma(kst)

def calculate_adaptive_moving_average(data: pd.DataFrame, window: int=10):
    change = data['close'].diff(window).abs()
    volatility = data['close'].diff().abs().rolling(window=window).sum()
    efficiency_ratio = change / volatility
    smoothing_constant = (efficiency_ratio * (2.0 / (2 + 1) - 2.0 / (30 + 1)) + 2.0 / (30 + 1)) ** 2
    ama = data['close'].copy().astype(float).tolist()
    for i in range(window+1, len(data)):
        ama[i] = ama[i - 1] + smoothing_constant.iloc[i] * (data['close'].iloc[i] - ama[i - 1])
    return pd.Series(ama, index= data.index)

def calculate_wilders_smoothing(data: pd.DataFrame, window: int=14):
    initial_ema = data['close'][:window].mean()
    smoothed_data = pd.Series(index=data.index)
    smoothed_data.iloc[window-1] = initial_ema
    alpha = 1 / window
    for i in range(window, len(data)):
        smoothed_data.iloc[i] = smoothed_data.iloc[i-1] + alpha * (data['close'].iloc[i] - smoothed_data.iloc[i-1])
    return smoothed_data

def calculate_elder_impulse_system(data: pd.DataFrame, ema_param: int = 13):
    ema = calculate_ema(data['close'], ema_param)
    macd_val, macd_signal = calculate_macd(data['close'])
    impulse = np.where((ema.diff() > 0) & (macd_val - macd_signal > 0), 1, np.where((ema.diff() < 0) & (macd_val - macd_signal < 0), -1, 0))
    return pd.Series(impulse, index= data.index)

def calculate_zero_lag_ema(data: pd.DataFrame, window: int=14):
    ema1 = calculate_ema(data['close'], window)
    ema2 = calculate_ema(ema1, window)
    zlema = ema1 + (ema1 - ema2)
    return zlema

def calculate_volume_price_trend(data: pd.DataFrame):
    vpt = (data['volume'] * ((data['close'] - data['close'].shift(1)) / data['close'].shift(1))).cumsum()
    return vpt

def calculate_panic_euphoria_indicator(data: pd.DataFrame, roll_period: int = 10):
    """
    Calculates the panic-euphoria indicator based on the given data.

    Args:
        data (pd.DataFrame): The input data containing the 'close' and 'volume' columns.
        roll_period (int, optional): The rolling period for calculating volatility and average volume. Defaults to 10.

    Returns:
        pd.Series: A series containing the panic-euphoria indicator values (0 or 1) for each data point.
    """
    p_change = data['close'].pct_change()
    vol = p_change.rolling(window=roll_period).std()
    high_vol = data['volume'] > data['volume'].rolling(window=roll_period).mean()

    peu = ((abs(p_change) > (2.0 * vol)) & high_vol)

    return peu.astype(int)

def calculate_disposition_effect_indicator(data: pd.DataFrame, threshold: float=0.01):
    """
    Calculates the disposition effect indicator for a given DataFrame of stock data.

    The disposition effect indicator measures the tendency of investors to sell stocks that have gained in value
    and continue holding stocks that have lost value.

    Args:
        data (pd.DataFrame): The DataFrame containing the stock data.
        threshold (float, optional): The threshold value for gains/losses. Defaults to 0.01.

    Returns:
        sell (pd.Series): A binary series indicating where assets are being sold after gains.
        hold (pd.Series): A binary series indicating where assets are continued to be held after losses.
    """
    # Calculate gains/losses from the purchase
    returns = data['close'].pct_change()
    # Identify where assets are being sold after gains or continued holding after losses
    sell = (returns > threshold) & (data['volume'] > data['volume'].shift(1))
    hold = (returns < -threshold) & (data['volume'].shift(1) > data['volume'])
    return sell.astype(int), hold.astype(int)

"""
THE FOLLOWING METHODS ONLY RETURN VALUES FOR ONE ITERATION (ROW)
"""

def calculate_rsi(data: pd.DataFrame, period: int = 5) -> float:
    """
    Calculate the Relative Strength Index (RSI) for the given stock data.

    Parameters:
    data (pandas.DataFrame): The stock market data.
    period (int): The period over which to calculate the RSI, typically 14 days.

    Returns:
    pandas.Series: The RSI values.
    """
    # Calculate daily price changes
    data['change'] = data['close'] - data['open']
    gains = data[data['change']>0]
    losses = data[data['change']<=0]

    # Calculate the average gains and losses
    avg_gain = gains['change'].rolling(window=period, min_periods=period).mean().tolist()[-1]
    avg_loss = -1 * losses['change'].rolling(window=period, min_periods=period).mean().tolist()[-1]

    # Calculate the Relative Strength (RS)
    if avg_loss==0:
        return 100
    
    rs = avg_gain / avg_loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calc_stop_loss(data: pd.DataFrame, stoploss_type: str = 'ema20', trade_type: str = "BUY", close_price: float = None) -> bool:
    close_price = data['close'].tolist()[-1] if not close_price else close_price
    if stoploss_type=='ema20':
        sl_cutoff = calculate_ema(data['close'], 20).tolist()[-1]
        return (close_price < sl_cutoff) if trade_type=="BUY" else (close_price>sl_cutoff)
    elif stoploss_type=='rsi5':
        sl_cutoff = calculate_rsi_ema(data, 5).tolist()[-1]
        return (sl_cutoff < 50) if trade_type=="BUY" else (sl_cutoff>50)
    else:
        print("\nERROR: OPTION UNAVAILABLE\n")
        return False

def three_line_strike_single(data: pd.DataFrame) -> int:
    """
    Returns 1 is bullish or -1 if bearish, 0 if neither
    """
    if data['close'].iloc[-4] < data['open'].iloc[-4] and \
        data['close'].iloc[-3] < data['open'].iloc[-3] and \
            data['close'].iloc[-2] < data['open'].iloc[-2] and \
                    data['close'].iloc[-1] > data['open'].iloc[-1]:
        return 1
    
    if data['close'].iloc[-4] > data['open'].iloc[-4] and \
        data['close'].iloc[-3] > data['open'].iloc[-3] and \
            data['close'].iloc[-2] > data['open'].iloc[-2] and \
                    data['close'].iloc[-1] < data['open'].iloc[-1]:
        return -1
    
    return 0

def engulfing_candle_single(open: pd.Series, close: pd.Series) -> int:
    """
    Returns 1 is bullish or -1 if bearish, 0 if neither
    """

    if open.iloc[-1] <= close.iloc[-2] and \
        open.iloc[-1] < open.iloc[-2] and \
            close.iloc[-1] > open.iloc[-2]:
        return 1
    
    if open.iloc[-1] >= close.iloc[-2] and \
        open.iloc[-1] > open.iloc[-2] and \
            close.iloc[-1] < open.iloc[-2]:
        return -1
    
    return 0

