from __future__ import annotations

from typing import Callable, Dict

import pandas as pd
from pandas import DataFrame, Series, NA

from . import quant_utils as qu


_PERIOD_MAP = {
    1: 'minute',
    3: '3minute',
    5: '5minute',
    10: '10minute',
    15: '15minute',
    30: '30minute',
    60: 'hour',
}


class _CandleDef:
    minute_num_dict = {"minute": 1, "3minute": 3, "5minute": 5, "10minute": 10,
                       "15minute": 15, "30minute": 30, "hour": 60, "day": 375}
    changes_threshold = {"minute": 0.005, '3minute': 0.007, "5minute": 0.01, "10minute": 0.015,
                         "15minute": 0.02, "30minute": 0.03, "hour": 0.04, "day": 0.05}

    @classmethod
    def sensitivity(cls, period: str) -> float:
        return cls.changes_threshold.get(period, 0.01)


def get_period_from_df(df: DataFrame) -> str:
    """Infer timeframe string from timestamp deltas."""
    if len(df) < 2:
        return '5minute'
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'date'
    try:
        delta = (pd.to_datetime(df[ts_col].iloc[-1]) - pd.to_datetime(df[ts_col].iloc[-2])).total_seconds() / 60.0
    except Exception:
        return '5minute'
    int_delta = int(round(delta))
    return _PERIOD_MAP.get(int_delta, 'day')


candle_def = _CandleDef

def pd_subtract(data: DataFrame, x: Series) -> Series:
    return (x - data['close'])    

def pd_scale(data: DataFrame, x: Series, scaling_factor: float = 100.0) -> Series:
    return ((x * scaling_factor) / data['close'])

def sub_and_scale(data: DataFrame, x: Series, scaling_factor: float = 100.0) -> Series:
    return pd_scale(data, pd_subtract(data, x), scaling_factor)

def ichi_conversion(data: DataFrame):
    tenkan_sen, kijun_sen, senkou_span_leading_a, senkou_span_a, senkou_span_b = qu.calculate_ichimoku_cloud(data['high'], data['low'])
    return (tenkan_sen)

def ichi_kijun(data: DataFrame):
    tenkan_sen, kijun_sen, senkou_span_leading_a, senkou_span_a, senkou_span_b = qu.calculate_ichimoku_cloud(data['high'], data['low'])
    return (kijun_sen)

def ichi_senkou_leading(data: DataFrame):
    tenkan_sen, kijun_sen, senkou_span_leading_a, senkou_span_a, senkou_span_b = qu.calculate_ichimoku_cloud(data['high'], data['low'])
    return (senkou_span_leading_a)

def ichi_senkou_a(data: DataFrame):
    tenkan_sen, kijun_sen, senkou_span_leading_a, senkou_span_a, senkou_span_b = qu.calculate_ichimoku_cloud(data['high'], data['low'])
    return (senkou_span_a)

def ichi_senkou_b(data: DataFrame):
    tenkan_sen, kijun_sen, senkou_span_leading_a, senkou_span_a, senkou_span_b = qu.calculate_ichimoku_cloud(data['high'], data['low'])
    return (senkou_span_b)

def price_movement(data: DataFrame):
    s, x = qu.calculate_volatility_oscillator(data)
    return (s)

def price_volatility(data: DataFrame):
    s, x = qu.calculate_volatility_oscillator(data)
    return (x)

def volume_osc(data: DataFrame):
    return qu.calculate_volume_oscillator(data['volume'])

def rsi_generic(data: DataFrame, i: int):
    return ((qu.calculate_rsi_ema(data['close'], i) - 50.0) / 50.0)

def stochastic_rsi_generic(data: DataFrame, i: int):
    return qu.calculate_stochastic_rsi(data, i)

def rsi_7(data: DataFrame):
    return rsi_generic(data, 7)

def rsi_5(data: DataFrame):
    return rsi_generic(data, 5)

def stochastic_rsi_7(data: DataFrame):
    return stochastic_rsi_generic(data, 7)

def stochastic_rsi_5(data: DataFrame):
    return stochastic_rsi_generic(data, 5)

def vol_ema_generic(data: DataFrame, i: int):
    v = data['volume'].replace(0, NA).ffill()
    return ((qu.calculate_ema(data['volume'], i) - v) *100.0)/ v

def vol_ema_3(data: DataFrame):
    return vol_ema_generic(data, 3)

def vol_ema_5(data: DataFrame):
    return vol_ema_generic(data, 5)

def vol_ema_7(data: DataFrame):
    return vol_ema_generic(data, 7)

def vol_ema_9(data: DataFrame):
    return vol_ema_generic(data, 9)

def cmo_generic(data: DataFrame, i: int):
    return qu.calculate_chande_momentum_oscillator(data['close'], i)

def cmo_5(data: DataFrame):
    return cmo_generic(data, 5)

def cmo_7(data: DataFrame):
    return cmo_generic(data, 7)

def cmo_9(data: DataFrame):
    return cmo_generic(data, 9)

def sthc_generic(data: DataFrame, i: int):
    super_limit_max, super_limit_min = qu.calculate_super_trend(data, i)
    return (super_limit_max)

def stlc_generic(data: DataFrame, i: int):
    super_limit_max, super_limit_min = qu.calculate_super_trend(data, i)
    return (super_limit_min)

def sthc_7(data: DataFrame):
    return sthc_generic(data, 7)

def sthc_10(data: DataFrame):
    return sthc_generic(data, 10)

def stlc_7(data: DataFrame):
    return stlc_generic(data, 7)

def stlc_10(data: DataFrame):
    return stlc_generic(data, 10)

def macd_val_generic(data: DataFrame, params: list[int] = [3,7,3]):
    macd_val_tmp, macd_signal_tmp = qu.calculate_macd(data['close'], params)
    return macd_val_tmp

def macd_signal_generic(data: DataFrame, params: list[int] = [3,7,3]):
    macd_val_tmp, macd_signal_tmp = qu.calculate_macd(data['close'], params)
    return macd_signal_tmp

def macd_histogram_generic(data: DataFrame, params: list[int] = [3,7,3]):
    macd_val_tmp, macd_signal_tmp = qu.calculate_macd(data['close'], params)
    return (macd_val_tmp - macd_signal_tmp)

def macd_val_3_7_3(data: DataFrame):
    return macd_val_generic(data, [3, 7, 3])

def macd_signal_3_7_3(data: DataFrame):
    return macd_signal_generic(data, [3,7,3])

def macd_histogram_3_7_3(data: DataFrame):
    return macd_histogram_generic(data, [3,7,3])

def macd_val_8_21_5(data: DataFrame):
    return macd_val_generic(data, [8, 21, 5])

def macd_signal_8_21_5(data: DataFrame):
    return macd_signal_generic(data, [8,21,5])

def macd_histogram_8_21_5(data: DataFrame):
    return macd_histogram_generic(data, [8,21,5])

def macd_val_5_20_30(data: DataFrame):
    return macd_val_generic(data, [5, 20, 30])

def macd_signal_5_20_30(data: DataFrame):
    return macd_signal_generic(data, [5,20,30])

def macd_histogram_5_20_30(data: DataFrame):
    return macd_histogram_generic(data, [5,20,30])

def blc(data: DataFrame):
    lower, middle, upper = qu.calculate_bollinger_bands(data['close'])
    return lower

def bmc(data: DataFrame):
    lower, middle, upper = qu.calculate_bollinger_bands(data['close'])
    return middle

def bhc(data: DataFrame):
    lower, middle, upper = qu.calculate_bollinger_bands(data['close'])
    return upper

def atr(data: DataFrame):
    return (qu.calculate_average_true_range(data))

def khc(data: DataFrame):
    keltner_lower, keltner_middle, keltner_upper = qu.calculate_keltner_channels(data)
    return (keltner_upper)

def kmc(data: DataFrame):
    keltner_lower, keltner_middle, keltner_upper = qu.calculate_keltner_channels(data)
    return (keltner_middle)

def klc(data: DataFrame):
    keltner_lower, keltner_middle, keltner_upper = qu.calculate_keltner_channels(data)
    return (keltner_lower)

def obv(data: DataFrame):
    return ((qu.calculate_on_balance_volume(data)  - data['volume']) *100.0)/ data['volume']

def cci(data: DataFrame):
    return qu.calculate_cci_from_pivot_points(qu.calculate_pivot_points(data))

def dpo(data: DataFrame):
    return qu.calculate_detrended_price_oscillator(data['close'])

def historical_volatility(data: DataFrame):
    return qu.calculate_historical_volatility(data)

def parkinson_volatility(data: DataFrame):
    return qu.calculate_parkinson_volatility(data)

def adr(data: DataFrame):
    return (qu.calculate_average_daily_range(data))

def bbw(data: DataFrame):
    return qu.calculate_bollinger_band_width(data)

def chandelier_long(data: DataFrame):
    return (qu.calculate_chandelier_exit(data= data, long_position= True))

def chandelier_short(data: DataFrame):
    return (qu.calculate_chandelier_exit(data= data, long_position= False))

def perc_k(data: DataFrame):
    pck, pcd = qu.calculate_stochastic_oscillator(data)
    return pck

def perc_d(data: DataFrame):
    pck, pcd = qu.calculate_stochastic_oscillator(data)
    return pcd

def awesome(data: DataFrame):
    return (qu.calculate_awesome_oscillator(data))

def mfi(data: DataFrame):
    mfi, mfr = qu.calculate_money_flow_index(data)
    return mfi

def mfr(data: DataFrame):
    mfi, mfr = qu.calculate_money_flow_index(data)
    return mfr

def pvo(data: DataFrame):
    pvo, pvo_signal, pvo_hist = qu.calculate_percentage_volume_oscillator(data)
    return pvo

def pvo_signal(data: DataFrame):
    pvo, pvo_signal, pvo_hist = qu.calculate_percentage_volume_oscillator(data)
    return pvo_signal

def pvo_hist(data: DataFrame):
    pvo, pvo_signal, pvo_hist = qu.calculate_percentage_volume_oscillator(data)
    return pvo_hist

def tsi(data: DataFrame):
    x, y = qu.calculate_true_strength_index(data)
    return x

def tsi_signal(data: DataFrame):
    x, y = qu.calculate_true_strength_index(data)
    return y

def tsi_hist(data: DataFrame):
    x, y = qu.calculate_true_strength_index(data)
    return x-y

def dymind(data: DataFrame):
    return ((qu.calculate_dynamic_momentum_index(data) - 50.0) / 50.0)

def hawkins(data: DataFrame):
    return (qu.calculate_hawkins_pressure_indicator(data) * 100.0) / data['volume']

def di_plus(data: DataFrame):
    di_p, di_m = qu.calculate_directional_movement_index(data)
    return di_p

def di_minus(data: DataFrame):
    di_p, di_m = qu.calculate_directional_movement_index(data)
    return di_m

def hmac(data: DataFrame):
    return qu.calculate_hull_moving_average(data)

def zigzag(data: DataFrame):
    return (qu.calculate_ZigZag(data, candle_def.sensitivity(get_period_from_df(data))))

def vi_plus(data: DataFrame):
    vi_plus, vi_minus = qu.calculate_vortex_indicator(data)
    return vi_plus

def vi_minus(data: DataFrame):
    vi_plus, vi_minus = qu.calculate_vortex_indicator(data)
    return vi_minus

def prings_kst(data: DataFrame):
    kst, kst_signal = qu.calculate_prings_know_sure_thing(data)
    return kst

def prings_kst_signal(data: DataFrame):
    kst, kst_signal = qu.calculate_prings_know_sure_thing(data)
    return kst_signal

def prings_kst_hist(data: DataFrame):
    kst, kst_signal = qu.calculate_prings_know_sure_thing(data)
    return (kst - kst_signal)

def dei_sell(data: DataFrame):
    dei_sell, dei_hold = qu.calculate_disposition_effect_indicator(data, candle_def.sensitivity(get_period_from_df(data)))
    return dei_sell

def dei_hold(data: DataFrame):
    dei_sell, dei_hold = qu.calculate_disposition_effect_indicator(data, candle_def.sensitivity(get_period_from_df(data)))
    return dei_hold


class indicator_def:
    key_function_def = {
        'ichi_conversion': ichi_conversion,
        'ichi_kijun': ichi_kijun,
        'ichi_senkou_leading': ichi_senkou_leading,
        'ichi_senkou_a': ichi_senkou_a,
        'ichi_senkou_b': ichi_senkou_b,
        'price_movement': price_movement,
        'price_volatility': price_volatility,
        'volume_osc': volume_osc,
        'rsi_5': rsi_5,
        'stochastic_rsi_5': stochastic_rsi_5,
        'rsi_7': rsi_7,
        'stochastic_rsi_7': stochastic_rsi_7,
        'vol_ema_3': vol_ema_3,
        'vol_ema_5': vol_ema_3,
        'vol_ema_7': vol_ema_7,
        'vol_ema_9': vol_ema_9,
        'cmo_5': cmo_5,
        'cmo_7': cmo_7,
        'cmo_9': cmo_9,
        'sthc_7': sthc_7,
        'sthc_10': sthc_10,
        'stlc_7': stlc_7,
        'stlc_10': stlc_10,
        'macd_val_3_7_3': macd_val_3_7_3,
        'macd_val_8_21_5': macd_val_8_21_5,
        'macd_val_5_20_30': macd_val_5_20_30,
        'macd_signal_3_7_3': macd_signal_3_7_3,
        'macd_signal_8_21_5': macd_signal_8_21_5,
        'macd_signal_5_20_30': macd_signal_5_20_30,
        'macd_histogram_3_7_3': macd_histogram_3_7_3,
        'macd_histogram_8_21_5': macd_histogram_8_21_5,
        'macd_histogram_5_20_30': macd_histogram_5_20_30,
        'blc': blc,
        'bmc': bmc,
        'bhc': bhc,
        'cmf': qu.calculate_cmf,
        'atr': atr,
        'khc': khc,
        'klc': klc,
        'kmc': kmc,
        'obv': obv,
        'pivot_point': qu.calculate_pivot_points,
        'cci': cci,
        'dpo': dpo,
        'chaikin_volatility': qu.calculate_chaikin_volatility,
        'historical_volatility': historical_volatility,
        'parkinson_volatility': parkinson_volatility,
        'rvi': qu.calculate_relative_volatility_index,
        'log_rvi': qu.calculate_log_relative_volatility_index,
        'adr': adr,
        'bbw': bbw,
        'chandelier_long': chandelier_long,
        'chandelier_short': chandelier_short,
        'trix': qu.calculate_trix,
        'yzv': qu.calculate_yang_zhang_volatility,
        'perc_k': perc_k,
        'perc_d': perc_d,
        'a_d_line': qu.calculate_accumulation_distribution_line,
        'swing_index': qu.calculate_swing_index,
        'awesome': awesome,
        'mfi': mfi,
        'mfr': mfr,
        'pvo': pvo,
        'pvo_signal': pvo_signal,
        'pvo_hist': pvo_hist,
        'tsi': tsi,
        'tsi_hist': tsi_hist,
        'tsi_signal': tsi_signal,
        'dymind': dymind,
        'hawkins': hawkins,
        'di_plus': di_plus,
        'di_minus': di_minus,
        'hmac': hmac,
        't3': qu.calculate_T3,
        'tema': qu.calculate_TEMA,
        'vwap': qu.calculate_VWAP,
        'zigzag': zigzag,
        'cfo': qu.calculate_chande_forecast_oscillator,
        'vi_plus': vi_plus,
        'vi_minus': vi_minus,
        'schaff': qu.calculate_schaff_trend_cycle,
        'mcc_osc': qu.calculate_mcclellan_oscillator,
        'mcc_si': qu.calculate_mcclellan_summation_index,
        'coppock': qu.calculate_coppock_curve,
        'prings_kst': prings_kst,
        'prings_kst_signal': prings_kst_signal,
        'prings_kst_hist': prings_kst_hist,
        'ama': qu.calculate_adaptive_moving_average,
        'wilders': qu.calculate_wilders_smoothing,
        'zero_lag_ema': qu.calculate_zero_lag_ema,
        'vpt': qu.calculate_volume_price_trend,
        'dei_sell': dei_sell,
        'dei_hold': dei_hold
        }


INDICATOR_FUNCTIONS: Dict[str, Callable[[DataFrame], Series]] = indicator_def.key_function_def
