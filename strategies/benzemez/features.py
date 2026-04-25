"""
Feature engineering — all indicators use only past bars (no look-ahead).

Rolling windows naturally exclude the current bar when combined with .shift(1)
where the current bar's value would contaminate the signal.
"""
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    # --- returns ---
    for n in (1, 5, 10, 20):
        df[f'ret_{n}'] = c.pct_change(n)

    # --- SMAs & normalised distance ---
    for n in (10, 20, 50, 100, 200):
        sma = c.rolling(n).mean()
        df[f'sma_{n}']       = sma
        df[f'sma_{n}_ratio'] = c / sma - 1

    # --- EMAs ---
    for n in (9, 21, 55):
        df[f'ema_{n}'] = c.ewm(span=n, adjust=False).mean()

    df['ema_9_21_ratio']  = df['ema_9']  / df['ema_21'] - 1
    df['ema_21_55_ratio'] = df['ema_21'] / df['ema_55'] - 1

    # --- RSI ---
    df['rsi_14'] = _rsi(c, 14)

    # --- MACD ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['macd_pct']        = df['macd']        / c
    df['macd_signal_pct'] = df['macd_signal'] / c
    df['macd_hist_pct']   = df['macd_hist']   / c

    # --- Bollinger Bands ---
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_pct']   = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width']  = (df['bb_upper'] - df['bb_lower']) / bb_mid

    # --- ATR ---
    df['atr_14'] = _atr(h, l, c, 14)
    df['atr_pct'] = df['atr_14'] / c

    # --- Volume ---
    df['vol_ratio'] = v / v.rolling(20).mean()

    # --- Candle shape ---
    df['hl_pct'] = (h - l) / c

    # --- Donchian channels — shifted by 1 to exclude the current bar ---
    df['donchian_high'] = h.rolling(20).max().shift(1)
    df['donchian_low']  = l.rolling(20).min().shift(1)

    return df.dropna()
