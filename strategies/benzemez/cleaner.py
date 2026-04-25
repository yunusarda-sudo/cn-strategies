"""
Removes bad bars: duplicates, zero prices, OHLC violations, price spikes.

Forward-fill is NOT applied to OHLC — filling open/high/low from a previous
bar violates OHLC semantics and corrupts ATR, Donchian channels, and
intra-bar stop/target checks in the backtester.
Small gaps in close-only data (max 3 consecutive) are filled; everything
else is dropped.
"""
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[~df.index.duplicated(keep='first')]

    positive = (df[['open', 'high', 'low', 'close']] > 0).all(axis=1)
    df = df[positive]

    valid_ohlc = (
        (df['high'] >= df['close']) &
        (df['low']  <= df['close']) &
        (df['high'] >= df['open'])  &
        (df['low']  <= df['open'])  &
        (df['high'] >= df['low'])
    )
    df = df[valid_ohlc]

    # Drop bars with a single-bar close move > 30 % (data error)
    spike = df['close'].pct_change().abs() > 0.30
    df = df[~spike]

    # Drop rows with any remaining NaN — do NOT forward-fill OHLC
    df = df.dropna()

    return df
