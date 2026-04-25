"""
Volatility-breakout strategy: Donchian channel breach with ATR expansion + volume surge.

Logic:
  BUY  — close breaks above N-bar high AND ATR expanding AND volume spike
  SELL — close breaks below N-bar low (exit)

Wins when volatility compresses then explodes (earnings, macro events, ranges ending).
Uncorrelated with both trend and reversion strategies.
"""
import pandas as pd

from .base import Strategy


class BreakoutStrategy(Strategy):
    def __init__(
        self,
        donchian_period: int  = 20,
        atr_period: int       = 14,
        atr_quantile: float   = 0.60,   # ATR must be above this rolling percentile
        vol_quantile: float   = 0.65,   # Volume must be above this rolling percentile
    ):
        self.donchian_period = donchian_period
        self.atr_period      = atr_period
        self.atr_quantile    = atr_quantile
        self.vol_quantile    = vol_quantile

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c, h, l, v = df['close'], df['high'], df['low'], df['volume']

        # Shift by 1 so today's bar doesn't self-reference
        dc_high = h.rolling(self.donchian_period).max().shift(1)
        dc_low  = l.rolling(self.donchian_period).min().shift(1)

        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / self.atr_period, adjust=False).mean()

        lookback = self.atr_period * 3
        # Percentile thresholds — regime-adaptive, scale-independent
        atr_thresh = atr.rolling(lookback).quantile(self.atr_quantile)
        vol_thresh = v.rolling(lookback).quantile(self.vol_quantile)

        atr_expand = atr > atr_thresh
        high_vol   = v > vol_thresh

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[(c > dc_high) & high_vol & atr_expand] = 1
        # Short also requires ATR expansion — prevents excessive shorts in
        # consolidating bull markets where downside volume alone is unreliable.
        signals[(c < dc_low)  & high_vol & atr_expand] = -1

        return signals

    def get_param_space(self) -> dict:
        return {
            'donchian_period': ('int',   5,    30),
            'atr_period':      ('int',   5,    21),
            'atr_quantile':    ('float', 0.45, 0.85),
            'vol_quantile':    ('float', 0.45, 0.85),
        }

    def set_params(self, params: dict):
        self.donchian_period = params.get('donchian_period', self.donchian_period)
        self.atr_period      = params.get('atr_period',      self.atr_period)
        self.atr_quantile    = params.get('atr_quantile',    self.atr_quantile)
        self.vol_quantile    = params.get('vol_quantile',    self.vol_quantile)

    def get_params(self) -> dict:
        return {
            'donchian_period': self.donchian_period,
            'atr_period':      self.atr_period,
            'atr_quantile':    self.atr_quantile,
            'vol_quantile':    self.vol_quantile,
        }
