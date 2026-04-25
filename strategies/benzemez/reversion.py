"""
Mean-reversion strategy: RSI extremes confirmed by Bollinger Band touch.

Logic:
  BUY  — RSI < oversold threshold AND close <= lower BB
  SELL — RSI > overbought threshold AND close >= upper BB

Wins in sideways, high-chop markets.
Uncorrelated with TrendStrategy (loses in strong trends).
"""
import numpy as np
import pandas as pd

from .base import Strategy


class ReversionStrategy(Strategy):
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c = df['close']

        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        mid    = c.rolling(self.bb_period).mean()
        std    = c.rolling(self.bb_period).std()
        lower  = mid - self.bb_std * std
        upper  = mid + self.bb_std * std

        # Only short when price is below the long-term trend EMA.
        # Without this guard, overbought RSI in a bull market triggers SELL,
        # which poisons the ensemble in 'ranging' regime (80% reversion weight).
        trend_ema = c.ewm(span=200, adjust=False).mean()

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[(rsi < self.rsi_oversold)  & (c <= lower * 1.005)] = 1
        signals[(rsi > self.rsi_overbought) & (c >= upper * 0.995) & (c < trend_ema)] = -1

        return signals

    def get_param_space(self) -> dict:
        return {
            'rsi_period':    ('int',   10,   21),
            'rsi_oversold':  ('float', 25.0, 38.0),
            'rsi_overbought':('float', 62.0, 82.0),
            'bb_period':     ('int',   14,   28),
            'bb_std':        ('float', 1.8,  2.8),
        }

    def set_params(self, params: dict):
        self.rsi_period     = params['rsi_period']
        self.rsi_oversold   = params['rsi_oversold']
        self.rsi_overbought = params['rsi_overbought']
        self.bb_period      = params['bb_period']
        self.bb_std         = params['bb_std']

    def get_params(self) -> dict:
        return {
            'rsi_period':     self.rsi_period,
            'rsi_oversold':   self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'bb_period':      self.bb_period,
            'bb_std':         self.bb_std,
        }
