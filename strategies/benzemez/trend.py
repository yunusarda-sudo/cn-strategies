"""
Trend-following strategy: EMA crossover filtered by long-term trend.

Logic:
  BUY  — fast EMA crosses above slow EMA AND close > trend EMA
  SELL — fast EMA crosses below slow EMA AND close < trend EMA

Both sides are filtered by the trend EMA — symmetric signal generation.
Wins in sustained directional moves.
Uncorrelated with ReversionStrategy (loses in ranging markets).
"""
import pandas as pd

from .base import Strategy


class TrendStrategy(Strategy):
    def __init__(self, fast: int = 9, slow: int = 21, trend: int = 200):
        self.fast = fast
        self.slow = slow
        self.trend = trend

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c = df['close']

        fast_ema = c.ewm(span=self.fast, adjust=False).mean()
        slow_ema = c.ewm(span=self.slow, adjust=False).mean()
        trend_ema = c.ewm(span=self.trend, adjust=False).mean()

        cross_up   = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        cross_down = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
        above_trend = c > trend_ema
        # B4 FIX: symmetric trend filter — short signals only fire in confirmed downtrend.
        # Without this filter, any EMA crossdown (even in a bull market) triggers a short,
        # generating false negatives that hurt ensemble Calmar score.
        below_trend = c < trend_ema

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[cross_up & above_trend]   = 1
        signals[cross_down & below_trend] = -1

        return signals

    def get_param_space(self) -> dict:
        return {
            'fast':  ('int',  5,  20),
            'slow':  ('int', 20,  60),
            'trend': ('int', 100, 400),
        }

    def set_params(self, params: dict):
        self.fast  = params['fast']
        self.slow  = params['slow']
        self.trend = params['trend']

    def get_params(self) -> dict:
        return {'fast': self.fast, 'slow': self.slow, 'trend': self.trend}
