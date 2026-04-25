"""
Market regime detector.

Regimes (assigned in priority order — last wins):
  1. ranging      — default (low slope, normal volatility)
  2. sideways     — very low ATR + very low slope (flat, no edge)
  3. volatile     — vol > 1.5× historical average
  4. trending_up  — slope > +2% (overrides volatile)
  5. trending_down— slope < -2% (overrides volatile)

Each regime maps to a different aggressiveness level used by the ensemble
combiner and position sizer:

  Regime           Aggressiveness   Rationale
  ───────────────  ───────────────  ──────────────────────────────────
  trending_up      1.0              full size — high-confidence bias
  trending_down    0.8              slightly reduced — whipsaw risk
  ranging          0.7              medium — reversion works but slow
  volatile         0.6              reduced — false signals common
  sideways         0.4              minimum — flat markets eat spreads

The aggressiveness multiplier is applied to the tier allocation fraction
in bot.py _decide() and backtest/engine.py _score_bar() callers.
"""
import pandas as pd

_REGIME_AGGRESSIVENESS: dict[str, float] = {
    'trending_up':   1.0,
    'trending_down': 0.8,
    'ranging':       0.7,
    'volatile':      0.6,
    'sideways':      0.4,
}


def detect(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    c   = df['close']
    ret = c.pct_change()

    sma   = c.rolling(lookback).mean()
    slope = (sma - sma.shift(lookback // 2)) / sma.shift(lookback // 2)

    vol      = ret.rolling(lookback).std()
    vol_hist = vol.rolling(lookback * 3).mean()
    high_vol = vol > vol_hist * 1.5

    # Sideways: very tight slope AND very low volatility
    atr_pct = (df['atr_pct'] if 'atr_pct' in df.columns
               else (df['high'] - df['low']) / df['close'])
    atr_median = atr_pct.rolling(lookback).median()
    low_atr    = atr_pct < atr_median * 0.6
    flat_slope = slope.abs() < 0.005

    regime = pd.Series('ranging', index=df.index, dtype=object)

    # Apply in priority order — each overwrites the previous
    regime[high_vol]                     = 'volatile'
    regime[low_atr & flat_slope]         = 'sideways'
    regime[slope >  0.02]                = 'trending_up'
    regime[slope < -0.02]                = 'trending_down'

    return regime


def aggressiveness(regime_series: pd.Series) -> pd.Series:
    """Map a regime Series to per-bar aggressiveness multipliers (0.0–1.0)."""
    return regime_series.map(_REGIME_AGGRESSIVENESS).fillna(0.7)
