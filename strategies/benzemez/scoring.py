"""
Canonical confidence scoring — single source of truth for backtest and live.

Both bot.py (backtest) and live/engine.py import from here.
Do not replicate this logic elsewhere.
"""
from __future__ import annotations

import pandas as pd
from .trend import TrendStrategy
from .breakout import BreakoutStrategy


def _zscore_last(series: pd.Series, lookback: int = 50) -> float:
    window = series.dropna().tail(lookback)
    if len(window) < 10:
        return 0.0
    return (float(window.iloc[-1]) - float(window.mean())) / max(float(window.std()), 1e-10)


def score_signal(df: pd.DataFrame, signal: int, raw_score: float = 0.0) -> float:
    """
    Confidence score for `signal` (+1 or -1) on the last bar of `df`.
    Returns 0.0 if signal == 0.

    Factors (max weighted sum = 8.5):
      F1  EMA-spread z-score          ≥1σ in signal direction          +1.0
      F2  TrendStrategy+Breakout agree both agree / one agrees         +1.5 / +0.75
      F3  MACD-hist-pct z-score       ≥0.5σ in signal direction        +1.0
      F4  Volume ratio                ≥1.5 / ≥1.0                      +1.5 / +0.75
      F5  ATR expanding               atr_pct[-1] > atr_pct[-2]        +0.75
      F6  5-bar momentum              ret5 in signal direction          +1.0–1.5
      F7  RSI not overextended        long<65 / short>35               +0.75
      F8  MACD-hist direction         rising+positive / falling+neg     +0.75
      F9  Bollinger band position     long<0.35 / short>0.65           +0.50
    """
    if signal == 0 or len(df) < 10:
        return 0.0

    factors = 0.0

    # F1 — EMA-spread z-score (ADX proxy, price-scale-free)
    if 'ema_9_21_ratio' in df.columns and len(df) >= 20:
        z = _zscore_last(df['ema_9_21_ratio'])
        if (signal == 1 and z >= 1.0) or (signal == -1 and z <= -1.0):
            factors += 1.0

    # F2 — trend & breakout strategy agreement
    try:
        t_sig = int(TrendStrategy().generate_signals(df).iloc[-1])
        b_sig = int(BreakoutStrategy().generate_signals(df).iloc[-1])
        if t_sig == signal and b_sig == signal:
            factors += 1.5
        elif t_sig == signal or b_sig == signal:
            factors += 0.75  # partial credit — mirrors bot.py _score_signal
    except Exception:
        pass

    # F3 — MACD-histogram z-score (price-scale-free)
    if 'macd_hist_pct' in df.columns and len(df) >= 20:
        z = _zscore_last(df['macd_hist_pct'])
        if (signal == 1 and z >= 0.5) or (signal == -1 and z <= -0.5):
            factors += 1.0

    # F4 — volume above rolling mean
    if 'vol_ratio' in df.columns:
        vr = float(df['vol_ratio'].iloc[-1])
        if vr >= 1.5:
            factors += 1.5
        elif vr >= 1.0:
            factors += 0.75

    # F5 — ATR expansion
    if 'atr_pct' in df.columns and len(df) >= 2:
        if float(df['atr_pct'].iloc[-1]) > float(df['atr_pct'].iloc[-2]):
            factors += 0.75

    # F6 — 5-bar close momentum in signal direction
    if len(df) >= 5:
        closes = df['close']
        ret5 = float(closes.iloc[-1] / closes.iloc[-5] - 1)
        if signal == 1 and ret5 > 0:
            factors += 1.0 + min(ret5 * 5, 0.5)
        elif signal == -1 and ret5 < 0:
            factors += 1.0 + min(abs(ret5) * 5, 0.5)

    # F7 — RSI not overextended
    if 'rsi_14' in df.columns:
        rsi = float(df['rsi_14'].iloc[-1])
        if signal == 1 and rsi < 65:
            factors += 0.75
        elif signal == -1 and rsi > 35:
            factors += 0.75

    # F8 — MACD-hist direction and sign
    if 'macd_hist_pct' in df.columns and len(df) >= 2:
        h_now  = float(df['macd_hist_pct'].iloc[-1])
        h_prev = float(df['macd_hist_pct'].iloc[-2])
        if signal == 1 and h_now > h_prev and h_now > 0:
            factors += 0.75
        elif signal == -1 and h_now < h_prev and h_now < 0:
            factors += 0.75

    # F9 — Bollinger band position (mean-reversion confirmation)
    if 'bb_pct' in df.columns:
        bp = float(df['bb_pct'].iloc[-1])
        if signal == 1 and bp < 0.35:
            factors += 0.5
        elif signal == -1 and bp > 0.65:
            factors += 0.5

    return round(min(factors / 8.5, 1.0), 4)


def signal_to_tier(score: float) -> str:
    """Map confidence score (0.0–1.0) to sizing tier."""
    if score < 0.20:
        return 'skip'
    if score < 0.55:
        return 'medium'
    if score < 0.75:
        return 'high'
    return 'ultra'
