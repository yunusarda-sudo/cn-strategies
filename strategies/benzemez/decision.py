"""
Decision filter layer — single source of truth for trade gating.

Called from bot.py predict() only. Has no persistent state — all state
lives in BenzemezStrategy so the cnlib interface stays clean.

Gates (applied in order in bot.py):
  1. Regime     — CHOPPY / LOW_VOLATILITY → hard block
  2. Confidence — conf < MIN_CONFIDENCE   → skip
  3. Edge       — combined ATR × conf < fee threshold → skip
  4. Cooldown   — bars since last close   → handled in bot.py state
  5. Trade cap  — rolling 100-bar count   → handled in bot.py state

Allocation scaling (multiplicative, applied after all gates pass):
  confidence_alloc_mult × regime_alloc_mult × damage_alloc_mult
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .regime import detect as _detect_raw

# ── Decision regime classes ───────────────────────────────────────────────────
TRENDING        = 'TRENDING'
CHOPPY          = 'CHOPPY'
HIGH_VOLATILITY = 'HIGH_VOLATILITY'
LOW_VOLATILITY  = 'LOW_VOLATILITY'
NORMAL          = 'NORMAL'

_RAW_TO_DECISION: dict[str, str] = {
    'trending_up':   TRENDING,
    'trending_down': TRENDING,
    'ranging':       NORMAL,
    'volatile':      HIGH_VOLATILITY,
    'sideways':      CHOPPY,
}

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_CONFIDENCE        = 0.55  # default minimum confidence to open a trade
MIN_CONFIDENCE_STRONG = 0.45  # reduced threshold when raw ensemble score > 0.4
COOLDOWN_BARS         = 3     # bars to wait after any position close
MAX_TRADES_100_BARS = 10     # rolling 100-bar trade cap per coin
FEE_PER_LEG         = 0.001  # 0.1% taker fee per side
MIN_ATR_FEE_MULT    = 3.0    # combined edge must cover ≥ 3× round-trip fee

# ER constants — used as adaptive bounds, not fixed thresholds
ER_LOOKBACK     = 30    # bars for single ER computation
ER_FLOOR        = 0.08  # adaptive threshold never goes below this
ER_CEIL         = 0.18  # adaptive threshold never goes above this


# ── Rolling helpers (stateless, vectorized) ───────────────────────────────────

def _rolling_er_series(closes: pd.Series, lookback: int = ER_LOOKBACK) -> pd.Series:
    """
    Vectorized Kaufman Efficiency Ratio — no future leakage.
    Returns a series aligned to `closes`.
    """
    net  = closes.diff(lookback).abs()
    path = closes.diff().abs().rolling(lookback, min_periods=5).sum()
    return (net / path.clip(lower=1e-10)).fillna(0.5)


def _adaptive_threshold(
    series: pd.Series,
    pct: float,
    lookback: int,
    floor: float,
    ceil: float,
) -> float:
    """
    Rolling-percentile threshold: P{pct} of the last `lookback` non-null values,
    clamped to [floor, ceil].

    Falls back to `floor` when fewer than 20 non-null observations are available.
    Coin-agnostic — no hardcoded asset-specific values.
    """
    recent = series.dropna().tail(lookback)
    if len(recent) < 20:
        return floor
    return float(np.clip(np.percentile(recent, pct), floor, ceil))


# ── Efficiency Ratio (scalar, for per-bar classify_regime) ───────────────────

def _efficiency_ratio(df: pd.DataFrame, lookback: int = ER_LOOKBACK) -> float:
    """
    Kaufman Efficiency Ratio for the last `lookback` bars of df.
    Returns 0.5 (neutral) if insufficient data.
    """
    if len(df) < lookback:
        return 0.5
    closes = df['close'].tail(lookback + 1)
    net    = abs(float(closes.iloc[-1]) - float(closes.iloc[0]))
    path   = float(closes.diff().abs().sum())
    return net / max(path, 1e-10)


# ── Regime classification ─────────────────────────────────────────────────────

def classify_regime(df: pd.DataFrame, lookback: int = 50) -> str:
    """
    Map the latest bar to one of 5 decision regime classes using 4 adaptive
    rolling-percentile thresholds. No coin-specific constants.

    Adaptive rules (applied in priority order):
      1. ER   < rolling P25 of ER history (clamped [0.08, 0.18])
               → CHOPPY  (unless base regime = TRENDING)
      2. ATR  < rolling P20 of ATR history (clamped [1.5×fee, 0.04])
               → LOW_VOLATILITY
      3. ATR compressing: ATR < 65% of 20-bar mean, base=NORMAL
               → CHOPPY
      4. EMA spread < rolling P20 of spread history (if available)
               → CHOPPY  (unless already TRENDING/HIGH_VOL)
    """
    raw      = _detect_raw(df, lookback=lookback)
    decision = _RAW_TO_DECISION.get(str(raw.iloc[-1]), NORMAL)
    trending = (decision == TRENDING)

    # ── Rule 1: Efficiency Ratio — primary chop detector ─────────────────────
    if len(df) >= ER_LOOKBACK + 20:
        er_series = _rolling_er_series(df['close'])
        er_thresh  = _adaptive_threshold(
            er_series, pct=25.0, lookback=200, floor=ER_FLOOR, ceil=ER_CEIL
        )
        if _efficiency_ratio(df) < er_thresh and not trending:
            decision = CHOPPY

    # ── Rule 2: ATR percentage — low-volatility / flat-market detector ───────
    round_trip = 2.0 * FEE_PER_LEG
    if 'atr_pct' in df.columns and len(df) >= 20:
        atr_now   = float(df['atr_pct'].iloc[-1])
        atr_thresh = _adaptive_threshold(
            df['atr_pct'], pct=15.0, lookback=200,
            floor=round_trip * 1.5, ceil=0.04,
        )
        if atr_now < atr_thresh:
            decision = LOW_VOLATILITY
        elif decision == NORMAL:
            # ATR compressing within a normal regime → choppy
            atr_20 = float(df['atr_pct'].tail(20).mean())
            if atr_20 > 0 and atr_now < atr_20 * 0.65:
                decision = CHOPPY

    # ── Rule 3: EMA spread — flat-price / no-direction detector ──────────────
    if ('ema_9_21_ratio' in df.columns and len(df) >= 20
            and decision not in (CHOPPY, LOW_VOLATILITY, TRENDING)):
        ema_spread = (df['ema_9_21_ratio'] - 1.0).abs()
        spread_thresh = _adaptive_threshold(
            ema_spread, pct=20.0, lookback=200, floor=0.0005, ceil=0.015
        )
        if float(ema_spread.iloc[-1]) < spread_thresh:
            decision = CHOPPY

    return decision


# ── Entry gates ───────────────────────────────────────────────────────────────

def can_enter(regime: str) -> bool:
    """Hard gate: block new positions in hostile regimes."""
    return regime not in (CHOPPY, LOW_VOLATILITY)


def has_min_edge(df: pd.DataFrame, leverage: int = 1, conf: float = 1.0) -> bool:
    """
    Combined ATR + confidence edge gate.

    Neither ATR nor confidence alone is sufficient — both must contribute.

    Formula:
      combined = (atr_pct × leverage / round_trip) × (conf / MIN_CONFIDENCE)
      gate     : combined ≥ MIN_ATR_FEE_MULT

    Effect:
      - Ultra-confidence (0.85+) can offset moderate ATR (but not negligible ATR)
      - Borderline confidence (0.55) requires strong ATR
      - Very low ATR is always blocked regardless of confidence
    """
    round_trip = 2.0 * FEE_PER_LEG
    if 'atr_pct' not in df.columns:
        return conf >= MIN_CONFIDENCE  # fallback: confidence gate only
    atr_pct    = float(df['atr_pct'].iloc[-1])
    atr_ratio  = (atr_pct * leverage) / max(round_trip, 1e-10)
    conf_ratio = conf / max(MIN_CONFIDENCE, 1e-10)  # ≥ 1.0 (Gate 2 already passed)
    return atr_ratio * conf_ratio >= MIN_ATR_FEE_MULT


# ── Allocation multipliers ────────────────────────────────────────────────────

def confidence_alloc_mult(conf: float) -> float:
    """
    Scale allocation by confidence score.
      conf < 0.55        → 0.00  (blocked before reaching this)
      0.55 ≤ conf < 0.70 → 0.40
      0.70 ≤ conf < 0.85 → 0.75
      conf ≥ 0.85        → 1.00
    """
    if conf >= 0.85:
        return 1.00
    if conf >= 0.70:
        return 0.75
    if conf >= 0.55:
        return 0.40
    return 0.00


def regime_alloc_mult(regime: str) -> float:
    """
    Scale allocation by regime.
      TRENDING        → 1.00  (full size — directional edge)
      NORMAL          → 0.85  (slight reduction)
      HIGH_VOLATILITY → 0.60  (reduced — false signal risk)
      CHOPPY          → 0.00  (blocked upstream by can_enter)
      LOW_VOLATILITY  → 0.00  (blocked upstream by can_enter)
    """
    return {
        TRENDING:        1.00,
        NORMAL:          0.85,
        HIGH_VOLATILITY: 0.60,
        CHOPPY:          0.00,
        LOW_VOLATILITY:  0.00,
    }.get(regime, 0.85)


def damage_alloc_mult(consecutive_losses: int, drawdown: float = 0.0) -> float:
    """
    Reduce allocation after consecutive per-coin losses.

    Consecutive losses (per coin):
      0  → 1.00  (no reduction)
      1  → 0.80  (mild caution)
      2+ → 0.55  (significant reduction)

    Portfolio drawdown supplement:
      DD > 15% → additional × 0.80 on top of the above.
    """
    base = (0.55 if consecutive_losses >= 2 else
            0.80 if consecutive_losses == 1 else
            1.00)
    if drawdown > 0.15:
        base *= 0.80
    return round(base, 2)
