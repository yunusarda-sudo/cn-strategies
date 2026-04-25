"""
Ensemble signal combiner.

Each regime gets its own weight vector [trend_w, reversion_w, breakout_w].
Weights were chosen based on strategy-regime affinity:
  - trending_up/down → trend + breakout dominate
  - ranging          → reversion dominates
  - volatile         → breakout + slight trend; reversion dangerous

Weighted score → discretised to {-1, 0, +1} via adaptive thresholds:
  long_threshold  = rolling 75th percentile of score (last 100 bars)
  short_threshold = rolling 25th percentile of score (last 100 bars)
Fixed thresholds are used as fallback for bars before the rolling window fills.
"""
from __future__ import annotations

import pandas as pd

from .regime import detect as detect_regime, aggressiveness as regime_agg


# [TrendStrategy, ReversionStrategy, BreakoutStrategy, MomentumStrategy]
# Each regime gets weights that match the regime's most reliable strategy type.
# Weights are normalised at runtime to the actual number of strategies provided.
_DEFAULT_WEIGHTS: dict[str, list[float]] = {
    'trending_up':   [0.40, 0.05, 0.30, 0.25],   # trend + momentum + breakout
    'trending_down': [0.25, 0.05, 0.45, 0.25],   # breakout + momentum; reversion unreliable
    'ranging':       [0.08, 0.70, 0.10, 0.12],   # reversion dominates; momentum helps
    'volatile':      [0.15, 0.05, 0.55, 0.25],   # breakout + momentum; reversion risky
    'sideways':      [0.05, 0.65, 0.20, 0.10],   # reversion with slight breakout
}

_ROLLING_WINDOW = 100   # bars used to compute adaptive percentile thresholds
_MIN_PERIODS    = 20    # minimum bars before adaptive mode kicks in


class EnsembleCombiner:
    def __init__(
        self,
        strategies: list,
        weights: dict[str, list[float]] | None = None,
        threshold: float = 0.12,        # fallback for startup period (<20 bars)
        trend_threshold: float = 0.08,  # kept for backward-compat; unused in adaptive path
        regime_lookback: int = 50,
    ):
        self.strategies       = strategies
        self.weights          = weights or _DEFAULT_WEIGHTS
        self.threshold        = threshold
        self.trend_threshold  = trend_threshold
        self.regime_lookback  = regime_lookback

    def _compute_score(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Returns (regimes, score) — shared by combine / combine_with_scores.

        Score is scaled by per-regime aggressiveness (0.4–1.0) so sideways
        and volatile bars produce weaker signals → fewer/smaller trades.
        """
        regimes  = detect_regime(df, lookback=self.regime_lookback)
        agg      = regime_agg(regimes)       # per-bar multiplier [0.4, 1.0]
        all_sigs = [s.generate_signals(df).astype(float) for s in self.strategies]
        n_strats = len(all_sigs)
        score    = pd.Series(0.0, index=df.index)
        for regime, w_full in self.weights.items():
            mask = regimes == regime
            if not mask.any():
                continue
            # Use only as many weights as there are strategies; renormalise to sum=1
            w = list(w_full[:n_strats])
            total = sum(w) or 1.0
            w = [x / total for x in w]
            for i, sig in enumerate(all_sigs):
                score[mask] += sig[mask] * w[i]
        # Scale by aggressiveness — suppresses entry signals in hostile regimes
        score = score * agg
        return regimes, score

    def _adaptive_thresholds(
        self, score: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Rolling 75th/25th percentile thresholds; fixed fallback before window fills."""
        roll = score.rolling(_ROLLING_WINDOW, min_periods=_MIN_PERIODS)
        long_th  = roll.quantile(0.75).fillna(self.threshold)
        short_th = roll.quantile(0.25).fillna(-self.threshold)
        return long_th, short_th

    def combine(self, df: pd.DataFrame) -> pd.Series:
        _, score            = self._compute_score(df)
        long_th, short_th   = self._adaptive_thresholds(score)
        result = pd.Series(0, index=df.index, dtype=int)
        result[score > long_th]  =  1
        result[score < short_th] = -1
        return result

    def combine_with_scores(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """Return (signals, raw_scores) in one pass — avoids double computation.

        raw_scores is the weighted ensemble score before thresholding.
        Callers can feed raw_scores into backtest.engine.run() for
        confidence-tier position sizing that mirrors live _score_signal logic.
        """
        _, score            = self._compute_score(df)
        long_th, short_th   = self._adaptive_thresholds(score)
        result = pd.Series(0, index=df.index, dtype=int)
        result[score > long_th]  =  1
        result[score < short_th] = -1
        return result, score

    def last_signal(self, df: pd.DataFrame) -> int:
        """Convenience for live engine — returns the signal for the latest bar."""
        return int(self.combine(df).iloc[-1])

    def last_raw_score(self, df: pd.DataFrame) -> float:
        """Return the weighted score (before thresholding) for the latest bar."""
        _, score = self._compute_score(df)
        return float(score.iloc[-1])
