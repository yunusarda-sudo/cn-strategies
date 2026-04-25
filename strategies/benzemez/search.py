"""
Hyperparameter optimiser — wraps Optuna.

Composite objective (production spec):
  score = 0.40 × return
        + 0.25 × norm_sharpe        (Sharpe / 3 — saturates at ~3σ strategies)
        + 0.20 × norm_calmar        (Calmar / 5 — saturates at high ratio)
        - 0.10 × |max_drawdown|     (drawdown fraction, 0–1)
        - 0.05 × overtrading        (excess trades beyond 1 per N bars)
        - overfit_penalty           (2× gap if train/val diff > 5%)
        - conc_penalty              (0.5 if ≤2 trades drive >80% of profit)
        - 10  × liq_count           (hard loss if DD < -35%)

Hard constraints (trial rejected immediately):
  - TrendStrategy: fast < slow
  - exit: take_profit > stop_loss
  - train_return - val_return > 20%  (curve-fitting, not skill)
  - val trades < _MIN_VAL_TRADES
  - total trades < _MIN_TRADES
"""
from __future__ import annotations

import numpy as np
import optuna
from typing import Type, Literal

import pandas as pd

from .engine import run as backtest_run
from .metrics import max_drawdown, sharpe as compute_sharpe, calmar as compute_calmar
from .base import Strategy

_MAX_DD_FLOOR   = -0.35   # hard reject threshold — matches 35% drawdown rule
_MIN_TRADES     = 4       # minimum total trades (train + val)
_MIN_VAL_TRADES = 2       # minimum val trades
_INNER_SPLIT    = 0.80    # 80% inner train, 20% inner val (time-ordered, never shuffled)

# Overtrading: allow at most 1 trade per N bars on the val set before penalising
_OVERTRADE_BARS_PER_TRADE = 15


def _build_params(trial: optuna.Trial, space: dict) -> dict:
    params = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == 'int':
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == 'float':
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == 'categorical':
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


def _composite_score(
    val_eq: pd.Series,
    val_trades: pd.DataFrame,
    val_return: float,
    val_mdd: float,
    train_return: float,
    n_val_bars: int,
) -> float:
    """
    Balanced composite metric per user specification:
      0.40 × return
    + 0.25 × norm_sharpe
    + 0.20 × norm_calmar
    - 0.10 × |mdd|
    - 0.05 × overtrading_penalty
    - overfit_penalty
    - conc_penalty
    - 10 × liq_count
    """
    # --- Sharpe and Calmar (normalised to ~[−1, 1] range) ---
    val_sharpe = compute_sharpe(val_eq)
    val_calmar = compute_calmar(val_eq)

    # Saturate: Sharpe 3 ≈ excellent; Calmar 5 ≈ excellent
    norm_sharpe = float(np.clip(val_sharpe / 3.0, -1.0, 1.0))
    norm_calmar = float(np.clip(val_calmar / 5.0, -1.0, 1.0))

    # --- Overtrading penalty ---
    max_trades = max(2, n_val_bars / _OVERTRADE_BARS_PER_TRADE)
    overtrading = max(0.0, len(val_trades) / max_trades - 1.0)

    # --- Liquidation proxy ---
    liq_count = 1 if val_mdd < -0.35 else 0

    # --- Overfit penalty (train/val return gap beyond 5% tolerance) ---
    overfit_gap     = max(0.0, train_return - val_return - 0.05)
    overfit_penalty = 2.0 * overfit_gap

    # --- Concentration penalty (≤2 trades drive >80% of gross val profit) ---
    conc_penalty = 0.0
    pos_pnl = val_trades.loc[val_trades['pnl'] > 0, 'pnl'] if len(val_trades) else pd.Series([], dtype=float)
    if len(pos_pnl) >= 1 and pos_pnl.sum() > 0:
        top2_share = pos_pnl.nlargest(2).sum() / pos_pnl.sum()
        if top2_share > 0.80:
            conc_penalty = 0.5

    score = (
        0.40 * val_return
        + 0.25 * norm_sharpe
        + 0.20 * norm_calmar
        - 0.10 * abs(val_mdd)
        - 0.05 * overtrading
        - overfit_penalty
        - conc_penalty
        - 10.0 * liq_count
    )
    return float(score)


def optimize(
    df: pd.DataFrame,
    strategy_class: Type[Strategy],
    n_trials: int           = 300,
    initial_capital: float  = 10_000.0,
    stop_loss_pct: float    = 1.0,
    take_profit_pct: float  = 2.0,
    commission_pct: float   = 0.001,
    position_pct: float     = 1.0,
    optimize_exits: bool    = True,
    objective: Literal['calmar', 'sharpe', 'return'] = 'calmar',
    seed: int               = 42,
    n_jobs: int             = 1,
    show_progress_bar: bool = True,
) -> dict:
    """
    Returns best parameter dict including optimised stop/target levels
    when optimize_exits=True.

    Every trial evaluates on an inner validation split (last 20% of df,
    time-ordered) to prevent in-sample overfitting.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = strategy_class().get_param_space()

    def objective_fn(trial: optuna.Trial) -> float:
        s      = strategy_class()
        params = _build_params(trial, space)

        # Enforce fast < slow for TrendStrategy
        if 'fast' in params and 'slow' in params:
            if params['fast'] >= params['slow']:
                return -5.0

        sl = trial.suggest_float('stop_loss_pct',   0.3, 4.0) if optimize_exits else stop_loss_pct
        tp = trial.suggest_float('take_profit_pct', 0.6, 8.0) if optimize_exits else take_profit_pct

        if tp <= sl:
            return -5.0

        # Time-ordered inner split — NEVER shuffle
        cut         = int(len(df) * _INNER_SPLIT)
        inner_train = df.iloc[:cut]
        inner_val   = df.iloc[cut:]

        if len(inner_train) < 150 or len(inner_val) < 50:
            return -5.0

        s.set_params(params)

        # Generate signals on the FULL df so indicators have correct history
        # at the split boundary (causal indicators — no look-ahead).
        all_signals = s.generate_signals(df)
        train_sig   = all_signals.iloc[:cut]
        val_sig     = all_signals.iloc[cut:]

        # --- Evaluate on inner validation (primary score source) ---
        val_eq, val_tr = backtest_run(
            inner_val, val_sig, initial_capital,
            position_pct=position_pct,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            commission_pct=commission_pct,
        )

        if len(val_tr) < _MIN_VAL_TRADES:
            return -5.0

        val_mdd = max_drawdown(val_eq)

        # Hard reject on catastrophic drawdown
        if val_mdd < _MAX_DD_FLOOR:
            return val_mdd

        val_return = float(val_eq.iloc[-1] / val_eq.iloc[0] - 1)

        # --- Evaluate on inner train (for overfit detection only) ---
        train_eq, train_tr = backtest_run(
            inner_train, train_sig, initial_capital,
            position_pct=position_pct,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            commission_pct=commission_pct,
        )
        train_return = float(train_eq.iloc[-1] / train_eq.iloc[0] - 1)

        # Reject if total trade count (train + val) is still too low
        if len(train_tr) + len(val_tr) < _MIN_TRADES:
            return -5.0

        # Hard overfit constraint: a gap > 20% is curve-fitting, not skill.
        if train_return - val_return > 0.20:
            return -5.0

        score = _composite_score(
            val_eq, val_tr, val_return, val_mdd,
            train_return, len(inner_val),
        )
        return float(score) if np.isfinite(score) else -5.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

    best = study.best_params.copy()

    if not optimize_exits:
        best['stop_loss_pct']   = stop_loss_pct
        best['take_profit_pct'] = take_profit_pct

    return best
