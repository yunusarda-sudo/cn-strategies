"""
Performance metrics.  All functions are pure — no side effects.
"""
import numpy as np
import pandas as pd


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    median_td = pd.Series(index).diff().median()
    seconds   = median_td.total_seconds()
    return 365.25 * 24 * 3600 / seconds


def sharpe(equity: pd.Series, risk_free: float = 0.0) -> float:
    r = equity.pct_change().dropna()
    std = r.std()
    if std == 0 or not np.isfinite(std):
        return 0.0
    ppy = _periods_per_year(equity.index)
    value = (r.mean() - risk_free / ppy) / std * np.sqrt(ppy)
    return float(value) if np.isfinite(value) else 0.0


def sortino(equity: pd.Series, risk_free: float = 0.0) -> float:
    r  = equity.pct_change().dropna()
    dd = r[r < 0]
    std = dd.std()
    if std == 0 or not np.isfinite(std):
        return 0.0
    ppy = _periods_per_year(equity.index)
    value = (r.mean() - risk_free / ppy) / std * np.sqrt(ppy)
    return float(value) if np.isfinite(value) else 0.0


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max
    return float(dd.min())


def calmar(equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    ppy    = _periods_per_year(equity.index)
    ann_r  = (equity.iloc[-1] / equity.iloc[0]) ** (ppy / len(equity)) - 1
    value = ann_r / abs(mdd)
    return float(value) if np.isfinite(value) else 0.0


def win_rate(trades: pd.DataFrame) -> float:
    if len(trades) == 0 or 'pnl' not in trades:
        return 0.0
    return float((trades['pnl'] > 0).mean())


def profit_factor(trades: pd.DataFrame) -> float:
    if len(trades) == 0 or 'pnl' not in trades:
        return 0.0
    wins   = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
    losses = trades.loc[trades['pnl'] < 0, 'pnl'].abs().sum()
    if losses == 0:
        return float('inf') if wins > 0 else 0.0
    return float(wins / losses)


def compute_all(equity: pd.Series, trades: pd.DataFrame) -> dict:
    if len(equity) < 2:
        return {}
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    return {
        'total_return':  total_return,
        'sharpe':        sharpe(equity),
        'sortino':       sortino(equity),
        'calmar':        calmar(equity),
        'max_drawdown':  max_drawdown(equity),
        'win_rate':      win_rate(trades),
        'profit_factor': profit_factor(trades),
        'n_trades':      len(trades),
    }
