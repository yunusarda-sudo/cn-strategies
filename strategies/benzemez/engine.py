"""
Bar-by-bar backtester — aligned with live BenzemezStrategy execution.

Live-matching features (enabled via optional parameters):
  1. Tier-based position sizing   — confidence scored per bar, allocation mapped to
                                    small/normal/aggressive fractions (0.20/0.40/0.65).
                                    Mirrors bot.py _score_signal + _score_to_tier.
  2. ATR trailing stop            — ratchets stop in profit direction each bar.
                                    Mirrors config.ATR_TRAIL_MULT intention.
  3. Signal persistence           — replays cached non-zero signal for N bars after
                                    ensemble returns 0. Mirrors bot.py _apply_persistence.
  4. Drawdown risk mode           — RECOVERY at 18% DD (allocation × 0.70),
                                    FLAT at 30% DD (no new entries, force-close open).
                                    Mirrors bot.py _risk_mode.

Confidence scoring replicates bot.py _score_signal factors F1, F3–F9.
F2 (trend+breakout re-run) is omitted for bar-by-bar efficiency; without it
the maximum achievable confidence is 7.0/8.5 ≈ 0.82, which is slightly
conservative and maps correctly to the aggressive tier.

Legacy API preserved: pass tier_alloc=None to use fixed position_pct.

SL/TP checked at candle CLOSE — matches cnlib portfolio.py and live bot
_update_equity. Commission charged on both legs.

Returns:
    equity  — pd.Series, mark-to-market equity at each bar's close
    trades  — pd.DataFrame, one row per completed trade
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_zscore(arr: np.ndarray, window: int = 50, min_periods: int = 10) -> np.ndarray:
    """Causal rolling z-score — no future data. Pre-computed before the main loop."""
    s     = pd.Series(arr, dtype=float)
    mu    = s.rolling(window, min_periods=min_periods).mean()
    sigma = s.rolling(window, min_periods=min_periods).std().clip(lower=1e-10)
    return ((s - mu) / sigma).fillna(0.0).values


# ── Risk-mode thresholds (mirrors bot.py constants) ──────────────────────────
_FLAT_THRESHOLD      = 0.30   # peak-to-trough drawdown → all entries suppressed
_RECOVERY_THRESHOLD  = 0.20   # matches bot.py _DD_RECOVERY / config.DD_RECOVERY_PCT
_RECOVERY_ALLOC_MULT = 0.70   # allocation multiplier in RECOVERY mode

# ── Default tier allocations (matches config.TIER_ALLOC) ─────────────────────
# Max single-coin allocation capped at 0.30 per user spec.
DEFAULT_TIER_ALLOC: dict[str, float] = {
    'skip':   0.00,
    'medium': 0.15,
    'high':   0.25,
    'ultra':  0.30,
}


# ── Confidence helpers ────────────────────────────────────────────────────────

def _conf_to_tier(conf: float) -> str:
    """Maps raw confidence to sizing tier. Matches bot.py _score_to_tier."""
    if conf < 0.20:
        return 'skip'
    if conf < 0.55:
        return 'medium'
    if conf < 0.75:
        return 'high'
    return 'ultra'


def _score_bar(
    sig: int,
    bar_idx: int,
    closes: np.ndarray,
    ema_9_21_z: float,       # rolling z-score of ema_9_21_ratio (for F1)
    vol_ratio: float,
    atr_now: float,
    atr_lag: float,
    rsi: float,
    macd_hist_pct: float,    # current macd_hist_pct value (for F8 direction check)
    macd_hist_pct_z: float,  # rolling z-score of macd_hist_pct (for F3)
    macd_hist_prev: float,
    bb_pct: float,
) -> float:
    """
    Per-bar confidence score — replicates bot.py _score_signal (F1, F3–F9).

    F2 (TrendStrategy / BreakoutStrategy agreement) is intentionally omitted:
    re-running both strategies on every bar is O(n²) in the backtest loop.
    Without F2 the max reachable score is 7.0/8.5 ≈ 0.82.

    F1 and F3 now use rolling z-scores (pre-computed in run()) so they are
    price-scale-independent — same formula as bot.py _score_signal.
    """
    if sig == 0:
        return 0.0

    factors = 0.0

    # F1 — EMA-spread z-score ≥ 1σ in signal direction (mirrors bot.py F1)
    if (sig == 1 and ema_9_21_z >= 1.0) or (sig == -1 and ema_9_21_z <= -1.0):
        factors += 1.0

    # F3 — MACD-hist-pct z-score ≥ 0.5σ in signal direction (mirrors bot.py F3)
    if (sig == 1 and macd_hist_pct_z >= 0.5) or (sig == -1 and macd_hist_pct_z <= -0.5):
        factors += 1.0

    # F4 — volume above 20-bar SMA
    if vol_ratio >= 1.5:
        factors += 1.5
    elif vol_ratio >= 1.0:
        factors += 0.75

    # F5 — ATR expanding (atr_pct compares % of price — scale-free)
    if atr_lag > 0 and atr_now > atr_lag * 1.05:
        factors += 0.75

    # F6 — 5-bar close return in signal direction
    if bar_idx >= 4:
        ret5 = float(closes[bar_idx] / closes[bar_idx - 4] - 1)
        if sig == 1 and ret5 > 0:
            factors += 1.0 + min(ret5 * 5.0, 0.5)
        elif sig == -1 and ret5 < 0:
            factors += 1.0 + min(abs(ret5) * 5.0, 0.5)

    # F7 — RSI not overextended
    if sig == 1 and rsi < 65.0:
        factors += 0.75
    elif sig == -1 and rsi > 35.0:
        factors += 0.75

    # F8 — MACD-hist-pct direction (sign + trend check, price-scale-free)
    if sig == 1 and macd_hist_pct > macd_hist_prev and macd_hist_pct > 0.0:
        factors += 0.75
    elif sig == -1 and macd_hist_pct < macd_hist_prev and macd_hist_pct < 0.0:
        factors += 0.75

    # F9 — Bollinger band position (mean-reversion confirmation)
    if sig == 1 and bb_pct < 0.35:
        factors += 0.5
    elif sig == -1 and bb_pct > 0.65:
        factors += 0.5

    return min(factors / 8.5, 1.0)


# ── Main engine ───────────────────────────────────────────────────────────────

def run(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float         = 10_000.0,
    position_pct: float            = 0.95,        # legacy fallback (tier_alloc=None)
    stop_loss_pct: float           = 1.0,
    take_profit_pct: float         = 2.0,
    commission_pct: float          = 0.001,
    # ── Live-matching parameters (all optional; defaults = legacy behaviour) ──
    raw_scores: pd.Series | None   = None,         # ensemble weighted score per bar
    atr_trail_mult: float | None   = None,         # None → fixed SL, no trailing
    persistence_bars: int          = 0,            # 0 → no persistence
    tier_alloc: dict | None        = None,         # None → use position_pct
    # ── Profit levers ────────────────────────────────────────────────────────
    min_hold_bars: int             = 5,   # ignore signal flip for first N bars held
    enable_pyramiding: bool        = True,  # add 50 % on trend continuation
    pyramid_atr_trigger: float     = 1.5,  # profit ≥ this × ATR fires pyramid
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Execute a bar-by-bar backtest.

    Parameters
    ----------
    df               Feature-engineered OHLCV DataFrame (data/features.compute output).
                     Must contain 'close'. Optional columns atr_14, ema_9_21_ratio,
                     vol_ratio, rsi_14, macd_hist, bb_pct unlock live-matching features.
    signals          Ensemble signal per bar (+1 long, -1 short, 0 flat).
    raw_scores       Ensemble weighted score before thresholding — feed from
                     EnsembleCombiner.combine_with_scores() for accurate F1 scoring.
    atr_trail_mult   ATR multiplier for trailing stop (2.5 matches live config).
    persistence_bars Bars to replay cached non-zero signal (4 matches live bot).
    tier_alloc       Allocation fractions per tier, or None for legacy position_pct.
                     Use DEFAULT_TIER_ALLOC for live-matching behaviour.
    """
    use_live = tier_alloc is not None

    # ── Pre-extract numpy arrays (avoids repeated iloc in hot loop) ───────────
    closes = df['close'].values
    sigs   = signals.reindex(df.index).fillna(0).values
    idx    = df.index
    n      = len(df)

    def _arr(col: str) -> np.ndarray | None:
        return df[col].values if col in df.columns else None

    atr_a    = _arr('atr_14')
    ema_a    = _arr('ema_9_21_ratio')
    vol_a    = _arr('vol_ratio')
    rsi_a    = _arr('rsi_14')
    macd_a   = _arr('macd_hist_pct')
    bb_a     = _arr('bb_pct')
    score_a  = (raw_scores.reindex(df.index).fillna(0.0).values
                if raw_scores is not None else None)

    # Pre-compute rolling z-scores (causal, 50-bar window) for F1 and F3.
    # Done once here so the inner loop stays O(1) per bar.
    _zeros   = np.zeros(n)
    ema_z_a  = _rolling_zscore(ema_a,  window=50) if ema_a  is not None else _zeros
    macd_z_a = _rolling_zscore(macd_a, window=50) if macd_a is not None else _zeros

    def _f(arr: np.ndarray | None, i: int, default: float) -> float:
        if arr is None:
            return default
        v = arr[i]
        return default if np.isnan(v) else float(v)

    # ── State ─────────────────────────────────────────────────────────────────
    capital  = initial_capital
    peak_cap = initial_capital
    position = None   # dict when open, None when flat
    equity   = []
    trades   = []

    # Signal persistence state
    p_cache = 0
    p_age   = 0

    for i in range(n):
        price = float(closes[i])

        # ── 1. Signal persistence ─────────────────────────────────────────────
        raw_sig = int(sigs[i])
        if raw_sig != 0:
            # New signal (or counter-signal) — reset cache
            p_cache = raw_sig
            p_age   = 0
            sig     = raw_sig
        elif p_cache != 0 and p_age < persistence_bars:
            # Ensemble returned 0 but cached signal still valid
            sig   = p_cache
            p_age += 1
        else:
            p_cache = 0
            p_age   = 0
            sig     = 0

        # ── 2. Mark-to-market equity (needed for drawdown check) ──────────────
        if position is not None:
            unrealized = (
                (price - position['entry']) * position['qty'] * position['side']
            )
            cur_equity = capital + position['capital'] + unrealized
        else:
            cur_equity = capital

        # ── 3. Drawdown risk mode (mirrors bot.py _risk_mode) ─────────────────
        if cur_equity > peak_cap:
            peak_cap = cur_equity
        dd   = (peak_cap - cur_equity) / peak_cap if peak_cap > 0 else 0.0
        mode = (
            'FLAT'     if dd >= _FLAT_THRESHOLD     else
            'RECOVERY' if dd >= _RECOVERY_THRESHOLD else
            'NORMAL'
        )

        # FLAT: force-close any open position, then suppress all new entries
        if mode == 'FLAT' and position is not None:
            side      = position['side']
            raw_pnl   = (price - position['entry']) * position['qty'] * side
            exit_comm = price * position['qty'] * commission_pct
            net_pnl   = raw_pnl - position['entry_commission'] - exit_comm
            capital  += position['capital'] + raw_pnl - exit_comm
            trades.append({
                'entry_time': position['time'],
                'exit_time':  idx[i],
                'entry':      position['entry'],
                'exit':       price,
                'qty':        position['qty'],
                'side':       side,
                'pnl':        net_pnl,
                'reason':     'flat_mode',
                'tier':       position.get('tier', 'legacy'),
                'pyramided':  position.get('pyramided', False),
            })
            position = None

        # ── 4. ATR trailing stop — ratchet in profit direction ────────────────
        if position is not None and atr_trail_mult is not None and atr_a is not None:
            atr_now = _f(atr_a, i, 0.0)
            if atr_now > 0.0:
                if position['side'] == 1:
                    candidate = price - atr_trail_mult * atr_now
                    if candidate > position['stop']:
                        position['stop'] = candidate
                else:
                    candidate = price + atr_trail_mult * atr_now
                    if candidate < position['stop']:
                        position['stop'] = candidate

        # ── 4b. Pyramiding — add 50 % on confirmed trend continuation ─────────
        if (position is not None
                and enable_pyramiding
                and not position.get('pyramided', False)
                and capital > 0
                and atr_a is not None):
            atr_now = _f(atr_a, i, 0.0)
            if atr_now > 0.0:
                side   = position['side']
                profit = (price - position['entry']) * side
                if profit >= pyramid_atr_trigger * atr_now:
                    # raise stop to break-even before adding new risk
                    be = position['entry']
                    position['stop'] = (max(position['stop'], be) if side == 1
                                        else min(position['stop'], be))
                    add_alloc = min(position['capital'] * 0.50, capital)
                    if add_alloc > 0:
                        add_comm = add_alloc * commission_pct
                        position['qty']              += add_alloc / price
                        position['capital']          += add_alloc
                        position['entry_commission'] += add_comm
                        capital -= add_alloc + add_comm
                        position['pyramided'] = True

        # ── 5. SL / TP / signal exits at candle CLOSE ────────────────────────
        if position is not None:
            side      = position['side']
            bars_held = i - position.get('entry_bar', i)

            if side == 1:
                hit_stop   = price <= position['stop']
                # Fixed TP is disabled when trailing stop is active (let winners run)
                hit_target = (atr_trail_mult is None) and (price >= position['target'])
                exit_sig   = (sig in (-1, 0)) and (bars_held >= min_hold_bars)
            else:
                hit_stop   = price >= position['stop']
                hit_target = (atr_trail_mult is None) and (price <= position['target'])
                exit_sig   = (sig in (1, 0)) and (bars_held >= min_hold_bars)

            if hit_stop or hit_target or exit_sig:
                exit_price = (position['stop']   if hit_stop   else
                              position['target'] if hit_target else price)

                raw_pnl   = (exit_price - position['entry']) * position['qty'] * side
                exit_comm = exit_price * position['qty'] * commission_pct
                net_pnl   = raw_pnl - position['entry_commission'] - exit_comm
                capital  += position['capital'] + raw_pnl - exit_comm

                trades.append({
                    'entry_time': position['time'],
                    'exit_time':  idx[i],
                    'entry':      position['entry'],
                    'exit':       exit_price,
                    'qty':        position['qty'],
                    'side':       side,
                    'pnl':        net_pnl,
                    'reason':     ('stop'   if hit_stop   else
                                   'target' if hit_target else 'signal'),
                    'tier':       position.get('tier', 'legacy'),
                    'pyramided':  position.get('pyramided', False),
                })
                position = None

        # ── 6. Open new position ──────────────────────────────────────────────
        if position is None and sig in (1, -1) and capital > 0 and mode != 'FLAT':

            if use_live:
                # ── Confidence scoring → tier → allocation fraction ────────────
                ema_z  = float(ema_z_a[i])
                macd_z = float(macd_z_a[i])
                vr   = _f(vol_a,    i,     1.0)
                atr0 = _f(atr_a,    i,     0.0)
                alag = _f(atr_a,    i - 2, atr0) if i >= 2 else atr0
                rsi  = _f(rsi_a,    i,    50.0)
                mh   = _f(macd_a,   i,     0.0)
                mhp  = _f(macd_a,   i - 1, 0.0) if i >= 1 else 0.0
                bp   = _f(bb_a,     i,     0.5)

                conf  = _score_bar(sig, i, closes, ema_z, vr, atr0, alag, rsi, mh, macd_z, mhp, bp)
                tier  = _conf_to_tier(conf)
                frac  = tier_alloc.get(tier, 0.0)

                if mode == 'RECOVERY':
                    frac *= _RECOVERY_ALLOC_MULT

                if frac <= 0.0:
                    equity.append(capital)
                    continue

                allocated = capital * frac
            else:
                # Legacy: fixed fraction of capital
                tier      = 'legacy'
                allocated = capital * position_pct

            qty        = allocated / price
            entry_comm = allocated * commission_pct
            capital   -= allocated + entry_comm

            if sig == 1:
                stop   = price * (1 - stop_loss_pct   / 100)
                target = price * (1 + take_profit_pct / 100)
            else:
                stop   = price * (1 + stop_loss_pct   / 100)
                target = price * (1 - take_profit_pct / 100)

            position = {
                'entry_bar':        i,
                'time':             idx[i],
                'entry':            price,
                'qty':              qty,
                'side':             sig,
                'stop':             stop,
                'target':           target,
                'capital':          allocated,
                'entry_commission': entry_comm,
                'tier':             tier,
                'pyramided':        False,
            }

        # ── 7. Mark-to-market equity (final value for this bar) ───────────────
        if position is not None:
            unrealized = (
                (price - position['entry']) * position['qty'] * position['side']
            )
            equity.append(capital + position['capital'] + unrealized)
        else:
            equity.append(capital)

    equity_s  = pd.Series(equity, index=idx, name='equity')
    trades_df = pd.DataFrame(
        trades,
        columns=['entry_time', 'exit_time', 'entry', 'exit',
                 'qty', 'side', 'pnl', 'reason', 'tier', 'pyramided'],
    )
    return equity_s, trades_df
