from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from cnlib.base_strategy import BaseStrategy

from .cleaner import clean
from .features import compute as compute_features
from .trend import TrendStrategy
from .reversion import ReversionStrategy
from .breakout import BreakoutStrategy
from .scoring import score_signal, signal_to_tier
from .decision import (
    classify_regime, can_enter, has_min_edge,
    confidence_alloc_mult as _conf_mult_fn,
    regime_alloc_mult     as _regime_mult_fn,
    damage_alloc_mult     as _damage_mult_fn,
    MIN_CONFIDENCE, MIN_CONFIDENCE_STRONG, COOLDOWN_BARS as _COOLDOWN_BARS,
    MAX_TRADES_100_BARS   as _MAX_TRADES_100,
    TRENDING              as _D_TRENDING,
    MIN_ATR_FEE_MULT      as _D_ATR_FEE_MULT,
    FEE_PER_LEG           as _D_FEE_PER_LEG,
)
from .combiner import EnsembleCombiner
from .risk import RiskManager
from .regime import detect as detect_regime
from .config import TIER_ALLOC, MAX_EXPOSURE, MAX_COIN_ALLOC, DD_CAUTION_PCT, DD_RECOVERY_PCT, DD_FLAT_PCT

COINS = [
    'kapcoin-usd_train',
    'metucoin-usd_train',
    'tamcoin-usd_train',
]

USE_REVERSION   = True

N_TRIALS        = 500
FEATURE_WINDOW  = 280
INITIAL_CAPITAL = 3_000.0

_VALID_LEVERAGE = {1, 2, 3, 5}

# Drawdown mode thresholds (mirrored from config for in-class use).
_DD_CAUTION  = DD_CAUTION_PCT   # >10%  → reduce risk
_DD_RECOVERY = DD_RECOVERY_PCT  # >20%  → half mode
_DD_FLAT     = DD_FLAT_PCT      # >30%  → stop trading

# Allocation multipliers per drawdown mode.
_CAUTION_ALLOC_MULT  = 0.70
_RECOVERY_ALLOC_MULT = 0.70  # aligned with backtest/engine.py

_PERSISTENCE_BARS = 8
_MAX_ZERO_HOLD = 3  # max bars to hold through signal_zero for profitable trending positions

# Attack mode: combined edge required to trigger the 1.25× allocation boost.
# At 1.5× the gate threshold, only trades with clear ATR × confidence surplus qualify.
_ATTACK_EDGE_MULT = 1.5

_MAX_SL_BY_LEV: dict[int, float] = {
    1:  6.0,
    2:  4.0,
    3:  3.0,
    5:  2.0,
    10: 0.8,
}

_STRATEGIES = [s for s in [
    ('trend',     TrendStrategy),
    ('reversion', ReversionStrategy if USE_REVERSION else None),
    ('breakout',  BreakoutStrategy),
] if s[1] is not None]


def _to_internal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    for col in ('open', 'high', 'low', 'close', 'volume'):
        df[col] = df[col].astype(float)
    return df[['open', 'high', 'low', 'close', 'volume']]


def _nearest_valid_lev(lev: int) -> int:
    return min(_VALID_LEVERAGE, key=lambda v: abs(v - lev))



def _atr_based_exits(
    df: pd.DataFrame,
    price: float,
    opt_sl: float,
    opt_tp: float,
    coin_lev: int,
) -> tuple[float, float]:
    sl_cap  = _MAX_SL_BY_LEV.get(coin_lev, 2.0)
    atr_pct = 0.0
    if df is not None and 'atr_pct' in df.columns:
        atr_pct = float(df['atr_pct'].iloc[-1]) * 100

    if atr_pct > 0:
        sl_pct = float(np.clip(atr_pct * 1.0, 0.3, sl_cap))
    else:
        sl_pct = min(opt_sl, sl_cap)

    tp_pct = max(sl_pct * 4.0, opt_tp)
    return round(sl_pct, 4), round(tp_pct, 4)


class BenzemezStrategy(BaseStrategy):

    def __init__(self):
        super().__init__()
        self.ensembles:   dict[str, EnsembleCombiner] = {}
        self.exit_levels: dict[str, dict]             = {}
        self._ready = False

        self._equity      = 1.0
        self._risk_equity = 1.0
        self._peak_equity = 1.0
        self._max_dd_seen = 0.0

        self._open_positions:   dict[str, dict] = {}
        self._spent_this_bar:    set[str]        = set()
        self._last_sent_signal: dict[str, int]  = {c: 0 for c in COINS}

        self._signal_cache: dict[str, int] = {c: 0 for c in COINS}
        self._signal_age:   dict[str, int] = {c: 0 for c in COINS}

        self._caution_entries  = 0
        self._recovery_entries = 0
        self._flat_entries     = 0
        self._prev_mode        = 'NORMAL'

        # ── Adaptive risk layer state ──────────────────────────────────────────
        self._risk = RiskManager(initial_capital=INITIAL_CAPITAL)
        self._zero_hold_count: dict[str, int]  = {c: 0 for c in COINS}
        self._last_regime:     dict[str, str]  = {c: 'ranging' for c in COINS}

        # ── Robustness filter state ────────────────────────────────────────────
        self._last_close_candle: dict[str, int]       = {c: -999  for c in COINS}
        self._coin_trade_candles: dict[str, list[int]] = {c: []    for c in COINS}

        # ── Decision layer state ───────────────────────────────────────────────
        self._decision_regime:    dict[str, str]      = {c: 'NORMAL' for c in COINS}
        self._cooldown:           dict[str, int]       = {c: 0        for c in COINS}
        self._bar_counter:        int                  = 0
        self._trade_bars:         dict[str, list[int]] = {c: []       for c in COINS}
        self._coin_consec_losses: dict[str, int]       = {c: 0        for c in COINS}
        self._attack_mode_count:  int                  = 0

        # Populated by egit() — used for score reporting
        self.coin_metrics:    dict[str, dict] = {}
        self.coin_rejections: dict[str, str]  = {}

    def _mark_to_market_equity(self, data: dict) -> float:
        equity = self._equity
        unrealized = 0.0
        for coin, pos in self._open_positions.items():
            df_raw = data.get(coin)
            if df_raw is None:
                continue
            close = float(df_raw['Close'].iloc[-1])
            ret = ((close - pos['entry']) / pos['entry']
                   * pos['signal'] * pos['leverage'])
            unrealized += pos['allocation'] * ret
        return equity * max(0.0, 1.0 + unrealized)

    def _update_equity(self, data: dict, raw_signals: dict[str, int]) -> None:
        self._spent_this_bar = set()

        for coin in list(self._open_positions.keys()):
            pos    = self._open_positions[coin]
            df_raw = data.get(coin)
            if df_raw is None:
                continue

            high  = float(df_raw['High'].iloc[-1])
            low   = float(df_raw['Low'].iloc[-1])
            close = float(df_raw['Close'].iloc[-1])
            sig   = pos['signal']

            exit_price: float | None = None
            exit_spends_bar = False

            # cnlib processes liquidations intrabar before close-based exits.
            if sig == 1:
                liquidation_price = pos['entry'] * (1.0 - 1.0 / pos['leverage'])
                if low <= liquidation_price:
                    _liq_pnl = -round(pos['allocation'] * self._equity * INITIAL_CAPITAL, 2)
                    self._equity *= max(0.0, 1.0 - pos['allocation'])
                    self._risk.record_trade_result(_liq_pnl)
                    self._zero_hold_count[coin] = 0
                    del self._open_positions[coin]
                    self._cooldown[coin] = _COOLDOWN_BARS
                    self._coin_consec_losses[coin] = self._coin_consec_losses.get(coin, 0) + 1
                    self._spent_this_bar.add(coin)
                    continue
            else:
                liquidation_price = pos['entry'] * (1.0 + 1.0 / pos['leverage'])
                if high >= liquidation_price:
                    _liq_pnl = -round(pos['allocation'] * self._equity * INITIAL_CAPITAL, 2)
                    self._equity *= max(0.0, 1.0 - pos['allocation'])
                    self._risk.record_trade_result(_liq_pnl)
                    self._zero_hold_count[coin] = 0
                    del self._open_positions[coin]
                    self._cooldown[coin] = _COOLDOWN_BARS
                    self._coin_consec_losses[coin] = self._coin_consec_losses.get(coin, 0) + 1
                    self._spent_this_bar.add(coin)
                    continue

            if sig == 1:
                if close <= pos['stop_loss']:
                    exit_price = pos['stop_loss']   # exit at stop level, not close
                    exit_spends_bar = True
                elif close >= pos['take_profit']:
                    exit_price = pos['take_profit']  # exit at target level
                    exit_spends_bar = True
                elif raw_signals.get(coin, 0) == 0:
                    # Task 4: Hold through signal_zero if profitable + trending
                    _unrealized = (close - pos['entry']) * sig
                    _regime     = self._last_regime.get(coin, 'ranging')
                    if (_unrealized > 0 and
                            _regime in ('trending_up', 'trending_down') and
                            self._zero_hold_count.get(coin, 0) < _MAX_ZERO_HOLD):
                        self._zero_hold_count[coin] = self._zero_hold_count.get(coin, 0) + 1
                    else:
                        self._zero_hold_count[coin] = 0
                        exit_price = close
                elif raw_signals.get(coin, 0) == -1:
                    self._zero_hold_count[coin] = 0
                    exit_price = close
            else:
                if close >= pos['stop_loss']:
                    exit_price = pos['stop_loss']   # exit at stop level
                    exit_spends_bar = True
                elif close <= pos['take_profit']:
                    exit_price = pos['take_profit']  # exit at target level
                    exit_spends_bar = True
                elif raw_signals.get(coin, 0) == 0:
                    # Task 4: Hold through signal_zero if profitable + trending
                    _unrealized = (pos['entry'] - close) * abs(sig)
                    _regime     = self._last_regime.get(coin, 'ranging')
                    if (_unrealized > 0 and
                            _regime in ('trending_up', 'trending_down') and
                            self._zero_hold_count.get(coin, 0) < _MAX_ZERO_HOLD):
                        self._zero_hold_count[coin] = self._zero_hold_count.get(coin, 0) + 1
                    else:
                        self._zero_hold_count[coin] = 0
                        exit_price = close
                elif raw_signals.get(coin, 0) == 1:
                    self._zero_hold_count[coin] = 0
                    exit_price = close

            if exit_price is None:
                pos['_bars_held'] = pos.get('_bars_held', 0) + 1
                continue

            ret = (exit_price - pos['entry']) / pos['entry'] * sig * pos['leverage']
            _pnl_usd = round(pos['allocation'] * ret * self._equity * INITIAL_CAPITAL, 2)
            self._equity *= (1.0 + pos['allocation'] * ret)
            self._risk.record_trade_result(_pnl_usd)
            self._zero_hold_count[coin] = 0
            del self._open_positions[coin]
            self._cooldown[coin] = _COOLDOWN_BARS
            if _pnl_usd > 0:
                self._coin_consec_losses[coin] = 0
            elif _pnl_usd < 0:
                self._coin_consec_losses[coin] = self._coin_consec_losses.get(coin, 0) + 1
            if exit_spends_bar:
                self._spent_this_bar.add(coin)

        self._risk_equity = self._mark_to_market_equity(data)
        if self._risk_equity > self._peak_equity:
            self._peak_equity = self._risk_equity
        self._risk.update_equity(self._risk_equity * INITIAL_CAPITAL)
        dd = self._drawdown()
        if dd > self._max_dd_seen:
            self._max_dd_seen = dd

    def _drawdown(self) -> float:
        if self._peak_equity == 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._risk_equity) / self._peak_equity)

    def _risk_mode(self) -> str:
        dd = self._drawdown()
        if dd >= _DD_FLAT:
            return 'FLAT'
        if dd >= _DD_RECOVERY:
            return 'RECOVERY'
        if dd >= _DD_CAUTION:
            return 'CAUTION'
        return 'NORMAL'

    def _apply_persistence(self, raw_signals: dict[str, int]) -> None:
        for coin in COINS:
            new_sig = raw_signals[coin]
            cached  = self._signal_cache[coin]
            age     = self._signal_age[coin]

            if new_sig != 0:
                # Counter-signal: immediately cancel old cache
                if cached != 0 and new_sig == -cached:
                    self._signal_cache[coin] = new_sig
                    self._signal_age[coin]   = 0
                else:
                    self._signal_cache[coin] = new_sig
                    self._signal_age[coin]   = 0
                # raw_signals[coin] already holds new_sig
            elif cached != 0 and age < _PERSISTENCE_BARS:
                raw_signals[coin]      = cached
                self._signal_age[coin] = age + 1
            else:
                self._signal_cache[coin] = 0
                self._signal_age[coin]   = 0

    def egit(self, n_trials: int = N_TRIALS, thresholds: dict | None = None, strict: bool = True) -> None:
        from .search import optimize
        from .engine import run as backtest_run
        from .metrics import compute_all as compute_metrics

        # Thresholds mirror train.py — same gate for both entry points.
        # Caller may pass `thresholds` dict to adjust gates for a re-run.
        _th = thresholds or {}
        _WF_MIN_SHARPE   = _th.get('wf_min_sharpe',   0.0)
        _WF_MIN_RETURN   = _th.get('wf_min_return',  -0.05)
        _WF_MAX_DRAWDOWN = _th.get('wf_max_drawdown', -0.35)
        _WF_MAX_FOLD_STD = _th.get('wf_max_fold_std',  0.40)
        _HO_MIN_RETURN   = _th.get('ho_min_return',    0.0)
        _HO_MIN_SHARPE   = _th.get('ho_min_sharpe',    0.0)
        _HO_MAX_DRAWDOWN = _th.get('ho_max_drawdown', -0.35)

        # Reset per-run state
        self.coin_metrics    = {}
        self.coin_rejections = {}

        for coin in COINS:
            if coin not in self.coin_data:
                print(f'[warn] {coin} veri yok — atlandı')
                continue

            print(f'\n══ {coin} ══')
            df_raw  = _to_internal(self.coin_data[coin])
            df_clean = clean(df_raw)
            total_bars  = len(df_clean)
            df          = compute_features(df_clean)
            usable_bars = len(df)
            warmup_bars = total_bars - usable_bars
            print(f'  [data] total={total_bars}  warmup_removed={warmup_bars}  usable={usable_bars}')

            if usable_bars < 500:
                msg = (
                    f'{coin}: {usable_bars} usable bars (after removing {warmup_bars} warmup bars) '
                    f'< 500 minimum required for reliable evaluation. '
                    f'Provide at least {500 + warmup_bars} raw bars.'
                )
                if strict:
                    raise ValueError(msg)
                print(f'[warn] {msg} — coin atlandı')
                self.coin_rejections[coin] = f'insufficient data: {usable_bars} bars'
                continue

            # ── Year-based time split (mirrors train.py — NEVER random) ─────
            n   = len(df)
            q1  = n // 4      # 25% — Year 2 start
            q2  = n // 2      # 50% — Year 3 start (end of train)
            q3  = 3 * n // 4  # 75% — Year 4 start (pseudo-test)

            year12 = df.iloc[:q2]         # Year 1+2 → primary training set
            year2  = df.iloc[q1:q2]       # Year 2 slice (for Fold 2 train)
            year3  = df.iloc[q2:q3]       # Year 3 → WFV fold 1 test
            year4  = df.iloc[q3:]         # Year 4 → WFV fold 2 test + holdout
            year23 = pd.concat([year2, year3])

            train_df = year12
            test_df  = year4

            # ── 1. Optimize ONCE on Year 1-2 (never touches Year 3-4) ───────
            best_params: dict[str, dict] = {}
            for name, cls in _STRATEGIES:
                print(f'  [{name}] optimize ediliyor (Year 1-2)...')
                best_params[name] = optimize(
                    train_df, cls,
                    n_trials=n_trials,
                    optimize_exits=True,
                    objective='calmar',
                    commission_pct=0.001,
                    position_pct=1.0,
                )
                print(f'    → {best_params[name]}')

            # Build ensemble with Year 1-2 optimised params
            strategies = []
            sl_list, tp_list = [], []
            for name, cls in _STRATEGIES:
                s = cls()
                s.set_params({k: v for k, v in best_params[name].items()
                              if k not in ('stop_loss_pct', 'take_profit_pct')})
                strategies.append(s)
                sl_list.append(best_params[name].get('stop_loss_pct',   1.0))
                tp_list.append(best_params[name].get('take_profit_pct', 2.5))
            avg_sl   = float(np.mean(sl_list))
            avg_tp   = float(np.mean(tp_list))
            ensemble = EnsembleCombiner(strategies, threshold=0.20)

            # ── 2. Walk-forward: 2 year-aligned folds (re-optimize per fold) ─
            # Fold 1: Year 1-2 train → Year 3 test
            # Fold 2: Year 2-3 train → Year 4 test
            print('  [wf] walk-forward (2 year-aligned folds, re-optimizing per fold)...')

            def _run_wf_fold(fold_train: pd.DataFrame, fold_test: pd.DataFrame,
                             fold_label: str) -> dict:
                fold_best: dict[str, dict] = {}
                for f_name, f_cls in _STRATEGIES:
                    fold_best[f_name] = optimize(
                        fold_train, f_cls,
                        n_trials=max(50, n_trials // 5),
                        optimize_exits=True, objective='calmar',
                        commission_pct=0.001, position_pct=1.0,
                        seed=42 + hash(f_name + fold_label) % 997,
                        show_progress_bar=False,
                    )
                fold_strats = []
                for f_name, f_cls in _STRATEGIES:
                    fs = f_cls()
                    fs.set_params({k: v for k, v in fold_best[f_name].items()
                                   if k not in ('stop_loss_pct', 'take_profit_pct')})
                    fold_strats.append(fs)
                fold_ens = EnsembleCombiner(fold_strats, threshold=0.20)
                f_sl = float(np.mean([fold_best[n].get('stop_loss_pct',   1.0) for n, _ in _STRATEGIES]))
                f_tp = float(np.mean([fold_best[n].get('take_profit_pct', 2.5) for n, _ in _STRATEGIES]))

                ctx_len = min(200, len(fold_train))
                ctx_df  = pd.concat([fold_train.iloc[-ctx_len:], fold_test])
                ctx_df  = ctx_df[~ctx_df.index.duplicated(keep='last')]
                fsigs   = fold_ens.combine(ctx_df).loc[fold_test.index]
                eq, tr  = backtest_run(
                    fold_test, fsigs, initial_capital=INITIAL_CAPITAL,
                    position_pct=1.0, stop_loss_pct=f_sl, take_profit_pct=f_tp,
                    commission_pct=0.001, atr_trail_mult=2.5, min_hold_bars=5,
                )
                m = compute_metrics(eq, tr)
                m['fold_label'] = fold_label
                m['period'] = (f'{fold_test.index[0].date()} → {fold_test.index[-1].date()}'
                               f'  ({len(fold_test)} bars)')
                return m

            fold1_metrics = _run_wf_fold(year12, year3, 'Year1-2→Year3')
            fold2_metrics = _run_wf_fold(year23, year4, 'Year2-3→Year4')
            wf_results = [fold1_metrics, fold2_metrics]

            print(f'  {"Fold":<20}  {"Period":<34}  {"Return":>8}  {"Sharpe":>7}  {"MDD":>7}  {"Trades":>6}')
            for r in wf_results:
                print(f'  {r.get("fold_label",""):<20}  {r.get("period",""):<34}'
                      f'  {r.get("total_return",0):>+8.2%}'
                      f'  {r.get("sharpe",0):>7.3f}'
                      f'  {r.get("max_drawdown",0):>+7.2%}'
                      f'  {r.get("n_trades",0):>6}')

            numeric_keys = [k for k in fold1_metrics if isinstance(fold1_metrics[k], (int, float))]
            wf_avg = {k: float(np.mean([r.get(k, 0.0) for r in wf_results])) for k in numeric_keys}

            print(f'    folds={len(wf_results)}'
                  f'  Sharpe={wf_avg.get("sharpe", 0):.3f}'
                  f'  return={wf_avg.get("total_return", 0):.2%}'
                  f'  MDD={wf_avg.get("max_drawdown", 0):.2%}')
            for r in wf_results:
                print(f'      fold {r.get("fold","?")}:'
                      f'  ret={r.get("total_return", 0):.2%}'
                      f'  Sharpe={r.get("sharpe", 0):.3f}'
                      f'  trades={r.get("n_trades", 0)}')

            # Hard rejection — same thresholds as train.py
            wf_sharpe = wf_avg.get('sharpe',       -999.0)
            wf_ret    = wf_avg.get('total_return',  -999.0)
            wf_mdd    = wf_avg.get('max_drawdown',  -999.0)

            rejection = None
            if wf_sharpe < _WF_MIN_SHARPE:
                rejection = f'WF Sharpe {wf_sharpe:.3f} < {_WF_MIN_SHARPE}'
            elif wf_ret < _WF_MIN_RETURN:
                rejection = f'WF return {wf_ret:.2%} < {_WF_MIN_RETURN:.0%}'
            elif wf_mdd < _WF_MAX_DRAWDOWN:
                rejection = f'WF max_drawdown {wf_mdd:.2%} < {_WF_MAX_DRAWDOWN:.0%}'
            elif len(wf_results) >= 2:
                fold_rets = [r.get('total_return', 0.0) for r in wf_results]
                std = float(np.std(fold_rets, ddof=1))
                if std > _WF_MAX_FOLD_STD:
                    rejection = f'WF return std {std:.3f} > {_WF_MAX_FOLD_STD} (inconsistent)'

            if rejection:
                self.coin_rejections[coin] = rejection
                print(f'  [REJECT] {coin} — {rejection} — coin atlandı')
                continue

            print(f'  [ACCEPT] WF: Sharpe={wf_sharpe:.3f}  '
                  f'return={wf_ret:.2%}  MDD={wf_mdd:.2%}')

            # ── 3. Holdout: Year 4 (pseudo-test — unseen during optimization) ──
            # Uses Year 1-2 trained ensemble (same params used in Fold 1).
            # Fold 2 already validated on Year 4 independently above.
            ctx_len  = min(200, len(year12))
            ctx_df   = pd.concat([year12.iloc[-ctx_len:], year4])
            ctx_df   = ctx_df[~ctx_df.index.duplicated(keep='last')]
            all_sigs = ensemble.combine(ctx_df)
            ho_sigs  = all_sigs.loc[all_sigs.index.isin(year4.index)]

            ho_eq, ho_tr = backtest_run(
                year4, ho_sigs,
                initial_capital=INITIAL_CAPITAL,
                position_pct=1.0,
                stop_loss_pct=avg_sl,
                take_profit_pct=avg_tp,
                commission_pct=0.001,
                atr_trail_mult=2.5,
                min_hold_bars=5,
            )
            ho_metrics = compute_metrics(ho_eq, ho_tr)

            ho_ret    = ho_metrics.get('total_return', -999.0)
            ho_sharpe = ho_metrics.get('sharpe',       -999.0)
            ho_mdd    = ho_metrics.get('max_drawdown', -999.0)

            ho_rejection = None
            if ho_ret < _HO_MIN_RETURN:
                ho_rejection = f'holdout return {ho_ret:.2%} < 0'
            elif ho_sharpe < _HO_MIN_SHARPE:
                ho_rejection = f'holdout Sharpe {ho_sharpe:.3f} < 0'
            elif ho_mdd < _HO_MAX_DRAWDOWN:
                ho_rejection = f'holdout max_drawdown {ho_mdd:.2%} > 35%'

            if ho_rejection:
                self.coin_rejections[coin] = ho_rejection
                print(f'  [REJECT] {coin} — {ho_rejection} — coin atlandı')
                continue

            print(f'  [ACCEPT] Holdout: Sharpe={ho_sharpe:.3f}  '
                  f'return={ho_ret:.2%}  MDD={ho_mdd:.2%}')

            # ── 4. Train backtest (for score/overfit reporting only) ─────────
            train_sigs = ensemble.combine(train_df)
            tr_eq, tr_tr = backtest_run(
                train_df, train_sigs,
                position_pct=1.0,
                stop_loss_pct=avg_sl,
                take_profit_pct=avg_tp,
                commission_pct=0.001,
            )
            tr_metrics   = compute_metrics(tr_eq, tr_tr)
            train_return = tr_metrics.get('total_return', 0.0)
            train_mdd    = tr_metrics.get('max_drawdown', 0.0)

            # ── 5. Commit — only reaches here if both gates passed ────────────
            self.ensembles[coin] = ensemble
            self.exit_levels[coin] = {
                'sl_pct': round(avg_sl, 4),
                'tp_pct': round(avg_tp, 4),
            }
            self.coin_metrics[coin] = {
                # Metadata
                'training_data_type': 'real',
                'train_period':       f'{year12.index[0].date()} → {year12.index[-1].date()}',
                'validation_period':  f'{year3.index[0].date()} → {year3.index[-1].date()}',
                'pseudotest_period':  f'{year4.index[0].date()} → {year4.index[-1].date()}',
                'folds':              2,
                'objective':          'composite (0.4×return + 0.25×sharpe + 0.2×calmar - 0.1×mdd - 0.05×overtrading)',
                'optuna_trials':      n_trials,
                # Per-fold walk-forward results
                'wf_fold1_return':    round(fold1_metrics.get('total_return', 0.0), 4),
                'wf_fold1_sharpe':    round(fold1_metrics.get('sharpe', 0.0),       4),
                'wf_fold1_mdd':       round(fold1_metrics.get('max_drawdown', 0.0), 4),
                'wf_fold1_trades':    int(fold1_metrics.get('n_trades', 0)),
                'wf_fold2_return':    round(fold2_metrics.get('total_return', 0.0), 4),
                'wf_fold2_sharpe':    round(fold2_metrics.get('sharpe', 0.0),       4),
                'wf_fold2_mdd':       round(fold2_metrics.get('max_drawdown', 0.0), 4),
                'wf_fold2_trades':    int(fold2_metrics.get('n_trades', 0)),
                # Avg WFV
                'wf_sharpe':          round(wf_sharpe, 4),
                'wf_return':          round(wf_ret,    4),
                'wf_mdd':             round(wf_mdd,    4),
                # Holdout (Year 4)
                'holdout_sharpe':     round(ho_sharpe, 4),
                'holdout_return':     round(ho_ret,    4),
                'holdout_mdd':        round(ho_mdd,    4),
                'holdout_n_trades':   int(ho_metrics.get('n_trades', 0)),
                # Train (Year 1-2 in-sample)
                'train_return':       round(train_return, 4),
                'train_mdd':          round(train_mdd,    4),
                # Exit levels
                'exit_sl':            round(avg_sl, 4),
                'exit_tp':            round(avg_tp, 4),
            }
            print(f'  exit → sl={avg_sl:.2f}%  tp={avg_tp:.2f}%')

        accepted = list(self.ensembles.keys())
        if not accepted:
            raise RuntimeError(
                '[REJECT] Hiçbir coin walk-forward + holdout doğrulamasını geçemedi. '
                'Model kaydedilmedi.'
            )

        self._ready = True
        print(f'\n[egit] tamamlandı ✓  kabul edilen coinler: {accepted}')

    def predict(self, data: dict) -> list[dict]:
        # Advance bar counter and decay cooldowns before any other logic.
        self._bar_counter += 1
        for _c in COINS:
            if self._cooldown[_c] > 0:
                self._cooldown[_c] -= 1

        if not self._ready:
            return [{'coin': c, 'signal': 0, 'allocation': 0.0, 'leverage': 1}
                    for c in COINS]

        raw_signals: dict[str, int]          = {}
        raw_scores:  dict[str, float]        = {}
        feature_dfs: dict[str, pd.DataFrame] = {}

        for coin in COINS:
            df_raw = data.get(coin)
            if df_raw is None or coin not in self.ensembles:
                raw_signals[coin] = 0
                raw_scores[coin]  = 0.0
                continue
            try:
                df = _to_internal(df_raw.iloc[-FEATURE_WINDOW:])
                df = compute_features(df)
                if len(df) >= 50:
                    raw_signals[coin] = self.ensembles[coin].last_signal(df)
                    raw_scores[coin]  = self.ensembles[coin].last_raw_score(df)
                    feature_dfs[coin] = df
                else:
                    raw_signals[coin] = 0
                    raw_scores[coin]  = 0.0
            except Exception as e:
                print(f'[predict error] {coin}: {e}')
                raw_signals[coin] = 0
                raw_scores[coin]  = 0.0

        # Compute and store current regime per coin (used for exit hold + allocation)
        for _c in COINS:
            _df_r = feature_dfs.get(_c)
            if _df_r is not None and len(_df_r) > 0:
                try:
                    self._last_regime[_c] = str(detect_regime(_df_r).iloc[-1])
                except Exception:
                    pass

        # Decision regime (5-class: TRENDING/CHOPPY/HIGH_VOLATILITY/LOW_VOLATILITY/NORMAL)
        for _c in COINS:
            _df_r = feature_dfs.get(_c)
            if _df_r is not None and len(_df_r) >= 50:
                try:
                    self._decision_regime[_c] = classify_regime(_df_r)
                except Exception:
                    pass

        self._apply_persistence(raw_signals)
        self._update_equity(data, raw_signals)

        mode = self._risk_mode()
        dd   = self._drawdown()

        if mode != self._prev_mode:
            if mode == 'CAUTION':
                self._caution_entries += 1
                print(f'[risk] CAUTION   dd={dd:.1%}  alloc×{_CAUTION_ALLOC_MULT}'
                      f'  (#{self._caution_entries})')
            elif mode == 'RECOVERY':
                self._recovery_entries += 1
                print(f'[risk] RECOVERY  dd={dd:.1%}  alloc×{_RECOVERY_ALLOC_MULT}'
                      f'  (#{self._recovery_entries})')
            elif mode == 'FLAT':
                self._flat_entries += 1
                print(f'[risk] FLAT      dd={dd:.1%}  stopping  (#{self._flat_entries})')
            else:
                print(f'[risk] NORMAL    dd={dd:.1%}')
        self._prev_mode = mode

        if mode == 'FLAT':
            for coin in list(self._open_positions.keys()):
                pos    = self._open_positions[coin]
                df_raw = data.get(coin)
                if df_raw is not None:
                    close = float(df_raw['Close'].iloc[-1])
                    ret   = ((close - pos['entry']) / pos['entry']
                             * pos['signal'] * pos['leverage'])
                    _pnl_usd = round(pos['allocation'] * ret * self._equity * INITIAL_CAPITAL, 2)
                    self._equity *= (1.0 + pos['allocation'] * ret)
                    self._risk.record_trade_result(_pnl_usd)
            self._open_positions.clear()
            self._zero_hold_count = {c: 0 for c in COINS}
            self._cooldown        = {c: _COOLDOWN_BARS for c in COINS}
            if self._equity > self._peak_equity:
                self._peak_equity = self._equity
            for coin in COINS:
                self._signal_cache[coin]     = 0
                self._signal_age[coin]       = 0
                self._last_sent_signal[coin] = 0
            return [{'coin': c, 'signal': 0, 'allocation': 0.0, 'leverage': 1}
                    for c in COINS]

        n_active = sum(1 for s in raw_signals.values() if s != 0)

        if n_active == 0:
            return [{'coin': c, 'signal': 0, 'allocation': 0.0, 'leverage': 1}
                    for c in COINS]

        # ── DD-mode allocation multiplier (smooth scaling) ───────────────────
        _pv_at_decision = self._risk_equity * INITIAL_CAPITAL
        dd_mult = self._risk.get_smooth_dd_multiplier(_pv_at_decision)

        # ── Current exposure tracking ─────────────────────────────────────────
        # exposure = sum(allocation × leverage) across already-open positions.
        exposure_used = sum(
            p['allocation'] * p['leverage']
            for p in self._open_positions.values()
        )
        # Strong-trade tracking now computed per-iteration from self._open_positions

        decisions = []
        for coin in COINS:
            sig = raw_signals[coin]

            if sig == 0:
                decisions.append({'coin': coin, 'signal': 0,
                                  'allocation': 0.0, 'leverage': 1})
                self._last_sent_signal[coin] = 0
                continue

            # ── Gate 1: Cooldown ─────────────────────────────────────────────
            if self._cooldown[coin] > 0:
                decisions.append({'coin': coin, 'signal': 0,
                                  'allocation': 0.0, 'leverage': 1})
                self._last_sent_signal[coin] = 0
                continue

            df_feat   = feature_dfs.get(coin)
            raw_score = raw_scores.get(coin, 0.0)
            conf      = score_signal(df_feat, sig, raw_score) if df_feat is not None else 0.35
            tier      = signal_to_tier(conf)

            # ── Gate 2: Composite confidence threshold ───────────────────────
            # Strong ensemble signal (|score| > 0.4) earns a relaxed floor
            # (0.45 vs 0.55). Only the top quartile of raw scores qualify,
            # so trade count cannot explode — weak signals still need 0.55.
            _min_conf = MIN_CONFIDENCE_STRONG if abs(raw_score) > 0.4 else MIN_CONFIDENCE
            if conf < _min_conf:
                self._signal_cache[coin] = 0
                self._signal_age[coin]   = 0
                decisions.append({'coin': coin, 'signal': 0,
                                  'allocation': 0.0, 'leverage': 1})
                self._last_sent_signal[coin] = 0
                continue

            # ── Gate 3: Regime filter ─────────────────────────────────────────
            _dec_regime = self._decision_regime.get(coin, 'NORMAL')
            if not can_enter(_dec_regime):
                self._signal_cache[coin] = 0
                self._signal_age[coin]   = 0
                decisions.append({'coin': coin, 'signal': 0,
                                  'allocation': 0.0, 'leverage': 1})
                self._last_sent_signal[coin] = 0
                continue

            # Compute leverage for this coin (smooth DD-based cap)
            _base_lev = {'medium': 2, 'high': 3, 'ultra': 5}
            coin_lev = _nearest_valid_lev(
                min(_base_lev.get(tier, 2), self._risk.get_leverage_cap(_pv_at_decision))
            )

            # ── Gate 4: Minimum expected edge (ATR × confidence combined) ─────
            if df_feat is not None and not has_min_edge(df_feat, coin_lev, conf):
                decisions.append({'coin': coin, 'signal': 0,
                                  'allocation': 0.0, 'leverage': 1})
                self._last_sent_signal[coin] = 0
                continue

            # ── Gate 5: Rolling trade-rate cap (per coin, 100-bar window) ─────
            self._trade_bars[coin] = [
                b for b in self._trade_bars[coin]
                if b > self._bar_counter - 100
            ]
            if len(self._trade_bars[coin]) >= _MAX_TRADES_100:
                decisions.append({'coin': coin, 'signal': 0,
                                  'allocation': 0.0, 'leverage': 1})
                self._last_sent_signal[coin] = 0
                continue

            # Regime flags used for strong-trade gate
            _coin_regime = self._last_regime.get(coin, 'ranging')
            _is_trending = _coin_regime in ('trending_up', 'trending_down')

            # Strong-trade gate: allow 2nd high/ultra only under strict conditions
            _strong_count = sum(
                1 for p in self._open_positions.values()
                if p.get('tier', 'medium') in ('high', 'ultra')
            )
            _max_strong = 1
            if (_strong_count == 1 and dd < 0.10 and _is_trending and
                    sum(p['allocation'] for p in self._open_positions.values())
                    + TIER_ALLOC.get(tier, 0.15) <= 1.0 and
                    exposure_used + TIER_ALLOC.get(tier, 0.15) * coin_lev <= MAX_EXPOSURE):
                _max_strong = 2

            if tier in ('high', 'ultra') and _strong_count >= _max_strong:
                decisions.append({'coin': coin, 'signal': 0,
                                  'allocation': 0.0, 'leverage': 1})
                self._last_sent_signal[coin] = 0
                continue

            # Adaptive multiplier (performance-based, bounded)
            _adaptive_mult = self._risk.get_adaptive_multiplier(tier, conf, _pv_at_decision)
            if dd >= 0.10:
                _adaptive_mult = min(1.0, _adaptive_mult)

            # Decision-layer regime, confidence, and damage multipliers
            _regime_mult = _regime_mult_fn(_dec_regime)
            if dd >= 0.10:
                _regime_mult = min(1.0, _regime_mult)  # no boost during drawdown
            _conf_mult   = _conf_mult_fn(conf)
            _damage_mult = _damage_mult_fn(self._coin_consec_losses.get(coin, 0), dd)

            # Combine multipliers, hard-cap per coin
            coin_alloc = min(
                round(TIER_ALLOC[tier] * dd_mult * _adaptive_mult
                      * _regime_mult * _conf_mult * _damage_mult, 4),
                MAX_COIN_ALLOC,
            )

            # ── Attack mode: 1.25× boost for high-conviction TRENDING trades ──
            # Applied AFTER all gates. Never changes trade count — only sizes up
            # trades that already passed every filter.
            # Conditions: TRENDING regime, conf≥0.75, ATR edge 1.5× gate
            #             threshold, no recent DD (< 5%), no consecutive losses.
            if (
                _dec_regime == _D_TRENDING
                and conf >= 0.75
                and df_feat is not None and 'atr_pct' in df_feat.columns
                and (float(df_feat['atr_pct'].iloc[-1]) * coin_lev
                     / (2.0 * _D_FEE_PER_LEG)
                     * (conf / MIN_CONFIDENCE)) >= _D_ATR_FEE_MULT * _ATTACK_EDGE_MULT
                and dd < 0.05
                and self._coin_consec_losses.get(coin, 0) == 0
            ):
                coin_alloc = min(round(coin_alloc * 1.25, 4), MAX_COIN_ALLOC)
                self._attack_mode_count += 1

            # Exposure cap: clip allocation so total stays ≤ MAX_EXPOSURE.
            trade_exp = coin_alloc * coin_lev
            headroom  = MAX_EXPOSURE - exposure_used
            if trade_exp > headroom:
                max_alloc = headroom / coin_lev
                if max_alloc < 0.05:
                    decisions.append({'coin': coin, 'signal': 0,
                                      'allocation': 0.0, 'leverage': 1})
                    self._last_sent_signal[coin] = 0
                    continue
                coin_alloc = round(max_alloc, 4)
                trade_exp  = coin_alloc * coin_lev

            price  = float(data[coin]['Close'].iloc[-1])
            levels = self.exit_levels.get(coin, {'sl_pct': 1.0, 'tp_pct': 2.5})
            sl_pct, tp_pct = _atr_based_exits(
                df_feat, price,
                levels['sl_pct'], levels['tp_pct'],
                coin_lev,
            )

            if sig == 1:
                sl_price = price * (1 - sl_pct / 100)
                tp_price = price * (1 + tp_pct / 100)
            else:
                sl_price = price * (1 + sl_pct / 100)
                tp_price = price * (1 - tp_pct / 100)

            if coin not in self._open_positions:
                if (coin not in self._spent_this_bar
                        and self._risk.can_open_new_trade(_pv_at_decision)):
                    self._open_positions[coin] = {
                        'signal':      sig,
                        'entry':       price,
                        'stop_loss':   sl_price,
                        'take_profit': tp_price,
                        'allocation':  coin_alloc,
                        'leverage':    coin_lev,
                        'tier':        tier,
                        '_bars_held':  0,
                    }
                    self._zero_hold_count[coin] = 0
                    self._risk.record_trade()
                    self._trade_bars[coin].append(self._bar_counter)
                    exposure_used += trade_exp

            print(f'[size] {coin}: conf={conf:.2f} tier={tier}'
                  f'  alloc={coin_alloc:.4f} lev={coin_lev}'
                  f'  sl={sl_pct:.2f}% tp={tp_pct:.2f}%'
                  f'  mode={mode} exp={exposure_used:.2f}/{MAX_EXPOSURE}')

            reported_sl = sl_price
            self._last_sent_signal[coin] = sig
            decisions.append({
                'coin':        coin,
                'signal':      sig,
                'allocation':  coin_alloc,
                'leverage':    coin_lev,
                'stop_loss':   round(reported_sl, 6),
                'take_profit': round(tp_price, 6),
            })

        return decisions

    def risk_summary(self) -> dict:
        total_return = (self._equity - 1.0) * 100
        final_equity = self._equity * INITIAL_CAPITAL
        return {
            'final_equity_usd':      round(final_equity, 2),
            'total_return_pct':      round(total_return, 2),
            'max_drawdown_pct':      round(self._max_dd_seen * 100, 2),
            'caution_mode_entries':  self._caution_entries,
            'recovery_mode_entries': self._recovery_entries,
            'flat_mode_entries':     self._flat_entries,
            'attack_mode_count':     self._attack_mode_count,
        }

    def print_risk_summary(self) -> None:
        s = self.risk_summary()
        print('\n── Portfolio Risk Özeti ' + '─' * 49)
        print(f'  final equity      : ${s["final_equity_usd"]:>10,.2f}')
        print(f'  total return      : {s["total_return_pct"]:>+10.2f}%')
        print(f'  max drawdown      : {s["max_drawdown_pct"]:>10.2f}%')
        print(f'  caution entries   : {s["caution_mode_entries"]:>10}')
        print(f'  recovery entries  : {s["recovery_mode_entries"]:>10}')
        print(f'  flat entries      : {s["flat_mode_entries"]:>10}')
        print(f'  attack mode fires : {s["attack_mode_count"]:>10}')
        print('─' * 72)


if __name__ == '__main__':
    import sys
    if '--train' in sys.argv:
        from cnlib import backtest as _cnlib_backtest
        strategy = BenzemezStrategy()
        strategy.get_data()
        strategy.egit(n_trials=N_TRIALS)
        result = _cnlib_backtest.run(strategy=strategy, initial_capital=INITIAL_CAPITAL)
        result.print_summary()
        strategy.print_risk_summary()
    elif '--backtest' in sys.argv:
        from cnlib import backtest as _cnlib_backtest
        strategy = BenzemezStrategy()
        result = _cnlib_backtest.run(strategy=strategy, initial_capital=INITIAL_CAPITAL)
        result.print_summary()
    else:
        strategy = BenzemezStrategy()
        print('BenzemezStrategy instantiated OK')
        print('No training/backtest started. Use --train or --backtest explicitly.')
