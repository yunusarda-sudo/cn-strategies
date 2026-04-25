"""
Risk management — confidence-tiered, drawdown-aware, zero-liquidation.

Position sizing (keyed to confidence tier from _score_to_tier()):
  skip   →  0 % of portfolio — no trade
  medium → 20 % of portfolio, 2× leverage → 0.40 exposure
  high   → 50 % of portfolio, 3× leverage → 1.50 exposure
  ultra  → 70 % of portfolio, 5× leverage → clipped to 3.0 by MAX_EXPOSURE

Drawdown modes (applied as a multiplier on nominal allocation):
  CAUTION  (DD >10%)  → allocation × 0.70, leverage capped at 3×
  RECOVERY (DD >20%)  → allocation × 0.50, leverage capped at 2×
  FLAT     (DD >30%)  → no new entries (hard stop)

Exposure control:
  total exposure ≤ MAX_EXPOSURE (3.0) at all times
  at most 1 strong (high / ultra) trade open simultaneously

Stop / exit logic:
  Initial stop  = entry ± max(MIN_STOP_PCT % of entry, 1.0 × ATR)
  Trailing stop = price peak ∓ ATR_TRAIL_MULT × ATR  (ratchets in profit direction)
  No fixed take-profit — winners run until trail or reversal signal.

Pyramiding:
  Trigger : unrealized gain ≥ ATR_PYRAMID_TRIGGER × ATR
  Add-on  : 50 % of original allocation, one add-on per position
  Pre-cond: stop raised to break-even before adding

ENHANCED FEATURES (v2):
  - Adaptive Position Sizing: scale based on recent performance
  - Trade Distribution Control: prevent profit concentration
  - Regime-Based Risk: trending vs sideways vs volatile
  - Dynamic Drawdown Control: smooth scaling (not binary)
  - Persistence Tuning: reduce early exits
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

from .config import (
    MAX_TRADES_PER_HOUR,
    MAX_DAILY_LOSS_PCT,
    DD_CAUTION_PCT,
    DD_RECOVERY_PCT,
    DD_FLAT_PCT,
    MIN_STOP_PCT,
    ATR_TRAIL_MULT,
    ATR_PYRAMID_TRIGGER,
    TIER_ALLOC,
    MAX_EXPOSURE,
    INITIAL_CAPITAL,
    CONSECUTIVE_LOSS_CAP,
)


class RiskManager:
    def __init__(self, initial_capital: float | None = None):
        self.trade_timestamps: list[float] = []
        self.daily_pnl: float = 0.0
        self.daily_start: float = time.time()

        _cap = initial_capital if initial_capital is not None else INITIAL_CAPITAL
        self._equity_peak: float = _cap
        self._pyramid_used: bool = False

        # === ENHANCED RISK MANAGEMENT (v2) ===

        # 1. Adaptive Position Sizing - track recent performance
        self._recent_trades: deque[float] = deque(maxlen=10)  # last 10 trades
        self._consecutive_wins: int = 0
        self._consecutive_losses: int = 0

        # 2. Trade Distribution Control
        self._trade_pnl_history: deque[float] = deque(maxlen=50)  # for distribution analysis
        self._max_trade_pnl_pct: float = 0.25  # max single trade can contribute 25% of total PnL

        # 3. Regime Detection
        self._price_history: deque[float] = deque(maxlen=50)
        self._volatility_history: deque[float] = deque(maxlen=20)

        # 4. Dynamic Drawdown Control - smooth scaling
        self._dd_multiplier: float = 1.0  # continuous multiplier, not binary

        # 5. Persistence Tuning
        self._min_hold_bars_override: int = 5  # reduce early exits

    # ─────────────────────────────────────────── internals

    def _prune(self):
        cutoff = time.time() - 3600
        self.trade_timestamps = [t for t in self.trade_timestamps if t > cutoff]
        if time.time() - self.daily_start > 86400:
            self.daily_pnl = 0.0
            self.daily_start = time.time()

    def _current_dd(self, portfolio_value: float) -> float:
        return (self._equity_peak - portfolio_value) / self._equity_peak

    # ─────────────────────────────────────────── entry gate

    def can_open_new_trade(self, portfolio_value: float | None = None) -> bool:
        """
        Hard-cap gate. All conditions must pass before any position is opened.

        Blocks when:
          - trades per hour exceeds MAX_TRADES_PER_HOUR
          - daily loss exceeds MAX_DAILY_LOSS_PCT × portfolio
          - drawdown ≥ DD_FLAT_PCT (>30%) — full stop
        """
        self._prune()

        if len(self.trade_timestamps) >= MAX_TRADES_PER_HOUR:
            print(f'[risk] BLOCKED trades/h: {len(self.trade_timestamps)}'
                  f'/{MAX_TRADES_PER_HOUR}')
            return False

        ref = portfolio_value if portfolio_value is not None else self._equity_peak
        daily_limit = ref * MAX_DAILY_LOSS_PCT
        if self.daily_pnl <= -daily_limit:
            print(f'[risk] BLOCKED daily loss: ${-self.daily_pnl:.2f}'
                  f' >= ${daily_limit:.2f}')
            return False

        if portfolio_value is not None:
            dd = self._current_dd(portfolio_value)
            if dd >= DD_FLAT_PCT:
                print(f'[risk] BLOCKED drawdown: {dd:.1%} >= {DD_FLAT_PCT:.1%} (FLAT)')
                return False

        if self._consecutive_losses >= CONSECUTIVE_LOSS_CAP:
            print(f'[risk] BLOCKED consecutive losses: {self._consecutive_losses}'
                  f'>= {CONSECUTIVE_LOSS_CAP}')
            return False

        return True

    def record_trade(self):
        self.trade_timestamps.append(time.time())
        self._pyramid_used = False

    def record_pnl(self, pnl: float, portfolio_value: float | None = None):
        self.daily_pnl += pnl
        if portfolio_value is not None:
            self._equity_peak = max(self._equity_peak, portfolio_value)

    def update_equity(self, portfolio_value: float):
        """Call every candle (flat or in position) to keep the peak current."""
        self._equity_peak = max(self._equity_peak, portfolio_value)

    # ─────────────────────────────────────────── position sizing

    def compute_allocation(
        self,
        tier: str,
        portfolio_value: float,
        regime: str = 'ranging',   # kept for API compatibility; no longer used
    ) -> float:
        """
        USD notional to deploy for a new position.
        Returns 0.0 for 'skip' — caller must abort the trade.

        DD-mode multipliers (CAUTION/RECOVERY) are applied in the caller
        (predict / live engine) before calling this method, so this function
        returns the raw tier allocation.
        """
        base_pct = TIER_ALLOC.get(tier, 0.0)
        return portfolio_value * base_pct

    # ─────────────────────────────────────────── stop management

    def initial_stop(self, entry: float, atr: float, signal: int) -> float:
        """
        Tight initial stop for asymmetric risk (small loss if wrong).
        Distance = max(MIN_STOP_PCT % of entry, 1.0 × ATR).
        """
        stop_dist = max(entry * MIN_STOP_PCT / 100.0, atr)
        return entry - stop_dist if signal == 1 else entry + stop_dist

    def update_trailing_stop(
        self,
        current_stop: float,
        price: float,
        atr: float,
        signal: int,
    ) -> float:
        """
        Ratchet trailing stop: only advances in the profitable direction, never retreats.
        Trail distance = ATR_TRAIL_MULT × ATR.
        Call every candle while in a position.
        """
        trail = ATR_TRAIL_MULT * atr
        if signal == 1:
            return max(current_stop, price - trail)
        else:
            return min(current_stop, price + trail)

    # ─────────────────────────────────────────── pyramiding

    def can_pyramid(
        self,
        position: dict,
        price: float,
        atr: float,
        confidence: float,
    ) -> bool:
        """
        Allow one add-on per position when:
          - Not yet pyramided this trade
          - Confidence still ≥ medium tier (≥ 0.30)
          - Unrealized gain ≥ ATR_PYRAMID_TRIGGER × ATR
        """
        if self._pyramid_used:
            return False
        if confidence < 0.30:
            return False
        side   = position.get('side', 1)
        profit = (price - position['entry']) * side
        return profit >= ATR_PYRAMID_TRIGGER * atr

    def pyramid_allocation(self, original_allocation: float) -> float:
        """50 % add-on. Marks pyramid consumed so it fires only once per position."""
        self._pyramid_used = True
        return original_allocation * 0.50

    def breakeven_stop(self, entry: float) -> float:
        """Return break-even stop price. Move stop here before pyramiding."""
        return entry

    # ─────────────────────────────────────────── status

    def status(self, portfolio_value: float | None = None) -> str:
        self._prune()
        dd_s = ''
        if portfolio_value is not None:
            dd = self._current_dd(portfolio_value)
            mode = ('FLAT' if dd >= DD_FLAT_PCT else
                    'RECOVERY' if dd >= DD_RECOVERY_PCT else
                    'CAUTION' if dd >= DD_CAUTION_PCT else 'NORMAL')
            dd_s = f'  dd={dd:.1%}({mode})'
        return (
            f'trades_1h={len(self.trade_timestamps)}/{MAX_TRADES_PER_HOUR}  '
            f'daily_pnl=${self.daily_pnl:+.2f}{dd_s}'
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ENHANCED RISK MANAGEMENT (v2) - New Methods
    # ═══════════════════════════════════════════════════════════════════════════

    # ─────────────────────────────────────────── 1. Adaptive Position Sizing

    def get_adaptive_multiplier(
        self,
        tier: str,
        confidence: float,
        portfolio_value: float,
    ) -> float:
        """
        Scale allocation based on recent performance.
        
        - Increase when: recent trades profitable, high confidence
        - Decrease when: recent losses, drawdown increases
        
        Returns multiplier in range [0.5, 1.5].
        """
        multiplier = 1.0
        
        # Track recent performance
        if len(self._recent_trades) >= 3:
            recent_avg = sum(self._recent_trades) / len(self._recent_trades)
            
            # Scale up on profit streak
            if recent_avg > 0:
                # Progressive increase: up to +50% for strong performers
                multiplier = min(1.5, 1.0 + (recent_avg / 100.0))
            
            # Scale down on loss streak
            elif recent_avg < 0:
                # Progressive decrease: down to 50% for poor performers
                multiplier = max(0.5, 1.0 + (recent_avg / 200.0))
        
        # Confidence bonus: ultra tier gets up to +20% boost
        if tier == 'ultra' and confidence >= 0.8:
            multiplier *= 1.1
        elif tier == 'high' and confidence >= 0.6:
            multiplier *= 1.05
        
        # Drawdown penalty (smooth scaling)
        dd = self._current_dd(portfolio_value)
        if dd > 0.10:
            # Gradual reduction from 10% DD onwards
            dd_penalty = max(0.5, 1.0 - (dd - 0.10) * 1.5)
            multiplier *= dd_penalty
        
        return round(multiplier, 3)

    def record_trade_result(self, pnl: float) -> None:
        """Record trade PnL for adaptive sizing."""
        self._recent_trades.append(pnl)
        self._trade_pnl_history.append(pnl)
        
        # Track consecutive wins/losses
        if pnl > 0:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        elif pnl < 0:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

    # ─────────────────────────────────────────── 2. Trade Distribution Control

    def get_distribution_multiplier(
        self,
        trade_pnl: float,
        total_realized_pnl: float,
    ) -> float:
        """
        Prevent profit concentration - limit max allocation for ultra-tier trades.
        
        If a single trade would contribute more than _max_trade_pnl_pct of total PnL,
        reduce the allocation for that trade.
        """
        if total_realized_pnl <= 0 or len(self._trade_pnl_history) < 5:
            return 1.0
        
        # Calculate what this trade would add
        projected_total = total_realized_pnl + trade_pnl
        contribution = trade_pnl / projected_total if projected_total != 0 else 0
        
        # If this trade would be > 25% of total, reduce allocation
        if contribution > self._max_trade_pnl_pct:
            # Smooth reduction
            excess = contribution - self._max_trade_pnl_pct
            return max(0.5, 1.0 - excess)
        
        return 1.0

    def get_max_allocation_for_tier(self, tier: str) -> float:
        """Get max allocation percentage per tier."""
        caps = {
            'skip': 0.0,
            'medium': 0.20,
            'high': 0.50,
            'ultra': 0.60,  # reduced from 0.70 to prevent concentration
        }
        return caps.get(tier, 0.0)

    # ─────────────────────────────────────────── 3. Regime-Based Risk

    def detect_regime(self, prices: list[float]) -> str:
        """
        Detect market regime: trending, sideways, or volatile.
        
        - trending: strong directional movement
        - ranging: low volatility, no clear direction
        - volatile: high volatility, uncertain
        """
        if len(prices) < 20:
            return 'ranging'
        
        # Calculate trend strength
        recent = prices[-20:]
        first_half = sum(recent[:10]) / 10
        second_half = sum(recent[10:]) / 10
        trend_pct = (second_half - first_half) / first_half if first_half != 0 else 0
        
        # Calculate volatility
        returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
        volatility = (max(returns) - min(returns)) / 2 if returns else 0
        
        # Classify regime
        if abs(trend_pct) > 0.05 and volatility < 0.03:
            return 'trending'
        elif volatility > 0.05:
            return 'volatile'
        else:
            return 'ranging'

    def get_regime_multiplier(self, regime: str) -> float:
        """
        Adjust risk based on regime.
        
        - trending: allow larger positions (+20%)
        - ranging: normal exposure
        - volatile: reduce exposure (-30%)
        """
        multipliers = {
            'trending': 1.2,
            'ranging': 1.0,
            'volatile': 0.7,
        }
        return multipliers.get(regime, 1.0)

    def update_regime_state(self, price: float, atr: float | None = None) -> None:
        """Update regime detection state with new price."""
        self._price_history.append(price)
        if atr is not None:
            self._volatility_history.append(atr)

    # ─────────────────────────────────────────── 4. Dynamic Drawdown Control

    def get_smooth_dd_multiplier(self, portfolio_value: float) -> float:
        """
        Smooth drawdown scaling - not binary FLAT.
        
        Instead of discrete CAUTION/RECOVERY/FLAT modes, use continuous scaling.
        """
        dd = self._current_dd(portfolio_value)
        
        if dd < 0.05:
            return 1.0
        elif dd < 0.10:
            # 5-10% DD: gradual reduction from 100% to 85%
            return 0.85 + (0.10 - dd) * 3
        elif dd < 0.20:
            # 10-20% DD: reduction from 85% to 60%
            return 0.60 + (0.20 - dd) * 2.5
        elif dd < 0.30:
            # 20-30% DD: reduction from 60% to 30%
            return 0.30 + (0.30 - dd) * 3
        else:
            # >30% DD: flat (no new trades)
            return 0.0

    def get_leverage_cap(self, portfolio_value: float) -> int:
        """Dynamic leverage cap based on drawdown."""
        dd = self._current_dd(portfolio_value)
        
        if dd < 0.10:
            return 5  # max 5x leverage
        elif dd < 0.20:
            return 3  # cap at 3x
        elif dd < 0.30:
            return 2  # cap at 2x
        else:
            return 0  # no leverage

    # ─────────────────────────────────────────── 5. Persistence Tuning

    def get_min_hold_bars(self, regime: str = 'ranging') -> int:
        """
        Reduce early exits - allow winners to run longer.
        
        Override MIN_HOLD_BARS based on regime.
        """
        # In trending markets, hold longer
        if regime == 'trending':
            return max(self._min_hold_bars_override, 8)
        elif regime == 'volatile':
            return max(self._min_hold_bars_override, 4)
        else:
            return self._min_hold_bars_override

    def should_extend_hold(
        self,
        bars_held: int,
        unrealized_pnl: float,
        atr: float,
        regime: str = 'ranging',
    ) -> bool:
        """
        Decide whether to extend hold beyond signal=0 exit.
        
        Allow winners to run if:
        - Strong unrealized profit (> 2 ATR)
        - In trending regime
        - Not yet held too long
        """
        min_bars = self.get_min_hold_bars(regime)
        
        # Always respect minimum hold
        if bars_held < min_bars:
            return True
        
        # Extend if strong profit and trending
        if unrealized_pnl > 2 * atr and regime == 'trending':
            return True
        
        # Extend if very strong profit (> 3 ATR)
        if unrealized_pnl > 3 * atr:
            return True
        
        return False

    # ─────────────────────────────────────────── Evaluation Metrics

    def get_performance_stats(self) -> dict:
        """Return current performance statistics."""
        recent_trades = list(self._recent_trades)
        
        if not recent_trades:
            return {
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'consecutive_wins': 0,
                'consecutive_losses': 0,
            }
        
        wins = sum(1 for p in recent_trades if p > 0)
        return {
            'win_rate': wins / len(recent_trades),
            'avg_pnl': sum(recent_trades) / len(recent_trades),
            'consecutive_wins': self._consecutive_wins,
            'consecutive_losses': self._consecutive_losses,
            'total_trades_tracked': len(recent_trades),
        }
