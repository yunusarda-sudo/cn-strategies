"""
All configuration in one place. Tune here, nowhere else.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---- API credentials ----
# Training never needs credentials. Live trading reads generic names first so
# hackathon-provided keys can be dropped into .env without touching code.
EXCHANGE_ID = os.getenv('EXCHANGE_ID', 'binance').lower()
API_KEY = (
    os.getenv('TRADING_API_KEY')
    or os.getenv('PARIBU_API_KEY')
    or os.getenv('BINANCE_API_KEY')
    or ''
)
API_SECRET = (
    os.getenv('TRADING_API_SECRET')
    or os.getenv('PARIBU_API_SECRET')
    or os.getenv('BINANCE_API_SECRET')
    or ''
)

# Keys are only required for live trading, not for train.py
def _require_keys():
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            'Missing TRADING_API_KEY/PARIBU_API_KEY or TRADING_API_SECRET/PARIBU_API_SECRET. '
            'Training does not need keys; live trading does.'
        )

# ---- Trading pair ----
SYMBOL   = 'BTCUSDT'
INTERVAL = '1h'          # candle interval fed to the live engine

# ---- Order sizing ----
ORDER_USD = 100.0        # notional per trade (USD)

# ---- Exit rules (fallback defaults — train.py will find the optimal values) ----
STOP_LOSS_PCT   = 1.0
TAKE_PROFIT_PCT = 2.0

# ---- Competition mode ----
# Set COMPETITION=1 in .env to disable conservative risk caps and
# maximise for raw return instead of risk-adjusted return.
COMPETITION = bool(int(os.getenv('COMPETITION', '0')))

# ---- Rate cap (trades per rolling hour) ----
MAX_TRADES_PER_HOUR = 999 if COMPETITION else 6

# ---- Dynamic risk parameters ----
# Daily loss limit as a fraction of current portfolio value.
MAX_DAILY_LOSS_PCT   = float(os.getenv('MAX_DAILY_LOSS_PCT',   '0.05'))  # 5 %

# Drawdown circuit-breakers (fraction of equity peak).
DD_CAUTION_PCT  = float(os.getenv('DD_CAUTION_PCT',  '0.10'))  # >10% → reduce risk (×0.70)
DD_RECOVERY_PCT = float(os.getenv('DD_RECOVERY_PCT', '0.20'))  # >20% → half mode  (×0.50)
DD_FLAT_PCT     = float(os.getenv('DD_FLAT_PCT',     '0.30'))  # >30% → stop trading

# Minimum initial stop distance (% of entry price).
MIN_STOP_PCT         = float(os.getenv('MIN_STOP_PCT',        '1.0'))

# Trailing stop trails this many ATRs behind the running peak.
ATR_TRAIL_MULT       = float(os.getenv('ATR_TRAIL_MULT',      '2.5'))

# Pyramid add-on fires when unrealized profit ≥ this many ATRs.
ATR_PYRAMID_TRIGGER  = float(os.getenv('ATR_PYRAMID_TRIGGER', '1.5'))

CONSECUTIVE_LOSS_CAP = int(os.getenv('CONSECUTIVE_LOSS_CAP',  '3'))

# Signal-flip exit is ignored for the first N bars held (prevents whipsaw exits).
MIN_HOLD_BARS = int(os.getenv('MIN_HOLD_BARS', '3'))

# Confidence-tiered allocation fractions (per coin, capped at MAX_COIN_ALLOC).
#   medium → 0.15 × capital  (low conviction)
#   high   → 0.25 × capital  (medium conviction)
#   ultra  → 0.30 × capital  (max — hard ceiling per user spec)
TIER_ALLOC: dict[str, float] = {
    'skip':   0.00,
    'medium': 0.15,
    'high':   0.25,
    'ultra':  0.30,
}

# Hard cap: no single coin position may use more than 30% of its own capital.
MAX_COIN_ALLOC = float(os.getenv('MAX_COIN_ALLOC', '0.30'))

# Hard cap: total deployed fraction (sum across all open positions) ≤ 1.0.
MAX_TOTAL_ALLOC = float(os.getenv('MAX_TOTAL_ALLOC', '1.00'))

# Hard ceiling on total open exposure: sum(allocation × leverage) across positions.
MAX_EXPOSURE = float(os.getenv('MAX_EXPOSURE', '3.0'))

# ---- Simulation settings ----
COMMISSION_PCT  = float(os.getenv('COMMISSION_PCT', '0.001'))  # 0.1% per leg (taker fee)
POSITION_PCT    = float(os.getenv('POSITION_PCT',     '1.0'))   # 1.0 = deploy full capital
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '3000.0')) # starting portfolio in USD

# ---- Files ----
TRAINED_PARAMS_FILE = 'trained_params.json'
STATE_DB_FILE       = 'state.db'
KILL_SWITCH_FILE    = 'KILL_SWITCH'
