"""
Production-Ready State Machine Strategy
========================================
Adapted from the v1_best research configuration for incremental live deployment.

Configuration (v1_best from 81-param sweep, Train 2015-2021 / Test 2022-2025):
  ADX threshold  = 25
  Breadth bull    = 45%
  BEAR->RANGE     = 5-day confirmation
  RANGE->BULL     = 2-day confirmation

Deployment adaptations vs research backtest:
  - Weekly rebalancing (first trading day of each calendar week)
  - ATR(14) trailing stop at 2x ATR from position HWM
  - 100-share round-lot constraint (A-share standard)
  - Minimum position size: 10,000 CNY notional
  - Dual-momentum filter: only buy stocks with positive 60-day return
  - Max positions: 8 (reduces concentration risk from Top-5)
  - 511880 money-market ETF as cash vehicle

Signal output: JSON file consumable by paper/live execution systems.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Configuration — single source of truth, tunable for deployment
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StateMachineConfig:
    """Immutable strategy configuration.  Change here to tune the live strategy."""

    # ── State machine (from v1_best) ──
    adx_threshold: float = 25.0
    breadth_bull_pct: float = 45.0
    confirm_bear_to_range_days: int = 5
    confirm_range_to_bull_days: int = 2

    # ── Position sizing & risk ──
    max_positions: int = 8          # max stocks held simultaneously
    atr_stop_multiplier: float = 2.0  # trailing stop = HWM - N * ATR(14)
    min_notional_per_position: float = 10_000.0  # minimum CNY per position
    round_lot: int = 100            # A-share standard lot size
    dual_momentum_filter: bool = True  # require ret_60d > 0 to enter

    # ── State exposure ──
    bull_exposure: float = 1.0      # fraction of capital in stocks during BULL
    range_exposure: float = 0.50    # fraction during RANGE
    bear_exposure: float = 0.0      # fraction during BEAR (all in 511880)

    # ── Execution ──
    rebalance_frequency: str = "weekly"  # "weekly" or "monthly"
    cash_etf: str = "511880"
    csi300_proxy: str = "510300"
    commission: float = 0.00025     # one-way

    # ── Signal output ──
    signal_output_dir: str = "data/signals"


# ═══════════════════════════════════════════════════════════════
# Pure functions for indicator computation
# ═══════════════════════════════════════════════════════════════

def rolling_mean(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average. Returns array of same length with NaN prefix."""
    if len(arr) < period:
        return np.full(len(arr), np.nan)
    out = np.full(len(arr), np.nan)
    cs = np.cumsum(np.insert(arr, 0, 0.0))
    out[period - 1:] = (cs[period:] - cs[:-period]) / period
    return out


def adx_series(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    """ADX indicator.  Returns array of same length with NaN prefix."""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))

    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]; dn = low[i - 1] - low[i]
        if up > dn and up > 0: pdm[i] = up
        if dn > up and dn > 0: mdm[i] = dn

    atr_arr = np.full(n, np.nan)
    atr_arr[period] = float(np.mean(tr[1:period + 1]))
    for i in range(period + 1, n):
        atr_arr[i] = (tr[i] + (period - 1) * atr_arr[i - 1]) / period

    ps = float(np.mean(pdm[1:period + 1]))
    ms = float(np.mean(mdm[1:period + 1]))
    pi_arr = np.full(n, np.nan); mi_arr = np.full(n, np.nan)
    pi_arr[period] = ps / max(atr_arr[period], 0.001) * 100
    mi_arr[period] = ms / max(atr_arr[period], 0.001) * 100
    for i in range(period + 1, n):
        ps = (pdm[i] + (period - 1) * ps) / period
        ms = (mdm[i] + (period - 1) * ms) / period
        pi_arr[i] = min(ps / max(atr_arr[i], 0.001) * 100, 1000)
        mi_arr[i] = min(ms / max(atr_arr[i], 0.001) * 100, 1000)

    dx_arr = np.abs(pi_arr - mi_arr) / (pi_arr + mi_arr + 1e-10) * 100
    ax_arr = np.full(n, np.nan)
    seed = float(np.nanmean(dx_arr[period:period * 2]))
    ax_arr[period * 2 - 1] = 0.0 if np.isnan(seed) else seed
    ds = ax_arr[period * 2 - 1]
    for i in range(period * 2, n):
        vi = dx_arr[i] if not np.isnan(dx_arr[i]) else ds
        ds = (vi + (period - 1) * ds) / period
        ax_arr[i] = ds
    return ax_arr


def atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    """Average True Range.  Returns array of same length with NaN prefix."""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr_arr = np.full(n, np.nan)
    atr_arr[period] = float(np.mean(tr[1:period + 1]))
    for i in range(period + 1, n):
        atr_arr[i] = (tr[i] + (period - 1) * atr_arr[i - 1]) / period
    return atr_arr


def realized_volatility(closes: np.ndarray, window: int = 60) -> float:
    """Annualized realized volatility over window."""
    if len(closes) < window + 2:
        return np.nan
    rets = np.diff(closes[-window - 1:]) / (closes[-window - 1:-1] + 1e-10)
    return float(np.nanstd(rets) * np.sqrt(252))


# ═══════════════════════════════════════════════════════════════
# State Machine — determines BULL / RANGE / BEAR each day
# ═══════════════════════════════════════════════════════════════

class DailyStateMachine:
    """Incrementally buildable state machine for live use.

    Call .update(date, csi_close, csi_high, csi_low, breadth) each day;
    the object maintains rolling state and returns the confirmed regime.
    """

    def __init__(self, config: StateMachineConfig):
        self.cfg = config
        # Rolling CSI300 data
        self._dates: list[str] = []
        self._closes: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._breadths: list[float] = []

        # Raw state history
        self._raw_states: list[int] = []
        self._confirmed_states: list[int] = [0]  # start in BEAR

    def update(self, date_str: str, csi_close: float, csi_high: float,
               csi_low: float, breadth: float) -> int:
        """Ingest one day of CSI300 data, return confirmed state (0=BEAR, 1=RANGE, 2=BULL)."""
        self._dates.append(date_str)
        self._closes.append(csi_close)
        self._highs.append(csi_high)
        self._lows.append(csi_low)
        self._breadths.append(breadth)

        # ── Compute raw state ──
        n = len(self._closes)
        raw_state = 0  # default BEAR

        if n >= 120:
            c_arr = np.array(self._closes, dtype=np.float64)
            h_arr = np.array(self._highs, dtype=np.float64)
            l_arr = np.array(self._lows, dtype=np.float64)

            ma120_arr = rolling_mean(c_arr, 120)
            adx_arr = adx_series(h_arr, l_arr, c_arr, 14)

            ma120_val = ma120_arr[-1]
            adx_val = adx_arr[-1]
            breadth_val = self._breadths[-1]

            if not np.isnan(ma120_val) and c_arr[-1] > ma120_val:
                adx_ok = (not np.isnan(adx_val)) and adx_val > self.cfg.adx_threshold
                breadth_ok = (not np.isnan(breadth_val)) and breadth_val > self.cfg.breadth_bull_pct

                if adx_ok and breadth_ok:
                    raw_state = 2  # BULL
                else:
                    raw_state = 1  # RANGE

        self._raw_states.append(raw_state)

        # ── Apply hysteresis for confirmed state ──
        prev_conf = self._confirmed_states[-1]
        cbr = self.cfg.confirm_bear_to_range_days
        crb = self.cfg.confirm_range_to_bull_days

        if prev_conf == 0:  # BEAR
            if n >= cbr:
                window = self._raw_states[-cbr:]
                if all(s >= 1 for s in window):
                    if n >= crb and all(s == 2 for s in self._raw_states[-crb:]):
                        conf = 2  # straight to BULL
                    else:
                        conf = 1  # to RANGE
                else:
                    conf = 0
            else:
                conf = 0

        elif prev_conf == 1:  # RANGE
            if raw_state == 0:
                conf = 0  # immediate BEAR
            elif n >= crb and all(s == 2 for s in self._raw_states[-crb:]):
                conf = 2  # to BULL
            else:
                conf = 1

        elif prev_conf == 2:  # BULL
            if raw_state == 0:
                conf = 0  # immediate BEAR
            elif raw_state == 1:
                conf = 1  # immediate RANGE
            else:
                conf = 2

        self._confirmed_states.append(conf)
        return conf

    @property
    def current_state(self) -> int:
        return self._confirmed_states[-1] if self._confirmed_states else 0

    @property
    def n_days(self) -> int:
        return len(self._dates)


# ═══════════════════════════════════════════════════════════════
# Stock scoring — same composite as research (60% mom, 30% trend, 10% low-vol)
# ═══════════════════════════════════════════════════════════════

def score_stock(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                volumes: np.ndarray,
                w_mom: float = 0.60, w_trend: float = 0.30, w_lowvol: float = 0.10
                ) -> tuple[float, float, bool]:
    """Composite trend-momentum score for a single stock.

    Returns (score, ret_60d, passes_dual_momentum).
    """
    n = len(closes)
    if n < 130:
        return -999.0, 0.0, False

    # ── Momentum (3M + 6M normalized) ──
    r3 = closes[-1] / closes[-63] - 1.0 if closes[-63] > 1e-6 else 0.0
    r6 = closes[-1] / closes[-126] - 1.0 if closes[-126] > 1e-6 else 0.0
    m3 = min(max(r3 / 0.5, 0.0), 1.0) if r3 > 0 else max(r3 / 0.3, -1.0)
    m6 = min(max(r6 / 0.8, 0.0), 1.0) if r6 > 0 else max(r6 / 0.5, -1.0)
    mom_score = (0.5 * m3 + 0.5 * m6) * 100.0

    # ── Trend quality ──
    ma120_arr = rolling_mean(closes, 120)
    above_ma = 1.0 if (not np.isnan(ma120_arr[-1]) and closes[-1] > ma120_arr[-1]) else 0.0

    rh = np.max(highs[-20:]); ph = np.max(highs[-60:-20])
    rl = np.min(lows[-20:]);  pl = np.min(lows[-60:-20])
    hhll = 1.0 if (rh > ph and rl > pl) else 0.0

    ma20_arr = rolling_mean(closes, 20)
    ma60_arr = rolling_mean(closes, 60)
    aligned = 0.0
    if (not np.isnan(ma20_arr[-1]) and not np.isnan(ma60_arr[-1])
        and not np.isnan(ma120_arr[-1])
        and ma20_arr[-1] > ma60_arr[-1] > ma120_arr[-1]):
        aligned = 1.0

    adx_val = adx_series(highs, lows, closes, 14)[-1]
    adx_norm = min(max((adx_val - 15.0) / 35.0, 0.0), 1.0) if not np.isnan(adx_val) else 0.0

    trend_comp = (0.35 * above_ma + 0.25 * aligned + 0.20 * adx_norm + 0.20 * hhll) * 100.0

    # ── Low-volatility ──
    w = closes[-61:]
    dr = np.diff(w) / (w[:-1] + 1e-10)
    vol_score = (1.0 - min(np.nanstd(dr) / 0.05, 1.0)) * 100.0

    # ── Dual momentum check ──
    ret_60d = (closes[-1] / closes[-60] - 1.0) if closes[-60] > 1e-6 else 0.0

    score = w_mom * mom_score + w_trend * trend_comp + w_lowvol * vol_score
    return float(score), float(ret_60d), (ret_60d > 0)


def atr_value(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    """Latest ATR(14) for stop-loss placement."""
    atr_arr = atr_series(highs, lows, closes, 14)
    return float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0


# ═══════════════════════════════════════════════════════════════
# Signal Output — JSON consumable by execution system
# ═══════════════════════════════════════════════════════════════

@dataclass
class LiveSignal:
    """Output of one signal generation run."""
    timestamp: str                          # ISO format
    state: str                              # "BEAR" | "RANGE" | "BULL"
    state_raw: str                          # raw state before hysteresis
    target_positions: list[dict]            # [{"symbol", "target_shares", "score", "stop_price"}, ...]
    current_positions: list[dict]           # positions that should be held (including stops)
    exit_positions: list[str]               # symbols to exit
    warnings: list[str]                     # human-readable warnings
    metrics: dict                           # {"breadth": float, "adx": float, "csi300_close": float, "ma120": float}


def to_json(signal: LiveSignal) -> str:
    return json.dumps({
        "timestamp": signal.timestamp,
        "state": signal.state,
        "state_raw": signal.state_raw,
        "target_positions": signal.target_positions,
        "current_positions": signal.current_positions,
        "exit_positions": signal.exit_positions,
        "warnings": signal.warnings,
        "metrics": signal.metrics,
    }, ensure_ascii=False, indent=2)
