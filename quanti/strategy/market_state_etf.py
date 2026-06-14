"""
Market-state-aware multi-sector ETF rotation strategy.

Full pipeline:
  1. CSI300 120MA confirmation system → market state (0-4)
  2. 5-condition ETF trend filter + momentum scoring → ranked candidates
  3. State-dependent position sizing with A43 decay
  4. Sharp3pct flash exit → bond/gold switch
  5. HWM stop-loss + DD circuit breaker
  6. Per-sector concentration limit (max 2/sector)

Implements the BaseStrategy interface so it plugs into BacktestEngine directly.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np

from quanti.config.etf_universe import get_available_etfs, get_sector
from quanti.indicators import sma as _sma_shared, adx_with_di
from quanti.strategy.base import BaseStrategy
from quanti.types import Bar, MarketData, Order, OrderSide, Portfolio, Signal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOND_ETF = "511880"
GOLD_ETF = "518880"
CSI300_CODE = "510300"

# Market state enum (matching run_backtest.py conventions)
ST_BEAR      = 0  # below 60MA
ST_CONFIRMING = 1  # above 120MA, accumulating N days
ST_BULL       = 2  # confirmed bull (N days + volume/return check)
ST_COOLDOWN   = 3  # M-day mandatory cooldown after 120MA break
ST_FAKE       = 4  # N days confirmed but volume/return failed


# ---------------------------------------------------------------------------
# Market state helper (pure function, no side effects)
# ---------------------------------------------------------------------------

def build_market_state(
    dates: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    n_confirm: int = 5,
    m_cooldown: int = 40,
    vol_threshold: float = 0.85,
    return_threshold: float = -0.02,
) -> dict[str, int]:
    """CSI300 120MA confirmation state machine.

    Returns a dict mapping ``trade_date`` (str) → state (int).

    Parameters
    ----------
    n_confirm : int
        Days above 120MA required for confirmation.
    m_cooldown : int
        Mandatory cooldown days after 120MA break.
    vol_threshold : float
        Minimum avg_volume / prev_volume ratio for confirmation.
    return_threshold : float
        Minimum price-return during confirmation window.
    """
    n = len(closes)
    ma120 = _sma_shared(closes, 120)
    ma60 = _sma_shared(closes, 60)
    above_120 = (closes > ma120) & (~np.isnan(ma120))
    above_60  = (closes > ma60)  & (~np.isnan(ma60))

    states = np.full(n, 0, dtype=int)
    cd_end = -1      # cooldown end index
    cf_start = -1    # confirmation start index
    bull_start = -1  # last confirmed bull start

    for i in range(121, n):
        # cooldown
        if i <= cd_end:
            states[i] = ST_COOLDOWN
            continue

        # sustained bull (still above 120MA after confirmation)
        if bull_start >= 0:
            if above_120[i]:
                states[i] = ST_BULL
                continue
            else:
                bull_start = -1

        # mid-confirmation
        if cf_start >= 0:
            if not above_120[i]:
                cd_end = i + m_cooldown - 1
                states[i] = ST_COOLDOWN
                cf_start = -1
                continue
            if i - cf_start + 1 == n_confirm:
                ws, we = cf_start, i + 1
                wvol = float(np.mean(volumes[ws:we]))
                pvol = float(np.mean(volumes[max(0, ws - 20):ws]))
                chg  = closes[i] / closes[cf_start] - 1.0
                if wvol >= vol_threshold * pvol and chg >= return_threshold:
                    states[i] = ST_BULL
                    bull_start = i
                else:
                    states[i] = ST_FAKE
                    cf_start = -1
            else:
                states[i] = ST_CONFIRMING
            continue

        # detect new break above 120MA
        if above_120[i] and not above_120[i - 1]:
            cf_start = i
            states[i] = ST_CONFIRMING
        elif above_60[i]:
            states[i] = ST_FAKE
        else:
            states[i] = ST_BEAR

    return {str(dates[j]): int(states[j]) for j in range(n)}


# ---------------------------------------------------------------------------
# ETF trend screening & scoring helpers
# ---------------------------------------------------------------------------

def _is_etf_uptrend(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray,
    min_score: int = 3,
) -> tuple[bool, int]:
    """5-condition ETF trend filter. Returns (passed, cond_count)."""
    n = len(closes)
    if n < 200:
        return False, 0

    ma120 = _sma_shared(closes, 120)
    if ma120 is None or np.isnan(ma120[-1]):
        return False, 0

    # 1. above 120MA
    above = closes[-1] > ma120[-1]

    # 2. structural highs/lows
    rh20 = float(np.max(highs[-20:]))
    ph20 = float(np.max(highs[-60:-20])) if n >= 80 else 0.0
    rl20 = float(np.min(lows[-20:]))
    pl20 = float(np.min(lows[-60:-20])) if n >= 80 else 0.0
    structure = rh20 > ph20 and rl20 > pl20

    # 3. MA alignment
    ma20 = _sma_shared(closes, 20)
    ma60_val = _sma_shared(closes, 60)
    alignment = (
        ma20 is not None and ma60_val is not None
        and not np.isnan(ma20[-1]) and not np.isnan(ma60_val[-1])
        and ma20[-1] > ma60_val[-1] > ma120[-1]
    )

    # 4. ADX > 25
    di_result = adx_with_di(highs, lows, closes, 14)
    adx_ok = di_result is not None and not np.isnan(di_result[0][-1]) and di_result[0][-1] > 25

    # 5. volume surge
    if n >= 22:
        v20 = float(np.mean(volumes[-21:-1]))
        surge = volumes[-1] > v20 * 1.2 if v20 > 0 else False
    else:
        surge = False

    conds = sum([above, structure, alignment, adx_ok, surge])
    return (above and adx_ok and conds >= min_score), conds


def _trend_score(closes: np.ndarray) -> float:
    """3M + 6M momentum + stability (0-100 scale)."""
    n = len(closes)
    if n < 130:
        return 0.0

    # momentum
    r3 = closes[-1] / closes[-63] - 1.0 if closes[-63] > 1e-6 else 0.0
    r6 = closes[-1] / closes[-126] - 1.0 if closes[-126] > 1e-6 else 0.0
    m3 = min(max(r3 / 0.5, 0.0), 1.0) if r3 > 0 else 0.0
    m6 = min(max(r6 / 0.8, 0.0), 1.0) if r6 > 0 else 0.0
    mom = (0.5 * m3 + 0.5 * m6) * 100.0

    # stability
    window = closes[-61:]
    daily_ret = np.diff(window) / (window[:-1] + 1e-10)
    vol = min(np.nanstd(daily_ret) / 0.04, 1.0) if len(daily_ret) > 1 else 1.0
    stability = (1.0 - vol) * 100.0

    return 0.6 * mom + 0.4 * stability


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class MarketStateETFStrategy(BaseStrategy):
    """Multi-sector ETF rotation with market-state-aware defense.

    Parameters
    ----------
    csi300_bars : list[ETFDailyBar]
        Full CSI300 (510300) history for market state computation.
    top_n : int
        Number of ETFs to hold in equity mode (default 3).
    n_confirm : int
        Days above 120MA to confirm a bull market (default 5).
    m_cooldown : int
        Mandatory cooldown days after bull break (default 40).
    sharp_threshold : float
        CSI300 5-day return threshold for flash exit (default -0.03 i.e. -3%).
    bond_pct : float
        Defensive allocation to bond ETF (default 0.80).
    gold_pct : float
        Defensive allocation to gold ETF (default 0.20).
    hwm_stop_pct : float
        Per-position high-water-mark stop-loss (default -10.0).
    dd_exit_pct : float
        Global drawdown circuit breaker (default 15.0).
    max_per_sector : int
        Maximum ETFs from a single industry sector (default 2).
    """

    name = "market_state_etf"

    def __init__(
        self,
        csi300_bars: list | None = None,
        top_n: int = 3,
        n_confirm: int = 5,
        m_cooldown: int = 40,
        sharp_threshold: float = -0.03,
        bond_pct: float = 0.80,
        gold_pct: float = 0.20,
        hwm_stop_pct: float = -10.0,
        dd_exit_pct: float = 15.0,
        max_per_sector: int = 2,
    ):
        self.top_n = top_n
        self.n_confirm = n_confirm
        self.m_cooldown = m_cooldown
        self.sharp_threshold = sharp_threshold
        self.bond_pct = bond_pct
        self.gold_pct = gold_pct
        self.hwm_stop_pct = hwm_stop_pct
        self.dd_exit_pct = dd_exit_pct
        self.max_per_sector = max_per_sector

        # --- precompute market state map from CSI300 bars ---
        self._state_map: dict[str, int] = {}
        if csi300_bars is not None:
            try:
                # csi300_bars may be ETFDailyBar or quanti.types.Bar
                dates_arr = np.array([b.trade_date for b in csi300_bars])
                closes_arr = np.array([b.close for b in csi300_bars], dtype=np.float64)
                volumes_arr = np.array([getattr(b, "volume", 0.0) for b in csi300_bars], dtype=np.float64)
                if len(closes_arr) >= 200:
                    self._state_map = build_market_state(
                        dates_arr, closes_arr, volumes_arr,
                        n_confirm=self.n_confirm,
                        m_cooldown=self.m_cooldown,
                    )
            except Exception:
                self._state_map = {}

        # --- CSI300 5-day return cache (for Sharp exit) ---
        self._csi_ret5: dict[str, float] = {}
        if csi300_bars is not None:
            try:
                closes_arr = np.array([b.close for b in csi300_bars], dtype=np.float64)
                dates_arr = [b.trade_date for b in csi300_bars]
                for i in range(5, len(closes_arr)):
                    self._csi_ret5[dates_arr[i]] = float(
                        closes_arr[i] / closes_arr[i - 5] - 1.0
                    )
            except Exception:
                self._csi_ret5 = {}

        # --- mutable state (persists across calls) ---
        self._hwm: Dict[str, float] = {}
        self._max_equity: float = 0.0
        self._dd_active: bool = False
        self._sharp_cd: str | None = None          # cooldown end date (YYYYMMDD)
        self._months_in_cycle: int = 0
        self._prev_mst: int = 0
        self._genuine_prev: int = 0
        self._last_rebalance_month: str = ""        # "YYYYMM"
        self._cached_signals: list[Signal] = []       # replay on non-rebalance days
        self._last_scores: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _month_of(date_str: str) -> str:
        return date_str[:6]

    def _is_month_end(self, date_str: str, market_data: MarketData) -> bool:
        """True when *date_str* is the last trading day in its calendar month
        among all bars in *market_data*."""
        this_month = date_str[:6]
        for bars in market_data.bars.values():
            for b in bars:
                bd = b.trade_date if hasattr(b, 'trade_date') else b.datetime.strftime("%Y%m%d")
                if bd[:6] == this_month and bd > date_str:
                    return False  # another day in this month exists
        return True

    # ------------------------------------------------------------------
    # Public API (BaseStrategy)
    # ------------------------------------------------------------------

    def generate_signals(self, market_data: MarketData) -> list[Signal]:
        """Score ETFs with market-state gating.

        Called by BacktestEngine for EVERY trading day.  Full recomputation
        only on month boundaries; mid-month returns the cached set so that
        size_positions does not rotate out existing holdings.  Sharp flash
        exit is checked every day and can override the cached set.
        """
        ts = market_data.timestamp
        date_str = ts.strftime("%Y%m%d")
        month_key = date_str[:6]
        is_month_end = self._is_month_end(date_str, market_data)

        # --- check Sharp exit every day (can fire mid-month) ---
        sharp_fired = (
            date_str in self._csi_ret5
            and self._csi_ret5[date_str] < self.sharp_threshold
        )
        if self._sharp_cd is not None and date_str >= self._sharp_cd:
            self._sharp_cd = None
        if sharp_fired:
            self._sharp_cd = _add_trading_days(date_str, self.m_cooldown)
        in_sharp_cd = self._sharp_cd is not None and date_str < self._sharp_cd

        # --- mid-month: replay cached signals (Sharp overrides) ---
        if not is_month_end:
            if not sharp_fired and not in_sharp_cd and self._cached_signals:
                return self._cached_signals

        # --- monthly state update ---
        raw_mst = self._state_map.get(date_str, 0)
        effective_mst = ST_COOLDOWN if in_sharp_cd else raw_mst

        if is_month_end:
            self._last_rebalance_month = month_key
            if effective_mst in (ST_BULL, ST_FAKE):
                if self._genuine_prev not in (ST_BULL, ST_FAKE):
                    self._months_in_cycle = 1
                else:
                    self._months_in_cycle += 1
                self._genuine_prev = effective_mst
            elif effective_mst == ST_COOLDOWN and raw_mst != ST_COOLDOWN:
                pass  # Sharp cooldown: don't touch cycle
            else:
                self._months_in_cycle = 0
                self._genuine_prev = effective_mst
        self._prev_mst = effective_mst

        # --- determine equity / defensive ---
        base_sm = {ST_BULL: 1.0, ST_FAKE: 0.5}.get(effective_mst, 0.0)

        if base_sm == 0.0 or sharp_fired or in_sharp_cd:
            signals: list[Signal] = []
            if self.bond_pct > 0:
                signals.append(Signal(
                    symbol=BOND_ETF, side=OrderSide.BUY, strength=1.0,
                    reason=f"defensive bond {self.bond_pct:.0%} "
                           + (f"sharp={self.sharp_threshold:+.0%}" if sharp_fired
                              else f"mst={effective_mst}"),
                ))
            if self.gold_pct > 0:
                signals.append(Signal(
                    symbol=GOLD_ETF, side=OrderSide.BUY, strength=0.8,
                    reason=f"defensive gold {self.gold_pct:.0%} "
                           + (f"sharp={self.sharp_threshold:+.0%}" if sharp_fired
                              else f"mst={effective_mst}"),
                ))
            self._last_scores = {}
            self._cached_signals = signals
            return signals

        # --- equity mode: screen + score + rank ---
        available = get_available_etfs(date_str)
        universe = [e["code"] for e in available]

        scored: list[Tuple[str, float]] = []
        for sym in universe:
            if sym not in market_data.bars:
                continue
            bars = market_data.bars[sym]
            if len(bars) < 200:
                continue
            closes = np.array([b.close for b in bars], dtype=np.float64)
            highs  = np.array([b.high for b in bars], dtype=np.float64)
            lows   = np.array([b.low for b in bars], dtype=np.float64)
            vols   = np.array([b.volume for b in bars], dtype=np.float64)
            passed, _ = _is_etf_uptrend(closes, highs, lows, vols)
            if not passed:
                continue
            score = _trend_score(closes)
            scored.append((sym, score))

        if not scored:
            self._last_scores = {}
            self._cached_signals = []
            return []

        scored.sort(key=lambda x: x[1], reverse=True)

        selected: set[str] = set()
        sector_counts: Dict[str, int] = {}
        for sym, s in scored:
            if len(selected) >= self.top_n:
                break
            sector = get_sector(sym)
            if sector not in ("宽基", "防御"):
                if sector_counts.get(sector, 0) >= self.max_per_sector:
                    continue
            selected.add(sym)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        self._last_scores = {s[0]: s[1] for s in scored[:20]}

        signals = []
        for sym in selected:
            score = next(s[1] for s in scored if s[0] == sym)
            sector = get_sector(sym)
            signals.append(Signal(
                symbol=sym, side=OrderSide.BUY, strength=round(score / 100.0, 4),
                reason=f"Score={score:.1f} mst={effective_mst} [{sector}]",
            ))
        self._cached_signals = signals
        return signals

    def size_positions(
        self,
        signals: list[Signal],
        capital: float,
        portfolio: Portfolio,
        market_data: Optional[MarketData] = None,
    ) -> list[Order]:
        """Size positions with A43 decay and equal weighting.

        - Equity ETFs get equal weight × decay(sm, months_in_cycle)
        - Bond/gold get fixed pct split
        - Non-selected positions are sold
        """
        orders: list[Order] = []
        buy_sigs = [s for s in signals if s.side == OrderSide.BUY]
        selected = {s.symbol for s in buy_sigs}

        # --- Decay multiplier ---
        decay = _a43_decay(self._months_in_cycle)
        base_sm = 0.0
        if self._prev_mst == ST_BULL:
            base_sm = 1.0
        elif self._prev_mst == ST_FAKE:
            base_sm = 0.5
        sm = base_sm * decay if base_sm > 0 else 1.0  # sm=1 for defensive (bond/gold)

        # --- Sell non-selected positions ---
        for sym, pos in portfolio.positions.items():
            if pos.quantity <= 0:
                continue
            should_sell = False
            reason = ""
            if sym not in selected:
                should_sell = True
                reason = "Rotated out"
            elif sym in self._hwm and pos.current_price > 0:
                loss = (pos.current_price / self._hwm[sym] - 1.0) * 100.0
                if loss < self.hwm_stop_pct:
                    should_sell = True
                    reason = f"HWM stop: {loss:.1f}%"

            if should_sell:
                orders.append(Order(
                    symbol=sym, side=OrderSide.SELL,
                    quantity=pos.quantity, price=None, order_type="market",
                    signal_ref=reason,
                ))

        # --- Buy new positions ---
        new_sigs = [s for s in buy_sigs if s.symbol not in portfolio.positions]
        if not new_sigs:
            return orders

        # Check if we're in defensive mode (bond/gold signals)
        is_defensive = BOND_ETF in selected or GOLD_ETF in selected

        if is_defensive:
            # Fixed split
            for sig in new_sigs:
                pct = self.bond_pct if sig.symbol == BOND_ETF else self.gold_pct
                alloc = capital * pct * 0.98
                p = _resolve_price(sig.symbol, portfolio, market_data)
                if p and p > 0.01:
                    q = int(alloc / p / 100) * 100
                    if q >= 100:
                        orders.append(Order(
                            symbol=sig.symbol, side=OrderSide.BUY,
                            quantity=q, price=p, order_type="limit",
                            signal_ref=f"defensive {pct:.0%} {sig.reason}",
                        ))
        else:
            # Equal weight with decay
            n = max(len(new_sigs), 1)
            total_capital = capital + sum(
                p.quantity * p.current_price
                for p in portfolio.positions.values()
                if p.symbol in selected
            )
            per = total_capital * sm / n * 0.90

            for sig in new_sigs:
                p = _resolve_price(sig.symbol, portfolio, market_data)
                if p and p > 0.01:
                    q = int(per / p / 100) * 100
                    if q >= 100:
                        orders.append(Order(
                            symbol=sig.symbol, side=OrderSide.BUY,
                            quantity=q, price=p, order_type="limit",
                            signal_ref=sig.reason,
                        ))

        return orders

    def risk_check(
        self,
        orders: list[Order],
        portfolio: Portfolio,
        market_data: Optional[MarketData] = None,
        risk_checker: object = None,
    ) -> list[Order]:
        """DD breaker + per-position HWM stops."""
        approved: list[Order] = []

        total = portfolio.cash + sum(
            p.quantity * p.current_price for p in portfolio.positions.values()
        )
        if total > self._max_equity:
            self._max_equity = total
            self._dd_active = False

        if self._max_equity > 0 and self.dd_exit_pct > 0:
            dd = (self._max_equity - total) / self._max_equity * 100.0
            if dd > self.dd_exit_pct:
                self._dd_active = True
            elif self._dd_active and total / self._max_equity > 0.92:
                self._dd_active = False

        if self._dd_active:
            for sym, pos in portfolio.positions.items():
                if pos.quantity > 0:
                    approved.append(Order(
                        symbol=sym, side=OrderSide.SELL,
                        quantity=pos.quantity, price=None, order_type="market",
                        signal_ref=f"DD breaker: -{dd:.1f}%",
                    ))
            return approved

        # HWM update + stop
        for sym, pos in portfolio.positions.items():
            if sym not in self._hwm or pos.current_price > self._hwm[sym]:
                self._hwm[sym] = pos.current_price
            if pos.current_price > 0 and self._hwm[sym] > 0:
                loss = (pos.current_price / self._hwm[sym] - 1.0) * 100.0
                if loss < self.hwm_stop_pct:
                    approved.append(Order(
                        symbol=sym, side=OrderSide.SELL,
                        quantity=pos.quantity, price=None, order_type="market",
                        signal_ref=f"HWM stop: {loss:.1f}%",
                    ))

        for o in orders:
            approved.append(o)
        return approved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _a43_decay(months_in_cycle: int) -> float:
    """A43 decay: 1.0 (m1-4) → 0.75 (m5-8) → 0.50 (m9+)."""
    if months_in_cycle <= 4:
        return 1.0
    if months_in_cycle <= 8:
        return 0.75
    return 0.50


def _add_trading_days(date_str: str, days: int) -> str:
    """Add N calendar days (crude proxy for trading days)."""
    dt = datetime.strptime(date_str, "%Y%m%d")
    from datetime import timedelta
    return (dt + timedelta(days=days)).strftime("%Y%m%d")


def _resolve_price(
    symbol: str,
    portfolio: Portfolio,
    market_data: Optional[MarketData],
) -> Optional[float]:
    """Get current price from portfolio position or market data."""
    pos = portfolio.positions.get(symbol)
    if pos and pos.current_price > 0:
        return pos.current_price
    if market_data and symbol in market_data.bars:
        bars = market_data.bars[symbol]
        if bars:
            return bars[-1].close
    return None
