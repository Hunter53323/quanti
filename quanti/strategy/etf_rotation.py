"""
ETFRotationStrategy: Rising MA ETF Trend Rotation with optional multi-sector support.

Monthly rebalance across ETFs with:
- Trend + ADX + momentum scoring (composite)
- 120-day rising MA filter
- HWM stop and DD breaker
- Multi-sector universe with per-sector concentration limits (new, default)
- Legacy backward-compatible mode (original 6-ETF pool)

Multi-sector mode uses the dynamic ETF universe from ``quanti.config.etf_universe``
with listing-date awareness and concentration limits that cap the number of ETFs
selected from any single industry sector (金融/科技/新能源/消费/资源/TMT/高端制造).
宽基 and 防御 sectors are exempt from concentration limits.
"""
from typing import Optional, Dict, List, Tuple
import numpy as np

from quanti.types import MarketData, Portfolio, Signal, Order, OrderSide, Bar
from quanti.strategy.base import BaseStrategy
from quanti.config import settings
from quanti.config.etf_universe import get_available_etfs, get_sector, ETF_UNIVERSE_LEGACY


class ETFRotationStrategy(BaseStrategy):
    """Rising MA ETF trend rotation with optional multi-sector support.

    Parameters
    ----------
    top_n : int
        Number of ETFs to select each period (default 3).
    ma_period : int
        Period for the SMA used in trend scoring (default 120).
    w_trend, w_adx, w_momentum : float
        Composite score weights (default 0.35, 0.40, 0.25).
    dd_exit_pct : float
        Maximum drawdown before full exit (default 15.0).
    equity_mandate : bool
        Legacy: force at least one equity ETF when all are rising. No-op when
        ``use_multi_sector=True`` (default False).
    use_multi_sector : bool
        When True (default), use the dynamic multi-sector ETF pool from
        ``quanti.config.etf_universe``. When False, use the original 6-ETF legacy
        hardcoded set.
    max_per_sector : int
        Maximum number of ETFs from a single industry sector in the selected set
        (default 2). Only applies when ``use_multi_sector=True`` and only to
        industry sectors (not 宽基 or 防御).
    """

    name = "etf_rotation"

    def __init__(
        self,
        top_n: int = 3,
        ma_period: int = 120,
        w_trend: float = 0.35,
        w_adx: float = 0.40,
        w_momentum: float = 0.25,
        dd_exit_pct: float = 15.0,
        equity_mandate: bool = False,
        use_multi_sector: bool = True,
        max_per_sector: int = 2,
    ):
        self.top_n = top_n
        self.ma_period = ma_period
        self.w_trend = w_trend
        self.w_adx = w_adx
        self.w_momentum = w_momentum
        self.dd_exit_pct = dd_exit_pct
        self.equity_mandate = equity_mandate
        self.use_multi_sector = use_multi_sector
        self.max_per_sector = max_per_sector

        self._max_equity: float = 0.0
        self._dd_exit_active: bool = False
        self._hwm: Dict[str, float] = {}
        self._last_scores: Dict[str, float] = {}
        self._sector_counts: Dict[str, int] = {}

    # ── Public API ───────────────────────────────────

    def generate_signals(self, market_data: MarketData) -> list[Signal]:
        """Score all ETFs, select top N with rising MA."""
        signals: list[Signal] = []
        if not market_data.bars:
            return signals

        # --- Determine ETF universe ---
        if self.use_multi_sector:
            date_str = market_data.timestamp.strftime("%Y%m%d")
            available = get_available_etfs(date_str)
            universe = [etf["code"] for etf in available]
        else:
            universe = ETF_UNIVERSE_LEGACY

        # --- Score ---
        scored: list[Tuple[str, float, bool]] = []
        for sym in universe:
            if sym not in market_data.bars:
                continue
            bars = market_data.bars[sym]
            if len(bars) < self.ma_period + 20:
                continue
            score = self._score_etf(bars)
            rising = self._ma_rising(bars)
            if score > 0 and rising:
                scored.append((sym, score, rising))

        if not scored:
            self._last_scores = {}
            return signals

        scored.sort(key=lambda x: x[1], reverse=True)

        # --- Select ---
        if self.use_multi_sector:
            selected, sector_counts = self._select_with_concentration(scored)
            self._sector_counts = sector_counts
            self._last_scores = {s[0]: s[1] for s in scored[:20]}

            for sym in selected:
                score = next(s[1] for s in scored if s[0] == sym)
                sector = get_sector(sym)
                signals.append(Signal(
                    symbol=sym,
                    side=OrderSide.BUY,
                    strength=round(score, 4),
                    reason=f"Score={score:.3f} trend+adx+momentum [{sector}]",
                ))
        else:
            # --- Legacy: simple top-N with optional equity mandate ---
            selected = {s[0] for s in scored[: self.top_n]}

            # Equity mandate: force at least one 宽基 ETF when all are rising
            if self.equity_mandate:
                equity_etfs = {"510300", "510500", "159915"}
                equity_rising = all(
                    any(s[0] == e and s[2] for s in scored)
                    for e in equity_etfs
                )
                if equity_rising and not (selected & equity_etfs):
                    best_eq = max(
                        (s for s in scored if s[0] in equity_etfs),
                        key=lambda x: x[1], default=None,
                    )
                    if best_eq:
                        worst_in = min(
                            (s for s in scored[: self.top_n] if s[0] in selected),
                            key=lambda x: x[1],
                        )
                        selected.discard(worst_in[0])
                        selected.add(best_eq[0])

            self._last_scores = {s[0]: s[1] for s in scored[:20]}
            self._sector_counts = {}

            for sym in selected:
                score = next(s[1] for s in scored if s[0] == sym)
                signals.append(Signal(
                    symbol=sym,
                    side=OrderSide.BUY,
                    strength=round(score, 4),
                    reason=f"Score={score:.3f} trend+adx+momentum",
                ))

        return signals

    def _select_with_concentration(
        self, scored: list[Tuple[str, float, bool]],
    ) -> Tuple[set[str], Dict[str, int]]:
        """Select top-N ETFs respecting per-sector concentration limits.

        Industry sectors (金融, 科技, 新能源, 消费, 资源, TMT, 高端制造) are
        capped at ``max_per_sector``. 宽基 (broad-based) and 防御 (defensive)
        sectors are exempt.
        """
        selected: set[str] = set()
        sector_counts: Dict[str, int] = {}

        for sym, _score, _rising in scored:
            if len(selected) >= self.top_n:
                break
            sector = get_sector(sym)
            if sector not in ("宽基", "防御"):
                if sector_counts.get(sector, 0) >= self.max_per_sector:
                    continue
            selected.add(sym)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        return selected, sector_counts

    def size_positions(
        self, signals: list[Signal], capital: float, portfolio: Portfolio,
        market_data: Optional[MarketData] = None,
    ) -> list[Order]:
        """Equal-weight. Sell if not in selected set."""
        orders: list[Order] = []
        buy_sigs = [s for s in signals if s.side == OrderSide.BUY]
        selected = {s.symbol for s in buy_sigs}

        for sym, pos in portfolio.positions.items():
            if pos.quantity <= 0:
                continue
            should_sell = False
            reason = ""
            if sym not in selected:
                should_sell = True
                reason = "Rotated out"
            elif sym in self._hwm and pos.current_price > 0:
                loss = (pos.current_price / self._hwm[sym] - 1) * 100
                if loss < -10:
                    should_sell = True
                    reason = f"Stop: {loss:.1f}%"

            if should_sell:
                orders.append(Order(
                    symbol=sym, side=OrderSide.SELL,
                    quantity=pos.quantity, price=None, order_type="market",
                    signal_ref=reason,
                ))

        new = [s for s in buy_sigs if s.symbol not in portfolio.positions]
        if not new:
            return orders

        n = max(len(buy_sigs), 1)
        per = capital / n * 0.92
        for sig in new:
            p = self._get_price(sig.symbol, portfolio, market_data)
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
        self, orders: list[Order], portfolio: Portfolio,
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
            self._dd_exit_active = False

        if self._max_equity > 0 and self.dd_exit_pct > 0:
            dd = (self._max_equity - total) / self._max_equity * 100
            if dd > self.dd_exit_pct:
                self._dd_exit_active = True
            elif self._dd_exit_active and total / self._max_equity > 0.92:
                self._dd_exit_active = False

        if self._dd_exit_active:
            for sym, pos in portfolio.positions.items():
                if pos.quantity > 0:
                    approved.append(Order(
                        symbol=sym, side=OrderSide.SELL,
                        quantity=pos.quantity, price=None, order_type="market",
                        signal_ref=f"DD breaker: -{dd:.1f}%",
                    ))
            return approved

        for sym, pos in portfolio.positions.items():
            if sym not in self._hwm or pos.current_price > self._hwm[sym]:
                self._hwm[sym] = pos.current_price
            if pos.current_price > 0 and self._hwm[sym] > 0:
                loss = (pos.current_price / self._hwm[sym] - 1) * 100
                if loss < -10:
                    approved.append(Order(
                        symbol=sym, side=OrderSide.SELL,
                        quantity=pos.quantity, price=None, order_type="market",
                        signal_ref=f"HWM stop: {loss:.1f}%",
                    ))

        for o in orders:
            cost = o.quantity * (o.price or 0)
            if cost > getattr(settings, "TRADING_CAPITAL", 90000) * 0.4:
                continue
            approved.append(o)
        return approved

    # ── Scoring ────────────────────────────────────

    @staticmethod
    def _sma(arr: np.ndarray, period: int) -> np.ndarray:
        out = np.full(len(arr), np.nan, dtype=np.float64)
        if len(arr) < period:
            return out
        cs = np.cumsum(np.insert(arr, 0, 0.0))
        out[period - 1:] = (cs[period:] - cs[:-period]) / period
        return out

    def _score_etf(self, bars: list[Bar]) -> float:
        closes = np.array([b.close for b in bars], dtype=np.float64)
        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)

        # Trend: above 120MA
        ma = self._sma(closes, self.ma_period)
        trend = 1.0 if (
            not np.isnan(ma[-1]) and closes[-1] > ma[-1]
        ) else 0.0

        # ADX (simplified)
        if len(closes) >= 28:
            adx_v = self._compute_adx(highs, lows, closes, 14)
            adx_val = min(float(adx_v) / 50.0, 1.0) if not np.isnan(adx_v) else 0.5
        else:
            adx_val = 0.5

        # 20-day momentum
        if closes[-21] > 1e-6:
            ret = (closes[-1] / closes[-21] - 1) * 100
            mom = min(max(ret / 15.0, 0), 1) if ret > 0 else 0
        else:
            mom = 0.5

        return 0.35 * trend + 0.40 * adx_val + 0.25 * mom

    def _ma_rising(self, bars: list[Bar]) -> bool:
        closes = np.array([b.close for b in bars], dtype=np.float64)
        if len(closes) < self.ma_period + 20:
            return False
        now = np.mean(closes[-self.ma_period:])
        ago = np.mean(closes[-self.ma_period - 20:-20])
        return now > ago

    @staticmethod
    def _compute_adx(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14,
    ) -> float:
        n = len(close)
        tr = np.zeros(n, dtype=np.float64)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        pdm = np.zeros(n, dtype=np.float64)
        mdm = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            up = high[i] - high[i-1]
            down = low[i-1] - low[i]
            if up > down and up > 0: pdm[i] = up
            if down > up and down > 0: mdm[i] = down

        atr = float(np.mean(tr[1:period+1])) if tr[1:period+1].any() else 0.001
        ps = float(np.mean(pdm[1:period+1]))
        ms = float(np.mean(mdm[1:period+1]))
        for i in range(period + 1, n):
            atr = (tr[i] + (period - 1) * atr) / period
            ps = (pdm[i] + (period - 1) * ps) / period
            ms = (mdm[i] + (period - 1) * ms) / period

        denom = max(atr, 0.001)
        pdi = min(ps / denom * 100, 1000.0)
        mdi = min(ms / denom * 100, 1000.0)
        return float(np.abs(pdi - mdi) / (pdi + mdi + 1e-10) * 100)

    @staticmethod
    def _get_price(
        symbol: str, portfolio: Portfolio, market_data: Optional[MarketData],
    ) -> Optional[float]:
        pos = portfolio.positions.get(symbol)
        if pos and pos.current_price > 0:
            return pos.current_price
        if market_data and symbol in market_data.bars:
            bars = market_data.bars[symbol]
            if bars:
                return bars[-1].close
        return None
