"""
Trend-First Momentum Strategy (V2 Final).

Monthly rebalance workflow:
1. Market gate: CSI300 must be above 120-day MA (bull market only)
2. Stock filter: only stocks in confirmed uptrend (5-condition check)
3. Rank trending stocks by momentum + low-vol composite
4. Buy top N, equal-weight, set 10% trailing stop from HWM
5. Portfolio drawdown breaker: if >20% from peak, liquidate to cash
"""

import numpy as np

from quanti.config import settings
from quanti.indicators import adx, sma
from quanti.strategy.base import BaseStrategy
from quanti.types import Bar, MarketData, Order, OrderSide, Portfolio, Signal


class StockMomentumStrategy(BaseStrategy):
    """
    Trend-first momentum strategy for CSI300+CSI500 constituents.

    Entry rules:
    - Market: CSI300 > SMA120
    - Stock: >=3 of 5 trend conditions met (price>MA120, higher highs/lows,
             MA alignment, ADX>25, volume expansion)
    - Selection: top N by momentum(60%) + low-vol(40%)

    Exit rules:
    - Individual: trailing stop at -10% from high-water-mark
    - Portfolio: if drawdown from peak > 20%, liquidate all
    - Rotation: sell if no longer in top N at monthly rebalance
    """

    name = "trend_first_momentum"

    def __init__(
        self,
        top_n: int = 5,
        stop_loss_pct: float = -10.0,
        min_trend_score: int = 3,
        market_trend_required: bool = True,
        dd_exit_pct: float = 20.0,
        ma_filter_period: int = 120,
    ):
        self.top_n = top_n
        self.stop_loss_pct = stop_loss_pct
        self.min_trend_score = min_trend_score
        self.market_trend_required = market_trend_required
        self.dd_exit_pct = dd_exit_pct
        self.ma_filter_period = ma_filter_period

        self._hwm: dict[str, float] = {}
        self._last_scores: dict[str, float] = {}
        self._max_equity: float = 0.0
        self._dd_exit_active: bool = False
        self._dd_exit_days: int = 0  # days spent in cash after DD exit triggered

    # ── Public API ───────────────────────────────────

    def generate_signals(self, market_data: MarketData) -> list[Signal]:
        """
        Monthly rebalance entry.
        Returns BUY for selected stocks, empty list in bear market.
        """
        signals: list[Signal] = []

        if not market_data.bars:
            return signals

        # ── Market gate: CSI300 > SMA120 ──
        mkt_ok = True
        if self.market_trend_required:
            mkt_ok = self._is_market_trending(market_data)

        if not mkt_ok:
            self._last_scores = {}
            return signals

        # ── Score all stocks ──
        trending: list[tuple[str, float, int]] = []
        for symbol, bars in market_data.bars.items():
            if len(bars) < 200:
                continue
            is_t, cond_count = self._is_stock_trending(bars)
            if not is_t or cond_count < self.min_trend_score:
                continue
            score = self._trend_strength_score(bars)
            if score > 0:
                trending.append((symbol, score, cond_count))

        if not trending:
            self._last_scores = {}
            return signals

        # ── Select top N ──
        trending.sort(key=lambda x: x[1], reverse=True)
        selected = {t[0] for t in trending[: self.top_n]}
        self._last_scores = {t[0]: t[1] for t in trending[:min(50, len(trending))]}

        for sym in selected:
            strength = min(self._last_scores[sym] / 100.0, 1.0)
            cond = next(t[2] for t in trending[: self.top_n] if t[0] == sym)
            signals.append(Signal(
                symbol=sym,
                side=OrderSide.BUY,
                strength=round(strength, 4),
                reason=f"Trend({cond}/5) score={self._last_scores[sym]:.1f}",
            ))

        return signals

    def size_positions(
        self,
        signals: list[Signal],
        capital: float,
        portfolio: Portfolio,
        market_data: MarketData | None = None,
    ) -> list[Order]:
        """Equal-weight among selected stocks, sell rotated-out positions."""
        orders: list[Order] = []
        buy_signals = [s for s in signals if s.side == OrderSide.BUY]
        selected_syms = {s.symbol for s in buy_signals}

        # ── Sell rotated-out ──
        for sym, pos in portfolio.positions.items():
            if pos.quantity <= 0:
                continue
            sell = False
            reason = ""
            if sym not in selected_syms:
                sell = True
                reason = "Rotated out of top N"
            elif sym in self._hwm and pos.current_price > 0:
                loss = (pos.current_price / self._hwm[sym] - 1) * 100
                if loss < self.stop_loss_pct:
                    sell = True
                    reason = f"Stop-loss: {loss:.1f}% from HWM"

            if sell:
                orders.append(Order(
                    symbol=sym, side=OrderSide.SELL,
                    quantity=pos.quantity, price=None, order_type="market",
                    signal_ref=reason,
                ))

        # ── Buy new entries (equal weight) ──
        new_entries = [s for s in buy_signals if s.symbol not in portfolio.positions]
        if not new_entries:
            return orders

        n_positions = len(buy_signals)
        per_stock = capital / max(n_positions, 1) * 0.92  # 8% buffer for slippage

        for sig in new_entries:
            price = self._get_price(sig.symbol, portfolio, market_data)
            if price and price > 0.01:
                qty = int(per_stock / price / 100) * 100
                if qty >= 100:
                    orders.append(Order(
                        symbol=sig.symbol, side=OrderSide.BUY,
                        quantity=qty, price=price, order_type="limit",
                        signal_ref=sig.reason,
                    ))

        return orders

    def risk_check(
        self,
        orders: list[Order],
        portfolio: Portfolio,
        market_data: MarketData | None = None,
        risk_checker: object = None,
    ) -> list[Order]:
        """Add stop-loss exits and drawdown breaker liquidation."""
        approved: list[Order] = []

        # Update portfolio HWM tracking
        total_value = portfolio.cash + sum(
            p.quantity * p.current_price for p in portfolio.positions.values()
        )
        if total_value > self._max_equity:
            self._max_equity = total_value
            self._dd_exit_active = False

        # Check drawdown breaker
        if self._max_equity > 0 and self.dd_exit_pct > 0:
            dd = (self._max_equity - total_value) / self._max_equity * 100
            if dd > self.dd_exit_pct:
                self._dd_exit_active = True
                self._dd_exit_days = 0
            elif self._dd_exit_active:
                self._dd_exit_days += 1
                # Recovery: either equity recovers past 92% of peak
                # OR 60+ trading days (~3 calendar months) in cash with no re-trigger
                if (total_value / self._max_equity > 0.92) or (self._dd_exit_days > 60 and dd <= self.dd_exit_pct):
                    self._dd_exit_active = False
                    self._max_equity = total_value  # reset peak to current level
                    self._dd_exit_days = 0

        # If drawdown breaker active, liquidate everything
        if self._dd_exit_active:
            for sym, pos in portfolio.positions.items():
                if pos.quantity > 0:
                    approved.append(Order(
                        symbol=sym, side=OrderSide.SELL,
                        quantity=pos.quantity, price=None, order_type="market",
                        signal_ref=f"DD breaker: -{dd:.1f}% from peak",
                    ))
            return approved

        # ── Per-stock stop-loss ──
        for sym, pos in portfolio.positions.items():
            if sym not in self._hwm or pos.current_price > self._hwm[sym]:
                self._hwm[sym] = pos.current_price

            if pos.current_price > 0 and self._hwm[sym] > 0:
                loss_pct = (pos.current_price / self._hwm[sym] - 1) * 100
                if loss_pct < self.stop_loss_pct:
                    approved.append(Order(
                        symbol=sym, side=OrderSide.SELL,
                        quantity=pos.quantity, price=None, order_type="market",
                        signal_ref=f"HWM stop: {loss_pct:.1f}% from peak",
                    ))

        # ── Pass through valid orders ──
        for o in orders:
            cost = o.quantity * (o.price or 0)
            trading_cap = getattr(settings, "TRADING_CAPITAL", 90000)
            if cost > trading_cap * 0.25:
                continue
            approved.append(o)

        return approved

    # ── Trend detection ─────────────────════════

    @staticmethod
    def _sma(arr: np.ndarray, period: int) -> np.ndarray | None:
        """SMA via shared indicators. Returns None if insufficient data."""
        if len(arr) < period:
            return None
        return sma(arr, period)

    @staticmethod
    def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int = 14) -> np.ndarray | None:
        """ADX via shared indicators. Returns array or None."""
        return adx(high, low, close, period)

    def _is_stock_trending(self, bars: list[Bar]) -> tuple[bool, int]:
        """Check 5 trend conditions. Returns (is_trending, conditions_met)."""
        if len(bars) < 200:
            return False, 0

        closes = np.array([b.close for b in bars], dtype=np.float64)
        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)
        vols = np.array([b.volume for b in bars], dtype=np.float64)

        count = 0

        # 1. Price above 120-day MA
        ma120 = self._sma(closes, self.ma_filter_period)
        if ma120 is not None and not np.isnan(ma120[-1]) and closes[-1] > ma120[-1]:
            count += 1

        # 2. Higher highs and higher lows
        recent_high = np.max(highs[-20:])
        prev_high = np.max(highs[-60:-20])
        recent_low = np.min(lows[-20:])
        prev_low = np.min(lows[-60:-20])
        if recent_high > prev_high and recent_low > prev_low:
            count += 1

        # 3. MA alignment: SMA20 > SMA60 > SMA120
        ma20 = self._sma(closes, 20)
        ma60 = self._sma(closes, 60)
        if (ma20 is not None and ma60 is not None and ma120 is not None and
            not np.isnan(ma20[-1]) and not np.isnan(ma60[-1]) and
            not np.isnan(ma120[-1]) and ma20[-1] > ma60[-1] > ma120[-1]):
            count += 1

        # 4. ADX > 25
        adx_arr = self._adx(highs, lows, closes, 14)
        if adx_arr is not None and not np.isnan(adx_arr[-1]) and adx_arr[-1] > 25:
            count += 1

        # 5. Volume expansion (institutional interest)
        vol_20 = np.mean(vols[-21:-1])
        if vol_20 > 0 and vols[-1] > vol_20 * 1.2:
            count += 1

        return (count >= self.min_trend_score and count >= 1), count

    def _is_market_trending(self, market_data: MarketData) -> bool:
        """Check if CSI300 proxy is trending. Uses first wide-market ETF found."""
        # Try CSI300 ETF as proxy
        for sym in ("510300", "159919"):
            if sym in market_data.bars:
                bars = market_data.bars[sym]
                if len(bars) >= 200:
                    is_t, _ = self._is_stock_trending(bars)
                    return is_t
        # Default: permissive (not enough index data)
        return True

    def _trend_strength_score(self, bars: list[Bar]) -> float:
        """For stocks that ARE trending, compute momentum + low-vol score (0-100)."""
        closes = np.array([b.close for b in bars], dtype=np.float64)

        # Momentum component (60%)
        if closes[-63] < 1e-6 or closes[-126] < 1e-6:
            mom_score = 0.0  # insufficient data: no momentum signal
        else:
            ret_3m = closes[-1] / closes[-63] - 1
            ret_6m = closes[-1] / closes[-126] - 1
            mom_3m = min(max(ret_3m / 0.5, 0), 1) if ret_3m > 0 else 0
            mom_6m = min(max(ret_6m / 0.8, 0), 1) if ret_6m > 0 else 0
            mom_score = (0.5 * mom_3m + 0.5 * mom_6m) * 100

        # Low volatility bonus (40%)
        if len(closes) >= 61:
            window_c = closes[-61:]
            daily_ret = np.diff(window_c) / (window_c[:-1] + 1e-10)
            vol = np.nanstd(daily_ret)
            vol_score = max(0, (1 - min(vol / 0.04, 1))) * 100
        else:
            vol_score = 50.0

        return 0.6 * mom_score + 0.4 * vol_score

    # ── Utilities ────────────────────────────────────

    @staticmethod
    def _get_price(
        symbol: str,
        portfolio: Portfolio,
        market_data: MarketData | None,
    ) -> float | None:
        pos = portfolio.positions.get(symbol)
        if pos and pos.current_price > 0:
            return pos.current_price
        if market_data and symbol in market_data.bars:
            bars = market_data.bars[symbol]
            if bars:
                return bars[-1].close
        return None
