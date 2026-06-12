"""
Convertible Bond Dual-Low Rotation Strategy (Phase 6).
Dual-low = low price + low premium rate.
Ranks bonds by composite score, buys top N, rebalances periodically.
"""
import numpy as np

from quanti.config import settings
from quanti.execution.risk import RiskChecker
from quanti.strategy.base import BaseStrategy
from quanti.types import MarketData, Order, OrderSide, Portfolio, Signal


class CBDualLowStrategy(BaseStrategy):
    """Ranks CBs by dual-low score. Buys top N, rotates on rebalance."""

    name = "cb_dual_low"

    def __init__(self, max_positions=None):
        self.max_positions = max_positions or settings.MAX_POSITIONS
        self._held_symbols = set()

    def generate_signals(self, market_data):
        signals = []
        symbol_data = {}

        for symbol, bars in market_data.bars.items():
            if len(bars) == 0:
                continue
            latest = bars[-1]
            price = latest.close
            premium = getattr(latest, "premium_rt", 99) or 99
            if price <= 0 or premium <= -50:
                continue
            symbol_data[symbol] = (price, premium)

        if not symbol_data:
            return signals

        prices = np.array([v[0] for v in symbol_data.values()])
        premiums = np.array([v[1] for v in symbol_data.values()])

        p_low, p_high = np.percentile(prices, 30), np.percentile(prices, 100)
        prem_low, prem_high = np.percentile(premiums, 10), np.percentile(premiums, 100)

        scores = {}
        for sym, (p, pr) in symbol_data.items():
            p_score = 1 - (p - p_low) / (p_high - p_low + 1e-10) if p_high > p_low else 0.5
            pr_score = 1 - (pr - prem_low) / (prem_high - prem_low + 1e-10) if prem_high > prem_low else 0.5
            scores[sym] = 0.6 * p_score + 0.4 * pr_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_n = ranked[:self.max_positions]
        top_symbols = {s for s, _ in top_n}

        for sym, score in top_n:
            if sym not in self._held_symbols:
                signals.append(Signal(symbol=sym, side=OrderSide.BUY, strength=score,
                    reason=f"Dual-low score={score:.3f}"))

        for sym in self._held_symbols - top_symbols:
            signals.append(Signal(symbol=sym, side=OrderSide.SELL, strength=1.0,
                reason=f"Rotated out of top {self.max_positions}"))

        self._held_symbols = top_symbols
        return signals

    def size_positions(self, signals, capital, portfolio, market_data=None):
        buy_signals = [s for s in signals if s.side == OrderSide.BUY]
        sell_signals = [s for s in signals if s.side == OrderSide.SELL]
        orders = []

        for sig in sell_signals:
            pos = portfolio.positions.get(sig.symbol)
            if pos and pos.quantity > 0:
                orders.append(Order(symbol=sig.symbol, side=OrderSide.SELL,
                    quantity=pos.quantity, price=None, order_type="market",
                    signal_ref=sig.reason))

        if buy_signals and capital > 0:
            new_symbols = [s for s in buy_signals if s.symbol not in portfolio.positions]
            n = max(len(new_symbols), 1)
            per_pos = capital / n
            for sig in new_symbols:
                price = None
                if market_data:
                    bars = market_data.bars.get(sig.symbol, [])
                    if bars:
                        price = bars[-1].close
                if price and price > 0:
                    qty = int(per_pos / price / 10) * 10
                    if qty > 0:
                        orders.append(Order(symbol=sig.symbol, side=OrderSide.BUY,
                            quantity=qty, price=price, order_type="limit",
                            signal_ref=sig.reason))
        return orders

    def risk_check(
        self,
        orders: list[Order],
        portfolio: Portfolio,
        market_data: "MarketData | None" = None,
        risk_checker: "RiskChecker | None" = None,
    ) -> list[Order]:
        """
        Unified risk & exit check (delegates to RiskChecker).

        Because the base strategy interface requires risk_check, and
        the project's architectural decision (AD-2) requires all strategies
        to run through identical risk paths, this method delegates to
        RiskChecker.check_all() for capital sufficiency, position limits,
        and duplicate detection.

        CB-specific filter: single position <= 15% of TRADING_CAPITAL
        (convertible bonds require broader diversification than ETFs).
        """
        # Delegate to RiskChecker for unified pre-trade checks
        if risk_checker is not None:
            orders, _ = risk_checker.check_all(orders, portfolio, market_data)
        else:
            temp_checker = RiskChecker()
            orders, _ = temp_checker.check_all(orders, portfolio, market_data)

        # CB-specific filters: diversification limit
        approved: list[Order] = []
        seen: set[str] = set()
        for o in orders:
            cost = o.quantity * (o.price or 0)
            if cost > settings.TRADING_CAPITAL * 0.15:
                continue
            key = f"{o.symbol}:{o.side.value}"
            if key in seen:
                continue
            seen.add(key)
            approved.append(o)
        return approved
