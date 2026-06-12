"""
Dividend Barbell Strategy.

Core-satellite allocation centered on 510880 (中证红利 ETF):
- 40%: 510880 (high-dividend equity) -- core income engine
- 40%: Bond ETF (国债ETF / 货币ETF) -- stability ballast
- 10%: Gold ETF (518880) -- diversification / crisis hedge
- 10%: Cash reserve

Optional dividend-yield tilt: when the dividend spread (dividend yield - 10Y bond yield)
widens, increase dividend ETF allocation; when it narrows, rotate toward bonds.

Rebalances quarterly to minimize churn and tax friction.

Inherits BaseStrategy so it works with the same backtest / live infrastructure.
"""


from quanti.config import settings
from quanti.strategy.base import BaseStrategy
from quanti.types import MarketData, Order, OrderSide, Portfolio, Position, Signal

# Sentinel for dict.get() defaults
_EMPTY_POS = Position(symbol="", quantity=0, avg_cost=0.0, current_price=0.0)


class DividendBarbell(BaseStrategy):
    """
    Dividend Barbell Allocation Strategy.

    Static allocation (default):
      40% 510880 (中证红利)
      40% Bond ETF
      10% 518880 (gold)
      10% Cash

    Dynamic tilt mode (when DYN_TILT=True):
      dividend_tilt = clamp((current_dividend_yield - risk_free_rate) * 5, -0.10, +0.10)
      adjusted_dividend_pct = base_dividend_pct + dividend_tilt
      adjusted_bond_pct = base_bond_pct - dividend_tilt

    Parameters:
    - dividend_etf: primary dividend ETF (default: 510880)
    - bond_etf: bond / money-market ETF (default: 511880)
    - gold_etf: gold ETF (default: 518880)
    - dividend_pct: base allocation to dividend ETF (default: 0.40)
    - bond_pct: base allocation to bond ETF (default: 0.40)
    - gold_pct: fixed gold allocation (default: 0.10)
    - cash_pct: cash reserve (default: 0.10)
    - dynamic_tilt: enable yield-based dynamic tilt (default: False)
    - rebalance_freq: trading days between rebalances (default: 63 = quarterly)
    """

    name = "dividend_barbell"

    def __init__(
        self,
        dividend_etf: str | None = None,
        bond_etf: str | None = None,
        gold_etf: str | None = None,
        dividend_pct: float | None = None,
        bond_pct: float | None = None,
        gold_pct: float | None = None,
        cash_pct: float | None = None,
        dynamic_tilt: bool = False,
        rebalance_freq: int = 63,  # quarterly
    ):
        self.dividend_etf = dividend_etf or "510880"
        self.bond_etf = bond_etf or getattr(settings, "PE_BAND_BOND_ETF", "511880")
        self.gold_etf = gold_etf or getattr(settings, "PE_BAND_GOLD_ETF", "518880")
        self.dividend_pct = dividend_pct if dividend_pct is not None else 0.40
        self.bond_pct = bond_pct if bond_pct is not None else 0.40
        self.gold_pct = gold_pct if gold_pct is not None else 0.10
        self.cash_pct = cash_pct if cash_pct is not None else 0.10
        self.dynamic_tilt = dynamic_tilt
        self.rebalance_freq = rebalance_freq

        # State tracking
        self._last_rebalance_date: str | None = None
        self._current_targets: dict[str, float] = {}
        self._days_since_rebalance = 0

    def get_allocation_targets(self, market_data: MarketData | None = None) -> dict[str, float]:
        """
        Compute target allocation fractions.

        Static targets by default. If dynamic_tilt is enabled and market_data
        contains dividend yield info in index_fundamentals, adjusts the
        dividend-bond split based on yield spread.

        Returns:
            {"510880": 0.40, "511880": 0.40, "518880": 0.10}
            plus implied cash of 0.10 (not an ETF, handled by capital constraint).
        """
        dividend_alloc = self.dividend_pct
        bond_alloc = self.bond_pct

        if self.dynamic_tilt and market_data and market_data.index_fundamentals:
            div_yield = market_data.index_fundamentals.get("dividend_yield")
            risk_free = market_data.index_fundamentals.get("risk_free_rate", 0.018)  # ~1.8% default

            if div_yield is not None and div_yield > 0:
                # Tilt: higher spread -> more dividend, less bond
                spread = float(div_yield) - float(risk_free)
                tilt = max(-0.10, min(0.10, spread * 5.0))  # 5x leverage on spread
                dividend_alloc = max(self.dividend_pct + tilt, 0.10)
                bond_alloc = self.bond_pct - (dividend_alloc - self.dividend_pct)
                bond_alloc = max(bond_alloc, 0.10)

        targets = {}
        assets = [
            (self.dividend_etf, dividend_alloc),
            (self.gold_etf, self.gold_pct),
            (self.bond_etf, bond_alloc),
        ]

        # Normalize to sum to 1.0 - cash_pct (remaining is intentional cash)
        total_non_cash = sum(pct for _, pct in assets)
        cash_reserve = self.cash_pct
        if total_non_cash + cash_reserve > 1.0:
            scale = (1.0 - cash_reserve) / total_non_cash
            for sym, pct in assets:
                targets[sym] = round(pct * scale, 4)
        else:
            for sym, pct in assets:
                targets[sym] = round(pct, 4)

        return targets

    def generate_signals(self, market_data: MarketData) -> list[Signal]:
        """
        Generate allocation signals on rebalance days.

        Only generates signals every rebalance_freq trading days.
        On other days, produces empty signal list (no action).
        """
        signals: list[Signal] = []
        self._days_since_rebalance += 1

        if self._days_since_rebalance < self.rebalance_freq:
            return signals

        self._days_since_rebalance = 0

        targets = self.get_allocation_targets(market_data)
        self._current_targets = targets

        for symbol, target_pct in targets.items():
            if target_pct > 0.01:
                signals.append(Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    strength=min(target_pct * 2.5, 1.0),
                    reason=(
                        f"DIV-BARBELL: rebalance "
                        f"{symbol.split('.')[0]}={target_pct*100:.0f}%"
                    ),
                ))

        return signals

    def size_positions(
        self, signals: list[Signal], capital: float, portfolio: Portfolio,
        market_data: MarketData | None = None,
    ) -> list[Order]:
        """
        Convert target allocation signals to sized orders.

        Only generates orders when signals are present (rebalance days).
        Targets are derived from signal strength (strength = target_pct * 2.5).
        Compares target to current holdings, issues BUY/SELL as needed.
        """
        if not signals:
            return []  # No rebalance signal -> no trade

        orders: list[Order] = []

        # Total portfolio value
        holdings_value = sum(
            p.quantity * p.current_price for p in portfolio.positions.values()
        )
        total_capital = capital + holdings_value

        if total_capital <= 0:
            return []

        # Extract targets from signals: strength encodes target% as strength / 2.5
        targets_from_signals = {}
        for sig in signals:
            pct = min(sig.strength / 2.5, 1.0)
            targets_from_signals[sig.symbol] = pct

        for symbol, target_pct in targets_from_signals.items():
            target_amount = total_capital * target_pct
            current_pos = portfolio.positions.get(symbol)
            current_amount = (
                current_pos.quantity * current_pos.current_price
                if current_pos and current_pos.quantity > 0
                else 0.0
            )

            diff = target_amount - current_amount
            if abs(diff) < total_capital * 0.005:  # Skip if < 0.5% change
                continue

            price = None
            if market_data and symbol in market_data.bars and market_data.bars[symbol]:
                price = market_data.bars[symbol][-1].close
            if price is None or price <= 0:
                continue

            quantity = int(abs(diff) / price / 100) * 100
            if quantity <= 0:
                continue

            if diff > 0:
                cost = quantity * price
                if cost <= capital:
                    orders.append(Order(
                        symbol=symbol, side=OrderSide.BUY,
                        quantity=quantity, price=price,
                        order_type="limit",
                        signal_ref=f"barbell buy: {symbol} target={target_pct*100:.0f}%",
                    ))
            else:
                if current_pos and quantity <= current_pos.quantity:
                    orders.append(Order(
                        symbol=symbol, side=OrderSide.SELL,
                        quantity=quantity, price=price,
                        order_type="limit",
                        signal_ref=f"barbell sell: {symbol} reduce={-diff/price:.0f}sh",
                    ))

        return orders

    def risk_check(
        self, orders: list[Order], portfolio: Portfolio,
        market_data: MarketData | None = None,
        risk_checker=None,
    ) -> list[Order]:
        """
        Risk checks:
        1. No single ETF over 70% of portfolio
        2. Dividend ETF never exceeds 65% (sector concentration limit)
        3. Capital sufficiency
        """
        total_capital = portfolio.cash + sum(
            p.quantity * p.current_price for p in portfolio.positions.values()
        )

        if total_capital <= 0:
            return orders

        approved: list[Order] = []

        for order in orders:
            cost = order.quantity * (order.price or 0)
            if cost > portfolio.cash and order.side == OrderSide.BUY:
                continue

            # Post-trade concentration check
            current_qty = (
                portfolio.positions.get(order.symbol, _EMPTY_POS).quantity
                if order.side == OrderSide.BUY else
                portfolio.positions.get(order.symbol, _EMPTY_POS).quantity - order.quantity
            )
            post_qty = current_qty + (order.quantity if order.side == OrderSide.BUY else -order.quantity)
            price = order.price or 0
            post_exposure = (post_qty * price) / total_capital if total_capital > 0 else 0

            if post_exposure > 0.70:
                continue  # Cap at 70%

            # Dividend ETF limit
            if order.symbol == self.dividend_etf and post_exposure > 0.65:
                continue

            approved.append(order)

        return approved

    def get_current_targets(self) -> dict[str, float]:
        """Return current allocation targets for reporting."""
        return dict(self._current_targets)

    def get_allocation_summary(self, market_data: MarketData | None = None) -> str:
        """Generate a human-readable summary of current allocation."""
        targets = self.get_allocation_targets(market_data)
        labels = {
            self.dividend_etf: "Dividend ETF",
            self.bond_etf: "Bond ETF",
            self.gold_etf: "Gold ETF",
        }
        parts = ["DIV-BARBELL"]
        for sym, pct in sorted(targets.items()):
            label = labels.get(sym, sym)
            parts.append(f"{label}: {pct*100:.0f}%")
        parts.append(f"Cash: {self.cash_pct*100:.0f}%")
        return " | ".join(parts)
