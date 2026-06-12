"""
PE-Band Dynamic Allocation Strategy.

Allocates between equity ETF and bond ETF based on CSI300 PE percentile:
    equity_pct = max_equity - (pe_percentile / 100) * (max_equity - min_equity)

When PE is low (cheap percentile < 25%) -> high equity (up to 80%)
When PE is high (expensive percentile > 75%) -> low equity (down to 20%)
Remaining allocated to bond ETF + fixed gold slice.

Rebalances on schedule (quarterly default) to minimize churn.

Inherits BaseStrategy so it works with the same backtest / live infrastructure.
"""


from quanti.config import settings
from quanti.strategy.base import BaseStrategy
from quanti.types import MarketData, Order, OrderSide, Portfolio, Position, Signal

# Sentinel for dict.get() defaults
_EMPTY_POS = Position(symbol="", quantity=0, avg_cost=0.0, current_price=0.0)

# Allowed rebalance frequency values
REBALANCE_FREQ_MAP = {
    "monthly": 21,     # ~21 trading days per month
    "quarterly": 63,   # ~63 trading days per quarter
    "weekly": 5,       # ~5 trading days per week
}


class PEBandAllocation(BaseStrategy):
    """
    PE-Band Dynamic Allocation Strategy.

    Uses CSI300 PE percentile to determine equity/bond/cash mix:
    - PE percentile 0%   -> 80% equity / 10% gold / 10% bond
    - PE percentile 50%  -> 50% equity / 10% gold / 40% bond
    - PE percentile 100% -> 20% equity / 10% gold / 70% bond

    Strategy output is an allocation target ratio. Rebalance schedule
    converts target to market orders on rebalance days only.

    Parameters:
    - source_index: index code for PE data (default: 000300.SH)
    - equity_etf: equity ETF symbol (default: 510300)
    - bond_etf: bond/money-market ETF (default: 511880)
    - gold_etf: gold ETF symbol (default: 518880)
    - gold_fixed_pct: fixed gold allocation (default: 0.10)
    - equity_max: max equity allocation (default: 0.80)
    - equity_min: min equity allocation (default: 0.20)
    - rebalance_freq: trading days between rebalances (default: 63 = quarterly)
    """

    name = "pe_band_allocation"

    def __init__(
        self,
        source_index: str | None = None,
        equity_etf: str | None = None,
        bond_etf: str | None = None,
        gold_etf: str | None = None,
        gold_fixed_pct: float | None = None,
        equity_max: float | None = None,
        equity_min: float | None = None,
        rebalance_freq: int = 63,  # quarterly by default
    ):
        self.source_index = source_index or getattr(settings, "PE_BAND_SOURCE_INDEX", "000300.SH")
        self.equity_etf = equity_etf or getattr(settings, "PE_BAND_EQUITY_ETF", "510300")
        self.bond_etf = bond_etf or getattr(settings, "PE_BAND_BOND_ETF", "511880")
        self.gold_etf = gold_etf or getattr(settings, "PE_BAND_GOLD_ETF", "518880")
        self.gold_fixed_pct = gold_fixed_pct or getattr(settings, "PE_BAND_GOLD_FIXED_PCT", 0.10)
        self.equity_max = equity_max or getattr(settings, "PE_BAND_EQUITY_MAX", 0.80)
        self.equity_min = equity_min or getattr(settings, "PE_BAND_EQUITY_MIN", 0.20)
        self.rebalance_freq = rebalance_freq

        # State tracking
        self._last_rebalance_date: str | None = None
        self._current_targets: dict[str, float] = {}  # symbol -> target fraction
        self._days_since_rebalance = 0

    def get_allocation_targets(self, market_data: MarketData) -> dict[str, float] | None:
        """
        Compute target allocation fractions from PE percentile data.

        Returns:
            {"510300": 0.55, "518880": 0.10, "511880": 0.35}
            or None if PE data unavailable.

        1. Read PE/PB percentiles from market_data.index_fundamentals
        2. Compute equity allocation % from PE percentile
        3. Fixed gold slice
        4. Remainder to bond ETF
        """
        if not market_data.index_fundamentals:
            return None

        pe_pctile = market_data.index_fundamentals.get("pe_percentile")
        if pe_pctile is None:
            return None

        # Clamp percentile to [5, 95]
        pe_pctile = max(5.0, min(95.0, float(pe_pctile)))

        # Equity allocation: linear interpolation between min and max
        equity_pct = self.equity_max - (pe_pctile / 100.0) * (self.equity_max - self.equity_min)
        equity_pct = max(self.equity_min, min(self.equity_max, equity_pct))

        # Remaining after equity + gold goes to bonds
        remaining_pct = 1.0 - equity_pct - self.gold_fixed_pct
        bond_pct = max(0.0, remaining_pct)

        return {
            self.equity_etf: equity_pct,
            self.gold_etf: self.gold_fixed_pct,
            self.bond_etf: bond_pct,
        }

    def generate_signals(self, market_data: MarketData) -> list[Signal]:
        """
        Generate allocation signals.

        On rebalance days: produce target ratio signals for each ETF.
        On non-rebalance days: produce only SELL signals if any position
        has a closing signal (e.g. gap risk, extreme PE move).

        This allows the engine to call generate_signals daily and only
        act on rebalance days.
        """
        signals: list[Signal] = []

        # Daily counter for rebalance timing
        self._days_since_rebalance += 1
        should_rebalance = self._days_since_rebalance >= self.rebalance_freq

        targets = self.get_allocation_targets(market_data)
        if targets is None:
            return signals  # No PE data = no signals

        # Track targets for position sizing
        self._current_targets = targets

        # Only generate signals on rebalance days
        if should_rebalance:
            self._days_since_rebalance = 0

            for symbol, target_pct in targets.items():
                if target_pct > 0.01:
                    signals.append(Signal(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        strength=min(target_pct * 2, 1.0),  # Scale to 0-1
                        reason=(
                            f"PE-BAND: pe_pctile={market_data.index_fundamentals.get('pe_percentile', '?')}% "
                            f"target={target_pct*100:.0f}%"
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
        Targets are derived from signal strength (strength = target_pct * 2).
        Compares target to current holdings, issues BUY/SELL as needed.

        Minimum lot size = 100 shares for A-share ETF.
        """
        if not signals:
            return []  # No rebalance signal -> no trade

        orders: list[Order] = []
        total_capital = capital + sum(
            p.quantity * p.current_price for p in portfolio.positions.values()
        )

        if total_capital <= 0:
            return []

        # Extract targets from signals: strength encodes target% as strength / 2
        targets_from_signals = {}
        for sig in signals:
            pct = min(sig.strength / 2.0, 1.0)  # Reverse the encoding from generate_signals
            targets_from_signals[sig.symbol] = pct

        # For each target symbol, compute delta
        for symbol, target_pct in targets_from_signals.items():
            target_amount = total_capital * target_pct
            current_pos = portfolio.positions.get(symbol)
            current_amount = (current_pos.quantity * current_pos.current_price
                              if current_pos and current_pos.quantity > 0 else 0.0)

            diff = target_amount - current_amount
            if abs(diff) < total_capital * 0.005:  # Skip if < 0.5% change
                continue

            price = None
            if market_data and symbol in market_data.bars and market_data.bars[symbol]:
                price = market_data.bars[symbol][-1].close
            if price is None or price <= 0:
                continue

            quantity = int(abs(diff) / price / 100) * 100  # Round to lots
            if quantity <= 0:
                continue

            if diff > 0:
                cost = quantity * price
                if cost <= capital:
                    orders.append(Order(
                        symbol=symbol, side=OrderSide.BUY,
                        quantity=quantity, price=price,
                        order_type="limit",
                        signal_ref=f"PE-band buy: target={target_pct*100:.0f}%",
                    ))
            else:
                if current_pos and quantity <= current_pos.quantity:
                    orders.append(Order(
                        symbol=symbol, side=OrderSide.SELL,
                        quantity=quantity, price=price,
                        order_type="limit",
                        signal_ref=f"PE-band sell: target={target_pct*100:.0f}%",
                    ))

        return orders

    def risk_check(
        self, orders: list[Order], portfolio: Portfolio,
        market_data: MarketData | None = None,
        risk_checker=None,
    ) -> list[Order]:
        """
        Risk checks for allocation strategy:
        1. No single ETF over 85% of portfolio (concentration)
        2. No more than 10% deviation from target allocation
        3. Standard capital sufficiency
        """
        if not self._current_targets or not market_data:
            return orders

        total_capital = portfolio.cash + sum(
            p.quantity * p.current_price for p in portfolio.positions.values()
        )

        if total_capital <= 0:
            return orders

        approved: list[Order] = []

        for order in orders:
            # Check: capital sufficiency
            cost = order.quantity * (order.price or 0)
            if cost > portfolio.cash and order.side == OrderSide.BUY:
                continue

            # Check: concentration (post-trade)
            symbol_target = self._current_targets.get(order.symbol, 0)
            post_trade_exposure = (
                (portfolio.positions.get(order.symbol, _EMPTY_POS).quantity
                 + (order.quantity if order.side == OrderSide.BUY else -order.quantity))
                * (order.price or 0)
            ) / total_capital if total_capital > 0 else 0

            if post_trade_exposure > 0.85:
                continue  # Max 85% in any single asset

            # Check: deviation from target
            if (symbol_target > 0 and
                abs(post_trade_exposure - symbol_target) > 0.10):
                continue  # Max 10% deviation from target

            approved.append(order)

        return approved

    def register_rebalance(self, date_str: str) -> None:
        """Manually record a rebalance date (used by engine)."""
        self._last_rebalance_date = date_str
        self._days_since_rebalance = 0

    def get_current_targets(self) -> dict[str, float]:
        """Return current allocation targets for reporting."""
        return dict(self._current_targets)

    def get_allocation_summary(self, market_data: MarketData) -> str:
        """
        Generate a human-readable summary of current allocation.
        """
        targets = self.get_allocation_targets(market_data)
        if targets is None:
            return "PE-BAND: No PE data available"

        pe_info = market_data.index_fundamentals or {}
        parts = [
            "PE-BAND ALLOCATION",
            f"CSI300 PE: {pe_info.get('pe', '?'):.1f} ({pe_info.get('pe_percentile', '?'):.0f}th pctile)",
        ]
        for sym, pct in sorted(targets.items()):
            label = sym
            if "510300" in sym: label = "CSI300 ETF"
            elif "518880" in sym: label = "Gold ETF"
            elif "511880" in sym: label = "Bond ETF"
            parts.append(f"  {label}: {pct*100:.0f}%")
        return " | ".join(parts)
