"""Tests for strategy signal generation (Phase 3)."""

from quanti.strategy.base import BaseStrategy
from quanti.types import MarketData, Order, OrderSide, Portfolio, Signal


class DummyStrategy(BaseStrategy):
    """Minimal strategy for testing the interface contract."""

    name = "dummy"

    def generate_signals(self, market_data: MarketData) -> list[Signal]:
        if not market_data.bars:
            return []
        return [
            Signal(
                symbol="510300.SH",
                side=OrderSide.BUY,
                strength=0.8,
                reason="test_signal",
            )
        ]

    def size_positions(
        self, signals: list[Signal], capital: float, portfolio: Portfolio
    ) -> list[Order]:
        orders = []
        for sig in signals:
            qty = int(capital * 0.1 / 3.50)  # 10% of capital at price 3.50
            if qty > 0:
                orders.append(Order(
                    symbol=sig.symbol,
                    side=sig.side,
                    quantity=qty,
                    price=3.50,
                    order_type="limit",
                    signal_ref=sig.reason,
                ))
        return orders

    def risk_check(self, orders: list[Order], portfolio: Portfolio) -> list[Order]:
        """Reject orders that exceed 20% of capital."""
        approved = []
        for o in orders:
            cost = o.quantity * (o.price or 0)
            if cost <= portfolio.total_capital * 0.2:
                approved.append(o)
        return approved


class TestStrategyInterface:

    def test_generate_signals_empty_market(self):
        strat = DummyStrategy()
        md = MarketData(bars={}, index_bars={}, timestamp=None)
        signals = strat.generate_signals(md)
        assert signals == []

    def test_size_positions_respects_capital(self):
        from datetime import datetime
        strat = DummyStrategy()
        md = MarketData(bars={"510300.SH": []}, index_bars={}, timestamp=datetime.now())
        signals = strat.generate_signals(md)
        portfolio = Portfolio(
            positions={}, cash=90000.0, total_capital=100000.0,
            timestamp=datetime.now(),
        )
        orders = strat.size_positions(signals, 90000.0, portfolio)
        assert len(orders) > 0
        for o in orders:
            cost = o.quantity * (o.price or 0)
            assert cost <= 90000.0  # Within trading capital
            assert cost <= 20000.0  # Within 20% single-position limit (risk check)

    def test_risk_check_filters_large_orders(self):
        from datetime import datetime
        strat = DummyStrategy()
        # Create orders exceeding 20% limit
        orders = [
            Order("510300.SH", OrderSide.BUY, 1000, 3.50, "limit", "test"),   # 3,500 = OK
            Order("510300.SH", OrderSide.BUY, 100000, 3.50, "limit", "test"),  # 350,000 = TOO BIG
        ]
        portfolio = Portfolio(
            positions={}, cash=90000.0, total_capital=100000.0,
            timestamp=datetime.now(),
        )
        approved = strat.risk_check(orders, portfolio)
        assert len(approved) == 1
        assert approved[0].quantity == 1000


class TestTypesSerialization:

    def test_signal_is_hashable(self):
        """Signal must be usable in dicts/sets for deduplication."""
        from dataclasses import asdict
        sig = Signal("510300.SH", OrderSide.BUY, 0.5, "test")
        d = asdict(sig)
        assert d["symbol"] == "510300.SH"
        assert d["strength"] == 0.5

    def test_order_optional_price(self):
        """Order with None price = market order."""
        order = Order("510300.SH", OrderSide.BUY, 1000, None, "market", "test")
        assert order.price is None
        assert order.order_type == "market"
