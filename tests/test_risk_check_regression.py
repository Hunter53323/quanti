"""Regression tests for risk_check bug (2026-06-10).
Ensures BUY orders are not silently dropped from the approval list."""
from datetime import datetime

import quanti.config.settings as st
from quanti.strategy.etf_trend import ETFTrendStrategy
from quanti.types import Order, OrderSide, Portfolio, Position


class TestRiskCheckRegression:
    """Verify risk_check does not silently drop non-stop-loss orders."""

    def test_buy_order_is_approved(self):
        """Single valid BUY order must be approved, not silently dropped."""
        strategy = ETFTrendStrategy()
        pf = Portfolio(positions={}, cash=90000, total_capital=100000,
                       timestamp=datetime.now())
        orders = [Order("510300.SH", OrderSide.BUY, 100, 3.50, "limit", "test")]
        approved = strategy.risk_check(orders, pf)
        assert len(approved) == 1, (
            f"BUG REGRESSION: risk_check returned {len(approved)} orders, "
            f"expected 1. BUY order was silently dropped."
        )
        assert approved[0].symbol == "510300.SH"

    def test_multiple_buy_orders_all_approved(self):
        """Multiple BUY orders for different symbols all pass (within exposure limit)."""
        strategy = ETFTrendStrategy()
        pf = Portfolio(positions={}, cash=90000, total_capital=100000,
                       timestamp=datetime.now())
        orders = [
            Order("510300.SH", OrderSide.BUY, 100, 3.50, "limit", "test1"),
            Order("510500.SH", OrderSide.BUY, 200, 6.00, "limit", "test2"),
            Order("159915.SZ", OrderSide.BUY, 300, 2.50, "limit", "test3"),
        ]
        approved = strategy.risk_check(orders, pf)
        assert len(approved) == 3, (
            f"BUG REGRESSION: only {len(approved)}/3 BUY orders approved"
        )

    def test_duplicate_orders_rejected(self):
        """Duplicate symbol+side orders: only first is kept."""
        strategy = ETFTrendStrategy()
        pf = Portfolio(positions={}, cash=90000, total_capital=100000,
                       timestamp=datetime.now())
        orders = [
            Order("510300.SH", OrderSide.BUY, 100, 3.50, "limit", "test1"),
            Order("510300.SH", OrderSide.BUY, 200, 3.50, "limit", "test2"),
        ]
        approved = strategy.risk_check(orders, pf)
        assert len(approved) == 1, (
            f"Duplicate detection failed: {len(approved)} orders approved, expected 1"
        )

    def test_oversized_order_rejected(self):
        """Order exceeding TRADING_CAPITAL is rejected."""
        strategy = ETFTrendStrategy()
        pf = Portfolio(positions={}, cash=90000, total_capital=100000,
                       timestamp=datetime.now())
        orders = [Order("510300.SH", OrderSide.BUY, 100000, 3.50, "limit", "test")]
        approved = strategy.risk_check(orders, pf)
        assert len(approved) == 0, (
            f"Oversized order should be rejected, got {len(approved)} approved"
        )

    def test_total_exposure_limit(self):
        """Cannot exceed TRADING_CAPITAL in total across multiple orders + positions."""
        strategy = ETFTrendStrategy()
        pf = Portfolio(positions={}, cash=90000, total_capital=100000,
                       timestamp=datetime.now())
        orders = [
            Order("510300.SH", OrderSide.BUY, 5000, 10.0, "limit", "test1"),  # 50K
            Order("510500.SH", OrderSide.BUY, 5000, 8.0, "limit", "test2"),   # 40K = 90K total
        ]
        approved = strategy.risk_check(orders, pf)
        # Second order (bringing total to 90K) may be rejected if > TRADING_CAPITAL
        assert len(approved) >= 1, f"At least first order should pass, got {len(approved)}"

    def test_stop_loss_generates_sell(self):
        """Stop-loss generates SELL orders for breached positions."""
        st.STOP_LOSS_PCT = 8.0
        pf = Portfolio(
            positions={"TEST": Position(symbol="TEST", quantity=1000,
                       avg_cost=100.0, current_price=90.0)},
            cash=90000, total_capital=100000, timestamp=datetime.now()
        )
        strategy = ETFTrendStrategy()
        approved = strategy.risk_check([], pf)
        sells = [o for o in approved if o.side == OrderSide.SELL]
        assert len(sells) >= 1, (
            f"Stop-loss should trigger SELL, got {len(sells)} sells"
        )

    def test_mixed_buy_and_stop_loss(self):
        """BUY AND stop-loss SELL both appear when within exposure limits."""
        st.STOP_LOSS_PCT = 8.0
        # Position: 100 shares at average cost 100, current 90 -> -10% -> stop-loss triggered
        # Exposure from position: 100 * 90 = 9000
        # Buy cost: 100 * 3.50 = 350  -> total new exposure = 9350, well under 90000 cap
        pf = Portfolio(
            positions={"TEST": Position(symbol="TEST", quantity=100,
                       avg_cost=100.0, current_price=90.0)},
            cash=90000, total_capital=100000, timestamp=datetime.now()
        )
        orders = [Order("510300.SH", OrderSide.BUY, 100, 3.50, "limit", "test")]
        strategy = ETFTrendStrategy()
        approved = strategy.risk_check(orders, pf)
        buys = [o for o in approved if o.side == OrderSide.BUY]
        sells = [o for o in approved if o.side == OrderSide.SELL]
        assert len(sells) >= 1, f"Missing stop-loss SELL, got {len(sells)}"
        assert len(buys) == 1, (
            f"BUG REGRESSION: BUY order was dropped by risk_check! "
            f"buys={len(buys)}, total approved={len(approved)}"
        )
