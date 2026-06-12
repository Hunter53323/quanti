"""Tests for PE-band allocation and dividend barbell strategies."""
from datetime import datetime, timedelta

from quanti.types import Bar, MarketData, Order, OrderSide, Portfolio, Position


def make_bars(symbol, prices):
    bars = []
    for i, p in enumerate(prices):
        bars.append(Bar(
            symbol=symbol,
            datetime=datetime(2024, 1, 1) + timedelta(days=i),
            open=p, high=p*1.01, low=p*0.99, close=p,
            volume=10000.0
        ))
    return bars


class TestPEBandAllocation:
    def test_allocation_formula(self):
        """PE percentile 20% -> equity around 68% (max 80%, min 20%)."""
        from quanti.strategy.pe_band import PEBandAllocation
        strat = PEBandAllocation(equity_max=0.80, equity_min=0.20, gold_fixed_pct=0.10)
        md = MarketData(
            bars={}, index_bars={},
            timestamp=datetime.now(),
            index_fundamentals={"pe_percentile": 20.0}
        )
        targets = strat.get_allocation_targets(md)
        assert targets is not None
        assert "510300" in targets
        # equity = 0.80 - (20/100) * (0.80 - 0.20) = 0.80 - 0.12 = 0.68
        assert abs(targets["510300"] - 0.68) < 0.01

    def test_expensive_allocation(self):
        """PE percentile 90% -> equity around 26%."""
        from quanti.strategy.pe_band import PEBandAllocation
        strat = PEBandAllocation(equity_max=0.80, equity_min=0.20, gold_fixed_pct=0.10)
        md = MarketData(
            bars={}, index_bars={},
            timestamp=datetime.now(),
            index_fundamentals={"pe_percentile": 90.0}
        )
        targets = strat.get_allocation_targets(md)
        # equity = 0.80 - (90/100) * 0.60 = 0.80 - 0.54 = 0.26
        assert abs(targets["510300"] - 0.26) < 0.01

    def test_no_pe_data_no_targets(self):
        from quanti.strategy.pe_band import PEBandAllocation
        strat = PEBandAllocation()
        md = MarketData(bars={}, index_bars={}, timestamp=datetime.now())
        targets = strat.get_allocation_targets(md)
        assert targets is None

    def test_rebalance_signals(self):
        """After rebalance_freq days, generate_signals produces BUY signals."""
        from quanti.strategy.pe_band import PEBandAllocation
        strat = PEBandAllocation(rebalance_freq=3, gold_fixed_pct=0.10)
        strat._days_since_rebalance = 3  # trigger rebalance
        md = MarketData(
            bars={}, index_bars={},
            timestamp=datetime.now(),
            index_fundamentals={"pe_percentile": 50.0}
        )
        signals = strat.generate_signals(md)
        assert len(signals) >= 1
        for s in signals:
            assert s.side == OrderSide.BUY
            assert s.strength > 0

    def test_signals_on_non_rebalance_day(self):
        from quanti.strategy.pe_band import PEBandAllocation
        strat = PEBandAllocation(rebalance_freq=63)
        strat._days_since_rebalance = 1  # not yet rebalance day
        md = MarketData(
            bars={}, index_bars={},
            timestamp=datetime.now(),
            index_fundamentals={"pe_percentile": 50.0}
        )
        signals = strat.generate_signals(md)
        assert len(signals) == 0  # No signals on non-rebalance days


class TestDividendBarbell:
    def test_allocation_targets(self):
        from quanti.strategy.dividend_barbell import DividendBarbell
        strat = DividendBarbell(dividend_pct=0.40, bond_pct=0.40, gold_pct=0.10, cash_pct=0.10)
        targets = strat.get_allocation_targets()
        assert "510880" in targets
        assert abs(targets["510880"] - 0.40) < 0.01
        assert abs(targets["518880"] - 0.10) < 0.01

    def test_rebalance_signals(self):
        from quanti.strategy.dividend_barbell import DividendBarbell
        strat = DividendBarbell(rebalance_freq=3)
        strat._days_since_rebalance = 3
        md = MarketData(bars={}, index_bars={}, timestamp=datetime.now())
        signals = strat.generate_signals(md)
        assert len(signals) >= 1
        for s in signals:
            assert s.side == OrderSide.BUY

    def test_non_rebalance_day_no_signals(self):
        from quanti.strategy.dividend_barbell import DividendBarbell
        strat = DividendBarbell(rebalance_freq=63)
        strat._days_since_rebalance = 1
        md = MarketData(bars={}, index_bars={}, timestamp=datetime.now())
        signals = strat.generate_signals(md)
        assert len(signals) == 0

    def test_concentration_limit_in_risk_check(self):
        from quanti.strategy.dividend_barbell import DividendBarbell
        strat = DividendBarbell()
        order = Order(symbol="510880", side=OrderSide.BUY, quantity=1000, price=100.0,
                      order_type="limit", signal_ref="test")
        pos = Position(symbol="510880", quantity=2000, avg_cost=95, current_price=100)
        pf = Portfolio(
            positions={"510880": pos}, cash=100000, total_capital=300000,
            timestamp=datetime.now()
        )
        approved = strat.risk_check([order], pf)
        # Order would bring 510880 share to (1000+2000)*100/300000 = 100% > 70% cap -> rejected
        assert len(approved) == 0


class TestStockMomentum:
    def test_stock_not_trending_flat(self):
        """Flat prices should not be trending."""
        from quanti.strategy.stock_momentum import StockMomentumStrategy
        strat = StockMomentumStrategy()
        prices = [10.0] * 250
        bars = [Bar(symbol="test", datetime=datetime(2024, 1, 1) + timedelta(days=i),
                     open=p, high=p+0.1, low=p-0.1, close=p, volume=5000.0)
                for i, p in enumerate(prices)]
        is_trending, count = strat._is_stock_trending(bars)
        # Flat prices: no MA alignment, no HH/HL, no ADX > 25, vol not surging
        assert count < 3

    def test_empty_signals_when_no_trend(self):
        from quanti.strategy.stock_momentum import StockMomentumStrategy
        strat = StockMomentumStrategy(market_trend_required=True)
        md = MarketData(bars={}, index_bars={}, timestamp=datetime.now())
        signals = strat.generate_signals(md)
        assert len(signals) == 0

    def test_drawdown_breaker_liquidates(self):
        from quanti.strategy.stock_momentum import StockMomentumStrategy
        strat = StockMomentumStrategy(dd_exit_pct=20.0)
        pos = Position(symbol="test", quantity=100, avg_cost=10, current_price=7)
        pf = Portfolio(
            positions={"test": pos}, cash=1000, total_capital=1700,
            timestamp=datetime.now()
        )
        strat._max_equity = 3000.0  # peak at 3000
        # dd = (3000-1700)/3000 = 43.3% > 20% -> trigger
        orders = strat.risk_check([], pf)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert "DD breaker" in orders[0].signal_ref
