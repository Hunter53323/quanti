"""Tests for RiskChecker in risk.py -- stop-loss, ATR trailing stop, RSI, helpers."""
from datetime import datetime

import numpy as np

from quanti.execution.risk import RiskChecker
from quanti.types import Bar, MarketData, Order, OrderSide, Portfolio, Position


def make_bars(closes, highs=None, lows=None, symbol="TEST"):
    """Create Bar objects from close/high/low arrays."""
    if highs is None:
        highs = [c + 0.15 for c in closes]
    if lows is None:
        lows = [c - 0.15 for c in closes]
    bars = []
    for i, (c, h, l) in enumerate(zip(closes, highs, lows, strict=False)):
        d = datetime(2024, 1, 1) + np.timedelta64(i, 'D')
        if isinstance(d, np.datetime64):
            d = d.astype(object)
        bars.append(Bar(symbol=symbol, datetime=d, open=c, high=h, low=l, close=c, volume=1_000_000))
    return bars


def make_pf(symbol, qty, avg_cost, current_price, cash=50000.0):
    """Create a simple portfolio with one position."""
    pos = Position(symbol=symbol, quantity=qty, avg_cost=avg_cost, current_price=current_price)
    mv = qty * current_price
    return Portfolio(
        positions={symbol: pos},
        cash=cash,
        total_capital=cash + mv,
        timestamp=datetime.now(),
    )


class TestStopLoss:
    """Flat percentage stop-loss check."""

    def test_stop_loss_triggers(self):
        rc = RiskChecker()
        pf = make_pf("TEST", 1000, 10.0, 9.0)  # 10% loss
        sells = rc._check_stop_loss(pf, stop_loss_pct=8.0)
        assert len(sells) == 1
        assert sells[0].side == OrderSide.SELL
        assert sells[0].symbol == "TEST"
        assert "stop-loss" in sells[0].signal_ref

    def test_stop_loss_not_triggered(self):
        rc = RiskChecker()
        pf = make_pf("TEST", 1000, 10.0, 9.5)  # 5% loss
        sells = rc._check_stop_loss(pf, stop_loss_pct=10.0)
        assert len(sells) == 0

    def test_zero_quantity_skipped(self):
        rc = RiskChecker()
        pf = make_pf("TEST", 0, 10.0, 5.0)
        sells = rc._check_stop_loss(pf, stop_loss_pct=5.0)
        assert len(sells) == 0

    def test_zero_price_skipped(self):
        rc = RiskChecker()
        pf = make_pf("TEST", 1000, 0.0, 5.0)
        sells = rc._check_stop_loss(pf, stop_loss_pct=5.0)
        assert len(sells) == 0

    def test_stop_loss_disabled(self):
        """When stop_loss_pct == 0, check_all() skips stop-loss entirely.
        Note: calling _check_stop_loss directly with 0.0 means 'trigger on any loss'
        which is correct -- the disable gate is in check_all()."""
        import quanti.config.settings as s
        s.STOP_LOSS_PCT = 0.0
        rc = RiskChecker()
        pf = make_pf("TEST", 1000, 10.0, 1.0)  # massive loss
        approved, rejected = rc.check_all([], pf)
        sells = [o for o in approved if o.side == OrderSide.SELL]
        assert len(sells) == 0, f"STOP_LOSS_PCT=0 should disable, got {len(sells)} sells"


class TestATRTrailingStop:
    """ATR trailing stop check."""

    def test_atr_stop_triggered(self):
        """Price drops below HWM - 2*ATR triggers SELL."""
        # Build bars: uptrend then sharp decline
        rng = np.random.RandomState(42)
        closes = np.concatenate([
            np.full(20, 10.0) + rng.randn(20) * 0.05,
            np.linspace(10.0, 15.0, 20),  # up
            np.linspace(15.0, 8.0, 10),   # sharp decline
        ])
        bars = make_bars(closes)
        rc = RiskChecker()
        rc._high_water_marks["TEST"] = 15.0  # set HWM high
        pf = make_pf("TEST", 1000, 10.0, closes[-1])
        md = MarketData(bars={"TEST": bars}, index_bars={}, timestamp=datetime.now())
        sells = rc._check_atr_trailing_stop(pf, md, atr_period=14, atr_mult=2.0)
        assert len(sells) == 1
        assert sells[0].side == OrderSide.SELL
        assert "ATR trailing stop" in sells[0].signal_ref

    def test_atr_stop_not_triggered_uptrend(self):
        """In uptrend, HWM rises with price, no stop triggered."""
        rng = np.random.RandomState(42)
        closes = np.concatenate([
            np.full(20, 10.0) + rng.randn(20) * 0.05,
            np.linspace(10.0, 15.0, 30),
        ])
        bars = make_bars(closes)
        rc = RiskChecker()
        rc._high_water_marks["TEST"] = closes[-1]  # HWM at current price
        pf = make_pf("TEST", 1000, 10.0, closes[-1])
        md = MarketData(bars={"TEST": bars}, index_bars={}, timestamp=datetime.now())
        sells = rc._check_atr_trailing_stop(pf, md, atr_period=14, atr_mult=2.0)
        assert len(sells) == 0

    def test_insufficient_bars(self):
        rc = RiskChecker()
        bars = make_bars([10.0] * 10)  # fewer than atr_period+1
        pf = make_pf("TEST", 1000, 10.0, 9.0)
        md = MarketData(bars={"TEST": bars}, index_bars={}, timestamp=datetime.now())
        sells = rc._check_atr_trailing_stop(pf, md, atr_period=14)
        assert len(sells) == 0

    def test_hwm_raises(self):
        """HWM should update when price makes new high."""
        np.random.RandomState(42)
        closes = np.linspace(10.0, 15.0, 30)
        bars = make_bars(closes)
        rc = RiskChecker()
        rc._high_water_marks["TEST"] = 10.0  # initial HWM
        pf = make_pf("TEST", 1000, 10.0, closes[-1])
        md = MarketData(bars={"TEST": bars}, index_bars={}, timestamp=datetime.now())
        rc._check_atr_trailing_stop(pf, md, atr_period=14, atr_mult=2.0)
        assert rc._high_water_marks["TEST"] > 10.0  # HWM should have risen

    def test_register_entry_exit(self):
        rc = RiskChecker()
        rc.register_entry("TEST", entry_price=10.0, entry_date=datetime.now(), entry_atr=0.15)
        assert rc.get_high_water_mark("TEST") == 10.0
        assert rc.get_entry_atr("TEST") == 0.15
        assert rc.get_entry_date("TEST") is not None

        rc.register_exit("TEST")
        assert rc.get_high_water_mark("TEST") is None
        assert rc.get_entry_atr("TEST") is None
        assert rc.get_entry_date("TEST") is None


class TestRSIComputation:
    """RSI computation helper."""

    def test_rsi_in_uptrend(self):
        closes = np.linspace(10.0, 18.0, 50)
        rsi = RiskChecker._compute_rsi(closes, period=14)
        assert rsi is not None
        assert rsi > 50  # strong uptrend = high RSI
        assert rsi <= 100

    def test_rsi_in_downtrend(self):
        closes = np.linspace(18.0, 10.0, 50)
        rsi = RiskChecker._compute_rsi(closes, period=14)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_rsi_flat(self):
        closes = np.full(50, 10.0) + np.random.RandomState(42).randn(50) * 0.01
        rsi = RiskChecker._compute_rsi(closes, period=14)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_rsi_insufficient_data(self):
        closes = np.array([10.0] * 10)
        rsi = RiskChecker._compute_rsi(closes, period=14)
        assert rsi is None


class TestATRComputation:
    """ATR computation helper."""

    def test_atr_positive(self):
        rng = np.random.RandomState(42)
        closes = np.linspace(10.0, 15.0, 30) + rng.randn(30) * 0.1
        bars = make_bars(closes)
        atr = RiskChecker._compute_atr(bars, period=14)
        assert atr is not None
        assert atr > 0

    def test_atr_insufficient_bars(self):
        bars = make_bars([10.0] * 10)
        atr = RiskChecker._compute_atr(bars, period=14)
        assert atr is None

    def test_atr_increases_with_volatility(self):
        rng = np.random.RandomState(42)
        # Low volatility
        low_vol = np.linspace(10.0, 12.0, 30) + rng.randn(30) * 0.05
        bars_low = make_bars(low_vol)
        atr_low = RiskChecker._compute_atr(bars_low, period=14)

        # Higher volatility
        high_vol = np.linspace(10.0, 12.0, 30) + rng.randn(30) * 0.5
        bars_high = make_bars(high_vol)
        atr_high = RiskChecker._compute_atr(bars_high, period=14)

        assert atr_low is not None and atr_high is not None
        assert atr_high > atr_low


class TestCheckAllIntegration:
    """Integration test: check_all() includes stop-loss and ATR checks."""

    def test_stop_loss_in_check_all(self):
        import quanti.config.settings as s
        s.STOP_LOSS_PCT = 8.0

        rc = RiskChecker()
        pf = make_pf("BAD", 1000, 10.0, 9.0, cash=90000.0)  # 10% loss > 8% stop
        approved, rejected = rc.check_all([], pf)

        sells = [o for o in approved if o.side == OrderSide.SELL]
        assert len(sells) >= 1, f"Expected stop-loss SELL in check_all, got {len(sells)} sells"
        assert "stop-loss" in sells[0].signal_ref

    def test_atr_stop_in_check_all(self):
        import quanti.config.settings as s
        s.STOP_LOSS_PCT = 0.0

        rng = np.random.RandomState(42)
        closes = np.concatenate([
            np.full(20, 10.0) + rng.randn(20) * 0.05,
            np.linspace(10.0, 15.0, 20),
            np.linspace(15.0, 8.0, 10),
        ])
        bars = make_bars(closes)
        rc = RiskChecker()
        rc._high_water_marks["TEST"] = 15.0
        pf = make_pf("TEST", 1000, 10.0, closes[-1])
        md = MarketData(bars={"TEST": bars}, index_bars={}, timestamp=datetime.now())

        # Enable ATR trailing stop via settings
        s.ATR_TRAILING_STOP_ENABLED = True
        approved, rejected = rc.check_all([], pf, market_data=md)

        sells = [o for o in approved if o.side == OrderSide.SELL]
        # In a sharp decline from HWM, should trigger ATR stop
        assert len(sells) >= 1, f"Expected ATR stop SELL, got {len(sells)} sells"

        s.ATR_TRAILING_STOP_ENABLED = False

    def test_buy_passes_through(self):
        """Valid BUY orders should pass through check_all."""
        rc = RiskChecker()
        pf = Portfolio(positions={}, cash=90000.0, total_capital=100000.0, timestamp=datetime.now())
        orders = [Order(symbol="GOOD", side=OrderSide.BUY, quantity=100, price=50.0, order_type="limit", signal_ref="test")]
        approved, rejected = rc.check_all(orders, pf)
        buys = [o for o in approved if o.side == OrderSide.BUY]
        assert len(buys) == 1
