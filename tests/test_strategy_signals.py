"""Known-input signal regression tests for ETFTrendStrategy.
Feeds controlled price data and verifies exact signal output."""
from datetime import datetime

import numpy as np

from quanti.strategy.etf_trend import ETFTrendStrategy
from quanti.types import Bar, MarketData, OrderSide


def _make_bars(close_prices: list[float], volumes: list[float] | None = None) -> list[Bar]:
    """Build Bar objects from close prices, with synthetic OHLCV."""
    if volumes is None:
        volumes = [1_000_000] * len(close_prices)
    bars = []
    for i, (c, v) in enumerate(zip(close_prices, volumes, strict=False)):
        d = datetime(2024, 1, 1) + np.timedelta64(i, 'D')
        if isinstance(d, np.datetime64):
            d = d.astype(object)
        bars.append(Bar(
            symbol='TEST.SH', datetime=d,
            open=c * 0.99, high=c * 1.01, low=c * 0.98,
            close=c, volume=v,
        ))
    return bars


class TestStrategySignals:
    """Verify correct signal generation for known market patterns."""

    def test_buy_signal_all_5_conditions_met(self):
        """All 5 conditions met (MA, ADX, BB, volume, market) -> BUY."""
        # 160 bars: tight range then sharp breakout for BB expansion + breakout
        rng = np.random.RandomState(42)
        # Days 1-150: tight range creates narrow Bollinger Bands
        tight = list(10.0 + rng.randn(150) * 0.05)
        # Days 151-160: sharp breakout drives price above upper band and expands BB width
        breakout = [10.0, 10.3, 10.7, 11.2, 11.8, 12.5, 13.3, 14.2, 15.2, 16.5]
        prices = tight + breakout
        vols = [1_000_000] * 159 + [4_000_000]  # volume surge on last day
        bars = _make_bars(prices, vols)
        strategy = ETFTrendStrategy(
            ma_fast=5, ma_slow=15, ma_long=60,
            adx_entry_threshold=15, di_diff_threshold=5,
            volume_surge_mult=1.5,
        )
        md = MarketData(bars={'TEST.SH': bars}, index_bars={}, timestamp=datetime.now())
        signals = strategy.generate_signals(md)
        buys = [s for s in signals if s.side == OrderSide.BUY]
        assert len(buys) > 0, f"Expected BUY signal, got {len(signals)} signals"

    def test_sell_signal_on_ma_crossover_down(self):
        """Fast MA crosses below slow MA -> SELL (baseline exit)."""
        # 160 days: first 150 trending up, last 10 reversing sharply down
        rng = np.random.RandomState(42)
        base = np.linspace(10.0, 18.0, 160)
        prices = list(base + rng.randn(160) * 0.15)
        # Sharp reversal at end
        prices[-15:] = [18.0, 17.5, 17.0, 16.5, 16.0, 15.5, 15.0, 14.5, 14.0, 13.5, 13.0, 12.5, 12.0, 11.5, 11.0]
        vols = [1_000_000] * 160
        bars = _make_bars(prices, vols)
        strategy = ETFTrendStrategy(ma_fast=10, ma_slow=30, ma_long=60, adx_entry_threshold=25)
        md = MarketData(bars={'TEST.SH': bars}, index_bars={}, timestamp=datetime.now())
        signals = strategy.generate_signals(md)
        sells = [s for s in signals if s.side == OrderSide.SELL]
        assert len(sells) > 0, f"Expected SELL signal, got {len(signals)} signals"

    def test_no_signal_when_no_trend(self):
        """Flat market -- no signals."""
        prices = [100.0] * 160
        bars = _make_bars(prices)
        strategy = ETFTrendStrategy(ma_fast=10, ma_slow=30)
        md = MarketData(bars={'TEST.SH': bars}, index_bars={}, timestamp=datetime.now())
        signals = strategy.generate_signals(md)
        assert len(signals) == 0, f"Expected 0 signals in flat market, got {len(signals)}"

    def test_insufficient_data_produces_no_signal(self):
        """Less than 120 bars -> no signals (need ma_long=120 by default)."""
        bars = _make_bars([100.0] * 60)
        strategy = ETFTrendStrategy(ma_fast=10, ma_slow=30)
        md = MarketData(bars={'TEST.SH': bars}, index_bars={}, timestamp=datetime.now())
        signals = strategy.generate_signals(md)
        assert len(signals) == 0, "Expected 0 signals with insufficient data"


class TestStopLoss:
    """Verify stop-loss generates correct sell orders."""

    def test_stop_loss_generates_sell(self):
        import quanti.config.settings as s
        s.STOP_LOSS_PCT = 10.0  # 10% loss trigger

        from quanti.types import Portfolio, Position
        pf = Portfolio(
            positions={'TEST.SH': Position(symbol='TEST.SH', quantity=1000, avg_cost=100.0, current_price=85.0)},
            cash=90000, total_capital=100000, timestamp=datetime.now(),
        )
        strategy = ETFTrendStrategy()
        approved = strategy.risk_check([], pf)
        sells = [o for o in approved if o.side == OrderSide.SELL]
        assert len(sells) == 1, f"Expected 1 stop-loss SELL, got {len(sells)}"

    def test_stop_loss_not_triggered_when_above_threshold(self):
        import quanti.config.settings as s
        s.STOP_LOSS_PCT = 10.0

        from quanti.types import Portfolio, Position
        pf = Portfolio(
            positions={'TEST.SH': Position(symbol='TEST.SH', quantity=1000, avg_cost=100.0, current_price=92.0)},
            cash=90000, total_capital=100000, timestamp=datetime.now(),
        )
        strategy = ETFTrendStrategy()
        approved = strategy.risk_check([], pf)
        sells = [o for o in approved if o.side == OrderSide.SELL]
        assert len(sells) == 0, f"Expected 0 stop-loss SELL (92 > 90), got {len(sells)}"
