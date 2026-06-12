"""Tests for multi-indicator entry signals."""
from datetime import datetime

import numpy as np

from quanti.strategy.etf_trend import ETFTrendStrategy
from quanti.types import Bar, MarketData, OrderSide


def make_bars(closes, volumes=None, symbol="510300"):
    """Create Bar objects from close prices and optional volumes."""
    if volumes is None:
        volumes = [1000000] * len(closes)
    bars = []
    for i, (c, v) in enumerate(zip(closes, volumes, strict=False)):
        d = datetime(2024, 1, 1) + np.timedelta64(i, 'D')
        if isinstance(d, np.datetime64):
            d = d.astype(object)
        bars.append(Bar(
            symbol=symbol,
            datetime=d,
            open=c,
            high=c * 1.01,
            low=c * 0.99,
            close=c,
            volume=v,
        ))
    return bars


class TestMAAlignment:
    """SMA alignment: SMA20 > SMA60 > SMA120 (both today and yesterday)."""

    def test_ma_alignment_true(self):
        """SMA20 > SMA60 > SMA120 on rising prices -- alignment should be True."""
        closes = np.concatenate([
            np.full(120, 1.0),
            np.linspace(1.0, 1.3, 40),
        ])
        strategy = ETFTrendStrategy(ma_fast=20, ma_slow=60)
        result = strategy._check_ma_alignment(closes)
        assert bool(result) is True

    def test_ma_alignment_false_downtrend(self):
        """SMA20 < SMA60 on falling prices -- alignment should be False."""
        closes = np.concatenate([
            np.full(120, 1.0),
            np.linspace(1.0, 0.7, 40),
        ])
        strategy = ETFTrendStrategy(ma_fast=20, ma_slow=60)
        result = strategy._check_ma_alignment(closes)
        assert bool(result) is False

    def test_ma_alignment_false_flat(self):
        """Flat market -- MAs cross each other, no clear alignment."""
        closes = np.full(160, 1.0) + np.random.RandomState(42).randn(160) * 0.005
        strategy = ETFTrendStrategy(ma_fast=20, ma_slow=60)
        result = strategy._check_ma_alignment(closes)
        assert bool(result) is False


class TestBBExpansion:
    """Bollinger Band expansion breakout detection."""

    def test_bb_expansion_detected(self):
        """BB bandwidth expanding and price above upper band."""
        rng = np.random.RandomState(42)
        closes = np.concatenate([
            np.full(20, 1.0) + rng.randn(20) * 0.01,
            np.linspace(1.0, 1.15, 10),
        ])
        strategy = ETFTrendStrategy()
        result = strategy._check_bb_expansion(closes)
        assert bool(result) is True

    def test_bb_no_expansion_in_range(self):
        """BB bandwidth not expanding when prices range-bound."""
        rng = np.random.RandomState(42)
        closes = np.full(30, 1.0) + rng.randn(30) * 0.01
        strategy = ETFTrendStrategy()
        result = strategy._check_bb_expansion(closes)
        assert bool(result) is False


class TestVolumeSurge:
    """Volume surge confirmation."""

    def test_volume_surge_detected(self):
        """Last day volume is Nx the 20-day average -- surge detected."""
        closes = [1.0] * 30
        volumes = [1000000] * 29 + [3000000]  # 3x last day
        bars = make_bars(closes, volumes)
        strategy = ETFTrendStrategy(volume_surge_mult=1.5)
        result = strategy._check_volume_surge(bars)
        assert bool(result) is True

    def test_volume_normal_no_surge(self):
        """Volume is within normal range -- no surge."""
        closes = [1.0] * 30
        volumes = [1000000] * 30
        bars = make_bars(closes, volumes)
        strategy = ETFTrendStrategy(volume_surge_mult=1.5)
        result = strategy._check_volume_surge(bars)
        assert bool(result) is False

    def test_volume_surge_edge_case_equal(self):
        """Volume exactly at multiplier threshold (uses strict > so is False)."""
        closes = [1.0] * 30
        avg_vol = 1000000
        volumes = [avg_vol] * 29 + [int(avg_vol * 1.5)]
        bars = make_bars(closes, volumes)
        strategy = ETFTrendStrategy(volume_surge_mult=1.5)
        result = strategy._check_volume_surge(bars)
        assert bool(result) is False  # strict >, not >=


class TestADXEntryFilter:
    """ADX-based entry filtering (ADX > threshold AND +DI > -DI AND DI diff > threshold)."""

    def test_adx_above_threshold_passes(self):
        """ADX > ADX_ENTRY_THRESHOLD, +DI > -DI, DI diff > threshold -> passes."""
        rng = np.random.RandomState(42)
        closes = np.concatenate([np.full(60, 10.0), np.linspace(10.0, 13.0, 40) + rng.randn(40) * 0.05])
        highs = closes + 0.15 + rng.random(len(closes)) * 0.1
        lows = closes - 0.15 - rng.random(len(closes)) * 0.1
        strategy = ETFTrendStrategy(adx_entry_threshold=15, di_diff_threshold=5)
        result = strategy._check_adx_trend(highs, lows, closes)
        assert bool(result) is True

    def test_adx_below_threshold_blocks(self):
        """ADX < ADX_ENTRY_THRESHOLD -> blocks entry."""
        rng = np.random.RandomState(42)
        closes = np.full(100, 10.0) + rng.randn(100) * 0.02  # flat/choppy
        highs = closes + 0.05
        lows = closes - 0.05
        strategy = ETFTrendStrategy(adx_entry_threshold=25, di_diff_threshold=10)
        result = strategy._check_adx_trend(highs, lows, closes)
        assert bool(result) is False


class TestDIDiffFilter:
    """DI difference filter (+DI - -DI > threshold)."""

    def test_di_diff_above_threshold_passes(self):
        """+DI - -DI > DI_DIFF_THRESHOLD -- directional bias confirmed."""
        rng = np.random.RandomState(42)
        closes = np.concatenate([np.full(60, 10.0), np.linspace(10.0, 13.0, 40) + rng.randn(40) * 0.05])
        highs = closes + 0.15 + rng.random(len(closes)) * 0.1
        lows = closes - 0.15 - rng.random(len(closes)) * 0.1
        strategy = ETFTrendStrategy(adx_entry_threshold=10, di_diff_threshold=5)
        result = strategy._check_adx_trend(highs, lows, closes)
        assert bool(result) is True

    def test_di_diff_below_threshold_blocks(self):
        """+DI - -DI < DI_DIFF_THRESHOLD -- no clear direction."""
        rng = np.random.RandomState(99)
        closes = np.full(100, 10.0) + rng.randn(100) * 0.15  # random walk
        highs = closes + 0.10
        lows = closes - 0.10
        strategy = ETFTrendStrategy(adx_entry_threshold=5, di_diff_threshold=30)  # impossibly high
        result = strategy._check_adx_trend(highs, lows, closes)
        assert bool(result) is False


class TestRSIFilter:
    """RSI-based exit: RSI > 80 tightens ATR multiplier from 2x to 1.5x."""

    def test_rsi_not_overbought_passes(self):
        """RSI < RSI_OVERBOUGHT: normal ATR multiplier."""
        rng = np.random.RandomState(42)
        closes = np.linspace(10.0, 12.0, 50) + rng.randn(50) * 0.1
        strategy = ETFTrendStrategy()
        rsi = strategy._compute_rsi(closes)
        assert rsi is not None
        assert rsi < 80, f"Expected RSI < 80, got {rsi}"

    def test_rsi_overbought(self):
        """RSI > RSI_OVERBOUGHT: should detect overbought condition."""
        # Build a steady uptrend that produces high RSI
        closes = np.concatenate([np.full(14, 10.0), np.linspace(10.0, 18.0, 50)])
        strategy = ETFTrendStrategy()
        rsi = strategy._compute_rsi(closes)
        assert rsi is not None
        assert rsi > 80, f"Expected RSI > 80 in strong uptrend, got {rsi}"


class TestCompositeEntry:
    """Full composite entry signal (all filters must pass)."""

    def test_all_filters_pass_produces_buy(self):
        """When MA alignment, BB expansion, volume surge all pass -> BUY."""
        rng = np.random.RandomState(42)
        # 150 days tight range (low vol = narrow BB), then 10-day sharp breakout
        tight = list(10.0 + rng.randn(150) * 0.05)
        breakout = [10.0, 10.3, 10.7, 11.2, 11.8, 12.5, 13.3, 14.2, 15.2, 16.5]
        prices = tight + breakout
        vols = [1_000_000] * 159 + [4_000_000]
        bars = make_bars(prices, vols)
        strategy = ETFTrendStrategy(
            ma_fast=5, ma_slow=15, ma_long=60,
            adx_entry_threshold=15, di_diff_threshold=5,
            volume_surge_mult=1.5,
        )
        md = MarketData(bars={'510300': bars}, index_bars={}, timestamp=datetime.now())
        signals = strategy.generate_signals(md)
        buys = [s for s in signals if s.side == OrderSide.BUY]
        assert len(buys) > 0, f"Expected BUY signal, got {len(signals)} signals"
        assert buys[0].strength > 0, "Signal strength should be positive"

    def test_one_filter_fails_no_buy(self):
        """When all conditions are weak (choppy, no breakout) -> no BUY (score < threshold)."""
        rng = np.random.RandomState(42)
        # Choppy flat market: no trend, no BB expansion, no volume surge
        prices = list(10.0 + rng.randn(160) * 0.03)
        vols = [1_000_000] * 160  # constant volume
        bars = make_bars(prices, vols)
        strategy = ETFTrendStrategy(
            ma_fast=5, ma_slow=15, ma_long=60,
            adx_entry_threshold=15, di_diff_threshold=5,
            volume_surge_mult=1.5,
        )
        md = MarketData(bars={'510300': bars}, index_bars={}, timestamp=datetime.now())
        signals = strategy.generate_signals(md)
        buys = [s for s in signals if s.side == OrderSide.BUY]
        assert len(buys) == 0, f"Expected no BUY (all conditions weak), got {len(buys)} buys"

    def test_entry_with_custom_parameters(self):
        """Strategy picks up custom __init__ parameters over settings defaults."""
        strategy = ETFTrendStrategy(
            ma_fast=10, ma_slow=30, ma_long=100,
            bb_period=15, bb_std=2.5,
        )
        assert strategy.ma_fast == 10
        assert strategy.ma_slow == 30
        assert strategy.ma_long == 100
        assert strategy.bb_period == 15
        assert strategy.bb_std == 2.5
        assert strategy.adx_threshold is not None
