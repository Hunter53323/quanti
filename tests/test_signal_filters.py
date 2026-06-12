"""Tests for MarketEnvironmentFilter in signal_filters.py."""
from datetime import date

import numpy as np

from quanti.strategy.etf_trend import ETFTrendStrategy
from quanti.strategy.signal_filters import MarketEnvironmentFilter


def make_index_bars(closes, highs=None, lows=None):
    """Create synthetic Bar-like objects for index testing."""
    class FakeBar:
        def __init__(self, c, h, l):
            self.close = c
            self.high = h
            self.low = l

    if highs is None:
        highs = [c + 0.1 for c in closes]
    if lows is None:
        lows = [c - 0.1 for c in closes]

    return [FakeBar(c, h, l) for c, h, l in zip(closes, highs, lows, strict=False)]


class TestIsTrending:
    """Market trending check: at least one index ADX > threshold."""

    def test_trending_market(self):
        """Strong trending index data -> is_trending returns True."""
        rng = np.random.RandomState(42)
        closes = np.concatenate([
            np.full(60, 10.0),
            np.linspace(10.0, 14.0, 40) + rng.randn(40) * 0.05,
        ])
        bars = make_index_bars(closes)
        flt = MarketEnvironmentFilter(market_adx_threshold=20.0)
        result = flt.is_trending({'CSI300': bars}, ETFTrendStrategy._adx)
        assert result is True

    def test_choppy_market(self):
        """Flat/choppy market -> is_trending returns False."""
        rng = np.random.RandomState(42)
        closes = np.full(100, 10.0) + rng.randn(100) * 0.02
        bars = make_index_bars(closes)
        flt = MarketEnvironmentFilter(market_adx_threshold=30.0)
        result = flt.is_trending({'CSI300': bars}, ETFTrendStrategy._adx)
        assert bool(result) is False

    def test_empty_bars_permissive(self):
        """No index data -> permissive (returns True)."""
        flt = MarketEnvironmentFilter()
        result = flt.is_trending({}, ETFTrendStrategy._adx)
        assert result is True

    def test_strict_threshold(self):
        """High ADX threshold blocks moderately trending market."""
        rng = np.random.RandomState(42)
        closes = np.concatenate([
            np.full(60, 10.0),
            np.linspace(10.0, 12.0, 40) + rng.randn(40) * 0.05,
        ])
        bars = make_index_bars(closes)
        flt = MarketEnvironmentFilter(market_adx_threshold=95.0)  # impossibly high
        result = flt.is_trending({'CSI300': bars}, ETFTrendStrategy._adx)
        assert bool(result) is False

    def test_multiple_indices_any_trending(self):
        """If any of multiple indices is trending, returns True."""
        rng = np.random.RandomState(42)
        # Choppy index
        choppy = np.full(100, 10.0) + rng.randn(100) * 0.02
        # Trending index
        trending = np.concatenate([np.full(60, 10.0), np.linspace(10.0, 14.0, 40)])

        flt = MarketEnvironmentFilter(market_adx_threshold=20.0)
        result = flt.is_trending(
            {'SH000001': make_index_bars(choppy), 'CSI300': make_index_bars(trending)},
            ETFTrendStrategy._adx,
        )
        assert result is True


class TestIsBearMarket:
    """Bear market detection: index 120-day MA declining."""

    def test_bull_market(self):
        """Rising prices -> SMA rising -> not bear."""
        closes = np.linspace(10.0, 15.0, 130)
        bars = make_index_bars(closes)
        flt = MarketEnvironmentFilter(index_sma_long=20)  # shorter for test
        result = flt.is_bear_market({'CSI300': bars}, ETFTrendStrategy._sma)
        assert bool(result) is False

    def test_bear_market(self):
        """Declining prices -> SMA declining -> bear."""
        closes = np.linspace(15.0, 10.0, 130)
        bars = make_index_bars(closes)
        flt = MarketEnvironmentFilter(index_sma_long=20)
        result = flt.is_bear_market({'CSI300': bars}, ETFTrendStrategy._sma)
        assert bool(result) is True

    def test_empty_bars_assumes_bull(self):
        """No data -> assume bull (returns False)."""
        flt = MarketEnvironmentFilter()
        result = flt.is_bear_market({}, ETFTrendStrategy._sma)
        assert bool(result) is False


class TestForbiddenPeriod:
    """Manual calendar blackout check."""

    def test_normal_date_not_forbidden(self):
        flt = MarketEnvironmentFilter()
        assert flt.is_forbidden_period(date(2026, 6, 11)) is False

    def test_added_date_is_forbidden(self):
        flt = MarketEnvironmentFilter()
        flt.add_forbidden_date(date(2026, 1, 1))
        assert flt.is_forbidden_period(date(2026, 1, 1)) is True
        assert flt.is_forbidden_period(date(2026, 1, 2)) is False

    def test_remove_date(self):
        flt = MarketEnvironmentFilter()
        flt.add_forbidden_date(date(2026, 1, 1))
        flt.remove_forbidden_date(date(2026, 1, 1))
        assert flt.is_forbidden_period(date(2026, 1, 1)) is False

    def test_defaults_to_today(self):
        flt = MarketEnvironmentFilter()
        # Today should not be forbidden (we didn't add it)
        assert flt.is_forbidden_period() is False


class TestPositionSizeMultiplier:
    """Position sizing during bull vs bear."""

    def test_bull_market_full_size(self):
        closes = np.linspace(10.0, 15.0, 130)
        bars = make_index_bars(closes)
        flt = MarketEnvironmentFilter(index_sma_long=20)
        mult = flt.get_position_size_multiplier({'CSI300': bars}, ETFTrendStrategy._sma)
        assert mult == 1.0

    def test_bear_market_defense_size(self):
        closes = np.linspace(15.0, 10.0, 130)
        bars = make_index_bars(closes)
        flt = MarketEnvironmentFilter(index_sma_long=20, defense_pct=0.25)
        mult = flt.get_position_size_multiplier({'CSI300': bars}, ETFTrendStrategy._sma)
        assert mult == 0.25


class TestShouldTrade:
    """Composite gate: trending + not forbidden."""

    def test_all_clear(self):
        np.random.RandomState(42)
        closes = np.concatenate([np.full(60, 10.0), np.linspace(10.0, 14.0, 40)])
        bars = make_index_bars(closes)
        flt = MarketEnvironmentFilter(market_adx_threshold=20.0)
        result = flt.should_trade({'CSI300': bars}, ETFTrendStrategy._adx, ETFTrendStrategy._sma)
        assert result is True

    def test_forbidden_blocked(self):
        np.random.RandomState(42)
        closes = np.concatenate([np.full(60, 10.0), np.linspace(10.0, 14.0, 40)])
        bars = make_index_bars(closes)
        flt = MarketEnvironmentFilter(market_adx_threshold=20.0)
        flt.add_forbidden_date(date.today())
        result = flt.should_trade({'CSI300': bars}, ETFTrendStrategy._adx, ETFTrendStrategy._sma)
        assert bool(result) is False

    def test_no_trend_blocked(self):
        rng = np.random.RandomState(42)
        closes = np.full(100, 10.0) + rng.randn(100) * 0.02
        bars = make_index_bars(closes)
        flt = MarketEnvironmentFilter(market_adx_threshold=30.0)
        result = flt.should_trade({'CSI300': bars}, ETFTrendStrategy._adx, ETFTrendStrategy._sma)
        assert bool(result) is False
