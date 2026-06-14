"""Tests for per-sector concentration limits in ETFRotationStrategy."""
from datetime import datetime, timedelta

import numpy as np

from quanti.strategy.etf_rotation import ETFRotationStrategy
from quanti.types import Bar, MarketData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(end_price: float, n: int = 150, symbol: str = "510300") -> list[Bar]:
    """Create ``n`` bars whose close prices increase linearly from 10 to
    *end_price*, ensuring a positive trend score and rising MA."""
    closes = np.linspace(10.0, end_price, n)
    start = datetime(2020, 1, 1)
    bars: list[Bar] = []
    for i, c in enumerate(closes):
        bars.append(Bar(
            symbol=symbol,
            datetime=start + timedelta(days=i),
            open=c * 0.99,
            high=c * 1.01,
            low=c * 0.98,
            close=c,
            volume=1_000_000,
        ))
    return bars


# ---------------------------------------------------------------------------
# Concentration cap enforcement
# ---------------------------------------------------------------------------

class TestConcentrationCap:
    """Verify that per-sector concentration limits are enforced."""

    def test_max_two_per_sector_respected(self):
        """3 新能源 ETFs rank top 3 by score, but only 2 are kept and the
        4th-ranked ETF from 科技 fills the remaining slot."""
        bars = {
            "515790": _make_bars(14.0, symbol="515790"),   # 新能源, highest score
            "516160": _make_bars(12.0, symbol="516160"),   # 新能源, 2nd
            "516110": _make_bars(11.0, symbol="516110"),   # 新能源, 3rd
            "512480": _make_bars(10.5, symbol="512480"),   # 科技, 4th
        }
        md = MarketData(bars=bars, index_bars={}, timestamp=datetime(2022, 1, 4))
        strategy = ETFRotationStrategy(top_n=3, max_per_sector=2)
        signals = strategy.generate_signals(md)

        symbols = {s.symbol for s in signals}
        assert len(signals) == 3
        assert "515790" in symbols
        assert "516160" in symbols
        # 516110 (3rd-ranked) is skipped -- exceeds cap; 512480 (4th) is in
        assert "512480" in symbols
        assert "516110" not in symbols

    def test_sector_counts_tracked(self):
        """sector_counts dict is populated after generate_signals."""
        bars = {
            "515790": _make_bars(14.0, symbol="515790"),   # 新能源
            "516160": _make_bars(12.0, symbol="516160"),   # 新能源
            "512480": _make_bars(10.5, symbol="512480"),   # 科技
        }
        md = MarketData(bars=bars, index_bars={}, timestamp=datetime(2022, 1, 4))
        strategy = ETFRotationStrategy(top_n=2, max_per_sector=2)
        strategy.generate_signals(md)

        assert strategy._sector_counts.get("新能源", 0) == 2
        assert strategy._sector_counts.get("科技", 0) == 0


# ---------------------------------------------------------------------------
# Exempt sectors (宽基 and 防御)
# ---------------------------------------------------------------------------

class TestExemptSectors:
    """宽基 (broad-based) and 防御 (defensive) sectors ignore caps."""

    def test_defensive_sector_exempt_from_cap(self):
        """3 防御 ETFs can all be selected even with max_per_sector=2."""
        bars = {
            "510880": _make_bars(14.0, symbol="510880"),   # 防御
            "518880": _make_bars(12.0, symbol="518880"),   # 防御
            "511880": _make_bars(11.0, symbol="511880"),   # 防御
            "512480": _make_bars(10.0, symbol="512480"),   # 科技
        }
        md = MarketData(bars=bars, index_bars={}, timestamp=datetime(2022, 1, 4))
        strategy = ETFRotationStrategy(top_n=3, max_per_sector=2)
        signals = strategy.generate_signals(md)

        symbols = {s.symbol for s in signals}
        assert len(signals) == 3
        assert "510880" in symbols
        assert "518880" in symbols
        assert "511880" in symbols  # 3rd defensive ETF selected despite cap

    def test_broad_based_sector_exempt_from_cap(self):
        """3 宽基 ETFs can all be selected even with max_per_sector=2."""
        bars = {
            "510300": _make_bars(14.0, symbol="510300"),   # 宽基
            "510500": _make_bars(12.0, symbol="510500"),   # 宽基
            "159915": _make_bars(11.0, symbol="159915"),   # 宽基
            "512480": _make_bars(10.0, symbol="512480"),   # 科技
        }
        md = MarketData(bars=bars, index_bars={}, timestamp=datetime(2022, 1, 4))
        strategy = ETFRotationStrategy(top_n=3, max_per_sector=2)
        signals = strategy.generate_signals(md)

        symbols = {s.symbol for s in signals}
        assert len(signals) == 3
        assert "510300" in symbols
        assert "510500" in symbols
        assert "159915" in symbols  # 3rd broad-based ETF selected despite cap


# ---------------------------------------------------------------------------
# Backward compatibility (legacy pool)
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """use_multi_sector=False uses the original 6-ETF hardcoded pool."""

    def test_legacy_mode_ignores_multi_sector_etfs(self):
        """515790 (新能源) is not in the legacy pool so it is never selected."""
        bars = {
            "515790": _make_bars(14.0, symbol="515790"),   # not in legacy
            "510300": _make_bars(12.0, symbol="510300"),   # in legacy
            "159915": _make_bars(11.0, symbol="159915"),   # in legacy
        }
        md = MarketData(bars=bars, index_bars={}, timestamp=datetime(2022, 1, 4))
        strategy = ETFRotationStrategy(top_n=2, use_multi_sector=False)
        signals = strategy.generate_signals(md)

        symbols = {s.symbol for s in signals}
        assert "515790" not in symbols
        assert "510300" in symbols
        assert "159915" in symbols

    def test_legacy_mode_no_concentration_limit(self):
        """Legacy mode does not enforce sector caps."""
        bars = {
            "510880": _make_bars(14.0, symbol="510880"),   # 防御, in legacy
            "518880": _make_bars(12.0, symbol="518880"),   # 防御, in legacy
            "159915": _make_bars(11.0, symbol="159915"),   # 宽基, in legacy
        }
        md = MarketData(bars=bars, index_bars={}, timestamp=datetime(2022, 1, 4))
        strategy = ETFRotationStrategy(top_n=2, use_multi_sector=False)
        signals = strategy.generate_signals(md)

        assert len(signals) == 2


# ---------------------------------------------------------------------------
# Signal reason string
# ---------------------------------------------------------------------------

class TestReasonString:
    """Sector info appears in signal reason for multi-sector mode."""

    def test_reason_contains_sector_name(self):
        """Each signal's reason includes the sector name in brackets."""
        bars = {
            "515790": _make_bars(14.0, symbol="515790"),   # 新能源
            "512480": _make_bars(12.0, symbol="512480"),   # 科技
        }
        md = MarketData(bars=bars, index_bars={}, timestamp=datetime(2022, 1, 4))
        strategy = ETFRotationStrategy(top_n=2)
        signals = strategy.generate_signals(md)

        for s in signals:
            if s.symbol == "515790":
                assert "[新能源]" in s.reason
            elif s.symbol == "512480":
                assert "[科技]" in s.reason
