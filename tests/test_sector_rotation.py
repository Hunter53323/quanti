"""Tests for SectorRotation in sector_rotation.py."""
import numpy as np
import pytest

from quanti.strategy.sector_rotation import SECTOR_ETF_MAP, SectorRotation


def make_fake_bars(symbol, closes):
    """Create Bar-like objects from close prices."""
    class FakeBar:
        def __init__(self, sym, cl):
            self.symbol = sym
            self.close = cl
    return [FakeBar(symbol, c) for c in closes]


class TestRankByMomentum:
    def test_ranks_by_return(self):
        sr = SectorRotation(top_n=3, lookback_days=20)
        bars = {
            "A": make_fake_bars("A", np.linspace(10.0, 15.0, 50)),
            "B": make_fake_bars("B", np.linspace(10.0, 12.0, 50)),
            "C": make_fake_bars("C", np.linspace(10.0, 18.0, 50)),
        }
        ranked = sr.rank_by_momentum(bars)
        assert len(ranked) == 3
        assert ranked[0][0] == "C"  # 18/10 = 80% return
        assert ranked[1][0] == "A"  # 15/10 = 50% return
        assert ranked[2][0] == "B"  # 12/10 = 20% return

    def test_empty_data(self):
        sr = SectorRotation()
        ranked = sr.rank_by_momentum({})
        assert len(ranked) == 0

    def test_insufficient_bars(self):
        sr = SectorRotation(lookback_days=20)
        bars = {"A": make_fake_bars("A", [10.0] * 10)}
        ranked = sr.rank_by_momentum(bars)
        assert len(ranked) == 0

    def test_negative_returns_ranked_below_positive(self):
        sr = SectorRotation(lookback_days=20)
        bars = {
            "A": make_fake_bars("A", np.linspace(10.0, 9.0, 50)),   # -10%
            "B": make_fake_bars("B", np.linspace(10.0, 10.5, 50)),  # +5%
        }
        ranked = sr.rank_by_momentum(bars)
        assert ranked[0][0] == "B"
        assert ranked[1][0] == "A"


class TestSelectCandidates:
    def test_select_top_n(self):
        sr = SectorRotation(top_n=2, lookback_days=20)
        bars = {
            "A": make_fake_bars("A", np.linspace(10.0, 18.0, 50)),
            "B": make_fake_bars("B", np.linspace(10.0, 15.0, 50)),
            "C": make_fake_bars("C", np.linspace(10.0, 12.0, 50)),
        }
        candidates = sr.select_candidates(bars, lambda s, b: True)
        assert len(candidates) == 2
        assert candidates[0] == "A"
        assert candidates[1] == "B"

    def test_filter_removes_failing(self):
        sr = SectorRotation(top_n=3, lookback_days=20)
        bars = {
            "A": make_fake_bars("A", np.linspace(10.0, 18.0, 50)),
            "B": make_fake_bars("B", np.linspace(10.0, 15.0, 50)),
            "C": make_fake_bars("C", np.linspace(10.0, 12.0, 50)),
        }
        candidates = sr.select_candidates(bars, lambda s, b: s != "A")
        assert len(candidates) == 2
        assert candidates[0] == "B"
        assert candidates[1] == "C"

    def test_not_enough_qualifying(self):
        sr = SectorRotation(top_n=3, lookback_days=20)
        bars = {
            "A": make_fake_bars("A", np.linspace(10.0, 15.0, 50)),
            "B": make_fake_bars("B", np.linspace(10.0, 12.0, 50)),
        }
        candidates = sr.select_candidates(bars, lambda s, b: s == "A")
        assert len(candidates) == 1

    def test_empty_data(self):
        sr = SectorRotation()
        candidates = sr.select_candidates({}, lambda s, b: True)
        assert len(candidates) == 0


class TestSectorETFMap:
    def test_known_symbols(self):
        assert SectorRotation.get_sector_name("512000") == "broker"
        assert SectorRotation.get_sector_name("512480") == "semiconductor"

    def test_unknown_symbol(self):
        assert SectorRotation.get_sector_name("999999") is None

    def test_get_symbol_by_name(self):
        assert SectorRotation.get_sector_symbol("bank") == "512800"
        assert SectorRotation.get_sector_symbol("consumer") == "159928"

    def test_all_symbols(self):
        symbols = SectorRotation.get_sector_etf_symbols()
        assert len(symbols) == 8
        assert "512000" in symbols
        assert "513130" in symbols

    def test_map_has_expected_sectors(self):
        expected = {"broker", "semiconductor", "new_energy", "consumer",
                    "pharma", "defense", "bank", "hang_seng_tech"}
        assert set(SECTOR_ETF_MAP.keys()) == expected


class TestRelativeStrength:
    def test_relative_strength_ranking(self):
        sr = SectorRotation(lookback_days=20)
        benchmark = make_fake_bars("CSI300", np.linspace(10.0, 11.0, 50))  # +10%
        bars = {
            "A": make_fake_bars("A", np.linspace(10.0, 12.0, 50)),  # +20%, RS=2.0
            "B": make_fake_bars("B", np.linspace(10.0, 10.5, 50)),  # +5%, RS=0.5
        }
        ranked = sr.rank_by_relative_strength(bars, benchmark)
        assert len(ranked) == 2
        assert ranked[0][0] == "A"

    def test_no_benchmark_falls_back_to_absolute(self):
        sr = SectorRotation(lookback_days=20)
        bars = {
            "A": make_fake_bars("A", np.linspace(10.0, 15.0, 50)),
            "B": make_fake_bars("B", np.linspace(10.0, 12.0, 50)),
        }
        ranked = sr.rank_by_relative_strength(bars, None)
        assert len(ranked) == 2
        assert ranked[0][0] == "A"


class TestConstructorValidation:
    def test_invalid_top_n(self):
        with pytest.raises(ValueError):
            SectorRotation(top_n=0)

    def test_invalid_lookback(self):
        with pytest.raises(ValueError):
            SectorRotation(lookback_days=1)
