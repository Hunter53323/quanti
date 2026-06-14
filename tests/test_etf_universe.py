"""Tests for the ETF universe definition module (quanti.config.etf_universe)."""
import pytest

from quanti.config.etf_universe import (
    ETF_UNIVERSE_LEGACY,
    ETF_UNIVERSE_MULTI,
    get_available_etfs,
    get_sector,
    get_sector_map,
)


class TestGetAvailableETFs:
    """Tests for get_available_etfs listing-date filtering."""

    def test_early_date_returns_legacy_like_set(self):
        """20150105 should return exactly 8 ETFs (legacy + early sector ETFs)."""
        etfs = get_available_etfs("20150105")
        codes = {e["code"] for e in etfs}
        assert len(etfs) == 8
        # Legacy ETFs should be present
        assert "510300" in codes
        assert "159915" in codes
        assert "510880" in codes
        # Early-listed sector ETFs should be present
        assert "159928" in codes  # 消费, listed 20130916
        assert "512010" in codes  # 医药 (消费 sector), listed 20130916

    def test_later_date_returns_full_universe(self):
        """20220104 should return all 25 ETFs (everyone has >120 days history)."""
        etfs = get_available_etfs("20220104")
        assert len(etfs) == 25

    def test_mid_date_returns_intermediate_count(self):
        """20191231 returns some count between 8 and 25."""
        etfs = get_available_etfs("20191231")
        codes = {e["code"] for e in etfs}
        assert 8 < len(etfs) < 25
        assert "159928" in codes
        assert "512010" in codes

    def test_invalid_date_format_raises_value_error(self):
        """YYYY-MM-DD format should raise ValueError."""
        with pytest.raises(ValueError):
            get_available_etfs("2022-01-04")


class TestGetSector:
    """Tests for get_sector mapping."""

    def test_known_code(self):
        assert get_sector("515790") == "新能源"

    def test_unknown_code_returns_unknown(self):
        assert get_sector("999999") == "未知"


class TestGetSectorMap:
    """Tests for get_sector_map."""

    def test_returns_exactly_nine_sectors(self):
        sector_map = get_sector_map()
        assert len(sector_map) == 9
        expected = {"宽基", "金融", "科技", "新能源", "消费", "资源", "TMT", "高端制造", "防御"}
        assert set(sector_map.keys()) == expected

    def test_sector_contains_expected_codes(self):
        sector_map = get_sector_map()
        assert "515790" in sector_map["新能源"]
        assert "510300" in sector_map["宽基"]
        assert "512480" in sector_map["科技"]


class TestUniverseConstants:
    """Tests for ETF_UNIVERSE_LEGACY and ETF_UNIVERSE_MULTI constants."""

    def test_legacy_has_exactly_six_codes(self):
        assert len(ETF_UNIVERSE_LEGACY) == 6

    def test_multi_has_exactly_25_entries(self):
        assert len(ETF_UNIVERSE_MULTI) == 25

    def test_multi_contains_legacy_codes(self):
        legacy_codes = ETF_UNIVERSE_LEGACY
        multi_codes = {e["code"] for e in ETF_UNIVERSE_MULTI}
        assert legacy_codes.issubset(multi_codes)
