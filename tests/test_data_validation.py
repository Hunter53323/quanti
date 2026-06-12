"""Tests for data validation pipeline."""

from quanti.data.validation import DataValidator


class TestDataValidator:

    def test_validates_clean_bars(self, sample_bars):
        validator = DataValidator()
        valid, warnings = validator.validate_bars(sample_bars)
        assert len(valid) == 3
        assert len(warnings) == 0

    def test_rejects_zero_price_bars(self, corrupt_bars):
        validator = DataValidator()
        valid, warnings = validator.validate_bars(corrupt_bars)
        # Bar 1: zero prices -> rejected
        # Bar 2: high < low -> rejected
        # Bar 3: OK
        # Bar 4: duplicate date -> rejected
        # Only bar 3 valid
        assert len(valid) == 1
        assert valid[0].trade_date == "20240103"
        assert len(warnings) >= 3  # zero price, high<low, duplicate

    def test_detects_missing_fields_threshold(self):
        from quanti.data.schema import ETFDailyBar

        # 3 out of 4 bars have zero close = 75% > 5% threshold
        bars = [
            ETFDailyBar("510300.SH", "20240101", 3.5, 3.6, 3.4, 0.0, 1000000.0, 3500000.0),
            ETFDailyBar("510300.SH", "20240102", 3.5, 3.6, 3.4, 0.0, 1000000.0, 3500000.0),
            ETFDailyBar("510300.SH", "20240103", 3.5, 3.6, 3.4, 0.0, 1000000.0, 3500000.0),
            ETFDailyBar("510300.SH", "20240104", 3.5, 3.6, 3.4, 3.5, 1000000.0, 3500000.0),
        ]
        validator = DataValidator(missing_threshold_pct=5.0)
        valid, warnings = validator.validate_bars(bars)
        # close field is missing in 75% -> triggered
        close_warnings = [w for w in warnings if "close" in w.lower()]
        assert len(close_warnings) > 0

    def test_compare_sources_no_discrepancy(self, sample_bars):
        validator = DataValidator()
        # Identical bars should produce no discrepancies
        disc = validator.compare_sources(sample_bars, sample_bars)
        assert len(disc) == 0

    def test_compare_sources_detects_price_diff(self, sample_bars):
        from quanti.data.schema import ETFDailyBar
        modified = [
            ETFDailyBar(
                symbol=b.symbol, trade_date=b.trade_date,
                open=b.open, high=b.high, low=b.low,
                close=b.close * 1.02,  # 2% higher
                volume=b.volume, amount=b.amount,
            )
            for b in sample_bars
        ]
        validator = DataValidator(price_discrepancy_pct=1.0)
        disc = validator.compare_sources(sample_bars, modified)
        assert len(disc) == 3  # All 3 bars differ by 2% > 1% threshold

    def test_freshness_check_stale(self):
        validator = DataValidator()
        # Use last month's date
        result = validator.check_freshness("510300.SH", "20250101", deadline_hour=0)
        assert result is not None
        assert "stale" in result.lower() or "510300" in result

    def test_empty_bars(self):
        validator = DataValidator()
        valid, warnings = validator.validate_bars([])
        assert len(valid) == 0
        assert len(warnings) > 0

