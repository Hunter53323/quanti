"""
Data validation pipeline.
Validates incoming data before it reaches the strategy layer.
"""

from datetime import datetime

from quanti.data.schema import ETFDailyBar


class DataValidator:
    """Validates and cleans ETF daily bar data."""

    def __init__(
        self,
        missing_threshold_pct: float = 5.0,
        price_discrepancy_pct: float = 1.0,
    ):
        self.missing_threshold_pct = missing_threshold_pct
        self.price_discrepancy_pct = price_discrepancy_pct

    def validate_bars(self, bars: list[ETFDailyBar]) -> tuple[list[ETFDailyBar], list[str]]:
        """
        Validate a list of bars. Returns (valid_bars, warnings).

        Checks:
        - Non-zero prices (open, high, low, close)
        - High >= Low
        - No duplicate dates
        - Missing fields threshold check
        """
        if not bars:
            return ([], ["No bars to validate"])

        valid: list[ETFDailyBar] = []
        warnings: list[str] = []

        seen_dates: set[str] = set()
        for bar in bars:
            issues = []

            # Non-zero price check
            if bar.open <= 0 or bar.high <= 0 or bar.low <= 0 or bar.close <= 0:
                issues.append("zero_or_negative_price")

            # High/low sanity
            if bar.high < bar.low:
                issues.append("high_below_low")

            # Duplicate date
            if bar.trade_date in seen_dates:
                issues.append("duplicate_date")
            else:
                seen_dates.add(bar.trade_date)

            # Volume zero check (zero volume = likely suspended or bad data)
            if bar.volume <= 0:
                issues.append("zero_volume")

            if issues:
                warnings.append(f"{bar.symbol} {bar.trade_date}: {', '.join(issues)}")
            else:
                valid.append(bar)

        # Missing fields threshold: warn if any field is zero across ALL bars
        total = len(bars)
        if total > 0:
            fields = {
                "open": sum(1 for b in bars if b.open <= 0),
                "high": sum(1 for b in bars if b.high <= 0),
                "low": sum(1 for b in bars if b.low <= 0),
                "close": sum(1 for b in bars if b.close <= 0),
                "volume": sum(1 for b in bars if b.volume <= 0),
            }
            for field, missing in fields.items():
                pct = (missing / total) * 100
                if pct > self.missing_threshold_pct:
                    warnings.append(f"Field '{field}' missing at {pct:.1f}% (threshold: {self.missing_threshold_pct}%)")

        return (valid, warnings)

    def compare_sources(
        self,
        tushare_bars: list[ETFDailyBar],
        akshare_bars: list[ETFDailyBar],
    ) -> list[str]:
        """
        Compare data from two sources for the same symbol/date.
        Returns list of discrepancies.
        """
        discrepancies: list[str] = []

        # Build lookup by date for each source
        ts_by_date = {b.trade_date: b for b in tushare_bars}
        ak_by_date = {b.trade_date: b for b in akshare_bars}

        for date, ts_bar in ts_by_date.items():
            ak_bar = ak_by_date.get(date)
            if ak_bar is None:
                continue

            # Compare close prices (most important field)
            if ts_bar.close > 0 and ak_bar.close > 0:
                diff_pct = abs(ts_bar.close - ak_bar.close) / ts_bar.close * 100
                if diff_pct > self.price_discrepancy_pct:
                    discrepancies.append(
                        f"{ts_bar.symbol} {date}: close {ts_bar.close} vs {ak_bar.close} ({diff_pct:.2f}%)"
                    )

        return discrepancies

    def check_freshness(
        self, symbol: str, last_date: str, deadline_hour: int = 17
    ) -> str | None:
        """
        Check if data is stale. Returns warning message or None.

        Args:
            symbol: ETF symbol
            last_date: Last available date (YYYYMMDD)
            deadline_hour: Hour after which we consider data late (default 17:00)
        """
        today = datetime.now()
        today_str = today.strftime("%Y%m%d")

        # Only check on weekdays (Mon=0..Fri=4)
        if today.weekday() >= 5:
            return None

        if last_date < today_str and today.hour >= deadline_hour:
            return f"{symbol}: Data stale. Last: {last_date}, expected: {today_str}"

        return None
