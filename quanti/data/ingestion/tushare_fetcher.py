"""
Tushare ETF data fetcher -- primary data source.
Fetches daily OHLCV bars for specified ETF symbols.

Prerequisites: TUSHARE_TOKEN must be set in .env
"""

import time
from datetime import datetime, timedelta

import tushare as ts

from quanti.config import settings
from quanti.data.schema import ETFDailyBar


class TushareETFetcher:
    """Fetches ETF daily bars from Tushare Pro API."""

    def __init__(self, token: str | None = None):
        token = token or settings.TUSHARE_TOKEN
        if not token or token == "your_token_here":
            raise ValueError(
                "TUSHARE_TOKEN not configured. Set it in .env file."
            )
        self._api = ts.pro_api(token)
        self._last_call = 0.0
        self._min_interval = 1.0 / 3  # Tushare free tier: ~3 calls/sec

    def _rate_limit(self):
        """Simple rate limiter with fixed delay."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def fetch_daily(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[ETFDailyBar]:
        """
        Fetch daily bars for one ETF symbol.

        Args:
            symbol: ETF code (e.g. "510300.SH" for CSI 300 ETF)
            start_date: YYYYMMDD format
            end_date: YYYYMMDD format
        """
        self._rate_limit()

        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

        try:
            df = ts.pro_bar(
                ts_code=symbol,
                asset="FD",  # Fund
                start_date=start_date,
                end_date=end_date,
                adj="qfq",   # Forward-adjusted
            )
        except Exception as e:
            raise RuntimeError(f"Tushare fetch failed for {symbol}: {e}") from e

        if df is None or df.empty:
            return []

        bars = []
        for _, row in df.iterrows():
            bar = ETFDailyBar(
                symbol=row.get("ts_code", symbol),
                trade_date=str(row.get("trade_date", "")),
                open=float(row.get("open", 0)),
                high=float(row.get("high", 0)),
                low=float(row.get("low", 0)),
                close=float(row.get("close", 0)),
                volume=float(row.get("vol", 0)),
                amount=float(row.get("amount", 0)),
            )
            bars.append(bar)

        return sorted(bars, key=lambda b: b.trade_date)

    def fetch_multiple(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, list[ETFDailyBar]]:
        """Fetch daily bars for multiple ETF symbols."""
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.fetch_daily(sym, start_date, end_date)
            except Exception as e:
                print(f"WARN: Failed to fetch {sym}: {e}")
                result[sym] = []
        return result
