"""
AkShare ETF data fetcher -- fallback data source.
Uses Sina API (fund_etf_hist_sina) for clean, reliable historical data.
"""
import os
import time
from datetime import datetime, date
import akshare as ak
from quanti.data.schema import ETFDailyBar


def _bypass_proxy():
    """Remove proxy env vars that prevent akshare from reaching Chinese financial APIs."""
    for k in list(os.environ.keys()):
        if k.upper() in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "REQUESTS_CA_BUNDLE"):
            os.environ.pop(k, None)


class AkShareETFetcher:

    def __init__(self):
        self._last_call = 0.0
        self._min_interval = 1.0
        _bypass_proxy()

    def _rate_limit(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def _to_sina_symbol(self, symbol: str) -> str:
        s = symbol.replace(".SH", "").replace(".SZ", "").replace(".sh", "").replace(".sz", "")
        if s.startswith(("sh", "sz")):
            return s
        prefix = "sh" if s.startswith(("51", "58", "60")) else "sz"
        return prefix + s

    @staticmethod
    def _parse_date(d) -> str:
        """Convert datetime.date or str to YYYYMMDD string."""
        if isinstance(d, date):
            return d.strftime("%Y%m%d")
        return str(d).replace("-", "")

    @staticmethod
    def _parse_start(s: str | None) -> date | None:
        if s is None:
            return None
        return datetime.strptime(s, "%Y%m%d").date()

    def fetch_daily(self, symbol, start_date=None, end_date=None):
        self._rate_limit()
        sina_sym = self._to_sina_symbol(symbol)

        try:
            df = ak.fund_etf_hist_sina(symbol=sina_sym)
        except Exception as e:
            raise RuntimeError(f"AkShare Sina fetch failed for {sina_sym}: {e}") from e

        if df is None or df.empty:
            return []

        start_d = self._parse_start(start_date)
        end_d = self._parse_start(end_date)

        bars = []
        for _, row in df.iterrows():
            d = row["date"]
            if start_d and d < start_d:
                continue
            if end_d and d > end_d:
                continue
            bar = ETFDailyBar(
                symbol=sina_sym,
                trade_date=self._parse_date(d),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                amount=float(row["amount"]),
            )
            bars.append(bar)

        return sorted(bars, key=lambda b: b.trade_date)

    def fetch_multiple(self, symbols, start_date=None, end_date=None):
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.fetch_daily(sym, start_date, end_date)
            except Exception as e:
                print(f"WARN: Failed to fetch {sym}: {e}")
                result[sym] = []
        return result
