"""
A-share stock daily data fetcher. Must be imported BEFORE any other akshare usage.
Patches requests.get/request with trust_env=False to bypass Windows proxy.
"""
import os

# ── Environment cleanup ──
for k in list(os.environ.keys()):
    if "proxy" in k.lower() or k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        del os.environ[k]

# ── Patch requests BEFORE akshare import ──
import requests as _r

# Fresh no-proxy session per top-level call.
# ak.stock_zh_a_hist() calls requests.get(), which internally creates a new
# Session with trust_env=True by default. We replace the module-level get()
# so it constructs a trust_env=False session each time.
_real_get = _r.get

def _patched_get(url, **kwargs):
    """Replace requests.get() with a no-proxy version. Creates a fresh
    session per call to avoid connection-pool exhaustion on long batch runs."""
    kwargs.setdefault("timeout", 30)
    with _r.Session() as s:
        s.trust_env = False
        return s.get(url, **kwargs)

_r.get = _patched_get

import time
from typing import Optional
import akshare as ak
from quanti.data.schema import ETFDailyBar


class StockFetcher:
    """Fetch A-share stock daily OHLCV via akshare stock_zh_a_hist."""

    def __init__(self):
        self._last_call = 0.0
        self._min_interval = 2.0  # Seconds between calls (Eastmoney rate limit)

    def _rate_limit(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def fetch_daily(
        self, symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_retries: int = 3,
    ) -> list[ETFDailyBar]:
        self._rate_limit()
        for attempt in range(max_retries):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol, period="daily",
                    start_date=start_date or "20000101",
                    end_date=end_date or "20991231",
                    adjust="qfq",
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 3
                    print(f"  Retry {symbol} in {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Failed to fetch {symbol}: {e}") from e

        if df is None or df.empty:
            return []

        bars = []
        for _, row in df.iterrows():
            td = str(row.get("日期", "")).replace("-", "")
            if not td:
                continue
            bars.append(ETFDailyBar(
                symbol=symbol, trade_date=td,
                open=float(row.get("开盘", 0)), high=float(row.get("最高", 0)),
                low=float(row.get("最低", 0)), close=float(row.get("收盘", 0)),
                volume=float(row.get("成交量", 0)), amount=float(row.get("成交额", 0)),
            ))
        return sorted(bars, key=lambda b: b.trade_date)

    def fetch_multiple(
        self, symbols: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, list[ETFDailyBar]]:
        result: dict[str, list[ETFDailyBar]] = {}
        for i, sym in enumerate(symbols):
            try:
                result[sym] = self.fetch_daily(sym, start_date, end_date)
                if (i + 1) % 20 == 0:
                    print(f"  {i+1}/{len(symbols)} done")
            except Exception as e:
                print(f"  WARN {sym}: {e}")
                result[sym] = []
        return result
