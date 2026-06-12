"""
CSI300 PE / PB percentile data provider.
Fetches index-level PE/PB data and computes rolling percentile ranks
for PE-band dynamic allocation.

Data sources (priority order):
1. Tushare Pro (index_dailybasic) -- needs TUSHARE_TOKEN in .env
2. AkShare (stock_index_pe_lg) -- free, no token needed

Usage:
    fetcher = IndexPEFetcher()
    stats = fetcher.get_stats("000300.SH")
    # Returns: {"pe": 13.9, "pe_percentile": 84.1, "pb": ..., "equity_allocation_pct": 15.9}
"""

import numpy as np
from datetime import datetime, timedelta, date

from quanti.config import settings


# Chinese index name mapping for AkShare
INDEX_NAME_MAP = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000016.SH": "上证50",
    "000688.SH": "科创50",
    "399006.SZ": "创业板指",
    "000001.SH": "上证指数",
}


class IndexPEFetcher:
    """Fetch and cache index PE/PB data, compute rolling percentiles."""

    def __init__(self):
        self._akshare_raw: list[dict] | None = None
        self._cached_at: date | None = None
        self._use_akshare = False

        # Try Tushare first
        self._ts_pro = None
        try:
            import tushare as ts
            token = settings.TUSHARE_TOKEN
            if token and token != "your_token_here":
                self._ts_pro = ts.pro_api(token)
        except Exception:
            pass

        # Fall back to AkShare
        if self._ts_pro is None:
            self._use_akshare = True

    def fetch_history(
        self, index_code: str = "000300.SH",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        """
        Fetch historical daily PE/PB for an index.

        Args:
            index_code: "000300.SH" (CSI300), "000905.SH" (CSI500), etc.
            start_date: YYYYMMDD. Defaults to 10 years ago.
            end_date: YYYYMMDD. Defaults to today.

        Returns:
            List of dicts: {trade_date, pe, pb} sorted ascending by date.
        """
        if self._use_akshare:
            return self._fetch_akshare(index_code)
        return self._fetch_tushare(index_code, start_date, end_date)

    def _fetch_tushare(
        self, index_code: str,
        start_date: str | None, end_date: str | None,
    ) -> list[dict]:
        """Fetch via Tushare Pro."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(
                days=settings.PE_BAND_WINDOW_YEARS * 365 + 30
            )).strftime("%Y%m%d")

        try:
            df = self._ts_pro.index_dailybasic(
                ts_code=index_code,
                start_date=start_date,
                end_date=end_date,
                fields="trade_date,pe,pb",
            )
        except Exception as e:
            raise RuntimeError(f"Tushare index_dailybasic failed: {e}") from e

        if df is None or df.empty:
            return []

        records = []
        for _, row in df.iterrows():
            pe = row.get("pe")
            pb = row.get("pb")
            if pe is not None and pb is not None:
                records.append({
                    "trade_date": str(row["trade_date"]),
                    "pe": float(pe),
                    "pb": float(pb),
                })

        return sorted(records, key=lambda r: r["trade_date"])

    def _fetch_akshare(self, index_code: str) -> list[dict]:
        """Fetch via AkShare (free, legulegu.com source)."""
        import akshare as ak

        cname = INDEX_NAME_MAP.get(index_code)
        if cname is None:
            for code, name in INDEX_NAME_MAP.items():
                if code.replace(".SH", "").replace(".SZ", "") == index_code.replace(".SH", "").replace(".SZ", ""):
                    cname = name
                    break
        if cname is None:
            raise ValueError(f"Unknown index code: {index_code}. Known: {list(INDEX_NAME_MAP.keys())}")

        try:
            df = ak.stock_index_pe_lg(symbol=cname)
        except Exception as e:
            raise RuntimeError(f"AkShare PE fetch failed for {cname}: {e}") from e

        if df is None or df.empty:
            return []

        records = []
        for _, row in df.iterrows():
            date_str = str(row.iloc[0]).replace("-", "")
            # TTM PE is column 6 (0-indexed): 滚动市盈率
            pe = float(row.iloc[6]) if row.iloc[6] and float(row.iloc[6]) > 0 else None
            # PB is not available in this dataset -- derive from index value
            # Use equal-weight rolling PE (column 5) as fallback
            if pe is None:
                pe = float(row.iloc[5]) if row.iloc[5] and float(row.iloc[5]) > 0 else None
            # Use static PE (column 3) as last resort for PB calculation
            pb_val = 0.0

            if pe is not None and pe > 0:
                records.append({
                    "trade_date": date_str,
                    "pe": pe,
                    "pb": pb_val,
                })

        return sorted(records, key=lambda r: r["trade_date"])

    def get_stats(
        self, index_code: str = "000300.SH",
        window_years: int | None = None,
    ) -> dict | None:
        """
        Get latest PE/PB with rolling percentile ranks.

        Returns:
            {
                "trade_date": "20260130",
                "pe": 13.9,
                "pe_percentile": 84.1,
                "pb": 1.5,
                "pb_percentile": 61.6,
                "equity_allocation_pct": 15.9,
            }
            or None if data unavailable.
        """
        if window_years is None:
            window_years = settings.PE_BAND_WINDOW_YEARS
        raw = self.fetch_history(index_code)
        if not raw:
            return None

        latest = raw[-1]
        pe_vals = np.array([r["pe"] for r in raw if r["pe"] > 0], dtype=np.float64)
        pb_vals = np.array([r["pb"] for r in raw if r["pb"] > 0], dtype=np.float64)

        if len(pe_vals) < 10:
            return None

        pe_pctile = float(np.sum(pe_vals <= latest["pe"]) / len(pe_vals)) * 100
        pb_pctile = 50.0  # Default neutral when PB unavailable
        if latest.get("pb", 0) > 0 and len(pb_vals) > 0:
            pb_pctile = float(np.sum(pb_vals <= latest["pb"]) / len(pb_vals)) * 100

        # Clamp to [5, 95] to avoid extreme allocations
        pe_pctile = max(5.0, min(95.0, pe_pctile))

        equity_max = settings.PE_BAND_EQUITY_MAX
        equity_min = settings.PE_BAND_EQUITY_MIN
        equity_pct = equity_max - (pe_pctile / 100.0) * (equity_max - equity_min)
        equity_pct = max(equity_min, min(equity_max, equity_pct)) * 100

        return {
            "trade_date": latest["trade_date"],
            "pe": latest["pe"],
            "pe_percentile": round(pe_pctile, 1),
            "pb": latest["pb"],
            "pb_percentile": round(pb_pctile, 1),
            "equity_allocation_pct": round(equity_pct, 1),
        }

    def get_pe_series(self, index_code: str = "000300.SH") -> list[dict]:
        """Return full PE time series for plotting / analysis."""
        raw = self.fetch_history(index_code)
        return [{"date": r["trade_date"], "pe": r["pe"], "pb": r["pb"]} for r in raw]
