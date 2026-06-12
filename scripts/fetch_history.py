"""
Batch historical data fetcher for backtesting.
Downloads all ETF and index data in one shot using AkShare (Sina source).

Usage: python scripts/fetch_history.py
"""
import os
import sys
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Bypass proxy BEFORE any other imports
for k in list(os.environ.keys()):
    if k.upper() in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "REQUESTS_CA_BUNDLE"):
        os.environ.pop(k, None)

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from quanti.data.ingestion.akshare_fetcher import AkShareETFetcher
from quanti.data.storage import DataStorage

# ── ETF Universe ──────────────────────────────────────
BASE_ETFS = {
    "510300": "沪深300",
    "510500": "中证500",
    "159915": "创业板50",
    "588000": "科创50",
}

SECTOR_ETFS = {
    "512000": "券商ETF",
    "512480": "半导体ETF",
    "516160": "新能源ETF",
    "159928": "消费ETF",
    "512010": "医药ETF",
    "512660": "军工ETF",
    "512800": "银行ETF",
    "513130": "恒生科技ETF",
}

ALL_ETFS = {**BASE_ETFS, **SECTOR_ETFS}


def fetch_and_save():
    fetcher = AkShareETFetcher()
    storage = DataStorage()

    total_bars = 0
    success = []
    failed = []

    # ── ETF 日线数据 ──
    print("=" * 60)
    print("  AKShare Batch History Download")
    print("=" * 60)
    print(f"\nTotal: {len(ALL_ETFS)} ETFs\n")

    for code, name in ALL_ETFS.items():
        # AkShare Sina format: sh510300 or sz159915
        prefix = "sh" if code.startswith(("51", "58", "56", "60")) else "sz"
        sina_sym = prefix + code

        try:
            bars = fetcher.fetch_daily(sina_sym)
            if bars:
                # Save with bare code (510300) not sina symbol (sh510300)
                storage.save_bars_clean(code, bars)
                last = bars[-1]
                print(f"  OK {code} ({name}) -> {len(bars):>5} bars | {bars[0].trade_date} ~ {last.trade_date}")
                total_bars += len(bars)
                success.append(code)
            else:
                print(f"  WARN {code} ({name}) -> 0 bars returned")
                failed.append(code)
        except Exception as e:
            print(f"  FAIL {code} ({name}) -> {e}")
            failed.append(code)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("  Download Complete")
    print(f"  Success: {len(success)}/{len(ALL_ETFS)}")
    print(f"  Failed:  {len(failed)}/{len(ALL_ETFS)}")
    print(f"  Total bars: {total_bars}")
    if failed:
        print(f"  Failed: {failed}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    return success, failed


if __name__ == "__main__":
    fetch_and_save()
