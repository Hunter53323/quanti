"""
Download ETF universe daily data via StockFetcher (akshare stock_zh_a_hist).

Downloads from each ETF's listing date through 2025-12-31, saves to data/clean/.
Rate limit: 2 seconds between calls (built into StockFetcher).

Usage:
    python scripts/download_etf_universe.py                  # full download
    python scripts/download_etf_universe.py --dry-run        # preview only
    python scripts/download_etf_universe.py --start-from 5   # resume from ETF #5
"""
import sys
sys.path.insert(0, r"C:\study\AIWorkspace\quanti")

import argparse
from datetime import datetime
from pathlib import Path

# Import StockFetcher first (patches requests + cleans proxy env at module level)
from quanti.data.ingestion.stock_fetcher import StockFetcher
from quanti.data.storage import DataStorage

# ── ETF Universe ──────────────────────────────────────────────────────────────

# Single source of truth: the etf_universe config module.
# No fallback -- wrong codes in a fallback would silently corrupt data.
from quanti.config.etf_universe import ETF_UNIVERSE_MULTI

ETF_UNIVERSE = ETF_UNIVERSE_MULTI


END_DATE = "20251231"


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins < 60:
        return f"{mins}m {secs}s"
    hours = mins // 60
    mins = mins % 60
    return f"{hours}h {mins}m {secs}s"


def _fmt_date(raw: str) -> str:
    """Normalise a date string: strip dashes so it becomes YYYYMMDD."""
    return raw.replace("-", "")


def main():
    parser = argparse.ArgumentParser(
        description="Download daily OHLCV data for the full ETF multi-industry universe."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List ETFs and date ranges without fetching."
    )
    parser.add_argument(
        "--start-from", type=int, default=0, metavar="N",
        help="Skip the first N ETFs (resume from the Nth, 0-indexed)."
    )
    args = parser.parse_args()

    total = len(ETF_UNIVERSE)
    etfs = list(ETF_UNIVERSE)

    if total == 0:
        print("ERROR: ETF universe is empty. Nothing to do.")
        sys.exit(1)

    # Apply start-from offset
    if args.start_from > 0:
        if args.start_from >= total:
            print(f"ERROR: --start-from {args.start_from} is >= total ETFs ({total}). Nothing to do.")
            sys.exit(1)
        etfs = etfs[args.start_from:]
        print(f"Resuming from ETF #{args.start_from + 1} of {total} "
              f"({len(etfs)} remaining)\n")

    # ── Dry-run / plan display ───────────────────────────────────────────────
    print(f"{'=' * 70}")
    print(f"  ETF Universe Download  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'=' * 70}")
    print(f"  Total ETFs: {total}")
    print(f"  Start from: ETF #{args.start_from + 1}")
    print(f"  Date range: listing date ~ {END_DATE}")
    print()

    if args.dry_run:
        estimated = len(etfs) * 2  # 2 seconds per ETF (built-in rate limit)
        print(f"{'ID':>3}  {'Code':>6}  {'Sector':<14}  {'Name':<28}  {'From':<10}  {'To':<10}")
        print(f"{'─' * 3}  {'─' * 6}  {'─' * 14}  {'─' * 28}  {'─' * 10}  {'─' * 10}")
        for i, etf in enumerate(etfs, args.start_from + 1):
            print(f"{i:3d}  {etf['code']:>6}  {etf['sector']:<14}  {etf['name']:<28}  "
                  f"{_fmt_date(etf['list_date']):<10}  {END_DATE:<10}")
        print()
        print(f"  Estimated download time: {format_duration(estimated)} "
              f"({len(etfs)} ETFs x 2s rate limit + retries)")
        print("  (dry-run -- no data was fetched)")
        return

    # ── Live download ─────────────────────────────────────────────────────────
    fetcher = StockFetcher()
    storage = DataStorage()

    total_bars = 0
    failures: list[tuple[str, str]] = []  # (code, error_message)

    for idx, etf in enumerate(etfs, args.start_from + 1):
        code = etf["code"]
        start_date = _fmt_date(etf["list_date"])

        try:
            bars = fetcher.fetch_daily(
                symbol=code,
                start_date=start_date,
                end_date=END_DATE,
                max_retries=3,
            )
            bar_count = len(bars)

            if bar_count == 0:
                print(f"[{idx:2d}/{total}] {code} ({etf['name']}): OK  (0 bars)")
                storage.log_ingestion(
                    "akshare", code, start_date, END_DATE, 0, "success",
                )
                continue

            storage.save_bars_clean(code, bars)
            storage.log_ingestion(
                "akshare", code, bars[0].trade_date, bars[-1].trade_date,
                bar_count, "success",
            )

            total_bars += bar_count
            print(f"[{idx:2d}/{total}] {code} ({etf['name']}): done, {bar_count} bars  "
                  f"{bars[0].trade_date} ~ {bars[-1].trade_date}")

        except Exception as e:
            msg = str(e)
            print(f"[{idx:2d}/{total}] {code} ({etf['name']}): FAILED  ({msg})")
            storage.log_ingestion(
                "akshare", code, start_date, END_DATE, 0, "error", msg,
            )
            failures.append((code, msg))

    # ── Summary ───────────────────────────────────────────────────────────────
    successful = total - len(failures)
    print()
    print(f"{'=' * 70}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total:      {total}")
    print(f"  Successful: {successful}")
    print(f"  Failed:     {len(failures)}")
    print(f"  Total bars: {total_bars}")
    if failures:
        print(f"  Failed ETFs:")
        for code, msg in failures:
            print(f"    - {code}: {msg}")
    print()
    print("  Per-ETF breakdown:")
    storage_base = Path(storage.clean_dir)
    for etf in ETF_UNIVERSE:
        parquet = storage_base / f"{etf['code']}.parquet"
        if parquet.exists():
            import pandas as pd  # noqa: late import
            df = pd.read_parquet(parquet)
            print(f"    {etf['code']} ({etf['name']}): {len(df)} bars  "
                  f"{df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print()
    print(f"  Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
