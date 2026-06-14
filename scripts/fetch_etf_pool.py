"""
Phase 1.3: Fetch historical data for all 24 available ETFs and store as Parquet.
Uses the existing AkShareETFetcher and DataStorage infrastructure.
"""
import json
import sys
import time
from pathlib import Path

# Ensure we can import quanti
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quanti.data.ingestion.akshare_fetcher import AkShareETFetcher
from quanti.data.storage import DataStorage


def main():
    # Load audit JSON to identify successful ETFs (skip 588000 which is "empty")
    audit_path = Path(__file__).resolve().parent.parent / "data" / "etf_listing_audit.json"
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    etf_symbols = sorted([
        info["symbol"]
        for info in audit["etfs"].values()
        if info["status"] == "ok"
    ])

    total = len(etf_symbols)
    print(f"Found {total} ETFs with available data. Starting batch fetch...")
    print(f"Symbols: {', '.join(etf_symbols)}")
    print()

    fetcher = AkShareETFetcher()
    storage = DataStorage()

    results = {"ok": 0, "fail": 0, "total_bars": 0}

    for i, symbol in enumerate(etf_symbols, 1):
        try:
            bars = fetcher.fetch_daily(symbol)
            bar_count = len(bars)

            if bar_count == 0:
                print(f"[{i:2d}/{total}] {symbol}: OK (0 bars - empty dataset)")
                storage.log_ingestion("akshare", symbol, "", "", 0, "success")
                results["ok"] += 1
                # Still rate-limit even on empty result
                if i < total:
                    time.sleep(2)
                continue

            first_date = bars[0].trade_date
            last_date = bars[-1].trade_date

            storage.save_bars_clean(symbol, bars)
            storage.log_ingestion("akshare", symbol, first_date, last_date, bar_count, "success")

            print(f"[{i:2d}/{total}] {symbol}: OK ({bar_count} bars)")
            results["ok"] += 1
            results["total_bars"] += bar_count

        except Exception as e:
            print(f"[{i:2d}/{total}] {symbol}: FAILED ({e})")
            storage.log_ingestion("akshare", symbol, "", "", 0, "error", str(e))
            results["fail"] += 1

        # Rate limiting: 2 seconds between calls
        if i < total:
            time.sleep(2)

        # Cooldown every 10 ETFs
        if i % 10 == 0 and i < total:
            print("  (5-second cooldown...)")
            time.sleep(5)

    # ── Summary ──
    print()
    print("=" * 50)
    print("BATCH FETCH SUMMARY")
    print("=" * 50)
    print(f"  Total:       {total}")
    print(f"  Successful:  {results['ok']}")
    print(f"  Failed:      {results['fail']}")
    print(f"  Total bars:  {results['total_bars']}")
    print("Done.")


if __name__ == "__main__":
    main()
