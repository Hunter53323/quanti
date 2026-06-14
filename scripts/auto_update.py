"""
Auto-Update + Backtest Pipeline for ETF Rotation v4 (Rising MA Filter)
======================================================================
Usage:
    python scripts/auto_update.py                   # full: fetch + backtest + report
    python scripts/auto_update.py --skip-fetch      # only backtest + report
    python scripts/auto_update.py --skip-backtest   # only fetch data
    python scripts/auto_update.py --skip-macro      # skip macro data fetch

Process:
  0. Fetch macro data (PMI, CGB yield) to data/macro/ (unless --skip-macro)
  1. Fetch latest daily bars for all 7 ETFs from AkShare (Sina source)
  2. Append new bars to data/clean/*.parquet (deduplicated by trade_date)
  3. Run full grid search (54 combos) on Train + Test
  4. Generate data/rising_ma_etf_rotation_report.md
"""

import os
import sys
import time
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

for k in list(os.environ.keys()):
    if k.upper() in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "REQUESTS_CA_BUNDLE"):
        os.environ.pop(k, None)

from quanti.data.ingestion.akshare_fetcher import AkShareETFetcher
from quanti.data.storage import DataStorage


ALL_ETFS = {
    "510300": "沪深300",
    "510500": "中证500",
    "159915": "创业板",
    "510880": "红利ETF",
    "511010": "国债ETF",
    "518880": "黄金ETF",
    "511880": "货币ETF",
}


def fetch_macro_data():
    """Fetch PMI and CGB yield data."""
    print("=" * 60)
    print("  STEP 0: FETCH MACRO DATA")
    print("=" * 60)
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        from scripts.fetch_macro import fetch_all_macro
        fetch_all_macro()
        print("  OK   Macro data fetched\n")
        return True
    except Exception as e:
        print(f"  WARN Macro fetch failed: {e}")
        print("  Continuing with existing macro data...\n")
        return False


def fetch_latest_data():
    """Fetch all 7 ETFs from AkShare and append to clean storage."""
    print("=" * 60)
    print("  STEP 1: FETCH LATEST ETF DATA")
    print("=" * 60)
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ETFs: {', '.join(ALL_ETFS.keys())}\n")

    fetcher = AkShareETFetcher()
    storage = DataStorage()

    results = {}
    for code, name in ALL_ETFS.items():
        prefix = "sh" if code.startswith(("51", "58", "56", "60")) else "sz"
        sina_sym = prefix + code
        try:
            bars = fetcher.fetch_daily(sina_sym)
            if not bars:
                print(f"  WARN  {code} ({name}) -> 0 bars returned")
                results[code] = {"status": "empty", "new": 0}
                continue
            existing_path = storage.clean_dir / f"{code}.parquet"
            old_count = 0
            old_last = "N/A"
            if existing_path.exists():
                import pandas as pd
                old_df = pd.read_parquet(existing_path)
                old_count = len(old_df)
                old_last = old_df["trade_date"].max()
            storage.save_bars_clean(code, bars)
            new_df = pd.read_parquet(existing_path)
            final_count = len(new_df)
            added = final_count - old_count
            new_last = bars[-1].trade_date
            print(f"  OK    {code} ({name:8s}) -> +{added:>4} new bars | "
                  f"{old_count:>5} -> {final_count:>5} | "
                  f"last: {old_last} -> {new_last}")
            results[code] = {"status": "ok", "new": added, "total": final_count,
                           "last_date": new_last}
        except Exception as e:
            print(f"  FAIL  {code} ({name}) -> {e}")
            results[code] = {"status": "error", "error": str(e)}
    return results


def run_backtest():
    """Run v4 backtest and generate report."""
    print("\n" + "=" * 60)
    print("  STEP 2: BACKTEST")
    print("=" * 60)
    import pandas as pd
    from _funcs import load, backtest, metrics, year_bt, bench, P
    data = load()
    train_start = "2022-01-01"
    test_end = datetime.now().strftime("%Y-%m-%d")
    print(f"  Period: {train_start} to {test_end}")
    bt = backtest(data, train_start, test_end,
                  **{k: v for k, v in P.items() if k != "vt"}, vt=P["vt"])
    m = metrics(bt)
    print(f"\n  Results:")
    print(f"    AnnRet: {m['annual_return']:.2%}")
    print(f"    MaxDD:  {m['max_drawdown']:.2%}")
    print(f"    Sharpe: {m['sharpe_ratio']:.2f}")
    print(f"    Calmar: {m['calmar_ratio']:.2f}")
    print(f"\n  Benchmarks:")
    for bm in ("510300", "510500", "159915", "518880"):
        b = bench(bm, train_start, test_end)
        if b:
            print(f"    {bm}: AnnRet={b['annual_return']:.2%} MaxDD={b['max_drawdown']:.2%}")
    return bt, m


def run_v6_signal():
    """Print v6 PE-Band live allocation targets."""
    print("\n" + "=" * 60)
    print("  STEP 2: v6 PE-BAND LIVE SIGNAL")
    print("=" * 60)
    try:
        import pandas as pd
        from scripts.v6_pe_band import pe_pct_at, trend, T as pe_t, GOLD
        today = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
        dr = pd.DatetimeIndex(sorted(set().union(*[df.index for df in pe_t.values()])))
        latest = dr[dr <= today][-1] if len(dr[dr <= today]) > 0 else today

        pp = pe_pct_at(latest)
        eq_pct = 0.60 - pp * (0.60 - 0.10)
        eq_pct = max(0.10, min(0.60, eq_pct))
        g_pct = 0.30 if trend(GOLD, latest, 50) else 0.0
        bd_pct = max(0.0, 1.0 - eq_pct - g_pct)

        print(f"  Date: {latest.date()}")
        print(f"  CSI300 PE: {pp*100:.0f}th percentile (5y)")
        print(f"  Gold: {'TRENDING' if g_pct > 0 else 'not trending'}")
        print(f"\n  v6 Allocation Targets:")
        print(f"    CSI300 (510300): {eq_pct*100:.0f}%")
        print(f"    Gold   (518880): {g_pct*100:.0f}%")
        print(f"    Bonds  (511010): {bd_pct*100:.0f}%")
    except Exception as e:
        print(f"  WARN v6 signal failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-update ETF rotation pipeline")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data fetching, only run backtest")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip backtest, only fetch data")
    parser.add_argument("--skip-macro", action="store_true",
                        help="Skip macro data fetch")
    parser.add_argument("--v6-signal", action="store_true",
                        help="Print v6 PE-Band live signal instead of v4 backtest")
    args = parser.parse_args()

    print("=" * 60)
    print("  ETF Rotation -- Auto Update Pipeline")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    now = time.time()
    if not args.skip_macro:
        fetch_macro_data()
    else:
        print("[SKIP] Macro data fetch (--skip-macro)\n")

    if not args.skip_fetch:
        fetch_latest_data()
    else:
        print("[SKIP] Data fetching (--skip-fetch)")

    if not args.skip_backtest:
        if args.v6_signal:
            run_v6_signal()
        else:
            run_backtest()
    else:
        print("[SKIP] Backtest (--skip-backtest)")

    elapsed = time.time() - now
    print(f"\n  Pipeline complete: {elapsed:.1f}s")
