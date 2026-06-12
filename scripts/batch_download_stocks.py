"""Batch download using stock_fetcher (proven to work) with big delays."""
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Must come first - patches requests
import akshare as ak

from quanti.data.ingestion.stock_fetcher import StockFetcher
from quanti.data.storage import DataStorage

# Get constituent lists
print("Get constituents...", flush=True)
df300 = ak.index_stock_cons(symbol="000300")
csi300 = sorted(df300["品种代码"].dropna().str[:6].unique().tolist())
df500 = ak.index_stock_cons(symbol="000905")
csi500 = sorted(df500["品种代码"].dropna().str[:6].unique().tolist())
all_codes = sorted(set(csi300 + csi500))
print(f"CSI300:{len(csi300)} CSI500:{len(csi500)} Total:{len(all_codes)}", flush=True)

fetcher = StockFetcher()
fetcher._min_interval = 5.0  # 5s between requests
storage = DataStorage()

ok, fail, skip = 0, 0, 0
start = time.monotonic()

print(f"Downloading {len(all_codes)} stocks (~{len(all_codes)*5/60:.0f} min)", flush=True)

for i, code in enumerate(all_codes):
    # Skip if already have data
    existing = storage.load_bars(code)
    if existing and len(existing) > 100:
        skip += 1
        continue

    try:
        bars = fetcher.fetch_daily(code, start_date="20150101", max_retries=2)
        if bars:
            storage.save_bars_clean(code, bars)
            ok += 1
        else:
            fail += 1
    except Exception as e:
        fail += 1
        if fail <= 5:
            print(f"\n  ERR {code}: {e}", flush=True)

    if (i + 1) % 50 == 0:
        elapsed = time.monotonic() - start
        done = ok + fail + skip
        eta = elapsed / max(done, 1) * (len(all_codes) - i - 1)
        print(f"\n[{done}/{len(all_codes)}] OK={ok} Skip={skip} Fail={fail} | {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)
    elif (i + 1) % 10 == 0:
        print(".", end="", flush=True)

elapsed = time.monotonic() - start
total = ok + skip
print(f"\n\nDONE {elapsed:.0f}s ({elapsed/60:.1f}m)")
print(f"OK:{ok} Skip:{skip} Fail:{fail} Total available:{total}")
