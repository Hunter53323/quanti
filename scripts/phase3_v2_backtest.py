"""V2 Stock Momentum Strategy backtest: Train/Validate/LiveTest."""
import gc
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from quanti.backtest.engine import BacktestEngine
from quanti.data.storage import DataStorage
from quanti.strategy.stock_momentum import StockMomentumStrategy

CAPITAL = 90000
PARAMS = dict(top_n=8, stop_loss_pct=15.0, ma_fast=20, ma_slow=60)

# ── Load data ──────────────────────────────────────
print("Loading data...", flush=True)
storage = DataStorage()
all_parquets = sorted(storage.clean_dir.glob("*.parquet"))
# Exclude ETFs (5-6 digit codes starting with 5/1/588) and non-stock
stocks = []
for p in all_parquets:
    code = p.stem
    # Stock codes are 6 digits, ETFs start with 51/58/15/56
    if len(code) == 6 and not code.startswith(("51", "58", "15", "56")):
        stocks.append(code)

print(f"  Found {len(stocks)} stocks", flush=True)

# ── Split by period ────────────────────────────────
PERIODS = [
    ("Train",    "20150101", "20201231"),
    ("Validate", "20210101", "20221231"),
    ("LiveTest", "20230101", "20251231"),
]

def load_period(period_start, period_end):
    """Load raw ETFDailyBar objects (what engine.run() expects as input)."""
    bars_dict = {}
    for code in stocks:
        raw = storage.load_bars(code)
        if not raw:
            continue
        filtered = [r for r in raw if period_start <= r.trade_date <= period_end]
        if len(filtered) < 130:
            continue
        bars_dict[code] = filtered
    return bars_dict

results = {}
all_start = time.monotonic()

for pname, pstart, pend in PERIODS:
    print(f"\n{'='*80}", flush=True)
    print(f"  [{pname}] {pstart}~{pend}", flush=True)
    print(f"{'='*80}", flush=True)

    t0 = time.monotonic()
    bars = load_period(pstart, pend)
    load_t = time.monotonic() - t0
    print(f"  Loaded {len(bars)} stocks in {load_t:.0f}s", flush=True)

    t0 = time.monotonic()
    engine = BacktestEngine(StockMomentumStrategy, PARAMS, CAPITAL)
    r = engine.run(list(bars.keys()), bars, period_label=pname)
    run_t = time.monotonic() - t0

    results[pname] = {
        "sharpe": round(r.sharpe_ratio, 3), "cagr": round(r.cagr_pct, 1),
        "maxdd": round(r.max_drawdown_pct, 1), "trades": len(r.trades),
        "winrate": round(r.win_rate_pct, 0), "calmar": round(r.calmar_ratio, 3),
        "turnover": round(r.annual_turnover_pct, 0), "n_stocks": len(bars),
        "start": r.start_date, "end": r.end_date,
    }

    print(f"  Sharpe={r.sharpe_ratio:.3f}  CAGR={r.cagr_pct:.1f}%  MaxDD={r.max_drawdown_pct:.1f}%  "
          f"WinRate={r.win_rate_pct:.0f}%  Trades={len(r.trades)}  TO/yr={r.annual_turnover_pct:.0f}%  "
          f"Calmar={r.calmar_ratio:.3f}  ({run_t:.0f}s)", flush=True)

    gc.collect()

# ── Summary ─────────────────────────────────────────
print(f"\n{'='*80}")
print("  SUMMARY (top_n=8, stop=-15%)")
print(f"{'='*80}")
print(f"  {'Period':<12s} | {'N Stocks':>8s} | {'Sharpe':>7s} | {'CAGR':>7s} | {'MaxDD':>5s} | {'Trades':>6s} | {'Win%':>5s} | {'Calmar':>6s}")
print(f"  {'-'*12} | {'-'*8} | {'-'*7} | {'-'*7} | {'-'*5} | {'-'*6} | {'-'*5} | {'-'*6}")
for pname, _, _ in PERIODS:
    r = results[pname]
    print(f"  {pname:<12s} | {r['n_stocks']:>8d} | {r['sharpe']:7.3f} | {r['cagr']:6.1f}% | {r['maxdd']:5.1f}% | {r['trades']:6d} | {r['winrate']:4.0f}% | {r['calmar']:6.3f}")

elapsed = time.monotonic() - all_start
print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

# Gates
r_lt = results.get("LiveTest", {})
r_tr = results.get("Train", {})
print("\nGATE CHECK (LiveTest):")
gates = [
    ("Sharpe >= 0.3", r_lt.get("sharpe", 0) >= 0.3),
    ("MaxDD < 30%", r_lt.get("maxdd", 100) < 30),
    ("Turnover < 400%", r_lt.get("turnover", 999) < 400),
]
all_ok = True
for g, ok in gates:
    s = "PASS" if ok else "FAIL"
    if not ok: all_ok = False
    print(f"  [{s}] {g}")
print(f"  >> {'ALL PASS' if all_ok else 'SOME FAIL'}")
